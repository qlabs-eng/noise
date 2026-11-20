import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _indices_kernel(Step, Pick, GROUPS: tl.constexpr, STRIDE: tl.constexpr, RANK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    groups = i.to(tl.uint32)
    step = tl.load(Step).to(tl.uint32)
    # step_signs' mix with a salt of its own (134), so the picks are not the probe signs' stream
    z = groups * tl.full((), 0x85EBCA77, tl.uint32) + step * tl.full((), 0x9E3779B1, tl.uint32) + 134
    z = (z ^ (z >> 15)) * tl.full((), 0x2C1B3C6D, tl.uint32)
    z = (z ^ (z >> 12)) * tl.full((), 0x297A2D39, tl.uint32)
    phase = ((z ^ (z >> 15)) + RANK) & (STRIDE - 1)
    tl.store(Pick + i, (groups * STRIDE + phase).to(tl.int64), i < GROUPS)


def indices(step, rank: int, stride: int, vocab: int):
    groups = vocab // stride
    out = torch.empty((groups,), device=step.device, dtype=torch.int64)
    _indices_kernel[(triton.cdiv(groups, 256),)](step, out, GROUPS=groups, STRIDE=stride, RANK=rank, BLOCK=256)
    return out

@torch.compile(dynamic=False)
def powers(c, pick):
    selected = c.index_select(1, pick)
    return (selected.to(torch.bfloat16), (selected * selected).to(torch.bfloat16),
            (selected * selected * selected).to(torch.bfloat16))

@triton.jit
def _sample_weights_kernel(Base, Inv0, Pick, FG1, Fast, Rows, Total, T: tl.constexpr, V: tl.constexpr,
    BLOCK: tl.constexpr, CAP: tl.constexpr, STRIDE: tl.constexpr):
    """The readout cycle's per-logit part for one token in one pass: f g1 (fp32), the five bf16 weight rows, and the row
    sums (total, and rows' three)."""
    token = tl.program_id(0).to(tl.int64)
    inv0 = tl.load(Inv0 + token)
    total = tl.zeros((BLOCK,), tl.float32)
    r1 = tl.zeros((BLOCK,), tl.float32)
    r2 = tl.zeros((BLOCK,), tl.float32)
    r3 = tl.zeros((BLOCK,), tl.float32)
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        z = tl.load(Base + token * V + cols) * inv0
        t = libdevice.tanh(z / CAP)
        f = libdevice.exp(CAP * t)
        g1 = 1 - t * t
        d1 = -2 * t * g1 / CAP
        d2 = -2 * g1 * (1 - 3 * t * t) / (CAP * CAP)
        g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2
        fz, fzz = f * z, f * z * z
        total += f
        r1 += fz * g1
        r2 += fzz * g2
        r3 += fzz * z * g3
        out = token * V + cols
        tl.store(FG1 + out, f * g1)
        compact = token * (V // STRIDE) + cols // STRIDE
        selected = cols == tl.load(Pick + cols // STRIDE)
        tl.store(Fast + compact, (fz * g2).to(tl.bfloat16), selected)
        tl.store(Fast + 1 * T * (V // STRIDE) + compact, (fzz * g3).to(tl.bfloat16), selected)
        tl.store(Fast + 2 * T * (V // STRIDE) + compact, (f * g2).to(tl.bfloat16), selected)
        tl.store(Fast + 3 * T * (V // STRIDE) + compact, (fz * g3).to(tl.bfloat16), selected)
        tl.store(Fast + 4 * T * (V // STRIDE) + compact, (f * g3).to(tl.bfloat16), selected)
    tl.store(Total + token, tl.sum(total, 0))
    tl.store(Rows + token * 3, tl.sum(r1, 0))
    tl.store(Rows + token * 3 + 1, tl.sum(r2, 0))
    tl.store(Rows + token * 3 + 2, tl.sum(r3, 0))

def weights(ops, x, base, h, pick, stride):
    steps, inv, inv0, delta, sigma = ops.cycle_scales(x, h)
    tokens, vocab = base.shape
    fg1 = torch.empty_like(base)
    fast = torch.empty((5, tokens, vocab // stride), device=base.device, dtype=torch.bfloat16)
    rows = torch.empty((tokens, 3), device=base.device)
    total = torch.empty((tokens, 1), device=base.device)
    _sample_weights_kernel[(tokens,)](base.contiguous(), inv0.contiguous(), pick, fg1, fast, rows, total,
        T=tokens, V=vocab, BLOCK=1024, CAP=ops.SOFTCAP, STRIDE=stride, num_warps=4)
    return fg1, fast, rows, total, steps, inv, inv0, delta, sigma


@torch.compile(dynamic=False)
def sampled_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h, stride: int):
    fast = [x.float() * stride for x in prods]
    ratio = (sigma * exact + delta * sigma * fast[0] + 0.5 * delta ** 2 * sigma * fast[1] + 0.5 * sigma ** 2 * fast[2]
      + 0.5 * delta * sigma ** 2 * fast[3] + sigma ** 3 / 6 * fast[4]
      + delta * rows[:, :1] + 0.5 * delta ** 2 * rows[:, 1:2] + delta ** 3 / 6 * rows[:, 2:])
    clean_partition = torch.log(total)
    base_y = base.gather(1, y[:, None])
    dirs_y = c.T[y]
    clean = clean_partition - (15 * torch.tanh(base_y * inv0 / 15))
    moved = clean_partition + torch.log1p(ratio / total) - (15 * torch.tanh((base_y + steps * dirs_y) * inv / 15))
    return ((moved - clean) / steps) @ h / 512


def credit(ops, model, x, y, stride):
    if stride not in (1, 2, 4):
        raise ValueError("Identity vocabulary stride must be 1, 2, or 4")
    h = model.H * ops.step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head
    base = x @ model.lm_head
    pick = indices(model.step_t, model.rank, stride, ops.VOCAB)
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = weights(ops, x, base, h, pick, stride)
    exact = fg1 @ c.T
    cb, c2, c3 = powers(c, pick)
    n = x.shape[0]
    p01 = (fast[:2].reshape(2*n, -1) @ cb.T).view(2, n, -1)
    p23 = (fast[2:4].reshape(2*n, -1) @ c2.T).view(2, n, -1)
    prods = [p01[0], p01[1], p23[0], p23[1], fast[4] @ c3.T]
    return sampled_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h, stride)
