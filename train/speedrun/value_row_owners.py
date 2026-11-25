"""Vocabulary-row ownership for the four Identity value tables, and the all-reduce and Adam step of the rest of the
Identity flat gradient."""
import torch
import torch.distributed as dist
import triton
import triton.language as tl

PARTNER = (2, 3, 0, 1)

# The identity flat gradient's all-reduce: lm_head and the scalars in fp32, wte and the gates in fp16 at scale 32.
PREFIX_N = 2097152
TAIL_START = 4194320


@triton.jit
def _copy_small(X, Y, N: tl.constexpr, PREFIX: tl.constexpr,
                GAP: tl.constexpr, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    physical = j + tl.where(j >= PREFIX, GAP, 0)
    if INVERSE:
        value = tl.load(X + j, j < N, 0).to(tl.float32) * 0.03125
        tl.store(Y + physical, value, j < N)
    else:
        value = tl.load(X + physical, j < N, 0).to(tl.float32) * 32.0
        tl.store(Y + j, value, j < N)


def bind(model, group):
    blocks = tuple(model.ve)
    values = torch.stack([model.ve[i] for i in blocks], dim=1).contiguous()
    vocab = values.shape[0]
    model.owner_bank = values
    model.owner_group = group
    model.owner_stream = torch.cuda.Stream()
    model.owner_slice = slice(model.rank * (vocab // 8), (model.rank + 1) * (vocab // 8))
    model.owner_in = torch.empty_like(values[model.owner_slice], dtype=torch.float16)
    small_count = model.flat_grad.numel() - values.numel() - (TAIL_START - PREFIX_N)
    model.owner_small_wire = torch.empty(small_count, device=values.device, dtype=torch.float16)
    model.owner_ve_start = model.flat_grad.numel() - values.numel()
    for k, block in enumerate(blocks):
        param = values[:, k]
        model.ve[block] = param
        _, grad, lr = model.adam[8 + k]
        model.adam[8 + k] = param, grad, lr


def copy_small(flat, wire, inverse=False):
    n = wire.numel()
    p = PREFIX_N
    gap = TAIL_START - p
    a, b = (wire, flat) if inverse else (flat, wire)
    _copy_small[(triton.cdiv(n, 2048),)](a, b, n, p, gap, inverse, 2048)


def reduce(model):
    head = dist.all_reduce(model.flat_grad[PREFIX_N:TAIL_START], async_op=True)
    copy_small(model.flat_grad, model.owner_small_wire)
    other = dist.all_reduce(model.owner_small_wire, async_op=True)
    return head, other


@torch.compile(dynamic=False)
def delta_rows(g, ms, vs, n, rate, step, beta1, beta2, eps):
    c1, c2 = 1 - beta1 ** (step + 1), 1 - beta2 ** (step + 1)
    out = []
    for j in range(4):
        grad = ((g[:, j].float() * 0.03125) / n).to(torch.bfloat16).float()
        m = beta1 * ms[j] + (1 - beta1) * grad
        v = beta2 * vs[j] + (1 - beta2) * grad.square()
        out.append(rate * (m / c1) / (torch.sqrt(v) / torch.sqrt(c2) + eps))
        ms[j].copy_(m)
        vs[j].copy_(v)
    return torch.stack(out)


@torch.compile(dynamic=False)
def _raw_stats(delta, frames):
    transform = torch.bmm(frames, frames[list(PARTNER)].transpose(1, 2))
    sent = torch.bmm(delta, transform)
    borrow = sent[list(PARTNER)]
    scalar = torch.stack([delta.square().sum((1, 2)), borrow.square().sum((1, 2)),
                          (delta * borrow).sum((1, 2))], dim=1)
    stats = torch.cat([delta.sum(1), borrow.sum(1), scalar], dim=1)
    return borrow, stats


@torch.compile(dynamic=False)
def _unpack_raw(delta, raw_borrow, stats, vocab: int):
    dim = delta.shape[-1]
    mu = stats[:, None, :dim] / vocab
    bm = stats[:, None, dim:2 * dim] / vocab
    total, borrow_total, cross = stats[:, 2 * dim:].unbind(1)
    c2 = (total - vocab * mu.square().sum((1, 2))).clamp_min(0.)
    b2 = (borrow_total - vocab * bm.square().sum((1, 2))).clamp_min(0.)
    cb = cross - vocab * (mu * bm).sum((1, 2))
    return mu, delta - mu, raw_borrow - bm, torch.stack([total, c2, b2, cb])


@torch.compile(dynamic=False)
def _shape(mu, centered, borrowed, stats, vocab: int, gain: float, transfer: float):
    total, csq, bsq, cross = stats[:, :, None, None].unbind(0)
    common = vocab * mu.square().sum((1, 2), keepdim=True)
    s = torch.sqrt(total / (total + (gain * gain - 1.) * common).clamp_min(1.e-30))
    energy = s.square() * csq
    b = torch.sqrt(energy / bsq.clamp_min(1.e-30))
    mixed = (1. + transfer * transfer) * energy + 2 * transfer * s * b * cross
    scale = torch.sqrt(energy / mixed.clamp_min(1.e-30))
    return gain * s * mu + scale * (s * centered + transfer * b * borrowed)


def shape(delta, frames, model, gain, transfer):
    vocab = model.owner_bank.shape[0]
    sent, stats = _raw_stats(delta, frames)
    dist.all_reduce(stats, group=model.owner_group)
    mu, centered, borrow, stats = _unpack_raw(delta, sent, stats, vocab)
    return _shape(mu, centered, borrow, stats, vocab, gain, transfer)


@torch.compile(dynamic=False)
def _apply(values, delta):
    return (values.float() - delta.transpose(0, 1)).to(values.dtype).contiguous()


def advance(model, n, frames, api):
    rows = model.owner_slice
    rate = api.LR_VE * api.id_rate_scale(model.step_t)
    delta = delta_rows(model.owner_in, [m[rows] for m in model.m[8:]],
                       [v[rows] for v in model.v[8:]], n, rate, model.step_t,
                       api.BETA1, api.BETA2, api.ADAM_EPS)
    if api.SHAPING_ON:
        delta = shape(delta, frames, model, api.VE_COMMON_GAIN, api.VE_PARTNER_GAIN)
    published = _apply(model.owner_bank[rows], delta)
    gathered = dist.all_gather_into_tensor(model.owner_bank, published,
                                           group=model.owner_group, async_op=True)
    gathered.wait()
    return published


def update(model, works, n, api):
    head, other = works['flat']
    main, side = torch.cuda.current_stream(), model.owner_stream
    head.wait()
    other.wait()
    copy_small(model.flat_grad, model.owner_small_wire, inverse=True)
    ps, gs, lrs = zip(*model.adam[:8])
    api.adam_math(list(ps), list(gs), n, model.m[:8], model.v[:8], lrs, model.step_t)
    main.wait_stream(side)
