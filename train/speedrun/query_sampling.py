import torch
import triton
import triton.language as tl
from pps_budget import importance_plan
from moment_kernels import _query_response, _by_token, _pair_operands, MOMENT_CONFIG


@triton.jit
def _sample_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, f2_ptr, f4_ptr, xy_ptr, cos_ptr, sin_ptr,
    lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, W_STRIDE, OUT_LAYER,
    OUT_TOKEN, out_map, D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    WINDOW: tl.constexpr, OUT_H: tl.constexpr, OUT_MAPPED: tl.constexpr, seeds, KEEP: tl.constexpr,
    GROUP: tl.constexpr):
    """moment_kernels' query moments on the sampled queries only: each program's J feature pairs (of D // 2) feed
    both raw coordinates of each pair (d = pair and d = pair + D // 2)."""
    parts: tl.constexpr = D // 2 // J
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S
    row, part = i // parts, i % parts
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    selected = s0 + tl.arange(0, BLOCK_S)
    offs_s = tl.load(seeds + row * (T // GROUP * KEEP) + selected)
    offs_d = tl.arange(0, D)
    pairs = part * J + tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    lse = tl.load(lse_ptr + row * T + offs_s)
    hub_out = tl.load(hubout_ptr + row * T + offs_s)
    n4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    d4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    nxy = tl.zeros((BLOCK_S, J), tl.float32)
    dxy = tl.zeros((BLOCK_S, J), tl.float32)
    l2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    m2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    s1n = tl.zeros((BLOCK_S,), tl.float32)
    s2n = tl.zeros((BLOCK_S,), tl.float32)
    s1d = tl.zeros((BLOCK_S,), tl.float32)
    s2d = tl.zeros((BLOCK_S,), tl.float32)
    last = (((s0 + BLOCK_S - 1) // KEEP) * GROUP + GROUP - 1) // BLOCK_T + 1
    start = tl.maximum((s0 // KEEP) * GROUP - WINDOW, 0) // BLOCK_T
    for it in range(start, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale
        c = tl.dot(hub, tl.trans(v)) - hub_out[:, None]
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= WINDOW), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
        f2 = tl.load(f2_ptr + base + offs_t[:, None] * D + (part * 2 * J + tl.arange(0, 2 * J))[None, :])
        f4 = tl.load(f4_ptr + 2 * base + offs_t[:, None] * (2 * D) + (part * 4 * J + tl.arange(0, 4 * J))[None, :])
        fxy = tl.load(xy_ptr + base // 2 + offs_t[:, None] * (D // 2) + pairs[None, :])
        wn = a * c
        wnl = (wn * logits).to(tl.bfloat16)
        al = (a * logits).to(tl.bfloat16)
        wnb, ab = wn.to(tl.bfloat16), a.to(tl.bfloat16)
        n4 += tl.dot(wnb, f4)
        nxy += tl.dot(wnb, fxy)
        l2 += tl.dot(wnl, f2)
        d4 += tl.dot(ab, f4)
        dxy += tl.dot(ab, fxy)
        m2 += tl.dot(al, f2)
        s1n += tl.sum(wn * logits, axis=1)
        s2n += tl.sum(wn * logits * logits, axis=1)
        s1d += tl.sum(a * logits, axis=1)
        s2d += tl.sum(a * logits * logits, axis=1)
    for half in tl.static_range(2):
        _query_response(n4, d4, nxy, dxy, l2, m2, s1n, s2n, s1d, s2d, raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr,
          out_ptr, row, BH, T, offs_s, pairs + half * (D // 2), M, RAW_GPU, RAW_STRIDE, W_STRIDE, D, J, BLOCK_S,
          OUT_H, OUT_LAYER, OUT_TOKEN, out_map, OUT_MAPPED)


@triton.jit
def _clear_queries(out, maps, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, OUT_LAYER: tl.constexpr,
                   OUT_TOKEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = _by_token(row, H, OUT_LAYER, OUT_TOKEN, D, maps, True)
    tl.store(out + start + (j // D) * OUT_TOKEN + j % D, 0., j < T * D)


@triton.jit
def _weight_queries(out, maps, positions, inverse, S: tl.constexpr,
                    H: tl.constexpr, D: tl.constexpr, OUT_LAYER: tl.constexpr,
                    OUT_TOKEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = j < S * D
    selected = j // D
    token = tl.load(positions + row * S + selected, live, 0)
    factor = tl.load(inverse + row * S + selected, live, 0)
    start = _by_token(row, H, OUT_LAYER, OUT_TOKEN, D, maps, True)
    pointer = out + start + token * OUT_TOKEN + j % D
    value = tl.load(pointer, live, 0)
    tl.store(pointer, value * factor, live)

def sample_query_scores_into(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hub: torch.Tensor, raw: torch.Tensor,
                            sq: torch.Tensor, side: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            lse: torch.Tensor, hub_out: torch.Tensor, steps: torch.Tensor, out: torch.Tensor,
                            seeds: torch.Tensor, strength: torch.Tensor, layers: torch.Tensor, window: int, keep: int,
                            group: int) -> None:
    rows, t, d = q.shape
    heads = out.shape[-1] // d
    c = MOMENT_CONFIG
    assert group in (16, 64) and keep in (group // 2, 3 * group // 8)
    positions, inverse = importance_plan(strength, seeds, layers, heads, group_size=group, keep=keep)
    assert d == 64 and 0 < keep <= group and t % group == 0
    assert (t // group * keep) % c['BLOCK_S'] == 0
    f2, f4, xy = _pair_operands(side.contiguous())
    _clear_queries[(rows, triton.cdiv(t * d, 1024))](out, layers, T=t, H=heads, D=d,
        OUT_LAYER=out.stride(0), OUT_TOKEN=out.stride(1), BLOCK=1024)
    _sample_queries_kernel[(rows * (d // 2 // c['PAIRS']), t // group * keep // c['BLOCK_S'])](
        q, k, v, hub, raw, sq, f2, f4, xy, cos, sin, lse, hub_out, steps, out,
        rows // raw.shape[1], t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), t, 0, 0,
        out.stride(0), out.stride(1), layers, D=d, J=c['PAIRS'], BLOCK_T=c['BLOCK_T'], BLOCK_S=c['BLOCK_S'],
        WINDOW=min(window, t), OUT_H=heads, OUT_MAPPED=True, seeds=positions, KEEP=keep, GROUP=group,
        num_warps=c['num_warps'], num_stages=c['num_stages'])
    selected = t // group * keep
    _weight_queries[(rows, triton.cdiv(selected * d, 512))](
        out, layers, positions, inverse, S=selected, H=heads, D=d,
        OUT_LAYER=out.stride(0), OUT_TOKEN=out.stride(1), BLOCK=512,
        num_warps=4)
