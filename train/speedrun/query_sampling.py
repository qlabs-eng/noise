import torch
import triton
import triton.language as tl
import depth_queries
from moment_kernels import _query_response, _by_token, MOMENT_CONFIG


@triton.jit
def _sample_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, f2_ptr, cos_ptr, sin_ptr,
    lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, OUT_LAYER,
    OUT_TOKEN, out_map, D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    WINDOW: tl.constexpr, OUT_H: tl.constexpr, seeds,
    GROUP: tl.constexpr, owners, weights):
    """moment_kernels' query moments on the sampled queries only: each program's J feature pairs (of D // 2) feed
    both raw coordinates of each pair (d = pair and d = pair + D // 2)."""
    parts: tl.constexpr = D // 2 // J
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S
    part = i % parts
    tile = s0 // BLOCK_S
    metadata = tl.load(owners + tile)
    row = metadata >> 12
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    selected = s0 + tl.arange(0, BLOCK_S)
    offs_s = tl.load(seeds + selected)
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
    first_group = (metadata >> 6) & 63
    last_group = metadata & 63
    start = tl.maximum(first_group * GROUP - WINDOW, 0) // BLOCK_T
    last = ((last_group + 1) * GROUP - 1) // BLOCK_T + 1
    for it in range(start, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale
        c = tl.dot(hub, tl.trans(v)) - hub_out[:, None]
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= WINDOW), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
        f2 = tl.load(f2_ptr + base + offs_t[:, None] * D + (part * 2 * J + tl.arange(0, 2 * J))[None, :])
        linear = f2.to(tl.float32)
        f4 = tl.reshape(tl.join(linear, linear * linear), (BLOCK_T, 4 * J)).to(tl.bfloat16)
        left, right = tl.split(tl.reshape(linear, (BLOCK_T, J, 2)))
        fxy = (left * right).to(tl.bfloat16)
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
    factors = tl.load(weights + selected)
    output = out_ptr + _by_token(row, OUT_H, OUT_LAYER, D, out_map) + offs_s[:, None] * OUT_TOKEN
    for half in tl.static_range(2):
        d = pairs + half * (D // 2)
        z = _query_response(n4, d4, nxy, dxy, l2, m2, s1n, s2n, s1d, s2d, raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr,
          row, BH, T, offs_s, d, M, RAW_GPU, RAW_STRIDE, D, J, BLOCK_S)
        tl.store(output + d[None, :], z.to(tl.bfloat16).to(tl.float32) * factors[:, None])


@triton.jit
def _clear_queries(out, maps, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, OUT_LAYER: tl.constexpr,
                   OUT_TOKEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = _by_token(row, H, OUT_LAYER, D, maps)
    tl.store(out + start + (j // D) * OUT_TOKEN + j % D, 0., j < T * D)


def sample_query_scores_into(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hub: torch.Tensor, raw: torch.Tensor,
                            sq: torch.Tensor, side: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            lse: torch.Tensor, hub_out: torch.Tensor, steps: torch.Tensor, out: torch.Tensor,
                            seeds: torch.Tensor, strength: torch.Tensor, layers: torch.Tensor, window: int,
                            group: int, *, draw_labels, depth) -> None:
    rows, t, d = q.shape
    heads = out.shape[-1] // d
    c = MOMENT_CONFIG
    batch_rows, low, high, panel = depth
    owner, positions, inverse = depth_queries.plan(
        strength, sq, seeds, layers, heads, batch_rows, low, high, panel, center=hub_out,
        labels=draw_labels)
    _clear_queries[(rows, triton.cdiv(t * d, 1024))](out, layers, T=t, H=heads, D=d,
        OUT_LAYER=out.stride(0), OUT_TOKEN=out.stride(1), BLOCK=1024)
    parts = d // 2 // c['PAIRS']
    grid = (parts, owner.numel())
    _sample_queries_kernel[grid](
        q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
        rows // raw.shape[1], t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), t, 0,
        out.stride(0), out.stride(1), layers, D=d, J=c['PAIRS'], BLOCK_T=c['BLOCK_T'], BLOCK_S=c['BLOCK_S'],
        WINDOW=min(window, t), OUT_H=heads, seeds=positions, GROUP=group,
        owners=owner, weights=inverse,
        num_warps=c['num_warps'], num_stages=c['num_stages'])
