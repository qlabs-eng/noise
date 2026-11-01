"""Faster key / value / query-moment credit kernels for run.py's Cut (snapshot 0beafd8), numerically the reference's.

Drop-in replacements (same inputs and outputs as the reference functions):
- implicit_key_context(q, k, values, hub, window) -> (lse, center): tiles of 64 queries (reference: 16).
- moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window): the
  reference's programs 2 and 3 of each token tile (repeat = 4) recomputed the moment sums of programs 0 and 1 (the
  feature pairs are i % 32): each program now sums one half of the pairs and writes both raw coordinates of each
  pair; the keys' bf16 pair operands are formed once per call (_pair_operands), not per program and key tile.
- key_credits(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma) -> key: batched_key_scores' key scan
  (tiles of 64 targets, wgmma).
- value_credits(context, corrected, window) -> (raw, terminal).

The value credit split by dependency (raw = sum p * hub needs no descending chain, and the chain only reads raw;
terminal = sum p * corrected needs the chain):
- raw_value_credits(q, k, lse, hub, window) / terminal_value_credits(q, k, lse, corrected, window): one 64-channel
  pass over a stack of layers (heads of several layers of one window size along the first axis).
- key_and_raw_value_credits(...) -> (key, raw): the key scan with the raw value credit in the same pass.

`*_by_token` variants (key_credits, key_and_raw_value_credits, moment_query_scores, raw/terminal_value_credits) write
(layers, T, heads * 64) fp32 directly, the layout the gradient reassembly reads (layer j's credit is out[j], a view),
and terminal_value_credits_by_token reads corrected in that layout too: no transpose / reshape / cat / .float() copies.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

PREC = 'tf32'  # run.py's KEY_DOT_PRECISION
# warp-specialized Gluon kernels, bitwise the Triton kernels' numbers; the Triton kernels stay for the calls they do
# not take (and with the switch at 0)
ATTN_WS = 1  # moment_query_scores_into's OUT_H path on moment_queries_ws.py: 1 persistent CTAs, 2 one per item
KEY_WS = 3  # tf32: bit 1 the key scan without value credits (key_scan_ws.py), bit 2 lse / center (implicit_ws.py)
VALUE_WS = 1  # the value credits (_value_pass, head_dim 64) on value_credit_ws.py: 1 persistent CTAs, 2 one per item
# measured (bench.py, one H100, CUDA graph replay)
IMPLICIT_CONFIG = dict(BM=64, BN=64, num_warps=4, num_stages=2)
MOMENT_CONFIG = dict(PAIRS=16, BLOCK_T=32, BLOCK_S=64, num_warps=4, num_stages=3)
KEY_CONFIG = dict(BT=64, BS=32, num_warps=4, num_stages=3)
VALUE_CONFIG = {64: dict(BT=128, BS=128, num_warps=8, num_stages=3),  # by channels
                128: dict(BT=64, BS=64, num_warps=4, num_stages=3)}


# --- per-token operands ------------------------------------------------------------------------------------------

@triton.jit
def _pair_features(X, F2, F4, XY, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """The query moments' bf16 dot operands of each key of X (rows, T, D), from its pairs' scaled coordinates
    x = X[i] / sqrt(D), y = X[i + D / 2], pair-major: F2 = (x, y), F4 = (x, x^2, y, y^2) and XY = x y, as the
    reference kernel forms them in its loop (there once per program and key tile)."""
    J: tl.constexpr = D // 2
    row, tile = tl.program_id(0), tl.program_id(1)
    token = row.to(tl.int64) * T + tile * BLOCK + tl.arange(0, BLOCK)
    pairs = tl.arange(0, J)
    dims = tl.arange(0, D)
    x = tl.load(X + token[:, None] * D + pairs[None, :]) * D ** (-0.5)
    y = tl.load(X + token[:, None] * D + (pairs + J)[None, :]) * D ** (-0.5)
    tl.store(F2 + token[:, None] * D + dims[None, :], tl.reshape(tl.join(x, y), (BLOCK, D)).to(tl.bfloat16))
    tl.store(XY + token[:, None] * J + pairs[None, :], (x * y).to(tl.bfloat16))
    f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BLOCK, 2 * D))
    tl.store(F4 + token[:, None] * (2 * D) + tl.arange(0, 2 * D)[None, :], f4.to(tl.bfloat16))


def _pair_operands(x):
    rows, t, d = x.shape
    f2 = x.new_empty((rows, t, d), dtype=torch.bfloat16)
    f4 = x.new_empty((rows, t, 2 * d), dtype=torch.bfloat16)
    xy = x.new_empty((rows, t, d // 2), dtype=torch.bfloat16)
    _pair_features[rows, t // 64](x, f2, f4, xy, T=t, D=d, BLOCK=64, num_warps=4)
    return f2, f4, xy


@triton.jit
def _by_token(row, H: tl.constexpr, LAYER, TOKEN, C: tl.constexpr, Map=None, MAPPED: tl.constexpr = False):
    """Offset of row's token 0 when the rows (head h of layer l = row // H) are stored by token in a
    (layers, T, >= H * C) view with strides (LAYER, TOKEN, 1): H = 1, LAYER = T * C, TOKEN = C is the stacked
    (rows, T, C) layout. With MAPPED, layer l is stored at index Map[l] of the view's first axis."""
    layer = row // H
    if MAPPED:
        layer = tl.load(Map + layer)
    return layer.to(tl.int64) * LAYER + (row % H) * C


# --- lse and center ------------------------------------------------------------------------------------------------


@triton.jit
def _implicit_block(q, hub, K, Values, base, query, block, maximum, total, score_sum, Window: tl.constexpr,
                    BN: tl.constexpr, PREC: tl.constexpr, MASKED: tl.constexpr):
    """One key block of _implicit_key_statistics' online softmax; MASKED for blocks the causal window cuts."""
    dims = tl.arange(0, 64)
    source = block * BN + tl.arange(0, BN)
    k = tl.load(K + base + source[:, None] * 64 + dims[None, :])
    v = tl.load(Values + base + source[:, None] * 64 + dims[None, :])
    logits = tl.dot(q, tl.trans(k), input_precision=PREC) * 0.125
    scores = tl.dot(hub, tl.trans(v), input_precision=PREC)
    if MASKED:
        allowed = (source[None, :] <= query[:, None]) & (query[:, None] - source[None, :] <= Window)
        logits = tl.where(allowed, logits, float('-inf'))
    next_max = tl.maximum(maximum, tl.max(logits, 1))
    safe_max = next_max
    if MASKED:
        safe_max = tl.where(next_max == float('-inf'), 0.0, next_max)
    rescale = tl.exp(maximum - safe_max)
    prob = tl.exp(logits - safe_max[:, None])
    total = total * rescale + tl.sum(prob, 1)
    score_sum = score_sum * rescale + tl.sum(prob * scores, 1)
    return next_max, total, score_sum


@triton.jit
def _implicit_key_statistics_edges(Q, K, Values, Hub, LSE, Center, T: tl.constexpr, Window: tl.constexpr,
                                   B: tl.constexpr, PREC: tl.constexpr):
    """_implicit_key_statistics for square tiles (B queries, B keys; Window % B == 0): only the diagonal block and the
    window's first block are masked, the blocks between are whole; the heaviest query tiles launch first."""
    tile, head = T // B - 1 - tl.program_id(0), tl.program_id(1)
    query = tile * B + tl.arange(0, B)
    dims = tl.arange(0, 64)
    base = head * T * 64
    q = tl.load(Q + base + query[:, None] * 64 + dims[None, :])
    hub = tl.load(Hub + base + query[:, None] * 64 + dims[None, :])
    maximum = tl.full((B,), float('-inf'), tl.float32)
    total = tl.zeros((B,), tl.float32)
    score_sum = tl.zeros((B,), tl.float32)
    maximum, total, score_sum = _implicit_block(q, hub, K, Values, base, query, tile, maximum, total, score_sum,
                                                Window, B, PREC, True)
    start = tile * B - Window
    lo = 0
    if start >= 0:
        lo = start // B
        maximum, total, score_sum = _implicit_block(q, hub, K, Values, base, query, lo, maximum, total, score_sum,
                                                    Window, B, PREC, True)
        lo += 1
    for block in range(lo, tile):
        maximum, total, score_sum = _implicit_block(q, hub, K, Values, base, query, block, maximum, total, score_sum,
                                                    Window, B, PREC, False)
    tl.store(LSE + head * T + query, maximum + tl.log(total))
    tl.store(Center + head * T + query, score_sum / total)


def _implicit_key_context(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, hub: torch.Tensor,
                          window: int) -> tuple[torch.Tensor, torch.Tensor]:
    if KEY_WS & 2 and PREC == 'tf32':
        import implicit_ws
        if implicit_ws.supported(q, window):
            return implicit_ws.implicit_key_context_ws(q, k, values, hub, window)
    h, t, _ = q.shape
    q, k, values, hub = (z.contiguous() for z in (q, k, values, hub))
    lse = torch.empty((h, t), device=q.device)
    center = torch.empty_like(lse)
    c = IMPLICIT_CONFIG
    assert c['BM'] == c['BN'] and window % c['BN'] == 0 and t % c['BM'] == 0
    _implicit_key_statistics_edges[(t // c['BM'], h)](q, k, values, hub, lse, center, T=t, Window=window,
        B=c['BM'], PREC=PREC, num_warps=c['num_warps'], num_stages=c['num_stages'])
    return lse, center


@torch.library.custom_op('moment_kernels::implicit_key_context_edges', mutates_args=())
def implicit_key_context_edges(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, hub: torch.Tensor,
                               window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """implicit_key_context (same outputs up to summation order) masking only the key blocks the window cuts."""
    return _implicit_key_context(q, k, values, hub, window)


@implicit_key_context_edges.register_fake
def _(q, k, values, hub, window):
    h, t, _ = q.shape
    return q.new_empty((h, t)), q.new_empty((h, t))


# --- query moments -------------------------------------------------------------------------------------------------

@triton.jit
def _query_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, d, M, RAW_GPU, RAW_STRIDE,
    D: tl.constexpr):
    """The reference's _closed_coefficients for raw coordinates d."""
    i = d % (D // 2)
    n = (row % BH).to(tl.int64) * T + offs
    raw = tl.load(raw_ptr + (n // M * RAW_GPU + n % M * RAW_STRIDE)[:, None] + d[None, :]).to(tl.float32)
    sq = tl.load(sq_ptr + row * T + offs)[:, None]
    steps = tl.load(steps_ptr + row * T + offs)[:, None]
    change = (2 * steps * raw + steps * steps) / D / (sq / D + 1e-6)
    a = libdevice.expm1(-0.5 * libdevice.log1p(change))
    b = steps * libdevice.rsqrt(sq / D + 1e-6) * (1 + a)
    cos = tl.load(cos_ptr + offs[:, None] * (D // 2) + i[None, :])
    sin = tl.load(sin_ptr + offs[:, None] * (D // 2) + i[None, :])
    first = (d < D // 2)[None, :]
    return a, b * tl.where(first, cos, sin), b * tl.where(first, -sin, cos)


@triton.jit
def _query_store(out_ptr, x, row, BH, T, offs, d, M, W_STRIDE, D: tl.constexpr):
    n = (row % BH).to(tl.int64) * T + offs
    tl.store(out_ptr + (n // M * W_STRIDE + n % M * D)[:, None] + d[None, :], x)


@triton.jit
def _split4(acc, BLOCK: tl.constexpr, J: tl.constexpr):
    first, second = tl.split(tl.reshape(acc, (BLOCK, J, 2, 2)))
    x, y = tl.split(first)
    xx, yy = tl.split(second)
    return x, y, xx, yy


@triton.jit
def _query_response(n4, d4, nxy, dxy, l2, m2, s1n, s2n, s1d, s2d, raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr,
    out_ptr, row, BH, T, offs_s, d, M, RAW_GPU, RAW_STRIDE, W_STRIDE, D: tl.constexpr, J: tl.constexpr,
    BLOCK_S: tl.constexpr, OUT_H: tl.constexpr, OUT_LAYER, OUT_TOKEN, out_map, OUT_MAPPED: tl.constexpr):
    ca, bu, bw = _query_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_s, d, M,
      RAW_GPU, RAW_STRIDE, D)
    nx0, ny0, nx1, ny1 = _split4(n4, BLOCK_S, J)
    dx0, dy0, dx1, dy1 = _split4(d4, BLOCK_S, J)
    lx0, ly0 = tl.split(tl.reshape(l2, (BLOCK_S, J, 2)))
    mx0, my0 = tl.split(tl.reshape(m2, (BLOCK_S, J, 2)))
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
      + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
      + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_s)
    z = zn / (1.0 + zd) / steps[:, None]
    if OUT_H:  # fp32 by token, the bf16 output's values
        start = _by_token(row, OUT_H, OUT_LAYER, OUT_TOKEN, D, out_map, OUT_MAPPED)
        tl.store(out_ptr + start + offs_s[:, None] * OUT_TOKEN + d[None, :], z.to(tl.bfloat16).to(tl.float32))
    else:
        _query_store(out_ptr, z, row, BH, T, offs_s, d, M, W_STRIDE, D)


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, f2_ptr, f4_ptr, xy_ptr, cos_ptr, sin_ptr,
    lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, W_STRIDE, OUT_LAYER,
    OUT_TOKEN, out_map, D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    WINDOW: tl.constexpr, OUT_H: tl.constexpr, OUT_MAPPED: tl.constexpr):
    """The reference's _moment_queries_kernel with each program's J feature pairs (of D // 2) feeding both raw
    coordinates of each pair (d = pair and d = pair + D // 2)."""
    parts: tl.constexpr = D // 2 // J
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S
    row, part = i // parts, i % parts
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_s = s0 + tl.arange(0, BLOCK_S)
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
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    start = tl.maximum(s0 - WINDOW, 0) // BLOCK_T
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


def _moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window=None,
                         out_heads=0, layers=None):
    """out: the reference's (1, rows, T, D) bf16, or with out_heads a (rows // out_heads, T, out_heads * D) fp32 view
    (last stride 1)."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    c = MOMENT_CONFIG
    assert repeat == 4 and d == 64, 'the reference call (repeat = head_dim // 16)'
    parts = d // 2 // c['PAIRS']
    if ATTN_WS and out_heads:
        from moment_queries_ws import moment_query_scores_ws
        return moment_query_scores_ws(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
                                      t if window is None else min(window, t), out_heads, layers, ATTN_WS == 1)
    f2, f4, xy = _pair_operands(side.contiguous())
    m, out_blk, w_stride = (t, 0, 0) if out_heads else (out.shape[2], out.stride(1), out.stride(0))
    assert not out_heads or out.stride(2) == 1
    _moment_queries_kernel[(rows * parts, t // c['BLOCK_S'])](q, k, v, hub, raw, sq, f2, f4, xy, cos, sin, lse, hub_out,
      steps, out, rows // blocks, t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), m, out_blk, w_stride,
      out.stride(0), out.stride(1), out if layers is None else layers,
      D=d, J=c['PAIRS'], BLOCK_T=c['BLOCK_T'], BLOCK_S=c['BLOCK_S'], WINDOW=t if window is None else min(window, t),
      OUT_H=out_heads, OUT_MAPPED=layers is not None, num_warps=c['num_warps'], num_stages=c['num_stages'])


@torch.library.custom_op('moment_kernels::moment_query_scores', mutates_args=('out',))
def moment_query_scores(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hub: torch.Tensor, raw: torch.Tensor,
                        sq: torch.Tensor, side: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lse: torch.Tensor,
                        hub_out: torch.Tensor, steps: torch.Tensor, out: torch.Tensor, repeat: int,
                        window: int | None = None) -> None:
    _moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window)


@moment_query_scores.register_fake
def _(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window=None):
    return None


@torch.library.custom_op('moment_kernels::moment_query_scores_into', mutates_args=('out',))
def moment_query_scores_into(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hub: torch.Tensor, raw: torch.Tensor,
                             sq: torch.Tensor, side: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                             lse: torch.Tensor, hub_out: torch.Tensor, steps: torch.Tensor, out: torch.Tensor,
                             window: int | None = None, layers: torch.Tensor | None = None) -> None:
    """moment_query_scores_by_token into out, a (layers, T, heads * D) fp32 view with last stride 1 (e.g. a column
    slice of a wider per-layer buffer); heads = out.shape[-1] // D; layers (int32, optional): the out index of each
    of the rows' layers."""
    _moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, q.shape[-1] // 16, window,
                         out.shape[-1] // q.shape[-1], layers)


@moment_query_scores_into.register_fake
def _(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, window=None, layers=None):
    return None


# --- identity-phase key moments -----------------------------------------------------------------------------------

@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
    steps_ptr, out_ptr, value_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    """run.py's _moment_keys_kernel (repeat = 2) with both of a token tile's programs in one: the reference's two
    programs formed the same moment sums (pairs i % 32) and differed only in the output coordinates (d = pair,
    pair + 32) and the 32 hub columns of the value sum, so the sums are formed once and the value sum over all 64."""
    J: tl.constexpr = D // 2
    row, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    value_ptr += blk.to(tl.int64) * OUT_BLK
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    side = side_ptr + base
    p2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    q2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    qxy = tl.zeros((BLOCK_T, J), tl.float32)
    r2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    val = tl.zeros((BLOCK_T, D), tl.float32)
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    for it in range(t0 // BLOCK_S, T // BLOCK_S):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where(lag >= 0, tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        x = tl.load(side + offs_s[:, None] * D + offs_j[None, :]) * scale
        y = tl.load(side + offs_s[:, None] * D + (offs_j + J)[None, :]) * scale
        w1 = a * c
        w2 = (0.5 * a - a * a) * c
        w1b, w2b, w2lb = w1.to(tl.bfloat16), w2.to(tl.bfloat16), (w2 * logits).to(tl.bfloat16)
        f2 = tl.reshape(tl.join(x, y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        g2 = tl.reshape(tl.join(x * x, y * y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        val += tl.dot(a.to(tl.bfloat16), hub)
        p2 += tl.dot(w1b, f2)
        q2 += tl.dot(w2b, g2)
        qxy += tl.dot(w2b, (x * y).to(tl.bfloat16))
        r2 += tl.dot(w2lb, f2)
        s1 += tl.sum(w1 * logits, axis=1)
        s2 += tl.sum(w2 * logits * logits, axis=1)
    px, py = tl.split(tl.reshape(p2, (BLOCK_T, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BLOCK_T, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BLOCK_T, J, 2)))
    val0, val1 = tl.split(tl.permute(tl.reshape(val, (BLOCK_T, 2, J)), (0, 2, 1)))
    steps = tl.load(steps_ptr + row * T + offs_t)
    for half in tl.static_range(2):
        d = offs_j + half * J
        ca, bu, bw = _query_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_t, d, M,
          RAW_GPU, RAW_STRIDE, D)
        z = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + qxy * (2.0 * bu * bw) + 2.0 * ca * (rx * bu + ry * bw)
          + s1[:, None] * ca + s2[:, None] * (ca * ca))
        _query_store(out_ptr, z / steps[:, None], row, BH, T, offs_t, d, M, W_STRIDE, D)
        _query_store(value_ptr, val0 if half == 0 else val1, row, BH, T, offs_t, d, M, W_STRIDE, D)


def _moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, repeat):
    rows, t, d = q.shape
    blocks = raw.shape[1]
    assert repeat == 2 and d == 64, 'the identity phase\'s call (repeat = head_dim // 32)'
    assert values.stride() == out.stride() and values.shape == out.shape
    _moment_keys_kernel[(rows, t // 64)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values,
      rows // blocks, t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2], out.stride(1),
      out.stride(0), D=d, BLOCK_T=64, BLOCK_S=32, num_warps=4, num_stages=3)


@torch.library.custom_op('moment_kernels::moment_key_scores', mutates_args=('out', 'values'))
def moment_key_scores(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hub: torch.Tensor, raw: torch.Tensor,
                      sq: torch.Tensor, side: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lse: torch.Tensor,
                      hub_out: torch.Tensor, steps: torch.Tensor, out: torch.Tensor, values: torch.Tensor,
                      repeat: int) -> None:
    """Drop-in for run.py's moment_key_scores (the identity phase's key credits and value sums)."""
    _moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, repeat)


@moment_key_scores.register_fake
def _(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, repeat):
    return None


# --- key credits (and the raw value credit) -----------------------------------------------------------------------

@triton.jit
def _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D: tl.constexpr, Sigma: tl.constexpr):
    raw = tl.load(Raw + base + target[:, None] * D + dims[None, :])
    raw_j = tl.load(Raw + base + target[:, None] * D + dirs[None, :])
    sq = tl.sum(raw * raw, 1)[:, None] / D
    step = Sigma * tl.maximum(tl.sqrt(sq), 1e-06)
    ratio = (2 * step * raw_j + step * step) / D / (sq + 1.1920928955078125e-07)
    ca = libdevice.expm1(-0.5 * libdevice.log1p(ratio))
    cb = step * libdevice.rsqrt(sq + 1.1920928955078125e-07) * (1 + ca)
    cs = tl.load(Cos + target[:, None] * (D // 2) + pair[None, :])
    sn = tl.load(Sin + target[:, None] * (D // 2) + pair[None, :])
    bu = cb * tl.where(dirs[None, :] < D // 2, cs, sn)
    bw = cb * tl.where(dirs[None, :] < D // 2, -sn, cs)
    return (ca, bu, bw, step)


@triton.jit
def _key_scan_kernel(Q, K, Raw, LSE, Values, Hub, Center, Cos, Sin, Out, RawValue, OUT_LAYER, OUT_TOKEN, RAW_LAYER,
    RAW_TOKEN, OutMap, T: tl.constexpr, D: tl.constexpr,
    Window: tl.constexpr, Sigma: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, PREC: tl.constexpr,
    VALUES: tl.constexpr, OUT_H: tl.constexpr, OUT_MAPPED: tl.constexpr):
    """The reference's _key_scan_kernel (J = D // 2); with VALUES also the raw value credit sum_s p * hub, as the
    reference's _value_credit_kernel forms it (bf16 probabilities times hub's bf16 high and low parts)."""
    J: tl.constexpr = D // 2
    head = tl.program_id(0)
    t0 = tl.program_id(1) * BT
    target = t0 + tl.arange(0, BT)
    dims = tl.arange(0, D)
    dirs = tl.arange(0, D)
    feature_pair = tl.arange(0, J)
    pair = dirs % (D // 2)
    base = head * T * D
    key = tl.load(K + base + target[:, None] * D + dims[None, :])
    value = tl.load(Values + base + target[:, None] * D + dims[None, :])
    p2 = tl.zeros((BT, 2 * J), tl.float32)
    q2 = tl.zeros((BT, 2 * J), tl.float32)
    qxy = tl.zeros((BT, J), tl.float32)
    r2 = tl.zeros((BT, 2 * J), tl.float32)
    acc = tl.zeros((BT, D), tl.float32)
    s1 = tl.zeros((BT,), tl.float32)
    s2 = tl.zeros((BT,), tl.float32)
    first = t0 // BS
    last = tl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    for it in range(first, last):
        source = it * BS + tl.arange(0, BS)
        query = tl.load(Q + base + source[:, None] * D + dims[None, :])
        logit = tl.dot(key, tl.trans(query), input_precision=PREC) * D ** (-0.5)
        valid = (source[None, :] >= target[:, None]) & (source[None, :] - target[:, None] <= Window)
        hub = tl.load(Hub + base + source[:, None] * D + dims[None, :])
        score = tl.dot(value, tl.trans(hub), input_precision=PREC)
        center = tl.load(Center + head * T + source)
        c = tl.where(valid, score - center[None, :], 0.0)
        lse = tl.load(LSE + head * T + source)
        a = tl.where(valid, tl.exp(logit - lse[None, :]), 0.0)
        x = tl.load(Q + base + source[:, None] * D + feature_pair[None, :]) * D ** (-0.5)
        y = tl.load(Q + base + source[:, None] * D + (feature_pair + D // 2)[None, :]) * D ** (-0.5)
        f2 = tl.reshape(tl.join(x, y), (BS, 2 * J))
        g2 = tl.reshape(tl.join(x * x, y * y), (BS, 2 * J))
        xy = x * y
        w1 = a * c
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logit
        w1d, w2d, w2ld = (w1.to(tl.bfloat16), w2.to(tl.bfloat16), w2l.to(tl.bfloat16))
        f2d, g2d, xyd = (f2.to(tl.bfloat16), g2.to(tl.bfloat16), xy.to(tl.bfloat16))
        p2 += tl.dot(w1d, f2d)
        q2 += tl.dot(w2d, g2d)
        qxy += tl.dot(w2d, xyd)
        r2 += tl.dot(w2ld, f2d)
        s1 += tl.sum(w1 * logit, 1)
        s2 += tl.sum(w2 * logit * logit, 1)
        if VALUES:
            prob = a.to(tl.bfloat16)
            hi = hub.to(tl.bfloat16)
            lo = (hub - hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(prob, hi)
            acc += tl.dot(prob, lo)
    ca, bu, bw, step = _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D, Sigma)
    px, py = tl.split(tl.reshape(p2, (BT, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BT, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BT, J, 2)))
    gather_pair = tl.broadcast_to(pair[None, :], (BT, D))
    px, py = (tl.gather(px, gather_pair, 1), tl.gather(py, gather_pair, 1))
    qx, qy = (tl.gather(qx, gather_pair, 1), tl.gather(qy, gather_pair, 1))
    rx, ry = (tl.gather(rx, gather_pair, 1), tl.gather(ry, gather_pair, 1))
    qxy = tl.gather(qxy, gather_pair, 1)
    response = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + 2 * qxy * bu * bw + 2 * ca * (rx * bu + ry * bw)
      + s1[:, None] * ca + s2[:, None] * ca * ca)
    start = _by_token(head, OUT_H, OUT_LAYER, OUT_TOKEN, D, OutMap, OUT_MAPPED)
    tl.store(Out + start + target[:, None] * OUT_TOKEN + dirs[None, :], response / step)
    if VALUES:
        start = _by_token(head, OUT_H, RAW_LAYER, RAW_TOKEN, D, OutMap, OUT_MAPPED)
        tl.store(RawValue + start + target[:, None] * RAW_TOKEN + dims[None, :], acc)


def _key_scan(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, with_values, out_heads=1, key=None,
              layers=None):
    """The key credits (and with_values the raw value credits), stacked (rows, T, D) or by token
    (rows // out_heads, T, out_heads * D); the key credits into key (such a view, last stride 1) when given."""
    h, t, d = k_raw.shape
    shape = (h // out_heads, t, out_heads * d)
    key = k_raw.new_empty(shape) if key is None else key
    assert key.shape[1:] == shape[1:] and key.stride(2) == 1 and (layers is not None or key.shape == shape)
    raw_value = k_raw.new_empty(shape) if with_values else key
    c = KEY_CONFIG
    if KEY_WS & 1 and not with_values and PREC == 'tf32':
        import key_scan_ws
        if key_scan_ws.supported(qp, kp, k_raw, lse, values, hub, center, cos, sin):
            key_scan_ws.key_scan_ws(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, key, out_heads,
                                    layers)
            key = key.view(h, t, d) if out_heads == 1 else key
            return key, key
    _key_scan_kernel[h, t // c['BT']](qp, kp, k_raw, lse, values, hub, center, cos, sin, key, raw_value,
        key.stride(0), key.stride(1), raw_value.stride(0), raw_value.stride(1), key if layers is None else layers,
        T=t, D=d, OUT_MAPPED=layers is not None,
        Window=window, Sigma=sigma, BT=c['BT'], BS=c['BS'], PREC=PREC, VALUES=with_values, OUT_H=out_heads,
        num_warps=c['num_warps'], num_stages=c['num_stages'])
    return key.view(h, t, d) if out_heads == 1 else key, raw_value.view(h, t, d) if out_heads == 1 else raw_value


def _by_token_shape(x, heads):
    rows, t, d = x.shape
    return (rows // heads, t, heads * d)


@torch.library.custom_op('moment_kernels::key_credits_into', mutates_args=('out',))
def key_credits_into(qp: torch.Tensor, kp: torch.Tensor, k_raw: torch.Tensor, lse: torch.Tensor, values: torch.Tensor,
                     hub: torch.Tensor, center: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int,
                     sigma: float, out: torch.Tensor, layers: torch.Tensor | None = None) -> None:
    """key_credits_by_token into out, a (layers, T, heads * D) view with last stride 1; layers (int32, optional): the
    out index of each of the rows' layers."""
    _key_scan(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, False, out.shape[-1] // k_raw.shape[-1],
              out, layers)


@key_credits_into.register_fake
def _(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, out, layers=None):
    return None


# --- value credits -------------------------------------------------------------------------------------------------

@triton.jit
def _value_credit_kernel(Q, K, LSE, Hub, Out, T: tl.constexpr, Window: tl.constexpr, BT: tl.constexpr,
    BS: tl.constexpr, Channels: tl.constexpr, PREC: tl.constexpr, IN_H: tl.constexpr, OUT_H: tl.constexpr,
    IN_LAYER, IN_TOKEN, OUT_LAYER, OUT_TOKEN, OutMap, OUT_MAPPED: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    target = tile * BT + tl.arange(0, BT)
    dims = tl.arange(0, 64)
    channels = tl.arange(0, Channels)
    base = head * T * 64
    key = tl.load(K + base + target[:, None] * 64 + dims[None, :])
    acc = tl.zeros((BT, Channels), tl.float32)
    hub_start = _by_token(head, IN_H, IN_LAYER, IN_TOKEN, Channels)
    first = tile * BT // BS
    last = tl.minimum((tile * BT + BT - 1 + Window) // BS + 1, T // BS)
    for it in range(first, last):
        receiver = it * BS + tl.arange(0, BS)
        query = tl.load(Q + base + receiver[:, None] * 64 + dims[None, :])
        hub = tl.load(Hub + hub_start + receiver[:, None] * IN_TOKEN + channels[None, :])
        lse = tl.load(LSE + head * T + receiver)
        logits = tl.dot(key, tl.trans(query), input_precision=PREC) * .125
        valid = (receiver[None, :] >= target[:, None]) & (receiver[None, :] - target[:, None] <= Window)
        prob = tl.where(valid, tl.exp(logits - lse[None, :]), 0.).to(tl.bfloat16)
        hi = hub.to(tl.bfloat16)
        lo = (hub - hi.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(prob, hi)
        acc += tl.dot(prob, lo)
    out_start = _by_token(head, OUT_H, OUT_LAYER, OUT_TOKEN, Channels, OutMap, OUT_MAPPED)
    tl.store(Out + out_start + target[:, None] * OUT_TOKEN + channels[None, :], acc)


def _value_pass(q, k, lse, source, window, in_heads=1, out_heads=1, out=None, layers=None):
    """sum_s p * source: source stacked (rows, T, C) or (in_heads > 1) by token (rows // in_heads, T, in_heads * C);
    the output stacked or (out_heads > 1) by token likewise, into out (such a view, last stride 1) when given."""
    h, t, _ = q.shape
    channels = source.shape[-1] // in_heads
    out = source.new_empty((h // out_heads, t, out_heads * channels)) if out is None else out
    assert source.stride(2) == 1 and out.stride(2) == 1
    if VALUE_WS and q.shape[-1] == channels == 64 and t % 128 == 0 and all(
            z.is_contiguous() for z in (q, k, lse, source)):
        from value_credit_ws import value_pass_ws
        return value_pass_ws(q, k, lse, source, window, in_heads, out_heads, out, layers, VALUE_WS == 1)
    c = VALUE_CONFIG[channels]
    _value_credit_kernel[h, t // c['BT']](q, k, lse, source, out, T=t, Window=window, BT=c['BT'], BS=c['BS'],
        Channels=channels, PREC=PREC, IN_H=in_heads, OUT_H=out_heads, IN_LAYER=source.stride(0),
        IN_TOKEN=source.stride(1), OUT_LAYER=out.stride(0), OUT_TOKEN=out.stride(1),
        OutMap=out if layers is None else layers, OUT_MAPPED=layers is not None, num_warps=c['num_warps'],
        num_stages=c['num_stages'])
    return out


@torch.library.custom_op('moment_kernels::raw_value_credits_by_token', mutates_args=())
def raw_value_credits_by_token(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor, hub: torch.Tensor, window: int,
                               heads: int) -> torch.Tensor:
    """raw_value_credits as (layers, T, heads * 64) (the rows are heads of rows // heads layers): out[j] is layer
    j's raw value credit as raw[rows of j].transpose(0, 1).reshape(T, -1)."""
    return _value_pass(q, k, lse, hub, window, 1, heads)


@raw_value_credits_by_token.register_fake
def _(q, k, lse, hub, window, heads):
    return hub.new_empty(_by_token_shape(hub, heads))


@torch.library.custom_op('moment_kernels::terminal_value_credits_into', mutates_args=('out',))
def terminal_value_credits_into(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor, corrected: torch.Tensor,
                                window: int, out: torch.Tensor, layers: torch.Tensor | None = None) -> None:
    """terminal_value_credits_by_token into out, a (layers, T, heads * 64) view with last stride 1; corrected may be
    such a view too (indexed by the rows' layers); layers (int32, optional): the out index of each of them."""
    heads = corrected.shape[-1] // q.shape[-1]
    _value_pass(q, k, lse, corrected, window, heads, heads, out, layers)


@terminal_value_credits_into.register_fake
def _(q, k, lse, corrected, window, out, layers=None):
    return None


