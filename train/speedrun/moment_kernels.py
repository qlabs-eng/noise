"""Softmax moment responses for the Identity phase and the sampled Cut queries."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

PREC = 'tf32'  # the lse / center pass's logits and scores (query_context)
IMPLICIT_CONFIG = dict(BM=64, BN=64, num_warps=4, num_stages=2)
MOMENT_CONFIG = dict(PAIRS=16, BLOCK_T=32, BLOCK_S=64, num_warps=4, num_stages=3)


# --- by-token output offsets -------------------------------------------------------------------------------------

@triton.jit
def _by_token(row, H: tl.constexpr, LAYER, C: tl.constexpr, Map):
    """Offset of the first token for this head of the mapped Cut layer."""
    layer = tl.load(Map + row // H)
    return layer.to(tl.int64) * LAYER + (row % H) * C


# --- query moments -------------------------------------------------------------------------------------------------

@triton.jit
def _query_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, d, M, RAW_GPU, RAW_STRIDE,
    D: tl.constexpr):
    """The moment response's closed-form coefficients for raw coordinates d."""
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
    row, BH, T, offs_s, d, M, RAW_GPU, RAW_STRIDE, D: tl.constexpr, J: tl.constexpr, BLOCK_S: tl.constexpr):
    """The degree-2 softmax moment response to a finite step of each raw (pre-RoPE) query coordinate d through RoPE
    and the query norm. The step moves both rotated coordinates of d's pair (bu and bw), so the model carries their
    mixed term (nxy, dxy)."""
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
    return zn / (1.0 + zd) / steps[:, None]


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, f2_ptr, f4_ptr, xy_ptr, cos_ptr, sin_ptr,
    lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    WINDOW: tl.constexpr):
    """The query moments: each program's J feature pairs (of D // 2) feed both raw coordinates of each pair
    (d = pair and d = pair + D // 2)."""
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
        d = pairs + half * (D // 2)
        z = _query_response(n4, d4, nxy, dxy, l2, m2, s1n, s2n, s1d, s2d, raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr,
          row, BH, T, offs_s, d, M, RAW_GPU, RAW_STRIDE, D, J, BLOCK_S)
        _query_store(out_ptr, z, row, BH, T, offs_s, d, M, W_STRIDE, D)


def _moment_query_scores(q, k, v, hub, raw, sq, pair_ops, cos, sin, lse, hub_out, steps, out):
    """Write the Identity phase's query response using the prepared key-pair features."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    c = MOMENT_CONFIG
    parts = d // 2 // c['PAIRS']
    f2, f4, xy = pair_ops
    _moment_queries_kernel[(rows * parts, t // c['BLOCK_S'])](q, k, v, hub, raw, sq, f2, f4, xy, cos, sin, lse, hub_out,
      steps, out, rows // blocks, t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
      out.stride(1), out.stride(0), D=d, J=c['PAIRS'], BLOCK_T=c['BLOCK_T'], BLOCK_S=c['BLOCK_S'], WINDOW=t,
      num_warps=c['num_warps'], num_stages=c['num_stages'])


# --- identity-phase key moments -----------------------------------------------------------------------------------

@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
    steps_ptr, out_ptr, value_ptr, BH, T, scale, RAW_BLK, RAW_GPU, RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, SHORT_WINDOW: tl.constexpr,
    Gates, Destinations, Lexical):
    """The key moments, one program per token tile: both output coordinates of a pair (d = pair,
    pair + 32) come from the same moment sums (pairs i % 32), formed once; the value sum runs over all 64 hub
    columns."""
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
    window = tl.where(blk % 4 == 3, T, SHORT_WINDOW)
    last = tl.cdiv(tl.minimum(T, t0 + BLOCK_T + window), BLOCK_S)
    for it in range(t0 // BLOCK_S, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
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
    if blk % 2 == 1:
        n = (row % BH) * T + offs_t
        pos = n % M
        head = ((row % BH) * T) // M
        table = blk // 2
        z = tl.load(Gates + (table * M + pos) * 8 + head).to(tl.float32)
        # val rounded to bf16, the precision of the value sums this kernel stores.
        gated = val.to(tl.bfloat16).to(tl.float32) * 2.0 * tl.sigmoid(z)[:, None]
        dst = tl.load(Destinations + pos).to(tl.int32)
        addr = Lexical + (((dst * 4 + table) * 8 + head) * D)[:, None] + offs_d[None, :]
        tl.store(addr, gated)
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


KEY_MOMENT_CONFIG = dict(BLOCK_T=64, BLOCK_S=32, num_warps=4, num_stages=3)  # the identity phase's key moments


def moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, short_window, lexical):
    """The identity phase's key credits and value sums."""
    c = KEY_MOMENT_CONFIG
    rows, t, d = q.shape
    blocks = raw.shape[1]
    gates, destinations, lexical_out = lexical
    _moment_keys_kernel[(rows, t // c['BLOCK_T'])](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
      values, rows // blocks, t, d ** -0.5, raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2], out.stride(1),
      out.stride(0), D=d, BLOCK_T=c['BLOCK_T'], BLOCK_S=c['BLOCK_S'], SHORT_WINDOW=short_window,
      Gates=gates, Destinations=destinations, Lexical=lexical_out, num_warps=c['num_warps'],
      num_stages=c['num_stages'])
