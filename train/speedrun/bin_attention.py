"""The attention replay's preparation (strong keys, their probabilities and value tables) for selected query rows."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_stage(keys, vals, valid, lane, SIZE: tl.constexpr, STRIDE: tl.constexpr, FINAL: tl.constexpr):
    """One compare-exchange stage of ATen's small bitonic sort (SortUtils.cuh bitonicSort / bitonicSwap, descending:
    a pair swaps iff ((kA > kB and validA) or not validB) == dir; invalid entries sort to the end)."""
    is_a = (lane & STRIDE) == 0
    a = tl.where(is_a, lane, lane - STRIDE)
    b = a + STRIDE
    t = (a & (STRIDE - 1)) | ((a >> 1) & ~(STRIDE - 1))  # the thread that owns the pair
    if FINAL:
        flag = lane < 0
    else:
        flag = (t & (SIZE // 2)) != 0
    ka, kb = tl.gather(keys, a, 0), tl.gather(keys, b, 0)
    va, vb = tl.gather(vals, a, 0), tl.gather(vals, b, 0)
    oka, okb = tl.gather(valid, a, 0), tl.gather(valid, b, 0)
    swap = (((ka > kb) & (oka != 0)) | (okb == 0)) == flag
    keys = tl.where(is_a, tl.where(swap, kb, ka), tl.where(swap, ka, kb))
    vals = tl.where(is_a, tl.where(swap, vb, va), tl.where(swap, va, vb))
    valid = tl.where(is_a, tl.where(swap, okb, oka), tl.where(swap, oka, okb))
    return keys, vals, valid


@triton.jit
def _moments(Scores, Keys, Values, Packed, Pos, Maxima, Sums, OutV, OutK,
             N: tl.constexpr, T: tl.constexpr, WINDOW: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    tile, h = tl.program_id(0), tl.program_id(1)
    rr = tile * BM + tl.arange(0, BM)
    pos = tl.load(Pos + rr, rr < N, other=0).to(tl.int32)
    ds = tl.arange(0, 64)
    ns = tl.arange(0, BN)
    # Exact dense32-row scan envelope for each original query.
    lo = tl.maximum((pos // 32 * 32 - WINDOW) // BN, 0)
    hi = tl.cdiv((pos // 32 + 1) * 32, BN)
    first = tl.min(tl.where(rr < N, lo, T // BN), 0)
    last = tl.max(tl.where(rr < N, hi, 0), 0)
    mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    winners = tl.full((BM, K), -2147483648, tl.int32)
    runners = tl.full((BM, K), -2147483648, tl.int32)
    for block in range(first, last):
        kk = block * BN + ns
        active = (rr < N) & (block >= lo) & (block < hi)
        lag = pos[:, None] - kk[None, :]
        legal = (lag >= 0) & (lag <= WINDOW) & (rr[:, None] < N)
        raw = tl.load(Scores + (h * N + rr[:, None]) * T + kk[None, :], legal, other=-float('inf'))
        sc = raw.to(tl.float32) * 0.125
        bits = raw.to(tl.int16, bitcast=True).to(tl.int32)
        ordered = tl.where(bits < 0, bits ^ 0x7fff, bits)
        pad = (pos[:, None] < K - 1) & (kk[None, :] < K)
        packed = tl.where(legal | pad, (ordered << 11) | (T - 1 - kk[None, :]), -2147483648)
        grouped = tl.reshape(packed, (BM, BN // K, K))
        best = tl.max(grouped, 1)
        second = tl.max(tl.where(grouped == best[:, None, :], -2147483648, grouped), 1)
        runners = tl.maximum(tl.maximum(runners, second), tl.minimum(winners, best))
        winners = tl.maximum(winners, best)
        new = tl.maximum(mx, tl.max(sc, 1))
        # Inactive leading/trailing blocks never alter the dense recurrence.
        safe = tl.where(active, new, 0.)
        den = tl.where(active, den * tl.exp(mx - safe) + tl.sum(tl.exp(sc - safe[:, None]), 1), den)
        mx = tl.where(active, new, mx)
    av = tl.zeros((BM, 64), tl.float32)
    ak = tl.zeros((BM, 64), tl.float32)
    for block in range(first, last):
        kk = block * BN + ns
        active = (rr < N) & (block >= lo) & (block < hi)
        lag = pos[:, None] - kk[None, :]
        legal = (lag >= 0) & (lag <= WINDOW) & (rr[:, None] < N)
        raw = tl.load(Scores + (h * N + rr[:, None]) * T + kk[None, :], legal, other=-float('inf'))
        sc = raw.to(tl.float32) * 0.125
        pp = tl.where(active[:, None], tl.exp(sc - mx[:, None]) / den[:, None], 0.).to(tl.bfloat16)
        key = tl.load(Keys + (h * T + kk[:, None]) * 64 + ds[None, :])
        val = tl.load(Values + (h * T + kk[:, None]) * 64 + ds[None, :])
        av += tl.dot(pp, val)
        ak += tl.dot(pp, key)
    off = (h * N + rr[:, None]) * K + tl.arange(0, K)[None, :]
    pool = tl.reshape(tl.join(winners, runners), (BM, 2 * K))
    selected = tl.topk(pool, K)
    tl.store(Packed + off, selected, rr[:, None] < N)
    tl.store(Maxima + h * N + rr, mx, rr < N)
    tl.store(Sums + h * N + rr, den, rr < N)
    dst = (h * N + rr[:, None]) * 64 + ds[None, :]
    tl.store(OutV + dst, av.to(tl.bfloat16).to(tl.float32), rr[:, None] < N)
    tl.store(OutK + dst, ak.to(tl.bfloat16).to(tl.float32), rr[:, None] < N)

@triton.jit
def _finish(Packed, Maxima, Sums, Probability, Indices, N: tl.constexpr, K: tl.constexpr, LOG_N: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    top = tl.load(Packed + row * K + lane, lane < K, other=-2147483648)
    o = top >> LOG_N
    col = N - 1 - (top & (N - 1))
    kth = tl.sum(tl.where(lane == K - 1, o, 0), 0)
    tie = (o == kth).to(tl.int32)
    # gather order: (tie, index) ascending; invalid slots last
    order = tl.where(lane < K, (tie << 28) | (col << 16) | (o + 32768), 1 << 30)
    order = tl.sort(order)
    col = (order >> 16) & 0xFFF
    o = (order & 0xFFFF) - 32768
    keys = tl.where(o < 0, o ^ 0x7FFF, o).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    valid = (lane < K).to(tl.int32)
    vals = col
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 2, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 4, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 4, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 4, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 8, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 4, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 16, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 8, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 4, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 2, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 1, True)
    out = lane < K
    tl.store(Indices + row * K + lane, vals.to(tl.int64), out)
    peak = tl.load(Maxima + row)
    den = tl.load(Sums + row)
    tl.store(Probability + row * K + lane, tl.exp(keys * 0.125 - peak) / den, out)


@triton.jit
def _table(V, W, Table, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, O: tl.constexpr,
           TB: tl.constexpr, BO: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    rr = b * TB + tl.arange(0, TB)
    dd = tl.arange(0, D)
    v = tl.load(V + (h * T + rr[:, None]) * D + dd[None, :])
    for ob in tl.range(0, O // BO, num_stages=3):
        oo = ob * BO + tl.arange(0, BO)
        w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
        tl.store(Table + (h * T + rr[:, None]) * O + oo[None, :], tl.dot(v, w.to(tl.bfloat16)).to(tl.bfloat16))

@triton.jit
def _query_tables(Origin, W, Idx, Corr, Rows, Keys, Prob, AllKeys, Shifted,
                  N: tl.constexpr, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                  D: tl.constexpr, O: tl.constexpr, TB: tl.constexpr, BO: tl.constexpr, TS: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    rr = b * TB + tl.arange(0, TB)
    dd = tl.arange(0, D)
    origin = tl.load(Origin + (h * N + rr[:, None]) * D + dd[None, :], rr[:, None] < N, other=0.)
    for ob in tl.range(0, O // BO, num_stages=3):
        oo = ob * BO + tl.arange(0, BO)
        w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
        neg = -tl.dot(origin, w, input_precision='tf32')
        high = neg.to(tl.bfloat16)
        corr = Corr + (rr[:, None] * (2 * H) + h) * O + oo[None, :]
        tl.store(corr, high, rr[:, None] < N)
        tl.store(corr + H * O, (neg - high.to(tl.float32)).to(tl.bfloat16), rr[:, None] < N)
    kk = tl.arange(0, K)
    for sb in tl.range(0, TB // TS, num_stages=3):
        ts = b * TB + sb * TS + tl.arange(0, TS)
        hk = (h * N + ts[:, None]) * K + kk[None, :]
        idx = tl.load(Idx + hk, ts[:, None] < N, other=0)
        tl.store(Rows + ts[:, None] * (H * K) + h * K + kk[None, :], (idx + h * T).to(tl.int32), ts[:, None] < N)
        top = tl.load(Keys + (h * T + idx)[:, :, None] * D + dd[None, None, :]).to(tl.float32)
        p = tl.load(Prob + hk, ts[:, None] < N, other=0.)
        tail = tl.load(AllKeys + (h * N + ts[:, None]) * D + dd[None, :], ts[:, None] < N, other=0.) - tl.sum(p[:, :,
            None] * top, 1)
        mass = tl.maximum(1 - tl.sum(p, 1), 0)[:, None]
        centroid = tl.where(mass > 1e-5, tail / tl.maximum(mass, 1e-5), 0)
        tl.store(Shifted + ((ts[:, None, None] * H + h) * K + kk[None, :, None]) * D + dd[None, None, :],
                 (top - centroid[:, None, :]).to(tl.bfloat16), ts[:, None, None] < N)


def moments(q, keys, values, positions, window, count=16):
    h, t, d = keys.shape
    n = q.shape[0]
    scores = torch.bmm(q.to(keys.dtype).transpose(0, 1), keys.transpose(1, 2))
    idx = torch.empty((h, n, count), device=q.device, dtype=torch.int64)
    prob = torch.empty((h, n, count), device=q.device)
    ov = torch.empty((h, n, d), device=q.device)
    ok = torch.empty_like(ov)
    packed = torch.empty((h, n, count), device=q.device, dtype=torch.int32)
    maxima = torch.empty((h, n), device=q.device)
    sums = torch.empty_like(maxima)
    _moments[(triton.cdiv(n, 32), h)](scores, keys, values, packed, positions, maxima, sums, ov, ok,
            n, t, window, count, 32, 128, num_warps=4, num_stages=2)
    _finish[(h * n,)](packed, maxima, sums, prob, idx, t, count, t.bit_length() - 1, num_warps=1)
    return prob, idx, ov, ok


def tables(values, origin, indices, w_proj_t, keys, prob, all_keys):
    h, t, d = values.shape
    n, k = indices.shape[1:]
    o = w_proj_t.shape[1]
    table = torch.empty((h * t, o), device=values.device, dtype=torch.bfloat16)
    corr = torch.empty((n, 2 * h, o), device=values.device, dtype=torch.bfloat16)
    rows = torch.empty((n, h * k), device=values.device, dtype=torch.int32)
    shifted = torch.empty((n, h, k, d), device=values.device, dtype=keys.dtype)
    _table[(h, t // 32)](values, w_proj_t, table, t, h, d, o, 32, 128, num_warps=4)
    _query_tables[(h, triton.cdiv(n, 32))](origin, w_proj_t, indices, corr, rows, keys, prob, all_keys, shifted,
            n, t, h, k, d, o, 32, 128, 16, num_warps=4)
    return shifted, (table, corr, rows)
