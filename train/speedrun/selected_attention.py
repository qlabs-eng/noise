"""The attention replay's preparation (strong keys, their probabilities and value tables) for selected query rows."""
import torch
import triton
import triton.language as tl
import window_topk

@triton.jit
def _mask(Scores, Pos, N: tl.constexpr, T: tl.constexpr, WINDOW: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr):
    h, m, n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rr = m * BM + tl.arange(0, BM)
    kk = n * BN + tl.arange(0, BN)
    pos = tl.load(Pos + rr, rr < N, other=0)
    lag = pos[:, None] - kk[None, :]
    tl.store(Scores + (h * N + rr[:, None]) * T + kk[None, :], float('-inf'),
             (rr[:, None] < N) & ((lag < 0) | (lag > WINDOW)))

@triton.jit
def _moments(Scores, Keys, Values, Top, Pos, Probability, OutV, OutK,
             N: tl.constexpr, T: tl.constexpr, WINDOW: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    tile, h = tl.program_id(0), tl.program_id(1)
    rr = tile * BM + tl.arange(0, BM)
    pos = tl.load(Pos + rr, rr < N, other=0)
    ds = tl.arange(0, 64)
    ns = tl.arange(0, BN)
    # Exact dense32-row scan envelope for each original query.
    lo = tl.maximum((pos // 32 * 32 - WINDOW) // BN, 0)
    hi = tl.cdiv((pos // 32 + 1) * 32, BN)
    first = tl.min(tl.where(rr < N, lo, T // BN), 0)
    last = tl.max(tl.where(rr < N, hi, 0), 0)
    mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    for block in range(first, last):
        kk = block * BN + ns
        active = (rr < N) & (block >= lo) & (block < hi)
        sc = tl.load(Scores + (h * N + rr[:, None]) * T + kk[None, :], rr[:, None] < N,
            other=float('-inf')).to(tl.float32) * 0.125
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
        sc = tl.load(Scores + (h * N + rr[:, None]) * T + kk[None, :], rr[:, None] < N,
            other=float('-inf')).to(tl.float32) * 0.125
        pp = tl.where(active[:, None], tl.exp(sc - mx[:, None]) / den[:, None], 0.).to(tl.bfloat16)
        key = tl.load(Keys + (h * T + kk[:, None]) * 64 + ds[None, :])
        val = tl.load(Values + (h * T + kk[:, None]) * 64 + ds[None, :])
        av += tl.dot(pp, val)
        ak += tl.dot(pp, key)
    off = (h * N + rr[:, None]) * K + tl.arange(0, K)[None, :]
    top = tl.load(Top + off, rr[:, None] < N, other=0.).to(tl.float32) * 0.125
    prob = tl.exp(top - mx[:, None]) / den[:, None]
    tl.store(Probability + off, prob, rr[:, None] < N)
    dst = (h * N + rr[:, None]) * 64 + ds[None, :]
    tl.store(OutV + dst, av.to(tl.bfloat16).to(tl.float32), rr[:, None] < N)
    tl.store(OutK + dst, ak.to(tl.bfloat16).to(tl.float32), rr[:, None] < N)

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
    _mask[(h, triton.cdiv(n, 32), t // 128)](scores, positions, n, t, window, 32, 128, num_warps=4)
    top, idx = window_topk.window_topk_pruned(scores, count)
    prob = torch.empty((h, n, count), device=q.device)
    ov = torch.empty((h, n, d), device=q.device)
    ok = torch.empty_like(ov)
    _moments[(triton.cdiv(n, 32), h)](scores, keys, values, top, positions, prob, ov, ok,
            n, t, window, count, 32, 128, num_warps=4, num_stages=2)
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


def prepare(recipe, raw_own, inside, count):
    """The strong keys of the query rows `inside` only: every token's queries by one GEMM, then the scores,
    top keys, moments and tables of the selected rows."""
    pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = raw_own
    selected_pos = pos[inside]
    selected_inputs = [None if x is None else x[inside] for x in inputs]
    top_keys, top_values, top_p = [], [], []
    for x, xs, wq, kk, vv, window, wp in zip(inputs, selected_inputs, w_q, keys, values, windows, w_proj_t):
        if x is None:
            top_keys.append(None)
            top_values.append(None)
            top_p.append(None)
            continue
        h = kk.shape[0]
        full_q = recipe.strong_queries(x, wq, cos, sin, h)
        q = full_q[inside]
        p, idx, out, all_keys = moments(q, kk, vv, selected_pos, window, count)
        shifted, tab = tables(vv, out, idx, wp, kk, p, all_keys)
        top_keys.append(shifted)
        top_values.append(tab)
        top_p.append(p.transpose(0, 1).contiguous())
    return (selected_pos, selected_inputs, w_q, top_keys, top_values, top_p, (cos, sin))
