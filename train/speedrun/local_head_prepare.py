import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(Q, K, V, O, TG, LseIn, Cos, Sin, RawQK,
             Q16, K16, V16, Hub16, Q32, K32, LSE, HubOut, SqQ, SqK, StepsQ, StepsK,
             layer, sq, sk, sv, so, st, ls0, ls1, ls2,
             W: tl.constexpr, L: tl.constexpr, N: tl.constexpr, T: tl.constexpr,
             SIGMA: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr,
             F2=None, F4=None, XY=None, PAIRS: tl.constexpr = False):
    tile, w = tl.program_id(0), tl.program_id(1)
    i = (tile * BM + tl.arange(0, BM)).to(tl.int64)
    src = ((w * L + layer) * N + i).to(tl.int64)
    dst = ((layer * W + w) * N + i).to(tl.int64)
    pos = i % T
    half = tl.arange(0, 32)
    dims = tl.arange(0, 64)
    c = tl.load(Cos + pos[:, None] * 32 + half[None, :])
    s = tl.load(Sin + pos[:, None] * 32 + half[None, :])
    for j in tl.static_range(2):
        raw = Q if j == 0 else K
        stride = sq if j == 0 else sk
        offset = i[:, None] * stride + w * 64 + half[None, :]
        raw1 = tl.load(raw + offset)
        raw2 = tl.load(raw + offset + 32)
        tl.store(RawQK + src[:, None] * 128 + j * 64 + half[None, :], raw1)
        tl.store(RawQK + src[:, None] * 128 + j * 64 + 32 + half[None, :], raw2)
        x1 = raw1.to(tl.float32)
        x2 = raw2.to(tl.float32)
        r1 = x1 * c + x2 * s
        r2 = x1 * -s + x2 * c
        inv = tl.rsqrt((tl.sum(r1 * r1, 1) + tl.sum(r2 * r2, 1)) / 64 + EPS)
        r1, r2 = r1 * inv[:, None], r2 * inv[:, None]
        sqr = tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)
        out16 = (Q16 if j == 0 else K16) + dst[:, None] * 64 + half[None, :]
        out32 = (Q32 if j == 0 else K32) + dst[:, None] * 64 + half[None, :]
        tl.store(out16, r1.to(tl.bfloat16))
        tl.store(out16 + 32, r2.to(tl.bfloat16))
        if PAIRS and j == 1:
            x, y = r1 * 64 ** (-0.5), r2 * 64 ** (-0.5)
            tl.store(F2 + dst[:, None] * 64 + dims[None, :], tl.reshape(tl.join(x, y), (BM, 64)).to(tl.bfloat16))
            tl.store(XY + dst[:, None] * 32 + half[None, :], (x * y).to(tl.bfloat16))
            f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BM, 128))
            tl.store(F4 + dst[:, None] * 128 + tl.arange(0, 128)[None, :], f4.to(tl.bfloat16))
        else:
            tl.store(out32, r1)
            tl.store(out32 + 32, r2)
        tl.store((SqQ if j == 0 else SqK) + dst, sqr)
        tl.store((StepsQ if j == 0 else StepsK) + dst, SIGMA * tl.maximum(tl.sqrt(sqr / 64), 1e-6))
    col = w * 64 + dims[None, :]
    v = tl.load(V + i[:, None] * sv + col)
    o = tl.load(O + i[:, None] * so + col).to(tl.float32)
    hub = tl.load(TG + i[:, None] * st + col)
    tl.store(V16 + dst[:, None] * 64 + dims[None, :], v)
    tl.store(Hub16 + dst[:, None] * 64 + dims[None, :], hub)
    tl.store(HubOut + dst, tl.sum(hub.to(tl.float32) * o, 1))
    tl.store(LSE + dst, tl.load(LseIn + (i // T) * ls0 + w * ls1 + pos * ls2))


def prepare(qs, ks, vals, outs, targets, lses, cos, sin, b, t, heads, sigma, eps):
    assert len(qs) > 0 and t > 0
    layers, n = len(qs), qs[0].shape[0]
    assert n % 32 == 0 and n % t == 0 and b * t == heads * n
    assert all(len(xs) == layers for xs in (ks, vals, outs, targets, lses))
    rows = layers * b
    new = lambda *s, dtype=torch.float32: torch.empty(*s, device=qs[0].device, dtype=dtype)
    raw = new(heads, layers, n, 128, dtype=torch.bfloat16)
    q16, k16, v16, hub16 = (new(rows, t, 64, dtype=torch.bfloat16) for _ in range(4))
    q32, k32 = new(rows, t, 64), new(rows, t, 64)
    lse, hub_out, sq_q, sq_k, steps_q, steps_k = (new(rows, t) for _ in range(6))
    pairs = (new(rows, t, 64, dtype=torch.bfloat16), new(rows, t, 128, dtype=torch.bfloat16),
             new(rows, t, 32, dtype=torch.bfloat16))
    for layer, (q, k, v, o, tg, li) in enumerate(zip(qs, ks, vals, outs, targets, lses)):
        assert all(z.dtype == torch.bfloat16 and z.shape == (n, heads * 64) and z.stride(1) == 1
                   for z in (q, k, v, o, tg))
        assert li.dtype == torch.float32 and li.shape == (n // t, heads, t)
        _prepare[(n // 32, heads)](q, k, v, o, tg, li, cos, sin, raw,
            q16, k16, v16, hub16, q32, k32, lse, hub_out, sq_q, sq_k, steps_q, steps_k,
            layer, q.stride(0), k.stride(0), v.stride(0), o.stride(0), tg.stride(0),
            li.stride(0), li.stride(1), li.stride(2), W=heads, L=layers, N=n, T=t,
            SIGMA=sigma, EPS=eps, BM=32, F2=pairs[0], F4=pairs[1], XY=pairs[2],
            PAIRS=True, num_warps=4)
    prepared = (q16, k16, v16, hub16, q32, pairs, lse, hub_out, sq_q, sq_k, steps_q, steps_k)
    return prepared, raw
