"""The clean forward's attention as one kernel: causal softmax attention over a left window (utils/model.py's
`attention`: a query at s sees keys s - window to s) for (rows, 2048, 8, 64) bf16 q, k and v.

A program takes 64 queries of one head and walks the 64-key tiles its window reaches, keeping a running maximum,
sum and weighted value sum (online softmax, scores in log2 units). Only the tiles at the window's two edges are
masked; the tiles between them are inside every query's window.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _step(q, kp, vp, m, l, a, qi, start, SK: tl.constexpr, SV: tl.constexpr, W: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, MASK: tl.constexpr):
    kj = start + tl.arange(0, BN)
    d = tl.arange(0, 64)
    k = tl.load(kp + kj[None, :] * SK + d[:, None])
    s = tl.dot(q, k) * (0.125 * 1.4426950408889634)
    if MASK:
        s = tl.where((qi[:, None] >= kj[None, :]) & (qi[:, None] - kj[None, :] <= W), s, float('-inf'))
    new = tl.maximum(m, tl.max(s, 1))
    pivot = tl.where(new == float('-inf'), 0., new)
    prob = tl.exp2(s - pivot[:, None])
    alpha = tl.exp2(m - pivot)
    l = l * alpha + tl.sum(prob, 1)
    a = a * alpha[:, None]
    v = tl.load(vp + kj[:, None] * SV + d[None, :])
    a = tl.dot(prob.to(tl.bfloat16), v, a)
    return new, l, a


@triton.jit
def _attention(Q, K, V, O, QB: tl.constexpr, QT: tl.constexpr, QH: tl.constexpr, KB: tl.constexpr,
               KT: tl.constexpr, KH: tl.constexpr, VB: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
               T: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 8
    h = bh % 8
    first = block * BM
    qi = first + tl.arange(0, BM)
    d = tl.arange(0, 64)
    q = tl.load(Q + b * QB + h * QH + qi[:, None] * QT + d[None, :])
    kp = K + b * KB + h * KH
    vp = V + b * VB + h * VH
    m = tl.full((BM,), float('-inf'), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    a = tl.zeros((BM, 64), tl.float32)
    lo = tl.maximum(0, (first - W) // BN * BN)
    hi = tl.minimum(T, tl.cdiv(first + BM, BN) * BN)
    midlo = tl.maximum(0, tl.cdiv(first + BM - 1 - W, BN) * BN)
    midhi = first // BN * BN
    for start in range(lo, midlo, BN):
        m, l, a = _step(q, kp, vp, m, l, a, qi, start, KT, VT, W, BM, BN, True)
    for start in range(midlo, midhi, BN):
        m, l, a = _step(q, kp, vp, m, l, a, qi, start, KT, VT, W, BM, BN, False)
    for start in range(midhi, hi, BN):
        m, l, a = _step(q, kp, vp, m, l, a, qi, start, KT, VT, W, BM, BN, True)
    tl.store(O + ((b * T + qi[:, None]) * 8 + h) * 64 + d[None, :], (a / l[:, None]).to(tl.bfloat16))


CONFIG = dict(BM=64, BN=64, num_warps=4, num_stages=3)


def attention(q, k, v, window):
    assert q.shape == k.shape == v.shape and q.shape[-2:] == (8, 64)
    assert q.shape[1] == 2048 and window in (1024, 2048)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _attention[(triton.cdiv(q.shape[1], CONFIG['BM']), q.shape[0] * 8)](
        q, k, v, out, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], q.shape[1], window, **CONFIG)
    return out
