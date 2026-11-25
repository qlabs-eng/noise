import torch
import triton
import triton.language as tl


@triton.jit
def _pick(ptrs, layer, L: tl.constexpr):
    out = ptrs[0]
    for i in tl.static_range(1, L):
        out = tl.where(layer == i, ptrs[i], out)
    return out


@triton.jit
def _direct(Blocks, Weights, Out, Dots, L: tl.constexpr, XS: tl.constexpr,
    N: tl.constexpr, D: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
           SIGMA: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr):
    tt = tl.program_id(0) * BT + tl.arange(0, BT)
    layer = tl.program_id(1)
    row = layer * N + tt
    X = _pick(Blocks, layer, L)
    W = _pick(Weights, layer, L)
    dd = tl.arange(0, D)
    kk = tl.arange(0, K)
    x = tl.load(X + tt[:, None] * XS + dd[None, :], tt[:, None] < N, 0).to(tl.float32)
    xk = tl.load(X + tt[:, None] * XS + kk[None, :], tt[:, None] < N, 0).to(tl.float32)
    sq = tl.sum(x * x, 1) / D
    radius = SIGMA * D**.5 * tl.maximum(tl.sqrt(sq), 1e-6)
    plus_inv = tl.rsqrt(sq[:, None] + (2 * radius[:, None] * x + radius[:, None] * radius[:, None]) / D + EPS)
    minus_inv = tl.rsqrt(sq[:, None] + (-2 * radius[:, None] * x + radius[:, None] * radius[:, None]) / D + EPS)
    acc = tl.full((BT, D), 0, tl.float32)
    for h in tl.static_range(H):
        wk = tl.load(W + h * K + kk).to(tl.float32)
        base = tl.sum(xk * wk[None, :], 1)[:, None]
        kappa = tl.load(Dots + row * H + h, tt < N, other=0.)[:, None]
        weight = tl.load(W + h * K + dd, dd < K, 0).to(tl.float32)[None, :]
        plus = (base + radius[:, None] * weight) * plus_inv
        minus = (base - radius[:, None] * weight) * minus_inv
        sp = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=r,r', [plus * .5], dtype=tl.float32,
            is_pure=True, pack=1)
        sm = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=r,r', [minus * .5], dtype=tl.float32,
            is_pure=True, pack=1)
        acc += (sp - sm) * (.5 * kappa)
    tl.store(Out + row[:, None] * D + dd[None, :], acc / radius[:, None], tt[:, None] < N)


def gate_inputs_direct(blocks, weights, sigma: float, dots):
    layers, bt = len(blocks), 4
    d = blocks[0].shape[-1]
    n = blocks[0].numel() // d
    h, k = weights[0].shape
    xs = blocks[0].stride(-2)
    out = torch.empty((layers, n, d), device=dots.device, dtype=torch.float32)
    _direct[(triton.cdiv(n, bt), layers)](tuple(blocks), tuple(weights), out, dots, layers, xs,
        n, d, h, k, sigma, 1.1920928955078125e-7, bt, num_warps=4, enable_fp_fusion=False)
    return out.unbind(0)
