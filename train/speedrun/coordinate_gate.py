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
def _direct(Blocks, Raws, Ves, Weights, Out, L: tl.constexpr, XS: tl.constexpr, RS: tl.constexpr, VS: tl.constexpr,
    N: tl.constexpr, D: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
           SIGMA: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr, TANH: tl.constexpr):
    tt = tl.program_id(0) * BT + tl.arange(0, BT)
    layer = tl.program_id(1)
    row = layer * N + tt
    X = _pick(Blocks, layer, L)
    Raw = _pick(Raws, layer, L)
    VE = _pick(Ves, layer, L)
    W = _pick(Weights, layer, L)
    dd = tl.arange(0, D)
    kk = tl.arange(0, K)
    hd = tl.arange(0, D // H)
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
        raw = tl.load(Raw + tt[:, None] * RS + h * (D // H) + hd[None, :], tt[:, None] < N, 0).to(tl.float32)
        ve = tl.load(VE + tt[:, None] * VS + h * (D // H) + hd[None, :], tt[:, None] < N, 0).to(tl.float32)
        kappa = tl.sum(raw * ve, 1)[:, None]
        weight = tl.load(W + h * K + dd, dd < K, 0).to(tl.float32)[None, :]
        plus = (base + radius[:, None] * weight) * plus_inv
        minus = (base - radius[:, None] * weight) * minus_inv
        if TANH:
            sp = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=r,r', [plus * .5], dtype=tl.float32,
                is_pure=True, pack=1)
            sm = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=r,r', [minus * .5], dtype=tl.float32,
                is_pure=True, pack=1)
            acc += (sp - sm) * (.5 * kappa)
        else:
            sp = 1. / (1. + tl.exp(-plus))
            sm = 1. / (1. + tl.exp(-minus))
            acc += (sp - sm) * kappa
    tl.store(Out + row[:, None] * D + dd[None, :], acc / radius[:, None], tt[:, None] < N)


def gate_inputs_direct(raws, ves, blocks, weights, sigma: float, bt: int = 4, tanh: bool = True):
    layers = len(raws)
    n, d = raws[0].shape
    h, k = weights[0].shape
    xs = blocks[0].stride(-2)
    rs = raws[0].stride(0)
    vs = ves[0].stride(0)
    assert layers in (4, 8) and d == 512 and h == 8 and k == 32
    assert all(x.stride(-1) == 1 and x.stride(-2) == xs and x.numel() == n * d for x in blocks)
    assert all(z.stride(1) == 1 and z.stride(0) == rs and z.shape == (n, d) for z in raws)
    assert all(z.stride(1) == 1 and z.stride(0) == vs and z.shape == (n, d) for z in ves)
    assert all(z.is_contiguous() for z in weights)
    out = torch.empty((layers, n, d), device=raws[0].device, dtype=torch.float32)
    _direct[(triton.cdiv(n, bt), layers)](tuple(blocks), tuple(raws), tuple(ves), tuple(weights), out, layers, xs, rs,
        vs,
        n, d, h, k, sigma, 1.1920928955078125e-7, bt, tanh, num_warps=4 if bt <= 4 else 8, enable_fp_fusion=False)
    return out.unbind(0)
