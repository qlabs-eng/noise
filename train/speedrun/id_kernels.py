"""Identity-phase kernels (round 18): fused pieces of run.py's identity_grad.

hidden_credits_fused: hidden_credits' batched GEMM (credb @ w_out_t, bf16 out) with hidden_slope_'s relu^2 central
difference slope as the GEMM epilogue: the bf16 product is rounded as cuBLAS rounds it, then scaled by the slope in
fp32 and rounded again, so the result is hidden_credits' up to the GEMM's fp32 accumulation order.
"""
import torch
import triton
import triton.language as tl

BIAS: tl.constexpr = tl.constexpr(1 << 40)  # added to offsets: they always pass as int64 scalars (no respecialization)

HIDDEN_CONFIG = dict(BM=128, BN=128, BK=64, GROUP=8, num_warps=8, num_stages=3)


@triton.jit
def _layer_offset(layer, O1, O2, O3, O4, O5, O6, O7):
    """Element offset of layer's tensor from layer 0's (eight separately allocated tensors), from offsets passed in
    units of 8 elements plus BIAS (layer_offsets): the products keep the loads' 16-byte alignment visible."""
    units = tl.where(layer == 1, O1, tl.where(layer == 2, O2, tl.where(layer == 3, O3, tl.where(
        layer == 4, O4, tl.where(layer == 5, O5, tl.where(layer == 6, O6, tl.where(layer == 7, O7, BIAS)))))))
    return (units - BIAS) * 8


@triton.jit(do_not_specialize=['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7'])
def _hidden_credit_kernel(A, B, U, O1, O2, O3, O4, O5, O6, O7, Step, Out, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    """Out[l] = bf16(slope(U_l, Step[l]) * float(bf16(A[l] @ B[l]))) for A (L, M, K), B (L, K, N), Out (L, M, N) bf16,
    U_l (M, N) bf16 at element offset O_l from U (O_0 = 0) and Step (L, M) fp32."""
    pid, layer = tl.program_id(0), tl.program_id(1).to(tl.int64)
    tiles_m, tiles_n = M // BM, N // BN
    width = GROUP * tiles_n
    group = pid // width
    first = group * GROUP
    size = min(tiles_m - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = A + layer * M * K + rm[:, None] * K + rk[None, :]
    b_ptr = B + layer * K * N + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptr), tl.load(b_ptr), acc)
        a_ptr += BK
        b_ptr += BK * N
    read = acc.to(tl.bfloat16).to(tl.float32)
    tile = rm[:, None] * N + rn[None, :]
    offs = layer * M * N + tile
    u = tl.load(U + _layer_offset(layer, O1, O2, O3, O4, O5, O6, O7) + tile).to(tl.float32)
    step = tl.load(Step + layer * M + rm)[:, None]
    up = tl.maximum(u + step, 0.0)
    down = tl.maximum(u - step, 0.0)
    slope = (up * up - down * down) / (2 * step)
    tl.store(Out + offs, (slope * read).to(tl.bfloat16))


def layer_offsets(xs):
    """Offsets of eight tensors' storage from the first's in 16-byte units (8 bf16 elements), plus BIAS (as kernel
    scalars; see _layer_offset)."""
    assert xs[0].element_size() == 2 and xs[0].data_ptr() % 16 == 0
    offsets = [(x.data_ptr() - xs[0].data_ptr()) // 16 + BIAS.value for x in xs]
    assert len(xs) == 8 and all(x.is_contiguous() and (x.data_ptr() - xs[0].data_ptr()) % 16 == 0 for x in xs)
    return offsets[1:]


def hidden_credits_fused(us, steps, credb, w_out_t, config=None):
    """us: eight (.., N) bf16 pre-activations with M rows each, steps (L, M) fp32."""
    c = config or HIDDEN_CONFIG
    layers, m, k = credb.shape
    n = w_out_t.shape[-1]
    assert credb.is_contiguous() and w_out_t.is_contiguous() and steps.is_contiguous()
    assert all(u.numel() == m * n for u in us)
    assert m % c['BM'] == 0 and n % c['BN'] == 0 and k % c['BK'] == 0
    out = torch.empty((layers, m, n), device=credb.device, dtype=torch.bfloat16)
    grid = ((m // c['BM']) * (n // c['BN']), layers)
    _hidden_credit_kernel[grid](credb, w_out_t, us[0], *layer_offsets(us), steps, out, M=m, N=n, K=k, BM=c['BM'],
                                BN=c['BN'], BK=c['BK'], GROUP=c['GROUP'], num_warps=c['num_warps'],
                                num_stages=c['num_stages'])
    return out


SQUARE_CONFIG = dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3)


@triton.jit(do_not_specialize=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'])
def _square_credit_kernel(A, A1, A2, A3, A4, A5, A6, A7, G, Out, L: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                          HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Out[j * L + l] = bf16(A_l^T @ B_jl) for A_l (M, D) bf16 at offset A_l from A (see _layer_offset) and
    B_jl[tok, w * HD + d] = G[w, l, j, tok, d]
    (G (W, L, 3, M, HD) bf16, the received heads_out): identity_grad's c_q / c_k / c_v gradients without credit_stacks'
    by-token copy."""
    pid, jl = tl.program_id(0), tl.program_id(1)
    j, layer = jl // L, jl % L
    tiles_n = (D // HD * HD) // BN
    pm, pn = pid // tiles_n, pid % tiles_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    col = (((rn // HD) * L + layer) * 3 + j).to(tl.int64) * M * HD + rn % HD
    a_ptr = A + _layer_offset(layer, A1, A2, A3, A4, A5, A6, A7) + rk[:, None] * D + rm[None, :]
    b_ptr = G + col[None, :] + rk[:, None] * HD
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, M, BK):
        acc = tl.dot(tl.trans(tl.load(a_ptr)), tl.load(b_ptr), acc)
        a_ptr += BK * D
        b_ptr += BK * HD
    out = Out + jl.to(tl.int64) * D * D + rm[:, None] * D + rn[None, :]
    tl.store(out, acc.to(tl.bfloat16))


def square_credit_grads(n1s, got2, out, config=None):
    """out (3L, D, W * HD) bf16 contiguous view: out[j * L + l] = n1s[l]^T @ (got2[:, l, j] by token); n1s: eight
    (M, D) bf16 matrices (a stack or separate tensors)."""
    c = config or SQUARE_CONFIG
    layers, (m, d) = len(n1s), n1s[0].shape
    world, _, three, _, hd = got2.shape
    assert three == 3 and world * hd == d and out.shape == (3 * layers, d, d) and out.is_contiguous()
    assert got2.is_contiguous() and m % c['BK'] == 0 and d % c['BM'] == 0
    grid = ((d // c['BM']) * (d // c['BN']), 3 * layers)
    _square_credit_kernel[grid](n1s[0], *layer_offsets(list(n1s)), got2, out, L=layers, M=m, D=d, HD=hd,
                                BM=c['BM'], BN=c['BN'], BK=c['BK'], num_warps=c['num_warps'],
                                num_stages=c['num_stages'])



AT_B_CONFIG = dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3)


@triton.jit(do_not_specialize=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
def _at_b_kernel(A, A1, A2, A3, A4, A5, A6, A7, B, B1, B2, B3, B4, B5, B6, B7, Out, K: tl.constexpr, P: tl.constexpr,
                 Q: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Out[l] = bf16(A_l^T @ B_l) for A_l (K, P) and B_l (K, Q) bf16 row-major at offsets A_l / B_l from A / B
    (see _layer_offset); Out (L, P, Q) bf16 contiguous."""
    pid, layer = tl.program_id(0), tl.program_id(1)
    tiles_n = Q // BN
    pm, pn = pid // tiles_n, pid % tiles_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = A + _layer_offset(layer, A1, A2, A3, A4, A5, A6, A7) + rk[:, None] * P + rm[None, :]
    b_ptr = B + _layer_offset(layer, B1, B2, B3, B4, B5, B6, B7) + rk[:, None] * Q + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.trans(tl.load(a_ptr)), tl.load(b_ptr), acc)
        a_ptr += BK * P
        b_ptr += BK * Q
    tl.store(Out + layer.to(tl.int64) * P * Q + rm[:, None] * Q + rn[None, :], acc.to(tl.bfloat16))


def at_b_layers(As, Bs, out, config=None):
    """out[l] = As[l]^T @ Bs[l] (bf16 out, fp32 accumulation) for eight (K, P) / (K, Q) bf16 matrices each (any
    leading shape flattened to K rows): torch.bmm(stack(As).transpose(1, 2), stack(Bs), out=out) without the stacks."""
    c = config or AT_B_CONFIG
    p, q = As[0].shape[-1], Bs[0].shape[-1]
    k = As[0].numel() // p
    assert out.shape == (8, p, q) and out.is_contiguous() and all(b.numel() == k * q for b in Bs)
    assert all(a.numel() == k * p for a in As) and k % c['BK'] == 0 and p % c['BM'] == 0 and q % c['BN'] == 0
    _at_b_kernel[((p // c['BM']) * (q // c['BN']), 8)](As[0], *layer_offsets(As), Bs[0], *layer_offsets(Bs), out, K=k,
                                                     P=p, Q=q, BM=c['BM'], BN=c['BN'], BK=c['BK'],
                                                     num_warps=c['num_warps'], num_stages=c['num_stages'])
