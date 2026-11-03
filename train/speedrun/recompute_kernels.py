"""block_recompute's MLP output GEMM with the relu^2 on its A load (RELU2_GEMM).

out[l] = mid[l] + relu(h[l])^2 @ w[l]^T in tf32 with cuBLAS's operand rounding (round to nearest, ties away from zero:
cvt.rna), so the result differs from torch.bmm's only by the fp32 accumulation order. The relu^2 tensor is never
written (inductor's pass wrote and re-read 4 bytes per hidden element).
"""
import torch
import triton
import triton.language as tl

RELU2_CONFIG = dict(BM=128, BN=256, BK=32, GROUP=8, num_warps=8, num_stages=3)


@triton.jit
def _rna_tf32(x):
    """x rounded to tf32 as cvt.rna (finite x): integer ops instead of the conversion instruction."""
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


def rna_tf32(x):
    """_rna_tf32 for a torch fp32 tensor (traceable by torch.compile)."""
    return ((x.view(torch.int32) + 0x1000) & -8192).view(torch.float32)


@triton.jit
def _relu2_gemm_kernel(H, W, Mid, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    """Out[l] = Mid[l] + rna(relu(H[l])^2) @ W[l]^T for H (L, M, K), W (L, N, K) (rounded to tf32 already), Mid / Out
    (L, M, N), all fp32 contiguous."""
    pid, layer = tl.program_id(0), tl.program_id(1).to(tl.int64)
    tiles_m, tiles_n = M // BM, N // BN
    width = GROUP * tiles_n
    first = (pid // width) * GROUP
    size = min(tiles_m - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = H + layer * M * K + rm[:, None] * K + rk[None, :]
    b_ptr = W + layer * N * K + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        a = tl.maximum(tl.load(a_ptr), 0.0)
        acc = tl.dot(_rna_tf32(a * a), tl.load(b_ptr), acc, input_precision='tf32')
        a_ptr += BK
        b_ptr += BK
    offs = layer * M * N + rm[:, None] * N + rn[None, :]
    tl.store(Out + offs, tl.load(Mid + offs) + acc)


def relu2_gemm(h, w_rounded, mid, config=None):
    """mid + bmm(relu(h)^2, w^T) for h (L, M, K), w_rounded = rna_tf32(w) (L, N, K), mid (L, M, N) fp32."""
    c = config or RELU2_CONFIG
    layers, m, k = h.shape
    n = w_rounded.shape[1]
    assert h.is_contiguous() and w_rounded.is_contiguous() and mid.is_contiguous() and mid.shape == (layers, m, n)
    assert m % c['BM'] == 0 and n % c['BN'] == 0 and k % c['BK'] == 0
    out = torch.empty_like(mid)
    _relu2_gemm_kernel[((m // c['BM']) * (n // c['BN']), layers)](h, w_rounded, mid, out, m, n, k, BM=c['BM'], BN=c['BN'],
                                                                  BK=c['BK'], GROUP=c['GROUP'], num_warps=c['num_warps'],
                                                                  num_stages=c['num_stages'])
    return out
