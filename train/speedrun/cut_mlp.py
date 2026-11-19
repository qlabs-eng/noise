"""Cut MLP credits with clean activation/probe-RMS fusion and measured finite responses."""
import torch
import triton
import triton.language as tl
from id_kernels import _layer_offset, layer_offsets

CONFIG = dict(BM=128, BN=128, BK=64, GROUP=8, num_warps=8, num_stages=3)


@triton.jit
def _clean_activation_rms_kernel(U, Act, Rms, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    u = tl.load(U + row * N + cols).to(tl.float32)
    rms = tl.maximum(tl.sqrt(tl.sum(u * u, 0) / N), 1.0e-6)
    rectified = tl.maximum(u, 0.0)
    tl.store(Act + row * N + cols, (rectified * rectified).to(tl.bfloat16))
    tl.store(Rms + row, rms)


@torch.library.custom_op('cut_mlp::activation_rms', mutates_args=())
def activation_rms(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert u.is_contiguous() and u.dtype == torch.bfloat16 and u.shape[-1] == 2048
    n = u.shape[-1]
    m = u.numel() // n
    act = torch.empty_like(u)
    rms = torch.empty((m,), device=u.device, dtype=torch.float32)
    _clean_activation_rms_kernel[(m,)](u, act, rms, N=n, num_warps=4)
    return act, rms


@activation_rms.register_fake
def _(u):
    return torch.empty_like(u), torch.empty((u.numel() // u.shape[-1],), device=u.device, dtype=torch.float32)


@triton.jit(do_not_specialize=['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7',
                              'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7',
                              'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'])
def _rms_hidden_kernel(A, W, W1, W2, W3, W4, W5, W6, W7,
                   U, U1, U2, U3, U4, U5, U6, U7, R, R1, R2, R3, R4, R5, R6, R7, Out, Sigma: tl.constexpr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid, layer = tl.program_id(0), tl.program_id(1)
    nm, nn = M // BM, N // BN
    group = pid // (GROUP * nn)
    first = group * GROUP
    size = tl.minimum(nm - first, GROUP)
    pm = first + (pid % (GROUP * nn)) % size
    pn = (pid % (GROUP * nn)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    wp = W + _layer_offset(layer, W1, W2, W3, W4, W5, W6, W7)
    ap = A + layer * M * K + rm[:, None] * K + rk[None, :]
    bp = wp + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BK
        bp += BK * N
    read = acc.to(tl.bfloat16).to(tl.float32)
    uptr = U + _layer_offset(layer, U1, U2, U3, U4, U5, U6, U7)
    u = tl.load(uptr + rm[:, None] * N + rn[None, :]).to(tl.float32)
    rptr = R + _layer_offset(layer, R1, R2, R3, R4, R5, R6, R7) // 2
    step = (Sigma * tl.load(rptr + rm))[:, None]
    up, down = tl.maximum(u + step, 0.0), tl.maximum(u - step, 0.0)
    response = ((up - down) * (up + down)) / (2.0 * step)
    tl.store(Out + layer * M * N + rm[:, None] * N + rn[None, :],
             (read * response).to(tl.bfloat16))



@torch.library.custom_op('cut_mlp::hidden_credits_rms', mutates_args=())
def hidden_credits_rms(us: list[torch.Tensor], a: torch.Tensor, weights: list[torch.Tensor],
                       rmss: list[torch.Tensor], sigma: float) -> torch.Tensor:
    layers, m, k = a.shape
    n = us[0].shape[-1]
    c = CONFIG
    assert layers == 8 and a.is_contiguous() and a.dtype == torch.bfloat16
    assert m % c['BM'] == 0 and n % c['BN'] == 0 and k % c['BK'] == 0
    assert all(w.shape == (k, n) and w.dtype == torch.bfloat16 for w in weights)
    assert all(r.shape == (m,) and r.is_contiguous() and r.dtype == torch.float32 for r in rmss)
    uoff, woff = layer_offsets(us), layer_offsets(weights)
    roff = [(r.data_ptr() - rmss[0].data_ptr()) // 16 + (1 << 40) for r in rmss[1:]]
    assert all(r.data_ptr() % 16 == 0 for r in rmss)
    output = torch.empty((layers, m, n), device=a.device, dtype=torch.bfloat16)
    _rms_hidden_kernel[(m // c['BM'] * (n // c['BN']), layers)](
        a, weights[0], *woff, us[0], *uoff, rmss[0], *roff, output,
        Sigma=sigma, M=m, N=n, K=k, enable_fp_fusion=False, **c)
    return output


@hidden_credits_rms.register_fake
def _(us, a, weights, rmss, sigma):
    return torch.empty((a.shape[0], a.shape[1], us[0].shape[-1]), device=a.device, dtype=torch.bfloat16)
