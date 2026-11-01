"""The routed root's decode in run.py: credit = control + boost B^T (B B^T + ridge d I)^-1 r, r = loss_change / step -
B control, for each token's own bf16 move rows B (K, D) (K = 128: 64 routed c_fc rows of each of two blocks, |b_k|^2 =
d), in one Triton kernel per token: the Gram from B by tl.dot (bf16 products, fp32 accumulation), r in the same pass,
ITERS conjugate-gradient steps on the Gram scaled by 1 / d (near the identity: the routed rows are close to
orthogonal; its diagonal is constant, so Jacobi preconditioning is a no-op), the Gram held in registers (never written
out), then the lift B^T z. (In place of an fp32 Gram, a batched Cholesky and two triangular solves: ~1e-5 of the fp64
credit residual at 16 steps per 128 moves.)"""
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _span_cg_kernel(Rows, Loss, Steps, Control, Boost, Out, boost_s, ridge, inv_d,
                    K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, ITERS: tl.constexpr, BOOST_T: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    KB: tl.constexpr = K
    rk = tl.arange(0, KB)
    base = Rows + t * K * D
    step = tl.load(Steps + t)
    eye = (rk[:, None] == rk[None, :]).to(tl.float32) * ridge
    g = tl.zeros([KB, KB], tl.float32)
    c0 = tl.zeros([KB], tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        b0 = tl.load(base + rk[:, None] * D + cols[None, :])
        g = tl.dot(b0, tl.trans(b0), g)
        c0 += tl.sum(b0.to(tl.float32) * tl.load(Control + t * D + cols)[None, :], 1)
    g = g * inv_d + eye
    r0 = (tl.load(Loss + t * K + rk) / step - c0) * inv_d
    z0 = tl.zeros([KB], tl.float32)
    p0 = r0
    rs = tl.sum(r0 * r0, 0)
    for _ in range(ITERS):
        q0 = tl.sum(g * p0[None, :], 1)
        alpha = rs / tl.maximum(tl.sum(p0 * q0, 0), 1e-30)
        z0 += alpha * p0
        r0 -= alpha * q0
        new = tl.sum(r0 * r0, 0)
        beta = new / tl.maximum(rs, 1e-30)
        rs = new
        p0 = r0 + beta * p0
    boost = tl.load(Boost + t) if BOOST_T else boost_s
    for d0 in range(0, D, BD):  # the lift: control + boost B^T z
        cols = d0 + tl.arange(0, BD)
        u = tl.sum(tl.load(base + rk[:, None] * D + cols[None, :]).to(tl.float32) * z0[:, None], 0)
        tl.store(Out + t * D + cols, tl.load(Control + t * D + cols) + boost * u)


CONFIGS = {128: dict(BD=64, num_warps=4)}


@torch.library.custom_op('visit_kernel::span_cg_decode', mutates_args=())
def span_cg_decode(loss_change: torch.Tensor, steps: torch.Tensor, control: torch.Tensor, rows: torch.Tensor,
                   boost: Optional[torch.Tensor], boost_s: float, ridge: float, iters: int) -> torch.Tensor:
    """The routed root's per-token decode by `iters` CG steps (fp32, the Gram from the bf16 rows): loss_change (n, K),
    steps (n, 1), control (n, D) fp32, rows (n, K, D) bf16, boost (n, 1) per-token weights or None (then the scalar
    boost_s); ridge relative to the unit diagonal (run.py's SPAN_RIDGE). Returns the credit (n, D) fp32."""
    n, k, d = rows.shape
    assert k in CONFIGS and d % 32 == 0
    out = torch.empty((n, d), device=rows.device, dtype=torch.float32)
    per_token = boost is not None
    cfg = CONFIGS[k]
    _span_cg_kernel[(n,)](rows.to(torch.bfloat16).contiguous(), loss_change.float().contiguous(),
                          steps.float().contiguous(), control.float().contiguous(),
                          boost.float().contiguous() if per_token else out, out,
                          float(boost_s), float(ridge), 1.0 / d,
                          K=k, D=d, ITERS=iters, BOOST_T=per_token, **cfg)
    return out


@span_cg_decode.register_fake
def _(loss_change, steps, control, rows, boost, boost_s, ridge, iters):
    return torch.empty_like(control, dtype=torch.float32)
