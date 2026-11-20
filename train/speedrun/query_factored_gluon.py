"""Factored query replay with one token/head-pair per CTA.

Each CTA gathers its token's factors directly from the small weight product once,
keeps them and the rotated keys resident, and iterates over all probe tiles.
Probe-specific changes and frame tiles arrive by TMA.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

EPS = torch.finfo(torch.float32).eps
from visit_chain import _bf16, _rbf16


@gluon.jit
def _head_keys_issue(kh, rh, kt, k_off, r_off, rmask):
    """cp.async of a token's rotated keys / r0' (KR rows at kt) for both heads of the pair: kh rows 16 h + key, rh
    row 8 h (the other rows stay zero)."""
    cp.async_copy_global_to_shared(kh, kt + k_off)
    cp.async_copy_global_to_shared(rh, kt + r_off, mask=rmask)
    cp.commit_group()


@gluon.jit
def _head_scores(a, kh, rh, h: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, mmak: gl.constexpr,
    mmad: gl.constexpr):
    """Head h's raw . k' and raw . r0' (column 0) wgmmas, in flight."""
    ak = gl.convert_layout(a, gl.DotOperandLayout(operand_index=0, parent=mmak, k_width=2))
    ad = gl.convert_layout(a, gl.DotOperandLayout(operand_index=0, parent=mmad, k_width=2))
    sc = warpgroup_mma(ak, kh.slice(h * K, K).permute((1, 0)), gl.zeros([BM, K], gl.float32, mmak), use_acc=False,
      is_async=True)
    q = warpgroup_mma(ad, rh.slice(h * 8, 8).permute((1, 0)), gl.zeros([BM, 8], gl.float32, mmad), use_acc=False,
      is_async=True)
    return sc, q, ak, ad


@gluon.jit
def _head_weights(sc, q, moved, g, shift0, sq, rk, prob, optr, D: gl.constexpr, EPS_: gl.constexpr):
    """One head's replay weights from raw . k' (sc), raw . r0' (q), |raw|^2 (moved) and the token's operands."""
    dots = gl.convert_layout(gl.sum(q, 1), g.type.layout)  # q's only nonzero column
    qdots = g * dots + shift0 * sq
    qmoved = g * (g * moved + 2 * shift0 * dots) + shift0 * shift0 * sq
    inv0 = gl.rsqrt(sq / D + EPS_)
    inv = gl.rsqrt((sq + 2 * qdots + qmoved) / D + EPS_)
    alpha = gl.expand_dims(g * inv, 1)
    beta = gl.expand_dims(shift0 * inv + (inv - inv0), 1)
    ds = _rbf16(alpha * sc + beta * rk[None, :]) * (D ** -.5)
    pu = prob[None, :] * (gl.exp(ds) - 1.0)
    gl.store(optr, _bf16(pu * gl.expand_dims(1.0 / (1.0 + gl.sum(pu, 1)), 1)))


@gluon.jit
def _issue_moves(change_desc, frame_desc, bar, csm, fsm, token, pair, pos0,
    P: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, WN: gl.constexpr):
    mbarrier.expect(bar, BM * (R + WN) * 2)
    tma.async_copy_global_to_shared(change_desc, [token * P + pos0, 0], bar, csm)
    tma.async_copy_global_to_shared(frame_desc, [pos0, pair * WN], bar, fsm)


@gluon.jit
def _qf_rot_kernel(change_desc, Prod, Routes, frame_desc, Steps, Scale, KR, RK, SQ, Inv, Shift, Prob, Out,
    P: gl.constexpr, C: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, R: gl.constexpr,
    BM: gl.constexpr, EPS_: gl.constexpr, num_warps: gl.constexpr):
    HP: gl.constexpr = H // 2
    WN: gl.constexpr = 2 * D
    KR_ROWS: gl.constexpr = 2 * K + 2
    gl.static_assert(BM == 16 * num_warps)
    pid = gl.program_id(0)
    pair = pid % HP
    token = pid // HP
    csm = gl.allocate_shared_memory(gl.bfloat16, [BM, R], change_desc.layout)
    asm = gl.allocate_shared_memory(gl.bfloat16, [R, WN],
      gl.NVMMASharedLayout.get_default_for([R, WN], gl.bfloat16))
    fsm = gl.allocate_shared_memory(gl.bfloat16, [BM, WN], frame_desc.layout)
    klay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([2 * K, D], gl.bfloat16)
    rlay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([16, D], gl.bfloat16)
    kh = gl.allocate_shared_memory(gl.bfloat16, [2 * K, D], klay)
    rh = gl.allocate_shared_memory(gl.bfloat16, [16, D], rlay)
    bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    mbarrier.init(bars.index(0), count=1)
    factor_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    factor_r = gl.arange(0, R, gl.SliceLayout(1, factor_layout))
    factor_c = gl.arange(0, WN, gl.SliceLayout(0, factor_layout))
    chosen = gl.load(Routes + token * R + factor_r).to(gl.int64)
    cp.async_copy_global_to_shared(asm, Prod + chosen[:, None] * C + (pair * WN + factor_c)[None, :])
    cp.commit_group()
    _issue_moves(change_desc, frame_desc, bars.index(0), csm, fsm, token, pair, 0, P, R, BM, WN)

    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, WN, 16])
    mmak: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, K, 16])
    mmad: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 8, 16])
    RL: gl.constexpr = gl.SliceLayout(1, mmak)
    KLY: gl.constexpr = gl.SliceLayout(0, mmak)
    bl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    kr = gl.arange(0, 2 * K, gl.SliceLayout(1, bl))
    kc = gl.arange(0, D, gl.SliceLayout(0, bl))
    k_off = ((2 * pair + kr // K) * KR_ROWS + kr % K)[:, None] * D + kc[None, :]
    rr = gl.arange(0, 16, gl.SliceLayout(1, bl))
    r_off = ((2 * pair + rr // 8) * KR_ROWS + 2 * K)[:, None] * D + kc[None, :]
    rmask = (rr % 8 == 0)[:, None] & (kc < D)[None, :]
    rh.store(gl.zeros([16, D], gl.bfloat16, bl))
    gl.thread_barrier()
    _head_keys_issue(kh, rh, KR + token * (H * KR_ROWS * D), k_off, r_off, rmask)
    scale = gl.load(Scale).to(gl.float32)
    step = gl.load(Steps + token)
    zero = gl.zeros([BM, WN], gl.float32, mma)
    rows = gl.arange(0, BM, RL)
    kk = gl.arange(0, K, KLY)
    hk = (token * H + 2 * pair) * K + kk
    rk0 = gl.load(RK + hk)
    rk1 = gl.load(RK + K + hk)
    pr0 = gl.load(Prob + hk)
    pr1 = gl.load(Prob + K + hk)
    sq0 = gl.load(SQ + token * H + 2 * pair)
    sq1 = gl.load(SQ + token * H + 2 * pair + 1)
    cp.wait_group(0)
    fence_async_shared()
    gl.thread_barrier()
    for pos0 in range(0, P, BM):
        mbarrier.wait(bars.index(0), (pos0 // BM) & 1)
        acc = warpgroup_mma(csm, asm, zero, use_acc=False, is_async=True)
        acc = warpgroup_mma_wait(0, deps=[acc])
        raw = scale * (acc + step * fsm.load(mma).to(gl.float32))
        gl.thread_barrier()
        if pos0 + BM < P:
            _issue_moves(change_desc, frame_desc, bars.index(0), csm, fsm, token, pair, pos0 + BM, P, R, BM, WN)
        x0, x1 = gl.split(gl.permute(gl.reshape(_bf16(raw), [BM, 2, D]), (0, 2, 1)))
        s0, q0, a0, d0 = _head_scores(x0, kh, rh, 0, K, BM, mmak, mmad)
        s1, q1, a1, d1 = _head_scores(x1, kh, rh, 1, K, BM, mmak, mmad)
        prow = token * P + pos0 + rows
        g = gl.load(Inv + prow)
        shift0 = gl.load(Shift + prow)
        f0 = x0.to(gl.float32)
        f1 = x1.to(gl.float32)
        m0 = gl.convert_layout(gl.sum(f0 * f0, 1), RL)
        m1 = gl.convert_layout(gl.sum(f1 * f1, 1), RL)
        s0, q0, s1, q1 = warpgroup_mma_wait(0, deps=[s0, q0, s1, q1, a0, d0, a1, d1])[:4]
        orow = gl.expand_dims(prow.to(gl.int64) * (H * K), 1) + (2 * pair * K + kk)[None, :]
        _head_weights(s0, q0, m0, g, shift0, sq0, rk0, pr0, Out + orow, D, EPS_)
        _head_weights(s1, q1, m1, g, shift0, sq1, rk1, pr1, Out + orow + K, D, EPS_)
    mbarrier.invalidate(bars.index(0))

@triton.jit
def _rot_keys_kernel(Keys, R0, Cos, Sin, KR, RK, SQ, H: tl.constexpr, K: tl.constexpr, D: tl.constexpr):
    """Per (token, head): the keys and r0 rotated back by the token's RoPE angles (k' with rope(x) . k = x . k'),
    w_q-interleaved columns, as bf16 hi / lo parts: KR rows [k' hi (K), k' lo (K), r0' hi, r0' lo]; RK = r0 . k
    and SQ = |r0|^2 (fp32)."""
    pid = tl.program_id(0)
    token = pid // H
    HD: tl.constexpr = D // 2
    kk = tl.arange(0, K)
    ii = tl.arange(0, HD)
    rows = Keys + pid * (K * D) + kk[:, None] * D
    ka = tl.load(rows + ii[None, :]).to(tl.float32)
    kb = tl.load(rows + HD + ii[None, :]).to(tl.float32)
    c = tl.load(Cos + token * HD + ii)
    s = tl.load(Sin + token * HD + ii)
    r0a = tl.load(R0 + pid * D + ii)
    r0b = tl.load(R0 + pid * D + HD + ii)
    tl.store(RK + pid * K + kk, tl.sum(ka * r0a[None, :] + kb * r0b[None, :], 1))
    tl.store(SQ + pid, tl.sum(r0a * r0a + r0b * r0b, 0))
    kp = tl.reshape(tl.join(c[None, :] * ka - s[None, :] * kb, s[None, :] * ka + c[None, :] * kb), [K, D])
    up = tl.reshape(tl.join(c * r0a - s * r0b, s * r0a + c * r0b), [D])
    out = KR + pid * ((2 * K + 2) * D)
    dd = tl.arange(0, D)
    khi = kp.to(tl.bfloat16)
    tl.store(out + kk[:, None] * D + dd[None, :], khi)
    tl.store(out + (K + kk)[:, None] * D + dd[None, :], (kp - khi.to(tl.float32)).to(tl.bfloat16))
    uhi = up.to(tl.bfloat16)
    tl.store(out + 2 * K * D + dd, uhi)
    tl.store(out + (2 * K + 1) * D + dd, (up - uhi.to(tl.float32)).to(tl.bfloat16))


ROT_CONFIG = dict(BM=64, num_warps=4, maxnreg=128)  # no spills at 128


@triton.jit
def _factor_at_kernel(Wo, Routes, Wq, At, M, C: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr):
    """At[t, c, r] = (w_q @ Wo[routes[t, r]]^T)[c] (bf16, fp32 accumulation): the gathered rows' GEMM, stored
    transposed per token (r contiguous). Called with every hidden unit as its own row (R = 1): the full product."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    live = rows < M
    chosen = tl.load(Routes + rows, mask=live, other=0).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        a = tl.load(Wo + chosen[:, None] * D + (k0 + kk)[None, :])
        b = tl.load(Wq + cols[:, None] * D + (k0 + kk)[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    token = rows // R
    dst = (token[:, None].to(tl.int64) * C + cols[None, :]) * R + (rows % R)[:, None]
    tl.store(At + dst, acc.to(tl.bfloat16), mask=live[:, None])


def interleave_wq(w_q, heads):
    """w_q's rows per head reordered a0 b0 a1 b1 ... (the RoPE halves interleaved), bf16."""
    c = w_q.shape[0]
    return w_q.to(torch.bfloat16).view(heads, 2, c // heads // 2, -1).transpose(1, 2).reshape(c, -1).contiguous()


FACTOR_CONFIG = dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3)


@torch.library.custom_op('visit_kernel::query_factors_table', mutates_args=())
def query_factors_table(wo: torch.Tensor, routes: torch.Tensor, basis: torch.Tensor,
    w_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full bf16 factor product and frame projection; routed rows are gathered inside the replay."""
    c = w_q.shape[0]
    wq = interleave_wq(w_q, c // 64)
    n, r = routes.shape
    rows, d = wo.shape
    cfg = dict(FACTOR_CONFIG)
    assert c % cfg['BN'] == 0
    prod = torch.empty((rows, c, 1), device=wo.device, dtype=torch.bfloat16)  # (Wo @ wq^T) as At with R = 1
    every = torch.arange(rows, device=wo.device, dtype=routes.dtype)
    _factor_at_kernel[(triton.cdiv(rows, cfg['BM']), c // cfg['BN'])](wo.contiguous(), every, wq, prod, rows, c, d, 1,
      **cfg)
    f = (basis.float() @ wq.float().T).to(torch.bfloat16)
    return prod.view(rows, c), f


@query_factors_table.register_fake
def _(wo, routes, basis, w_q):
    return (torch.empty((wo.shape[0], w_q.shape[0]), device=wo.device, dtype=torch.bfloat16),
      torch.empty((basis.shape[0], w_q.shape[0]), device=wo.device, dtype=torch.bfloat16))


@torch.library.custom_op('visit_kernel::fused_query_replay_factored_rot', mutates_args=())
def fused_query_replay_factored_rot(change: torch.Tensor, steps: torch.Tensor, scale: torch.Tensor,
    at: torch.Tensor, f: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor,
    shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    """The factored query replay by _qf_rot_kernel (the scores against rotated keys, no per-element
    RoPE; dq unrounded, one reciprocal per row): one _rot_keys_kernel pass over the keys, then the replay."""
    n, p, r = change.shape
    _, heads, count, width = keys.shape
    c = heads * width
    cfg = dict(ROT_CONFIG)
    bm = cfg.pop('BM')
    assert p % bm == 0 and width == 64 and heads % 2 == 0 and at.ndim == 2 and at.shape[1] == c and routes.shape == (n,
        r) and f.shape == (p, c)
    f32 = lambda z: z.float().contiguous()
    kr = torch.empty((n, heads, 2 * count + 2, width), device=change.device, dtype=torch.bfloat16)
    rk = torch.empty((n, heads, count), device=change.device, dtype=torch.float32)
    sq = torch.empty((n, heads), device=change.device, dtype=torch.float32)
    _rot_keys_kernel[(n * heads,)](keys.contiguous(), f32(r0), f32(cos), f32(sin), kr, rk, sq, heads, count, width,
      num_warps=4)
    out = torch.empty((n, p, heads * count), device=change.device, dtype=torch.bfloat16)
    change_desc = TensorDescriptor.from_tensor(change.contiguous().view(n * p, r), [bm, r],
      gl.NVMMASharedLayout.get_default_for([bm, r], gl.bfloat16))
    frame_desc = TensorDescriptor.from_tensor(f.contiguous(), [bm, 2 * width],
      gl.NVMMASharedLayout.get_default_for([bm, 2 * width], gl.bfloat16))
    grid = (n * (heads // 2),)
    _qf_rot_kernel[grid](change_desc, at.contiguous(), routes.contiguous(), frame_desc, f32(steps), scale, kr, rk, sq,
        f32(inv), f32(shift),
      f32(prob), out, p, c, heads, width, count, r, BM=bm, EPS_=EPS, **cfg)
    return out


@fused_query_replay_factored_rot.register_fake
def _(change, steps, scale, at, f, r0, cos, sin, inv, shift, keys, prob, routes):
    return torch.empty((*change.shape[:2], keys.shape[1] * keys.shape[2]), device=change.device, dtype=torch.bfloat16)
