"""Gluon (Hopper) factored query replay: query_factored.fused_query_replay_factored's math with the K = 64 GEMM on
wgmma and several CTAs per SM to hide the replay epilogue.

A CTA (one warpgroup) takes BM = 64 moves x one head pair for TOK consecutive tokens: per token the change tile
(BM x 64) and the token's transposed factor slice A_t^T (2 D x 64, w_q's rows interleaved per head so each RoPE pair
sits in one thread) come in by TMA (two stages), raw = scale * (change @ A_t + step * F) with F's tile kept in shared
memory for all the CTA's tokens, then query_gluon's replay epilogue (both heads' key scores in one wgmma against a
block-diagonal key matrix). ~72 KB of shared memory: three CTAs per SM.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from query_gluon import EPS, _replay_pair, interleave_wq
from visit_chain import _bf16, _rbf16

CONFIG = dict(BM=64, TOK=8, STAGES=1, num_warps=4)  # one TMA stage: 49 KB, 4 CTAs/SM (two stages, 3 CTAs: 0.78 ms)
_SMS = {}


@gluon.jit
def _issue(change_desc, at_desc, bar, csm, asm, s, token, pair, pos0, P: gl.constexpr, C: gl.constexpr,
    R: gl.constexpr, BM: gl.constexpr, WN: gl.constexpr):
    """The token's change tile and its A_t^T slice for the head pair into stage s."""
    mbarrier.expect(bar.index(s), (BM + WN) * R * 2)
    tma.async_copy_global_to_shared(change_desc, [token * P + pos0, 0], bar.index(s), csm.index(s))
    tma.async_copy_global_to_shared(at_desc, [token * C + pair * WN, 0], bar.index(s), asm.index(s))


@gluon.jit
def _qf_kernel(change_desc, at_desc, F, Steps, Scale, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out, N,
    P: gl.constexpr, C: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, R: gl.constexpr,
    BM: gl.constexpr, TOK: gl.constexpr, EPS_: gl.constexpr, num_warps: gl.constexpr, STAGES: gl.constexpr = 2,
    FGLOBAL: gl.constexpr = False):
    HP: gl.constexpr = H // 2
    MT: gl.constexpr = P // BM
    WN: gl.constexpr = 2 * D
    gl.static_assert(BM == 16 * num_warps)  # one warpgroup per 64 moves
    pid = gl.program_id(0)
    pair = pid % HP
    pos0 = ((pid // HP) % MT) * BM
    t0 = (pid // (HP * MT)) * TOK
    t1 = gl.minimum(t0 + TOK, N)

    csm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, R], change_desc.layout)
    asm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, WN, R], at_desc.layout)
    kb = gl.allocate_shared_memory(gl.bfloat16, [2 * K, WN],
      gl.NVMMASharedLayout.get_default_for([2 * K, WN], gl.bfloat16))
    fsm = gl.allocate_shared_memory(gl.bfloat16, [1 if FGLOBAL else BM, WN],
      gl.NVMMASharedLayout.get_default_for([BM, WN], gl.bfloat16))
    bar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(bar.index(i), count=1)
    _issue(change_desc, at_desc, bar, csm, asm, 0, t0, pair, pos0, P, C, R, BM, WN)

    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, WN, 16])
    mmak: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 2 * K, 16])
    bl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    fr = gl.arange(0, BM, gl.SliceLayout(1, bl))
    fc = gl.arange(0, WN, gl.SliceLayout(0, bl))
    if not FGLOBAL:  # F's tile in shared memory, for every token
        fsm.store(gl.load(F + (pos0 + fr)[:, None] * C + (pair * WN + fc)[None, :]))
    f_off = (pos0 + gl.arange(0, BM, gl.SliceLayout(1, mma)))[:, None] * C + \
      (pair * WN + gl.arange(0, WN, gl.SliceLayout(0, mma)))[None, :]
    kr = gl.arange(0, 2 * K, gl.SliceLayout(1, bl))
    kdiag = (kr[:, None] // K) == (fc[None, :] // D)
    kcol = (fc % D) // 2 + (fc % 2) * (D // 2)  # interleaved column 2 i + e = key column 32 e + i
    scale = gl.load(Scale).to(gl.float32)
    zero = gl.zeros([BM, WN], gl.float32, mma)

    for token in range(t0, t1):
        ti = token - t0
        s = ti % STAGES
        gl.thread_barrier()  # every warp is done with the previous token's stage and with kb
        if STAGES == 2:
            if token + 1 < t1:
                _issue(change_desc, at_desc, bar, csm, asm, 1 - s, token + 1, pair, pos0, P, C, R, BM, WN)
        elif ti > 0:  # one stage: this token's tiles now (other CTAs on the SM hide the wait)
            _issue(change_desc, at_desc, bar, csm, asm, 0, token, pair, pos0, P, C, R, BM, WN)
        kv = gl.load(Keys + ((token * H + 2 * pair + kr // K) * K + kr % K)[:, None] * D + kcol[None, :],
          mask=kdiag, other=0.0)
        kb.store(kv)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.wait(bar.index(s), (ti // STAGES) & 1)
        acc = warpgroup_mma(csm.index(s), asm.index(s).permute((1, 0)), zero, use_acc=False, is_async=True)
        acc = warpgroup_mma_wait(0, deps=[acc])
        step = gl.load(Steps + token)
        if FGLOBAL:  # (from L2)
            frame_q = gl.load(F + f_off).to(gl.float32)
        else:
            frame_q = fsm.load(mma).to(gl.float32)
        raw = scale * (acc + step * frame_q)
        _replay_pair(raw, kb, token, pair, token * P + pos0, pos0, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out, P, H,
          D, K, BM, EPS_, num_warps, mmak)
    for i in gl.static_range(2):
        mbarrier.invalidate(bar.index(i))


@gluon.jit
def _head_keys_issue(kh, kl, rh, rl, kt, k_off, r_off, rmask, K: gl.constexpr, D: gl.constexpr,
    LO: gl.constexpr = True):
    """cp.async of a token's rotated keys / r0' (KR rows at kt) for both heads of the pair: kh / kl rows 16 h + key,
    rh / rl row 8 h (the other rows stay zero)."""
    cp.async_copy_global_to_shared(kh, kt + k_off)
    if LO:
        cp.async_copy_global_to_shared(kl, kt + K * D + k_off)
    cp.async_copy_global_to_shared(rh, kt + r_off, mask=rmask)
    if LO:
        cp.async_copy_global_to_shared(rl, kt + D + r_off, mask=rmask)
    cp.commit_group()


@gluon.jit
def _head_scores(a, kh, kl, rh, rl, h: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, mmak: gl.constexpr,
    mmad: gl.constexpr, LO: gl.constexpr = True):
    """Head h's raw . k' (hi + lo parts accumulated) and raw . r0' (column 0) wgmmas, in flight."""
    ak = gl.convert_layout(a, gl.DotOperandLayout(operand_index=0, parent=mmak, k_width=2))
    ad = gl.convert_layout(a, gl.DotOperandLayout(operand_index=0, parent=mmad, k_width=2))
    sc = warpgroup_mma(ak, kh.slice(h * K, K).permute((1, 0)), gl.zeros([BM, K], gl.float32, mmak), use_acc=False,
      is_async=True)
    if LO:
        sc = warpgroup_mma(ak, kl.slice(h * K, K).permute((1, 0)), sc, is_async=True)
    q = warpgroup_mma(ad, rh.slice(h * 8, 8).permute((1, 0)), gl.zeros([BM, 8], gl.float32, mmad), use_acc=False,
      is_async=True)
    if LO:
        q = warpgroup_mma(ad, rl.slice(h * 8, 8).permute((1, 0)), q, is_async=True)
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
def _qf_rot_kernel(change_desc, at_desc, F, Steps, Scale, KR, RK, SQ, Inv, Shift, Prob, Out, N,
    P: gl.constexpr, C: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, R: gl.constexpr,
    BM: gl.constexpr, TOK: gl.constexpr, EPS_: gl.constexpr, num_warps: gl.constexpr, LO: gl.constexpr = True):
    """_qf_kernel's tiles with the scores against rotated keys: with x = rope(raw), q = x g + shift0 r0 and
    dq = q inv + r0 shift, a key's score is dq . k = g inv (raw . k') + (shift0 inv + shift) (r0 . k), k' the key
    rotated back (rope is orthogonal), and q's norm sums follow from raw . r0', |raw|^2 and |r0|^2. The per-element
    RoPE, scaling and norm sums give way to small wgmmas of each head's bf16 raw queries (split per head in
    registers) against its rotated keys and r0' (_rot_keys_kernel's KR rows: K hi, K lo, r0' hi, r0' lo; bf16 hi /
    lo parts, by cp.async); dq is not rounded to bf16. The next token's change / A_t^T (TMA) and keys (cp.async)
    load under this one's epilogue. 53 KB of shared memory, 128 registers: four CTAs per SM."""
    HP: gl.constexpr = H // 2
    MT: gl.constexpr = P // BM
    WN: gl.constexpr = 2 * D
    KR_ROWS: gl.constexpr = 2 * K + 2
    gl.static_assert(BM == 16 * num_warps)
    pid = gl.program_id(0)
    pair = pid % HP
    pos0 = ((pid // HP) % MT) * BM
    t0 = (pid // (HP * MT)) * TOK
    t1 = gl.minimum(t0 + TOK, N)

    csm = gl.allocate_shared_memory(gl.bfloat16, [1, BM, R], change_desc.layout)
    asm = gl.allocate_shared_memory(gl.bfloat16, [1, WN, R], at_desc.layout)
    klay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([2 * K, D], gl.bfloat16)
    rlay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([16, D], gl.bfloat16)
    kh = gl.allocate_shared_memory(gl.bfloat16, [2 * K, D], klay)
    kl = gl.allocate_shared_memory(gl.bfloat16, [2 * K, D], klay)
    rh = gl.allocate_shared_memory(gl.bfloat16, [16, D], rlay)
    rl = gl.allocate_shared_memory(gl.bfloat16, [16, D], rlay)
    fsm = gl.allocate_shared_memory(gl.bfloat16, [BM, WN], gl.NVMMASharedLayout.get_default_for([BM, WN], gl.bfloat16))
    bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    mbarrier.init(bar.index(0), count=1)
    _issue(change_desc, at_desc, bar, csm, asm, 0, t0, pair, pos0, P, C, R, BM, WN)

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
    rh.store(gl.zeros([16, D], gl.bfloat16, bl))  # rows 8 h + 1.. stay zero
    rl.store(gl.zeros([16, D], gl.bfloat16, bl))
    gl.thread_barrier()
    _head_keys_issue(kh, kl, rh, rl, KR + t0 * (H * KR_ROWS * D), k_off, r_off, rmask, K, D, LO)
    fr = gl.arange(0, BM, gl.SliceLayout(1, bl))
    fc = gl.arange(0, WN, gl.SliceLayout(0, bl))
    fsm.store(gl.load(F + (pos0 + fr)[:, None] * C + (pair * WN + fc)[None, :]))
    scale = gl.load(Scale).to(gl.float32)
    zero = gl.zeros([BM, WN], gl.float32, mma)
    rows = gl.arange(0, BM, RL)
    kk = gl.arange(0, K, KLY)

    for token in range(t0, t1):
        ti = token - t0
        step = gl.load(Steps + token)
        cp.wait_group(0)
        fence_async_shared()
        gl.thread_barrier()  # the token's keys (and F's tile) in every warp's view
        mbarrier.wait(bar.index(0), ti & 1)
        acc = warpgroup_mma(csm.index(0), asm.index(0).permute((1, 0)), zero, use_acc=False, is_async=True)
        acc = warpgroup_mma_wait(0, deps=[acc])
        gl.thread_barrier()  # every warp is done with the change / A_t^T tiles
        if token + 1 < t1:
            _issue(change_desc, at_desc, bar, csm, asm, 0, token + 1, pair, pos0, P, C, R, BM, WN)
        raw = scale * (acc + step * fsm.load(mma).to(gl.float32))
        x0, x1 = gl.split(gl.permute(gl.reshape(_bf16(raw), [BM, 2, D]), (0, 2, 1)))  # per head
        s0, q0, a0, d0 = _head_scores(x0, kh, kl, rh, rl, 0, K, BM, mmak, mmad, LO)
        s1, q1, a1, d1 = _head_scores(x1, kh, kl, rh, rl, 1, K, BM, mmak, mmad, LO)
        prow = token * P + pos0 + rows
        g = gl.load(Inv + prow)
        shift0 = gl.load(Shift + prow)
        hk = (token * H + 2 * pair) * K + kk
        rk0 = gl.load(RK + hk)
        rk1 = gl.load(RK + K + hk)
        pr0 = gl.load(Prob + hk)
        pr1 = gl.load(Prob + K + hk)
        sq0 = gl.load(SQ + token * H + 2 * pair)
        sq1 = gl.load(SQ + token * H + 2 * pair + 1)
        f0 = x0.to(gl.float32)
        f1 = x1.to(gl.float32)
        m0 = gl.convert_layout(gl.sum(f0 * f0, 1), RL)  # |raw|^2 per (move, head) = |rope(raw)|^2
        m1 = gl.convert_layout(gl.sum(f1 * f1, 1), RL)
        s0, q0, s1, q1 = warpgroup_mma_wait(0, deps=[s0, q0, s1, q1, a0, d0, a1, d1])[:4]
        gl.thread_barrier()  # every warp is done with the keys
        if token + 1 < t1:
            _head_keys_issue(kh, kl, rh, rl, KR + (token + 1) * (H * KR_ROWS * D), k_off, r_off, rmask, K, D, LO)
        orow = gl.expand_dims(prow.to(gl.int64) * (H * K), 1) + (2 * pair * K + kk)[None, :]
        _head_weights(s0, q0, m0, g, shift0, sq0, rk0, pr0, Out + orow, D, EPS_)
        _head_weights(s1, q1, m1, g, shift0, sq1, rk1, pr1, Out + orow + K, D, EPS_)
    mbarrier.invalidate(bar.index(0))


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


ROT_CONFIG = dict(BM=64, TOK=8, num_warps=4, maxnreg=128, LO=True)  # no spills at 128; LO False (run.py's QF_KEYS_HI):
# the rotated keys' / r0's bf16 high parts only (not bitwise)


@triton.jit
def _factor_at_kernel(Wo, Routes, Wq, At, M, C: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr):
    """At[t, c, r] = (w_q @ Wo[routes[t, r]]^T)[c] (bf16, fp32 accumulation): the gathered rows' GEMM, stored
    transposed per token (r contiguous)."""
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


FACTOR_CONFIG = dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3)


@torch.library.custom_op('visit_kernel::query_factors_t', mutates_args=())
def query_factors_t(wo: torch.Tensor, routes: torch.Tensor, basis: torch.Tensor,
    w_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The Gluon replay's factors, w_q's rows interleaved per head (heads = C / 64): At (n, C, R) bf16 = the token's
    (Wo[routes_t] @ w_q^T)^T, F (P, C) bf16 = frame @ w_q^T (fp32)."""
    c = w_q.shape[0]
    wq = interleave_wq(w_q, c // 64)
    n, r = routes.shape
    d = wq.shape[1]
    at = torch.empty((n, c, r), device=wo.device, dtype=torch.bfloat16)
    cfg = dict(FACTOR_CONFIG)
    assert c % cfg['BN'] == 0
    _factor_at_kernel[(triton.cdiv(n * r, cfg['BM']), c // cfg['BN'])](wo.contiguous(), routes.contiguous(), wq, at,
      n * r, c, d, r, **cfg)
    f = (basis.float() @ wq.float().T).to(torch.bfloat16)
    return at, f


@query_factors_t.register_fake
def _(wo, routes, basis, w_q):
    return (torch.empty((routes.shape[0], w_q.shape[0], routes.shape[1]), device=wo.device, dtype=torch.bfloat16),
      torch.empty((basis.shape[0], w_q.shape[0]), device=wo.device, dtype=torch.bfloat16))


@triton.jit
def _gather_at_kernel(Prod, Routes, At, C: tl.constexpr, R: tl.constexpr, BC: tl.constexpr):
    """At[t, c, r] = Prod[routes[t, r], c]: token t's rows of the full product, stored transposed (r contiguous)."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    rr = tl.arange(0, R)
    chosen = tl.load(Routes + token * R + rr).to(tl.int64)
    tile = tl.load(Prod + chosen[:, None] * C + cols[None, :])
    tl.store(At + (token * C + cols[None, :]) * R + rr[:, None], tile)


@torch.library.custom_op('visit_kernel::query_factors_t_gathered', mutates_args=())
def query_factors_t_gathered(wo: torch.Tensor, routes: torch.Tensor, basis: torch.Tensor,
    w_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """query_factors_t with At gathered from the full product Wo @ w_q^T (every hidden row once, by
    _factor_at_kernel's own tiles and accumulation order, so bitwise) instead of one GEMM row per routed row."""
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
    at = torch.empty((n, c, r), device=wo.device, dtype=torch.bfloat16)
    _gather_at_kernel[(n, c // 64)](prod, routes.contiguous(), at, c, r, BC=64, num_warps=4)
    f = (basis.float() @ wq.float().T).to(torch.bfloat16)
    return at, f


@query_factors_t_gathered.register_fake
def _(wo, routes, basis, w_q):
    return (torch.empty((routes.shape[0], w_q.shape[0], routes.shape[1]), device=wo.device, dtype=torch.bfloat16),
      torch.empty((basis.shape[0], w_q.shape[0]), device=wo.device, dtype=torch.bfloat16))


@torch.library.custom_op('visit_kernel::fused_query_replay_factored_gluon', mutates_args=())
def fused_query_replay_factored_gluon(change: torch.Tensor, steps: torch.Tensor, scale: torch.Tensor,
    at: torch.Tensor, f: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor,
    shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """query_factored.fused_query_replay_factored with at / f from query_factors_t -> weights (n, P, H * K) bf16."""
    n, p, r = change.shape
    _, heads, count, width = keys.shape
    c = heads * width
    cfg = dict(CONFIG)
    bm, tok = cfg.pop('BM'), cfg.pop('TOK')
    assert p % bm == 0 and width == 64 and heads % 2 == 0 and at.shape == (n, c, r) and f.shape == (p, c)
    out = torch.empty((n, p, heads * count), device=change.device, dtype=torch.bfloat16)
    f32 = lambda z: z.float().contiguous()
    change_desc = TensorDescriptor.from_tensor(change.contiguous().view(n * p, r), [bm, r],
      gl.NVMMASharedLayout.get_default_for([bm, r], gl.bfloat16))
    at_desc = TensorDescriptor.from_tensor(at.contiguous().view(n * c, r), [2 * width, r],
      gl.NVMMASharedLayout.get_default_for([2 * width, r], gl.bfloat16))
    grid = (triton.cdiv(n, tok) * (p // bm) * (heads // 2),)
    _qf_kernel[grid](change_desc, at_desc, f.contiguous(), f32(steps), scale, f32(r0), f32(cos), f32(sin), f32(inv),
      f32(shift), keys.contiguous(), f32(prob), out, n, p, c, heads, width, count, r, BM=bm, TOK=tok, EPS_=EPS,
      **cfg)
    return out


@fused_query_replay_factored_gluon.register_fake
def _(change, steps, scale, at, f, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((*change.shape[:2], keys.shape[1] * keys.shape[2]), device=change.device, dtype=torch.bfloat16)


@torch.library.custom_op('visit_kernel::fused_query_replay_factored_rot', mutates_args=())
def fused_query_replay_factored_rot(change: torch.Tensor, steps: torch.Tensor, scale: torch.Tensor,
    at: torch.Tensor, f: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor,
    shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """fused_query_replay_factored_gluon's API by _qf_rot_kernel (the scores against rotated keys, no per-element
    RoPE; dq unrounded, one reciprocal per row): one _rot_keys_kernel pass over the keys, then the replay."""
    n, p, r = change.shape
    _, heads, count, width = keys.shape
    c = heads * width
    cfg = dict(ROT_CONFIG)
    bm, tok = cfg.pop('BM'), cfg.pop('TOK')
    assert p % bm == 0 and width == 64 and heads % 2 == 0 and at.shape == (n, c, r) and f.shape == (p, c)
    f32 = lambda z: z.float().contiguous()
    kr = torch.empty((n, heads, 2 * count + 2, width), device=change.device, dtype=torch.bfloat16)
    rk = torch.empty((n, heads, count), device=change.device, dtype=torch.float32)
    sq = torch.empty((n, heads), device=change.device, dtype=torch.float32)
    _rot_keys_kernel[(n * heads,)](keys.contiguous(), f32(r0), f32(cos), f32(sin), kr, rk, sq, heads, count, width,
      num_warps=4)
    out = torch.empty((n, p, heads * count), device=change.device, dtype=torch.bfloat16)
    change_desc = TensorDescriptor.from_tensor(change.contiguous().view(n * p, r), [bm, r],
      gl.NVMMASharedLayout.get_default_for([bm, r], gl.bfloat16))
    at_desc = TensorDescriptor.from_tensor(at.contiguous().view(n * c, r), [2 * width, r],
      gl.NVMMASharedLayout.get_default_for([2 * width, r], gl.bfloat16))
    grid = (triton.cdiv(n, tok) * (p // bm) * (heads // 2),)
    _qf_rot_kernel[grid](change_desc, at_desc, f.contiguous(), f32(steps), scale, kr, rk, sq, f32(inv), f32(shift),
      f32(prob), out, n, p, c, heads, width, count, r, BM=bm, TOK=tok, EPS_=EPS, **cfg)
    return out


@fused_query_replay_factored_rot.register_fake
def _(change, steps, scale, at, f, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((*change.shape[:2], keys.shape[1] * keys.shape[2]), device=change.device, dtype=torch.bfloat16)
