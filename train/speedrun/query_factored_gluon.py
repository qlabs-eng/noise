"""Gluon (Hopper) factored query replay: the routed query replay right after a root, from the root's factors. The root
output is delta = bf16(bf16(step * frame + bf16(change @ Wo[routes])) * scale), so its raw queries delta @ w_q^T are,
up to delta's two bf16 roundings, scale * (change @ A_t + step * F[move]) with A_t = Wo[routes_t] @ w_q^T (64 x C per
token) and F = frame @ w_q^T (P x C): a K = 64 GEMM per tile (on wgmma) instead of K = C, and no delta read; several
CTAs per SM hide the replay epilogue.

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
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from query_gluon import EPS, _replay_pair, interleave_wq

# one TMA stage: 49 KB, 4 CTAs/SM (two stages, 3 CTAs: 0.78 ms); TOK 2 tokens per CTA (bitwise the same as 8, faster)
CONFIG = dict(BM=64, TOK=2, STAGES=1, num_warps=4)


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


@torch.library.custom_op('visit_kernel::fused_query_replay_factored_gluon', mutates_args=())
def fused_query_replay_factored_gluon(change: torch.Tensor, steps: torch.Tensor, scale: torch.Tensor,
    at: torch.Tensor, f: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor,
    shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """The factored replay's response weights (n, P, H * K) bf16, at / f from query_factors_t."""
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
