"""Gluon (Hopper) lse / center pass: moment_kernels._implicit_key_statistics_edges as a TMA + wgmma kernel.

CTA = BT queries of one head (q and hub resident in shared memory, fp32 in 32-column halves); the window's key blocks
(BS keys) stream through a STAGES-deep TMA ring of keys and values. Per block: logit = q @ keys^T / 8 and
score = hub @ values^T (tf32 wgmmas from shared memory, as the Triton kernel's tf32 dots), masked only on the blocks
the causal window cuts, then the same online max / sum / score sum. lse = max + log(sum), center = score sum / sum.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# r5base162 shapes (96 heads at window 1024 + 32 at 2048; visit-kernel/r20 bench_implicit.py): Triton edges kernel
# 0.357 ms; this kernel BS 128 0.336, + FOLD 0.312; BS 64 0.351-0.377; block n + 1's wgmmas under block n's softmax
# (PIPE): no faster
CONFIG = dict(BT=128, BS=128, STAGES=2, PIPE=0, FOLD=1)
SCALE2 = gl.constexpr(0.125 * 1.4426950408889634)  # logit scale 1 / sqrt(64) in log2 units
LN2 = gl.constexpr(0.6931471805599453)


@gluon.jit
def _issue(k_desc, v_desc, bar, ksm, vsm, slot, row, pred, BS: gl.constexpr, QK16: gl.constexpr = False):
    mbarrier.expect(bar.index(slot), 2 * BS * 64 * (2 if QK16 else 4), pred=pred)
    tma.async_copy_global_to_shared(k_desc, [row, 0], bar.index(slot), ksm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(k_desc, [row, 32], bar.index(slot), ksm.index(2 * slot + 1), pred=pred)
    tma.async_copy_global_to_shared(v_desc, [row, 0], bar.index(slot), vsm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(v_desc, [row, 32], bar.index(slot), vsm.index(2 * slot + 1), pred=pred)


@gluon.jit
def _mmas(qsm, hsm, ksm, vsm, bar, slot, phase, zero_s):
    """The block's logit and score wgmmas (async) once its keys and values have landed."""
    mbarrier.wait(bar.index(slot), phase)
    logit = warpgroup_mma(qsm.index(0), ksm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False, is_async=True)
    logit = warpgroup_mma(qsm.index(1), ksm.index(2 * slot + 1).permute((1, 0)), logit, is_async=True)
    score = warpgroup_mma(hsm.index(0), vsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False, is_async=True)
    score = warpgroup_mma(hsm.index(1), vsm.index(2 * slot + 1).permute((1, 0)), score, is_async=True)
    return logit, score


@gluon.jit
def _implicit_gluon_pipe_kernel(q_desc, h_desc, k_desc, v_desc, LSE, Center, T: gl.constexpr, Window: gl.constexpr,
    BT: gl.constexpr, BS: gl.constexpr, STAGES: gl.constexpr, num_warps: gl.constexpr, FOLD: gl.constexpr = 0):
    """_implicit_gluon_kernel with block n + 1's wgmmas in flight under block n's softmax (two accumulator pairs)."""
    tiles: gl.constexpr = T // BT
    head = gl.program_id(1)
    t0 = (tiles - 1 - gl.program_id(0)) * BT
    base = head * T
    qsm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], q_desc.layout)
    hsm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], q_desc.layout)
    ksm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], k_desc.layout)
    vsm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], k_desc.layout)
    qbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(qbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    first = gl.maximum(t0 - Window, 0) // BS
    last = (t0 + BT - 1) // BS + 1
    mbarrier.expect(qbar, 2 * BT * 64 * 4)
    tma.async_copy_global_to_shared(q_desc, [base + t0, 0], qbar, qsm.index(0))
    tma.async_copy_global_to_shared(q_desc, [base + t0, 32], qbar, qsm.index(1))
    tma.async_copy_global_to_shared(h_desc, [base + t0, 0], qbar, hsm.index(0))
    tma.async_copy_global_to_shared(h_desc, [base + t0, 32], qbar, hsm.index(1))
    for i in gl.static_range(STAGES):
        _issue(k_desc, v_desc, bar, ksm, vsm, i, base + (first + i) * BS, first + i < last, BS)

    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BS, 8])
    query = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_s))
    cols = gl.arange(0, BS, gl.SliceLayout(0, mma_s))
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    maximum = gl.full([BT], float('-inf'), gl.float32, gl.SliceLayout(1, mma_s))
    total = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    score_sum = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    mbarrier.wait(qbar, 0)
    lg_next, sc_next = _mmas(qsm, hsm, ksm, vsm, bar, 0, 0, zero_s)
    n = 0
    for it in range(first, last):
        logit, score = warpgroup_mma_wait(0, deps=[lg_next, sc_next])
        gl.thread_barrier()  # every warp's block n wgmmas are done: its slot takes block n + STAGES
        nxt = it + STAGES
        _issue(k_desc, v_desc, bar, ksm, vsm, n % STAGES, base + nxt * BS, nxt < last, BS)
        if it + 1 < last:
            lg_next, sc_next = _mmas(qsm, hsm, ksm, vsm, bar, (n + 1) % STAGES, ((n + 1) // STAGES) & 1, zero_s)
        edge = (it * BS + BS - 1 > t0) | (t0 + BT - 1 - it * BS > Window)
        if edge:
            source = it * BS + cols
            lag = query[:, None] - source[None, :]
            logit = gl.where((lag >= 0) & (lag <= Window), logit, float('-inf'))
        if FOLD:  # raw logits' max (log2 units: raw / 8 * log2 e), exp2 of one FMA per element
            next_max = gl.maximum(maximum, gl.max(logit, 1) * SCALE2)
            safe_max = gl.where(next_max == float('-inf'), 0.0, next_max)
            rescale = gl.exp2(maximum - safe_max)
            prob = gl.exp2(logit * SCALE2 - safe_max[:, None])
        else:
            logit = logit * 0.125
            next_max = gl.maximum(maximum, gl.max(logit, 1))
            safe_max = gl.where(next_max == float('-inf'), 0.0, next_max)
            rescale = gl.exp(maximum - safe_max)
            prob = gl.exp(logit - safe_max[:, None])
        total = total * rescale + gl.sum(prob, 1)
        score_sum = score_sum * rescale + gl.sum(prob * score, 1)
        maximum = next_max
        n += 1
    mbarrier.invalidate(qbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    if FOLD:
        gl.store(LSE + base + query, maximum * LN2 + gl.log(total))
    else:
        gl.store(LSE + base + query, maximum + gl.log(total))
    gl.store(Center + base + query, score_sum / total)


@gluon.jit
def _implicit_gluon_kernel(q_desc, h_desc, k_desc, v_desc, LSE, Center, T: gl.constexpr, Window: gl.constexpr,
    BT: gl.constexpr, BS: gl.constexpr, STAGES: gl.constexpr, num_warps: gl.constexpr, FOLD: gl.constexpr = 0,
    QK16: gl.constexpr = False):
    tiles: gl.constexpr = T // BT
    head = gl.program_id(1)
    t0 = (tiles - 1 - gl.program_id(0)) * BT  # the heaviest query tiles first
    base = head * T
    if QK16:  # bf16 operands (the query moments' copies): bf16 wgmmas
        qsm = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 32], q_desc.layout)
        hsm = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 32], q_desc.layout)
        ksm = gl.allocate_shared_memory(gl.bfloat16, [2 * STAGES, BS, 32], k_desc.layout)
        vsm = gl.allocate_shared_memory(gl.bfloat16, [2 * STAGES, BS, 32], k_desc.layout)
    else:
        qsm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], q_desc.layout)
        hsm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], q_desc.layout)
        ksm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], k_desc.layout)
        vsm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], k_desc.layout)
    qbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(qbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    first = gl.maximum(t0 - Window, 0) // BS
    last = (t0 + BT - 1) // BS + 1
    mbarrier.expect(qbar, 2 * BT * 64 * (2 if QK16 else 4))
    tma.async_copy_global_to_shared(q_desc, [base + t0, 0], qbar, qsm.index(0))
    tma.async_copy_global_to_shared(q_desc, [base + t0, 32], qbar, qsm.index(1))
    tma.async_copy_global_to_shared(h_desc, [base + t0, 0], qbar, hsm.index(0))
    tma.async_copy_global_to_shared(h_desc, [base + t0, 32], qbar, hsm.index(1))
    for i in gl.static_range(STAGES - 1):
        _issue(k_desc, v_desc, bar, ksm, vsm, i, base + (first + i) * BS, first + i < last, BS, QK16)

    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BS, 16 if QK16 else 8])
    query = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_s))
    cols = gl.arange(0, BS, gl.SliceLayout(0, mma_s))
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    maximum = gl.full([BT], float('-inf'), gl.float32, gl.SliceLayout(1, mma_s))
    total = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    score_sum = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    mbarrier.wait(qbar, 0)
    n = 0
    for it in range(first, last):
        slot = n % STAGES
        gl.thread_barrier()  # every warp is done with the slot refilled now
        nxt = it + STAGES - 1
        _issue(k_desc, v_desc, bar, ksm, vsm, (n + STAGES - 1) % STAGES, base + nxt * BS, nxt < last, BS, QK16)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        logit = warpgroup_mma(qsm.index(0), ksm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)
        logit = warpgroup_mma(qsm.index(1), ksm.index(2 * slot + 1).permute((1, 0)), logit, is_async=True)
        score = warpgroup_mma(hsm.index(0), vsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)
        score = warpgroup_mma(hsm.index(1), vsm.index(2 * slot + 1).permute((1, 0)), score, is_async=True)
        logit, score = warpgroup_mma_wait(0, deps=[logit, score])
        edge = (it * BS + BS - 1 > t0) | (t0 + BT - 1 - it * BS > Window)
        if edge:
            source = it * BS + cols
            lag = query[:, None] - source[None, :]
            logit = gl.where((lag >= 0) & (lag <= Window), logit, float('-inf'))
        if FOLD:  # raw logits' max (log2 units: raw / 8 * log2 e), exp2 of one FMA per element
            next_max = gl.maximum(maximum, gl.max(logit, 1) * SCALE2)
            safe_max = gl.where(next_max == float('-inf'), 0.0, next_max)
            rescale = gl.exp2(maximum - safe_max)
            prob = gl.exp2(logit * SCALE2 - safe_max[:, None])
        else:
            logit = logit * 0.125
            next_max = gl.maximum(maximum, gl.max(logit, 1))
            safe_max = gl.where(next_max == float('-inf'), 0.0, next_max)
            rescale = gl.exp(maximum - safe_max)
            prob = gl.exp(logit - safe_max[:, None])
        total = total * rescale + gl.sum(prob, 1)
        score_sum = score_sum * rescale + gl.sum(prob * score, 1)
        maximum = next_max
        n += 1
    mbarrier.invalidate(qbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    if FOLD:
        gl.store(LSE + base + query, maximum * LN2 + gl.log(total))
    else:
        gl.store(LSE + base + query, maximum + gl.log(total))
    gl.store(Center + base + query, score_sum / total)


def implicit_key_context_gluon(q, k, values, hub, window, **over):
    """moment_kernels.implicit_key_context_edges' outputs (lse, center), (heads, tokens) fp32."""
    h, t, d = q.shape
    assert d == 64
    cfg = dict(CONFIG, **over)
    bt, bs = cfg['BT'], cfg['BS']
    assert t % bt == 0 and t % bs == 0 and window % bs == 0
    lse = torch.empty((h, t), device=q.device, dtype=torch.float32)
    center = torch.empty_like(lse)
    qk16 = all(z.dtype == torch.bfloat16 for z in (q, k, values, hub))  # (the query moments' bf16 copies)
    assert qk16 or not cfg.get('PIPE', 1), 'bf16 operands: the plain kernel only'
    f32 = (lambda z: z.contiguous().view(h * t, 64)) if qk16 else (lambda z: z.float().contiguous().view(h * t, 64))
    qk_gl = gl.bfloat16 if qk16 else gl.float32
    lay_q = gl.NVMMASharedLayout.get_default_for([bt, 32], qk_gl)
    lay_k = gl.NVMMASharedLayout.get_default_for([bs, 32], qk_gl)
    q_desc = TensorDescriptor.from_tensor(f32(q), [bt, 32], lay_q)
    h_desc = TensorDescriptor.from_tensor(f32(hub), [bt, 32], lay_q)
    k_desc = TensorDescriptor.from_tensor(f32(k), [bs, 32], lay_k)
    v_desc = TensorDescriptor.from_tensor(f32(values), [bs, 32], lay_k)
    kernel = _implicit_gluon_pipe_kernel if cfg.get('PIPE', 1) else _implicit_gluon_kernel
    kernel[(t // bt, h)](q_desc, h_desc, k_desc, v_desc, lse, center, T=t, Window=window, BT=bt, BS=bs,
      STAGES=cfg['STAGES'], num_warps=bt // 16, FOLD=cfg.get('FOLD', 0), **({'QK16': True} if qk16 else {}))
    return lse, center


@torch.library.custom_op('implicit_gluon::implicit_key_context', mutates_args=())
def implicit_key_context(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, hub: torch.Tensor,
                         window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """implicit_key_context_gluon as a custom op."""
    return implicit_key_context_gluon(q, k, values, hub, window)


@implicit_key_context.register_fake
def _(q, k, values, hub, window):
    h, t, _ = q.shape
    return q.new_empty((h, t), dtype=torch.float32), q.new_empty((h, t), dtype=torch.float32)
