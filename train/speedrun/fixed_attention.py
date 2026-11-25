"""The clean forward's attention as one kernel: causal softmax attention over a left window (utils/model.py's
`attention`: a query at s sees keys s - window to s) for (rows, 2048, 8, 64) bf16 q, k and v.

A program takes 64 queries of one head and walks the 64-key tiles its window reaches, keeping a running maximum,
sum and weighted value sum (online softmax, scores in log2 units). Only the tiles at the window's two edges are
masked; the tiles between them are inside every query's window.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _load_pair(keys, values, barrier, ksm, vsm, slot, row, col, active, BS: gl.constexpr):
    mbarrier.expect(barrier.index(slot), 2 * BS * 64 * 2, pred=active)
    tma.async_copy_global_to_shared(keys, [row, col], barrier.index(slot), ksm.index(slot), pred=active)
    tma.async_copy_global_to_shared(values, [row, col], barrier.index(slot), vsm.index(slot), pred=active)


@gluon.jit
def _accumulate(scores, values, old_max, numer, denom, qpos, kpos, start, block, W: gl.constexpr,
                BQ: gl.constexpr, BS: gl.constexpr, v_left: gl.constexpr):
    logits = scores * (0.125 * 1.4426950408889634)
    edge = (block * BS + BS - 1 > start) | (start + BQ - 1 - block * BS > W)
    if edge:
        lag = qpos[:, None] - (block * BS + kpos)[None, :]
        logits = gl.where((lag >= 0) & (lag <= W), logits, -float('inf'))
    new_max = gl.maximum(old_max, gl.max(logits, axis=1))
    pivot = gl.where(new_max == -float('inf'), 0., new_max)
    if edge:
        exponent = logits - pivot[:, None]
    else:
        exponent = gl.fma(scores, 0.125 * 1.4426950408889634, -pivot[:, None])
    mass = gl.exp2(exponent)
    mass_sum = gl.sum(mass, axis=1)
    alpha = gl.exp2(old_max - pivot)
    denom = denom * alpha + mass_sum
    numer = numer * alpha[:, None]
    prob = gl.convert_layout(mass.to(gl.bfloat16), v_left)
    numer = warpgroup_mma(prob, values, numer, is_async=True)
    return numer, denom, new_max, prob


@gluon.jit
def _direct_attention(queries, keys, values, Out, T: gl.constexpr, W: gl.constexpr,
                      BQ: gl.constexpr, BS: gl.constexpr, STAGES: gl.constexpr):
    head = gl.program_id(0)
    tile = T // BQ - 1 - gl.program_id(1)
    start = tile * BQ
    base = head // 8 * T
    channel = (head % 8) * 64
    first = gl.maximum(0, (start - W) // BS)
    stop = (start + BQ - 1) // BS + 1

    qsm = gl.allocate_shared_memory(gl.bfloat16, [BQ, 64], queries.layout)
    ksm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], keys.layout)
    vsm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], values.layout)
    qbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(qbar, count=1)
    for j in gl.static_range(STAGES):
        mbarrier.init(bar.index(j), count=1)
    mbarrier.expect(qbar, BQ * 64 * 2)
    tma.async_copy_global_to_shared(queries, [base + start, channel], qbar, qsm)
    for j in gl.static_range(STAGES - 1):
        _load_pair(keys, values, bar, ksm, vsm, j, base + (first + j) * BS, channel, first + j < stop, BS)

    s_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BS, 16])
    o_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, 64, 16])
    v_left: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_layout, k_width=2)
    qpos = start + gl.arange(0, BQ, gl.SliceLayout(1, s_layout))
    kpos = gl.arange(0, BS, gl.SliceLayout(0, s_layout))
    zero = gl.full([BQ, BS], 0, gl.float32, s_layout)
    denom = gl.full([BQ], 0, gl.float32, gl.SliceLayout(1, s_layout))
    numer = gl.full([BQ, 64], 0, gl.float32, o_layout)
    old_max = gl.full([BQ], -float('inf'), gl.float32, gl.SliceLayout(1, s_layout))
    mbarrier.wait(qbar, 0)
    mbarrier.wait(bar.index(0), 0)
    scores = warpgroup_mma(qsm, ksm.index(0).permute((1, 0)), zero, use_acc=False, is_async=True)
    scores = warpgroup_mma_wait(0, deps=[scores])
    index = 0
    for block in range(first, stop - 1):
        slot = index % STAGES
        nxt = (index + 1) % STAGES
        upcoming = block + STAGES - 1
        gl.thread_barrier()
        _load_pair(keys, values, bar, ksm, vsm, (index + STAGES - 1) % STAGES,
                   base + upcoming * BS, channel, upcoming < stop, BS)
        mbarrier.wait(bar.index(nxt), ((index + 1) // STAGES) & 1)
        following = warpgroup_mma(qsm, ksm.index(nxt).permute((1, 0)), zero, use_acc=False, is_async=True)
        numer, denom, old_max, prob = _accumulate(scores, vsm.index(slot), old_max, numer, denom, qpos, kpos,
                                         start, block, W, BQ, BS, v_left)
        scores, numer, prob = warpgroup_mma_wait(0, deps=[following, numer, prob])
        index += 1
    numer, denom, old_max, prob = _accumulate(scores, vsm.index(index % STAGES), old_max, numer, denom, qpos, kpos,
                                     start, stop - 1, W, BQ, BS, v_left)
    numer, prob = warpgroup_mma_wait(0, deps=[numer, prob])
    mbarrier.invalidate(qbar)
    for j in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(j))
    row = base + start + gl.arange(0, BQ, gl.SliceLayout(1, o_layout))
    col = channel + gl.arange(0, 64, gl.SliceLayout(0, o_layout))
    den = gl.convert_layout(denom, gl.SliceLayout(1, o_layout))
    gl.store(Out + row[:, None] * 512 + col[None, :], (numer / den[:, None]).to(gl.bfloat16))


@torch.library.custom_op('fixed_attention::attention', mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    b, t, h, d = q.shape
    bq, bs = 64, 64
    q_layout = gl.NVMMASharedLayout.get_default_for([bq, d], gl.bfloat16)
    kv_layout = gl.NVMMASharedLayout.get_default_for([bs, d], gl.bfloat16)
    qdesc = TensorDescriptor.from_tensor(q.reshape(b * t, h * d), [bq, d], q_layout)
    kdesc = TensorDescriptor.from_tensor(k.reshape(b * t, h * d), [bs, d], kv_layout)
    vdesc = TensorDescriptor.from_tensor(v.reshape(b * t, h * d), [bs, d], kv_layout)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _direct_attention[(b * h, t // bq)](qdesc, kdesc, vdesc, output, t, window, bq, bs, 3, num_warps=4)
    return output


@attention.register_fake
def _(q, k, v, window):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)
