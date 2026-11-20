import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _issue(q_desc, h_desc, lse_desc, bar, qsm, hsm, lcsm, slot, row, pred,
           BS: gl.constexpr):
    mbarrier.expect(bar.index(slot), 2 * BS * 64 * 2 + BS * 4, pred=pred)
    tma.async_copy_global_to_shared(lse_desc, [row // BS, 0], bar.index(slot), lcsm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 0], bar.index(slot), qsm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(h_desc, [row, 0], bar.index(slot), hsm.index(slot), pred=pred)


@gluon.jit
def _raw_kernel(k_desc, q_desc, h_desc, lse_desc, Out,
                T: gl.constexpr, Window: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
                STAGES: gl.constexpr, OUT_H: gl.constexpr, num_warps: gl.constexpr):
    head = gl.program_id(0)
    t0 = gl.program_id(1) * BT
    base = head * T
    ksm = gl.allocate_shared_memory(gl.bfloat16, [BT, 64], k_desc.layout)
    qsm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], q_desc.layout)
    hsm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], q_desc.layout)
    lcsm = gl.allocate_shared_memory(gl.float32, [STAGES, 1, BS], lse_desc.layout)
    kbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(kbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    first = t0 // BS
    last = gl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    mbarrier.expect(kbar, BT * 64 * 2)
    tma.async_copy_global_to_shared(k_desc, [base + t0, 0], kbar, ksm)
    for i in gl.static_range(STAGES - 1):
        _issue(q_desc, h_desc, lse_desc, bar, qsm, hsm, lcsm, i,
               base + (first + i) * BS, first + i < last, BS)
    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                 instr_shape=[16, BS, 16])
    mma_f: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                 instr_shape=[16, 64, 16])
    a_f: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_f, k_width=2)
    target = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_s))
    cols = gl.arange(0, BS, gl.SliceLayout(0, mma_s))
    raw64 = gl.zeros([BT, 64], gl.float32, mma_f)
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    mbarrier.wait(kbar, 0)
    n = 0
    for it in range(first, last):
        slot = n % STAGES
        gl.thread_barrier()
        nxt = it + STAGES - 1
        _issue(q_desc, h_desc, lse_desc, bar, qsm, hsm, lcsm,
               (n + STAGES - 1) % STAGES, base + nxt * BS, nxt < last, BS)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        logit = warpgroup_mma(ksm, qsm.index(slot).permute((1, 0)), zero_s, use_acc=False, is_async=True)
        receiver = it * BS + cols
        lc = lcsm._reinterpret(gl.float32, [STAGES, BS], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        lse = lc.index(slot).load(gl.SliceLayout(0, mma_s))
        logit = warpgroup_mma_wait(0, deps=[logit])
        logit = logit * 0.125
        a = gl.exp2((logit - lse[None, :]) * 1.4426950408889634)
        edge = (it * BS < t0 + BT - 1) | (it * BS + BS - 1 - t0 > Window)
        if edge:
            lag = receiver[None, :] - target[:, None]
            valid = (lag >= 0) & (lag <= Window)
            a = gl.where(valid, a, 0.0)
        probability = gl.convert_layout(a.to(gl.bfloat16), a_f)
        raw64 = warpgroup_mma(probability, hsm.index(slot), raw64, is_async=True)
        raw64, probability = warpgroup_mma_wait(0, deps=[raw64, probability])
        n += 1
    mbarrier.invalidate(kbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    rows = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_f))
    dims = gl.arange(0, 64, gl.SliceLayout(0, mma_f))
    start = (head // OUT_H).to(gl.int64) * T * OUT_H * 64 + (head % OUT_H) * 64
    gl.store(Out + start + rows[:, None].to(gl.int64) * OUT_H * 64 + dims[None, :], raw64)


def raw_values(q, k, lse, hub, window, heads):
    h, t, d = q.shape
    assert d == 64 and all(z.dtype == torch.bfloat16 for z in (q, k, hub))
    bt, bs, stages = 64, 32, 2
    out = torch.empty((h // heads, t, heads * d), device=q.device, dtype=torch.float32)
    kl = gl.NVMMASharedLayout.get_default_for([bt, d], gl.bfloat16)
    ql = gl.NVMMASharedLayout.get_default_for([bs, d], gl.bfloat16)
    kd = TensorDescriptor.from_tensor(k.contiguous().view(h * t, d), [bt, d], kl)
    qd = TensorDescriptor.from_tensor(q.contiguous().view(h * t, d), [bs, d], ql)
    hd = TensorDescriptor.from_tensor(hub.contiguous().view(h * t, d), [bs, d], ql)
    ld = TensorDescriptor.from_tensor(lse.contiguous().view(h * t // bs, bs), [1, bs],
              gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2))
    _raw_kernel[(h, t // bt)](kd, qd, hd, ld, out, T=t, Window=window,
                            BT=bt, BS=bs, STAGES=stages, OUT_H=heads, num_warps=4)
    return out
