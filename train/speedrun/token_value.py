"""The Cut's value credit pass as a Gluon (Hopper) kernel: each key's sum over its window's queries of attention
probability times the source row, with the gated credits of the value tables added per vocabulary row."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

CONFIG = dict(BT=128, BS=64, STAGES=2)
LOG2E = gl.constexpr(1.4426950408889634)


@gluon.jit
def _issue(q_desc, h_desc, bar, qsm, hsm, slot, row, source_row, column, pred, BS: gl.constexpr,
           lse_desc, lsesm):
    mbarrier.expect(bar.index(slot), BS * 64 * 2 + BS * 64 * 2 + BS * 4, pred=pred)
    # the row of BS receivers' lse ((h T / BS, BS) view)
    tma.async_copy_global_to_shared(lse_desc, [row // BS, 0], bar.index(slot), lsesm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 0], bar.index(slot), qsm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 32], bar.index(slot), qsm.index(2 * slot + 1), pred=pred)
    tma.async_copy_global_to_shared(h_desc, [source_row, column], bar.index(slot), hsm.index(slot), pred=pred)


@gluon.jit
def _value_gluon_kernel(k_desc, q_desc, h_desc, Out, OutMap, OUT_LAYER, OUT_TOKEN, lse_desc,
    T: gl.constexpr, Window: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr, STAGES: gl.constexpr,
    OUT_H: gl.constexpr, num_warps: gl.constexpr,
    Gates, Ids, Lexical, ROWS: gl.constexpr, VOCAB: gl.constexpr):
    head = gl.program_id(0)
    tile = gl.program_id(1)
    t0 = tile * BT
    base = head * T
    source_base = gl.load(OutMap + head // OUT_H) * T
    source_column = (head % OUT_H) * 64
    ksm = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 32], k_desc.layout)
    qsm = gl.allocate_shared_memory(gl.bfloat16, [2 * STAGES, BS, 32], q_desc.layout)
    hsm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], h_desc.layout)
    lsesm = gl.allocate_shared_memory(gl.float32, [STAGES, 1, BS], lse_desc.layout)
    kbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(kbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    first = t0 // BS
    last = gl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    mbarrier.expect(kbar, BT * 64 * 2)
    tma.async_copy_global_to_shared(k_desc, [base + t0, 0], kbar, ksm.index(0))
    tma.async_copy_global_to_shared(k_desc, [base + t0, 32], kbar, ksm.index(1))
    for i in gl.static_range(STAGES - 1):
        _issue(q_desc, h_desc, bar, qsm, hsm, i, base + (first + i) * BS,
               source_base + (first + i) * BS, source_column, first + i < last, BS, lse_desc, lsesm)

    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BS, 16])
    mma_o: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 64, 16])
    p_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_o, k_width=2)
    target = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_s))
    cols = gl.arange(0, BS, gl.SliceLayout(0, mma_s))
    acc = gl.zeros([BT, 64], gl.float32, mma_o)
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    mbarrier.wait(kbar, 0)
    n = 0
    for it in range(first, last):
        slot = n % STAGES
        # every warp is done with the slot refilled now (its previous block's wgmmas were waited)
        gl.thread_barrier()
        nxt = it + STAGES - 1
        _issue(q_desc, h_desc, bar, qsm, hsm, (n + STAGES - 1) % STAGES, base + nxt * BS,
               source_base + nxt * BS, source_column, nxt < last, BS, lse_desc, lsesm)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        s = warpgroup_mma(ksm.index(0), qsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)
        s = warpgroup_mma(ksm.index(1), qsm.index(2 * slot + 1).permute((1, 0)), s, is_async=True)
        receiver = it * BS + cols
        lse = lsesm._reinterpret(gl.float32, [STAGES, BS], gl.SwizzledSharedLayout(1, 1, 1, [0])).index(
          slot).load(gl.SliceLayout(0, mma_s))
        s = warpgroup_mma_wait(0, deps=[s])
        x = s * (0.125 * LOG2E) - (lse * LOG2E)[None, :]
        edge = (it * BS < t0 + BT - 1) | (it * BS + BS - 1 - t0 > Window)
        if edge:
            lag = receiver[None, :] - target[:, None]
            prob = gl.where((lag >= 0) & (lag <= Window), gl.exp2(x), 0.0)
        else:
            prob = gl.exp2(x)
        p = gl.convert_layout(prob.to(gl.bfloat16), p_layout)
        acc = warpgroup_mma(p, hsm.index(slot), acc, is_async=True)
        acc, p = warpgroup_mma_wait(0, deps=[acc, p])
        n += 1
    mbarrier.invalidate(kbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    layer = head // OUT_H
    layer = gl.load(OutMap + layer)
    start = layer.to(gl.int64) * OUT_LAYER + (head % OUT_H) * 64
    rows = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_o))
    dims = gl.arange(0, 64, gl.SliceLayout(0, mma_o))
    gl.store(Out + start + rows[:, None].to(gl.int64) * OUT_TOKEN + dims[None, :], acc)

    physical_layer = layer // ROWS
    if physical_layer % 2 == 1:
        table = physical_layer // 2
        token = (layer % ROWS) * T + rows
        h = head % OUT_H
        gated_row = table * (ROWS * T) + token
        z = gl.load(Gates + gated_row * OUT_H + h).to(gl.float32)
        gate = 2.0 * (1.0 / (1.0 + gl.exp(-z)))
        contribution = acc * gate[:, None]
        ids = gl.load(Ids + token).to(gl.int32)
        dest = table * VOCAB + ids
        offset = (dest[:, None] * OUT_H + h) * 64 + dims[None, :]
        gl.atomic_add(Lexical + offset, contribution, sem='relaxed')


def transport(q, k, lse, source, window, out, layers, lexical):
    h, t, width = q.shape
    heads = source.shape[-1] // width
    bt, bs = CONFIG['BT'], CONFIG['BS']
    def desc(tensor, shape, dtype):
        return TensorDescriptor.from_tensor(tensor, shape, gl.NVMMASharedLayout.get_default_for(shape, dtype))
    kd = desc(k.contiguous().view(-1, 64), [bt, 32], gl.bfloat16)
    qd = desc(q.contiguous().view(-1, 64), [bs, 32], gl.bfloat16)
    hd = desc(source.view(-1, heads * width), [bs, 64], gl.bfloat16)
    ld = TensorDescriptor.from_tensor(lse.contiguous().view(-1, bs), [1, bs],
        gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2))
    gates, ids, destination = lexical
    rows = ids.shape[0]
    size = destination.shape[1]
    _value_gluon_kernel[(h, t // bt)](kd, qd, hd, out, layers, out.stride(0), out.stride(1), ld,
        T=t, Window=window, BT=bt, BS=bs, STAGES=CONFIG['STAGES'], OUT_H=heads, num_warps=bt // 16,
        Gates=gates, Ids=ids, Lexical=destination, ROWS=rows, VOCAB=size)


def lexical_storage(ids, gate_values, vocab, width):
    gates = torch.stack(gate_values)
    destination = torch.zeros((4, vocab, width), device=ids.device, dtype=torch.float32)
    return gates, ids, destination
