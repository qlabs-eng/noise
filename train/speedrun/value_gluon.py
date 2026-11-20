"""Gluon (Hopper) value credit pass: sum_s p(s, t) * source(s) (the window's attention
probabilities of query s on key t times the source rows: bf16 queries and keys, bf16 probabilities times the source's
bf16 high part, fp32 accumulation), as an FA3-style kernel.

CTA = 128 keys (two warpgroups, 64 each) of one head; the receiver blocks (64 queries) of its window stream through a
TMA ring shared by both warpgroups: the queries (as two 32-column halves), the source's bf16 high part (formed once
per call by _split_kernel) and the receivers' lse row. Per block: S = keys @ queries^T (wgmma from shared memory),
P = exp(S / 8 - lse) (masked only on the window's edge blocks), O += P @ high (P from registers).
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

CONFIG = dict(BT=128, BS=64, STAGES=2)
LOG2E = gl.constexpr(1.4426950408889634)


@triton.jit
def _split_kernel(Src, Hi, T: tl.constexpr, IN_H: tl.constexpr, IN_LAYER, IN_TOKEN, BLOCK: tl.constexpr):
    """Hi (rows, T, 64) bf16: each source row's bf16 high part; the source stacked or by token (IN_H heads of a
    layer side by side, strides IN_LAYER, IN_TOKEN)."""
    row, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, 64)
    x = tl.load(Src + (row // IN_H).to(tl.int64) * IN_LAYER + (row % IN_H) * 64 + token[:, None] * IN_TOKEN
                + dims[None, :])
    hi = x.to(tl.bfloat16)
    dst = row.to(tl.int64) * T * 64 + token[:, None] * 64 + dims[None, :]
    tl.store(Hi + dst, hi)


@gluon.jit
def _issue(q_desc, h_desc, bar, qsm, hsm, slot, row, pred, BS: gl.constexpr, lse_desc=None, lsesm=None):
    mbarrier.expect(bar.index(slot), BS * 64 * 2 + BS * 64 * 2 + BS * 4, pred=pred)
    # the row of BS receivers' lse ((h T / BS, BS) view)
    tma.async_copy_global_to_shared(lse_desc, [row // BS, 0], bar.index(slot), lsesm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 0], bar.index(slot), qsm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 32], bar.index(slot), qsm.index(2 * slot + 1), pred=pred)
    tma.async_copy_global_to_shared(h_desc, [row, 0], bar.index(slot), hsm.index(slot), pred=pred)


@gluon.jit
def _value_gluon_kernel(k_desc, q_desc, h_desc, Out, OutMap, OUT_LAYER, OUT_TOKEN, lse_desc,
    T: gl.constexpr, Window: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr, STAGES: gl.constexpr,
    OUT_H: gl.constexpr, num_warps: gl.constexpr):
    head = gl.program_id(0)
    tile = gl.program_id(1)
    t0 = tile * BT
    base = head * T
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
        _issue(q_desc, h_desc, bar, qsm, hsm, i, base + (first + i) * BS, first + i < last, BS, lse_desc,
          lsesm)

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
        _issue(q_desc, h_desc, bar, qsm, hsm, (n + STAGES - 1) % STAGES, base + nxt * BS, nxt < last,
          BS, lse_desc, lsesm)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        s = warpgroup_mma(ksm.index(0), qsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)  # fp32 operands: tf32 (the default input precision)
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


def value_pass_gluon(q, k, lse, source, window, out, layers):
    """sum_s p * source for (rows, T, 64) bf16 q / k, the rows heads of rows // heads layers: source and out are
    (layers, T, heads * 64) views with last stride 1, source indexed by the rows' layers and out at the index
    layers (int32) gives each of them."""
    h, t, d = q.shape
    heads = source.shape[-1] // d
    assert d == 64 and source.shape[-1] // heads == 64 and source.stride(2) == 1
    bt, bs = CONFIG['BT'], CONFIG['BS']
    assert out.stride(2) == 1 and t % bt == 0 and t % bs == 0
    hi = torch.empty((h, t, 64), device=q.device, dtype=torch.bfloat16)
    _split_kernel[(h, t // 64)](source, hi, T=t, IN_H=heads, IN_LAYER=source.stride(0),
      IN_TOKEN=source.stride(1), BLOCK=64, num_warps=4)
    assert q.dtype == torch.bfloat16 and layers is not None
    q2, k2 = q.contiguous().view(h * t, 64), k.contiguous().view(h * t, 64)
    k_desc = TensorDescriptor.from_tensor(k2, [bt, 32], gl.NVMMASharedLayout.get_default_for([bt, 32], gl.bfloat16))
    q_desc = TensorDescriptor.from_tensor(q2, [bs, 32], gl.NVMMASharedLayout.get_default_for([bs, 32], gl.bfloat16))
    hl = gl.NVMMASharedLayout.get_default_for([bs, 64], gl.bfloat16)
    h_desc = TensorDescriptor.from_tensor(hi.view(h * t, 64), [bs, 64], hl)
    lse_desc = TensorDescriptor.from_tensor(lse.contiguous().view(h * t // bs, bs), [1, bs],
      gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2))
    _value_gluon_kernel[(h, t // bt)](k_desc, q_desc, h_desc, out,
      layers, out.stride(0), out.stride(1), lse_desc, T=t, Window=window, BT=bt, BS=bs,
      STAGES=CONFIG['STAGES'], OUT_H=heads, num_warps=bt // 16)
