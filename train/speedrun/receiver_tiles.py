"""The Cut's key credit scan as a Gluon (Hopper) kernel: each raw key coordinate's degree-2 moment response, with the
raw value credits and the value gates' dots from the same scan."""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
import gate_head_dot

CONFIG = dict(BT=64, BS=32, STAGES=2)
LOG2E = gl.constexpr(1.4426950408889634)


@triton.jit
def _pack(Q, H, Lse, Center, Data, Stats, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 64)
    local = rows[:, None] * 64 + cols[None, :]
    offset = tile * BLOCK * 64 + local
    record = Data + tile * (2 * BLOCK) * 64
    q = tl.load(Q + offset)
    tl.store(record + local, q)
    tl.store(record + BLOCK * 64 + local, tl.load(H + offset))
    lse = tl.load(Lse + tile * BLOCK + rows)
    center = tl.load(Center + tile * BLOCK + rows)
    tl.store(Stats + tile * 2 * BLOCK + rows, lse)
    tl.store(Stats + tile * 2 * BLOCK + BLOCK + rows, center)


@gluon.jit
def _issue_packet(data_desc, stats_desc, bar, ring, stats, slot, tile, pred,
                  BS: gl.constexpr):
    mbarrier.expect(bar.index(slot), 2 * BS * 64 * 2 + 2 * BS * 4, pred=pred)
    tma.async_copy_global_to_shared(data_desc, [tile * (2 * BS), 0], bar.index(slot), ring.index(slot), pred=pred)
    tma.async_copy_global_to_shared(stats_desc, [tile, 0], bar.index(slot), stats.index(slot), pred=pred)


@gluon.jit
def _half_response(px, py, qx, qy, qxy, rx, ry, s1, s2, Raw, Cos, Sin, base, target, half: gl.constexpr,
    Sigma: gl.constexpr, layout: gl.constexpr):
    """The key response of raw (pre-RoPE) coordinates half * 32 + j (pair j): a finite step of the raw key through RoPE
    and the key norm. The step moves both rotated coordinates of the pair (bu and bw), so the degree-2 model carries
    their mixed term (qxy)."""
    j = gl.arange(0, 32, gl.SliceLayout(0, layout))
    dims = gl.arange(0, 64, gl.SliceLayout(0, layout))
    raw = gl.load(Raw + base + target[:, None] * 64 + dims[None, :])
    raw_j = gl.load(Raw + base + target[:, None] * 64 + (half * 32 + j)[None, :])
    sq = gl.convert_layout(gl.sum(raw * raw, 1), gl.SliceLayout(1, layout))[:, None] / 64
    step = Sigma * gl.maximum(gl.sqrt(sq), 1e-06)
    ratio = (2 * step * raw_j + step * step) / 64 / (sq + 1.1920928955078125e-07)
    root = gl.sqrt(1 + ratio)
    ca = -ratio / (root * (1 + root))  # = expm1(-0.5 log1p(r)) (the Triton kernel's form), without cancellation
    cb = step * gl.rsqrt(sq + 1.1920928955078125e-07) * (1 + ca)
    cs = gl.load(Cos + target[:, None] * 32 + j[None, :])
    sn = gl.load(Sin + target[:, None] * 32 + j[None, :])
    if half == 0:
        bu, bw = cb * cs, -cb * sn
    else:
        bu, bw = cb * sn, cb * cs
    response = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + 2 * qxy * bu * bw + 2 * ca * (rx * bu + ry * bw)
      + s1[:, None] * ca + s2[:, None] * ca * ca)
    return response / step


@gluon.jit
def _key_gluon_kernel(k_desc, v_desc, data_desc, stats_desc, Raw, Cos, Sin, Out,
    OutMap, OUT_LAYER, OUT_TOKEN, ValueOut, VALUE_LAYER, VALUE_TOKEN, ROWS: gl.constexpr,
    Lexical, Dots,
    T: gl.constexpr, Window: gl.constexpr, Sigma: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
    STAGES: gl.constexpr, OUT_H: gl.constexpr, num_warps: gl.constexpr):
    head = gl.program_id(0)
    t0 = gl.program_id(1) * BT
    base = head * T
    ksm = gl.allocate_shared_memory(gl.bfloat16, [BT, 64], k_desc.layout)
    vsm = gl.allocate_shared_memory(gl.bfloat16, [BT, 64], k_desc.layout)
    plane: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BS, 64], gl.bfloat16)
    ring = gl.allocate_shared_memory(gl.bfloat16, [STAGES, 2 * BS, 64], data_desc.layout)
    g2sm = gl.allocate_shared_memory(gl.bfloat16, [BS, 64], plane)
    xysm = gl.allocate_shared_memory(gl.bfloat16, [BS, 32],
        gl.NVMMASharedLayout.get_default_for([BS, 32], gl.bfloat16))
    pair_layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [num_warps, 1], [1, 0])
    lcsm = gl.allocate_shared_memory(gl.float32, [STAGES, 1, 2 * BS], stats_desc.layout)
    kbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(kbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    layer_row = gl.load(OutMap + head // OUT_H)
    first = t0 // BS
    last = gl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    mbarrier.expect(kbar, 2 * BT * 64 * 2)
    tma.async_copy_global_to_shared(k_desc, [base + t0, 0], kbar, ksm)
    tma.async_copy_global_to_shared(v_desc, [base + t0, 0], kbar, vsm)
    for i in gl.static_range(STAGES - 1):
        _issue_packet(data_desc, stats_desc, bar, ring, lcsm, i, base // BS + first + i, first + i < last, BS)

    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BS, 16])
    mma_f: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 64, 16])
    mma_x: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 32, 16])
    a_f: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_f, k_width=2)
    a_x: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_x, k_width=2)
    target = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_s))
    cols = gl.arange(0, BS, gl.SliceLayout(0, mma_s))
    p2 = gl.zeros([BT, 64], gl.float32, mma_f)
    q2 = gl.zeros([BT, 64], gl.float32, mma_f)
    r2 = gl.zeros([BT, 64], gl.float32, mma_f)
    qxy = gl.zeros([BT, 32], gl.float32, mma_x)
    raw64 = gl.zeros([BT, 64], gl.float32, mma_f)
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    mbarrier.wait(kbar, 0)
    n = 0
    for it in range(first, last):
        slot = n % STAGES
        gl.thread_barrier()  # every warp is done with the slot refilled now
        nxt = it + STAGES - 1
        _issue_packet(data_desc, stats_desc, bar, ring, lcsm, (n + STAGES - 1) % STAGES, base // BS + nxt, nxt < last, BS)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        record = ring.index(slot)
        query = record.slice(0, BS)
        hub = record.slice(BS, BS)
        logit = warpgroup_mma(ksm, query.permute((1, 0)), zero_s, use_acc=False, is_async=True)
        score = warpgroup_mma(vsm, hub.permute((1, 0)), zero_s, use_acc=False, is_async=True)
        receiver = it * BS + cols
        lc = lcsm._reinterpret(gl.float32, [2 * STAGES, BS], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        lse = lc.index(2 * slot).load(gl.SliceLayout(0, mma_s))
        center = lc.index(2 * slot + 1).load(gl.SliceLayout(0, mma_s))
        logit, score = warpgroup_mma_wait(0, deps=[logit, score])
        logit = logit * 0.125
        a = gl.exp2((logit - lse[None, :]) * LOG2E)
        c = score - center[None, :]
        edge = (it * BS < t0 + BT - 1) | (it * BS + BS - 1 - t0 > Window)
        if edge:
            lag = receiver[None, :] - target[:, None]
            valid = (lag >= 0) & (lag <= Window)
            a = gl.where(valid, a, 0.0)
            c = gl.where(valid, c, 0.0)
        w1 = a * c
        w1d = gl.convert_layout(w1.to(gl.bfloat16), a_f)
        probability = gl.convert_layout(a.to(gl.bfloat16), a_f)
        p2 = warpgroup_mma(w1d, query, p2, is_async=True)
        raw64 = warpgroup_mma(probability, hub, raw64, is_async=True)
        p2, raw64, w1d, probability = warpgroup_mma_wait(0, deps=[p2, raw64, w1d, probability])
        linear = query.load(pair_layout).to(gl.float32)
        linear = linear * 0.125
        x, y = gl.split(gl.permute(gl.reshape(linear, [BS, 2, 32]), (0, 2, 1)))
        g2sm.store((linear * linear).to(gl.bfloat16))
        xysm.store((x * y).to(gl.bfloat16))
        fence_async_shared()
        gl.thread_barrier()
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logit
        w2d = gl.convert_layout(w2.to(gl.bfloat16), a_f)
        w2x = gl.convert_layout(w2.to(gl.bfloat16), a_x)
        w2ld = gl.convert_layout(w2l.to(gl.bfloat16), a_f)
        q2 = warpgroup_mma(w2d, g2sm, q2, is_async=True)
        qxy = warpgroup_mma(w2x, xysm, qxy, is_async=True)
        r2 = warpgroup_mma(w2ld, query, r2, is_async=True)
        q2, qxy, r2, w2d, w2x, w2ld = warpgroup_mma_wait(0, deps=[q2, qxy, r2, w2d, w2x, w2ld])
        n += 1
    mbarrier.invalidate(kbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    value_rows = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma_f))
    value_cols = gl.arange(0, 64, gl.SliceLayout(0, mma_f))
    local = value_rows[:, None].to(gl.int64) * VALUE_TOKEN + value_cols[None, :] + (head % OUT_H) * 64
    gate_head_dot.emit(raw64, Lexical, Dots, layer_row, head % OUT_H,
                       value_rows, value_cols, T, ROWS)
    gl.store(ValueOut + layer_row.to(gl.int64) * VALUE_LAYER + local, raw64)
    out_l: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [num_warps, 1], [1, 0])
    p = gl.convert_layout(p2, pair_layout)
    q = gl.convert_layout(q2, pair_layout)
    r = gl.convert_layout(r2, pair_layout)
    px, py = gl.split(gl.permute(gl.reshape(p * 0.125, [BT, 2, 32]), (0, 2, 1)))
    qx, qy = gl.split(gl.permute(gl.reshape(q, [BT, 2, 32]), (0, 2, 1)))
    rx, ry = gl.split(gl.permute(gl.reshape(r * 0.125, [BT, 2, 32]), (0, 2, 1)))
    px = gl.convert_layout(px, out_l)
    py = gl.convert_layout(py, out_l)
    qx = gl.convert_layout(qx, out_l)
    qy = gl.convert_layout(qy, out_l)
    rx = gl.convert_layout(rx, out_l)
    ry = gl.convert_layout(ry, out_l)
    qxy = gl.convert_layout(qxy, out_l)
    kx, ky = gl.split(gl.permute(gl.reshape(ksm.load(pair_layout).to(gl.float32), [BT, 2, 32]), (0, 2, 1)))
    kx = gl.convert_layout(kx, out_l)
    ky = gl.convert_layout(ky, out_l)
    s1 = gl.sum(kx * px + ky * py, 1)
    s2 = gl.sum(kx * rx + ky * ry, 1)
    rows = t0 + gl.arange(0, BT, gl.SliceLayout(1, out_l))
    layer = head // OUT_H
    layer = gl.load(OutMap + layer)
    start = layer.to(gl.int64) * OUT_LAYER + (head % OUT_H) * 64
    j = gl.arange(0, 32, gl.SliceLayout(0, out_l))
    for half in gl.static_range(2):
        resp = _half_response(px, py, qx, qy, qxy, rx, ry, s1, s2, Raw, Cos, Sin, base * 64, rows, half, Sigma, out_l)
        gl.store(Out + start + rows[:, None].to(gl.int64) * OUT_TOKEN + (half * 32 + j)[None, :], resp)


def key_values_into(q, k, k_raw, lse, values, hub, center, cos, sin, window, sigma, heads, key_out, layers, packed,
                    batch_rows, gate_args):
    """One bf16 attention scan for the key moments, written into key_out, the raw value credits, into packed, and the
    value gates' dots (gate_args): the (rows, T, 64) operands are heads of rows // heads layers, and both outputs are by
    token, (layers, T, heads * 64), each layer at its index in layers (int32)."""
    h, t, _ = k_raw.shape
    bt, bs = CONFIG['BT'], CONFIG['BS']
    q2 = q.contiguous().view(h * t, 64)
    pitch = 2 * bs
    count = h * t // bs
    data = torch.empty((count, pitch, 64), device=q.device, dtype=torch.bfloat16)
    stats = torch.empty((count, 2 * bs), device=q.device, dtype=torch.float32)
    _pack[(count,)](q2, hub.contiguous(), lse.contiguous(), center.contiguous(), data, stats,
                    BLOCK=bs, num_warps=4)
    f32k = gl.NVMMASharedLayout.get_default_for([bt, 64], gl.bfloat16)
    k_desc = TensorDescriptor.from_tensor(k.contiguous().view(h * t, 64), [bt, 64], f32k)
    v_desc = TensorDescriptor.from_tensor(values.contiguous().view(h * t, 64), [bt, 64], f32k)
    box = [pitch, 64]
    data_desc = TensorDescriptor.from_tensor(data.view(-1, 64), box,
                                             gl.NVMMASharedLayout.get_default_for(box, gl.bfloat16))
    rows32 = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)
    stats_desc = TensorDescriptor.from_tensor(stats, [1, 2 * bs], rows32)
    gate_sources, dots = gate_args
    _key_gluon_kernel[(h, t // bt)](k_desc, v_desc, data_desc, stats_desc, k_raw.contiguous(),
      cos.contiguous(), sin.contiguous(), key_out, layers, key_out.stride(0), key_out.stride(1),
      packed, packed.stride(0), packed.stride(1), batch_rows,
      gate_sources, dots, T=t, Window=window, Sigma=sigma, BT=bt, BS=bs,
      STAGES=CONFIG['STAGES'], OUT_H=heads, num_warps=bt // 16)
