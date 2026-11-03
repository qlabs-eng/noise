"""Gluon (Hopper) key credit scan: moment_kernels._key_scan (without the raw value credits) as an FA3-style kernel.

CTA = 128 keys (two warpgroups) of one head; the window's receiver blocks (64 queries) stream through a 2-stage TMA
ring: the queries and the hub (fp32, 32-column halves) and the queries' bf16 pair features (formed once per call by
_features_kernel, the same bf16 operands the Triton kernel forms per block). Per block: logit = keys @ queries^T and
score = values @ hub^T (tf32 wgmmas from shared memory), the weights w1 = a c, w2 = (a / 2 - a^2) c and w2 * logit
(masked only on the window's edge blocks), then p2 += w1 @ f2, q2 += w2 @ g2, qxy += w2 @ xy, r2 += w2l @ f2 (bf16,
A from registers). The epilogue forms each direction's response from its pair's sums (directions d and d + 32 share
pair d) without gathers.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

CONFIG = dict(BT=128, BS=64, STAGES=2, PF=0)  # PF: the receivers' lse / center rows by TMA with the ring (bitwise)
LOG2E = gl.constexpr(1.4426950408889634)


@triton.jit
def _features_kernel(Q, F2, G2, XY, BLOCK: tl.constexpr):
    """Per query (rows of 64): x = q[:32] / 8, y = q[32:] / 8; F2 = (x_j, y_j) interleaved, G2 = (x_j^2, y_j^2)
    interleaved, XY = x_j y_j, bf16 (the Triton key scan's operands)."""
    token = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pairs = tl.arange(0, 32)
    x = tl.load(Q + token[:, None] * 64 + pairs[None, :]).to(tl.float32) * 0.125
    y = tl.load(Q + token[:, None] * 64 + (pairs + 32)[None, :]).to(tl.float32) * 0.125
    cols = tl.arange(0, 64)
    tl.store(F2 + token[:, None] * 64 + cols[None, :], tl.reshape(tl.join(x, y), (BLOCK, 64)).to(tl.bfloat16))
    tl.store(G2 + token[:, None] * 64 + cols[None, :], tl.reshape(tl.join(x * x, y * y), (BLOCK, 64)).to(tl.bfloat16))
    tl.store(XY + token[:, None] * 32 + pairs[None, :], (x * y).to(tl.bfloat16))


@gluon.jit
def _issue(q_desc, h_desc, f2_desc, g2_desc, xy_desc, bar, qsm, hsm, f2sm, g2sm, xysm, slot, row, pred,
    BS: gl.constexpr, lse_desc=None, center_desc=None, lcsm=None, PF: gl.constexpr = False, QK16: gl.constexpr = False):
    mbarrier.expect(bar.index(slot), 2 * BS * 64 * (2 if QK16 else 4) + 2 * BS * 64 * 2 + BS * 32 * 2 + (2 * BS * 4 if PF else 0),
      pred=pred)
    if PF:  # rows of 64 receivers' lse / center ((h T / BS, BS) views)
        tma.async_copy_global_to_shared(lse_desc, [row // BS, 0], bar.index(slot), lcsm.index(2 * slot), pred=pred)
        tma.async_copy_global_to_shared(center_desc, [row // BS, 0], bar.index(slot), lcsm.index(2 * slot + 1),
          pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 0], bar.index(slot), qsm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(q_desc, [row, 32], bar.index(slot), qsm.index(2 * slot + 1), pred=pred)
    tma.async_copy_global_to_shared(h_desc, [row, 0], bar.index(slot), hsm.index(2 * slot), pred=pred)
    tma.async_copy_global_to_shared(h_desc, [row, 32], bar.index(slot), hsm.index(2 * slot + 1), pred=pred)
    tma.async_copy_global_to_shared(f2_desc, [row, 0], bar.index(slot), f2sm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(g2_desc, [row, 0], bar.index(slot), g2sm.index(slot), pred=pred)
    tma.async_copy_global_to_shared(xy_desc, [row, 0], bar.index(slot), xysm.index(slot), pred=pred)


@gluon.jit
def _half_response(px, py, qx, qy, qxy, rx, ry, s1, s2, Raw, Cos, Sin, base, target, half: gl.constexpr,
    Sigma: gl.constexpr, layout: gl.constexpr):
    """The key response of directions half * 32 + j (pair j): moment_kernels._key_perturbation_geometry's step."""
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
def _key_gluon_kernel(k_desc, v_desc, q_desc, h_desc, f2_desc, g2_desc, xy_desc, Raw, LSE, Center, Cos, Sin, Out,
    OutMap, OUT_LAYER, OUT_TOKEN, lse_desc, center_desc,
    T: gl.constexpr, Window: gl.constexpr, Sigma: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
    STAGES: gl.constexpr, OUT_H: gl.constexpr, OUT_MAPPED: gl.constexpr, num_warps: gl.constexpr,
    PF: gl.constexpr = False, QK16: gl.constexpr = False):
    head = gl.program_id(0)
    t0 = gl.program_id(1) * BT
    base = head * T
    if QK16:  # bf16 keys / values / queries / hub (the query moments' copies)
        ksm = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 32], k_desc.layout)
        vsm = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 32], k_desc.layout)
        qsm = gl.allocate_shared_memory(gl.bfloat16, [2 * STAGES, BS, 32], q_desc.layout)
        hsm = gl.allocate_shared_memory(gl.bfloat16, [2 * STAGES, BS, 32], q_desc.layout)
    else:
        ksm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], k_desc.layout)
        vsm = gl.allocate_shared_memory(gl.float32, [2, BT, 32], k_desc.layout)
        qsm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], q_desc.layout)
        hsm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, BS, 32], q_desc.layout)
    f2sm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], f2_desc.layout)
    g2sm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 64], f2_desc.layout)
    xysm = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BS, 32], xy_desc.layout)
    if PF:
        lcsm = gl.allocate_shared_memory(gl.float32, [2 * STAGES, 1, BS], lse_desc.layout)
    else:
        lcsm = None
    kbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    mbarrier.init(kbar, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(bar.index(i), count=1)
    first = t0 // BS
    last = gl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    mbarrier.expect(kbar, 2 * BT * 64 * (2 if QK16 else 4))
    tma.async_copy_global_to_shared(k_desc, [base + t0, 0], kbar, ksm.index(0))
    tma.async_copy_global_to_shared(k_desc, [base + t0, 32], kbar, ksm.index(1))
    tma.async_copy_global_to_shared(v_desc, [base + t0, 0], kbar, vsm.index(0))
    tma.async_copy_global_to_shared(v_desc, [base + t0, 32], kbar, vsm.index(1))
    for i in gl.static_range(STAGES - 1):
        _issue(q_desc, h_desc, f2_desc, g2_desc, xy_desc, bar, qsm, hsm, f2sm, g2sm, xysm, i,
          base + (first + i) * BS, first + i < last, BS, lse_desc, center_desc, lcsm, PF, QK16)

    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BS, 16 if QK16 else 8])
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
    s1 = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    s2 = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma_s))
    zero_s = gl.zeros([BT, BS], gl.float32, mma_s)
    mbarrier.wait(kbar, 0)
    n = 0
    for it in range(first, last):
        slot = n % STAGES
        gl.thread_barrier()  # every warp is done with the slot refilled now
        nxt = it + STAGES - 1
        _issue(q_desc, h_desc, f2_desc, g2_desc, xy_desc, bar, qsm, hsm, f2sm, g2sm, xysm,
          (n + STAGES - 1) % STAGES, base + nxt * BS, nxt < last, BS, lse_desc, center_desc, lcsm, PF, QK16)
        mbarrier.wait(bar.index(slot), (n // STAGES) & 1)
        logit = warpgroup_mma(ksm.index(0), qsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)
        logit = warpgroup_mma(ksm.index(1), qsm.index(2 * slot + 1).permute((1, 0)), logit, is_async=True)
        score = warpgroup_mma(vsm.index(0), hsm.index(2 * slot).permute((1, 0)), zero_s, use_acc=False,
          is_async=True)
        score = warpgroup_mma(vsm.index(1), hsm.index(2 * slot + 1).permute((1, 0)), score, is_async=True)
        receiver = it * BS + cols
        if PF:
            lc = lcsm._reinterpret(gl.float32, [2 * STAGES, BS], gl.SwizzledSharedLayout(1, 1, 1, [0]))
            lse = lc.index(2 * slot).load(gl.SliceLayout(0, mma_s))
            center = lc.index(2 * slot + 1).load(gl.SliceLayout(0, mma_s))
        else:
            lse = gl.load(LSE + base + receiver)
            center = gl.load(Center + base + receiver)
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
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logit
        s1 += gl.sum(w1 * logit, 1)
        s2 += gl.sum(w2l * logit, 1)
        w1d = gl.convert_layout(w1.to(gl.bfloat16), a_f)
        w2d = gl.convert_layout(w2.to(gl.bfloat16), a_f)
        w2x = gl.convert_layout(w2.to(gl.bfloat16), a_x)
        w2ld = gl.convert_layout(w2l.to(gl.bfloat16), a_f)
        p2 = warpgroup_mma(w1d, f2sm.index(slot), p2, is_async=True)
        q2 = warpgroup_mma(w2d, g2sm.index(slot), q2, is_async=True)
        qxy = warpgroup_mma(w2x, xysm.index(slot), qxy, is_async=True)
        r2 = warpgroup_mma(w2ld, f2sm.index(slot), r2, is_async=True)
        p2, q2, qxy, r2, w1d, w2d, w2x, w2ld = warpgroup_mma_wait(0, deps=[p2, q2, qxy, r2, w1d, w2d, w2x, w2ld])
        n += 1
    mbarrier.invalidate(kbar)
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(bar.index(i))
    # the response per direction: pair j's sums at columns 2 j (x) and 2 j + 1 (y)
    out_l: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [num_warps, 1], [1, 0])
    px, py = gl.split(gl.reshape(gl.convert_layout(p2, gl.BlockedLayout([1, 2], [4, 8], [num_warps, 1], [1, 0])),
      [BT, 32, 2]))
    qx, qy = gl.split(gl.reshape(gl.convert_layout(q2, gl.BlockedLayout([1, 2], [4, 8], [num_warps, 1], [1, 0])),
      [BT, 32, 2]))
    rx, ry = gl.split(gl.reshape(gl.convert_layout(r2, gl.BlockedLayout([1, 2], [4, 8], [num_warps, 1], [1, 0])),
      [BT, 32, 2]))
    px = gl.convert_layout(px, out_l)
    py = gl.convert_layout(py, out_l)
    qx = gl.convert_layout(qx, out_l)
    qy = gl.convert_layout(qy, out_l)
    rx = gl.convert_layout(rx, out_l)
    ry = gl.convert_layout(ry, out_l)
    qxy = gl.convert_layout(qxy, out_l)
    s1 = gl.convert_layout(s1, gl.SliceLayout(1, out_l))
    s2 = gl.convert_layout(s2, gl.SliceLayout(1, out_l))
    rows = t0 + gl.arange(0, BT, gl.SliceLayout(1, out_l))
    layer = head // OUT_H
    if OUT_MAPPED:
        layer = gl.load(OutMap + layer)
    start = layer.to(gl.int64) * OUT_LAYER + (head % OUT_H) * 64
    j = gl.arange(0, 32, gl.SliceLayout(0, out_l))
    for half in gl.static_range(2):
        resp = _half_response(px, py, qx, qy, qxy, rx, ry, s1, s2, Raw, Cos, Sin, base * 64, rows, half, Sigma, out_l)
        gl.store(Out + start + rows[:, None].to(gl.int64) * OUT_TOKEN + (half * 32 + j)[None, :], resp)


def key_scan_gluon(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, with_values, out_heads=1,
                   key=None, layers=None, **over):
    """moment_kernels._key_scan's arguments and result (with_values False only)."""
    assert not with_values
    h, t, d = k_raw.shape
    assert d == 64
    cfg = dict(CONFIG, **over)
    bt, bs = cfg['BT'], cfg['BS']
    shape = (h // out_heads, t, out_heads * d)
    key = k_raw.new_empty(shape) if key is None else key
    assert key.stride(2) == 1 and t % bt == 0 and t % bs == 0
    q2 = qp.contiguous().view(h * t, 64)
    f2 = torch.empty((h * t, 64), device=qp.device, dtype=torch.bfloat16)
    g2 = torch.empty_like(f2)
    xy = torch.empty((h * t, 32), device=qp.device, dtype=torch.bfloat16)
    _features_kernel[(h * t // 64,)](q2, f2, g2, xy, BLOCK=64, num_warps=4)
    qk16 = kp.dtype == torch.bfloat16  # (the query moments' bf16 q / k / values / hub)
    qk_gl = gl.bfloat16 if qk16 else gl.float32
    f32k = gl.NVMMASharedLayout.get_default_for([bt, 32], qk_gl)
    f32s = gl.NVMMASharedLayout.get_default_for([bs, 32], qk_gl)
    k_desc = TensorDescriptor.from_tensor(kp.contiguous().view(h * t, 64), [bt, 32], f32k)
    v_desc = TensorDescriptor.from_tensor(values.contiguous().view(h * t, 64), [bt, 32], f32k)
    q_desc = TensorDescriptor.from_tensor(q2, [bs, 32], f32s)
    h_desc = TensorDescriptor.from_tensor(hub.contiguous().view(h * t, 64), [bs, 32], f32s)
    b64 = gl.NVMMASharedLayout.get_default_for([bs, 64], gl.bfloat16)
    f2_desc = TensorDescriptor.from_tensor(f2, [bs, 64], b64)
    g2_desc = TensorDescriptor.from_tensor(g2, [bs, 64], b64)
    xy_desc = TensorDescriptor.from_tensor(xy, [bs, 32], gl.NVMMASharedLayout.get_default_for([bs, 32], gl.bfloat16))
    rows32 = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)
    lse_desc = TensorDescriptor.from_tensor(lse.contiguous().view(h * t // bs, bs), [1, bs], rows32)
    center_desc = TensorDescriptor.from_tensor(center.contiguous().view(h * t // bs, bs), [1, bs], rows32)
    _key_gluon_kernel[(h, t // bt)](k_desc, v_desc, q_desc, h_desc, f2_desc, g2_desc, xy_desc, k_raw.contiguous(),
      lse.contiguous(), center.contiguous(), cos.contiguous(), sin.contiguous(), key,
      key if layers is None else layers, key.stride(0), key.stride(1), lse_desc, center_desc, T=t, Window=window,
      Sigma=sigma, BT=bt, BS=bs, STAGES=cfg['STAGES'], OUT_H=out_heads, OUT_MAPPED=layers is not None,
      PF=bool(cfg['PF']), QK16=qk16, num_warps=bt // 16)
    return key.view(h, t, d) if out_heads == 1 else key, key.view(h, t, d) if out_heads == 1 else key
