"""Gluon (Hopper) key scan: moment_kernels._key_scan_kernel's key credits (VALUES off), bitwise the same numbers.

The same tiles (64 targets x 32 sources), the same MMA instructions in the same k order and the same layouts for every
row sum, so each output gets the same arithmetic. What changes:
- the source tiles (queries, hub, as two 32-column blocks each) come by TMA into a two-slot ring, the key and value
  tiles once; under the last source tile the epilogue's raw keys, cos and sin come by TMA into the key / value buffers;
- the pair operands (x, y, x^2, y^2, x y of the source queries, bf16) are formed from the query tile in shared memory
  (the reference reloads them from global memory), and the moment GEMMs take them as column blocks (x | y), (x^2 |
  y^2), x y instead of interleaved: each sum is the same, and the epilogue needs no gathers;
- only the tiles the diagonal or the window edge cut are masked;
- the logit and score GEMMs are waited for together, and the moment GEMMs stay in flight under the next tile's operand
  formation: one wait per tile (the reference waits after each of its six GEMMs);
- lse and center load one tile ahead (a load in flight would hold up the proxy fence);
- the response expression keeps the reference's fused multiply-adds: ptxas fuses across the reference's layout
  conversions differently, so the same boundaries are set here (_opaque).
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, tma, warpgroup_mma,
                                                              warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

CONFIG = dict(STAGES=2)


@gluon.jit
def _load_tile(q_desc, h_desc, q0, q1, h0, h1, ready, row0, it, i, BS: gl.constexpr, J: gl.constexpr,
               S: gl.constexpr):
    """Source tile it (this CTA's i-th) into ring slot i % S: its query and hub column blocks."""
    s = i % S
    mbarrier.expect(ready.index(s), 4 * BS * J * 4)
    tma.async_copy_global_to_shared(q_desc, [row0 + it * BS, 0], ready.index(s), q0.index(s))
    tma.async_copy_global_to_shared(q_desc, [row0 + it * BS, J], ready.index(s), q1.index(s))
    tma.async_copy_global_to_shared(h_desc, [row0 + it * BS, 0], ready.index(s), h0.index(s))
    tma.async_copy_global_to_shared(h_desc, [row0 + it * BS, J], ready.index(s), h1.index(s))


@gluon.jit
def _tile(i, it, first, last, acc, s1, s2, cl, q_desc, h_desc, r_desc, c_desc, s_desc, q0, q1, h0, h1, ks, vs, ops,
          ready, epbar, LSE, Center, head, row0, t0, target, T: gl.constexpr, Window: gl.constexpr, BT: gl.constexpr,
          BS: gl.constexpr, J: gl.constexpr, S: gl.constexpr, MASKED: gl.constexpr):
    """Source tile it (the reference's loop body)."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BS, 8])
    mma16: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, J, 16])
    mma64: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 2 * J, 16])
    op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma16, k_width=2)
    op64: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma64, k_width=2)
    xyl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    p2, q2, qxy, r2 = acc
    s = i % S
    mbarrier.wait(ready.index(s), (i // S) & 1)
    # the pair operands of the source queries (moment_kernels: x = Q[:, :J] / sqrt(D), y = Q[:, J:] / sqrt(D)),
    # read before the GEMMs queue on the same pipe
    xr = q0.index(s).load(xyl)
    yr = q1.index(s).load(xyl)
    zero = gl.zeros([BT, BS], gl.float32, mma)
    lt = warpgroup_mma(ks.index(0), q0.index(s).permute((1, 0)), zero, use_acc=False, is_async=True)
    lt = warpgroup_mma(ks.index(1), q1.index(s).permute((1, 0)), lt, is_async=True)
    st = warpgroup_mma(vs.index(0), h0.index(s).permute((1, 0)), zero, use_acc=False, is_async=True)
    st = warpgroup_mma(vs.index(1), h1.index(s).permute((1, 0)), st, is_async=True)
    source = it * BS + gl.arange(0, BS, gl.SliceLayout(0, mma))
    center, lse = cl  # loaded one tile ahead (a load still in flight would hold up the proxy fence below)
    x = xr * (2 * J) ** (-0.5)
    y = yr * (2 * J) ** (-0.5)
    b = i % 2
    of2, og2, oxy = ops  # (x | y), (x^2 | y^2) and x y: column blocks (the reference interleaves x and y)
    of2.index(b).store(_cat(x.to(gl.bfloat16), y.to(gl.bfloat16), BS, J))
    og2.index(b).store(_cat((x * x).to(gl.bfloat16), (y * y).to(gl.bfloat16), BS, J))
    oxy.index(b).store((x * y).to(gl.bfloat16))
    fence_async_shared()
    gl.thread_barrier()
    lt, st, p2, q2, qxy, r2 = warpgroup_mma_wait(0, deps=[lt, st, p2, q2, qxy, r2])
    if i + S < last - first:  # slot s is free again: tile it + S into it
        _load_tile(q_desc, h_desc, q0, q1, h0, h1, ready, row0, it + S, i + S, BS, J, S)
    if it == last - 1:  # the key and value tiles are free: the epilogue's (raw keys, cos, sin) load under this tile
        mbarrier.expect(epbar, 4 * BT * J * 4)
        tma.async_copy_global_to_shared(r_desc, [row0 + t0, 0], epbar, ks.index(0))
        tma.async_copy_global_to_shared(r_desc, [row0 + t0, J], epbar, ks.index(1))
        tma.async_copy_global_to_shared(c_desc, [t0, 0], epbar, vs.index(0))
        tma.async_copy_global_to_shared(s_desc, [t0, 0], epbar, vs.index(1))
    logit = lt * (2 * J) ** (-0.5)
    if MASKED:
        valid = (source[None, :] >= target[:, None]) & (source[None, :] - target[:, None] <= Window)
        c = gl.where(valid, st - center[None, :], 0.0)
        a = gl.where(valid, gl.exp(logit - lse[None, :]), 0.0)
    else:
        c = st - center[None, :]
        a = gl.exp(logit - lse[None, :])
    w1 = a * c
    w2 = (0.5 * a - a * a) * c
    w2l = w2 * logit
    w1d = gl.convert_layout(w1.to(gl.bfloat16), op64)
    w2d = gl.convert_layout(w2.to(gl.bfloat16), op64)
    w2ld = gl.convert_layout(w2l.to(gl.bfloat16), op64)
    p2 = warpgroup_mma(w1d, of2.index(b), p2, is_async=True)
    q2 = warpgroup_mma(w2d, og2.index(b), q2, is_async=True)
    qxy = warpgroup_mma(gl.convert_layout(w2d, op), oxy.index(b), qxy, is_async=True)
    r2 = warpgroup_mma(w2ld, of2.index(b), r2, is_async=True)
    s1 += gl.sum(w1 * logit, axis=1)
    s2 += gl.sum(w2l * logit, axis=1)
    return (p2, q2, qxy, r2), s1, s2, _center_lse(LSE, Center, head, gl.minimum(it + 1, last - 1), T, BS, mma)


@gluon.jit
def _center_lse(LSE, Center, head, it, T: gl.constexpr, BS: gl.constexpr, mma: gl.constexpr):
    source = it * BS + gl.arange(0, BS, gl.SliceLayout(0, mma))
    return gl.load(Center + head * T + source), gl.load(LSE + head * T + source)


@gluon.jit
def _cat(x, y, BLOCK: gl.constexpr, J: gl.constexpr):
    """(x | y) along the columns, in registers."""
    return gl.reshape(gl.permute(gl.join(x, y), (0, 2, 1)), (BLOCK, 2 * J))


@gluon.jit
def _blocks2(acc, BLOCK: gl.constexpr, J: gl.constexpr, lay: gl.constexpr):
    """The two J-column blocks of a (BLOCK, 2 J) accumulator, in its own registers."""
    x, y = gl.split(gl.permute(gl.reshape(acc, (BLOCK, 2, J)), (0, 2, 1)))
    return gl.convert_layout(x, lay), gl.convert_layout(y, lay)


@gluon.jit
def _opaque(x):
    """x, a value the compiler cannot see through (no multiply-add is fused across it)."""
    return gl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _half(px, py, qx, qy, qxy, rx, ry, s1, s2, sq, step, raw_j, cs, sn, Out, start, target, OUT_TOKEN, half,
          D: gl.constexpr, mma: gl.constexpr):
    """Coordinates half * D / 2 + (0..D / 2) of the response (moment_kernels' expression; pair = coordinate % J); a
    runtime half, so that the epilogue's code is emitted once."""
    J: gl.constexpr = D // 2
    dirs = half * J + gl.arange(0, J, gl.SliceLayout(0, mma))
    ratio = ((2 * step)[:, None] * raw_j + (step * step)[:, None]) / D / (sq + 1.1920928955078125e-07)[:, None]
    ca = libdevice.expm1(-0.5 * libdevice.log1p(ratio))
    cb = (step * libdevice.rsqrt(sq + 1.1920928955078125e-07))[:, None] * (1 + ca)
    first = half == 0
    bu = cb * gl.where(first, cs, sn)
    bw = cb * gl.where(first, -sn, cs)
    # the reference forms bu, bw, 2 ca and the first six terms in other layouts than the rest (the layout conversions
    # bound the multiply-adds the compiler fuses): the same boundaries here, so that it fuses the same ones
    bu, bw, ca2 = _opaque(bu), _opaque(bw), _opaque(2 * ca)
    response = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + 2 * qxy * bu * bw + ca2 * (rx * bu + ry * bw))
    ca = _opaque(ca)
    response = _opaque(response) + s1[:, None] * ca + s2[:, None] * ca * ca
    gl.store(Out + start + target[:, None] * OUT_TOKEN + dirs[None, :], response / step[:, None])


@gluon.jit
def _key_scan_ws_kernel(q_desc, k_desc, v_desc, h_desc, r_desc, c_desc, s_desc, LSE, Center, Out, OUT_LAYER,
                        OUT_TOKEN, OutMap, T: gl.constexpr, D: gl.constexpr, Window: gl.constexpr,
                        Sigma: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr, OUT_H: gl.constexpr,
                        OUT_MAPPED: gl.constexpr, S: gl.constexpr):
    J: gl.constexpr = D // 2
    gl.static_assert(BT % BS == 0)
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BS, 8])
    mma16: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, J, 16])
    blocked2: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    head = gl.program_id(0)
    t0 = gl.program_id(1) * BT
    row0 = head * T
    ks = gl.allocate_shared_memory(gl.float32, [2, BT, J], k_desc.layout)
    vs = gl.allocate_shared_memory(gl.float32, [2, BT, J], v_desc.layout)
    q0 = gl.allocate_shared_memory(gl.float32, [S, BS, J], q_desc.layout)
    q1 = gl.allocate_shared_memory(gl.float32, [S, BS, J], q_desc.layout)
    h0 = gl.allocate_shared_memory(gl.float32, [S, BS, J], h_desc.layout)
    h1 = gl.allocate_shared_memory(gl.float32, [S, BS, J], h_desc.layout)
    olay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BS, J], gl.bfloat16)
    olay2: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BS, 2 * J], gl.bfloat16)
    ops = (gl.allocate_shared_memory(gl.bfloat16, [2, BS, 2 * J], olay2),
           gl.allocate_shared_memory(gl.bfloat16, [2, BS, 2 * J], olay2),
           gl.allocate_shared_memory(gl.bfloat16, [2, BS, J], olay))
    blay: gl.constexpr = mbarrier.MBarrierLayout()
    ready = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    kvbar = gl.allocate_shared_memory(gl.int64, [1], blay)
    epbar = gl.allocate_shared_memory(gl.int64, [1], blay)
    for i in gl.static_range(S):
        mbarrier.init(ready.index(i), count=1)
    mbarrier.init(kvbar, count=1)
    mbarrier.init(epbar, count=1)
    first = t0 // BS
    last = gl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    mbarrier.expect(kvbar, 4 * BT * J * 4)
    for b in gl.static_range(2):
        tma.async_copy_global_to_shared(k_desc, [row0 + t0, b * J], kvbar, ks.index(b))
        tma.async_copy_global_to_shared(v_desc, [row0 + t0, b * J], kvbar, vs.index(b))
    for i in gl.static_range(S):
        if first + i < last:
            _load_tile(q_desc, h_desc, q0, q1, h0, h1, ready, row0, first + i, i, BS, J, S)
    target = t0 + gl.arange(0, BT, gl.SliceLayout(1, mma))
    mma64: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 2 * J, 16])
    zero = gl.zeros([BT, J], gl.float32, mma16)
    zero2 = gl.zeros([BT, 2 * J], gl.float32, mma64)
    acc = (warpgroup_mma_init(zero2), warpgroup_mma_init(zero2), warpgroup_mma_init(zero), warpgroup_mma_init(zero2))
    s1 = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma))
    s2 = gl.zeros([BT], gl.float32, gl.SliceLayout(1, mma))
    cl = _center_lse(LSE, Center, head, first, T, BS, mma)
    mbarrier.wait(kvbar, 0)
    # tiles the diagonal cuts, whole tiles, tiles the window edge cuts
    head_end = gl.minimum(first + BT // BS, last)
    mid_end = gl.maximum(gl.minimum(first + (Window - BS + 1) // BS + 1, last), head_end)
    for it in range(first, head_end):
        acc, s1, s2, cl = _tile(it - first, it, first, last, acc, s1, s2, cl, q_desc, h_desc, r_desc, c_desc, s_desc,
                                q0, q1, h0, h1, ks, vs, ops, ready, epbar, LSE, Center, head, row0, t0, target, T,
                                Window, BT, BS, J, S, True)
    for it in range(head_end, mid_end):
        acc, s1, s2, cl = _tile(it - first, it, first, last, acc, s1, s2, cl, q_desc, h_desc, r_desc, c_desc, s_desc,
                                q0, q1, h0, h1, ks, vs, ops, ready, epbar, LSE, Center, head, row0, t0, target, T,
                                Window, BT, BS, J, S, False)
    for it in range(mid_end, last):
        acc, s1, s2, cl = _tile(it - first, it, first, last, acc, s1, s2, cl, q_desc, h_desc, r_desc, c_desc, s_desc,
                                q0, q1, h0, h1, ks, vs, ops, ready, epbar, LSE, Center, head, row0, t0, target, T,
                                Window, BT, BS, J, S, True)
    p2, q2, qxy, r2 = acc
    p2, q2, qxy, r2 = warpgroup_mma_wait(0, deps=[p2, q2, qxy, r2])
    px, py = _blocks2(p2, BT, J, mma)
    qx, qy = _blocks2(q2, BT, J, mma)
    rx, ry = _blocks2(r2, BT, J, mma)
    mbarrier.wait(epbar, 0)
    for i in gl.static_range(S):
        mbarrier.invalidate(ready.index(i))
    mbarrier.invalidate(kvbar)
    mbarrier.invalidate(epbar)
    # the perturbation geometry's row terms in the reference's layout (the same reduction tree for sq): the raw keys
    # as one (BT, D) tile (the two column blocks are its shared memory layout)
    raw = ks._reinterpret(gl.float32, [BT, D], gl.NVMMASharedLayout.get_default_for([BT, D], gl.float32)).load(blocked2)
    raw0 = ks.index(0).load(mma)
    raw1 = ks.index(1).load(mma)
    cs = vs.index(0).load(mma)
    sn = vs.index(1).load(mma)
    sq = gl.sum(raw * raw, axis=1) / D
    step = Sigma * gl.maximum(gl.sqrt(sq), 1e-06)
    sq = gl.convert_layout(sq, gl.SliceLayout(1, mma))
    step = gl.convert_layout(step, gl.SliceLayout(1, mma))
    layer = head // OUT_H
    if OUT_MAPPED:
        layer = gl.load(OutMap + layer)
    start = layer.to(gl.int64) * OUT_LAYER + (head % OUT_H) * D
    qxy = gl.convert_layout(qxy, mma)
    for half in range(2):
        _half(px, py, qx, qy, qxy, rx, ry, s1, s2, sq, step, gl.where(half == 0, raw0, raw1), cs, sn, Out, start,
              target, OUT_TOKEN, half, D, mma)


def _desc(x, block):
    return TensorDescriptor.from_tensor(x, block, gl.NVMMASharedLayout.get_default_for(block, gl.float32))


def supported(qp, kp, k_raw, lse, values, hub, center, cos, sin, bt=64):
    h, t, d = k_raw.shape
    return d == 64 and t % bt == 0 and all(x.is_contiguous() and x.dtype == torch.float32
                                           for x in (qp, kp, k_raw, lse, values, hub, center, cos, sin))


def key_scan_ws(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, sigma, key, out_heads, layers, bt=64,
                bs=32):
    """moment_kernels._key_scan's key credits (with_values off) into key, the (rows // out_heads, T, out_heads * D)
    view _key_scan writes (layers: its layer map): qp, kp, values, hub, k_raw (rows, T, D), lse, center (rows, T), cos,
    sin (T, D / 2), all fp32 contiguous (supported)."""
    h, t, d = k_raw.shape
    j = d // 2
    q2, k2, v2, h2 = (x.view(h * t, d) for x in (qp, kp, values, hub))
    args = (_desc(q2, [bs, j]), _desc(k2, [bt, j]), _desc(v2, [bt, j]), _desc(h2, [bs, j]),
            _desc(k_raw.view(h * t, d), [bt, j]), _desc(cos, [bt, j]), _desc(sin, [bt, j]), lse, center, key,
            key.stride(0), key.stride(1), key if layers is None else layers)
    _key_scan_ws_kernel[(h, t // bt)](*args, T=t, D=d, Window=window, Sigma=sigma, BT=bt, BS=bs, OUT_H=out_heads,
                                      OUT_MAPPED=layers is not None, S=CONFIG['STAGES'], num_warps=4)
    return key
