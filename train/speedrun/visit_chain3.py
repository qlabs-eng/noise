"""Gluon (Hopper) three-block chain: the replay-response visit (block A: a routed visit with the attention replay's
response bf16(add_weights @ add_values) added on load, formed in the kernel), a plain routed visit (block M) and
the readout visit (block B) back to back per tile, the tile's delta kept in shared memory throughout.

fused_resp_visit_readout_fast returns the readout moments of the three visits. Ring of routed rows: per tile Fc A,
Wo A, Fc M, Wo M, Fc B, Wo B (6 KC items). Tiles of 128 moves (two warpgroups). Response staging (2 x 32 KB slots):
block A's 8 value items per tile alternate between the two slots, the response weights have a buffer of their own,
and while blocks M and B run the slots hold the tensor-core parts and the readout vectors.

The response's values are gathered, not projected: each chunk's 128 value rows are gathered by cp.async
from a per-key table bf16(v @ W_h) (completion on the slot's barrier, one arrival per thread), and the centring enters
as a k16 wgmma: [s, s] @ [-Op_hi; -Op_lo] (s the weights' per-head sums, by a wgmma against a 0/1 head selector; -Op =
-(out @ W_h) in bf16 hi / lo, TMA'd into the weights' buffer while block A's gate runs; the next tile's weights load
after the gate).
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from visit_chain import (EPS, _bf16, _rbf16, _change, _diag_issue, _gate_done, _gate_tc_done,
  _gate_tc_init, _issue_rows, _output_readout_tc, _readout_tc_done, _reload, _row_sum, _rows_issue, _stats_store,
  split_parts)

CONFIG = dict(BM=128, NS=4, num_warps=8)
# Blocks M's and B's gate dots and the readout's linear moments run on the tensor cores (visit_chain.TC); the
# parts live in the response staging slots while blocks M and B run: x_m's at [0, 8 KB), x_b's at [8, 16 KB), the
# readout vectors' at [16, 24 KB), diag (fp32) at [24, 26 KB); the next tile's response weights (slot 0) then load
# after block B's gate.


@gluon.jit
def _rslot_phase(r):
    """Slot and phase of the tile's response item r: its 8 value items alternate between two slots, 4 per slot (the
    weights have their own buffer)."""
    return r % 2, ((r - 1) // 2) & 1


@gluon.jit
def _w_issue_wb(w_desc, wbar, wbuf, row0, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """The tile's response weights (BM x 2 BK) into their own buffer, as two BK-column halves."""
    mbarrier.expect(wbar, 2 * NBYTES, pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, 0], wbar, wbuf.index(0), pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, BK], wbar, wbuf.index(1), pred=pred)


@gluon.jit
def _gv_issue(astage, abar, slot, prow, cols, c, BK: gl.constexpr):
    """Chunk c's columns of the token's 128 gathered key rows (prow) into slot; every thread's completion arrives
    on the slot's barrier."""
    cp.async_copy_global_to_shared(astage.index(slot), prow[:, None] + (c * BK + cols)[None, :])
    cp.mbarrier_arrive(abar.index(slot), increment_count=False)


@gluon.jit
def _resp_take_gv(abar, astage, slot, phase, w0, w1, a2, obuf, c, zero, BK: gl.constexpr):
    """bf16(weights @ gathered rows chunk + [s, s] @ [-Op_hi; -Op_lo] chunk c)."""
    mbarrier.wait(abar.index(slot), phase)
    fence_async_shared()
    acc = warpgroup_mma(w0, astage.index(slot).slice(0, BK), zero, use_acc=False, is_async=True)
    acc = warpgroup_mma(w1, astage.index(slot).slice(BK, BK), acc, is_async=True)
    acc = warpgroup_mma(a2, obuf.index(c), acc, is_async=True)
    acc = warpgroup_mma_wait(0, deps=[acc, w0, w1, a2])[0]
    out = _rbf16(acc)
    return out


@gluon.jit
def _gate_resp(j, acc, sq_x, dots, moved_sq, x_next, h, dbar, dsm, wsm, abar, astage, w0, w1, zero, X, Clean,
    it, base, fc_rows, wo_rows, cols, xcols, token, nxt_token,
    D: gl.constexpr, R: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr,
    mma1: gl.constexpr, mma2: gl.constexpr, a_layout: gl.constexpr,
    prow=None, a2=None, obuf=None, wbar=None):
    """Block A's gate chunk j: d = delta + response chunk j (fp32), acc += bf16(d) @ Fc[routes]^T (A from registers,
    waited at once); the norm sums of d. bf16(d) also replaces the delta chunk (each thread its own elements),
    for the residual (no second response bmm; the tile's response items are the weights and the gate's KC chunks)."""
    slot = (it * KC + j) % KC
    mbarrier.wait(dbar.index(slot), it & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    r = 1 + j
    _issue_rows(wsm.index((base + j + NS - 1) % NS), gl.where(j + NS - 1 < KC, fc_rows, wo_rows), cols,
      ((j + NS - 1) % KC) * BK)
    # the next chunk's gathered rows (a thread's cp.async arrive tracks every copy it has in flight: the ring
    # item just issued is waited with them)
    if j + 1 < KC:
        _gv_issue(astage, abar, (r + 1) % 2, prow, cols, j + 1, BK)
    s, ph = _rslot_phase(r)
    if j == 0:  # -Op landed (issued at the tile's start, waited here under the first gather's latency)
        mbarrier.wait(wbar, 1)
    addv = _resp_take_gv(abar, astage, s, ph, w0, w1, a2, obuf, j, zero, BK)
    d = dsm.index(slot).load(mma2).to(gl.float32) + addv.to(gl.float32)
    moved = _bf16(d)
    dsm.index(slot).store(moved)
    a_op = gl.convert_layout(moved, a_layout)
    acc = warpgroup_mma(a_op, wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    acc, a_op = warpgroup_mma_wait(0, deps=[acc, a_op])
    x = x_next
    x_next = gl.load(X + gl.where(j + 1 < KC, token, nxt_token) * D + ((j + 1) % KC) * BK + xcols)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    sq_x += gl.sum(gl.expand_dims(x * x, 0), 1)
    dots += gl.sum(d * x[None, :], 1)
    moved_sq += gl.sum(d * d, 1)
    return acc, sq_x, dots, moved_sq, x_next, h


@gluon.jit
def _output_resp(c, dsm, wsm, change, zero, scale, it, base, wo_rows, next_rows,
    cols, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, mma2: gl.constexpr):
    """Block A's output chunk c: bf16(bf16(d + change @ Wo[routes][:, c]) * scale) back into delta chunk c, d the
    gate's stored bf16(delta + response chunk c)."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    slot = (it * KC + c) % KC
    old = dsm.index(slot).load(mma2).to(gl.float32)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    dsm.index(slot).store((value.to(gl.float32) * scale).to(gl.bfloat16))


@gluon.jit
def _fused_issue(buf, item, wo_rows, fc_rows, next_rows, cols,
    BK: gl.constexpr, KC: gl.constexpr):
    rows = gl.where(item < 2 * KC, gl.where(item % 2 == 0, wo_rows, fc_rows), next_rows)
    chunk = gl.where(item < 2 * KC, item // 2, item - 2 * KC)
    _issue_rows(buf, rows, cols, chunk * BK)


@gluon.jit
def _gate_to_fused(j, acc, accx, moved_sq, h, dsm, wsm, x3, Clean, it, base,
    fc_rows, wo_rows, next_fc_rows, cols, token, R: gl.constexpr, XOFF: gl.constexpr,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr,
    mma1: gl.constexpr, s_layout: gl.constexpr, BM: gl.constexpr):
    slot = (it * KC + j) % (KC + E)
    cp.wait_group(NS - 2)
    fence_async_shared()
    acc, accx = warpgroup_mma_wait(0, deps=[acc, accx])
    gl.thread_barrier()
    future = j + NS - 1
    following = future - KC
    rows = gl.where(future < KC, fc_rows, gl.where(following % 2 == 0, wo_rows, next_fc_rows))
    chunk = gl.where(future < KC, future, following // 2)
    _issue_rows(wsm.index((base + future) % NS), rows, cols, chunk * BK)
    acc = warpgroup_mma(dsm.index(slot), wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    accx = warpgroup_mma(dsm.index(slot), x3.index(XOFF + j).permute((1, 0)), accx, is_async=True)
    d = dsm.index(slot).load(s_layout).to(gl.float32)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    moved_sq += gl.sum(gl.reshape(d * d, [BM, BK // 8, 8]), 2)
    return acc, accx, moved_sq, h


@gluon.jit
def _output_gate(c, acc, accx, moved_sq, h, dsm, wsm, x3, Clean,
    change, zero, scale, base, wo_rows, fc_rows, next_rows, cols, token,
    R: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, BM: gl.constexpr,
    mma1: gl.constexpr, mma2: gl.constexpr, mmax: gl.constexpr, s_layout: gl.constexpr):
    item = 2 * c
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    old = dsm.index(c).load(mma2).to(gl.float32)
    out = warpgroup_mma(change, wsm.index((base + item) % NS), zero, use_acc=False, is_async=True)
    out, _, acc, accx = warpgroup_mma_wait(0, deps=[out, change, acc, accx])
    value = (old + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    moved = (value.to(gl.float32) * scale).to(gl.bfloat16)
    dsm.index(c).store(moved)
    cp.wait_group(NS - 3)
    fence_async_shared()
    gl.thread_barrier()
    _fused_issue(wsm.index((base + item + NS - 1) % NS), item + NS - 1,
        wo_rows, fc_rows, next_rows, cols, BK, KC)
    _fused_issue(wsm.index((base + item + NS) % NS), item + NS,
        wo_rows, fc_rows, next_rows, cols, BK, KC)
    operand = gl.convert_layout(moved, gl.DotOperandLayout(operand_index=0, parent=mma1, k_width=2))
    xoperand = gl.convert_layout(moved, gl.DotOperandLayout(operand_index=0, parent=mmax, k_width=2))
    acc = warpgroup_mma(operand, wsm.index((base + item + 1) % NS).permute((1, 0)), acc, is_async=True)
    accx = warpgroup_mma(xoperand, x3.index(8 + c).permute((1, 0)), accx, is_async=True)
    d = dsm.index(c).load(s_layout).to(gl.float32)
    acc, accx, operand, xoperand = warpgroup_mma_wait(0, deps=[acc, accx, operand, xoperand])
    acc = warpgroup_mma_init(acc)
    accx = warpgroup_mma_init(accx)
    if c == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    moved_sq += gl.sum(gl.reshape(d * d, [BM, BK // 8, 8]), 2)
    return acc, accx, moved_sq, h


@gluon.jit
def _chain3_kernel(delta_desc, w_desc, XA, FcA, WoA, RoutesA, CleanA, ScaleA,
    FcM, WoM, RoutesM, CleanM, ScaleM, FcB, WoB, RoutesB, CleanB,
    Diag, Stats, XPM, SQM, XPB, SQB, VP, TILES, VT, RIX, op_desc,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, NS: gl.constexpr,
    EPSILON: gl.constexpr, num_warps: gl.constexpr):
    BK: gl.constexpr = 64
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    NBYTES: gl.constexpr = BM * BK * 2
    IT: gl.constexpr = 6 * KC
    E: gl.constexpr = 0
    gl.static_assert(NS >= 2 and NS <= KC + 1)
    # 128 moves per tile (two warpgroups): the response weights in their own buffer
    gl.static_assert(BM == 2 * BK and KC == 8)
    G = gl.num_programs(0)
    pid = gl.program_id(0)

    dsm = gl.allocate_shared_memory(gl.bfloat16, [KC, BM, BK], delta_desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NS, R, BK],
      gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [KC, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(KC):
        mbarrier.init(dbar.index(i), count=1)
    astage = gl.allocate_shared_memory(gl.bfloat16, [2, 2 * BK, BK], delta_desc.layout)
    abar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        # the gathered rows: every thread's cp.async completion arrives
        mbarrier.init(abar.index(i), count=num_warps * 32)
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=num_warps * 32)
    wbuf = gl.allocate_shared_memory(gl.bfloat16, [2, BM, BK], delta_desc.layout)
    wbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(wbar, count=1)
    # -Op (hi, lo) of the tile's token per chunk, in the weights' buffer while block A's gate runs
    obuf = wbuf._reinterpret(gl.bfloat16, [KC, 16, BK], op_desc.layout)
    # the head selector E (2 BK x 16, E[16 h + k, c] = [c % 8 == h]) in the buffer's other half: the head
    # sums [s, s] = weights @ E by wgmma (fp32: exact sums of the bf16 weights)
    ebuf = wbuf.index(1)._reinterpret(gl.bfloat16, [2 * BK, 16],
      gl.NVMMASharedLayout.get_default_for([2 * BK, 16], gl.bfloat16))
    # the parts in the response slots: [32, 8, BK] tiles (x_m 0-7, x_b 8-15, vectors 16-23), diag after them
    t3 = astage._reinterpret(gl.bfloat16, [32, 8, BK], delta_desc.layout)
    q4 = astage._reinterpret(gl.bfloat16, [4, 8 * KC, BK], delta_desc.layout)
    dgv = astage.index(1)._reinterpret(gl.float32, [8, D],
      gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])).index(4)
    dg2 = astage.index(1)._reinterpret(gl.float32, [8 * KC, BK],
      gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))  # diag chunk c: row 4 KC + c
    mmax: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8, 16])
    row_first = (pid // MT) * P + (pid % MT) * BM
    _w_issue_wb(w_desc, wbar, wbuf, row_first, True, NBYTES, BK)
    for i in gl.static_range(KC):
        _reload(delta_desc, dbar, dsm, i, i, row_first, True, NBYTES, BK)

    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen_a = gl.load(RoutesA + (pid // MT) * R + rr).to(gl.int32)
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), FcA + chosen_a * D, cols, i * BK)

    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, R, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BK, 16])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma1, k_width=2)
    w_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
    # the first tile's gathered row indices (later tiles': prefetched a tile ahead)
    gv_bar = wbar
    r128 = gl.arange(0, 2 * BK, gl.SliceLayout(1, g_layout))
    rix_next = gl.load(RIX + (pid // MT) * (2 * BK) + r128)
    e_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [num_warps, 1], [1, 0])
    sel = (gl.arange(0, 2 * BK, gl.SliceLayout(1, e_layout))[:, None] // 16 ==
           gl.arange(0, 16, gl.SliceLayout(0, e_layout))[None, :] % 8)
    mmas: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 16, 16])
    sw_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mmas, k_width=2)
    s_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    xcols2 = gl.arange(0, BK, gl.SliceLayout(0, mma2))
    xa_next = gl.load(XA + (pid // MT) * D + xcols2)
    scale_a = gl.load(ScaleA).to(gl.float32)
    scale_m = gl.load(ScaleM).to(gl.float32)
    zero = gl.zeros([BM, BK], gl.float32, mma2)

    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        row0 = token * P + (tile % MT) * BM
        base = it * IT
        nxt_tile = tile + G
        has_next = nxt_tile < TILES
        nxt_token = gl.where(has_next, nxt_tile // MT, token)
        nxt_row0 = nxt_token * P + (nxt_tile % MT) * BM
        chosen_m = gl.load(RoutesM + token * R + rr).to(gl.int32)
        chosen_b = gl.load(RoutesB + token * R + rr).to(gl.int32)
        chosen_next = gl.load(RoutesA + nxt_token * R + rr).to(gl.int32)

        # block A: the response weights into registers, the first gathered-values chunk in flight
        fence_async_shared()
        gl.thread_barrier()  # every warp is done with the readout vectors in slot 1
        s, ph = _rslot_phase(1)
        prow = VT + rix_next.to(gl.int64) * D
        _gv_issue(astage, abar, s, prow, cols, 0, BK)
        rix_next = gl.load(RIX + nxt_token * (2 * BK) + r128)
        # the weights into registers, -Op into their buffer (the next tile's weights after the gate)
        mbarrier.wait(wbar, 0)
        w0 = wbuf.index(0).load(w_layout)
        w1 = wbuf.index(1).load(w_layout)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.expect(wbar, KC * 16 * BK * 2)
        for c in gl.static_range(KC):
            tma.async_copy_global_to_shared(op_desc, [token * 16, c * BK], wbar, obuf.index(c))
        ebuf.store(gl.where(sel, 1.0, 0.0).to(gl.bfloat16))
        fence_async_shared()
        gl.thread_barrier()
        sw0 = gl.convert_layout(w0, sw_layout)  # (the same registers: A operands of the N = 16 wgmma)
        sw1 = gl.convert_layout(w1, sw_layout)
        sums = warpgroup_mma(sw0, ebuf.slice(0, BK), gl.zeros([BM, 16], gl.float32, mmas), use_acc=False,
          is_async=True)
        sums = warpgroup_mma(sw1, ebuf.slice(BK, BK), sums, is_async=True)
        sums = warpgroup_mma_wait(0, deps=[sums, sw0, sw1])[0]
        a2 = gl.convert_layout(sums.to(gl.bfloat16), w_layout)  # (in registers: accumulator -> A operand)
        sq_x = gl.zeros([1], gl.float32, gl.SliceLayout(1, mma2))
        dots = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        moved_sq = dots
        acc = gl.zeros([BM, R], gl.float32, mma1)
        h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        for j in range(KC):
            acc, sq_x, dots, moved_sq, xa_next, h = _gate_resp(j, acc, sq_x, dots, moved_sq, xa_next, h, dbar, dsm,
              wsm, abar, astage, w0, w1, zero, XA, CleanA, it, base % NS, FcA + chosen_a * D,
              WoA + chosen_a * D, cols, xcols2, token, nxt_token, D=D, R=R, BK=BK, KC=KC, NS=NS,
              mma1=mma1, mma2=mma2, a_layout=a_layout, prow=prow,
              a2=a2, obuf=obuf, wbar=gv_bar)
        sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, mma1)
        # the response slots are free: this tile's parts in flight under block A's output
        fence_async_shared()
        gl.thread_barrier()
        # -Op is spent: the next tile's weights
        _w_issue_wb(w_desc, wbar, wbuf, nxt_row0, has_next, NBYTES, BK)
        _rows_issue(q4.index(0), XPM, token, True, D, BK, num_warps)
        _rows_issue(q4.index(1), XPB, token, True, D, BK, num_warps)
        _rows_issue(q4.index(2), VP, token, True, D, BK, num_warps)
        _diag_issue(dgv, Diag, token, True, D, num_warps)
        cp.mbarrier_arrive(vbar, increment_count=False)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2)
        for c in range(KC):
            _output_resp(c, dsm, wsm, change, zero, scale_a, it, (base + KC) % NS,
              WoA + chosen_a * D, FcM + chosen_m * D, cols, BK=BK, KC=KC, NS=NS, mma2=mma2)


        kc = KC + (TILES >> 31)  # = KC, opaque: LLVM unrolls the (short) TC gate loops otherwise (spills)
        mbarrier.wait(vbar, it & 1)  # the parts
        acc, accx, moved_sq, h = _gate_tc_init(BM, BK, R, s_layout, mma1, mmax)
        for j in range(kc):
            acc, accx, moved_sq, h = _gate_to_fused(j, acc, accx, moved_sq, h, dsm, wsm, t3, CleanM, it,
              (base + 2 * KC) % NS, FcM + chosen_m * D, WoM + chosen_m * D, FcB + chosen_b * D, cols,
              token, R=R, XOFF=0, BK=BK, KC=KC, NS=NS, E=E, mma1=mma1, s_layout=s_layout, BM=BM)
        acc, sq_x, dots, moved_sq = _gate_tc_done(acc, accx, moved_sq, gl.load(SQM + token), mma1, mmax)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2)
        gl.static_assert(NS >= 3)
        acc, accx, moved_sq, h = _gate_tc_init(BM, BK, R, s_layout, mma1, mmax)
        for c in range(kc):
            acc, accx, moved_sq, h = _output_gate(c, acc, accx, moved_sq, h, dsm, wsm, t3, CleanB,
                change, zero, scale_m, (base + 3 * KC) % NS, WoM + chosen_m * D,
                FcB + chosen_b * D, WoB + chosen_b * D, cols, token,
                R, BK, KC, NS, BM, mma1, mma2, mmax, s_layout)
        acc, sq_x, dots, moved_sq = _gate_tc_done(acc, accx, moved_sq, gl.load(SQB + token), mma1, mmax)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2)
        accr = gl.zeros([BM, 8], gl.float32, mmax)
        s1 = _row_sum(zero, BM, BK)
        s5 = s1
        for c in range(kc):
            accr, s1, s5 = _output_readout_tc(c, accr, s1, s5, delta_desc, dbar, dsm, wsm, t3, dg2, change, zero,
              it, (base + 5 * KC) % NS, WoB + chosen_b * D, FcA + chosen_next * D, cols, nxt_row0, has_next,
              BK=BK, KC=KC, NS=NS, E=E, NBYTES=NBYTES, mma2=mma2, mmax=mmax, BM=BM, VOFF=16, DOFF=4 * KC)
        s0, s1, s2, s3, s4, s5 = _readout_tc_done(accr, s1, s5, mma2, mmax)
        _stats_store(Stats, row0, s0, s1, s2, s3, s4, s5, BM, mma2)
        chosen_a = chosen_next
        it += 1
    cp.wait_group(0)
    for i in gl.static_range(KC):
        mbarrier.invalidate(dbar.index(i))
    for i in gl.static_range(2):
        mbarrier.invalidate(abar.index(i))
    mbarrier.invalidate(vbar)
    mbarrier.invalidate(wbar)


@torch.library.custom_op('visit_kernel::fused_resp_visit_readout_fast', mutates_args=())
def fused_resp_visit_readout_fast(delta: torch.Tensor, x_a: torch.Tensor, fc_a: torch.Tensor, wo_a: torch.Tensor,
    routes_a: torch.Tensor, clean_a: torch.Tensor, scale_a: torch.Tensor, add_weights: torch.Tensor,
    add_values: torch.Tensor, x_m: torch.Tensor, fc_m: torch.Tensor, wo_m: torch.Tensor, routes_m: torch.Tensor,
    clean_m: torch.Tensor, scale_m: torch.Tensor, x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor,
    routes_b: torch.Tensor, clean_b: torch.Tensor, xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, diag: torch.Tensor, gathered_rows: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    """Block A's visit of delta with the replay response add_weights @ add_values added, block M's visit and block
    B's readout visit in one kernel: the readout moments (n, P, 6) fp32. The response is computed once per chunk
    (block A's residual reuses the gate's bf16(delta + response); <= 1 bf16 ulp from the two-pass result).
    add_values is the per-key table (rows, D) the response's value rows are gathered from: row j of token t is
    table[gathered_rows[t, j]] + correction[t, h] + correction[t, 8 + h], h = j // 16 (gathered_rows (n, 128) int32,
    correction (n, 16, D) bf16); the correction enters weighted by each head's sum of add_weights."""
    f32 = lambda z: z.float().contiguous()
    blk = lambda x, fc, wo, routes, clean: (f32(x), fc.contiguous(), wo.contiguous(), routes.contiguous(), f32(clean))
    stats = torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
    delta = delta.contiguous()
    x_a, fc_a, wo_a, routes_a, clean_a = blk(x_a, fc_a, wo_a, routes_a, clean_a)
    add_weights, add_values = add_weights.contiguous(), add_values.contiguous()
    x_m, fc_m, wo_m, routes_m, clean_m = blk(x_m, fc_m, wo_m, routes_m, clean_m)
    x_b, fc_b, wo_b, routes_b, clean_b = blk(x_b, fc_b, wo_b, routes_b, clean_b)
    xl, target, mu, nu, diag = (f32(z) for z in (xl, target, mu, nu, diag))
    rows, correction = gathered_rows.contiguous(), correction.contiguous()
    n, p, d = delta.shape
    r = routes_a.shape[1]
    assert routes_m.shape[1] == routes_b.shape[1] == r == 64, 'R = 64 only (the shared-memory budget)'
    cfg = dict(CONFIG)
    bm = cfg.pop('BM')
    hk = add_weights.shape[-1]
    assert p % bm == 0 and d % 64 == 0 and hk == 128
    assert delta.is_contiguous() and delta.dtype == torch.bfloat16
    layout = gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16)
    delta_desc = TensorDescriptor.from_tensor(delta.view(n * p, d), [bm, 64], layout)
    w_desc = TensorDescriptor.from_tensor(add_weights.reshape(n * p, hk), [bm, 64], layout)
    assert add_values.dim() == 2 and add_values.shape[1] == d
    assert rows.shape == (n, hk) and rows.dtype == torch.int32 and correction.shape == (n, 16, d)
    op_desc = TensorDescriptor.from_tensor(correction.reshape(n * 16, d), [16, 64],
      gl.NVMMASharedLayout.get_default_for([16, 64], gl.bfloat16))
    tiles, sms = n * (p // bm), torch.cuda.get_device_properties(delta.device).multi_processor_count
    xpm, sqm = split_parts([x_m], 3, square=True)
    xpb, sqb = split_parts([x_b], 3, square=True)
    vp, _ = split_parts([xl, target, mu, nu], 2)
    _chain3_kernel[(min(tiles, sms),)](delta_desc, w_desc, x_a, fc_a, wo_a, routes_a, clean_a,
      scale_a, fc_m, wo_m, routes_m, clean_m, scale_m, fc_b, wo_b, routes_b, clean_b,
      diag, stats, xpm, sqm, xpb, sqb, vp, tiles, add_values, rows, op_desc, p, d, r, bm, EPSILON=EPS, **cfg)
    return stats


@fused_resp_visit_readout_fast.register_fake
def _(delta, x_a, fc_a, wo_a, routes_a, clean_a, scale_a, add_weights, add_values, x_m, fc_m, wo_m, routes_m, clean_m,
    scale_m, x_b, fc_b, wo_b, routes_b, clean_b, xl, target, mu, nu, diag, gathered_rows, correction):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
