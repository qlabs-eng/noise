"""Gluon (Hopper) three-block chain: the replay-response visit (block A: MODE 1 with the attention replay's response
bf16(add_weights @ add_values) added on load, formed in the kernel), a plain MODE 1 visit (block M) and the MODE 2
readout visit (block B) back to back per tile, the tile's delta kept in shared memory throughout.

fused_resp_visit_readout_fast is numerically visit.fused_visit_fast(add_weights=, add_values=) then
chain.fused_visit_readout_fast (bitwise with DEFER 0). Ring of routed rows: per tile Fc A, Wo A, Fc M, Wo M, Fc B, Wo B
(6 KC items). Response staging ring (2 x 32 KB TMA slots): per tile the weights (slot 0) and the projected values for
block A's gate and residual chunks (16 items alternating slots 1, 0), padded to 18 items so the weights always land in
slot 0; slot 1 is free while blocks M and B run and holds the readout vectors then (shared memory: 2 CTAs per SM).
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from visit_chain import (EPS, CTAS_PER_SM, _SMS, _change, _gate, _gate_done, _gate_init, _output_keep,
  _output_readout, _bf2f, _issue_rows, _pick, _rbf, _reload, _row_done, _row_sum, _rows, _vec_issue)
import visit_chain

CONFIG = dict(BM=64, NS=2, DEFER=0, num_warps=4)  # DEFER 1 (as chain.py): no faster here (255 registers)


@gluon.jit
def _ld(buf, layout: gl.constexpr, FAST: gl.constexpr):
    """A bf16 chunk from shared memory as f32 (FAST 32: one ALU op per element)."""
    if FAST // 32 % 2 == 1:
        return _bf2f(buf.load(layout))
    else:
        return buf.load(layout).to(gl.float32)


@gluon.jit
def _rslot_phase(it, r):
    """Slot and phase of the tile's response item r (slot 0: 9 loads per tile, slot 1: 8)."""
    return r % 2, gl.where(r % 2 == 0, (9 * it + r // 2) & 1, (r // 2) & 1)


@gluon.jit
def _w_issue(w_desc, abar, astage, row0, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """The tile's response weights (BM x 2 BK) into slot 0, as two BK-column halves."""
    mbarrier.expect(abar.index(0), 2 * NBYTES, pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, 0], abar.index(0), astage.index(0).slice(0, BK), pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, BK], abar.index(0), astage.index(0).slice(BK, BK), pred=pred)


@gluon.jit
def _pv_issue(pv_desc, abar, astage, slot, token, c, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """The token's projected values (2 BK x BK), columns of chunk c, into slot."""
    mbarrier.expect(abar.index(slot), 2 * NBYTES, pred=pred)
    tma.async_copy_global_to_shared(pv_desc, [token * 2 * BK, c * BK], abar.index(slot), astage.index(slot),
      pred=pred)


@gluon.jit
def _resp_take(abar, astage, slot, phase, w0, w1, zero, BK: gl.constexpr, FAST: gl.constexpr = 0):
    """The add chunk from slot: bf16(weights @ projected values chunk), as the reference's bmm rounds it (FAST 1:
    returned as f32, the same values)."""
    mbarrier.wait(abar.index(slot), phase)
    acc = warpgroup_mma(w0, astage.index(slot).slice(0, BK), zero, use_acc=False, is_async=True)
    acc = warpgroup_mma(w1, astage.index(slot).slice(BK, BK), acc, is_async=True)
    acc = warpgroup_mma_wait(0, deps=[acc, w0, w1])[0]
    if FAST % 2 == 1:
        return _rbf(acc)
    else:
        return acc.to(gl.bfloat16)


@gluon.jit
def _gate_resp(j, acc, sq_x, dots, moved_sq, x_next, h, dbar, dsm, wsm, abar, astage, pv_desc, w0, w1, zero, X, Clean,
    it, base, fc_rows, wo_rows, cols, xcols, token, nxt_token,
    D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr,
    NBYTES: gl.constexpr, DEFER: gl.constexpr, mma1: gl.constexpr, mma2: gl.constexpr, a_layout: gl.constexpr,
    FAST: gl.constexpr = 0):
    """Block A's gate chunk j: d = delta + response chunk j (fp32), acc += bf16(d) @ Fc[routes]^T (A from registers,
    waited at once); the norm sums of d."""
    slot = (it * KC + j) % KC
    if FAST // 64 % 2 == 1:  # x chunk j loaded here, under the waits (no prefetch registers rotating in the loop)
        x = gl.load(X + token * D + j * BK + xcols)
    mbarrier.wait(dbar.index(slot), it & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + j + NS - 1) % NS), _pick(j + NS - 1 < KC, fc_rows, wo_rows, FAST), cols,
      ((j + NS - 1) % KC) * BK, FAST)
    r = 1 + j
    s, ph = _rslot_phase(it, r + 1)
    _pv_issue(pv_desc, abar, astage, s, token, (j + 1) % KC, True, NBYTES, BK)  # the next item (residual's first)
    s, ph = _rslot_phase(it, r)
    addv = _resp_take(abar, astage, s, ph, w0, w1, zero, BK, FAST)
    if FAST % 2 == 1:
        d = _ld(dsm.index(slot), mma2, FAST) + addv
    else:
        d = dsm.index(slot).load(mma2).to(gl.float32) + addv.to(gl.float32)
    a_op = gl.convert_layout(d.to(gl.bfloat16), a_layout)
    acc = warpgroup_mma(a_op, wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    acc, a_op = warpgroup_mma_wait(0, deps=[acc, a_op])
    if FAST // 64 % 2 == 0:
        x = x_next
        x_next = gl.load(X + gl.where(j + 1 < KC, token, nxt_token) * D + ((j + 1) % KC) * BK + xcols)
    if FAST // 128 % 2 == 0:
        if j == KC - 2:
            h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    if DEFER:
        sq_x += x * x
        dots += _row_sum(d * x[None, :], BM, BK, True)
        moved_sq += _row_sum(d * d, BM, BK, True)
    else:
        sq_x += gl.sum(gl.expand_dims(x * x, 0), 1)
        dots += gl.sum(d * x[None, :], 1)
        moved_sq += gl.sum(d * d, 1)
    return acc, sq_x, dots, moved_sq, x_next, h


@gluon.jit
def _output_resp(c, dsm, wsm, abar, astage, pv_desc, w0, w1, change, zero, scale, it, base, token, wo_rows, next_rows,
    cols, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, NBYTES: gl.constexpr, mma2: gl.constexpr,
    FAST: gl.constexpr = 0):
    """Block A's output chunk c: bf16(bf16(delta + response chunk c + change @ Wo[routes][:, c]) * scale) back into
    delta chunk c."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), _pick(c + NS - 1 < KC, wo_rows, next_rows, FAST), cols,
      ((c + NS - 1) % KC) * BK, FAST)
    slot = (it * KC + c) % KC
    r = 1 + KC + c
    s, ph = _rslot_phase(it, r + 1)
    _pv_issue(pv_desc, abar, astage, s, token, (c + 1) % KC, c + 1 < KC, NBYTES, BK)
    s, ph = _rslot_phase(it, r)
    addv = _resp_take(abar, astage, s, ph, w0, w1, zero, BK, FAST)
    if FAST % 2 == 1:
        old = _ld(dsm.index(slot), mma2, FAST) + addv
        out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
        out = warpgroup_mma_wait(0, deps=[out, change])[0]
        value = _rbf(old + _rbf(out))
        dsm.index(slot).store((value * scale).to(gl.bfloat16))
    else:
        old = dsm.index(slot).load(mma2).to(gl.float32) + addv.to(gl.float32)
        out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
        out = warpgroup_mma_wait(0, deps=[out, change])[0]
        value = (old + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
        dsm.index(slot).store((value.to(gl.float32) * scale).to(gl.bfloat16))


@gluon.jit
def _chain3_kernel(delta_desc, w_desc, pv_desc, XA, FcA, WoA, RoutesA, CleanA, ScaleA,
    XM, FcM, WoM, RoutesM, CleanM, ScaleM, XB, FcB, WoB, RoutesB, CleanB,
    XL, Target, Mu, Nu, Diag, Stats, TILES,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, NS: gl.constexpr, DEFER: gl.constexpr,
    EPSILON: gl.constexpr, num_warps: gl.constexpr, FAST: gl.constexpr = 0):
    BK: gl.constexpr = 64
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    NBYTES: gl.constexpr = BM * BK * 2
    IT: gl.constexpr = 6 * KC
    E: gl.constexpr = 0
    DG: gl.constexpr = DEFER % 2 == 1
    DR: gl.constexpr = DEFER >= 2
    gl.static_assert(NS >= 2 and NS <= KC + 1 and BM == BK)
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
        mbarrier.init(abar.index(i), count=1)
    # the readout vectors (xl, target, mu, nu, diag) in response slot 1, free while blocks M and B run
    vsm = astage.index(1)._reinterpret(gl.float32, [5, D],
      gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=num_warps * 32)
    row_first = (pid // MT) * P + (pid % MT) * BM
    _w_issue(w_desc, abar, astage, row_first, True, NBYTES, BK)
    for i in gl.static_range(KC):
        _reload(delta_desc, dbar, dsm, i, i, row_first, True, NBYTES, BK)

    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen_a = gl.load(RoutesA + (pid // MT) * R + rr).to(gl.int32)
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), _rows(FcA, chosen_a, D, FAST), cols, i * BK, FAST)

    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, R, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BK, 16])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma1, k_width=2)
    w_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
    s_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    xcols2 = gl.arange(0, BK, gl.SliceLayout(0, mma2))
    xcols = gl.arange(0, BK, gl.SliceLayout(0, s_layout))
    if FAST // 64 % 2 == 1:
        xa_next = 0.0
        xm_next = 0.0
        xm_after = 0.0
        xb_next = 0.0
        xb_after = 0.0
    else:
        xa_next = gl.load(XA + (pid // MT) * D + xcols2)
        xm_next = gl.load(XM + (pid // MT) * D + xcols)
        xm_after = gl.load(XM + (pid // MT) * D + BK + xcols)
        xb_next = gl.load(XB + (pid // MT) * D + xcols)
        xb_after = gl.load(XB + (pid // MT) * D + BK + xcols)
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

        # block A: the response weights into registers, the first projected-values chunk in flight
        fence_async_shared()
        gl.thread_barrier()  # every warp is done with the readout vectors in slot 1
        s, ph = _rslot_phase(it, 1)
        _pv_issue(pv_desc, abar, astage, s, token, 0, True, NBYTES, BK)
        mbarrier.wait(abar.index(0), it & 1)
        w0 = astage.index(0).slice(0, BK).load(w_layout)
        w1 = astage.index(0).slice(BK, BK).load(w_layout)
        if DG:
            sq_x = gl.zeros([BK], gl.float32, gl.SliceLayout(0, mma2))
            dots = _row_sum(zero, BM, BK, True)
            moved_sq = dots
        else:
            sq_x = gl.zeros([1], gl.float32, gl.SliceLayout(1, mma2))
            dots = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
            moved_sq = dots
        acc = gl.zeros([BM, R], gl.float32, mma1)
        if FAST // 128 % 2 == 1:  # the clean hidden row before the loop, not a predicated load in every step
            h = gl.load(CleanA + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
        else:
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        for j in range(KC):
            acc, sq_x, dots, moved_sq, xa_next, h = _gate_resp(j, acc, sq_x, dots, moved_sq, xa_next, h, dbar, dsm,
              wsm, abar, astage, pv_desc, w0, w1, zero, XA, CleanA, it, base % NS, _rows(FcA, chosen_a, D, FAST),
              _rows(WoA, chosen_a, D, FAST), cols, xcols2, token, nxt_token, D=D, R=R, BM=BM, BK=BK, KC=KC, NS=NS,
              NBYTES=NBYTES, DEFER=DG, mma1=mma1, mma2=mma2, a_layout=a_layout, FAST=FAST)
        sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, FAST)
        for c in range(KC):
            _output_resp(c, dsm, wsm, abar, astage, pv_desc, w0, w1, change, zero, scale_a, it, (base + KC) % NS,
              token, _rows(WoA, chosen_a, D, FAST), _rows(FcM, chosen_m, D, FAST), cols, BK=BK, KC=KC, NS=NS,
              NBYTES=NBYTES, mma2=mma2, FAST=FAST)

        # the response slots are free: the next tile's weights into slot 0, this tile's readout vectors into slot 1
        fence_async_shared()
        gl.thread_barrier()
        _w_issue(w_desc, abar, astage, nxt_row0, has_next, NBYTES, BK)
        _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, token, True, D, num_warps)

        # block M: a plain visit of block A's output in shared memory, written back into the tile
        sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
        acc = warpgroup_mma_init(gl.zeros([BM, R], gl.float32, mma1))
        if FAST // 128 % 2 == 1:  # the clean hidden row before the loop, not a predicated load in every step
            h = gl.load(CleanM + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
        else:
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        for j in range(KC):
            acc, sq_x, dots, moved_sq, xm_next, xm_after, h = _gate(j, acc, sq_x, dots, moved_sq, xm_next, xm_after,
              h, dbar, dsm, wsm, XM, CleanM, it, (base + 2 * KC) % NS, _rows(FcM, chosen_m, D, FAST),
              _rows(WoM, chosen_m, D, FAST), cols, xcols, token, nxt_token, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E,
              LOADED=False, mma1=mma1, s_layout=s_layout, BM=BM, DEFER=DG, FAST=FAST)
        acc = warpgroup_mma_wait(0, deps=[acc])
        sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, FAST)
        for c in range(KC):
            _output_keep(c, dbar, dsm, wsm, change, zero, 0.0, scale_m, it, (base + 3 * KC) % NS,
              _rows(WoM, chosen_m, D, FAST), _rows(FcB, chosen_b, D, FAST), cols, BK=BK, KC=KC, NS=NS, E=E, ROOT=False,
              mma2=mma2, FAST=FAST)

        # block B: the readout visit; its chunks reload with the next tile's delta
        sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
        acc = warpgroup_mma_init(gl.zeros([BM, R], gl.float32, mma1))
        if FAST // 128 % 2 == 1:  # the clean hidden row before the loop, not a predicated load in every step
            h = gl.load(CleanB + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
        else:
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        for j in range(KC):
            acc, sq_x, dots, moved_sq, xb_next, xb_after, h = _gate(j, acc, sq_x, dots, moved_sq, xb_next, xb_after,
              h, dbar, dsm, wsm, XB, CleanB, it, (base + 4 * KC) % NS, _rows(FcB, chosen_b, D, FAST),
              _rows(WoB, chosen_b, D, FAST), cols, xcols, token, nxt_token, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E,
              LOADED=False, mma1=mma1, s_layout=s_layout, BM=BM, DEFER=DG, FAST=FAST)
        acc = warpgroup_mma_wait(0, deps=[acc])
        sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, FAST)
        s0 = _row_sum(zero, BM, BK, DR)
        s1 = s0
        s2 = s0
        s3 = s0
        s4 = s0
        s5 = s0
        for c in gl.static_range(KC):  # unrolled: the readout vectors are sliced at constant offsets
            s0, s1, s2, s3, s4, s5 = _output_readout(c, s0, s1, s2, s3, s4, s5, delta_desc, dbar, dsm, wsm, vsm,
              vbar, change, zero, it, (base + 5 * KC) % NS, _rows(WoB, chosen_b, D, FAST),
              _rows(FcA, chosen_next, D, FAST), cols, nxt_row0, has_next, BK=BK, KC=KC, NS=NS, E=E, NBYTES=NBYTES,
              mma2=mma2, BM=BM, DEFER=DR, FAST=FAST)
        if DR:
            s0, s1, s2, s3, s4, s5 = (_row_done(s0, mma2), _row_done(s1, mma2), _row_done(s2, mma2),
              _row_done(s3, mma2), _row_done(s4, mma2), _row_done(s5, mma2))
        stats_rows = Stats + (row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))).to(gl.int64) * 6
        gl.store(stats_rows, s0)
        gl.store(stats_rows + 1, s1)
        gl.store(stats_rows + 2, s2)
        gl.store(stats_rows + 3, s3)
        gl.store(stats_rows + 4, s4)
        gl.store(stats_rows + 5, s5)
        chosen_a = chosen_next
        it += 1
    cp.wait_group(0)
    for i in gl.static_range(KC):
        mbarrier.invalidate(dbar.index(i))
    for i in gl.static_range(2):
        mbarrier.invalidate(abar.index(i))
    mbarrier.invalidate(vbar)


def launch_chain3(delta, block_a, add_weights, add_values, block_m, block_b, vectors, stats, **overrides):
    """block_a / block_m: (x, fc, wo, routes, clean, scale); block_b: (x, fc, wo, routes, clean)."""
    n, p, d = delta.shape
    r = block_a[3].shape[1]
    assert block_m[3].shape[1] == block_b[3].shape[1] == r == 64, 'R = 64 only (the shared-memory budget)'
    cfg = dict(CONFIG, **overrides)
    if visit_chain.CHAIN_DEFER:  # (visit_chain.py lists the FAST bits)
        cfg.setdefault('FAST', visit_chain.CHAIN3_FAST)
        if visit_chain.CHAIN_DEFER == 1 and 'DEFER' not in overrides:
            cfg['DEFER'] = 2
    bm = cfg.pop('BM')
    hk = add_weights.shape[-1]
    assert p % bm == 0 and d % 64 == 0 and bm == 64 and hk == 128 and add_values.shape[1] == hk
    assert delta.is_contiguous() and delta.dtype == torch.bfloat16
    layout = gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16)
    delta_desc = TensorDescriptor.from_tensor(delta.view(n * p, d), [bm, 64], layout)
    w_desc = TensorDescriptor.from_tensor(add_weights.reshape(n * p, hk), [bm, 64], layout)
    pv_desc = TensorDescriptor.from_tensor(add_values.reshape(n * hk, d), [hk, 64],
      gl.NVMMASharedLayout.get_default_for([hk, 64], gl.bfloat16))
    dev = delta.device.index
    if dev not in _SMS:
        _SMS[dev] = (132 if delta.device.type == 'cpu'
          else torch.cuda.get_device_properties(delta.device).multi_processor_count)
    tiles = n * (p // bm)
    _chain3_kernel[(min(tiles, CTAS_PER_SM * _SMS[dev]),)](delta_desc, w_desc, pv_desc, *block_a, *block_m, *block_b,
      *vectors, stats, tiles, p, d, r, bm, EPSILON=EPS, **cfg)
    return stats


@torch.library.custom_op('visit_kernel::fused_resp_visit_readout_fast', mutates_args=())
def fused_resp_visit_readout_fast(delta: torch.Tensor, x_a: torch.Tensor, fc_a: torch.Tensor, wo_a: torch.Tensor,
    routes_a: torch.Tensor, clean_a: torch.Tensor, scale_a: torch.Tensor, add_weights: torch.Tensor,
    add_values: torch.Tensor, x_m: torch.Tensor, fc_m: torch.Tensor, wo_m: torch.Tensor, routes_m: torch.Tensor,
    clean_m: torch.Tensor, scale_m: torch.Tensor, x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor,
    routes_b: torch.Tensor, clean_b: torch.Tensor, xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """chain.fused_visit_readout_fast(visit.fused_visit_fast(delta, x_a, fc_a, wo_a, routes_a, clean_a, scale_a,
    add_weights=add_weights, add_values=add_values), x_m, fc_m, wo_m, routes_m, clean_m, scale_m, x_b, fc_b, wo_b,
    routes_b, clean_b, xl, target, mu, nu, diag) in one kernel: the readout moments (n, P, 6) fp32."""
    f32 = lambda z: z.float().contiguous()
    blk = lambda x, fc, wo, routes, clean: (f32(x), fc.contiguous(), wo.contiguous(), routes.contiguous(), f32(clean))
    stats = torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
    return launch_chain3(delta.contiguous(), (*blk(x_a, fc_a, wo_a, routes_a, clean_a), scale_a),
      add_weights.contiguous(), add_values.contiguous(), (*blk(x_m, fc_m, wo_m, routes_m, clean_m), scale_m),
      blk(x_b, fc_b, wo_b, routes_b, clean_b), tuple(f32(z) for z in (xl, target, mu, nu, diag)), stats)


@fused_resp_visit_readout_fast.register_fake
def _(delta, x_a, fc_a, wo_a, routes_a, clean_a, scale_a, add_weights, add_values, x_m, fc_m, wo_m, routes_m, clean_m,
    scale_m, x_b, fc_b, wo_b, routes_b, clean_b, xl, target, mu, nu, diag):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
