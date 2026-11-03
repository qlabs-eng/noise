"""Gluon (Hopper) kernel for one fused chain MLP visit, numerically the reference _fused_visit_kernel: MODE 1, 2 and 3,
an injected signed-Hadamard partner move (INJECT) and an added stream (ADD, MODE 1).

Persistent CTAs (two per SM) walk tiles of BM = 64 moves of one token. A tile's delta stays in shared memory for both
GEMMs (TMA in, 8 chunks of 64 columns); the routed c_fc / c_proj^T rows stream through a cp.async ring; both GEMMs run
on wgmma. The moved block input d = delta (+ add) (+ step * frame) feeds the gate GEMM straight from shared memory, or,
with INJECT / ADD, from registers, and is formed again for the residual. MODE 1 writes bf16(bf16(d + routed MLP change)
* scale) into the delta chunk in shared memory and sends it out by TMA; MODE 3 also sums the stored output against the
next block's clean input; MODE 2 reduces the moved stream to the six readout moments instead. While a tile's output
GEMM runs, the next tile's delta reloads into the chunks whose stores have left and its first c_fc chunks enter the
ring.
"""
from typing import Optional

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

EPS = torch.finfo(torch.float32).eps
# NS: weight ring slots; SL: TMA stores a delta chunk's slot may leave pending before it takes the next tile's chunk
# (the last SL - E chunks load during the next tile's gate GEMM); E: spare delta slots (KC + E rotate); per routes
# count R. R = 64 sweep (n = 2048): SL 7 / E 2 best for plain visits (-1.5% vs SL 6 / E 0), SL 2 / E 2 for injected
# ones (-3%)
CONFIGS = {128: dict(BM=64, NS=3, SL=6, E=0, num_warps=4), 64: dict(BM=64, NS=3, SL=7, E=2, num_warps=4)}
# chunk loops unrolled for the plain MODE 1 visits and R = 64 MODE 2 (measured faster there), runtime loops otherwise
_SMS = {}
ADD_VIA_INJECT = False  # add-only visits through the add + inject build (zero steps): the fallback from before the add
# staging ring, when the add-only build (add loaded into registers) faulted
CTAS_PER_SM = 2


@gluon.jit
def _issue_rows(buf, rows, cols, c0):
    cp.async_copy_global_to_shared(buf, rows[:, None] + (c0 + cols)[None, :])
    cp.commit_group()


@gluon.jit
def _reload(delta_desc, dbar, dsm, slot, c, row0, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """TMA of delta chunk c of the tile at row0 into shared memory slot `slot`."""
    mbarrier.expect(dbar.index(slot), NBYTES, pred=pred)
    tma.async_copy_global_to_shared(delta_desc, [row0, c * BK], dbar.index(slot), dsm.index(slot), pred=pred)


@gluon.jit
def _store_wait(pending, SL: gl.constexpr):
    """tma.store_wait with a runtime count of pending stores (< SL)."""
    for v in gl.static_range(SL):
        if pending == v:
            tma.store_wait(v)


@gluon.jit
def _add_issue(add_desc, abar, astage, k, row0, c, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """TMA load k of the add stream (per tile: the gate's chunks 0..KC-1, then the residual's) into slot k % 2."""
    mbarrier.expect(abar.index(k % 2), NBYTES, pred=pred)
    tma.async_copy_global_to_shared(add_desc, [row0, c * BK], abar.index(k % 2), astage.index(k % 2), pred=pred)


@gluon.jit
def _add_take(abar, astage, k, layout: gl.constexpr):
    mbarrier.wait(abar.index(k % 2), (k // 2) & 1)
    return astage.index(k % 2).load(layout)


@gluon.jit
def _w_issue(w_desc, abar, astage, k, row0, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """RESP ring item k: the tile's response weights (BM x 2 BK) as two BK-column halves."""
    mbarrier.expect(abar.index(k % 2), 2 * NBYTES, pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, 0], abar.index(k % 2), astage.index(k % 2).slice(0, BK),
      pred=pred)
    tma.async_copy_global_to_shared(w_desc, [row0, BK], abar.index(k % 2), astage.index(k % 2).slice(BK, BK),
      pred=pred)


@gluon.jit
def _pv_issue(pv_desc, abar, astage, k, token, c, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """RESP ring item k: the token's projected values (2 BK x BK), columns of chunk c."""
    mbarrier.expect(abar.index(k % 2), 2 * NBYTES, pred=pred)
    tma.async_copy_global_to_shared(pv_desc, [token * 2 * BK, c * BK], abar.index(k % 2), astage.index(k % 2),
      pred=pred)


@gluon.jit
def _w_take(abar, astage, k, layout: gl.constexpr, BK: gl.constexpr):
    mbarrier.wait(abar.index(k % 2), (k // 2) & 1)
    return astage.index(k % 2).slice(0, BK).load(layout), astage.index(k % 2).slice(BK, BK).load(layout)


@gluon.jit
def _resp_take(abar, astage, k, w0, w1, zero, BK: gl.constexpr):
    """The add chunk from ring item k: bf16(weights @ projected values chunk), as the reference's bmm rounds it."""
    mbarrier.wait(abar.index(k % 2), (k // 2) & 1)
    acc = warpgroup_mma(w0, astage.index(k % 2).slice(0, BK), zero, use_acc=False, is_async=True)
    acc = warpgroup_mma(w1, astage.index(k % 2).slice(BK, BK), acc, is_async=True)
    acc = warpgroup_mma_wait(0, deps=[acc, w0, w1])[0]
    return acc.to(gl.bfloat16)


@gluon.jit
def _frame_step(step, SignBits, moves, c, lcols, BK: gl.constexpr):
    """step * the signed Sylvester Hadamard frame at (moves, c * BK + lcols), exactly: frame entries are
    (-1)^(popcount(move & col) + signbit[col]), signs packed 32 per int32 word (BK = 64: two words per chunk)."""
    lo = gl.load(SignBits + 2 * c)
    hi = gl.load(SignBits + 2 * c + 1)
    sbit = (gl.where(lcols < 32, lo, hi) >> (lcols & 31)) & 1
    bits = moves[:, None] & (c * BK + lcols)[None, :]
    odd = gl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [bits], dtype=gl.int32, is_pure=True, pack=1)
    return (step.to(gl.int32, bitcast=True) ^ ((odd ^ sbit[None, :]) << 31)).to(gl.float32, bitcast=True)


def sign_bits(signs):
    """signs (D,) of +-1 as bits (1 = negative), 32 per int32 word."""
    neg = (signs < 0).view(-1, 32).to(torch.int64) << torch.arange(32, device=signs.device)
    return neg.sum(1).to(torch.int32)


@gluon.jit
def _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, token, pred, D: gl.constexpr, num_warps: gl.constexpr):
    """MODE 2: cp.async of the token's 5 readout vectors into vsm; every thread's completion arrives on vbar."""
    lay: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
    dd = gl.arange(0, D, lay)
    off = token * D + dd
    live = (dd < D) & pred
    cp.async_copy_global_to_shared(vsm.index(0), XL + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(1), Target + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(2), Mu + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(3), Nu + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(4), Diag + off, mask=live)
    cp.mbarrier_arrive(vbar, increment_count=False)


@gluon.jit
def _moved(old, layout: gl.constexpr, addv, Signs, step, move0, c, opaque,
    BM: gl.constexpr, BK: gl.constexpr, INJECT: gl.constexpr, ADD: gl.constexpr):
    """Delta chunk c (bf16, in layout) as the fp32 moved stream, in the reference's order: + add chunk c (addv, from
    the TMA staging ring), + step * frame (_frame_step, exact). opaque: a zero the compiler cannot see through,
    different at each call site: it hides that move0 is a multiple of BM and keeps the gate's and the residual's frame
    entries apart, else the compiler hoists tile-invariant parts out of the tile loop or keeps the gate's 8 chunks of
    entries live until the output GEMM reuses them, and spills."""
    d = old.to(gl.float32)
    rows = gl.arange(0, BM, gl.SliceLayout(1, layout))
    cols = c * BK + gl.arange(0, BK, gl.SliceLayout(0, layout))
    if ADD:
        d += addv.to(gl.float32)
    if INJECT:
        d += _frame_step(step, Signs, move0 + opaque + rows, c, gl.arange(0, BK, gl.SliceLayout(0, layout)), BK)
    return d


@gluon.jit
def _gate_chunk(j, acc, sq_x, dots, moved_sq, x_next, x_after, h,
    delta_desc, dbar, dsm, wsm, add_desc, pv_desc, w0, w1, zero, abar, astage, Signs, X, Clean, step, row0, move0, it,
    phase, base, fc_rows,
    wo_rows, cols, xcols, token, nxt_token,
    D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr,
    SL: gl.constexpr, E: gl.constexpr, NBYTES: gl.constexpr, STORES: gl.constexpr, FRESH: gl.constexpr,
    INJECT: gl.constexpr, ADD: gl.constexpr, RESP: gl.constexpr, mma1: gl.constexpr, mma2: gl.constexpr, a_layout: gl.constexpr,
    s_layout: gl.constexpr):
    """Gate GEMM chunk j: acc += d @ Fc[routes]^T over columns j; chunk j's norm sums run under its wgmma."""
    if STORES:
        if j < SL - E:  # the previous tile's last stores have left: their slots take this tile's last chunks
            _store_wait(SL - 1 - j, SL)
            _reload(delta_desc, dbar, dsm, (it * KC + KC - SL + E + j) % (KC + E), KC - SL + E + j, row0, it > 0,
              NBYTES, BK)
    g = it * KC + j  # chunk j's slot in the rotation over KC + E slots
    slot = g % (KC + E)
    mbarrier.wait(dbar.index(slot), (g // (KC + E)) & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    acc = warpgroup_mma_wait(0, deps=[acc])
    gl.thread_barrier()
    _issue_rows(wsm.index((base + j + NS - 1) % NS), gl.where(j + NS - 1 < KC, fc_rows, wo_rows), cols,
      ((j + NS - 1) % KC) * BK)
    w_op = wsm.index((base + j) % NS).permute((1, 0))
    if FRESH:
        addv = 0.0
        if RESP:  # ring items per tile: weights, then the projected values for the gate's and the residual's chunks
            k = it * (2 * KC + 1) + 1 + j
            _pv_issue(pv_desc, abar, astage, k + 1, token, (j + 1) % KC, True, NBYTES, BK)
            addv = _resp_take(abar, astage, k, w0, w1, zero, BK)
        elif ADD:  # the add stream's next chunk (slot free: read last iteration, before the barrier above)
            k = it * 2 * KC + j
            _add_issue(add_desc, abar, astage, k + 1, row0, (j + 1) % KC, True, NBYTES, BK)
            addv = _add_take(abar, astage, k, mma2)
        d = _moved(dsm.index(slot).load(mma2), mma2, addv, Signs, step, move0, j, row0 >> 30, BM, BK, INJECT, ADD)
        # a wgmma reading A from registers is waited for at once: the wait's deps keep a_op alive but not
        # the packed registers the wgmma reads, which the compiler may otherwise reuse while it runs
        a_op = gl.convert_layout(d.to(gl.bfloat16), a_layout)
        acc = warpgroup_mma(a_op, w_op, acc, is_async=True)
        acc, a_op = warpgroup_mma_wait(0, deps=[acc, a_op])
    else:
        acc = warpgroup_mma(dsm.index(slot), w_op, acc, is_async=True)
        d = dsm.index(slot).load(s_layout).to(gl.float32)
    x = x_next
    if FRESH:
        x_next = gl.load(X + gl.where(j + 1 < KC, token, nxt_token) * D + ((j + 1) % KC) * BK + xcols)
    else:
        x_next = x_after
        x_after = gl.load(X + gl.where(j + 2 < KC, token, nxt_token) * D + ((j + 2) % KC) * BK + xcols)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    sq_x += gl.sum(gl.expand_dims(x * x, 0), 1)
    dots += gl.sum(d * x[None, :], 1)
    moved_sq += gl.sum(d * d, 1)
    return acc, sq_x, dots, moved_sq, x_next, x_after, h


@gluon.jit
def _output_chunk(c, q0, q1, s0, s1, s2, s3, s4, s5,
    delta_desc, out_desc, dbar, dsm, wsm, change, zero, add_desc, pv_desc, w_desc, w0, w1, abar, astage, vsm, vbar,
    Signs, XL, Target, Mu, Nu, Diag, Fc, step, scale, row0, move0, it, token, nxt_row0, has_next, base, wo_rows,
    chosen_next, cols,
    D: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, SL: gl.constexpr,
    E: gl.constexpr, NBYTES: gl.constexpr, MODE: gl.constexpr, STORES: gl.constexpr, FRESH: gl.constexpr, INJECT: gl.constexpr,
    ADD: gl.constexpr, RESP: gl.constexpr, mma2: gl.constexpr, s_layout: gl.constexpr):
    """Output GEMM chunk c = change @ Wo[routes][:, chunk c], added to the moved stream's chunk c."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    # the last chunks of the ring fetch the next tile's first c_fc chunks
    _issue_rows(wsm.index((base + KC + c + NS - 1) % NS),
      gl.where(c + NS - 1 < KC, wo_rows, Fc + chosen_next * D), cols, ((c + NS - 1) % KC) * BK)
    slot = (it * KC + c) % (KC + E)
    old = dsm.index(slot).load(mma2)
    if FRESH:
        addv = 0.0
        if RESP:  # the residual's chunk c; then the next item (the next tile's weights after the last)
            k = it * (2 * KC + 1) + 1 + KC + c
            if c + 1 < KC:
                _pv_issue(pv_desc, abar, astage, k + 1, token, c + 1, True, NBYTES, BK)
            else:
                _w_issue(w_desc, abar, astage, k + 1, nxt_row0, has_next, NBYTES, BK)
            addv = _resp_take(abar, astage, k, w0, w1, zero, BK)
        elif ADD:  # the residual's add chunk c; then the next one (the next tile's first gate chunk after the last)
            k = it * 2 * KC + KC + c
            _add_issue(add_desc, abar, astage, k + 1, gl.where(c + 1 < KC, row0, nxt_row0), (c + 1) % KC,
              (c + 1 < KC) | has_next, NBYTES, BK)
            addv = _add_take(abar, astage, k, mma2)
        old = _moved(old, mma2, addv, Signs, step, move0, c, row0 >> 31, BM, BK, INJECT, ADD)
    out = warpgroup_mma(change, wsm.index((base + KC + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old.to(gl.float32) + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    if STORES:
        result = (value.to(gl.float32) * scale).to(gl.bfloat16)
        dsm.index(slot).store(result)
        fence_async_shared()
        gl.thread_barrier()
        tma.async_copy_shared_to_global(out_desc, [row0, c * BK], dsm.index(slot))
        if MODE == 3:
            stored = result.to(gl.float32)
            nx = gl.load(XL + token * D + c * BK + gl.arange(0, BK, gl.SliceLayout(0, mma2)))
            q0 += gl.sum(stored * nx[None, :], 1)
            q1 += gl.sum(stored * stored, 1)
        if c >= SL - E:  # chunk c - SL's store has left its slot: the next tile's chunk c - SL + E goes there
            tma.store_wait(SL)
            _reload(delta_desc, dbar, dsm, (it * KC + KC + c - SL + E) % (KC + E), c - SL + E, nxt_row0, has_next,
              NBYTES, BK)
    else:  # the readout moments, from the tile's 5 vectors staged in shared memory
        if c == 0:
            mbarrier.wait(vbar, it & 1)
        fence_async_shared()
        gl.thread_barrier()  # every warp has read delta chunk c: its slot takes the next tile's chunk
        _reload(delta_desc, dbar, dsm, (it * KC + c + KC) % (KC + E), c, nxt_row0, has_next, NBYTES, BK)
        v = value.to(gl.float32)
        vcols: gl.constexpr = gl.SliceLayout(0, mma2)
        s0 += gl.sum(v * vsm.index(0).slice(c * BK, BK).load(vcols)[None, :], 1)
        s1 += gl.sum(v * v, 1)
        s2 += gl.sum(v * vsm.index(1).slice(c * BK, BK).load(vcols)[None, :], 1)
        s3 += gl.sum(v * vsm.index(2).slice(c * BK, BK).load(vcols)[None, :], 1)
        s4 += gl.sum(v * vsm.index(3).slice(c * BK, BK).load(vcols)[None, :], 1)
        s5 += gl.sum(v * v * vsm.index(4).slice(c * BK, BK).load(vcols)[None, :], 1)
    return q0, q1, s0, s1, s2, s3, s4, s5


@gluon.jit
def _visit_kernel(delta_desc, out_desc, X, Fc, Wo, Routes, Clean, Scale, XL, Target, Mu, Nu, Diag, Stats,
    Steps, Signs, add_desc, w_desc, pv_desc, TILES,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, NS: gl.constexpr, SL: gl.constexpr,
    E: gl.constexpr, MODE: gl.constexpr, INJECT: gl.constexpr, ADD: gl.constexpr, RESP: gl.constexpr, UNROLL: gl.constexpr,
    EPSILON: gl.constexpr,
    num_warps: gl.constexpr):
    BK: gl.constexpr = 64
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    NBYTES: gl.constexpr = BM * BK * 2
    STORES: gl.constexpr = MODE != 2
    FRESH: gl.constexpr = INJECT or ADD  # the gate operand differs from delta: form it in registers
    gl.static_assert(NS >= 2 and NS <= KC + 1 and SL <= KC)
    G = gl.num_programs(0)
    pid = gl.program_id(0)

    # delta chunk slots: KC, or KC + E rotating (E spare slots: the next tile's chunk c loads while this tile's chunk c
    # is stored, E stores in flight)
    gl.static_assert(E <= SL)
    dsm = gl.allocate_shared_memory(gl.bfloat16, [KC + E, BM, BK], delta_desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NS, R, BK],
      gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [KC + E, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(KC + E):
        mbarrier.init(dbar.index(i), count=1)
    # MODE 2: the tile's readout vectors (xl, target, mu, nu, diag)
    vsm = gl.allocate_shared_memory(gl.float32, [5 if MODE == 2 else 1, D if MODE == 2 else 32],
      gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=num_warps * 32)
    if MODE == 2:
        _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, pid // MT, True, D, num_warps)
    # the add stream's 2-slot staging ring (a token slot without ADD): add chunks, or (RESP) the response's weights
    # and projected values
    AS: gl.constexpr = 2 if ADD else 1
    AR: gl.constexpr = 2 * BK if RESP else (BM if ADD else 8)
    astage = gl.allocate_shared_memory(gl.bfloat16, [AS, AR, BK], delta_desc.layout)
    abar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(abar.index(i), count=1)
    if RESP:
        gl.static_assert(BM == BK)
        _w_issue(w_desc, abar, astage, 0, (pid // MT) * P + (pid % MT) * BM, True, NBYTES, BK)
    elif ADD:
        _add_issue(add_desc, abar, astage, 0, (pid // MT) * P + (pid % MT) * BM, 0, True, NBYTES, BK)
    for i in gl.static_range(KC):
        _reload(delta_desc, dbar, dsm, i, i, (pid // MT) * P + (pid % MT) * BM, True, NBYTES, BK)

    # routed weight rows: R rows of BK contiguous bf16, 16 bytes per thread
    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen = gl.load(Routes + (pid // MT) * R + rr).to(gl.int32)
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), Fc + chosen * D, cols, i * BK)

    # wgmma layouts: one warpgroup per 64 rows; with 8 warps the second warpgroup takes the other half of N
    WN: gl.constexpr = num_warps // 4
    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN],
      instr_shape=[16, R // WN, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, WN],
      instr_shape=[16, BK // WN, 16])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma1, k_width=2)
    s_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    # with INJECT / ADD the moved stream is formed (and summed) in the output GEMM's accumulator layout, which
    # holds the same elements per thread as the gate's register A operand (global loads in the operand layout fault)
    n_layout: gl.constexpr = mma2 if FRESH else s_layout
    row_layout: gl.constexpr = gl.SliceLayout(1, s_layout)
    xcols = gl.arange(0, BK, gl.SliceLayout(0, n_layout))
    x_next = gl.load(X + (pid // MT) * D + xcols)
    if FRESH:  # x two chunks ahead (one ahead with INJECT / ADD, short of registers)
        x_after = x_next
    else:
        x_after = gl.load(X + (pid // MT) * D + BK + xcols)
    if STORES:
        scale = gl.load(Scale).to(gl.float32)
    else:
        scale = 1.0
    zero = gl.zeros([BM, BK], gl.float32, mma2)

    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        move0 = (tile % MT) * BM
        row0 = token * P + move0
        phase = it & 1
        base = (it * 2 * KC) % NS  # ring slot of this tile's first chunk (2 * KC chunks per tile)
        nxt_tile = tile + G
        has_next = nxt_tile < TILES
        nxt_token = gl.where(has_next, nxt_tile // MT, token)
        nxt_row0 = nxt_token * P + (nxt_tile % MT) * BM
        chosen_next = gl.load(Routes + nxt_token * R + rr).to(gl.int32)
        fc_rows = Fc + chosen * D
        wo_rows = Wo + chosen * D
        if INJECT:
            step = gl.load(Steps + token)
        else:
            step = 0.0

        # gate GEMM: acc = d @ Fc[routes]^T over chunk j of K; chunk j's norm sums run under its wgmma
        sq_x = gl.zeros([1], gl.float32, gl.SliceLayout(1, n_layout))
        dots = gl.zeros([BM], gl.float32, gl.SliceLayout(1, n_layout))
        moved_sq = gl.zeros([BM], gl.float32, gl.SliceLayout(1, n_layout))
        acc = gl.zeros([BM, R], gl.float32, mma1)
        h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        if not FRESH and not UNROLL:  # the smem-operand wgmma stays in flight across iterations
            acc = warpgroup_mma_init(acc)
        if RESP:  # the tile's response weights into registers (the wgmma A operand); the gate's first values in flight
            fence_async_shared()
            gl.thread_barrier()
            _pv_issue(pv_desc, abar, astage, it * (2 * KC + 1) + 1, token, 0, True, NBYTES, BK)
            w0, w1 = _w_take(abar, astage, it * (2 * KC + 1), gl.DotOperandLayout(operand_index=0, parent=mma2,
              k_width=2), BK)
        else:
            w0 = 0.0
            w1 = 0.0
        gate_args = (delta_desc, dbar, dsm, wsm, add_desc, pv_desc, w0, w1, zero, abar, astage, Signs, X, Clean, step,
          row0, move0, it, phase, base, fc_rows, wo_rows, cols, xcols, token, nxt_token)
        # unrolled chunk loops, or (UNROLL false) runtime ones: unrolled, the injected and added variants overflow the
        # instruction cache
        if UNROLL:
            for j in gl.static_range(KC):
                acc, sq_x, dots, moved_sq, x_next, x_after, h = _gate_chunk(j, acc, sq_x, dots, moved_sq, x_next,
                  x_after, h, *gate_args, D=D, R=R, BM=BM, BK=BK, KC=KC, NS=NS, SL=SL, E=E, NBYTES=NBYTES, STORES=STORES,
                  FRESH=FRESH, INJECT=INJECT, ADD=ADD, RESP=RESP, mma1=mma1, mma2=mma2, a_layout=a_layout,
                  s_layout=s_layout)
        else:
            for j in range(KC):
                acc, sq_x, dots, moved_sq, x_next, x_after, h = _gate_chunk(j, acc, sq_x, dots, moved_sq, x_next,
                  x_after, h, *gate_args, D=D, R=R, BM=BM, BK=BK, KC=KC, NS=NS, SL=SL, E=E, NBYTES=NBYTES, STORES=STORES,
                  FRESH=FRESH, INJECT=INJECT, ADD=ADD, RESP=RESP, mma1=mma1, mma2=mma2, a_layout=a_layout,
                  s_layout=s_layout)
        acc = warpgroup_mma_wait(0, deps=[acc])

        inv0 = gl.rsqrt(sq_x / D + EPSILON)
        inv_moved = gl.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
        inv = gl.convert_layout(inv_moved, gl.SliceLayout(1, mma1))
        ratio = gl.convert_layout(inv_moved / inv0 - 1, gl.SliceLayout(1, mma1))
        dh = inv[:, None] * acc.to(gl.bfloat16).to(gl.float32) + ratio[:, None] * h[None, :]
        up = gl.maximum(h[None, :] + dh, 0)
        down = gl.maximum(h[None, :], 0)
        change = ((up - down) * (up + down)).to(gl.bfloat16)
        change = gl.convert_layout(change, gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))

        # output GEMM: chunk c = change @ Wo[routes][:, chunk c], added to the moved stream's chunk c
        s0 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))  # MODE 2's moments
        s1 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        s2 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        s3 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        s4 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        s5 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        q0 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))  # MODE 3's sums
        q1 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        out_args = (delta_desc, out_desc, dbar, dsm, wsm, change, zero, add_desc, pv_desc, w_desc, w0, w1, abar, astage,
          vsm, vbar, Signs, XL, Target, Mu, Nu, Diag, Fc, step, scale, row0, move0, it, token, nxt_row0, has_next, base, wo_rows,
          chosen_next, cols)
        if UNROLL:
            for c in gl.static_range(KC):
                q0, q1, s0, s1, s2, s3, s4, s5 = _output_chunk(c, q0, q1, s0, s1, s2, s3, s4, s5, *out_args, D=D,
                  BM=BM, BK=BK, KC=KC, NS=NS, SL=SL, E=E, NBYTES=NBYTES, MODE=MODE, STORES=STORES, FRESH=FRESH,
                  INJECT=INJECT, ADD=ADD, RESP=RESP, mma2=mma2, s_layout=s_layout)
        else:
            for c in range(KC):
                q0, q1, s0, s1, s2, s3, s4, s5 = _output_chunk(c, q0, q1, s0, s1, s2, s3, s4, s5, *out_args, D=D,
                  BM=BM, BK=BK, KC=KC, NS=NS, SL=SL, E=E, NBYTES=NBYTES, MODE=MODE, STORES=STORES, FRESH=FRESH,
                  INJECT=INJECT, ADD=ADD, RESP=RESP, mma2=mma2, s_layout=s_layout)
        if MODE == 2:
            gl.thread_barrier()  # every warp is done with vsm: the next tile's vectors in flight
            _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, nxt_token, has_next, D, num_warps)
            stats_rows = Stats + (row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))).to(gl.int64) * 6
            gl.store(stats_rows, s0)
            gl.store(stats_rows + 1, s1)
            gl.store(stats_rows + 2, s2)
            gl.store(stats_rows + 3, s3)
            gl.store(stats_rows + 4, s4)
            gl.store(stats_rows + 5, s5)
        if MODE == 3:
            stats_rows = Stats + (row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))).to(gl.int64) * 2
            gl.store(stats_rows, q0)
            gl.store(stats_rows + 1, q1)
        chosen = chosen_next
        it += 1
    tma.store_wait(0)
    cp.wait_group(0)
    for i in gl.static_range(KC + E):
        mbarrier.invalidate(dbar.index(i))
    for i in gl.static_range(2):
        mbarrier.invalidate(abar.index(i))


def launch_visit(delta, x, fc, wo, routes, clean, scale, out, mode=1, vectors=None, stats=None, steps=None,
    signs=None, add=None, add_weights=None, add_values=None):
    """MODE 1 writes out (n, P, D) bf16; MODE 3 also stats (n, P, 2) fp32 against vectors = (next_x,); MODE 2 writes
    stats (n, P, 6) fp32 from vectors = (xl, target, mu, nu, diag). steps (n, 1) and signs (D,) (frame[0] of the
    signed Hadamard frame) inject step * frame on load; add (n, P, D) bf16 is added on load, or (RESP) the response
    bf16(add_weights @ add_values), add_weights (n, P, 2 * 64) and add_values (n, 2 * 64, D) bf16, formed in the
    kernel. Vectors, steps and signs fp32 contiguous."""
    n, p, d = delta.shape
    r = routes.shape[1]
    resp = add_weights is not None
    assert not (resp and add is not None)
    cfg = dict(CONFIGS[r])
    if cfg.get('maxnreg') is None:
        cfg.pop('maxnreg', None)
    if resp:  # its 2 x 32 KB staging ring leaves room for 2 weight slots and no spare delta slots at two CTAs per SM
        cfg.update(NS=2, SL=6, E=0)
    elif steps is not None and r == 64:
        cfg.update(SL=2)
    if mode == 2:  # the 10 KB readout-vector stage leaves room for one spare delta slot
        cfg.update(E=min(cfg['E'], 1))
    if add is not None and r == 64:  # the add staging ring leaves no room for spare delta slots
        cfg.update(E=0, SL=6)
    if add is not None and r == 128:  # with the add staging ring, 3 weight slots would leave one CTA per SM
        cfg['NS'] = 2
    bm = cfg.pop('BM')
    assert p % bm == 0 and d % 64 == 0 and delta.is_contiguous() and delta.dtype == torch.bfloat16
    if add is not None and steps is None and ADD_VIA_INJECT:
        # add without inject through the add + inject path with zero steps (d + 0.0 == d)
        steps = torch.zeros((n, 1), device=delta.device, dtype=torch.float32)
        signs = torch.ones(d, device=delta.device, dtype=torch.float32)
    assert steps is None or p == d, 'the in-kernel signed Hadamard frame needs P == D'
    layout = gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16)
    delta_desc = TensorDescriptor.from_tensor(delta.view(n * p, d), [bm, 64], layout)
    out_desc = TensorDescriptor.from_tensor(out.view(n * p, d), [bm, 64], layout) if mode != 2 else delta_desc
    add_desc = TensorDescriptor.from_tensor(add.view(n * p, d), [bm, 64], layout) if add is not None else delta_desc
    if resp:
        hk = add_weights.shape[-1]
        assert hk == 128 and bm == 64 and add_values.shape[1] == hk
        w_desc = TensorDescriptor.from_tensor(add_weights.reshape(n * p, hk), [bm, 64], layout)
        pv_desc = TensorDescriptor.from_tensor(add_values.reshape(n * hk, d), [hk, 64],
          gl.NVMMASharedLayout.get_default_for([hk, 64], gl.bfloat16))
    else:
        w_desc = pv_desc = delta_desc
    dev = delta.device.index
    if dev not in _SMS:
        _SMS[dev] = (132 if delta.device.type == 'cpu'
          else torch.cuda.get_device_properties(delta.device).multi_processor_count)
    tiles = n * (p // bm)
    vecs = tuple(vectors or ()) + (x,) * (5 - len(vectors or ()))
    _visit_kernel[(min(tiles, CTAS_PER_SM * _SMS[dev]),)](delta_desc, out_desc, x, fc, wo, routes, clean, scale, *vecs,
      x if stats is None else stats, x if steps is None else steps, x if signs is None else sign_bits(signs),
      add_desc, w_desc, pv_desc, tiles, p, d, r, bm, MODE=mode, INJECT=steps is not None,
      ADD=add is not None or resp, RESP=resp,
      UNROLL=mode == 2 or (steps is None and add is None and not resp and (mode == 1 or r == 64)),
      EPSILON=EPS, **cfg)
    return out if mode == 1 else stats if mode == 2 else (out, stats)


def _inputs(delta, x, fc, wo, routes, clean, steps=None, signs=None):
    f32 = lambda z: None if z is None else z.float().contiguous()
    return (delta.contiguous(), f32(x), fc.contiguous(), wo.contiguous(), routes.contiguous(), f32(clean),
      f32(steps), f32(signs))


@torch.library.custom_op('visit_kernel::fused_visit_fast', mutates_args=())
def fused_visit_fast(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, scale: torch.Tensor, steps: Optional[torch.Tensor] = None,
    signs: Optional[torch.Tensor] = None, add: Optional[torch.Tensor] = None,
    add_weights: Optional[torch.Tensor] = None, add_values: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fused_visit (MODE 1): bf16(bf16(delta (+ add) (+ steps * frame) + routed MLP change) * scale); signs =
    frame[0] of the signed Hadamard frame; add_weights (n, P, 128) / add_values (n, 128, D) in place of add =
    bmm(add_weights, add_values) (the attention replay's response, formed in the kernel)."""
    delta, x, fc, wo, routes, clean, steps, signs = _inputs(delta, x, fc, wo, routes, clean, steps, signs)
    contig = lambda z: None if z is None else z.contiguous()
    return launch_visit(delta, x, fc, wo, routes, clean, scale, torch.empty_like(delta), steps=steps, signs=signs,
      add=contig(add), add_weights=contig(add_weights), add_values=contig(add_values))


@fused_visit_fast.register_fake
def _(delta, x, fc, wo, routes, clean, scale, steps=None, signs=None, add=None, add_weights=None, add_values=None):
    return torch.empty_like(delta)


@torch.library.custom_op('visit_kernel::fused_query_visit_fast', mutates_args=())
def fused_query_visit_fast(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, scale: torch.Tensor,
    next_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fused_query_visit (MODE 3): the scaled output and, per move, its dot with next_x and its square, viewed as
    (token, move, statistic, 1) partials."""
    delta, x, fc, wo, routes, clean, _, _ = _inputs(delta, x, fc, wo, routes, clean)
    stats = torch.empty((*delta.shape[:2], 2), device=delta.device, dtype=torch.float32)
    out, stats = launch_visit(delta, x, fc, wo, routes, clean, scale, torch.empty_like(delta), mode=3,
      vectors=(next_x.float().contiguous(),), stats=stats)
    return out, stats.unsqueeze(-1)


@fused_query_visit_fast.register_fake
def _(delta, x, fc, wo, routes, clean, scale, next_x):
    return torch.empty_like(delta), torch.empty((*delta.shape[:2], 2, 1), device=delta.device, dtype=torch.float32)


@torch.library.custom_op('visit_kernel::fused_readout_visit_fast', mutates_args=())
def fused_readout_visit_fast(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, diag: torch.Tensor, steps: Optional[torch.Tensor] = None,
    signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fused_readout_visit (MODE 2): the readout-moment sums of the moved final stream, (n, P, 6)."""
    delta, x, fc, wo, routes, clean, steps, signs = _inputs(delta, x, fc, wo, routes, clean, steps, signs)
    stats = torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
    return launch_visit(delta, x, fc, wo, routes, clean, x, None, mode=2,
      vectors=tuple(z.float().contiguous() for z in (xl, target, mu, nu, diag)), stats=stats, steps=steps,
      signs=signs)


@fused_readout_visit_fast.register_fake
def _(delta, x, fc, wo, routes, clean, xl, target, mu, nu, diag, steps=None, signs=None):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)
