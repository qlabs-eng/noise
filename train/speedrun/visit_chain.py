"""Gluon (Hopper) chain kernels: block A (a plain MODE 1 visit, or the root visit) and the MODE 2 readout visit
(block B, the last block) back to back per tile, the tile's delta kept in shared memory between them (no HBM round
trip of block A's output).

fused_visit_readout_fast is numerically visit.fused_visit_fast followed by visit.fused_readout_visit_fast,
fused_root_readout_fast visit_root.fused_root_visit_fast followed by it (the same per-element ops in the same order):
block A's bf16(bf16(d + change) * scale) is written into the tile's delta chunks in shared memory and read there as
block B's delta. Persistent CTAs (two per SM) walk tiles of BM = 64 moves of one token in token-major order; the
routed rows of both blocks stream through one cp.async ring (per tile: Fc A (not for the root), Wo A, Fc B, Wo B).
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

EPS = torch.finfo(torch.float32).eps
CTAS_PER_SM = 2
_SMS = {}

# runtime chunk loops: unrolled (as visit.py's plain visits) the kernel is 1.4x slower and computes wrong, unrepeatable
# block B sums with E = 1 or NS = 2 (illegal instruction at n = 17; unresolved, bitwise right only at E = 0, NS = 3)
# DEFER 1: the gates' norm sums as per-thread partials, one reduction per tile (-3-4%; fp32 order differs from the
# two-kernel path: stats as close to an fp64 readout as Triton's; bitwise equal to it only with DEFER 0)
CONFIG = dict(BM=64, NS=3, E=0, UNROLL=False, DEFER=1, num_warps=4)
# the root's change goes to the wgmma operand layout through an 8 KB scratch buffer: no spare delta slot for 2 CTAs/SM
ROOT_CONFIG = dict(BM=64, NS=3, E=0, UNROLL=False, DEFER=1, num_warps=4, TC=0, PK=0)
# PK: packed fp32 -> bf16 conversions (cvt.rn.bf16x2.f32) where ptxas emits scalar F2F; bitwise
# TC (root chain, chain3): dot products on the tensor cores: the gates of blocks fed from shared memory take d . x as a
# wgmma against x in 3 bf16 parts (8-row B tiles; x's sum of squares per token from the host), the readout its four
# linear moments against xl / target / mu / nu in 2 bf16 parts (|residual| <= 2^-17); squares stay on the FMA pipes.
# Not bitwise (fp32 order, the 2-part split).


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
def _gate(j, acc, sq_x, dots, moved_sq, x_next, x_after, h, dbar, dsm, wsm, X, Clean, it, base, fc_rows, wo_rows,
    cols, xcols, token, nxt_token,
    D: gl.constexpr, R: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr,
    LOADED: gl.constexpr, mma1: gl.constexpr, s_layout: gl.constexpr, BM: gl.constexpr = 64, DEFER: gl.constexpr = False):
    """Gate GEMM chunk j of one block: acc += d @ Fc[routes]^T over columns j; chunk j's norm sums run under its
    wgmma. LOADED: chunk j comes in by TMA (block A); otherwise the previous block wrote it (block B)."""
    g = it * KC + j
    slot = g % (KC + E)
    if LOADED:
        mbarrier.wait(dbar.index(slot), (g // (KC + E)) & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    acc = warpgroup_mma_wait(0, deps=[acc])
    gl.thread_barrier()
    _issue_rows(wsm.index((base + j + NS - 1) % NS), gl.where(j + NS - 1 < KC, fc_rows, wo_rows), cols,
      ((j + NS - 1) % KC) * BK)
    acc = warpgroup_mma(dsm.index(slot), wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    d = dsm.index(slot).load(s_layout).to(gl.float32)
    x = x_next
    x_next = x_after
    x_after = gl.load(X + gl.where(j + 2 < KC, token, nxt_token) * D + ((j + 2) % KC) * BK + xcols)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    if DEFER:  # per-thread partials (x * x per column, d sums per row and 8-column group), reduced once per tile
        sq_x += x * x
        dots += gl.sum(gl.reshape(d * x[None, :], [BM, BK // 8, 8]), 2)
        moved_sq += gl.sum(gl.reshape(d * d, [BM, BK // 8, 8]), 2)
    else:
        sq_x += gl.sum(gl.expand_dims(x * x, 0), 1)
        dots += gl.sum(d * x[None, :], 1)
        moved_sq += gl.sum(d * d, 1)
    return acc, sq_x, dots, moved_sq, x_next, x_after, h


@gluon.jit
def _gate_init(BM: gl.constexpr, BK: gl.constexpr, DEFER: gl.constexpr, s_layout: gl.constexpr):
    """Zero norm sums (sq_x, dots, moved_sq), or (DEFER) their per-thread partials."""
    if DEFER:
        part = gl.sum(gl.reshape(gl.zeros([BM, BK], gl.float32, s_layout), [BM, BK // 8, 8]), 2)
        return gl.zeros([BK], gl.float32, gl.SliceLayout(0, s_layout)), part, part
    else:
        rows: gl.constexpr = gl.SliceLayout(1, s_layout)
        return gl.zeros([1], gl.float32, rows), gl.zeros([BM], gl.float32, rows), gl.zeros([BM], gl.float32, rows)


@gluon.jit
def _gate_done(sq_x, dots, moved_sq, DEFER: gl.constexpr, mma1: gl.constexpr):
    """The norm sums per row, in the gate accumulator's row layout."""
    rows: gl.constexpr = gl.SliceLayout(1, mma1)
    if DEFER:
        sq_x = gl.sum(gl.expand_dims(sq_x, 0), 1)
        dots = gl.sum(dots, 1)
        moved_sq = gl.sum(moved_sq, 1)
    return gl.convert_layout(sq_x, rows), gl.convert_layout(dots, rows), gl.convert_layout(moved_sq, rows)


@gluon.jit
def _bf16(x):
    """x.to(bfloat16) as packed cvt.rn.bf16x2.f32 (bitwise the same; ptxas emits a scalar F2F per element, on the
    quarter-rate XU pipe, where the result goes straight back to fp32)."""
    return gl.inline_asm_elementwise("cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,r,r", [x], dtype=gl.bfloat16, is_pure=True,
      pack=2)


@gluon.jit
def _rbf16(x):
    """x.to(bfloat16).to(float32) (bitwise): one packed cvt.rn.bf16x2.f32 and two bit ops per pair."""
    return gl.inline_asm_elementwise(
      "{ .reg .b32 t; cvt.rn.bf16x2.f32 t, $3, $2; shl.b32 $0, t, 16; and.b32 $1, t, 0xffff0000; }",
      "=r,=r,r,r", [x], dtype=gl.float32, is_pure=True, pack=2)


@gluon.jit
def _change(acc, sq_x, dots, moved_sq, h, D: gl.constexpr, EPSILON: gl.constexpr, mma1: gl.constexpr,
    mma2: gl.constexpr, PK: gl.constexpr = False):
    """The routed MLP's change relu(h + dh)^2 - relu(h)^2 as the output GEMM's register A operand."""
    inv0 = gl.rsqrt(sq_x / D + EPSILON)
    inv_moved = gl.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
    inv = gl.convert_layout(inv_moved, gl.SliceLayout(1, mma1))
    ratio = gl.convert_layout(inv_moved / inv0 - 1, gl.SliceLayout(1, mma1))
    if PK:
        dh = inv[:, None] * _rbf16(acc) + ratio[:, None] * h[None, :]
    else:
        dh = inv[:, None] * acc.to(gl.bfloat16).to(gl.float32) + ratio[:, None] * h[None, :]
    up = gl.maximum(h[None, :] + dh, 0)
    down = gl.maximum(h[None, :], 0)
    if PK:
        change = _bf16((up - down) * (up + down))
    else:
        change = ((up - down) * (up + down)).to(gl.bfloat16)
    return gl.convert_layout(change, gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))


@gluon.jit
def _output_keep(c, dbar, dsm, wsm, change, zero, step, scale, it, base, wo_rows, next_rows, cols,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, ROOT: gl.constexpr, mma2: gl.constexpr,
    PK: gl.constexpr = False):
    """Block A's output chunk c: bf16(bf16(d + change @ Wo[routes][:, c]) * scale) back into delta chunk c (base:
    the ring slot of the output's first item). ROOT: d = step * the frame chunk TMA-loaded there (exact: +-1)."""
    g = it * KC + c
    slot = g % (KC + E)
    if ROOT:
        mbarrier.wait(dbar.index(slot), (g // (KC + E)) & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    if ROOT:
        old = step * dsm.index(slot).load(mma2).to(gl.float32)
    else:
        old = dsm.index(slot).load(mma2).to(gl.float32)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    if PK and ROOT:  # packed conversions (old is fp32 here: no bf16 add)
        value = _rbf16(old + _rbf16(out))
        dsm.index(slot).store(_bf16(value * scale))
    else:
        value = (old + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
        dsm.index(slot).store((value.to(gl.float32) * scale).to(gl.bfloat16))  # each thread its own elements


@gluon.jit
def _output_readout(c, s0, s1, s2, s3, s4, s5, delta_desc, dbar, dsm, wsm, vsm, vbar, change, zero, it, base,
    wo_rows, next_rows, cols, nxt_row0, has_next,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, NBYTES: gl.constexpr,
    mma2: gl.constexpr, BM: gl.constexpr = 64, DEFER: gl.constexpr = False):
    """Block B's output chunk c, reduced to the readout moments; its slot then takes the next tile's chunk c (base:
    the ring slot of the output's first item)."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    slot = (it * KC + c) % (KC + E)
    old = dsm.index(slot).load(mma2)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old.to(gl.float32) + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    if c == 0:
        mbarrier.wait(vbar, it & 1)
    fence_async_shared()
    gl.thread_barrier()  # every warp has read chunk c: its slot takes the next tile's chunk
    _reload(delta_desc, dbar, dsm, (it * KC + c + KC) % (KC + E), c, nxt_row0, has_next, NBYTES, BK)
    v = value.to(gl.float32)
    vcols: gl.constexpr = gl.SliceLayout(0, mma2)
    s0 += _row_sum(v * vsm.index(0).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s1 += _row_sum(v * v, BM, BK, DEFER)
    s2 += _row_sum(v * vsm.index(1).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s3 += _row_sum(v * vsm.index(2).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s4 += _row_sum(v * vsm.index(3).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s5 += _row_sum(v * v * vsm.index(4).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    return s0, s1, s2, s3, s4, s5


@gluon.jit
def _row_done(part, mma2: gl.constexpr):
    return gl.convert_layout(gl.sum(part, 1), gl.SliceLayout(1, mma2))


@gluon.jit
def _row_sum(t, BM: gl.constexpr, BK: gl.constexpr, DEFER: gl.constexpr):
    """Row sums of an accumulator-layout chunk, or (DEFER) per-thread partials: column 8 g + k of a thread's rows sits
    in its registers for all 8 g, so summing over g needs no shuffles (reduced once per tile)."""
    if DEFER:
        return gl.sum(gl.reshape(t, [BM, 8, BK // 8]), 1)
    else:
        return gl.sum(t, 1)


@triton.jit
def _parts_kernel(S0, S1, S2, S3, Dst, Sq, D: tl.constexpr, NSRC: tl.constexpr, SPLIT: tl.constexpr,
    SQ: tl.constexpr):
    """Dst (n, 8, D) bf16: the token's NSRC fp32 vectors, each as SPLIT bf16 parts (row s * SPLIT + p; the parts sum
    to the vector to the split's precision), other rows zero; SQ: Sq (n,) the first vector's sum of squares."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, D)
    out = Dst + t * 8 * D + cols
    for s in tl.static_range(NSRC):
        if s == 0:
            x = tl.load(S0 + t * D + cols)
        elif s == 1:
            x = tl.load(S1 + t * D + cols)
        elif s == 2:
            x = tl.load(S2 + t * D + cols)
        else:
            x = tl.load(S3 + t * D + cols)
        if SQ and s == 0:
            tl.store(Sq + t, tl.sum(x * x, 0))
        for q in tl.static_range(SPLIT):
            part = x.to(tl.bfloat16)
            tl.store(out + (s * SPLIT + q) * D, part)
            x = x - part.to(tl.float32)
    for k in tl.static_range(NSRC * SPLIT, 8):
        tl.store(out + k * D, tl.zeros([D], tl.bfloat16))


def split_parts(vectors, split, square=False):
    """(n, 8, D) bf16 parts of up to 4 (n, D) fp32 vectors (_parts_kernel) and (square) the first's sums of squares."""
    n, d = vectors[0].shape
    dst = torch.empty((n, 8, d), device=vectors[0].device, dtype=torch.bfloat16)
    sq = torch.empty((n,), device=vectors[0].device, dtype=torch.float32) if square else dst
    srcs = list(vectors) + [vectors[0]] * (4 - len(vectors))
    _parts_kernel[(n,)](*srcs, dst, sq, d, len(vectors), split, square, num_warps=4)
    return dst, (sq if square else None)


@gluon.jit
def _rows_issue(buf, Src, token, pred, D: gl.constexpr, BK: gl.constexpr, num_warps: gl.constexpr):
    """cp.async of the token's (8, D) bf16 parts into buf [D // BK * 8, BK] (chunk c's 8 rows at 8 c), no commit."""
    lay: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rows = gl.arange(0, D // BK * 8, gl.SliceLayout(1, lay))
    cols = gl.arange(0, BK, gl.SliceLayout(0, lay))
    off = token.to(gl.int64) * 8 * D + ((rows % 8) * D + (rows // 8) * BK)[:, None] + cols[None, :]
    cp.async_copy_global_to_shared(buf, Src + off, mask=(cols < BK)[None, :] & pred)


@gluon.jit
def _diag_issue(dg, Diag, token, pred, D: gl.constexpr, num_warps: gl.constexpr):
    """cp.async of the token's readout diag (fp32) into dg, no commit."""
    lay: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
    dd = gl.arange(0, D, lay)
    cp.async_copy_global_to_shared(dg, Diag + token * D + dd, mask=(dd < D) & pred)


@gluon.jit
def _gate_tc(j, acc, accx, moved_sq, h, dsm, wsm, x3, Clean, it, base, fc_rows, wo_rows, cols, token,
    R: gl.constexpr, XOFF: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, mma1: gl.constexpr,
    s_layout: gl.constexpr, BM: gl.constexpr):
    """TC gate chunk j of a block fed from shared memory: acc += d @ Fc[routes]^T and accx += d @ x-parts^T (both
    wgmmas from the delta chunk in shared memory); per-thread partial sums of d^2 under them."""
    slot = (it * KC + j) % (KC + E)
    cp.wait_group(NS - 2)
    fence_async_shared()
    acc, accx = warpgroup_mma_wait(0, deps=[acc, accx])
    gl.thread_barrier()
    _issue_rows(wsm.index((base + j + NS - 1) % NS), gl.where(j + NS - 1 < KC, fc_rows, wo_rows), cols,
      ((j + NS - 1) % KC) * BK)
    acc = warpgroup_mma(dsm.index(slot), wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    accx = warpgroup_mma(dsm.index(slot), x3.index(XOFF + j).permute((1, 0)), accx, is_async=True)
    d = dsm.index(slot).load(s_layout).to(gl.float32)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    moved_sq += gl.sum(gl.reshape(d * d, [BM, BK // 8, 8]), 2)
    return acc, accx, moved_sq, h


@gluon.jit
def _gate_tc_init(BM: gl.constexpr, BK: gl.constexpr, R: gl.constexpr, s_layout: gl.constexpr, mma1: gl.constexpr,
    mmax: gl.constexpr):
    """Zero gate accumulators (routes, x parts), per-thread partial sums of d^2 and h."""
    acc = warpgroup_mma_init(gl.zeros([BM, R], gl.float32, mma1))
    accx = warpgroup_mma_init(gl.zeros([BM, 8], gl.float32, mmax))
    part = gl.sum(gl.reshape(gl.zeros([BM, BK], gl.float32, s_layout), [BM, BK // 8, 8]), 2)
    return acc, accx, part, gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))


@gluon.jit
def _gate_tc_done(acc, accx, moved_sq, sq, mma1: gl.constexpr, mmax: gl.constexpr):
    """The gate (acc, sq_x, dots, moved_sq), sums per row in the accumulator's row layout; dots: x parts 0-2."""
    acc, accx = warpgroup_mma_wait(0, deps=[acc, accx])
    rows: gl.constexpr = gl.SliceLayout(1, mma1)
    cx = gl.arange(0, 8, gl.SliceLayout(0, mmax))
    dots = gl.convert_layout(gl.sum(gl.where((cx < 3)[None, :], accx, 0.0), 1), rows)
    moved_sq = gl.convert_layout(gl.sum(moved_sq, 1), rows)
    return acc, dots * 0.0 + sq, dots, moved_sq


@gluon.jit
def _output_readout_tc(c, accr, s1, s5, delta_desc, dbar, dsm, wsm, v3, dg, change, zero, it, base,
    wo_rows, next_rows, cols, nxt_row0, has_next,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, NBYTES: gl.constexpr,
    mma2: gl.constexpr, mmax: gl.constexpr, BM: gl.constexpr = 64, DEFER: gl.constexpr = False, VOFF: gl.constexpr = 0,
    DOFF: gl.constexpr = 0):
    """TC readout chunk c (runtime c: dg is [*, BK] fp32, chunk c's diag in row DOFF + c): as _output_readout, but the
    four linear moments as one wgmma of the output chunk (A from registers, waited at once) against the vector parts
    (accr columns 2k, 2k + 1: xl, target, mu, nu); v^2 and v^2 diag on the FMA pipes."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    slot = (it * KC + c) % (KC + E)
    old = dsm.index(slot).load(mma2)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old.to(gl.float32) + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    fence_async_shared()
    gl.thread_barrier()  # every warp has read chunk c: its slot takes the next tile's chunk
    _reload(delta_desc, dbar, dsm, (it * KC + c + KC) % (KC + E), c, nxt_row0, has_next, NBYTES, BK)
    vop = gl.convert_layout(value, gl.DotOperandLayout(operand_index=0, parent=mmax, k_width=2))
    accr = warpgroup_mma(vop, v3.index(VOFF + c).permute((1, 0)), accr, is_async=True)
    accr, vop = warpgroup_mma_wait(0, deps=[accr, vop])
    v = value.to(gl.float32)
    vcols: gl.constexpr = gl.SliceLayout(0, mma2)
    s1 += _row_sum(v * v, BM, BK, DEFER)
    s5 += _row_sum(v * v * dg.index(DOFF + c).load(vcols)[None, :], BM, BK, DEFER)
    return accr, s1, s5


@gluon.jit
def _readout_tc_done(accr, s1, s5, mma2: gl.constexpr, mmax: gl.constexpr, DEFER: gl.constexpr):
    """(s0, ..., s5) per row in mma2's row layout from the readout accumulator and the squares' sums."""
    rows: gl.constexpr = gl.SliceLayout(1, mma2)
    pr = gl.arange(0, 8, gl.SliceLayout(0, mmax)) // 2
    s0 = gl.convert_layout(gl.sum(gl.where((pr == 0)[None, :], accr, 0.0), 1), rows)
    s2 = gl.convert_layout(gl.sum(gl.where((pr == 1)[None, :], accr, 0.0), 1), rows)
    s3 = gl.convert_layout(gl.sum(gl.where((pr == 2)[None, :], accr, 0.0), 1), rows)
    s4 = gl.convert_layout(gl.sum(gl.where((pr == 3)[None, :], accr, 0.0), 1), rows)
    if DEFER:
        s1 = _row_done(s1, mma2)
        s5 = _row_done(s5, mma2)
    return s0, s1, s2, s3, s4, s5


@gluon.jit
def _stats_store(Stats, row0, s0, s1, s2, s3, s4, s5, BM: gl.constexpr, mma2: gl.constexpr):
    stats_rows = Stats + (row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))).to(gl.int64) * 6
    gl.store(stats_rows, s0)
    gl.store(stats_rows + 1, s1)
    gl.store(stats_rows + 2, s2)
    gl.store(stats_rows + 3, s3)
    gl.store(stats_rows + 4, s4)
    gl.store(stats_rows + 5, s5)


@gluon.jit
def _chain_kernel(delta_desc, XA, FcA, WoA, RoutesA, CleanA, ScaleA, XB, FcB, WoB, RoutesB, CleanB,
    XL, Target, Mu, Nu, Diag, Stats, Steps, Proj, Inv0, Inv, XPB, SQB, VP, TILES,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, NS: gl.constexpr, E: gl.constexpr,
    ROOT: gl.constexpr, UNROLL: gl.constexpr, EPSILON: gl.constexpr, num_warps: gl.constexpr,
    DEFER: gl.constexpr = 0, TC: gl.constexpr = 0, PK: gl.constexpr = 0):
    """Block A (a plain visit, or ROOT: the root visit, delta_desc then the bf16 frame (P, D)) and block B (the
    readout visit) per tile. DEFER: 1 the gates' norm sums, 2 the readout sums, 3 both kept as per-thread partials
    across chunks (one reduction per tile; a different fp32 order). TC: block B's gate dots and the readout's linear
    moments on the tensor cores (XPB / SQB: x_b's parts and sums of squares, VP: the readout vectors' parts)."""
    BK: gl.constexpr = 64
    DG: gl.constexpr = DEFER % 2 == 1
    DR: gl.constexpr = DEFER >= 2
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    NBYTES: gl.constexpr = BM * BK * 2
    GA: gl.constexpr = 0 if ROOT else KC  # ring items per tile: block A's gate (none for the root), then 3 x KC
    IT: gl.constexpr = GA + 3 * KC
    gl.static_assert(NS >= 2 and NS <= KC + 1 and E <= 1)
    G = gl.num_programs(0)
    pid = gl.program_id(0)

    dsm = gl.allocate_shared_memory(gl.bfloat16, [KC + E, BM, BK], delta_desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NS, R, BK],
      gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [KC + E, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(KC + E):
        mbarrier.init(dbar.index(i), count=1)
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=num_warps * 32)
    if TC:  # x_b's parts, the readout vectors' parts ([KC * 8, BK] bf16 each; chunk c's 8 rows at 8 c) and diag
        p_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        xbuf = gl.allocate_shared_memory(gl.bfloat16, [KC * 8, BK], p_layout)
        vbuf = gl.allocate_shared_memory(gl.bfloat16, [KC * 8, BK], p_layout)
        x3 = xbuf._reinterpret(gl.bfloat16, [KC, 8, BK], p_layout)
        v3 = vbuf._reinterpret(gl.bfloat16, [KC, 8, BK], p_layout)
        dg = gl.allocate_shared_memory(gl.float32, [D], gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
          order=[0]))
        dg2 = dg._reinterpret(gl.float32, [KC, BK], gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
          order=[0]))
        _rows_issue(xbuf, XPB, pid // MT, True, D, BK, num_warps)
        _rows_issue(vbuf, VP, pid // MT, True, D, BK, num_warps)
        _diag_issue(dg, Diag, pid // MT, True, D, num_warps)
        cp.mbarrier_arrive(vbar, increment_count=False)
    else:
        vsm = gl.allocate_shared_memory(gl.float32, [5, D],
          gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
        _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, pid // MT, True, D, num_warps)
    for i in gl.static_range(KC):  # the first tile's delta (ROOT: frame) chunks
        _reload(delta_desc, dbar, dsm, i, i, (0 if ROOT else (pid // MT) * P) + (pid % MT) * BM, True, NBYTES, BK)

    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen_a = gl.load(RoutesA + (pid // MT) * R + rr).to(gl.int32)
    first_a = WoA if ROOT else FcA  # block A's first ring items
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), first_a + chosen_a * D, cols, i * BK)

    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, R, 16])  # num_warps 8: BM = 128, two warpgroups sharing every routed-row gather
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BK, 16])
    mmax: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, 8, 16])
    s_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, s_layout)
    q_layout: gl.constexpr = gl.BlockedLayout([4, 1], [16, 2], [1, num_warps], [0, 1])  # ROOT: Proj move-contiguous
    qrows = gl.arange(0, BM, gl.SliceLayout(1, q_layout))
    qcols = gl.arange(0, R, gl.SliceLayout(0, q_layout))
    xcols = gl.arange(0, BK, gl.SliceLayout(0, s_layout))
    if ROOT:
        xa_next = 0.0
        xa_after = 0.0
    else:
        xa_next = gl.load(XA + (pid // MT) * D + xcols)
        xa_after = gl.load(XA + (pid // MT) * D + BK + xcols)
    if TC:
        xb_next = 0.0
        xb_after = 0.0
    else:
        xb_next = gl.load(XB + (pid // MT) * D + xcols)
        xb_after = gl.load(XB + (pid // MT) * D + BK + xcols)
    scale = gl.load(ScaleA).to(gl.float32)
    zero = gl.zeros([BM, BK], gl.float32, mma2)

    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        move0 = (tile % MT) * BM
        row0 = token * P + move0
        base = it * IT  # ring position of the tile's first item
        nxt_tile = tile + G
        has_next = nxt_tile < TILES
        nxt_token = gl.where(has_next, nxt_tile // MT, token)
        nxt_move0 = (nxt_tile % MT) * BM
        chosen_b = gl.load(RoutesB + token * R + rr).to(gl.int32)
        chosen_next = gl.load(RoutesA + nxt_token * R + rr).to(gl.int32)

        if ROOT:  # block A's gate change from the projection: moved = bf16(q * bf16(step)), relu^2 at the clean h
            step = gl.load(Steps + token)
            q = gl.load(Proj + gl.load(RoutesA + token * R + qcols).to(gl.int32)[None, :] * P + (move0 + qrows)[:, None])
            if PK:
                moved = _rbf16(q * step.to(gl.bfloat16).to(gl.float32))
            else:
                moved = (q * step.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
            inv = gl.load(Inv + row0 + qrows)
            inv0 = gl.load(Inv0 + token)
            h = gl.load(CleanA + token * R + qcols).to(gl.float32)
            dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
            up = gl.maximum(h[None, :] + dh, 0)
            down = gl.maximum(h[None, :], 0)
            change = gl.convert_layout(_bf16((up - down) * (up + down)) if PK else ((up - down) * (up + down)).to(gl.bfloat16),
              gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))
        else:  # block A: gate GEMM over the TMA-loaded tile
            step = 0.0
            sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
            acc = gl.zeros([BM, R], gl.float32, mma1)
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
            if not UNROLL:
                acc = warpgroup_mma_init(acc)
            gate_a = (dbar, dsm, wsm, XA, CleanA, it, base % NS, FcA + chosen_a * D, WoA + chosen_a * D, cols, xcols,
              token, nxt_token)
            if UNROLL:
                for j in gl.static_range(KC):
                    acc, sq_x, dots, moved_sq, xa_next, xa_after, h = _gate(j, acc, sq_x, dots, moved_sq, xa_next,
                      xa_after, h, *gate_a, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=True, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            else:
                for j in range(KC):
                    acc, sq_x, dots, moved_sq, xa_next, xa_after, h = _gate(j, acc, sq_x, dots, moved_sq, xa_next,
                      xa_after, h, *gate_a, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=True, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            acc = warpgroup_mma_wait(0, deps=[acc])
            sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
            change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, PK)
        # block A's output, written back into the tile; the ring's tail fetches block B's first c_fc chunks
        out_a = (dbar, dsm, wsm, change, zero, step, scale, it, (base + GA) % NS, WoA + chosen_a * D,
          FcB + chosen_b * D, cols)
        if UNROLL:
            for c in gl.static_range(KC):
                _output_keep(c, *out_a, BK=BK, KC=KC, NS=NS, E=E, ROOT=ROOT, mma2=mma2, PK=PK)
        else:
            for c in range(KC):
                _output_keep(c, *out_a, BK=BK, KC=KC, NS=NS, E=E, ROOT=ROOT, mma2=mma2, PK=PK)

        if TC:  # block B with the dot products on the tensor cores
            mbarrier.wait(vbar, it & 1)  # x_b's parts and the readout vectors
            acc, accx, moved_sq, h = _gate_tc_init(BM, BK, R, s_layout, mma1, mmax)
            for j in range(KC):
                acc, accx, moved_sq, h = _gate_tc(j, acc, accx, moved_sq, h, dsm, wsm, x3, CleanB, it,
                  (base + GA + KC) % NS, FcB + chosen_b * D, WoB + chosen_b * D, cols, token, R=R, XOFF=0, BK=BK,
                  KC=KC, NS=NS, E=E, mma1=mma1, s_layout=s_layout, BM=BM)
            acc, sq_x, dots, moved_sq = _gate_tc_done(acc, accx, moved_sq, gl.load(SQB + token), mma1, mmax)
            change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, PK)
            accr = gl.zeros([BM, 8], gl.float32, mmax)
            s1 = _row_sum(zero, BM, BK, DR)
            s5 = s1
            for c in range(KC + (TILES >> 31)):  # opaque bound: rolled (unrolled, the register-A wgmmas take 70 more
                # registers)
                accr, s1, s5 = _output_readout_tc(c, accr, s1, s5, delta_desc, dbar, dsm, wsm, v3, dg2, change, zero, it,
                  (base + GA + 2 * KC) % NS, WoB + chosen_b * D, first_a + chosen_next * D, cols,
                  (0 if ROOT else nxt_token * P) + nxt_move0, has_next, BK=BK, KC=KC, NS=NS, E=E, NBYTES=NBYTES,
                  mma2=mma2, mmax=mmax, BM=BM, DEFER=DR)
            s0, s1, s2, s3, s4, s5 = _readout_tc_done(accr, s1, s5, mma2, mmax, DR)
            gl.thread_barrier()  # every warp is done with the parts: the next tile's in flight
            _rows_issue(xbuf, XPB, nxt_token, has_next, D, BK, num_warps)
            _rows_issue(vbuf, VP, nxt_token, has_next, D, BK, num_warps)
            _diag_issue(dg, Diag, nxt_token, has_next, D, num_warps)
            cp.mbarrier_arrive(vbar, increment_count=False)
            _stats_store(Stats, row0, s0, s1, s2, s3, s4, s5, BM, mma2)
        else:
            # block B: gate GEMM over block A's output in shared memory, output reduced to the readout moments
            sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
            acc = gl.zeros([BM, R], gl.float32, mma1)
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
            if not UNROLL:
                acc = warpgroup_mma_init(acc)
            gate_b = (dbar, dsm, wsm, XB, CleanB, it, (base + GA + KC) % NS, FcB + chosen_b * D, WoB + chosen_b * D, cols,
              xcols, token, nxt_token)
            if UNROLL:
                for j in gl.static_range(KC):
                    acc, sq_x, dots, moved_sq, xb_next, xb_after, h = _gate(j, acc, sq_x, dots, moved_sq, xb_next,
                      xb_after, h, *gate_b, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=False, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            else:
                for j in range(KC):
                    acc, sq_x, dots, moved_sq, xb_next, xb_after, h = _gate(j, acc, sq_x, dots, moved_sq, xb_next,
                      xb_after, h, *gate_b, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=False, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            acc = warpgroup_mma_wait(0, deps=[acc])
            sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
            change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2, PK)
            s0 = _row_sum(zero, BM, BK, DR)
            s1 = s0
            s2 = s0
            s3 = s0
            s4 = s0
            s5 = s0
            # the ring's tail fetches the next tile's first block A items; the consumed chunks reload with its delta
            out_b = (delta_desc, dbar, dsm, wsm, vsm, vbar, change, zero, it, (base + GA + 2 * KC) % NS,
              WoB + chosen_b * D, first_a + chosen_next * D, cols, (0 if ROOT else nxt_token * P) + nxt_move0, has_next)
            for c in gl.static_range(KC):  # unrolled: the readout vectors are sliced at constant offsets
                s0, s1, s2, s3, s4, s5 = _output_readout(c, s0, s1, s2, s3, s4, s5, *out_b, BK=BK, KC=KC, NS=NS, E=E,
                  NBYTES=NBYTES, mma2=mma2, BM=BM, DEFER=DR)
            if DR:
                s0, s1, s2, s3, s4, s5 = (_row_done(s0, mma2), _row_done(s1, mma2), _row_done(s2, mma2),
                  _row_done(s3, mma2), _row_done(s4, mma2), _row_done(s5, mma2))
            gl.thread_barrier()  # every warp is done with vsm: the next tile's vectors in flight
            _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, nxt_token, has_next, D, num_warps)
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
    for i in gl.static_range(KC + E):
        mbarrier.invalidate(dbar.index(i))
    mbarrier.invalidate(vbar)


def launch_chain(n, p, d, block_a, block_b, vectors, stats, root=None, **overrides):
    """block_a: (delta, x, fc, wo, routes, clean, scale), or with root = (steps, projection, inv0, inv) (frame,
    None, None, wo, routes, clean, scale) with the bf16 frame (P, D); block_b: (x, fc, wo, routes, clean)."""
    first, xa, fca, woa, routesa, cleana, scalea = block_a
    r = routesa.shape[1]
    assert block_b[3].shape[1] == r == 64, 'R = 64 only (the shared-memory budget is sized for it)'
    cfg = dict(CONFIG if root is None else ROOT_CONFIG, **overrides)
    bm = cfg.pop('BM')
    assert p % bm == 0 and d % 64 == 0 and first.is_contiguous() and first.dtype == torch.bfloat16
    layout = gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16)
    first_desc = TensorDescriptor.from_tensor(first.view(-1, d), [bm, 64], layout)
    dev = woa.device.index
    if dev not in _SMS:
        _SMS[dev] = (132 if woa.device.type == 'cpu'
          else torch.cuda.get_device_properties(woa.device).multi_processor_count)
    tiles = n * (p // bm)
    pad = stats  # unused pointer arguments
    if cfg.get('TC'):
        assert root is not None, 'TC: the root chain only'
        xpb, sqb = split_parts([block_b[0]], 3, square=True)
        vp, _ = split_parts(list(vectors[:4]), 2)
        tc_parts = (xpb, sqb, vp)
    else:
        tc_parts = (pad,) * 3
    _chain_kernel[(min(tiles, (CTAS_PER_SM if bm == 64 else 1) * _SMS[dev]),)](first_desc, pad if xa is None else xa,
      pad if fca is None else fca, woa, routesa, cleana, scalea, *block_b, *vectors, stats, *(root or (pad,) * 4),
      *tc_parts, tiles, p, d, r, bm, ROOT=root is not None, EPSILON=EPS, **cfg)
    return stats


@torch.library.custom_op('visit_kernel::fused_visit_readout_fast', mutates_args=())
def fused_visit_readout_fast(delta: torch.Tensor, x_a: torch.Tensor, fc_a: torch.Tensor, wo_a: torch.Tensor,
    routes_a: torch.Tensor, clean_a: torch.Tensor, scale_a: torch.Tensor, x_b: torch.Tensor, fc_b: torch.Tensor,
    wo_b: torch.Tensor, routes_b: torch.Tensor, clean_b: torch.Tensor, xl: torch.Tensor, target: torch.Tensor,
    mu: torch.Tensor, nu: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """fused_readout_visit_fast(fused_visit_fast(delta, x_a, fc_a, wo_a, routes_a, clean_a, scale_a), x_b, fc_b,
    wo_b, routes_b, clean_b, xl, target, mu, nu, diag) in one kernel: the readout moments (n, P, 6) fp32."""
    f32 = lambda z: z.float().contiguous()
    n, p, d = delta.shape
    stats = torch.empty((n, p, 6), device=delta.device, dtype=torch.float32)
    return launch_chain(n, p, d, (delta.contiguous(), f32(x_a), fc_a.contiguous(), wo_a.contiguous(),
      routes_a.contiguous(), f32(clean_a), scale_a), (f32(x_b), fc_b.contiguous(), wo_b.contiguous(),
      routes_b.contiguous(), f32(clean_b)), tuple(f32(z) for z in (xl, target, mu, nu, diag)), stats)


@fused_visit_readout_fast.register_fake
def _(delta, x_a, fc_a, wo_a, routes_a, clean_a, scale_a, x_b, fc_b, wo_b, routes_b, clean_b, xl, target, mu, nu,
    diag):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)


@torch.library.custom_op('visit_kernel::fused_root_readout_fast', mutates_args=())
def fused_root_readout_fast(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo_a: torch.Tensor,
    routes_a: torch.Tensor, clean_a: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor, scale_a: torch.Tensor,
    x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor, routes_b: torch.Tensor, clean_b: torch.Tensor,
    xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """fused_readout_visit_fast(fused_root_visit_fast(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv,
    scale_a, None)[0], x_b, fc_b, wo_b, routes_b, clean_b, xl, target, mu, nu, diag) in one kernel: the readout
    moments (n, P, 6) fp32; basis (P, D), the signed Hadamard frame, read as bf16 (exact: its entries are +-1)."""
    f32 = lambda z: z.float().contiguous()
    n, p, d = steps.shape[0], basis.shape[0], wo_a.shape[1]
    stats = torch.empty((n, p, 6), device=wo_a.device, dtype=torch.float32)
    return launch_chain(n, p, d, (basis.to(torch.bfloat16).contiguous(), None, None, wo_a.contiguous(),
      routes_a.contiguous(), f32(clean_a), scale_a), (f32(x_b), fc_b.contiguous(), wo_b.contiguous(),
      routes_b.contiguous(), f32(clean_b)), tuple(f32(z) for z in (xl, target, mu, nu, diag)), stats,
      root=(f32(steps), f32(projection), f32(inv0), f32(inv)))


@fused_root_readout_fast.register_fake
def _(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv, scale_a, x_b, fc_b, wo_b, routes_b, clean_b, xl,
    target, mu, nu, diag):
    return torch.empty((steps.shape[0], basis.shape[0], 6), device=wo_a.device, dtype=torch.float32)
