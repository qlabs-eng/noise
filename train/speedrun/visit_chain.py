"""Pieces the Gluon (Hopper) chain kernels share (visit_chain3.py, terminal_panels.py): the cp.async ring of routed
rows, a block's gate and output chunks with the tile's delta kept in shared memory between them, the readout chunks
and their sums, packed bf16 conversions, and the readout vectors split into bf16 parts for the tensor cores.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)

EPS = torch.finfo(torch.float32).eps

# _bf16 / _rbf16: packed fp32 -> bf16 conversions (cvt.rn.bf16x2.f32) where ptxas emits scalar F2F; bitwise
# TC (chain3): dot products on the tensor cores: the gates of blocks fed from shared memory take d . x as a
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
def _gate_done(sq_x, dots, moved_sq, mma1: gl.constexpr):
    """The norm sums per row, in the gate accumulator's row layout."""
    rows: gl.constexpr = gl.SliceLayout(1, mma1)
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
    mma2: gl.constexpr):
    """The routed MLP's change relu(h + dh)^2 - relu(h)^2 as the output GEMM's register A operand."""
    inv0 = gl.rsqrt(sq_x / D + EPSILON)
    inv_moved = gl.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
    inv = gl.convert_layout(inv_moved, gl.SliceLayout(1, mma1))
    ratio = gl.convert_layout(inv_moved / inv0 - 1, gl.SliceLayout(1, mma1))
    dh = inv[:, None] * _rbf16(acc) + ratio[:, None] * h[None, :]
    up = gl.maximum(h[None, :] + dh, 0)
    down = gl.maximum(h[None, :], 0)
    change = _bf16((up - down) * (up + down))
    return gl.convert_layout(change, gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))


@gluon.jit
def _row_done(part, mma2: gl.constexpr):
    return gl.convert_layout(gl.sum(part, 1), gl.SliceLayout(1, mma2))


@gluon.jit
def _row_sum(t, BM: gl.constexpr, BK: gl.constexpr):
    """Per-thread partials of an accumulator-layout chunk's row sums: column 8 g + k of a thread's rows sits in its
    registers for all 8 g, so summing over g needs no shuffles (reduced once per tile)."""
    return gl.sum(gl.reshape(t, [BM, 8, BK // 8]), 1)


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
def _readout_diag(dg, c, vcols: gl.constexpr):
    return dg.index(c).load(vcols)


@gluon.jit
def _output_readout_tc(c, accr, s1, s5, delta_desc, dbar, dsm, wsm, v3, dg, change, zero, it, base,
    wo_rows, next_rows, cols, nxt_row0, has_next,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, NBYTES: gl.constexpr,
    mma2: gl.constexpr, mmax: gl.constexpr, BM: gl.constexpr = 64, VOFF: gl.constexpr = 0,
    DOFF: gl.constexpr = 0):
    """Block B's output chunk c (runtime c: dg is [*, BK] fp32, chunk c's diag in row DOFF + c), reduced to the readout
    moments; its slot then takes the next tile's chunk c (base: the ring slot of the output's first item). The four
    linear moments are one wgmma of the output chunk (A from registers, waited at once) against the vector parts
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
    s1 += _row_sum(v * v, BM, BK)
    s5 += _row_sum(v * v * _readout_diag(dg, DOFF + c, vcols)[None, :], BM, BK)
    return accr, s1, s5


@gluon.jit
def _readout_tc_done(accr, s1, s5, mma2: gl.constexpr, mmax: gl.constexpr):
    """(s0, ..., s5) per row in mma2's row layout from the readout accumulator and the squares' sums."""
    rows: gl.constexpr = gl.SliceLayout(1, mma2)
    pr = gl.arange(0, 8, gl.SliceLayout(0, mmax)) // 2
    s0 = gl.convert_layout(gl.sum(gl.where((pr == 0)[None, :], accr, 0.0), 1), rows)
    s2 = gl.convert_layout(gl.sum(gl.where((pr == 1)[None, :], accr, 0.0), 1), rows)
    s3 = gl.convert_layout(gl.sum(gl.where((pr == 2)[None, :], accr, 0.0), 1), rows)
    s4 = gl.convert_layout(gl.sum(gl.where((pr == 3)[None, :], accr, 0.0), 1), rows)
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


