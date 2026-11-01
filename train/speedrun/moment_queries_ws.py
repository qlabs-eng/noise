"""Warp-specialized (Gluon, Hopper) query moments: the numbers of moment_kernels._moment_queries_kernel (bitwise), its
OUT_H path (moment_query_scores_into), with each softmax weight formed once instead of twice.

The Triton kernel runs two programs per (row, 64-query tile), parts 0 and 1 (feature pairs 0-15 and 16-31). Both form
the same logits, centered scores, softmax weights, their bf16 products (wn, a, wn * logit, a * logit) and the same
four row sums; they differ only in the moment GEMMs' feature columns. Here one CTA takes a whole (row, query tile)
work item with four partitions:
- softmax (default, one warpgroup): per key tile both K = 64 GEMMs (wgmma), the weights, the four row sums (in the
  reference's MMA layout, so the same reduction tree) and the four bf16 weight tiles into a shared memory slot;
- moments (two warpgroups, parts 0 and 1): per key tile the part's six moment GEMMs with the weights loaded into
  registers (ldmatrix, as the reference holds them) and the features from shared memory (the same instruction shapes,
  k-steps and accumulation order), then the part's response epilogue (the reference's code: row sums handed over);
- loads (one warp): every TMA load (query tiles double-buffered, then rings of key / value and feature tiles).
The pair features come in per-part column blocks (x | y | x^2 | y^2, pair_operands_blocked: the reference's values;
the column order changes no sum), so that the epilogue splits the moment sums in registers. The weights reach the
moment warpgroups through shared memory and mbarriers (no proxy fence: they are read with ldmatrix). The K = 64 GEMMs
of the next key tile are issued before each hand-over, ahead of the moment GEMMs it releases (the softmax loop is
unrolled by two tiles so that the in-flight accumulators keep their registers; ptxas serializes every wgmma once an
in-flight accumulator is copied). With PERSISTENT, one CTA per SM walks the items (heaviest first, dealt in a
zigzag), the next item's query tile and keys loading under the current item's tail.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# key / value / feature ring depth, weight slots, register budgets (the softmax warpgroup gets the rest)
CONFIG = dict(STAGES=4, SLOTS=4, MOMENT_REGS=168, LOAD_REGS=24)


@gluon.jit
def _item_count(ITEMS, PERSISTENT: gl.constexpr):
    """How many work items this CTA takes (persistent: of the zigzag-dealt items)."""
    if PERSISTENT:
        p = gl.program_id(0)
        P = gl.num_programs(0)
        R = ITEMS // P
        return R + gl.where(gl.where(R % 2 == 0, p, P - 1 - p) < ITEMS % P, 1, 0)
    else:
        return 1


@gluon.jit
def _item(k, ROWS, NT: gl.constexpr, BLOCK_S: gl.constexpr, BLOCK_T: gl.constexpr, WINDOW: gl.constexpr,
          PERSISTENT: gl.constexpr):
    """This CTA's k-th work item (items heaviest first, the reference's grid order; persistent: dealt to the CTAs in a
    zigzag, so that each takes a mix of heavy and light ones): its row, first query, first key tile, key tile count."""
    w = gl.program_id(0)
    if PERSISTENT:
        P = gl.num_programs(0)
        w = k * P + gl.where(k % 2 == 0, w, P - 1 - w)
    row = w % ROWS
    s0 = (NT - 1 - w // ROWS) * BLOCK_S
    start = gl.maximum(s0 - WINDOW, 0) // BLOCK_T
    n = (s0 + BLOCK_S - 1) // BLOCK_T + 1 - start
    return row, s0, start, n


# --- softmax partition ---------------------------------------------------------------------------------------------

@gluon.jit
def _issue(qs, hs, ks, vs, kvready, g, i, n, zero, S: gl.constexpr):
    """Key tile i's K = 64 GEMMs (ring index g + i; past the last tile: tile n - 1's again, drained unused)."""
    live = i < n
    r = gl.where(live, g + i, g + n - 1)
    mbarrier.wait(kvready.index(r % S), (r // S) & 1, pred=live)
    la = warpgroup_mma(qs, ks.index(r % S).permute((1, 0)), zero, use_acc=False, is_async=True)
    ca = warpgroup_mma(hs, vs.index(r % S).permute((1, 0)), zero, use_acc=False, is_async=True)
    return la, ca


@gluon.jit
def _step(la, ca, qs, hs, ks, vs, kvready, kvfree, g, i, n, it, scale, hub_out, lse, offs_s, wns, ans, wnls, als,
          aready, afree, s1n, s2n, s1d, s2d, zero, BLOCK_T: gl.constexpr, WINDOW: gl.constexpr, S: gl.constexpr,
          SA: gl.constexpr, mma: gl.constexpr):
    """Key tile i (the reference's loop body; ring index g + i): its weights into weight slot (g + i) % SA, tile i + 1's
    K = 64 GEMMs issued, the slot handed over, then tile i's row sums."""
    lt, ct = warpgroup_mma_wait(0, deps=[la, ca])
    r = g + i
    mbarrier.arrive(kvfree.index(r % S))
    logits = lt * scale
    c = ct - hub_out[:, None]
    offs_t = it * BLOCK_T + gl.arange(0, BLOCK_T, gl.SliceLayout(0, mma))
    lag = offs_s[:, None] - offs_t[None, :]
    a = gl.where((lag >= 0) & (lag <= WINDOW), gl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
    wn = a * c
    wnl = wn * logits
    al = a * logits
    sa = r % SA
    mbarrier.wait(afree.index(sa), ((r // SA) & 1) ^ 1)
    wnls.index(sa).store(wnl.to(gl.bfloat16))
    als.index(sa).store(al.to(gl.bfloat16))
    wns.index(sa).store(wn.to(gl.bfloat16))
    ans.index(sa).store(a.to(gl.bfloat16))
    gl.thread_barrier()
    la, ca = _issue(qs, hs, ks, vs, kvready, g, i + 1, n, zero, S)
    mbarrier.arrive(aready.index(sa))
    s1n += gl.sum(wnl, axis=1)
    s2n += gl.sum(wnl * logits, axis=1)
    s1d += gl.sum(al, axis=1)
    s2d += gl.sum(al * logits, axis=1)
    return la, ca, s1n, s2n, s1d, s2d


@gluon.jit
def _softmax_partition(lse_ptr, hubout_ptr, scale, qs2, hs2, ks, vs, kvready, kvfree, qready, qfree, wns, ans, wnls,
                       als, aready, afree, sums, sready, sfree, ITEMS, ROWS, T: gl.constexpr, NT: gl.constexpr,
                       BLOCK_T: gl.constexpr, BLOCK_S: gl.constexpr, WINDOW: gl.constexpr, S: gl.constexpr,
                       SA: gl.constexpr, PERSISTENT: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_T, 16])
    zero = gl.zeros([BLOCK_S, BLOCK_T], gl.float32, mma)
    g = 0
    for k in range(_item_count(ITEMS, PERSISTENT)):
        row, s0, start, n = _item(k, ROWS, NT, BLOCK_S, BLOCK_T, WINDOW, PERSISTENT)
        qs = qs2.index(k % 2)
        hs = hs2.index(k % 2)
        offs_s = s0 + gl.arange(0, BLOCK_S, gl.SliceLayout(1, mma))
        lse = gl.load(lse_ptr + row * T + offs_s)
        hub_out = gl.load(hubout_ptr + row * T + offs_s)
        s1n = gl.zeros([BLOCK_S], gl.float32, gl.SliceLayout(1, mma))
        s2n = gl.zeros([BLOCK_S], gl.float32, gl.SliceLayout(1, mma))
        s1d = gl.zeros([BLOCK_S], gl.float32, gl.SliceLayout(1, mma))
        s2d = gl.zeros([BLOCK_S], gl.float32, gl.SliceLayout(1, mma))
        mbarrier.wait(qready.index(k % 2), (k // 2) & 1)
        la, ca = _issue(qs, hs, ks, vs, kvready, g, 0, n, zero, S)
        for j in range(n // 2):
            i = 2 * j
            lb, cb, s1n, s2n, s1d, s2d = _step(la, ca, qs, hs, ks, vs, kvready, kvfree, g, i, n, start + i, scale,
                                               hub_out, lse, offs_s, wns, ans, wnls, als, aready, afree, s1n, s2n,
                                               s1d, s2d, zero, BLOCK_T, WINDOW, S, SA, mma)
            la, ca, s1n, s2n, s1d, s2d = _step(lb, cb, qs, hs, ks, vs, kvready, kvfree, g, i + 1, n, start + i + 1,
                                               scale, hub_out, lse, offs_s, wns, ans, wnls, als, aready, afree, s1n,
                                               s2n, s1d, s2d, zero, BLOCK_T, WINDOW, S, SA, mma)
        if n % 2 == 1:
            i = n - 1
            la, ca, s1n, s2n, s1d, s2d = _step(la, ca, qs, hs, ks, vs, kvready, kvfree, g, i, n, start + i, scale,
                                               hub_out, lse, offs_s, wns, ans, wnls, als, aready, afree, s1n, s2n,
                                               s1d, s2d, zero, BLOCK_T, WINDOW, S, SA, mma)
        warpgroup_mma_wait(0, deps=[la, ca])
        mbarrier.arrive(qfree.index(k % 2))
        sb = sums.index(k % 2)
        mbarrier.wait(sfree.index(k % 2), ((k // 2) & 1) ^ 1)
        sb.slice(0 * BLOCK_S, BLOCK_S).store(s1n)
        sb.slice(1 * BLOCK_S, BLOCK_S).store(s2n)
        sb.slice(2 * BLOCK_S, BLOCK_S).store(s1d)
        sb.slice(3 * BLOCK_S, BLOCK_S).store(s2d)
        gl.thread_barrier()
        mbarrier.arrive(sready.index(k % 2))
        g += n


# --- load partition ------------------------------------------------------------------------------------------------

@gluon.jit
def _load_partition(q_desc, k_desc, v_desc, hub_desc, f2_desc, f4_desc, xy_desc, qs2, hs2, ks, vs, f2s, f4s, xys,
                    qready, qfree, kvready, kvfree, fready, ffree, ITEMS, ROWS, T: gl.constexpr, D: gl.constexpr,
                    J: gl.constexpr, NT: gl.constexpr, BLOCK_T: gl.constexpr, BLOCK_S: gl.constexpr,
                    WINDOW: gl.constexpr, S: gl.constexpr, PERSISTENT: gl.constexpr):
    g = 0
    for k in range(_item_count(ITEMS, PERSISTENT)):
        row, s0, start, n = _item(k, ROWS, NT, BLOCK_S, BLOCK_T, WINDOW, PERSISTENT)
        base = row * T
        mbarrier.wait(qfree.index(k % 2), ((k // 2) & 1) ^ 1)
        mbarrier.expect(qready.index(k % 2), 2 * BLOCK_S * D * 2)
        tma.async_copy_global_to_shared(q_desc, [base + s0, 0], qready.index(k % 2), qs2.index(k % 2))
        tma.async_copy_global_to_shared(hub_desc, [base + s0, 0], qready.index(k % 2), hs2.index(k % 2))
        for i in range(n):
            r = g + i
            s = r % S
            free = ((r // S) & 1) ^ 1
            key = base + (start + i) * BLOCK_T
            mbarrier.wait(kvfree.index(s), free)
            mbarrier.expect(kvready.index(s), 2 * BLOCK_T * D * 2)
            tma.async_copy_global_to_shared(k_desc, [key, 0], kvready.index(s), ks.index(s))
            tma.async_copy_global_to_shared(v_desc, [key, 0], kvready.index(s), vs.index(s))
            mbarrier.wait(ffree.index(s), free)
            mbarrier.expect(fready.index(s), 2 * BLOCK_T * 7 * J * 2)
            for part in gl.static_range(2):
                tma.async_copy_global_to_shared(f2_desc, [key, part * 2 * J], fready.index(s), f2s.index(part * S + s))
                tma.async_copy_global_to_shared(f4_desc, [key, part * 4 * J], fready.index(s), f4s.index(part * S + s))
                tma.async_copy_global_to_shared(xy_desc, [key, part * J], fready.index(s), xys.index(part * S + s))
        g += n


# --- moment partitions ---------------------------------------------------------------------------------------------

@gluon.jit
def _coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, RAW_BLK, row, T: gl.constexpr, offs_s, d,
                  D: gl.constexpr):
    """moment_kernels._query_coefficients (BH = 1, M = T)."""
    i = d % (D // 2)
    raw = gl.load(raw_ptr + row.to(gl.int64) * RAW_BLK + (offs_s * D)[:, None] + d[None, :]).to(gl.float32)
    sq = gl.load(sq_ptr + row * T + offs_s)[:, None]
    steps = gl.load(steps_ptr + row * T + offs_s)[:, None]
    change = (2 * steps * raw + steps * steps) / D / (sq / D + 1e-6)
    a = libdevice.expm1(-0.5 * libdevice.log1p(change))
    b = steps * libdevice.rsqrt(sq / D + 1e-6) * (1 + a)
    cos = gl.load(cos_ptr + offs_s[:, None] * (D // 2) + i[None, :])
    sin = gl.load(sin_ptr + offs_s[:, None] * (D // 2) + i[None, :])
    first = (d < D // 2)[None, :]
    return a, b * gl.where(first, cos, sin), b * gl.where(first, -sin, cos)


@gluon.jit
def _blocks4(acc, BLOCK: gl.constexpr, J: gl.constexpr, lay: gl.constexpr):
    """The four J-column blocks (x | y | x^2 | y^2 moments) of a blocked-feature accumulator, in its own registers."""
    b = gl.reshape(gl.permute(gl.reshape(acc, (BLOCK, 2, 2, J)), (0, 3, 1, 2)), (BLOCK, J, 2, 2))
    lo, hi = gl.split(b)  # (x, x^2), (y, y^2)
    x, xx = gl.split(lo)
    y, yy = gl.split(hi)
    return (gl.convert_layout(x, lay), gl.convert_layout(y, lay), gl.convert_layout(xx, lay),
            gl.convert_layout(yy, lay))


@gluon.jit
def _blocks2(acc, BLOCK: gl.constexpr, J: gl.constexpr, lay: gl.constexpr):
    x, y = gl.split(gl.permute(gl.reshape(acc, (BLOCK, 2, J)), (0, 2, 1)))
    return gl.convert_layout(x, lay), gl.convert_layout(y, lay)


@gluon.jit
def _response(nx0, ny0, nx1, ny1, dx0, dy0, dx1, dy1, nxy, dxy, lx0, ly0, mx0, my0, s1n, s2n, s1d, s2d, raw_ptr, sq_ptr,
              steps_ptr, cos_ptr, sin_ptr, out_ptr, out_map, RAW_BLK, row, T: gl.constexpr, offs_s, d,
              OUT_H: gl.constexpr, OUT_LAYER, OUT_TOKEN, OUT_MAPPED: gl.constexpr, D: gl.constexpr):
    """moment_kernels._query_response (OUT_H path) for coordinates d, the moment sums already split (kept as the
    reference writes it: its fused multiply-adds, as ptxas forms them, depend on the expression structure)."""
    ca, bu, bw = _coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, RAW_BLK, row, T, offs_s, d, D)
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
          + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
          + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = gl.load(steps_ptr + row * T + offs_s)
    z = zn / (1.0 + zd) / steps[:, None]
    layer = row // OUT_H
    if OUT_MAPPED:
        layer = gl.load(out_map + layer)
    start = layer.to(gl.int64) * OUT_LAYER + (row % OUT_H) * D
    gl.store(out_ptr + start + offs_s[:, None] * OUT_TOKEN + d[None, :], z.to(gl.bfloat16).to(gl.float32))


@gluon.jit
def _moment_partition(f2s, f4s, xys, fready, ffree, wns, ans, wnls, als, aready, afree, sums, sready, sfree, raw_ptr,
                      sq_ptr, steps_ptr, cos_ptr, sin_ptr, out_ptr, out_map, RAW_BLK, OUT_LAYER, OUT_TOKEN, ITEMS, ROWS,
                      NT: gl.constexpr, WINDOW: gl.constexpr, T: gl.constexpr, D: gl.constexpr, J: gl.constexpr,
                      BLOCK_T: gl.constexpr, BLOCK_S: gl.constexpr, S: gl.constexpr, SA: gl.constexpr,
                      PART: gl.constexpr, OUT_H: gl.constexpr, OUT_MAPPED: gl.constexpr, PERSISTENT: gl.constexpr):
    mma4: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 4 * J, 16])
    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, J, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 2 * J, 16])
    op4: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma4, k_width=2)
    op1: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma1, k_width=2)
    op2: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
    g = 0
    for k in range(_item_count(ITEMS, PERSISTENT)):
        row, s0, start, n = _item(k, ROWS, NT, BLOCK_S, BLOCK_T, WINDOW, PERSISTENT)
        n4 = gl.zeros([BLOCK_S, 4 * J], gl.float32, mma4)
        d4 = gl.zeros([BLOCK_S, 4 * J], gl.float32, mma4)
        nxy = gl.zeros([BLOCK_S, J], gl.float32, mma1)
        dxy = gl.zeros([BLOCK_S, J], gl.float32, mma1)
        l2 = gl.zeros([BLOCK_S, 2 * J], gl.float32, mma2)
        m2 = gl.zeros([BLOCK_S, 2 * J], gl.float32, mma2)
        for i in range(n):
            r = g + i
            s = r % S
            sa = r % SA
            f = PART * S + s
            mbarrier.wait(fready.index(s), (r // S) & 1)
            mbarrier.wait(aready.index(sa), (r // SA) & 1)
            w0 = wns.index(sa).load(op4)  # the weights into registers once (two at a time)
            w1 = ans.index(sa).load(op4)
            n4 = warpgroup_mma(w0, f4s.index(f), n4, is_async=True)
            nxy = warpgroup_mma(gl.convert_layout(w0, op1), xys.index(f), nxy, is_async=True)
            d4 = warpgroup_mma(w1, f4s.index(f), d4, is_async=True)
            dxy = warpgroup_mma(gl.convert_layout(w1, op1), xys.index(f), dxy, is_async=True)
            n4, nxy, d4, dxy = warpgroup_mma_wait(0, deps=[n4, nxy, d4, dxy])
            w2 = wnls.index(sa).load(op2)
            w3 = als.index(sa).load(op2)
            l2 = warpgroup_mma(w2, f2s.index(f), l2, is_async=True)
            m2 = warpgroup_mma(w3, f2s.index(f), m2, is_async=True)
            l2, m2 = warpgroup_mma_wait(0, deps=[l2, m2])
            mbarrier.arrive(afree.index(sa))
            mbarrier.arrive(ffree.index(s))
        sb = sums.index(k % 2)
        mbarrier.wait(sready.index(k % 2), (k // 2) & 1)
        s1n = sb.slice(0 * BLOCK_S, BLOCK_S).load(gl.SliceLayout(1, mma1))
        s2n = sb.slice(1 * BLOCK_S, BLOCK_S).load(gl.SliceLayout(1, mma1))
        s1d = sb.slice(2 * BLOCK_S, BLOCK_S).load(gl.SliceLayout(1, mma1))
        s2d = sb.slice(3 * BLOCK_S, BLOCK_S).load(gl.SliceLayout(1, mma1))
        gl.thread_barrier()
        mbarrier.arrive(sfree.index(k % 2))
        nx0, ny0, nx1, ny1 = _blocks4(n4, BLOCK_S, J, mma1)
        dx0, dy0, dx1, dy1 = _blocks4(d4, BLOCK_S, J, mma1)
        lx0, ly0 = _blocks2(l2, BLOCK_S, J, mma1)
        mx0, my0 = _blocks2(m2, BLOCK_S, J, mma1)
        offs_s = s0 + gl.arange(0, BLOCK_S, gl.SliceLayout(1, mma1))
        pairs = PART * J + gl.arange(0, J, gl.SliceLayout(0, mma1))
        for half in gl.static_range(2):
            _response(nx0, ny0, nx1, ny1, dx0, dy0, dx1, dy1, nxy, dxy, lx0, ly0, mx0, my0, s1n, s2n, s1d, s2d, raw_ptr,
                      sq_ptr, steps_ptr, cos_ptr, sin_ptr, out_ptr, out_map, RAW_BLK, row, T, offs_s,
                      pairs + half * (D // 2), OUT_H, OUT_LAYER, OUT_TOKEN, OUT_MAPPED, D)
        g += n


# --- blocked feature operands ---------------------------------------------------------------------------------------

@triton.jit
def _pair_features_blocked(X, F2, F4, XY, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """moment_kernels._pair_features' values (x = X[i] / sqrt(D), y = X[i + D / 2]) with each part's feature columns
    in blocks of its D / 4 pairs: F2 (x | y), F4 (x | y | x^2 | y^2) per part, XY = x y as there."""
    J: tl.constexpr = D // 2
    row, tile = tl.program_id(0), tl.program_id(1)
    token = row.to(tl.int64) * T + tile * BLOCK + tl.arange(0, BLOCK)
    pairs = tl.arange(0, J)
    x = tl.load(X + token[:, None] * D + pairs[None, :]) * D ** (-0.5)
    y = tl.load(X + token[:, None] * D + (pairs + J)[None, :]) * D ** (-0.5)
    f2 = tl.reshape(tl.join(x, y), (BLOCK, 2, J // 2, 2))  # (part, pair, x / y)
    f2 = tl.reshape(tl.permute(f2, (0, 1, 3, 2)), (BLOCK, D))
    tl.store(F2 + token[:, None] * D + tl.arange(0, D)[None, :], f2.to(tl.bfloat16))
    # (part, pair, x / y, first / second power)
    f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BLOCK, 2, J // 2, 2, 2))
    f4 = tl.reshape(tl.permute(f4, (0, 1, 4, 3, 2)), (BLOCK, 2 * D))
    tl.store(F4 + token[:, None] * (2 * D) + tl.arange(0, 2 * D)[None, :], f4.to(tl.bfloat16))
    tl.store(XY + token[:, None] * J + pairs[None, :], (x * y).to(tl.bfloat16))


def pair_operands_blocked(x):
    rows, t, d = x.shape
    f2 = x.new_empty((rows, t, d), dtype=torch.bfloat16)
    f4 = x.new_empty((rows, t, 2 * d), dtype=torch.bfloat16)
    xy = x.new_empty((rows, t, d // 2), dtype=torch.bfloat16)
    _pair_features_blocked[rows, t // 64](x, f2, f4, xy, T=t, D=d, BLOCK=64, num_warps=4)
    return f2, f4, xy


# --- kernel --------------------------------------------------------------------------------------------------------

@gluon.jit
def _moment_queries_ws_kernel(q_desc, k_desc, v_desc, hub_desc, f2_desc, f4_desc, xy_desc, raw_ptr, sq_ptr, cos_ptr,
                              sin_ptr, lse_ptr, hubout_ptr, steps_ptr, out_ptr, out_map, scale, RAW_BLK, OUT_LAYER,
                              OUT_TOKEN, ITEMS, ROWS, T: gl.constexpr, D: gl.constexpr, J: gl.constexpr,
                              BLOCK_T: gl.constexpr, BLOCK_S: gl.constexpr, WINDOW: gl.constexpr, OUT_H: gl.constexpr,
                              OUT_MAPPED: gl.constexpr, S: gl.constexpr, SA: gl.constexpr, REGS: gl.constexpr,
                              LOAD_REGS: gl.constexpr, PERSISTENT: gl.constexpr):
    gl.static_assert(D // 2 == 2 * J)  # two parts
    NT: gl.constexpr = T // BLOCK_S
    qs2 = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_S, D], q_desc.layout)
    hs2 = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_S, D], hub_desc.layout)
    ks = gl.allocate_shared_memory(gl.bfloat16, [S, BLOCK_T, D], k_desc.layout)
    vs = gl.allocate_shared_memory(gl.bfloat16, [S, BLOCK_T, D], v_desc.layout)
    f2s = gl.allocate_shared_memory(gl.bfloat16, [2 * S, BLOCK_T, 2 * J], f2_desc.layout)
    f4s = gl.allocate_shared_memory(gl.bfloat16, [2 * S, BLOCK_T, 4 * J], f4_desc.layout)
    xys = gl.allocate_shared_memory(gl.bfloat16, [2 * S, BLOCK_T, J], xy_desc.layout)
    alay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_S, BLOCK_T], gl.bfloat16)
    wns = gl.allocate_shared_memory(gl.bfloat16, [SA, BLOCK_S, BLOCK_T], alay)
    ans = gl.allocate_shared_memory(gl.bfloat16, [SA, BLOCK_S, BLOCK_T], alay)
    wnls = gl.allocate_shared_memory(gl.bfloat16, [SA, BLOCK_S, BLOCK_T], alay)
    als = gl.allocate_shared_memory(gl.bfloat16, [SA, BLOCK_S, BLOCK_T], alay)
    sums = gl.allocate_shared_memory(gl.float32, [2, 4 * BLOCK_S], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    blay: gl.constexpr = mbarrier.MBarrierLayout()
    kvready = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    kvfree = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    fready = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    ffree = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    aready = gl.allocate_shared_memory(gl.int64, [SA, 1], blay)
    afree = gl.allocate_shared_memory(gl.int64, [SA, 1], blay)
    qready = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    qfree = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    sready = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    sfree = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    for i in gl.static_range(S):
        mbarrier.init(kvready.index(i), count=1)
        mbarrier.init(kvfree.index(i), count=1)
        mbarrier.init(fready.index(i), count=1)
        mbarrier.init(ffree.index(i), count=2)
    for i in gl.static_range(SA):
        mbarrier.init(aready.index(i), count=1)
        mbarrier.init(afree.index(i), count=2)
    for i in gl.static_range(2):
        mbarrier.init(qready.index(i), count=1)
        mbarrier.init(qfree.index(i), count=1)
        mbarrier.init(sready.index(i), count=1)
        mbarrier.init(sfree.index(i), count=2)
    gl.warp_specialize([
        (_softmax_partition, (lse_ptr, hubout_ptr, scale, qs2, hs2, ks, vs, kvready, kvfree, qready, qfree, wns, ans,
                              wnls, als, aready, afree, sums, sready, sfree, ITEMS, ROWS, T, NT, BLOCK_T, BLOCK_S,
                              WINDOW, S, SA, PERSISTENT)),
        (_moment_partition, (f2s, f4s, xys, fready, ffree, wns, ans, wnls, als, aready, afree, sums, sready, sfree,
                             raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, out_ptr, out_map, RAW_BLK, OUT_LAYER,
                             OUT_TOKEN, ITEMS, ROWS, NT, WINDOW, T, D, J, BLOCK_T, BLOCK_S, S, SA, 0, OUT_H,
                             OUT_MAPPED, PERSISTENT)),
        (_moment_partition, (f2s, f4s, xys, fready, ffree, wns, ans, wnls, als, aready, afree, sums, sready, sfree,
                             raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, out_ptr, out_map, RAW_BLK, OUT_LAYER,
                             OUT_TOKEN, ITEMS, ROWS, NT, WINDOW, T, D, J, BLOCK_T, BLOCK_S, S, SA, 1, OUT_H,
                             OUT_MAPPED, PERSISTENT)),
        (_load_partition, (q_desc, k_desc, v_desc, hub_desc, f2_desc, f4_desc, xy_desc, qs2, hs2, ks, vs, f2s, f4s,
                           xys, qready, qfree, kvready, kvfree, fready, ffree, ITEMS, ROWS, T, D, J, NT, BLOCK_T,
                           BLOCK_S, WINDOW, S, PERSISTENT)),
    ], [4, 4, 1], [REGS, REGS, LOAD_REGS])


_SMS = {}


def _desc(x, block):
    return TensorDescriptor.from_tensor(x, block, gl.NVMMASharedLayout.get_default_for(block, gl.bfloat16))


def moment_query_scores_ws(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, window, out_heads, layers,
                           persistent=True):
    """moment_kernels._moment_query_scores' OUT_H path: q, k, v, hub (rows, T, 64) bf16, raw (1, rows, T, 64) fp32,
    side (rows, T, 64) fp32 (the pair features' source), out a (layers, T, heads * 64) fp32 view (last stride 1)."""
    rows, t, d = q.shape
    j, block_t, block_s = d // 4, 32, 64
    c = CONFIG
    assert d == 64 and t % block_s == 0 and raw.shape[:2] == (1, rows) and raw.stride(2) == d and raw.stride(3) == 1
    assert out.stride(2) == 1
    f2, f4, xy = pair_operands_blocked(side.contiguous())
    flat = lambda z: z.reshape(rows * t, z.shape[-1])
    items = rows * (t // block_s)
    if q.device.index not in _SMS:
        _SMS[q.device.index] = torch.cuda.get_device_properties(q.device).multi_processor_count
    _moment_queries_ws_kernel[(min(items, _SMS[q.device.index]) if persistent else items,)](
        _desc(flat(q), [block_s, d]), _desc(flat(k), [block_t, d]), _desc(flat(v), [block_t, d]),
        _desc(flat(hub), [block_s, d]), _desc(flat(f2), [block_t, 2 * j]), _desc(flat(f4), [block_t, 4 * j]),
        _desc(flat(xy), [block_t, j]), raw, sq, cos, sin, lse, hub_out, steps, out, out if layers is None else layers,
        d ** -0.5, raw.stride(1), out.stride(0), out.stride(1), items, rows, T=t, D=d, J=j, BLOCK_T=block_t,
        BLOCK_S=block_s, WINDOW=window, OUT_H=out_heads, OUT_MAPPED=layers is not None, S=c['STAGES'], SA=c['SLOTS'],
        REGS=c['MOMENT_REGS'], LOAD_REGS=c['LOAD_REGS'], PERSISTENT=persistent, num_warps=4)
