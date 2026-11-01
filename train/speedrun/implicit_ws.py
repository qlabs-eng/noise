"""Gluon (Hopper) lse / center: moment_kernels._implicit_key_statistics_edges' outputs, bitwise the same numbers, with
each key block's GEMMs issued one block ahead of its online-softmax update.

The same tiles (64 queries x 64 keys), the same blocks in the same order (the diagonal block, the window's first
block, then the whole blocks in order), the same MMA instructions in the same k order and the same layout for every
row reduction. What changes: the query, hub, key and value tiles come by TMA (keys and values in a two-slot ring), and
in the loop over whole blocks block j + 1's GEMMs run under block j's softmax update (the reference waits for each
GEMM right away; the cut blocks keep that). No in-flight accumulator crosses a branch or the loop's back edge (two
blocks per iteration): ptxas would copy it and then serialize every wgmma in the kernel.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _block(j, nb, tile, lo):
    """The j-th key block of the query tile (j = 0: the diagonal block, then lo, lo + 1, ..., tile - 1)."""
    return gl.where(j == 0, tile, lo + j - 1)


@gluon.jit
def _load_block(k_desc, v_desc, ks, vs, ready, row0, blk, j, B: gl.constexpr, J: gl.constexpr):
    s = j % 2
    mbarrier.expect(ready.index(s), 4 * B * J * 4)
    for b in gl.static_range(2):
        tma.async_copy_global_to_shared(k_desc, [row0 + blk * B, b * J], ready.index(s), ks.index(2 * s + b))
        tma.async_copy_global_to_shared(v_desc, [row0 + blk * B, b * J], ready.index(s), vs.index(2 * s + b))


@gluon.jit
def _issue(qs, hs, ks, vs, ready, j):
    """Block j's GEMMs (ring slot j % 2), once its tiles are in."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 8])
    zero = gl.zeros([64, 64], gl.float32, mma)
    s = j % 2
    mbarrier.wait(ready.index(s), (j // 2) & 1)
    lt = warpgroup_mma(qs.index(0), ks.index(2 * s).permute((1, 0)), zero, use_acc=False, is_async=True)
    lt = warpgroup_mma(qs.index(1), ks.index(2 * s + 1).permute((1, 0)), lt, is_async=True)
    st = warpgroup_mma(hs.index(0), vs.index(2 * s).permute((1, 0)), zero, use_acc=False, is_async=True)
    st = warpgroup_mma(hs.index(1), vs.index(2 * s + 1).permute((1, 0)), st, is_async=True)
    return lt, st


@gluon.jit
def _update(lt, st, blk, maximum, total, score_sum, query, Window: gl.constexpr, B: gl.constexpr,
            MASKED: gl.constexpr):
    """The reference's online-softmax update for one key block (moment_kernels._implicit_block)."""
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, B, 8])
    logits = lt * 0.125
    if MASKED:
        source = blk * B + gl.arange(0, B, gl.SliceLayout(0, mma))
        allowed = (source[None, :] <= query[:, None]) & (query[:, None] - source[None, :] <= Window)
        logits = gl.where(allowed, logits, float('-inf'))
    next_max = gl.maximum(maximum, gl.max(logits, 1))
    safe_max = next_max
    if MASKED:
        safe_max = gl.where(next_max == float('-inf'), 0.0, next_max)
    rescale = gl.exp(maximum - safe_max)
    prob = gl.exp(logits - safe_max[:, None])
    total = total * rescale + gl.sum(prob, 1)
    score_sum = score_sum * rescale + gl.sum(prob * st, 1)
    return next_max, total, score_sum


@gluon.jit
def _sync_step(j, nb, tile, lo, maximum, total, score_sum, qs, hs, ks, vs, ready, k_desc, v_desc, row0, query,
               Window: gl.constexpr, B: gl.constexpr, J: gl.constexpr, MASKED: gl.constexpr):
    """Key block j without overlap (its GEMMs waited for at once): the blocks outside the pipelined loop."""
    lt, st = _issue(qs, hs, ks, vs, ready, j)
    lt, st = warpgroup_mma_wait(0, deps=[lt, st])
    if j + 2 < nb:
        _load_block(k_desc, v_desc, ks, vs, ready, row0, _block(j + 2, nb, tile, lo), j + 2, B, J)
    return _update(lt, st, _block(j, nb, tile, lo), maximum, total, score_sum, query, Window, B, MASKED)


@gluon.jit
def _step(j, nb, tile, lo, lt, st, maximum, total, score_sum, qs, hs, ks, vs, ready, k_desc, v_desc, row0, query,
          Window: gl.constexpr, B: gl.constexpr, J: gl.constexpr, NEXT: gl.constexpr):
    """Whole key block j in the pipelined loop: its GEMMs' results, block j + 2's load into the freed slot, block
    j + 1's GEMMs issued (NEXT), then the softmax update under them."""
    lt, st = warpgroup_mma_wait(0, deps=[lt, st])
    if j + 2 < nb:  # block j's slot is free (only its GEMMs read it)
        _load_block(k_desc, v_desc, ks, vs, ready, row0, _block(j + 2, nb, tile, lo), j + 2, B, J)
    if NEXT:
        nlt, nst = _issue(qs, hs, ks, vs, ready, j + 1)
        maximum, total, score_sum = _update(lt, st, _block(j, nb, tile, lo), maximum, total, score_sum, query,
                                            Window, B, False)
        return nlt, nst, maximum, total, score_sum
    else:
        return _update(lt, st, _block(j, nb, tile, lo), maximum, total, score_sum, query, Window, B, False)


@gluon.jit
def _implicit_ws_kernel(q_desc, k_desc, v_desc, h_desc, LSE, Center, T: gl.constexpr, Window: gl.constexpr,
                        B: gl.constexpr):
    J: gl.constexpr = 32
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, B, 8])
    tile = T // B - 1 - gl.program_id(0)  # the heaviest query tiles first
    head = gl.program_id(1)
    row0 = head * T
    qs = gl.allocate_shared_memory(gl.float32, [2, B, J], q_desc.layout)
    hs = gl.allocate_shared_memory(gl.float32, [2, B, J], h_desc.layout)
    ks = gl.allocate_shared_memory(gl.float32, [4, B, J], k_desc.layout)
    vs = gl.allocate_shared_memory(gl.float32, [4, B, J], v_desc.layout)
    blay: gl.constexpr = mbarrier.MBarrierLayout()
    qbar = gl.allocate_shared_memory(gl.int64, [1], blay)
    ready = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    mbarrier.init(qbar, count=1)
    for i in gl.static_range(2):
        mbarrier.init(ready.index(i), count=1)
    start = tile * B - Window
    has_lo = start >= 0
    lo = gl.where(has_lo, start // B, 0)
    nb = 1 + tile - lo  # the diagonal block, then lo .. tile - 1
    mbarrier.expect(qbar, 4 * B * J * 4)
    for b in gl.static_range(2):
        tma.async_copy_global_to_shared(q_desc, [row0 + tile * B, b * J], qbar, qs.index(b))
        tma.async_copy_global_to_shared(h_desc, [row0 + tile * B, b * J], qbar, hs.index(b))
    _load_block(k_desc, v_desc, ks, vs, ready, row0, tile, 0, B, J)
    if nb > 1:
        _load_block(k_desc, v_desc, ks, vs, ready, row0, lo, 1, B, J)
    query = tile * B + gl.arange(0, B, gl.SliceLayout(1, mma))
    maximum = gl.full([B], float('-inf'), gl.float32, gl.SliceLayout(1, mma))
    total = gl.zeros([B], gl.float32, gl.SliceLayout(1, mma))
    score_sum = gl.zeros([B], gl.float32, gl.SliceLayout(1, mma))
    mbarrier.wait(qbar, 0)
    # the blocks the window cuts (the diagonal one, the window's first) and, with an odd count of whole blocks, the
    # first whole one, without overlap
    maximum, total, score_sum = _sync_step(0, nb, tile, lo, maximum, total, score_sum, qs, hs, ks, vs, ready, k_desc,
                                           v_desc, row0, query, Window, B, J, True)
    first = 1
    if has_lo:
        maximum, total, score_sum = _sync_step(1, nb, tile, lo, maximum, total, score_sum, qs, hs, ks, vs, ready,
                                               k_desc, v_desc, row0, query, Window, B, J, True)
        first = 2
    if (nb - first) % 2 == 1:
        maximum, total, score_sum = _sync_step(first, nb, tile, lo, maximum, total, score_sum, qs, hs, ks, vs, ready,
                                               k_desc, v_desc, row0, query, Window, B, J, False)
        first += 1
    if first < nb:  # an even count of whole blocks, two per iteration (the accumulators alternate register sets)
        lt, st = _issue(qs, hs, ks, vs, ready, first)
        for jj in range((nb - first) // 2 - 1):
            j = first + 2 * jj
            lt, st, maximum, total, score_sum = _step(j, nb, tile, lo, lt, st, maximum, total, score_sum, qs, hs, ks,
                                                      vs, ready, k_desc, v_desc, row0, query, Window, B, J, True)
            lt, st, maximum, total, score_sum = _step(j + 1, nb, tile, lo, lt, st, maximum, total, score_sum, qs, hs,
                                                      ks, vs, ready, k_desc, v_desc, row0, query, Window, B, J, True)
        lt, st, maximum, total, score_sum = _step(nb - 2, nb, tile, lo, lt, st, maximum, total, score_sum, qs, hs, ks,
                                                  vs, ready, k_desc, v_desc, row0, query, Window, B, J, True)
        maximum, total, score_sum = _step(nb - 1, nb, tile, lo, lt, st, maximum, total, score_sum, qs, hs, ks, vs,
                                          ready, k_desc, v_desc, row0, query, Window, B, J, False)
    mbarrier.invalidate(qbar)
    for i in gl.static_range(2):
        mbarrier.invalidate(ready.index(i))
    gl.store(LSE + head * T + query, maximum + gl.log(total))
    gl.store(Center + head * T + query, score_sum / total)


def _desc(x, block):
    return TensorDescriptor.from_tensor(x, block, gl.NVMMASharedLayout.get_default_for(block, gl.float32))


def supported(q, window, b=64):
    h, t, d = q.shape
    return d == 64 and window % b == 0 and window >= b and t % b == 0 and q.dtype == torch.float32


def implicit_key_context_ws(q, k, values, hub, window, b=64):
    """moment_kernels._implicit_key_context: q, k, values, hub (rows, T, 64) fp32 -> lse, center (rows, T)."""
    h, t, d = q.shape
    q, k, values, hub = (z.contiguous() for z in (q, k, values, hub))
    lse = torch.empty((h, t), device=q.device)
    center = torch.empty_like(lse)
    q2, k2, v2, h2 = (z.view(h * t, d) for z in (q, k, values, hub))
    _implicit_ws_kernel[(t // b, h)](_desc(q2, [b, d // 2]), _desc(k2, [b, d // 2]), _desc(v2, [b, d // 2]),
                                     _desc(h2, [b, d // 2]), lse, center, T=t, Window=window, B=b, num_warps=4)
    return lse, center
