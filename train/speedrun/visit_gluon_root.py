"""Gluon (Hopper) kernel for a chain replay's root visit:
each move is the signed step along a frame direction, so the routed gate change comes from the precomputed projection
of the c_fc rows on the frame, and the output adds the move itself; the stored output is also summed against the
next block's clean input.

Persistent CTAs walk tiles of BM = 64 moves of one token. Per tile: gather q = Proj[routes, moves] (move-contiguous),
form the relu^2 change and store it to shared memory as the output wgmma's A operand; stream Wo[routes] chunks of 64
columns through a cp.async ring; per chunk add step * frame (its bf16 chunk TMA-loaded into the output staging slot),
scale, write back into the slot and send out by TMA.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from visit_chain import _bf16, _issue_rows, _rbf16

# NS: Wo ring slots, NST: output staging slots, CTAS: CTAs per SM (shared memory permitting)
ROOT_CONFIG = dict(NS=2, NST=3, CTAS=4, maxnreg=128)


@gluon.jit
def _frame_issue(frame_desc, fbar, stg, k, move0, c, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """TMA load of frame chunk (moves move0.., columns c) into staging slot k % NST."""
    slot = k % stg.shape[0]
    mbarrier.expect(fbar.index(slot), NBYTES, pred=pred)
    tma.async_copy_global_to_shared(frame_desc, [move0, c * BK], fbar.index(slot), stg.index(slot), pred=pred)


@gluon.jit
def _root_chunk(out, c, it, step, scale, token, row0, move0, nxt_move0, nxt_tile, nxt_token, TILES, q0, q1, nx, NextX,
    out_desc, frame_desc, fbar, stg, ocols, KC: gl.constexpr, NST: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr,
    D: gl.constexpr, mma2: gl.constexpr):
    """Output chunk c's epilogue: + step * frame chunk (in staging slot k % NST), scale, back into the slot, out by TMA;
    the slot of chunk k - 1 refills with frame chunk k + NST - 1 once its store has left."""
    k = it * KC + c
    slot = k % NST
    mbarrier.wait(fbar.index(slot), (k // NST) & 1)
    old = step * stg.index(slot).load(mma2).to(gl.float32)  # frame entries are +-1: exact
    value = _rbf16(old + _rbf16(out))
    stored = _rbf16(value * scale)
    result = _bf16(stored)  # exact
    # nx: next_x's chunk c, loaded a chunk ahead; the next one goes in flight
    q0 += gl.sum(stored * nx[None, :], 1)
    q1 += gl.sum(stored * stored, 1)
    nx = gl.load(NextX + gl.where(c + 1 < KC, token, nxt_token) * D + ((c + 1) % KC) * BK + ocols)
    stg.index(slot).store(result)  # each warp rewrites the rows it read
    fence_async_shared()
    gl.thread_barrier()
    tma.async_copy_shared_to_global(out_desc, [row0, c * BK], stg.index(slot))
    tma.store_wait(1)
    ahead = c + NST - 1
    _frame_issue(frame_desc, fbar, stg, k + NST - 1, gl.where(ahead < KC, move0, nxt_move0), ahead % KC,
      (ahead < KC) | (nxt_tile < TILES), BM * BK * 2, BK)
    return q0, q1, nx


@gluon.jit
def _root_kernel(out_desc, frame_desc, change_desc, Steps, Proj, Wo, Routes, Clean, Inv0, Inv, Scale, NextX, Stats, TILES,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, NS: gl.constexpr, NST: gl.constexpr,
    num_warps: gl.constexpr):
    BM: gl.constexpr = 64
    BK: gl.constexpr = 64
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    G = gl.num_programs(0)
    pid = gl.program_id(0)

    csm = gl.allocate_shared_memory(gl.bfloat16, [BM, R], gl.NVMMASharedLayout.get_default_for([BM, R], gl.bfloat16))
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NS, R, BK],
      gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    # output staging slots: slot of chunk k (per tile KC chunks, numbered across tiles) first receives the frame chunk
    # by TMA, then the output, which leaves by TMA; it refills with frame chunk k + NST once that store has left
    stg = gl.allocate_shared_memory(gl.bfloat16, [NST, BM, BK], out_desc.layout)
    fbar = gl.allocate_shared_memory(gl.int64, [NST, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NST):
        mbarrier.init(fbar.index(i), count=1)
    for i in gl.static_range(NST - 1):
        _frame_issue(frame_desc, fbar, stg, i, (pid % MT) * BM, i, True, BM * BK * 2, BK)

    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen = gl.load(Routes + (pid // MT) * R + rr).to(gl.int32)
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), Wo + chosen * D, cols, i * BK)

    # q tiles move-contiguous (Proj is (F, P)); the output wgmma's accumulator layout
    q_layout: gl.constexpr = gl.BlockedLayout([4, 1], [16, 2], [1, num_warps], [0, 1])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
      instr_shape=[16, BK, 16])
    qrows = gl.arange(0, BM, gl.SliceLayout(1, q_layout))
    qcols = gl.arange(0, R, gl.SliceLayout(0, q_layout))
    orows = gl.arange(0, BM, gl.SliceLayout(1, mma2))
    ocols = gl.arange(0, BK, gl.SliceLayout(0, mma2))
    scale = gl.load(Scale).to(gl.float32)
    zero = gl.zeros([BM, BK], gl.float32, mma2)

    nx = gl.load(NextX + (pid // MT) * D + ocols)
    # the first tile's projection gather; later tiles' are in flight under the previous tile's GEMM
    q = gl.load(Proj + gl.load(Routes + (pid // MT) * R + qcols).to(gl.int32)[None, :] * P
      + ((pid % MT) * BM + qrows)[:, None])
    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        move0 = (tile % MT) * BM
        row0 = token * P + move0
        base = (it * KC) % NS
        nxt_tile = tile + G
        nxt_token = gl.where(nxt_tile < TILES, nxt_tile // MT, token)
        nxt_move0 = (nxt_tile % MT) * BM
        chosen_next = gl.load(Routes + nxt_token * R + rr).to(gl.int32)
        wo_rows = Wo + chosen * D

        # gate change from the projection: moved = bf16(q * bf16(step)), the relu^2 change at the clean h
        step = gl.load(Steps + token)
        moved = _rbf16(q * step.to(gl.bfloat16).to(gl.float32))
        inv = gl.load(Inv + row0 + qrows)
        inv0 = gl.load(Inv0 + token)
        h = gl.load(Clean + token * R + qcols).to(gl.float32)
        dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
        up = gl.maximum(h[None, :] + dh, 0)
        down = gl.maximum(h[None, :], 0)
        gl.thread_barrier()  # the previous tile's output wgmmas (waited) are done with csm in every warp
        csm.store(_bf16((up - down) * (up + down)))
        fence_async_shared()
        gl.thread_barrier()
        # the change (BM x R bf16) out by TMA
        tma.async_copy_shared_to_global(change_desc, [row0, 0], csm)

        q0 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        q1 = gl.zeros([BM], gl.float32, gl.SliceLayout(1, mma2))
        q = gl.load(Proj + gl.load(Routes + nxt_token * R + qcols).to(gl.int32)[None, :] * P
          + (nxt_move0 + qrows)[:, None])
        for c in range(KC):
            cp.wait_group(NS - 2)
            fence_async_shared()
            gl.thread_barrier()
            _issue_rows(wsm.index((base + c + NS - 1) % NS),
              gl.where(c + NS - 1 < KC, wo_rows, Wo + chosen_next * D), cols, ((c + NS - 1) % KC) * BK)
            out = warpgroup_mma(csm, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
            out = warpgroup_mma_wait(0, deps=[out])
            q0, q1, nx = _root_chunk(out, c, it, step, scale, token, row0, move0, nxt_move0, nxt_tile, nxt_token,
              TILES, q0, q1, nx, NextX, out_desc, frame_desc, fbar, stg, ocols, KC, NST, BM, BK, D, mma2)
        stats_rows = Stats + (row0 + orows).to(gl.int64) * 2
        gl.store(stats_rows, q0)
        gl.store(stats_rows + 1, q1)
        chosen = chosen_next
        it += 1
    tma.store_wait(0)
    cp.wait_group(0)
    for i in gl.static_range(NST):
        mbarrier.invalidate(fbar.index(i))


@torch.library.custom_op('visit_kernel::fused_root_visit_change', mutates_args=())
def fused_root_visit_change(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor, scale: torch.Tensor,
    next_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The root visit, which also writes the routed change (n, P, R) bf16 (the output GEMM's A operand) and sums
    the output against next_x, the next block's clean input: (out (n, P, D) bf16, change, stats). R = 64."""
    n, p, d, r = steps.shape[0], basis.shape[0], wo.shape[1], routes.shape[1]
    assert p % 64 == 0 and d % 64 == 0 and r == 64
    cfg = dict(ROOT_CONFIG)
    ctas = cfg.pop('CTAS')
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    change = torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16)
    stats = torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32)
    layout = gl.NVMMASharedLayout.get_default_for([64, 64], gl.bfloat16)
    out_desc = TensorDescriptor.from_tensor(out.view(-1, d), [64, 64], layout)
    change_desc = TensorDescriptor.from_tensor(change.view(n * p, r), [64, r],
      gl.NVMMASharedLayout.get_default_for([64, r], gl.bfloat16))
    tiles, sms = n * (p // 64), torch.cuda.get_device_properties(wo.device).multi_processor_count
    f32 = lambda z: z.float().contiguous()
    frame_desc = TensorDescriptor.from_tensor(basis.to(torch.bfloat16).contiguous(), [64, 64], layout)
    _root_kernel[(min(tiles, ctas * sms),)](out_desc, frame_desc, change_desc, f32(steps),
      f32(projection), wo.contiguous(), routes.contiguous(), f32(clean), f32(inv0), f32(inv), scale,
      f32(next_x), stats, tiles, p, d, r, num_warps=4, **cfg)
    return out, change, stats


@fused_root_visit_change.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, scale, next_x):
    n, p, d, r = steps.shape[0], basis.shape[0], wo.shape[1], routes.shape[1]
    return (torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16),
      torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32))
