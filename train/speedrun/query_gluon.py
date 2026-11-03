"""Gluon (Hopper) routed query replay: query_replay.fused_query_replay (fused_query_weights(delta @ w_q.T, ...)) with
the delta tile read once.

Persistent CTAs (one per SM, two warpgroups of 64 moves) walk tiles of BM = 128 moves of one token. The tile's delta
(BM x C bf16) comes in by TMA and stays in shared memory for all heads; w_q streams through a TMA ring, a pair of
heads (128 rows) by 64 columns per item; per head pair raw = bf16(delta @ w_q[pair]^T) accumulates on wgmma, then the
reference's replay epilogue runs on it in registers: w_q's rows are interleaved per head (a0 b0 a1 b1 ...) so each
RoPE pair sits in one thread; the key scores of both heads come from one wgmma against a block-diagonal key matrix
(exact: the off-diagonal zeros add nothing), its A operand the epilogue's dq in the accumulator layout.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

EPS = torch.finfo(torch.float32).eps
CONFIG = dict(BM=128, NW=4, num_warps=8)
_SMS = {}


@gluon.jit
def _w_issue(wq_desc, wbar, wsm, g, pred, KC: gl.constexpr, BK: gl.constexpr, NW: gl.constexpr,
    WROWS: gl.constexpr):
    """w_q ring item g: head pair (g // KC) % HP's rows, columns of chunk g % KC (the same items every tile)."""
    slot = g % NW
    pair = (g // KC) % (wq_desc.shape[0] // WROWS)
    mbarrier.expect(wbar.index(slot), WROWS * BK * 2, pred=pred)
    tma.async_copy_global_to_shared(wq_desc, [pair * WROWS, (g % KC) * BK], wbar.index(slot), wsm.index(slot),
      pred=pred)


@gluon.jit
def _replay_pair(acc, kb, token, pair, row0, pos0, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out,
    P: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, EPS_: gl.constexpr,
    num_warps: gl.constexpr, mmak: gl.constexpr):
    """The reference's replay epilogue for heads 2 pair, 2 pair + 1 of the tile (acc: fp32 raw queries, BM x 2 D,
    each head's RoPE halves interleaved)."""
    HD: gl.constexpr = D // 2
    raw = acc.to(gl.bfloat16).to(gl.float32)
    a, b = gl.split(gl.reshape(raw, [BM, D, 2]))
    a = gl.reshape(a, [BM, 2, HD])  # (move, head of the pair, i)
    b = gl.reshape(b, [BM, 2, HD])
    L3: gl.constexpr = a.type.layout
    vl: gl.constexpr = gl.SliceLayout(0, L3)
    hh = gl.arange(0, 2, gl.SliceLayout(1, vl))
    ii = gl.arange(0, HD, gl.SliceLayout(0, vl))
    c = gl.expand_dims(gl.expand_dims(gl.load(Cos + token * HD + ii), 0), 0)
    s = gl.expand_dims(gl.expand_dims(gl.load(Sin + token * HD + ii), 0), 0)
    r0_rows = R0 + ((token * H + 2 * pair + hh) * D)[:, None]
    r0a = gl.load(r0_rows + ii[None, :])[None, :, :]
    r0b = gl.load(r0_rows + HD + ii[None, :])[None, :, :]
    rl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    pp = pos0 + gl.arange(0, BM, rl)
    g = gl.expand_dims(gl.expand_dims(gl.load(Inv + token * P + pp), 1), 2)
    shift0 = gl.expand_dims(gl.expand_dims(gl.load(Shift + token * P + pp), 1), 2)
    ra = a * c + b * s
    rb = -a * s + b * c
    ra, rb = ra * g + shift0 * r0a, rb * g + shift0 * r0b
    sq = gl.sum(r0a * r0a, 2) + gl.sum(r0b * r0b, 2)
    dots = gl.sum(ra * r0a, 2) + gl.sum(rb * r0b, 2)
    moved = gl.sum(ra * ra, 2) + gl.sum(rb * rb, 2)
    inv0 = gl.rsqrt(sq / D + EPS_)
    inv = gl.rsqrt((sq + 2 * dots + moved) / D + EPS_)
    shift = gl.expand_dims(inv - inv0, 2)
    inv = gl.expand_dims(inv, 2)
    dqa = (ra * inv + r0a * shift).to(gl.bfloat16)
    dqb = (rb * inv + r0b * shift).to(gl.bfloat16)
    dq = gl.reshape(gl.join(gl.reshape(dqa, [BM, D]), gl.reshape(dqb, [BM, D])), [BM, 2 * D])
    dq = gl.convert_layout(dq, gl.DotOperandLayout(operand_index=0, parent=mmak, k_width=2))
    ds = warpgroup_mma(dq, kb.permute((1, 0)), gl.zeros([BM, 2 * K], gl.float32, mmak), use_acc=False,
      is_async=True)
    ds = warpgroup_mma_wait(0, deps=[ds, dq])[0]
    ds = ds.to(gl.bfloat16).to(gl.float32) * (D ** -.5)
    u = gl.reshape(gl.exp(ds) - 1.0, [BM, 2, K])  # (move, head of the pair, key)
    LK: gl.constexpr = u.type.layout
    kl: gl.constexpr = gl.SliceLayout(0, LK)
    kh = gl.arange(0, 2, gl.SliceLayout(1, kl))
    kk = gl.arange(0, K, gl.SliceLayout(0, kl))
    prob = gl.load(Prob + (token * H + 2 * pair + kh)[:, None] * K + kk[None, :])[None, :, :]
    z = gl.expand_dims(gl.sum(prob * u, 2), 2)
    weights = prob * u / (1.0 + z)
    orow = gl.arange(0, BM, gl.SliceLayout(1, gl.SliceLayout(2, LK)))
    optr = Out + gl.expand_dims(gl.expand_dims((row0 + orow).to(gl.int64) * (H * K), 1), 2) \
      + (((2 * pair + kh) * K)[:, None] + kk[None, :])[None, :, :]
    gl.store(optr, weights.to(gl.bfloat16))


@gluon.jit
def _query_kernel(delta_desc, wq_desc, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out, TILES,
    P: gl.constexpr, C: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, BM: gl.constexpr,
    NW: gl.constexpr, EPS_: gl.constexpr, num_warps: gl.constexpr, ABL: gl.constexpr = 0):
    BK: gl.constexpr = 64
    KC: gl.constexpr = C // BK
    HP: gl.constexpr = H // 2
    WROWS: gl.constexpr = 2 * D
    MT: gl.constexpr = P // BM
    IT: gl.constexpr = HP * KC  # ring items per tile
    gl.static_assert(num_warps == 8 and BM == 128 and NW >= 2 and 2 * K <= 256)
    G = gl.num_programs(0)
    pid = gl.program_id(0)
    my_items = ((TILES - pid + G - 1) // G) * IT

    dsm = gl.allocate_shared_memory(gl.bfloat16, [KC, BM, BK], delta_desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NW, WROWS, BK], wq_desc.layout)
    kb = gl.allocate_shared_memory(gl.bfloat16, [2 * K, WROWS],
      gl.NVMMASharedLayout.get_default_for([2 * K, WROWS], gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [KC, 1], mbarrier.MBarrierLayout())
    wbar = gl.allocate_shared_memory(gl.int64, [NW, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(KC):
        mbarrier.init(dbar.index(i), count=1)
    for i in gl.static_range(NW):
        mbarrier.init(wbar.index(i), count=1)
    for i in gl.static_range(KC):
        mbarrier.expect(dbar.index(i), BM * BK * 2)
        tma.async_copy_global_to_shared(delta_desc, [pid * BM, i * BK], dbar.index(i), dsm.index(i))
    for i in gl.static_range(NW - 1):
        _w_issue(wq_desc, wbar, wsm, i, i < my_items, KC, BK, NW, WROWS)

    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, WROWS, 16])
    mmak: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 2 * K, 16])
    # the pair's block-diagonal key matrix (2 K x 2 D), columns interleaved as w_q's rows: row 16 h + r holds key r of
    # head h at columns 2 D h + (2 i + e) = keys[.., 32 e + i]
    kl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    kr = gl.arange(0, 2 * K, gl.SliceLayout(1, kl))
    kc = gl.arange(0, WROWS, gl.SliceLayout(0, kl))
    kdiag = (kr[:, None] // K) == (kc[None, :] // D)
    kcol = (kc % D) // 2 + (kc % 2) * (D // 2)

    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        pos0 = (tile % MT) * BM
        row0 = tile * BM
        has_next = tile + G < TILES
        for pair in range(HP):
            acc = gl.zeros([BM, WROWS], gl.float32, mma)
            for j in range(KC):
                gi = it * IT + pair * KC + j  # ring item
                if pair == 0:
                    mbarrier.wait(dbar.index(j), it & 1)
                mbarrier.wait(wbar.index(gi % NW), (gi // NW) & 1)
                if ABL != 2:  # ablation 2: no GEMM
                    acc = warpgroup_mma(dsm.index(j), wsm.index(gi % NW).permute((1, 0)), acc, is_async=True)
                    acc = warpgroup_mma_wait(1, deps=[acc])
                gl.thread_barrier()  # both warpgroups are done with item gi - 1 (and delta chunk j - 1)
                _w_issue(wq_desc, wbar, wsm, gi + NW - 1, gi + NW - 1 < my_items, KC, BK, NW, WROWS)
                if pair == HP - 1:
                    if j > 0:  # the next tile's delta chunk j - 1
                        mbarrier.expect(dbar.index(j - 1), BM * BK * 2, pred=has_next)
                        tma.async_copy_global_to_shared(delta_desc, [(tile + G) * BM, (j - 1) * BK],
                          dbar.index(j - 1), dsm.index(j - 1), pred=has_next)
            acc = warpgroup_mma_wait(0, deps=[acc])
            # the pair's key matrix (every warp is done with the previous pair's key wgmma: waited at once)
            kv = gl.load(Keys + ((token * H + 2 * pair + kr // K) * K + kr % K)[:, None] * D + kcol[None, :],
              mask=kdiag, other=0.0)
            gl.thread_barrier()
            kb.store(kv)
            if pair == HP - 1:
                mbarrier.expect(dbar.index(KC - 1), BM * BK * 2, pred=has_next)
                tma.async_copy_global_to_shared(delta_desc, [(tile + G) * BM, (KC - 1) * BK], dbar.index(KC - 1),
                  dsm.index(KC - 1), pred=has_next)
            fence_async_shared()
            gl.thread_barrier()
            if ABL == 1:  # ablation: no epilogue (and no GEMM: its result unused)
                acc = acc
            elif ABL == 3:  # ablation: no epilogue, the GEMM kept (its sum stored when never true)
                if token < 0:
                    gl.store(Out + gl.arange(0, BM, gl.SliceLayout(1, mma)), gl.sum(acc, 1).to(gl.bfloat16))
            else:
                _replay_pair(acc, kb, token, pair, row0, pos0, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out, P, H, D, K, BM,
                  EPS_, num_warps, mmak)
        it += 1
    for i in gl.static_range(KC):
        mbarrier.invalidate(dbar.index(i))
    for i in gl.static_range(NW):
        mbarrier.invalidate(wbar.index(i))


def interleave_wq(w_q, heads):
    """w_q's rows per head reordered a0 b0 a1 b1 ... (the RoPE halves interleaved), bf16."""
    c = w_q.shape[0]
    return w_q.to(torch.bfloat16).view(heads, 2, c // heads // 2, -1).transpose(1, 2).reshape(c, -1).contiguous()


@torch.library.custom_op('visit_kernel::fused_query_replay_gluon', mutates_args=())
def fused_query_replay_gluon(delta: torch.Tensor, w_q: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor,
    sin: torch.Tensor, inv: torch.Tensor, shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """fused_query_weights(delta @ w_q.T, r0, cos, sin, inv, shift, keys, prob): delta (n, P, C) bf16, w_q (C, C);
    r0 (n, H, D); cos / sin (n, D / 2); inv / shift (n, P); keys (n, H, K, D) bf16; prob (n, H, K) ->
    weights (n, P, H * K) bf16 (query_replay.fused_query_replay's API)."""
    n, p, c = delta.shape
    _, heads, count, width = keys.shape
    cfg = dict(CONFIG)
    bm = cfg.pop('BM')
    assert p % bm == 0 and heads % 2 == 0 and heads * width == c and width == 64 and c % 64 == 0
    out = torch.empty((n, p, heads * count), device=delta.device, dtype=torch.bfloat16)
    f32 = lambda z: z.float().contiguous()
    delta_desc = TensorDescriptor.from_tensor(delta.contiguous().view(n * p, c), [bm, 64],
      gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16))
    wq_desc = TensorDescriptor.from_tensor(interleave_wq(w_q, heads), [2 * width, 64],
      gl.NVMMASharedLayout.get_default_for([2 * width, 64], gl.bfloat16))
    dev = delta.device.index
    if dev not in _SMS:
        _SMS[dev] = (132 if delta.device.type == 'cpu'
          else torch.cuda.get_device_properties(delta.device).multi_processor_count)
    tiles = n * p // bm
    _query_kernel[(min(tiles, _SMS[dev]),)](delta_desc, wq_desc, f32(r0), f32(cos), f32(sin), f32(inv), f32(shift),
      keys.contiguous(), f32(prob), out, tiles, p, c, heads, width, count, BM=bm, EPS_=EPS, **cfg)
    return out


@fused_query_replay_gluon.register_fake
def _(delta, w_q, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((*delta.shape[:2], keys.shape[1] * keys.shape[2]), device=delta.device, dtype=torch.bfloat16)
