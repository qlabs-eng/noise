"""The routed query replay in one kernel: per (token, BM moves) the moved raw queries raw = bf16(delta @ w_q^T) are
formed per head on the tensor cores and go straight into the replay (RoPE, norm folding, the K-key response weights),
numerically fused_query_weights(delta @ w_q.T, ...) without the raw round trip. Triton (not Gluon)."""
import torch
import triton
import triton.language as tl

EPS = torch.finfo(torch.float32).eps
# measured best of ~20 (BM 64-256, 2 or 4 heads per CTA, K-steps 32-128, 4-16 warps, 2-5 stages); WIDE: one
# N = 2 D dot per K-step for the 2 heads (-5% vs two N = D dots); slower: INTERLEAVE (RoPE halves interleaved by
# permuting w_q's rows), 4 heads in one N = 4 D dot, a persistent tile loop (with or without Triton's warp
# specialization: +40%)
CONFIG = dict(BM=128, HB=2, BKK=64, num_warps=8, num_stages=3, WIDE=True)


@triton.jit
def _replay_head(acc, head, token, rows, c, s, g, shift0, R0, Keys, Prob, Out,
    H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, EPS_: tl.constexpr,
    INTERLEAVE: tl.constexpr):
    """The reference's replay epilogue for one head of the tile (acc: the fp32 raw query, BM x D; INTERLEAVE: its
    columns are the RoPE halves interleaved, a0 b0 a1 b1 ..., from row-permuted w_q, so the halves split in
    registers)."""
    hh = tl.arange(0, D // 2)
    dh = tl.arange(0, D)
    rr = tl.arange(0, K)
    raw = acc.to(tl.bfloat16).to(tl.float32)
    if INTERLEAVE:
        a_, b_ = tl.split(tl.reshape(raw, (BM, D // 2, 2)))
    else:
        a_, b_ = tl.split(tl.permute(tl.reshape(raw, (BM, 2, D // 2)), (0, 2, 1)))
    r0a = tl.load(R0 + (token * H + head) * D + hh)
    r0b = tl.load(R0 + (token * H + head) * D + D // 2 + hh)
    ra = a_ * c[None, :] + b_ * s[None, :]
    rb = -a_ * s[None, :] + b_ * c[None, :]
    ra, rb = ra * g + shift0 * r0a[None, :], rb * g + shift0 * r0b[None, :]
    sq = tl.sum(r0a * r0a, 0) + tl.sum(r0b * r0b, 0)
    dots = tl.sum(ra * r0a[None, :], 1) + tl.sum(rb * r0b[None, :], 1)
    moved = tl.sum(ra * ra, 1) + tl.sum(rb * rb, 1)
    inv0 = tl.math.rsqrt(sq / D + EPS_)
    inv = tl.math.rsqrt((sq + 2 * dots + moved) / D + EPS_)
    shift = (inv - inv0)[:, None]
    dqa = (ra * inv[:, None] + r0a[None, :] * shift).to(tl.bfloat16)
    dqb = (rb * inv[:, None] + r0b[None, :] * shift).to(tl.bfloat16)
    if INTERLEAVE:  # dq and the keys' columns both interleaved
        dq = tl.reshape(tl.join(dqa, dqb), (BM, D))
        kcol = (dh % 2) * (D // 2) + dh // 2
    else:
        dq = tl.reshape(tl.trans(tl.join(dqa, dqb), (0, 2, 1)), (BM, D))
        kcol = dh
    keys = tl.load(Keys + ((token * H + head) * K + rr[:, None]) * D + kcol[None, :])
    prob = tl.load(Prob + (token * H + head) * K + rr)
    ds = tl.dot(dq, tl.trans(keys)).to(tl.bfloat16).to(tl.float32) * (D ** -.5)
    u = tl.exp(ds) - 1.0
    z = tl.sum(prob[None, :] * u, 1)
    weights = prob[None, :] * u / (1.0 + z[:, None])
    tl.store(Out + (rows[:, None] * H + head) * K + rr[None, :], weights.to(tl.bfloat16))


@triton.jit
def _query_replay_kernel(Delta, Wq, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out,
    P: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, HB: tl.constexpr, BKK: tl.constexpr, EPS_: tl.constexpr,
    INTERLEAVE: tl.constexpr = False, WIDE: tl.constexpr = False):
    """One CTA: BM moves of one token x HB (2 or 4) heads. raw_h = delta_tile @ w_q[head h]^T (C = model width,
    K-loop in BKK columns, the delta tile shared by the HB heads), then per head the reference's replay epilogue."""
    pid = tl.program_id(0)
    groups: tl.constexpr = H // HB
    hg = pid % groups  # the head groups of one move tile are neighbours in launch order: the tile stays in L2
    row0 = (pid // groups) * BM
    token = row0 // P
    pp = row0 % P + tl.arange(0, BM)
    rows = row0 + tl.arange(0, BM)
    dh = tl.arange(0, D)
    kk = tl.arange(0, BKK)
    acc0 = tl.zeros((BM, D), tl.float32)
    acc1 = tl.zeros((BM, D), tl.float32)
    acc2 = tl.zeros((BM, D), tl.float32)
    acc3 = tl.zeros((BM, D), tl.float32)
    wrow = (hg * HB * D + dh)[:, None] * C
    if WIDE:  # one N = HB D dot per K-step, the heads split after the loop
        acc = tl.zeros((BM, HB * D), tl.float32)
        wrow2 = (hg * HB * D + tl.arange(0, HB * D))[:, None] * C
        for k0 in range(0, C, BKK):
            a = tl.load(Delta + rows[:, None] * C + (k0 + kk)[None, :])
            acc = tl.dot(a, tl.trans(tl.load(Wq + wrow2 + (k0 + kk)[None, :])), acc)
        if HB == 4:
            lo, hi = tl.split(tl.permute(tl.reshape(acc, (BM, 2, 2 * D)), (0, 2, 1)))
            acc0, acc1 = tl.split(tl.permute(tl.reshape(lo, (BM, 2, D)), (0, 2, 1)))
            acc2, acc3 = tl.split(tl.permute(tl.reshape(hi, (BM, 2, D)), (0, 2, 1)))
        else:
            acc0, acc1 = tl.split(tl.permute(tl.reshape(acc, (BM, 2, D)), (0, 2, 1)))
    for k0 in range(0, C if not WIDE else 0, BKK):
        a = tl.load(Delta + rows[:, None] * C + (k0 + kk)[None, :])
        acc0 = tl.dot(a, tl.trans(tl.load(Wq + wrow + (k0 + kk)[None, :])), acc0)
        acc1 = tl.dot(a, tl.trans(tl.load(Wq + wrow + D * C + (k0 + kk)[None, :])), acc1)
        if HB == 4:
            acc2 = tl.dot(a, tl.trans(tl.load(Wq + wrow + 2 * D * C + (k0 + kk)[None, :])), acc2)
            acc3 = tl.dot(a, tl.trans(tl.load(Wq + wrow + 3 * D * C + (k0 + kk)[None, :])), acc3)
    hh = tl.arange(0, D // 2)
    c = tl.load(Cos + token * (D // 2) + hh)
    s = tl.load(Sin + token * (D // 2) + hh)
    g = tl.load(Inv + token * P + pp)[:, None]
    shift0 = tl.load(Shift + token * P + pp)[:, None]
    _replay_head(acc0, hg * HB, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, INTERLEAVE)
    _replay_head(acc1, hg * HB + 1, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, INTERLEAVE)
    if HB == 4:
        _replay_head(acc2, hg * HB + 2, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, INTERLEAVE)
        _replay_head(acc3, hg * HB + 3, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, INTERLEAVE)


@torch.library.custom_op('visit_kernel::fused_query_replay', mutates_args=())
def fused_query_replay(delta: torch.Tensor, w_q: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor,
    sin: torch.Tensor, inv: torch.Tensor, shift: torch.Tensor, keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """fused_query_weights(delta @ w_q.T, r0, cos, sin, inv, shift, keys, prob): delta (n, P, C) bf16, w_q (C, C);
    r0 (n, H, D); cos / sin (n, D / 2); inv / shift (n, P); keys (n, H, K, D) bf16; prob (n, H, K) ->
    weights (n, P, H * K) bf16."""
    n, p, cdim = delta.shape
    _, heads, count, width = keys.shape
    cfg = dict(CONFIG)
    bm = cfg.pop('BM')
    assert p % bm == 0 and cfg['HB'] in (2, 4) and heads % cfg['HB'] == 0 and heads * width == cdim
    out = torch.empty((n, p, heads * count), device=delta.device, dtype=torch.bfloat16)
    f32 = lambda z: z.float().contiguous()
    wq = w_q.to(torch.bfloat16)
    if cfg.get('INTERLEAVE', False):  # each head's rows: the RoPE halves interleaved
        wq = wq.view(heads, 2, width // 2, cdim).transpose(1, 2)
    _query_replay_kernel[(n * p // bm * (heads // cfg['HB']),)](
      delta.contiguous(), wq.contiguous(), f32(r0), f32(cos), f32(sin), f32(inv), f32(shift),
      keys.contiguous(), f32(prob), out, p, cdim, heads, width, count, BM=bm, EPS_=EPS, **cfg)
    return out


@fused_query_replay.register_fake
def _(delta, w_q, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((*delta.shape[:2], keys.shape[1] * keys.shape[2]), device=delta.device, dtype=torch.bfloat16)
