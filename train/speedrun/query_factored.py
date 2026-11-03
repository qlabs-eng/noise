"""The routed query replay from the block-4 root's factors (Triton): the root output is
delta = bf16(bf16(step * frame + bf16(change @ Wo[routes])) * scale), so its raw queries delta @ w_q^T are, up to
delta's two bf16 roundings, scale * (change @ A_t + step * F[move]) with A_t = Wo[routes_t] @ w_q^T (64 x C per token)
and F = frame @ w_q^T (P x C): a K = 64 GEMM per tile instead of K = C, and no delta read. The replay epilogue is
query_replay's (_replay_head). Not bitwise with fused_query_replay(delta, ...): it skips delta's roundings.
"""
import torch
import triton
import triton.language as tl

from query_replay import EPS, _replay_head

CONFIG = dict(BM=128, HB=2, TOK=8, num_warps=8, num_stages=3)  # TOK: tokens per CTA (0: one)


@triton.jit
def _query_factored_kernel(Change, A, F, Steps, Scale, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out,
    P: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BM: tl.constexpr, HB: tl.constexpr, EPS_: tl.constexpr, ABL: tl.constexpr = 0):
    """One CTA: BM moves of one token x HB = 2 heads: raw = scale * (change @ A_t[:, heads] + step * F[moves, heads])
    in fp32 (one N = 2 D dot, K = R), then per head the reference's replay epilogue."""
    gl_ = tl.program_id(0)
    groups: tl.constexpr = H // HB
    hg = gl_ % groups  # the head groups of one move tile are neighbours in launch order: change stays in L2
    row0 = (gl_ // groups) * BM
    token = row0 // P
    pp = row0 % P + tl.arange(0, BM)
    rows = row0 + tl.arange(0, BM)
    rk = tl.arange(0, R)
    cols = hg * HB * D + tl.arange(0, HB * D)
    change = tl.load(Change + rows[:, None] * R + rk[None, :])
    a_t = tl.load(A + (token * R + rk)[:, None] * C + cols[None, :])
    acc = tl.dot(change, a_t)
    step = tl.load(Steps + token)
    scale = tl.load(Scale).to(tl.float32)
    if ABL != 1:  # ablation 1: no frame term
        frame_q = tl.load(F + pp[:, None] * C + cols[None, :]).to(tl.float32)
        acc = scale * (acc + step * frame_q)
    if ABL == 2:  # ablation 2: no replay epilogue (one bf16 per move and head stored)
        tl.store(Out + rows * H * K + hg, tl.sum(acc, 1).to(tl.bfloat16))
        return
    acc0, acc1 = tl.split(tl.permute(tl.reshape(acc, (BM, 2, D)), (0, 2, 1)))
    hh = tl.arange(0, D // 2)
    c = tl.load(Cos + token * (D // 2) + hh)
    s = tl.load(Sin + token * (D // 2) + hh)
    g = tl.load(Inv + token * P + pp)[:, None]
    shift0 = tl.load(Shift + token * P + pp)[:, None]
    _replay_head(acc0, hg * HB, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, False)
    _replay_head(acc1, hg * HB + 1, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, False)


@triton.jit
def _query_factored_tok_kernel(Change, A, F, Steps, Scale, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out, N,
    P: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BM: tl.constexpr, HB: tl.constexpr, TOK: tl.constexpr, EPS_: tl.constexpr):
    """One CTA: BM moves x HB = 2 heads for TOK consecutive tokens; the frame term's F tile (moves x heads) is loaded
    once and kept in registers across the tokens."""
    pid = tl.program_id(0)
    groups: tl.constexpr = H // HB
    MT: tl.constexpr = P // BM
    hg = pid % groups
    mt = (pid // groups) % MT
    t0 = (pid // (groups * MT)) * TOK
    pp = mt * BM + tl.arange(0, BM)
    rk = tl.arange(0, R)
    cols = hg * HB * D + tl.arange(0, HB * D)
    frame_q = tl.load(F + pp[:, None] * C + cols[None, :])
    scale = tl.load(Scale).to(tl.float32)
    hh = tl.arange(0, D // 2)
    for token in range(t0, tl.minimum(t0 + TOK, N)):
        rows = token * P + pp
        change = tl.load(Change + rows[:, None] * R + rk[None, :])
        a_t = tl.load(A + (token * R + rk)[:, None] * C + cols[None, :])
        acc = tl.dot(change, a_t)
        step = tl.load(Steps + token)
        acc = scale * (acc + step * frame_q.to(tl.float32))
        acc0, acc1 = tl.split(tl.permute(tl.reshape(acc, (BM, 2, D)), (0, 2, 1)))
        c = tl.load(Cos + token * (D // 2) + hh)
        s = tl.load(Sin + token * (D // 2) + hh)
        g = tl.load(Inv + token * P + pp)[:, None]
        shift0 = tl.load(Shift + token * P + pp)[:, None]
        _replay_head(acc0, hg * HB, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, False)
        _replay_head(acc1, hg * HB + 1, token, rows, c, s, g, shift0, R0, Keys, Prob, Out, H, D, K, BM, EPS_, False)


@triton.jit
def _factor_a_kernel(Wo, Routes, Wq, A, M, C: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr):
    """A[rows] = Wo[routes.flat[rows]] @ w_q^T (bf16, fp32 accumulation): the gathered rows' GEMM in one pass."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    live = rows < M
    chosen = tl.load(Routes + rows, mask=live, other=0).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        a = tl.load(Wo + chosen[:, None] * D + (k0 + kk)[None, :])
        b = tl.load(Wq + cols[:, None] * D + (k0 + kk)[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(A + rows[:, None].to(tl.int64) * C + cols[None, :], acc.to(tl.bfloat16), mask=live[:, None])


FACTOR_CONFIG = dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3)


@torch.library.custom_op('visit_kernel::query_factors', mutates_args=())
def query_factors(wo: torch.Tensor, routes: torch.Tensor, basis: torch.Tensor,
    w_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A (n, R, C) bf16 = Wo[routes_t] @ w_q^T per token (fp32 accumulation) and F (P, C) bf16 = frame @ w_q^T
    (fp32; ~2 us, no need to hoist)."""
    wq = w_q.to(torch.bfloat16).contiguous()
    n, r = routes.shape
    c, d = wq.shape
    a = torch.empty((n, r, c), device=wo.device, dtype=torch.bfloat16)
    cfg = dict(FACTOR_CONFIG)
    assert c % cfg['BN'] == 0
    _factor_a_kernel[(triton.cdiv(n * r, cfg['BM']), c // cfg['BN'])](wo.contiguous(), routes.contiguous(), wq, a,
      n * r, c, d, **cfg)
    f = (basis.float() @ w_q.float().T).to(torch.bfloat16)
    return a, f


@query_factors.register_fake
def _(wo, routes, basis, w_q):
    return (torch.empty((*routes.shape, w_q.shape[0]), device=wo.device, dtype=torch.bfloat16),
      torch.empty((basis.shape[0], w_q.shape[0]), device=wo.device, dtype=torch.bfloat16))


@torch.library.custom_op('visit_kernel::fused_query_replay_factored', mutates_args=())
def fused_query_replay_factored(change: torch.Tensor, steps: torch.Tensor, scale: torch.Tensor, a: torch.Tensor,
    f: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor, shift: torch.Tensor,
    keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """fused_query_replay(delta, w_q, ...) for the root output delta, from its factors: change (n, P, R) bf16 (the
    root's routed change), steps (n, 1), scale (the root's output scale), a / f from query_factors; the rest as
    fused_query_replay -> weights (n, P, H * K) bf16."""
    n, p, r = change.shape
    _, heads, count, width = keys.shape
    cdim = heads * width
    cfg = dict(CONFIG)
    bm = cfg.pop('BM')
    assert p % bm == 0 and heads % cfg['HB'] == 0 and a.shape == (n, r, cdim) and f.shape == (p, cdim)
    out = torch.empty((n, p, heads * count), device=change.device, dtype=torch.bfloat16)
    f32 = lambda z: z.float().contiguous()
    tok = cfg.pop('TOK', 0)
    if tok:  # token loop: the F tile stays in registers
        _query_factored_tok_kernel[(triton.cdiv(n, tok) * (p // bm) * (heads // cfg['HB']),)](
          change.contiguous(), a.contiguous(), f.contiguous(), f32(steps), scale, f32(r0), f32(cos), f32(sin),
          f32(inv), f32(shift), keys.contiguous(), f32(prob), out, n, p, cdim, heads, width, count, r, BM=bm, TOK=tok,
          EPS_=EPS, **cfg)
    else:
        _query_factored_kernel[(n * p // bm * (heads // cfg['HB']),)](
          change.contiguous(), a.contiguous(), f.contiguous(), f32(steps), scale, f32(r0), f32(cos), f32(sin),
          f32(inv), f32(shift), keys.contiguous(), f32(prob), out, p, cdim, heads, width, count, r, BM=bm, EPS_=EPS,
          **cfg)
    return out


@fused_query_replay_factored.register_fake
def _(change, steps, scale, a, f, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((*change.shape[:2], keys.shape[1] * keys.shape[2]), device=change.device, dtype=torch.bfloat16)
