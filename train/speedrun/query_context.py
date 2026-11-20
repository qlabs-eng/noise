import torch
import triton
import triton.language as tl
from moment_kernels import IMPLICIT_CONFIG, PREC

@triton.jit
def _block_stats(q, hub, K, Values, base, query, block, maximum, total, score_sum, square_sum, Window: tl.constexpr,
                    BN: tl.constexpr, PREC: tl.constexpr, MASKED: tl.constexpr):
    """One key block of the context pass's online softmax; MASKED for blocks the causal window cuts."""
    dims = tl.arange(0, 64)
    source = block * BN + tl.arange(0, BN)
    k = tl.load(K + base + source[:, None] * 64 + dims[None, :])
    v = tl.load(Values + base + source[:, None] * 64 + dims[None, :])
    logits = tl.dot(q, tl.trans(k), input_precision=PREC) * 0.125
    scores = tl.dot(hub, tl.trans(v), input_precision=PREC)
    if MASKED:
        allowed = (source[None, :] <= query[:, None]) & (query[:, None] - source[None, :] <= Window)
        logits = tl.where(allowed, logits, float('-inf'))
    next_max = tl.maximum(maximum, tl.max(logits, 1))
    safe_max = next_max
    if MASKED:
        safe_max = tl.where(next_max == float('-inf'), 0.0, next_max)
    rescale = tl.exp(maximum - safe_max)
    prob = tl.exp(logits - safe_max[:, None])
    total = total * rescale + tl.sum(prob, 1)
    score_sum = score_sum * rescale + tl.sum(prob * scores, 1)
    square_sum = square_sum * rescale + tl.sum(prob * scores * scores, 1)
    return next_max, total, score_sum, square_sum


@triton.jit
def _context_stats(Q, K, Values, Hub, LSE, Center, Variance, T: tl.constexpr, Window: tl.constexpr,
                                   B: tl.constexpr, PREC: tl.constexpr):
    """The lse / center / score-variance pass for square tiles (B queries, B keys; Window % B == 0): only the diagonal
    block and the window's first block are masked, the blocks between are whole; the heaviest query tiles launch
    first."""
    tile, head = T // B - 1 - tl.program_id(0), tl.program_id(1)
    query = tile * B + tl.arange(0, B)
    dims = tl.arange(0, 64)
    base = head * T * 64
    q = tl.load(Q + base + query[:, None] * 64 + dims[None, :])
    hub = tl.load(Hub + base + query[:, None] * 64 + dims[None, :])
    maximum = tl.full((B,), float('-inf'), tl.float32)
    total = tl.zeros((B,), tl.float32)
    score_sum = tl.zeros((B,), tl.float32)
    square_sum = tl.zeros((B,), tl.float32)
    maximum, total, score_sum, square_sum = _block_stats(q, hub, K, Values, base, query, tile, maximum, total,
        score_sum, square_sum,
                                                Window, B, PREC, True)
    start = tile * B - Window
    lo = 0
    if start >= 0:
        lo = start // B
        maximum, total, score_sum, square_sum = _block_stats(q, hub, K, Values, base, query, lo, maximum, total,
            score_sum, square_sum,
                                                    Window, B, PREC, True)
        lo += 1
    for block in range(lo, tile):
        maximum, total, score_sum, square_sum = _block_stats(q, hub, K, Values, base, query, block, maximum, total,
            score_sum, square_sum,
                                                    Window, B, PREC, False)
    tl.store(LSE + head * T + query, maximum + tl.log(total))
    tl.store(Center + head * T + query, score_sum / total)


    center = score_sum / total
    tl.store(Variance + head * T + query, tl.maximum(square_sum / total - center * center, 0.))

def context(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, hub: torch.Tensor,
    window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, t, _ = q.shape
    q, k, values, hub = (x.contiguous() for x in (q, k, values, hub))
    lse = torch.empty((h, t), device=q.device, dtype=torch.float32)
    center = torch.empty_like(lse)
    variance = torch.empty_like(lse)
    c = IMPLICIT_CONFIG
    assert c['BM'] == c['BN'] and t % c['BM'] == 0 and window % c['BN'] == 0
    _context_stats[(t // c['BM'], h)](q, k, values, hub, lse, center, variance, T=t, Window=window, B=c['BM'],
        PREC=PREC,
        num_warps=c['num_warps'], num_stages=c['num_stages'])
    return lse, center, variance
