"""Root 4's positions: a fixed-count systematic sample along a Morton order of each row's input and target tokens."""
import torch
import triton
import triton.language as tl


@triton.jit
def _plan(Size, Offset, Order, Indices, Inverse, N: tl.constexpr, K: tl.constexpr,
          Floor: tl.constexpr, Q: tl.constexpr):
    pos = tl.arange(0, N)
    permutation = tl.load(Order + pos).to(tl.int32)
    size = tl.load(Size + pos).to(tl.float64)
    total = tl.sum(size, 0)
    share = Floor / N + (1.0 - Floor) * size / total
    share = tl.where(total > 0.0, share, 1.0 / N)
    pi = tl.minimum(K * share, 1.0)
    for _ in tl.static_range(6):
        capped = pi >= 1.0
        rest = tl.where(capped, 0.0, share)
        pi = tl.minimum(tl.where(capped, 1., share * (K - tl.sum(capped.to(tl.int64), 0)) / tl.sum(rest, 0)), 1.)
    # A deficit left by an unfinished cap cascade is spread over the remaining capacity.
    mass = tl.sum(pi, 0)
    if mass < K - 1.0e-10:
        pi += (1.0 - pi) * ((K - mass) / (N - mass))

    cumulative = tl.cumsum(tl.gather(pi, permutation, axis=0), axis=0)
    cdf = tl.floor(cumulative * Q + 0.5).to(tl.int64)
    cdf = tl.where(pos == N - 1, K * Q, cdf)
    previous = tl.where(pos == 0, 0, tl.gather(cdf, tl.maximum(pos - 1, 0), axis=0))
    ticks = tl.minimum(cdf - previous, Q)  # (at most one ticket per position)
    deficit = K * Q - tl.sum(ticks, 0)
    if deficit != 0:  # ticks cut off by that cap go to the first positions with room
        capacity = Q - ticks
        prefix = tl.cumsum(capacity, 0)
        ticks += tl.minimum(prefix, deficit) - tl.minimum(prefix - capacity, deficit)
        cdf = tl.cumsum(ticks, 0)
    shift = tl.floor(tl.load(Offset) * Q).to(tl.int64)
    # In a systematic draw, this interval contains a ticket if its first
    # phase-matching tick lies below the interval's upper endpoint.
    picked = ((shift - (cdf - ticks)) & (Q - 1)) < ticks
    key = tl.sort(tl.where(picked, permutation * N + pos, N * N), descending=False)
    mass = tl.gather(ticks, key % N, 0)
    tl.store(Indices + pos, (key // N).to(tl.int64), pos < K)
    tl.store(Inverse + pos, (Q / mass.to(tl.float64)).to(tl.float32), pos < K)


def plan(size, offset, count, floor, order):
    length = size.numel()
    inside = torch.empty(count, dtype=torch.int64, device=size.device)
    inverse = torch.empty((count, 1), dtype=torch.float32, device=size.device)
    _plan[(1,)](size, offset, order, inside, inverse, length, count, floor,
                1 << 24, num_warps=8, enable_fp_fusion=False)
    return inside, inverse


@triton.jit
def _transition_orders(Inputs, Targets, Order, BX: tl.constexpr, BY: tl.constexpr,
                       SX: tl.constexpr, SY: tl.constexpr, LANES: tl.constexpr,
                       N: tl.constexpr, V: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, N)
    source = tl.load(Inputs + (row // LANES) * BX + (row % LANES) * SX + pos).to(tl.int32)
    target = tl.load(Targets + (row // LANES) * BY + (row % LANES) * SY + pos).to(tl.int32)

    # Each coordinate is its token's midrank among the row's positions, ties included, in 64 integer bins.
    src_bins = tl.histogram(source, V)
    dst_bins = tl.histogram(target, V)
    src_mid = 2 * tl.cumsum(src_bins, axis=0) - src_bins
    dst_mid = 2 * tl.cumsum(dst_bins, axis=0) - dst_bins
    x = tl.gather(src_mid, source, axis=0) // 64
    y = tl.gather(dst_mid, target, axis=0) // 64
    key = tl.full((N,), 0, tl.int32)
    for i in tl.static_range(6):
        key |= ((x >> i) & 1) << (2 * i + 1)
        key |= ((y >> i) & 1) << (2 * i)
    physical = tl.sort(key * N + pos) % N
    tl.store(Order + row * N + pos, physical)


def transition_orders(inputs, targets):
    """Each row's positions (inputs, targets: (batches, lanes, T) views with unit last stride) in Morton order
    of the input token's and the target token's midrank among the row's tokens, on a 64-by-64 grid."""
    batches, lanes, length = inputs.shape
    order = torch.empty((batches, lanes, length), dtype=torch.int16, device=inputs.device)
    _transition_orders[(batches * lanes,)](inputs, targets, order, inputs.stride(0),
                                         targets.stride(0), inputs.stride(1), targets.stride(1),
                                         lanes, length, 4096, num_warps=8)
    return order
