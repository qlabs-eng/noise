"""A fixed-count draw of tokens with probability proportional to size (PPS), in fp64."""
import torch
import triton
import triton.language as tl

@triton.jit
def _plan(Size, Offset, Inside, Boost, T: tl.constexpr, K: tl.constexpr,
          Floor: tl.constexpr, BK: tl.constexpr):
    token = tl.arange(0, T)
    size = tl.load(Size + token).to(tl.float64)
    total = tl.sum(size, 0)
    floor64 = tl.full((), Floor, tl.float64)
    share = floor64 / T + (1.0 - floor64) * size / total
    pi = tl.minimum(K * share, 1.0)
    for _ in tl.static_range(6):
        capped = pi >= 1.0
        rest = tl.where(capped, 0.0, share)
        pi = tl.minimum(tl.where(capped, 1.0, share * (K - tl.sum(capped.to(tl.int64), 0)) / tl.sum(rest, 0)), 1.0)
    pi = pi * (K / tl.sum(pi, 0))
    pi = tl.where((total > 0.0) & (total < float('inf')), pi, float('nan'))
    cdf = tl.cumsum(pi, 0)
    slot = tl.arange(0, BK)
    target = tl.load(Offset) + slot.to(tl.float64)
    low = tl.full((BK,), 0, tl.int32)
    high = tl.full((BK,), T, tl.int32)
    for _ in tl.static_range(12):
        middle = (low + high) // 2
        value = tl.gather(cdf, tl.minimum(middle, T - 1), axis=0)
        below = (middle < T) & (value < target)
        low = tl.where(below, middle + 1, low)
        high = tl.where(below, high, middle)
    index = tl.minimum(low, T - 1)
    weight = (1.0 / tl.gather(pi, index, axis=0)).to(tl.float32)
    tl.store(Inside + slot, index.to(tl.int64), slot < K)
    tl.store(Boost + slot, weight, slot < K)

def plan(size: torch.Tensor, offset: torch.Tensor, count: int, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`count` of the 2,048 tokens drawn with probability in proportion to floor / T + (1 - floor) * size / sum(size)
    (capped at 1 and rescaled), by systematic sampling from `offset`: their indices and inverse probabilities."""
    assert size.shape == (2048,) and size.dtype == torch.float64 and size.is_contiguous()
    assert offset.shape == (1,) and offset.dtype == torch.float64 and offset.device == size.device
    assert 0 < count <= 2048
    inside = torch.empty((count,), device=size.device, dtype=torch.int64)
    boost = torch.empty((count, 1), device=size.device, dtype=torch.float32)
    _plan[(1,)](size, offset, inside, boost, T=size.numel(), K=count, Floor=floor,
        BK=triton.next_power_of_2(count), num_warps=8, enable_fp_fusion=False)
    return inside, boost
