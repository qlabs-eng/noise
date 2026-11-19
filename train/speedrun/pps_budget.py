"""The sampled query moments' token plan: for each (layer, row, head) and each group of consecutive tokens, a fixed
number of token positions and their inverse inclusion probabilities, drawn from the tokens' strengths by a
counter-based hash of the seeds.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


REGULARIZATION = 2.0 ** -20


@triton.jit
def _plan_kernel(strength, seeds, maps, positions, inverse, T: tl.constexpr, H: tl.constexpr,
                 REG: tl.constexpr, GP: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1) * GP + tl.arange(0, GP)
    token = tl.arange(0, N)
    valid_group = group < T // N
    x = tl.load(strength + row * T + group[:, None] * N + token[None, :],
                valid_group[:, None], other=0.0).to(tl.float64)
    finite = (x == x) & (tl.abs(x) <= 3.4028234663852886e38)
    clean = tl.where(finite, tl.maximum(x, 0.0), 0.0)
    largest = tl.max(clean, axis=1)
    invalid = (tl.sum((~finite).to(tl.int32), axis=1) != 0) | (largest <= 0.0)
    normalized = clean / tl.where(largest > 0.0, largest, 1.0)[:, None]
    weight = libdevice.sqrt(normalized)
    weight += (REG / N) * tl.sum(weight, axis=1)[:, None]
    weight = tl.where(invalid[:, None], 1.0, weight)

    # pi = 1/32 + min(lambda * weight, 31/32), sum(pi) = K.
    # At most floor((K-N/32)/(31/32)) entries saturate.
    saturated = tl.full((GP, N), False, tl.int1)
    for _ in tl.static_range((32 * K - N) // 31):
        count = tl.sum(saturated.to(tl.int32), axis=1).to(tl.float64)
        remaining = (K - N / 32.0) - 0.96875 * count
        active_sum = tl.sum(tl.where(saturated, 0.0, weight), axis=1)
        lam = remaining / active_sum
        saturated = saturated | (lam[:, None] * weight >= 0.96875)
    count = tl.sum(saturated.to(tl.int32), axis=1).to(tl.float64)
    remaining = (K - N / 32.0) - 0.96875 * count
    active_sum = tl.sum(tl.where(saturated, 0.0, weight), axis=1)
    lam = remaining / active_sum
    pi = 0.03125 + tl.where(saturated, 0.96875, lam[:, None] * weight)
    pi = tl.minimum(tl.maximum(pi, 0.03125), 1.0)

    # Positive float-to-int conversion truncates, hence floors. Repair the
    # total in original token order before applying random pair orientations.
    unit: tl.constexpr = 4294967296
    mass = (pi * unit).to(tl.int64)
    mass = tl.minimum(tl.maximum(mass, unit // 32), unit)
    deficit = tl.full((GP,), K * unit, tl.int64) - tl.sum(mass, axis=1)
    eligible = mass < unit
    eligible_rank = tl.cumsum(eligible.to(tl.int32), axis=1)
    eligible_count = tl.sum(eligible.to(tl.int32), axis=1)
    bad_rounding = (deficit < 0) | (deficit > eligible_count)
    mass += (eligible & (eligible_rank <= deficit[:, None])).to(tl.int64)
    fallback = invalid | bad_rounding
    mass = tl.where(fallback[:, None], K * (unit // N), mass)

    layer = tl.load(maps + row // H)
    seed = tl.load(seeds + layer)
    counter = ((row % H) * (T // N) + group).to(tl.uint32)
    orientation = tl.randint(seed, 2 * counter).to(tl.uint32) & ((1 << (N // 2)) - 1)
    start = tl.randint(seed, 2 * counter + 1).to(tl.uint32).to(tl.int64)
    flip = (orientation[:, None] >> (token[None, :] // 2)) & 1
    order = token[None, :] ^ flip.to(tl.int32)
    ordered_mass = tl.gather(mass, order, axis=1)
    hi = tl.cumsum(ordered_mass, axis=1)
    slot = tl.arange(0, BK)
    # Padded lanes use an in-range threshold and never store.
    threshold = start[:, None] + tl.where(slot < K, slot, 0)[None, :].to(tl.int64) * unit
    low = tl.full((GP, BK), 0, tl.int32)
    high = tl.full((GP, BK), N - 1, tl.int32)
    for _ in tl.static_range(4 if N == 16 else 6):
        middle = (low + high) // 2
        endpoint = tl.gather(hi, middle, axis=1)
        below = endpoint <= threshold
        low = tl.where(below, middle + 1, low)
        high = tl.where(below, high, middle)
    picked = tl.gather(order, low, axis=1)
    picked_mass = tl.gather(ordered_mass, low, axis=1)
    out_slot = row * (T // N * K) + group[:, None] * K + slot[None, :]
    out_mask = valid_group[:, None] & (slot[None, :] < K)
    tl.store(positions + out_slot, group[:, None] * N + picked, out_mask)
    tl.store(inverse + out_slot, (unit / picked_mass.to(tl.float64)).to(tl.float32),
             out_mask)


def importance_plan(strength, seeds, maps, heads, group_size=64, keep=None):
    """Return group-major indices and inverses, each [rows,T/group_size*keep].

    Maps entries must index seeds. One seed belongs to each mapped layer/row;
    heads receive disjoint counters. No PyTorch RNG draw is made.
    With keep = 3/8 of the group, output tiles can straddle groups; scan bounds include both.
    """
    keep = group_size // 2 if keep is None else keep
    assert group_size in (16, 64)
    assert keep in (group_size // 2, 3 * group_size // 8)
    assert strength.ndim == 2 and strength.dtype == torch.float32
    rows, t = strength.shape
    assert rows > 0 and heads > 0 and rows % heads == 0
    # A response tile may straddle sampling groups; its causal scan uses
    # the first and last covered group. The final output tile must be full.
    assert t > 0 and t % group_size == 0 and (t // group_size * keep) % 64 == 0
    assert 2 * heads * (t // group_size) < (1 << 32)
    assert maps.ndim == 1 and maps.numel() == rows // heads
    assert seeds.ndim == 1 and seeds.dtype == torch.int32
    assert maps.dtype in (torch.int32, torch.int64)
    assert strength.is_cuda and seeds.device == strength.device == maps.device
    assert strength.is_contiguous() and seeds.is_contiguous() and maps.is_contiguous()
    positions = torch.empty((rows, t // group_size * keep), device=strength.device, dtype=torch.int32)
    inverse = torch.empty((rows, t // group_size * keep), device=strength.device, dtype=torch.float32)
    gp = 4 if group_size == 16 else 1
    _plan_kernel[(rows, triton.cdiv(t // group_size, gp))](
        strength, seeds, maps, positions, inverse,
        T=t, H=heads, REG=REGULARIZATION, GP=gp, N=group_size, K=keep, BK=triton.next_power_of_2(keep),
        num_warps=4, enable_fp_fusion=False)
    return positions, inverse

