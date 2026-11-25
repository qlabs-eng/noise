"""The sampled queries' plan: the measurements split between paired blocks on the same attention window, and each
64-position group's draw (a fixed-size exchange of two offset systematic samples) with its inverse probabilities."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _larger_index(a, b):
    return tl.maximum(a, b)


@triton.jit
def _trail(Labels, batch_row, LABEL_ROW: tl.constexpr, group, T: tl.constexpr,
           first, second, n, seed, counter, slot):
    lane = tl.arange(0, 64)
    pos = tl.where((lane & 1) == 0, tl.gather(first, lane >> 1, 0), tl.gather(second, lane >> 1, 0))
    local = group * 64 + pos
    labels = Labels + batch_row * LABEL_ROW
    label = tl.load(labels + local).to(tl.int64)
    following = tl.load(labels + tl.minimum(local + 1, T - 1)).to(tl.int64)
    key = (((label * 4096 + following) * 64 + pos) * 64) + lane.to(tl.int64)
    ordered = tl.sort(tl.where(lane < 2 * n, key, (1 << 42) + lane.to(tl.int64)), descending=False)
    endpoint = (ordered & 63).to(tl.int32)
    label = ordered >> 24
    prev = tl.maximum(lane - 1, 0)
    starts = (lane == 0) | (label != tl.gather(label, prev, 0))
    begin = tl.associative_scan(tl.where(starts, lane, 0), 0, _larger_index)
    pair = tl.where(((lane - begin) & 1) == 0, lane + 1, lane - 1)
    bounded = tl.minimum(tl.maximum(pair, 0), 63)
    matched = (lane < 2 * n) & (pair < 2 * n) & (label == tl.gather(label, bounded, 0))
    neighbor = tl.where(matched, tl.gather(endpoint, bounded, 0), 64)
    peer = tl.sort(endpoint * 128 + neighbor, descending=False) & 127

    across = tl.gather(peer, lane ^ 1, 0)
    nxt = tl.where(across < 64, across, lane)
    representative = lane
    for _ in tl.static_range(5):
        representative = tl.minimum(representative, tl.gather(representative, nxt, 0))
        nxt = tl.gather(nxt, nxt, 0)
    forward = tl.gather(representative, 2 * slot, 0)
    backward = tl.gather(representative, 2 * slot + 1, 0) ^ 1
    root = tl.minimum(forward, backward)
    word = tl.randint(seed, 0x20000 + counter * 32 + (root >> 1).to(tl.uint32))
    return ((word.to(tl.uint32) & 1) != 0) ^ ((root & 1) != 0)


@triton.jit
def draw(mass, order, start, seed, counter, Center, row, group, T: tl.constexpr, keep,
         Labels, batch_row, LABEL_ROW: tl.constexpr):
    token = tl.arange(0, 64)
    slot = tl.arange(0, 32)
    unit: tl.constexpr = 4294967296
    ordered_mass = tl.gather(mass, order, 0)
    lo = tl.cumsum(ordered_mass, 0) - ordered_mass
    phase = (start - lo) & (unit - 1)
    left = phase < ordered_mass
    right = ((phase + unit // 2) & (unit - 1)) < ordered_mass
    both = left & right
    n = tl.sum((left & ~right).to(tl.int32), 0)
    center = tl.load(Center + row * T + group * 64 + order)
    guide = (center.to(tl.float64) * (unit / ordered_mass.to(tl.float64))).to(tl.float32)
    bits = guide.to(tl.uint32, bitcast=True)
    sortable = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)
    key = (sortable.to(tl.int64) << 6) | order.to(tl.int64)
    absent: tl.constexpr = 1 << 42
    a = tl.sort(tl.where(left & ~right, key, absent + token.to(tl.int64)), descending=False)
    b = tl.sort(tl.where(right & ~left, key, absent + token.to(tl.int64)), descending=False)
    common = tl.sort(tl.where(both, order, 64 + token), descending=False)
    bslot = tl.maximum(0, n - 1 - slot)
    bslot = tl.minimum(bslot, 31)
    first = tl.gather(a, slot, 0).to(tl.int32) & 63
    second = tl.gather(b, bslot, 0).to(tl.int32) & 63
    coin = _trail(Labels, batch_row, LABEL_ROW, group, T, first, second, n, seed, counter, slot)
    winner = tl.where(coin, first, second)
    ids = tl.where(slot < n, winner, tl.gather(common, tl.maximum(0, slot - n), 0))
    ids = tl.sort(tl.where(slot < keep, ids, 64), descending=False)
    return tl.minimum(ids, 63)


@triton.jit
def _support(Strength, Square, pos):
    variance = tl.load(Strength + pos).to(tl.float64)
    square = tl.load(Square + pos).to(tl.float64)
    clean = tl.where((variance >= 0) & (variance <= 3.402823466e38) &
                     (square >= 0) & (square <= 3.402823466e38), variance, 0.)
    a = libdevice.sqrt(clean / tl.maximum(square, 1.e-20))
    total = tl.sum(a, 0)
    energy = tl.sum(a * a, 0)
    return tl.where(energy > 0, total * total / energy, 0.)


@triton.jit
def _assign(Strength, Square, Address, Owner, T: tl.constexpr, UNITS: tl.constexpr,
            LAYERS: tl.constexpr, PANEL: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr):
    pair, unit, panel = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pairs: tl.constexpr = LAYERS // 2
    panels: tl.constexpr = T // PANEL
    per_tile: tl.constexpr = 4096 // PANEL
    slots: tl.constexpr = (LOW + HIGH) // per_tile
    per_panel: tl.constexpr = UNITS * (pairs * slots + (LAYERS % 2) * (32 // per_tile))
    if pair < pairs:
        first = pair * UNITS + unit
        second = (pair + (LAYERS + 1) // 2) * UNITS + unit
        lanes = panel * PANEL + tl.arange(0, PANEL)
        first_score = _support(Strength, Square, first * T + lanes)
        second_score = _support(Strength, Square, second * T + lanes)
        first_high = first_score >= second_score
        first_keep = tl.where(first_high, HIGH, LOW)
        second_keep = LOW + HIGH - first_keep
        start = panel * per_panel + (pair * UNITS + unit) * slots
        split = first_keep // per_tile
        tl.store(Address + first * panels + panel, start * 64 + first_keep)
        tl.store(Address + second * panels + panel, (start + split) * 64 + second_keep)
        j = tl.arange(0, triton.next_power_of_2(slots))
        is_first = j < split
        row = tl.where(is_first, first, second)
        quota = tl.where(is_first, first_keep, second_keep)
        span = tl.where(is_first, j, j - split)
        left = panel * (PANEL // 64) + (span * 64) // quota
        right = panel * (PANEL // 64) + (span * 64 + 63) // quota
        tl.store(Owner + start + j, (row * 64 + left) * 64 + right, j < slots)
    else:
        middle = pairs * UNITS + unit
        start = panel * per_panel + pairs * UNITS * slots + unit * (32 // per_tile)
        tl.store(Address + middle * panels + panel, start * 64 + 32)
        single_tiles = tl.arange(0, 32 // per_tile)
        single_left = panel * (PANEL // 64) + 2 * single_tiles
        tl.store(Owner + start + single_tiles, (middle * 64 + single_left) * 64 + single_left + 1)


@triton.jit
def _plan(Strength, Seeds, Map, Address, Position, Inverse, T: tl.constexpr, H: tl.constexpr, PANEL: tl.constexpr,
          Labels, ROWS: tl.constexpr, LABEL_ROW: tl.constexpr, Center):
    row, group = tl.program_id(0), tl.program_id(1)
    code = tl.load(Address + row * (T // PANEL) + group // (PANEL // 64))
    keep = code % 64
    base = (code // 64) * 64 + (group % (PANEL // 64)) * keep
    p = tl.arange(0, 64)
    x = tl.load(Strength + row * T + group * 64 + p).to(tl.float64)
    finite = (x == x) & (tl.abs(x) <= 3.4028234663852886e38)
    clean = tl.where(finite, tl.maximum(x, 0.), 0.)
    biggest = tl.max(clean, 0)
    invalid = (tl.sum((~finite).to(tl.int32), 0) != 0) | (biggest <= 0.)
    weight = libdevice.sqrt(clean / tl.where(biggest > 0., biggest, 1.))
    weight += (1.490116119384765625e-8) * tl.sum(weight, 0)
    weight = tl.where(invalid, 1., weight)
    cap = tl.full((64,), False, tl.int1)
    for fixed_pass in tl.static_range(6):
        scale = (keep - 2. - .96875 * tl.sum(cap.to(tl.int32), 0)) / tl.sum(tl.where(cap, 0., weight), 0)
        cap = cap | (scale * weight >= .96875)
    scale = (keep - 2. - .96875 * tl.sum(cap.to(tl.int32), 0)) / tl.sum(tl.where(cap, 0., weight), 0)
    # Rare multi-scale strata may still need additional active-set passes.
    if tl.sum(((~cap) & (scale * weight > .96875)).to(tl.int32), 0) > 0:
        for extra_pass in range(32):
            cap = cap | (scale * weight >= .96875)
            scale = (keep - 2. - .96875 * tl.sum(cap.to(tl.int32), 0)) / tl.sum(tl.where(cap, 0., weight), 0)
    pi = .03125 + tl.where(cap, .96875, scale * weight)
    unit: tl.constexpr = 4294967296
    mass = tl.minimum(tl.maximum((pi * unit).to(tl.int64), unit // 32), unit)
    deficit = keep.to(tl.int64) * unit - tl.sum(mass, 0)
    eligible = mass < unit
    bad = (deficit < 0) | (deficit > tl.sum(eligible.to(tl.int32), 0))
    mass += (eligible & (tl.cumsum(eligible.to(tl.int32), 0) <= deficit)).to(tl.int64)
    mass = tl.where(bad | invalid, keep.to(tl.int64) * (unit // 64), mass)
    seed = tl.load(Seeds + tl.load(Map + row // H))
    counter = ((row % H) * (T // 64) + group).to(tl.uint32)
    flip = (tl.randint(seed, 2 * counter).to(tl.uint32) >> (p // 2)) & 1
    start = tl.randint(seed, 2 * counter + 1).to(tl.uint32).to(tl.int64)
    order = p ^ flip.to(tl.int32)
    j = tl.arange(0, 32)
    ids = draw(mass, order, start, seed, counter, Center, row, group, T, keep, Labels, (row // H) % ROWS, LABEL_ROW)
    denom = tl.gather(mass, ids, 0)
    tl.store(Position + base + j, group * 64 + ids, j < keep)
    tl.store(Inverse + base + j, (unit / denom.to(tl.float64)).to(tl.float32), j < keep)


def plan(strength, square, seeds, maps, heads, batch_rows, low, high, panel, center, labels):
    rows, t = strength.shape
    units = batch_rows * heads
    layers = rows // units
    per_tile = 4096 // panel
    panels = t // panel
    tiles = (layers // 2 * ((low + high) // per_tile) + layers % 2 * (32 // per_tile)) * units * panels
    address = torch.empty((rows, panels), dtype=torch.int32, device=strength.device)
    owner = torch.empty(tiles, dtype=torch.int32, device=strength.device)
    pos = torch.empty((tiles, 64), dtype=torch.int32, device=strength.device)
    inv = torch.empty_like(pos, dtype=torch.float32)
    _assign[((layers + 1) // 2, units, panels)](strength, square, address, owner, t, units, layers,
                                              panel, low, high, num_warps=4, enable_fp_fusion=False)
    _plan[(rows, t // 64)](strength, seeds, maps, address, pos, inv, t, heads, panel,
                         labels, batch_rows, labels.stride(0), Center=center, num_warps=4, enable_fp_fusion=False)
    return owner, pos, inv
