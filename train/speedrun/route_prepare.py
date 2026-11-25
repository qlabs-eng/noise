import torch
import triton
import triton.language as tl

@triton.jit
def _ordered_bits(x):
    """fp32 -> int32 in the same order (-inf lowest, -0 below +0, as torch's radix select orders them)."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _route_prepare_kernel(Inputs, Radius, OutNorm, Indices, Clean, Scratch, L: tl.constexpr, ROWS: tl.constexpr,
    N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                              CAP: tl.constexpr, SCALE: tl.constexpr, RowIds=None, PartnerIds=None,
                              T: tl.constexpr = 0):
    """torch.topk(scores, K).indices of one row, exactly: floor = the K-th largest of the N / G group maxima (count(key
    >= floor) >= K, so every top-K key is >= it); the candidates key >= floor, if at most CAP (else the binary search
    from the floor), compacted and top-K'd as (key, N - 1 - col) (values descending, equal values by ascending
    index)."""
    row = tl.program_id(0).to(tl.int64)
    source_row = row
    if PartnerIds is not None:
        if row < ROWS // 2:
            source_row = tl.load(RowIds + row).to(tl.int64)
        else:
            source_row = tl.load(PartnerIds + row - ROWS // 2).to(tl.int64) + T
    NG: tl.constexpr = N // G
    cols = tl.arange(0, NG)[:, None] * G + tl.arange(0, G)[None, :]
    layer = tl.program_id(1)
    radius = tl.load(Radius + layer * N + cols) * SCALE
    out_norm = tl.load(OutNorm + layer * N + cols)
    h = tl.full((NG, G), 0, tl.float32)
    for j in tl.static_range(L):
        if layer == j:
            h = tl.load(Inputs[j] + source_row * N + cols).to(tl.float32)
    up = tl.maximum(h + radius, 0)
    down = tl.maximum(h - radius, 0)
    score = tl.abs(up * up - down * down) * out_norm
    key = _ordered_bits(score)
    row = layer.to(tl.int64) * ROWS + row
    floor = tl.min(tl.topk(tl.max(key, 1), K), 0)
    packed = (key.to(tl.int64) << 32) | (N - 1 - cols).to(tl.int64)
    cand = key >= floor
    per_group = tl.sum(cand.to(tl.int32), 1)
    count = tl.sum(per_group, 0)
    if count <= CAP:
        slot = (tl.cumsum(per_group, 0) - per_group)[:, None] + tl.cumsum(cand.to(tl.int32), 1) - 1
        tl.store(Scratch + row * CAP + slot, packed, cand)
        tl.debug_barrier()
        c = tl.arange(0, CAP)
        top = tl.topk(tl.load(Scratch + row * CAP + c, c < count, other=-9223372036854775808), K)
    else:  # many ties at the floor: the K-th largest key by binary search, the keys above it and the first ties
        lo = floor.to(tl.int64)
        hi = tl.full((), 2147483647, tl.int64)
        for _ in tl.static_range(32):
            mid = (lo + hi + 1) // 2
            enough = tl.sum(tl.sum((key >= mid.to(tl.int32)).to(tl.int32), 1), 0) >= K
            lo = tl.where(enough, mid, lo)
            hi = tl.where(enough, hi, mid - 1)
        kth = lo.to(tl.int32)
        chosen = key > kth
        need = K - tl.sum(tl.sum(chosen.to(tl.int32), 1), 0)
        equal = (key == kth).to(tl.int32)
        equal_groups = tl.sum(equal, 1)
        rank = (tl.cumsum(equal_groups, 0) - equal_groups)[:, None] + tl.cumsum(equal, 1)
        chosen = chosen | ((equal != 0) & (rank <= need))
        chosen_groups = tl.sum(chosen.to(tl.int32), 1)
        slot = (tl.cumsum(chosen_groups, 0) - chosen_groups)[:, None] + tl.cumsum(chosen.to(tl.int32), 1) - 1
        tl.store(Scratch + row * CAP + slot, packed, chosen)
        tl.debug_barrier()
        top = tl.sort(tl.load(Scratch + row * CAP + tl.arange(0, K)), descending=True)
    index = (N - 1 - (top & 0xFFFFFFFF)).to(tl.int32)
    tl.store(Indices + row * K + tl.arange(0, K), index)
    selected = tl.full((K,), 0, tl.float32)
    for j in tl.static_range(L):
        if layer == j:
            selected = tl.load(Inputs[j] + source_row * N + index).to(tl.float32)
    tl.store(Clean + row * K + tl.arange(0, K), selected)


def prepare(inputs: list[torch.Tensor], radius: torch.Tensor, out_norm: torch.Tensor,
            count: int, scale: float, row_ids=None, partner_ids=None,
            row_length=0) -> tuple[torch.Tensor, torch.Tensor]:
    n = inputs[0].shape[-1]
    rows = inputs[0].numel() // n
    if partner_ids is not None:
        rows = 2 * row_ids.numel()
    layers = len(inputs)
    indices = torch.empty((layers, rows, count), device=radius.device, dtype=torch.int64)
    clean = torch.empty((layers, rows, count), device=radius.device, dtype=torch.float32)
    cap = 2 * count
    groups = min(4 * count, n)
    scratch = torch.empty((layers, rows, cap), device=radius.device, dtype=torch.int64)
    _route_prepare_kernel[(rows, layers)](tuple(inputs), radius.contiguous(), out_norm.contiguous(),
        indices, clean, scratch, L=layers, ROWS=rows, N=n, K=count, G=n//groups,
        CAP=cap, SCALE=scale, RowIds=row_ids, PartnerIds=partner_ids, T=row_length, num_warps=1, enable_fp_fusion=True)
    return indices, clean
