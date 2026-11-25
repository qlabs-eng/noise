"""Identity's batch order from update 196 on: within each window of six updates, the order with the least summed
distance between batches that share a bigram occurring exactly twice in the window."""
import itertools

import numpy as np
import torch
import triton
import triton.language as tl


WINDOW = 6
FIRST = 195
VOCAB = 4096


@triton.jit
def _count_edges(Keys, Out, SIZE: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    part, group = tl.program_id(0), tl.program_id(1)
    at = part * BLOCK + tl.arange(0, BLOCK)
    base = Keys + group * SIZE
    left = tl.load(base + at - 1, (at > 0) & (at <= SIZE), other=-8)
    first = tl.load(base + at, at < SIZE, other=-16)
    second = tl.load(base + at + 1, at + 1 < SIZE, other=-24)
    right = tl.load(base + at + 2, at + 2 < SIZE, other=-32)
    x, y = first & 7, second & 7
    token_pair = first >> 3
    twice = (at + 1 < SIZE) & (token_pair == second >> 3) & (token_pair != left >> 3) & (token_pair != right >> 3)
    codes = tl.where(twice & (x != y), 8 * x + y, 0)
    count = tl.histogram(codes, 64)
    bins = tl.arange(0, 64)
    tl.store(Out + (group * PARTS + part) * 64 + bins, tl.where(bins == 0, 0, count))


def edge_counts(rows):
    groups, width, _, _ = rows.shape
    owners = torch.arange(width, dtype=torch.int32, device=rows.device)[None, :, None, None]
    keys = (rows[..., :-1].int() * VOCAB + rows[..., 1:].int()) * 8 + owners
    keys = keys.flatten(1).sort(dim=1).values
    n = keys.size(1)
    parts = triton.cdiv(n, 512)
    partial = torch.empty((groups, parts, 64), dtype=torch.int32, device=keys.device)
    _count_edges[(parts, groups)](keys, partial, n, parts, 512, num_warps=4)
    return partial.sum(dim=1, dtype=torch.int32)


def tables(device):
    serial = np.arange(WINDOW)
    perms = sorted(itertools.permutations(serial), key=lambda p: sum(abs(i - j) for i, j in enumerate(p)))
    perms = np.asarray(perms, dtype=np.int64)
    positions = np.argsort(perms, axis=1)
    distances = np.zeros((64, len(perms)), dtype=np.float64)
    for i in range(WINDOW):
        for j in range(i + 1, WINDOW):
            distances[8 * i + j] = np.abs(positions[:, j] - positions[:, i])
    return torch.from_numpy(perms).to(device), torch.from_numpy(distances).to(device)


def choose(counts, options, distances):
    costs = counts.double() @ distances
    which = costs.argmin(1)
    return options[which]


def arrange(rows, options, distances):
    source = rows[FIRST:].view(-1, WINDOW, *rows.shape[1:])
    counts = edge_counts(source)
    order = choose(counts, options, distances)
    offsets = torch.arange(len(source), device=rows.device)[:, None] * WINDOW
    physical = (order + offsets).flatten()
    rows[FIRST:].copy_(rows[FIRST:].index_select(0, physical))
