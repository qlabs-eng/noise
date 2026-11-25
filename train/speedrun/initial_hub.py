"""The attention hubs of one window's layers: each block's credit through its c_proj weight, per head, in bf16."""
import torch
import triton
import triton.language as tl

@triton.jit
def _window(Inputs, Writer, Hub, Layers: tl.constexpr, ROWS: tl.constexpr, T: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    g = tl.program_id(2)
    src = Inputs[0]
    layer = tl.full((), Layers[0], tl.int32)
    for j in tl.static_range(1, len(Layers)):
        src = tl.where(g == j, Inputs[j], src)
        layer = tl.where(g == j, Layers[j], layer)
    k = tl.arange(0, BK)
    lhs = src + m[:, None] * 512 + k[None, :]
    rhs = Writer + layer * 512 * 512 + k[:, None] * 512 + n[None, :]
    acc = tl.full((BM, BN), 0., tl.float32)
    for _ in range(512 // BK):
        source = tl.load(lhs)
        acc = tl.dot(source.to(tl.bfloat16), tl.load(rhs), acc)
        lhs += BK
        rhs += BK * 512
    row = g * ROWS + m // T
    dst = (row[:, None] * 8 + n[None, :] // 64) * (T * 64) + (m % T)[:, None] * 64 + n[None, :] % 64
    tl.store(Hub + dst, acc.to(tl.bfloat16))


def window(credits, writers, layers):
    rows, t, _ = credits[0].shape
    src = tuple(x.contiguous() for x in credits)
    hub = torch.empty((len(layers) * rows * 8, t, 64), dtype=torch.bfloat16, device=writers.device)
    _window[(rows * t // 128, 4, len(layers))](src, writers, hub, tuple(layers), rows, t, 128, 128, 64,
                                               num_warps=8, num_stages=3)
    return hub
