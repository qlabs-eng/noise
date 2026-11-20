"""The MLP weight credits' bf16 operands, packed for all 8 blocks in one pass."""
import torch
import triton
import triton.language as tl
from id_kernels import _layer_offset

CONFIG = dict(BLOCK=1024, num_warps=4, enable_fp_fusion=True)


@triton.jit(do_not_specialize=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                              'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'])
def _pack_kernel(C, C1, C2, C3, C4, C5, C6, C7,
                 E, E1, E2, E3, E4, E5, E6, E7, G, Lam,
                 Mid, Out, Pure, N: tl.constexpr, BLOCK: tl.constexpr):
    layer = tl.program_id(1)
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = at < N
    cp = C + _layer_offset(layer, C1, C2, C3, C4, C5, C6, C7) // 2
    ep = E + _layer_offset(layer, E1, E2, E3, E4, E5, E6, E7) // 2
    above = tl.minimum(layer + 1, 7)
    nextp = C + _layer_offset(above, C1, C2, C3, C4, C5, C6, C7) // 2
    nextp = tl.where(layer == 7, G, nextp)
    scale = tl.where(layer == 7, 1.0, tl.load(Lam + above))
    current = tl.load(cp + at, live, 0)
    extra = tl.load(ep + at, live, 0)
    credit = tl.load(nextp + at, live, 0) * scale
    destination = layer * N + at
    tl.store(Mid + destination, (current + extra).to(tl.bfloat16), live)
    tl.store(Out + destination, (credit + extra).to(tl.bfloat16), live)
    tl.store(Pure + destination, credit.to(tl.bfloat16), live)


def _offsets(tensors):
    assert all(t.data_ptr() % 16 == 0 for t in tensors)
    return [(t.data_ptr() - tensors[0].data_ptr()) // 16 + (1 << 40)
            for t in tensors[1:]]


@torch.library.custom_op('mlp_pack::pack_credits', mutates_args=())
def pack_credits(chains: list[torch.Tensor], g_top: torch.Tensor,
                 lam: torch.Tensor, extras: list[torch.Tensor]
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(chains) == len(extras) == 8
    m, k = chains[0].shape
    assert k == 512 and g_top.shape == (m, k) and lam.shape == (8,)
    assert all(t.is_contiguous() and t.dtype == torch.float32
               and t.shape == (m, k) for t in [*chains, *extras, g_top])
    assert lam.is_contiguous() and lam.dtype == torch.float32
    outputs = [torch.empty((8, m, k), dtype=torch.bfloat16, device=g_top.device)
               for _ in range(3)]
    _pack_kernel[(triton.cdiv(m * k, CONFIG['BLOCK']), 8)](
        chains[0], *_offsets(chains), extras[0], *_offsets(extras),
        g_top, lam, *outputs, N=m * k, **CONFIG)
    return tuple(outputs)


@pack_credits.register_fake
def _(chains, g_top, lam, extras):
    return tuple(torch.empty((8, *g_top.shape), device=g_top.device,
                             dtype=torch.bfloat16) for _ in range(3))
