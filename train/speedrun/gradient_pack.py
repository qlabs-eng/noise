"""The Cut update's gradient wire: the estimate's weight gradients divided by the row count, cast to bf16 and laid end
to end, the tensors of one shape and layout packed by one kernel launch."""
import collections
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(Inputs, Wire, Divisor, Offsets: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(1)
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = at // C * S0 + at % C * S1
    divisor = tl.load(Divisor).to(tl.float32)
    for j in tl.static_range(GROUP):
        if group == j:
            value = tl.load(Inputs[j] + source, at < N, 0).to(tl.float32) / divisor
            tl.store(Wire + Offsets[j] + at, value.to(tl.bfloat16), at < N)


def pack(gradients: list[torch.Tensor], divisor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gradients (1-D or 2-D, any strides) / divisor as one flat bf16 wire, in the order given. Also returns an fp32
    buffer of the same length, left unwritten: Cut.update copies into it the parts of the reduced wire it reads."""
    groups = collections.defaultdict(list)
    offset = 0
    for g in gradients:
        assert g.ndim in (1, 2)
        cols, s0, s1 = (g.numel(), 0, g.stride(0)) if g.ndim == 1 else (g.shape[1], *g.stride())
        groups[g.numel(), cols, s0, s1, g.dtype].append((g, offset))
        offset += g.numel()
    flat = torch.empty(offset, dtype=torch.float32, device=gradients[0].device)
    wire = torch.empty(offset, dtype=torch.bfloat16, device=gradients[0].device)
    for (n, cols, s0, s1, dtype), entries in groups.items():
        _pack[(triton.cdiv(n, 4096), len(entries))](
            tuple(e[0] for e in entries), wire, divisor, Offsets=tuple(e[1] for e in entries),
            N=n, C=cols, S0=s0, S1=s1, GROUP=len(entries), BLOCK=4096, num_warps=4)
    return flat, wire
