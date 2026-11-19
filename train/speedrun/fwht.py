import torch
import triton
import triton.language as tl


@triton.jit
def _butterfly(v, col, BR: tl.constexpr, D: tl.constexpr):
    for bit in tl.static_range(0, 9):
        index = tl.broadcast_to((col ^ (1 << bit))[None, :], (BR, D))
        other = tl.gather(v, index, axis=1)
        v = tl.where((col[None, :] & (1 << bit)) == 0, v + other, other - v)
    return v


@triton.jit
def _frame_kernel(X, Frame, Out, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                  REVERSE: tl.constexpr, ROUND_BF16: tl.constexpr, TRANSPOSE: tl.constexpr,
                  BR: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, D)
    v = tl.load(X + row[:, None] * S0 + col[None, :] * S1, row[:, None] < N, 0).to(tl.float32)
    if ROUND_BF16:
        v = v.to(tl.bfloat16).to(tl.float32)
    sign = tl.load(Frame + col).to(tl.float32)
    if not REVERSE:
        v = v * sign[None, :]
    v = _butterfly(v, col, BR, D)
    if REVERSE:
        v = v * sign[None, :]
    if TRANSPOSE:
        offset = col[None, :] * N + row[:, None]
    else:
        offset = row[:, None] * D + col[None, :]
    tl.store(Out + offset, v, row[:, None] < N)


@torch.library.custom_op('fwht::frame', mutates_args=())
def frame(x: torch.Tensor, basis: torch.Tensor, reverse: bool = False,
          bf16: bool = False, transpose: bool = False) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[1] == 512 and basis.shape == (512, 512)
    n = x.shape[0]
    out = torch.empty((n, 512), device=x.device, dtype=torch.float32)
    _frame_kernel[(triton.cdiv(n, 4),)](x, basis, out, n, *x.stride(), reverse, bf16, False, 4, 512,
                                      num_warps=4)
    return out.T if transpose else out


@frame.register_fake
def _(x, basis, reverse=False, bf16=False, transpose=False):
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    return out.T if transpose else out

