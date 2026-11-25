"""Blocks 6 and 7's MLP reader and writer rows in root 6's signed Walsh frame, as fp16 pairs packed into one int32."""
import torch
import triton
import triton.language as tl
import fwht


@triton.jit
def _pack(Reader6, Reader7, Writer6, Writer7, Frame, Output, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, 512)
    layer = tl.program_id(1)
    reader, writer = Reader6, Writer6
    if layer == 1:
        reader, writer = Reader7, Writer7
    index = row[:, None] * 512 + col[None, :]
    sign = tl.load(Frame + col).to(tl.float32)
    r = fwht._butterfly(tl.load(reader + index).to(tl.float32) * sign[None, :], col, BR, 512)
    w = fwht._butterfly(tl.load(writer + index).to(tl.float32) * sign[None, :], col, BR, 512)
    lo = r.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi = w.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(Output + layer * 2048 * 512 + index, (lo | (hi << 16)).to(tl.int32, bitcast=True))


@torch.library.custom_op('paired_walsh::packed_weights', mutates_args=())
def packed_weights(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
                   frame: torch.Tensor) -> torch.Tensor:
    out = torch.empty((2, 2048, 512), dtype=torch.int32, device=a.device)
    _pack[(512, 2)](a, b, c, d, frame, out, 4, num_warps=4)
    return out


@packed_weights.register_fake
def _(a, b, c, d, frame):
    return a.new_empty((2, 2048, 512), dtype=torch.int32)
