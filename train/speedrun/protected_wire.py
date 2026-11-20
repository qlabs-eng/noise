"""The identity phase's flat gradient all-reduce: the head and the scalars in fp32, the rest in fp16 at scale 32."""
import torch
import torch.distributed as dist
import triton
import triton.language as tl

PREFIX_N = 2097152
TAIL_START = 4194320
FLAT_N = 12583952
SMALL_N = PREFIX_N + FLAT_N - TAIL_START


@triton.jit
def _copy_small(X, Y, N: tl.constexpr, PREFIX: tl.constexpr,
                GAP: tl.constexpr, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    physical = j + tl.where(j >= PREFIX, GAP, 0)
    if INVERSE:
        value = tl.load(X + j, j < N, 0).to(tl.float32) * 0.03125
        tl.store(Y + physical, value, j < N)
    else:
        value = tl.load(X + physical, j < N, 0).to(tl.float32) * 32.0
        tl.store(Y + j, value, j < N)


def copy_small(flat, wire, inverse=False):
    assert flat.numel() == FLAT_N and wire.numel() == SMALL_N
    assert flat.dtype == torch.float32 and wire.dtype == torch.float16
    assert flat.is_contiguous() and wire.is_contiguous()
    src, dst = (wire, flat) if inverse else (flat, wire)
    _copy_small[(triton.cdiv(SMALL_N, 2048),)](
        src, dst, SMALL_N, PREFIX_N, TAIL_START - PREFIX_N, inverse, 2048)


class Reduced:
    def __init__(self, head_work, small_work, wire, output):
        self.head_work, self.small_work = head_work, small_work
        self.wire, self.output = wire, output

    def wait(self):
        self.head_work.wait()
        self.small_work.wait()
        copy_small(self.output, self.wire, inverse=True)


def reduce(model):
    assert dist.get_world_size() == 8
    assert model.flat_grad.dtype == torch.float32 and model.flat_grad.numel() == FLAT_N
    if not hasattr(model, '_identity_protected_wire'):
        model._identity_protected_wire = torch.empty(SMALL_N, device=model.flat_grad.device,
                                                     dtype=torch.float16)
    wire = model._identity_protected_wire
    head_work = dist.all_reduce(model.flat_grad[PREFIX_N:TAIL_START], async_op=True)
    copy_small(model.flat_grad, wire)
    small_work = dist.all_reduce(wire, async_op=True)
    return Reduced(head_work, small_work, wire, model.flat_grad)
