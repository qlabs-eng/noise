"""Finite RMS value responses and the linear descending residual scan."""
import torch
import triton
import triton.language as tl


@triton.jit
def _response(x, v):
    xx = tl.sum(x * x, 1)
    vv = tl.sum(v * v, 1)
    s = 0.01 * tl.sqrt(tl.maximum(xx / 512.0, 1.e-30) /
                       tl.maximum(vv / 512.0, 1.e-30))
    move = s[:, None] * v
    dots = tl.sum(x * move, 1)
    moved_sq = tl.sum(move * move, 1)
    inv0 = tl.rsqrt(xx / 512.0 + 1.1920928955078125e-7)
    up = tl.rsqrt((xx + 2.0 * dots + moved_sq) / 512.0 + 1.1920928955078125e-7)
    down = tl.rsqrt((xx - 2.0 * dots + moved_sq) / 512.0 + 1.1920928955078125e-7)
    positive = move * up[:, None] + x * (up - inv0)[:, None]
    negative = -move * down[:, None] + x * (down - inv0)[:, None]
    return (positive - negative) / (2.0 * s[:, None])


@triton.jit
def _fused(Blocks, Values, Gates, Previous, X0, Lambda, Extra, Partial,
           HAS_GATE: tl.constexpr, N: tl.constexpr, BT: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, 512)
    off = t[:, None] * 512 + c[None, :]
    ok = t[:, None] < N
    x0 = tl.load(X0 + off, ok, other=0).to(tl.float32)
    extra = tl.full((BT, 512), 0, tl.float32)
    count: tl.constexpr = triton.cdiv(N, BT)
    for i in tl.static_range(7, -1, -1):
        x = tl.load(Blocks[i] + off, ok, other=0).to(tl.float32)
        v = tl.load(Values[i] + off, ok, other=0).to(tl.float32)
        inp = _response(x, v)
        if HAS_GATE[i]:
            inp = inp + tl.load(Gates[i] + off, ok, other=0).to(tl.float32)
        tl.store(Extra + i * N * 512 + off, extra, ok)
        credit = inp + extra
        prev = tl.load(Previous[i] + off, ok, other=0).to(tl.float32)
        p0 = tl.sum(tl.sum(tl.where(ok, credit * prev, 0.0), 1), 0)
        p1 = tl.sum(tl.sum(tl.where(ok, credit * x0, 0.0), 1), 0)
        tl.store(Partial + i * count + tl.program_id(0), p0)
        tl.store(Partial + (8 + i) * count + tl.program_id(0), p1)
        extra = tl.load(Lambda + i).to(tl.float32) * credit


@triton.jit
def _reduce(Partial, Path, COUNT: tl.constexpr, BS: tl.constexpr):
    a = tl.arange(0, BS)
    x = tl.load(Partial + tl.program_id(0) * COUNT + a, a < COUNT, other=0)
    tl.store(Path + tl.program_id(0), tl.sum(x, 0))


def descending(gates, blocks, values, lam, x0, outs, bt=4):
    """The descending value chain over the 8 blocks in one kernel: each block's input response to its value
    response values[i] (+ its gate's input credit, None where the block has no value embedding), the extra carried
    down the residual scalars lam, and the mixing scalars' path contractions. Returns the per-block extras
    (tokens, 512) and the (2, 8) path sums."""
    assert len(blocks) == len(values) == len(gates) == 8
    n, d = x0.shape
    assert d == 512 and all(z.is_contiguous() for z in (*blocks, *values, x0, *outs))
    assert all(z is None or z.is_contiguous() for z in gates)
    flags = tuple(z is not None for z in gates)
    gate_ptrs = tuple(z if z is not None else x0 for z in gates)
    previous = (x0, *outs[:7])
    extra = torch.empty((8, n, 512), device=x0.device, dtype=torch.float32)
    count = triton.cdiv(n, bt)
    partial = torch.empty((2, 8, count), device=x0.device, dtype=torch.float32)
    path = torch.empty((2, 8), device=x0.device, dtype=torch.float32)
    _fused[(count,)](tuple(blocks), tuple(values), gate_ptrs, previous, x0, lam, extra, partial,
                     HAS_GATE=flags, N=n, BT=bt, num_warps=4, enable_fp_fusion=False)
    _reduce[(16,)](partial, path, count, triton.next_power_of_2(count),
                    num_warps=4, enable_fp_fusion=False)
    return list(extra.unbind(0)), path
