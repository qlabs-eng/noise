"""The Cut's descending pass: finite RMS value responses and the linear descending residual scan, which leaves the
writers' and MLP readers' credit operands and the residual scalars' path sums."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
def _reduce(Partial, Path, COUNT: tl.constexpr, BS: tl.constexpr):
    a = tl.arange(0, BS)
    x = tl.load(Partial + tl.program_id(0) * COUNT + a, a < COUNT, other=0)
    tl.store(Path + tl.program_id(0), tl.sum(x, 0))


@triton.jit
def _measure(Blocks, Values, Gates, Measured, GATED: tl.constexpr, N: tl.constexpr, BT: tl.constexpr):
    layer = tl.program_id(1)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    off = t[:, None] * 512 + tl.arange(0, 512)[None, :]
    ok = t[:, None] < N
    for i in tl.static_range(8):
        if layer == i:
            x = tl.load(Blocks[i] + off, ok, other=0).to(tl.float32)
            value = tl.load(Values[i] + off, ok, other=0).to(tl.float32)
            response = _response(x, value)
            if GATED[i]:
                response += tl.load(Gates[i] + off, ok, other=0).to(tl.float32)
            tl.store(Measured + i * N * 512 + off, response, ok)


@gluon.jit
def _direct(Previous, X0, Lambda, Chains, GTop, Pure, Mid, Writer, Partial, Measured,
            N: gl.constexpr, BT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 4], [1, 0])
    tile = gl.program_id(0)
    t = tile * BT + gl.arange(0, BT, gl.SliceLayout(1, layout))
    off = t[:, None] * 512 + gl.arange(0, 512, gl.SliceLayout(0, layout))[None, :]
    ok = t[:, None] < N
    x0 = gl.load(X0 + off, ok, other=0).to(gl.float32)
    extra = gl.full((BT, 512), 0, gl.float32, layout)
    following = gl.load(GTop + off, ok, other=0).to(gl.float32)
    count: gl.constexpr = triton.cdiv(N, BT)
    for i in gl.static_range(7, -1, -1):
        response = gl.load(Measured + i * N * 512 + off, ok, other=0)
        native = response + extra
        if i == 7:
            gain = 1.
        else:
            gain = gl.load(Lambda + i + 1)
        if i != 4 and i != 6 and i != 7:
            current = following * gain
        else:
            current = gl.load(Chains[i] + off, ok, other=0)
        address = i * N * 512 + off
        gl.store(Pure + address, (following * gain).to(gl.bfloat16), ok)
        gl.store(Mid + address, (current + extra).to(gl.bfloat16), ok)
        gl.store(Writer + address, gl.fma(following, gain, extra).to(gl.bfloat16), ok)
        following = current
        prev = gl.load(Previous[i] + off, ok, other=0).to(gl.float32)
        credit = native + current
        p0 = gl.sum(gl.sum(gl.where(ok, credit * prev, 0.), 1), 0)
        p1 = gl.sum(gl.sum(gl.where(ok, credit * x0, 0.), 1), 0)
        gl.store(Partial + i * count + tile, p0)
        gl.store(Partial + (8 + i) * count + tile, p1)
        extra = gl.load(Lambda + i).to(gl.float32) * native


def run(gates, blocks, values, lam, x0, outs, chains, top, bt):
    n = x0.shape[0]
    shape = (8, n, 512)
    packed = tuple(torch.empty(shape, device=x0.device, dtype=torch.bfloat16) for _ in range(3))
    count = triton.cdiv(n, bt)
    partial = torch.empty((2, 8, count), device=x0.device, dtype=torch.float32)
    path = torch.empty((2, 8), device=x0.device, dtype=torch.float32)
    measured = torch.empty(shape, device=x0.device, dtype=torch.float32)
    flags = tuple(g is not None for g in gates)
    gate_ptrs = tuple(g if g is not None else x0 for g in gates)
    _measure[(count, 8)](tuple(blocks), tuple(values), gate_ptrs, measured, flags,
                         n, bt, num_warps=4, enable_fp_fusion=False)
    _direct[(count,)]((x0, *outs[:7]), x0, lam, tuple(chains), top, *packed, partial, measured,
                      n, bt, num_warps=4, enable_fp_fusion=False)
    _reduce[(16,)](partial, path, count, triton.next_power_of_2(count),
                   num_warps=4, enable_fp_fusion=False)
    return packed, path
