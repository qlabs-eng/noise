"""The Cut's readout credit (the top root) and the head moments the root replays read, both rows in one pass: each row
keeps its own Walsh signs and class draws."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import fwht


@triton.jit
def _finish(X, Y, Frame, Head, Base, Square, Step, Inv0, B1, B2, Mu, Nu, Quad, Output,
            N: tl.constexpr, QSTRIDE: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, 512)
    pos = row[:, None] * 512 + col[None, :]
    keep = row < N
    sign = tl.load(Frame + col)[None, :]
    label = tl.load(Y + row, keep, -1)
    target = tl.maximum(label, 0)
    w = tl.load(Head + target[:, None] * 512 + col[None, :])
    w = fwht._butterfly(w.to(tl.bfloat16).to(tl.float32) * sign, col, BR, 512)
    x = tl.load(X + pos, keep[:, None], 0).to(tl.bfloat16).to(tl.float32)
    mu = tl.load(Mu + pos, keep[:, None], 0).to(tl.bfloat16).to(tl.float32)
    nu = tl.load(Nu + pos, keep[:, None], 0).to(tl.bfloat16).to(tl.float32)
    xh = fwht._butterfly(x * sign, col, BR, 512)
    muh = fwht._butterfly(mu * sign, col, BR, 512)
    nuh = fwht._butterfly(nu * sign, col, BR, 512)
    square = tl.load(Square + row, keep, 0)[:, None]
    step = tl.load(Step + row, keep, 1)[:, None]
    inv0 = tl.load(Inv0 + row, keep, 1)[:, None]
    inv = tl.rsqrt((square + (2. * step) * xh + (step * step) * 512.) / 512. + EPS)
    a = inv - inv0
    b1 = tl.load(B1 + row, keep, 0)[:, None]
    b2 = tl.load(B2 + row, keep, 0)[:, None]
    quad = tl.load(Quad + row[:, None] * QSTRIDE + col[None, :], keep[:, None], 0)
    lin = a * b1 + inv * step * muh
    change = lin + a * a * b2 + 2. * a * inv * step * nuh + (inv * inv) * (step * step) * quad - .5 * lin * lin
    base = tl.load(Base + row, keep, 0)[:, None]
    moved = 15. * libdevice.tanh((base + step * w) * inv / 15.)
    clean = 15. * libdevice.tanh(base * inv0 / 15.)
    derivative = tl.where(label[:, None] >= 0, (change - (moved - clean)) / step, 0.)
    credit = fwht._butterfly(derivative.to(tl.bfloat16).to(tl.float32), col, BR, 512) * sign / 512.
    tl.store(Output + pos, credit, keep[:, None])


@torch.library.custom_op('clean_anchor::finish', mutates_args=())
def finish(x: torch.Tensor, y: torch.Tensor, frame: torch.Tensor, head: torch.Tensor,
           base: torch.Tensor, square: torch.Tensor, steps: torch.Tensor, inv0: torch.Tensor,
           b1: torch.Tensor, b2: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor,
           quad: torch.Tensor, epsilon: float) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    n = x.shape[0]
    _finish[(triton.cdiv(n, 4),)](x, y, frame, head, base, square, steps, inv0, b1, b2, mu, nu,
                                  quad, out, n, quad.stride(0), epsilon, 4,
                                  num_warps=4, enable_fp_fusion=False)
    return out


@finish.register_fake
def _(x, y, frame, head, base, square, steps, inv0, b1, b2, mu, nu, quad, epsilon):
    return torch.empty_like(x, dtype=torch.float32)


def class_indices(tail_draws, v, device):
    indices = []
    for draw in tail_draws:
        first, rest = draw // 15, draw % 15
        loc = torch.stack((first, rest + (rest >= first).long()), 1)
        indices.append((torch.arange(v // 16, device=device)[:, None] * 16 + loc).flatten())
    ix = torch.stack(indices)
    return ix


def contract(x, y, weight, wb, frames, ix, square, steps, inv0, c1, quadratic, cross, stats, epsilon):
    rows = len(frames)
    n, d = x.shape
    t = n // rows
    mu = torch.mm(c1, wb, out_dtype=torch.float32)
    selected = weight[ix]
    sw = selected.bfloat16()
    projected = torch.stack([fwht.frame(sw[r], frames[r]) for r in range(rows)])
    sqweights = torch.cat((selected.square(), projected.square()), 2).bfloat16()
    nu = torch.bmm(cross.reshape(rows, t, -1), sw, out_dtype=torch.float32).flatten(0, 1)
    second = torch.bmm(quadratic.reshape(rows, t, -1), sqweights, out_dtype=torch.float32).flatten(0, 1)
    curv, quad = second[:, :d].contiguous(), second[:, d:]
    credit = []
    for r, frame in enumerate(frames):
        sl = slice(r * t, (r + 1) * t)
        # stats[2]: each token's target logit with the norm's scale divided out.
        credit.append(finish(x[sl], y[sl], frame, weight, stats[2, sl].view(t, 1),
                     square[sl], steps[sl], inv0[sl], stats[0, sl], stats[1, sl], mu[sl], nu[sl], quad[sl], epsilon))
    return torch.cat(credit), (stats[0], stats[1], mu, nu, curv)


@triton.jit
def _factors(Clean, Labels, Inv, Indices, Linear, Quadratic, Cross, Stats,
             N: tl.constexpr, T: tl.constexpr, V: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, V)
    z = tl.load(Clean + row * V + columns)
    label = tl.load(Labels + row)
    target_col = tl.maximum(label, 0)
    target = tl.load(Clean + row * V + target_col)
    inv = tl.load(Inv + row)
    base = z.to(tl.float32) / inv
    tangent = libdevice.tanh(z.to(tl.float32) / 15.)
    mass = tl.exp(15. * tangent)
    partition = tl.sum(mass, 0)
    reciprocal = 1. / partition
    probability = mass * reciprocal
    slope = 1. - tangent * tangent
    first = probability * slope
    second = .5 * probability * (slope * slope - (2. / 15.) * tangent * slope)
    tl.store(Linear + row * V + columns, first)
    tl.store(Stats + row, tl.sum(first * base, 0))
    tl.store(Stats + N + row, tl.sum(second * base * base, 0))
    tl.store(Stats + 2 * N + row, target.to(tl.float32) / inv)
    slots = tl.arange(0, V // 8)
    ids = tl.load(Indices + (row // T) * (V // 8) + slots).to(tl.int32)
    selected = tl.gather(second, ids, 0) * 8.
    tl.store(Quadratic + row * (V // 8) + slots, selected)
    tl.store(Cross + row * (V // 8) + slots, selected * tl.gather(base, ids, 0))


def outputs(z):
    n, v = z.shape
    linear = z.new_empty((n, v))
    quad = z.new_empty((n, v // 8))
    return (linear, quad, torch.empty_like(quad), z.new_empty((3, n), dtype=torch.float32))


@torch.library.custom_op('clean_anchor::factors', mutates_args=())
def factors(z: torch.Tensor, labels: torch.Tensor, inv: torch.Tensor, ix: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, v = z.shape
    out = outputs(z)
    _factors[(n,)](z, labels, inv, ix, *out, n, n // ix.shape[0], v,
                    num_warps=8, enable_fp_fusion=False)
    return out


@factors.register_fake
def _(z, labels, inv, ix):
    return outputs(z)


@torch.compile(dynamic=False)
def readout(x, y, weight, frames, draws, eps, clean):
    d = x.shape[1]
    v = weight.shape[0]
    ix = class_indices(draws, v, x.device)
    sq = x.square().sum(-1, keepdim=True)
    inv = torch.rsqrt(sq / d + eps)
    steps = .005 * (sq / d).sqrt().clamp_min(1.e-6)
    c1, quad, cross, stats = factors(clean, y, inv, ix)
    return contract(x, y, weight, weight.bfloat16(), frames, ix, sq, steps, inv,
                    c1, quad, cross, stats, eps)
