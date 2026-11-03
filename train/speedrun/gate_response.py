"""stacked_gate_input_responses' elementwise core as one Triton kernel.

response[l, t, r] = sum_h (sigmoid(zp) - sigmoid(zm)) kappa[l, t, h] / steps[l, t] with
zp / zm = (base[l, t, h] +- steps[l, t] projected[l, r, h]) * rsqrt(square[l, t] +- 2 steps cross[l, t, r]
+ steps^2 row_square[l, r] + EPS). Inductor's kernel recomputed the per-(layer, token) scalars (IEEE divides, sqrt)
for every (row, head); here they are hoisted, and (TANH) sigmoid(a) - sigmoid(b) = (tanh(a / 2) - tanh(b / 2)) / 2
by tanh.approx: 2 MUFU ops per pair instead of 4 (max relative error ~3e-4 of the largest response; without TANH,
exp2 and an approximate reciprocal: a few ulps, but slower than inductor's). r5base162 shapes (8 x 2048 x 512 x 8;
visit-kernel/r20 mb_gate.py): inductor's core 0.119 ms, TANH 0.059; whole function 0.626 -> 0.566."""
import torch
import triton
import triton.language as tl

CONFIG = dict(BT=32, BR=128, num_warps=4, TANH=1)
LOG2E = 1.4426950408889634


@triton.jit
def _sigmoid(z):
    return tl.fdiv(1.0, 1.0 + tl.exp2(-z * 1.4426950408889634), ieee_rounding=False)


@triton.jit
def _tanh(x):
    return tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=r,r', [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _gate_response_kernel(Base, Proj, Kappa, Steps, Square, Cross, RowSq, Out, T: tl.constexpr, R: tl.constexpr,
                          H: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, TANH: tl.constexpr):
    layer, tb, rb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tt = tb * BT + tl.arange(0, BT)
    rr = rb * BR + tl.arange(0, BR)
    lt = layer * T + tt
    steps = tl.load(Steps + lt)
    square = tl.load(Square + lt)
    cross = tl.load(Cross + (lt.to(tl.int64) * R)[:, None] + rr[None, :])
    row_sq = tl.load(RowSq + layer * R + rr)
    lin = 2 * steps[:, None] * cross
    quad = (steps * steps)[:, None] * row_sq[None, :]
    positive = tl.math.rsqrt(square[:, None] + lin + quad + EPS)
    negative = tl.math.rsqrt(square[:, None] - lin + quad + EPS)
    if TANH:  # sigmoid(a) - sigmoid(b) = (tanh(a / 2) - tanh(b / 2)) / 2: the halves folded into the rsqrt factors
        positive = positive * 0.5
        negative = negative * 0.5
    acc = tl.zeros((BT, BR), tl.float32)
    for h in tl.static_range(H):
        base = tl.load(Base + lt * H + h)[:, None]
        move = steps[:, None] * tl.load(Proj + (layer * R + rr) * H + h)[None, :]
        kappa = tl.load(Kappa + lt * H + h)[:, None]
        if TANH:
            acc += (_tanh((base + move) * positive) - _tanh((base - move) * negative)) * kappa
        else:
            acc += (_sigmoid((base + move) * positive) - _sigmoid((base - move) * negative)) * kappa
    if TANH:
        acc = acc * 0.5
    tl.store(Out + (lt.to(tl.int64) * R)[:, None] + rr[None, :], acc / steps[:, None])


@torch.library.custom_op('gate_response::responses', mutates_args=())
def responses(base: torch.Tensor, projected: torch.Tensor, kappa: torch.Tensor, steps: torch.Tensor,
              square: torch.Tensor, cross: torch.Tensor, row_square: torch.Tensor, eps: float) -> torch.Tensor:
    """base / kappa (L, T, H), projected (L, R, H), steps / square (L, T), cross (L, T, R), row_square (L, R), all
    fp32 -> response (L, T, R) fp32."""
    layers, t, h = base.shape
    r = projected.shape[1]
    bt, br = CONFIG['BT'], CONFIG['BR']
    assert t % bt == 0 and r % br == 0
    out = torch.empty((layers, t, r), device=base.device, dtype=torch.float32)
    f = lambda z: z.float().contiguous()
    _gate_response_kernel[(layers, t // bt, r // br)](f(base), f(projected), f(kappa), f(steps), f(square), f(cross),
      f(row_square), out, T=t, R=r, H=h, EPS=eps, BT=bt, BR=br, TANH=CONFIG['TANH'], num_warps=CONFIG['num_warps'])
    return out


@responses.register_fake
def _(base, projected, kappa, steps, square, cross, row_square, eps):
    return base.new_empty((base.shape[0], base.shape[1], projected.shape[1]), dtype=torch.float32)
