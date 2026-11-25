"""The root replays' readout: each token's clean target terms, and a replay's loss change from its readout sums."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

EPS = torch.finfo(torch.float32).eps


@gluon.jit
def _target_rows(X, Normalized, Weight, Labels, FullClean, Records, Target,
                 WIDTH: gl.constexpr, SX: gl.constexpr, SN: gl.constexpr, SW: gl.constexpr):
    row = gl.program_id(0)
    col = gl.arange(0, WIDTH, gl.BlockedLayout([4], [32], [4], [0]))
    label = gl.maximum(gl.load(Labels + row), 0)
    x = gl.load(X + row * SX + col).to(gl.float32)
    normalized = gl.load(Normalized + row * SN + col).to(gl.bfloat16).to(gl.float32)
    weight = gl.load(Weight + label * SW + col).to(gl.float32)
    square = gl.sum(x * x, 0)
    inv = gl.rsqrt(square / 512.0 + 1.1920928955078125e-7)
    change_x = gl.sum((x * inv - normalized) * weight, 0)
    change_w = gl.sum(normalized * (weight - weight.to(gl.bfloat16).to(gl.float32)), 0)
    actual = gl.load(FullClean + row * 4096 + label).to(gl.float32)
    clean = actual + (change_x + change_w)
    gl.store(Records + row * 3, square)
    gl.store(Records + row * 3 + 1, gl.sum(x * weight, 0))
    gl.store(Records + row * 3 + 2, clean)
    gl.store(Target + row * WIDTH + col, weight)


def indexed_tokens(x, normalized, weight, labels, full_clean):
    n, d = x.shape
    normalized = normalized.reshape(n, d)
    records = x.new_empty((n, 3), dtype=torch.float32)
    target = x.new_empty((n, d), dtype=torch.float32)
    _target_rows[(n,)](x, normalized, weight, labels, full_clean, records, target, d,
                        x.stride(0), normalized.stride(0), weight.stride(0),
                        num_warps=4, enable_fp_fusion=False)
    return records, target


def context(fields, labels, b1, b2):
    square, raw, clean = fields.unbind(1)
    return torch.stack((square, raw, b1, b2, (labels >= 0).float(), clean), 1)


def loss(stats, fields, labels, b1, b2):
    square, raw, clean = fields.unbind(1)
    dots, dsq, target_dot, mu_dot, nu_dot, quad = stats.unbind(-1)
    inv = torch.rsqrt((square[:, None] + 2 * dots + dsq) / 512 + EPS)
    inv0 = torch.rsqrt(square[:, None] / 512 + EPS)
    a = inv - inv0
    lin = a * b1[:, None] + inv * mu_dot
    change = lin + a.square() * b2[:, None] + 2 * a * inv * nu_dot + inv.square() * quad - .5 * lin.square()
    moved = clean[:, None] + (a * raw[:, None] + inv * target_dot)
    cap_difference = 15 * torch.tanh(moved / 15) - 15 * torch.tanh(clean[:, None] / 15)
    return (change - cap_difference) * (labels >= 0)[:, None]
