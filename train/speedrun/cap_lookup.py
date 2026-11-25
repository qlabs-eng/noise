"""The lm_head credit of both phases: the loss and each logit's finite-difference response, with the cap function read
from a 65,536-entry table indexed by the logit's bf16 bits."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fill(Moves, Target, B: tl.constexpr):
    code = tl.program_id(0) * B + tl.arange(0, B)
    raw = code.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    clean = 15.0 * libdevice.tanh(raw / 15.0)
    positive = 15.0 * libdevice.tanh((raw + 0.1) / 15.0)
    negative = 15.0 * libdevice.tanh((raw - 0.1) / 15.0)
    up = libdevice.expm1(positive - clean)
    down = libdevice.expm1(negative - clean)
    exp_clean = tl.exp(clean)
    up *= exp_clean
    down *= exp_clean
    tl.store(Moves + 4 * code, tl.exp(clean))
    tl.store(Moves + 4 * code + 1, up)
    tl.store(Moves + 4 * code + 2, down)
    tl.store(Moves + 4 * code + 3, clean)
    tl.store(Target + 2 * code, clean)
    tl.store(Target + 2 * code + 1, positive - negative)


def allocate(device):
    moves = torch.empty((65536, 4), device=device, dtype=torch.float32)
    target = torch.empty((65536, 2), device=device, dtype=torch.float32)
    return moves, target


def fill(tables):
    _fill[(256,)](*tables, 256, enable_fp_fusion=False)


@triton.jit
def _row(Raw, Y, Moves, Target, LossOut, Out, V: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, V)
    y = tl.load(Y + row)
    target_code = tl.load(Raw + row * V + tl.maximum(y, 0)).to(tl.uint16, bitcast=True).to(tl.int32)
    target_clean = tl.load(Target + target_code * 2)
    target_gap = tl.load(Target + target_code * 2 + 1)
    offset = row * V + cols
    code = tl.load(Raw + offset).to(tl.uint16, bitcast=True).to(tl.int32)
    # A contiguous four-float record allows the three mass fields to be gathered together.
    record = tl.load(Moves + code[:, None] * 4 + tl.arange(0, 4)[None, :])
    a, b = tl.split(tl.reshape(record, (V, 2, 2)))
    clean_mass, down = tl.split(a)
    up, unused = tl.split(b)
    partition = tl.sum(clean_mass, 0)
    tl.store(LossOut + row, tl.where(y >= 0, tl.log(partition) - target_clean, 0.))
    inv_partition = 1. / partition
    difference = libdevice.log1p(up * inv_partition) - libdevice.log1p(down * inv_partition)
    delta = difference - tl.where(cols == y, target_gap, 0.)
    tl.store(Out + offset, tl.where(y >= 0, delta / 0.2, 0.) * 64.)


def run(raw, labels, tables):
    vocab = raw.shape[-1]
    n = raw.numel() // vocab
    out = torch.empty_like(raw, dtype=torch.float16)
    loss = torch.empty_like(labels, dtype=torch.float32)
    _row[(n,)](raw, labels, *tables, loss, out, vocab, num_warps=8, enable_fp_fusion=False)
    return loss, out
