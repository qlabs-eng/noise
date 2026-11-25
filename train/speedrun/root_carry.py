"""The Cut's three root credits (4, 6 and the top) carried down the residual stream to each block and the embedding."""
import torch
import triton
import triton.language as tl
import fwht


@triton.jit
def _embedding(Root4, Root6, Top, Residual, Skip, Input, Emb, Operand,
               M: tl.constexpr, SIGMA: tl.constexpr):
    col = tl.arange(0, 512)
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    off = row[:, None] * 512 + col[None, :]
    mask = off < M
    four = tl.load(Root4 + off, mask, 0.)
    three = four * tl.load(Residual + 4)
    two = three * tl.load(Residual + 3)
    one = two * tl.load(Residual + 2)
    zero = one * tl.load(Residual + 1)
    six = tl.load(Root6 + off, mask, 0.)
    five = six * tl.load(Residual + 6)
    seven = tl.load(Top + off, mask, 0.)
    layers = (zero, one, two, three, four, five, six, seven)
    embedding = tl.full((4, 512), 0., tl.float32)
    for i in tl.static_range(8):
        c = layers[i]
        embedding += tl.load(Skip + i) * c
        tl.store(Operand + i * M + off, c.to(tl.bfloat16), mask)
    embedding = embedding + tl.load(Residual) * zero
    x = tl.load(Input + off, mask, 0.).to(tl.float32)
    sq = tl.sum(x * x, 1)[:, None]
    dot = tl.sum(x * embedding, 1)[:, None]
    xh = fwht._butterfly(x, col, 4, 512)
    gh = fwht._butterfly(embedding, col, 4, 512)
    eps: tl.constexpr = 1.1920928955078125e-7
    step = SIGMA * tl.maximum(tl.sqrt(sq / 512.), 1.e-6)
    inv = tl.rsqrt(sq / 512. + eps)
    up_inv = tl.rsqrt((sq + 2. * step * xh + step * step * 512.) / 512. + eps)
    down_inv = tl.rsqrt((sq - 2. * step * xh + step * step * 512.) / 512. + eps)
    up = step * gh * up_inv + dot * (up_inv - inv)
    down = -step * gh * down_inv + dot * (down_inv - inv)
    embedding = fwht._butterfly((up - down) / (2. * step), col, 4, 512) / 512.
    tl.store(Emb + off, embedding, mask)


def embed(roots, residual, skip, x, sigma):
    n, d = x.shape
    emb = torch.empty_like(roots[4])
    operands = x.new_empty((8, n, d), dtype=torch.bfloat16)
    _embedding[(n // 4,)](roots[4], roots[6], roots[7], residual, skip, x, emb,
        operands, n * d, sigma, num_warps=4, enable_fp_fusion=False)
    return emb, operands
