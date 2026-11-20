import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _head_response(Raw, Y, Loss, Out, T: tl.constexpr, V: tl.constexpr,
                   SIGMA: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    ts = tl.program_id(0) * BM + tl.arange(0, BM)
    vs = tl.program_id(1) * BN + tl.arange(0, BN)
    target = tl.load(Y + ts, ts < T, other=-1)
    rt = tl.load(Raw + ts * V + tl.maximum(target, 0), ts < T, other=0).to(tl.float32)
    clean = tl.load(Loss + ts, ts < T, other=0)
    log_z = tl.where(target >= 0, clean + 15.0 * libdevice.tanh(rt / 15.0), 30.0)
    raw = tl.load(Raw + ts[:, None] * V + vs[None, :],
                  (ts[:, None] < T) & (vs[None, :] < V), other=0).to(tl.float32)
    z = 15.0 * libdevice.tanh(raw / 15.0)
    up = 15.0 * libdevice.tanh((raw + SIGMA) / 15.0)
    down = 15.0 * libdevice.tanh((raw - SIGMA) / 15.0)
    probability = tl.exp(z - log_z[:, None])
    positive_loss = libdevice.log1p(probability * libdevice.expm1(up - z))
    negative_loss = libdevice.log1p(probability * libdevice.expm1(down - z))
    change = positive_loss - negative_loss
    change -= tl.where(vs[None, :] == target[:, None], up - down, 0.0)
    credit = tl.where(target[:, None] >= 0, change / (2.0 * SIGMA), 0.0)
    tl.store(Out + ts[:, None] * V + vs[None, :], credit * SCALE,
             (ts[:, None] < T) & (vs[None, :] < V))


def response(raw, targets, clean_loss, sigma=0.1, bf16=False, bm=4, bn=256, fp16=False, scale=1.0):
    assert raw.is_contiguous() and targets.is_contiguous() and clean_loss.is_contiguous()
    vocab = raw.shape[-1]
    tokens = raw.numel() // vocab
    out = torch.empty_like(raw, dtype=torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
    _head_response[(triton.cdiv(tokens, bm), triton.cdiv(vocab, bn))](
        raw, targets, clean_loss, out, tokens, vocab, sigma, bm, bn, scale, num_warps=4,
        enable_fp_fusion=False)
    return out


@torch.compile(dynamic=False)
def contract_fp16(error, inputs, scale=64.0):
    return torch.mm(error.flatten(0, 1).T, inputs.flatten(0, 1).to(torch.float16),
                    out_dtype=torch.float32) / scale
