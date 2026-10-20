"""A PyTorch reference for PR #29 (not for merging): #29's trainer (@RobertTLange's identity-path credit with moment
kernels and one head per GPU, #25's recipe), ported from JAX to PyTorch for comparison.

Same estimator, recipe and work split as #29. One process per GPU: each runs the clean pass on its own rows (two GPUs
share a 2048-token row and keep half each), the readout cycle (a degree-3 expansion as six matrix products) and the
MLP steps on its tokens, and one head's queries, keys and values over the batch: their inputs arrive in one
all-to-all, the exact experiments run as attention-weighted moments on tensor cores (degree 2, kernels.py: Triton
versions of #29's Pallas kernels), and the credits go back by token in a second all-to-all. The ANVIL matrices'
gradients are reduce-scattered to the GPU that steps them. Each update is one CUDA graph per row length (collectives
captured inside) that reads its batch and step index from device memory.

    torchrun --standalone --nproc_per_node=8 train/speedrun/run.py

Weights are kept in the JAX trainer's layout ((in, out) matrices, the head (d, vocab)) so formulas read as in #29;
`state_dict()` exports utils/model.py's layout.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.prepare_data import TRAIN_ROWS, load_rows  # noqa: E402
from kernels import (moment_coefficients, moment_features, moment_key_scores, moment_query_scores,  # noqa: E402
                     value_credit)

F32, BF16 = torch.float32, torch.bfloat16

# recipe (#25 / #29: 975 updates of 4 rows, cooldown the last 40%, 1024-token rows until 80%)
BATCH, SEQ, STEPS, SEED = 4, 2048, 975, 0
WARMUP_LEN, WARMUP_UNTIL = 1024, STEPS * 4 // 5
COOLDOWN = STEPS * 2 // 5
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.003, 0.7, 0.98, 1e-8
LR_WTE, LR_VE, LR_LAMBDAS = 0.9, 0.9, 0.003
ANVIL_LR, ANVIL_WD = 0.04, 2.25

GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
EPS = float(np.finfo(np.float32).eps)  # the readout norm's epsilon
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1

ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
              (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
              (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240, 50, 514)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')


def has_ve(layer):
    return layer % 2 == (L - 1) % 2


def window(layer):
    if layer == L - 1:
        return SEQ
    return SEQ if 'SSSL'[layer % 4] == 'L' else SEQ // 2


def norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)


def softcap(x):
    return SOFTCAP * torch.tanh(x / SOFTCAP)


def hadamard(d, device):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


# ------------------------------------------------------------------------------------------------------------- model

class Model:
    """Parameters (JAX layout), optimizer state and the static buffers of the captured step."""

    def __init__(self, device, rank, world):
        self.device, self.rank, self.world = device, rank, world
        g = torch.Generator(device='cpu').manual_seed(SEED)
        s = math.sqrt(3.0) * D ** -0.5
        uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
        proj = lambda fan_in, fan_out: uni((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
        # block matrices live in two stacks, oriented wide side out as ANVIL steps them: square [kind * L + layer]
        # (q, k, v, proj), wide [c_fc layers, then c_proj layers transposed]
        self.square = torch.stack([uni((D, D), s) if k != 'attn.c_proj' else proj(D, D)
                                   for k in SQUARE for _ in range(L)]).to(device)
        self.wide = torch.stack([uni((D, 4 * D), s) for _ in range(L)] + [proj(4 * D, D).T for _ in range(L)]).to(device)
        self.gates = {i: torch.zeros(GATE_CHANNELS, H, device=device) for i in range(L) if has_ve(i)}
        self.wte = torch.randn(VOCAB, D, generator=g).to(device, BF16)
        self.lm_head = (0.001 * torch.randn(D, VOCAB, generator=g)).to(device)
        self.ve = {i: uni((VOCAB, D), s).to(device, BF16) for i in range(L) if has_ve(i)}
        self.lam_r = torch.ones(L, device=device)
        self.lam_0 = torch.full((L,), 0.1, device=device)
        # the ANVIL matrices' gradient stacks (each GPU's partial sums, reduce-scattered to the GPU that steps them),
        # and one flat buffer (one all-reduce) for the rest
        shapes = dict(wte=self.wte.shape, lm_head=self.lm_head.shape,
                      lam_r=(L,), lam_0=(L,), **{f'gate{i}': (GATE_CHANNELS, H) for i in self.gates},
                      **{f've{i}': (VOCAB, D) for i in self.ve})
        sizes = {k: math.prod(v) for k, v in shapes.items()}
        self.flat_grad = torch.zeros(sum(sizes.values()), device=device)
        self.grad, offset = {}, 0
        for k, shape in shapes.items():
            self.grad[k] = self.flat_grad[offset:offset + sizes[k]].view(shape)
            offset += sizes[k]
        self.grad['square'], self.grad['wide'] = torch.zeros_like(self.square), torch.zeros_like(self.wide)
        self.owned = {name: torch.zeros(w.shape[0] // world, *w.shape[1:], device=device)
                      for name, w in (('square', self.square), ('wide', self.wide))}  # this GPU's summed share
        # Adam on everything but the ANVIL matrices; ANVIL state per stacked matrix
        self.adam = [(self.wte, self.grad['wte'], LR_WTE), (self.lm_head, self.grad['lm_head'], LR),
                     (self.lam_r, self.grad['lam_r'], LR_LAMBDAS), (self.lam_0, self.grad['lam_0'], LR_LAMBDAS)]
        self.adam += [(self.gates[i], self.grad[f'gate{i}'], LR) for i in self.gates]
        self.adam += [(self.ve[i], self.grad[f've{i}'], LR_VE) for i in self.ve]
        self.m = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.v = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.anvil = {name: dict(fast=torch.zeros_like(w), slow=torch.zeros_like(w),
                                 energy=torch.zeros(w.shape[0], 1, w.shape[2], device=device))
                      for name, w in (('square', self.square), ('wide', self.wide))}
        self.H = hadamard(D, device)
        self.dims = torch.arange(HD, device=device)  # every head dimension (one head per GPU)
        self.rows = torch.randint(0, VOCAB, (STEPS, BATCH, SEQ + 1), device=device)  # the batches (random until training)
        self.losses = torch.zeros(STEPS, device=device)
        self.step_t = torch.zeros((), device=device)  # the update index (0-based)
        self.scale_t = torch.ones((), device=device)  # the warmup-stable-decay factor
        self.rates = dict(square=torch.ones(4 * L, device=device),
                          wide=torch.tensor([2.0] * L + [1.0] * L, device=device))  # sqrt(max(1, out / in))
        self.square_bf, self.wide_bf = self.square.to(BF16), self.wide.to(BF16)  # the forward's bf16 weights
        self.lm_head_bf = self.lm_head.to(BF16)
        self.gates_bf = {i: g.to(BF16) for i, g in self.gates.items()}
        mod = lambda w: (lambda b, h, qi, ki: (qi >= ki) & (qi - ki <= w))
        self.masks = {(w, t): create_block_mask(mod(w), None, None, t, t, device=device)
                      for w in (SEQ // 2, SEQ) for t in (WARMUP_LEN, SEQ)}

    # views in the JAX trainer's layout
    def w(self, i, kind):
        if kind in SQUARE:
            return self.square[SQUARE.index(kind) * L + i]
        return self.wide[i] if kind == 'mlp.c_fc' else self.wide[L + i].T

    def wb(self, i, kind):
        if kind in SQUARE:
            return self.square_bf[SQUARE.index(kind) * L + i]
        return self.wide_bf[i] if kind == 'mlp.c_fc' else self.wide_bf[L + i].T

    @torch.no_grad()
    def refresh_bf16(self):
        self.square_bf.copy_(self.square)
        self.wide_bf.copy_(self.wide)
        self.lm_head_bf.copy_(self.lm_head)
        for i, g in self.gates.items():
            self.gates_bf[i].copy_(g)

    def tensors(self):
        """Every tensor the training state consists of (for snapshots)."""
        out = [self.square, self.wide, self.wte, self.lm_head, self.lam_r, self.lam_0, *self.gates.values(),
               *self.ve.values(), *self.m, *self.v]
        return out + [x for st in self.anvil.values() for x in st.values()]

    def state_dict(self):
        """utils/model.py's state dict (float32, its layout)."""
        sd = {'transformer.wte.weight': self.wte.float(), 'lm_head.weight': self.lm_head.T,
              'resid_lambdas': self.lam_r, 'x0_lambdas': self.lam_0}
        for i in range(L):
            for kind in (*SQUARE, 'mlp.c_fc', 'mlp.c_proj'):
                sd[f'transformer.h.{i}.{kind}.weight'] = self.w(i, kind).T
            if has_ve(i):
                sd[f'transformer.h.{i}.attn.ve_gate.weight'] = self.gates[i].T
                sd[f'value_embeds.{i}.weight'] = self.ve[i].float()
        return {k: v.detach().float().contiguous().cpu() for k, v in sd.items()}


def rotary(t, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, HD, 2, dtype=F32, device=device) / HD)
    freqs = torch.outer(torch.arange(t, dtype=F32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()  # [t, HD / 2]


def rope_norm(raw, cos, sin):
    """RoPE then QK norm in fp32 -> bf16 [b, h, t, d]. raw [b, t, h, d]; cos, sin [t, d/2]."""
    x = raw.float()
    c, s_ = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * c + x2 * s_, x1 * -s_ + x2 * c], -1)
    return (rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)).to(BF16).transpose(1, 2).contiguous()


def forward_impl(model, idx, targets, cos, sin, lo: int, hi: int):
    """The clean pass on this GPU's rows idx [b, t]: (cache, mean loss of its tokens lo..hi of the flattened rows)."""
    b, t = idx.shape
    cache = {}
    x = norm(model.wte[idx])
    x0, pending = x, None
    cache['x0'] = x0
    for i in range(L):
        mixed = model.lam_r[i] * (x if pending is None else x + pending.float()) + model.lam_0[i] * x0
        n1 = norm(mixed).to(BF16)
        q, k, v = (n1 @ model.wb(i, f'attn.c_{s}') for s in 'qkv')
        values = v.view(b, t, H, HD)
        if has_ve(i):
            ve = model.ve[i][idx]
            z = n1[..., :GATE_CHANNELS] @ model.gates_bf[i]
            values = values + (2 * torch.sigmoid(z.float())).unsqueeze(-1).to(BF16) * ve.view(b, t, H, HD)
            cache[f'h{i}.ve_gate'], cache[f'h{i}.ve'] = z, ve
        qp, kp = rope_norm(q.view(b, t, H, HD), cos, sin), rope_norm(k.view(b, t, H, HD), cos, sin)
        o, lse = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.masks[(window(i), t)],
                                return_lse=True)
        o = o.transpose(1, 2).reshape(b, t, D)
        cache.update({f'h{i}.in.attn': n1, f'h{i}.q': q, f'h{i}.k': k, f'h{i}.values': values,
                      f'h{i}.lse': lse.float(), f'h{i}.in.attn.c_proj': o})
        x = mixed + (o @ model.wb(i, 'attn.c_proj')).float()
        n2 = norm(x).to(BF16)
        hidden = n2 @ model.wb(i, 'mlp.c_fc')
        act = F.relu(hidden).square()
        pending = act @ model.wb(i, 'mlp.c_proj')
        cache.update({f'h{i}.in.mlp.c_fc': n2, f'h{i}.mlp.c_fc': hidden, f'h{i}.in.mlp.c_proj': act,
                      f'h{i}.stream': x + pending.float()})
    final = norm(x + pending.float()).to(BF16).reshape(b * t, D)[lo:hi]
    raw = final @ model.lm_head_bf
    cache['in.lm_head'], cache['lm_head'] = final, raw
    loss = F.cross_entropy(softcap(raw.float()), targets.reshape(-1)[lo:hi])
    return cache, loss


forward = torch.compile(forward_impl, dynamic=False)


# --------------------------------------------------------------------------------------------------------- estimator

@torch.compile(dynamic=False)
def coordinate_change(raw, targets):
    """The LM head's credit on raw logits: each logit moved up by HEAD_SIGMA minus moved down, over 2 sigma."""
    raw = raw.float()
    z = softcap(raw)
    up, down = softcap(raw + HEAD_SIGMA), softcap(raw - HEAD_SIGMA)
    weight = torch.softmax(z, -1)
    change = torch.log1p(weight * torch.expm1(up - z)) - torch.log1p(weight * torch.expm1(down - z))
    change = change - F.one_hot(targets, VOCAB).float() * (up - down)
    return change / (2 * HEAD_SIGMA)


def step_signs(step_t, device):
    """D random signs from the update index (a counter hash, so a replayed graph draws fresh signs every update)."""
    x = (step_t.to(torch.int64) * 0x9E3779B1 + torch.arange(D, device=device, dtype=torch.int64) * 0x85EBCA77
         + SEED) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x = x ^ (x >> 12)
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    return 1.0 - 2.0 * (x & 1).float()


@torch.compile(dynamic=False)
def cycle_weights(x, base, h):
    """The readout cycle's moment weights (#29's cycle_derivatives): per token, f(z) = exp(15 tanh(z / 15)) at the
    clean logits z and its log-derivatives; per move j, the norm's change delta_j and the step sigma_j."""
    steps = SIGMA * torch.clamp(torch.sqrt(x.square().mean(-1, keepdim=True)), min=1e-6)
    sq = x.square().sum(-1, keepdim=True)
    dots = x @ h.T
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * D) / D + EPS)
    inv0 = torch.rsqrt(sq / D + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv
    z = base * inv0
    t = torch.tanh(z / SOFTCAP)
    f = torch.exp(SOFTCAP * t)
    g1 = 1 - t * t
    d1 = -2 * t * g1 / SOFTCAP
    d2 = -2 * g1 * (1 - 3 * t * t) / SOFTCAP ** 2
    g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2
    fz, fzz = f * z, f * z * z
    total = f.sum(-1, keepdim=True)
    rows = torch.cat([(fz * g1).sum(-1, keepdim=True), (fzz * g2).sum(-1, keepdim=True),
                      (fzz * z * g3).sum(-1, keepdim=True)], 1)
    fast = torch.stack([fz * g2, fzz * g3, f * g2, fz * g3, f * g3]).to(BF16)  # [5, n, V]
    return f * g1, fast, rows, total, steps, inv, inv0, delta, sigma


@torch.compile(dynamic=False)
def cycle_finish(exact, fast, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h):
    ratio = (sigma * exact + delta * sigma * fast[0] + 0.5 * delta ** 2 * sigma * fast[1] + 0.5 * sigma ** 2 * fast[2]
             + 0.5 * delta * sigma ** 2 * fast[3] + sigma ** 3 / 6 * fast[4]
             + delta * rows[:, :1] + 0.5 * delta ** 2 * rows[:, 1:2] + delta ** 3 / 6 * rows[:, 2:])
    clean_partition = torch.log(total)
    base_y = base.gather(1, y[:, None])
    dirs_y = c.T[y]
    clean = clean_partition - softcap(base_y * inv0)
    moved = clean_partition + torch.log1p(ratio / total) - softcap((base_y + steps * dirs_y) * inv)
    return ((moved - clean) / steps) @ h / D


def top_credit(model, x, y):
    """Each token's readout credit at the top stream x (m, d): a whole sign-flipped Hadamard cycle, every moved
    stream's partition function from a degree-3 expansion around the clean logits (six matrix products)."""
    h = model.H * step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head  # (P, V): row j is direction j's logits
    base = x @ model.lm_head
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = cycle_weights(x, base, h)
    torch.backends.cuda.matmul.allow_tf32 = False  # the first-order product in full fp32, as #29's HIGHEST
    exact = fg1 @ c.T
    torch.backends.cuda.matmul.allow_tf32 = True
    cb, c2 = c.to(BF16), (c * c).to(BF16)
    c3 = (c * c * c).to(BF16)
    prods = torch.stack([fast[0] @ cb.T, fast[1] @ cb.T, fast[2] @ c2.T, fast[3] @ c2.T, fast[4] @ c3.T]).float()
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def closed_coefficients(raw, cos, sin, dims, sigma: float):
    """Moves of raw [BH, T, D] by sigma times its RMS along dims [J]: coefficients [BH, 3, T, J], steps [BH, T, 1]."""
    half = HD // 2
    sq = (raw * raw).sum(-1, keepdim=True)
    steps = sigma * torch.clamp(torch.sqrt(sq / HD), min=1e-6)
    change = (2 * steps * raw[..., dims] + steps * steps) / HD / (sq / HD + 1e-6)
    a = torch.expm1(-0.5 * torch.log1p(change))
    b = steps * torch.rsqrt(sq / HD + 1e-6) * (1 + a)
    i = dims % half
    first = (dims < half)[None, :]
    u = torch.where(first, cos[:, i], sin[:, i])
    w = torch.where(first, -sin[:, i], cos[:, i])
    return torch.stack([a, b * u, b * w], 1), steps


def rotated_columns(processed, dims):
    i = dims % (HD // 2)
    return torch.stack([processed[..., i], processed[..., i + HD // 2]], 1).float()


def rope32(x, cos, sin):
    """RoPE then QK norm in fp32 over [b, t, d] (one head)."""
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


CHUNK, REPEAT = 16, HD // 16  # directions per moment-kernel row; chunks of a head's dims ride in the batch axis
PACK = 5 * HD + 2  # q_raw, k_raw, values, attention output, target (bf16), log-sum-exp (fp32 bits)


@torch.compile(dynamic=False)
def pack_heads(qs, ks, vals, outs, targets, lse_bits):
    """This GPU's tokens' inputs to every block's exact kernels, by head: [h, L, m, PACK] bf16 (lists over blocks:
    q, k (m, d), values (m, h, hd), outputs (m, d), targets (m, d) bf16, the lse's fp32 bits (m, 2h) bf16)."""
    parts = []
    for q, k, v, o, tg, lb in zip(qs, ks, vals, outs, targets, lse_bits):
        m = q.shape[0]
        bits = lb.view(m, H, 2)
        parts.append(torch.cat([q.view(m, H, HD), k.view(m, H, HD), v.view(m, H, HD), o.view(m, H, HD),
                                tg.view(m, H, HD), bits], -1))
    return torch.stack(parts).permute(2, 0, 1, 3).contiguous()


@torch.compile(dynamic=False)
def head_prep(whole, lse, cos, sin, dims):
    """One block's exact-kernel inputs for this GPU's head over the batch, from whole [b, t, PACK], lse [b, t]."""
    b, t = whole.shape[:2]
    q_raw, k_raw, values, out, hub = (whole[..., j * HD:(j + 1) * HD].float() for j in range(5))
    q1, k1 = rope32(q_raw, cos, sin), rope32(k_raw, cos, sin)
    hub_out = (hub * out).sum(-1)
    batch = lambda x: torch.movedim(x.reshape(*x.shape[:-1], REPEAT, CHUNK), -2, 1).reshape(-1, *x.shape[1:-1], CHUNK)
    coef_q, steps_q = closed_coefficients(q_raw, cos, sin, dims, KEY_SIGMA)
    coef_k, steps_k = closed_coefficients(k_raw, cos, sin, dims, KEY_SIGMA)
    cq, ck = moment_coefficients(batch(coef_q)).contiguous(), moment_coefficients(batch(coef_k)).contiguous()
    fk = moment_features(batch(rotated_columns(k1, dims)), HD).contiguous()
    fq = moment_features(batch(rotated_columns(q1, dims)), HD).contiguous()
    q1b, k1b = q1.to(BF16).contiguous(), k1.to(BF16).contiguous()
    return (q1b, k1b, values.to(BF16).contiguous(), hub.to(BF16).contiguous(), lse.contiguous(), hub_out.contiguous(),
            cq, ck, fq, fk, steps_q, steps_k)


@torch.compile(dynamic=False)
def head_finish(qs, ks, vs, steps_q, steps_k):
    """[3, b t, hd]: the head's query, key and value credits by token."""
    b, t = steps_q.shape[:2]
    unbatch = lambda x: torch.movedim(x.reshape(b, REPEAT, t, CHUNK), 1, 2).reshape(b, t, HD)
    return torch.stack([unbatch(qs) / steps_q, unbatch(ks) / steps_k, vs]).reshape(3, b * t, HD)


def head_credits(whole, rope32, win):
    lse_whole = whole[..., 5 * HD:].contiguous().view(F32)[..., 0]  # the lse's fp32 bits back
    q1, k1, values, hub, lse, hub_out, cq, ck, fq, fk, steps_q, steps_k = head_prep(whole, lse_whole, *rope32,
                                                                                    MODEL.dims)
    qs = moment_query_scores(q1, k1, values, hub, cq, fk, lse, hub_out, win, REPEAT, block_s=64, block_t=32,
                             num_warps=4, num_stages=2)
    ks = moment_key_scores(q1, k1, values, hub, ck, fq, lse, hub_out, win, REPEAT, block_t=64, block_s=64,
                           num_warps=4, num_stages=2)
    vs = value_credit(q1, k1, hub, lse, win)
    return head_finish(qs, ks, vs, steps_q, steps_k)


@torch.compile(dynamic=False)
def layer_stacks(n1, o, n2, act, u, prev, lo: int, hi: int):
    """Every block's inputs on this GPU's tokens lo..hi, stacked [L, m, ...] in f32."""
    st = lambda xs: torch.stack([x.reshape(-1, x.shape[-1])[lo:hi] for x in xs]).float()
    return st(n1), st(o), st(n2), st(act), st(u), st(prev)


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    """Each block output's credit: the top credit times the residual scalars above it, [L, m, d]."""
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])  # above[i] = lam_r[i] ... lam_r[last]
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    return credit_top[None] * scale[:, None, None]


@torch.compile(dynamic=False)
def hidden_credits(u, cred, w_out_t):
    """Every block's MLP hidden credit: each unit's central difference through relu^2, read through its output row
    (credit @ W_out^T; w_out_t [L, d, 4d] bf16 is W_out^T)."""
    step = FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1, keepdim=True)), min=1e-6)
    relu2 = lambda z: F.relu(z).square()
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    return slope * torch.bmm(cred.to(BF16), w_out_t).float()


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    """The mixing scalars' gradients (resid, x0) per block, and the credit of x0."""
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


def identity_grad(model, idx, targets, lo, hi, rows_total):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens lo..hi, one head's queries, keys and values over the batch. The ANVIL matrices' gradients stay this
    GPU's partial sums (reduce-scattered later); the rest are all-reduced here. Returns the clean loss of its tokens."""
    world, rank = model.world, model.rank
    t = idx.shape[1]
    n, m = rows_total * t, hi - lo
    rope32 = rotary(t, idx.device)
    cache, loss = forward(model, idx, targets, *rope32, lo, hi)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])[lo:hi]
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1)[lo:hi], idx.reshape(-1)[lo:hi]
    coords = coordinate_change(cache['lm_head'], y)
    model.grad['lm_head'].copy_(cache['in.lm_head'].float().T @ coords)
    credit_top = top_credit(model, stream(f'h{L - 1}.stream'), y)
    gsq, gwide = model.grad['square'], model.grad['wide']
    # every block at once: its inputs on this GPU's tokens stacked [L, m, ...] (f32), its output credit
    xs = layer_stacks([cache[f'h{i}.in.attn'] for i in range(L)], [cache[f'h{i}.in.attn.c_proj'] for i in range(L)],
                      [cache[f'h{i}.in.mlp.c_fc'] for i in range(L)], [cache[f'h{i}.in.mlp.c_proj'] for i in range(L)],
                      [cache[f'h{i}.mlp.c_fc'] for i in range(L)], [cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)],
                      lo, hi)
    n1s, os_, n2s, acts, us, prevs = xs
    cred = block_credits(credit_top, model.lam_r)  # [L, m, d]: the top credit down the residual path
    hidden = hidden_credits(us, cred, model.wide_bf[L:])
    torch.bmm(cred.transpose(1, 2), acts, out=gwide[L:])  # mlp.c_proj, oriented (out, in)
    torch.bmm(n2s.transpose(1, 2), hidden, out=gwide[:L])  # mlp.c_fc
    torch.bmm(os_.transpose(1, 2), cred, out=gsq[3 * L:])  # attn.c_proj
    targets_o = torch.bmm(cred, model.square[3 * L:].transpose(1, 2)).to(BF16)  # attention-output targets (bf16)
    send = pack_heads([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
                      [mine(cache[f'h{i}.values']) for i in range(L)],
                      [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)], list(targets_o.unbind(0)),
                      [cache[f'h{i}.lse'][0].T[lo:hi].contiguous().view(BF16) for i in range(L)])
    got = torch.empty_like(send)
    dist.all_to_all_single(got, send)  # [source GPU (its tokens), L, m, PACK]: this GPU's head over the batch
    whole = got.permute(1, 0, 2, 3).reshape(L, rows_total, t, PACK)
    res = torch.stack([head_credits(whole[i], rope32, window(i)) for i in range(L)])  # [L, 3, n, hd]
    back = res.view(L, 3, world, m, HD).permute(2, 0, 1, 3, 4).contiguous()
    got2 = torch.empty_like(back)
    dist.all_to_all_single(got2, back)  # [source GPU (its head), L, 3, m, hd]
    by_token = got2.permute(1, 2, 3, 0, 4).reshape(L, 3, m, D)
    for j in range(3):  # queries, keys, values
        torch.bmm(n1s.transpose(1, 2), by_token[:, j], out=gsq[j * L:(j + 1) * L])
    for i in range(L):
        if has_ve(i):
            z, ve = mine(cache[f'h{i}.ve_gate']).float(), mine(cache[f'h{i}.ve']).float().view(-1, H, HD)
            vcredit = by_token[i, 2].view(-1, H, HD)
            slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
            model.grad[f'gate{i}'].copy_(n1s[i, :, :GATE_CHANNELS].T @ (slope * (vcredit * ve).sum(-1)))
            model.grad[f've{i}'].zero_().index_add_(0, tokens,
                                                    (vcredit * 2 * torch.sigmoid(z)[..., None]).view(-1, D))
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    dist.all_reduce(model.flat_grad)
    model.flat_grad.div_(n)
    for k in ['wte', *[f've{i}' for i in model.ve]]:  # cast to the parameter's dtype, as the JAX trainer does
        model.grad[k].copy_(model.grad[k].to(BF16).float())
    for name in ('square', 'wide'):  # the ANVIL matrices: summed gradients of this GPU's share
        dist.reduce_scatter_tensor(model.owned[name], model.grad[name])
        model.owned[name].div_(n)
    return loss


# -------------------------------------------------------------------------------------------------------- optimizers

def anvil_betas(t):
    total, warm, cool, engage = (round(k * STEPS / ANVIL_RUN[0]) for k in ANVIL_RUN)
    lo, hi = 0.85, 0.93
    beta = torch.where(t < warm, lo + (hi - lo) * t / max(warm, 1), torch.full_like(t, hi))
    beta = torch.where(t > total - cool, hi - (hi - lo) * (t - (total - cool)) / max(cool, 1), beta)
    engaged = t >= engage
    return beta, torch.where(engaged, 0.85, beta), torch.where(engaged, 0.4385, 1.0)


@torch.compile(dynamic=False)
def anvil_math(w, g, fast, slow, energy, lr, beta, fast_beta, weight, rates):
    """ANVIL II on a stack of wide-oriented matrices [k, rows <= cols]."""
    fast = fast + (1 - fast_beta) * (g - fast)
    slow = slow + (1 - 0.98) * (g - slow)
    x = g + beta * (weight * fast + (1 - weight) * slow - g)
    x = (x / (POLAR_MARGIN * torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-6)).to(BF16)
    for a, b_, c in ANVIL_MAPS:
        sq = x @ x.transpose(1, 2)
        x = a * x + (b_ * sq + c * (sq @ sq)) @ x
    update = x.float()
    energy = energy + (1 - 0.9) * (update.square().mean(1, keepdim=True) - energy)
    gained = update * torch.rsqrt(torch.clamp(energy, min=1e-10))
    update = gained * (torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
                       / torch.clamp(torch.linalg.vector_norm(gained, dim=(1, 2), keepdim=True), min=1e-10))
    decay = ((slow * w) >= 0) * w * (ANVIL_WD * lr)
    return w - lr * rates[:, None, None] * (update + decay), fast, slow, energy


def anvil_update(model):
    beta, fast_beta, weight = anvil_betas(model.step_t)
    lr = model.scale_t * ANVIL_LR
    world, rank = model.world, model.rank
    for name, w in (('square', model.square), ('wide', model.wide)):
        rates = model.rates[name]
        per = w.shape[0] // world
        sl = slice(rank * per, (rank + 1) * per)
        st = model.anvil[name]
        new, fast, slow, energy = anvil_math(w[sl], model.owned[name], st['fast'][sl], st['slow'][sl],
                                             st['energy'][sl], lr, beta, fast_beta, weight, rates[sl])
        st['fast'][sl].copy_(fast)
        st['slow'][sl].copy_(slow)
        st['energy'][sl].copy_(energy)
        dist.all_gather_into_tensor(w, new.contiguous())


@torch.compile(dynamic=False)
def adam_math(p, g, m, v, lr, c1, c2):
    m = BETA1 * m + (1 - BETA1) * g
    v = BETA2 * v + (1 - BETA2) * g.square()
    p = (p.float() - lr * (m / c1) / (torch.sqrt(v) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype)
    return p, m, v


def adam_update(model):
    t = model.step_t + 1
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    for (p, g, lr), m, v in zip(model.adam, model.m, model.v):
        newp, newm, newv = adam_math(p, g, m, v, lr * model.scale_t, c1, c2)
        p.copy_(newp)
        m.copy_(newm)
        v.copy_(newv)


def train_step(model, length):
    """One update, everything on the device: this step's batch from model.rows, this GPU's rows of it."""
    world, rank = model.world, model.rank
    model.scale_t.copy_(torch.clamp((STEPS - model.step_t) / COOLDOWN, max=1.0))  # warmup-stable-decay
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]  # [4, 2049]
    inputs, targets = batch[:, :-1].reshape(-1, length), batch[:, 1:].reshape(-1, length)  # rows cut into pieces
    rows_total = inputs.shape[0]
    if rows_total % world == 0:  # whole rows per GPU
        per = rows_total // world
        idx, tgt = inputs[rank * per:(rank + 1) * per], targets[rank * per:(rank + 1) * per]
        lo, hi = 0, per * length
    else:  # GPUs share a row: each runs it and keeps its part of the tokens
        share = world // rows_total
        idx, tgt = inputs[rank // share:rank // share + 1], targets[rank // share:rank // share + 1]
        part = length // share
        lo, hi = (rank % share) * part, (rank % share + 1) * part
    loss = identity_grad(model, idx, tgt, lo, hi, rows_total)
    anvil_update(model)  # from the weights before this step's Adam (the gradients are this step's)
    adam_update(model)
    model.refresh_bf16()
    model.losses.index_copy_(0, model.step_t.long().view(1), loss.view(1))
    model.step_t.add_(1)


# ---------------------------------------------------------------------------------------------------------- training

MODEL = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--steps', type=int, default=0, help='Stop after this many updates (smoke tests).')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', rank)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    global MODEL
    model = MODEL = Model(device, rank, world)
    rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS).numpy()
    order = np.random.default_rng((SEED + 1, 0)).permutation(len(rows))
    batches = np.stack([rows[np.sort(order[i:i + BATCH])] for i in range(0, len(rows) - BATCH + 1, BATCH)][:STEPS])
    total = args.steps or STEPS
    snapshot = [x.clone() for x in model.tensors()]
    graphs = {}
    for length in (WARMUP_LEN, SEQ):  # warmup on random tokens (model.rows is random until training), then capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                train_step(model, length)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            train_step(model, length)
        graphs[length] = g
    with torch.no_grad():  # training starts from the untouched state, on the real batches
        for x, s in zip(model.tensors(), snapshot):
            x.copy_(s)
        model.refresh_bf16()
        model.rows.copy_(torch.from_numpy(batches.astype(np.int64)))
        model.step_t.zero_()
        model.losses.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    log = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script='train/speedrun/run.py', reference_for='#29', world=world, batch=BATCH, steps=STEPS, seed=SEED,
            seq_warmup=f'{WARMUP_LEN}:{WARMUP_UNTIL}', cooldown=COOLDOWN, lr=LR, betas=[BETA1, BETA2], lr_wte=LR_WTE,
            lr_value_embeds=LR_VE, lr_lambdas=LR_LAMBDAS, anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD), indent=2) + '\n')
        log = (args.output / 'metrics.jsonl').open('w')
    start = time.perf_counter()
    for step in range(total):
        length = WARMUP_LEN if step < WARMUP_UNTIL else SEQ
        graphs[length].replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if rank == 0:
        losses = model.losses.tolist()
        for s in list(range(0, total, 25)) + [total - 1]:
            log.write(json.dumps(dict(step=s + 1, train_loss=round(losses[s], 4))) + '\n')
        log.close()
        torch.save(model.state_dict(), args.output / 'final.pt')
        print(f'trained {total} updates in {elapsed:.2f} s ({elapsed / total * 1000:.2f} ms per update)', flush=True)
    if rank != 0:  # NCCL teardown after captured collectives can hang: leave without it
        os._exit(0)
    if rank == 0:  # utils/eval.py on the final weights, in a fresh process
        env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                                                                 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK')}
        out = subprocess.run([sys.executable, str(root / 'utils' / 'eval.py'), str(args.output / 'final.pt'),
                              '--data-dir', str(args.data_dir)], env=env, capture_output=True, text=True, check=True)
        val = json.loads(out.stdout.strip().splitlines()[-1])['val_loss']
        result = dict(val_loss=val, train_seconds=round(elapsed, 2), steps=total, tokens=total * BATCH * SEQ,
                      reached_target=val <= 4.2)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        os._exit(0)


if __name__ == '__main__':
    main()
