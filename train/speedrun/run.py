"""Track 2: an identity-path phase hands off to Cut; held-out training rows then select a merge of weight averages.

1. Identity phase: 300 updates of 8 rows of #44's 1831-update plan (#44: 660), with #29's identity-path credit
   (full-cycle readout credit, finite MLP responses and #29's attention moments). Both hand off at full rate; these
   300 updates lie inside the plan's 368-update ANVIL momentum ramp, which #44's 660 outlast, and end before the
   late skip, tail average and cooldown that #44 reaches. ANVIL updates the matrices and Adam the rest. Listed under
   "identity" in config.json: each attn.c_proj starts as the norm-matched transpose of its c_v, and each wte row
   starts with 1/256 of its energy in the 32 gate channels; for the first 195 updates, wte rows keep that split,
   the value embeddings' vocabulary-common Adam motion is amplified 4x at fixed energy, the VE gates' head-mean
   credit is pooled over the four VE layers, and each VE layer's centered Adam motion is joined by its partner
   layer's (1-5, 3-7), mapped through the partner's c_proj and back through its own and scaled to half the
   receiving layer's norm, then rescaled to the receiving layer's energy.
2. Cut phase: 1523 updates of 8 of the remaining rows. Each batch takes one row from each of 8 difficulty strata:
   rows ranked by the mean add-one-smoothed unigram surprisal of their target tokens, with the counts taken over the
   Cut rows inside the timer. The Cut estimator:
   - #44's measured full-cycle readout probe;
   - MLP roots at blocks 0, 1, 2, 4, 6 and 7, measured in pairs (0-1, 2-4, 6-7): one forward replay moves both roots
     of a pair along signed Hadamard frames with independent random signs per token (the same on every GPU, whose
     rows differ), through routed MLPs (each token's 128 units with the largest |relu^2(h + r) - relu^2(h - r)| times
     output-weight norm, h the unit's pre-activation and r = chain_sigma times its input-weight norm; #44: the 192
     largest relu^2(h) times output-weight norm) and routed attention at block 3 only (16 strong keys shifted by the
     tail centroid; #44: 32 keys at every later block), and scores it with diagonal readout moments (the diagonal of
     #44's per-token readout curvature). Each root's identity-path prediction (the readout credit times the residual
     scalars above it) is subtracted from the replay and added back to that root's credit, so the partner enters only
     through its deviation from its prediction, with a zero-mean random sign. Blocks 3 and 5 take the next block's
     credit times its residual scalar;
   - mixing scalars: each block's mid credit contracted with the scalar's two inputs, a linear step (#44's Cut
     probed them; #29's identity-path credit contracts them the same way), plus #44's path term (the value- and
     gate-path input credits contracted the same way);
   - the embedding's credit: every block's mid credit summed along its x0 skip (#44: block 0's only), one finite step
     back through the norm;
   - #44's finite query probes and its value and gate paths (exact linear value credit; one finite norm or gate
     step whose credit feeds only linear consumers). The key credit moves each raw key coordinate by a finite step
     through the key norm and RoPE in closed form and scores the softmax's response with its degree-2 moment expansion
     (the moment model of #29's and #44's identity-phase key credit; #44's Cut used the exact single-key
     response).
   ANVIL weight decay is zero after the first 275/1043 of the updates.
3. Weight averages kept over the last 40% of the Cut updates are merged part by part, chosen on 64 held-out rows
   (Nystrom-sketched input-covariance bases; candidates screened on 8 of the rows, finalists scored on all 64).

config.json lists the changes against #44 (qlabs-eng/noise@3343973b) under "additions".

Run on one 8xH100 node, seed 0; measured results are in train/speedrun/log/.
As in #44, train.pt is read (with its pinned-hash check) and split by seeded permutations, and the identity batches
are put on the GPUs, before the timer; apart from the organizers' pinned-hash check, the difficulty order is the only
computation on the rows' contents before the first update, and it is timed. The random-token warmup (the identity
graph, a throwaway Cut, the Cut-estimate graph capture and the merge scorer, whose inputs the held-out rows overwrite
before scoring) is discarded or reset before timed training starts from seed 0; final validation is untimed.
metrics.jsonl writes the Cut records every 16 updates, each with the time of its flush.

    uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from functools import cache
import ast
import contextlib
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import threading
import time
from typing import Optional
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, has_ve, norm, window_mask
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visit_gluon  # noqa: E402  (next to this file)
import visit_gluon_root  # noqa: E402
import query_replay  # noqa: E402
import visit_chain  # noqa: E402
import visit_chain3  # noqa: E402
import query_factored  # noqa: E402
import query_factored_gluon  # noqa: E402
import window_topk  # noqa: E402
import gate_response  # noqa: E402
import implicit_gluon  # noqa: E402
import project_values  # noqa: E402
import moment_kernels  # noqa: E402
import key_gluon  # noqa: E402
import id_kernels  # noqa: E402
import recompute_kernels  # noqa: E402
import value_gluon  # noqa: E402
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate
BEGIN_MARKER = 'NOISE_TRAINING_BEGIN'
END_MARKER = 'NOISE_TRAINING_END'


def _boundary(marker: str) -> None:
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if int(os.environ['RANK']) == 0:
        print(marker, flush=True)


def begin_training() -> None:
    """Call after random-token warmup and rebuilding every training state from seed 0."""
    _boundary(BEGIN_MARKER)


def end_training() -> None:
    """Call after the last update and any weight averaging or training-data merge."""
    _boundary(END_MARKER)


SEED = 0
IDENTITY_UPDATES, IDENTITY_SCHEDULE = 355, 1831  # hand off to Cut after 355 of 1831 planned identity updates
EPS = torch.finfo(torch.float32).eps
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


def model_config():
    return GPTConfig(sequence_len=2048, vocab_size=4096, n_layer=8, n_head=8,
      n_kv_head=8, n_embd=512, window_pattern='SSSL')


@dataclass
class Settings:
    model: GPTConfig = field(default_factory=model_config)
    late_cut_decay_multiplier: float = 0.0  # ANVIL weight-decay factor after the first 275/1043 of Cut updates
    cut_batch: int = 8
    wide_rows: int = 16  # rows per Cut update from wide_from on (each GPU's rows by one estimate replay each)
    wide_from: float = 0.1  # fraction of the Cut's cut_batch-row steps after which updates take wide_rows (1: off)
    wide_lr_scale: float = 1.3  # AdamW's rates in the wide updates, relative to the schedule
    wide_anvil_scale: float = 1.3  # ANVIL's rate in the wide updates, relative to its schedule
    late_rows: int = 32  # rows per Cut update from late_from on (after the wide ones)
    late_from: float = 1.0  # fraction of the Cut's cut_batch-row steps after which updates take late_rows (1: off)
    late_lr_scale: float = 2.0  # AdamW's rates in the late updates, relative to the schedule
    late_anvil_scale: float = 2.0  # ANVIL's rate in the late updates, relative to its schedule
    lr: float = 0.0028
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.55
    cut_warmup: int = 0  # Cut updates over which both optimizers' rates ramp up linearly from the hand-off
    final: float = 0.5
    anvil_lr: float = 0.036
    anvil_wd: float = 2.25
    fast_weight: float = 1.0
    polar_margin: float = 1.1
    telemetry_batch: int = 16
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    fused_chain: bool = True  # chain replays' MLP visits in one kernel each (_fused_visit_kernel)
    compiled_glue: bool = True  # key_context's per-head tensors from one compiled kernel
    batched_keys: bool = True  # key contexts, key credits and query moments batched over layers (per window size)
    batched_routes: bool = True
    route_k_late: int = 64  # with batched_routes: this many routes from block route_split on (0: route_k everywhere)
    route_split: int = 2
    route_schedule: tuple = ()  # with batched_routes: routes per block (powers of two, at most route_k)  # every block's MLP routes from one top-k (exact)
    probe_moments: bool = True  # readout probe from the second-order logit expansion (exact quadratic term)
    query_moments: bool = True  # query credit per raw coordinate from the degree-2 moment model (no probes)
    query_moment_sigma: float = 0.003
    chain_sigma: float = 0.01
    chain_depth_roots: tuple = (4, 6)
    joint_root_groups: tuple = ((4,), (6,))
    route_k: int = 128
    chain_fraction: float = 1.0  # tokens replayed per row (a random subset; unbiased: their deviation scaled up)
    chain_moves: int = 512  # frame directions replayed per root (a random subset of the signed Hadamard's rows)
    chain_tokens: int = 2048
    query_replay_long_layers: tuple = (5,)
    gate_path_sigma: float = 0.01
    query_keys: int = 16
    replay_groups: Optional[tuple] = None  # heads of the root groups that replay attention (None: every group)
    key_sigma: float = 0.003
    merge_screen_rows: int = 8
    merge_screen_global: int = 16
    merge_screen_coordinate: int = 2
    merge_keys: int = 4
    merge_sweeps: int = 0
    emb_scale: float = 750.0
    lambda_scale: float = 15.0
    def __post_init__(self):
        if self.wide_rows % self.cut_batch or self.late_rows % self.cut_batch:
            raise ValueError('wide_rows and late_rows must be multiples of cut_batch')
        flattened_groups = tuple(site for group in self.joint_root_groups for site in group)
        if (tuple(sorted(flattened_groups)) != self.chain_depth_roots
        or any(tuple(sorted(group)) != group or not group for group in self.joint_root_groups)):
            raise ValueError('joint_root_groups must split chain_depth_roots into ascending groups')
        last = self.model.n_layer - 1
        if any(p < last and (p + 1 in group or p + 1 in self.query_replay_long_layers)
               for group in self.joint_root_groups for p in group[1:]):
            raise ValueError('A partner root must be the last block or be followed by a block '
                             'that is neither replayed with attention nor another partner')


@torch.compile
def coordinate_differences(raw, y, sigma):
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    if (d, device, dtype) not in HADAMARDS:
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor.div_(torch.distributed.get_world_size())
    return tensor


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


def query_probe_attention(moved, keys, values, window, count, batch):
    _, length, heads, width = keys.shape
    repeat = lambda z: z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*batch, length, heads, width)
    return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores, directions):
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    """Collapse the full Hadamard central-difference cycle for the separable hidden objective.
    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's
    scalar finite difference. Keep that finite step, including ReLU crossings;
    contract in FP32 instead of rounding every direction's projected response to BF16.
    u and out_weight contain all units.
    """
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope


def norm_inverses(x, delta):
    sq = x.square().sum(-1, keepdim=True)
    if delta.dtype == x.dtype:
        dots = torch.einsum('md,mpd->mp', x, delta)
    else:
        dots = (x.unsqueeze(1) * delta.float()).sum(-1)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    return inv0, torch.rsqrt((sq + 2 * dots + delta.float().square().sum(-1)) / x.shape[-1] + EPS)


def norm_difference(x, delta):
    inv0, inv = norm_inverses(x, delta)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
      / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    if NORM_MOVE_DOT:  # the one-move dot as a fused row reduction (the einsum ran as a batched cuBLAS gemv + copies)
        move = s * v
        d = x.shape[-1]
        sq = x.square().sum(-1, keepdim=True)
        dots = (x * move).sum(-1, keepdim=True)
        moved_sq = move.square().sum(-1, keepdim=True)
        inv0 = torch.rsqrt(sq / d + EPS)
        up = torch.rsqrt((sq + 2 * dots + moved_sq) / d + EPS)
        down = torch.rsqrt((sq - 2 * dots + moved_sq) / d + EPS)
        return ((move * up + x * (up - inv0)) - (-move * down + x * (down - inv0))) / (2 * s)
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


NORM_MOVE_DOT = True  # norm_move's dot products as a fused reduction instead of einsum's batched gemv


@torch.compile(dynamic=False)
def gate_input_responses(xs, weights, kappas, rows, sigma: float):
    """prepared_gate_input_response for several layers in one compiled graph (lists of per-layer tensors)."""
    out = []
    for x, weight, kappa, r in zip(xs, weights, kappas, rows):
        projected = F.linear(r[:, :weight.shape[1]], weight)
        row_square = r.square().mean(-1)
        steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        square = x.square().mean(-1, keepdim=True)
        cross = (x @ r.T) / x.shape[-1]
        base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
        positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * row_square.unsqueeze(0) + EPS)
        negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * row_square.unsqueeze(0) + EPS)
        zp = (base + steps.unsqueeze(-1) * projected.unsqueeze(0)) * positive.unsqueeze(-1)
        zm = (base - steps.unsqueeze(-1) * projected.unsqueeze(0)) * negative.unsqueeze(-1)
        response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
        out.append(response @ r / len(r))
    return out


STACKED_GATES = True  # fused_descending_key_values' value-gate input credits with the layers stacked (one graph, bmm)


@torch.compile(dynamic=False)
def stacked_gate_input_responses(raws, ves, blocks, weights, rows, sigma: float):
    """gate_input_responses with the layers stacked along a batch axis (each GEMM once, as a bmm) and the kappas
    (raws (tokens, width) and ves the layers' value embeddings) and casts inside the graph."""
    x = torch.stack(blocks).float().flatten(1, 2)
    weight = torch.stack(weights).float()
    r = torch.stack(rows)
    heads, k = weight.shape[1:]
    raw = torch.stack(raws)
    kappa = (raw.view(*raw.shape[:2], heads, -1) * torch.stack(ves).float().view(*raw.shape[:2], heads, -1)).sum(-1)
    projected = r[..., :k] @ weight.transpose(1, 2)
    row_square = r.square().mean(-1)
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ r.transpose(1, 2)) / x.shape[-1]
    base = (x[..., :k] @ weight.transpose(1, 2)).unsqueeze(2)
    if GATE_RESPONSE_KERNEL:  # the elementwise core by gate_response (scalars hoisted, fast sigmoid)
        response = gate_response.responses(base.squeeze(2), projected, kappa, steps.squeeze(-1), square.squeeze(-1),
                                           cross, row_square, EPS)
        return (response @ r / r.shape[1]).unbind(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * row_square.unsqueeze(1) + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * row_square.unsqueeze(1) + EPS)
    zp = (base + steps.unsqueeze(-1) * projected.unsqueeze(1)) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * projected.unsqueeze(1)) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(2)).sum(-1) / steps
    return (response @ r / r.shape[1]).unbind(0)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ rows.T) / x.shape[-1]
    direction_square = row_square.unsqueeze(0)
    base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
    direction = projected_rows.unsqueeze(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * direction_square + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * direction_square + EPS)
    zp = (base + steps.unsqueeze(-1) * direction) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * direction) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
    return decode_local_scores(response, rows) / len(rows)


def center_routed_values(values, probabilities, rest):
    origin = (probabilities.unsqueeze(-1) * values.float()).sum(-2) + rest
    return (values.float() - origin.unsqueeze(-2)).to(values.dtype)


TOKEN_MAJOR_VALUES = True  # the replay's projected top values written token-major by one kernel (no transpose copy)


@triton.jit
def _project_values_kernel(C, W, Out, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                           O: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    kk, dd, oo = tl.arange(0, K), tl.arange(0, D), tl.arange(0, O)
    c = tl.load(C + ((h * T + t) * K + kk[:, None]) * D + dd[None, :])
    w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
    out = tl.dot(c, w)
    tl.store(Out + ((t * H + h) * K + kk[:, None]) * O + oo[None, :], out.to(tl.bfloat16))


def project_routed_values_token_major(values, probabilities, rest, w_proj_t):
    """preproject_routed_values(...).transpose(0, 1).contiguous() without the transposed copy."""
    centered = center_routed_values_compiled(values, probabilities, rest).contiguous()
    if BLOCKED_PROJECTION:  # 4 tokens x 256 columns per program: each head's weight slice read once (bitwise)
        return project_values.project_values(centered, w_proj_t)
    heads, tokens, count, width = centered.shape
    output_width = w_proj_t.shape[1]
    out = torch.empty((tokens, heads, count, output_width), device=values.device, dtype=centered.dtype)
    _project_values_kernel[(tokens, heads)](centered, w_proj_t.to(centered.dtype).contiguous(), out, T=tokens, H=heads,
                                            K=count, D=width, O=output_width, num_warps=4)
    return out


center_routed_values_compiled = torch.compile(lambda v, p, r: center_routed_values(v, p, r), dynamic=False)


@torch.compile(dynamic=False)
def preproject_routed_values(values, probabilities, rest, w_proj_t):
    centered = center_routed_values(values, probabilities, rest)
    heads, tokens, count, width = centered.shape
    output_width = w_proj_t.shape[1]
    projected = torch.bmm(centered.reshape(heads, tokens * count, width),
      w_proj_t.to(centered.dtype).reshape(heads, width, output_width))
    return projected.view(heads, tokens, count, output_width)


def gathered_values_dense(table, correction, rows):
    """A GATHERED_VALUES triple as projected values (tokens, heads * count, O), for consumers other than chain3."""
    heads = correction.shape[1] // 2
    shift = (correction[:, :heads].float() + correction[:, heads:].float()).repeat_interleave(rows.shape[1] // heads, 1)
    return (table[rows.long()].float() + shift).to(table.dtype)


def token_slice(z, local):
    """z[local] along tokens; a GATHERED_VALUES triple keeps its per-key table whole."""
    if isinstance(z, tuple):
        return z[0], z[1][local], z[2][local]
    return z[local]


@triton.jit
def _fused_query_replay(Raw, R0, Cos, Sin, Inv, Shift, KeysPtr, Prob, Out,
    P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BP: tl.constexpr, BK: tl.constexpr, EPS_: tl.constexpr):
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    pp = tl.program_id(0) * BP + tl.arange(0, BP)
    hh = tl.arange(0, D // 2)
    dd = tl.arange(0, D)
    base = ((token * P + pp[:, None]) * H + head) * D
    a = tl.load(Raw + base + hh[None, :], pp[:, None] < P, 0).to(tl.float32)
    b = tl.load(Raw + base + D // 2 + hh[None, :], pp[:, None] < P, 0).to(tl.float32)
    c = tl.load(Cos + token * (D // 2) + hh)
    s = tl.load(Sin + token * (D // 2) + hh)
    r0a = tl.load(R0 + (token * H + head) * D + hh)
    r0b = tl.load(R0 + (token * H + head) * D + D // 2 + hh)
    ra = a * c[None, :] + b * s[None, :]
    rb = -a * s[None, :] + b * c[None, :]
    g = tl.load(Inv + token * P + pp, pp < P, 0)[:, None]
    h = tl.load(Shift + token * P + pp, pp < P, 0)[:, None]
    ra, rb = ra * g + h * r0a[None, :], rb * g + h * r0b[None, :]
    sq = tl.sum(r0a * r0a, 0) + tl.sum(r0b * r0b, 0)
    dots = tl.sum(ra * r0a[None, :], 1) + tl.sum(rb * r0b[None, :], 1)
    moved = tl.sum(ra * ra, 1) + tl.sum(rb * rb, 1)
    inv0 = tl.math.rsqrt(sq / D + EPS_)
    inv = tl.math.rsqrt((sq + 2 * dots + moved) / D + EPS_)
    shift = (inv - inv0)[:, None]
    dqa = (ra * inv[:, None] + r0a[None, :] * shift).to(tl.bfloat16)
    dqb = (rb * inv[:, None] + r0b[None, :] * shift).to(tl.bfloat16)
    dq = tl.reshape(tl.trans(tl.join(dqa, dqb), (0, 2, 1)), (BP, D))
    rr = tl.arange(0, BK)
    offsets = ((token * H + head) * K + rr[:, None]) * D + dd[None, :]
    keys = tl.load(KeysPtr + offsets, rr[:, None] < K, 0)
    prob = tl.load(Prob + (token * H + head) * K + rr, rr < K, 0)
    ds = tl.dot(dq, tl.trans(keys)).to(tl.bfloat16).to(tl.float32) * (D ** -.5)
    u = tl.exp(ds) - 1.0
    z = tl.sum(prob[None, :] * u, 1)
    weights = prob[None, :] * u / (1.0 + z[:, None])
    tl.store(Out + ((token * P + pp[:, None]) * H + head) * K + rr[None, :],
      weights.to(tl.bfloat16), (pp[:, None] < P) & (rr[None, :] < K))


@torch.library.custom_op('noise_record::fused_query_weights', mutates_args=())
def fused_query_weights(raw: torch.Tensor, r0: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor, shift: torch.Tensor,
    keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    m, probes = raw.shape[:2]
    heads, width = r0.shape[1:]
    count = keys.shape[2]
    out = torch.empty((m, probes, heads * count), device=raw.device, dtype=torch.bfloat16)
    _fused_query_replay[(triton.cdiv(probes, 32), m * heads)](
      raw.contiguous(), r0.float().contiguous(), cos.float().contiguous(),
      sin.float().contiguous(), inv.float().contiguous(), shift.float().contiguous(),
      keys.contiguous(), prob.contiguous(), out,
      probes, heads, width, count, 32, max(16, triton.next_power_of_2(count)), EPS, num_warps=4)
    return out


@fused_query_weights.register_fake
def _(raw, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((raw.shape[0], raw.shape[1], keys.shape[1] * keys.shape[2]),
      device=raw.device, dtype=torch.bfloat16)


def fused_projected_response(raw, r0, cos, sin, inv, shift, keys, projected_values, prob, add_to=None,
    raw_bf16=False):
    weights = fused_query_weights(raw, r0, cos, sin, inv, shift, keys, prob)
    if raw_bf16 == 'parts':  # the response's two factors, multiplied inside the next visit
        return weights, projected_values.flatten(1, 2)
    if raw_bf16:
        return torch.bmm(weights, projected_values.flatten(1, 2))
    if add_to is not None:  # delta + response in one GEMM epilogue, bf16 out
        return torch.baddbmm(add_to, weights, projected_values.flatten(1, 2))
    return torch.bmm(weights, projected_values.flatten(1, 2)).float()


def query_change_routed(x, delta, pos, w_q, top_keys, top_values, top_p, cos, sin, query_inv, add_to=None,
    raw_bf16=False, factored=None):
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None], sin[pos][:, None]
    q0 = (norm(x) @ w_q.T).view(m, heads, width)
    r0 = torch.cat([q0[..., :width // 2] * c + q0[..., width // 2:] * s,
      q0[..., :width // 2] * -s + q0[..., width // 2:] * c], -1)
    inv0, inv = query_inv
    gathered = isinstance(top_values, tuple)  # GATHERED_VALUES: the triple kept for chain3 (weights-only paths)
    if factored is not None and raw_bf16 == 'parts':  # delta = the root output: q change from its routed factors
        change, steps, scale, wo, routes, basis = factored
        if GLUON_FACTORED:  # A_t transposed and w_q's rows interleaved per head: its own factors op
            at, f = (query_factored_gluon.query_factors_t_gathered if FACTOR_GATHER else
                     query_factored_gluon.query_factors_t)(wo, routes, basis, w_q)
            replay = (query_factored_gluon.fused_query_replay_factored_rot if QF_ROTATED_KEYS else
                      query_factored_gluon.fused_query_replay_factored_gluon)
            weights = replay(change, steps, scale, at, f, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_p)
        else:
            a, f = query_factored.query_factors(wo, routes, basis, w_q)
            weights = query_factored.fused_query_replay_factored(change, steps, scale, a, f, r0, cos[pos], sin[pos],
                                                                 inv, inv / inv0 - 1, top_keys, top_p)
        return weights, top_values if gathered else top_values.flatten(1, 2)
    if FUSED_QUERY_REPLAY and raw_bf16 == 'parts':  # q projection and replay in one kernel, only weights stored
        weights = query_replay.fused_query_replay(delta, w_q, r0, cos[pos], sin[pos], inv, inv / inv0 - 1,
                                                  top_keys, top_p)
        return weights, top_values if gathered else top_values.flatten(1, 2)
    if gathered:
        top_values = gathered_values_dense(*top_values).unflatten(1, (heads, -1))
    if QUERY_FP8:  # rowwise-scaled e4m3 operands, bf16 out
        flat = delta.reshape(-1, delta.shape[-1])
        sa = (flat.abs().amax(-1, keepdim=True).float() / 448).clamp_min(1e-12)
        sb = (w_q.abs().amax(-1, keepdim=True).float() / 448).clamp_min(1e-12)
        a8 = (flat.float() / sa).to(torch.float8_e4m3fn)
        b8 = (w_q.float() / sb).to(torch.float8_e4m3fn)
        raw = torch._scaled_mm(a8, b8.T, scale_a=sa, scale_b=sb.T, out_dtype=torch.bfloat16).view(
            m, probes, heads, width)
    else:
        raw = (delta @ w_q.T.to(torch.bfloat16)).view(m, probes, heads, width)
    return fused_projected_response(raw, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_values, top_p,
      add_to, raw_bf16)


@torch.compile(dynamic=False)
def key_context_tensors(q, k, v, ve, z, target, cos, sin, heads: int):
    """key_context's per-head tensors (heads, tokens, head_dim), contiguous: RoPE'd unit queries and keys, raw keys,
    values with the gated value embedding, and the attention output's credit."""
    t = q.shape[0]
    hd = q.shape[1] // heads
    split = lambda z_: z_.float().view(t, heads, hd).transpose(0, 1)
    rot = lambda z_: torch.cat([z_[..., :hd // 2] * cos + z_[..., hd // 2:] * sin,
                                -z_[..., :hd // 2] * sin + z_[..., hd // 2:] * cos], -1)
    unit = lambda z_: z_ * torch.rsqrt(z_.square().mean(-1, keepdim=True) + EPS)
    q_raw, k_raw, values = split(q), split(k), split(v)
    if ve is not None:
        values = values + 2 * torch.sigmoid(z.float().view(t, heads)).T.unsqueeze(-1) * split(ve)
    return (unit(rot(q_raw)).contiguous(), unit(rot(k_raw)).contiguous(), k_raw.contiguous(), values.contiguous(),
            split(target).contiguous())


@torch.compile(dynamic=False)
def key_context_group(qs, ks, vs, ves, gates, targets, cos, sin, heads: int):
    """key_context_tensors for several layers, stacked along the head axis in the same compiled graph."""
    t = qs[0].shape[0]
    hd = qs[0].shape[1] // heads
    split = lambda z_: z_.float().view(t, heads, hd).transpose(0, 1)
    rot = lambda z_: torch.cat([z_[..., :hd // 2] * cos + z_[..., hd // 2:] * sin,
                                -z_[..., :hd // 2] * sin + z_[..., hd // 2:] * cos], -1)
    unit = lambda z_: z_ * torch.rsqrt(z_.square().mean(-1, keepdim=True) + EPS)
    out = [], [], [], [], []
    for q, k, v, ve, z, target in zip(qs, ks, vs, ves, gates, targets):
        k_raw, values = split(k), split(v)
        if ve is not None:
            values = values + 2 * torch.sigmoid(z.float().view(t, heads)).T.unsqueeze(-1) * split(ve)
        for acc, part in zip(out, (unit(rot(split(q))), unit(rot(k_raw)), k_raw, values, split(target))):
            acc.append(part)
    return tuple(torch.cat(acc, 0) for acc in out)


EARLY_MOMENT_INPUTS = True  # with FUSED_GLUE: query_moment_inputs' outputs formed in key_context_group's graph


@torch.compile(dynamic=False)
def key_context_group_moments(qs, ks, vs, ves, gates, targets, cos, sin, heads: int, sigma: float):
    """key_context_group, and query_moment_inputs' outputs for the stacked layers, in one compiled graph (the bf16
    copies written while the fp32 tensors are formed, not reread)."""
    qp, kp, k_raw, values, hub = key_context_group(qs, ks, vs, ves, gates, targets, cos, sin, heads)
    q_out = torch.cat(qs, 1)
    return (qp, kp, k_raw, values, hub), query_moment_inputs(q_out, qp, kp, values, hub, sigma)


@torch.compile(dynamic=False)
def query_moment_inputs(q_out, qp, kp, values, hub, sigma: float):
    heads, t, hd = qp.shape
    q_raw = q_out.float().view(t, heads, hd).transpose(0, 1)
    sq = q_raw.square().sum(-1)
    steps = sigma * (sq / hd).sqrt().clamp(min=1e-6)
    b16 = lambda z_: z_.to(torch.bfloat16).contiguous()
    return b16(qp), b16(kp), b16(values), b16(hub), q_raw.contiguous().unsqueeze(0), sq.contiguous(), steps.contiguous()


@torch.compile(dynamic=False)
def response_scores(hpre, radius, out_norm):
    plus = F.relu(hpre + radius).square()
    minus = F.relu(hpre - radius).square()
    return (plus - minus).abs() * out_norm


@torch.compile(dynamic=False)
def norm_input_credit(x, g, sigma, H):
    width = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    step = sigma * (sq / width).sqrt().clamp(min=1e-6)
    xh, gh, gx = x @ H.T, g @ H.T, (g * x).sum(-1, keepdim=True)
    inv = torch.rsqrt(sq / width + EPS)
    inv_up = torch.rsqrt((sq + 2 * step * xh + step.square() * width) / width + EPS)
    inv_down = torch.rsqrt((sq - 2 * step * xh + step.square() * width) / width + EPS)
    up = step * gh * inv_up + gx * (inv_up - inv)
    down = -step * gh * inv_down + gx * (inv_down - inv)
    return ((up - down) / (2 * step)) @ H / width


@torch.no_grad()
@torch.compile(dynamic=False)
def readout_curvature_diagonal(weight, c2):
    """Diagonal of each token's readout curvature W^T diag(c2) W; the readout moments keep only this diagonal."""
    return c2 @ weight.square()


@torch.no_grad()
def head_moments(xL, base, weight):
    width = xL.shape[-1]
    inv0 = torch.rsqrt(xL.square().sum(-1, keepdim=True) / width + EPS)
    t = torch.tanh(base * inv0 / 15)
    f, f1 = 15 * t, 1 - t.square()
    f2 = -(2 / 15) * t * f1
    prob = torch.softmax(f, -1)
    c1, c2 = prob * f1, prob * (f2 + f1.square()) / 2
    b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
    mu1, nu = c1 @ weight, (c2 * base) @ weight
    second = readout_curvature_diagonal(weight, c2)
    return b1, b2, mu1, nu, second


@triton.jit
def _route_gate_kernel(Delta, Weight, Routes, Clean, Inv0, Inv, Out, Steps, Projection,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    rr = tl.program_id(1) * BN + tl.arange(0, BN)
    chosen = tl.load(Routes + token * R + rr, rr < R, 0)
    acc = tl.zeros((BM, BN), tl.float32)
    if ROOT:
        q = tl.load(Projection + chosen[:, None] * P + pp[None, :],
        (rr[:, None] < R) & (pp[None, :] < P), 0)
        step16 = tl.load(Steps + token).to(tl.bfloat16).to(tl.float32)
        acc = tl.trans(q) * step16
    else:
        for k0 in range(0, D, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(Delta + token * S0 + pp[:, None] * S1 + kk[None, :] * S2,
          (pp[:, None] < P) & (kk[None, :] < D), 0).to(tl.bfloat16)
            b = tl.load(Weight + chosen[:, None] * D + kk[None, :],
          (rr[:, None] < R) & (kk[None, :] < D), 0).to(tl.bfloat16)
            acc += tl.dot(a, tl.trans(b))
    moved = acc.to(tl.bfloat16).to(tl.float32)
    inv = tl.load(Inv + token * P + pp, pp < P, 0)
    inv0 = tl.load(Inv0 + token)
    h = tl.load(Clean + token * R + rr, rr < R, 0).to(tl.float32)
    dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    changed = ((up - down) * (up + down)).to(tl.bfloat16)
    tl.store(Out + (token * P + pp[:, None]) * R + rr[None, :], changed,
      (pp[:, None] < P) & (rr[None, :] < R))


@triton.jit
def _route_output_parts(Change, Weight, Routes, Delta, Steps, Basis,
    token, pp, dd, P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INJECT: tl.constexpr):
    acc = tl.zeros((BM, BN), tl.float32)
    for r0 in range(0, R, BK):
        rr = r0 + tl.arange(0, BK)
        chosen = tl.load(Routes + token * R + rr, rr < R, 0)
        a = tl.load(Change + (token * P + pp[:, None]) * R + rr[None, :],
        (pp[:, None] < P) & (rr[None, :] < R), 0)
        b = tl.load(Weight + chosen[:, None] * D + dd[None, :],
        (rr[:, None] < R) & (dd[None, :] < D), 0).to(tl.bfloat16)
        acc += tl.dot(a, b)
    moved = acc.to(tl.bfloat16).to(tl.float32)
    if ROOT:
        step = tl.load(Steps + token)
        basis = tl.load(Basis + pp[:, None] * D + dd[None, :],
        (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
        old = step * basis
    else:
        old = tl.load(Delta + token * S0 + pp[:, None] * S1 + dd[None, :] * S2,
        (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
        if INJECT:
            step = tl.load(Steps + token)
            frame = tl.load(Basis + pp[:, None] * D + dd[None, :],
          (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
            old = old + step * frame
    return old, moved


@triton.jit
def _route_output_kernel(Change, Weight, Routes, Delta, Out, Steps, Basis, Scale,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, ROOT, BM, BN, BK, False)
    result = (old + moved).to(tl.bfloat16)
    result = (result.to(tl.float32) * tl.load(Scale)).to(tl.bfloat16)
    tl.store(Out + (token * P + pp[:, None]) * D + dd[None, :], result,
      (pp[:, None] < P) & (dd[None, :] < D))


@torch.library.custom_op('noise_record::route_mlp', mutates_args=())
def route_mlp(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor,
    inv0: torch.Tensor, inv: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean = routes.contiguous(), clean.contiguous()
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=delta.device, dtype=torch.bfloat16)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      delta, fc, routes, clean, inv0.contiguous(), inv.contiguous(), change, delta, delta,
      p, d, r, *delta.stride(), False, BM=64, BN=min(128, r), BK=64, num_warps=8, num_stages=3)
    _route_output_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, delta, out, delta, delta, out_scale, p, d, r, *delta.stride(),
      False, BM=64, BN=128, BK=r, num_warps=8, num_stages=2)
    return out


@route_mlp.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, out_scale):
    return torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16)


def mlp_statistics_storage(n, p, ns, nb, device):
    # Stored as (token, statistic, tile, probe) and viewed as (token, probe, statistic, tile).
    return torch.empty((n, ns, nb, p), device=device, dtype=torch.float32).permute(0, 3, 1, 2)


@triton.jit
def _mlp_output_stats_kernel(Change, Weight, Routes, Delta, Out, Stats, X, Lambda,
    Target, Mu, Nu, Diag, Steps, Basis,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    MODE: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr, INJECT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, False, BM, BN, BK, INJECT)
    value = (old + moved).to(tl.bfloat16).to(tl.float32)
    if MODE == 1:
        value = value * tl.load(Lambda).to(tl.float32)
        tl.store(Out + (token * P + pp[:, None]) * D + dd[None, :], value.to(tl.bfloat16),
          (pp[:, None] < P) & (dd[None, :] < D))
    xx = tl.load(X + token * D + dd, dd < D, 0).to(tl.float32)
    pos = (token * NS * NB + tl.program_id(1)) * P + pp
    ss: tl.constexpr = NB * P
    tl.store(Stats + pos, tl.sum(value * xx[None, :], 1), pp < P)
    tl.store(Stats + pos + ss, tl.sum(value * value, 1), pp < P)
    if MODE == 2:
        target = tl.load(Target + token * D + dd, dd < D, 0).to(tl.float32)
        mu = tl.load(Mu + token * D + dd, dd < D, 0).to(tl.float32)
        nu = tl.load(Nu + token * D + dd, dd < D, 0).to(tl.float32)
        diag = tl.load(Diag + token * D + dd, dd < D, 0).to(tl.float32)
        tl.store(Stats + pos + 2 * ss, tl.sum(value * target[None, :], 1), pp < P)
        tl.store(Stats + pos + 3 * ss, tl.sum(value * mu[None, :], 1), pp < P)
        tl.store(Stats + pos + 4 * ss, tl.sum(value * nu[None, :], 1), pp < P)
        tl.store(Stats + pos + 5 * ss, tl.sum(value * value * diag[None, :], 1), pp < P)


def launch_mlp_output_stats(out, delta, fc, wo, routes, clean, inv0, inv, x, scale, target, mu, nu, diag, mode,
    steps, frame, packed):
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean = routes.contiguous(), clean.contiguous()
    ns, nb = (2 if mode == 1 else 6), triton.cdiv(d, 64 if mode == 1 else 128)
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    stats = mlp_statistics_storage(n, p, ns, nb, delta.device)
    tile = min(64, r)
    inject = steps is not None
    steps = delta if steps is None else steps.contiguous()
    frame = delta if frame is None else frame.contiguous()
    gate_input = delta if packed is None else packed
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      gate_input, fc, routes, clean, inv0.contiguous(), inv.contiguous(), change, delta, delta,
      p, d, r, *gate_input.stride(), False, BM=64, BN=min(128, r), BK=64, num_warps=8, num_stages=3)
    _mlp_output_stats_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, delta, out, stats, x.contiguous(), scale, target.contiguous(),
      mu.contiguous(), nu.contiguous(), diag.contiguous(), steps, frame, p, d, r, *delta.stride(),
      mode, ns, nb, inject, BM=64, BN=64 if mode == 1 else 128, BK=r if mode == 1 else tile,
      num_warps=4, num_stages=2 if mode == 1 else 3, enable_fp_fusion=False)
    return stats


@torch.library.custom_op('noise_record::mlp_output_stats', mutates_args=())
def mlp_output_stats(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor,
    inv: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor], packed: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Routed MLP output scaled by the next block's residual scalar, with the partials of the next block's norm."""
    out = torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16)
    stats = launch_mlp_output_stats(out, delta, fc, wo, routes, clean, inv0, inv, x, scale, x, x, x, x, 1,
      steps, frame, packed)
    return out, stats


@mlp_output_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, x, scale, steps, frame, packed):
    n, p, d = delta.shape
    return (torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), delta.device))


@torch.library.custom_op('noise_record::mlp_readout_stats', mutates_args=())
def mlp_readout_stats(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor,
    x: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, diag: torch.Tensor,
    steps: Optional[torch.Tensor], frame: Optional[torch.Tensor], packed: Optional[torch.Tensor]) -> torch.Tensor:
    """Last block's routed MLP: partials of the readout moments only (the output itself is not written)."""
    return launch_mlp_output_stats(delta, delta, fc, wo, routes, clean, inv0, inv, x, x, target, mu, nu, diag, 2,
      steps, frame, packed)


@mlp_readout_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, x, target, mu, nu, diag, steps, frame, packed):
    n, p, d = delta.shape
    return mlp_statistics_storage(n, p, 6, triton.cdiv(d, 128), delta.device)


def root_norm_inverses(x, steps, basis):
    d = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    dots = steps * (x @ basis.T)
    moved_sq = steps.square() * basis.square().sum(-1).unsqueeze(0)
    return torch.rsqrt(sq / d + EPS), torch.rsqrt((sq + 2 * dots + moved_sq) / d + EPS)


@torch.library.custom_op('noise_record::route_mlp_root', mutates_args=())
def route_mlp_root(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor,
    wo: torch.Tensor, routes: torch.Tensor, clean: torch.Tensor,
    inv0: torch.Tensor, inv: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    n, d, p = steps.shape[0], wo.shape[1], basis.shape[0]
    r = routes.shape[1]
    steps, basis = steps.contiguous(), basis.contiguous()
    routes, clean = routes.contiguous(), clean.contiguous()
    change = torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      steps, steps, routes, clean, inv0.contiguous(), inv.contiguous(), change, steps, projection,
      p, d, r, 0, 0, 0, True, BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
    _route_output_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, steps, out, steps, basis, out_scale, p, d, r, 0, 0, 0,
      True, BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
    return out


@route_mlp_root.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, out_scale):
    return torch.empty((steps.shape[0], basis.shape[0], wo.shape[1]), device=wo.device, dtype=torch.bfloat16)


@triton.jit
def _injected_norm_kernel(X, Delta, Steps, Basis, Inv0, Inv, Mixed,
    P: tl.constexpr, D: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    BM: tl.constexpr, BD: tl.constexpr, EPSILON: tl.constexpr):
    token = tl.program_id(0)
    pp = tl.program_id(1) * BM + tl.arange(0, BM)
    dd = tl.arange(0, BD)
    x = tl.load(X + token * D + dd, dd < D, 0).to(tl.float32)
    delta = tl.load(Delta + token * S0 + pp[:, None] * S1 + dd[None, :] * S2,
      (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
    frame = tl.load(Basis + pp[:, None] * D + dd[None, :],
      (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
    step = tl.load(Steps + token)
    moved = delta + step * frame
    tl.store(Mixed + (token * P + pp[:, None]) * D + dd[None, :], moved.to(tl.bfloat16),
      (pp[:, None] < P) & (dd[None, :] < D))
    square = tl.sum(x * x, 0)
    dot = tl.sum(moved * x[None, :], 1)
    move_square = tl.sum(moved * moved, 1)
    inv = tl.rsqrt((square + 2 * dot + move_square) / D + EPSILON)
    tl.store(Inv + token * P + pp, inv, pp < P)
    if tl.program_id(1) == 0:
        tl.store(Inv0 + token, tl.rsqrt(square / D + EPSILON))


@torch.library.custom_op('noise_record::injected_norm_packed', mutates_args=())
def injected_norm_packed(x: torch.Tensor, delta: torch.Tensor, steps: torch.Tensor,
    basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, p, d = delta.shape
    x, steps, basis = x.contiguous(), steps.contiguous(), basis.contiguous()
    inv0 = torch.empty((n, 1), device=x.device, dtype=torch.float32)
    inv = torch.empty((n, p), device=x.device, dtype=torch.float32)
    mixed = torch.empty((n, p, d), device=x.device, dtype=torch.bfloat16)
    _injected_norm_kernel[(n, triton.cdiv(p, 8))](
      x, delta, steps, basis, inv0, inv, mixed, p, d, *delta.stride(),
      BM=8, BD=triton.next_power_of_2(d), EPSILON=EPS, num_warps=4, enable_fp_fusion=False)
    return inv0, inv, mixed


@injected_norm_packed.register_fake
def _(x, delta, steps, basis):
    return (torch.empty((delta.shape[0], 1), device=x.device, dtype=torch.float32),
      torch.empty(delta.shape[:2], device=x.device, dtype=torch.float32),
      torch.empty(delta.shape, device=x.device, dtype=torch.bfloat16))


@triton.jit
def _fused_visit_kernel(Delta, Steps, Frame, X, Fc, Wo, Routes, Clean, Scale, Out,
    Target, Mu, Nu, Diag, XL, Stats, Add,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr, INJECT: tl.constexpr, MODE: tl.constexpr,
    ADD: tl.constexpr, HADAMARD: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, EPSILON: tl.constexpr):
    """One routed MLP visit of a chain replay for BM moves of one token, the moved-stream change kept on chip: the
    gate GEMM over the routed c_fc rows accumulates the norm sums of the moved block input as it streams delta (plus
    an injected partner move), the relu^2 change stays in registers for the output GEMM. MODE 1 writes the change
    of the next block's input scaled by its residual scalar; MODE 2 (the last block) writes the readout-moment
    partial sums of the moved final stream instead."""
    token = tl.program_id(1)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    live = pp < P
    rr = tl.arange(0, R)
    chosen = tl.load(Routes + token * R + rr)
    acc = tl.zeros((BM, R), tl.float32)
    sq_x = tl.zeros((1,), tl.float32)
    dots = tl.zeros((BM,), tl.float32)
    moved_sq = tl.zeros((BM,), tl.float32)
    if INJECT:
        step = tl.load(Steps + token)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(Delta + (token * P + pp[:, None]) * D + kk[None, :], live[:, None], 0).to(tl.float32)
        if ADD:
            d += tl.load(Add + (token * P + pp[:, None]) * D + kk[None, :], live[:, None], 0).to(tl.float32)
        if INJECT:
            if HADAMARD:
                d += step * _signed_hadamard(Frame, pp, kk)
            else:
                d += step * tl.load(Frame + pp[:, None] * D + kk[None, :], live[:, None], 0)
        x = tl.load(X + token * D + kk)
        sq_x += tl.sum(x * x, 0)
        dots += tl.sum(d * x[None, :], 1)
        moved_sq += tl.sum(d * d, 1)
        w = tl.load(Fc + chosen[:, None] * D + kk[None, :])
        acc = tl.dot(d.to(tl.bfloat16), tl.trans(w), acc)
    inv0 = tl.math.rsqrt(sq_x / D + EPSILON)
    inv = tl.math.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
    h = tl.load(Clean + token * R + rr).to(tl.float32)
    dh = inv[:, None] * acc.to(tl.bfloat16).to(tl.float32) + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    change = ((up - down) * (up + down)).to(tl.bfloat16)
    if MODE != 2:
        scale = tl.load(Scale).to(tl.float32)
        q0 = tl.zeros((BM,), tl.float32)
        q1 = tl.zeros((BM,), tl.float32)
    else:
        s0 = tl.zeros((BM,), tl.float32)
        s1 = tl.zeros((BM,), tl.float32)
        s2 = tl.zeros((BM,), tl.float32)
        s3 = tl.zeros((BM,), tl.float32)
        s4 = tl.zeros((BM,), tl.float32)
        s5 = tl.zeros((BM,), tl.float32)
    for n0 in range(0, D, BN):
        nn = n0 + tl.arange(0, BN)
        w = tl.load(Wo + chosen[:, None] * D + nn[None, :])
        moved = tl.dot(change, w).to(tl.bfloat16).to(tl.float32)
        old = tl.load(Delta + (token * P + pp[:, None]) * D + nn[None, :], live[:, None], 0).to(tl.float32)
        if ADD:
            old += tl.load(Add + (token * P + pp[:, None]) * D + nn[None, :], live[:, None], 0).to(tl.float32)
        if INJECT:
            if HADAMARD:
                old += step * _signed_hadamard(Frame, pp, nn)
            else:
                old += step * tl.load(Frame + pp[:, None] * D + nn[None, :], live[:, None], 0)
        value = (old + moved).to(tl.bfloat16).to(tl.float32)
        if MODE != 2:
            result = (value * scale).to(tl.bfloat16)
            tl.store(Out + (token * P + pp[:, None]) * D + nn[None, :], result, live[:, None])
            if MODE == 3:
                stored = result.to(tl.float32)
                q0 += tl.sum(stored * tl.load(XL + token * D + nn)[None, :], 1)
                q1 += tl.sum(stored * stored, 1)
        else:
            xl = tl.load(XL + token * D + nn)
            s0 += tl.sum(value * xl[None, :], 1)
            s1 += tl.sum(value * value, 1)
            s2 += tl.sum(value * tl.load(Target + token * D + nn)[None, :], 1)
            s3 += tl.sum(value * tl.load(Mu + token * D + nn)[None, :], 1)
            s4 += tl.sum(value * tl.load(Nu + token * D + nn)[None, :], 1)
            s5 += tl.sum(value * value * tl.load(Diag + token * D + nn)[None, :], 1)
    if MODE == 3:
        tl.store(Stats + (token * P + pp) * 2, q0, live)
        tl.store(Stats + (token * P + pp) * 2 + 1, q1, live)
    if MODE == 2:
        base = (token * P + pp) * 6
        tl.store(Stats + base, s0, live)
        tl.store(Stats + base + 1, s1, live)
        tl.store(Stats + base + 2, s2, live)
        tl.store(Stats + base + 3, s3, live)
        tl.store(Stats + base + 4, s4, live)
        tl.store(Stats + base + 5, s5, live)


@triton.jit
def _signed_hadamard(Signs, pp, dd):
    """Entries (pp, dd) of the signed Sylvester Hadamard frame H * signs: signs[d] * (-1)^popcount(p & d)."""
    bits = pp[:, None] & dd[None, :]
    bits ^= bits >> 8
    bits ^= bits >> 4
    bits ^= bits >> 2
    bits ^= bits >> 1
    return (1 - 2 * (bits & 1)).to(tl.float32) * tl.load(Signs + dd)[None, :]


@triton.jit
def _fused_root_kernel(Steps, Basis, Proj, Wo, Routes, Clean, Inv0, Inv, Scale, Out, NextX, Stats,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr, QSTATS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    HADAMARD: tl.constexpr):
    """A chain replay's root visit for BM moves of one token: each move is the signed step along a frame
    direction, so the routed gate change is read from the precomputed projection of the c_fc rows on the frame
    and the output GEMM adds the move itself; QSTATS also writes the next block's query-norm sums."""
    token = tl.program_id(1)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    live = pp < P
    rr = tl.arange(0, R)
    chosen = tl.load(Routes + token * R + rr)
    q = tl.load(Proj + chosen[None, :] * P + pp[:, None], live[:, None], 0)
    step = tl.load(Steps + token)
    moved = (q * step.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = tl.load(Inv + token * P + pp, live, 0)
    inv0 = tl.load(Inv0 + token)
    h = tl.load(Clean + token * R + rr).to(tl.float32)
    dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    change = ((up - down) * (up + down)).to(tl.bfloat16)
    scale = tl.load(Scale).to(tl.float32)
    q0 = tl.zeros((BM,), tl.float32)
    q1 = tl.zeros((BM,), tl.float32)
    for n0 in range(0, D, BN):
        nn = n0 + tl.arange(0, BN)
        out_moved = tl.dot(change, tl.load(Wo + chosen[:, None] * D + nn[None, :])).to(tl.bfloat16).to(tl.float32)
        if HADAMARD:
            old = step * _signed_hadamard(Basis, pp, nn)
        else:
            old = step * tl.load(Basis + pp[:, None] * D + nn[None, :], live[:, None], 0)
        result = ((old + out_moved).to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(Out + (token * P + pp[:, None]) * D + nn[None, :], result, live[:, None])
        if QSTATS:
            stored = result.to(tl.float32)
            q0 += tl.sum(stored * tl.load(NextX + token * D + nn)[None, :], 1)
            q1 += tl.sum(stored * stored, 1)
    if QSTATS:
        tl.store(Stats + (token * P + pp) * 2, q0, live)
        tl.store(Stats + (token * P + pp) * 2 + 1, q1, live)


@torch.library.custom_op('noise_record::fused_root_visit', mutates_args=())
def fused_root_visit(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor, scale: torch.Tensor,
    next_x: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """route_mlp_root (and its query-stats variant when next_x is given) in one kernel."""
    n, p, d, r = steps.shape[0], basis.shape[0], wo.shape[1], routes.shape[1]
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    qstats = next_x is not None
    stats = torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32)
    _fused_root_kernel[(triton.cdiv(p, 64), n)](
      steps.contiguous(), basis[0].float().contiguous() if FRAME_HADAMARD and p == d else basis.float().contiguous(),
      projection.contiguous(), wo, routes.contiguous(),
      clean.contiguous(), inv0.contiguous(), inv.contiguous(), scale, out,
      next_x.float().contiguous() if qstats else out, stats, p, d, r, qstats, BM=64, BN=64,
      HADAMARD=FRAME_HADAMARD and p == d, num_warps=4, num_stages=3)
    return out, stats


@fused_root_visit.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, scale, next_x):
    n, p, d = steps.shape[0], basis.shape[0], wo.shape[1]
    return (torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32))


KEY_DOT_PRECISION = 'tf32'  # the key / value credit kernels' logits and scores (tf32x3: cos 1.000000, query credits' norms within 0.35%)
FRAME_HADAMARD = True  # frame entries computed in-kernel: a full frame is the signed Sylvester Hadamard (row 0 = signs)
FUSED_VISIT_CONFIG = dict(BM=64, BK=64, BN=64, num_warps=4, num_stages=3)


def launch_fused_visit(delta, steps, frame, x, fc, wo, routes, clean, scale, xl, target, mu, nu, diag, mode,
    add=None):
    n, p, d = delta.shape
    r = routes.shape[1]
    inject = steps is not None
    out = delta if mode == 2 else torch.empty_like(delta)
    stats = (x if mode == 1 else torch.empty((n, p, 6 if mode == 2 else 2), device=delta.device,
      dtype=torch.float32))
    cfg = dict(FUSED_VISIT_CONFIG)
    bm = cfg.pop('BM')
    _fused_visit_kernel[(triton.cdiv(p, bm), n)](
      delta, steps.contiguous() if inject else delta,
      (frame[0].float().contiguous() if FRAME_HADAMARD and p == d else frame.float().contiguous()) if inject else delta,
      x.float().contiguous(), fc, wo, routes.contiguous(), clean.contiguous(), scale, out,
      target, mu, nu, diag, xl, stats, delta if add is None else add.contiguous(), p, d, r, inject, mode,
      add is not None, inject and FRAME_HADAMARD and p == d, BM=bm, EPSILON=EPS, **cfg)
    return out if mode == 1 else stats if mode == 2 else (out, stats)


@torch.library.custom_op('noise_record::fused_visit', mutates_args=())
def fused_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor, routes: torch.Tensor,
    clean: torch.Tensor, scale: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor], add: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A routed MLP visit with the next block's residual scalar applied (MODE 1); add: the attention replay's
    response to this block's input, added to delta on load."""
    return launch_fused_visit(delta.contiguous(), steps, frame, x, fc, wo, routes, clean, scale,
      x, x, x, x, x, 1, add)


@fused_visit.register_fake
def _(delta, x, fc, wo, routes, clean, scale, steps, frame, add=None):
    return torch.empty_like(delta)


@torch.library.custom_op('noise_record::fused_query_visit', mutates_args=())
def fused_query_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor, routes: torch.Tensor,
    clean: torch.Tensor, scale: torch.Tensor, next_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A routed MLP visit before an attention replay (MODE 3): the scaled output and, per move, its dot with the
    next block's clean input and its square, viewed as (token, move, statistic, 1) partials."""
    out, stats = launch_fused_visit(delta.contiguous(), None, None, x, fc, wo, routes, clean, scale,
      next_x.float().contiguous(), x, x, x, x, 3)
    return out, stats.unsqueeze(-1)


@fused_query_visit.register_fake
def _(delta, x, fc, wo, routes, clean, scale, next_x):
    return torch.empty_like(delta), torch.empty((*delta.shape[:2], 2, 1), device=delta.device, dtype=torch.float32)


@torch.library.custom_op('noise_record::fused_readout_visit', mutates_args=())
def fused_readout_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, diag: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor]) -> torch.Tensor:
    """The last block's routed MLP visit: the readout-moment sums of the moved final stream (MODE 2)."""
    f32 = lambda z: z.float().contiguous()
    return launch_fused_visit(delta.contiguous(), steps, frame, x, fc, wo, routes, clean, x,
      f32(xl), f32(target), f32(mu), f32(nu), f32(diag), 2)


@fused_readout_visit.register_fake
def _(delta, x, fc, wo, routes, clean, xl, target, mu, nu, diag, steps, frame):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)


def attention_replay_layers(roots, layers, long_layers):
    return tuple(b for b in range(roots[0] + 1, layers) if b in long_layers)


@triton.jit
def _route_output_query_stats_kernel(
  Change, Weight, Routes, Delta, Out, Steps, Basis,
  NextX, Scale, Stats,
  P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
  ROOT: tl.constexpr,
  NB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, ROOT, BM, BN, BK, False)
    result = (old + moved).to(tl.bfloat16)
    result = (result.to(tl.float32) * tl.load(Scale)).to(tl.bfloat16)
    tl.store(
      Out + (token * P + pp[:, None]) * D + dd[None, :],
      result, (pp[:, None] < P) & (dd[None, :] < D),
    )
    value = result.to(tl.float32)
    x = tl.load(NextX + token * D + dd, dd < D, 0).to(tl.float32)
    dot = tl.sum(value * x[None, :], 1)
    square = tl.sum(value * value, 1)
    position = (token * 2 * NB + tl.program_id(1)) * P + pp
    ss: tl.constexpr = NB * P
    tl.store(Stats + position, dot, pp < P)
    tl.store(Stats + position + ss, square, pp < P)


@torch.library.custom_op('noise_record::route_mlp_query_stats', mutates_args=())
def route_mlp_query_stats(
  delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
  routes: torch.Tensor, clean: torch.Tensor,
  inv0: torch.Tensor, inv: torch.Tensor,
  out_scale: torch.Tensor, next_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean, next_x = routes.contiguous(), clean.contiguous(), next_x.contiguous()
    tile, nb = min(64, r), triton.cdiv(d, 64)
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=delta.device, dtype=torch.bfloat16)
    partials = mlp_statistics_storage(n, p, 2, nb, delta.device)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      delta, fc, routes, clean, inv0.contiguous(), inv.contiguous(),
      change, delta, delta,
      p, d, r, *delta.stride(), False,
      BM=64, BN=tile, BK=64, num_warps=4, num_stages=3,
    )
    _route_output_query_stats_kernel[
      lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)
    ](
      change, wo, routes, delta, out, delta, delta,
      next_x, out_scale, partials,
      p, d, r, *delta.stride(), False, nb,
      BM=64, BN=64, BK=tile, num_warps=4, num_stages=3,
    )
    return out, partials


@route_mlp_query_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, out_scale, next_x):
    n, p, d = delta.shape
    return (
      torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), delta.device),
    )


@torch.library.custom_op('noise_record::route_mlp_root_query_stats', mutates_args=())
def route_mlp_root_query_stats(
  steps: torch.Tensor, basis: torch.Tensor,
  projection: torch.Tensor, wo: torch.Tensor,
  routes: torch.Tensor, clean: torch.Tensor,
  inv0: torch.Tensor, inv: torch.Tensor,
  out_scale: torch.Tensor, next_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d, p = steps.shape[0], wo.shape[1], basis.shape[0]
    r = routes.shape[1]
    steps, basis = steps.contiguous(), basis.contiguous()
    routes, clean, next_x = routes.contiguous(), clean.contiguous(), next_x.contiguous()
    nb = triton.cdiv(d, 64)
    change = torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    partials = mlp_statistics_storage(n, p, 2, nb, wo.device)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      steps, steps, routes, clean, inv0.contiguous(), inv.contiguous(),
      change, steps, projection,
      p, d, r, 0, 0, 0, True,
      BM=64, BN=64, BK=64, num_warps=4, num_stages=3,
    )
    _route_output_query_stats_kernel[
      lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)
    ](
      change, wo, routes, steps, out, steps, basis,
      next_x, out_scale, partials,
      p, d, r, 0, 0, 0, True, nb,
      BM=64, BN=64, BK=64, num_warps=4, num_stages=3,
    )
    return out, partials


@route_mlp_root_query_stats.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, out_scale, next_x):
    n, p, d = steps.shape[0], basis.shape[0], wo.shape[1]
    return (
      torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), wo.device),
    )


def query_inverses_from_output_partials(next_x, partials):
    stats = partials.sum(-1)
    square = next_x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(square / next_x.shape[-1] + EPS)
    inv = torch.rsqrt(
      (square + 2 * stats[:, :, 0] + stats[:, :, 1]) / next_x.shape[-1] + EPS
    )
    return inv0, inv


def chain_chunk_eager(site, root_moves, mids, xL, idx, hs, lam_r, w_fc, w_out_t, base,
    target_logit_w, y, own, moments, root_projection, injections, query_layers):
    """Replay one root group: the first root's signed frame moves enter at site and each partner's at its own
    block, through routed MLPs, with attention replayed only at query_layers. A partner must be the last block
    or be followed by a block that is neither replayed with attention nor another partner. Returns each token's
    moved-minus-clean loss for every move, from diagonal readout moments."""
    L = len(mids)
    pending_inv = pending_query_inv = replay_add = factored = None
    delta = None
    # CHAIN_READOUT: the visit (or root) at block L-2 and the readout at L-1 in one kernel (visit_chain)
    chain_tail = (CHAIN_READOUT and FUSED_CHAIN and L - 1 not in query_layers
                  and all(i != L - 1 for i, _, _ in injections) and idx[L - 1].shape[1] == 64
                  and idx[L - 2] is not None and idx[L - 2].shape[1] == 64)
    tail = lambda: (mids[L - 1], w_fc[L - 1], w_out_t[L - 1], idx[L - 1], hs[L - 1], xL, target_logit_w,
                    moments[2], moments[3], moments[4])
    for b in range(site, L):
        query_inv, pending_query_inv = pending_query_inv, None
        next_scale = lam_r[b + 1] if b + 1 < L else None
        next_query_x = own[1][b + 1] if b + 1 in query_layers else None
        if b > site:
            if b in query_layers:
                pos, inputs, w_q, keys, values, prob, cos_sin = own
                if FUSED_CHAIN and FUSED_ADD_ON_LOAD:
                    parts = FUSED_RESPONSE and keys[b].shape[1] * keys[b].shape[2] == 128
                    replay_add = query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv, raw_bf16='parts' if parts else True,
                        factored=factored if b == site + 1 else None)
                elif FUSED_CHAIN and FUSED_REPLAY_ADD:
                    delta = query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv, add_to=delta.to(torch.bfloat16))
                else:
                    delta = delta + query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv)
            delta = delta.to(torch.bfloat16)
        if b == site:
            inv0, inv = root_norm_inverses(mids[b], *root_moves)
            if chain_tail and b == L - 2 and next_query_x is None and FUSED_ROOT:
                terminal_stats = visit_chain.fused_root_readout_fast(*root_moves, root_projection, w_out_t[b], idx[b],
                    hs[b], inv0, inv, next_scale, *tail())
                break
            if (FACTORED_QUERY and FUSED_CHAIN and FUSED_ROOT and GLUON_ROOT and next_query_x is not None
                    and idx[b].shape[1] == 64):  # the root's routed change too, for the factored query replay
                delta, change, partials = visit_gluon_root.fused_root_visit_change(*root_moves, root_projection,
                    w_out_t[b], idx[b], hs[b], inv0, inv, next_scale, next_query_x)
                factored = (change, root_moves[0], next_scale, w_out_t[b], idx[b], root_moves[1])
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
                continue
            if FUSED_CHAIN and FUSED_ROOT:
                root_visit = visit_gluon_root.fused_root_visit_fast if GLUON_ROOT else fused_root_visit
                delta, partials = root_visit(*root_moves, root_projection, w_out_t[b], idx[b], hs[b], inv0,
                    inv, next_scale, next_query_x)
                if next_query_x is not None:
                    pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif next_query_x is None:
                delta = route_mlp_root(*root_moves, root_projection, w_out_t[b],
                    idx[b], hs[b], inv0, inv, next_scale)
            else:
                delta, partials = route_mlp_root_query_stats(*root_moves, root_projection,
                    w_out_t[b], idx[b], hs[b], inv0, inv, next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            continue
        injected = next(((steps, frame) for i, steps, frame in injections if i == b), None)
        if FUSED_CHAIN:
            steps, frame = injected if injected is not None else (None, None)
            pending_inv = None
            if (CHAIN_RESPONSE and chain_tail and b == L - 3 and steps is None and isinstance(replay_add, tuple)
                    and next_query_x is None and L - 2 not in query_layers and idx[b].shape[1] == 64
                    and all(i != L - 2 for i, _, _ in injections)):  # blocks L-3 (with the replay) to L-1 in one kernel
                m = L - 2
                add_weights, add_values = replay_add
                gathered = isinstance(add_values, tuple)
                terminal_stats = visit_chain3.fused_resp_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b],
                    idx[b], hs[b], next_scale, add_weights, add_values[0] if gathered else add_values, mids[m],
                    w_fc[m], w_out_t[m], idx[m], hs[m], lam_r[m + 1], *tail(), shared=CHAIN3_SHARED_RESP,
                    gathered_rows=add_values[2] if gathered else None, correction=add_values[1] if gathered else None)
                break
            if isinstance(replay_add, tuple) and isinstance(replay_add[1], tuple):  # not chain3: dense values
                replay_add = (replay_add[0], gathered_values_dense(*replay_add[1]))
            if chain_tail and b == L - 2 and steps is None and replay_add is None and next_query_x is None:
                terminal_stats = visit_chain.fused_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                    hs[b], next_scale, *tail())
                break
            if next_query_x is not None and not FUSED_QUERY_VISIT:
                inv0, inv = norm_inverses(mids[b], delta)
                delta, partials = route_mlp_query_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                    next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif next_query_x is not None:
                visit = visit_gluon.fused_query_visit_fast if GLUON_QUERY_VISIT else fused_query_visit
                delta, partials = visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif b == L - 1:
                assert replay_add is None, 'an attention replay must be followed by a routed visit that takes it'
                if (GLUON_INJECT and steps is not None) or (GLUON_READOUT and steps is None and idx[b].shape[1] == 64):
                    terminal_stats = visit_gluon.fused_readout_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                        hs[b], xL, target_logit_w, moments[2], moments[3], moments[4], steps,
                        None if frame is None else frame[0])
                else:
                    terminal_stats = fused_readout_visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], xL,
                        target_logit_w, moments[2], moments[3], moments[4], steps, frame)
            elif GLUON_VISIT and steps is None and replay_add is None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale)
            elif GLUON_INJECT and steps is not None and replay_add is None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, frame[0])
            elif GLUON_ADD and isinstance(replay_add, tuple):
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, None if frame is None else frame[0], add_weights=replay_add[0], add_values=replay_add[1])
                replay_add = None
            elif GLUON_ADD and replay_add is not None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, None if frame is None else frame[0], replay_add)
                replay_add = None
            else:
                delta = fused_visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale, steps, frame,
                    replay_add)
                replay_add = None
            continue
        if injected is not None:
            steps, frame = injected
            inv0, inv, packed = injected_norm_packed(mids[b], delta, steps, frame)
        else:
            steps = frame = packed = None
            inv0, inv = norm_inverses(mids[b], delta) if pending_inv is None else pending_inv
            pending_inv = None
        next_attention_held = (b + 1 < L and b + 1 not in query_layers
            and all(i != b + 1 for i, _, _ in injections))
        if next_attention_held:
            delta, partials = mlp_output_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                mids[b+1], lam_r[b+1], steps, frame, packed)
            stats = partials.sum(-1)
            clean_sq = mids[b+1].square().sum(-1, keepdim=True)
            pending_inv = (torch.rsqrt(clean_sq / mids[b+1].shape[-1] + EPS),
                torch.rsqrt((clean_sq + 2*stats[:, :, 0] + stats[:, :, 1]) / mids[b+1].shape[-1] + EPS))
        elif b == L - 1:
            terminal_stats = mlp_readout_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                xL, target_logit_w, moments[2], moments[3], moments[4], steps, frame, packed).sum(-1)
        else:
            if next_query_x is None:
                delta = route_mlp(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv, next_scale)
            else:
                delta, partials = route_mlp_query_stats(delta, w_fc[b], w_out_t[b], idx[b],
                    hs[b], inv0, inv, next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
    sq = xL.square().sum(-1, keepdim=True)
    dots, dsq, target_dot = terminal_stats[:, :, 0], terminal_stats[:, :, 1], terminal_stats[:, :, 2]
    inv = torch.rsqrt((sq + 2 * dots + dsq) / xL.shape[-1] + EPS)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + target_dot) * inv
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    b1, b2 = moments[:2]
    a = inv - inv0
    mu_dot = terminal_stats[:, :, 3]
    nu_dot = terminal_stats[:, :, 4]
    lin = a * b1.unsqueeze(1) + inv * mu_dot
    quad = terminal_stats[:, :, 5]
    change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * nu_dot
        + inv.square() * quad - 0.5 * lin.square())
    target0 = base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0
    moved_minus_clean = change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(target0 / 15))
    return moved_minus_clean * (y >= 0).unsqueeze(1)


FUSED_CHAIN = False  # set from Settings.fused_chain in Cut.__init__ (read when chain_chunk compiles)
FUSED_QUERY_VISIT = True  # the visit before the attention replay fused too (MODE 3)
FUSED_REPLAY_ADD = False  # measured slower (round 4)
FUSED_ADD_ON_LOAD = True  # the attention replay's response added inside the next fused visit's loads
FUSED_ROOT = True  # root visits in one kernel (_fused_root_kernel)
GLUON_QUERY_VISIT = True  # the visit before the attention replay (MODE 3) by visit_gluon too
FUSED_QUERY_REPLAY = True  # with FUSED_RESPONSE: the replay's q projection inside query_replay's kernel
GLUON_ROOT = True  # root visits by visit_gluon_root's Hopper kernel
FUSED_RESPONSE = True  # with GLUON_ADD and 128 replay key-heads: the replay's response formed inside the visit
GLUON_ADD = True  # the visit that takes the attention replay's response by visit_gluon too
GLUON_INJECT = True  # chain visits with an injected partner move (MODE 1 and the readout) by visit_gluon too
BLOCKED_PROJECTION = True  # the routed values' projection by project_values (bitwise)
WINDOW_TOPK_PRUNED = True  # with WINDOW_TOPK: the strong keys' top-k from group-maxima-pruned candidates (bitwise)
WINDOW_TOPK = True  # the strong keys' top-k by window_topk (bitwise equal to torch.topk)
FACTOR_GATHER = True  # GLUON_FACTORED's At factors gathered from the full Wo @ w_q^T product (bitwise; one small GEMM)
GLUON_FACTORED = True  # with FACTORED_QUERY: the Gluon factored replay (query_factored_gluon)
QF_KEYS_HI = True  # with QF_ROTATED_KEYS: the replay's rotated keys and r0' as bf16 high parts only (no low parts)
QF_ROTATED_KEYS = True  # with GLUON_FACTORED: the replay's scores against keys rotated back by the token's RoPE
# angles (query_factored_gluon.fused_query_replay_factored_rot: no per-element RoPE, norm sums and dq on wgmmas; dq not
# rounded to bf16, so not bitwise)
FACTORED_QUERY = True  # the replay right after a root: its query change from the root's routed factors
CHAIN_RESPONSE = True  # with CHAIN_READOUT: also the replay-response visit before them (visit_chain3)
ROOT_CHAIN_NS2 = True  # the root chain's routed-row ring at 2 slots (bitwise; 8 KB less shared memory)
GATHERED_VALUES = True  # chain3 (with CHAIN3_WIDE and CHAIN3_TC): the replay's projected top values never formed: the
# response gathers each token's 128 key rows of the per-key table bf16(v @ W_h) and adds the centring as one k16 wgmma
# ([s, s] @ -(out @ W_h) in bf16 hi / lo, s the per-head weight sums); no centring, no vtop gather (not bitwise)
FUSED_KEY_TABLES = True  # with GATHERED_VALUES: strong_key_shifted's shifted top keys and gathered_tables' outputs by
# one kernel (project_values.gathered_key_tables; the keys' fp32 sum order and a tf32, not tf32x3, correction differ)
CHAIN3_SHARED_RESP = True  # chain3's response once per chunk (the residual reuses the gate's bf16(delta + resp))
CHAIN3_DEFER_READOUT = True  # chain3's readout sums as per-thread partials (DEFER 2; fp32 order differs)
ROOT_CHAIN_TC = True  # root chain: block B's gate dots and the readout's 4 linear moments as wgmmas (x in 3, the readout
# vectors in 2 bf16 parts; fp32 order and the 2-part split differ: not bitwise)
ROOT_CHAIN_DEFER_READOUT = True  # root chain: the readout sums as per-thread partials (DEFER 3; fp32 order differs;
# -6% with ROOT_CHAIN_TC)
CHAIN3_TC = True  # chain3 (with CHAIN3_SHARED_RESP): blocks M's and B's gate dots and the readout's as ROOT_CHAIN_TC
CHAIN3_WIDE = True  # chain3 (with CHAIN3_SHARED_RESP): tiles of 128 moves (two warpgroups sharing each routed-row
# gather, one CTA per SM), a 4-slot routed-row ring, the response weights in their own buffer (bitwise per row)
CHAIN_PACKED_CVT = True  # chain kernels and the Gluon root: fp32 -> bf16 conversions as packed cvt.rn.bf16x2 (ptxas:
# scalar F2F); the roundings are the same, the fp32 contraction order around them may differ
CHAIN_READOUT = True  # the last two blocks' visits of a replay in one kernel (visit_chain, 64 routes)
GLUON_READOUT = True  # plain readout visits at 64 routes by visit_gluon (staged readout vectors)
GLUON_VISIT = True  # plain MODE 1 chain visits by visit_gluon's Hopper kernel (delta tile in shared memory)
chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)


@triton.jit
def _implicit_key_statistics(Q, K, Values, Hub, LSE, Center,
    T: tl.constexpr, Window: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    query = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 64)
    src = tl.arange(0, BN)
    base = head * T * 64
    q = tl.load(Q + base + query[:, None] * 64 + dims[None, :])
    hub = tl.load(Hub + base + query[:, None] * 64 + dims[None, :])
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    score_sum = tl.zeros((BM,), tl.float32)
    first = tl.maximum(tile * BM - Window, 0) // BN
    last = tl.cdiv((tile + 1) * BM, BN)
    for block in range(first, last):
        source = block * BN + src
        k = tl.load(K + base + source[:, None] * 64 + dims[None, :])
        v = tl.load(Values + base + source[:, None] * 64 + dims[None, :])
        logits = tl.dot(q, tl.trans(k), input_precision=PREC) * 0.125
        scores = tl.dot(hub, tl.trans(v), input_precision=PREC)
        allowed = (source[None, :] <= query[:, None]) & (query[:, None] - source[None, :] <= Window)
        logits = tl.where(allowed, logits, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == float('-inf'), 0.0, next_max)
        rescale = tl.exp(maximum - safe_max)
        prob = tl.exp(logits - safe_max[:, None])
        total = total * rescale + tl.sum(prob, 1)
        score_sum = score_sum * rescale + tl.sum(prob * scores, 1)
        maximum = next_max
    tl.store(LSE + head * T + query, maximum + tl.log(total))
    tl.store(Center + head * T + query, score_sum / total)


def implicit_key_context(q, k, values, hub, window):
    h, t, _ = q.shape
    q, k, values, hub = (z.contiguous() for z in (q, k, values, hub))
    lse = torch.empty((h, t), device=q.device)
    center = torch.empty_like(lse)
    _implicit_key_statistics[(t // 16, h)](q, k, values, hub, lse, center,
      T=t, Window=window, BM=16, BN=64, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    return lse, center


@triton.jit
def _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D: tl.constexpr, Sigma: tl.constexpr):
    raw = tl.load(Raw + base + target[:, None] * D + dims[None, :])
    raw_j = tl.load(Raw + base + target[:, None] * D + dirs[None, :])
    sq = tl.sum(raw * raw, 1)[:, None] / D
    step = Sigma * tl.maximum(tl.sqrt(sq), 1e-06)
    ratio = (2 * step * raw_j + step * step) / D / (sq + 1.1920928955078125e-07)
    ca = libdevice.expm1(-0.5 * libdevice.log1p(ratio))
    cb = step * libdevice.rsqrt(sq + 1.1920928955078125e-07) * (1 + ca)
    cs = tl.load(Cos + target[:, None] * (D // 2) + pair[None, :])
    sn = tl.load(Sin + target[:, None] * (D // 2) + pair[None, :])
    bu = cb * tl.where(dirs[None, :] < D // 2, cs, sn)
    bw = cb * tl.where(dirs[None, :] < D // 2, -sn, cs)
    return (ca, bu, bw, step)


@triton.jit
def _key_scan_kernel(Q, K, Raw, LSE, Values, Hub, Center, Cos, Sin, Out, T: tl.constexpr, D: tl.constexpr,
    Window: tl.constexpr, Sigma: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, J: tl.constexpr, PREC: tl.constexpr):
    head = tl.program_id(0)
    t0 = tl.program_id(1) * BT
    target = t0 + tl.arange(0, BT)
    dims = tl.arange(0, D)
    dirs = tl.arange(0, D)
    feature_pair = tl.arange(0, J)
    pair = dirs % (D // 2)
    base = head * T * D
    key = tl.load(K + base + target[:, None] * D + dims[None, :])
    value = tl.load(Values + base + target[:, None] * D + dims[None, :])
    p2 = tl.zeros((BT, 2 * J), tl.float32)
    q2 = tl.zeros((BT, 2 * J), tl.float32)
    qxy = tl.zeros((BT, J), tl.float32)
    r2 = tl.zeros((BT, 2 * J), tl.float32)
    s1 = tl.zeros((BT,), tl.float32)
    s2 = tl.zeros((BT,), tl.float32)
    first = t0 // BS
    last = tl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    for it in range(first, last):
        source = it * BS + tl.arange(0, BS)
        query = tl.load(Q + base + source[:, None] * D + dims[None, :])
        logit = tl.dot(key, tl.trans(query), input_precision=PREC) * D ** (-0.5)
        valid = (source[None, :] >= target[:, None]) & (source[None, :] - target[:, None] <= Window)
        hub = tl.load(Hub + base + source[:, None] * D + dims[None, :])
        score = tl.dot(value, tl.trans(hub), input_precision=PREC)
        center = tl.load(Center + head * T + source)
        c = tl.where(valid, score - center[None, :], 0.0)
        lse = tl.load(LSE + head * T + source)
        a = tl.where(valid, tl.exp(logit - lse[None, :]), 0.0)
        x = tl.load(Q + base + source[:, None] * D + feature_pair[None, :]) * D ** (-0.5)
        y = tl.load(Q + base + source[:, None] * D + (feature_pair + D // 2)[None, :]) * D ** (-0.5)
        f2 = tl.reshape(tl.join(x, y), (BS, 2 * J))
        g2 = tl.reshape(tl.join(x * x, y * y), (BS, 2 * J))
        xy = x * y
        w1 = a * c
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logit
        w1d, w2d, w2ld = (w1.to(tl.bfloat16), w2.to(tl.bfloat16), w2l.to(tl.bfloat16))
        f2d, g2d, xyd = (f2.to(tl.bfloat16), g2.to(tl.bfloat16), xy.to(tl.bfloat16))
        p2 += tl.dot(w1d, f2d)
        q2 += tl.dot(w2d, g2d)
        qxy += tl.dot(w2d, xyd)
        r2 += tl.dot(w2ld, f2d)
        s1 += tl.sum(w1 * logit, 1)
        s2 += tl.sum(w2 * logit * logit, 1)
    ca, bu, bw, step = _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D, Sigma)
    px, py = tl.split(tl.reshape(p2, (BT, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BT, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BT, J, 2)))
    gather_pair = tl.broadcast_to(pair[None, :], (BT, D))
    px, py = (tl.gather(px, gather_pair, 1), tl.gather(py, gather_pair, 1))
    qx, qy = (tl.gather(qx, gather_pair, 1), tl.gather(qy, gather_pair, 1))
    rx, ry = (tl.gather(rx, gather_pair, 1), tl.gather(ry, gather_pair, 1))
    qxy = tl.gather(qxy, gather_pair, 1)
    response = px * bu + py * bw + qx * bu * bu + qy * bw * bw + 2 * qxy * bu * bw + 2 * ca * (rx * bu + ry * bw) + s1[:, None] * ca + s2[:, None] * ca * ca
    tl.store(Out + base + target[:, None] * D + dirs[None, :], response / step)


@triton.jit
def _value_credit_kernel(Q,K,LSE,Hub,Out,T:tl.constexpr,Window:tl.constexpr,
    BT:tl.constexpr,BS:tl.constexpr,Channels:tl.constexpr,PREC:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    target=tile*BT+tl.arange(0,BT)
    dims=tl.arange(0,64)
    channels=tl.arange(0,Channels)
    base=head*T*64
    key=tl.load(K+base+target[:,None]*64+dims[None,:])
    acc=tl.zeros((BT,Channels),tl.float32)
    first=tile*BT//BS
    last=tl.minimum((tile*BT+BT-1+Window)//BS+1,T//BS)
    for it in range(first,last):
        receiver=it*BS+tl.arange(0,BS)
        query=tl.load(Q+base+receiver[:,None]*64+dims[None,:])
        hub=tl.load(Hub+head*T*Channels+receiver[:,None]*Channels+channels[None,:])
        lse=tl.load(LSE+head*T+receiver)
        logits=tl.dot(key,tl.trans(query),input_precision=PREC)*.125
        valid=(receiver[None,:]>=target[:,None]) & (receiver[None,:]-target[:,None]<=Window)
        prob=tl.where(valid,tl.exp(logits-lse[None,:]),0.).to(tl.bfloat16)
        hi=hub.to(tl.bfloat16)
        lo=(hub-hi.to(tl.float32)).to(tl.bfloat16)
        acc+=tl.dot(prob,hi)
        acc+=tl.dot(prob,lo)
    tl.store(Out+head*T*Channels+target[:,None]*Channels+channels[None,:],acc)


def value_credits(context, corrected, window):
    """key_and_value_credits' value part alone (the key part batched across layers by batched_key_contexts)."""
    q, k, raw, cos, sin, lse, (v, hub, center) = context
    h, t, d = raw.shape
    paired = torch.cat((hub.contiguous(), corrected.contiguous()), dim=-1)
    out = torch.empty_like(paired)
    _value_credit_kernel[h, t // 32](q.contiguous(), k.contiguous(), lse.contiguous(), paired, out, T=t, Window=window,
        BT=32, BS=64, Channels=128, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    return out[..., :64].contiguous(), out[..., 64:].contiguous()


def key_and_value_credits(context, corrected, sigma, window):
    q, k, raw, cos, sin, lse, (v, hub, center) = context
    q, k, raw, cos, sin, lse, v, hub, center, corrected = (z.contiguous() for z in (q, k, raw, cos, sin, lse, v, hub, center, corrected))
    h, t, d = raw.shape
    key = torch.empty_like(raw)
    _key_scan_kernel[h, t // 32](
        q, k, raw, lse, v, hub, center, cos, sin, key, T=t, D=d, Window=window, Sigma=sigma,
        BT=32, BS=64, J=32, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2,
    )
    paired = torch.cat((hub, corrected), dim=-1)
    out = torch.empty_like(paired)
    _value_credit_kernel[h, t // 32](q, k, lse, paired, out, T=t, Window=window, BT=32, BS=64, Channels=128, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    value, terminal = (out[..., :64].contiguous(), out[..., 64:].contiguous())
    return (key, value, terminal)


QUERY_FP8 = False  # the attention replay's moved-query projection in fp8 (rowwise scales)
COMPILED_CAPTURE = True  # Capture.capture by one compiled forward that returns what the hooks record
QKV_GEMM = True  # CAST_ONCE: c_q / c_k / c_v in one GEMM against their concatenated bf16 weight (outputs: column slices)
# the capture's flex attention tiles: BLOCK_N 128 as the default keeps it bitwise, BLOCK_M 64 / 4 warps / 3 stages
# fastest (None: the defaults)
CAPTURE_FLEX_OPTIONS = dict(BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=3)


def capture_attention(q, k, v, window):
    """utils.model.attention with CAPTURE_FLEX_OPTIONS (inside capture_forward's graph)."""
    if CAPTURE_FLEX_OPTIONS is None:
        return attention(q, k, v, window)
    mask = window_mask(window, q.shape[1], q.device)
    return flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask,
                          kernel_options=CAPTURE_FLEX_OPTIONS).transpose(1, 2)


@torch.compile(dynamic=False, options={'combo_kernels': True})
def capture_forward(model, idx, targets, weights=None):
    """GPT.forward (utils/model.py) under bf16 autocast, returning the tensors Capture's hooks record: each Linear's
    input (fp32) and output, the embeddings' outputs, each block's input and the per-token loss. weights: the Linears'
    bf16 weights by module name (cast once by the caller), else autocast casts them."""
    inputs, outputs, blocks = {}, {}, {}
    BF16_KINDS = ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj') if CAPTURE_BF16_INPUTS else ()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        b, t = idx.shape
        cos, sin = model.cos[:, :t], model.sin[:, :t]
        def linear(name, module, z):
            if not (CAPTURE_PROJ_UNSTACKED and name.endswith('mlp.c_proj')):  # (weight_credits_bf16 forms relu^2)
                inputs[name] = z.to(torch.bfloat16) if name.endswith(BF16_KINDS) else z.float()
            out = module(z) if weights is None else F.linear(z, weights[name])
            outputs[name] = out
            return out
        e = model.transformer.wte(idx)
        inputs['transformer.wte'], outputs['transformer.wte'] = idx, e
        x = norm(e)
        x0 = x
        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            blocks[i] = x
            p = f'transformer.h.{i}.'
            attn = block.attn
            ve = None
            if str(i) in model.value_embeds:
                ve = model.value_embeds[str(i)](idx)
                inputs[f'value_embeds.{i}'], outputs[f'value_embeds.{i}'] = idx, ve
            xn = norm(x)
            h, hd = attn.n_head, attn.head_dim
            if weights is not None and p + 'attn.qkv' in weights:  # QKV_GEMM: one GEMM, outputs its column slices
                inputs[p + 'attn.c_q'] = xn.float()
                qkv = F.linear(xn, weights[p + 'attn.qkv'])
                c, ck = attn.c_q.weight.shape[0], attn.c_k.weight.shape[0]
                q, k, v = qkv[..., :c], qkv[..., c:c + ck], qkv[..., c + ck:]
                outputs[p + 'attn.c_q'], outputs[p + 'attn.c_k'], outputs[p + 'attn.c_v'] = q, k, v
                q, k, v = q.view(b, t, h, hd), k.view(b, t, attn.n_kv_head, hd), v.view(b, t, attn.n_kv_head, hd)
            else:
                q = linear(p + 'attn.c_q', attn.c_q, xn).view(b, t, h, hd)
                k = linear(p + 'attn.c_k', attn.c_k, xn).view(b, t, attn.n_kv_head, hd)
                v = linear(p + 'attn.c_v', attn.c_v, xn).view(b, t, attn.n_kv_head, hd)
            if ve is not None:
                gate = 2 * torch.sigmoid(linear(p + 'attn.ve_gate', attn.ve_gate, xn[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve.view(b, t, attn.n_kv_head, hd)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            y = capture_attention(q, k, v, model.window_sizes[i][0]).contiguous().view(b, t, -1)
            x = x + linear(p + 'attn.c_proj', attn.c_proj, y)
            hid = linear(p + 'mlp.c_fc', block.mlp.c_fc, norm(x))
            x = x + linear(p + 'mlp.c_proj', block.mlp.c_proj, F.relu(hid).square())
        logits = linear('lm_head', model.lm_head, norm(x))[..., :model.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                               reduction='none')
    if FUSED_GLUE:  # each kind of block input stacked over the blocks ('stack:<kind>'; the per-block ones are views)
        L = len(model.transformer.h)
        for kind in STACKED_INPUTS:
            if kind == 'mlp.c_proj' and CAPTURE_PROJ_UNSTACKED:  # (the per-block relu^2 GEMM inputs are kept)
                continue
            stack = torch.stack([inputs[f'transformer.h.{i}.{kind}'] for i in range(L)])
            inputs['stack:' + kind] = stack
            if kind == 'attn.c_q' and INTERNALS_BF16:  # the c_q GEMM's bf16 input (autocast's cast of the same values)
                inputs['stack:attn.c_q:bf16'] = stack.to(torch.bfloat16)
            for i in range(L):
                inputs[f'transformer.h.{i}.{kind}'] = stack[i]
        for i in range(L):  # the same normed block input
            inputs[f'transformer.h.{i}.attn.c_k'] = inputs[f'transformer.h.{i}.attn.c_v'] = (
                inputs[f'transformer.h.{i}.attn.c_q'])
    return inputs, outputs, blocks, loss


STACKED_INPUTS = ('attn.c_q', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
INTERNALS_BF16 = True  # internals' batched q / k / v weight-credit GEMM on bf16 operands (fp32 accumulation and output):
# the c_q input stack is the forward's bf16 GEMM input (exact), the side-by-side errors are rounded to bf16


@torch.compile(dynamic=False)
def qkv_weight_credits_bf16(qkv, inputs_bf16):
    """INTERNALS_BF16: torch.bmm(qkv.transpose(1, 2), inputs.flatten(1, 2)) with bf16 operands, fp32 out."""
    return torch.bmm(qkv.to(torch.bfloat16).transpose(1, 2), inputs_bf16.flatten(1, 2), out_dtype=torch.float32)
HIDDEN_GW_BF16 = True  # weight_credits_bf16: the hidden credit's c_proj back-projection g_w written in bf16 (one more
# rounding before the slope: not bitwise), not fp32 (a 32 MB write and read per block)
GATE_RESPONSE_KERNEL = True  # stacked_gate_input_responses' sigmoid core by gate_response's kernel (the per-token
# scalars hoisted; sigmoid differences as tanh.approx differences: ~3e-4 of the largest response, not bitwise)
CAPTURE_PROJ_UNSTACKED = True  # with WEIGHT_CREDITS_BF16: the capture records no c_proj inputs and no 'stack:mlp.c_proj'
# (a 128 MB relu^2 copy recomputed from the c_fc outputs); weight_credits_bf16 forms each block's relu^2 from its c_fc
# output and takes its c_proj credit by one mm (bitwise with the bmm)
CAPTURE_BF16_INPUTS = True  # with WEIGHT_CREDITS_BF16 (and FUSED_GLUE, not LAYER_TAILS): the compiled capture records
# the c_proj / c_fc inputs as bf16 (the values weight_credits_bf16 casts them to; its only consumer), not fp32 copies


class Capture:
    def __init__(self, model, seed):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.weights = None  # (CAST_ONCE: the Linears' bf16 weights, set by Cut.estimate)
        self.capturing = False
        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(layer)))
    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
        return hook
    def block_hook(self, layer):
        def hook(module, args):
            if self.capturing:
                self.blocks[layer] = args[0]
        return hook
    def close(self):
        for hook in self.hooks:
            hook.remove()
    def signs(self, *shape, shared=False):
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1
    def query_noise(self, draws, b, t, d):
        patterns = self.signs(1, d, shared=True)
        offset = torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator)
        signs = self.signs(draws, b, t, 1)
        index = torch.arange(draws, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        return hadamard(d, self.device, self.dtype)[rows] * patterns[0] * signs
    def capture(self, x, y):
        self.y = y
        self.batch, self.length = x.shape
        if COMPILED_CAPTURE:
            self.inputs, self.outputs, self.blocks, loss = capture_forward(self.model, x, y, self.weights)
            self.clean_loss = loss.view_as(y).float()
            return
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
    def head_error(self, sigma=0.1):
        return coordinate_differences(self.outputs['lm_head'], self.y, sigma)
    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        if REASSEMBLE_BF16 and site == 'lm_head':  # x is the head GEMM's bf16 input (exact); the error is rounded
            return torch.mm(error.flatten(0, 1).T.to(torch.bfloat16), x.flatten(0, 1).to(torch.bfloat16),
                            out_dtype=torch.float32)
        return torch.einsum('btd,bti->di', error, x)


@triton.jit
def _route_moment_stream_kernel(Scores, K, V, Top, Probability, OutV, OutK,
    T: tl.constexpr, WINDOW: tl.constexpr, KEYS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    qs = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, 64)
    ns = tl.arange(0, BN)
    running_max = tl.full((BM,), float('-inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    first = tl.maximum(tile * BM - WINDOW, 0) // BN
    last = tl.cdiv((tile + 1) * BM, BN)
    for block in range(first, last):
        ks = block * BN + ns
        score = tl.load(Scores + (head * T + qs[:, None]) * T + ks[None, :]).to(tl.float32) * 0.125
        new_max = tl.maximum(running_max, tl.max(score, 1))
        denom = denom * tl.exp(running_max-new_max) + tl.sum(tl.exp(score-new_max[:, None]), 1)
        running_max = new_max
    av = tl.zeros((BM, 64), tl.float32)
    ak = tl.zeros((BM, 64), tl.float32)
    for block in range(first, last):
        ks = block * BN + ns
        score = tl.load(Scores + (head * T + qs[:, None]) * T + ks[None, :]).to(tl.float32) * 0.125
        prob = (tl.exp(score-running_max[:, None])/denom[:, None]).to(tl.bfloat16)
        key = tl.load(K + (head * T + ks[:, None]) * 64 + ds[None, :])
        val = tl.load(V + (head * T + ks[:, None]) * 64 + ds[None, :])
        av += tl.dot(prob, val)
        ak += tl.dot(prob, key)
    rows = (head * T + qs[:, None]) * KEYS + tl.arange(0, KEYS)[None, :]
    top = tl.load(Top + rows).to(tl.float32) * 0.125
    pp = tl.exp(top-running_max[:, None])/denom[:, None]
    tl.store(Probability + rows, pp)
    dest = (head * T + qs[:, None]) * 64 + ds[None, :]
    tl.store(OutV + dest, av.to(tl.bfloat16).to(tl.float32))
    tl.store(OutK + dest, ak.to(tl.bfloat16).to(tl.float32))


FAST_MOMENTS = True  # moment_kernels' implicit key context and query moments (subagent's tiles)
FAST_KEYS = True  # moment_kernels' key scan (larger tiles)
BATCHED_VALUES = True  # with OVERLAP_KEYS: raw value credits before the chain, terminal ones after, batched
OVERLAP_KEYS = True  # with batched_keys: key credits and query moments on a side stream under the value chain
TRITON_TOPK = False  # the replay's strong keys by a row top-k kernel (ties may resolve differently than torch.topk)
LEAN_STRONG_KEYS = True  # strong keys / values gathered contiguously, their centring in one compiled graph


COMPILED_DESCENT = False  # with BATCHED_VALUES and BATCHED_GATES: the descending chain's glue in one compiled graph


@torch.compile(dynamic=False)
def descending_chain(chains, raws, gates, xins, c_projs, c_vs, lam_r, length: int):
    """descending_key_values' chain without its attention kernels: each layer's output credit corrected by the
    deeper layers' measured input responses, the value input response through the norm (plus the gate's), and the
    carried extra for the layer below."""
    L = len(chains)
    corrected, inputs, extra = [None] * L, [None] * L, [None] * L
    extra[L - 1] = torch.zeros_like(chains[0])
    for i in reversed(range(L)):
        corrected[i] = ((chains[i] + extra[i]) @ c_projs[i]).view(1, length, 8, 64).transpose(1, 2)[0]
        ev = raws[i].transpose(0, 1).reshape(-1, chains[0].shape[-1])
        inputs[i] = norm_move(xins[i], ev @ c_vs[i])
        if gates[i] is not None:
            inputs[i] = inputs[i] + gates[i]
        if i:
            extra[i - 1] = lam_r[i] * (inputs[i] + extra[i])
    return corrected, inputs, extra


BATCHED_GATES = True  # with BATCHED_VALUES: the value gates' input credits of all layers before the chain
RECOMPUTE_BMM = True  # block_recompute's GEMMs as two batched GEMMs over the blocks (fp32 weights stacked per call)
FLAT_GRADIENTS = True  # with STACKED_ANVIL, LEAN_FLAT and SHARDED_ANVIL: the estimate packs Cut.step's flat buffer


@torch.compile(dynamic=False)
def pack_gradients(gradients, divisor, bf16: bool):
    """Cut.step's flat gradient buffer, torch.cat(...) / divisor, and (bf16) its bf16 copy, in one compiled graph."""
    flat = torch.cat([g.float().reshape(-1) for g in gradients]) / divisor
    return flat, flat.to(torch.bfloat16) if bf16 else flat


CHAIN_GLUE = True  # chain_credits' weight casts / transposes and root projections in compiled graphs


@torch.compile(dynamic=False)
def chain_weights(fc_weights, proj_weights):
    """chain_credits' bf16 c_fc weights and transposed bf16 c_proj weights (each contiguous)."""
    return ([w.to(torch.bfloat16) for w in fc_weights],
            [w.to(torch.bfloat16).T.contiguous() for w in proj_weights])


@torch.compile(dynamic=False)
def chain_transposes(weights):
    """Each weight transposed, contiguous."""
    return [w.T.contiguous() for w in weights]


CAST_ONCE = True  # Cut.estimate casts every Linear weight to bf16 once (capture forward and chain_credits share them)


@torch.compile(dynamic=False)
def root_projections_of(fc_weights, frames):
    """Each root's bf16 c_fc rows projected on its frame (fp32, contiguous)."""
    return [(w.float() @ frame.T).contiguous() for w, frame in zip(fc_weights, frames)]


IMPLICIT_EDGES = True  # batched_key_contexts' lse and center masking only the key blocks the window cuts
IMPLICIT_GLUON = True  # batched_key_contexts' lse / center by implicit_gluon (TMA + tf32 wgmma; fp32 order and the
# exp2 form differ)
IMPLICIT_GLUON_BF16 = False  # with IMPLICIT_BF16 1: its bf16 lse / center pass by implicit_gluon (bf16 wgmmas) instead of
# the Triton kernel
IMPLICIT_BF16 = 1  # with EARLY_MOMENT_INPUTS: the lse / center pass on the query moments' bf16 q / k / values / hub
# copies (bf16 dots, fp32 accumulation; not bitwise: tf32 on fp32 before)
FUSED_GLUE = True  # with batched_keys, OVERLAP_KEYS, BATCHED_VALUES and BATCHED_GATES: descending_key_values' credit
# kernels write the (layers, tokens, width) layout the reassembly reads, the chain and the path scalars in one graph


def block_recompute(blocks, attn_outs, fc_weights, proj_weights, first: int, chain_sigma: float, fc_outs=None,
                    proj_outs=None):
    """block_recompute_compiled; RELU2_GEMM (with RECOMPUTE_BMM): the output GEMM by recompute_kernels.relu2_gemm;
    RECOMPUTE_FROM_CAPTURE: the capture's own bf16 c_fc / c_proj outputs (fc_outs / proj_outs), no GEMMs."""
    if RECOMPUTE_FROM_CAPTURE:
        return recompute_from_capture(blocks, attn_outs, fc_outs, proj_outs, fc_weights, proj_weights, first,
                                      chain_sigma)
    if not (RELU2_GEMM and RECOMPUTE_BMM):
        return block_recompute_compiled(blocks, attn_outs, fc_weights, proj_weights, first, chain_sigma)
    mid, hpre, w_proj, scores = recompute_front(blocks, attn_outs, fc_weights, proj_weights, first, chain_sigma)
    out = recompute_kernels.relu2_gemm(hpre, w_proj, mid)
    return list(mid.unbind(0)), list(hpre.unbind(0)), list(out.unbind(0)), scores


RELU2_GEMM = True  # with RECOMPUTE_BMM: block_recompute's output GEMM with relu^2 on its A load (Triton, tf32 rna)
RECOMPUTE_FROM_CAPTURE = True  # block_recompute's pre-activations and outputs from the capture forward's bf16 c_fc /
# c_proj outputs instead of fp32 / tf32 GEMMs (bf16 roundings: not bitwise)
CAPTURE_ROUTES = True  # with RECOMPUTE_FROM_CAPTURE: the route values gathered from the capture's bf16 pre-activations
# (no fp32 copy of them; the same values)


@torch.compile(dynamic=False)
def recompute_from_capture(blocks, attn_outs, fc_outs, proj_outs, fc_weights, proj_weights, first: int,
                           chain_sigma: float):
    """RECOMPUTE_FROM_CAPTURE: block_recompute_compiled's mid streams, pre-activations, outputs and route scores, the
    pre-activations and the MLP outputs read from the capture (the forward's bf16 GEMM outputs)."""
    mid = torch.stack([b.float().flatten(0, 1) + a.float().flatten(0, 1) for b, a in zip(blocks, attn_outs)])
    out = mid + torch.stack([o.float().flatten(0, 1) for o in proj_outs])
    scores = [response_scores(fc_outs[i].float().flatten(0, 1), chain_sigma * fc_weights[i].float().norm(dim=1),
                              proj_weights[i].float().norm(dim=0)) for i in range(first, len(blocks))]
    if CAPTURE_ROUTES:  # the pre-activations stay the capture's bf16 tensors (the route values gathered from them)
        return list(mid.unbind(0)), [f.flatten(0, 1) for f in fc_outs], list(out.unbind(0)), torch.stack(scores)
    hpre = torch.stack([f.float().flatten(0, 1) for f in fc_outs])
    return list(mid.unbind(0)), list(hpre.unbind(0)), list(out.unbind(0)), torch.stack(scores)


@torch.compile(dynamic=False)
def recompute_front(blocks, attn_outs, fc_weights, proj_weights, first: int, chain_sigma: float):
    """RELU2_GEMM: block_recompute's mid stream, pre-activations and route scores, and the stacked c_proj weights
    rounded to tf32 for the output GEMM."""
    mid = torch.stack([b.float().flatten(0, 1) + a.float().flatten(0, 1) for b, a in zip(blocks, attn_outs)])
    hpre = torch.bmm(norm(mid), torch.stack([w.float() for w in fc_weights]).transpose(1, 2))
    scores = [response_scores(hpre[i], chain_sigma * fc_weights[i].float().norm(dim=1),
                              proj_weights[i].float().norm(dim=0)) for i in range(first, len(blocks))]
    w_proj = recompute_kernels.rna_tf32(torch.stack([w.float() for w in proj_weights]))
    return mid, hpre, w_proj, torch.stack(scores)


@torch.compile(dynamic=False)
def block_recompute_compiled(blocks, attn_outs, fc_weights, proj_weights, first: int, chain_sigma: float):
    """estimate's fp32 recompute of each block's mid stream, MLP pre-activations and output, and (from block first on)
    the stacked route scores (route_scores) for the batched MLP routes."""
    if RECOMPUTE_BMM:  # the blocks' GEMMs batched (the blocks are independent here: mid = block input + attention)
        mid = torch.stack([b.float().flatten(0, 1) + a.float().flatten(0, 1) for b, a in zip(blocks, attn_outs)])
        hpre = torch.bmm(norm(mid), torch.stack([w.float() for w in fc_weights]).transpose(1, 2))
        out = mid + torch.bmm(F.relu(hpre).square(), torch.stack([w.float() for w in proj_weights]).transpose(1, 2))
        scores = [response_scores(hpre[i], chain_sigma * fc_weights[i].float().norm(dim=1),
                                  proj_weights[i].float().norm(dim=0)) for i in range(first, len(blocks))]
        return list(mid.unbind(0)), list(hpre.unbind(0)), list(out.unbind(0)), torch.stack(scores)
    mids, hpres, outs, scores = [], [], [], []
    for i, (block, attn_out, w_fc, w_proj) in enumerate(zip(blocks, attn_outs, fc_weights, proj_weights)):
        mid = block.float().flatten(0, 1) + attn_out.float().flatten(0, 1)
        hpre = F.linear(norm(mid), w_fc.float())
        out = mid + F.linear(F.relu(hpre).square(), w_proj.float())
        mids.append(mid), hpres.append(hpre), outs.append(out)
        if i >= first:
            radius = chain_sigma * w_fc.float().norm(dim=1)
            scores.append(response_scores(hpre, radius, w_proj.float().norm(dim=0)))
    return mids, hpres, outs, torch.stack(scores)


@torch.compile(dynamic=False)
def weight_credits(chains, g_top, lam_r, extras, attn_proj_in, mlp_proj_in, fc_outs, fc_in, fc_out_weights,
                   sigma: float):
    """layer_weight_credits for every block at once: three batched GEMMs over the blocks' stacked credits and the
    capture's stacked inputs ((blocks, 1, tokens, width)). Returns (blocks, out, in) weight credits."""
    L = len(chains)
    d_out = [lam_r[i + 1] * chains[i + 1] for i in range(L - 1)] + [g_top]
    x_mid = torch.stack([chains[i] + extras[i] for i in range(L)])
    x_out = torch.stack([d_out[i] + extras[i] for i in range(L)])
    hidden = []
    for i in range(L):
        u = fc_outs[i].float().flatten(0, 1)
        steps = sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        hidden.append(hidden_local_credit(u, steps, d_out[i], fc_out_weights[i].float()))
    hidden = torch.stack(hidden)
    return (torch.bmm(x_mid.transpose(1, 2), attn_proj_in.flatten(1, 2)),
            torch.bmm(x_out.transpose(1, 2), mlp_proj_in.flatten(1, 2)),
            torch.bmm(hidden.transpose(1, 2), fc_in.flatten(1, 2)))


KEY_GLUON = True  # the key credit scan by key_gluon (Gluon, FA3-style; fp32 order differs)
VALUE_LSE_TMA = True  # with VALUE_GLUON: the receivers' lse rows by TMA with the query ring (as KEY_LSE_TMA; bitwise)
KEY_LSE_TMA = True  # with KEY_GLUON: the receivers' lse / center rows by TMA with the query ring (bitwise)
TOPK_PRUNED = True  # route_topk by a candidate-pruned kernel (group-maxima floor, top-k of <= 2K candidates; exact)
VALUE_HI_ONLY = True  # with VALUE_GLUON: the source's bf16 high part only (P @ high; the low part dropped: not bitwise)
VALUE_QK16 = True  # with VALUE_GLUON and EARLY_MOMENT_INPUTS: the value passes' logits from the query moments' bf16 q / k
# copies (bf16 wgmmas; tf32 on fp32 before: not bitwise)
KEY_QK16 = False  # with KEY_GLUON and EARLY_MOMENT_INPUTS: the key scan's logits and scores from the query moments' bf16
# q / k / values / hub copies (bf16 wgmmas; tf32 on fp32 before: not bitwise)
VALUE_GLUON = True  # the raw / terminal value credit passes by value_gluon (Gluon, FA3-style; fp32 order differs)
WEIGHT_CREDITS_BF16 = True  # weight_credits' GEMMs on bf16 operands (fp32 accumulation and output): the block inputs are
# bf16 values already (exact), the credits and c_proj's weight are rounded to bf16


@torch.compile(dynamic=False)
def weight_credits_bf16(chains, g_top, lam_r, extras, attn_proj_in, mlp_proj_in, fc_outs, fc_in, fc_out_weights,
                        sigma: float):
    """weight_credits with bf16 GEMM operands."""
    L = len(chains)
    bf = torch.bfloat16
    d_out = [lam_r[i + 1] * chains[i + 1] for i in range(L - 1)] + [g_top]
    x_mid = torch.stack([chains[i] + extras[i] for i in range(L)]).to(bf)
    x_out = torch.stack([d_out[i] + extras[i] for i in range(L)]).to(bf)
    hidden = []
    for i in range(L):
        u = fc_outs[i].float().flatten(0, 1)
        steps = sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        up, down = F.relu(u + steps), F.relu(u - steps)
        slope = (up - down) * (up + down) / (2 * steps)
        g_w = torch.mm(d_out[i].to(bf), fc_out_weights[i].to(bf), out_dtype=bf if HIDDEN_GW_BF16 else torch.float32)
        hidden.append((g_w * slope).to(bf))
    hidden = torch.stack(hidden)
    if mlp_proj_in is None:  # CAPTURE_PROJ_UNSTACKED: each block's relu^2 (bf16, as the forward's c_proj GEMM input)
        # formed in the slope pass that reads the same c_fc output, then one mm per block (bitwise with the bmm)
        mlp_proj = [torch.mm(x_out[i].T, F.relu(fc_outs[i].flatten(0, 1)).square().to(bf), out_dtype=torch.float32)
                    for i in range(L)]
    else:
        mlp_proj = torch.bmm(x_out.transpose(1, 2), mlp_proj_in.flatten(1, 2).to(bf), out_dtype=torch.float32)
    return (torch.bmm(x_mid.transpose(1, 2), attn_proj_in.flatten(1, 2).to(bf), out_dtype=torch.float32),
            mlp_proj,
            torch.bmm(hidden.transpose(1, 2), fc_in.flatten(1, 2).to(bf), out_dtype=torch.float32))


@torch.compile(dynamic=False)
def layer_weight_credits(d_mid, d_next, lam, extra, attn_proj_in, mlp_proj_in, fc_out, fc_in, fc_out_weight,
                         sigma: float):
    """estimate's per-block weight credits in one compiled graph: attn.c_proj from the mid credit, mlp.c_proj from the
    out credit (lam * the next block's mid credit, or the readout's with lam = 1) and mlp.c_fc through
    hidden_credit's finite step (fc_out the bf16 hidden pre-activations)."""
    d_out = lam * d_next
    x_mid, x_out = d_mid + extra, d_out + extra
    u = fc_out.float().flatten(0, 1)
    steps = sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    hidden = hidden_local_credit(u, steps, d_out, fc_out_weight.float())
    return (x_mid.T @ attn_proj_in.flatten(0, 1), x_out.T @ mlp_proj_in.flatten(0, 1),
            hidden.T @ fc_in.flatten(0, 1))


def probe_mm(a, b):
    """a @ b; PROBE_BF16: on bf16 operands, fp32 accumulation and output."""
    if PROBE_BF16:
        return torch.mm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)
    return a @ b


PROBE_BF16 = True  # probe_moment_terms' GEMMs on bf16 operands (fp32 out; tf32 before: not bitwise)
MERGED_PROBE_GEMMS = True  # probe_moment_terms' GEMMs that share an operand merged (bitwise the same)


def probe_moment_terms_eager(x, y, weight, H):
    """Cut.probe_moments_eager's computation (frame H drawn by the caller) in one compiled graph: the readout credit
    and the head moments."""
    mm = probe_mm
    C = mm(H, weight.T)
    width = x.shape[-1]
    steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    base = mm(x, weight.T)
    sq = x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    t = torch.tanh(base * inv0 / 15)
    f1 = 1 - t.square()
    prob = torch.softmax(15 * t, -1)
    c1, c2 = prob * f1, prob * (-(2 / 15) * t * f1 + f1.square()) / 2
    b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
    if MERGED_PROBE_GEMMS:  # products sharing an operand as one GEMM: [c1; c2 base] @ W, c2 @ [W^2 | C^2 T], .. @ H^T
        n = x.shape[0]
        first = mm(torch.cat([c1, c2 * base]), weight)
        mu1, nu = first[:n], first[n:]
        second = mm(c2, torch.cat([weight.square(), C.square().T], 1))
        curvature, quad = second[:, :width].contiguous(), second[:, width:]
        mu1_h, nu_h = mm(first, H.T).split(n)
    else:
        mu1, nu = c1 @ weight, (c2 * base) @ weight
        curvature, quad = readout_curvature_diagonal(weight, c2), c2 @ C.square().T
        mu1_h, nu_h = mu1 @ H.T, nu @ H.T
    moments = (b1, b2, mu1, nu, curvature)
    inv = torch.rsqrt((sq + 2 * steps * mm(x, H.T) + steps.square() * width) / width + EPS)
    a = inv - inv0
    lin = a * b1.unsqueeze(1) + inv * steps * mu1_h
    change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * steps * nu_h
              + inv.square() * steps.square() * quad - 0.5 * lin.square())
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    zy = (base_y + steps * C.T[target]) * inv
    derivative = (change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(base_y * inv0 / 15))) / steps
    return mm(derivative * (y >= 0).unsqueeze(1), H) / width, moments


probe_moment_terms = torch.compile(probe_moment_terms_eager, dynamic=False)


@torch.compile(dynamic=False)
def probe_moment_terms_rows(x, y, weight, frames):
    """probe_moment_terms for consecutive rows of tokens, row r with its own frame frames[r] (NATIVE_WIDE), the
    rows' credits and moments concatenated by token."""
    t = x.shape[0] // len(frames)
    parts = [probe_moment_terms_eager(x[r * t:(r + 1) * t], y[r * t:(r + 1) * t], weight, H)
             for r, H in enumerate(frames)]
    return (torch.cat([part[0] for part in parts]),
            tuple(torch.cat([part[1][k] for part in parts]) for k in range(len(parts[0][1]))))


DESCENT_GRAPH = False  # with FUSED_GLUE: descending_inputs as one compiled graph (else per-layer compiled pieces)
LAYER_TAILS = False  # with FUSED_GLUE: the weight credits per block (layer_weight_credits) instead of batched


def descending_inputs_eager(raws, gates, blocks, c_vs, lam_r, x0, outs, moved: bool = False):
    """descending_chain's input responses and carried extras (corrected is formed by the caller), and
    path_scalar_credit's contractions: raws (tokens, width) per layer, blocks the bf16 block inputs. moved
    (DESCENT_BMM): c_vs holds the value responses raws[i] @ c_v[i] already."""
    L = len(raws)
    inputs, extra, path = [None] * L, [None] * L, [None] * L
    extra[L - 1] = torch.zeros_like(x0)
    for i in reversed(range(L)):
        inputs[i], carried, path[i] = chain_step(blocks[i], c_vs[i] if moved else raws[i] @ c_vs[i], gates[i], extra[i],
                                                 lam_r[i],
                                                 x0 if i == 0 else outs[i - 1], x0)
        if i:
            extra[i - 1] = carried
    return inputs, extra, torch.stack(path, dim=1)


@torch.compile(dynamic=False)
def chain_step(block, v, gate, extra, lam, previous, x0):
    """One layer of the descending chain: the value input response (+ the gate's), the extra carried down
    (lam * credit) and path_scalar_credit's two contractions of the credit (inputs + extra)."""
    inputs = norm_move(block.float().flatten(0, 1), v)
    if gate is not None:
        inputs = inputs + gate
    credit = inputs + extra
    return inputs, lam * credit, torch.stack([(credit * previous).sum(), (credit * x0).sum()])


descending_inputs_graph = torch.compile(descending_inputs_eager, dynamic=False)
TARGETS_BF16 = False  # chain_gradients' attention targets chain @ c_proj on bf16 operands (fp32 out; tf32 before)
REASSEMBLE_BF16 = False  # the head's weight credit (reassemble's einsum) on bf16 operands (fp32 out; tf32 before)
DESCENT_BF16 = False  # with DESCENT_BMM: its two batched GEMMs on bf16 operands (fp32 out; tf32 before)
DESCENT_BMM = True  # fused_descending_key_values' per-layer value-response and corrected-credit GEMMs batched per window
# group (the raw credits and the corrected buffers are stacked by group already)


@torch.compile(dynamic=False)
def stacked_sums(chains, extras):
    return torch.stack([c + e for c, e in zip(chains, extras)])
LEAN_CHAIN_TAIL = True  # the chain credits' carry, mixing-scalar sums and x0 credit in one compiled graph


@torch.compile(dynamic=False)
def chain_tail(measured, g_top, lam_r, lam_0, x0, outs):
    """Each block's mid credit (measured, else carried from the next block by its residual scalar; the last block
    from the readout), the mixing scalars' contractions and the credit reaching x0."""
    L = len(measured)
    credits = [None] * L
    credits[L - 1] = measured[L - 1] if measured[L - 1] is not None else g_top
    for i in reversed(range(L - 1)):
        credits[i] = measured[i] if measured[i] is not None else lam_r[i + 1] * credits[i + 1]
    lams = [torch.stack([(credits[i] * (x0 if i == 0 else outs[i - 1])).sum(), (credits[i] * x0).sum()])
            for i in range(L)]
    x0_credit = lam_r[0] * credits[0] + sum(lam_0[i] * credits[i] for i in range(L))
    return credits, lams, x0_credit


LEAN_GROUP_CREDITS = True  # a replay group's credits from its loss changes in one compiled graph


@torch.compile(dynamic=False)
def group_credits(loss_change, steps, controls, frames, boost: float):
    """Each root's credit from its replay group's loss changes: the identity-path predictions of every root removed,
    the remainder read along the root's frame and added back to its own prediction."""
    residual = loss_change
    for step, control, frame in zip(steps, controls, frames):
        residual = residual - step * (control @ frame.T)
    return [control + boost * ((residual / step) @ frame / frame.shape[0])
            for step, control, frame in zip(steps, controls, frames)]


@torch.compile(dynamic=False)
def gate_error(gate_out, ve_out, credit, heads: int):
    """Cut.exact_values' value-gate error: the gate's central-difference slope times the raw value credit's
    contraction with the value embedding."""
    b, t, _ = gate_out.shape
    z = gate_out.float()
    ve = ve_out.float().view(b, t, heads, -1)
    slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
    return slope * (ve * credit).sum(-1)


@torch.compile(dynamic=False)
def strong_queries(x, wq, cos, sin, heads: int):
    """Cut.strong_keys' RoPE'd unit queries (tokens, heads, head_dim) in one compiled graph."""
    length = x.shape[0]
    width = wq.shape[0] // heads
    c, s = cos[:, None], sin[:, None]
    r = (norm(x) @ wq.T).view(length, heads, width)
    return norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                           r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))


@torch.compile(dynamic=False)
def own_query_tensors(k_out, v_out, gate_out, ve_out, cos, sin, heads: int):
    """Cut.own_query_inputs' RoPE'd unit keys and gated values of one layer, (heads, tokens, head_dim) bf16."""
    length = k_out.shape[0]
    width = k_out.shape[1] // heads
    k = k_out.float().view(length, heads, width)
    k1, k2 = k[..., :width // 2], k[..., width // 2:]
    c, s = cos[:, None], sin[:, None]
    k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
    v = v_out.float().view(length, heads, width)
    if gate_out is not None:
        v = v + 2 * torch.sigmoid(gate_out.float().view(length, heads, 1)) * ve_out.float().view(length, heads, width)
    return k.transpose(0, 1).to(torch.bfloat16).contiguous(), v.transpose(0, 1).to(torch.bfloat16).contiguous()


@torch.compile(dynamic=False)
def strong_key_tensors(kk, vv, idx, p, out, all_keys):
    """The top keys (shifted by the tail centroid) and values of each query, and the values' tail remainder."""
    heads, length, width = kk.shape
    count = idx.shape[-1]
    flat = idx.reshape(heads, length * count, 1).expand(-1, -1, width)
    ktop = kk.gather(1, flat).view(heads, length, count, width)
    vtop = vv.gather(1, flat).view(heads, length, count, width)
    if FUSED_GLUE:  # the weighted sums over the top keys as fused reductions (not batched gemvs)
        rest = out - (p.unsqueeze(-1) * vtop.float()).sum(-2)
        tail_keys = all_keys - (p.unsqueeze(-1) * ktop.float()).sum(-2)
    else:
        rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
        tail_keys = all_keys - torch.einsum('htk,htkd->htd', p, ktop.float())
    mass = (1 - p.sum(-1, keepdim=True)).clamp_min(0)
    centroid = torch.where(mass > 1e-5, tail_keys / mass.clamp_min(1e-5), 0)
    shifted = (ktop.float() - centroid.unsqueeze(-2)).to(kk.dtype)
    if CHAIN_GLUE:  # the shifted top keys token-major, (tokens, heads, keys, width), as the replay reads them
        return shifted.transpose(0, 1).contiguous(), vtop, rest
    return shifted, vtop, rest


@torch.compile(dynamic=False)
def strong_key_shifted(kk, idx, p, all_keys):
    """strong_key_tensors' shifted top keys alone (GATHERED_VALUES: no vtop, no rest)."""
    heads, length, width = kk.shape
    count = idx.shape[-1]
    flat = idx.reshape(heads, length * count, 1).expand(-1, -1, width)
    ktop = kk.gather(1, flat).view(heads, length, count, width)
    tail_keys = all_keys - (p.unsqueeze(-1) * ktop.float()).sum(-2)
    mass = (1 - p.sum(-1, keepdim=True)).clamp_min(0)
    centroid = torch.where(mass > 1e-5, tail_keys / mass.clamp_min(1e-5), 0)
    shifted = (ktop.float() - centroid.unsqueeze(-2)).to(kk.dtype)
    return shifted.transpose(0, 1).contiguous()


@cache
def window_blocked(length, window, device):
    """The causal left-window mask's complement (True where a query may not see a key), built once."""
    positions = torch.arange(length, device=device)
    lag = positions[:, None] - positions[None, :]
    return ~((lag >= 0) & (lag <= window))


@triton.jit
def _row_topk_kernel(Scores, Tops, Indices, T: tl.constexpr, WINDOW: tl.constexpr, K: tl.constexpr):
    """Each query row's K largest scores inside its causal left window, largest first (repeated argmax)."""
    row = tl.program_id(0)
    query = row % T
    keys = tl.arange(0, T)
    allowed = (keys <= query) & (query - keys <= WINDOW)
    x = tl.load(Scores + row * T + keys, allowed, float('-inf')).to(tl.float32)
    for j in tl.static_range(K):
        best = tl.argmax(x, 0)
        tl.store(Tops + row * K + j, tl.max(x, 0).to(Tops.dtype.element_ty))
        tl.store(Indices + row * K + j, best.to(tl.int64))
        x = tl.where(keys == best, float('-inf'), x)


@triton.jit
def _window_fill_kernel(Scores, T: tl.constexpr, WINDOW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """-inf at every (query, key) outside the causal left window (scores (heads, T, T), written only there): tiles of
    BM queries x BN keys, fully blocked tiles by unmasked (vector) stores, edge tiles masked."""
    head, tile, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query = tile * BM + tl.arange(0, BM)
    keys = block * BN + tl.arange(0, BN)
    q0, q1, k0, k1 = tile * BM, tile * BM + BM - 1, block * BN, block * BN + BN - 1
    ptr = Scores + (head * T + query[:, None]).to(tl.int64) * T + keys[None, :]
    fill = tl.full((BM, BN), float('-inf'), Scores.dtype.element_ty)
    if (k0 > q1) | (q0 - k1 > WINDOW):  # every key after every query, or before every query's window
        tl.store(ptr, fill)
    elif (k1 > q0) | (q1 - k0 > WINDOW):  # an edge tile
        lag = query[:, None] - keys[None, :]
        tl.store(ptr, fill, (lag < 0) | (lag > WINDOW))


def window_fill_(scores, window):
    """scores.masked_fill_(window_blocked(...), -inf) writing only the blocked entries."""
    heads, length, _ = scores.shape
    assert scores.is_contiguous() and length % 64 == 0
    _window_fill_kernel[(heads, length // 32, length // 64)](scores, T=length, WINDOW=window, BM=32, BN=64,
                                                              num_warps=4)
    return scores


@triton.jit
def _ordered_bits(x):
    """fp32 -> int32 in the same order (-inf lowest, -0 below +0, as torch's radix select orders them)."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _route_topk_kernel(Scores, Indices, Scratch, N: tl.constexpr, K: tl.constexpr):
    """torch.topk(scores, K).indices of one row exactly (values descending, equal values by ascending index): the K-th
    largest value by binary search over its ordered bits, the values above it and the lowest-index ties, sorted."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    key = _ordered_bits(tl.load(Scores + row * N + cols).to(tl.float32))
    lo = tl.full((), -2147483648, tl.int64)
    hi = tl.full((), 2147483647, tl.int64)
    for _ in tl.static_range(32):  # the largest t with count(key >= t) >= K
        mid = (lo + hi + 1) // 2
        enough = tl.sum((key >= mid.to(tl.int32)).to(tl.int32), 0) >= K
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid - 1)
    kth = lo.to(tl.int32)
    equal = key == kth
    need = K - tl.sum((key > kth).to(tl.int32), 0)
    chosen = (key > kth) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= need))
    slot = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(Scratch + row * K + slot, (key.to(tl.int64) << 32) | (N - 1 - cols).to(tl.int64), chosen)
    tl.debug_barrier()
    top = tl.sort(tl.load(Scratch + row * K + tl.arange(0, K)), descending=True)
    tl.store(Indices + row * K + tl.arange(0, K), N - 1 - (top & 0xFFFFFFFF))


@triton.jit
def _route_topk_pruned_kernel(Scores, Indices, Scratch, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                              CAP: tl.constexpr):
    """_route_topk_kernel's indices, bitwise: floor = the K-th largest of the N / G group maxima (count(key >= floor)
    >= K, so every top-K key is >= it); the candidates key >= floor, if at most CAP (else the binary search from the
    floor), compacted and top-K'd as (key, N - 1 - col) (values descending, equal values by ascending index)."""
    row = tl.program_id(0).to(tl.int64)
    NG: tl.constexpr = N // G
    cols = tl.arange(0, NG)[:, None] * G + tl.arange(0, G)[None, :]
    key = _ordered_bits(tl.load(Scores + row * N + cols).to(tl.float32))
    floor = tl.min(tl.topk(tl.max(key, 1), K), 0)
    packed = (key.to(tl.int64) << 32) | (N - 1 - cols).to(tl.int64)
    cand = key >= floor
    per_group = tl.sum(cand.to(tl.int32), 1)
    count = tl.sum(per_group, 0)
    if count <= CAP:
        slot = (tl.cumsum(per_group, 0) - per_group)[:, None] + tl.cumsum(cand.to(tl.int32), 1) - 1
        tl.store(Scratch + row * CAP + slot, packed, cand)
        tl.debug_barrier()
        c = tl.arange(0, CAP)
        top = tl.topk(tl.load(Scratch + row * CAP + c, c < count, other=-9223372036854775808), K)
    else:  # many ties at the floor: the K-th largest key by binary search, the keys above it and the first ties
        lo = floor.to(tl.int64)
        hi = tl.full((), 2147483647, tl.int64)
        for _ in tl.static_range(32):
            mid = (lo + hi + 1) // 2
            enough = tl.sum(tl.sum((key >= mid.to(tl.int32)).to(tl.int32), 1), 0) >= K
            lo = tl.where(enough, mid, lo)
            hi = tl.where(enough, hi, mid - 1)
        kth = lo.to(tl.int32)
        chosen = key > kth
        need = K - tl.sum(tl.sum(chosen.to(tl.int32), 1), 0)
        equal = (key == kth).to(tl.int32)
        equal_groups = tl.sum(equal, 1)
        rank = (tl.cumsum(equal_groups, 0) - equal_groups)[:, None] + tl.cumsum(equal, 1)
        chosen = chosen | ((equal != 0) & (rank <= need))
        chosen_groups = tl.sum(chosen.to(tl.int32), 1)
        slot = (tl.cumsum(chosen_groups, 0) - chosen_groups)[:, None] + tl.cumsum(chosen.to(tl.int32), 1) - 1
        tl.store(Scratch + row * CAP + slot, packed, chosen)
        tl.debug_barrier()
        top = tl.sort(tl.load(Scratch + row * CAP + tl.arange(0, K)), descending=True)
    tl.store(Indices + row * K + tl.arange(0, K), N - 1 - (top & 0xFFFFFFFF))


def route_topk(scores, count):
    """scores.topk(count, dim=-1).indices for (..., N) fp32 scores (N a power of two), exactly."""
    n = scores.shape[-1]
    rows = scores.numel() // n
    indices = torch.empty((*scores.shape[:-1], count), device=scores.device, dtype=torch.int64)
    if TOPK_PRUNED:
        cap, groups = 2 * count, min(8 * count, n)
        scratch = torch.empty((rows, cap), device=scores.device, dtype=torch.int64)
        _route_topk_pruned_kernel[(rows,)](scores.contiguous(), indices, scratch, N=n, K=count, G=n // groups,
                                           CAP=cap, num_warps=1)
        return indices
    scratch = torch.empty((rows, count), device=scores.device, dtype=torch.int64)
    _route_topk_kernel[(rows,)](scores.contiguous(), indices, scratch, N=n, K=count, num_warps=2)
    return indices


def stream_route_moments(q, keys, values, window, count):
    heads, length, width = keys.shape
    if LEAN_STRONG_KEYS:  # (heads, query, key) straight from a batched matmul, no permuted copy
        scores = torch.bmm(q.to(keys.dtype).transpose(0, 1), keys.transpose(1, 2))
    else:
        scores = torch.einsum('thd,hsd->hts', q.to(keys.dtype), keys).contiguous()
    if FUSED_GLUE:  # the stream kernel reads it masked
        window_fill_(scores, window)
    else:
        scores.masked_fill_(window_blocked(length, window, q.device), float('-inf'))
    if TRITON_TOPK:
        tops = torch.empty((heads, length, count), device=q.device, dtype=scores.dtype)
        indices = torch.empty((heads, length, count), device=q.device, dtype=torch.int64)
        _row_topk_kernel[(heads * length,)](scores, tops, indices, T=length, WINDOW=window, K=count, num_warps=4)
    elif WINDOW_TOPK:  # torch.topk's selection and order, bitwise (bf16 rows, count <= 32)
        tops, indices = (window_topk.window_topk_pruned if WINDOW_TOPK_PRUNED else window_topk.window_topk)(scores, count)
    else:
        tops, indices = scores.topk(count, dim=-1)
    probabilities = torch.empty((heads, length, count), device=q.device)
    out = torch.empty((heads, length, width), device=q.device)
    all_keys = torch.empty_like(out)
    _route_moment_stream_kernel[(length // 32, heads)](scores, keys, values, tops, probabilities, out, all_keys,
      T=length, WINDOW=window, KEYS=count, BM=32, BN=128, num_warps=4, num_stages=2)
    return probabilities, indices, out, all_keys


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    """PR #44's readout probe, verbatim: forward differences of each token's readout loss along every direction
    of a full cycle; the moved stream's RMS is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


class Cut:
    def __init__(self, model, settings, seed, rank, world):
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model, seed + 1000 * rank)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * rank)
        self.move_generator = torch.Generator(self.device).manual_seed(seed + 23 + 1000 * rank)
        self.sign_generator = torch.Generator(self.device).manual_seed(seed + 29 + 1000 * rank)
        self.partner = None  # CONCURRENT_WIDE: a second Cut on the same model whose graph estimates a wide update's row 2
        self._estimate_graph = None
        self._telemetry_health = []
        self._query_moment_errors = {}
        self._side_stream = torch.cuda.Stream()
        self._sources = [self]  # each estimated row's draw source (NATIVE_WIDE: row 2 draws from the partner's generators)
        self._amortize = None  # AMORTIZED_ROOTS: this estimate 'measure's the roots or 'skip's them
        self._modes = {}  # AMORTIZED_ROOTS: each mode's captured graphs and buffers (Cut.use_mode)
        self._updates = 0
        self._window = (0, 1 << 30)  # AMORTIZED_ROOTS: the amortized updates (Cut.amortized_window)
        self._root_scale = torch.full((), float(ROOT_EVERY), device=self.device)  # ROOT_BOOST: the boost, a graph input
        self._boosting = True
        self._measured = False  # AMORTIZED_ROOTS: a deviation measured (bind_flat_layout makes the EMA's buffers)
        global FUSED_CHAIN
        if FUSED_CHAIN != settings.fused_chain:
            FUSED_CHAIN = settings.fused_chain
            torch.compiler.reset()
        visit_chain.ROOT_CONFIG['NS'] = 2 if ROOT_CHAIN_NS2 else 3  # read at each launch (outside the compiled graph)
        visit_chain3.CONFIG['DEFER'] = 2 if CHAIN3_DEFER_READOUT else 0  # likewise
        visit_chain.ROOT_CONFIG['TC'] = int(ROOT_CHAIN_TC)
        visit_chain.ROOT_CONFIG['DEFER'] = 3 if ROOT_CHAIN_DEFER_READOUT else 1
        visit_chain3.CONFIG['TC'] = int(CHAIN3_TC)
        moment_kernels.VALUE_IMPL = value_gluon.value_pass_gluon if VALUE_GLUON else None
        moment_kernels.KEY_IMPL = key_gluon.key_scan_gluon if KEY_GLUON else None
        key_gluon.CONFIG['PF'] = int(KEY_LSE_TMA)
        value_gluon.CONFIG['PF'] = int(VALUE_LSE_TMA)
        value_gluon.CONFIG['HI'] = int(VALUE_HI_ONLY)
        query_factored_gluon.ROT_CONFIG['LO'] = not QF_KEYS_HI
        visit_chain3.CONFIG.update(dict(BM=128, num_warps=8, NS=4) if CHAIN3_WIDE else dict(BM=64, num_warps=4, NS=2))
        visit_chain.ROOT_CONFIG['PK'] = visit_chain3.CONFIG['PK'] = int(CHAIN_PACKED_CVT)
        visit_gluon_root.ROOT_CONFIGS[64]['PK'] = int(CHAIN_PACKED_CVT)
        self._gate_rows = {i: torch.empty(self.width, self.width, device=self.device)
        for i in range(self.model.config.n_layer)
        if has_ve(i, self.model.config.n_layer)}
    def root_boost(self, group):
        """ROOT_BOOST, a measuring update: an amortized group's deviations count self._root_scale times (ROOT_EVERY
        inside the amortized window, 1 outside it; a graph input)."""
        if ROOT_BOOST and self._amortize == 'measure' and set(group) & set(AMORTIZED_ROOTS):
            return self._root_scale
        return 1.0
    def amortized_window(self, updates):
        """AMORTIZED_ROOTS: the updates [ROOT_FROM, ROOT_UNTIL) of the Cut's `updates` skip the roots on all but every
        ROOT_EVERY-th; the rest measure them (ROOT_BOOST: with the deviation counted once)."""
        self._window = (round(ROOT_FROM * updates), round(ROOT_UNTIL * updates))
    def measured_groups(self):
        """The replayed root groups: all, or (AMORTIZED_ROOTS, a skipping update) those without an amortized root."""
        if self._amortize != 'skip':
            return self.s.joint_root_groups
        return tuple(group for group in self.s.joint_root_groups if not set(group) & set(AMORTIZED_ROOTS))
    MODE_STATE = ('_estimate_graph', '_static_gradients', '_packed', '_static_x', '_static_y', '_wide_graph',
                  '_wide_packed', '_wide_loss', '_wide_gradients', '_wide_x', '_wide_y', '_row_offsets')
    CAPTURE_STATE = ('inputs', 'outputs', 'blocks', 'clean_loss', 'y', 'batch', 'length')
    def use_mode(self, mode):
        """AMORTIZED_ROOTS: switch this Cut (and its partner's draws) to mode's estimate and its captured graphs."""
        if self._amortize is not None and self._amortize in self._modes:
            self._modes[self._amortize] = self.mode_state()
        self._amortize = mode
        if self.partner is not None:
            self.partner._amortize = mode
        for name, value in self._modes.get(mode, {}).items():
            if name.startswith('capture.'):
                setattr(self.capture, name[8:], value)
            else:
                setattr(self, name, value)
    def mode_state(self):
        state = {name: getattr(self, name, None) for name in self.MODE_STATE}
        state.update({'capture.' + name: getattr(self.capture, name, None) for name in self.CAPTURE_STATE})
        return state
    def capture_modes(self, capture):
        """AMORTIZED_ROOTS: capture(self) once per mode (each mode's graphs and buffers kept apart)."""
        for mode in ('skip', 'measure'):
            self._modes[mode] = {}
            self.use_mode(mode)
            for name in self.MODE_STATE:
                setattr(self, name, None)
            capture(self)
            self._modes[mode] = self.mode_state()
    def next_mode(self):
        """AMORTIZED_ROOTS: inside the window, measure on every ROOT_EVERY-th update (its first included) and skip on
        the rest; outside it, measure (ROOT_BOOST: unboosted; the EMA mode keeps the window at all updates)."""
        first, last = self._window
        inside = first <= self._updates < last
        mode = 'measure' if not inside or (self._updates - first) % ROOT_EVERY == 0 else 'skip'
        if ROOT_BOOST and inside != self._boosting:
            self._root_scale.fill_(float(ROOT_EVERY) if inside else 1.0)
            self._boosting = inside
        self._updates += 1
        if mode != self._amortize:
            self.use_mode(mode)
    def reset_deviation(self):
        """AMORTIZED_ROOTS: no deviation measured yet (the next update measures and takes its deviation whole)."""
        self._deviation.zero_()
        self._root_weight.fill_(1.0)
        self._root_scale.fill_(float(ROOT_EVERY))
        self._boosting = True
        self._updates = 0
        self._measured = False
    def measured(self):
        """AMORTIZED_ROOTS: after a measuring update, later measurements move the EMA by 1 - ROOT_BETA."""
        if self._amortize == 'measure' and not self._measured:
            self._root_weight.fill_(1 - ROOT_BETA)
            self._measured = True
    def basis(self, row=0):
        generator = self._sources[row].generator
        signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator).float() * 2 - 1
        return hadamard(self.width, self.device, torch.float32) * signs
    def query_errors(self, draws, targets):
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L):
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            fixed, vals = kp.to(dt), values.to(dt)
            baseline = attention(processed(q_raw).to(dt), fixed, vals, window)
            a = d.query_noise(draws, b, t, cfg.n_embd).view(draws, b, t, h, hd)
            moved = moved_heads(q_raw, a, sigma, cos, sin)
            out = query_probe_attention(moved, fixed, vals, window, draws, b)
            estimate = code_estimate(head_scores(out, baseline, target), a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors
    def hidden_credit(self, u, g, out_weight):
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return hidden_local_credit(u, steps, g, out_weight.float())
    def batched_key_contexts(self, targets, defer=False):
        """key_context, the query moments and the key credits of every layer, the layers of one window size stacked
        along the head axis so that each kernel launches once per window size. Returns per-layer contexts and key
        credits (heads, tokens, head_dim); the query moment credits go to self._query_moment_errors."""
        d, model, s = self.capture, self.model, self.s
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        windows = {}
        for i in range(L):
            windows.setdefault(model.window_sizes[i][0], []).append(i)
        contexts, work = {}, []
        early = EARLY_MOMENT_INPUTS and self.fuses_glue()  # (read by batched_key_scores_by_token)
        self._moment_inputs = {}
        rows = d.batch  # (several rows: each layer's rows stacked along the head axis after it, layer-major)
        if rows > 1 and not (early and defer):
            raise ValueError('a several-row estimate needs the fused glue')
        for window, layers in windows.items():
            pairs = [(i, r) for i in layers for r in range(rows)]
            pick = lambda name: [d.outputs[f'transformer.h.{i}.attn.{name}'][r] for i, r in pairs]
            ves = [d.outputs[f'value_embeds.{i}'][r] if has_ve(i, L) else None for i, r in pairs]
            gates = [d.outputs[f'transformer.h.{i}.attn.ve_gate'][r] if has_ve(i, L) else None for i, r in pairs]
            if early:
                (qp, kp, k_raw, values, hub), self._moment_inputs[window] = key_context_group_moments(
                    pick('c_q'), pick('c_k'), pick('c_v'), ves, gates, [targets[i][r] for i, r in pairs], cos, sin, h,
                    s.query_moment_sigma)
            else:
                qp, kp, k_raw, values, hub = key_context_group(pick('c_q'), pick('c_k'), pick('c_v'), ves, gates,
                                                               [targets[i][0] for i in layers], cos, sin, h)
            operands = (qp, kp, values, hub)
            if IMPLICIT_BF16 and early:  # from the query moments' bf16 copies (mode 1: all four; 2: values and hub;
                # 3: q and k)
                q16, k16, v16, hub16 = self._moment_inputs[window][:4]
                operands = {1: (q16, k16, v16, hub16), 2: (qp, kp, v16, hub16), 3: (q16, k16, values, hub)}[IMPLICIT_BF16]
            if IMPLICIT_GLUON and (IMPLICIT_GLUON_BF16 or not (IMPLICIT_BF16 and early)):  # (Gluon: fp32, or bf16 with
                # IMPLICIT_GLUON_BF16 and IMPLICIT_BF16 1)
                lse, center = implicit_gluon.implicit_key_context(*operands, window)
            else:
                lse, center = ((moment_kernels.implicit_key_context_edges if IMPLICIT_EDGES
                                else moment_kernels.implicit_key_context) if FAST_MOMENTS else implicit_key_context)(
                    *operands, window)
            for j, i in enumerate(layers if rows == 1 else ()):
                heads = slice(j * h, (j + 1) * h)
                contexts[i] = (qp[heads], kp[heads], k_raw[heads], cos, sin, lse[heads],
                               (values[heads], hub[heads], center[heads]))
            work.append((window, layers, qp, kp, k_raw, values, hub, lse, center))
        if defer:
            return contexts, work
        return contexts, self.batched_key_scores(work)
    def value_logit_inputs(self, window, qp, kp):
        """The value passes' q / k: VALUE_QK16, the query moments' bf16 copies."""
        if VALUE_QK16 and VALUE_GLUON and window in self._moment_inputs:
            return self._moment_inputs[window][:2]
        return qp, kp
    def batched_key_scores(self, work):
        """The key credits and query moments of batched_key_contexts' window groups."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys = {}
        for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
            if FAST_KEYS:
                key = moment_kernels.key_credits(qp, kp, k_raw, lse, values, hub, center, cos, sin, window,
                                                 s.key_sigma)
            else:
                key = torch.empty_like(k_raw)
                _key_scan_kernel[len(layers) * h, t // 32](
                    qp, kp, k_raw, lse, values, hub, center, cos, sin, key, T=t, D=hd, Window=window,
                    Sigma=s.key_sigma, BT=32, BS=64, J=32, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
            if s.query_moments:
                q_out = torch.cat([d.outputs[f'transformer.h.{i}.attn.c_q'][0].float().view(t, h, hd).transpose(0, 1)
                                   for i in layers], 0)
                q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(
                    q_out.transpose(0, 1).reshape(t, -1), qp, kp, values, hub, s.query_moment_sigma)
                out = torch.empty(1, len(layers) * h, t, hd, device=self.device, dtype=torch.bfloat16)
                (moment_kernels.moment_query_scores if FAST_MOMENTS else moment_query_scores)(q16, k16, v16, hub16, q_raw, sq, kp.contiguous(), cos[:, :hd // 2].contiguous(),
                                    sin[:, :hd // 2].contiguous(), lse, center, steps, out, hd // 16, window=window)
            for j, i in enumerate(layers):
                rows = slice(j * h, (j + 1) * h)
                keys[i] = key[rows]
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = (
                        out[0, rows].float().transpose(0, 1).reshape(1, t, h * hd))
        return keys
    def batched_key_scores_by_token(self, work, qkv, maps):
        """batched_key_scores with each layer's query and key credits written into columns [0, width) and
        [width, 2 width) of the (layers, tokens, 3 width) buffer qkv (window group g's layers at maps[g]); returns the
        key credits as (tokens, width) views (the query credits go to self._query_moment_errors as
        (1, tokens, width) views)."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys, w = {}, self.width
        rows = d.batch  # (qkv's (layer, row) stretches as (layers x rows, tokens, 3 width), maps' indices into it)
        by_row = qkv.view(-1, t, qkv.shape[-1])
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            if window in self._moment_inputs:
                q16, k16, v16, hub16, q_raw, sq, steps = self._moment_inputs[window]
            else:
                q_out = torch.cat([d.outputs[f'transformer.h.{i}.attn.c_q'][0] for i in layers], 1)
                q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(q_out, qp, kp, values, hub,
                                                                             s.query_moment_sigma)
            key_inputs = (q16, k16, k_raw, lse, v16, hub16, center) if KEY_QK16 and KEY_GLUON else (
                qp, kp, k_raw, lse, values, hub, center)
            moment_kernels.key_credits_into(*key_inputs, cos, sin, window, s.key_sigma, by_row[:, :, w:2 * w], maps[g])
            moment_kernels.moment_query_scores_into(
                q16, k16, v16, hub16, q_raw, sq, kp, cos[:, :hd // 2].contiguous(), sin[:, :hd // 2].contiguous(),
                lse, center, steps, by_row[:, :, :w], window=window, layers=maps[g])
            for i in layers:
                keys[i] = qkv[i, :, w:2 * w]
                self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = qkv[i, :, :w].view(rows, t, w)
        return keys
    def query_moment_credit(self, i, context):
        """Each raw query coordinate moved by a finite step through RoPE and the query norm, the attention's response
        to it from the degree-2 moment model (the identity phase's query credit, with this layer's window)."""
        qp, kp, _, cos, sin, lse, (values, hub, center) = context
        heads, t, hd = qp.shape
        q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(
            self.capture.outputs[f'transformer.h.{i}.attn.c_q'][0], qp, kp, values, hub, self.s.query_moment_sigma)
        out = torch.empty(1, heads, t, hd, device=self.device, dtype=torch.bfloat16)
        half = hd // 2
        moment_query_scores(q16, k16, v16, hub16, q_raw, sq, kp.contiguous(), cos[:, :half].contiguous(),
                            sin[:, :half].contiguous(), lse.contiguous(), center.contiguous(), steps, out, hd // 16,
                            window=self.model.window_sizes[i][0])
        return out[0].float().transpose(0, 1).reshape(1, t, heads * hd)
    def internals(self, gradients, targets, b, t, key_errors, raw_values, terminal_values):
        d, s = self.capture, self.s
        errors = self._query_moment_errors if s.query_moments else self.query_errors(s.query_draws, targets)
        errors.update(key_errors)
        for i in targets:
            errors.update(self.exact_values(i, raw_values[i], terminal_values[i]))
        fused = self.fuses_glue()
        if fused:  # every layer's query, key and value weight credits from one batched GEMM of the side-by-side errors
            if INTERNALS_BF16:
                weight = qkv_weight_credits_bf16(self._qkv, d.inputs['stack:attn.c_q:bf16'])
            else:
                weight = torch.bmm(self._qkv.transpose(1, 2), d.inputs['stack:attn.c_q'].flatten(1, 2))
            for i in targets:
                p = f'transformer.h.{i}.attn.'
                for k, name in enumerate(('c_q', 'c_k', 'c_v')):
                    gradients[p + name + '.weight'] = weight[i, k * self.width:(k + 1) * self.width]
        for site, error in errors.items():
            if not (fused and site.endswith(('attn.c_q', 'attn.c_k', 'attn.c_v'))):
                gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
    def bind_flat_layout(self, optimizer):
        """Cut.step's flat gradient layout (ANVIL's shape groups first, in its batch order). Bound before the estimate
        graph's capture, it lets the estimate pack its gradients there (FLAT_GRADIENTS)."""
        params = list(self.model.named_parameters())
        anvil = optimizer.optimizers[0]
        anvil_ids = ([id(q) for q in anvil.rank_major_order()] if RANK_MAJOR
                     else [id(q) for batch in anvil.batches for q in batch[0]])
        rank_of = {pid: k for k, pid in enumerate(anvil_ids)}
        self._flat_order = sorted(params, key=lambda item: rank_of.get(id(item[1]), len(anvil_ids)))
        self._packs = FLAT_GRADIENTS and STACKED_ANVIL and LEAN_FLAT and SHARDED_ANVIL and not DELAYED_UPDATE
        if AMORTIZED_ROOTS:  # the deviations' EMA and its update weight (1 at the first measurement), graph inputs
            self._deviation = torch.zeros(sum(p.numel() for _, p in params), device=self.device)
            self._root_weight = torch.ones((), device=self.device)
    def packs_gradients(self):
        return getattr(self, '_packs', False)
    def step(self, x, y, optimizer):
        if AMORTIZED_ROOTS:
            self.next_mode()
        gradients = self.estimate_rows(x, y)
        if AMORTIZED_ROOTS and self.packs_gradients():
            self.measured()
        return self.update(gradients, y, optimizer, lambda: torch.where(y != -1, self.capture.clean_loss, 0).sum())
    def estimate_rows(self, x, y):
        if self._estimate_graph is None:
            if PREFETCH_GATE_ROWS:
                self.take_gate_rows()
            else:
                self.prepare_gate_rows()
            return self.estimate(x, y)
        with self.staging():
            if PREFETCH_GATE_ROWS:
                self.take_gate_rows()
            else:
                self.prepare_gate_rows()
            self._static_x.copy_(x)
            self._static_y.copy_(y)
        self.replay_graph(self._estimate_graph)
        self.staged_replay_done()
        if PREFETCH_GATE_ROWS:
            self.prefetch_gate_rows()
        return self._static_gradients
    @contextlib.contextmanager
    def staging(self):
        """PRE_STAGE: the ops inside run on a side stream once the previous estimate replay is done with its inputs (not
        after the previous update's step, which the main stream is still running), and the main stream waits for them."""
        if not PRE_STAGE:
            yield
            return
        main = torch.cuda.current_stream()
        if not hasattr(self, '_stage_stream'):  # (first use: after everything the main stream has queued, e.g. the rows' copy)
            self._stage_stream, self._replay_done = torch.cuda.Stream(priority=PREFETCH_PRIORITY), torch.cuda.Event()
            self._stage_stream.wait_stream(main)
        stage = self._stage_stream
        stage.wait_event(self._replay_done)
        with torch.cuda.stream(stage):
            yield
        if not EARLY_REPLAY:
            main.wait_stream(stage)
    def staged_replay_done(self):
        if PRE_STAGE:
            self._replay_done.record()
    def prepare_step_event(self):
        """EARLY_REPLAY (before a capture): the external event the estimate graphs open with, recorded once."""
        if EARLY_REPLAY and not hasattr(self, '_step_done'):
            if not PRE_STAGE:
                raise ValueError('EARLY_REPLAY needs PRE_STAGE')
            self._step_done = torch.cuda.Event(external=True)
            self._step_done.record()
    def wait_step_in_graph(self):
        """EARLY_REPLAY (inside a capture): the graph's first node waits for self._step_done's last record."""
        if EARLY_REPLAY:
            torch.cuda.current_stream().wait_event(self._step_done)
    def replay_graph(self, graph):
        """Replay an estimate graph (EARLY_REPLAY: on the staging stream, after its staged copies; the graph waits for
        the main stream's queue so far through self._step_done, and the main stream for the graph)."""
        if not EARLY_REPLAY:
            graph.replay()
            return
        main = torch.cuda.current_stream()
        self._step_done.record()
        with torch.cuda.stream(self._stage_stream):
            graph.replay()
        main.wait_stream(self._stage_stream)
    def wide_step(self, x, y, optimizer):
        """Cut.step over several rows per GPU (wide_rows): the one-row estimate graph replayed once per row, the packed
        gradients averaged over the rows, then one update."""
        if self._estimate_graph is None or not self.packs_gradients() or DELAYED_UPDATE:
            raise ValueError('wide updates need the captured estimate graph and packed gradients')
        if AMORTIZED_ROOTS:
            if not (self.partner is not None and NATIVE_WIDE and (x.shape[0] == 2 or ROOT_BOOST)):
                raise ValueError('AMORTIZED_ROOTS: wide updates need NATIVE_WIDE (and, with the EMA, two rows per GPU)')
            self.next_mode()
        if self.partner is not None and NATIVE_WIDE:  # the update reads the pair graph's packed buffers
            loss_sum = self.native_wide_estimate(x, y)
            if AMORTIZED_ROOTS:
                self.measured()
            one_row, self._packed = self._packed, self._wide_packed
            try:
                return self.update(self._static_gradients, y, optimizer,
                                   loss_sum if callable(loss_sum) else lambda: loss_sum)
            finally:
                self._packed = one_row
        if self.partner is not None:
            loss_sum = self.concurrent_wide_estimate(x, y)
            return self.update(self._static_gradients, y, optimizer, lambda: loss_sum)
        loss_sum = self.wide_estimate(x, y)
        return self.update(self._static_gradients, y, optimizer, lambda: loss_sum)
    def wide_estimate(self, x, y):
        """The rows' estimates one after another, their mean packed in self._packed; returns the rows' loss sum."""
        rows, loss_sum = x.shape[0], 0
        for j in range(rows):
            gradients = self.estimate_rows(x[j:j + 1], y[j:j + 1])
            flat, wire = self._packed
            loss_sum = loss_sum + torch.where(y[j:j + 1] != -1, self.capture.clean_loss, 0).sum()
            if j == 0:
                if getattr(self, '_wide_sum', None) is None:
                    self._wide_sum = torch.empty_like(flat)
                self._wide_sum.copy_(flat)
            elif j < rows - 1:
                self._wide_sum.add_(flat)
            else:
                flat.add_(self._wide_sum).div_(rows)
                wire.copy_(flat)
        return loss_sum
    def concurrent_wide_estimate(self, x, y):
        """wide_estimate with the rows in pairs: each pair's second row on the partner's graph, replayed on a side stream
        under the first's. The partner's generators take this Cut's states after the first replay and hand theirs back
        after the second, so every row draws what wide_estimate's replays would, and the rows' sum keeps its order
        (bitwise the same mean: ((A + B) + C) + D, with x + y = y + x)."""
        rows = x.shape[0]
        if rows % 2:
            raise ValueError('CONCURRENT_WIDE estimates an even number of rows per GPU')
        partner, main = self.partner, torch.cuda.current_stream()
        side = partner._side_replay = getattr(partner, '_side_replay', None) or torch.cuda.Stream(
            priority=WIDE_SIDE_PRIORITY)
        if WIDE_MAIN_PRIORITY:  # this Cut's replay on its own stream of that priority (joined right after)
            lane = self._main_replay = getattr(self, '_main_replay', None) or torch.cuda.Stream(
                priority=WIDE_MAIN_PRIORITY)
        loss_sum = None
        for first in range(0, rows, 2):
            mine_rows, their_rows = slice(first, first + 1), slice(first + 1, first + 2)
            if PREFETCH_GATE_ROWS:
                self.take_gate_rows()
            else:
                self.prepare_gate_rows()
            self._static_x.copy_(x[mine_rows])
            self._static_y.copy_(y[mine_rows])
            partner._static_x.copy_(x[their_rows])
            partner._static_y.copy_(y[their_rows])
            side.wait_stream(main)
            start = self.generator.get_offset()
            if WIDE_MAIN_PRIORITY:
                lane.wait_stream(main)
                with torch.cuda.stream(lane):
                    self._estimate_graph.replay()
                main.wait_stream(lane)
            else:
                self._estimate_graph.replay()
            replay_offset = self.generator.get_offset() - start
            for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
                theirs.set_state(mine.get_state())
            with torch.cuda.stream(side):
                if PREFETCH_GATE_ROWS:
                    partner.take_gate_rows()
                else:
                    partner.prepare_gate_rows()
                partner._estimate_graph.replay()
            for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
                mine.set_state(theirs.get_state())
            if PREFETCH_GATE_ROWS:  # the next pair's rows: this Cut's from its state, the partner's one replay on
                self.prefetch_gate_rows()
                ahead = torch.Generator(self.device)
                ahead.set_state(self.generator.get_state())
                ahead.set_offset(ahead.get_offset() + replay_offset)
                partner.prefetch_gate_rows(ahead.get_state())
            main.wait_stream(side)
            flat, wire = self._packed
            theirs_flat = partner._packed[0]
            if rows == 2:
                flat.add_(theirs_flat).div_(2)
            elif first == 0:
                self._wide_sum.copy_(flat).add_(theirs_flat)
            elif first + 2 < rows:
                self._wide_sum.add_(flat).add_(theirs_flat)
            else:
                self._wide_sum.add_(flat)
                torch.add(theirs_flat, self._wide_sum, out=flat).div_(rows)
            pair_loss = (torch.where(y[mine_rows] != -1, self.capture.clean_loss, 0).sum(),
                         torch.where(y[their_rows] != -1, partner.capture.clean_loss, 0).sum())
            loss_sum = pair_loss[0] + pair_loss[1] if loss_sum is None else loss_sum + pair_loss[0] + pair_loss[1]
        wire.copy_(flat)
        return loss_sum
    def native_wide_estimate(self, x, y):
        """NATIVE_WIDE: wide_estimate with the rows in pairs, each pair by one replay of the two-row estimate graph
        (capture_pair_graph). Row 2 draws from the partner's generators, set to this Cut's states one row's draws on
        (what concurrent_wide_estimate's partner replay draws), and this Cut's take theirs after the replay. The pair
        graph packs the pair's mean into self._wide_packed; returns the rows' loss sum."""
        rows, partner = x.shape[0], self.partner
        if rows % 2:
            raise ValueError('NATIVE_WIDE estimates an even number of rows per GPU')
        flat, wire = self._wide_packed
        loss_sum = None
        for first in range(0, rows, 2):
            pair = slice(first, first + 2)
            for mine, theirs, step in zip(self.graph_generators(), partner.graph_generators(), self._row_offsets):
                theirs.set_state(mine.get_state())
                theirs.set_offset(mine.get_offset() + step)
            with self.staging():
                if PREFETCH_GATE_ROWS:
                    self.take_gate_rows()
                    partner.take_gate_rows()
                else:
                    self.prepare_gate_rows()
                    partner.prepare_gate_rows()
                self._wide_x.copy_(x[pair])
                self._wide_y.copy_(y[pair])
            self.replay_graph(self._wide_graph)
            self.staged_replay_done()
            for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
                mine.set_state(theirs.get_state())
            if PREFETCH_GATE_ROWS:  # the next update's rows: this Cut's from its state, the partner's one row on
                self.prefetch_gate_rows()
                ahead = torch.Generator(self.device)
                ahead.set_state(self.generator.get_state())
                ahead.set_offset(ahead.get_offset() + self._row_offsets[0])
                partner.prefetch_gate_rows(ahead.get_state())
            if LATE_LOSS and rows == 2:  # (summed when the update asks: the graph's loss buffer is read before the next replay)
                losses = self._wide_loss
                return lambda: torch.where(y != -1, losses, 0).sum()
            pair_loss = torch.where(y[pair] != -1, self._wide_loss, 0).sum()
            loss_sum = pair_loss if loss_sum is None else loss_sum + pair_loss
            if rows == 2:
                break
            if first == 0:
                self._wide_sum.copy_(flat)
            elif first + 2 < rows:
                self._wide_sum.add_(flat)
            else:
                flat.add_(self._wide_sum).div_(rows // 2)
                wire.copy_(flat)
        return loss_sum
    def capture_pair_graph(self, x, y):
        """NATIVE_WIDE: capture the estimate of two rows (x, y: (2, T)) as one graph, row 2 drawing its frames, signs and
        gate rows from the partner's generators. This Cut's one-row graph state (packed buffers, the capture's
        tensors) is kept; the pair graph's are self._wide_*. Measures each generator's draws per row."""
        partner = self.partner
        kept = (self._packed, self.capture.inputs, self.capture.outputs, self.capture.blocks, self.capture.clean_loss,
                self.capture.y, self.capture.batch, self.capture.length)
        self._wide_x, self._wide_y = x.clone(), y.clone()
        generators = self.graph_generators() + partner.graph_generators()
        states = [g.get_state() for g in generators]
        default_state = torch.cuda.get_rng_state()
        def restore_rng():
            for generator, state in zip(generators, states):
                generator.set_state(state)
            torch.cuda.set_rng_state(default_state)
        self._sources = [self, partner]
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.prepare_gate_rows()
                    partner.prepare_gate_rows()
                    self.estimate(self._wide_x, self._wide_y)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            restore_rng()
            graph = torch.cuda.CUDAGraph()
            for generator in generators:
                graph.register_generator_state(generator)
            torch._C._cuda_clearCublasWorkspaces()
            self.prepare_step_event()
            with torch.cuda.graph(graph):
                self.wait_step_in_graph()
                self._wide_gradients = self.estimate(self._wide_x, self._wide_y)
        finally:
            self._sources = [self]
        self._wide_graph, self._wide_packed, self._wide_loss = graph, self._packed, self.capture.clean_loss
        (self._packed, self.capture.inputs, self.capture.outputs, self.capture.blocks, self.capture.clean_loss,
         self.capture.y, self.capture.batch, self.capture.length) = kept
        restore_rng()
        before = [g.get_offset() for g in generators]
        self.prepare_gate_rows()
        partner.prepare_gate_rows()
        graph.replay()
        torch.cuda.synchronize()
        steps = [g.get_offset() - b for g, b in zip(generators, before)]
        half = len(steps) // 2
        if steps[:half] != steps[half:]:
            raise ValueError(f'the pair graph draws unevenly per row: {steps}')
        self._row_offsets = steps[:half]
        restore_rng()
    def update(self, gradients, y, optimizer, loss_sum):
        if getattr(self, '_means_done', None) is not None:  # MEANS_SIDE: the averages read the weights this step writes
            torch.cuda.current_stream().wait_event(self._means_done)
            self._means_done = None
        if DELAYED_UPDATE:  # screen: apply the previous estimate (the val cost of overlapping update and estimate)
            current = {name: g.clone() for name, g in gradients.items()}
            gradients = getattr(self, '_previous_gradients', current)
            self._previous_gradients = current
        # LATE_LOSS: counted only for the diagnostic loss, on a side stream (late_loss)
        valid = None if LATE_LOSS and self.packs_gradients() else (y != -1).sum()
        if valid is None:
            self._loss_from = torch.cuda.Event()
            self._loss_from.record()
        optimizer.zero_grad(set_to_none=True)
        params = list(self.model.named_parameters())
        packed = getattr(self, '_packed', None) if self.packs_gradients() else None
        if STACKED_ANVIL:  # ANVIL's shape groups contiguous in the flat buffer, so their stacked grads are views
            if not hasattr(self, '_flat_order'):
                self.bind_flat_layout(optimizer)
            params = self._flat_order
            if packed is not None:  # the estimate already formed the flat buffer (and its bf16 copy)
                flat = packed[0]
            else:
                flat = torch.cat([gradients[name].float().reshape(-1) for name, _ in params]).div_(
                    valid * self.world if LEAN_FLAT else valid)
        else:
            flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if SHARDED_ANVIL and STACKED_ANVIL and LEAN_FLAT:  # ANVIL's grads reduce-scattered, the rest all-reduced
            anvil = optimizer.optimizers[0]
            sizes = [sum(q.numel() for q in batch[0]) for batch in anvil.batches]
            wire = packed[1] if packed is not None else (  # COMMS_BF16: half the bytes on the wire
                flat.to(torch.bfloat16) if COMMS_BF16 else flat)
            # ASYNC_COMMS: ANVIL's shards first, each batch's update waits only for its own shard, and the rest's
            # all-reduce runs under ANVIL's update
            anvil.shard_grads, anvil.shard_works = [None] * len(sizes), [None] * len(sizes)
            offsets = [sum(sizes[:k]) for k in range(len(sizes))]
            order = [] if RANK_MAJOR else list(range(len(sizes)))
            if RANK_MAJOR:  # a reduce-scatter per comm group: the flat layout is rank-major too (bind_flat_layout)
                anvil.receive_rank_major(wire.dtype)
                starts = [sum(len(anvil.rank_stores[h]) for h in range(g)) for g in range(len(anvil.rank_groups))]
                works = {}
                for g in reversed(range(len(anvil.rank_groups))):  # the last unit (the merged polar) first
                    works[g] = torch.distributed.reduce_scatter_tensor(
                        anvil.rank_receives[g], wire[starts[g]:starts[g] + len(anvil.rank_stores[g])],
                        async_op=ASYNC_COMMS)
                anvil.shard_works = [works[anvil.rank_group_of[k]] for k in range(len(sizes))]
                anvil.shard_grads = list(anvil.shard_buffers)
            elif ANVIL_OVERLAP and getattr(anvil, '_unit_graphs', None):  # the last unit's shards first (see graphed_step)
                order = [k for ks in reversed(anvil.shard_units()) for k in ks]
            for k in order:
                (batch_params, *_), size, offset = anvil.batches[k], sizes[k], offsets[k]
                full = wire[offset:offset + size].view(len(batch_params), *batch_params[0].shape)
                if getattr(anvil, 'shard_buffers', None) is not None:  # ANVIL_GRAPHS: the graphs' static inputs
                    shard = anvil.shard_buffers[k]
                else:
                    shard = torch.empty((len(batch_params) // self.world, *batch_params[0].shape),
                                        device=flat.device, dtype=wire.dtype)
                anvil.shard_works[k] = torch.distributed.reduce_scatter_tensor(shard, full, async_op=ASYNC_COMMS)
                anvil.shard_grads[k] = shard
            rest = torch.distributed.all_reduce(wire[sum(sizes):], async_op=ASYNC_COMMS)
            if ASYNC_COMMS:
                anvil_ids = {id(q) for batch in anvil.batches for q in batch[0]}

                def finish_rest():  # between ANVIL's and the other optimizers' steps
                    rest.wait()
                    if COMMS_BF16 and LEAN_REST:
                        # bf16 params' grads are views of the reduced wire (what .to(bf16) of its fp32 copy gives),
                        # fp32 params' slices copied once; the finite check reads the bf16 wire
                        if FINITE_SIDE:  # the check's sum under AdamW on a side stream (joined at the update's end)
                            main = torch.cuda.current_stream()
                            check = self._finite_stream = getattr(self, '_finite_stream', None) or torch.cuda.Stream()
                            check.wait_stream(main)
                            with torch.cuda.stream(check):
                                health = torch.isfinite(torch.sum(wire, dtype=torch.float32))
                            health.record_stream(main)
                            self._telemetry_health.append(health)
                        else:
                            self._telemetry_health.append(torch.isfinite(torch.sum(wire, dtype=torch.float32)))
                        offset = 0
                        for _, p in params:
                            if id(p) not in anvil_ids:
                                part = slice(offset, offset + p.numel())
                                if p.dtype == wire.dtype:
                                    p.grad = wire[part].view_as(p)
                                else:
                                    flat[part] = wire[part]
                                    p.grad = flat[part].view_as(p).to(p.dtype)
                            offset += p.numel()
                        return
                    if COMMS_BF16:
                        flat[sum(sizes):] = wire[sum(sizes):]
                    self._telemetry_health.append(torch.isfinite(flat.sum()))
                    offset = 0
                    for _, p in params:
                        if id(p) not in anvil_ids:
                            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
                        offset += p.numel()
                if ADAM_SIDE:
                    optimizer.step_side(finish_rest)
                else:
                    optimizer.step(between=finish_rest)
                if FINITE_SIDE and getattr(self, '_finite_stream', None) is not None:
                    torch.cuda.current_stream().wait_stream(self._finite_stream)  # before the wire is rewritten
                for work in anvil.gather_works:
                    work.wait()
                anvil.gather_works = []
                if valid is None:
                    return self.late_loss(loss_sum, y)
                diagnostic_loss = loss_sum() / valid
                return diagnostic_loss if LAZY_LOSS else average_ranks(diagnostic_loss.reshape(1)).reshape(())
            if COMMS_BF16:
                flat[sum(sizes):] = wire[sum(sizes):]
        elif ALLREDUCE_BF16:
            half = flat.to(torch.bfloat16)
            torch.distributed.all_reduce(half)
            flat = half.float()
        else:
            torch.distributed.all_reduce(flat)
        if LEAN_FLAT and STACKED_ANVIL:  # pre-divided by the world size; one reduction for the finite check
            finite = torch.isfinite(flat.sum())
        else:
            flat /= self.world
            finite = torch.isfinite(flat).all()
        self._telemetry_health.append(finite)
        offset = 0
        for _, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        diagnostic_loss = loss_sum() / valid
        return diagnostic_loss if LAZY_LOSS else average_ranks(diagnostic_loss.reshape(1)).reshape(())
    def late_loss(self, loss_sum, y):
        """LATE_LOSS: the diagnostic loss from the replay's outputs on a side stream (its all-reduce queued after the
        update's collectives); the main stream waits only for the reads of the replay's loss buffer."""
        side = self._loss_stream = getattr(self, '_loss_stream', None) or torch.cuda.Stream()
        side.wait_event(self._loss_from)
        with torch.cuda.stream(side):
            diagnostic_loss = loss_sum() / (y != -1).sum()
            read = torch.cuda.Event()
            read.record()
            if not LAZY_LOSS:
                diagnostic_loss = average_ranks(diagnostic_loss.reshape(1)).reshape(())
        torch.cuda.current_stream().wait_event(read)  # (the next replay rewrites the loss buffer)
        return diagnostic_loss
    def join_losses(self, losses):
        """LATE_LOSS: the main stream waits for the side-stream losses before reading them."""
        side = getattr(self, '_loss_stream', None)
        if side is not None:
            main = torch.cuda.current_stream()
            main.wait_stream(side)
            for loss in losses:
                loss.record_stream(main)
    def fused_descending_key_values(self, chain, targets, x0, outs):
        """descending_key_values (batched keys on the side stream, batched values and gates) with the credits in the
        layout their consumers read, and the chain's glue and path_scalar_credit in one compiled graph. Returns its
        outputs and the path scalars."""
        d, s, model = self.capture, self.s, self.model
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        errors, raw_values, terminal_values = {}, {}, {}
        contexts, work = self.batched_key_contexts(targets, defer=True)
        # every layer's query, key and value credits by token, side by side: one batched reassembly GEMM
        rows = d.batch  # (several rows: (layer, row) pairs in place of layers, row-major within a layer)
        qkv = torch.empty(L, rows * t, 3 * self.width, device=self.device)
        maps = [self.layer_map([i * rows + r for i in item[1] for r in range(rows)]) for item in work]
        main, side = torch.cuda.current_stream(), self._side_stream
        side.wait_stream(main)
        qkv.record_stream(side)
        for item in work:
            for z in item[2:]:
                z.record_stream(side)
        for inputs in self._moment_inputs.values():
            for z in inputs:
                z.record_stream(side)
        with torch.cuda.stream(side):
            keys = self.batched_key_scores_by_token(work, qkv, maps)
        raws, place, raw_groups = {}, {}, []
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            raw = moment_kernels.raw_value_credits_by_token(*self.value_logit_inputs(window, qp, kp), lse, hub, window, h)
            raw_groups.append(raw)
            for j, i in enumerate(layers):
                raws[i], place[i] = raw[j] if rows == 1 else raw[j * rows:(j + 1) * rows].flatten(0, 1), (g, j)
        ve_layers = [i for i in range(L) if has_ve(i, L)]
        if STACKED_GATES and rows > 1:  # each (layer, row) with its row's gate rows, stacked
            pairs = [(i, r) for i in ve_layers for r in range(rows)]
            responses = stacked_gate_input_responses(
                [raws[i][r * t:(r + 1) * t] for i, r in pairs], [d.outputs[f'value_embeds.{i}'][r] for i, r in pairs],
                [d.blocks[i][r:r + 1] for i, r in pairs], [model.transformer.h[i].attn.ve_gate.weight for i, r in pairs],
                [self._sources[r]._gate_rows[i] for i, r in pairs], s.gate_path_sigma)
            gates = {i: torch.cat(responses[k * rows:(k + 1) * rows]) for k, i in enumerate(ve_layers)}
        elif STACKED_GATES:
            gates = dict(zip(ve_layers, stacked_gate_input_responses(
                [raws[i] for i in ve_layers], [d.outputs[f'value_embeds.{i}'][0] for i in ve_layers],
                [d.blocks[i] for i in ve_layers], [model.transformer.h[i].attn.ve_gate.weight for i in ve_layers],
                [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        else:
            kappas = [(raws[i].view(t, h, hd) * d.outputs[f'value_embeds.{i}'].float().view(t, h, hd)).sum(-1)
                      for i in ve_layers]
            gates = dict(zip(ve_layers, gate_input_responses(
                [d.blocks[i].float().flatten(0, 1) for i in ve_layers],
                [model.transformer.h[i].attn.ve_gate.weight.float() for i in ve_layers], kappas,
                [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        c_vs = [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)]
        if DESCENT_BMM:  # each window group's value responses raw @ c_v by one batched GEMM (raw: its layers stacked)
            moved = [None] * L
            for (window, layers, *_), raw in zip(work, raw_groups):
                if DESCENT_BF16:
                    v = torch.bmm(raw.view(len(layers), rows * t, -1).to(torch.bfloat16),
                                  torch.stack([c_vs[i] for i in layers]).to(torch.bfloat16), out_dtype=torch.float32)
                else:
                    v = torch.bmm(raw.view(len(layers), rows * t, -1), torch.stack([c_vs[i] for i in layers]))
                for j, i in enumerate(layers):
                    moved[i] = v[j]
            c_vs = moved
        input_list, extra_list, path = (descending_inputs_graph if DESCENT_GRAPH else descending_inputs_eager)(
            [raws[i] for i in range(L)], [gates.get(i) for i in range(L)], [d.blocks[i] for i in range(L)],
            c_vs, model.resid_lambdas.float(), x0, outs, moved=DESCENT_BMM)
        corrected = [torch.empty(len(item[1]) * rows, t, self.width, device=self.device) for item in work]
        if DESCENT_BMM:  # each group's corrected credits by one batched GEMM into the terminal pass's input
            for g, (window, layers, *_) in enumerate(work):
                if DESCENT_BF16:
                    corrected[g].view(len(layers), rows * t, -1).copy_(torch.bmm(
                        stacked_sums([chain[i] for i in layers], [extra_list[i] for i in layers]).to(torch.bfloat16),
                        torch.stack([model.transformer.h[i].attn.c_proj.weight for i in layers]).to(torch.bfloat16),
                        out_dtype=torch.float32))
                    continue
                torch.bmm(stacked_sums([chain[i] for i in layers], [extra_list[i] for i in layers]),
                          torch.stack([model.transformer.h[i].attn.c_proj.weight.float() for i in layers]),
                          out=corrected[g].view(len(layers), rows * t, -1))
        for i in (() if DESCENT_BMM else range(L)):  # corrected by token, straight into the terminal pass's input
            g, j = place[i]
            torch.mm(chain[i] + extra_list[i], model.transformer.h[i].attn.c_proj.weight.float(),
                     out=corrected[g][j] if rows == 1 else corrected[g][j * rows:(j + 1) * rows].view(rows * t, -1))
        w = self.width
        by_row = qkv.view(L * rows, t, 3 * w)
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            moment_kernels.terminal_value_credits_into(*self.value_logit_inputs(window, qp, kp), lse, corrected[g], window,
                                                       by_row[:, :, 2 * w:], maps[g])
            for i in layers:
                terminal_values[i] = qkv[i, :, 2 * w:].view(rows, t, h, hd)
        self._qkv = qkv
        for i in range(L):
            raw_values[i] = raws[i].view(rows, t, h, hd)
        main.wait_stream(side)
        for i in range(L):
            keys[i].record_stream(main)
            errors[f'transformer.h.{i}.attn.c_k'] = keys[i].view(rows, t, w)
            self._query_moment_errors[f'transformer.h.{i}.attn.c_q'].record_stream(main)
        return (errors, raw_values, terminal_values, dict(enumerate(extra_list)), dict(enumerate(input_list)),
                path)
    def layer_map(self, layers):
        """The layers' indices as an int32 device tensor (made once, outside graph capture)."""
        if not hasattr(self, '_layer_maps'):
            self._layer_maps = {}
        key = tuple(layers)
        if key not in self._layer_maps:
            self._layer_maps[key] = torch.tensor(key, device=self.device, dtype=torch.int32)
        return self._layer_maps[key]
    def fuses_glue(self):
        s = self.s
        return (FUSED_GLUE and s.batched_keys and s.query_moments and OVERLAP_KEYS and BATCHED_VALUES and BATCHED_GATES
                and FAST_KEYS and FAST_MOMENTS)
    def descending_key_values(self, chain, targets):
        """Topologically schedule independent raw probes; corrected outputs have terminal consumers only."""
        d,s,model=self.capture,self.s,self.model
        L=model.config.n_layer
        errors,raw_values,terminal_values,inputs={},{},{},{}
        extra={L-1:torch.zeros_like(chain[0])}
        if s.batched_keys and OVERLAP_KEYS:  # key credits and query moments on a side stream, under the value chain
            contexts, work = self.batched_key_contexts(targets, defer=True)
            main, side = torch.cuda.current_stream(), self._side_stream
            side.wait_stream(main)
            for item in work:
                for z in item[2:]:
                    z.record_stream(side)
            with torch.cuda.stream(side):
                keys = self.batched_key_scores(work)
        elif s.batched_keys:
            contexts, keys = self.batched_key_contexts(targets)
        deferred_keys = s.batched_keys and OVERLAP_KEYS
        batched_values = deferred_keys and BATCHED_VALUES
        if batched_values:  # raw value credits need no chain: one pass per window group up front
            raw_pre, corrected_all = {}, {}
            for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
                raw_group = moment_kernels.raw_value_credits(qp, kp, lse, hub, window)
                for j, i in enumerate(layers):
                    raw_pre[i] = raw_group[j * model.config.n_head:(j + 1) * model.config.n_head]
            if BATCHED_GATES:  # the value gates' input credits read only raw values: all layers at once
                ve_layers = [i for i in range(L) if has_ve(i, L)]
                heads = model.config.n_head
                kappas = []
                for i in ve_layers:
                    value = raw_pre[i].transpose(0, 1).reshape(-1, heads, self.width // heads)
                    kappas.append((value * d.outputs[f'value_embeds.{i}'].float().reshape_as(value)).sum(-1))
                gate_pre = dict(zip(ve_layers, gate_input_responses(
                    [d.blocks[i].float().flatten(0, 1) for i in ve_layers],
                    [model.transformer.h[i].attn.ve_gate.weight.float() for i in ve_layers], kappas,
                    [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        if batched_values and BATCHED_GATES and COMPILED_DESCENT:  # the whole chain as one compiled graph
            corrected_list, input_list, extra_list = descending_chain(
                [chain[i] for i in range(L)], [raw_pre[i] for i in range(L)],
                [gate_pre.get(i) for i in range(L)], [d.blocks[i].float().flatten(0, 1) for i in range(L)],
                [model.transformer.h[i].attn.c_proj.weight.float() for i in range(L)],
                [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)],
                model.resid_lambdas.float(), d.length)
            for i in range(L):
                corrected_all[i] = corrected_list[i]
                inputs[i] = input_list[i]
                extra[i] = extra_list[i]
                raw_values[i] = raw_pre[i].transpose(0, 1).unsqueeze(0)
        for i in (reversed(range(L)) if not (batched_values and BATCHED_GATES and COMPILED_DESCENT) else ()):
            block=model.transformer.h[i]
            # extra[i] contains only deeper raw independently measured input responses.
            corrected=((chain[i]+extra[i])@block.attn.c_proj.weight.float()).view(1,d.length,8,64).transpose(1,2)[0]
            if batched_values:
                raw, terminal, key = raw_pre[i], None, None
                corrected_all[i] = corrected
            elif s.batched_keys:
                context = contexts[i]
                raw, terminal = value_credits(context, corrected, model.window_sizes[i][0])
                key = None if deferred_keys else keys[i]
            else:
                context=self.key_context(i,targets[i])
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = self.query_moment_credit(i, context)
                key,raw,terminal=key_and_value_credits(context,corrected,s.key_sigma,model.window_sizes[i][0])
            if key is not None:
                errors[f'transformer.h.{i}.attn.c_k']=key.transpose(0,1).reshape(1,d.length,self.width)
            raw_values[i]=raw.transpose(0,1).unsqueeze(0)
            if terminal is not None:
                terminal_values[i]=terminal.transpose(0,1).unsqueeze(0)
            ev=raw_values[i].reshape(-1,self.width)
            xin=d.blocks[i].float().flatten(0,1)
            # Only original raw values enter either nonlinear finite-response branch.
            inputs[i]=norm_move(xin,ev@block.attn.c_v.weight.float())
            if has_ve(i,L):
                inputs[i] = inputs[i] + (gate_pre[i] if batched_values and BATCHED_GATES
                                         else self.gate_input_credit(i, raw_values[i], self._gate_rows[i]))
            if i: extra[i-1]=model.resid_lambdas[i].float()*(inputs[i]+extra[i])
        if batched_values:  # terminal value credits after the chain: one pass per window group
            h = model.config.n_head
            for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
                terminal_group = moment_kernels.terminal_value_credits(
                    qp, kp, lse, torch.cat([corrected_all[i] for i in layers], 0), window)
                for j, i in enumerate(layers):
                    terminal_values[i] = terminal_group[j * h:(j + 1) * h].transpose(0, 1).unsqueeze(0)
        if deferred_keys:
            main.wait_stream(side)
            for i in range(L):
                keys[i].record_stream(main)
                errors[f'transformer.h.{i}.attn.c_k']=keys[i].transpose(0,1).reshape(1,d.length,self.width)
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'].record_stream(main)
        return errors,raw_values,terminal_values,extra,inputs
    def exact_values(self, i, credit, value_credit):
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer) and FUSED_GLUE:
            errors[prefix + 've_gate'] = gate_error(d.outputs[prefix + 've_gate'], d.outputs[f'value_embeds.{i}'],
                                                    credit, h)
        elif has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, cfg.n_embd // h)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors
    def key_context(self, i, target):
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        if self.s.compiled_glue:
            prefix = f'transformer.h.{i}.attn.'
            ve = has_ve(i, model.config.n_layer)
            qp, kp, k_raw, values, hub = key_context_tensors(
                d.outputs[prefix + 'c_q'][0], d.outputs[prefix + 'c_k'][0], d.outputs[prefix + 'c_v'][0],
                d.outputs[f'value_embeds.{i}'][0] if ve else None, d.outputs[prefix + 've_gate'][0] if ve else None,
                target[0], cos, sin, h)
            lse, center = implicit_key_context(qp, kp, values, hub, model.window_sizes[i][0])
            return qp, kp, k_raw, cos, sin, lse, (values, hub, center)
        rot = lambda z: torch.cat([z[..., :hd//2]*cos + z[..., hd//2:]*sin,
        -z[..., :hd//2]*sin + z[..., hd//2:]*cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix+name][0].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][0].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix+'ve_gate'][0].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        hub = target[0].view(t, h, hd).transpose(0, 1)
        lse, center = implicit_key_context(qp, kp, values, hub, model.window_sizes[i][0])
        return qp, kp, k_raw, cos, sin, lse, (values, hub, center)
    def probe_moments(self, x, y):
        if self.fuses_glue() and self.capture.batch > 1:  # NATIVE_WIDE: each row with its own frame
            frames = [self.basis(r) for r in range(self.capture.batch)]
            derivative, self._head_moments = probe_moment_terms_rows(x, y, self.model.lm_head.weight.float(), frames)
            return derivative
        if self.fuses_glue():
            derivative, self._head_moments = probe_moment_terms(x, y, self.model.lm_head.weight.float(), self.basis())
            return derivative
        return self.probe_moments_eager(x, y)
    def probe_moments_eager(self, x, y):
        """probe() with each move's readout change from its second-order expansion in the moved logits: the
        softcapped log-sum-exp to second order, its quadratic term exact over the whole vocabulary for every frame
        direction (c2 @ (C * C)^T, C the head's rows in the frame), the moved norm and the target logit exact. Also
        keeps the head moments for the chain replays (self._head_moments)."""
        weight = self.model.lm_head.weight.float()
        H = self.basis()
        C = H @ weight.T  # (direction, vocab): each head row's component along each frame direction
        width = self.width
        steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # #44 sigma
        base = x @ weight.T
        sq = x.square().sum(-1, keepdim=True)
        inv0 = torch.rsqrt(sq / width + EPS)
        t = torch.tanh(base * inv0 / 15)
        f1 = 1 - t.square()
        prob = torch.softmax(15 * t, -1)
        c1, c2 = prob * f1, prob * (-(2 / 15) * t * f1 + f1.square()) / 2
        b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
        mu1, nu = c1 @ weight, (c2 * base) @ weight
        self._head_moments = (b1, b2, mu1, nu, readout_curvature_diagonal(weight, c2))
        inv = torch.rsqrt((sq + 2 * steps * (x @ H.T) + steps.square() * width) / width + EPS)
        a = inv - inv0
        lin = a * b1.unsqueeze(1) + inv * steps * (mu1 @ H.T)
        quad = c2 @ C.square().T
        change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * steps * (nu @ H.T)
                  + inv.square() * steps.square() * quad - 0.5 * lin.square())
        target = y.clamp(min=0)
        base_y = base.gather(1, target.unsqueeze(1))
        zy = (base_y + steps * C.T[target]) * inv
        derivative = (change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(base_y * inv0 / 15))) / steps
        return (derivative * (y >= 0).unsqueeze(1)) @ H / width
    def probe(self, x, y):
        """Readout credit: PR #44's Cut probe, forward differences along a full Hadamard cycle."""
        weight = self.model.lm_head.weight.float()
        H = self.basis()
        stream_dirs = H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # #44 sigma
        n, vocab = x.shape[0], C.shape[1]
        credit = torch.empty(n, self.width, device=self.device)
        per = max(1, 134217728 // (self.width * vocab))  # #44 chunk
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ H / self.width
        return credit
    def close(self):
        if self._estimate_graph is not None:
            torch.cuda.synchronize()
            self._estimate_graph = None
            self._static_gradients = None
            self._static_x = self._static_y = None
            self._gate_rows.clear()
        self.capture.close()
    def prepare_gate_rows(self, state=None, into=None):
        """The gate rows the next estimate reads (from self.generator's state, or `state`), into self._gate_rows (or
        the same-keyed buffers `into`)."""
        generator = torch.Generator(self.device).set_state(self.generator.get_state() if state is None else state)
        self.draw_gate_rows(generator, self._gate_rows if into is None else into)
    def draw_gate_rows(self, generator, into):
        """prepare_gate_rows' draws from `generator` (at the state the next estimate starts from) into `into`."""
        # Skip the frames the next estimate draws from self.generator: the readout probe's, then one per chain root.
        for _ in range(1 + len(self.s.chain_depth_roots)):
            torch.randint(0, 2, (self.width,), device=self.device, generator=generator)
        frame = hadamard(self.width, self.device, torch.float32)
        for key, rows in into.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            rows.copy_(frame[indices] * signs)
    def prefetch_gate_rows(self, state=None):
        """PREFETCH_GATE_ROWS: the next update's gate rows, from self.generator's state after this update's estimate
        (or `state`), formed on a side stream under the estimate (into staging buffers; take_gate_rows copies them in)."""
        if not hasattr(self, '_staged_rows'):
            self._staged_rows = {key: torch.empty_like(rows) for key, rows in self._gate_rows.items()}
            self._prefetch_stream = torch.cuda.Stream(priority=PREFETCH_PRIORITY)
            self._staged_free = torch.cuda.Event()
            self._staged_free.record()
        state = self.generator.get_state() if state is None else state
        side = self._prefetch_stream
        side.wait_event(self._staged_free)  # the previous take's copies out of the staging buffers
        with torch.cuda.stream(side):
            if getattr(self, '_gate_graph', None) is not None:
                self._gate_generator.set_state(state)
                self._gate_graph.replay()
            else:
                self.prepare_gate_rows(state, self._staged_rows)
            ready = torch.cuda.Event()
            ready.record()
        self._prefetched = (state, ready)
    def capture_gate_graph(self):
        """GATE_GRAPH (before the timer): prefetch_gate_rows' draws into the staging buffers as one CUDA graph, on a
        generator registered with it (set to the prefetch's state before each replay)."""
        if not hasattr(self, '_staged_rows'):
            self.prefetch_gate_rows()
            torch.cuda.synchronize()
            self._prefetched = None
        self._gate_generator = torch.Generator(self.device)
        self._gate_generator.set_state(self.generator.get_state())
        side = self._prefetch_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.draw_gate_rows(self._gate_generator, self._staged_rows)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self._gate_generator)
        with torch.cuda.graph(graph, stream=side):
            self.draw_gate_rows(self._gate_generator, self._staged_rows)
        torch.cuda.synchronize()
        self._gate_graph = graph
        self._prefetched = None  # (the staging buffers now hold warm-up draws: the next take draws eagerly)
    def take_gate_rows(self):
        """prepare_gate_rows, from the prefetched rows when they were drawn from the generator's current state."""
        prefetched, self._prefetched = getattr(self, '_prefetched', None), None
        if prefetched is None or not torch.equal(prefetched[0], self.generator.get_state()):
            self.prepare_gate_rows()
            return
        torch.cuda.current_stream().wait_event(prefetched[1])
        for key, rows in self._gate_rows.items():
            rows.copy_(self._staged_rows[key])
        self._staged_free.record()
    @torch.no_grad()
    def graph_generators(self):
        generators = [self.generator, self.move_generator, self.capture.shared_generator, self.capture.generator]
        return generators + [self.sign_generator] if CUT_SIGN_GENERATOR else generators
    def capture_estimate_graph(self, x, y):
        self._static_x, self._static_y = x.clone(), y.clone()
        generators = self.graph_generators()
        states = [g.get_state() for g in generators]
        default_state = torch.cuda.get_rng_state()
        def restore_rng():
            for generator, state in zip(generators, states):
                generator.set_state(state)
            torch.cuda.set_rng_state(default_state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.prepare_gate_rows()
                self.estimate(self._static_x, self._static_y)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        for generator in generators:
            graph.register_generator_state(generator)
        torch._C._cuda_clearCublasWorkspaces()
        self.prepare_step_event()
        with torch.cuda.graph(graph):
            self.wait_step_in_graph()
            self._static_gradients = self.estimate(self._static_x, self._static_y)
        self._estimate_graph = graph
        restore_rng()
        self.prepare_gate_rows()
        graph.replay()
        torch.cuda.synchronize()
        restore_rng()
    def gate_input_credit(self, i, raw_value, rows):
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        value = raw_value.reshape(-1, heads, self.width // heads)
        ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
        kappa = (value * ve).sum(-1)
        xin = d.blocks[i].float().flatten(0, 1)
        weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
        projected = F.linear(rows[:, :weight.shape[1]], weight)
        row_square = rows.square().mean(-1)
        result = torch.empty_like(xin)
        for t0 in range(0, len(xin), s.chain_tokens):
            sl = slice(t0, t0 + s.chain_tokens)
            result[sl] = prepared_gate_input_response(
                xin[sl], weight, kappa[sl], rows, projected, row_square, s.gate_path_sigma)
        return result
    @staticmethod
    def path_scalar_credit(inputs, extra, x0, outs):
        credits = []
        for i in range(len(outs)):
            credit = inputs[i]+extra[i]
            previous = x0 if i == 0 else outs[i-1]
            credits.append(torch.stack([(credit*previous).sum(), (credit*x0).sum()]))
        return torch.stack(credits, dim=1)
    def chain_credits(self, e, x0, mids, outs, routes, y, g_top):
        model, s = self.model, self.s
        L = model.config.n_layer
        weight = model.lm_head.weight.float()
        if CAST_ONCE and CHAIN_GLUE:  # the bf16 weights cast_weights made; c_proj's transposed once more
            names = [f'transformer.h.{i}.mlp.' for i in range(L)]
            w_fc = [self._bf16_weights[p + 'c_fc'] for p in names]
            w_out_t = chain_transposes([self._bf16_weights[p + 'c_proj'] for p in names])
        elif CHAIN_GLUE:
            w_fc, w_out_t = chain_weights([blk.mlp.c_fc.weight for blk in model.transformer.h],
                                          [blk.mlp.c_proj.weight for blk in model.transformer.h])
        else:
            w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        moments = self._head_moments if s.probe_moments else head_moments(xL, base, weight)
        n, m = e.shape[0], s.chain_tokens
        rows = self.capture.batch  # (NATIVE_WIDE: each row with its own frames and signs, from its own generators)
        t = n // rows
        if rows > 1 and (t % m or s.chain_fraction < 1 or not CUT_SIGN_GENERATOR):
            raise ValueError('a several-row estimate needs whole-row chunks, every token and CUT_SIGN_GENERATOR')
        by_site = {}
        for site in s.chain_depth_roots:
            frames = []
            for r in range(rows):
                H = self.basis(r)
                if s.chain_moves < self.width:
                    picks = torch.randperm(self.width, device=self.device, generator=self._sources[r].move_generator)
                    H = H[picks[:s.chain_moves]].contiguous()
                frames.append(H)
            steps = s.chain_sigma * mids[site].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            by_site[site] = (frames, steps, torch.empty(n, self.width, device=self.device))
        signed_steps = {}
        controls = {}
        groups = self.measured_groups()
        for group in s.joint_root_groups:  # (every group's signs: a skipping update draws as a measuring one)
            for site in group:
                steps = by_site[site][1]
                signs = [torch.randint(0, 2, (t, 1), device=self.device,
                                       generator=self._sources[r].sign_generator if CUT_SIGN_GENERATOR else None)
                         for r in range(rows)]
                sign = ((signs[0] if rows == 1 else torch.cat(signs)) * 2 - 1).float()
                signed_steps[site] = steps * sign
                factor = lam_r[site + 1:].prod()
                controls[site] = g_top * factor
        roots = [group[0] for group in groups]
        if CHAIN_GLUE:  # per root, one projection per row's frame
            projected = root_projections_of([w_fc[r] for r in roots for _ in range(rows)],
                                            [frame for r in roots for frame in by_site[r][0]])
            root_projections = {r: projected[k * rows:(k + 1) * rows] for k, r in enumerate(roots)}
        else:
            root_projections = {r: [(w_fc[r].float() @ frame.T).contiguous() for frame in by_site[r][0]]
                                for r in roots}
        replay_layers = lambda group: (attention_replay_layers(group, L, s.query_replay_long_layers)
                                       if s.replay_groups is None or group[0] in s.replay_groups else ())
        active_query_layers = tuple(sorted({b for group in groups for b in replay_layers(group)}))
        owns = [self.strong_keys(self.own_query_inputs(active_query_layers, r), s.query_keys) for r in range(rows)]
        boost = 1.0
        chunks = [slice(t0, t0 + m) for t0 in range(0, n, m)]
        if s.chain_fraction < 1:  # replay a random subset of the tokens; the rest keep their identity-path control
            count = int(n * s.chain_fraction)
            chunks = [torch.randperm(n, device=self.device, generator=self.move_generator)[:count].sort().values]
            boost = n / count
            for site, (_, _, result) in by_site.items():
                result.copy_(controls[site])
        main = torch.cuda.current_stream()
        lanes = [None] + [self.pair_lane(r) for r in range(1, rows)] if PAIR_CHAIN_STREAMS and rows > 1 else [None] * rows
        for lane in lanes[1:]:  # PAIR_CHAIN_STREAMS: each further row's replays on its own stream, joined below
            lane.wait_stream(main)
        for sl in chunks:
            row = 0 if rows == 1 else sl.start // t  # (whole-row chunks: the row's own queries, keys and frames)
            with contextlib.nullcontext() if lanes[row] is None else torch.cuda.stream(lanes[row]):
                own, local = owns[row], sl if rows == 1 else slice(sl.start - row * t, sl.stop - row * t)
                own_sl = (own[0][local], [None if z is None else z[local] for z in own[1]], own[2]) + tuple(
                    [None if z is None else token_slice(z, local) for z in part] for part in own[3:6]) + own[6:]
                for group in groups:
                    site = group[0]
                    H = by_site[site][0][row]
                    steps = signed_steps[site][sl]
                    injections = tuple((other, signed_steps[other][sl], by_site[other][0][row])
              for other in group[1:])
                    loss_change = chain_chunk(
                        site, (steps, H), [z[sl] for z in mids], xL[sl], [None if r[0] is None else r[0][sl] for r in routes],
                        [None if r[1] is None else r[1][sl] for r in routes], lam_r, w_fc, w_out_t, base[sl], target_w[sl], y[sl], own_sl,
                        moments=tuple(z[sl] for z in moments),
                        root_projection=root_projections[site][row], injections=injections,
                        query_layers=replay_layers(group),
                    )
                    if LEAN_GROUP_CREDITS:
                        news = group_credits(loss_change, [signed_steps[o][sl] for o in group],
                                             [controls[o][sl] for o in group], [by_site[o][0][row] for o in group],
                                             boost * self.root_boost(group))
                        for other, new in zip(group, news):
                            by_site[other][2][sl] = new
                        continue
                    residual = loss_change
                    for other in group:
                        frame = by_site[other][0][row]
                        residual = residual - signed_steps[other][sl] * (controls[other][sl] @ frame.T)
                    for other in group:
                        frame, result = by_site[other][0][row], by_site[other][2]
                        result[sl] = controls[other][sl] + boost * ((residual / signed_steps[other][sl]) @ frame
                                                                    / frame.shape[0])
        for lane in lanes[1:]:
            main.wait_stream(lane)
        del own_sl, own, owns
        measured_sites = {site for group in groups for site in group}
        credits = {site: credit for site, (_, _, credit) in by_site.items() if site in measured_sites}
        if LEAN_CHAIN_TAIL:
            def tail(sites):
                measured = [credits.get(i) if i in sites else None for i in range(L)]
                mids_c, lams, x0_credit = chain_tail(measured, g_top, lam_r, lam_0, x0, outs)
                out = dict(enumerate(mids_c))
                for i in range(L):
                    out['lam', i] = lams[i]
                out['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
                                               hadamard(self.width, self.device, torch.float32))
                return out
            if self._amortize == 'measure' and not ROOT_BOOST:  # AMORTIZED_ROOTS: also the credits without theirs
                return tail(measured_sites), tail(measured_sites - set(AMORTIZED_ROOTS))
            return tail(measured_sites)
        if AMORTIZED_ROOTS:
            raise ValueError('AMORTIZED_ROOTS needs LEAN_CHAIN_TAIL')
        if L - 1 not in credits:  # last block unmeasured: its mid credit is the readout credit
            credits[L - 1] = g_top
        for i in reversed(range(L - 1)):
            if i not in credits:
                credits[i] = lam_r[i + 1] * credits[i + 1]
        for i in range(L):
            previous = x0 if i == 0 else outs[i - 1]
            credits['lam', i] = torch.stack([(credits[i] * previous).sum(),
          (credits[i] * x0).sum()])
        # x0 = norm(e) enters every block input (x_in_i = lam_r[i] * prev_i + lam_0[i] * x0), so each block's mid
        # credit (measured at a root, else carried from the next block) reaches x0 along its linear skip; one finite
        # step through the norm follows.
        x0_credit = lam_r[0] * credits[0] + sum(lam_0[i] * credits[i] for i in range(L))
        credits['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
      hadamard(self.width, self.device, torch.float32))
        return credits
    def pair_lane(self, row):
        """PAIR_CHAIN_STREAMS: the stream of row's chain replays (made once, outside graph capture's first use)."""
        if not hasattr(self, '_pair_lanes'):
            self._pair_lanes = {}
        if row not in self._pair_lanes:
            self._pair_lanes[row] = torch.cuda.Stream()
        return self._pair_lanes[row]
    def own_query_inputs(self, active_layers, row=0):
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            if i not in active_layers:
                inputs.append(None), w_q.append(None), keys.append(None), values.append(None)
                w_proj_t.append(None), windows.append(model.window_sizes[i][0])
                continue
            prefix = f'transformer.h.{i}.attn.'
            if FUSED_GLUE:
                gated = block.attn.ve_gate is not None
                k, v = own_query_tensors(d.outputs[prefix + 'c_k'][row], d.outputs[prefix + 'c_v'][row],
                                         d.outputs[prefix + 've_gate'][row] if gated else None,
                                         d.outputs[f'value_embeds.{i}'][row] if gated else None, cos, sin, heads)
                inputs.append(d.blocks[i][row].float())
                w_q.append(block.attn.c_q.weight.float())
                keys.append(k)
                values.append(v)
                w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
                windows.append(model.window_sizes[i][0])
                continue
            k = d.outputs[prefix + 'c_k'][row].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            c, s = cos[:, None], sin[:, None]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][row].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][row].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][row].float().view(length, heads, width)
            inputs.append(d.blocks[i][row].float())
            w_q.append(block.attn.c_q.weight.float())
            keys.append(k.transpose(0, 1).to(torch.bfloat16).contiguous())
            values.append(v.transpose(0, 1).to(torch.bfloat16).contiguous())
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            windows.append(model.window_sizes[i][0])
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)
    @staticmethod
    def strong_keys(own, k):
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        length = len(pos)
        top_keys, top_values, top_p = [], [], []
        for x, wq, kk, vv, window, wp in zip(inputs, w_q, keys, values, windows, w_proj_t):
            if x is None:
                top_keys.append(None), top_values.append(None), top_p.append(None)
                continue
            heads, _, width = kk.shape
            q = strong_queries(x, wq, cos, sin, heads) if FUSED_GLUE else None
            if q is None:
                c, s = cos[:, None], sin[:, None]
                r = (norm(x) @ wq.T).view(length, heads, width)
                q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
              r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            p, idx, out, all_keys = stream_route_moments(q, kk, vv, window, k)
            if (GATHERED_VALUES and CHAIN3_WIDE and CHAIN3_TC and CHAIN3_SHARED_RESP and LEAN_STRONG_KEYS and CHAIN_GLUE
                    and FUSED_GLUE and TOKEN_MAJOR_VALUES):
                if FUSED_KEY_TABLES:  # the shifted top keys from the tables' launch
                    shifted, tables = project_values.gathered_key_tables(vv, out, idx, wp, kk, p, all_keys)
                    top_keys.append(shifted), top_values.append(tables)
                else:
                    top_keys.append(strong_key_shifted(kk, idx, p, all_keys))
                    top_values.append(project_values.gathered_tables(vv, out, idx, wp))
                top_p.append(p.transpose(0, 1).contiguous())
                continue
            if LEAN_STRONG_KEYS:
                ktop, vtop, rest = strong_key_tensors(kk, vv, idx, p, out, all_keys)
            else:
                gather = lambda z: z.unsqueeze(1).expand(-1, length, -1, -1).gather(
              2, idx.unsqueeze(-1).expand(-1, -1, -1, width))
                ktop, vtop = gather(kk), gather(vv)
                rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
                mass = (1 - p.sum(-1, keepdim=True)).clamp_min(0)
                tail_keys = all_keys - torch.einsum('htk,htkd->htd', p, ktop.float())
                centroid = torch.where(mass > 1e-5, tail_keys / mass.clamp_min(1e-5), 0)
                ktop = (ktop.float() - centroid.unsqueeze(-2)).to(kk.dtype)
            top_keys.append(ktop if LEAN_STRONG_KEYS and CHAIN_GLUE else ktop.transpose(0, 1).contiguous())
            if TOKEN_MAJOR_VALUES:
                top_values.append(project_routed_values_token_major(vtop, p, rest, wp))
            else:
                vtop = preproject_routed_values(vtop, p, rest, wp)
                top_values.append(vtop.transpose(0, 1).contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
        return (pos, inputs, w_q, top_keys, top_values, top_p, (cos, sin))
    def route_scores(self, layer, hpre):
        mlp = self.model.transformer.h[layer].mlp
        radius = self.s.chain_sigma * mlp.c_fc.weight.float().norm(dim=1)
        return response_scores(hpre, radius, mlp.c_proj.weight.float().norm(dim=0))
    def select_mlp_routes(self, layer, hpre):
        mlp = self.model.transformer.h[layer].mlp
        weight = mlp.c_proj.weight.float()
        radius = self.s.chain_sigma * mlp.c_fc.weight.float().norm(dim=1)
        plus = F.relu(hpre + radius).square()
        minus = F.relu(hpre - radius).square()
        response = (plus - minus).abs()
        return (response * weight.norm(dim=0)).topk(self.s.route_k, dim=-1).indices
    @torch.no_grad()
    def cast_weights(self):
        """CAST_ONCE: every Linear's bf16 weight in persistent buffers, by one multi-tensor copy (the capture forward
        and chain_credits read them instead of casting their own)."""
        if not hasattr(self, '_bf16_weights'):
            linears = {n: m.weight for n, m in self.capture.modules.items() if isinstance(m, nn.Linear)}
            assert all(m.bias is None for m in self.capture.modules.values() if isinstance(m, nn.Linear))
            self._bf16_weights = {}
            if QKV_GEMM:  # each block's c_q / c_k / c_v buffers: row blocks of one (3 C, C) buffer, 'attn.qkv'
                for i in range(self.model.config.n_layer):
                    p = f'transformer.h.{i}.attn.'
                    parts = [linears[p + n] for n in ('c_q', 'c_k', 'c_v')]
                    cat = torch.empty((sum(w.shape[0] for w in parts), parts[0].shape[1]), device=parts[0].device,
                                      dtype=torch.bfloat16)
                    self._bf16_weights[p + 'qkv'] = cat
                    for n, rows in zip(('c_q', 'c_k', 'c_v'), cat.split([w.shape[0] for w in parts])):
                        self._bf16_weights[p + n] = rows
            for n, w in linears.items():
                if n not in self._bf16_weights:
                    self._bf16_weights[n] = torch.empty_like(w, dtype=torch.bfloat16)
            self._bf16_targets = [self._bf16_weights[n] for n in linears]
            self._fp32_weights = list(linears.values())
        torch._foreach_copy_(self._bf16_targets, self._fp32_weights)
        self.capture.weights = self._bf16_weights
    def estimate(self, x, y):
        d, model = self.capture, self.model
        if CAST_ONCE:
            self.cast_weights()
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        first = min(self.s.chain_depth_roots)
        if self.fuses_glue() and self.s.batched_routes:
            mids, routes, outs, scores = block_recompute(
                [d.blocks[i] for i in range(L)], [d.outputs[f'transformer.h.{i}.attn.c_proj'] for i in range(L)],
                [blk.mlp.c_fc.weight for blk in model.transformer.h],
                [blk.mlp.c_proj.weight for blk in model.transformer.h], first, self.s.chain_sigma,
                [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)],
                [d.outputs[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)])
        for i in (range(L) if not (self.fuses_glue() and self.s.batched_routes) else ()):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            if self.s.batched_routes:
                mids.append(mid), outs.append(out), routes.append(hpre)
                continue
            top = self.select_mlp_routes(i, hpre)
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        if self.s.batched_routes:  # one top-k over every replayed block's scores
            hpres = routes
            if not self.fuses_glue():
                scores = torch.stack([self.route_scores(i, h) for i, h in enumerate(hpres) if i >= first])
            count = self.s.route_k
            if self.fuses_glue() and not self.s.route_schedule and self.s.route_k_late and first >= self.s.route_split:
                count = self.s.route_k_late  # every scored block keeps only its strongest route_k_late
            tops = route_topk(scores, count) if self.fuses_glue() else scores.topk(count, dim=-1).indices
            tops = [None] * first + list(tops)
            if self.s.route_schedule:  # per block: the strongest route_schedule[i] of them
                tops = [None if top is None else top[:, :k].contiguous() for top, k in zip(tops, self.s.route_schedule)]
            elif self.s.route_k_late:  # the strongest route_k_late of them from block route_split on
                tops = [top if top is None or i < self.s.route_split else top[:, :self.s.route_k_late].contiguous()
                        for i, top in enumerate(tops)]
            routes = [(None, None) if top is None else (top, h.gather(1, top).float()) for top, h in zip(tops, hpres)]
        g_top = (self.probe_moments if self.s.probe_moments else self.probe)(outs[-1], yf)
        chain = self.chain_credits(e, x0, mids, outs, routes, yf, g_top)
        if self._amortize == 'skip' and self.packs_gradients() and not ROOT_BOOST:  # plus the deviations' EMA
            gradients = self.chain_gradients(gradients, chain, g_top, x0, outs, y)
            self._packed = skipping_pack(self._packed[0], self._deviation, COMMS_BF16)
            return gradients
        if self._amortize != 'measure' or ROOT_BOOST:
            return self.chain_gradients(gradients, chain, g_top, x0, outs, y)
        # AMORTIZED_ROOTS, a measuring update: the gradients with the roots' measured credits (packed as usual) and
        # their packed difference from the gradients without them (the roots' deviation, Cut.amortize's input)
        chain, dropped = chain
        if not self.packs_gradients():  # (a throwaway Cut's first update: no flat layout bound yet)
            return self.chain_gradients(gradients, chain, g_top, x0, outs, y)
        base = dict(gradients)
        gradients = self.chain_gradients(dict(base), chain, g_top, x0, outs, y)
        full = self._packed[0]
        self.chain_gradients(base, dropped, g_top, x0, outs, y)
        self._packed = measuring_pack(full, self._packed[0], self._deviation, self._root_weight, ROOT_FRESH,
                                      COMMS_BF16)
        return gradients
    def chain_gradients(self, gradients, chain, g_top, x0, outs, y):
        """The estimate's weight gradients from the chain's block credits (into gradients; FLAT_GRADIENTS: packed)."""
        d, model = self.capture, self.model
        L = model.config.n_layer
        b, t = y.shape
        lam_r = model.resid_lambdas.float()
        if TARGETS_BF16:
            targets = {i: torch.mm(chain[i].to(torch.bfloat16), block.attn.c_proj.weight.to(torch.bfloat16),
                                   out_dtype=torch.float32).view(b, t, self.width)
                       for i, block in enumerate(model.transformer.h)}
        else:
            targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                       for i, block in enumerate(model.transformer.h)}
        if self.fuses_glue():
            key_errors, raw_values, terminal_values, extra, inputs, path = self.fused_descending_key_values(
                chain, targets, x0, outs)
        else:
            key_errors,raw_values,terminal_values,extra,inputs=self.descending_key_values(chain,targets)
            path = None
        scalars = torch.empty(2, L, device=self.device)
        if self.fuses_glue() and not LAYER_TAILS:  # every block's weight credits from three batched GEMMs
            blocks = model.transformer.h
            attn_proj, mlp_proj, fc = (weight_credits_bf16 if WEIGHT_CREDITS_BF16 else weight_credits)(
                [chain[i] for i in range(L)], g_top, lam_r, [extra[i] for i in range(L)],
                d.inputs['stack:attn.c_proj'], d.inputs.get('stack:mlp.c_proj'),
                [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], d.inputs['stack:mlp.c_fc'],
                [blk.mlp.c_proj.weight for blk in blocks], self.s.fc_sigma)
            for i in range(L):
                p = f'transformer.h.{i}.'
                gradients[p + 'attn.c_proj.weight'], gradients[p + 'mlp.c_proj.weight'] = attn_proj[i], mlp_proj[i]
                gradients[p + 'mlp.c_fc.weight'] = fc[i]
                scalars[:, i] = chain['lam', i]
        for i in (range(L) if self.fuses_glue() and LAYER_TAILS else ()):
            p = f'transformer.h.{i}.'
            last = i == L - 1
            lam = torch.ones((), device=self.device) if last else lam_r[i + 1]
            (gradients[p + 'attn.c_proj.weight'], gradients[p + 'mlp.c_proj.weight'],
             gradients[p + 'mlp.c_fc.weight']) = layer_weight_credits(
                chain[i], g_top if last else chain[i + 1], lam, extra[i], d.inputs[p + 'attn.c_proj'],
                d.inputs[p + 'mlp.c_proj'], d.outputs[p + 'mlp.c_fc'],
                d.inputs[p + 'mlp.c_fc'], model.transformer.h[i].mlp.c_proj.weight, self.s.fc_sigma)
            scalars[:, i] = chain['lam', i]
        for i in (range(L) if not self.fuses_glue() else ()):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            if i == L - 1:
                d_out = g_top
            else:
                d_out = lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = d_mid + extra[i], d_out + extra[i]
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            scalars[:, i] = chain['lam', i]
        scalars += self.path_scalar_credit(inputs, extra, x0, outs) if path is None else path
        self.internals(gradients, targets, b, t, key_errors, raw_values, terminal_values)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.packs_gradients():  # Cut.step's flat buffer (and bf16 copy) formed here, inside the estimate graph
            self._packed = pack_gradients([gradients[name] for name, _ in self._flat_order], (y != -1).sum() * self.world,
                                          COMMS_BF16 and (not AMORTIZED_ROOTS or ROOT_BOOST))
        return gradients


MAPS = (
  (3.923798038567, -6.095026865488, 3.905234618423),
  (3.278126713798, -3.328923386476, 0.989127286973),
  (3.505298394150, -5.137358782410, 1.968325560615),
  (2.815058591845, -3.685181239622, 1.417196497642),
  (2.245503932403, -2.443826979899, 0.963091710461),
  (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)


POLAR_FUSED = False  # the Newton-Schulz maps' scaled sums in the GEMMs' epilogues (baddbmm), not separate passes


def newton_schulz(x):
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        if POLAR_FUSED:
            x = torch.baddbmm(x, torch.baddbmm(s, s, s, beta=b, alpha=c), x, beta=a)
        else:
            x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    R, Q = x @ x.transpose(1, 2), eye
    for t, (a, b, c) in enumerate(MAPS):
        if t in restarts:
            x = Q @ x
            R, Q = x @ x.transpose(1, 2), eye
        if POLAR_FUSED:
            Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
            Q = Z + a * Q if Q is eye else torch.baddbmm(Q, Q, Z, beta=a)
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)
            continue
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return Q @ x


HEAD_POLAR = ('q', 'k')  # subset of ('q', 'k', 'v', 'proj'): the Cut ANVIL's polar for those attention matrices per head (each
# head's rows, for attn.c_proj its input columns, orthogonalized on their own, Noise PR #54's anvil_head_polar), in their
# own ANVIL batch
MLP_POLAR_GROUPS = 1  # > 1: the Cut ANVIL's polar for mlp.c_fc's rows and mlp.c_proj's columns in this many blocks of
# hidden units each (HEAD_POLAR's analogue), in their own ANVIL batches
LANE_ROWS = False  # ANVIL's lane energy per output row for wide matrices too (mlp.c_proj; else per input column): NorMuon
LANE_BETA = 0.9  # ANVIL's lane energy EMA decay (NorMuon's beta2: 0.95)
HEAD_POLAR_JOINT = True  # with HEAD_POLAR=('q', 'k'): each head's c_q and c_k rows in one polar (their batch interleaves
# q and k per block)
LANE_CROSS = False  # after ANVIL's lane gain, each lane of the other axis to equal RMS (instantaneous; two-sided NorMuon)


def head_polar(g, margin, heads: int):
    """[B, r, c] -> each matrix's `heads` row blocks orthogonalized on their own (each slice scaled by margin x its norm)."""
    B, r, c = g.shape
    if HEAD_POLAR_JOINT:  # (q, k) pairs: [B/2, 2, heads, r/heads, c] -> one [2r/heads, c] slice per pair and head
        x = g.reshape(B // 2, 2, heads, r // heads, c).transpose(1, 2).reshape(B // 2 * heads, 2 * r // heads, c)
    else:
        x = g.reshape(B * heads, r // heads, c)
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = gram_newton_schulz(x).float()
    if HEAD_POLAR_JOINT:
        return x.reshape(B // 2, heads, 2, r // heads, c).transpose(1, 2).reshape(B, r, c)
    return x.reshape(B, r, c)


def polar_batch(g, margin, heads: int = 1):
    """heads > 1: head_polar of the row blocks; heads < -1: of the -heads column blocks."""
    if heads > 1:
        return head_polar(g, margin, heads)
    if heads < -1:
        return head_polar(g.transpose(1, 2), margin, -heads).transpose(1, 2)
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


def cross_lanes(gained, lanes: int):
    """LANE_CROSS: each lane of the axis the lane energy does not average over rescaled to equal RMS."""
    if not LANE_CROSS:
        return gained
    return gained * gained.square().mean(2 - lanes, keepdim=True).clamp_min(1e-20).rsqrt()


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
    ratio: float, margin: float, heads: int = 1):
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin, heads)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = cross_lanes(update * energy.clamp_min(1e-10).rsqrt(), lanes)
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


@torch.compile(dynamic=False)
def anvil_update_merged(gs, ws, fasts, slows, energies, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd,
                        lanes: tuple, ratios: tuple, margin: float):
    """anvil_update for several batches whose polar inputs share a (non-square) shape after polar_batch's transpose:
    their Gram Newton-Schulz iterations run as one batched polar."""
    inputs, talls = [], []
    for g, fast, slow in zip(gs, fasts, slows):
        fast.lerp_(g, 1 - fast_beta)
        slow.lerp_(g, 1 - slow_beta)
        x = g.lerp(weight * fast + (1 - weight) * slow, beta)
        talls.append(x.size(1) > x.size(2))
        x = x.transpose(1, 2) if talls[-1] else x
        inputs.append(x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6))
    polar = gram_newton_schulz(torch.cat(inputs)).float().split([len(g) for g in gs])
    news = []
    for update, tall, w, slow, energy, lane, ratio in zip(polar, talls, ws, slows, energies, lanes, ratios):
        update = update.transpose(1, 2) if tall else update
        energy.lerp_(update.square().mean(lane + 1, keepdim=True), 1 - lane_beta)
        gained = cross_lanes(update * energy.clamp_min(1e-10).rsqrt(), lane)
        update = gained * (update.norm(dim=(1, 2), keepdim=True)
                           / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
        decay = (slow * w >= 0) * w * (wd * lr)
        news.append(w - lr * ratio * (update + decay))
    return news


FINITE_SIDE = True  # with LEAN_REST: the update's finite-check sum of the wire on a side stream under AdamW (exact)
NCCL_HIGH_PRIORITY = True  # the process group's NCCL streams at high priority (exact)
# Launch environment, set in main() before CUDA and NCCL start (scheduling only): MAX_CONNECTIONS the device's hardware
# work queues (CUDA_DEVICE_MAX_CONNECTIONS; CUDA's default 8 makes streams beyond 8 share queues, so a kernel can wait
# behind another stream's in launch order), NCCL_ENV extra NCCL variables (e.g. {'NCCL_PROTO': 'LL128'}).
MAX_CONNECTIONS = 0
# ADAM_SIDE: the Cut's AdamW (and the rest's gradient views) on a side stream that waits only for the rest's
# all-reduce, under ANVIL's units, instead of after them (disjoint parameters; exact).
# GATHER_EVENTS: with RANK_MAJOR_UNITS and ANVIL_OVERLAP, each side unit's all-gather waits only for its own unit
# (an event), issued in the order the units finish: this stream's unit first, then the side units in run order (exact).
ADAM_SIDE = False
GATHER_EVENTS = False
NCCL_ENV = {'NCCL_PROTO': 'LL128'}


def set_launch_environment():
    if (MAX_CONNECTIONS or NCCL_ENV) and torch.cuda.is_initialized():
        raise RuntimeError('MAX_CONNECTIONS / NCCL_ENV must be set before CUDA starts')
    if MAX_CONNECTIONS:
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = str(MAX_CONNECTIONS)
    os.environ.update({key: str(value) for key, value in NCCL_ENV.items()})


def nccl_options():
    """NCCL_HIGH_PRIORITY: the collectives' internal stream at high priority (their CTAs take free SMs first)."""
    if not NCCL_HIGH_PRIORITY:
        return None
    options = torch.distributed.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    return options
RANK_MAJOR_UNITS = True  # with RANK_MAJOR: one reduce-scatter / all-gather per shard unit (each unit's gather right after its polar)
RANK_MAJOR = True  # with ANVIL_GRAPHS: ANVIL's weights stored rank-major (each rank's shard rows of every batch side by side), one reduce-scatter and one all-gather per update
ANVIL_OVERLAP = True  # with ANVIL_GRAPHS: the units concurrently (last unit's reduce-scatters first, others' gathers first)
ANVIL_GRAPHS = True  # with SHARDED_ANVIL: each shard unit's polar and state update as a CUDA graph (bitwise)
CONCURRENT_POLAR = False  # with MERGED_POLAR: the other sharded ANVIL batches' updates on a side stream under it
MERGED_POLAR = True  # with SHARDED_ANVIL: ANVIL batches with the same polar shape (c_fc^T, c_proj) in one polar
FUSED_ADAM = True  # AdamW's fused CUDA kernel instead of the foreach path
LAZY_LOSS = False  # the logged Cut loss averaged over ranks once per telemetry flush, not per update
ASYNC_COMMS = True  # with SHARDED_ANVIL: gradient collectives async, waited where first read
COMMS_BF16 = True  # with SHARDED_ANVIL: the gradient reduce-scatter and all-reduce in bf16
CUT_SIGN_GENERATOR = True  # the chain roots' step signs from the Cut's own generator (not the default CUDA one)
WIDE_SIDE_PRIORITY = 0  # CONCURRENT_WIDE: the partner's replay stream priority (lower = higher; 0 = default)
WIDE_MAIN_PRIORITY = 0  # CONCURRENT_WIDE: nonzero: this Cut's replay on its own stream of this priority
CONCURRENT_WIDE = True  # with CUT_SIGN_GENERATOR: a wide update's second row by a partner graph on a side stream
AMORTIZED_ROOTS = (4,)  # these chain roots replayed on every ROOT_EVERY-th update only (their groups must not hold others)
ROOT_EVERY = 2  # AMORTIZED_ROOTS: the updates per measurement; the rest add the EMA of the measured deviations
ROOT_BETA = 0.5  # AMORTIZED_ROOTS: the EMA's decay per measurement
@torch.compile(dynamic=False)
def measuring_pack(full, dropped, deviation, weight, fresh: bool, bf16: bool):
    """AMORTIZED_ROOTS, a measuring update: the EMA moves toward the roots' deviation (full - dropped) by weight; the
    update takes the full gradient (fresh) or the gradient without the roots plus the EMA (and its bf16 copy)."""
    deviation.lerp_(full - dropped, weight)
    flat = full if fresh else dropped + deviation
    return flat, flat.to(torch.bfloat16) if bf16 else flat


@torch.compile(dynamic=False)
def skipping_pack(dropped, deviation, bf16: bool):
    """AMORTIZED_ROOTS, a skipping update: the gradient without the roots plus the deviations' EMA."""
    flat = dropped + deviation
    return flat, flat.to(torch.bfloat16) if bf16 else flat


ROOT_FROM, ROOT_UNTIL = 0.1, 0.7  # AMORTIZED_ROOTS: the fractions of the Cut updates amortized (ROOT_BOOST: else measured)
ROOT_BOOST = True  # AMORTIZED_ROOTS: no EMA; a measuring update's amortized deviations count ROOT_EVERY times
ROOT_FRESH = False  # AMORTIZED_ROOTS: a measuring update keeps its own deviation (else the EMA's, as the others)
PAIR_CHAIN_STREAMS = True  # with NATIVE_WIDE: the pair graph's second row's chain replays on a side stream
NATIVE_WIDE = True  # with CUT_SIGN_GENERATOR: a wide update's rows in pairs, each pair by one two-row estimate graph
PREFETCH_GATE_ROWS = True  # Cut.step: the next update's gate rows formed on a side stream under this estimate
LATE_LOSS = True  # with packed gradients: the update's loss sums and loss all-reduce on a side stream from the replay's end
MEANS_SIDE = True  # with FOREACH_MEANS: the merge averages' lerp on a side stream under the next estimate (exact)
GATE_GRAPH = True  # with PREFETCH_GATE_ROWS: the gate-row prefetch's draws as one CUDA graph (captured before the timer)
PREFETCH_PRIORITY = -1  # the gate-row prefetch and PRE_STAGE streams' priority (-1: high; their kernels take free SMs first)
EARLY_REPLAY = True  # with PRE_STAGE: estimate graphs open with an external event wait (the previous update's step) and
# replay on the staging stream, so their RNG prologue and launch run under the previous update's tail (exact)
PRE_STAGE = True  # an estimate replay's input copies (rows, gate rows) on a side stream under the previous update's tail
LEAN_REST = True  # with ASYNC_COMMS and COMMS_BF16: Cut.step's non-ANVIL grads from the reduced bf16 wire (no full copy)
DELAYED_UPDATE = False  # screen only: each Cut update applies the previous estimate
LEAN_FLAT = True  # with STACKED_ANVIL: divide by tokens x world before the all-reduce, check finiteness by a sum
ALLREDUCE_BF16 = False  # the update's gradient all-reduce in bf16
SHARDED_ANVIL = True  # with STACKED_ANVIL and LEAN_FLAT: each rank updates 1/world of every ANVIL batch
STACKED_ANVIL = True  # ANVIL's weights and grads as views of stacked storage (no per-update stacking copies)


def stacked_view(tensors):
    """The (n, *shape) tensor whose rows are `tensors`, as a view when they are consecutive in memory, else a copy."""
    first = tensors[0]
    n = first.numel()
    if all(t.is_contiguous() and t.data_ptr() == first.data_ptr() + i * n * t.element_size()
           for i, t in enumerate(tensors)):
        return first.as_strided((len(tensors), *first.shape), (n, *first.stride()))
    return torch.stack(tensors)


class BatchedAnvil(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, steps, polar_margin, fast_weight):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.polar_margin, self.fast_weight = polar_margin, fast_weight
        self.t = 0

    def prepare_scalar_schedule(self, warmdown, final, decay_factors, warmup=0):
        group = self.param_groups[0]
        device = group['params'][0].device
        rows = []
        for t in range(self.total):
            lo, hi, cool = .85, .93, self.total - self.cooldown
            beta = lo + (hi - lo) * t / self.warmup if t < self.warmup else hi
            if t > cool:
                beta = hi - (hi - lo) * (t - cool) / self.cooldown
            fast_beta, weight = (.85, self.fast_weight) if t >= self.engage else (beta, 1.0)
            left = (self.total - (t + 1) + 1) / self.total
            factor = (final + (1 - final) * min(1.0, left / warmdown)) * (min(1.0, (t + 1) / warmup) if warmup else 1.0)
            rows.append([beta, fast_beta, .98, weight, LANE_BETA,
                group['lr'] * factor, group['weight_decay'] * decay_factors[t]])
        self.scalar_schedule = torch.tensor(rows, dtype=torch.float32, device=device)
        by_shape = {}
        head_ids = getattr(self, 'head_ids', {})  # HEAD_POLAR, MLP_POLAR_GROUPS: id -> polar_batch's heads (make_optimizer)
        for p in group['params']:
            by_shape.setdefault((tuple(p.shape), head_ids.get(id(p), 1)), []).append(p)
        self.batches, self.batch_heads = [], []
        for (shape, heads), params in by_shape.items():
            self.batch_heads.append(heads)
            lanes = 1 if shape[0] >= shape[1] or LANE_ROWS else 0
            if STACKED_ANVIL:  # the group's weights become views of one stacked tensor
                stacked = torch.stack([p.data for p in params])
                for i, p in enumerate(params):
                    p.data = stacked[i]
            fast = torch.zeros(len(params), *shape, device=device)
            state = dict(fast=fast, slow=torch.zeros_like(fast),
                energy=torch.zeros_like(fast.mean(lanes + 1, keepdim=True)))
            self.batches.append((params, state, lanes, max(1, shape[0] / shape[1]) ** .5))
        if RANK_MAJOR and dist.is_initialized():
            self.store_rank_major()

    def store_rank_major(self):
        """RANK_MAJOR: each comm group's weights (all batches; RANK_MAJOR_UNITS: each shard unit's) moved into one
        buffer, rank r's chunk holding its shard rows of the group's batches (for the record's model: block r's
        matrices), so one all-gather fills a group."""
        if not (SHARDED_ANVIL and STACKED_ANVIL and LEAN_FLAT and MERGED_POLAR):
            raise ValueError('RANK_MAJOR needs SHARDED_ANVIL, STACKED_ANVIL, LEAN_FLAT and MERGED_POLAR')
        world = dist.get_world_size()
        if any(len(params) % world for params, *_ in self.batches):
            raise ValueError('RANK_MAJOR needs every ANVIL batch divisible by the world size')
        self.rank_groups = self.shard_units() if RANK_MAJOR_UNITS else [tuple(range(len(self.batches)))]
        first = self.batches[0][0][0]
        self.rank_group_of, self.rank_offsets, self.rank_chunks = {}, {}, []
        self.rank_stores, self.rank_sends, self.rank_receives = [], [], [None] * len(self.rank_groups)
        for g, ks in enumerate(self.rank_groups):
            chunk = 0
            for k in ks:
                params = self.batches[k][0]
                self.rank_group_of[k], self.rank_offsets[k] = g, chunk
                chunk += len(params) // world * params[0].numel()
            self.rank_chunks.append(chunk)
            self.rank_stores.append(torch.empty(world * chunk, device=first.device, dtype=first.dtype))
            self.rank_sends.append(torch.zeros(chunk, device=first.device, dtype=first.dtype))
        for k, (params, *_) in enumerate(self.batches):
            g, per = self.rank_group_of[k], len(params) // world
            for i, p in enumerate(params):
                r, j = divmod(i, per)
                start = r * self.rank_chunks[g] + self.rank_offsets[k] + j * p.numel()
                slot = self.rank_stores[g][start:start + p.numel()].view_as(p)
                slot.copy_(p.data)
                p.data = slot

    def receive_rank_major(self, wire_dtype):
        """RANK_MAJOR: each group's reduce-scatter output chunk and each batch's shard as a view of it."""
        if self.rank_receives[0] is None or self.rank_receives[0].dtype != wire_dtype:
            self.rank_receives = [torch.zeros(chunk, dtype=wire_dtype, device=self.rank_stores[0].device)
                                  for chunk in self.rank_chunks]
            self.shard_buffers = [self.shard_view(k, self.rank_receives) for k in range(len(self.batches))]

    def rank_major_order(self):
        """RANK_MAJOR: the batches' parameters in storage order (groups, rank chunks, batches, shard rows)."""
        world = dist.get_world_size()
        return [p for ks in self.rank_groups for r in range(world) for k in ks
                for p in self.batches[k][0][r * (len(self.batches[k][0]) // world):
                                            (r + 1) * (len(self.batches[k][0]) // world)]]

    def shard_view(self, k, buffers=None):
        """RANK_MAJOR: this rank's shard rows of batch k, in its group's chunk of `buffers` (one rank chunk per group)
        or in the weights' storage."""
        params, g = self.batches[k][0], self.rank_group_of[k]
        per = len(params) // dist.get_world_size()
        start = self.rank_offsets[k] + (dist.get_rank() * self.rank_chunks[g] if buffers is None else 0)
        source = self.rank_stores[g] if buffers is None else buffers[g]
        return source[start:start + per * params[0].numel()].view(per, *params[0].shape)

    def polar_partners(self):
        """Groups of batch indices whose polar inputs (polar_batch's transpose of tall shapes) share a non-square
        shape: MERGED_POLAR runs each group's polar as one."""
        groups = {}
        for k, (params, *_) in enumerate(self.batches):
            rows, cols = params[0].shape
            if rows != cols and self.batch_heads[k] == 1:
                groups.setdefault((min(rows, cols), max(rows, cols)), []).append(k)
        return [ks for ks in groups.values() if len(ks) > 1]

    def merged_shard_step(self, ks, scalars):
        """The sharded step of batches ks (polar partners) with one polar: wait each shard, update, all-gather."""
        world, rank = dist.get_world_size(), dist.get_rank()
        gs, ws, views, fasts, slows, energies, lanes, ratios = [], [], [], [], [], [], [], []
        for k in ks:
            params, state, lane, ratio = self.batches[k]
            per = len(params) // world
            sl = slice(rank * per, (rank + 1) * per)
            w = stacked_view([p.data for p in params])
            if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                self.shard_works[k].wait()
            gs.append(self.shard_grads[k].float())
            ws.append(w[sl].float())
            views.append(w)
            fasts.append(state['fast'][sl])
            slows.append(state['slow'][sl])
            energies.append(state['energy'][sl])
            lanes.append(lane)
            ratios.append(ratio)
        news = anvil_update_merged(gs, ws, fasts, slows, energies, *scalars, tuple(lanes), tuple(ratios),
                                   self.polar_margin)
        for w, new in zip(views, news):
            work = dist.all_gather_into_tensor(w, new.to(w.dtype).contiguous(), async_op=ASYNC_COMMS)
            if work is not None:
                self.gather_works = getattr(self, 'gather_works', []) + [work]

    def shard_units(self):
        """The sharded step's units in step order: each polar-partner group at its last batch, the others alone."""
        merged = {k: ks for ks in self.polar_partners() for k in ks} if MERGED_POLAR else {}
        return [tuple(merged.get(k, (k,))) for k in range(len(self.batches)) if k not in merged or k == merged[k][-1]]

    def unit_sources(self, ks, scalars):
        """ANVIL_GRAPHS: this rank's new shard rows of batches ks as all-gather sources, by the ops (and compiled
        functions) of step's sharded path."""
        world, rank = dist.get_world_size(), dist.get_rank()
        views, slices, weights = [], [], []
        for k in ks:
            params = self.batches[k][0]
            per = len(params) // world
            slices.append(slice(rank * per, (rank + 1) * per))
            views.append(None if RANK_MAJOR else stacked_view([p.data for p in params]))
            weights.append(self.shard_view(k) if RANK_MAJOR else views[-1][slices[-1]])
        states = [self.batches[k][1] for k in ks]
        if len(ks) > 1:
            news = anvil_update_merged([self.shard_grads[k].float() for k in ks], [w.float() for w in weights],
                                       [st['fast'][sl] for st, sl in zip(states, slices)],
                                       [st['slow'][sl] for st, sl in zip(states, slices)],
                                       [st['energy'][sl] for st, sl in zip(states, slices)], *scalars,
                                       tuple(self.batches[k][2] for k in ks), tuple(self.batches[k][3] for k in ks),
                                       self.polar_margin)
        else:
            (k,), w, sl, st = ks, weights[0], slices[0], states[0]
            news = [anvil_update(self.shard_grads[k].float(), w.float(), st['fast'][sl], st['slow'][sl],
                                 st['energy'][sl], *scalars, self.batches[k][2], self.batches[k][3], self.polar_margin,
                                 self.batch_heads[k])]
        if RANK_MAJOR:  # into this rank's chunk of the all-gather's source
            sends = [self.shard_view(k, self.rank_sends) for k in ks]
            for send, new in zip(sends, news):
                send.copy_(new)
            return sends
        return [new.to(w.dtype).contiguous() for new, w in zip(news, views)]

    @torch.no_grad()
    def capture_shard_graphs(self, wire_dtype):
        """ANVIL_GRAPHS (before the timer): persistent reduce-scatter shards and one CUDA graph per shard unit
        (the polar and state update; the collectives stay eager), the scalars read from a static row. The eager
        warm-up's state changes are undone."""
        world = dist.get_world_size()
        device = self.scalar_schedule.device
        if RANK_MAJOR:  # the reduce-scatter's output chunk; each batch's shard a view of it
            self.receive_rank_major(wire_dtype)
        else:
            for params, *_ in self.batches:
                w = stacked_view([p.data for p in params])
                assert w._base is not None or w.data_ptr() == params[0].data.data_ptr(), 'ANVIL_GRAPHS needs STACKED_ANVIL views'
            self.shard_buffers = [torch.zeros((len(params) // world, *params[0].shape), dtype=wire_dtype,
                                              device=device) for params, *_ in self.batches]
        saved = [{key: x.clone() for key, x in state.items()} for _, state, *_ in self.batches]
        self._scalars = self.scalar_schedule[0].clone()
        scalars = self._scalars.unbind()
        self.shard_grads = self.shard_buffers
        units = self.shard_units()
        main, side = torch.cuda.current_stream(), torch.cuda.Stream()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:  # as Optimizers.step runs it (deterministic algorithms change the compiled kernels)
            torch.use_deterministic_algorithms(True)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for ks in units:
                    self.unit_sources(ks, scalars)
            main.wait_stream(side)
            torch.cuda.synchronize()
            self._unit_graphs = {}
            for ks in units:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    sources = self.unit_sources(ks, scalars)
                self._unit_graphs[ks] = (graph, sources)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for (_, state, *_), copy in zip(self.batches, saved):
            for key, x in state.items():
                x.copy_(copy[key])
        self.shard_grads = None
        torch.cuda.synchronize()

    def graphed_step(self):
        """step's sharded path by the captured unit graphs: per unit, wait its shards, replay, all-gather.
        ANVIL_OVERLAP: the units but the last on a side stream (their all-gathers issued first), the last unit (its
        reduce-scatters issued first by Cut.update) on this stream at the same time."""
        self._scalars.copy_(self.scalar_schedule[self.t])
        if ANVIL_OVERLAP:
            return self.overlapped_graphed_step()
        for ks in self.shard_units():
            for k in ks:
                if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                    self.shard_works[k].wait()
            graph, sources = self._unit_graphs[ks]
            graph.replay()
            for k, source in zip(ks, sources):
                w = stacked_view([p.data for p in self.batches[k][0]])
                work = dist.all_gather_into_tensor(w, source, async_op=ASYNC_COMMS)
                if work is not None:
                    self.gather_works = getattr(self, 'gather_works', []) + [work]
        self.shard_grads = None
        self.t += 1

    def unit_replay(self, ks):
        for k in ks:
            if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                self.shard_works[k].wait()
        graph, sources = self._unit_graphs[ks]
        graph.replay()
        for k, source in zip(ks, sources):
            w = stacked_view([p.data for p in self.batches[k][0]])
            work = dist.all_gather_into_tensor(w, source, async_op=ASYNC_COMMS)
            if work is not None:
                self.gather_works = getattr(self, 'gather_works', []) + [work]

    def rank_major_step(self):
        """RANK_MAJOR: per unit, wait its reduce-scatter and run it (its graph, or eagerly before the graphs are
        captured); ANVIL_OVERLAP: the last unit (the merged polar) on this stream, the others on a side stream at the
        same time, else the units in reverse order on this stream. Then the all-gathers (RANK_MAJOR_UNITS: one per
        unit's group, the last unit's first, each issued from the stream that ran it)."""
        units = self.shard_units()
        graphs = getattr(self, '_unit_graphs', None)
        scalars = None if graphs else self.scalar_schedule[self.t].unbind()
        waited = set()
        def run(n):
            for k in units[n]:
                work = self.shard_works[k] if getattr(self, 'shard_works', None) else None
                if work is not None and id(work) not in waited:
                    work.wait()
                    waited.add(id(work))
            graphs[units[n]][0].replay() if graphs else self.unit_sources(units[n], scalars)
        main = torch.cuda.current_stream()
        streams, done = {}, {}
        if ANVIL_OVERLAP:
            side = self._unit_side = getattr(self, '_unit_side', None) or torch.cuda.Stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for n in range(len(units) - 1):
                    run(n)
                    streams[n] = side
                    if GATHER_EVENTS:
                        done[n] = torch.cuda.Event()
                        done[n].record()
            run(len(units) - 1)
            streams[len(units) - 1] = main
        else:
            for n in reversed(range(len(units))):
                run(n)
                streams[n] = main
        if RANK_MAJOR_UNITS:
            last_unit = {g: max(n for n, u in enumerate(units) if set(u) & set(ks))
                         for g, ks in enumerate(self.rank_groups)}
            if done:  # GATHER_EVENTS: this stream's unit's gather first, then the side units' in the order they ran
                gather = self._gather_stream = getattr(self, '_gather_stream', None) or torch.cuda.Stream()
                for g in sorted(last_unit, key=lambda g: (last_unit[g] in done, last_unit[g])):
                    if last_unit[g] in done:
                        gather.wait_event(done[last_unit[g]])
                    with torch.cuda.stream(gather if last_unit[g] in done else main):
                        self.rank_gather(g)
                main.wait_stream(gather)
            else:
                for g in sorted(last_unit, key=lambda g: -last_unit[g]):
                    with torch.cuda.stream(streams[last_unit[g]]):
                        self.rank_gather(g)
        else:
            if ANVIL_OVERLAP:
                main.wait_stream(side)
            self.rank_gather(0)
        if ANVIL_OVERLAP:
            main.wait_stream(side)
        self.shard_grads = None
        self.t += 1

    def rank_gather(self, g):
        work = dist.all_gather_into_tensor(self.rank_stores[g], self.rank_sends[g], async_op=ASYNC_COMMS)
        if work is not None:
            self.gather_works = getattr(self, 'gather_works', []) + [work]

    def overlapped_graphed_step(self):
        units = self.shard_units()
        main = torch.cuda.current_stream()
        side = self._unit_side = getattr(self, '_unit_side', None) or torch.cuda.Stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for ks in units[:-1]:
                self.unit_replay(ks)
        self.unit_replay(units[-1])
        main.wait_stream(side)
        self.shard_grads = None
        self.t += 1

    @torch.no_grad()
    def step(self):
        if RANK_MAJOR and getattr(self, 'shard_grads', None) is not None:
            if getattr(self, '_unit_graphs', None):
                self._scalars.copy_(self.scalar_schedule[self.t])
            return self.rank_major_step()
        if (ANVIL_GRAPHS and getattr(self, '_unit_graphs', None) and SHARDED_ANVIL
                and getattr(self, 'shard_grads', None) is not None and not CONCURRENT_POLAR):
            return self.graphed_step()
        scalars = self.scalar_schedule[self.t].unbind()
        merged, joins = {}, []
        if MERGED_POLAR and SHARDED_ANVIL and getattr(self, 'shard_grads', None) is not None:
            for ks in self.polar_partners():
                merged.update({k: ks for k in ks})
        for k, (params, state, lanes, ratio) in enumerate(self.batches):
            if k in merged:
                if k == merged[k][-1]:  # one polar for the whole group, at its last batch
                    self.merged_shard_step(merged[k], scalars)
                continue
            if SHARDED_ANVIL and getattr(self, 'shard_grads', None) is not None:
                world, rank = dist.get_world_size(), dist.get_rank()
                per = len(params) // world
                sl = slice(rank * per, (rank + 1) * per)
                w = stacked_view([p.data for p in params])
                concurrent = CONCURRENT_POLAR and bool(merged)  # under the merged group's polar, on a side stream
                if concurrent:
                    if not hasattr(self, '_side_stream'):
                        self._side_stream = torch.cuda.Stream()
                    side = self._side_stream
                    side.wait_stream(torch.cuda.current_stream())
                    self.shard_grads[k].record_stream(side)
                with torch.cuda.stream(side) if concurrent else contextlib.nullcontext():
                    if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                        self.shard_works[k].wait()
                    new = anvil_update(self.shard_grads[k].float(), w[sl].float(), state['fast'][sl],
                                       state['slow'][sl], state['energy'][sl], *scalars, lanes, ratio,
                                       self.polar_margin, self.batch_heads[k])
                    work = dist.all_gather_into_tensor(w, new.to(w.dtype).contiguous(), async_op=ASYNC_COMMS)
                if work is not None:
                    self.gather_works = getattr(self, 'gather_works', []) + [work]
                if concurrent:
                    joins.append(side)
                continue
            if STACKED_ANVIL:
                w, g = stacked_view([p.data for p in params]), stacked_view([p.grad for p in params])
                new = anvil_update(g.float(), w.float(), state['fast'], state['slow'], state['energy'], *scalars,
                                   lanes, ratio, self.polar_margin, self.batch_heads[k])
                if w.data_ptr() == params[0].data.data_ptr() and w._base is not None:
                    w.copy_(new)
                else:
                    torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
                continue
            new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                state['fast'], state['slow'], state['energy'], *scalars, lanes, ratio, self.polar_margin,
                self.batch_heads[k])
            torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        for side in joins:
            torch.cuda.current_stream().wait_stream(side)
        self.shard_grads = None
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
    def zero_grad(self, set_to_none):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
    def step_side(self, between):
        """ADAM_SIDE: between() and the second optimizer on a side stream (it waits there for the rest's all-reduce),
        under the first one's step; this stream joins it afterwards."""
        main = torch.cuda.current_stream()
        side = self._side = getattr(self, '_side', None) or torch.cuda.Stream()
        side.wait_stream(main)
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.cuda.stream(side):
                between()
                self.optimizers[1].step()
            self.optimizers[0].step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        main.wait_stream(side)
    def step(self, between=None):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for k, o in enumerate(self.optimizers):
                if k == 1 and between is not None:
                    between()
                o.step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_model(device, seed, config):
    # Only the random-token warmup reads these init values: the identity handoff overwrites every weight of the timed
    # model. The seed and the init draws stay because they set the default CUDA generator's state, which the Cut
    # estimate's per-token root signs (chain_credits) draw from.
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        nn.init.normal_(model.lm_head.weight, std=0.02)
    return model.requires_grad_(False)


TRITON_ADAM = True  # the Cut's AdamW as ExactAdamW (one Triton pass per tensor, the fused kernel's arithmetic)


@triton.jit
def _adamw_kernel(P, G, M, V, N, step_size, bc2_sqrt, beta1: tl.constexpr, beta2: tl.constexpr, eps: tl.constexpr,
                  BLOCK: tl.constexpr):
    """torch's fused AdamW step (weight_decay 0, no amsgrad) per element: the moments in fp64 from the stored values,
    the update in fp32 with IEEE division and square root (step_size = fp32(lr / bias_correction1) and bc2_sqrt
    rounded on the host as the fused kernel rounds them)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p = tl.load(P + offs, mask=mask).to(tl.float32)
    g = tl.load(G + offs, mask=mask).to(tl.float32).to(tl.float64)
    m = tl.load(M + offs, mask=mask).to(tl.float32).to(tl.float64)
    v = tl.load(V + offs, mask=mask).to(tl.float32).to(tl.float64)
    m_new = (beta1 * m + (1 - beta1) * g).to(tl.float32)
    v_new = (beta2 * v + (1 - beta2) * g * g).to(tl.float32)
    denom = (tl.div_rn(tl.sqrt_rn(v_new), bc2_sqrt).to(tl.float64) + eps).to(tl.float32)
    tl.store(P + offs, (p - tl.div_rn(step_size * m_new, denom)).to(P.dtype.element_ty), mask=mask)
    tl.store(M + offs, m_new.to(M.dtype.element_ty), mask=mask)
    tl.store(V + offs, v_new.to(V.dtype.element_ty), mask=mask)


class ExactAdamW(torch.optim.Optimizer):
    """AdamW(fused=True) with weight_decay 0, numerically the same (bf16 tensors bitwise; fp32 ones up to the fp64
    moments' last-bit rounding), at full memory bandwidth (the fused kernel's 8-byte bf16 accesses run at ~1 TB/s).
    State as torch's AdamW keeps it (step, exp_avg, exp_avg_sq; the step count may be a CPU tensor)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=None):
        assert weight_decay == 0.0
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.0), exp_avg=torch.zeros_like(p, memory_format=torch.preserve_format),
                                 exp_avg_sq=torch.zeros_like(p, memory_format=torch.preserve_format))
                if state['step'].is_cuda:  # (once: the count lives on the host from here on)
                    state['step'] = state['step'].cpu()
                state['step'] += 1
                step = state['step'].item()
                bc1 = float(np.float32(1 - beta1 ** step))
                step_size = float(np.float32(group['lr'] / bc1))
                bc2_sqrt = float(np.float32(math.sqrt(1 - beta2 ** step)))
                n = p.numel()
                assert all(z.is_contiguous() for z in (p, p.grad, state['exp_avg'], state['exp_avg_sq']))
                _adamw_kernel[(triton.cdiv(n, 2048),)](p, p.grad, state['exp_avg'], state['exp_avg_sq'], n,
                                                       step_size, bc2_sqrt, beta1, beta2, group['eps'], BLOCK=2048,
                                                       num_warps=8)


def make_optimizer(model, recipe, steps):
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = (recipe.emb_scale if ('wte' in name or 'value_embeds' in name)
        else recipe.lambda_scale if 'lambdas' in name else 1)
        groups.setdefault(scale, []).append(parameter)
    adam = (ExactAdamW if TRITON_ADAM else torch.optim.AdamW)(
      [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
      betas=recipe.betas, eps=1e-8, weight_decay=0.0, fused=FUSED_ADAM or None,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
      polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    anvil.head_ids = {id(getattr(block.attn, f'c_{kind}').weight): -model.config.n_head if kind == 'proj' else
                      model.config.n_head for block in model.transformer.h for kind in HEAD_POLAR}
    if HEAD_POLAR_JOINT and tuple(HEAD_POLAR) != ('q', 'k'):
        raise ValueError("HEAD_POLAR_JOINT needs HEAD_POLAR=('q', 'k')")
    if MLP_POLAR_GROUPS > 1:
        for block in model.transformer.h:
            anvil.head_ids[id(block.mlp.c_fc.weight)] = MLP_POLAR_GROUPS
            anvil.head_ids[id(block.mlp.c_proj.weight)] = -MLP_POLAR_GROUPS
    # ANVIL weight decay is scaled by late_cut_decay_multiplier (zero) after the first 275/1043 of the Cut updates.
    late_from = round(275 / 1043 * steps)
    decay_factors = tuple(recipe.late_cut_decay_multiplier if t >= late_from else 1.0 for t in range(steps))
    anvil.prepare_scalar_schedule(recipe.warmdown, recipe.final, decay_factors, recipe.cut_warmup)
    return Optimizers(anvil, adam)


def cut_updates(steps, settings):
    """The Cut's updates as tuples of the cut_batch-row steps (1-based) each takes: one step each until wide_from of
    them, then wide_rows // cut_batch each until late_from, then late_rows // cut_batch each (each phase's first
    update moved later so that its updates fill the phase)."""
    phases = [(round(start * steps), rows // settings.cut_batch)
              for start, rows in ((settings.wide_from, settings.wide_rows), (settings.late_from, settings.late_rows))
              if start < 1 and rows > settings.cut_batch]
    updates, end = [], steps
    for start, per in reversed(phases):
        first = end - (end - min(start, end)) // per * per
        updates = [tuple(range(t, t + per)) for t in range(first + 1, end + 1, per)] + updates
        end = first
    return [(t,) for t in range(1, end + 1)] + updates


def rate_scales(settings, group):
    """(AdamW's, ANVIL's) rate scale for an update of these steps."""
    per = len(group)
    if per == 1:
        return 1.0, 1.0
    if per == settings.late_rows // settings.cut_batch and settings.late_from < 1:
        return settings.late_lr_scale, settings.late_anvil_scale
    return settings.wide_lr_scale, settings.wide_anvil_scale


def widen_anvil_schedule(optimizer, updates, settings):
    """ANVIL's per-update scalars for these updates: each the mean of its steps' rows, the rate scaled in wide ones."""
    anvil = optimizer.optimizers[0]
    table, rows = anvil.scalar_schedule, []
    for group in updates:
        row = table[group[0] - 1:group[-1]].mean(0)
        if len(group) > 1:
            row[5] *= rate_scales(settings, group)[1]
        rows.append(row)
    anvil.scalar_schedule = torch.stack(rows)


def set_rate(optimizer, step, steps, warmdown, final, warmup=0):
    left = (steps - step + 1) / steps
    ramp = min(1.0, step / warmup) if warmup else 1.0
    for group in optimizer.optimizers[1].param_groups:  # AdamW; ANVIL's rates are precomputed in its scalar schedule
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * ramp * (final + (1 - final) * min(1.0, left / warmdown))


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 64
MERGE_FROM, MERGE_EVERY = 0.3, 5
MERGE_DECAYS = (0.02,)
MERGE_RHOS = (1.0,)
MERGE_K, MERGE_TOKENS = 64, 512


MERGE_SKETCH_COLUMNS = 128


@torch.no_grad()
def merge_sketch_frame(width, columns, device, generator):
    raw = torch.randn(width, columns, device=device,
      generator=generator, dtype=torch.float32)
    return torch.linalg.qr(raw, mode='reduced')[0].contiguous()


@torch.no_grad()
def merge_sketch_add(x, omega):
    with torch.autocast('cuda', enabled=False):
        xf = x.reshape(-1, x.shape[-1]).float()
        return (xf.T @ (xf @ omega)).double()


@torch.no_grad()
def merge_nystrom_basis(y, omega, k):
    middle = omega.double().T @ y
    eigen, vectors = torch.linalg.eigh((middle + middle.T) * .5)
    floor = eigen[-1].clamp_min(1e-30) * 1e-6
    inverse = torch.where(eigen > floor, eigen.clamp_min(floor).rsqrt(), 0.)
    factor = y @ (vectors * inverse.unsqueeze(0))
    q, r = torch.linalg.qr(factor, mode='reduced')
    _, u = torch.linalg.eigh(r @ r.T)
    return (q @ u[:, -k:].flip(1)).float().contiguous()


@torch.no_grad()
def curvature_bases(model, rows, rank, k=MERGE_K):
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
      if isinstance(mod, nn.Linear) and 've_gate' not in name]
    bases = {key: torch.zeros(mod.weight.shape[1], k, device=device) for key, mod in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(SEED)
        frames = {key: merge_sketch_frame(mod.weight.shape[1], MERGE_SKETCH_COLUMNS,
        device, generator) for key, mod in layers}
        acc = {key: torch.zeros_like(omega, dtype=torch.float64) for key, omega in frames.items()}
        hooks = []
        for key, mod in layers:
            def hook(module, inputs, output, key=key):
                acc[key].add_(merge_sketch_add(inputs[0], frames[key]))
            hooks.append(mod.register_forward_hook(hook))
        counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)
        for batch in rows.split(4):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            counts += torch.bincount(batch[:, :-1].reshape(-1).long(), minlength=len(counts)).double()
        for h in hooks:
            h.remove()
        for key, _ in layers:
            bases[key] = merge_nystrom_basis(acc[key], frames[key], k)
        order = torch.argsort(counts, descending=True)
    for key, _ in layers:
        torch.distributed.broadcast(bases[key], 0)
    torch.distributed.broadcast(order, 0)
    return bases, order


def subspace_move(name, move, bases, order):
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


MERGE_COEFS = (0.75, 1.0)


@torch.no_grad()
def joint_parts(named, params, final, move, candidates, plain, scorer, settings, sweeps=2, shortlist=8):
    parts = sorted({merge_part(n) for n, _ in named})
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in MERGE_COEFS]
    cache = {}
    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]
    members = {}
    for i, (n, _) in enumerate(named):
        members.setdefault(merge_part(n), []).append(i)
    held = dict.fromkeys(parts)
    def show(choice):
        for p in parts:
            if held[p] != choice[p]:
                for i in members[p]:
                    params[i].copy_(value(i, choice[p]).to(params[i].dtype))
                held[p] = choice[p]
    def loss_for(trial, count):
        show(trial)
        return scorer.score(count)
    best_key, best, global_losses, _ = race_merge_keys(
      keys, None, plain, lambda key, count: loss_for(dict.fromkeys(parts, key), count),
      settings.merge_screen_rows, HOLDOUT, settings.merge_screen_global)
    choice = dict.fromkeys(parts, best_key)
    keys = [None] + [k for _, k in sorted(global_losses, key=lambda x: x[0]) if k is not None][:settings.merge_keys]
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            incumbent = choice[p]
            candidates_p = keys if sweep == 0 else ranked[p]
            def loss_of_key(key, count):
                trial = dict(choice)
                trial[p] = key
                return loss_for(trial, count)
            key, complete_loss, _, proxy_order = race_merge_keys(
          candidates_p, incumbent, best, loss_of_key, settings.merge_screen_rows,
          HOLDOUT, settings.merge_screen_coordinate)
            choice[p], best = key, complete_loss
            if sweep == 0:
                ranked[p] = proxy_order[:shortlist]
                if key not in ranked[p]:
                    ranked[p].append(key)
    show(choice)
    return dict(holdout_plain=plain, holdout_merged=best,
      merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]])
      for p, k in choice.items()})


def holdout_sum(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, y, loss_reduction='sum')


holdout_sum_captured = torch.compile(holdout_sum, dynamic=False)


def score_merge_keys(keys, loss_of_key, count):
    # Each score is an owned scalar clone, recorded before the next candidate writes.
    losses = torch.stack([loss_of_key(key, count) for key in keys]).cpu().tolist()
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError('Non-finite merge scores')
    return list(zip(losses, keys))


def race_merge_keys(keys, incumbent, incumbent_full_loss, loss_of_key, screen_rows, full_rows, keep):
    screened = score_merge_keys(keys, loss_of_key, screen_rows)
    ranked = [key for _, key in sorted(screened, key=lambda item: item[0])]
    finalists = ranked[:keep]
    if incumbent not in finalists:
        finalists.append(incumbent)
    measured = dict((key, loss) for loss, key in score_merge_keys(
      [key for key in finalists if key != incumbent], loss_of_key, full_rows))
    complete = [(incumbent_full_loss if key == incumbent else measured[key], key) for key in finalists]
    best_key, best_loss = incumbent, incumbent_full_loss
    for loss, key in complete:
        if loss < best_loss:
            best_key, best_loss = key, loss
    return best_key, best_loss, complete, ranked


class ResidentMergeScore:
    """Held-out loss on two fixed row cohorts (the screening rows and all held-out rows, split across ranks). Each
    cohort's forward and all-reduce are captured once as a CUDA graph on random tokens before the timer; bind()
    loads the held-out rows into the graph inputs."""
    def __init__(self, model, rank, world, sequence_len, screen_rows):
        self.model, self.rank, self.world = model, rank, world
        self.device = next(model.parameters()).device
        self.buffers, self.graphs = {}, {}
        for count in (screen_rows, HOLDOUT):
            shape = (count // world, sequence_len)
            x = torch.zeros(shape, device=self.device, dtype=torch.long)
            y = torch.zeros_like(x)
            total = torch.zeros(2, device=self.device, dtype=torch.float64)
            result = torch.zeros((), device=self.device, dtype=torch.float64)
            self.buffers[count] = (x, y, total, result)

    def bind(self, rows):
        for count, (x, y, _, _) in self.buffers.items():
            batch = rows[:count][self.rank::self.world].to(self.device)
            x.copy_(batch[:, :-1])
            y.copy_(batch[:, 1:])

    def execute(self, count):
        x, y, total, result = self.buffers[count]
        total.zero_()
        loss = holdout_sum_captured(self.model, x, y)
        total[0].add_(loss.double())
        total[1].add_(y.numel())
        dist.all_reduce(total)
        result.copy_(total[0] / total[1])

    def capture(self):
        for count in self.buffers:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.execute(count)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            dist.barrier()
            torch._C._cuda_clearCublasWorkspaces()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                self.execute(count)
            self.graphs[count] = graph
        torch.cuda.synchronize()

    def score(self, count):
        self.graphs[count].replay()
        return self.buffers[count][3].clone()

    def close(self):
        torch.cuda.synchronize()
        self.graphs.clear()


@torch.no_grad()
def prepare_resident_merge_score(model, settings, rank, world):
    noise = torch.randint(0, settings.model.vocab_size, (HOLDOUT, settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(SEED + 270))
    curvature_bases(model, noise, rank)  # random tokens; discarded
    scorer = ResidentMergeScore(model, rank, world, settings.model.sequence_len, settings.merge_screen_rows)
    scorer.bind(noise)
    scorer.capture()
    if QUAD_MERGE:
        quad_warmup(model, noise[:QUAD_ROWS].to(scorer.device), rank, world)
    return scorer


@torch.no_grad()
def end_merge(model, means, rows, rank, scorer, settings):
    scorer.bind(rows)
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows, rank) if any(rho < 1 for rho in MERGE_RHOS) else ({}, None)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]
    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)
    plain = scorer.score(HOLDOUT).item()
    return joint_parts(named, params, final, move, candidates, plain, scorer, settings, sweeps=settings.merge_sweeps)


# QUAD_MERGE: the end merge as per-part coefficients a in w + a (avg - w), chosen inside the run. The parts are split
# over the ranks; each rank fits a parabola through a = C0 - DELTA, C0, C0 + DELTA for each of its parts on its own
# QUAD_ROWS rows (one forward from the part's first block, the stream below it cached from the C0 forward), one
# all-reduce shares the vertices (clipped to QUAD_RANGE), and the joint move C0 -> vertices is shrunk by the best of
# QUAD_SHRINKS on the 64 held-out rows.
QUAD_MERGE = True
QUAD_DECAY, QUAD_C0, QUAD_DELTA, QUAD_RANGE = 0.02, 0.75, 0.5, (-0.25, 1.25)
QUAD_ROWS, QUAD_SOURCE, QUAD_SHRINKS = 24, 'identity', (0.25, 0.5, 0.75)
SETUP_THREAD = True  # with OVERLAP_SETUP: the rows' host setup on a thread while the identity replays are enqueued
IDENTITY_KEEP_CACHE = True  # the identity phase's freed memory kept cached (no empty_cache in the timer; exact)
OVERLAP_SETUP = True  # the timed host setup (row order, checks, the rows' copy) under the identity phase's replays
GC_FREEZE = True  # collect and freeze the warm-up's Python objects before the timer (no full collection inside it)
FOREACH_MEANS = True  # the merge averages' updates as one multi-tensor lerp per average (exact)
# TRANSITION_LAZY: identity -> Cut without a host sync: the identity record's loss and time are read at the first
# telemetry flush and the identity graphs destroyed after the timer (their destruction took ~1.3 ms of host time).
TRANSITION_LAZY = True
CUT_RESERVE_GB = 8  # a block this large allocated and freed before the timer: the timed allocations reuse the cache
quad_reserve = []  # the warm-up's outputs, held until the merge so their memory stays in the allocator's cache


def quad_start(name, layers):
    """First block a parameter acts on: -1 the embedding (x0), `layers` the head only."""
    if name == 'transformer.wte.weight':
        return -1
    if name == 'lm_head.weight':
        return layers
    if name.startswith('value_embeds.'):
        return int(name.split('.')[1])
    if name.startswith('transformer.h.'):
        return int(name.split('.')[2])
    return 0  # resid_lambdas, x0_lambdas


def quad_tail(model, start, idx, targets, x, x0, keep):
    """Summed loss of the forward from block `start` given the stream x entering it (start -1: from the tokens).
    keep: also return x0 and the stream entering each block from `start` on, and the last block's output."""
    layers, vocab = model.config.n_layer, model.config.vocab_size
    cos_sin = (model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if start < 0:
            x = x0 = norm(model.transformer.wte(idx))
            start = 0
        streams = []
        for i in range(start, layers):
            streams.append(x)
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
            x = model.transformer.h[i](x, ve, cos_sin, model.window_sizes[i])
        streams.append(x)
        logits = model.lm_head(norm(x))[..., :vocab].float()
        logits = 15 * torch.tanh(logits / 15)
        loss = F.cross_entropy(logits.view(-1, vocab), targets.reshape(-1), reduction='sum')
    return (loss, x0, streams) if keep else loss


quad_tail_compiled = torch.compile(quad_tail, dynamic=False)


def quad_plan(named, world, layers):
    """Parts, their members and first blocks, and each rank's parts: the parts ordered by first block, cut into
    `world` consecutive runs of least maximal cost (two probes each, plus one center per first block a rank holds;
    a forward from block s costs layers + 1 - s blocks)."""
    members = {}
    for i, (n, _) in enumerate(named):
        members.setdefault(merge_part(n), []).append(i)
    parts = sorted(members)
    first = {p: min(quad_start(named[i][0], layers) for i in members[p]) for p in parts}
    cost = lambda start: layers + 1 - max(start, 0)
    ordered = sorted(parts, key=lambda p: (first[p], p))
    def split(capacity):
        runs, load, starts = [[]], 0, set()
        for p in ordered:
            extra = 2 * cost(first[p]) + (0 if first[p] in starts else cost(first[p]))
            if runs[-1] and load + extra > capacity:
                runs.append([])
                load, starts = 0, set()
                extra = 3 * cost(first[p])
            runs[-1].append(p)
            load += extra
            starts.add(first[p])
        return runs
    capacity = 3 * cost(-1)
    while len(split(capacity)) > world:
        capacity += 1
    owned = split(capacity)
    return parts, members, first, owned + [[] for _ in range(world - len(owned))]


@torch.no_grad()
def quad_warmup(model, rows, rank, world):
    """Compile the forward from every start on random tokens (before the timer)."""
    idx, targets = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    _, x0, streams = quad_tail_compiled(model, -1, idx, targets, None, None, True)
    for start in range(-1, model.config.n_layer + 1):
        x = streams[max(start, 0)] if start >= 0 else None
        quad_tail_compiled(model, start, idx, targets, x, x0, False)
    torch.cuda.synchronize()
    quad_reserve[:] = [x0, streams] + [[torch.empty_like(q, dtype=torch.float32) for q in model.parameters()]
                                       for _ in range(2)]


@torch.no_grad()
def quad_merge(model, means, rows, rank, world, scorer):
    """rows: this rank's pick rows. The scorer is bound to the held-out rows."""
    quad_reserve.clear()
    clock = [time.perf_counter()]
    def lap():
        torch.cuda.synchronize()
        clock.append(time.perf_counter())
    named = list(model.named_parameters())
    params = [q for _, q in named]
    layers = model.config.n_layer
    mean = means[MERGE_DECAYS.index(QUAD_DECAY)]
    final = [q.detach().float().clone() for q in params]
    move = [m - f for m, f in zip(mean, final)]
    def put(i, a):
        params[i].copy_(torch.add(final[i], move[i], alpha=a).to(params[i].dtype))
    parts, members, first, owned = quad_plan(named, world, layers)
    for i in range(len(params)):
        put(i, QUAD_C0)
    idx, targets = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    lap()
    _, x0, streams = quad_tail_compiled(model, -1, idx, targets, None, None, True)
    lap()
    # Each probe is compared with the center scored by the same forward (its start's graph rounds alike).
    centers, probes = {}, []
    for p in owned[rank]:
        start = first[p]
        x = streams[start] if start >= 0 else None
        if start not in centers:
            centers[start] = quad_tail_compiled(model, start, idx, targets, x, x0, False)
        probes.append(centers[start])
        for a in (QUAD_C0 - QUAD_DELTA, QUAD_C0 + QUAD_DELTA):
            for i in members[p]:
                put(i, a)
            probes.append(quad_tail_compiled(model, start, idx, targets, x, x0, False))
        for i in members[p]:
            put(i, QUAD_C0)
    vertex = torch.zeros(len(parts), device=x0.device, dtype=torch.float64)
    if probes:
        center, low, high = torch.stack(probes).double().view(-1, 3).unbind(1)
        curve, slope = low + high - 2 * center, high - low
        step = torch.where(curve > 0, -QUAD_DELTA * slope / (2 * curve.clamp_min(1e-30)),
                           torch.where(slope < 0, 2 * QUAD_DELTA, -2 * QUAD_DELTA))
        own = torch.tensor([parts.index(p) for p in owned[rank]], device=x0.device)
        vertex[own] = (QUAD_C0 + step).clamp(*QUAD_RANGE)
    lap()
    dist.all_reduce(vertex)
    del streams, x0
    shift = (vertex - QUAD_C0).tolist()
    def show(s):
        for p, d in zip(parts, shift):
            for i in members[p]:
                put(i, QUAD_C0 + s * d)
    losses = []
    for s in QUAD_SHRINKS:
        show(s)
        losses.append(scorer.score(HOLDOUT))
    losses = torch.stack(losses).tolist()
    best = min(range(len(QUAD_SHRINKS)), key=lambda k: losses[k])
    show(QUAD_SHRINKS[best])
    lap()
    laps = [round(b - a, 4) for a, b in zip(clock, clock[1:])]
    return dict(quad_laps=dict(zip(('setup', 'center', 'probes', 'shrink'), laps)),quad_shrink=QUAD_SHRINKS[best], quad_holdout=dict(zip(map(str, QUAD_SHRINKS), losses)),
                holdout_merged=losses[best],
                quad_vertices={p: round(QUAD_C0 + d, 4) for p, d in zip(parts, shift)})


@triton.jit
def _side_features(side_ptr, offs, part, D: tl.constexpr, J: tl.constexpr, offs_j, scale):
    i = (part * J + offs_j) % (D // 2)
    x = tl.load(side_ptr + offs[:, None] * D + i[None, :]) * scale
    y = tl.load(side_ptr + offs[:, None] * D + (i + D // 2)[None, :]) * scale
    return x, y


@triton.jit
def _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, part, M, RAW_GPU,
    RAW_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    d = part * J + offs_j
    i = d % (D // 2)
    n = (row % BH).to(tl.int64) * T + offs
    raw = tl.load(raw_ptr + (n // M * RAW_GPU + n % M * RAW_STRIDE)[:, None] + d[None, :]).to(tl.float32)
    sq = tl.load(sq_ptr + row * T + offs)[:, None]
    steps = tl.load(steps_ptr + row * T + offs)[:, None]
    change = (2 * steps * raw + steps * steps) / D / (sq / D + 1e-6)
    a = libdevice.expm1(-0.5 * libdevice.log1p(change))
    b = steps * libdevice.rsqrt(sq / D + 1e-6) * (1 + a)
    cos = tl.load(cos_ptr + offs[:, None] * (D // 2) + i[None, :])
    sin = tl.load(sin_ptr + offs[:, None] * (D // 2) + i[None, :])
    first = (d < D // 2)[None, :]
    return a, b * tl.where(first, cos, sin), b * tl.where(first, -sin, cos)


@triton.jit
def _store_by_token(out_ptr, x, row, BH, T, offs, part, M, W_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    n = (row % BH).to(tl.int64) * T + offs
    tl.store(out_ptr + (n // M * W_STRIDE + n % M * D + part * J)[:, None] + offs_j[None, :], x)


@triton.jit
def _split4(acc, BLOCK: tl.constexpr, J: tl.constexpr):
    first, second = tl.split(tl.reshape(acc, (BLOCK, J, 2, 2)))
    x, y = tl.split(first)
    xx, yy = tl.split(second)
    return x, y, xx, yy


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
    RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, WINDOW: tl.constexpr):
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S
    row, part = i // REPEAT, i % REPEAT
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_s = s0 + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    lse = tl.load(lse_ptr + row * T + offs_s)
    hub_out = tl.load(hubout_ptr + row * T + offs_s)
    side = side_ptr + base
    n4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    d4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    nxy = tl.zeros((BLOCK_S, J), tl.float32)
    dxy = tl.zeros((BLOCK_S, J), tl.float32)
    l2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    m2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    s1n = tl.zeros((BLOCK_S,), tl.float32)
    s2n = tl.zeros((BLOCK_S,), tl.float32)
    s1d = tl.zeros((BLOCK_S,), tl.float32)
    s2d = tl.zeros((BLOCK_S,), tl.float32)
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    for it in range(tl.maximum(s0 - WINDOW, 0) // BLOCK_T, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= WINDOW), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
        c = tl.dot(hub, tl.trans(v)) - hub_out[:, None]
        x, y = _side_features(side, offs_t, part, D, J, offs_j, scale)
        f2 = tl.reshape(tl.join(x, y), (BLOCK_T, 2 * J)).to(tl.bfloat16)
        f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BLOCK_T, 4 * J)).to(tl.bfloat16)
        fxy = (x * y).to(tl.bfloat16)
        wn = a * c
        wnl = (wn * logits).to(tl.bfloat16)
        al = (a * logits).to(tl.bfloat16)
        wnb, ab = wn.to(tl.bfloat16), a.to(tl.bfloat16)
        n4 += tl.dot(wnb, f4)
        nxy += tl.dot(wnb, fxy)
        l2 += tl.dot(wnl, f2)
        d4 += tl.dot(ab, f4)
        dxy += tl.dot(ab, fxy)
        m2 += tl.dot(al, f2)
        s1n += tl.sum(wn * logits, axis=1)
        s2n += tl.sum(wn * logits * logits, axis=1)
        s1d += tl.sum(a * logits, axis=1)
        s2d += tl.sum(a * logits * logits, axis=1)
    ca, bu, bw = _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_s, part, M,
      RAW_GPU, RAW_STRIDE, D, J, offs_j)
    nx0, ny0, nx1, ny1 = _split4(n4, BLOCK_S, J)
    dx0, dy0, dx1, dy1 = _split4(d4, BLOCK_S, J)
    lx0, ly0 = tl.split(tl.reshape(l2, (BLOCK_S, J, 2)))
    mx0, my0 = tl.split(tl.reshape(m2, (BLOCK_S, J, 2)))
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
      + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
      + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_s)
    _store_by_token(out_ptr, zn / (1.0 + zd) / steps[:, None], row, BH, T, offs_s, part, M, W_STRIDE, D, J, offs_j)


def moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window=None):
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_queries_kernel[(rows * repeat, t // 64)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps,
      out, rows // blocks, t, d ** -0.5, repeat,
      raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
      out.stride(1), out.stride(0), D=d, J=d // repeat,
      BLOCK_T=32, BLOCK_S=64, WINDOW=t if window is None else min(window, t), num_warps=4, num_stages=3)


@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
    steps_ptr, out_ptr, value_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
    RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    row, part = i // REPEAT, i % REPEAT
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    side = side_ptr + base
    p2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    q2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    qxy = tl.zeros((BLOCK_T, J), tl.float32)
    r2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    val = tl.zeros((BLOCK_T, J), tl.float32)
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    first = t0 // BLOCK_S
    last = T // BLOCK_S
    for it in range(first, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub_part = tl.load(hub_ptr + base + offs_s[:, None] * D + part * J + offs_j[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where(lag >= 0, tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        x, y = _side_features(side, offs_s, part, D, J, offs_j, scale)
        w1 = a * c
        w2 = (0.5 * a - a * a) * c
        w1b, w2b, w2lb = w1.to(tl.bfloat16), w2.to(tl.bfloat16), (w2 * logits).to(tl.bfloat16)
        f2 = tl.reshape(tl.join(x, y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        g2 = tl.reshape(tl.join(x * x, y * y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        val += tl.dot(a.to(tl.bfloat16), hub_part)
        p2 += tl.dot(w1b, f2)
        q2 += tl.dot(w2b, g2)
        qxy += tl.dot(w2b, (x * y).to(tl.bfloat16))
        r2 += tl.dot(w2lb, f2)
        s1 += tl.sum(w1 * logits, axis=1)
        s2 += tl.sum(w2 * logits * logits, axis=1)
    ca, bu, bw = _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_t, part, M,
      RAW_GPU, RAW_STRIDE, D, J, offs_j)
    px, py = tl.split(tl.reshape(p2, (BLOCK_T, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BLOCK_T, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BLOCK_T, J, 2)))
    z = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + qxy * (2.0 * bu * bw) + 2.0 * ca * (rx * bu + ry * bw)
      + s1[:, None] * ca + s2[:, None] * (ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_t)
    _store_by_token(out_ptr, z / steps[:, None], row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)
    _store_by_token(value_ptr + blk.to(tl.int64) * OUT_BLK, val, row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)


def moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, repeat):
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_keys_kernel[(rows * repeat, t // 64)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
      values, rows // blocks, t, d ** -0.5, repeat,
      raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
      out.stride(1), out.stride(0), D=d, J=d // repeat,
      BLOCK_T=64, BLOCK_S=32, num_warps=4, num_stages=3)


F32, BF16 = torch.float32, torch.bfloat16
BATCH, SEQ = 8, 2048
ID_SEQ_LEN = 1024  # identity sequence length: each 2049-token row trains as two 1024-token sequences
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8
LR_WTE, LR_VE, LR_LAMBDAS = 1.8, 1.8, 0.006
ANVIL_LR, ANVIL_WD = 0.04, 2.25
ANVIL_DECAY = 0.09  # identity-phase ANVIL decay (lr x wd for this rate)
GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1
ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240)  # #44's ANVIL momentum warmup: the first 240 of 1194 updates (see anvil_betas)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_LAG, START_RHO, START_CONTENT = 1, 1, 0.9, 0.25


def id_has_ve(layer):
    return layer % 2 == (L - 1) % 2


VE_BLOCKS = tuple(i for i in range(L) if id_has_ve(i))


def id_norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)


def softcap(x):
    return SOFTCAP * torch.tanh(x / SOFTCAP)


def id_hadamard(d, device):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


@torch.no_grad()
def pair_value_output(model):
    # Each attn.c_proj becomes c_v^T at the random c_proj's Frobenius norm (the second rescale absorbs rounding).
    for i in range(L):
        value = model.w(i, 'attn.c_v')
        output = model.w(i, 'attn.c_proj')
        original_norm = output.norm()
        paired = value.T * (original_norm / value.norm())
        output.copy_(paired * (original_norm / paired.norm()))


SHAPING_UPDATES = 195  # the four update-shaping mechanisms below act on identity updates 0..194
SPLIT_SHAPING = True  # the identity phase as two graphs: with the shaping mechanisms for updates 0..194, without after
SHAPING_ON = True  # (set while each graph is traced: False drops the mechanisms, a no-op from update 195 on)
VE_COMMON_GAIN = 4.0


def amplify_common_motion(delta):
    # Amplify the vocabulary-common part of a value-embedding Adam update VE_COMMON_GAIN-fold at fixed energy.
    common = delta.mean(dim=0, keepdim=True)
    amplified = delta + (VE_COMMON_GAIN - 1.0) * common
    original_energy = delta.square().sum()
    amplified_energy = amplified.square().sum()
    return amplified * torch.sqrt(original_energy / amplified_energy)


WTE_GATE_ENERGY = 0.00390625  # 1/256 of each wte row's energy in its GATE_CHANNELS gate-input channels


def constrain_wte_gate_energy(wte, step_t):
    if not SHAPING_ON:
        return
    x = wte.float()
    sector = torch.arange(D, device=x.device) < GATE_CHANNELS
    inside = torch.where(sector, x.square(), 0.0).sum(-1, keepdim=True)
    outside = torch.where(sector, 0.0, x.square()).sum(-1, keepdim=True)
    total = inside + outside
    inside_scale = torch.sqrt(WTE_GATE_ENERGY * total / inside)
    outside_scale = torch.sqrt((1.0 - WTE_GATE_ENERGY) * total / outside)
    projected = x * torch.where(sector, inside_scale, outside_scale)
    wte.copy_(torch.where(step_t < SHAPING_UPDATES, projected, x).to(wte.dtype))


constrain_wte_gate_energy_compiled = torch.compile(constrain_wte_gate_energy, dynamic=False)


@torch.compile(dynamic=False)
def pool_gate_credits(grads, step_t):
    # Acts on the all-reduced gate credits, which feed only Adam: each VE gate's head-mean credit becomes the mean
    # over the four VE layers.
    if not SHAPING_ON:
        return
    common = torch.stack(grads).mean(dim=(0, 2))[:, None]
    for grad in grads:
        pooled = grad - grad.mean(dim=1, keepdim=True) + common
        grad.copy_(torch.where(step_t < SHAPING_UPDATES, pooled, grad))


VE_PARTNER_GAIN = 0.5
VE_PARTNERS = (2, 3, 0, 1)  # each VE layer's partner by index in VE_BLOCKS (1, 3, 5, 7): 1<->5, 3<->7
ID_SHAPING_GRAM = True  # amplify_common_motion + transfer_partner_motion in closed form (shape_values_closed)


def shape_values_closed(deltas, frames, step_t):
    """amplify_common_motion per table then transfer_partner_motion, the same values up to rounding, in closed form.
    With mu the column mean and c = delta - mu (rows: vocabulary), amplification by g = VE_COMMON_GAIN is
    s * (c + g mu) with s = sqrt(|d|^2 / (|d|^2 + (g^2 - 1) V |mu|^2)): its common part is g s mu, its centered part
    s c (energy e = s^2 |c|^2). The response's normalization cancels in the transfer's rescale to e, so the borrowed
    motion is sqrt(e) * p / |p| with p the partner's centered delta through W = F_partner F^T (its row mean is zero up
    to rounding, as is the mix's, so neither is subtracted), and the mix's energy is 1.25 e + s b <c, p> (b = sqrt(e) / |p|): one GEMM, no materialized intermediates."""
    v = deltas.shape[1]
    perm = list(VE_PARTNERS)
    mu = deltas.mean(dim=1, keepdim=True)
    total = deltas.square().sum(dim=(1, 2), keepdim=True)
    mu_sq = v * mu.square().sum(dim=(1, 2), keepdim=True)
    gain = VE_COMMON_GAIN
    s = torch.sqrt(total / (total + (gain * gain - 1.0) * mu_sq))
    centered = deltas - mu
    energy = s * s * centered.square().sum(dim=(1, 2), keepdim=True)
    to_partner = torch.bmm(frames, frames[perm].transpose(1, 2))  # [j] = F_j F_p(j)^T: table j's delta to p(j)
    # sent[j] = c_j F_j F_p(j)^T, the borrowed direction of table p(j) (row mean 0: c_j is centered over rows)
    sent = torch.bmm(centered, to_partner)
    b = torch.sqrt(energy / sent.square().sum(dim=(1, 2), keepdim=True)[perm])
    borrowed = sent[perm]
    cross = (centered * borrowed).sum(dim=(1, 2), keepdim=True)
    mixed_energy = (1.0 + VE_PARTNER_GAIN ** 2) * energy + 2 * VE_PARTNER_GAIN * s * b * cross
    scale = torch.sqrt(energy / mixed_energy)
    candidate = gain * s * mu + scale * (s * centered + VE_PARTNER_GAIN * b * borrowed)
    return torch.where(step_t < SHAPING_UPDATES, candidate, deltas)


def transfer_partner_motion(deltas, frames, step_t):
    # Completed Adam motion only. No transformed tensor returns to a nonlinear probe.
    common = deltas.mean(dim=1, keepdim=True)
    centered = deltas - common
    energy = centered.square().sum(dim=(1, 2), keepdim=True)
    response = torch.bmm(centered, frames)
    response = response * torch.rsqrt(response.square().sum(dim=(1, 2), keepdim=True))
    # Each layer borrows its partner's motion (same SSSL window size), mapped through the partner's c_proj frame and
    # back through its own.
    donor = torch.stack([response[j] for j in VE_PARTNERS])
    borrowed = torch.bmm(donor, frames.transpose(1, 2))
    borrowed = borrowed - borrowed.mean(dim=1, keepdim=True)
    borrowed = borrowed * torch.sqrt(energy / borrowed.square().sum(dim=(1, 2), keepdim=True))
    mixed = centered + VE_PARTNER_GAIN * borrowed
    mixed = mixed - mixed.mean(dim=1, keepdim=True)
    mixed_energy = mixed.square().sum(dim=(1, 2), keepdim=True)
    mixed = mixed * torch.sqrt(energy / mixed_energy)
    candidate = common + mixed
    return torch.where(step_t < SHAPING_UPDATES, candidate, deltas)


class Model:
    def __init__(self, device, rank, world):
        self.device, self.rank, self.world = device, rank, world
        g = torch.Generator(device='cpu').manual_seed(SEED)
        s = math.sqrt(3.0) * D ** -0.5
        uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
        proj = lambda fan_in, fan_out: uni((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
        self.square = torch.stack([uni((D, D), s) if k != 'attn.c_proj' else proj(D, D)
        for k in SQUARE for _ in range(L)]).to(device)
        self.wide = torch.stack([uni((D, 4 * D), s) for _ in range(L)] + [proj(4 * D, D).T for _ in range(L)]).to(device)
        self.gates = {i: torch.zeros(GATE_CHANNELS, H, device=device) for i in range(L) if id_has_ve(i)}
        self.wte = torch.randn(VOCAB, D, generator=g).to(device, BF16)
        self.lm_head = (0.001 * torch.randn(D, VOCAB, generator=g)).to(device)
        self.ve = {i: uni((VOCAB, D), s).to(device, BF16) for i in range(L) if id_has_ve(i)}
        self.lam_r = torch.ones(L, device=device)
        self.lam_0 = torch.full((L,), 0.1, device=device)
        pair_value_output(self)
        constrain_wte_gate_energy(self.wte, torch.full((), -1.0, device=device))
        structured_start(self)
        shapes = dict(wte=self.wte.shape, lm_head=self.lm_head.shape,
        lam_r=(L,), lam_0=(L,), **{f'gate{i}': (GATE_CHANNELS, H) for i in self.gates},
        **{f've{i}': (VOCAB, D) for i in self.ve})
        sizes = {k: math.prod(v) for k, v in shapes.items()}
        self.flat_grad = torch.zeros(sum(sizes.values()), device=device)
        self.grad, offset = {}, 0
        for k, shape in shapes.items():
            self.grad[k] = self.flat_grad[offset:offset + sizes[k]].view(shape)
            offset += sizes[k]
        self.grad['square'] = torch.zeros(self.square.shape, device=device, dtype=BF16)
        self.grad['wide'] = torch.zeros(self.wide.shape, device=device, dtype=BF16)
        self.owned = {name: torch.zeros(w.shape[0] // world, *w.shape[1:], device=device, dtype=BF16)
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.adam = [(self.wte, self.grad['wte'], LR_WTE), (self.lm_head, self.grad['lm_head'], LR),
        (self.lam_r, self.grad['lam_r'], LR_LAMBDAS), (self.lam_0, self.grad['lam_0'], LR_LAMBDAS)]
        self.adam += [(self.gates[i], self.grad[f'gate{i}'], LR) for i in self.gates]
        self.adam += [(self.ve[i], self.grad[f've{i}'], LR_VE) for i in self.ve]
        self.m = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.v = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.anvil = {name: dict(fast=torch.zeros_like(w), slow=torch.zeros_like(w),
        energy=torch.zeros(w.shape[0], 1, w.shape[2], device=device))
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.H = id_hadamard(D, device)
        self.rows = torch.randint(0, VOCAB, (IDENTITY_UPDATES, BATCH, SEQ + 1), device=device)
        self.loss = torch.zeros(1, device=device)
        self.step_t = torch.zeros((), device=device)
        self.rates = dict(square=torch.ones(4 * L, device=device),
        wide=torch.tensor([2.0] * L + [1.0] * L, device=device))
        self.square_bf, self.wide_bf = self.square.to(BF16), self.wide.to(BF16)
        self.shard = {name: slice(rank * (w.shape[0] // world), (rank + 1) * (w.shape[0] // world))
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.lm_head_bf = self.lm_head.to(BF16)
        self.gates_bf = {i: g.to(BF16) for i, g in self.gates.items()}
        self.gather_group = dist.new_group(pg_options=nccl_options()) if ID_GATHER_GROUP else None
        # 1024-token sequences: every SSSL window (1024 or 2048 tokens) covers the whole causal context.
        self.mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, ID_SEQ_LEN, ID_SEQ_LEN, device=device)
        self.rope = rotary(ID_SEQ_LEN, device)
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
        self.lm_head_bf.copy_(self.lm_head)
        for i, g in self.gates.items():
            self.gates_bf[i].copy_(g)
    @torch.no_grad()
    def cast_all(self):
        self.square_bf.copy_(self.square)
        self.wide_bf.copy_(self.wide)
        self.refresh_bf16()
    @torch.no_grad()
    def gather_fp32(self):
        for name, w in (('square', self.square), ('wide', self.wide)):
            dist.all_gather_into_tensor(w, w[self.shard[name]].clone())
    def tensors(self):
        out = [self.square, self.wide, self.wte, self.lm_head, self.lam_r, self.lam_0, *self.gates.values(),
        *self.ve.values(), *self.m, *self.v]
        return out + [x for st in self.anvil.values() for x in st.values()]
    def state_dict(self):
        sd = {'transformer.wte.weight': self.wte.float(), 'lm_head.weight': self.lm_head.T,
        'resid_lambdas': self.lam_r, 'x0_lambdas': self.lam_0}
        for i in range(L):
            for kind in (*SQUARE, 'mlp.c_fc', 'mlp.c_proj'):
                sd[f'transformer.h.{i}.{kind}.weight'] = self.w(i, kind).T
            if id_has_ve(i):
                sd[f'transformer.h.{i}.attn.ve_gate.weight'] = self.gates[i].T
                sd[f'value_embeds.{i}.weight'] = self.ve[i].float()
        return {k: v.detach().float().contiguous() for k, v in sd.items()}


def rope_pair(lag, rho, generator):
    half = HD // 2
    phase = 2 * math.pi * torch.rand(half, generator=generator)
    amplitude = rho ** (torch.arange(half, dtype=F32) / 2)
    q = torch.cat([amplitude * phase.cos(), amplitude * phase.sin()])
    q = q * math.sqrt(HD) / q.norm()
    angle = lag / 10000 ** (torch.arange(0, HD, 2, dtype=F32) / HD)
    q1, q2 = q[:half], q[half:]
    return q, torch.cat([q1 * angle.cos() + q2 * angle.sin(), q2 * angle.cos() - q1 * angle.sin()])


def relu2_moments(mu, sd):
    z = mu / sd
    cdf, pdf = 0.5 * (1 + torch.erf(z / math.sqrt(2))), torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    m2 = (mu ** 2 + sd ** 2) * cdf + mu * sd * pdf
    m4 = (mu ** 4 + 6 * mu ** 2 * sd ** 2 + 3 * sd ** 4) * cdf + (mu ** 3 + 5 * mu * sd ** 2) * sd * pdf
    return m2, m4


@torch.no_grad()
def shared_mean(model, block):
    mean, energy, x0 = torch.zeros(D, device=model.device), float(D), 1.0
    for i in range(block + 1):
        r, lam = model.lam_r[i].item(), model.lam_0[i].item()
        energy = r * r * energy + lam * lam * D + 2 * r * lam * x0 * D
        mean, x0 = r * mean, r * x0 + lam
        normed = mean / math.sqrt(energy / D)
        if i == block:
            return normed
        attended = (normed @ model.w(i, 'attn.c_v')) @ model.w(i, 'attn.c_proj')
        energy, mean = energy + (attended @ attended).item() + 2 * (mean @ attended).item(), mean + attended
        shared = mean / math.sqrt(energy / D)
        fc, proj = model.w(i, 'mlp.c_fc'), model.w(i, 'mlp.c_proj')
        spread = fc.norm(dim=0) * max(1 - (shared @ shared).item() / D, 1e-3) ** 0.5
        m2, m4 = relu2_moments(shared @ fc, spread)
        written = m2 @ proj
        energy += (written @ written).item() + (proj.square().sum(1) @ (m4 - m2 * m2)).item()
        energy, mean = energy + 2 * (mean @ written).item(), mean + written


@torch.no_grad()
def structured_start(model):
    generator = torch.Generator(device='cpu').manual_seed(SEED + 31)
    mean = shared_mean(model, START_BLOCK)
    for h in range(H):
        cols = slice(h * HD, (h + 1) * HD)
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAG, START_RHO, generator)):
            w = model.w(START_BLOCK, kind)
            w[:, cols] = START_CONTENT * w[:, cols] + torch.outer(mean, v.to(mean.device)) / (mean @ mean)


def rotary(t, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, HD, 2, dtype=F32, device=device) / HD)
    freqs = torch.outer(torch.arange(t, dtype=F32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()


def rope_norm(raw, cos, sin):
    x = raw.float()
    c, s_ = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * c + x2 * s_, x1 * -s_ + x2 * c], -1)
    return (rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)).to(BF16).transpose(1, 2).contiguous()


# the identity forward's flex attention kernel_options (None: the defaults; BLOCK_N 128 as the default keeps it bitwise)
ID_FLEX_TILES = dict(BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=3)
ID_COMBO = True  # the identity forward compiled with inductor's combo kernels (horizontal fusion of small kernels)


def forward_impl(model, idx, targets, cos, sin):
    b, t = idx.shape
    cache = {}
    x = id_norm(model.wte[idx])
    x0, pending = x, None
    cache['x0'] = x0
    steps = []
    for i in range(L):
        mixed = model.lam_r[i] * (x if pending is None else x + pending.float()) + model.lam_0[i] * x0
        n1 = id_norm(mixed).to(BF16)
        q, k, v = torch.bmm(n1.reshape(1, b * t, D).expand(3, -1, -1), model.square_bf[i:3 * L:L]).view(3, b, t, D)
        values = v.view(b, t, H, HD)
        if id_has_ve(i):
            ve = model.ve[i][idx]
            z = n1[..., :GATE_CHANNELS] @ model.gates_bf[i]
            values = values + (2 * torch.sigmoid(z.float())).unsqueeze(-1).to(BF16) * ve.view(b, t, H, HD)
            cache[f'h{i}.ve_gate'], cache[f'h{i}.ve'] = z, ve
        qp, kp = rope_norm(q.view(b, t, H, HD), cos, sin), rope_norm(k.view(b, t, H, HD), cos, sin)
        o, aux = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.mask,
        return_aux=AuxRequest(lse=True), kernel_options=ID_FLEX_TILES)
        lse = aux.lse
        o = o.transpose(1, 2).reshape(b, t, D)
        cache.update({f'h{i}.in.attn': n1, f'h{i}.q': q, f'h{i}.k': k, f'h{i}.values': values,
        f'h{i}.lse': lse.float(), f'h{i}.in.attn.c_proj': o})
        x = mixed + (o @ model.wb(i, 'attn.c_proj')).float()
        n2 = id_norm(x).to(BF16)
        hidden = n2 @ model.wb(i, 'mlp.c_fc')
        act = F.relu(hidden).square()
        pending = act @ model.wb(i, 'mlp.c_proj')
        cache.update({f'h{i}.in.mlp.c_fc': n2, f'h{i}.mlp.c_fc': hidden, f'h{i}.in.mlp.c_proj': act,
        f'h{i}.stream': x + pending.float()})
        if ID_HIDDEN_FUSED:  # hidden_slope_'s per-token step
            u = hidden.reshape(-1, hidden.shape[-1]).float()
            steps.append(FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1)), min=1e-6))
    if ID_HIDDEN_FUSED:
        cache['fc_steps'] = torch.stack(steps)
    final = id_norm(x + pending.float()).to(BF16).reshape(b * t, D)
    raw = final @ model.lm_head_bf
    cache['in.lm_head'], cache['lm_head'] = final, raw
    loss = F.cross_entropy(softcap(raw.float()), targets.reshape(-1))
    return cache, loss


forward = torch.compile(forward_impl, dynamic=False, options={'combo_kernels': True} if ID_COMBO else {})


@torch.compile(dynamic=False)
def coordinate_change(raw, targets):
    raw = raw.float()
    z = softcap(raw)
    up, down = softcap(raw + HEAD_SIGMA), softcap(raw - HEAD_SIGMA)
    weight = torch.softmax(z, -1)
    change = torch.log1p(weight * torch.expm1(up - z)) - torch.log1p(weight * torch.expm1(down - z))
    change = change - F.one_hot(targets, VOCAB).float() * (up - down)
    return change / (2 * HEAD_SIGMA)


def step_signs(step_t, device):
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
    fast = torch.stack([fz * g2, fzz * g3, f * g2, fz * g3, f * g3]).to(BF16)
    return f * g1, fast, rows, total, steps, inv, inv0, delta, sigma


FUSED_CYCLE = True  # top_credit's readout-cycle weights in one pass over the logits (_cycle_weights_kernel)


@torch.compile(dynamic=False)
def cycle_scales(x, h):
    """cycle_weights' per-token step, norm inverses and their ratios."""
    steps = SIGMA * torch.clamp(torch.sqrt(x.square().mean(-1, keepdim=True)), min=1e-6)
    sq = x.square().sum(-1, keepdim=True)
    dots = x @ h.T
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * D) / D + EPS)
    inv0 = torch.rsqrt(sq / D + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv
    return steps, inv, inv0, delta, sigma


@triton.jit
def _cycle_weights_kernel(Base, Inv0, FG1, Fast, Rows, Total, T: tl.constexpr, V: tl.constexpr,
    BLOCK: tl.constexpr, CAP: tl.constexpr):
    """cycle_weights' per-logit part for one token in one pass: f g1 (fp32), the five bf16 weight rows, and the row
    sums (total, and rows' three)."""
    token = tl.program_id(0).to(tl.int64)
    inv0 = tl.load(Inv0 + token)
    total = tl.zeros((BLOCK,), tl.float32)
    r1 = tl.zeros((BLOCK,), tl.float32)
    r2 = tl.zeros((BLOCK,), tl.float32)
    r3 = tl.zeros((BLOCK,), tl.float32)
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        z = tl.load(Base + token * V + cols) * inv0
        t = libdevice.tanh(z / CAP)
        f = libdevice.exp(CAP * t)
        g1 = 1 - t * t
        d1 = -2 * t * g1 / CAP
        d2 = -2 * g1 * (1 - 3 * t * t) / (CAP * CAP)
        g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2
        fz, fzz = f * z, f * z * z
        total += f
        r1 += fz * g1
        r2 += fzz * g2
        r3 += fzz * z * g3
        out = token * V + cols
        tl.store(FG1 + out, f * g1)
        tl.store(Fast + out, (fz * g2).to(tl.bfloat16))
        tl.store(Fast + T * V + out, (fzz * g3).to(tl.bfloat16))
        tl.store(Fast + 2 * T * V + out, (f * g2).to(tl.bfloat16))
        tl.store(Fast + 3 * T * V + out, (fz * g3).to(tl.bfloat16))
        tl.store(Fast + 4 * T * V + out, (f * g3).to(tl.bfloat16))
    tl.store(Total + token, tl.sum(total, 0))
    tl.store(Rows + token * 3, tl.sum(r1, 0))
    tl.store(Rows + token * 3 + 1, tl.sum(r2, 0))
    tl.store(Rows + token * 3 + 2, tl.sum(r3, 0))


def cycle_weights_fused(x, base, h):
    """cycle_weights (same outputs) with its per-logit part in one kernel pass."""
    steps, inv, inv0, delta, sigma = cycle_scales(x, h)
    tokens, vocab = base.shape
    fg1 = torch.empty_like(base)
    fast = torch.empty((5, tokens, vocab), device=base.device, dtype=BF16)
    rows = torch.empty((tokens, 3), device=base.device)
    total = torch.empty((tokens, 1), device=base.device)
    _cycle_weights_kernel[(tokens,)](base.contiguous(), inv0.contiguous(), fg1, fast, rows, total, T=tokens, V=vocab,
                                     BLOCK=1024, CAP=SOFTCAP, num_warps=4)
    return fg1, fast, rows, total, steps, inv, inv0, delta, sigma


@torch.compile(dynamic=False)
def cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h):
    fast = [x.float() for x in prods]
    ratio = (sigma * exact + delta * sigma * fast[0] + 0.5 * delta ** 2 * sigma * fast[1] + 0.5 * sigma ** 2 * fast[2]
      + 0.5 * delta * sigma ** 2 * fast[3] + sigma ** 3 / 6 * fast[4]
      + delta * rows[:, :1] + 0.5 * delta ** 2 * rows[:, 1:2] + delta ** 3 / 6 * rows[:, 2:])
    clean_partition = torch.log(total)
    base_y = base.gather(1, y[:, None])
    dirs_y = c.T[y]
    clean = clean_partition - softcap(base_y * inv0)
    moved = clean_partition + torch.log1p(ratio / total) - softcap((base_y + steps * dirs_y) * inv)
    return ((moved - clean) / steps) @ h / D


@torch.compile(dynamic=False)
def cycle_powers(c):
    return c.to(BF16), (c * c).to(BF16), (c * c * c).to(BF16)


def top_credit(model, x, y):
    h = model.H * step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head
    base = x @ model.lm_head
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = (cycle_weights_fused if FUSED_CYCLE else cycle_weights)(
        x, base, h)
    exact = fg1 @ c.T
    cb, c2, c3 = cycle_powers(c)
    if FUSED_CYCLE:  # the products sharing a right operand as one GEMM (stacked along tokens; bitwise the same)
        n = x.shape[0]
        p01 = (fast[:2].reshape(2 * n, -1) @ cb.T).view(2, n, -1)
        p23 = (fast[2:4].reshape(2 * n, -1) @ c2.T).view(2, n, -1)
        prods = [p01[0], p01[1], p23[0], p23[1], fast[4] @ c3.T]
    else:
        prods = [fast[0] @ cb.T, fast[1] @ cb.T, fast[2] @ c2.T, fast[3] @ c2.T, fast[4] @ c3.T]
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def rope32(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


REPEAT, REPEAT_K = HD // 16, HD // 32
PACK = 5 * HD + 2


@torch.compile(dynamic=False)
def pack_heads(qs, ks, vals, outs, targets, lse_bits):
    parts = []
    for q, k, v, o, tg, lb in zip(qs, ks, vals, outs, targets, lse_bits):
        m = q.shape[0]
        bits = lb.view(m, H, 2)
        parts.append(torch.cat([q.view(m, H, HD), k.view(m, H, HD), v.view(m, H, HD), o.view(m, H, HD),
        tg.view(m, H, HD), bits], -1))
    return torch.stack(parts).permute(2, 0, 1, 3).contiguous()


PACK_KERNEL = True  # identity_grad's all_to_all send buffer packed by _pack_heads_kernel (one pass, bitwise pack_heads)


@triton.jit
def _pack_heads_kernel(Q, K, V, O, TG, LSE, Out16, Out32, layer, sq, sk, sv, so, st, L: tl.constexpr,
                       M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr):
    """One layer's rows of pack_heads' (H, L, M, PACK) output for BM tokens of head h: the five 64-wide pieces and
    the lse's fp32 bits (LSE (rows, H, T) fp32 as int32), copied as raw 16-bit words."""
    tile, h = tl.program_id(0), tl.program_id(1)
    tok = (tile * BM + tl.arange(0, BM)).to(tl.int64)
    d = tl.arange(0, 64)
    row = (h * L + layer) * M + tok
    dst = Out16 + row[:, None] * PACK + d[None, :]
    col = h * 64 + d[None, :]
    tl.store(dst, tl.load(Q + tok[:, None] * sq + col))
    tl.store(dst + 64, tl.load(K + tok[:, None] * sk + col))
    tl.store(dst + 128, tl.load(V + tok[:, None] * sv + col))
    tl.store(dst + 192, tl.load(O + tok[:, None] * so + col))
    tl.store(dst + 256, tl.load(TG + tok[:, None] * st + col))
    lse = tl.load(LSE + ((tok // T) * H + h) * T + tok % T)
    tl.store(Out32 + row * (PACK // 2) + 5 * 32, lse)


def pack_heads_kernel(qs, ks, vals, outs, targets, lses):
    """pack_heads (bitwise) from each layer's (tokens, D) pieces and its lse (rows, H, t), one launch per layer."""
    n, t = qs[0].shape[0], lses[0].shape[-1]
    out = torch.empty(H, len(qs), n, PACK, device=qs[0].device, dtype=BF16)
    o16, o32 = out.view(torch.int16), out.view(torch.int32)
    for i, (q, k, v, o, tg, lse) in enumerate(zip(qs, ks, vals, outs, targets, lses)):
        assert all(z.stride(1) == 1 for z in (q, k, v, o, tg)) and n % 16 == 0
        _pack_heads_kernel[(n // 16, H)](q.view(torch.int16), k.view(torch.int16), v.view(torch.int16),
                                         o.view(torch.int16), tg.view(torch.int16), lse.contiguous().view(torch.int32),
                                         o16, o32, i, q.stride(0), k.stride(0), v.stride(0), o.stride(0), tg.stride(0),
                                         L=len(qs), M=n, T=t, H=H, PACK=PACK, BM=16, num_warps=1)
    return out


@torch.compile(dynamic=False)
def head_prep(got, lse, cos, sin, b: int, t: int):
    k = got.shape[1]
    whole, lse = got.transpose(0, 1).reshape(k * b, t, PACK), lse.transpose(0, 1).reshape(k * b, t)
    q_raw, k_raw, values, out, hub = (whole[..., j * HD:(j + 1) * HD].float() for j in range(5))
    q1, k1 = rope32(q_raw, cos, sin), rope32(k_raw, cos, sin)
    sq_q, sq_k = (q_raw * q_raw).sum(-1), (k_raw * k_raw).sum(-1)
    steps = lambda sq: KEY_SIGMA * torch.clamp(torch.sqrt(sq / HD), min=1e-6)
    return (q1.to(BF16), k1.to(BF16), values.to(BF16), hub.to(BF16), q1, k1, lse.contiguous(), (hub * out).sum(-1),
      sq_q, sq_k, steps(sq_q), steps(sq_k))


FUSED_HEAD_PREP = True  # id_head_scores' head_prep in one Triton pass (_head_prep_kernel)
ID_PAIR_PREP = True  # with FUSED_HEAD_PREP and FAST_ID_MOMENTS: the query moments' key pair operands made by head_prep


@triton.jit
def _head_prep_kernel(Got, Got32, Cos, Sin, Q16, K16, V16, Hub16, Q32, K32, LSE, HubOut, SqQ, SqK, StepsQ, StepsK,
                      W: tl.constexpr, L: tl.constexpr, N: tl.constexpr, T: tl.constexpr, PACK: tl.constexpr,
                      SIGMA: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr, F2=None, F4=None, XY=None,
                      PAIRS: tl.constexpr = False):
    """head_prep for BM tokens of rank w's layer l (got (W, L, N, PACK)): output token (l * W + w) * N + i."""
    tile, wl = tl.program_id(0), tl.program_id(1)
    w, layer = wl % W, wl // W
    i = tile * BM + tl.arange(0, BM)
    src = ((w * L + layer) * N + i).to(tl.int64)
    dst = ((layer * W + w) * N + i).to(tl.int64)
    pos = i % T
    half = tl.arange(0, 32)
    dims = tl.arange(0, 64)
    c = tl.load(Cos + pos[:, None] * 32 + half[None, :])
    s = tl.load(Sin + pos[:, None] * 32 + half[None, :])
    row = Got + src[:, None] * PACK
    for j in tl.static_range(2):
        x1 = tl.load(row + j * 64 + half[None, :]).to(tl.float32)
        x2 = tl.load(row + j * 64 + 32 + half[None, :]).to(tl.float32)
        r1 = x1 * c + x2 * s
        r2 = x1 * -s + x2 * c
        inv = tl.rsqrt((tl.sum(r1 * r1, 1) + tl.sum(r2 * r2, 1)) / 64 + EPS)
        r1, r2 = r1 * inv[:, None], r2 * inv[:, None]
        sq = tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)
        out16 = (Q16 if j == 0 else K16) + dst[:, None] * 64 + half[None, :]
        out32 = (Q32 if j == 0 else K32) + dst[:, None] * 64 + half[None, :]
        tl.store(out16, r1.to(tl.bfloat16))
        tl.store(out16 + 32, r2.to(tl.bfloat16))
        if PAIRS and j == 1:  # moment_kernels._pair_features of the fp32 keys, in place of storing them
            x, y = r1 * 64 ** (-0.5), r2 * 64 ** (-0.5)
            tl.store(F2 + dst[:, None] * 64 + dims[None, :], tl.reshape(tl.join(x, y), (BM, 64)).to(tl.bfloat16))
            tl.store(XY + dst[:, None] * 32 + half[None, :], (x * y).to(tl.bfloat16))
            f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BM, 128))
            tl.store(F4 + dst[:, None] * 128 + tl.arange(0, 128)[None, :], f4.to(tl.bfloat16))
        else:
            tl.store(out32, r1)
            tl.store(out32 + 32, r2)
        tl.store((SqQ if j == 0 else SqK) + dst, sq)
        tl.store((StepsQ if j == 0 else StepsK) + dst, SIGMA * tl.maximum(tl.sqrt(sq / 64), 1e-6))
    v = tl.load(row + 128 + dims[None, :])
    o = tl.load(row + 192 + dims[None, :]).to(tl.float32)
    hub = tl.load(row + 256 + dims[None, :])
    tl.store(V16 + dst[:, None] * 64 + dims[None, :], v)
    tl.store(Hub16 + dst[:, None] * 64 + dims[None, :], hub)
    tl.store(HubOut + dst, tl.sum(hub.to(tl.float32) * o, 1))
    tl.store(LSE + dst, tl.load(Got32 + src * (PACK // 2) + 160))


def head_prep_fused(got, cos, sin, b: int, t: int):
    """head_prep's outputs (the same up to summation order) from got in one kernel pass."""
    w, layers, n, _ = got.shape
    rows = layers * b
    new = lambda *s, dtype=torch.float32: torch.empty(*s, device=got.device, dtype=dtype)
    q16, k16, v16, hub16 = (new(rows, t, HD, dtype=BF16) for _ in range(4))
    q32, k32 = new(rows, t, HD), new(rows, t, HD)
    lse, hub_out, sq_q, sq_k, steps_q, steps_k = (new(rows, t) for _ in range(6))
    pairs = (new(rows, t, HD, dtype=BF16), new(rows, t, 2 * HD, dtype=BF16), new(rows, t, HD // 2, dtype=BF16)) \
        if ID_PAIR_PREP else (None, None, None)
    _head_prep_kernel[(n // 32, w * layers)](got, got.view(torch.float32), cos, sin, q16, k16, v16, hub16, q32, k32,
                                             lse, hub_out, sq_q, sq_k, steps_q, steps_k, W=w, L=layers, N=n, T=t,
                                             PACK=PACK, SIGMA=KEY_SIGMA, EPS=NORM_EPS, BM=32, F2=pairs[0],
                                             F4=pairs[1], XY=pairs[2], PAIRS=ID_PAIR_PREP, num_warps=4)
    if ID_PAIR_PREP:
        return q16, k16, v16, hub16, q32, pairs, lse, hub_out, sq_q, sq_k, steps_q, steps_k
    return q16, k16, v16, hub16, q32, k32, lse, hub_out, sq_q, sq_k, steps_q, steps_k


def id_head_scores(got, rope32, back, b, t):
    if FUSED_HEAD_PREP:
        q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep_fused(got, *rope32, b, t)
    else:
        lse_bits = got[..., 5 * HD:].view(F32)[..., 0]
        q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep(got, lse_bits, *rope32,
                                                                                              b, t)
    queries = moment_kernels.moment_query_scores if FAST_ID_MOMENTS else moment_query_scores
    keys = moment_kernels.moment_key_scores if FAST_ID_MOMENTS else moment_key_scores
    if ID_PAIR_PREP:  # k1f holds the keys' pair operands (head_prep_fused wrote them in place of the fp32 keys)
        moment_kernels._moment_query_scores(q1, k1, values, hub, got, sq_q, None, *rope32, lse, hub_out, steps_q,
                                            back[:, :, 0], REPEAT, pair_ops=k1f)
    else:
        queries(q1, k1, values, hub, got, sq_q, k1f, *rope32, lse, hub_out, steps_q, back[:, :, 0], REPEAT)
    keys(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k, back[:, :, 1], back[:, :, 2],
         REPEAT_K)


FAST_ID_MOMENTS = True  # the identity phase's query / key moment kernels by moment_kernels (each pair's sums once)


@torch.compile(dynamic=False)
def credit_stacks(got, ve_blocks: tuple):
    world, _, _, m, hd = got.shape
    by_token = got.permute(1, 2, 3, 0, 4).reshape(L, 3, m, world * hd)
    stacks = by_token.transpose(0, 1).contiguous()
    return stacks, [by_token[i, 2].view(m, H, HD).float() for i in ve_blocks]


def f32_stack(xs):
    return torch.stack([x.reshape(-1, x.shape[-1]) for x in xs]).float()


def bf16_stacks(n1, o, n2, act):
    st = lambda xs: torch.stack([x.reshape(-1, x.shape[-1]) for x in xs]).to(BF16)
    return st(n1), st(o), st(n2), st(act)


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


@torch.compile(dynamic=False)
def hidden_slope_(u, read):
    u = u.reshape(-1, u.shape[-1]).float()
    step = FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1, keepdim=True)), min=1e-6)
    relu2 = lambda z: F.relu(z).square()
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    read.copy_((slope * read.float()).to(BF16))


ID_HIDDEN_FUSED = True  # hidden_credits as one Triton GEMM with hidden_slope_ in its epilogue (id_kernels; fp32 order)


def hidden_credits(us, credb, w_out_t, steps=None):
    if ID_HIDDEN_FUSED:
        return id_kernels.hidden_credits_fused(us, steps, credb, w_out_t)
    hidden = torch.bmm(credb, w_out_t)
    for u, read in zip(us, hidden):
        hidden_slope_(u, read)
    return hidden


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


ID_SQUARE_DIRECT = True  # identity_grad's c_q / c_k / c_v gradients by id_kernels.square_credit_grads (fp32 order)
ID_STACKLESS = True  # with ID_SQUARE_DIRECT: no bf16_stacks; the gradient GEMMs by id_kernels.at_b_layers (fp32 order)


@torch.compile(dynamic=False)
def value_embed_credits(z, ve, vcredit, n1):
    z, ve, vcredit = z.float(), ve.float().view(-1, H, HD), vcredit.float()
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    gate = n1[:, :GATE_CHANNELS].float().T @ (slope * (vcredit * ve).sum(-1))
    return gate, (vcredit * 2 * torch.sigmoid(z)[..., None]).reshape(-1, D)


# identity_grad's collectives overlap: the wide gradients (hidden credits + two bmms) in heads_in's window (False) or in
# heads_out's (True); the order of the other independent work is fixed (numerically identical either way)
ID_WIDE_LATE = False
# ID_EARLY_WIDE: the wide gradients' reduce-scatter issued as soon as they are complete (in heads_in's window, where the
# collectives' stream idles), and their ANVIL update and all-gather on a side stream from there (exact: same
# kernels, same inputs; the step's last read of wide_bf, hidden_credits, precedes the fork).
# ID_GATHER_GROUP: the ANVIL all-gathers on a second communicator, so they do not queue behind the flat all-reduce
# (exact: copies).
ID_EARLY_WIDE = True
ID_GATHER_GROUP = True


def id_gather_group(model):
    return model.gather_group if ID_GATHER_GROUP else None


def early_wide_update(model, work, n):
    """ID_EARLY_WIDE: the wide matrices' ANVIL update and all-gather on a side stream once their reduce-scatter
    lands. Returns (gather work, its input, the side stream): train_step waits for the gather and joins the stream."""
    if not hasattr(model, 'wide_stream'):
        model.wide_stream = torch.cuda.Stream()
    side = model.wide_stream
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        work.wait()
        sl, st = model.shard['wide'], model.anvil['wide']
        new = anvil_math(model.wide[sl], model.owned['wide'], n, st['fast'][sl], st['slow'][sl], st['energy'][sl],
                         model.step_t, model.rates['wide'][sl])
        gather = dist.all_gather_into_tensor(model.wide_bf, new, group=id_gather_group(model), async_op=True)
    return gather, new, side


def identity_grad(model, idx, targets, rows_total):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens, one head's queries, keys and values over the batch. The ANVIL matrices' gradients stay this
    GPU's partial sums, reduce-scattered; the rest are all-reduced. Returns the clean loss of its tokens, the collectives
    in flight and the token count."""
    world = model.world
    t = idx.shape[1]
    n, m = rows_total * t, idx.numel()
    rope32 = model.rope
    cache, loss = forward(model, idx, targets, *rope32)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1), idx.reshape(-1)
    gsq, gwide = model.grad['square'], model.grad['wide']
    # heads_in's payload first (it needs the readout credit for targets_o); the work that does not feed it runs while
    # it is in flight (same calls as before, reordered: numerically identical)
    credit_top = top_credit(model, stream(f'h{L - 1}.stream'), y)
    cred, credb = block_credits(credit_top, model.lam_r)
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))
    if PACK_KERNEL:
        send = pack_heads_kernel([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
                                 [mine(cache[f'h{i}.values']) for i in range(L)],
                                 [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)],
                                 [targets_o[i] for i in range(L)], [cache[f'h{i}.lse'] for i in range(L)])
    else:
        send = pack_heads([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
        [mine(cache[f'h{i}.values']) for i in range(L)],
        [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)], [targets_o[i] for i in range(L)],
        [cache[f'h{i}.lse'].transpose(1, 2).reshape(-1, H).view(BF16)
        for i in range(L)])
    got = torch.empty_like(send)
    heads_in = dist.all_to_all_single(got, send, async_op=True)
    coords = coordinate_change(cache['lm_head'], y)
    prevs = f32_stack([cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)])
    if ID_STACKLESS:  # the layers' bf16 inputs read in place by id_kernels.at_b_layers / square_credit_grads
        assert ID_SQUARE_DIRECT
        n1s, os_, n2s, acts = ([mine(cache[f'h{i}.{key}']) for i in range(L)]
                               for key in ('in.attn', 'in.attn.c_proj', 'in.mlp.c_fc', 'in.mlp.c_proj'))
    else:
        n1s, os_, n2s, acts = bf16_stacks([cache[f'h{i}.in.attn'] for i in range(L)],
          [cache[f'h{i}.in.attn.c_proj'] for i in range(L)],
          [cache[f'h{i}.in.mlp.c_fc'] for i in range(L)],
          [cache[f'h{i}.in.mlp.c_proj'] for i in range(L)])
    hidden = lambda: hidden_credits([cache[f'h{i}.mlp.c_fc'] for i in range(L)], credb, model.wide_bf[L:],
                                    cache.get('fc_steps'))
    if ID_STACKLESS:
        wide = lambda: (id_kernels.at_b_layers(n2s, list(hidden()), gwide[:L]),
                        id_kernels.at_b_layers(list(credb), acts, gwide[L:]))
    else:
        wide = lambda: (torch.bmm(n2s.transpose(1, 2), hidden(), out=gwide[:L]),
                        torch.bmm(credb.transpose(1, 2), acts, out=gwide[L:]))
    if not ID_WIDE_LATE:
        wide()
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    works = {}
    if ID_EARLY_WIDE:
        if ID_WIDE_LATE:
            raise ValueError('ID_EARLY_WIDE needs the wide gradients in heads_in\'s window (ID_WIDE_LATE=False)')
        works['wide'] = dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True)
        works['wide_update'] = early_wide_update(model, works['wide'], n)
    heads_in.wait()
    back = torch.empty(world, L, 3, m, HD, device=idx.device, dtype=BF16)
    id_head_scores(got, rope32, back, rows_total, t)
    got2 = torch.empty_like(back)
    heads_out = dist.all_to_all_single(got2, back, async_op=True)
    if ID_WIDE_LATE:
        wide()
    # the wide gradients are complete: their reduce-scatter queues right behind heads_out
    if not ID_EARLY_WIDE:
        works['wide'] = dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True)
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    if ID_STACKLESS:
        id_kernels.at_b_layers(os_, list(credb), gsq[3 * L:])
    else:
        torch.bmm(os_.transpose(1, 2), credb, out=gsq[3 * L:])
    heads_out.wait()
    if ID_SQUARE_DIRECT:  # the c_q / c_k / c_v gradients read heads_out's buffer directly (no by-token copy)
        id_kernels.square_credit_grads(n1s, got2, gsq[:3 * L])
        vcredits = [got2[:, i, 2].permute(1, 0, 2) for i in VE_BLOCKS]
    else:
        by_token, vcredits = credit_stacks(got2, VE_BLOCKS)
        for j in range(3):
            torch.bmm(n1s.transpose(1, 2), by_token[j], out=gsq[j * L:(j + 1) * L])
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        gate, rows = value_embed_credits(mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(gate)
        model.grad[f've{i}'].zero_().index_add_(0, tokens, rows)
    works['flat'] = dist.all_reduce(model.flat_grad, async_op=True)
    return loss, works, n


def anvil_betas(t):
    # ANVIL_RUN's momentum warmup (0.85 -> 0.93 over 240 of 1194 updates) scaled to the 1831-update identity
    # schedule lasts 368 updates, so all 300 identity updates are on the ramp.
    warm = round(ANVIL_RUN[1] * IDENTITY_SCHEDULE / ANVIL_RUN[0])
    lo, hi = 0.85, 0.93
    if ID_BETA_HOLD:  # the ramp ends at hi (unclamped, beta passes 1 after ~740 updates: the momentum freezes)
        return torch.clamp(lo + (hi - lo) * t / warm, max=hi)
    return lo + (hi - lo) * t / warm


ID_BETA_HOLD = False  # identity ANVIL momentum held at 0.93 after its ramp (longer identity phases)
ID_DECAY = 150  # the identity phase's last ID_DECAY updates: its Adam and ANVIL rates fall linearly towards ID_DECAY_FINAL
ID_DECAY_FINAL = 0.3


def id_rate_scale(step_t):
    if not ID_DECAY:
        return 1.0
    left = (IDENTITY_UPDATES - step_t) / ID_DECAY
    return ID_DECAY_FINAL + (1 - ID_DECAY_FINAL) * torch.clamp(left, max=1.0)


@torch.compile(dynamic=False)
def anvil_math(w, g, n: int, fast, slow, energy, step_t, rates):
    beta = anvil_betas(step_t)
    g = g.float() / n
    fast_new = fast + (1 - beta) * (g - fast)
    slow_new = slow + (1 - 0.98) * (g - slow)
    x = g + beta * (fast_new - g)
    x = (x / (POLAR_MARGIN * torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-6)).to(BF16)
    for a, b_, c in ANVIL_MAPS:
        sq = x @ x.transpose(1, 2)
        x = a * x + (b_ * sq + c * (sq @ sq)) @ x
    update = x.float()
    energy_new = energy + (1 - 0.9) * (update.square().mean(1, keepdim=True) - energy)
    gained = update * torch.rsqrt(torch.clamp(energy_new, min=1e-10))
    update = gained * (torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
      / torch.clamp(torch.linalg.vector_norm(gained, dim=(1, 2), keepdim=True), min=1e-10))
    decay = ((slow_new * w) >= 0) * w * ANVIL_DECAY
    new = w - ANVIL_LR * id_rate_scale(step_t) * rates[:, None, None] * (update + decay)
    fast.copy_(fast_new)
    slow.copy_(slow_new)
    energy.copy_(energy_new)
    w.copy_(new)
    return new.to(BF16)


ID_CONCURRENT_ANVIL = True  # id_anvil_update's square-matrix update on a side stream under the wide one


def id_anvil_update(model, works, n):
    gathers = []
    main = torch.cuda.current_stream()
    side = None
    if ID_CONCURRENT_ANVIL:
        if not hasattr(model, 'side_stream'):
            model.side_stream = torch.cuda.Stream()
        side = model.side_stream
        side.wait_stream(main)
    for name, w, w_bf in (('square', model.square, model.square_bf), ('wide', model.wide, model.wide_bf)) if side \
            else (('wide', model.wide, model.wide_bf), ('square', model.square, model.square_bf)):
        if name == 'wide' and 'wide_update' in works:  # (ID_EARLY_WIDE: issued in identity_grad)
            continue
        with torch.cuda.stream(side) if side is not None and name == 'square' else contextlib.nullcontext():
            works[name].wait()
            sl, st = model.shard[name], model.anvil[name]
            new = anvil_math(w[sl], model.owned[name], n, st['fast'][sl], st['slow'][sl], st['energy'][sl],
                             model.step_t, model.rates[name][sl])
            gathers.append((dist.all_gather_into_tensor(w_bf, new, group=id_gather_group(model), async_op=True), new))
    if side is not None:
        main.wait_stream(side)
    if 'wide_update' in works:
        gather, new, wide_side = works['wide_update']
        gathers.append((gather, new))
        main.wait_stream(wide_side)
    return gathers


@torch.compile(dynamic=False)
def adam_math(ps, gs, n: int, ms, vs, lrs: tuple, step_t, frames):
    t = step_t + 1
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    value_deltas = []
    scale = id_rate_scale(step_t)
    for index, (p, g, m, v, lr) in enumerate(zip(ps, gs, ms, vs, lrs)):
        lr = lr * scale
        g = g / n
        if p.dtype == BF16:
            g = g.to(BF16).float()
        m_new = BETA1 * m + (1 - BETA1) * g
        v_new = BETA2 * v + (1 - BETA2) * g.square()
        if 8 <= index < 12:  # the value-embedding tables
            delta = lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)
            if SHAPING_ON and not ID_SHAPING_GRAM:
                amplified = amplify_common_motion(delta)
                delta = torch.where(step_t < SHAPING_UPDATES, amplified, delta)
            value_deltas.append(delta)
        else:
            p.copy_((p.float() - lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)
    if SHAPING_ON and ID_SHAPING_GRAM:
        transferred = shape_values_closed(torch.stack(value_deltas), frames, step_t)
    else:
        transferred = transfer_partner_motion(torch.stack(value_deltas), frames, step_t) if SHAPING_ON \
            else value_deltas
    for i in range(4):
        ps[8 + i].copy_((ps[8 + i].float() - transferred[i]).to(ps[8 + i].dtype))


def adam_update(model, works, n, frames):
    works['flat'].wait()
    ps, gs, lrs = zip(*model.adam)
    pool_gate_credits(list(gs[4:8]), model.step_t)
    adam_math(list(ps), list(gs), n, model.m, model.v, lrs, model.step_t, frames)


def train_step(model):
    world, rank = model.world, model.rank
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]
    inputs, targets = batch[:, :-1].reshape(-1, ID_SEQ_LEN), batch[:, 1:].reshape(-1, ID_SEQ_LEN)
    rows_total = inputs.shape[0]
    per = rows_total // world
    idx, tgt = inputs[rank * per:(rank + 1) * per], targets[rank * per:(rank + 1) * per]
    loss, works, n = identity_grad(model, idx, tgt, rows_total)
    # Capture gathered current output maps before ANVIL/all-gather overwrites square_bf.
    transfer_frames = model.square_bf[3 * L + 1:4 * L:2].float()
    gathers = id_anvil_update(model, works, n)
    adam_update(model, works, n, transfer_frames)
    constrain_wte_gate_energy_compiled(model.wte, model.step_t)
    for work, _ in gathers:
        work.wait()
    model.refresh_bf16()
    model.loss.copy_(loss.view(1))
    model.step_t.add_(1)


class IdentityPhase:
    """The identity-path phase: compiled and captured as one CUDA graph on random tokens, then reset to its fresh
    state. bind_rows() puts its batches on the GPU before the timer; run() replays the graph once per update."""
    def __init__(self, device, rank, world):
        self.order = np.random.default_rng((SEED + 1, 0)).permutation(TRAIN_ROWS)
        model = self.model = Model(device, rank, world)
        snapshot = [x.clone() for x in model.tensors()]
        # Compile and capture the identity step on the random tokens model.rows holds until bind_rows.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                train_step(model)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            train_step(model)
        self.plain_graph = None
        if SPLIT_SHAPING:  # the same step without the shaping mechanisms (no-ops from update SHAPING_UPDATES on)
            global SHAPING_ON
            SHAPING_ON = False
            try:
                with torch.cuda.stream(side):
                    for _ in range(2):
                        train_step(model)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.plain_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.plain_graph, pool=self.graph.pool()):
                    train_step(model)
            finally:
                SHAPING_ON = True
        # Restore the fresh state: weights, optimizer state, bf16 copies and the update counter.
        with torch.no_grad():
            for x, saved in zip(model.tensors(), snapshot):
                x.copy_(saved)
            model.cast_all()
            model.step_t.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    def bind_rows(self, rows):
        """Put the identity batches on the GPU: update i trains on rows order[BATCH * i:BATCH * (i + 1)], sorted by row
        index (the captured step reads model.rows[step_t])."""
        batches = rows[np.sort(self.order[:IDENTITY_UPDATES * BATCH].reshape(IDENTITY_UPDATES, BATCH), axis=1)]
        with torch.no_grad():
            self.model.rows.copy_(torch.from_numpy(batches))
    def config(self):
        """config.json "identity" entry. schedule_updates is the planned length that scales ANVIL's momentum warm-up;
        a shaping mechanism with `until` acts on identity updates 0..until-1."""
        return dict(updates=IDENTITY_UPDATES, schedule_updates=IDENTITY_SCHEDULE, batch=BATCH, sequence_len=ID_SEQ_LEN,
        init='prev8', value_output_pairing=dict(layers=list(range(L)), free_after_init=True),
        wte_gate_energy=dict(energy=WTE_GATE_ENERGY, channels=GATE_CHANNELS, init=True, until=SHAPING_UPDATES),
        value_embed_common_gain=dict(gain=VE_COMMON_GAIN, until=SHAPING_UPDATES),
        gate_credit_pooling=dict(layers=list(VE_BLOCKS), shared_component='head_mean_credit_only',
        until=SHAPING_UPDATES),
        value_embed_partner_transfer=dict(gain=VE_PARTNER_GAIN, donor='equal_window_partner', until=SHAPING_UPDATES),
        lr=LR, betas=[BETA1, BETA2], lr_wte=LR_WTE, lr_value_embeds=LR_VE, lr_lambdas=LR_LAMBDAS,
        anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD)
    def run(self):
        nsys_from = int(os.environ.get('NSYS_ID_FROM', -1))  # screen: nsys capture from this identity update on
        for i in range(IDENTITY_UPDATES):
            if i == nsys_from:
                torch.cuda.profiler.start()
            if nsys_from >= 0:
                torch.cuda.nvtx.range_push(f'id{i}')
            (self.plain_graph if self.plain_graph is not None and i >= SHAPING_UPDATES else self.graph).replay()
            if nsys_from >= 0:
                torch.cuda.nvtx.range_pop()
        self.model.gather_fp32()
    def close(self, free=True):
        self.graph = self.plain_graph = None
        self.model = None
        if free:  # (IDENTITY_KEEP_CACHE: the blocks stay in the caching allocator for the Cut; ~70 cudaFree in the timer)
            torch.cuda.empty_cache()


WARM_START = False  # Cut's ANVIL and Adam states start from the identity phase's (momenta, energies, moments)
WARM_START_SCALE = 1.0  # multiplies the carried momenta (fast, slow, Adam's first moment)


@torch.no_grad()
def warm_start_cut(idm, model, optimizer):
    """Copy the identity phase's optimizer states into Cut's. Identity matrices are stored as x @ W (Cut's weights
    transposed), except the MLP c_proj, stored as Cut's weight; ANVIL states are sharded across ranks there."""
    full = {}
    for name in ('square', 'wide'):
        sl = idm.shard[name]
        full[name] = {}
        for key, st in idm.anvil[name].items():
            out = torch.empty_like(st)
            dist.all_gather_into_tensor(out, st[sl].contiguous())
            full[name][key] = out
    def anvil_state(param_name):
        parts = param_name.split('.')
        i, kind = int(parts[2]), '.'.join(parts[3:5])
        if kind in SQUARE:
            index, name, flip = SQUARE.index(kind) * L + i, 'square', True
        elif kind == 'mlp.c_fc':
            index, name, flip = i, 'wide', True
        else:
            index, name, flip = L + i, 'wide', False
        f = lambda z: z[index].transpose(-1, -2) if flip else z[index]
        return f(full[name]['fast']), f(full[name]['slow']), f(full[name]['energy'])
    anvil, adam = optimizer.optimizers
    names = {id(p): n for n, p in model.named_parameters()}
    for params, state, lanes, ratio in anvil.batches:
        for k, p in enumerate(params):
            fast, slow, energy = anvil_state(names[id(p)])
            state['fast'][k].copy_(WARM_START_SCALE * fast)
            state['slow'][k].copy_(WARM_START_SCALE * slow)
            if energy.numel() != state['energy'][k].numel():  # LANE_ROWS: mlp.c_proj's per-column energy, averaged
                energy = energy.mean()
            state['energy'][k].copy_(energy.reshape(state['energy'][k].shape) if energy.ndim else energy)
    moments = {}
    for (p, _, _), m, v in zip(idm.adam, idm.m, idm.v):
        moments[id(p)] = (m, v)
    by_name = {'transformer.wte.weight': (idm.wte, False), 'lm_head.weight': (idm.lm_head, True),
               'resid_lambdas': (idm.lam_r, False), 'x0_lambdas': (idm.lam_0, False)}
    for i in idm.gates:
        by_name[f'transformer.h.{i}.attn.ve_gate.weight'] = (idm.gates[i], True)
        by_name[f'value_embeds.{i}.weight'] = (idm.ve[i], False)
    for group in adam.param_groups:
        for p in group['params']:
            source, flip = by_name[names[id(p)]]
            m, v = moments[id(source)]
            m, v = (m.T, v.T) if flip else (m, v)
            adam.state[p] = dict(step=torch.tensor(float(IDENTITY_UPDATES)),
                                 exp_avg=(WARM_START_SCALE * m).to(p.dtype).contiguous(),
                                 exp_avg_sq=v.to(p.dtype).contiguous().clone())


def difficulty_order(rows, settings):
    n, batch = (len(rows), settings.cut_batch)
    token_ids = rows.numpy()[:, 1:]
    counts = np.bincount(token_ids.reshape(-1), minlength=settings.model.vocab_size)
    surprisal = -np.log((counts.astype(np.float64) + 1.0) / (counts.sum() + len(counts)))
    score = np.empty(n, dtype=np.float64)
    for start in range(0, n, 256):
        score[start:start + 256] = surprisal[token_ids[start:start + 256]].mean(axis=1)
    ordered = np.argsort(score, kind='stable')
    rng = np.random.default_rng(SEED)
    strata = ordered.reshape(batch, n // batch)
    plan = np.stack([rng.permutation(group) for group in strata], axis=1)
    plan = np.take_along_axis(plan, np.argsort(rng.random(plan.shape), axis=1), axis=1)
    return plan.reshape(-1)


def train_stages(args, settings, device, rank, world):
    if world != 8:
        raise ValueError('This speedrun recipe requires eight GPUs')
    identity = IdentityPhase(device, rank, world)
    # The identity rows are left out of the Cut and held-out rows (one epoch over both phases).
    identity_indices = torch.from_numpy(identity.order[:IDENTITY_UPDATES * BATCH])
    keep = torch.ones(TRAIN_ROWS, dtype=torch.bool)
    keep[identity_indices] = False
    indices = torch.arange(TRAIN_ROWS)[keep]
    indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout_indices, train_indices = indices[-HOLDOUT:], indices[:-HOLDOUT]
    means = None
    horizon = len(train_indices) // settings.cut_batch  # every remaining row: 14648 - 8*300 - 64 = 8*1523
    stop = min(args.steps or horizon, horizon)
    schedule = cut_updates(horizon, settings)  # one ANVIL schedule row per update
    wide = any(len(group) > 1 for group in schedule)
    updates = [group for group in schedule if group[-1] <= stop]
    stop = updates[-1][-1]
    train_count = stop * settings.cut_batch
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(
            json.dumps(
                dict(
                    script=__file__, world=world, seed=SEED, recipe=asdict(settings),
                    steps=len(updates) + IDENTITY_UPDATES,
                    overrides=args.set,
                    epoch_updates=horizon + IDENTITY_UPDATES,
                    train_rows=train_count + IDENTITY_UPDATES * BATCH,
                    upstream='qlabs-eng/noise@3343973b6f1af119effcdea2622f1fe672c51b1e',
                    additions=[
                        'identity_handoff_at_300', 'identity_init_and_early_update_shaping',
                        'difficulty_balanced_row_order', 'joint_partial_chain_roots', 'reduced_forward_replay',
                        'diagonal_readout_moments', 'x0_skip_embedding_credit', 'fused_forward_kernels',
                        'late_zero_anvil_decay', 'sketched_screened_merge', 'cut_key_degree2_moments',
                        'cut_mixing_scalar_contraction', 'cut_estimate_cuda_graph', 'resident_merge_scorer',
                        'response_ranked_mlp_routes', 'batched_update_telemetry',
                    ],
                    torch_version=torch.__version__,
                    merge_sketch=dict(columns=MERGE_SKETCH_COLUMNS, output_rank=MERGE_K, seed=SEED),
                    identity=identity.config(),
                    cut_anvil_decay=dict(
                        early_multiplier=1.0, late_multiplier=settings.late_cut_decay_multiplier,
                        late_after_updates=round(275 / 1043 * horizon), horizon=horizon,
                    ),
                ),
                indent=2,
            )
            + '\n'
        )

    def rank_slice(batch):
        per = len(batch) // world
        first = rank * per
        return batch[first : first + per].to(device)

    # Compile Cut's update on random tokens with a throwaway model, optimizer and Cut. The timed ones are rebuilt from
    # seed below, and the timed Cut captures its estimate graph on the same tokens.
    noise = torch.randint(0, settings.model.vocab_size, (settings.cut_batch, settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(SEED))
    model = make_model(device, SEED, settings.model)
    optimizer = make_optimizer(model, settings, horizon)
    estimator = Cut(model, settings, SEED, rank, world)
    batch = rank_slice(noise)
    inp, target = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    for _ in range(4):
        estimator.step(inp, target, optimizer)
        torch.cuda.synchronize()
    estimator.close()
    del model, optimizer, estimator, inp, target, batch
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(device, SEED, settings.model)
    optimizer, estimator = (
        make_optimizer(model, settings, horizon),
        Cut(model, settings, SEED, rank, world),
    )
    if wide:
        widen_anvil_schedule(optimizer, schedule, settings)
    graph_batch = rank_slice(noise)
    estimator.bind_flat_layout(optimizer)  # (FLAT_GRADIENTS: the graph packs the gradients in step's layout)
    if AMORTIZED_ROOTS:  # both modes' graphs (one row, and NATIVE_WIDE's pair), each on the random tokens
        if not (estimator.packs_gradients() and (not wide or NATIVE_WIDE)):
            raise ValueError('AMORTIZED_ROOTS needs packed gradients and (wide) NATIVE_WIDE')
        if any(set(group) & set(AMORTIZED_ROOTS) and not set(group) <= set(AMORTIZED_ROOTS)
               for group in settings.joint_root_groups):
            raise ValueError('AMORTIZED_ROOTS: a joint root group mixes amortized and replayed roots')
        if wide:
            estimator.partner = Cut(model, settings, SEED, rank, world)
        pair_rows = noise[:2].to(device)
        def capture_mode(cut):
            cut.capture_estimate_graph(graph_batch[:, :-1].contiguous(), graph_batch[:, 1:].contiguous())
            if wide:
                cut.capture_pair_graph(pair_rows[:, :-1].contiguous(), pair_rows[:, 1:].contiguous())
        estimator.capture_modes(capture_mode)
        estimator.reset_deviation()  # (the captures' warmups moved it)
        estimator.amortized_window(len(updates))
        del pair_rows
    else:
        estimator.capture_estimate_graph(graph_batch[:, :-1].contiguous(), graph_batch[:, 1:].contiguous())
    if wide:  # wide_step's row sum, allocated before the timer
        estimator._wide_sum = torch.empty_like(estimator._packed[0])
    if wide and NATIVE_WIDE and not AMORTIZED_ROOTS:  # the two-row graph (the partner only draws row 2's randomness), on random tokens
        if not CUT_SIGN_GENERATOR or any(len(group) * settings.cut_batch // world % 2 for group in schedule
                                         if len(group) > 1):
            raise ValueError('NATIVE_WIDE needs CUT_SIGN_GENERATOR and an even number of rows per GPU')
        estimator.partner = Cut(model, settings, SEED, rank, world)
        pair_rows = noise[:2].to(device)
        estimator.capture_pair_graph(pair_rows[:, :-1].contiguous(), pair_rows[:, 1:].contiguous())
        del pair_rows
    elif wide and CONCURRENT_WIDE and not AMORTIZED_ROOTS:  # the partner's graph, captured on the same random tokens
        if not CUT_SIGN_GENERATOR or any(len(group) * settings.cut_batch // world % 2 for group in schedule
                                         if len(group) > 1):
            raise ValueError('CONCURRENT_WIDE needs CUT_SIGN_GENERATOR and an even number of rows per GPU')
        estimator.partner = Cut(model, settings, SEED, rank, world)
        estimator.partner.bind_flat_layout(optimizer)
        estimator.partner.capture_estimate_graph(graph_batch[:, :-1].contiguous(), graph_batch[:, 1:].contiguous())
    if GATE_GRAPH and PREFETCH_GATE_ROWS:
        for cut in (estimator, estimator.partner):
            if cut is not None:
                cut.capture_gate_graph()
    if ANVIL_GRAPHS and SHARDED_ANVIL and estimator.packs_gradients():
        optimizer.optimizers[0].capture_shard_graphs(torch.bfloat16 if COMMS_BF16 else torch.float32)
    log = (args.output / 'metrics.jsonl').open('w') if rank == 0 else None
    scorer = prepare_resident_merge_score(model, settings, rank, world)
    del noise, graph_batch
    # Untimed, as in #44 (qlabs-eng/noise@3343973b, train_stages): read train.pt, split it by the seeded permutations
    # above and put the identity phase's batches on the GPU. No split or order here depends on the rows' contents;
    # load_rows only verifies their pinned hash.
    all_rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    cut_rows = all_rows.index_select(0, train_indices)
    holdout_rows = all_rows.index_select(0, holdout_indices)
    # QUAD_MERGE's pick rows (this rank's; identity rows by a seeded permutation, or held-out rows), copied to the GPU
    # in the merge. 'unused' (the Cut rows a --steps run leaves) is taken from the timed row order below.
    if QUAD_SOURCE == 'identity':
        shuffle = torch.randperm(len(identity_indices), generator=torch.Generator().manual_seed(SEED + 3))
        picks = identity_indices[shuffle]
        quad_rows = all_rows.index_select(0, picks[rank * QUAD_ROWS:(rank + 1) * QUAD_ROWS])
    elif QUAD_SOURCE == 'holdout':
        quad_rows = holdout_rows[rank::world][:QUAD_ROWS]
    identity.bind_rows(all_rows.numpy())
    del all_rows
    if OVERLAP_SETUP:  # the Cut rows' staging buffers (sizes only: their contents are chosen inside the timer)
        staged_rows = torch.empty((train_count, cut_rows.shape[1]), dtype=cut_rows.dtype, pin_memory=True)
        device_rows = torch.empty(staged_rows.shape, dtype=staged_rows.dtype, device=device)
        copy_stream = torch.cuda.Stream()
    if CUT_RESERVE_GB:  # (the Cut's first update otherwise spent ~7 ms in two cudaMalloc calls)
        torch.empty(int(CUT_RESERVE_GB * 2 ** 30), dtype=torch.uint8, device=device)
    torch.cuda.reset_peak_memory_stats()
    if GC_FREEZE:
        gc.collect()
        gc.freeze()
    begin_training()
    start = time.perf_counter()
    if TRANSITION_LAZY:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    def prepare_rows():
        """The timed host setup of the Cut rows (no CUDA calls): their order, the unused and quad rows, the ordered rows
        (OVERLAP_SETUP: into the pinned staging buffer), the disjointness check and row_usage.json."""
        order = torch.from_numpy(difficulty_order(cut_rows, settings))  # timed: it reads the rows' tokens
        ordered = train_indices[order]
        unused, used = ordered[train_count:], ordered[:train_count]
        quad = None
        if QUAD_SOURCE == 'unused':
            if len(unused) < world * QUAD_ROWS:
                raise ValueError(f'{len(unused)} unused rows: QUAD_SOURCE=unused needs {world * QUAD_ROWS}')
            quad = cut_rows.index_select(0, order[train_count:][rank * QUAD_ROWS:(rank + 1) * QUAD_ROWS])
        if OVERLAP_SETUP:
            rows = torch.index_select(cut_rows, 0, order[:train_count], out=staged_rows)
        else:
            rows = cut_rows.index_select(0, order[:train_count])
        all_used = torch.cat([identity_indices, used, holdout_indices, unused])
        assert len(torch.unique(all_used)) == len(all_used) == TRAIN_ROWS
        if rank == 0:
            digest = lambda ids: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            usage = dict(identity_rows=len(identity_indices), cut_rows=train_count, merge_rows=len(holdout_indices),
                         unused_rows=len(unused), unused_sha256=digest(unused), disjoint=True,
                         identity_sha256=digest(identity_indices), cut_sha256=digest(used),
                         merge_sha256=digest(holdout_indices))
            (args.output / 'row_usage.json').write_text(json.dumps(usage, indent=2) + '\n')
        return used, unused, quad, rows

    try:
        if OVERLAP_SETUP and SETUP_THREAD:  # the setup on a host thread while this one enqueues the identity replays
            box = {}

            def prepare_into_box():
                try:
                    box['rows'] = prepare_rows()
                except BaseException as error:  # re-raised on the main thread
                    box['error'] = error
            worker = threading.Thread(target=prepare_into_box)
            worker.start()
            identity.run()
            worker.join()
            if 'error' in box:
                raise box['error']
            prepared = box['rows']
        else:
            if OVERLAP_SETUP:  # the identity phase's replays enqueued first: the host prepares the rows under them
                identity.run()
            prepared = prepare_rows()
        train_indices, unused_indices, quad, rows = prepared
        if quad is not None:
            quad_rows = quad
        if OVERLAP_SETUP:  # pinned staging, copied on a side stream under the identity phase
            with torch.cuda.stream(copy_stream):
                training = device_rows.copy_(rows, non_blocking=True)
            torch.cuda.current_stream().wait_stream(copy_stream)
        else:
            training = rows.to(device)
        del cut_rows, rows
        if not OVERLAP_SETUP:
            identity.run()
        with torch.no_grad():
            weights = model.state_dict()
            for name, value in identity.model.state_dict().items():
                weights[name].copy_(value)
            if WARM_START:
                warm_start_cut(identity.model, model, optimizer)
        identity_record = dict(step=IDENTITY_UPDATES, phase='identity', phase_step=IDENTITY_UPDATES,
                               seqs=IDENTITY_UPDATES * BATCH,
                               tokens=IDENTITY_UPDATES * BATCH * settings.model.sequence_len)
        def write_identity_record(loss, elapsed):
            identity_record.update(train_loss=loss, train_seconds=round(elapsed, 2))
            if rank == 0:
                print(json.dumps(identity_record), flush=True)
                log.write(json.dumps(identity_record) + '\n')
                log.flush()
        if TRANSITION_LAZY:
            identity_loss = average_ranks(identity.model.loss.clone())
            identity_end = torch.cuda.Event(enable_timing=True)
            identity_end.record()
        else:
            identity_loss = average_ranks(identity.model.loss.clone()).item()
            identity.close(free=not IDENTITY_KEEP_CACHE)
            torch.cuda.synchronize()
            write_identity_record(identity_loss, time.perf_counter() - start)
        pending_records = []
        means_stream = torch.cuda.Stream() if MEANS_SIDE else None
        for step, group in enumerate(updates, 1):
            begin, end = (group[0] - 1) * settings.cut_batch, group[-1] * settings.cut_batch
            # a wide update's rate: the mean of its steps' (the schedule is linear within one), scaled
            set_rate(optimizer, group[0] if len(group) == 1 else sum(group) / len(group), horizon, settings.warmdown,
                     settings.final, settings.cut_warmup)
            if len(group) > 1:
                for param_group in optimizer.optimizers[1].param_groups:
                    param_group['lr'] *= rate_scales(settings, group)[0]
            batch = rank_slice(training[begin:end])
            nsys = [int(z) for z in os.environ.get('NSYS_CUT', '').split(',') if z]  # screen: stop, start, stop, ... steps
            if nsys and step in nsys[0::2]:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            if nsys and step in nsys[1::2]:
                torch.cuda.profiler.start()
            if nsys:
                torch.cuda.nvtx.range_push(f'cut{step}' + ('w' if len(group) > 1 else ''))
            prof_step = int(os.environ.get('PROF_CUT_STEP', 0))
            if prof_step and step == prof_step:  # screen: profile 8 Cut updates on rank 0
                from torch.profiler import ProfilerActivity, profile
                profiler = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
                profiler.__enter__()
            # PRE_STAGE: views of the device rows, copied into the graphs' inputs on the staging stream
            rows_in, rows_out = (batch[:, :-1], batch[:, 1:]) if PRE_STAGE else (
                batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
            loss = (estimator.wide_step if len(group) > 1 else estimator.step)(rows_in, rows_out, optimizer)
            if nsys:
                torch.cuda.nvtx.range_pop()
            if prof_step and step == prof_step + 7:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                if rank == 0:
                    print(profiler.key_averages().table(sort_by='cuda_time_total', row_limit=45,
                                                        max_name_column_width=70), flush=True)
            record = dict(step=step + IDENTITY_UPDATES, phase='cut', phase_step=step,
                          seqs=end + IDENTITY_UPDATES * BATCH,
                          tokens=(end + IDENTITY_UPDATES * BATCH) * settings.model.sequence_len, train_loss=loss,
                          peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1000000000.0, 3))
            pending_records.append(record)
            flush = len(pending_records) >= settings.telemetry_batch or step == len(updates)
            if flush:
                if not bool(torch.stack(estimator._telemetry_health).all()):
                    raise FloatingPointError('Non-finite estimator credit')
                estimator._telemetry_health.clear()
                estimator.join_losses([entry['train_loss'] for entry in pending_records])
                losses = torch.stack([entry['train_loss'] for entry in pending_records])
                losses = (average_ranks(losses) if LAZY_LOSS else losses).cpu().tolist()
                if any(not math.isfinite(value) for value in losses):
                    raise FloatingPointError('Non-finite training loss')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if TRANSITION_LAZY and 'train_loss' not in identity_record:
                    write_identity_record(identity_loss.item(), start_event.elapsed_time(identity_end) / 1000)
                for entry, measured_loss in zip(pending_records, losses):
                    entry['train_loss'] = measured_loss
                    entry['train_seconds'] = round(elapsed, 2)
                if rank == 0:
                    records_text = ''.join(json.dumps(entry) + '\n' for entry in pending_records)
                    print(records_text, end='', flush=True)
                    log.write(records_text)
                    log.flush()
                pending_records.clear()
            # Weight averages for the end merge: every MERGE_EVERY steps of cut_batch rows (and the last) from MERGE_FROM
            # of the rows on (a wide update counts as its steps: the averages sample the same rows at the same decays).
            if ((any(k % MERGE_EVERY == 0 for k in group) or step == len(updates))
                    and begin / len(training) >= MERGE_FROM):
                with torch.no_grad():
                    if means is None:
                        means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                    elif FOREACH_MEANS and MEANS_SIDE:  # the next update's step waits for it (Cut.update)
                        means_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(means_stream):
                            current = [q.detach().float() for q in model.parameters()]
                            for a, mean in zip(MERGE_DECAYS, means):
                                torch._foreach_lerp_(mean, current, a)
                            estimator._means_done = torch.cuda.Event()
                            estimator._means_done.record()
                        del current
                    elif FOREACH_MEANS:
                        current = [q.detach().float() for q in model.parameters()]
                        for a, mean in zip(MERGE_DECAYS, means):
                            torch._foreach_lerp_(mean, current, a)
                    else:
                        for a, mean in zip(MERGE_DECAYS, means):
                            for m, q in zip(mean, model.parameters()):
                                m.lerp_(q.detach().float(), a)
        if getattr(estimator, '_means_done', None) is not None:  # MEANS_SIDE: the last lerp
            torch.cuda.current_stream().wait_event(estimator._means_done)
            estimator._means_done = None
        extra = {}
        if means and args.save_premerge is not None and rank == 0:  # offline merge screens only; timed, not a record
            names = [n for n, _ in model.named_parameters()]
            torch.save(dict(state={n: v.detach().cpu() for n, v in model.state_dict().items()}, decays=MERGE_DECAYS,
                            means=[{n: m.cpu() for n, m in zip(names, mean)} for mean in means],
                            model_config=asdict(settings.model), holdout_indices=holdout_indices,
                            identity_indices=identity_indices, unused_indices=unused_indices,
                            cut_steps=stop), args.save_premerge)
        if means:
            if os.environ.get('NSYS_MERGE'):  # screen: the end merge as its own capture range
                torch.cuda.profiler.start()
            merge_start = time.perf_counter()
            if QUAD_MERGE:
                scorer.bind(holdout_rows)
                extra.update(quad_merge(model, means, quad_rows.to(device), rank, world, scorer))
            else:
                extra.update(end_merge(model, means, holdout_rows, rank, scorer, settings))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
            means = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        end_training()
        if TRANSITION_LAZY:
            identity.close(free=not IDENTITY_KEEP_CACHE)
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        result = dict(
            val_loss=evaluate(state, args.data_dir), train_seconds=round(elapsed, 2),
            steps=len(updates) + IDENTITY_UPDATES, identity_steps=IDENTITY_UPDATES, cut_steps=len(updates),
            seqs=train_count + IDENTITY_UPDATES * BATCH,
            tokens=(train_count + IDENTITY_UPDATES * BATCH) * settings.model.sequence_len,
            peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1000000000.0, 3), **extra,
        )
        if rank == 0:
            if not args.no_ckpt:
                torch.save(state, args.output / 'final.pt')
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    finally:
        scorer.close()
        estimator.close()
        if estimator.partner is not None:
            estimator.partner.close()
        if log is not None:
            log.close()


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--no-ckpt', action='store_true')
    parser.add_argument('--save-premerge', type=Path, help='save weights and averages before the merge (screens)')
    parser.add_argument('--steps', type=int,
                        help='Cut updates to run after the identity phase; preserves the full-epoch LR schedule.')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='screening overrides: a Settings field (lower case) or a module constant (upper case)')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('steps must be positive')
    settings = Settings()
    for item in args.set:
        key, text = item.split('=', 1)
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text
        if key.isupper():
            if key not in globals():
                parser.error(f'unknown constant {key}')
            globals()[key] = value
        else:
            if not hasattr(settings, key):
                parser.error(f'unknown setting {key}')
            setattr(settings, key, value)
    settings.__post_init__()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    set_launch_environment()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.distributed.init_process_group('nccl', device_id=device, pg_options=nccl_options())
    train_stages(args, settings, device, rank, world)
    torch.cuda.synchronize()
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
