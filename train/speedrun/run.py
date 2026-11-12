"""Track 2: an identity-path phase hands off to Cut; an in-run merge of a weight average, chosen part by part, ends it.

Builds on qlabs-eng/noise#52 (itself on #51 and #44); the kernels it calls live next to this file (visit_chain*.py,
visit_gluon*.py, query_*.py, moment_kernels.py, key_gluon.py, value_gluon.py, id_kernels.py, gate_response.py,
window_topk.py, project_values.py).

1. Identity phase: 315 updates of 8 rows of #44's 1831-update plan, with #29's identity-path credit (full-cycle readout
   credit, finite MLP responses and #29's attention moments). These 315 updates lie inside the plan's 368-update ANVIL
   momentum ramp; over the last 150 of them the Adam and ANVIL rates fall linearly to 0.3x (#52: handed off at full
   rate). ANVIL updates the matrices and Adam the rest. Block 1's c_q and c_k start with a planted previous-token
   query / key pair per head, its RoPE-frequency amplitudes decaying as 0.9^(i/2) (#52: 0.8^(i/2)). Listed under
   "identity" in config.json: each attn.c_proj starts as the norm-matched transpose of its c_v, and each wte row with
   1/256 of its energy in the 32 gate channels; for the first 195 updates, wte rows keep that split, the value embeddings' vocabulary-common Adam motion is amplified 4x at
   fixed energy, the VE gates' head-mean credit is pooled over the four VE layers, and each VE layer's centered Adam
   motion is joined by its partner layer's (1-5, 3-7). Those mechanisms are exact no-ops after update 195, so a
   second captured graph without them runs the remaining updates.
2. Cut phase: 830 updates of the remaining rows: 152 of 8 rows, then 678 of 16 (#52: 1468 of 8), the 16-row updates
   at 1.3x both optimizers' rates. A 16-row update gives each GPU two rows, estimated together by one replay of a
   two-row estimate graph (the second row drawing its frames and signs from a partner Cut's generators). Each
   8-row step takes one row from each of 8 difficulty strata: rows ranked by the mean add-one-smoothed unigram
   surprisal of their target tokens, with the counts taken over the Cut rows inside the timer. The Cut estimator
   is #52's:
   - the readout probe from the second-order expansion of the softcapped log-sum-exp in the moved logits (its
     quadratic term exact over the vocabulary for every frame direction), the target logit and norm exact;
   - two single MLP roots, at blocks 4 and 6: each forward replay moves its root's mid stream along a full
     512-direction signed Hadamard frame with random signs per token (the same on every GPU, whose rows differ),
     through routed MLPs (each token's 64 units from block 2 on with the largest |relu^2(h + r) - relu^2(h - r)|
     times output-weight norm, h the unit's pre-activation and r = chain_sigma times its input-weight norm), to the
     readout, scored with diagonal readout moments. The block-4 replay also replays attention at block 5 (16 strong
     keys per moved query, shifted by the tail centroid); its moved queries are computed from the root's 64-wide
     routed change against each token's Wo[routes] @ W_q^T. Each root's identity-path prediction (the readout credit
     times the residual scalars above it) is subtracted from the replay and added back to that root's credit. New:
     from 10% to 65% of the Cut updates, block 4's root is replayed on every second update only, its measured
     deviation from the prediction counted twice there (the other updates take the prediction alone). Block 7
     takes the readout credit, the other blocks the next block's credit times its residual scalar, with block 5
     (sandwiched between measured roots 4 and 6) linearly blending 20% of the upward inverse-scalar residual credit
     from root 4;
   - mixing scalars: each block's mid credit contracted with the scalar's two inputs, a linear step, plus #44's path
     term (the value- and gate-path input credits contracted the same way);
   - the embedding's credit: every block's mid credit summed along its x0 skip, one finite step back through the
     norm;
   - query and key credits from the degree-2 moment model of the softmax's response to a finite step of each raw
     coordinate through the norm and RoPE; #44's value and gate paths (exact linear value credit; one finite norm or
     gate step whose credit feeds only linear consumers).
   The ANVIL step orthogonalizes each head's c_q and c_k rows together with soft cross-lane RMS equalization (one
   polar per head and block; #52: whole matrices); its weight decay is zero after the first 275/1043 of the updates; warmdown 0.55 of the updates (#52:
   0.5). Each rank updates 1/8 of every ANVIL shape batch, its weights stored rank-major so that one bf16
   reduce-scatter and one all-gather move each shard unit, each unit's polar and state update a CUDA graph; the
   other gradients are all-reduced in bf16; NCCL runs the LL128 protocol (set in this process before NCCL starts).
3. One weight average (decay 0.02, every 5 updates over the last 70% of the Cut rows) is merged part by part inside
   the run: each part's coefficient a in w + a (avg - w) from a parabola through three trial values on 24 of the 64
   held-out rows per rank (a window starting at row 8 * rank; the parts split over the ranks), the joint move shrunk
   by the best of three factors on all 64 held-out rows (#52: a race over 4 keys at 0.75 or 1.0, decays 0.03 and
   0.02). The merge reads only held-out rows: no row it reads feeds an update.

config.json lists the changes against #44 (qlabs-eng/noise@3343973b) under "additions".

Run on one 8xH100 node, seed 0; measured results are in train/speedrun/log/.
As in #44, train.pt is read (with its pinned-hash check) and split by seeded permutations, and the identity batches
are put on the GPUs, before the timer; apart from the organizers' pinned-hash check, the difficulty order is the only
computation on the rows' contents before the first update, and it is timed (on a host thread while the identity
graphs replay). The random-token warmup (both identity graphs, a throwaway Cut, the Cut-estimate and pair graph
captures, the ANVIL unit graphs, the merge's forwards and the merge scorer, whose inputs the held-out rows overwrite
before scoring) is discarded or reset before timed training starts from seed 0; final validation is untimed.
metrics.jsonl writes the Cut records every 16 updates, each with the time of its flush.

    uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
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
import query_factored_gluon  # noqa: E402
import window_topk  # noqa: E402
import gate_response  # noqa: E402
import project_values  # noqa: E402
import moment_kernels  # noqa: E402
import key_gluon  # noqa: E402
import id_kernels  # noqa: E402
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
IDENTITY_UPDATES, IDENTITY_SCHEDULE = 315, 1831  # hand off to Cut after 315 of 1831 planned identity updates
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
    wide_rows: int = 16  # rows per Cut update from wide_from on (two per GPU, by one two-row estimate replay)
    wide_from: float = 0.1  # fraction of the Cut's cut_batch-row steps after which updates take wide_rows
    wide_lr_scale: float = 1.3  # AdamW's rates in the wide updates, relative to the schedule
    wide_anvil_scale: float = 1.3  # ANVIL's rate in the wide updates, relative to its schedule
    lr: float = 0.0028
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.55
    final: float = 0.5
    anvil_lr: float = 0.036
    anvil_wd: float = 2.25
    fast_weight: float = 1.0
    polar_margin: float = 1.1
    telemetry_batch: int = 16
    fc_sigma: float = 0.05
    route_k_late: int = 64  # this many routes from block route_split on
    route_split: int = 2
    query_moment_sigma: float = 0.003
    chain_sigma: float = 0.01
    chain_depth_roots: tuple = (4, 6)
    joint_root_groups: tuple = ((4,), (6,))
    route_k: int = 128
    chain_tokens: int = 2048
    query_replay_long_layers: tuple = (5,)
    gate_path_sigma: float = 0.01
    query_keys: int = 16
    key_sigma: float = 0.003
    emb_scale: float = 750.0
    lambda_scale: float = 15.0
    def __post_init__(self):
        if self.wide_rows % self.cut_batch:
            raise ValueError('wide_rows must be a multiple of cut_batch')
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
def norm_move(x, v, sigma=0.01):
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
      / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = s * v
    d = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    dots = (x * move).sum(-1, keepdim=True)
    moved_sq = move.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / d + EPS)
    up = torch.rsqrt((sq + 2 * dots + moved_sq) / d + EPS)
    down = torch.rsqrt((sq - 2 * dots + moved_sq) / d + EPS)
    return ((move * up + x * (up - inv0)) - (-move * down + x * (down - inv0))) / (2 * s)


@torch.compile(dynamic=False)
def stacked_gate_input_responses(raws, ves, blocks, weights, rows, sigma: float):
    """The value gates' input credits of several layers in one compiled graph: each gate's finite response to signed
    steps of its block input along the gate rows, through the norm (the layers stacked along a batch axis, each GEMM
    once as a bmm; the kappas from raws (tokens, width) and ves, the layers' value embeddings)."""
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
    response = gate_response.responses(base.squeeze(2), projected, kappa, steps.squeeze(-1), square.squeeze(-1),
                                       cross, row_square, EPS)
    return (response @ r / r.shape[1]).unbind(0)


def gathered_values_dense(table, correction, rows):
    """A gathered-values triple (table, correction, rows) as projected values (tokens, heads * count, O), for consumers
    other than chain3."""
    heads = correction.shape[1] // 2
    shift = (correction[:, :heads].float() + correction[:, heads:].float()).repeat_interleave(rows.shape[1] // heads, 1)
    return (table[rows.long()].float() + shift).to(table.dtype)


def token_slice(z, local):
    """z[local] along tokens; a gathered-values triple keeps its per-key table whole."""
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
    gathered = isinstance(top_values, tuple)  # the triple kept for chain3 (weights-only paths)
    if factored is not None and raw_bf16 == 'parts':  # delta = the root output: q change from its routed factors
        change, steps, scale, wo, routes, basis = factored
        at, f = query_factored_gluon.query_factors_t_gathered(wo, routes, basis, w_q)
        replay = query_factored_gluon.fused_query_replay_factored_rot
        weights = replay(change, steps, scale, at, f, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_p)
        return weights, top_values if gathered else top_values.flatten(1, 2)
    if raw_bf16 == 'parts':  # q projection and replay in one kernel, only weights stored
        weights = query_replay.fused_query_replay(delta, w_q, r0, cos[pos], sin[pos], inv, inv / inv0 - 1,
                                                  top_keys, top_p)
        return weights, top_values if gathered else top_values.flatten(1, 2)
    if gathered:
        top_values = gathered_values_dense(*top_values).unflatten(1, (heads, -1))
    raw = (delta @ w_q.T.to(torch.bfloat16)).view(m, probes, heads, width)
    return fused_projected_response(raw, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_values, top_p,
      add_to, raw_bf16)


@torch.compile(dynamic=False)
def key_context_group(qs, ks, vs, ves, gates, targets, cos, sin, heads: int):
    """Per-head tensors (heads, tokens, head_dim) of several layers, stacked along the head axis: RoPE'd unit queries
    and keys, raw keys, values with the gated value embedding, and the attention output's credit."""
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
      steps.contiguous(), basis[0].float().contiguous() if p == d else basis.float().contiguous(),
      projection.contiguous(), wo, routes.contiguous(),
      clean.contiguous(), inv0.contiguous(), inv.contiguous(), scale, out,
      next_x.float().contiguous() if qstats else out, stats, p, d, r, qstats, BM=64, BN=64,
      HADAMARD=p == d, num_warps=4, num_stages=3)
    return out, stats


@fused_root_visit.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, scale, next_x):
    n, p, d = steps.shape[0], basis.shape[0], wo.shape[1]
    return (torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32))
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
      (frame[0].float().contiguous() if p == d else frame.float().contiguous()) if inject else delta,
      x.float().contiguous(), fc, wo, routes.contiguous(), clean.contiguous(), scale, out,
      target, mu, nu, diag, xl, stats, delta if add is None else add.contiguous(), p, d, r, inject, mode,
      add is not None, inject and p == d, BM=bm, EPSILON=EPS, **cfg)
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
    target_logit_w, y, own, moments, root_projection, query_layers):
    """Replay one root: its signed frame moves enter at site and go through routed MLPs, with attention replayed
    only at query_layers. Returns each token's moved-minus-clean loss for every move, from diagonal readout
    moments."""
    L = len(mids)
    pending_query_inv = replay_add = factored = None
    delta = None
    # the visit (or root) at block L-2 and the readout at L-1 in one kernel (visit_chain)
    chain_tail = (L - 1 not in query_layers and idx[L - 1].shape[1] == 64
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
                parts = keys[b].shape[1] * keys[b].shape[2] == 128
                replay_add = query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                    prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv, raw_bf16='parts' if parts else True,
                    factored=factored if b == site + 1 else None)
            delta = delta.to(torch.bfloat16)
        if b == site:
            inv0, inv = root_norm_inverses(mids[b], *root_moves)
            if chain_tail and b == L - 2 and next_query_x is None:
                terminal_stats = visit_chain.fused_root_readout_fast(*root_moves, root_projection, w_out_t[b], idx[b],
                    hs[b], inv0, inv, next_scale, *tail())
                break
            if (next_query_x is not None
                    and idx[b].shape[1] == 64):  # the root's routed change too, for the factored query replay
                delta, change, partials = visit_gluon_root.fused_root_visit_change(*root_moves, root_projection,
                    w_out_t[b], idx[b], hs[b], inv0, inv, next_scale, next_query_x)
                factored = (change, root_moves[0], next_scale, w_out_t[b], idx[b], root_moves[1])
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
                continue
            root_visit = visit_gluon_root.fused_root_visit_fast
            delta, partials = root_visit(*root_moves, root_projection, w_out_t[b], idx[b], hs[b], inv0,
                inv, next_scale, next_query_x)
            if next_query_x is not None:
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            continue
        if (chain_tail and b == L - 3 and isinstance(replay_add, tuple)
                and next_query_x is None and L - 2 not in query_layers
                and idx[b].shape[1] == 64):  # blocks L-3 (with the replay) to L-1 in one kernel
            m = L - 2
            add_weights, add_values = replay_add
            gathered = isinstance(add_values, tuple)
            terminal_stats = visit_chain3.fused_resp_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b],
                idx[b], hs[b], next_scale, add_weights, add_values[0] if gathered else add_values, mids[m],
                w_fc[m], w_out_t[m], idx[m], hs[m], lam_r[m + 1], *tail(), shared=True,
                gathered_rows=add_values[2] if gathered else None, correction=add_values[1] if gathered else None)
            break
        if isinstance(replay_add, tuple) and isinstance(replay_add[1], tuple):  # not chain3: dense values
            replay_add = (replay_add[0], gathered_values_dense(*replay_add[1]))
        if chain_tail and b == L - 2 and replay_add is None and next_query_x is None:
            terminal_stats = visit_chain.fused_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                hs[b], next_scale, *tail())
            break
        if next_query_x is not None:
            visit = visit_gluon.fused_query_visit_fast
            delta, partials = visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale, next_query_x)
            pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
        elif b == L - 1:
            assert replay_add is None, 'an attention replay must be followed by a routed visit that takes it'
            if idx[b].shape[1] == 64:
                terminal_stats = visit_gluon.fused_readout_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                    hs[b], xL, target_logit_w, moments[2], moments[3], moments[4], None, None)
            else:
                terminal_stats = fused_readout_visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], xL,
                    target_logit_w, moments[2], moments[3], moments[4], None, None)
        elif replay_add is None:
            delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale)
        elif isinstance(replay_add, tuple):
            delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                None, None, add_weights=replay_add[0], add_values=replay_add[1])
            replay_add = None
        else:
            delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                None, None, replay_add)
            replay_add = None
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
chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)
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
    BF16_KINDS = ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        b, t = idx.shape
        cos, sin = model.cos[:, :t], model.sin[:, :t]
        def linear(name, module, z):
            if not name.endswith('mlp.c_proj'):  # (weight_credits_bf16 forms relu^2)
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
            if weights is not None and p + 'attn.qkv' in weights:  # one GEMM, outputs its column slices
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
    L = len(model.transformer.h)
    for kind in STACKED_INPUTS:
        if kind == 'mlp.c_proj':  # (the per-block relu^2 GEMM inputs are kept)
            continue
        stack = torch.stack([inputs[f'transformer.h.{i}.{kind}'] for i in range(L)])
        inputs['stack:' + kind] = stack
        if kind == 'attn.c_q':  # the c_q GEMM's bf16 input (autocast's cast of the same values)
            inputs['stack:attn.c_q:bf16'] = stack.to(torch.bfloat16)
        for i in range(L):
            inputs[f'transformer.h.{i}.{kind}'] = stack[i]
    for i in range(L):  # the same normed block input
        inputs[f'transformer.h.{i}.attn.c_k'] = inputs[f'transformer.h.{i}.attn.c_v'] = (
            inputs[f'transformer.h.{i}.attn.c_q'])
    return inputs, outputs, blocks, loss


STACKED_INPUTS = ('attn.c_q', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


@torch.compile(dynamic=False)
def qkv_weight_credits_bf16(qkv, inputs_bf16):
    """torch.bmm(qkv.transpose(1, 2), inputs.flatten(1, 2)) with bf16 operands, fp32 out."""
    return torch.bmm(qkv.to(torch.bfloat16).transpose(1, 2), inputs_bf16.flatten(1, 2), out_dtype=torch.float32)


class Capture:
    def __init__(self, model, seed):
        self.model = model
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.weights = None  # (the Linears' bf16 weights, set by Cut.estimate)
    def capture(self, x, y):
        """One compiled forward that returns the Linears' inputs and outputs and the blocks' inputs."""
        self.y = y
        self.batch, self.length = x.shape
        self.inputs, self.outputs, self.blocks, loss = capture_forward(self.model, x, y, self.weights)
        self.clean_loss = loss.view_as(y).float()
    def head_error(self, sigma=0.1):
        return coordinate_differences(self.outputs['lm_head'], self.y, sigma)
    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
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


@torch.compile(dynamic=False)
def pack_gradients(gradients, divisor):
    """Cut.step's flat gradient buffer, torch.cat(...) / divisor, and its bf16 copy, in one compiled graph."""
    flat = torch.cat([g.float().reshape(-1) for g in gradients]) / divisor
    return flat, flat.to(torch.bfloat16)


@torch.compile(dynamic=False)
def chain_transposes(weights):
    """Each weight transposed, contiguous."""
    return [w.T.contiguous() for w in weights]


@torch.compile(dynamic=False)
def root_projections_of(fc_weights, frames):
    """Each root's bf16 c_fc rows projected on its frame (fp32, contiguous)."""
    return [(w.float() @ frame.T).contiguous() for w, frame in zip(fc_weights, frames)]


@torch.compile(dynamic=False)
def recompute_from_capture(blocks, attn_outs, fc_outs, proj_outs, fc_weights, proj_weights, first: int,
                           chain_sigma: float):
    """The blocks' mid streams, MLP pre-activations, outputs and route scores, the pre-activations and the MLP outputs
    read from the capture (the forward's bf16 GEMM outputs; no GEMMs)."""
    mid = torch.stack([b.float().flatten(0, 1) + a.float().flatten(0, 1) for b, a in zip(blocks, attn_outs)])
    out = mid + torch.stack([o.float().flatten(0, 1) for o in proj_outs])
    scores = [response_scores(fc_outs[i].float().flatten(0, 1), chain_sigma * fc_weights[i].float().norm(dim=1),
                              proj_weights[i].float().norm(dim=0)) for i in range(first, len(blocks))]
    return list(mid.unbind(0)), [f.flatten(0, 1) for f in fc_outs], list(out.unbind(0)), torch.stack(scores)


@torch.compile(dynamic=False)
def weight_credits_bf16(chains, g_top, lam_r, extras, attn_proj_in, mlp_proj_in, fc_outs, fc_in, fc_out_weights,
                        sigma: float):
    """The blocks' attn.c_proj, mlp.c_proj and mlp.c_fc weight credits from the chain's block credits, the GEMMs on bf16
    operands (fp32 accumulation and output)."""
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
        g_w = torch.mm(d_out[i].to(bf), fc_out_weights[i].to(bf), out_dtype=bf)
        hidden.append((g_w * slope).to(bf))
    hidden = torch.stack(hidden)
    if mlp_proj_in is None:  # each block's relu^2 (bf16, as the forward's c_proj GEMM input)
        # formed in the slope pass that reads the same c_fc output, then one mm per block (bitwise with the bmm)
        mlp_proj = [torch.mm(x_out[i].T, F.relu(fc_outs[i].flatten(0, 1)).square().to(bf), out_dtype=torch.float32)
                    for i in range(L)]
    else:
        mlp_proj = torch.bmm(x_out.transpose(1, 2), mlp_proj_in.flatten(1, 2).to(bf), out_dtype=torch.float32)
    return (torch.bmm(x_mid.transpose(1, 2), attn_proj_in.flatten(1, 2).to(bf), out_dtype=torch.float32),
            mlp_proj,
            torch.bmm(hidden.transpose(1, 2), fc_in.flatten(1, 2).to(bf), out_dtype=torch.float32))


def probe_mm(a, b):
    """a @ b; on bf16 operands, fp32 accumulation and output."""
    return torch.mm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)


def probe_moment_terms_eager(x, y, weight, H):
    """The readout probe (frame H drawn by the caller) in one compiled graph: each move's readout change from its
    second-order expansion in the moved logits (the softcapped log-sum-exp to second order, its quadratic term exact
    over the whole vocabulary for every frame direction, the moved norm and the target logit exact), and the head
    moments the chain replays read."""
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
    n = x.shape[0]
    first = mm(torch.cat([c1, c2 * base]), weight)
    mu1, nu = first[:n], first[n:]
    second = mm(c2, torch.cat([weight.square(), C.square().T], 1))
    curvature, quad = second[:, :width].contiguous(), second[:, width:]
    mu1_h, nu_h = mm(first, H.T).split(n)
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
    """probe_moment_terms for consecutive rows of tokens, row r with its own frame frames[r], the
    rows' credits and moments concatenated by token."""
    t = x.shape[0] // len(frames)
    parts = [probe_moment_terms_eager(x[r * t:(r + 1) * t], y[r * t:(r + 1) * t], weight, H)
             for r, H in enumerate(frames)]
    return (torch.cat([part[0] for part in parts]),
            tuple(torch.cat([part[1][k] for part in parts]) for k in range(len(parts[0][1]))))


def descending_inputs_eager(raws, gates, blocks, c_vs, lam_r, x0, outs, moved: bool = False):
    """The descending value chain's input responses and carried extras (corrected is formed by the caller), and the
    mixing scalars' path contractions: raws (tokens, width) per layer, blocks the bf16 block inputs. moved: c_vs holds
    the value responses raws[i] @ c_v[i] already."""
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
    (lam * credit) and the mixing scalars' two contractions of the credit (inputs + extra)."""
    inputs = norm_move(block.float().flatten(0, 1), v)
    if gate is not None:
        inputs = inputs + gate
    credit = inputs + extra
    return inputs, lam * credit, torch.stack([(credit * previous).sum(), (credit * x0).sum()])


@torch.compile(dynamic=False)
def stacked_sums(chains, extras):
    return torch.stack([c + e for c, e in zip(chains, extras)])


@torch.compile(dynamic=False)
def chain_tail(measured, g_top, lam_r, lam_0, x0, outs):
    """Each block's mid credit (measured, else carried from the next block by its residual scalar; the last block
    from the readout), the mixing scalars' contractions and the credit reaching x0."""
    L = len(measured)
    credits = [None] * L
    credits[L - 1] = measured[L - 1] if measured[L - 1] is not None else g_top
    for i in reversed(range(L - 1)):
        credits[i] = measured[i] if measured[i] is not None else lam_r[i + 1] * credits[i + 1]
    for i in range(1, L - 1):
        if measured[i] is None and measured[i - 1] is not None and measured[i + 1] is not None:
            credits[i] = credits[i].lerp(credits[i - 1] / lam_r[i].clamp_min(1e-6), 0.20)
    lams = [torch.stack([(credits[i] * (x0 if i == 0 else outs[i - 1])).sum(), (credits[i] * x0).sum()])
            for i in range(L)]
    x0_credit = lam_r[0] * credits[0] + sum(lam_0[i] * credits[i] for i in range(L))
    return credits, lams, x0_credit


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
    """Sets the entries outside the causal left window to -inf in place, writing only those entries."""
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
def _route_topk_pruned_kernel(Scores, Indices, Scratch, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                              CAP: tl.constexpr):
    """torch.topk(scores, K).indices of one row, exactly: floor = the K-th largest of the N / G group maxima (count(key
    >= floor) >= K, so every top-K key is >= it); the candidates key >= floor, if at most CAP (else the binary search
    from the floor), compacted and top-K'd as (key, N - 1 - col) (values descending, equal values by ascending
    index)."""
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
    cap, groups = 2 * count, min(8 * count, n)
    scratch = torch.empty((rows, cap), device=scores.device, dtype=torch.int64)
    _route_topk_pruned_kernel[(rows,)](scores.contiguous(), indices, scratch, N=n, K=count, G=n // groups,
                                       CAP=cap, num_warps=1)
    return indices


def stream_route_moments(q, keys, values, window, count):
    heads, length, width = keys.shape
    scores = torch.bmm(q.to(keys.dtype).transpose(0, 1), keys.transpose(1, 2))
    window_fill_(scores, window)
    tops, indices = window_topk.window_topk_pruned(scores, count)
    probabilities = torch.empty((heads, length, count), device=q.device)
    out = torch.empty((heads, length, width), device=q.device)
    all_keys = torch.empty_like(out)
    _route_moment_stream_kernel[(length // 32, heads)](scores, keys, values, tops, probabilities, out, all_keys,
      T=length, WINDOW=window, KEYS=count, BM=32, BN=128, num_warps=4, num_stages=2)
    return probabilities, indices, out, all_keys


class Cut:
    def __init__(self, model, settings, seed, rank, world):
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model, seed + 1000 * rank)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * rank)
        self.move_generator = torch.Generator(self.device).manual_seed(seed + 23 + 1000 * rank)
        self.sign_generator = torch.Generator(self.device).manual_seed(seed + 29 + 1000 * rank)
        self.partner = None  # a second Cut on the same model whose graph estimates a wide update's row 2
        self._estimate_graph = None
        self._telemetry_health = []
        self._query_moment_errors = {}
        self._side_stream = torch.cuda.Stream()
        self._sources = [self]  # each estimated row's draw source (row 2 draws from the partner's generators)
        self._amortize = None  # AMORTIZED_ROOTS: this estimate 'measure's the roots or 'skip's them
        self._modes = {}  # AMORTIZED_ROOTS: each mode's captured graphs and buffers (Cut.use_mode)
        self._updates = 0
        self._window = (0, 1 << 30)  # AMORTIZED_ROOTS: the amortized updates (Cut.amortized_window)
        self._root_scale = torch.full((), float(ROOT_EVERY), device=self.device)  # the boost, a graph input
        self._boosting = True
        # the kernels' variants (read at each launch, outside the compiled graphs): root chain with a 2-slot routed-row
        # ring, tensor-core gate dots and per-thread readout partials; chain3 at 128-move tiles with tensor-core dots
        # and deferred readout sums; packed bf16 conversions; the Gluon key scan and value passes (lse / center rows by
        # TMA, the value source's bf16 high part only); the factored replay's rotated keys as bf16 high parts only
        visit_chain.ROOT_CONFIG.update(NS=2, TC=1, DEFER=3, PK=1)
        visit_chain3.CONFIG.update(DEFER=2, TC=1, BM=128, num_warps=8, NS=4, PK=1)
        visit_gluon_root.ROOT_CONFIGS[64]['PK'] = 1
        moment_kernels.VALUE_IMPL = value_gluon.value_pass_gluon
        moment_kernels.KEY_IMPL = key_gluon.key_scan_gluon
        key_gluon.CONFIG['PF'] = 1
        value_gluon.CONFIG.update(PF=1, HI=1)
        query_factored_gluon.ROT_CONFIG['LO'] = False
        self._gate_rows = {i: torch.empty(self.width, self.width, device=self.device)
                           for i in range(self.model.config.n_layer) if has_ve(i, self.model.config.n_layer)}
    def root_boost(self, group):
        """A measuring update: an amortized group's deviations count self._root_scale times (ROOT_EVERY inside the
        amortized window, 1 outside it; a graph input)."""
        if self._amortize == 'measure' and set(group) & set(AMORTIZED_ROOTS):
            return self._root_scale
        return 1.0
    def amortized_window(self, updates):
        """AMORTIZED_ROOTS: the updates [ROOT_FROM, ROOT_UNTIL) of the Cut's `updates` skip the roots on all but every
        ROOT_EVERY-th; the rest measure them (with the deviation counted once)."""
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
        the rest; outside it, measure (unboosted)."""
        first, last = self._window
        inside = first <= self._updates < last
        mode = 'measure' if not inside or (self._updates - first) % ROOT_EVERY == 0 else 'skip'
        if inside != self._boosting:
            self._root_scale.fill_(float(ROOT_EVERY) if inside else 1.0)
            self._boosting = inside
        self._updates += 1
        if mode != self._amortize:
            self.use_mode(mode)
    def reset_amortization(self):
        """AMORTIZED_ROOTS: the update count and the boost back at their start (the captures' warmups moved them)."""
        self._root_scale.fill_(float(ROOT_EVERY))
        self._boosting = True
        self._updates = 0
    def basis(self, row=0):
        generator = self._sources[row].generator
        signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator).float() * 2 - 1
        return hadamard(self.width, self.device, torch.float32) * signs
    def batched_key_contexts(self, targets):
        """The key contexts (lse and center) and the query moments' inputs of every layer, the layers of one window size
        stacked along the head axis so that each kernel launches once per window size. Returns per-layer contexts and
        key credits (heads, tokens, head_dim); the query moment credits go to self._query_moment_errors."""
        d, model, s = self.capture, self.model, self.s
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        windows = {}
        for i in range(L):
            windows.setdefault(model.window_sizes[i][0], []).append(i)
        contexts, work = {}, []
        self._moment_inputs = {}  # (read by batched_key_scores_by_token)
        rows = d.batch  # (several rows: each layer's rows stacked along the head axis after it, layer-major)
        for window, layers in windows.items():
            pairs = [(i, r) for i in layers for r in range(rows)]
            pick = lambda name: [d.outputs[f'transformer.h.{i}.attn.{name}'][r] for i, r in pairs]
            ves = [d.outputs[f'value_embeds.{i}'][r] if has_ve(i, L) else None for i, r in pairs]
            gates = [d.outputs[f'transformer.h.{i}.attn.ve_gate'][r] if has_ve(i, L) else None for i, r in pairs]
            (qp, kp, k_raw, values, hub), self._moment_inputs[window] = key_context_group_moments(
                pick('c_q'), pick('c_k'), pick('c_v'), ves, gates, [targets[i][r] for i, r in pairs], cos, sin, h,
                s.query_moment_sigma)
            # the lse / center pass on the query moments' bf16 q / k / values / hub
            lse, center = moment_kernels.implicit_key_context_edges(*self._moment_inputs[window][:4], window)
            for j, i in enumerate(layers if rows == 1 else ()):
                heads = slice(j * h, (j + 1) * h)
                contexts[i] = (qp[heads], kp[heads], k_raw[heads], cos, sin, lse[heads],
                               (values[heads], hub[heads], center[heads]))
            work.append((window, layers, qp, kp, k_raw, values, hub, lse, center))
        return contexts, work
    def value_logit_inputs(self, window):
        """The value passes' q / k: the query moments' bf16 copies."""
        return self._moment_inputs[window][:2]
    def batched_key_scores_by_token(self, work, qkv, maps):
        """batched_key_contexts' key credits and query moments, each layer's query and key credits written into columns
        [0, width) and [width, 2 width) of the (layers, tokens, 3 width) buffer qkv (window group g's layers at
        maps[g]); returns the key credits as (tokens, width) views (the query credits go to self._query_moment_errors as
        (1, tokens, width) views)."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys, w = {}, self.width
        rows = d.batch  # (qkv's (layer, row) stretches as (layers x rows, tokens, 3 width), maps' indices into it)
        by_row = qkv.view(-1, t, qkv.shape[-1])
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            q16, k16, v16, hub16, q_raw, sq, steps = self._moment_inputs[window]
            moment_kernels.key_credits_into(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, s.key_sigma,
                                            by_row[:, :, w:2 * w], maps[g])
            moment_kernels.moment_query_scores_into(
                q16, k16, v16, hub16, q_raw, sq, kp, cos[:, :hd // 2].contiguous(), sin[:, :hd // 2].contiguous(),
                lse, center, steps, by_row[:, :, :w], window=window, layers=maps[g])
            for i in layers:
                keys[i] = qkv[i, :, w:2 * w]
                self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = qkv[i, :, :w].view(rows, t, w)
        return keys
    def internals(self, gradients, targets, b, t, key_errors, raw_values, terminal_values):
        d = self.capture
        errors = self._query_moment_errors
        errors.update(key_errors)
        for i in targets:
            errors.update(self.exact_values(i, raw_values[i], terminal_values[i]))
        fused = True
        if fused:  # every layer's query, key and value weight credits from one batched GEMM of the side-by-side errors
            weight = qkv_weight_credits_bf16(self._qkv, d.inputs['stack:attn.c_q:bf16'])
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
        graph's capture, it lets the estimate pack its gradients there ."""
        params = list(self.model.named_parameters())
        anvil = optimizer.optimizers[0]
        anvil_ids = ([id(q) for q in anvil.rank_major_order()])
        rank_of = {pid: k for k, pid in enumerate(anvil_ids)}
        self._flat_order = sorted(params, key=lambda item: rank_of.get(id(item[1]), len(anvil_ids)))
        self._packs = True
    def packs_gradients(self):
        return getattr(self, '_packs', False)
    def step(self, x, y, optimizer):
        self.next_mode()
        gradients = self.estimate_rows(x, y)
        return self.update(gradients, y, optimizer, lambda: torch.where(y != -1, self.capture.clean_loss, 0).sum())
    def estimate_rows(self, x, y):
        if self._estimate_graph is None:
            self.take_gate_rows()
            return self.estimate(x, y)
        with self.staging():
            self.take_gate_rows()
            self._static_x.copy_(x)
            self._static_y.copy_(y)
        self.replay_graph(self._estimate_graph)
        self.staged_replay_done()
        self.prefetch_gate_rows()
        return self._static_gradients
    @contextlib.contextmanager
    def staging(self):
        """The ops inside run on a side stream once the previous estimate replay is done with its inputs (not
        after the previous update's step, which the main stream is still running), and the main stream waits for them."""
        main = torch.cuda.current_stream()
        if not hasattr(self, '_stage_stream'):  # (first use: after everything the main stream has queued, e.g. the rows' copy)
            self._stage_stream, self._replay_done = torch.cuda.Stream(priority=PREFETCH_PRIORITY), torch.cuda.Event()
            self._stage_stream.wait_stream(main)
        stage = self._stage_stream
        stage.wait_event(self._replay_done)
        with torch.cuda.stream(stage):
            yield
    def staged_replay_done(self):
        self._replay_done.record()
    def prepare_step_event(self):
        """The external event the estimate graphs open with, recorded once."""
        if not hasattr(self, '_step_done'):
            self._step_done = torch.cuda.Event(external=True)
            self._step_done.record()
    def wait_step_in_graph(self):
        """The graph's first node waits for self._step_done's last record."""
        torch.cuda.current_stream().wait_event(self._step_done)
    def replay_graph(self, graph):
        """Replay an estimate graph (on the staging stream, after its staged copies; the graph waits for
        the main stream's queue so far through self._step_done, and the main stream for the graph)."""
        main = torch.cuda.current_stream()
        self._step_done.record()
        with torch.cuda.stream(self._stage_stream):
            graph.replay()
        main.wait_stream(self._stage_stream)
    def wide_step(self, x, y, optimizer):
        """Cut.step over several rows per GPU (wide_rows): the rows in pairs, each pair by one replay of the two-row
        estimate graph, the packed gradients averaged over the rows, then one update."""
        if self._estimate_graph is None or not self.packs_gradients() or self.partner is None:
            raise ValueError('wide updates need the captured estimate graphs, packed gradients and the partner Cut')
        self.next_mode()
        loss_sum = self.native_wide_estimate(x, y)
        one_row, self._packed = self._packed, self._wide_packed  # the update reads the pair graph's packed buffers
        try:
            return self.update(self._static_gradients, y, optimizer, loss_sum)
        finally:
            self._packed = one_row
    def native_wide_estimate(self, x, y):
        """A wide update's two rows by one replay of the two-row estimate graph (capture_pair_graph). Row 2 draws from
        the partner's generators, set to this Cut's states one row's draws on, and this Cut's take theirs after the
        replay. The pair graph packs the pair's mean into self._wide_packed; returns the rows' loss sum (a callable:
        the graph's loss buffer is read when the update asks, before the next replay)."""
        partner = self.partner
        if x.shape[0] != 2:
            raise ValueError('a wide update estimates two rows per GPU')
        for mine, theirs, step in zip(self.graph_generators(), partner.graph_generators(), self._row_offsets):
            theirs.set_state(mine.get_state())
            theirs.set_offset(mine.get_offset() + step)
        with self.staging():
            self.take_gate_rows()
            partner.take_gate_rows()
            self._wide_x.copy_(x)
            self._wide_y.copy_(y)
        self.replay_graph(self._wide_graph)
        self.staged_replay_done()
        for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
            mine.set_state(theirs.get_state())
        self.prefetch_gate_rows()
        ahead = torch.Generator(self.device)
        ahead.set_state(self.generator.get_state())
        ahead.set_offset(ahead.get_offset() + self._row_offsets[0])
        partner.prefetch_gate_rows(ahead.get_state())
        losses = self._wide_loss
        return lambda: torch.where(y != -1, losses, 0).sum()
    def capture_pair_graph(self, x, y):
        """Capture the estimate of two rows (x, y: (2, T)) as one graph, row 2 drawing its frames, signs and
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
        if getattr(self, '_means_done', None) is not None:  # the averages read the weights this step writes
            torch.cuda.current_stream().wait_event(self._means_done)
            self._means_done = None
        # counted only for the diagnostic loss, on a side stream (late_loss)
        valid = None if self.packs_gradients() else (y != -1).sum()
        if valid is None:
            self._loss_from = torch.cuda.Event()
            self._loss_from.record()
        optimizer.zero_grad(set_to_none=True)
        packed = getattr(self, '_packed', None) if self.packs_gradients() else None
        if not hasattr(self, '_flat_order'):
            self.bind_flat_layout(optimizer)
        params = self._flat_order
        if packed is not None:  # the estimate already formed the flat buffer (and its bf16 copy)
            flat = packed[0]
        else:
            flat = torch.cat([gradients[name].float().reshape(-1) for name, _ in params]).div_(
                valid * self.world)
        anvil = optimizer.optimizers[0]
        sizes = [sum(q.numel() for q in batch[0]) for batch in anvil.batches]
        wire = packed[1] if packed is not None else flat.to(torch.bfloat16)  # half the bytes on the wire
        # ANVIL's shards first, each batch's update waits only for its own shard, and the rest's
        # all-reduce runs under ANVIL's update
        anvil.receive_rank_major(wire.dtype)
        starts = [sum(len(anvil.rank_stores[h]) for h in range(g)) for g in range(len(anvil.rank_groups))]
        works = {}
        for g in reversed(range(len(anvil.rank_groups))):  # the last unit (the merged polar) first
            works[g] = torch.distributed.reduce_scatter_tensor(
                anvil.rank_receives[g], wire[starts[g]:starts[g] + len(anvil.rank_stores[g])],
                async_op=True)
        anvil.shard_works = [works[anvil.rank_group_of[k]] for k in range(len(sizes))]
        anvil.shard_grads = list(anvil.shard_buffers)
        rest = torch.distributed.all_reduce(wire[sum(sizes):], async_op=True)
        anvil_ids = {id(q) for batch in anvil.batches for q in batch[0]}

        def finish_rest():  # between ANVIL's and the other optimizers' steps
            rest.wait()
            # bf16 params' grads are views of the reduced wire (what .to(bf16) of its fp32 copy gives),
            # fp32 params' slices copied once; the finite check reads the bf16 wire
            main = torch.cuda.current_stream()
            check = self._finite_stream = getattr(self, '_finite_stream', None) or torch.cuda.Stream()
            check.wait_stream(main)
            with torch.cuda.stream(check):
                health = torch.isfinite(torch.sum(wire, dtype=torch.float32))
            health.record_stream(main)
            self._telemetry_health.append(health)
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
        optimizer.step(between=finish_rest)
        if getattr(self, '_finite_stream', None) is not None:
            torch.cuda.current_stream().wait_stream(self._finite_stream)  # before the wire is rewritten
        for work in anvil.gather_works:
            work.wait()
        anvil.gather_works = []
        if valid is None:
            return self.late_loss(loss_sum, y)
        diagnostic_loss = loss_sum() / valid
        return average_ranks(diagnostic_loss.reshape(1)).reshape(())
    def late_loss(self, loss_sum, y):
        """The diagnostic loss from the replay's outputs on a side stream (its all-reduce queued after the
        update's collectives); the main stream waits only for the reads of the replay's loss buffer."""
        side = self._loss_stream = getattr(self, '_loss_stream', None) or torch.cuda.Stream()
        side.wait_event(self._loss_from)
        with torch.cuda.stream(side):
            diagnostic_loss = loss_sum() / (y != -1).sum()
            read = torch.cuda.Event()
            read.record()
            diagnostic_loss = average_ranks(diagnostic_loss.reshape(1)).reshape(())
        torch.cuda.current_stream().wait_event(read)  # (the next replay rewrites the loss buffer)
        return diagnostic_loss
    def join_losses(self, losses):
        """The main stream waits for the side-stream losses before reading them."""
        side = getattr(self, '_loss_stream', None)
        if side is not None:
            main = torch.cuda.current_stream()
            main.wait_stream(side)
            for loss in losses:
                loss.record_stream(main)
    def fused_descending_key_values(self, chain, targets, x0, outs):
        """The attention credits below the chain: key credits and query moments on a side stream, raw value credits,
        the value gates' input credits and the descending value chain batched, the terminal value credits after it, in
        the layout their consumers read, and the chain's glue and the path scalars in one compiled graph. Returns its
        outputs and the path scalars."""
        d, s, model = self.capture, self.s, self.model
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        errors, raw_values, terminal_values = {}, {}, {}
        contexts, work = self.batched_key_contexts(targets)
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
            raw = moment_kernels.raw_value_credits_by_token(*self.value_logit_inputs(window), lse, hub, window, h)
            raw_groups.append(raw)
            for j, i in enumerate(layers):
                raws[i], place[i] = raw[j] if rows == 1 else raw[j * rows:(j + 1) * rows].flatten(0, 1), (g, j)
        ve_layers = [i for i in range(L) if has_ve(i, L)]
        if rows > 1:  # each (layer, row) with its row's gate rows, stacked
            pairs = [(i, r) for i in ve_layers for r in range(rows)]
            responses = stacked_gate_input_responses(
                [raws[i][r * t:(r + 1) * t] for i, r in pairs], [d.outputs[f'value_embeds.{i}'][r] for i, r in pairs],
                [d.blocks[i][r:r + 1] for i, r in pairs], [model.transformer.h[i].attn.ve_gate.weight for i, r in pairs],
                [self._sources[r]._gate_rows[i] for i, r in pairs], s.gate_path_sigma)
            gates = {i: torch.cat(responses[k * rows:(k + 1) * rows]) for k, i in enumerate(ve_layers)}
        else:
            gates = dict(zip(ve_layers, stacked_gate_input_responses(
                [raws[i] for i in ve_layers], [d.outputs[f'value_embeds.{i}'][0] for i in ve_layers],
                [d.blocks[i] for i in ve_layers], [model.transformer.h[i].attn.ve_gate.weight for i in ve_layers],
                [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        c_vs = [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)]
        moved = [None] * L
        for (window, layers, *_), raw in zip(work, raw_groups):
            v = torch.bmm(raw.view(len(layers), rows * t, -1), torch.stack([c_vs[i] for i in layers]))
            for j, i in enumerate(layers):
                moved[i] = v[j]
        c_vs = moved
        input_list, extra_list, path = descending_inputs_eager(
            [raws[i] for i in range(L)], [gates.get(i) for i in range(L)], [d.blocks[i] for i in range(L)],
            c_vs, model.resid_lambdas.float(), x0, outs, moved=True)
        corrected = [torch.empty(len(item[1]) * rows, t, self.width, device=self.device) for item in work]
        for g, (window, layers, *_) in enumerate(work):
            torch.bmm(stacked_sums([chain[i] for i in layers], [extra_list[i] for i in layers]),
                      torch.stack([model.transformer.h[i].attn.c_proj.weight.float() for i in layers]),
                      out=corrected[g].view(len(layers), rows * t, -1))
        w = self.width
        by_row = qkv.view(L * rows, t, 3 * w)
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            moment_kernels.terminal_value_credits_into(*self.value_logit_inputs(window), lse, corrected[g], window,
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
    def exact_values(self, i, credit, value_credit):
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            errors[prefix + 've_gate'] = gate_error(d.outputs[prefix + 've_gate'], d.outputs[f'value_embeds.{i}'],
                                                    credit, h)
        return errors
    def probe_moments(self, x, y):
        if self.capture.batch > 1:  # each row with its own frame
            frames = [self.basis(r) for r in range(self.capture.batch)]
            derivative, self._head_moments = probe_moment_terms_rows(x, y, self.model.lm_head.weight.float(), frames)
            return derivative
        derivative, self._head_moments = probe_moment_terms(x, y, self.model.lm_head.weight.float(), self.basis())
        return derivative
    def close(self):
        if self._estimate_graph is not None:
            torch.cuda.synchronize()
            self._estimate_graph = None
            self._static_gradients = None
            self._static_x = self._static_y = None
            self._gate_rows.clear()
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
        """The next update's gate rows, from self.generator's state after this update's estimate
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
        """Prefetch_gate_rows' draws into the staging buffers as one CUDA graph, on a
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
        return generators + [self.sign_generator]
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
    def chain_credits(self, e, x0, mids, outs, routes, y, g_top):
        model, s = self.model, self.s
        L = model.config.n_layer
        weight = model.lm_head.weight.float()
        names = [f'transformer.h.{i}.mlp.' for i in range(L)]
        w_fc = [self._bf16_weights[p + 'c_fc'] for p in names]
        w_out_t = chain_transposes([self._bf16_weights[p + 'c_proj'] for p in names])
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        moments = self._head_moments
        n, m = e.shape[0], s.chain_tokens
        rows = self.capture.batch  # (each row with its own frames and signs, from its own generators)
        t = n // rows
        if rows > 1 and t % m:
            raise ValueError('a several-row estimate needs whole-row chunks')
        by_site = {}
        for site in s.chain_depth_roots:
            frames = []
            for r in range(rows):
                frames.append(self.basis(r))
            steps = s.chain_sigma * mids[site].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            by_site[site] = (frames, steps, torch.empty(n, self.width, device=self.device))
        signed_steps = {}
        controls = {}
        groups = self.measured_groups()
        for group in s.joint_root_groups:  # (every group's signs: a skipping update draws as a measuring one)
            for site in group:
                steps = by_site[site][1]
                signs = [torch.randint(0, 2, (t, 1), device=self.device,
                                       generator=self._sources[r].sign_generator)
                         for r in range(rows)]
                sign = ((signs[0] if rows == 1 else torch.cat(signs)) * 2 - 1).float()
                signed_steps[site] = steps * sign
                factor = lam_r[site + 1:].prod()
                controls[site] = g_top * factor
        roots = [group[0] for group in groups]
        projected = root_projections_of([w_fc[r] for r in roots for _ in range(rows)],
                                        [frame for r in roots for frame in by_site[r][0]])
        root_projections = {r: projected[k * rows:(k + 1) * rows] for k, r in enumerate(roots)}
        replay_layers = lambda group: (attention_replay_layers(group, L, s.query_replay_long_layers))
        active_query_layers = tuple(sorted({b for group in groups for b in replay_layers(group)}))
        owns = [self.strong_keys(self.own_query_inputs(active_query_layers, r), s.query_keys) for r in range(rows)]
        chunks = [slice(t0, t0 + m) for t0 in range(0, n, m)]
        main = torch.cuda.current_stream()
        lanes = [None] + [self.pair_lane(r) for r in range(1, rows)] if rows > 1 else [None] * rows
        for lane in lanes[1:]:  # each further row's replays on its own stream, joined below
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
                    loss_change = chain_chunk(
                        site, (steps, H), [z[sl] for z in mids], xL[sl], [None if r[0] is None else r[0][sl] for r in routes],
                        [None if r[1] is None else r[1][sl] for r in routes], lam_r, w_fc, w_out_t, base[sl], target_w[sl], y[sl], own_sl,
                        moments=tuple(z[sl] for z in moments),
                        root_projection=root_projections[site][row],
                        query_layers=replay_layers(group),
                    )
                    news = group_credits(loss_change, [signed_steps[o][sl] for o in group],
                                         [controls[o][sl] for o in group], [by_site[o][0][row] for o in group],
                                         self.root_boost(group))
                    for other, new in zip(group, news):
                        by_site[other][2][sl] = new
        for lane in lanes[1:]:
            main.wait_stream(lane)
        del own_sl, own, owns
        measured_sites = {site for group in groups for site in group}
        credits = {site: credit for site, (_, _, credit) in by_site.items() if site in measured_sites}
        def tail(sites):
            measured = [credits.get(i) if i in sites else None for i in range(L)]
            mids_c, lams, x0_credit = chain_tail(measured, g_top, lam_r, lam_0, x0, outs)
            out = dict(enumerate(mids_c))
            for i in range(L):
                out['lam', i] = lams[i]
            out['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
                                           hadamard(self.width, self.device, torch.float32))
            return out
        return tail(measured_sites)
    def pair_lane(self, row):
        """The stream of row's chain replays (made once, outside graph capture's first use)."""
        if not hasattr(self, '_pair_lanes'):
            self._pair_lanes = {}
        if row not in self._pair_lanes:
            self._pair_lanes[row] = torch.cuda.Stream()
        return self._pair_lanes[row]
    def own_query_inputs(self, active_layers, row=0):
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            if i not in active_layers:
                inputs.append(None), w_q.append(None), keys.append(None), values.append(None)
                w_proj_t.append(None), windows.append(model.window_sizes[i][0])
                continue
            prefix = f'transformer.h.{i}.attn.'
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
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)
    @staticmethod
    def strong_keys(own, k):
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        top_keys, top_values, top_p = [], [], []
        for x, wq, kk, vv, window, wp in zip(inputs, w_q, keys, values, windows, w_proj_t):
            if x is None:
                top_keys.append(None), top_values.append(None), top_p.append(None)
                continue
            heads = kk.shape[0]
            q = strong_queries(x, wq, cos, sin, heads)
            p, idx, out, all_keys = stream_route_moments(q, kk, vv, window, k)
            shifted, tables = project_values.gathered_key_tables(vv, out, idx, wp, kk, p, all_keys)
            top_keys.append(shifted), top_values.append(tables)
            top_p.append(p.transpose(0, 1).contiguous())
        return (pos, inputs, w_q, top_keys, top_values, top_p, (cos, sin))
    @torch.no_grad()
    def cast_weights(self):
        """Every Linear's bf16 weight in persistent buffers, by one multi-tensor copy (the capture forward
        and chain_credits read them instead of casting their own)."""
        if not hasattr(self, '_bf16_weights'):
            linears = {n: m.weight for n, m in self.capture.modules.items() if isinstance(m, nn.Linear)}
            assert all(m.bias is None for m in self.capture.modules.values() if isinstance(m, nn.Linear))
            self._bf16_weights = {}
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
        first = min(self.s.chain_depth_roots)
        mids, hpres, outs, scores = recompute_from_capture(
            [d.blocks[i] for i in range(L)], [d.outputs[f'transformer.h.{i}.attn.c_proj'] for i in range(L)],
            [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)],
            [d.outputs[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)],
            [blk.mlp.c_fc.weight for blk in model.transformer.h],
            [blk.mlp.c_proj.weight for blk in model.transformer.h], first, self.s.chain_sigma)
        count = self.s.route_k
        if first >= self.s.route_split:
            count = self.s.route_k_late  # every scored block keeps only its strongest route_k_late
        tops = route_topk(scores, count)
        tops = [None] * first + list(tops)
        tops = [top if top is None or i < self.s.route_split else top[:, :self.s.route_k_late].contiguous()
                for i, top in enumerate(tops)]
        routes = [(None, None) if top is None else (top, h.gather(1, top).float()) for top, h in zip(tops, hpres)]
        g_top = self.probe_moments(outs[-1], yf)
        chain = self.chain_credits(e, x0, mids, outs, routes, yf, g_top)
        return self.chain_gradients(gradients, chain, g_top, x0, outs, y)
    def chain_gradients(self, gradients, chain, g_top, x0, outs, y):
        """The estimate's weight gradients from the chain's block credits (into gradients; packed)."""
        d, model = self.capture, self.model
        L = model.config.n_layer
        b, t = y.shape
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        key_errors, raw_values, terminal_values, extra, inputs, path = self.fused_descending_key_values(
            chain, targets, x0, outs)
        scalars = torch.empty(2, L, device=self.device)
        blocks = model.transformer.h
        attn_proj, mlp_proj, fc = weight_credits_bf16(
            [chain[i] for i in range(L)], g_top, lam_r, [extra[i] for i in range(L)],
            d.inputs['stack:attn.c_proj'], d.inputs.get('stack:mlp.c_proj'),
            [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], d.inputs['stack:mlp.c_fc'],
            [blk.mlp.c_proj.weight for blk in blocks], self.s.fc_sigma)
        for i in range(L):
            p = f'transformer.h.{i}.'
            gradients[p + 'attn.c_proj.weight'], gradients[p + 'mlp.c_proj.weight'] = attn_proj[i], mlp_proj[i]
            gradients[p + 'mlp.c_fc.weight'] = fc[i]
            scalars[:, i] = chain['lam', i]
        scalars += path
        self.internals(gradients, targets, b, t, key_errors, raw_values, terminal_values)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.packs_gradients():  # Cut.step's flat buffer (and bf16 copy) formed here, inside the estimate graph
            self._packed = pack_gradients([gradients[name] for name, _ in self._flat_order],
                                          (y != -1).sum() * self.world)
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


def newton_schulz(x):
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
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
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return Q @ x


HEAD_POLAR = ('q', 'k')  # the Cut ANVIL's polar per head for c_q and c_k: each head's c_q and c_k rows orthogonalized
# together (their own ANVIL batch, interleaving q and k per block; Noise PR #54's anvil_head_polar, joint)
LANE_BETA = 0.9  # ANVIL's lane energy EMA decay (NorMuon's beta2: 0.95)


def head_polar(g, margin, heads: int):
    """[B, r, c] -> each head's row blocks of a c_q / c_k pair (the batch interleaves them) orthogonalized as one
    (each slice scaled by margin x its norm)."""
    B, r, c = g.shape
    x = g.reshape(B // 2, 2, heads, r // heads, c).transpose(1, 2).reshape(B // 2 * heads, 2 * r // heads, c)
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = gram_newton_schulz(x).float()
    return x.reshape(B // 2, heads, 2, r // heads, c).transpose(1, 2).reshape(B, r, c)


def polar_batch(g, margin, heads: int = 1):
    """heads > 1: head_polar of the row blocks."""
    if heads > 1:
        return head_polar(g, margin, heads)
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


def cross_lanes(gained, lanes: int):
    """Each lane of the axis the lane energy does not average over rescaled to equal RMS."""
    rms = gained.square().mean(2 - lanes, keepdim=True).sqrt()
    return gained * rms.lerp(rms.mean((1, 2), keepdim=True), 0.75).clamp_min(1e-10).rsqrt()


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
NCCL_ENV = {'NCCL_PROTO': 'LL128'}


def set_launch_environment():
    """The collectives' protocol (NCCL_ENV), set in this process before NCCL initializes."""
    if torch.cuda.is_initialized():
        raise RuntimeError('NCCL_ENV must be set before CUDA starts')
    os.environ.update({key: str(value) for key, value in NCCL_ENV.items()})


def nccl_options():
    """The collectives' internal stream at high priority (their CTAs take free SMs first)."""
    options = torch.distributed.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    return options
AMORTIZED_ROOTS = (4,)  # these chain roots replayed on every ROOT_EVERY-th update only (their groups must not hold others)
ROOT_EVERY = 2  # AMORTIZED_ROOTS: the updates per measurement; a measuring update counts its deviation ROOT_EVERY times
ROOT_FROM, ROOT_UNTIL = 0.1, 0.65  # AMORTIZED_ROOTS: the fractions of the Cut updates amortized (else measured)
PREFETCH_PRIORITY = -1  # the gate-row prefetch and staging streams' priority (high: their kernels take free SMs first)


class BatchedAnvil(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, steps, polar_margin, fast_weight):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.polar_margin, self.fast_weight = polar_margin, fast_weight
        self.t = 0

    def prepare_scalar_schedule(self, warmdown, final, decay_factors):
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
            factor = final + (1 - final) * min(1.0, left / warmdown)
            rows.append([beta, fast_beta, .98, weight, LANE_BETA,
                group['lr'] * factor, group['weight_decay'] * decay_factors[t]])
        self.scalar_schedule = torch.tensor(rows, dtype=torch.float32, device=device)
        by_shape = {}
        head_ids = getattr(self, 'head_ids', {})  # HEAD_POLAR: id -> polar_batch's heads (make_optimizer)
        for p in group['params']:
            by_shape.setdefault((tuple(p.shape), head_ids.get(id(p), 1)), []).append(p)
        self.batches, self.batch_heads = [], []
        for (shape, heads), params in by_shape.items():
            self.batch_heads.append(heads)
            lanes = 1 if shape[0] >= shape[1] else 0
            stacked = torch.stack([p.data for p in params])
            for i, p in enumerate(params):
                p.data = stacked[i]
            fast = torch.zeros(len(params), *shape, device=device)
            state = dict(fast=fast, slow=torch.zeros_like(fast),
                energy=torch.zeros_like(fast.mean(lanes + 1, keepdim=True)))
            self.batches.append((params, state, lanes, max(1, shape[0] / shape[1]) ** .5))
        if dist.is_initialized():
            self.store_rank_major()

    def store_rank_major(self):
        """Each comm group's weights (each shard unit's batches) moved into one buffer, rank r's chunk holding its shard
        rows of the group's batches (for the record's model: block r's matrices), so one all-gather fills a group."""
        world = dist.get_world_size()
        if any(len(params) % world for params, *_ in self.batches):
            raise ValueError('the rank-major layout needs every ANVIL batch divisible by the world size')
        self.rank_groups = self.shard_units()
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
        """Each group's reduce-scatter output chunk and each batch's shard as a view of it."""
        if self.rank_receives[0] is None or self.rank_receives[0].dtype != wire_dtype:
            self.rank_receives = [torch.zeros(chunk, dtype=wire_dtype, device=self.rank_stores[0].device)
                                  for chunk in self.rank_chunks]
            self.shard_buffers = [self.shard_view(k, self.rank_receives) for k in range(len(self.batches))]

    def rank_major_order(self):
        """The batches' parameters in storage order (groups, rank chunks, batches, shard rows)."""
        world = dist.get_world_size()
        return [p for ks in self.rank_groups for r in range(world) for k in ks
                for p in self.batches[k][0][r * (len(self.batches[k][0]) // world):
                                            (r + 1) * (len(self.batches[k][0]) // world)]]

    def shard_view(self, k, buffers=None):
        """This rank's shard rows of batch k, in its group's chunk of `buffers` (one rank chunk per group)
        or in the weights' storage."""
        params, g = self.batches[k][0], self.rank_group_of[k]
        per = len(params) // dist.get_world_size()
        start = self.rank_offsets[k] + (dist.get_rank() * self.rank_chunks[g] if buffers is None else 0)
        source = self.rank_stores[g] if buffers is None else buffers[g]
        return source[start:start + per * params[0].numel()].view(per, *params[0].shape)

    def polar_partners(self):
        """Groups of batch indices whose polar inputs (polar_batch's transpose of tall shapes) share a non-square
        shape: each group's polar runs as one."""
        groups = {}
        for k, (params, *_) in enumerate(self.batches):
            rows, cols = params[0].shape
            if rows != cols and self.batch_heads[k] == 1:
                groups.setdefault((min(rows, cols), max(rows, cols)), []).append(k)
        return [ks for ks in groups.values() if len(ks) > 1]

    def shard_units(self):
        """The sharded step's units in step order: each polar-partner group at its last batch, the others alone."""
        merged = {k: ks for ks in self.polar_partners() for k in ks}
        return [tuple(merged.get(k, (k,))) for k in range(len(self.batches)) if k not in merged or k == merged[k][-1]]

    def unit_sources(self, ks, scalars):
        """This rank's new shard rows of batches ks as all-gather sources, by the ops (and compiled
        functions) of step's sharded path."""
        world, rank = dist.get_world_size(), dist.get_rank()
        views, slices, weights = [], [], []
        for k in ks:
            params = self.batches[k][0]
            per = len(params) // world
            slices.append(slice(rank * per, (rank + 1) * per))
            views.append(None)
            weights.append(self.shard_view(k))
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
        sends = [self.shard_view(k, self.rank_sends) for k in ks]
        for send, new in zip(sends, news):
            send.copy_(new)
        return sends

    @torch.no_grad()
    def capture_shard_graphs(self, wire_dtype):
        """Persistent reduce-scatter shards and one CUDA graph per shard unit
        (the polar and state update; the collectives stay eager), the scalars read from a static row. The eager
        warm-up's state changes are undone."""
        self.receive_rank_major(wire_dtype)
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

    def rank_major_step(self):
        """Per unit, wait its reduce-scatter and run it (its graph, or eagerly before the graphs are captured); the
        last unit (the merged polar) on this stream, the others on a side stream at the same time. Then the
        all-gathers (one per unit's group, the last unit's first, each issued from the stream that ran it)."""
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
        streams = {}
        side = self._unit_side = getattr(self, '_unit_side', None) or torch.cuda.Stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for n in range(len(units) - 1):
                run(n)
                streams[n] = side
        run(len(units) - 1)
        streams[len(units) - 1] = main
        last_unit = {g: max(n for n, u in enumerate(units) if set(u) & set(ks))
                     for g, ks in enumerate(self.rank_groups)}
        for g in sorted(last_unit, key=lambda g: -last_unit[g]):
            with torch.cuda.stream(streams[last_unit[g]]):
                self.rank_gather(g)
        main.wait_stream(side)
        self.shard_grads = None
        self.t += 1

    def rank_gather(self, g):
        work = dist.all_gather_into_tensor(self.rank_stores[g], self.rank_sends[g], async_op=True)
        if work is not None:
            self.gather_works = getattr(self, 'gather_works', []) + [work]

    @torch.no_grad()
    def step(self):
        """This rank's shard rows of every batch, from the gradients' reduce-scatter (Cut.update)."""
        if getattr(self, 'shard_grads', None) is None:
            raise RuntimeError('BatchedAnvil steps from the reduce-scattered shards only')
        if getattr(self, '_unit_graphs', None):
            self._scalars.copy_(self.scalar_schedule[self.t])
        return self.rank_major_step()


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
    def zero_grad(self, set_to_none):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
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
    adam = ExactAdamW(
      [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
      betas=recipe.betas, eps=1e-8, weight_decay=0.0, fused=True,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
      polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    anvil.head_ids = {id(getattr(block.attn, f'c_{kind}').weight): model.config.n_head
                      for block in model.transformer.h for kind in HEAD_POLAR}
    if tuple(HEAD_POLAR) != ('q', 'k'):
        raise ValueError("the joint head polar needs HEAD_POLAR=('q', 'k')")
    # ANVIL weight decay is scaled by late_cut_decay_multiplier (zero) after the first 275/1043 of the Cut updates.
    late_from = round(275 / 1043 * steps)
    decay_factors = tuple(recipe.late_cut_decay_multiplier if t >= late_from else 1.0 for t in range(steps))
    anvil.prepare_scalar_schedule(recipe.warmdown, recipe.final, decay_factors)
    return Optimizers(anvil, adam)


def cut_updates(steps, settings):
    """The Cut's updates as tuples of the cut_batch-row steps (1-based) each takes: one step each until wide_from of
    them, then wide_rows // cut_batch each (the first wide update moved later so that the wide updates fill the
    rest)."""
    start, per = round(settings.wide_from * steps), settings.wide_rows // settings.cut_batch
    first = steps - (steps - min(start, steps)) // per * per
    return [(t,) for t in range(1, first + 1)] + [tuple(range(t, t + per)) for t in range(first + 1, steps + 1, per)]


def widen_anvil_schedule(optimizer, updates, settings):
    """ANVIL's per-update scalars for these updates: each the mean of its steps' rows, the rate scaled in wide ones."""
    anvil = optimizer.optimizers[0]
    table, rows = anvil.scalar_schedule, []
    for group in updates:
        row = table[group[0] - 1:group[-1]].mean(0)
        if len(group) > 1:
            row[5] *= settings.wide_anvil_scale
        rows.append(row)
    anvil.scalar_schedule = torch.stack(rows)


def set_rate(optimizer, step, steps, warmdown, final):
    left = (steps - step + 1) / steps
    for group in optimizer.optimizers[1].param_groups:  # AdamW; ANVIL's rates are precomputed in its scalar schedule
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown))


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 64
MERGE_FROM, MERGE_EVERY = 0.3, 5
MERGE_DECAYS = (0.02,)
def holdout_sum(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, y, loss_reduction='sum')


holdout_sum_captured = torch.compile(holdout_sum, dynamic=False)


class ResidentMergeScore:
    """Held-out loss on the held-out rows (split across ranks). The forward and all-reduce are captured once as a
    CUDA graph on random tokens before the timer; bind() loads the held-out rows into the graph inputs."""
    def __init__(self, model, rank, world, sequence_len):
        self.model, self.rank, self.world = model, rank, world
        self.device = next(model.parameters()).device
        self.buffers, self.graphs = {}, {}
        for count in (HOLDOUT,):
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
    scorer = ResidentMergeScore(model, rank, world, settings.model.sequence_len)
    scorer.bind(noise)
    scorer.capture()
    quad_warmup(model, noise[:QUAD_ROWS].to(scorer.device), rank, world)
    return scorer


# The end merge as per-part coefficients a in w + a (avg - w), chosen inside the run. The parts are split over the
# ranks; each rank fits a parabola through a = C0 - DELTA, C0, C0 + DELTA for each of its parts on its own QUAD_ROWS
# rows (one forward from the part's first block, the stream below it cached from the C0 forward), one all-reduce
# shares the vertices (clipped to QUAD_RANGE), and the joint move C0 -> vertices is shrunk by the best of QUAD_SHRINKS
# on the 64 held-out rows.
QUAD_DECAY, QUAD_C0, QUAD_DELTA, QUAD_RANGE = 0.02, 0.75, 0.5, (-0.25, 1.25)
QUAD_ROWS, QUAD_SHRINKS = 24, (0.25, 0.5, 0.75)  # pick rows per rank: the held-out rows from rank * 8 on (wrapping)
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
SHAPING_ON = True  # (set while each graph is traced: False drops the mechanisms, a no-op from update 195 on)
VE_COMMON_GAIN = 4.0


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


def shape_values_closed(deltas, frames, step_t):
    """Each value-embedding table's vocabulary-common Adam motion amplified VE_COMMON_GAIN-fold at fixed energy, then
    transfer_partner_motion, in closed form.
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
        self.gather_group = dist.new_group(pg_options=nccl_options())
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
        u = hidden.reshape(-1, hidden.shape[-1]).float()
        steps.append(FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1)), min=1e-6))
    cache['fc_steps'] = torch.stack(steps)
    final = id_norm(x + pending.float()).to(BF16).reshape(b * t, D)
    raw = final @ model.lm_head_bf
    cache['in.lm_head'], cache['lm_head'] = final, raw
    loss = F.cross_entropy(softcap(raw.float()), targets.reshape(-1))
    return cache, loss


forward = torch.compile(forward_impl, dynamic=False, options={'combo_kernels': True})


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
def cycle_scales(x, h):
    """The readout cycle's per-token step, norm inverses and their ratios."""
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
    """The readout cycle's per-logit part for one token in one pass: f g1 (fp32), the five bf16 weight rows, and the row
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
    """The readout cycle's weights, its per-logit part in one kernel pass."""
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
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = cycle_weights_fused(
        x, base, h)
    exact = fg1 @ c.T
    cb, c2, c3 = cycle_powers(c)
    n = x.shape[0]
    p01 = (fast[:2].reshape(2 * n, -1) @ cb.T).view(2, n, -1)
    p23 = (fast[2:4].reshape(2 * n, -1) @ c2.T).view(2, n, -1)
    prods = [p01[0], p01[1], p23[0], p23[1], fast[4] @ c3.T]
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def rope32(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


REPEAT, REPEAT_K = HD // 16, HD // 32
PACK = 5 * HD + 2


@triton.jit
def _pack_heads_kernel(Q, K, V, O, TG, LSE, Out16, Out32, layer, sq, sk, sv, so, st, L: tl.constexpr,
                       M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr):
    """One layer's rows of the (H, L, M, PACK) all_to_all send buffer for BM tokens of head h: the five 64-wide pieces
    and the lse's fp32 bits (LSE (rows, H, T) fp32 as int32), copied as raw 16-bit words."""
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
    """identity_grad's (H, L, M, PACK) all_to_all send buffer (each head's q, k, v, output and target pieces and its lse
    bits) from each layer's (tokens, D) pieces and its lse (rows, H, t), one launch per layer."""
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


@triton.jit
def _head_prep_kernel(Got, Got32, Cos, Sin, Q16, K16, V16, Hub16, Q32, K32, LSE, HubOut, SqQ, SqK, StepsQ, StepsK,
                      W: tl.constexpr, L: tl.constexpr, N: tl.constexpr, T: tl.constexpr, PACK: tl.constexpr,
                      SIGMA: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr, F2=None, F4=None, XY=None,
                      PAIRS: tl.constexpr = False):
    """head_prep_fused for BM tokens of rank w's layer l (got (W, L, N, PACK)): output token (l * W + w) * N + i."""
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
    """The received heads' RoPE'd queries and keys, values, hubs, lse and square norms and steps from got (W, L, N,
    PACK), in one kernel pass."""
    w, layers, n, _ = got.shape
    rows = layers * b
    new = lambda *s, dtype=torch.float32: torch.empty(*s, device=got.device, dtype=dtype)
    q16, k16, v16, hub16 = (new(rows, t, HD, dtype=BF16) for _ in range(4))
    q32, k32 = new(rows, t, HD), new(rows, t, HD)
    lse, hub_out, sq_q, sq_k, steps_q, steps_k = (new(rows, t) for _ in range(6))
    pairs = (new(rows, t, HD, dtype=BF16), new(rows, t, 2 * HD, dtype=BF16), new(rows, t, HD // 2, dtype=BF16))
    _head_prep_kernel[(n // 32, w * layers)](got, got.view(torch.float32), cos, sin, q16, k16, v16, hub16, q32, k32,
                                             lse, hub_out, sq_q, sq_k, steps_q, steps_k, W=w, L=layers, N=n, T=t,
                                             PACK=PACK, SIGMA=KEY_SIGMA, EPS=NORM_EPS, BM=32, F2=pairs[0],
                                             F4=pairs[1], XY=pairs[2], PAIRS=True, num_warps=4)
    return q16, k16, v16, hub16, q32, pairs, lse, hub_out, sq_q, sq_k, steps_q, steps_k


def id_head_scores(got, rope32, back, b, t):
    q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep_fused(got, *rope32, b, t)
    moment_kernels._moment_query_scores(q1, k1, values, hub, got, sq_q, None, *rope32, lse, hub_out, steps_q,
                                        back[:, :, 0], REPEAT, pair_ops=k1f)
    moment_kernels.moment_key_scores(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k,
                                     back[:, :, 1], back[:, :, 2], REPEAT_K)


def f32_stack(xs):
    return torch.stack([x.reshape(-1, x.shape[-1]) for x in xs]).float()


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


def hidden_credits(us, credb, w_out_t, steps=None):
    return id_kernels.hidden_credits_fused(us, steps, credb, w_out_t)


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


@torch.compile(dynamic=False)
def value_embed_credits(z, ve, vcredit, n1):
    z, ve, vcredit = z.float(), ve.float().view(-1, H, HD), vcredit.float()
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    gate = n1[:, :GATE_CHANNELS].float().T @ (slope * (vcredit * ve).sum(-1))
    return gate, (vcredit * 2 * torch.sigmoid(z)[..., None]).reshape(-1, D)


# identity_grad's collectives: the wide gradients' reduce-scatter is issued as soon as they are complete (in heads_in's
# window, where the collectives' stream idles), and their ANVIL update and all-gather run on a side stream from there
# (the step's last read of wide_bf, hidden_credits, precedes the fork). The ANVIL all-gathers use a second
# communicator, so they do not queue behind the flat all-reduce.
def id_gather_group(model):
    return model.gather_group


def early_wide_update(model, work, n):
    """The wide matrices' ANVIL update and all-gather on a side stream once their reduce-scatter
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
    send = pack_heads_kernel([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
                             [mine(cache[f'h{i}.values']) for i in range(L)],
                             [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)],
                             [targets_o[i] for i in range(L)], [cache[f'h{i}.lse'] for i in range(L)])
    got = torch.empty_like(send)
    heads_in = dist.all_to_all_single(got, send, async_op=True)
    coords = coordinate_change(cache['lm_head'], y)
    prevs = f32_stack([cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)])
    n1s, os_, n2s, acts = ([mine(cache[f'h{i}.{key}']) for i in range(L)]
                           for key in ('in.attn', 'in.attn.c_proj', 'in.mlp.c_fc', 'in.mlp.c_proj'))
    hidden = lambda: hidden_credits([cache[f'h{i}.mlp.c_fc'] for i in range(L)], credb, model.wide_bf[L:],
                                    cache.get('fc_steps'))
    wide = lambda: (id_kernels.at_b_layers(n2s, list(hidden()), gwide[:L]),
                    id_kernels.at_b_layers(list(credb), acts, gwide[L:]))
    wide()
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    works = {}
    works['wide'] = dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True)
    works['wide_update'] = early_wide_update(model, works['wide'], n)
    heads_in.wait()
    back = torch.empty(world, L, 3, m, HD, device=idx.device, dtype=BF16)
    id_head_scores(got, rope32, back, rows_total, t)
    got2 = torch.empty_like(back)
    heads_out = dist.all_to_all_single(got2, back, async_op=True)
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    id_kernels.at_b_layers(os_, list(credb), gsq[3 * L:])
    heads_out.wait()
    id_kernels.square_credit_grads(n1s, got2, gsq[:3 * L])
    vcredits = [got2[:, i, 2].permute(1, 0, 2) for i in VE_BLOCKS]
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
    return lo + (hi - lo) * t / warm
ID_DECAY = 150  # the identity phase's last ID_DECAY updates: its Adam and ANVIL rates fall linearly towards ID_DECAY_FINAL
ID_DECAY_FINAL = 0.3


def id_rate_scale(step_t):
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


def id_anvil_update(model, works, n):
    gathers = []
    main = torch.cuda.current_stream()
    side = None
    if not hasattr(model, 'side_stream'):
        model.side_stream = torch.cuda.Stream()
    side = model.side_stream
    side.wait_stream(main)
    for name, w, w_bf in (('square', model.square, model.square_bf), ('wide', model.wide, model.wide_bf)) if side \
            else (('wide', model.wide, model.wide_bf), ('square', model.square, model.square_bf)):
        if name == 'wide' and 'wide_update' in works:  # (issued in identity_grad)
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
            value_deltas.append(delta)
        else:
            p.copy_((p.float() - lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)
    if SHAPING_ON:
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
        anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD, rate_decay=dict(last_updates=ID_DECAY, final=ID_DECAY_FINAL))
    def run(self):
        for i in range(IDENTITY_UPDATES):
            (self.plain_graph if self.plain_graph is not None and i >= SHAPING_UPDATES else self.graph).replay()
        self.model.gather_fp32()
    def close(self):
        """Drops the graphs and the model; their blocks stay in the caching allocator for the Cut (no cudaFree in the
        timer)."""
        self.graph = self.plain_graph = None
        self.model = None


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
                        'identity_handoff_at_315', 'identity_init_and_early_update_shaping', 'identity_rate_decay',
                        'difficulty_balanced_row_order', 'single_chain_roots_4_6', 'amortized_block4_root',
                        'reduced_forward_replay', 'diagonal_readout_moments', 'x0_skip_embedding_credit',
                        'fused_forward_kernels', 'late_zero_anvil_decay', 'in_run_part_merge',
                        'cut_key_degree2_moments', 'cut_query_degree2_moments', 'readout_probe_moments',
                        'factored_query_replay', 'cut_mixing_scalar_contraction', 'cut_estimate_cuda_graph',
                        'resident_merge_scorer', 'response_ranked_mlp_routes', 'batched_update_telemetry',
                        'fused_gluon_chain_kernels', 'gluon_key_value_kernels', 'bf16_weight_credits',
                        'sharded_anvil_bf16_async_collectives', 'rank_major_anvil_graphs', 'joint_qk_head_polar',
                        'split_identity_graph', 'paired_wide_cut_updates', 'nccl_ll128',
                        'bidirectional_chain_tail_lerp', 'soft_rms_cross_lanes',
                    ],
                    torch_version=torch.__version__,
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
    estimator.bind_flat_layout(optimizer)  # (the graph packs the gradients in step's layout)
    if not estimator.packs_gradients():
        raise ValueError('AMORTIZED_ROOTS needs packed gradients')
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
    estimator.reset_amortization()
    estimator.amortized_window(len(updates))
    del pair_rows
    for cut in (estimator, estimator.partner):
        if cut is not None:
            cut.capture_gate_graph()
    if estimator.packs_gradients():
        optimizer.optimizers[0].capture_shard_graphs(torch.bfloat16)
    log = (args.output / 'metrics.jsonl').open('w') if rank == 0 else None
    scorer = prepare_resident_merge_score(model, settings, rank, world)
    del noise, graph_batch
    # Untimed, as in #44 (qlabs-eng/noise@3343973b, train_stages): read train.pt, split it by the seeded permutations
    # above and put the identity phase's batches on the GPU. No split or order here depends on the rows' contents;
    # load_rows only verifies their pinned hash.
    all_rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    cut_rows = all_rows.index_select(0, train_indices)
    holdout_rows = all_rows.index_select(0, holdout_indices)
    # the in-run merge's pick rows: QUAD_ROWS held-out rows from row 8 * rank on, wrapping (no row the merge reads feeds
    # an update)
    quad_rows = holdout_rows.roll(-rank * (HOLDOUT // world), 0)[:QUAD_ROWS]
    identity.bind_rows(all_rows.numpy())
    del all_rows
    staged_rows = torch.empty((train_count, cut_rows.shape[1]), dtype=cut_rows.dtype, pin_memory=True)
    device_rows = torch.empty(staged_rows.shape, dtype=staged_rows.dtype, device=device)
    copy_stream = torch.cuda.Stream()
    torch.empty(int(CUT_RESERVE_GB * 2 ** 30), dtype=torch.uint8, device=device)
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.freeze()
    begin_training()
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    def prepare_rows():
        """The timed host setup of the Cut rows (no CUDA calls): their order, the unused and quad rows, the ordered rows
        (into the pinned staging buffer), the disjointness check and row_usage.json."""
        order = torch.from_numpy(difficulty_order(cut_rows, settings))  # timed: it reads the rows' tokens
        ordered = train_indices[order]
        unused, used = ordered[train_count:], ordered[:train_count]
        rows = torch.index_select(cut_rows, 0, order[:train_count], out=staged_rows)
        all_used = torch.cat([identity_indices, used, holdout_indices, unused])
        assert len(torch.unique(all_used)) == len(all_used) == TRAIN_ROWS
        if rank == 0:
            digest = lambda ids: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            usage = dict(identity_rows=len(identity_indices), cut_rows=train_count, merge_rows=len(holdout_indices),
                         unused_rows=len(unused), unused_sha256=digest(unused), disjoint=True,
                         identity_sha256=digest(identity_indices), cut_sha256=digest(used),
                         merge_sha256=digest(holdout_indices),
                         merge_pick_rows=f'{QUAD_ROWS} of merge_rows per rank, from row 8 * rank on (wrapping)')
            (args.output / 'row_usage.json').write_text(json.dumps(usage, indent=2) + '\n')
        return rows

    try:
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
        rows = box['rows']
        with torch.cuda.stream(copy_stream):
            training = device_rows.copy_(rows, non_blocking=True)
        torch.cuda.current_stream().wait_stream(copy_stream)
        del cut_rows, rows
        with torch.no_grad():
            weights = model.state_dict()
            for name, value in identity.model.state_dict().items():
                weights[name].copy_(value)
        identity_record = dict(step=IDENTITY_UPDATES, phase='identity', phase_step=IDENTITY_UPDATES,
                               seqs=IDENTITY_UPDATES * BATCH,
                               tokens=IDENTITY_UPDATES * BATCH * settings.model.sequence_len)
        def write_identity_record(loss, elapsed):
            identity_record.update(train_loss=loss, train_seconds=round(elapsed, 2))
            if rank == 0:
                print(json.dumps(identity_record), flush=True)
                log.write(json.dumps(identity_record) + '\n')
                log.flush()
        identity_loss = average_ranks(identity.model.loss.clone())
        identity_end = torch.cuda.Event(enable_timing=True)
        identity_end.record()
        pending_records = []
        means_stream = torch.cuda.Stream()
        for step, group in enumerate(updates, 1):
            begin, end = (group[0] - 1) * settings.cut_batch, group[-1] * settings.cut_batch
            # a wide update's rate: the mean of its steps' (the schedule is linear within one), scaled
            set_rate(optimizer, group[0] if len(group) == 1 else sum(group) / len(group), horizon, settings.warmdown,
                     settings.final)
            if len(group) > 1:
                for param_group in optimizer.optimizers[1].param_groups:
                    param_group['lr'] *= settings.wide_lr_scale
            batch = rank_slice(training[begin:end])
            # views of the device rows, copied into the graphs' inputs on the staging stream
            rows_in, rows_out = (batch[:, :-1], batch[:, 1:])
            loss = (estimator.wide_step if len(group) > 1 else estimator.step)(rows_in, rows_out, optimizer)
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
                losses = (losses).cpu().tolist()
                if any(not math.isfinite(value) for value in losses):
                    raise FloatingPointError('Non-finite training loss')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if 'train_loss' not in identity_record:
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
                    else:  # the next update's step waits for it (Cut.update)
                        means_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(means_stream):
                            current = [q.detach().float() for q in model.parameters()]
                            for a, mean in zip(MERGE_DECAYS, means):
                                torch._foreach_lerp_(mean, current, a)
                            estimator._means_done = torch.cuda.Event()
                            estimator._means_done.record()
                        del current
        if getattr(estimator, '_means_done', None) is not None:  # the last lerp
            torch.cuda.current_stream().wait_event(estimator._means_done)
            estimator._means_done = None
        extra = {}
        if means:
            merge_start = time.perf_counter()
            scorer.bind(holdout_rows)
            extra.update(quad_merge(model, means, quad_rows.to(device), rank, world, scorer))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
            means = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        end_training()
        identity.close()
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
