"""Track 2 (30M): 3.4948 in 479.34 s (7 min 59 s), adding full multi-layer linear x0 skip pullback into the
embedding credit and a 10x finer readout finite-difference step (sigma = 0.01) to the identity-path -> Cut-lite
recipe (#4586b02).

What changes against the previous 8 min 20 s record (3.4988 in 499.62 s, below):
  - Multi-layer linear x0 pullback (`emb_credit = 'mid0'`): every block i mixes the initial normalized embedding x0 =
    norm(e) linearly into its input stream as `lam_r[i] * x_{i-1} + lam_0[i] * x0`. Previously only block 0's measured
    mid-stream credit `(lam_r[0] + lam_0[0]) * credits[0]` was pulled back to x0; now all blocks' measured mid-stream
    credits contribute along their direct linear `lam_0[i]` skip paths (`x0_credit = (lam_r[0] + lam_0[0]) * credits[0]
    + sum_{i=1}^{L-1} lam_0[i] * credits[i]`) before taking the single nonlinear step through the input RMSNorm
    (`norm_input_credit`). Zero extra forward evaluations.
  - Readout coordinate finite-difference step (`head_error(0.01)` instead of `0.1`): reduces third-order Taylor
    truncation error on the soft-capped logits `15 * tanh(z / 15)` by 100x at zero compute cost.
  - IdentityPhase graph capture scoped to `max_run`: skips untimed warmup/capture of full-length and EMA graphs that
    lie beyond `--identity-updates`.

The previous record's description:

The 10M record's identity-path trainer on 5,280 rows, then Cut without its future-token and input-key
probes at 8 rows per update, and the end merge: 3.5 in ~8 min (the previous record: 1 h 43 min).

What changes against the previous record (Boost -> Cut with input-K/gate and future-token probes, below):
  - An identity-path phase first (the section below: the 10M track's record trainer, its CUDA graphs captured before the
    timer): 660 updates of 8 rows, of a 1,831-update schedule (it hands off at full rate, its own EMA and skips
    as planned for that length), ~3 s. Its weights become Cut's starting point; its rows feed no Cut update. No Boost.
  - Cut-lite: no future-token probes and no input-key probes (`future_draws = key_path_draws = 0`); the chain credits
    (own-query keys, exact values, routed MLPs) carry the internal credit. 8 rows per update, one sequence per GPU, at
    the original rates (Adam 0.0021, ANVIL 0.026), Adam multipliers x750 on the tables and x15 on the scalars. The
    embedding's credit comes from block 0's measured mid-stream credit, one finite step back through the input norm
    (`emb_credit = 'mid0'`), instead of its own chain root. Chain reruns keep 32 keys per moved query and 192 MLP
    units per token.
  - Chain-rerun kernels (each checked against the previous estimate at a checkpoint: cosine >= 0.9999 per tensor):
    one exponential per logit in the routed own-query response and its projection in bf16 (`chain_fast`); the moved
    stream stored in bf16 between blocks (`chain_act`); the readout's log-sum-exp change from per-token second
    moments of the clean readout (`head_moments`, its W^T diag(c) W in one Triton GEMM, `moment_kernel`); the moved
    queries rotated and renormalized in one Triton pass (`fused_query`); the RMSNorm change folded into the next
    projection, inv * (delta W) + (inv / inv0 - 1) * clean (`fold_norm`); each root's moves built inside the compiled
    chunk (`fused_moves`); 1,024-token chunks; the input-key response kernel tiled over probes x sources.
  - End merge on the last 64 training rows (32), its first sweep over the 8 best keys, loading only the parts a trial
    changes, the held-out forward compiled with CUDA graphs on random tokens before the timer (`merge_graph`): ~6 s.

Every signal is still measured forward against the loss and takes at most one step through a nonlinear part.

The previous record's description:

Boost -> Cut with independent input-K/gate and future-token probes (#31), trained at batch 4 on two GPUs per
sequence with the backprop baseline's recipe, and merged at the end along its own trajectory.

The own-query and value-path measurements of PR #23 remain the starting point.
Input-K/gate probes supply additional input credit to linear consumers. Separate
single-source hidden/Q/embedding probes are scored at later, original measured
attention hubs. Corrected credits never become another probe's scoring target.

This record adds to #31 (nothing here changes what is measured or how):
  - Record #11's recipe: Cut at batch 4, each sequence split over two GPUs (each takes half of every cycle of moves
    and draws; the pair adds its parts, equal to one GPU), 0.7x rates, Adam betas (0.5, 0.98), a warmdown over the
    last 40% that ends at 30% of each rate (the end merge anneals the rest), and the LM head initialized at std 0.02.
  - Future probes: 128 central pairs per source token (#31: 512), in larger token chunks.
  - An end merge along the trajectory. The last 32 training sequences feed no update: they only score the merge. From
    60% of the epoch, every 5 updates, nine exponential means of the weights are kept (weight 0.5 ... 0.01 on the
    newest). After the last update, for each part (each block matrix, each table, the scalars) a candidate moves the
    final weights toward one mean, in full inside the part's 64 top input-covariance directions (for a table, the rows
    of the 512 most frequent tokens; both from forward passes on the held-out rows) and a fraction 0, 1/2 or 1 outside,
    times 0.75, 1 or 1.25: 81 candidates and no move. The best single candidate for every part comes first, then two
    coordinate sweeps over the parts (the second over each part's best 8) keep any change that lowers the held-out
    loss. About 5,000 forward passes on 32 sequences (~1 min), inside the timed region.

This recipe uses 20% Boost (global batch 4), then Cut (global batch 4), consuming
every training sequence but the last 32 once (those score the end merge). Future probes use all
permitted receivers and the first four hidden/Q layers plus the embedding.
No receiver sampling or active-space direction encoding is used.

    uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from dataclasses import asdict, dataclass, field
from functools import cache, partial
from math import gcd
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Optional

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate

SEED = 0
IDENTITY_UPDATES, IDENTITY_SCHEDULE = 660, 1831  # the identity-path phase: updates run of its planned schedule
EPS = torch.finfo(torch.float32).eps
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


def model_config():
    """The fixed GPT topology used by the current val experiment."""
    return GPTConfig(sequence_len=2048, vocab_size=4096, n_layer=8, n_head=8,
                     n_kv_head=8, n_embd=512, window_pattern='SSSL')


@dataclass
class Settings:
    """Own-query/value-path credit, full hidden FD, and optional independent input-K probes."""
    model: GPTConfig = field(default_factory=model_config)
    estimator: str = 'cut'
    boost_hub: str = field(default='projected', init=False)  # persisted algorithm version, not an ablation flag
    mlp_output_credit: str = field(default='next_mid', init=False)
    switch: float = 0.0  # no Boost: the identity-path phase (IdentityPhase) comes first, then Cut
    boost_batch: int = 4
    cut_batch: int = 8
    projection_std: float = 1.5
    lr: float = 0.0021
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.4
    final: float = 0.3  # the warmdown ends at this fraction of each rate; the end merge anneals the rest
    head_std: float = 0.02  # the LM head's init std (the fixed init's 0.001)
    anvil_lr: float = 0.026
    anvil_wd: float = 2.25
    fast_weight: float = 1.0  # ANVIL fast rail
    polar_margin: float = 1.1
    sigma: float = 0.005
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    query_central: bool = False  # same directions, paired +/- attention evaluations
    nrep: int = 64
    emb_chunk: int = 16
    scalar_step: float = 0.01
    chunk: int = 2 ** 27
    chain_sigma: float = 0.01
    route_k: int = 192
    chain_tokens: int = 1024
    chain_precision: str = 'bf16'
    own_query: bool = True  # PR #19's measured own-query paths in later attention blocks
    value_path: bool = True  # PR #23: one RMSNorm step, then only linear writers/V/VE
    path_scalars: bool = True  # also contract input-path credit into the linear mixing coefficients
    key_path_draws: int = 0  # independent input-K central pairs per token (0: off)
    key_path_sigma: float = 0.01  # relative input RMS step
    key_path_tokens: int = 64  # source-token chunk; every allowed query contributes
    gate_path: bool = True  # independent full-cycle input probes through the VE gate
    gate_path_sigma: float = 0.01  # relative input RMS step
    future_draws: int = 0  # independent hidden/Q/WTE probes scored at every later raw hub (0: off)
    future_layers: int = 4  # first blocks whose hidden and Q outputs receive this correction
    future_tokens: int = 128
    future_chunk: int = 512
    future_sigma: float = 0.01  # source-relative RMS; Q includes its own complete MLP response
    query_keys: int = 32   # moving key logits per head; 0 reruns all allowed logits
    key_sigma: float = 0.003
    key_chunk: int = 32
    merge_keys: int = 8  # end merge: keys per part in the first sweep (0 = all 82)
    merge_graph: bool = True  # end merge: the held-out forward compiled with CUDA graphs (captured before the timer)
    chain_fast: bool = True  # routed own-query responses with one exponential per logit, their projection in bf16
    chain_act: str = 'bf16'  # 'bf16': the chain reruns' moved-stream changes stored in bf16, the query projection too
    fused_query: bool = True  # chain reruns: rotate + renormalize the moved queries in one Triton pass (bf16 moves)
    fold_norm: bool = True  # chain reruns: the norm's change folded into the next projection (d's, rescaled, + clean)
    fused_moves: bool = True  # chain reruns: each root's moves built inside the compiled chunk (no fp32 copies)
    head_moments: bool = True  # chain reruns: the readout's partition-function change from per-token moments
    moment_kernel: bool = True  # head_moments: each token's W^T diag(c) W in one Triton GEMM (no scaled copies)
    emb_credit: str = 'mid0'  # 'mid0': the embedding's credit from block 0's measured one (no embedding chain root)
    emb_scale: float = 750.0  # Adam rate multiplier of the embedding tables
    lambda_scale: float = 15.0  # Adam rate multiplier of the residual scalars

    def __post_init__(self):
        width, hidden = self.model.n_embd, 4 * self.model.n_embd
        if self.chain_act not in ('fp32', 'bf16'):
            raise ValueError('Unknown chain activation dtype')
        if self.emb_credit not in ('chain', 'mid0'):
            raise ValueError('Unknown embedding credit')
        if self.estimator not in ('boost', 'cut') or not 0 <= self.switch <= 1:
            raise ValueError('Invalid stage estimator or Boost fraction')
        if not 0 <= self.fast_weight <= 1:
            raise ValueError('ANVIL fast weight must be in [0,1]')
        if width < 1 or width & (width - 1):
            raise ValueError('Full cycles require power-of-two model width')
        if not 1 <= self.route_k <= hidden or self.chain_precision not in ('bf16', 'fp32'):
            raise ValueError('Invalid sparse route budget or chain precision')
        if self.query_draws != width // self.model.n_head:
            raise ValueError('Queries use one complete cycle per head')
        if not 0 < self.query_sigma < float('inf'):
            raise ValueError('Query step must be finite and positive')
        if not 0 < self.gate_path_sigma < float('inf') or (self.gate_path and self.estimator != 'cut'):
            raise ValueError('Gate input probes require Cut and a finite positive step')
        if (type(self.future_draws) is not int or not 0 <= self.future_draws <= width
                or (self.future_draws and self.estimator != 'cut')):
            raise ValueError('Future probes require Cut and a budget no greater than model width')
        if (type(self.future_layers) is not int or self.future_layers < 1
                or (self.future_draws and self.future_layers > self.model.n_layer)
                or type(self.future_tokens) is not int or self.future_tokens < 1
                or type(self.future_chunk) is not int or self.future_chunk < 1
                or not 0 < self.future_sigma < float('inf')):
            raise ValueError('Invalid future source layers, batching limits or finite step')
        if not isinstance(self.query_keys, int) or self.query_keys < 0:
            raise ValueError('Own-query key count must be a nonnegative integer')
        if type(self.key_path_draws) is not int or not 0 <= self.key_path_draws <= width:
            raise ValueError('Input-K pairs must be an integer between zero and model width')
        if (type(self.key_path_tokens) is not int or self.key_path_tokens < 1
                or not 0 < self.key_path_sigma < float('inf')):
            raise ValueError('Input-K chunk and finite step must be positive')
        if any(v <= 0 for v in (self.nrep, self.emb_chunk, self.chain_tokens, self.key_chunk, self.chunk,
                               self.sigma, self.fc_sigma, self.chain_sigma, self.key_sigma, self.scalar_step)):
            raise ValueError('Probe sizes and batching limits must be positive')

    @property
    def chain_dtype(self):
        return torch.bfloat16 if self.chain_precision == 'bf16' else torch.float32


# ---------------------------------------------------------------------------------------------------------- helpers


@torch.compile
def coordinate_differences(raw, y, sigma):
    """Finite differences of each token's loss along each of its own logits: for every token and every logit j,
    the loss with raw logit j moved up by sigma minus the loss with it moved down by sigma, over 2 sigma.

    Each of the two is a full forward evaluation of the soft-capped cross-entropy at the moved logit, written in
    closed form: moving one logit changes the partition function Z = sum_k exp(z_k) by one term, so the moved
    loss differs from the clean loss by log(Z - exp(z_j) + exp(z_j')) - log Z = log1p(exp(z_j) / Z * expm1(z_j' - z_j))
    in the partition and by -(z_j' - z_j) at the target logit. Nothing is differentiated: exp(z_j) / Z are the
    clean forward's probabilities, used only to evaluate the perturbed losses. Zero at ignored tokens (y < 0)."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)  # exp(z_j) / Z of the clean forward
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))  # loss(z_j -> new) - loss(z), partition
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))  # the target logit's own term
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1, built once per process."""
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)  # +-1 is exact in bfloat16
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


def attention_state(qraw, kraw, values, cos, sin, nhead, window):
    """Dense clean forward context for independently scored local attention trials."""
    def heads(raw):
        shaped = raw.reshape(*raw.shape[:2], nhead, -1)
        return norm(apply_rotary_emb(shaped, cos, sin)).transpose(1, 2)
    q, k = heads(qraw), heads(kraw)
    pos = torch.arange(qraw.shape[1], device=qraw.device)
    lag = pos[:, None] - pos[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -.5).masked_fill(
        ~allowed, -torch.inf).softmax(-1)
    return q, k, a, a @ values


def chunk_size(draws, cap):
    """Largest divisor of draws up to cap."""
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    """The attention each token would see if only its own state moved: every query attends to the other positions'
    clean keys and values and to its own moved ones, by swapping its own clean key and value through the
    log-sum-exp of one attention over the clean keys and values. Tensors are [batch, token, head, dim]."""
    if not q.is_cuda:
        t = q.shape[1]
        lag = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
        logits = q.transpose(1, 2) @ k_clean.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** .5
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), -torch.inf)
        own = (q * k_own).sum(-1).transpose(1, 2) / q.shape[-1] ** .5
        logits = logits.diagonal_scatter(own, dim1=-2, dim2=-1)
        weights = logits.softmax(-1)
        out = (weights @ v_clean.transpose(1, 2)).transpose(1, 2)
        return out + weights.diagonal(dim1=-2, dim2=-1).transpose(1, 2).unsqueeze(-1) * (v_own-v_clean)
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = flex(q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask,
                    return_lse=True)
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)

frozen_attention_compiled = torch.compile(frozen_attention)

@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)

@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)

@torch.compile(dynamic=False)
def query_probe_heads(raw, a, sigma, cos, sin):
    """Preserve the original finite norm calculation, then group probe heads."""
    moved = moved_heads(raw, a, sigma, cos, sin)
    batch, length, heads, width = raw.shape
    if raw.is_cuda and batch > 1:
        return moved.view(len(a), batch, length, heads, width).permute(1, 3, 0, 2, 4).reshape(
            batch, heads*len(a), length, width)
    return moved


def query_probe_attention(moved, keys, values, window, count, batch):
    """All probes share clean K/V; group probe heads instead of copying K/V."""
    _, length, heads, width = keys.shape
    if not moved.is_cuda or batch == 1:
        repeat = lambda z: z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*batch, length, heads, width)
        return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)
    out = flex(moved, keys.transpose(1, 2), values.transpose(1, 2),
               block_mask=window_mask(window, length, moved.device), enable_gqa=True)
    return out.view(batch, heads, count, length, width).permute(2, 0, 3, 1, 4)

@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    """Score either moved-minus-clean or plus-minus-minus by ordinary broadcasting."""
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)

@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores, directions):
    """Sum scalar probe scores times directions; the caller owns budget normalization.

    scores [..., P], directions [..., P, D] (or shared [P, D]).
    Keep this reduction inside the caller's compiled kernel, without creating
    a materialized score-times-direction tensor.
    """
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def decode_local_response(scores, steps, directions, central=True):
    """Decode independent trials; one-sided scores must already subtract clean."""
    if central:
        probes = directions.shape[-2]
        directional = (scores[..., :probes] - scores[..., probes:]) / (2 * steps)
    else:
        directional = scores / steps
    return decode_local_scores(directional, directions)


@dataclass(frozen=True)
class LocalExperiment:
    """A prepared forward response and its fixed-credit scalar score.

    response(rows, steps) returns concatenated +/- finite responses (central),
    or moved-minus-clean responses (one-sided). score(response) returns
    [..., 2P] or [..., P] independent scalar scores. No gradient or
    recursive credit propagation is implicit in this interface. Exact linear
    and separable local objectives can keep their specialized collapsed form.
    """
    response: Callable
    score: Callable
    central: bool = True

    def __call__(self, rows, steps):
        return decode_local_response(self.score(self.response(rows, steps)), steps, rows, self.central)

@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    """Collapse the full Hadamard central-difference cycle for the separable hidden objective.

    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's
    scalar finite difference. Keep that finite step, including ReLU crossings;
    contract in FP32 instead of rounding every direction's projected response to BF16.
    u and out_weight contain all units (or a GPU's contiguous slice).
    """
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope

@torch.compile(dynamic=False)
def embedding_moves(e, steps, rows, sign, wq, wk, wv, cos, sin):
    n, (b, t, width) = rows.shape[0], e.shape
    hd = 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + sign * steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)
    heads = lambda z: z.view(n * b, t, -1, hd)
    q = norm(apply_rotary_emb(heads(F.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(F.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(F.linear(x0, wv))

@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)

@torch.compile(dynamic=False)
def stage_readout(x0, out, mix, wproj, weight, y):
    xmid = mix * x0 + F.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)

def norm_inverses(x, delta):
    """1 / rms(x) (m, 1) and 1 / rms(x + delta) (m, P), exactly: x (m, W) clean, delta (m, P, W) moves."""
    sq = x.square().sum(-1, keepdim=True)
    if delta.dtype == x.dtype:
        dots = torch.einsum('md,mpd->mp', x, delta)
    else:  # bf16 moves (chain_act): a fused product-sum instead of a mixed-dtype product
        dots = (x.unsqueeze(1) * delta.float()).sum(-1)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    return inv0, torch.rsqrt((sq + 2 * dots + delta.float().square().sum(-1)) / x.shape[-1] + EPS)


def norm_difference(x, delta):
    """norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves."""
    inv0, inv = norm_inverses(x, delta)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


def folded_projection(x, delta, w, clean):
    """(norm(x + delta) - norm(x)) @ w.T without forming the difference (fold_norm): the norm has no weight, so it is
    inv delta @ w.T + (inv / inv0 - 1) clean, with clean = norm(x) @ w.T (m, O). delta's projection in w's dtype,
    batched per token when w is (m, O, W). Returns (m, P, O) fp32."""
    inv0, inv = norm_inverses(x, delta)
    moved = torch.bmm(delta.to(w.dtype), w.transpose(1, 2)) if w.dim() == 3 else delta.to(w.dtype) @ w.T
    return inv.unsqueeze(-1) * moved.float() + (inv / inv0 - 1).unsqueeze(-1) * clean.unsqueeze(1)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    """PR #23's central forward RMSNorm response along v.

    RMSNorm's Jacobian is symmetric: this finite directional response tends
    to its credit pullback. This is one nonlinear step, not an analytic VJP.
    The tiny floor also defines a finite response for a zero input or credit.
    """
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
                 / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    """Token chunks share the direction projections and squared lengths."""
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


def query_change(x, delta, pos, rel, w_q, keys, values, w_proj_t, cos, sin, window):
    """PR #19's finite own-query response; all keys and values stay at their clean values.

    Carry finite changes through both RMSNorms and softmax. The difference form
    preserves small probe responses without replacing them by a local derivative.
    """
    m, probes, _ = delta.shape
    heads, length, width = keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    q0, dq = r0 * inv0, dr * inv + r0 * (inv - inv0)
    lag = rel[:, None] - torch.arange(length, device=rel.device)[None]
    allowed = ((lag >= 0) & (lag <= window))[:, None, None]
    scores = torch.einsum('mphd,htd->mpht', q0.to(keys.dtype), keys).float() * width ** -0.5
    p0 = torch.softmax(scores.masked_fill(~allowed, float('-inf')), -1)
    ds = torch.einsum('mphd,htd->mpht', dq.to(keys.dtype), keys).float() * width ** -0.5
    dlse = torch.log1p((p0 * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mpht,htd->mphd', (p0 * torch.expm1(ds - dlse)).to(values.dtype), values).float()
    return dout.flatten(-2) @ w_proj_t


@cache
def finite_attention_kernels():
    """Finite-response reductions. Neither kernel differentiates an attention operation."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def routed(Q, K, V, P, Rest, Out,
               Probes: tl.constexpr, Heads: tl.constexpr, Keys: tl.constexpr, D: tl.constexpr,
               BP: tl.constexpr, BK: tl.constexpr, ONE_EXP: tl.constexpr):
        token, head = tl.program_id(1) // Heads, tl.program_id(1) % Heads
        probes = tl.program_id(0) * BP + tl.arange(0, BP)
        keys, features = tl.arange(0, BK), tl.arange(0, D)
        q = tl.load(Q + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                    probes[:, None] < Probes, 0)
        offsets = ((token * Heads + head) * Keys + keys[:, None]) * D + features[None, :]
        k = tl.load(K + offsets, keys[:, None] < Keys, 0)
        v = tl.load(V + offsets, keys[:, None] < Keys, 0)
        p = tl.load(P + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        # Match both BF16 matmul output roundings of the original implementation.
        ds = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * (D ** -.5)
        if ONE_EXP:  # one exponential per logit: expm1(ds - dlse) = exp(ds) exp(-dlse) - 1
            grown = tl.exp(ds)
            dlse = libdevice.log1p(tl.sum(p[None, :] * (grown - 1), 1))
            weights = (p[None, :] * (grown * tl.exp(-dlse)[:, None] - 1)).to(v.dtype)
        else:
            dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
            weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(Rest + (token * Heads + head) * D + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(Out + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                 moved.to(Out.dtype.element_ty), probes[:, None] < Probes)

    @triton.jit
    def key_response(Q, DK, A, C, Out,
                     H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                     QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                     AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                     CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                     SourceStart, QueryStart, Window: tl.constexpr,
                     BP: tl.constexpr, BM: tl.constexpr, BQ: tl.constexpr):
        # One program: one head, BP probes x BM sources; the attention and contrast tiles are loaded once per
        # query tile and broadcast over the probes.
        head = tl.program_id(2)
        probe = tl.program_id(1) * BP + tl.arange(0, BP)
        source = tl.program_id(0) * BM + tl.arange(0, BM)
        col = (probe[:, None] * M + source[None, :]).reshape(BP * BM)
        live = ((probe[:, None] < P) & (source[None, :] < M)).reshape(BP * BM)
        features = tl.arange(0, D)
        changed = tl.load(DK + (head * P * M + col[None, :]) * D + features[:, None], live[None, :], 0)
        s0 = tl.program_id(0) * BM
        first = tl.maximum(0, SourceStart + s0 - QueryStart) // BQ * BQ
        last = tl.minimum(T, SourceStart + s0 + BM + Window - QueryStart)
        result = tl.full((BP * BM,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start + tl.arange(0, BQ)
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD,
                        query[:, None] < T, 0)
            delta = tl.dot(q, changed, input_precision='tf32x3') * (D ** -.5)
            keep = (query[:, None] < T) & (source[None, :] < M)
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, keep, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, keep, 0)
            a = tl.broadcast_to(a[:, None, :], (BQ, BP, BM)).reshape(BQ, BP * BM)
            c = tl.broadcast_to(c[:, None, :], (BQ, BP, BM)).reshape(BQ, BP * BM)
            change = a * libdevice.expm1(delta)
            result += tl.sum(change / (1 + change) * c, 0)
        tl.store(Out + head * P * M + col, result, live)

    return routed, key_response


@torch.library.custom_op('noise_record::routed_response', mutates_args=())
def routed_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    probabilities: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    import triton
    m, probes, heads, width = dq.shape
    count = keys.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device,
                      dtype=torch.bfloat16 if OWN_PROJECTION_BF16 else torch.float32)  # bf16: what the projection reads
    finite_attention_kernels()[0][(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(), keys.contiguous(), values.contiguous(), probabilities.contiguous(), rest.contiguous(),
        out, probes, heads, count, width, 32, max(16, triton.next_power_of_2(count)), ROUTED_ONE_EXP, num_warps=4)
    return out


@routed_response.register_fake
def _(dq, keys, values, probabilities, rest):
    return torch.empty(dq.shape, device=dq.device, dtype=torch.bfloat16 if OWN_PROJECTION_BF16 else torch.float32)


SCORE_PROBES, SCORE_SOURCES, SCORE_QUERIES = 4, 16, 64  # key_response tiles: probes x sources per program, query tile


@torch.library.custom_op('noise_record::key_response', mutates_args=())
def fused_key_response(q: torch.Tensor, dk: torch.Tensor, A: torch.Tensor, C: torch.Tensor,
                       source_start: int, query_start: int, window: int) -> torch.Tensor:
    import triton
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    finite_attention_kernels()[1][(triton.cdiv(sources, SCORE_SOURCES), triton.cdiv(probes, SCORE_PROBES), heads)](
        q, dk.contiguous(), A, C, out, heads, q.shape[1], sources, probes, width,
        *q.stride(), *A.stride(), *C.stride(), source_start, query_start, window,
        SCORE_PROBES, SCORE_SOURCES, SCORE_QUERIES, num_warps=4, num_stages=2, enable_fp_fusion=False)
    return out


@fused_key_response.register_fake
def _(q, dk, A, C, source_start, query_start, window):
    return torch.empty(dk.shape[:3], device=q.device, dtype=torch.float32)


@cache
def moved_query_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Raw, R0, Cos, Sin, Inv, Shift, Out, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               BP: tl.constexpr, EPS_: tl.constexpr, FOLDED: tl.constexpr):
        # One program: BP moves of one (token, head). Rotate each moved query projection, renormalize it against the
        # clean rotated query r0 (the query-norm change, exactly) and store the change in bf16, in one pass. FOLDED:
        # Raw is the bare moves' projection; the input norm's change is inv * Raw + shift * (clean query).
        token, head = tl.program_id(1) // H, tl.program_id(1) % H
        rows = tl.program_id(0) * BP + tl.arange(0, BP)
        half = tl.arange(0, D // 2)
        c = tl.load(Cos + token * (D // 2) + half)
        s = tl.load(Sin + token * (D // 2) + half)
        r0a = tl.load(R0 + (token * H + head) * D + half)
        r0b = tl.load(R0 + (token * H + head) * D + D // 2 + half)
        base = ((token * P + rows[:, None]) * H + head) * D
        keep = rows[:, None] < P
        a = tl.load(Raw + base + half[None, :], keep, 0).to(tl.float32)
        b = tl.load(Raw + base + D // 2 + half[None, :], keep, 0).to(tl.float32)
        ra = a * c[None, :] + b * s[None, :]
        rb = -a * s[None, :] + b * c[None, :]
        if FOLDED:
            g = tl.load(Inv + token * P + rows, rows < P, 0)[:, None]
            h = tl.load(Shift + token * P + rows, rows < P, 0)[:, None]
            ra, rb = ra * g + h * r0a[None, :], rb * g + h * r0b[None, :]
        sq = tl.sum(r0a * r0a, 0) + tl.sum(r0b * r0b, 0)
        dots = tl.sum(ra * r0a[None, :], 1) + tl.sum(rb * r0b[None, :], 1)
        moved = tl.sum(ra * ra, 1) + tl.sum(rb * rb, 1)
        inv0 = tl.math.rsqrt(sq / D + EPS_)
        inv = tl.math.rsqrt((sq + 2 * dots + moved) / D + EPS_)
        shift = (inv - inv0)[:, None]
        tl.store(Out + base + half[None, :], (ra * inv[:, None] + r0a[None, :] * shift).to(tl.bfloat16), keep)
        tl.store(Out + base + D // 2 + half[None, :], (rb * inv[:, None] + r0b[None, :] * shift).to(tl.bfloat16), keep)
    return kernel


@torch.library.custom_op("noise_record::moved_query_bf16", mutates_args=())
def moved_query_bf16(raw: torch.Tensor, r0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     inv: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None) -> torch.Tensor:
    """dq for routed_response from the moved query projections raw (m, P, H, D) and the clean rotated query r0
    (m, H, D): rotate, renormalize against r0, bf16 (fused_query). With inv and shift (m, P) (fold_norm), raw is the
    bare moves' projection and the moved projection is inv * raw + shift * (r0 before the rotation)."""
    import triton
    m, probes, heads, width = raw.shape
    out = torch.empty(raw.shape, device=raw.device, dtype=torch.bfloat16)
    folded = inv is not None
    moved_query_kernel()[(triton.cdiv(probes, 16), m * heads)](
        raw.contiguous(), r0.float().contiguous(), cos.float().contiguous(), sin.float().contiguous(),
        inv.float().contiguous() if folded else raw, shift.float().contiguous() if folded else raw, out,
        probes, heads, width, 16, EPS, folded, num_warps=4)
    return out


@moved_query_bf16.register_fake
def _(raw, r0, cos, sin, inv=None, shift=None):
    return torch.empty(raw.shape, device=raw.device, dtype=torch.bfloat16)


def query_change_routed(x, delta, pos, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos, sin):
    """Move only the selected key logits; retain the exact rescaling of the held output."""
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    q0 = norm(x) @ w_q.T
    r0 = rotate(q0.view(m, 1, heads, width))
    if (FUSED_QUERY and CHAIN_ACT_BF16 and delta.dtype == torch.bfloat16 and delta.is_cuda
            and top_keys.dtype == torch.bfloat16 and width in (32, 64) and top_keys.shape[2] <= 256):
        if FOLD_NORM:  # the bare moves' projection; the kernel rescales it and adds the clean query's share
            inv0, inv = norm_inverses(x, delta)
            raw = (delta.to(torch.bfloat16) @ w_q.T.to(torch.bfloat16)).view(m, probes, heads, width)
            dq = moved_query_bf16(raw, r0[:, 0], cos[pos], sin[pos], inv, inv / inv0 - 1)
        else:
            moved = norm_difference(x, delta).to(torch.bfloat16)
            raw = (moved @ w_q.T.to(torch.bfloat16)).view(m, probes, heads, width)
            dq = moved_query_bf16(raw, r0[:, 0], cos[pos], sin[pos])
        dout = routed_response(dq, top_keys, top_values, top_p, rest)
        if OWN_PROJECTION_BF16:
            return (dout.flatten(-2) @ w_proj_t.to(torch.bfloat16)).float()
        return dout.flatten(-2) @ w_proj_t
    if FOLD_NORM:
        dr = rotate(folded_projection(x, delta, w_q.to(torch.bfloat16) if CHAIN_ACT_BF16 else w_q, q0)
                    .view(m, probes, heads, width))
    elif CHAIN_ACT_BF16:  # the moved query's input projection on bf16 operands
        moved = norm_difference(x, delta).to(torch.bfloat16)
        dr = rotate((moved @ w_q.T.to(torch.bfloat16)).float()
                    .view(m, probes, heads, width))
    else:
        dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    if dq.is_cuda and top_keys.dtype == torch.bfloat16 and width in (32, 64) and top_keys.shape[2] <= 256:
        dout = routed_response(dq.to(top_keys.dtype), top_keys, top_values, top_p, rest)
        if OWN_PROJECTION_BF16:  # the held attention output's change through c_proj on bf16 operands
            return (dout.flatten(-2) @ w_proj_t.to(torch.bfloat16)).float()  # dout is already bf16
        return dout.flatten(-2) @ w_proj_t
    ds = torch.einsum('mphd,mhkd->mphk', dq.to(top_keys.dtype), top_keys).float() * width ** -0.5
    p = top_p.unsqueeze(1)
    dlse = torch.log1p((p * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mphk,mhkd->mphd', (p * torch.expm1(ds - dlse)).to(top_values.dtype), top_values).float()
    dout = dout + torch.expm1(-dlse) * rest.unsqueeze(1)
    return dout.flatten(-2) @ w_proj_t


@torch.compile(dynamic=False)
def norm_input_credit(x, g, sigma, H):
    """Each token's credit at an RMSNorm's input, from the credit g at its output: central differences of
    g . norm(x +- s h) along every row h of a full Hadamard cycle (s = sigma * rms(x)), decoded. One finite step through
    the norm; closed form, since g . norm(x + d) only needs x . h, g . h, g . x and |x|^2."""
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


@cache
def moment_matrix_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(W, C, Out, V, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # One program: a (BM, BN) tile of token t's W^T diag(c_t) W, the scaling applied to W's tiles as they load.
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        t = tl.program_id(2)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, V, BK):
            k = k0 + tl.arange(0, BK)
            c = tl.load(C + t * V + k).to(tl.float32)
            a = (tl.load(W + k[:, None] * D + rm[None, :]).to(tl.float32) * c[:, None]).to(tl.bfloat16)
            b = tl.load(W + k[:, None] * D + rn[None, :])
            acc += tl.dot(tl.trans(a), b)
        tl.store(Out + t * D * D + rm[:, None] * D + rn[None, :], acc.to(tl.bfloat16))
    return kernel


def moment_matrices(w16, c16):
    """W^T diag(c_t) W per token in bf16 (moment_kernel): w16 (V, D), c16 (T, V), both bf16. The same products as
    scaling W^T by c_t in bf16 and multiplying, without storing the scaled (T, D, V) copies."""
    vocab, width = w16.shape
    assert vocab % 64 == 0 and width % 128 == 0
    out = torch.empty(len(c16), width, width, device=w16.device, dtype=torch.bfloat16)
    moment_matrix_kernel()[(width // 128, width // 128, len(c16))](
        w16.contiguous(), c16.contiguous(), out, vocab, width, 128, 128, 64, num_warps=8, num_stages=3)
    return out


MOMENT_KERNEL = False  # set from Settings.moment_kernel


@torch.no_grad()
def head_moments(xL, base, weight, block=64):
    """Per-token moments of the softcapped readout for head_moments: with z0 = base * inv0, f = 15 tanh(z / 15) and
    p = softmax(f(z0)), the change of log sum exp f(z0 + dz) for dz = a * base + inv * (W delta) is, to second order,
    lin + a^2 B2 + 2 a inv nu.delta + inv^2 delta^T M delta - lin^2 / 2 with lin = a B1 + inv mu1.delta. Returns
    (B1, B2, mu1, nu, M), M in bf16 (tokens, width, width)."""
    width = xL.shape[-1]
    inv0 = torch.rsqrt(xL.square().sum(-1, keepdim=True) / width + EPS)
    t = torch.tanh(base * inv0 / 15)
    f, f1 = 15 * t, 1 - t.square()
    f2 = -(2 / 15) * t * f1
    prob = torch.softmax(f, -1)
    c1, c2 = prob * f1, prob * (f2 + f1.square()) / 2
    b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
    mu1, nu = c1 @ weight, (c2 * base) @ weight
    w16 = weight.to(torch.bfloat16)
    if MOMENT_KERNEL and xL.is_cuda:
        return b1, b2, mu1, nu, moment_matrices(w16, c2.to(torch.bfloat16))
    second = torch.empty(len(xL), width, width, device=xL.device, dtype=torch.bfloat16)
    for t0 in range(0, len(xL), block):
        c = c2[t0:t0 + block].to(torch.bfloat16)
        second[t0:t0 + block] = torch.matmul(w16.T.unsqueeze(0) * c.unsqueeze(1), w16)
    return b1, b2, mu1, nu, second


def chain_chunk_eager(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                      target_logit_w, y, own=None, from_output=False, packed=False, moments=None):
    """Forward differences of each token's loss along every move of one site, K/V held, MLPs restricted to
    each token's top-k units. site: -1 for the raw embedding, else the block whose mid stream moves. move (m, P, W);
    scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block (m, W); idx, hs per block
    (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block (H, W); weight_t (W, V);
    base (m, V) clean top-stream logits before the norm; target_logit_w (m, W).
    own optionally supplies PR #19's own-query responses in every entered attention.
    from_output starts at the block's MLP output instead: skip its MLP, then enter
    the next block through BOTH its residual scaling and its own-query response.
    move may be (steps (m, 1), rows (P0, W), extra (m, P - P0, W) or None) (fused_moves): the moves steps * rows, then
    extra, formed here so the compiled graph never stores them in fp32.
    Returns (m, P)."""
    L = len(mids)
    if isinstance(move, tuple):
        steps, rows, extra = move
        move = steps.unsqueeze(-1) * rows.unsqueeze(0)
        if extra is not None:
            move = torch.cat([move, extra], 1)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site + int(from_output)
    for b in range(first, L):
        if b > first or from_output:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        if own is not None and (site < 0 or b > first or from_output):
            if own[0] == 'routed':
                _, pos, inputs, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos_sin = own
                delta = delta + query_change_routed(inputs[b], delta, pos, w_q[b], top_keys[b], top_values[b],
                                                    top_p[b], rest[b], w_proj_t[b], cos_sin[0], cos_sin[1])
                if CHAIN_ACT_BF16:
                    delta = delta.to(torch.bfloat16)
            else:
                _, pos, rels, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
                delta = delta + query_change(inputs[b], delta, pos, rels[b], w_q[b], keys[b], values[b],
                                             w_proj_t[b], cos, sin, windows[b])
        fc = w_fc[b] if packed else w_fc[b][idx[b]]
        wo = w_out_t[b] if packed else w_out_t[b][idx[b]]
        if FOLD_NORM:  # hs are the clean pre-activations norm(mid) @ fc.T of the selected units
            dh = folded_projection(mids[b], delta, fc, hs[b])
        else:
            dz = norm_difference(mids[b], delta)
            dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
        if CHAIN_ACT_BF16:  # the moved stream's change stored in bf16 between blocks (the head reads it in bf16)
            delta = delta.to(torch.bfloat16)
    sq = xL.square().sum(-1, keepdim=True)
    dots = (delta.float() * xL.unsqueeze(1)).sum(-1) if CHAIN_ACT_BF16 else torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.float().square().sum(-1)) / xL.shape[-1] + EPS)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    if moments is not None:  # the partition function's change to second order in the logit change (head_moments)
        b1, b2, mu1, nu, second = moments
        a = inv - inv0
        d = delta.float()
        lin = a * b1.unsqueeze(1) + inv * (d * mu1.unsqueeze(1)).sum(-1)
        quad = (torch.bmm(delta.to(torch.bfloat16), second).float() * d).sum(-1)
        change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * (d * nu.unsqueeze(1)).sum(-1)
                  + inv.square() * quad - 0.5 * lin.square())
        target0 = base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0
        moved_minus_clean = change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(target0 / 15))
        return moved_minus_clean / scale * (y >= 0).unsqueeze(1)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)
chain_chunk_own = torch.compile(chain_chunk_eager, dynamic=True)
KEY_BUCKET = 256
ROUTED_ONE_EXP = False  # both set from Settings.chain_fast when a Cut estimator is built
OWN_PROJECTION_BF16 = False
CHAIN_ACT_BF16 = False  # set from Settings.chain_act
FUSED_QUERY = False  # set from Settings.fused_query
FOLD_NORM = False  # set from Settings.fold_norm


@torch.compile(dynamic=False)
def key_scores(qp, kp, k_raw, steps, cos, sin, A, C, basis):
    """Single-key experiments, each evaluated exactly: the raw key at position t moves by steps_t e_j and nothing
    else moves, so each query's attention changes in one term, o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'),
    a' = a exp(dl). Scored against each query's target (C[s, t] = hub_s . (v_t - o_s)): one local step, as the
    record's coordinate head scores single-logit moves. The processed key's change is formed exactly before the
    logit product. qp, kp, k_raw (h, T, d); steps (h, T, 1); cos, sin (T, d/2); A, C (h, T, T), A zero outside the
    window; basis (J, d). Returns (h, T, J)."""
    d = k_raw.shape[-1]
    moved = k_raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    x1, x2 = moved[..., :d // 2], moved[..., d // 2:]
    moved = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    dk = moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)
    dl = torch.einsum('hsd,hjtd->hjst', qp, dk) / d ** 0.5
    change = A[:, None] * torch.expm1(dl)
    return (change / (1 + change) * C[:, None]).sum(2).transpose(1, 2) / steps


@torch.compile(dynamic=False)
def raw_key_moves(kp, raw, steps, cos, sin, basis):
    moved = raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    half = raw.shape[-1] // 2
    moved = torch.cat((moved[..., :half] * cos + moved[..., half:] * sin,
                       -moved[..., :half] * sin + moved[..., half:] * cos), -1)
    return moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)


@torch.compile(dynamic=False)
def prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin):
    """Same central experiments, reusing the linear projections of x and each direction.

    N(x +/- s*h) W - N(x) W =
        (inv_plusminus - inv0) * (x W) +/- s*inv_plusminus * (h W).
    This is a finite forward identity, including both RMSNorm responses.
    Positive and negative trials share one batched query/key product.
    """
    m, probes, width = x.shape[0], rows.shape[0], x.shape[-1]
    heads, _, dh = k_raw.shape
    square = x.square().sum(-1, keepdim=True)
    dots = x @ rows.T
    moved_square = square + steps.square() * rows.square().sum(-1)
    inv0 = torch.rsqrt(square / width + EPS)
    inv_up = torch.rsqrt((moved_square + 2 * steps * dots) / width + EPS)
    inv_down = torch.rsqrt((moved_square - 2 * steps * dots) / width + EPS)
    up = ((inv_up - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
          + (steps * inv_up).unsqueeze(-1) * projected_rows.unsqueeze(0))
    down = ((inv_down - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
            - (steps * inv_down).unsqueeze(-1) * projected_rows.unsqueeze(0))
    raw_change = torch.cat((up, down), dim=1).reshape(m, 2 * probes, heads, dh).permute(2, 1, 0, 3)
    rotate = lambda z: torch.cat([z[..., :dh//2]*cos + z[..., dh//2:]*sin,
                                  -z[..., :dh//2]*sin + z[..., dh//2:]*cos], -1)
    r0 = rotate(k_raw)
    dr = rotate(raw_change)
    dk = norm_difference(r0.reshape(heads*m, dh),
                         dr.permute(0, 2, 1, 3).reshape(heads*m, 2 * probes, dh))
    return dk.reshape(heads, m, 2 * probes, dh).permute(0, 2, 1, 3)


@torch.compile(dynamic=False)
def key_input_prepared_derivatives(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin, qp, A, C):
    dk = prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin)
    probes, dh = rows.shape[0], k_raw.shape[-1]
    dl = torch.einsum('hsd,hpmd->hpsm', qp, dk) / dh**.5
    change = A[:, None] * torch.expm1(dl)
    scores = (change / (1 + change) * C[:, None]).sum((0, 2)).T
    return (scores[:, :probes] - scores[:, probes:]) / (2 * steps)


@torch.compile(dynamic=False)
def decode_key_pairs(scores, steps, rows):
    return decode_local_response(scores.sum(0).T, steps, rows) / len(rows)


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames = {}
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
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

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws start..start + n of this step's world * draws at `site`: rows of a sign-flipped Hadamard matrix,
        orthogonal across chunks and GPUs within each cycle."""
        if site not in self.frames:
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma=0.1):
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

class Boost:
    """Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        if model.config.n_head != model.config.n_kv_head or model.config.n_layer % 2:
            raise ValueError('Stage estimators require equal Q/KV heads and an even number of blocks')
        if any(k % split for k in (model.config.n_embd, settings.query_draws)):
            raise ValueError('Readout and query cycles must divide across the sequence group')
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.shared_generator = self.capture.shared_generator
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)
        self.maps = {}

    def close(self):
        self.capture.close()

    def share(self, total):
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor):
        if self.split > 1:
            torch.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def probe(self, x, y, mix=None, probed=None):
        """Each token's readout credit in the probed space, measured along a full cycle of the step's basis; the
        GPUs of a sequence each take a slice of the cycle and add their parts."""
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        offset = torch.randint(0, self.width, (n, 1), device=self.device, generator=self.generator)  # each token's
        # first row (used only by partial cycles; drawn either way so the random stream is the same)
        credit = torch.zeros(n, self.width, device=self.device)
        lo, hi = self.share(self.width)
        C, stream_dirs, dir_sq, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ Hs / self.width
        return self.group_sum(credit)

    def embedding_block0(self, e, y, x):
        """The embedding's credit for the first attention stage's readout along each token's own path (every query
        sees the other positions' clean keys and values and its own moved ones)."""
        model = self.model
        cfg = model.config
        attn = model.transformer.h[0].attn
        b, t = x.shape
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        window = model.window_sizes[0][0]
        mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
        weight = model.lm_head.weight.float()[:cfg.vocab_size]
        W = {name: getattr(attn, 'c_' + name).weight.float() for name in ('q', 'k', 'v', 'proj')}
        e = e.float().view(b, t, self.width)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, self.width, device=self.device)
        n = chunk_size(self.width // self.split, self.s.emb_chunk)
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, self.width).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, self.width), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(self.width)
        for j0 in range(lo, hi, n):
            rows = H[j0:j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / self.width).reshape(b * t, self.width)

    def attention_maps(self, i):
        """Clean attention A[b, h, query, source]; no matched-filter matrix is needed."""
        if i in self.maps:
            return self.maps[i]
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (norm(apply_rotary_emb(d.outputs[prefix + name].float().view(b, t, h, hd), cos, sin))
                for name in ('c_q', 'c_k'))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        window = model.window_sizes[i][0]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        A = logits.softmax(-1)
        del logits
        self.maps[i] = A.to(torch.bfloat16)
        return self.maps[i]

    def query_errors(self, draws, targets, layers=None):
        """A complete per-head query cycle, scored against each query's own output target."""
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        n = chunk = chunk_size(draws, self.s.nrep)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = None if self.s.query_central else attention(processed(q_raw).to(dt), kp.to(dt), values.to(dt), window)
            fixed, vals = kp.to(dt), values.to(dt)
            raw = q_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = query_probe_heads(raw, a, sigma, cos, sin)
                out = query_probe_attention(moved, fixed, vals, window, n, b)
                if self.s.query_central:
                    minus = query_probe_attention(
                        query_probe_heads(raw, a, -sigma, cos, sin), fixed, vals, window, n, b)
                    score = head_scores(out, minus, target)
                else:
                    score = head_scores(out, baseline, target)
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma * (2 if self.s.query_central else 1))).reshape(b, t, cfg.n_embd)
        return errors

    def key_context(self, i, sequence, target):
        """Clean FP32 single-key score context, shared by raw-K and input-K experiments."""
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd//2]*cos + z[..., hd//2:]*sin,
                                   -z[..., :hd//2]*sin + z[..., hd//2:]*cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix+name][sequence].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][sequence].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix+'ve_gate'][sequence].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        logits = (qp @ kp.transpose(1, 2)) / hd**.5
        A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
        hub = target[sequence].view(t, h, hd).transpose(0, 1)
        C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
        return qp, kp, k_raw, cos, sin, A, C

    def exact_key_errors(self, targets, layers=None):
        """Key credit, one local step from the targets, by single-key experiments (key_scores): exact up to the
        one-sided step, with no interference between positions. The GPUs of a sequence split the directions."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                credit_all[r] = self.key_errors_from_context(i, self.key_context(i, r, targets[i]))
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def key_errors_from_context(self, layer, context):
        """Raw-K finite experiments, sharing the input-K context and skipping masked query blocks."""
        qp, kp, raw, cos, sin, A, C = context
        h, t, hd = raw.shape
        lo, hi = self.share(hd)
        result = torch.zeros_like(raw)
        eye = torch.eye(hd, device=self.device)
        steps = self.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        for t0 in range(0, t, KEY_BUCKET):
            t1 = min(t0 + KEY_BUCKET, t)
            q1 = min(t, t1 + self.model.window_sizes[layer][0])
            source, query = slice(t0, t1), slice(t0, q1)
            if raw.is_cuda and hd in (32, 64):
                moves = raw_key_moves(kp[:, source], raw[:, source], steps[:, source],
                                      cos[source], sin[source], eye[lo:hi])
                scored = fused_key_response(qp[:, query], moves, A[:, query, source], C[:, query, source],
                                           t0, t0, self.model.window_sizes[layer][0])
                result[:, source, lo:hi] = scored.transpose(1, 2) / steps[:, source]
                continue
            for j0 in range(lo, hi, self.s.key_chunk):
                j1 = min(j0 + self.s.key_chunk, hi)
                result[:, source, j0:j1] = key_scores(
                    qp[:, query], kp[:, source], raw[:, source], steps[:, source],
                    cos[source], sin[source], A[:, query, source], C[:, query, source], eye[j0:j1])
        return result

    def value_credit(self, i, target):
        """Fixed-attention value credit; all query tokens contribute, with no discount."""
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        A = self.attention_maps(i)  # A[b, h, query, source]
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return (A.float().transpose(-1, -2) @ hub).transpose(1, 2)

    def exact_values(self, i, target, value_target=None, raw_credit=None):
        """Layer i's value and gate errors, the local step in closed form: a value move at t reaches query s's output
        as A[s, t] times the move, so the value credit is sum_s A[s, t] target_s; a gate move scales the value
        embedding by the gate's slope (the record's central difference, odd in the move)."""
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        credit = self.value_credit(i, target) if raw_credit is None else raw_credit
        # A corrected target has already crossed RMSNorm. Only V/VE may use it:
        # the gate's sigmoid experiment must retain the directly measured target.
        value_credit = credit if value_target is None else self.value_credit(i, value_target)
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """All hidden units' local central differences, partitioned across a sequence's GPUs."""
        # Preserve full-state checkpoint RNG advancement from the retired hidden cycle.
        self.signs(256)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        lo, hi = self.share(u.shape[-1])
        local = hidden_local_credit(u[:, lo:hi], steps, g, out_weight[:, lo:hi].float())
        return local if self.split == 1 else F.pad(local, (lo, u.shape[-1] - hi))

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def internals(self, gradients, targets, b, t, value_targets=None, key_errors=None, raw_values=None):
        """Q/K/gate use measured targets; only linear V/VE paths accept corrected targets."""
        d, s = self.capture, self.s
        errors = self.query_errors(s.query_draws // self.split, targets)
        errors = {site: error / self.split for site, error in errors.items()}
        errors.update(self.exact_key_errors(targets) if key_errors is None else key_errors)
        for i, target in targets.items():
            value_target = None if value_targets is None else value_targets[i]
            raw = None if raw_values is None else raw_values[i]
            errors.update({site: error / self.split
                           for site, error in self.exact_values(i, target, value_target, raw).items()})
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        x0 = d.x0
        targets, streams = {}, []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i]
            xmid = xin + d.outputs[p + 'attn.c_proj']
            xout = xmid + d.outputs[p + 'mlp.c_proj']
            streams.append((xin, xmid, xout))
            xm, xo = xmid.float().flatten(0, 1), xout.float().flatten(0, 1)
            block = model.transformer.h[i]
            att_in = d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            g_mid = self.probe(xm, yf)
            gradients[p + 'attn.c_proj.weight'] = g_mid.T @ att_in
            # One local linear step from the measured stream credit (PR #15).
            hub = g_mid @ block.attn.c_proj.weight.float()
            # Keep later probes on the same RNG stream as the retired hub cycle.
            self.signs(self.width)
            torch.randint(0, self.width, (xm.shape[0], 1), device=self.device, generator=self.generator)
            targets[i] = hub.view(b, t, self.width)
            g_out = self.probe(xo, yf)
            gradients[p + 'mlp.c_proj.weight'] = g_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, g_out, model.transformer.h[i].mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            if i < L - 1:  # the head learns at every intermediate readout too
                for stream in (xm, xo):
                    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + EPS)
                    error = coordinate_differences(normed @ weight.T, yf, 0.1)
                    gradients['lm_head.weight'][:weight.shape[0]] += error.T @ normed
        self.internals(gradients, targets, b, t)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, self.width))
        scalars = torch.zeros(2, L, device=self.device)
        step = s.scalar_step
        for i in range(self.part, L, self.split):
            x_prev = x0 if i == 0 else streams[i - 1][2]
            ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
            lr_, l0 = model.resid_lambdas[i], model.x0_lambdas[i]
            for which in range(2):
                up = self.stage_losses(i, lr_ + step * (which == 0), l0 + step * (which == 1), x_prev, x0, ve, y)
                down = self.stage_losses(i, lr_ - step * (which == 0), l0 - step * (which == 1), x_prev, x0, ve, y)
                scalars[which, i] = (up - down).sum() / (2 * step)
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.part:  # computed in full by every GPU of the sequence: kept by its first GPU only
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight'):
                    gradients[name] = torch.zeros_like(gradients[name])
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

    def step(self, x, y, optimizer):
        gradients = self.estimate(x, y)
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        sequences = self.world // self.split
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if self.world > 1:
            torch.distributed.all_reduce(flat)
        flat /= sequences
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite estimate')
        offset = 0
        for name, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        self.maps.clear()
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()

class Cut(Boost):
    """Own-query forward loss probes, optionally augmented by PR #23's value path."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        global ROUTED_ONE_EXP, OWN_PROJECTION_BF16, CHAIN_ACT_BF16, FUSED_QUERY, FOLD_NORM, MOMENT_KERNEL
        ROUTED_ONE_EXP = OWN_PROJECTION_BF16 = settings.chain_fast
        CHAIN_ACT_BF16 = settings.chain_act == 'bf16'
        FUSED_QUERY = settings.fused_query
        FOLD_NORM = settings.fold_norm
        MOMENT_KERNEL = settings.moment_kernel
        # split > 1: the GPUs of a sequence each take a share of every cycle of moves and draws
        super().__init__(model, settings, seed, rank, world, split=split, group=group)

    def value_path_credit(self, chain, *, return_inputs=False, raw_values=None):
        """Independent measured mid credits -> V -> one RMSNorm step -> residual identities.

        Never feed accumulated extra credit into another norm or local nonlinear
        experiment. Reuse this capture's attention maps later for V/gate credit.
        """
        model, d = self.model, self.capture
        L = model.config.n_layer
        extra = {L - 1: torch.zeros_like(chain[0])}
        inputs = {}
        for i in reversed(range(0 if return_inputs else 1, L)):
            attn = model.transformer.h[i].attn
            ev = (self.value_credit(i, chain[i] @ attn.c_proj.weight.float())
                  if raw_values is None else raw_values[i]).reshape(-1, self.width)
            vp = norm_move(d.blocks[i].float().flatten(0, 1), ev @ attn.c_v.weight.float())
            inputs[i] = vp
            if i:
                extra[i - 1] = model.resid_lambdas[i].float() * (vp + extra[i])
        return (extra, inputs) if return_inputs else extra

    def key_input_credits(self, targets):
        """Independent central probes at each block input, moving only its K branch.

        Use a private copy of the saved probe RNG so these optional draws neither
        shift Q/hidden probes nor require another checkpoint RNG stream.
        Partial Hadamard cycles use randomized rows and column signs.
        """
        d, s = self.capture, self.s
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        raw_errors = {}
        for i, target in targets.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)[:s.key_path_draws]
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator)*2-1
            rows = frame[indices] * signs
            total_rows = len(rows)
            r0, r1 = self.share(total_rows)  # this GPU's share of the draws
            rows = rows[r0:r1]
            estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
            raw_credit = torch.empty_like(estimate)
            w_key = self.model.transformer.h[i].attn.c_k.weight.float()
            projected_rows = rows @ w_key.T
            for r in range(d.batch):
                context = self.key_context(i, r, target)
                qp, _, raw, cos, sin, A, C = context
                raw_credit[r] = self.key_errors_from_context(i, context).transpose(0, 1).reshape(d.length, self.width)
                xin = d.blocks[i][r].float()
                projected_x = xin @ w_key.T
                steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                for t0 in range(0, d.length, s.key_path_tokens):
                    t1 = min(t0+s.key_path_tokens, d.length)
                    sl = slice(t0, t1)
                    # Queries outside this union of source windows have exactly
                    # zero response. Round the retained range to kernel buckets.
                    q1 = min(d.length, t1 + self.model.window_sizes[i][0])
                    qlength = min(d.length, -(-(q1-t0) // KEY_BUCKET) * KEY_BUCKET)
                    qs = slice(max(0, q1-qlength), q1)
                    if xin.is_cuda and raw.shape[-1] in (32, 64):
                        # All directions share a source tile. The fused scorer
                        # reduces query tiles without writing H*P*Q*M intermediates.
                        moves = prepared_key_moves(xin[sl], steps[sl], rows, projected_x[sl],
                                                   projected_rows, raw[:, sl], cos[sl], sin[sl])
                        scores = fused_key_response(qp[:, qs], moves, A[:, qs, sl], C[:, qs, sl],
                                                    t0, qs.start, self.model.window_sizes[i][0])
                        estimate[r, sl] = decode_key_pairs(scores, steps[sl], rows) * (len(rows) / total_rows)
                        continue
                    derivatives = torch.empty(len(xin[sl]), len(rows), device=self.device)
                    for j0 in range(0, len(rows), s.key_chunk):
                        js = slice(j0, min(j0+s.key_chunk, len(rows)))
                        derivatives[:, js] = key_input_prepared_derivatives(
                            xin[sl], steps[sl], rows[js], projected_x[sl], projected_rows[js],
                            raw[:, sl], cos[sl], sin[sl],
                            qp[:, qs], A[:, qs, sl], C[:, qs, sl])
                    estimate[r, sl] = derivatives @ rows / total_rows
            credits[i] = self.group_sum(estimate.flatten(0, 1))
            raw_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
        self._input_key_errors = raw_errors
        return credits

    def path_credits(self, chain, targets, raw_values=None):
        """Sum independent input responses, then transport only along residual identities."""
        L = self.model.config.n_layer
        inputs = {i: torch.zeros_like(chain[i]) for i in range(L)}
        if self.s.value_path:
            _, inputs = self.value_path_credit(chain, return_inputs=True, raw_values=raw_values)
        if self.s.key_path_draws:
            keys = self.key_input_credits(targets)
            inputs = {i: inputs[i] + keys[i] for i in range(L)}
        if self.s.gate_path:
            for i, credit in self.gate_input_credits(targets, raw_values).items():
                inputs[i] = inputs[i] + credit
        extra = {L-1: torch.zeros_like(chain[0])}
        for i in reversed(range(1, L)):
            extra[i-1] = self.model.resid_lambdas[i].float() * (inputs[i]+extra[i])
        return extra, inputs

    def gate_input_credits(self, targets, raw_values=None):
        """Independent gate-input experiments from raw hubs; only linear consumers follow."""
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        for i, target in targets.items():
            # Advance for every layer to retain the tested private pilot bases.
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            if not has_ve(i, self.model.config.n_layer):
                continue
            rows = frame[indices] * signs
            value = (self.value_credit(i, target) if raw_values is None else raw_values[i])
            value = value.reshape(-1, heads, self.width // heads)
            ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
            kappa = (value * ve).sum(-1)
            xin = d.blocks[i].float().flatten(0, 1)
            weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
            projected_rows = F.linear(rows[:, :weight.shape[1]], weight)
            row_square = rows.square().mean(-1)
            result = torch.empty_like(xin)
            for t0 in range(0, len(xin), s.chain_tokens):
                sl = slice(t0, t0 + s.chain_tokens)
                result[sl] = prepared_gate_input_response(
                    xin[sl], weight, kappa[sl], rows, projected_rows, row_square, s.gate_path_sigma)
            credits[i] = result
        return credits

    @staticmethod
    def path_scalar_credit(inputs, extra, x0, outs):
        """Linear mixing coefficients see input responses plus later identity-path credit."""
        credits = []
        for i in range(len(outs)):
            credit = inputs[i]+extra[i]
            previous = x0 if i == 0 else outs[i-1]
            credits.append(torch.stack([(credit*previous).sum(), (credit*x0).sum()]))
        return torch.stack(credits, dim=1)

    def chain_credits(self, e, x0, mids, outs, routes, y):
        """Measure embedding and mid-stream credits from loss.

        Each root uses a full orthogonal cycle. Mixing scalars retain their
        central mid-stream probes. MLP outputs reuse the next mid credit;
        the final output uses the existing head probe.
        """
        model, s = self.model, self.s
        L = model.config.n_layer
        vocab = model.config.vocab_size
        weight = model.lm_head.weight.float()[:vocab]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        moments = head_moments(xL, base, weight) if s.head_moments else None
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        # Draw bases in the original site order. Replay one sequence at a time
        # so routed K/V storage does not grow with the per-GPU batch size.
        probes = []
        lo, hi = self.share(self.width)  # this GPU's share of each cycle; the scalar moves on the first GPU
        P, scalars_here = hi - lo, self.part == 0
        for site in range(-1, L):
            H = self.basis()[lo:hi]
            if site < 0 and s.emb_credit == 'mid0':  # the embedding's credit comes from block 0's instead (below)
                continue
            probed = e if site < 0 else mids[site]
            steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            credit = torch.empty(n, self.width, device=self.device)
            lam = torch.zeros(2, device=self.device)
            probes.append((site, H, steps, credit, lam))
        length = self.capture.length if s.own_query else n
        for start in range(0, n, length):
            own = None
            if s.own_query:
                own = self.own_query_inputs(start // length)
                if s.query_keys:
                    own = self.strong_keys(own, s.query_keys)
            for t0 in range(start, start + length, m):
                t1 = min(t0 + m, start + length)
                sl = slice(t0, t1)
                own_sl = None
                if own is not None:
                    begin, end = t0 - start, t1 - start
                    if own[0] == 'routed':
                        local = slice(begin, end)
                        own_sl = ('routed', own[1][local], [z[local] for z in own[2]], own[3]) + tuple(
                            [z[local] for z in part] for part in own[4:8]) + own[8:]
                    else:
                        own_sl = self.own_query_window(own, begin, end)
                kernel = chain_chunk_own if own_sl is not None and own_sl[0] == 'window' else chain_chunk
                # All roots use the same selected units. Gather once per token block,
                # not again for each of the 44 layer visits of the nine roots.
                selected_fc = [w[r[0][sl]] for w, r in zip(w_fc, routes)]
                selected_out = [w[r[0][sl]] for w, r in zip(w_out_t, routes)]
                for site, H, steps, credit, lam in probes:
                    move = None if s.fused_moves else steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                    extra = None
                    scale = steps[sl].expand(-1, P)
                    if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        extra = torch.cat([h * prev, h * x0[sl].unsqueeze(1), -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        if move is not None:
                            move = torch.cat([move, extra], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                           torch.full_like(scale[:, :2], -h)], 1)
                    derivative = kernel(
                        site, move if move is not None else (steps[sl], H, extra), scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, selected_fc, selected_out, weight_t,
                        base[sl], target_w[sl], y[sl], own_sl, packed=True,
                        moments=None if moments is None else tuple(z[sl] for z in moments))
                    credit[sl] = derivative[:, :P] @ H / self.width
                    if site >= 0 and scalars_here:
                        lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                del selected_fc, selected_out
            # Drop views as well as the owning cache before constructing the
            # next sequence's routed keys and values.
            del own_sl, own
        credits = {}
        for site, _, _, credit, lam in probes:
            credits['emb' if site < 0 else site] = self.group_sum(credit)
            if site >= 0:
                credits['lam', site] = self.group_sum(lam)
        if s.emb_credit == 'mid0':  # measured mid-stream credits back along all linear x0 skip paths, one step through the norm
            x0_credit = (lam_r[0] + lam_0[0]) * credits[0]
            for i in range(1, L):
                x0_credit = x0_credit + lam_0[i] * credits[i]
            credits['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
                                               hadamard(self.width, self.device, torch.float32))
        return credits

    def own_query_inputs(self, sequence=0):
        """Clean keys/values for one sequence; preserve upstream math and token positions."""
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            k = d.outputs[prefix + 'c_k'][sequence].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            c, s = cos[:, None], sin[:, None]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][sequence].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][sequence].float().view(length, heads, width)
            inputs.append(d.blocks[i][sequence].float())
            w_q.append(block.attn.c_q.weight.float())
            keys.append(k.transpose(0, 1).to(torch.bfloat16).contiguous())
            values.append(v.transpose(0, 1).to(torch.bfloat16).contiguous())
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            windows.append(model.window_sizes[i][0])
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)

    @staticmethod
    def strong_keys(own, k):
        """Only top-k logits move; keep the remaining output for exact renormalization."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        length = len(pos)
        top_keys, top_values, top_p, rests = [], [], [], []
        for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
            heads, _, width = kk.shape
            c, s = cos[:, None], sin[:, None]
            r = (norm(x) @ wq.T).view(length, heads, width)
            q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                                r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            lag = pos[:, None] - pos[None]
            allowed = (lag >= 0) & (lag <= window)
            prob = torch.softmax((torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width ** -0.5)
                                 .masked_fill(~allowed, float('-inf')), -1)
            p, idx = prob.topk(min(k, length), dim=-1)
            gather = lambda z: z.unsqueeze(1).expand(-1, length, -1, -1).gather(
                2, idx.unsqueeze(-1).expand(-1, -1, -1, width))
            ktop, vtop = gather(kk), gather(vv)
            out = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
            rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
            top_keys.append(ktop.transpose(0, 1).contiguous())
            top_values.append(vtop.transpose(0, 1).contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
            rests.append(rest.transpose(0, 1).contiguous())
        return ('routed', pos, inputs, w_q, top_keys, top_values, top_p, rests, w_proj_t, (cos, sin))

    @staticmethod
    def own_query_window(own, t0, t1):
        """Slice clean context windows, with the same 256-token buckets as PR #19."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        total = keys[0].shape[1]
        rels, ks, vs = [], [], []
        for k, v, window in zip(keys, values, windows):
            need = t1 - max(0, t0 - window)
            length = min(total, -(-need // KEY_BUCKET) * KEY_BUCKET)
            start = max(0, t1 - length)
            rels.append(pos[t0:t1] - start)
            ks.append(k[:, start:start + length])
            vs.append(v[:, start:start + length])
        return ('window', pos[t0:t1], rels, [z[t0:t1] for z in inputs],
                w_q, ks, vs, w_proj_t, cos, sin, windows)

    def select_mlp_routes(self, layer, mid, hpre, activation):
        """Select the fixed replay units; experiments may override only this ranking."""
        weight = self.model.transformer.h[layer].mlp.c_proj.weight.float()
        return (activation * weight.norm(dim=0)).topk(self.s.route_k, dim=-1).indices

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        self._input_key_errors = None
        self.measured = {}  # Last step's diagnostic credits are not inputs to this estimate.
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error(0.01)  # coordinate differences at the final readout
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        for i in range(L):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            top = self.select_mlp_routes(i, mid, hpre, r)
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        chain = self.chain_credits(e, x0, mids, outs, routes, yf)
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        # Reuse only directly measured hub credit, scoped to this capture.
        # Corrected targets still take a separate A.T @ target in exact_values.
        raw_values = {i: self.value_credit(i, target) for i, target in targets.items()}
        extra, inputs = self.path_credits(chain, targets, raw_values) if s.value_path or s.key_path_draws or s.gate_path else (None, None)
        value_targets = {}
        scalars = torch.zeros(2, L, device=self.device)
        for i in range(L):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            if i == L - 1:
                d_out = g_top
            else:
                d_out = lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = (d_mid, d_out) if extra is None else (d_mid + extra[i], d_out + extra[i])
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            if extra is not None:
                value_targets[i] = (x_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            # the internals' local step through the attention module, scored as c_proj sees the mid-stream credit
            scalars[:, i] = chain['lam', i]
        if s.path_scalars and extra is not None:
            scalars += self.path_scalar_credit(inputs, extra, x0, outs)
        self.measured = dict(chain=chain, g_top=g_top, targets=targets)
        self.internals(gradients, targets, b, t, value_targets or None, self._input_key_errors, raw_values)
        self._input_key_errors = None
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        if self.part:  # split: formed in full by every GPU of the sequence, kept by its first GPU (the
            # internals split or average their own: keys share dims, queries share draws, hidden units share units)
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight') or name in (
                        'transformer.wte.weight', 'resid_lambdas', 'x0_lambdas'):
                    gradients[name] = torch.zeros_like(gradients[name])
        if s.future_draws:
            # Use this capture's original measured hub targets, never corrected
            # targets. Only the source parameter gradients receive the addition.
            add_future_gradients(self, gradients)
        return gradients


# -------------------------------------------------------------------------------------------------- future probes


SCORE_COLUMNS = 32  # Smaller tiles avoid shared-memory/occupancy pressure on RTX 5090.


def norm_delta(x, delta):
    width = x.shape[-1]
    eps = torch.finfo(x.dtype).eps
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + eps)
    moved_inverse = torch.rsqrt(
        (square + 2 * (x * delta).sum(-1, keepdim=True)
         + delta.square().sum(-1, keepdim=True)) / width + eps)
    return delta * moved_inverse + x * (moved_inverse-inverse)


def rotate(x, cos, sin):
    half = x.shape[-1]//2
    return torch.cat((x[..., :half]*cos+x[..., half:]*sin,
                      -x[..., :half]*sin+x[..., half:]*cos), -1)


def receiver_changes(x, move, kraw, gate, ve, wkv, wg, cos, sin):
    """Finite K and effective-V changes of each independent source token."""
    m, p, width = move.shape
    heads, hd = kraw.shape[-2:]
    dn = norm_delta(x[:, None], move)
    projected = F.linear(dn, wkv).reshape(m, p, 2, heads, hd)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(kraw[:, None], c, s)
    dr = rotate(projected[:, :, 0], c, s)
    dk = norm_delta(r0, dr)
    dv = projected[:, :, 1]
    if gate is not None:
        dg = F.linear(dn[..., :wg.shape[-1]], wg)
        gate_change = 2*((gate[:, None]+dg).sigmoid()-gate[:, None].sigmoid())
        dv = dv + gate_change[..., None]*ve[:, None]
    return dk.permute(2, 1, 0, 3).contiguous(), dv.permute(2, 1, 0, 3).contiguous()


def joint_scores(q, target, dk, dv, a, contrast, source_start, window):
    """Exact single-column softmax update; output shape H,P,M."""
    dh = q.shape[-1]
    ds = torch.einsum("hsd,hpmd->hpsm", q, dk)*dh**-.5
    changed = a[:, None]*torch.expm1(ds)
    value_score = torch.einsum("hsd,hpmd->hpsm", target, dv)
    scores = (changed*contrast[:, None] + (a[:, None]+changed)*value_score)/(1+changed)
    s = torch.arange(q.shape[1], device=q.device)[:, None]
    t = source_start+torch.arange(dk.shape[2], device=q.device)[None]
    allowed = (s>t)&(s-t<=window)
    return (scores*allowed[None, None]).sum(2)


@cache
def score_kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(Q, G, DK, DV, A, C, Out,
               T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
               GH: tl.constexpr, GT: tl.constexpr, GD: tl.constexpr,
               AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
               CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
               Start, Window: tl.constexpr, BC: tl.constexpr, BQ: tl.constexpr):
        head = tl.program_id(1)
        col = tl.program_id(0)*BC+tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        offsets = (head*P*M+col[None, :])*D+features[:, None]
        dk = tl.load(DK+offsets, col[None, :]<P*M, 0)
        dv = tl.load(DV+offsets, col[None, :]<P*M, 0)
        first = (Start+tl.min(source, 0))//BQ*BQ
        last = tl.minimum(T, Start+tl.max(source, 0)+Window+1)
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start+tl.arange(0, BQ)
            q = tl.load(Q+head*QH+query[:, None]*QT+features[None, :]*QD,
                        query[:, None]<T, 0)
            g = tl.load(G+head*GH+query[:, None]*GT+features[None, :]*GD,
                        query[:, None]<T, 0)
            ds = tl.dot(q, dk, input_precision="tf32x3")*(D**-.5)
            vs = tl.dot(g, dv, input_precision="tf32x3")
            valid = (query[:, None]<T)&(col[None, :]<P*M)
            a = tl.load(A+head*AH+query[:, None]*AT+source[None, :]*AM, valid, 0)
            c = tl.load(C+head*CH+query[:, None]*CT+source[None, :]*CM, valid, 0)
            changed = a*libdevice.expm1(ds)
            allowed = (query[:, None]>Start+source[None, :])&(
                query[:, None]-Start-source[None, :]<=Window)
            response = (changed*c+(a+changed)*vs)/(1+changed)
            result += tl.sum(tl.where(valid&allowed, response, 0), 0)
        tl.store(Out+head*P*M+col, result, col<P*M)

    return kernel


@torch.library.custom_op("noise_record::future_scores", mutates_args=())
def fused_scores(q: torch.Tensor, target: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                 a: torch.Tensor, contrast: torch.Tensor, source_start: int,
                 window: int) -> torch.Tensor:
    if not q.is_cuda:
        return joint_scores(q, target, dk, dv, a, contrast, source_start, window)
    import triton
    h, p, m, d = dk.shape
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    score_kernel()[(triton.cdiv(p*m, SCORE_COLUMNS), h)](
        q, target, dk.contiguous(), dv.contiguous(), a, contrast, out,
        q.shape[1], m, p, d, *q.stride(), *target.stride(),
        *a.stride(), *contrast.stride(), source_start, window, SCORE_COLUMNS, 64,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@fused_scores.register_fake
def _(q, target, dk, dv, a, contrast, source_start, window):
    return torch.empty(dk.shape[:-1], device=q.device,
                       dtype=torch.float32 if q.is_cuda else q.dtype)


def signed_hidden_moves(raw, directions, step, weight):
    """Exact paired squared-ReLU responses for sign directions, with one large GEMM.

    The even component is independent of each +/-1 direction. This identity
    includes threshold crossings; it is not a derivative approximation.
    """
    base = raw.relu()
    plus, minus = (raw+step).relu(), (raw-step).relu()
    dp, dm = (plus-base)*(plus+base), (minus-base)*(minus+base)
    odd, even = (dp-dm)*.5, (dp+dm)*.5
    moved = F.linear(directions*odd[:, None], weight)
    center = F.linear(even, weight)[:, None]
    return torch.cat((center+moved, center-moved), 1)


def future_embedding_moves(raw, directions, step):
    delta = step[:, None]*directions
    return torch.cat((norm_delta(raw[:, None], delta), norm_delta(raw[:, None], -delta)), 1)


@cache
def query_kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(DQ, K, V, A, Base, Out,
               M: tl.constexpr, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               T: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
               VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
               AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
               OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
               BP: tl.constexpr, BK: tl.constexpr):
        token, head = tl.program_id(1) // H, tl.program_id(1) % H
        probe = tl.program_id(0)*BP+tl.arange(0, BP)
        feature = tl.arange(0, D)
        delta = tl.load(DQ+((token*P+probe[:, None])*H+head)*D+feature[None, :],
                        probe[:, None]<P, 0)
        numerator = tl.full((BP, D), 0, tl.float32)
        partition = tl.full((BP,), 0, tl.float32)
        for start in range(0, T, BK):
            key = start+tl.arange(0, BK)
            a = tl.load(A+head*AH+token*AM+key*AT, key<T, 0)
            # Causal/window-masked tiles are exactly zero, not a top-k truncation.
            if tl.sum(a, 0) > 0:
                k = tl.load(K+head*KH+key[None, :]*KT+feature[:, None]*KD, key[None, :]<T, 0)
                v = tl.load(V+head*VH+key[:, None]*VT+feature[None, :]*VD, key[:, None]<T, 0)
                ds = tl.dot(delta, k, input_precision="tf32x3")*(D**-.5)
                change = a[None, :]*libdevice.expm1(ds)
                partition += tl.sum(change, 1)
                numerator += tl.dot(change, v, input_precision="tf32x3")
        base = tl.load(Base+head*OH+token*OM+feature*OD)
        response = (numerator-partition[:, None]*base[None, :])/(1+partition[:, None])
        tl.store(Out+((token*P+probe[:, None])*H+head)*D+feature[None, :],
                 response, probe[:, None]<P)
    return kernel


@torch.library.custom_op("noise_record::future_query_response", mutates_args=())
def query_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   probabilities: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Stream the exact finite softmax response without a token/probe/key tensor."""
    if not dq.is_cuda:
        ds = torch.einsum("mphd,hsd->mphs", dq, keys)*dq.shape[-1]**-.5
        changed = probabilities.permute(1, 0, 2)[:, None]*torch.expm1(ds)
        partition = changed.sum(-1, keepdim=True)
        numerator = torch.einsum("mphs,hsd->mphd", changed, values)
        return (numerator-partition*base.transpose(0, 1)[:, None])/(1+partition)
    import triton
    m, p, h, d = dq.shape
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    query_kernel()[(triton.cdiv(p, 32), m*h)](
        dq.contiguous(), keys, values, probabilities, base, out, m, p, h, d, keys.shape[1],
        *keys.stride(), *values.stride(), *probabilities.stride(), *base.stride(),
        32, 64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@query_response.register_fake
def _(dq, keys, values, probabilities, base):
    return torch.empty_like(dq, memory_format=torch.contiguous_format)


def streamed_query_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight, base):
    m, p, width = directions.shape
    heads, _, hd = keys.shape
    delta = torch.cat((step[:, None]*directions, -step[:, None]*directions), 1)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = rotate(delta.reshape(m, 2*p, heads, hd), c, s)
    response = query_response(norm_delta(r0, dr), keys, values, probabilities, base)
    return F.linear(response.flatten(-2), weight)


def mlp_output_moves(mid, move, hidden, w_fc, w_out):
    """Carry each signed Q trial through its own block's complete MLP.

    Anchor at the captured hidden pre-activation, exactly as receiver_changes
    anchors K/V. This uses finite forward responses, including ReLU crossings.
    The incoming attention/residual move remains part of the block output.
    """
    dh = F.linear(norm_delta(mid[:, None], move), w_fc)
    base = hidden[:, None].relu()
    moved = (hidden[:, None] + dh).relu()
    return move + F.linear((moved-base)*(moved+base), w_out)


compiled_hidden = torch.compile(signed_hidden_moves, dynamic=False)
compiled_embedding = torch.compile(future_embedding_moves, dynamic=False)
compiled_query = torch.compile(streamed_query_moves, dynamic=False)


@torch.compile(dynamic=True)
def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin,
                    q, target, a, contrast, start, window):
    dk, dv = receiver_changes(x, move*coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return fused_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T


@torch.compile(dynamic=False)
def query_mlp_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight,
                    base, mid, hidden, w_fc, w_out):
    move = streamed_query_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight, base)
    return mlp_output_moves(mid, move, hidden, w_fc, w_out)


class FiniteFuture:
    def __init__(self, estimator, source_layers=4, tokens=64, chunk=512, sigma=.01, seed=9181,
                 query_mlp=True, timing=True):
        self.e = estimator
        self.source_layers, self.tokens, self.chunk = source_layers, tokens, chunk
        self.sigma, self.seed = sigma, seed
        self.query_mlp = query_mlp
        self.timing = timing
        self.contexts = None
        if tokens <= 0 or chunk <= 0 or sigma <= 0:
            raise ValueError("tokens, chunk and sigma must be positive")

    @torch.no_grad()
    def prepare(self, sequence=0):
        e = self.e
        d, model = e.capture, e.model
        t, heads = d.length, model.config.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        contexts = []
        for i, block in enumerate(model.transformer.h):
            prefix = f"transformer.h.{i}.attn."
            qraw, kraw, vraw = [d.outputs[prefix+n][sequence:sequence+1].float()
                               for n in ("c_q", "c_k", "c_v")]
            v = vraw.reshape(1, t, heads, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix+"ve_gate"][sequence].float()
                ve = d.outputs[f"value_embeds.{i}"][sequence].float().reshape(t, heads, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2*gate.sigmoid())[None, ..., None]*ve[None]
            values = v.transpose(1, 2)[0]
            qp, kp, a, out = attention_state(
                qraw, kraw, values[None], cos, sin, heads, model.window_sizes[i][0])
            q, k, a, out = qp[0], kp[0], a[0], out[0]
            target = e.measured["targets"][i][sequence].reshape(t, heads, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2))-(target*out).sum(-1, keepdim=True)
            contexts.append(dict(x=d.blocks[i][sequence].float(), q=q, k=k,
                kraw=kraw[0].reshape(t, heads, -1), values=values, a=a, out=out, contrast=contrast,
                target=target, gate=gate, ve=ve, wg=wg,
                wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(),
                window=model.window_sizes[i][0]))
            if self.query_mlp and i < self.source_layers:
                contexts[-1]["source_mlp"] = (
                    d.blocks[i][sequence].float()+d.outputs[prefix+"c_proj"][sequence].float(),
                    d.outputs[f"transformer.h.{i}.mlp.c_fc"][sequence].float(),
                    block.mlp.c_fc.weight.float(), block.mlp.c_proj.weight.float())
        self.contexts, self.sequence = contexts, sequence
        self.cos, self.sin = cos[0, :, 0], sin[0, :, 0]
        return self

    def sources(self, families):
        for family in families:
            if family == "wte":
                yield family, -1, "transformer.wte"
            else:
                for layer in range(self.source_layers):
                    suffix = "mlp.c_fc" if family == "hidden" else "attn.c_q"
                    yield family, layer, f"transformer.h.{layer}.{suffix}"

    def experiment(self, family, layer, raw, sl, receivers, start):
        """Bind clean tensors once per token chunk, outside the direction loop."""
        model = self.e.model
        if family == "hidden":
            response = partial(compiled_hidden, raw[sl],
                               weight=model.transformer.h[layer].mlp.c_proj.weight.float())
        elif family == "wte":
            response = partial(compiled_embedding, raw[sl])
        else:
            ctx = self.contexts[layer]
            kwargs = dict(cos=self.cos[sl], sin=self.sin[sl], keys=ctx["k"], values=ctx["values"],
                          probabilities=ctx["a"][:, sl], base=ctx["out"][:, sl],
                          weight=model.transformer.h[layer].attn.c_proj.weight.float())
            if self.query_mlp:
                mid, hidden, w_fc, w_out = ctx["source_mlp"]
                kwargs.update(mid=mid[sl], hidden=hidden[sl], w_fc=w_fc, w_out=w_out)
            response = partial(query_mlp_moves if self.query_mlp else compiled_query, raw[sl], **kwargs)
        scorers = []
        for j, coefficient in receivers:
            ctx = self.contexts[j]
            scorers.append(partial(receiver_scores, coefficient=coefficient,
                x=ctx["x"][sl], kraw=ctx["kraw"][sl],
                gate=None if ctx["gate"] is None else ctx["gate"][sl],
                ve=None if ctx["ve"] is None else ctx["ve"][sl],
                wkv=ctx["wkv"], wg=ctx["wg"], cos=self.cos[sl], sin=self.sin[sl],
                q=ctx["q"], target=ctx["target"], a=ctx["a"][:, :, sl],
                contrast=ctx["contrast"][:, :, sl], start=start, window=ctx["window"]))

        def score(move):
            if not scorers:
                return move.new_zeros(move.shape[:2])
            total = scorers[0](move)
            for receiver in scorers[1:]:
                total = total + receiver(move)
            return total
        return LocalExperiment(response, score)

    @torch.no_grad()
    def estimate(self, budgets=(512,), families=("hidden", "query", "wte")):
        assert self.contexts is not None
        e, model, d = self.e, self.e.model, self.e.capture
        budgets = sorted(set(budgets))
        if not budgets or budgets[0] <= 0:
            raise ValueError("Probe budgets must be positive")
        # Preserve every requested prefix when the execution tile is larger.
        chunk = gcd(self.chunk, *budgets)
        split, part = getattr(e, 'split', 1), getattr(e, 'part', 0)
        if split > 1:  # each GPU of the sequence takes a share of the draws, decoded at the full budget
            if len(budgets) != 1 or budgets[0] % split:
                raise ValueError('Split future probes need one budget divisible by the GPUs of a sequence')
            share = budgets[0] // split
            chunk = gcd(self.chunk, share)
            span = range(part * share, (part + 1) * share, chunk)
        else:
            span = range(0, max(budgets), chunk)
        result = {p: {} for p in budgets}
        timings = {}
        lam, lam0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        for family, layer, name in self.sources(families):
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw = d.outputs[name][self.sequence].float()
            t, width = raw.shape
            if budgets[-1] > width:
                raise ValueError(f"{name}: budget {budgets[-1]} exceeds the {width}-direction cycle")
            gen = torch.Generator(raw.device).manual_seed(self.seed+7919*(layer+2)+101*("hidden query wte".split().index(family)))
            # Per-token random signs/offsets prevent a shared low-rank projection
            # from persisting in the batch's parameter gradient.
            frame = hadamard(width, raw.device, torch.float32)
            permutation = torch.randperm(width, device=raw.device, generator=gen)
            offsets = torch.randint(width, (t,), device=raw.device, generator=gen)
            signs = 2*torch.randint(2, (t, width), device=raw.device, generator=gen)-1
            steps = self.sigma*raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            credits = {p: torch.zeros_like(raw) for p in budgets}
            receivers = []
            coeff = torch.ones((), device=raw.device)
            if family == "wte":
                for j in range(len(self.contexts)):
                    coeff = lam[j]*coeff+lam0[j]
                    receivers.append((j, coeff))
            else:
                for j in range(layer+1, len(self.contexts)):
                    coeff = coeff*lam[j]
                    receivers.append((j, coeff))
            for t0 in range(0, t, self.tokens):
                sl = slice(t0, min(t0+self.tokens, t))
                trial = self.experiment(family, layer, raw, sl, receivers, t0)
                accumulator = torch.zeros_like(raw[sl])
                for p0 in span:
                    p1 = p0+chunk
                    index = (offsets[sl, None]+permutation[None, p0:p1]) % width
                    rows = frame[index]*signs[sl, None]
                    accumulator += trial(rows, steps[sl])
                    if split > 1:
                        if p1 == (part + 1) * share:
                            credits[budgets[0]][sl] = accumulator/budgets[0]
                    elif p1 in credits:
                        credits[p1][sl] = accumulator/p1
            for p in budgets:
                result[p][name] = credits[p][None]
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            timings[name] = time.perf_counter()-start
        self.timings = timings
        return result


@torch.no_grad()
def add_future_gradients(estimator, gradients):
    """Add independent forward responses, releasing dense contexts per sequence.

    Draw a fresh frame seed from the checkpointed rank RNG each update. Hidden
    uses a partial cycle, so fixing that frame across training would bias it
    toward a persistent subspace. Never overwrite the measured hub targets.
    """
    e, s, d = estimator, estimator.s, estimator.capture
    e.maps.clear()
    seed = int(torch.randint(2**31, (), device=e.device,
                             generator=d.shared_generator if e.split > 1 else d.generator).item())
    future = FiniteFuture(e, source_layers=s.future_layers, tokens=s.future_tokens,
                          chunk=s.future_chunk, sigma=s.future_sigma, seed=seed,
                          query_mlp=True, timing=False)
    for sequence in range(d.batch):
        future.seed = seed + 104729 * sequence
        future.prepare(sequence)
        corrections = future.estimate((s.future_draws,))[s.future_draws]
        for site, credit in corrections.items():
            inputs = d.inputs[site][sequence]
            gradient = gradients[site + '.weight']
            if 'mlp.c_fc' in site:
                raw = d.outputs[site][sequence].float()
                step = s.future_sigma*raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                # Coordinates that are inactive in every signed trial have no
                # response; remove their partial-cycle decoding cross-talk.
                credit = credit * (raw+step > 0)[None]
            if site == 'transformer.wte':
                gradient.index_add_(0, inputs.flatten(), credit[0].flatten(0, -2))
            else:
                gradient.addmm_(credit[0].T, inputs)
        future.contexts = None
        del corrections


# ------------------------------------------------------------------------------------------------------ optimizer


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


RUN = (1194, 240, 50, 514)


def polar(g):
    """Approximate orthogonalization with six quintic maps."""
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (1.05 * x.norm() + 1e-6)).bfloat16()
    for a, b, c in MAPS:
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360."""

    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t = 0

    def rail_beta(self):
        """Scheduled fast-rail and Nesterov beta."""
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                g = p.grad.float()
                lanes = 1 if p.size(0) >= p.size(1) else 0
                state = self.state[p]
                if not state:
                    state.update(fast=torch.zeros_like(g), slow=torch.zeros_like(g),
                                 energy=torch.zeros_like(g.mean(lanes, keepdim=True)))
                fast, slow, energy = state['fast'], state['slow'], state['energy']
                fast.lerp_(g, 1 - fast_beta)
                slow.lerp_(g, 1 - self.slow_beta)
                update = polar(g.lerp(weight * fast + (1 - weight) * slow, beta))
                energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                gained = update * energy.clamp_min(1e-10).rsqrt()
                update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                w = p.float()
                decay = (slow * w >= 0) * w * (wd * lr)
                p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
        self.t += 1


def newton_schulz(x):
    """The record's six quintic maps on a stack (B, n, n) of square matrices, in bfloat16."""
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    """The same maps on wide matrices (n < m), iterated on the n x n Gram matrix (Gram Newton-Schulz, Dao lab 2026).
    R is rebuilt from Q X before the maps in `restarts` to bound the float16 error."""
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


def polar_batch(g, margin=1.1):
    """Approximate orthogonalization of a stack (B, r, c) of same-shape matrices."""
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
                 ratio: float, margin: float):
    """One ANVIL II step for a stack of same-shape matrices; updates fast, slow and energy in place."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class BatchedAnvil(Anvil):
    """ANVIL II from modded-nanogpt PR #360, applied to all same-shape matrices at once."""

    def __init__(self, *args, polar_margin=1.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.polar_margin, self.stacks = polar_margin, {}

    def load_state_dict(self, state_dict):
        # Cached stacks must be rebuilt from the freshly restored per-parameter state.
        self.stacks.clear()
        return super().load_state_dict(state_dict)

    def stacked_state(self, shape, params):
        """The moving averages of one shape group as stacked tensors. Each parameter's state holds views into them,
        so the Boost -> Cut handoff, which copies per-parameter state, carries them over."""
        key = tuple(params)
        stack = self.stacks.get(key)
        if stack is not None and all(self.state[p].get('fast') is not None for p in params):
            return stack
        lanes = 1 if shape[0] >= shape[1] else 0
        weights = torch.stack([p.float() for p in params])

        def gathered(key, like):
            held = [self.state[p].get(key) for p in params]
            return torch.stack(held).clone() if all(h is not None for h in held) else torch.zeros_like(like)

        stack = dict(fast=gathered('fast', weights), slow=gathered('slow', weights),
                     energy=gathered('energy', weights.mean(lanes + 1, keepdim=True)))
        for i, p in enumerate(params):
            self.state[p].update(fast=stack['fast'][i], slow=stack['slow'][i], energy=stack['energy'][i])
        self.stacks[key] = stack
        return stack

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        device = self.param_groups[0]['params'][0].device
        value = lambda v: torch.tensor(float(v), device=device)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            by_shape = {}
            for p in group['params']:
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, params in by_shape.items():
                stack = self.stacked_state(shape, params)
                new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                                   stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                                   value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                                   1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5, self.polar_margin)
                torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        # Identical all-reduced gradients must produce identical rank replicas.
        # Scope this to updates so embedding scatter and probes keep their fast paths.
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for o in self.optimizers:
                o.step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_model(device, seed, config, projection_std, head_std=0.001):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # Output projections: uniform with the declared std / sqrt(fan_in).
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = projection_std * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        if head_std != 0.001:
            nn.init.normal_(model.lm_head.weight, std=head_std)
    return model.requires_grad_(False)


def make_optimizer(model, recipe, steps):
    """ANVIL II on block matrices, Adam elsewhere."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = (recipe.emb_scale if ('wte' in name or 'value_embeds' in name)
                 else recipe.lambda_scale if 'lambdas' in name else 1)
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
        betas=recipe.betas, eps=1e-8, weight_decay=0.0,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
                         polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    return Optimizers(anvil, adam)


def set_rate(optimizer, step, steps, warmdown, final=0.0):
    """Set the original val linear warmdown rate, with one-based training steps (down to `final` of the rate)."""
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


# ------------------------------------------------------------------------------------------------------- training


def save_training_state(path, model, optimizer, estimator, *, step, train_seconds, recipe, progress=None):
    """Atomically save common training state and each rank's independent RNGs."""
    device = estimator.device
    random = dict(generator=estimator.generator.get_state(),
                  shared_generator=estimator.shared_generator.get_state(),
                  cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    if isinstance(estimator, Boost):
        random['capture_generator'] = estimator.capture.generator.get_state()
    states = [None] * estimator.world
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(states, random)
    else:
        states[0] = random
    if estimator.rank == 0:
        memory = getattr(estimator, 'memory', None)
        checkpoint = dict(step=step, train_seconds=train_seconds, recipe=recipe, world=estimator.world,
            schedule_steps=optimizer.optimizers[0].total,
            model=model.state_dict(), rng=states,
            estimator=type(estimator).__name__, progress=progress,
            optimizers=[dict(state=o.state_dict(), t=getattr(o, 't', None), kind=type(o).__name__)
                        for o in optimizer.optimizers],
            memory=None if memory is None else {name:getattr(memory, name) for name in
                    ('sites', 'beta', 'ridge', 'alpha_beta', 'key_moment', 'cross_moments', 'alpha', 'calibration')})
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def restore_training_state(path, model, optimizer, estimator, *, recipe):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    canonical = lambda x: json.loads(json.dumps(x))
    if canonical(checkpoint['recipe']) != canonical(recipe) or checkpoint['world'] != estimator.world:
        raise ValueError('Resume requires the same recipe and number of ranks')
    if checkpoint['estimator'] != type(estimator).__name__:
        raise ValueError('Resume requires the same estimator phase')
    if checkpoint['schedule_steps'] != optimizer.optimizers[0].total or checkpoint['memory'] is not None:
        raise ValueError('Resume requires the same schedule and no auxiliary memory')
    model.load_state_dict(checkpoint['model'], strict=True)
    for active, saved in zip(optimizer.optimizers, checkpoint['optimizers'], strict=True):
        if saved['kind'] != type(active).__name__:
            raise ValueError('Resume requires the same optimizer implementation')
        active.load_state_dict(saved['state'])
        if saved['t'] is not None:
            active.t = saved['t']
    optimizer.param_groups = [g for active in optimizer.optimizers for g in active.param_groups]
    random = checkpoint['rng'][estimator.rank]
    estimator.generator.set_state(random['generator'])
    estimator.shared_generator.set_state(random['shared_generator'])
    estimator.capture.generator.set_state(random['capture_generator'])
    torch.set_rng_state(random['cpu'])
    if random['cuda'] is not None:
        torch.cuda.set_rng_state(random['cuda'], estimator.device)
    return checkpoint['step'], checkpoint['train_seconds']


def stage_schedule(rows, settings):
    """Disjoint data slices, monotonically numbered updates, and phase-specific scheduler clocks."""
    b1, b2 = settings.boost_batch, settings.cut_batch
    if rows % b2 or b2 % b1:
        raise ValueError('Data rows must divide into complete Cut batches, and Cut batch must divide by Boost batch')
    n1, n2 = rows // b1, rows // b2
    alignment = b2 // b1
    boost = n1 if settings.estimator == 'boost' else round(settings.switch * n1) // alignment * alignment
    batches = [('boost', i, (i-1)*b1, i*b1, n1) for i in range(1, boost+1)]
    if settings.estimator != 'boost':
        batches += [(settings.estimator, i, (i-1)*b2, i*b2, n2) for i in range(boost*b1//b2+1, n2+1)]
    return batches


def handoff_optimizer(old, new, phase_step):
    """Keep ANVIL rails and Adam moments; change only the new phase's ANVIL scheduler clock."""
    for new_o, old_o in zip(new.optimizers, old.optimizers, strict=True):
        new_o.state.update(old_o.state)
    new.optimizers[0].t = phase_step - 1


def holdout_sum(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, y, loss_reduction='sum')


holdout_sum_graphed = torch.compile(holdout_sum, mode='reduce-overhead', dynamic=False)
MERGE_GRAPH = False  # set from Settings.merge_graph


def holdout_loss(model, rows, rank, world):
    """Mean next-token loss over the held-out training sequences, split across ranks."""
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    forward = holdout_sum_graphed if MERGE_GRAPH else holdout_sum
    with torch.no_grad():
        for batch in rows[rank::world].split(8):
            batch = batch.to(device)
            loss = forward(model, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
            total[0] += loss.double()
            total[1] += batch[:, 1:].numel()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])  # one matrix of block i (attn.c_q, ..., mlp.c_proj)
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 64  # training sequences that feed no update and score the end merge
MERGE_FROM, MERGE_EVERY = 0.6, 5  # the exponential means follow the weights from 60% of the epoch, every 5 updates
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)  # weight of the newest weights in each mean
MERGE_RHOS = (0.0, 0.5, 1.0)  # how far a candidate moves outside the high-curvature subspace
MERGE_K, MERGE_TOKENS = 64, 512  # input-covariance directions per linear layer; frequent-token rows per table


@torch.no_grad()
def curvature_bases(model, rows, rank, world, k=MERGE_K):
    """Per linear layer, the top-k eigenvectors of its input covariance E[x x^T], and the token order of
    `rows` by frequency (for the embedding tables), from forward passes of the current weights on `rows` (the
    held-out training sequences; no gradient). Computed on rank 0 and broadcast, so every rank merges identically."""
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
              if isinstance(mod, nn.Linear) and 've_gate' not in name]
    bases = {key: torch.zeros(mod.weight.shape[1], k, device=device) for key, mod in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        acc, hooks = {}, []
        for key, mod in layers:
            def hook(module, inputs, output, key=key):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                acc[key] = acc.get(key, 0) + (x.T @ x).double()
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
            bases[key] = torch.linalg.eigh(acc[key])[1].flip(1)[:, :k].float().contiguous()
        order = torch.argsort(counts, descending=True)
    if world > 1:
        for key, _ in layers:
            torch.distributed.broadcast(bases[key], 0)
        torch.distributed.broadcast(order, 0)
    return bases, order


def subspace_move(name, move, bases, order):
    """The part of a weight move inside its high-curvature subspace: the top input-covariance directions of a
    linear layer, the rows of the most frequent tokens of an embedding table; other tensors whole."""
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


MERGE_COEFS = (0.75, 1.0, 1.25)
MERGE_FIRST_KEYS = 0  # 0: the first sweep tries every key per part; else the best this many (Settings.merge_keys)  # multiples of each move in the joint search


@torch.no_grad()
def joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain,
                sweeps=2, shortlist=8, first_keys=0):
    """The joint search over keys (rho, j, c) = final + c * move(rho, j) per part (None = the final weights):
    the best single key for every part, then coordinate sweeps (all keys per part, then each part's best `shortlist`)."""
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
    held = dict.fromkeys(parts)  # the key each part's weights hold now (the final weights on entry)

    def show(choice):  # load only the parts whose key changed: the same weights as loading all
        for p in parts:
            if held[p] != choice[p]:
                for i in members[p]:
                    params[i].copy_(value(i, choice[p]).to(params[i].dtype))
                held[p] = choice[p]
    best_key, best = None, plain
    global_losses = []
    for key in keys[1:]:
        show(dict.fromkeys(parts, key))
        loss = holdout_loss(model, rows, rank, world)
        global_losses.append((loss, key))
        if loss < best:
            best_key, best = key, loss
    choice = {p: best_key for p in parts}
    if first_keys:  # the first sweep tries only the keys that did best for all parts at once (and no move)
        keys = [None] + [k for _, k in sorted(global_losses, key=lambda x: x[0])[:first_keys]]
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            tried = []
            for key in (keys if sweep == 0 else ranked[p]):
                if key == choice[p]:
                    tried.append((best, key))
                    continue
                trial = dict(choice)
                trial[p] = key
                show(trial)
                loss = holdout_loss(model, rows, rank, world)
                tried.append((loss, key))
                if loss < best:
                    best, choice = loss, trial
            if sweep == 0:
                ranked[p] = [k for _, k in sorted(tried, key=lambda x: x[0])[:shortlist]]
    show(choice)
    return dict(holdout_plain=plain, holdout_merged=best,
                merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]]) for p, k in choice.items()})


@torch.no_grad()
def end_merge(model, means, rows, rank, world):
    """The end merge (module docstring): per part, candidates final + c * (inside + rho * (full - inside)), full = an
    exponential mean minus the final weights and inside its part in the high-curvature subspace (curvature_bases on the
    held-out rows), chosen on the held-out loss by joint_parts. Every rank takes the same decisions (the losses are
    all-reduced)."""
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows, rank, world)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]

    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)
    plain = holdout_loss(model, rows, rank, world)
    return joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain,
                       first_keys=MERGE_FIRST_KEYS)


# ----------------------------------------------------------------------------------------- identity-path moment kernels
#
# Triton kernels for track 2's trainer: Triton versions of #29's Pallas kernels.
#
# - `moment_query_scores` / `moment_key_scores`: the exact single-coordinate query (key) experiments as attention-weighted
#   moments on tensor cores, degree 2 in the logit change (#29's kernels/exact_moments.py). The features (X, Y, X^2,
#   Y^2, XY of the other side's rotated columns) are built per tile in the kernels from the other side's processed rows,
#   the X and Y sides as separate products (the shared XY once), summed in the epilogue with the moved side's closed
#   coefficients (from its raw rows); the values' credit sum_s A[s, t] hub_s rides in the key kernel, and the credits
#   are stored by token in the head credits' all-to-all layout.

# moment kernels
# PR #29's exact query / key experiments as attention-weighted moments on tensor cores (degree 2 in the logit change):
# per (moved position, direction), dl = A L + Bu X + Bw Y; the response's polynomial in dl splits into products of a
# per-pair weight (a c L^m, a L^m) and per-(other position, direction) features ([X|Y], [X^2|Y^2], [XY|XY]); the sums
# over the other positions are [moved, other] x [other, 2J] products and the coefficients apply once at the end.


@triton.jit
def _side_features(side_ptr, offs, part, D: tl.constexpr, J: tl.constexpr, offs_j, scale):
    """The other side's rotated columns X, Y at rows `offs` of side (its processed rows [T, D] f32, at this row) for
    directions part J + offs_j: the pair (d mod D/2, d mod D/2 + D/2) each direction d moves, in logit units."""
    i = (part * J + offs_j) % (D // 2)
    x = tl.load(side_ptr + offs[:, None] * D + i[None, :]) * scale
    y = tl.load(side_ptr + offs[:, None] * D + (i + D // 2)[None, :]) * scale
    return x, y


@triton.jit
def _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, part, M, RAW_GPU,
                         RAW_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    """The moved side's closed coefficients A, Bu, Bw [BLOCK, J] at rows offs for directions part J + offs_j: a move of
    the raw row (raw [world, M, .] bf16 by token as received, GPUs RAW_GPU and tokens RAW_STRIDE apart; its squared
    norm sq and step size steps [BH, T] f32) along d changes
    the processed row by A (itself) + B (u, w) (the rotated pair), as #29's closed form (`closed_coefficients`)."""
    d = part * J + offs_j
    i = d % (D // 2)
    n = (row % BH).to(tl.int64) * T + offs  # token n of the block's flattened rows: GPU n // M, place n % M
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
    """Store x [BLOCK, J] (row `row`'s tokens offs, dims part J ..) into out [world, .., M, D] by token: token n = row T
    + t of the flattened rows sits at GPU n // M, place n % M (the head credits' all-to-all layout; W_STRIDE apart)."""
    n = (row % BH).to(tl.int64) * T + offs
    tl.store(out_ptr + (n // M * W_STRIDE + n % M * D + part * J)[:, None] + offs_j[None, :], x)


@triton.jit
def _split4(acc, BLOCK: tl.constexpr, J: tl.constexpr):
    """The products with the interleaved features [X Y X^2 Y^2] apart: (X, Y, X^2, Y^2), each [BLOCK, J]."""
    first, second = tl.split(tl.reshape(acc, (BLOCK, J, 2, 2)))  # [X Y] and [X^2 Y^2]
    x, y = tl.split(first)
    xx, yy = tl.split(second)
    return x, y, xx, yy


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr,
                                 hubout_ptr, steps_ptr, out_ptr, win_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
                                 RAW_STRIDE, M, OUT_BLK, W_STRIDE,
                                 D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S  # the longest causal sums first
    row, part = i // REPEAT, i % REPEAT  # rows of every block's batch, BH a block
    blk = row // BH
    window = tl.load(win_ptr + blk)
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
    side = side_ptr + base  # the processed keys [BH, T, D] (f32): their rotated columns
    # numerator / denominator products per (power of L, feature), the features interleaved per direction so that
    # each weight takes one product: [X Y X^2 Y^2] (n4, d4), [X Y] (l2, m2), and the shared XY (nxy, dxy)
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
    first = tl.maximum(s0 - window, 0) // BLOCK_T
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    for it in range(first, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale  # [BS, BT]
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
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
                                      RAW_GPU, RAW_STRIDE, D,
                                      J, offs_j)  # the raw queries'
    nx0, ny0, nx1, ny1 = _split4(n4, BLOCK_S, J)
    dx0, dy0, dx1, dy1 = _split4(d4, BLOCK_S, J)
    lx0, ly0 = tl.split(tl.reshape(l2, (BLOCK_S, J, 2)))
    mx0, my0 = tl.split(tl.reshape(m2, (BLOCK_S, J, 2)))
    # X side: P = Bu, Q = Bw; Y side: P = Bw, Q = Bu (degree 2: P, P^2 / 2, PQ / 2 each, the L-sums' A / 2, A^2 / 4)
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
          + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
          + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_s)
    _store_by_token(out_ptr, zn / (1.0 + zd) / steps[:, None], row, BH, T, offs_s, part, M, W_STRIDE, D, J, offs_j)


def moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, windows, repeat, block_s=64,
                        block_t=32, num_warps=4, num_stages=2):
    """Query credits of every running block at once into out: q, k, v, hub [K BH, T, D] bf16 (q, k processed; the K
    blocks' batches of BH rows stacked); raw [world, K, M, .] bf16 (the raw queries by token as received, any strides);
    sq, steps [K BH, T] f32 their squared norms and move sizes (the scores are divided by them); side [K BH, T, D] f32
    the processed keys; cos, sin [T, D / 2] the rotary tables; lse, hub_out [K BH, T] f32; out [world, K, M, D]
    (f32 or bf16) by token (see `_store_by_token`), M the tokens per GPU; windows [K] int32 each block's attention
    window. The features and the closed coefficients are built tile by tile in the kernel, D / repeat directions per
    program; the query blocks run longest causal sum first."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_queries_kernel[(rows * repeat, t // block_s)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps,
                                                          out, windows, rows // blocks, t, d ** -0.5, repeat,
                                                          raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
                                                          out.stride(1), out.stride(0), D=d, J=d // repeat,
                                                          BLOCK_T=block_t, BLOCK_S=block_s, num_warps=num_warps,
                                                          num_stages=num_stages)


@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
                              steps_ptr, out_ptr, value_ptr, win_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
                              RAW_STRIDE, M, OUT_BLK, W_STRIDE,
                              D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    row, part = i // REPEAT, i % REPEAT  # rows of every block's batch, BH a block
    blk = row // BH
    window = tl.load(win_ptr + blk)
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    side = side_ptr + base  # the processed queries [BH, T, D] (f32): their rotated columns
    p2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w1 [X Y] (interleaved per direction)
    q2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w2 [X^2 Y^2]
    qxy = tl.zeros((BLOCK_T, J), tl.float32)  # w2 XY (shared)
    r2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w2 L [X Y]
    val = tl.zeros((BLOCK_T, J), tl.float32)  # the value credit sum_s A[s, t] hub_s, this row's J of the D dims
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    first = t0 // BLOCK_S
    last = tl.minimum((t0 + BLOCK_T - 1 + window) // BLOCK_S + 1, T // BLOCK_S)
    for it in range(first, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub_part = tl.load(hub_ptr + base + offs_s[:, None] * D + part * J + offs_j[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale  # [BT, BS]
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        x, y = _side_features(side, offs_s, part, D, J, offs_j, scale)
        w1 = a * c  # ch / (1 + ch) = a dl + (a / 2 - a^2) dl^2 + ...
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
                                      RAW_GPU, RAW_STRIDE, D,
                                      J, offs_j)  # the raw keys'
    px, py = tl.split(tl.reshape(p2, (BLOCK_T, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BLOCK_T, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BLOCK_T, J, 2)))
    z = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + qxy * (2.0 * bu * bw) + 2.0 * ca * (rx * bu + ry * bw)
         + s1[:, None] * ca + s2[:, None] * (ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_t)
    _store_by_token(out_ptr, z / steps[:, None], row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)
    _store_by_token(value_ptr + blk.to(tl.int64) * OUT_BLK, val, row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)


def moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, windows, repeat,
                      block_t=64, block_s=64, num_warps=4, num_stages=2):
    """Key credits of every running block into out: raw, sq, steps of the raw keys, side the processed queries; as
    above. Also the values' credit sum_s A[s, t] hub_s into values (out's layout), each of a head row's programs
    D / repeat of its D dims."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_keys_kernel[(rows * repeat, t // block_t)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
                                                       values, windows, rows // blocks, t, d ** -0.5, repeat,
                                                       raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
                                                       out.stride(1), out.stride(0), D=d, J=d // repeat,
                                                       BLOCK_T=block_t, BLOCK_S=block_s, num_warps=num_warps,
                                                       num_stages=num_stages)


# -------------------------------------------------------------------------------------------- identity-path phase
#
# The identity-path phase of the hybrid record (FineWeb-30M / Noise track 2): the 10M track-2 record's trainer
# (identity-path credit, one head per GPU, moment kernels, structured start, tail average, late skip) as a phase that
# runs before Cut.
#
# Its history, from Track 2: #37 (#32's trainer with #33's structured start, tail average and late skip, overlapped collectives and a
# compiled forward, 450 updates) with a faster runtime: the moment kernels build their features in-kernel, the
# weight-gradient products run on bf16 operands, and the optimizer steps write their state in place.
#
# The recipe is #37's, unchanged. The runtime, against #37's:
#   - The moment kernels (the section above) take the raw queries' / keys' closed coefficients [3, T, J] and the other side's
#     rotated columns [2, T, J] as they are and build the features (X, Y, X^2, Y^2, XY) tile by tile, the X and Y sides
#     as separate products (the shared XY once), summed in the epilogue; #37 built [3, T, 2J] feature and coefficient
#     stacks before every call. The query kernel runs its latest query blocks first (their causal sums are the longest,
#     so the kernel no longer waits on them at its tail), the key kernel takes 32 directions per call (16 before), and
#     the tiles come from a sweep (both kernels 3 pipeline stages, keys 64 x 32). Same experiments: against #37's
#     kernels, cos 1.0 and 3e-4 / 5e-8 relative error (queries / keys).
#   - The weight-gradient products (input^T @ error per block matrix) run on bf16 stacks with fp32 accumulation and
#     write bf16 gradient stacks, which are reduce-scattered in bf16 and cast and scaled inside ANVIL II's compiled step;
#     #37 ran them on fp32 stacks (TF32) into fp32 stacks. Each gradient is rounded to bf16 once.
#   - Adam's and ANVIL II's compiled steps write the parameters and their state in place, instead of returning new
#     tensors copied back one by one; the bf16 embeddings' gradients are rounded inside Adam's step (three eager kernels
#     each before).
#
# #37's description of its own changes follows.
#
# #32's trainer (#13 at 8 rows per update, @Euraba; #13 = #29's identity-path credit with moment kernels and one head
# per GPU, @RobertTLange, in PyTorch) with #33's structured start, tail average and late skip (#34's skip start), in 450
# updates (#36), with a faster runtime: overlapped collectives and a forward compiled as one graph.
#
# The estimator, the optimizers and the work split are #32's, unchanged (below). What this adds, each switchable:
#   - The start (`--init prev8`, #12's formula and constants): every head of block 1 starts as a local RoPE head peaking
#     at lag 1. Its query and key columns get a rank-one term on the block input's token-shared direction
#     (`shared_means`, in closed form from the initial weights) and keep a quarter of their random map. A fixed formula
#     of the seed and the architecture: no data, statistic or forward pass.
#   - The tail average (`--ema fixed`): an fp32 EMA of every weight from 60% of the run (rate 1 / round(0.25 N)), kept
#     inside each later update's CUDA graph; after the last update, inside the timer, the weights move 0.4 of the way to
#     it. A fixed mix: nothing is measured or selected.
#   - The late skip (`--skip late`): from 20% of the run blocks 0, 2, 4 and 6 run no query / key / value experiments
#     (they leave the head pack and both all-to-alls), so their queries, keys and values get no credit and keep their
#     weights (the optimizer's moves on them are undone). Fewer measurements and no new signal. Each skip phase is its
#     own CUDA graph, captured in the warmup.
#   - The run length (`--updates`): N = 450 updates instead of 600, the first N batches of #32's data order, with every
#     schedule (the row-length switch at 0.8 N, the cooldown over the last 0.4 N, ANVIL's phases) on that length.
# With `--updates 600 --init none --ema none --skip none` the script runs #32's update: the same math, but the forward
# is compiled as one graph (#32's runs eagerly, below), so its floating-point rounding differs.
#
# The runtime, against #36's (the same arithmetic otherwise, op for op):
#   - The MLP matrices' gradients are reduce-scattered while the head credits run. The square stack's reduce-scatter,
#     the all-reduce of the other gradients and both ANVIL all-gathers are asynchronous, each under the next step's work
#     (ANVIL steps the MLP stack first, as its gradients arrive first; Adam runs while the matrices gather).
#   - The rotary tables are computed once per row length.
#   - The forward compiles as one graph: flex attention's log-sum-exp comes through `return_aux=AuxRequest(lse=True)`.
#     `return_lse=True` warns, and the warning broke the compiled forward inside its block loop, so the eight blocks
#     ran eagerly as many small kernels. Fused, the forward rounds differently (same math).
#
# #32's trainer: #13's estimator and work split at 8 rows per update (600 updates, ANVIL II's lr 0.032). One process per
# GPU: each runs the clean pass on its own rows (with fewer rows than GPUs, GPUs share a row and keep their part of it),
# the readout cycle (a degree-3 expansion as six matrix products) and the MLP steps on its tokens, and one head's
# queries, keys and values over the batch: their inputs arrive in one all-to-all, the exact experiments run as
# attention-weighted moments on tensor cores (degree 2, the section above: Triton versions of #29's Pallas kernels), and the
# credits go back by token in a second all-to-all. The ANVIL matrices' gradients are reduce-scattered to the GPU that
# steps them. Each update is one CUDA graph per row length (here also per tail-average and skip phase; collectives
# captured inside) that reads its batch and step index from device memory.
#
#     torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
#
# Weights are kept in the JAX trainer's layout ((in, out) matrices, the head (d, vocab)) so formulas read as in #29;
# `state_dict()` exports utils/model.py's layout.
#
# Its globals are its own (prefixed id_ / ID_ where they would collide with Cut's).

F32, BF16 = torch.float32, torch.bfloat16

# recipe (#25 / #29's shape at 8 rows per update (#32): cooldown the last 40%, 1024-token rows until 80%)
BATCH, SEQ, STEPS, ID_SEED = 8, 2048, 405, 0  # the run length: 405 updates (#36 / #37: 450, #32: 600)
WARMUP_LEN, WARMUP_UNTIL = 1024, STEPS * 4 // 5
COOLDOWN = STEPS * 2 // 5
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8  # Adam twice #37's rates (0.003, 0.9, 0.003)
LR_WTE, LR_VE, LR_LAMBDAS = 1.8, 1.8, 0.006
ANVIL_LR, ANVIL_WD = 0.032, 2.25  # 0.8x #29's: at 8 rows 0.04 spikes on some seeds

GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1

ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
              (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
              (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240, 50, 514)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_LAG, START_RHO, START_CONTENT = 1, 1, 0.8, 0.25  # every head of block 1 a local RoPE head at lag 1
EMA_FROM, EMA_HORIZON, EMA_MIX = 0.6, 0.25, 0.4  # tail average from 60% of the run over a quarter of it; final mix
START, EMA = True, True  # the structured start and the tail average (--init none / --ema none: #32's recipe)
SKIP_BLOCKS, SKIP_FROM, SKIP = (0, 2, 4, 6), 0.2, True  # late skip: no attention experiments in these blocks


def id_has_ve(layer):
    return layer % 2 == (L - 1) % 2


VE_BLOCKS = tuple(i for i in range(L) if id_has_ve(i))


def window(layer):
    if layer == L - 1:
        return SEQ
    return SEQ if 'SSSL'[layer % 4] == 'L' else SEQ // 2


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


# ------------------------------------------------------------------------------------------------------------- model

class Model:
    """Parameters (JAX layout), optimizer state and the static buffers of the captured step."""

    def __init__(self, device, rank, world):
        self.device, self.rank, self.world = device, rank, world
        g = torch.Generator(device='cpu').manual_seed(ID_SEED)
        s = math.sqrt(3.0) * D ** -0.5
        uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
        proj = lambda fan_in, fan_out: uni((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
        # block matrices live in two stacks, oriented wide side out as ANVIL steps them: square [kind * L + layer]
        # (q, k, v, proj), wide [c_fc layers, then c_proj layers transposed]
        self.square = torch.stack([uni((D, D), s) if k != 'attn.c_proj' else proj(D, D)
                                   for k in SQUARE for _ in range(L)]).to(device)
        self.wide = torch.stack([uni((D, 4 * D), s) for _ in range(L)] + [proj(4 * D, D).T for _ in range(L)]).to(device)
        self.gates = {i: torch.zeros(GATE_CHANNELS, H, device=device) for i in range(L) if id_has_ve(i)}
        self.wte = torch.randn(VOCAB, D, generator=g).to(device, BF16)
        self.lm_head = (0.001 * torch.randn(D, VOCAB, generator=g)).to(device)
        self.ve = {i: uni((VOCAB, D), s).to(device, BF16) for i in range(L) if id_has_ve(i)}
        self.lam_r = torch.ones(L, device=device)
        self.lam_0 = torch.full((L,), 0.1, device=device)
        if START:
            structured_start(self)
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
        # the block matrices' gradients in bf16 end to end: written by the bf16 products, reduce-scattered in bf16,
        # cast and scaled inside ANVIL's compiled step
        self.grad['square'] = torch.zeros(self.square.shape, device=device, dtype=BF16)
        self.grad['wide'] = torch.zeros(self.wide.shape, device=device, dtype=BF16)
        self.owned = {name: torch.zeros(w.shape[0] // world, *w.shape[1:], device=device, dtype=BF16)
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
        self.H = id_hadamard(D, device)
        self.rows = torch.randint(0, VOCAB, (STEPS, BATCH, SEQ + 1), device=device)  # the batches (random until training)
        self.losses = torch.zeros(STEPS, device=device)
        self.step_t = torch.zeros((), device=device)  # the update index (0-based)
        self.scale_t = torch.ones((), device=device)  # the warmup-stable-decay factor
        self.rates = dict(square=torch.ones(4 * L, device=device),
                          wide=torch.tensor([2.0] * L + [1.0] * L, device=device))  # sqrt(max(1, out / in))
        self.square_bf, self.wide_bf = self.square.to(BF16), self.wide.to(BF16)  # the forward's bf16 weights
        # during training only this GPU's share of the fp32 stacks is current (ANVIL steps it and all-gathers its bf16
        # cast into the forward's weights); the whole fp32 stacks are gathered once at the end (`gather_fp32`)
        self.shard = {name: slice(rank * (w.shape[0] // world), (rank + 1) * (w.shape[0] // world))
                      for name, w in (('square', self.square), ('wide', self.wide))}
        self.lm_head_bf = self.lm_head.to(BF16)
        self.gates_bf = {i: g.to(BF16) for i, g in self.gates.items()}
        self.ema = [torch.zeros(w.shape, device=device) for w in self.averaged()] if EMA else None
        self.windows = {run: torch.tensor([window(i) for i in run], device=device, dtype=torch.int32)
                        for run in (tuple(range(L)), tuple(i for i in range(L) if i not in SKIP_BLOCKS))}
        self.skip_rows = {SKIP_BLOCKS: torch.tensor([kind * L + i for kind in range(3) for i in SKIP_BLOCKS],
                                                    device=device)}  # the square stack's q, k, v rows of the skip
        mod = lambda w: (lambda b, h, qi, ki: (qi >= ki) & (qi - ki <= w))
        self.masks = {(w, t): create_block_mask(mod(w), None, None, t, t, device=device)
                      for w in (SEQ // 2, SEQ) for t in (WARMUP_LEN, SEQ)}
        self.rope = {t: rotary(t, device) for t in (WARMUP_LEN, SEQ)}  # cos, sin per row length

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
        """The Adam parameters' bf16 copies (ANVIL gathers the stacks' bf16 weights itself)."""
        self.lm_head_bf.copy_(self.lm_head)
        for i, g in self.gates.items():
            self.gates_bf[i].copy_(g)

    @torch.no_grad()
    def cast_all(self):
        """Every bf16 copy from the whole fp32 weights (at the start)."""
        self.square_bf.copy_(self.square)
        self.wide_bf.copy_(self.wide)
        self.refresh_bf16()

    @torch.no_grad()
    def gather_fp32(self):
        """The whole fp32 stacks from every GPU's share (at the end)."""
        for name, w in (('square', self.square), ('wide', self.wide)):
            dist.all_gather_into_tensor(w, w[self.shard[name]].clone())

    def averaged(self):
        """The weights the tail average follows (every parameter, in a fixed order; of the stacks this GPU's share)."""
        return [self.square[self.shard['square']], self.wide[self.shard['wide']], self.wte, self.lm_head, self.lam_r,
                self.lam_0, *self.gates.values(), *self.ve.values()]

    def tensors(self):
        """Every tensor the training state consists of (for snapshots)."""
        out = [self.square, self.wide, self.wte, self.lm_head, self.lam_r, self.lam_0, *self.gates.values(),
               *self.ve.values(), *self.m, *self.v]
        return out + [x for st in self.anvil.values() for x in st.values()]

    def state_dict(self, device='cpu'):
        """utils/model.py's state dict (float32, its layout) on `device`."""
        sd = {'transformer.wte.weight': self.wte.float(), 'lm_head.weight': self.lm_head.T,
              'resid_lambdas': self.lam_r, 'x0_lambdas': self.lam_0}
        for i in range(L):
            for kind in (*SQUARE, 'mlp.c_fc', 'mlp.c_proj'):
                sd[f'transformer.h.{i}.{kind}.weight'] = self.w(i, kind).T
            if id_has_ve(i):
                sd[f'transformer.h.{i}.attn.ve_gate.weight'] = self.gates[i].T
                sd[f'value_embeds.{i}.weight'] = self.ve[i].float()
        return {k: v.detach().float().contiguous().to(device) for k, v in sd.items()}


def rope_pair(lag, rho, generator):
    """A query and key whose rotary logit is a bump at `lag` whatever the position: energy rho^j on frequency pair j
    (j = 0 the fastest), random phases, and the key is the query moved `lag` positions ahead. RMS 1 each."""
    half = HD // 2
    phase = 2 * math.pi * torch.rand(half, generator=generator)
    amplitude = rho ** (torch.arange(half, dtype=F32) / 2)
    q = torch.cat([amplitude * phase.cos(), amplitude * phase.sin()])
    q = q * math.sqrt(HD) / q.norm()
    angle = lag / 10000 ** (torch.arange(0, HD, 2, dtype=F32) / HD)
    q1, q2 = q[:half], q[half:]
    return q, torch.cat([q1 * angle.cos() + q2 * angle.sin(), q2 * angle.cos() - q1 * angle.sin()])


def relu2_moments(mu, sd):
    """E relu(u)^2 and E relu(u)^4 for u ~ N(mu, sd^2)."""
    z = mu / sd
    cdf, pdf = 0.5 * (1 + torch.erf(z / math.sqrt(2))), torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    m2 = (mu ** 2 + sd ** 2) * cdf + mu * sd * pdf
    m4 = (mu ** 4 + 6 * mu ** 2 * sd ** 2 + 3 * sd ** 4) * cdf + (mu ** 3 + 5 * mu * sd ** 2) * sd * pdf
    return m2, m4


@torch.no_grad()
def shared_means(model):
    """Each block's normalized input, averaged over tokens, in closed form from the weights: the part of the stream
    every token shares, written by the relu^2 means of the MLPs below. Mean field: each stream is a shared mean plus
    an isotropic token part; near-uniform attention writes c_v c_proj of the mean; each MLP unit's input is Gaussian."""
    mean, energy, x0, means = torch.zeros(D, device=model.device), float(D), 1.0, []  # E x, E|x|^2, x0's share
    for i in range(L):
        r, lam = model.lam_r[i].item(), model.lam_0[i].item()
        energy = r * r * energy + lam * lam * D + 2 * r * lam * x0 * D
        mean, x0 = r * mean, r * x0 + lam
        means.append(mean / math.sqrt(energy / D))
        attended = (means[-1] @ model.w(i, 'attn.c_v')) @ model.w(i, 'attn.c_proj')
        energy, mean = energy + (attended @ attended).item() + 2 * (mean @ attended).item(), mean + attended
        shared = mean / math.sqrt(energy / D)
        fc, proj = model.w(i, 'mlp.c_fc'), model.w(i, 'mlp.c_proj')
        spread = fc.norm(dim=0) * max(1 - (shared @ shared).item() / D, 1e-3) ** 0.5
        m2, m4 = relu2_moments(shared @ fc, spread)
        written = m2 @ proj
        energy += (written @ written).item() + (proj.square().sum(1) @ (m4 - m2 * m2)).item()
        energy, mean = energy + 2 * (mean @ written).item(), mean + written
    return means


@torch.no_grad()
def structured_start(model):
    """Every head of block 1 starts as a local RoPE head peaking at lag 1: its query and key columns get a rank-one
    term on the block input's token-shared direction (shared_means), so the raw query is the same vector for every
    token, rotated to attend one position back, and keep a quarter of their random map. A fixed formula of the seed
    and the architecture: no data, statistic or forward pass."""
    generator = torch.Generator(device='cpu').manual_seed(ID_SEED + 31)
    mean = shared_means(model)[START_BLOCK]
    for h in range(H):
        cols = slice(h * HD, (h + 1) * HD)
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAG, START_RHO, generator)):
            w = model.w(START_BLOCK, kind)
            w[:, cols] = START_CONTENT * w[:, cols] + torch.outer(mean, v.to(mean.device)) / (mean @ mean)


def rotary(t, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, HD, 2, dtype=F32, device=device) / HD)
    freqs = torch.outer(torch.arange(t, dtype=F32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()  # [t, HD / 2]


def rope_norm(raw, cos, sin):
    """RoPE then QK id_norm in fp32 -> bf16 [b, h, t, d]. raw [b, t, h, d]; cos, sin [t, d/2]."""
    x = raw.float()
    c, s_ = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * c + x2 * s_, x1 * -s_ + x2 * c], -1)
    return (rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)).to(BF16).transpose(1, 2).contiguous()


def forward_impl(model, idx, targets, cos, sin, lo: int, hi: int):
    """The clean pass on this GPU's rows idx [b, t]: (cache, mean loss of its tokens lo..hi of the flattened rows)."""
    b, t = idx.shape
    cache = {}
    x = id_norm(model.wte[idx])
    x0, pending = x, None
    cache['x0'] = x0
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
        o, aux = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.masks[(window(i), t)],
                                return_aux=AuxRequest(lse=True))  # (return_lse's warning breaks the compiled graph)
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
    final = id_norm(x + pending.float()).to(BF16).reshape(b * t, D)[lo:hi]
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
         + ID_SEED) & 0xFFFFFFFF
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
def cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h):
    fast = [x.float() for x in prods]  # the five bf16 products
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
    """The cycle directions' logits c and their squares and cubes in bf16 (the fast products' operands)."""
    return c.to(BF16), (c * c).to(BF16), (c * c * c).to(BF16)


def top_credit(model, x, y):
    """Each token's readout credit at the top stream x (m, d): a whole sign-flipped Hadamard cycle, every moved
    stream's partition function from a degree-3 expansion around the clean logits (six matrix products)."""
    h = model.H * step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head  # (P, V): row j is direction j's logits
    base = x @ model.lm_head
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = cycle_weights(x, base, h)
    exact = fg1 @ c.T  # the first-order product in TF32 (#29: fp32, HIGHEST)
    cb, c2, c3 = cycle_powers(c)
    prods = [fast[0] @ cb.T, fast[1] @ cb.T, fast[2] @ c2.T, fast[3] @ c2.T, fast[4] @ c3.T]
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def rope32(x, cos, sin):
    """RoPE then QK id_norm in fp32 over [b, t, d] (one head)."""
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


REPEAT, REPEAT_K = HD // 16, HD // 32  # moment-kernel programs per head row: 16 query / 32 key directions each
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
def head_prep(got, lse, cos, sin, b: int, t: int):
    """The running blocks' exact-kernel inputs for this GPU's head over the batch, from got [world, k, m, PACK] (as
    received) and lse [world, k, m] (each block's b rows of t tokens, by GPU), stacked over the blocks [k b, t, .]: the
    processed queries and keys (bf16 for the logits, f32 for the rotated columns), values and hub (bf16), the raw
    rows' squared norms and move sizes, hub . output per token (f32)."""
    k = got.shape[1]
    whole, lse = got.transpose(0, 1).reshape(k * b, t, PACK), lse.transpose(0, 1).reshape(k * b, t)
    q_raw, k_raw, values, out, hub = (whole[..., j * HD:(j + 1) * HD].float() for j in range(5))
    q1, k1 = rope32(q_raw, cos, sin), rope32(k_raw, cos, sin)
    sq_q, sq_k = (q_raw * q_raw).sum(-1), (k_raw * k_raw).sum(-1)
    steps = lambda sq: KEY_SIGMA * torch.clamp(torch.sqrt(sq / HD), min=1e-6)
    return (q1.to(BF16), k1.to(BF16), values.to(BF16), hub.to(BF16), q1, k1, lse.contiguous(), (hub * out).sum(-1),
            sq_q, sq_k, steps(sq_q), steps(sq_k))


def id_head_scores(got, rope32, windows, back, b, t):
    """The running blocks' query, key and value credits for this GPU's head over the batch (b rows of t tokens each),
    from got [world, k, m, PACK] as received, into back [world, k, 3, m, hd] (the tokens grouped by the GPU that holds
    them); windows [k] int32 the blocks' attention windows. Every block in one launch per kernel."""
    lse_bits = got[..., 5 * HD:].view(F32)[..., 0]  # the lse's fp32 bits back (a view)
    q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep(got, lse_bits, *rope32, b, t)
    moment_query_scores(q1, k1, values, hub, got, sq_q, k1f, *rope32, lse, hub_out, steps_q, back[:, :, 0], windows,
                        REPEAT, block_s=64, block_t=32, num_warps=4, num_stages=3)
    moment_key_scores(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k, back[:, :, 1],
                      back[:, :, 2], windows, REPEAT_K, block_t=64, block_s=32, num_warps=4, num_stages=3)


@torch.compile(dynamic=False)
def credit_stacks(got, run: tuple, ve_blocks: tuple):
    """From got [source GPU (its head), k, 3, m, hd] bf16: every block's query, key and value credits by token
    [3, L, m, d] (zero in the skipped blocks), the weight-gradient products' operand, and the value-embedding blocks'
    value credits [m, h, hd] in f32."""
    world, _, _, m, hd = got.shape
    by_token = got.permute(1, 2, 3, 0, 4).reshape(len(run), 3, m, world * hd)
    zero = torch.zeros_like(by_token[0])
    stacks = torch.stack([by_token[run.index(i)] if i in run else zero for i in range(L)], 1)
    return stacks, [(by_token[run.index(i), 2] if i in run else zero[2]).view(m, H, HD).float()
                    for i in ve_blocks]


def f32_stack(xs, lo: int, hi: int):
    """The blocks' input streams on this GPU's tokens, stacked [L, m, d] in f32."""
    return torch.stack([x.reshape(-1, x.shape[-1])[lo:hi] for x in xs]).float()


def bf16_stacks(n1, o, n2, act, lo: int, hi: int):
    """The weight-gradient products' inputs on this GPU's tokens, stacked [L, m, ...] in bf16."""
    st = lambda xs: torch.stack([x.reshape(-1, x.shape[-1])[lo:hi] for x in xs]).to(BF16)
    return st(n1), st(o), st(n2), st(act)


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    """Each block output's credit: the top credit times the residual scalars above it, [L, m, d] (f32, and bf16 for the
    products)."""
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])  # above[i] = lam_r[i] ... lam_r[last]
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


@torch.compile(dynamic=False)
def hidden_slope_(u, read, lo: int, hi: int):
    """One block's MLP hidden credit in place of read (its credit read through the output rows, credit @ W_out^T,
    [m, 4d] bf16): each unit's central difference through relu^2 at its pre-activation (u [b, t, 4d])."""
    u = u.reshape(-1, u.shape[-1])[lo:hi].float()
    step = FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1, keepdim=True)), min=1e-6)
    relu2 = lambda z: F.relu(z).square()
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    read.copy_((slope * read.float()).to(BF16))


def hidden_credits(us, credb, w_out_t, lo, hi):
    """Every block's MLP hidden credit [L, m, 4d] (bf16, the product's operand; w_out_t [L, d, 4d] bf16 is W_out^T):
    one product for every block, then each block's slopes in place (one pass over its own pre-activations)."""
    hidden = torch.bmm(credb, w_out_t)
    for u, read in zip(us, hidden):
        hidden_slope_(u, read, lo, hi)
    return hidden


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    """The mixing scalars' gradients (resid, x0) per block, and the credit of x0."""
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


@torch.compile(dynamic=False)
def value_embed_credits(z, ve, vcredit, n1):
    """A value-embedding block's gate gradient [GATE_CHANNELS, h] (the gate's central difference) and its embedding
    rows' credits by token [m, d]: z [m, h] its gate's logits, ve [m, d] its embedding rows, vcredit [m, h, hd] the
    values' credit, n1 [m, d] the block's normed input."""
    z, ve = z.float(), ve.float().view(-1, H, HD)
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    gate = n1[:, :GATE_CHANNELS].float().T @ (slope * (vcredit * ve).sum(-1))
    return gate, (vcredit * 2 * torch.sigmoid(z)[..., None]).view(-1, D)


def identity_grad(model, idx, targets, lo, hi, rows_total, skip=()):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens lo..hi, one head's queries, keys and values over the batch. The ANVIL matrices' gradients stay this
    GPU's partial sums, reduce-scattered; the rest are all-reduced. Returns the clean loss of its tokens, the collectives
    in flight and the token count."""
    world = model.world
    t = idx.shape[1]
    n, m = rows_total * t, hi - lo
    rope32 = model.rope[t]
    cache, loss = forward(model, idx, targets, *rope32, lo, hi)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])[lo:hi]
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1)[lo:hi], idx.reshape(-1)[lo:hi]
    coords = coordinate_change(cache['lm_head'], y)
    credit_top = top_credit(model, stream(f'h{L - 1}.stream'), y)
    gsq, gwide = model.grad['square'], model.grad['wide']
    # every block at once: its inputs on this GPU's tokens stacked [L, m, ...] (f32), its output credit
    prevs = f32_stack([cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)], lo, hi)
    n1s, os_, n2s, acts = bf16_stacks([cache[f'h{i}.in.attn'] for i in range(L)],
                                      [cache[f'h{i}.in.attn.c_proj'] for i in range(L)],
                                      [cache[f'h{i}.in.mlp.c_fc'] for i in range(L)],
                                      [cache[f'h{i}.in.mlp.c_proj'] for i in range(L)], lo, hi)
    cred, credb = block_credits(credit_top, model.lam_r)  # [L, m, d]: the top credit down the residual path
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))  # attention-output targets (bf16)
    run = [i for i in range(L) if i not in skip]  # the blocks whose attention experiments run (the late skip)
    k = len(run)
    if k:
        send = pack_heads([mine(cache[f'h{i}.q']) for i in run], [mine(cache[f'h{i}.k']) for i in run],
                          [mine(cache[f'h{i}.values']) for i in run],
                          [mine(cache[f'h{i}.in.attn.c_proj']) for i in run], [targets_o[i] for i in run],
                          [cache[f'h{i}.lse'].transpose(1, 2).reshape(-1, H)[lo:hi].contiguous().view(BF16)
                           for i in run])  # [b, h, t] -> by token, every row of this GPU's
        got = torch.empty_like(send)
        # [source GPU (its tokens), k, m, PACK]: this GPU's head over the batch
        heads_in = dist.all_to_all_single(got, send, async_op=True)
    # while the heads' inputs travel: the weight-gradient products on bf16 operands (fp32 sums)
    hidden = hidden_credits([cache[f'h{i}.mlp.c_fc'] for i in range(L)], credb, model.wide_bf[L:], lo, hi)
    torch.bmm(credb.transpose(1, 2), acts, out=gwide[L:])  # mlp.c_proj, oriented (out, in)
    torch.bmm(n2s.transpose(1, 2), hidden, out=gwide[:L])  # mlp.c_fc
    torch.bmm(os_.transpose(1, 2), credb, out=gsq[3 * L:])  # attn.c_proj
    # the MLP matrices' gradients are complete: reduce them while the head credits run
    works = dict(wide=dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True))
    if not k:
        got2 = torch.zeros(world, 0, 3, m, HD, device=idx.device, dtype=BF16)
    else:  # one head per GPU
        heads_in.wait()
        back = torch.empty(world, k, 3, m, HD, device=idx.device, dtype=BF16)  # [GPU that holds the tokens, ...]
        id_head_scores(got, rope32, model.windows[tuple(run)], back, rows_total, t)
        got2 = torch.empty_like(back)
        heads_out = dist.all_to_all_single(got2, back, async_op=True)  # [source GPU (its head), k, 3, m, hd]
    # while the credits travel back: the readout's, the mixing scalars' and the embedding's gradients
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    if k:
        heads_out.wait()
    # the skipped blocks' queries, keys and values get no credit
    by_token, vcredits = credit_stacks(got2, tuple(run), VE_BLOCKS)
    for j in range(3):  # queries, keys, values
        torch.bmm(n1s.transpose(1, 2), by_token[j], out=gsq[j * L:(j + 1) * L])
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        gate, rows = value_embed_credits(mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(gate)
        model.grad[f've{i}'].zero_().index_add_(0, tokens, rows)
    works['flat'] = dist.all_reduce(model.flat_grad, async_op=True)
    return loss, works, n


# -------------------------------------------------------------------------------------------------------- optimizers

def anvil_betas(t):
    total, warm, cool, engage = (round(k * STEPS / ANVIL_RUN[0]) for k in ANVIL_RUN)
    lo, hi = 0.85, 0.93
    beta = torch.where(t < warm, lo + (hi - lo) * t / max(warm, 1), torch.full_like(t, hi))
    beta = torch.where(t > total - cool, hi - (hi - lo) * (t - (total - cool)) / max(cool, 1), beta)
    engaged = t >= engage
    return beta, torch.where(engaged, 0.85, beta), torch.where(engaged, 0.4385, 1.0)


@torch.compile(dynamic=False)
def anvil_math(w, g, n: int, fast, slow, energy, step_t, scale_t, rates):
    """ANVIL II on a stack of wide-oriented matrices [k, rows <= cols] from their summed bf16 gradients g over n
    tokens: w, fast, slow and energy are updated in place (in the compiled step's kernels); returns the new weights'
    bf16 cast (the forward's). The schedule (rate, momenta) from the update index inside the step."""
    beta, fast_beta, weight = anvil_betas(step_t)
    lr = scale_t * ANVIL_LR
    g = g.float() / n
    fast_new = fast + (1 - fast_beta) * (g - fast)
    slow_new = slow + (1 - 0.98) * (g - slow)
    x = g + beta * (weight * fast_new + (1 - weight) * slow_new - g)
    x = (x / (POLAR_MARGIN * torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-6)).to(BF16)
    for a, b_, c in ANVIL_MAPS:
        sq = x @ x.transpose(1, 2)
        x = a * x + (b_ * sq + c * (sq @ sq)) @ x
    update = x.float()
    energy_new = energy + (1 - 0.9) * (update.square().mean(1, keepdim=True) - energy)
    gained = update * torch.rsqrt(torch.clamp(energy_new, min=1e-10))
    update = gained * (torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
                       / torch.clamp(torch.linalg.vector_norm(gained, dim=(1, 2), keepdim=True), min=1e-10))
    decay = ((slow_new * w) >= 0) * w * (ANVIL_WD * lr)
    new = w - lr * rates[:, None, None] * (update + decay)
    fast.copy_(fast_new)
    slow.copy_(slow_new)
    energy.copy_(energy_new)
    w.copy_(new)
    return new.to(BF16)


def id_anvil_update(model, works, n):
    """Steps this GPU's share of each stack once its gradients arrive; returns the all-gathers of the new bf16 weights
    in flight."""
    gathers = []
    for name, w, w_bf in (('wide', model.wide, model.wide_bf), ('square', model.square, model.square_bf)):
        works[name].wait()
        sl, st = model.shard[name], model.anvil[name]
        new = anvil_math(w[sl], model.owned[name], n, st['fast'][sl], st['slow'][sl], st['energy'][sl], model.step_t,
                         model.scale_t, model.rates[name][sl])
        gathers.append((dist.all_gather_into_tensor(w_bf, new, async_op=True), new))
    return gathers


@torch.compile(dynamic=False)
def adam_math(ps, gs, n: int, ms, vs, lrs: tuple, step_t, scale_t):
    """One Adam step on every Adam parameter from its summed gradient over n tokens, the parameters and their moments
    written in place (a bf16 parameter's gradient rounded to bf16 first, as the JAX trainer does); the bias corrections
    and the schedule's factor inside the step."""
    t = step_t + 1
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    for p, g, m, v, lr in zip(ps, gs, ms, vs, lrs):
        g = g / n
        if p.dtype == BF16:
            g = g.to(BF16).float()
        m_new = BETA1 * m + (1 - BETA1) * g
        v_new = BETA2 * v + (1 - BETA2) * g.square()
        p.copy_((p.float() - lr * scale_t * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)


def adam_update(model, works, n):
    works['flat'].wait()
    ps, gs, lrs = zip(*model.adam)
    adam_math(list(ps), list(gs), n, model.m, model.v, lrs, model.step_t, model.scale_t)


def average_step(model, length, skip=()):
    """An update, then the tail average follows the new weights."""
    train_step(model, length, skip)
    rate = 1 / max(1, round(EMA_HORIZON * STEPS))
    torch._foreach_lerp_(model.ema, [w.float() for w in model.averaged()], rate)


def train_step(model, length, skip=()):
    """One update, everything on the device: this step's batch from model.rows, this GPU's rows of it."""
    world, rank = model.world, model.rank
    model.scale_t.copy_(torch.clamp((STEPS - model.step_t) / COOLDOWN, max=1.0))  # warmup-stable-decay
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]  # [BATCH, 2049]
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
    loss, works, n = identity_grad(model, idx, tgt, lo, hi, rows_total, skip)
    if skip:  # a skipped block's queries, keys and values get no credit, so no step at all (moves undone below)
        rows_qkv = model.skip_rows[skip]
        frozen, frozen_bf = model.square.index_select(0, rows_qkv), model.square_bf.index_select(0, rows_qkv)
    gathers = id_anvil_update(model, works, n)  # from the weights before this step's Adam (the gradients are this step's)
    adam_update(model, works, n)  # while the ANVIL matrices gather
    for work, _ in gathers:
        work.wait()
    if skip:
        model.square.index_copy_(0, rows_qkv, frozen)
        model.square_bf.index_copy_(0, rows_qkv, frozen_bf)
    model.refresh_bf16()
    model.losses.index_copy_(0, model.step_t.long().view(1), loss.view(1))
    model.step_t.add_(1)


# ---------------------------------------------------------------------------------------------------------- training

MODEL = None


class IdentityPhase:
    """The identity-path phase of a run: built, compiled and captured (CUDA graphs) on random tokens when constructed,
    then reset to its fresh state on the real batches; `run()` is the timed part. `rows` are the training rows (numpy
    [n, 2049]); the phase trains on the first `updates` x BATCH rows of the order permutation(n) under rng (ID_SEED + 1, 0)
    (`used_rows()`), so a later phase can leave them out."""

    def __init__(self, rows, updates, device, rank, world, seed=ID_SEED, start=True, ema=True, skip=True, max_run=None):
        global STEPS, WARMUP_UNTIL, COOLDOWN, ID_SEED, START, EMA, SKIP, MODEL
        STEPS, WARMUP_UNTIL, COOLDOWN, ID_SEED = updates, updates * 4 // 5, updates * 2 // 5, seed
        START, EMA, SKIP = start, ema, skip
        if world != H:
            raise SystemExit(f'one head per GPU: run on {H} GPUs, not {world}')
        torch.backends.cuda.matmul.allow_tf32 = True
        self.rank, self.updates = rank, updates
        run_limit = max_run or updates
        self.order = np.random.default_rng((ID_SEED + 1, 0)).permutation(len(rows))
        batches = np.stack([rows[np.sort(self.order[i:i + BATCH])]
                            for i in range(0, len(rows) - BATCH + 1, BATCH)][:STEPS])
        model = self.model = MODEL = Model(device, rank, world)
        snapshot = [x.clone() for x in model.tensors()]
        graphs = self.graphs = {}
        first_ema = round(EMA_FROM * STEPS)
        first_skip = round(SKIP_FROM * STEPS)
        lengths = (WARMUP_LEN,) if run_limit <= WARMUP_UNTIL else (WARMUP_LEN, SEQ)
        steps_of = (train_step, average_step) if (EMA and run_limit > first_ema + 1) else (train_step,)  # the tail average's updates: own graphs
        skips = ((), SKIP_BLOCKS) if (SKIP and run_limit > first_skip) else ((),)  # and the late skip's
        for length in lengths:  # warmup on random tokens (model.rows is random until training), capture
            for update in steps_of:
                for skip_blocks in skips:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            update(model, length, skip_blocks)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        update(model, length, skip_blocks)
                    graphs[length, update is average_step, bool(skip_blocks)] = g
        with torch.no_grad():  # training starts from the untouched state, on the real batches
            for x, s in zip(model.tensors(), snapshot):
                x.copy_(s)
            model.cast_all()
            model.rows.copy_(torch.from_numpy(batches.astype(np.int64)))
            model.step_t.zero_()
            model.losses.zero_()
        torch.cuda.synchronize()
        dist.barrier()

    def used_rows(self):
        """Indices of the training rows this phase consumes."""
        return self.order[:self.updates * BATCH]

    def config(self):
        return dict(updates=STEPS, batch=BATCH, seed=ID_SEED, init='prev8' if START else 'none',
                    skip=dict(blocks=SKIP_BLOCKS, start=round(SKIP_FROM * STEPS)) if SKIP else None,
                    ema=dict(start=round(EMA_FROM * STEPS), horizon=round(EMA_HORIZON * STEPS), mix=EMA_MIX)
                    if EMA else None, seq_warmup=f'{WARMUP_LEN}:{WARMUP_UNTIL}', cooldown=COOLDOWN, lr=LR,
                    betas=[BETA1, BETA2], lr_wte=LR_WTE, lr_value_embeds=LR_VE, lr_lambdas=LR_LAMBDAS,
                    anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD)

    def run(self, total=None):
        """The timed updates; ends with the tail average mixed in and the whole fp32 weights on every GPU."""
        model, graphs, total = self.model, self.graphs, total or STEPS
        first = round(EMA_FROM * STEPS)  # the update at which the tail average starts
        for step in range(total):
            length = WARMUP_LEN if step < WARMUP_UNTIL else SEQ
            graphs[length, EMA and step > first, SKIP and step >= round(SKIP_FROM * STEPS)].replay()
            if EMA and step == first:  # the average starts at the weights after this update
                torch._foreach_copy_(model.ema, [w.float() for w in model.averaged()])
        if EMA and total > first:  # the final weights: w + EMA_MIX (EMA - w), inside the timer
            with torch.no_grad():
                for w, e in zip(model.averaged(), model.ema):
                    w.copy_(torch.lerp(w.float(), e, EMA_MIX).to(w.dtype))
        model.gather_fp32()

    def close(self):
        """Release the graphs and the phase's tensors."""
        global MODEL
        self.graphs.clear()
        self.model = MODEL = None
        torch.cuda.empty_cache()



def train_stages(args, settings, device, rank, world):
    """Boost -> own-query Cut; each row is consumed exactly once, including across resume."""
    if world not in (1, 2, 4, 8):
        raise ValueError('Stage training supports 1, 2, 4 or 8 GPUs')
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    identity = None
    if args.identity_updates and not args.resume:  # the identity-path phase, built and captured before the timer
        identity = IdentityPhase(training.numpy(), args.identity_schedule or args.identity_updates, device, rank, world,
                                 seed=SEED, max_run=args.identity_updates)
    if args.identity_updates:  # its rows feed no Cut update: one epoch over both phases
        import numpy as np
        used = np.random.default_rng((SEED + 1, 0)).permutation(len(training))[:args.identity_updates * 8]
        keep = torch.ones(len(training), dtype=torch.bool)
        keep[torch.from_numpy(used)] = False
        training = training[keep]
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout_rows, training = training[-HOLDOUT:], training[:-HOLDOUT]  # these feed no update, only the end merge
    means = None  # the exponential means of the end merge, one per MERGE_DECAYS entry
    global MERGE_FIRST_KEYS, MERGE_GRAPH
    MERGE_FIRST_KEYS = settings.merge_keys
    MERGE_GRAPH = settings.merge_graph
    schedule = stage_schedule(len(training), settings)
    steps = min(args.steps or len(schedule), len(schedule))
    stop = min(steps, args.stop_after_step or steps)
    split = max(1, world // settings.boost_batch)
    groups = ([torch.distributed.new_group(list(range(i, i+split))) for i in range(0, world, split)]
              if split > 1 else [None])
    group = groups[rank // split] if split > 1 else None
    cut_split = max(1, world // settings.cut_batch)  # GPUs per sequence in Cut
    cut_groups = groups if cut_split == split else (
        [torch.distributed.new_group(list(range(i, i+cut_split))) for i in range(0, world, cut_split)]
        if cut_split > 1 else [None])
    cut_group = cut_groups[rank // cut_split] if cut_split > 1 else None
    recipe_state = asdict(settings)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, seed=SEED, recipe=recipe_state, steps=steps,
            epoch_updates=len(schedule), train_rows=len(training),
            upstream='qlabs-eng/noise@98dbf6339c7768b2a6c2b9022aa25c80eb22a259',
            additions=['linear_path_scalars', 'input_key_probes', 'input_gate_probes',
                       'independent_future_hidden_query_embedding_probes', 'cut_batch4_split2', 'recipe_11',
                       'end_merge', 'multi_layer_x0_credit', 'readout_sigma_001'],
            resume=str(args.resume.resolve()) if args.resume else None, torch_version=torch.__version__,
            identity=identity.config() if identity else None,
        ), indent=2)+'\n')

    def estimator_for(model, phase):
        if phase == 'boost':
            return Boost(model, settings, SEED, rank, world, split, group)
        return Cut(model, settings, SEED, rank, world, cut_split, cut_group)

    def rank_slice(batch, phase):
        group_size = split if phase == 'boost' else cut_split
        per = len(batch) // (world // group_size)
        first = (rank // group_size) * per
        return batch[first:first+per].to(device)

    # Compile every active phase on random tokens; recreate all model/optimizer/RNG state afterward.
    phases = {phase: horizon for phase, _, _, _, horizon in schedule}
    noise = torch.randint(0, settings.model.vocab_size,
                          (max(settings.boost_batch, settings.cut_batch), settings.model.sequence_len+1),
                          generator=torch.Generator().manual_seed(SEED))
    for phase, horizon in phases.items():
        model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
        optimizer, estimator = make_optimizer(model, settings, horizon), estimator_for(model, phase)
        size = settings.boost_batch if phase == 'boost' else settings.cut_batch
        batch = rank_slice(noise[:size], phase)
        estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        estimator.close()
        del model, optimizer, estimator
    torch.cuda.empty_cache()

    completed, previous_seconds = 0, 0.
    phase = schedule[0][0]
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        completed = checkpoint['step']
        if not 0 < completed < stop:
            raise ValueError('Resume checkpoint must precede the requested stop and follow at least one update')
        saved_phase, phase_step, _, seqs, _ = schedule[completed-1]
        expected = dict(phase=saved_phase, phase_step=phase_step, seqs=seqs, epoch_updates=len(schedule))
        if checkpoint.get('progress') != expected:
            raise ValueError('Resume data position or phase schedule differs')
        phase = saved_phase
        del checkpoint
    model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
    optimizer, estimator = make_optimizer(model, settings, phases[phase]), estimator_for(model, phase)
    if args.resume:
        completed, previous_seconds = restore_training_state(args.resume, model, optimizer, estimator, recipe=recipe_state)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    if MERGE_GRAPH and HOLDOUT:  # compile and capture the merge's held-out forward on random tokens, this model's weights
        for _ in range(3):
            holdout_loss(model, torch.randint(0, settings.model.vocab_size, (HOLDOUT, settings.model.sequence_len + 1),
                                              generator=torch.Generator().manual_seed(SEED + 2)), rank, world)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, paused, elapsed = time.perf_counter(), 0., previous_seconds
    try:
        if identity is not None:  # timed: the identity-path updates, then their weights into Cut's model
            identity.run(args.identity_updates)  # of a schedule of --identity-schedule updates (default: the same)
            with torch.no_grad():
                weights = model.state_dict()
                for name, value in identity.model.state_dict(device).items():
                    weights[name].copy_(value)
            n = args.identity_updates
            identity_loss = average_ranks(identity.model.losses[n - 1:n].clone()).item()
            identity.close()
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.perf_counter()-start
            record = dict(step=0, phase='identity', phase_step=args.identity_updates,
                          seqs=args.identity_updates * 8, tokens=args.identity_updates * 8 * settings.model.sequence_len,
                          train_loss=identity_loss, train_seconds=round(elapsed, 2))
            if rank == 0:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+'\n'); log.flush()
        for step in range(completed+1, stop+1):
            current, phase_step, begin, end, horizon = schedule[step-1]
            if current != phase:
                estimator.close()
                next_optimizer = make_optimizer(model, settings, horizon)
                handoff_optimizer(optimizer, next_optimizer, phase_step)
                optimizer, estimator, phase = next_optimizer, estimator_for(model, current), current
            batch = rank_slice(training[begin:end], phase)
            set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.perf_counter()-start-paused
            record = dict(step=step, phase=phase, phase_step=phase_step, seqs=end,
                          tokens=end*settings.model.sequence_len, train_loss=loss, train_seconds=round(elapsed, 2),
                          peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3))
            if args.eval_every and step % args.eval_every == 0 and step < steps:
                pause = time.perf_counter()
                record['val_loss'] = evaluate(model.state_dict(), args.data_dir)
                paused += time.perf_counter()-pause
            if rank == 0:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+'\n'); log.flush()
            if (step % MERGE_EVERY == 0 or step == stop) and begin / len(training) >= MERGE_FROM:
                with torch.no_grad():  # a resumed run restarts the means at its resumed weights
                    if means is None:
                        means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                    else:
                        for a, mean in zip(MERGE_DECAYS, means):
                            for m, q in zip(mean, model.parameters()):
                                m.lerp_(q.detach().float(), a)
            if (args.checkpoint_every and (step % args.checkpoint_every == 0 or step == stop)) or step == stop < steps:
                pause = time.perf_counter()
                save_training_state(args.output/'checkpoint.pt', model, optimizer, estimator, step=step,
                                    train_seconds=elapsed, recipe=recipe_state,
                                    progress=dict(phase=phase, phase_step=phase_step, seqs=end, epoch_updates=len(schedule)))
                paused += time.perf_counter()-pause
        if stop < steps:
            if rank == 0:
                (args.output/'segment.json').write_text(json.dumps(dict(step=stop, total_steps=steps, seqs=end,
                    train_seconds=elapsed, checkpoint=str((args.output/'checkpoint.pt').resolve())), indent=2)+'\n')
            return
        extra = {}
        if means:  # the end merge, timed with the training (the plain weights' validation is not)
            plain = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            pause = time.perf_counter()
            extra['val_loss_plain'] = evaluate(plain, args.data_dir)
            paused += time.perf_counter()-pause
            merge_start = time.perf_counter()
            extra.update(end_merge(model, means, holdout_rows, rank, world))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter()-merge_start, 2)
            elapsed = previous_seconds + time.perf_counter()-start-paused
            if rank == 0:
                torch.save(plain, args.output/'final_plain.pt')
            means = None
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        result = dict(val_loss=evaluate(state, args.data_dir), train_seconds=round(elapsed, 2), steps=stop,
                      seqs=end, tokens=end*settings.model.sequence_len, estimator=settings.estimator,
                      peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3), **extra)
        if rank == 0:
            torch.save(state, args.output/'final.pt')
            (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    finally:
        estimator.close()
        if log is not None:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--steps', type=int, help='Diagnostic prefix; preserves the full-epoch LR schedule.')
    # Execution chunks only: the default command reproduces the logged algorithm and budgets.
    parser.add_argument('--chain-tokens', type=int, default=Settings.chain_tokens)
    parser.add_argument('--key-chunk', type=int, default=32)
    parser.add_argument('--key-path-tokens', type=int, default=64)
    parser.add_argument('--future-tokens', type=int, default=128)
    parser.add_argument('--future-chunk', type=int, default=512)
    parser.add_argument('--identity-updates', type=int, default=IDENTITY_UPDATES,
                        help='Identity-path updates of 8 rows first (IdentityPhase); Cut trains on the other '
                             'rows. 0: Cut (and Boost) over the whole epoch.')
    parser.add_argument('--identity-schedule', type=int, default=IDENTITY_SCHEDULE,
                        help='Plan the identity phase for this many updates and hand off after --identity-updates.')
    args = parser.parse_args()
    for key in ('steps', 'stop_after_step'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(key + ' must be positive')
    if min(args.checkpoint_every, args.eval_every) < 0:
        parser.error('Logging intervals must be nonnegative')
    settings = Settings(chain_tokens=args.chain_tokens, key_chunk=args.key_chunk,
                        key_path_tokens=args.key_path_tokens, future_tokens=args.future_tokens,
                        future_chunk=args.future_chunk)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    try:
        train_stages(args, settings, device, rank, world)
    except BaseException:
        if args.identity_updates:  # the hard exit below would swallow the traceback
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if args.identity_updates:  # NCCL teardown after collectives captured in CUDA graphs can hang: leave without it
            sys.stdout.flush()
            os._exit(0)
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
