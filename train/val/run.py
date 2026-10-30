"""Track 1 (30M): 3.2998 in 43 min on 8xH100 (seed 0, merge included; seeds 0-4 average 3.3014). Record #3's
forward readout of cross-token credit, carried to blocks 0-6 and decoded through a transit solve, with two-hop
twin rows, shared logit pairs, early head transplants, momentum distillation, a planted lag-1 head and per-head
polar steps for c_q / c_k.

Every learning signal is a loss change or a measured-hub score of a forward experiment; no autograd and no backward
pass anywhere in training. A decoded credit feeds only linear consumers or, through the outer product, its own
source site; it never becomes another experiment's scoring target.

The method, in file order (each section below is one module with its own docstring):
  recipe        record #2 (Boost -> Cut with #31's estimator: own queries, exact values, input-K / gate probes,
                future-token probes, ANVIL II + Adam, the end merge), with Settings for everything below.
  onesided      future probes measured one-sided (+step only): the same directions, half the experiments.
  fast_kernels, fast_chain, fast_moments, fast_chain2
                the same experiments, faster: wider scorer tiles, the chain's roots advanced together, second-order
                moment tables for the receivers s > t (s = t exact), with running float64 checks and a fallback.
  readout, readout_kv, readout_hi, fast_readout, shared_push, fast_e3
                the true-path readout: every chain move is pushed up the token's real path and, at every later block,
                its key / value / gate change is scored as the exact single-column softmax change of each later
                receiver against that block's measured hub. q / c_fc source experiments, c_k / ve_gate receiver
                experiments, blocks 0-6; the chain shares its pushes with the readout roots.
  s2_*, fast_s2 the chain head to second order (float64 running check), moment tables as GEMMs, empty receiver slots
                skipped, the receiver plan in one kernel.
  head_edit     early head transplants (updates 50-450, every 25): dormant heads get a strong head's weights
                (clone / split / transfer), kept only if held-back rows of the update agree.
  mom_distill   momentum distillation: from 10% of the epoch every measured loss is the cross-entropy against
                0.7 the label + 0.3 an EMA teacher's distribution (beta 0.99).
  twin_rows, combo_twin
                two-hop credit: each experiment's row gets a twin in the next token's slot that carries the exact
                attention change it causes there forward.
  speed_stack   exact reuse (c_fc cycles sized to the active units, shared tables) and lossless storage.
  transit_base, transit_in0
                transit decode: the mid roots take their credit from one stream-only complete cycle at block 0's mid
                stream, solved per token over the displacements the push records.
  transit_gap, transit_gap2
                Boost's embedding credit adds block 0's receivers; block-input credits in_2..in_7 by their own solves.
  shared_pairs  c_q and c_k of blocks 0-4 from one experiment per measured (head, key, receiver) logit pair, plus one
                q-only tail row per receiver and head.
  plant_init, hq_plant
                block 4's head 0 starts as a content-free lag-1 (previous-token) head; init only.
  polarfar, plant_polar
                ANVIL's polar step for c_q / c_k taken per head; the mixing scalars' Adam rate x60.

Recipe: Boost for 6.68% of the epoch, then Cut at batch 4 (each sequence split over two GPUs); lr 0.0021, ANVIL lr
0.026 and weight decay 0.75, a linear decay over the whole run to 30% of each rate, the LM head initialized at std
0.02; the end merge: the last 32 training sequences feed no update and score it; from 60% of the epoch, every 5
updates, exponential means of the weights are kept, and after the last update each part moves toward one mean
chosen by the held-out loss (inside the timed region). The settings are RUN_SETTINGS in the recipe section.

This file holds its modules as sections (`# ==== module <name>`), loaded by name by the bootstrap below; each runs
from this file at its own line numbers. The run installs them in order (onesided ... plant_polar), then the kernels.

    uv run torchrun --standalone --nproc_per_node=8 train/val/run.py
"""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
MARK = '# ' + '=' * 40 + ' module '
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # the allocator the record ran with


class Sections(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """The modules below this bootstrap, one per `MARK name` section, imported by name. Each is compiled from this
    file at its own line numbers, so tracebacks, inspect.getsource and Triton's source reads see the real lines."""

    def __init__(self):
        self.lines = FILE.read_text().splitlines(keepends=True)
        starts = [(i, line[len(MARK):].strip()) for i, line in enumerate(self.lines) if line.startswith(MARK)]
        self.spans = {name: (i + 1, j) for (i, name), (j, _) in zip(starts, starts[1:] + [(len(self.lines), '')])}

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self, origin=str(FILE)) if name in self.spans else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        a, b = self.spans[module.__name__]
        module.__file__ = str(FILE)
        exec(compile('\n' * a + ''.join(self.lines[a:b]), str(FILE), 'exec'), module.__dict__)


LOADER = Sections()


def main():
    sys.path.insert(0, str(ROOT))  # utils/
    sys.meta_path.insert(0, LOADER)
    import torch
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    import fast_moments as FM
    import fast_kernels as FK
    import fast_chain as FC
    plain = FK.install

    def install_kernels(R=None, **kwargs):  # every speedup and the moment tables, after the method's own install
        R = plain(R, **(dict(tf32=True, fused=True) | kwargs))
        FC.install(R)
        FM.install(R)
        if os.environ.get('RANK', '0') == '0':
            print('fast_moments wrap:', json.dumps({k: v for k, v in FM.describe().items() if k != 'checks'}), flush=True)
        return R
    FK.install = install_kernels
    import plant_polar as PP
    PP.install()
    R = PP.R
    R.__file__ = str(FILE)  # config.json's 'script'
    if os.environ.get('RANK', '0') == '0':
        print(f'run.py: installed {sorted(n for n in sys.modules if n in LOADER.spans)}', flush=True)
    FK.install(R)
    PP.SP2.FE.install_chain2(R)
    R.main()


if __name__ == '__main__':
    main()
    sys.exit(0)
raise ImportError('run.py is a script; its modules load through Sections')


# ======================================== module recipe
"""Record #2 (track 1 #2, the 10M record #14): Boost -> Cut with own queries, exact values, input-K / gate probes,
future-token probes and an end merge, as the research copy this record was built on (Settings fields for the
rate floor, the head's init and the merge; RUN_SETTINGS at the bottom are this run's).
"""
import argparse
from dataclasses import asdict, dataclass, field
from functools import cache, partial
from math import gcd
import json
import math
import os
from pathlib import Path
import time
from typing import Callable

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate

SEED = 0
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
    switch: float = 0.2
    boost_batch: int = 4
    cut_batch: int = 8
    projection_std: float = 1.5
    lr: float = 0.003
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.3
    final: float = 0.0  # research: the warmdown ends at this fraction of each rate
    head_std: float = 0.001  # research: the head's init std (the fixed init's 0.001)
    future_window: tuple = (0.0, 1.0)  # research: future probes only while this fraction of the epoch is done
    future_families: tuple = ('hidden', 'query', 'wte')  # research: which future-probe sources run
    cross_alpha: float = 1.0  # research (D40): the transported cross-token credit (extra) enters the writers times this
    cross_alpha_layers: tuple = ()  # research (D41): one alpha per block (overrides cross_alpha)
    boost_head: str = 'sum'  # research (D24 J2): 'avg' = Boost's head gradient is the mean of its 2L - 1 readouts, not their sum;
    #   'final' = only the last readout's (the head's exact gradient, as in backprop)
    anvil_lr: float = 0.037
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
    route_k: int = 512
    chain_tokens: int = 256
    chain_precision: str = 'bf16'
    own_query: bool = True  # PR #19's measured own-query paths in later attention blocks
    value_path: bool = True  # PR #23: one RMSNorm step, then only linear writers/V/VE
    path_scalars: bool = True  # also contract input-path credit into the linear mixing coefficients
    key_path_draws: int = 512  # independent input-K central pairs per token
    key_path_sigma: float = 0.01  # relative input RMS step
    key_path_tokens: int = 32  # source-token chunk; every allowed query contributes
    gate_path: bool = True  # independent full-cycle input probes through the VE gate
    gate_path_sigma: float = 0.01  # relative input RMS step
    future_draws: int = 512  # independent hidden/Q/WTE probes scored at every later raw hub
    future_layers: int = 4  # first blocks whose hidden and Q outputs receive this correction
    future_tokens: int = 64
    future_chunk: int = 512
    future_sigma: float = 0.01  # source-relative RMS; Q includes its own complete MLP response
    query_keys: int = 256   # moving key logits per head; 0 reruns all allowed logits
    key_sigma: float = 0.003
    key_chunk: int = 32

    def __post_init__(self):
        width, hidden = self.model.n_embd, 4 * self.model.n_embd
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

def norm_difference(x, delta):
    """norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves."""
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / x.shape[-1] + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


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
               BP: tl.constexpr, BK: tl.constexpr):
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
        dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
        weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(Rest + (token * Heads + head) * D + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(Out + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                 moved, probes[:, None] < Probes)

    @triton.jit
    def key_response(Q, DK, A, C, Out,
                     H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                     QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                     AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                     CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                     SourceStart, QueryStart, Window: tl.constexpr,
                     BC: tl.constexpr, BQ: tl.constexpr):
        head = tl.program_id(1)
        col = tl.program_id(0) * BC + tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        changed = tl.load(DK + (head * P * M + col[None, :]) * D + features[:, None],
                          col[None, :] < P * M, 0)
        first = tl.maximum(0, SourceStart + tl.min(source, 0) - QueryStart)
        last = tl.minimum(T, SourceStart + tl.max(source, 0) + Window + 1 - QueryStart)
        first = first // BQ * BQ
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start + tl.arange(0, BQ)
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD,
                        query[:, None] < T, 0)
            delta = tl.dot(q, changed, input_precision='tf32x3') * (D ** -.5)
            valid = (query[:, None] < T) & (col[None, :] < P * M)
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
            change = a * libdevice.expm1(delta)
            result += tl.sum(change / (1 + change) * c, 0)
        tl.store(Out + head * P * M + col, result, col < P * M)

    return routed, key_response


@torch.library.custom_op('noise_record::routed_response', mutates_args=())
def routed_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    probabilities: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    import triton
    m, probes, heads, width = dq.shape
    count = keys.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device, dtype=torch.float32)
    finite_attention_kernels()[0][(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(), keys.contiguous(), values.contiguous(), probabilities.contiguous(), rest.contiguous(),
        out, probes, heads, count, width, 32, max(16, triton.next_power_of_2(count)), num_warps=4)
    return out


@routed_response.register_fake
def _(dq, keys, values, probabilities, rest):
    return torch.empty(dq.shape, device=dq.device, dtype=torch.float32)


@torch.library.custom_op('noise_record::key_response', mutates_args=())
def fused_key_response(q: torch.Tensor, dk: torch.Tensor, A: torch.Tensor, C: torch.Tensor,
                       source_start: int, query_start: int, window: int) -> torch.Tensor:
    import triton
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    finite_attention_kernels()[1][(triton.cdiv(probes * sources, 64), heads)](
        q, dk.contiguous(), A, C, out, heads, q.shape[1], sources, probes, width,
        *q.stride(), *A.stride(), *C.stride(), source_start, query_start, window,
        64, 64, num_warps=4, enable_fp_fusion=False)
    return out


@fused_key_response.register_fake
def _(q, dk, A, C, source_start, query_start, window):
    return torch.empty(dk.shape[:3], device=q.device, dtype=torch.float32)


def query_change_routed(x, delta, pos, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos, sin):
    """Move only the selected key logits; retain the exact rescaling of the held output."""
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    if dq.is_cuda and top_keys.dtype == torch.bfloat16 and width in (32, 64) and top_keys.shape[2] <= 256:
        dout = routed_response(dq.to(top_keys.dtype), top_keys, top_values, top_p, rest)
        return dout.flatten(-2) @ w_proj_t
    ds = torch.einsum('mphd,mhkd->mphk', dq.to(top_keys.dtype), top_keys).float() * width ** -0.5
    p = top_p.unsqueeze(1)
    dlse = torch.log1p((p * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mphk,mhkd->mphd', (p * torch.expm1(ds - dlse)).to(top_values.dtype), top_values).float()
    dout = dout + torch.expm1(-dlse) * rest.unsqueeze(1)
    return dout.flatten(-2) @ w_proj_t


def chain_chunk_eager(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                      target_logit_w, y, own=None, from_output=False, packed=False):
    """Forward differences of each token's loss along every move of one site, K/V held, MLPs restricted to
    each token's top-k units. site: -1 for the raw embedding, else the block whose mid stream moves. move (m, P, W);
    scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block (m, W); idx, hs per block
    (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block (H, W); weight_t (W, V);
    base (m, V) clean top-stream logits before the norm; target_logit_w (m, W).
    own optionally supplies PR #19's own-query responses in every entered attention.
    from_output starts at the block's MLP output instead: skip its MLP, then enter
    the next block through BOTH its residual scaling and its own-query response.
    Returns (m, P)."""
    L = len(mids)
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
            else:
                _, pos, rels, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
                delta = delta + query_change(inputs[b], delta, pos, rels[b], w_q[b], keys[b], values[b],
                                             w_proj_t[b], cos, sin, windows[b])
        dz = norm_difference(mids[b], delta)
        fc = w_fc[b] if packed else w_fc[b][idx[b]]
        wo = w_out_t[b] if packed else w_out_t[b][idx[b]]
        dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)
chain_chunk_own = torch.compile(chain_chunk_eager, dynamic=True)
KEY_BUCKET = 256


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
            if i < L - 1 and s.boost_head != 'final':  # the head learns at every intermediate readout too
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
        if s.boost_head == 'avg':  # D24 J2: Adam ignores the scale inside Boost; only the hand-off to Cut changes
            gradients['lm_head.weight'] = gradients['lm_head.weight'] / (2 * L - 1)
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
        # research: split > 1 = the GPUs of a sequence each take a share of every cycle of moves and draws
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
            r0, r1 = self.share(total_rows)  # research: this GPU's share of the draws
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
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        # Draw bases in the original site order. Replay one sequence at a time
        # so routed K/V storage does not grow with the per-GPU batch size.
        probes = []
        lo, hi = self.share(self.width)  # research: this GPU's share of each cycle; the scalar moves on the first GPU
        P, scalars_here = hi - lo, self.part == 0
        for site in range(-1, L):
            H = self.basis()[lo:hi]
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
                    move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                    scale = steps[sl].expand(-1, P)
                    if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1),
                                          -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                           torch.full_like(scale[:, :2], -h)], 1)
                    derivative = kernel(
                        site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, selected_fc, selected_out, weight_t,
                        base[sl], target_w[sl], y[sl], own_sl, packed=True)
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
        head = d.head_error()  # coordinate differences at the final readout
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
            alpha = s.cross_alpha_layers[i] if s.cross_alpha_layers else s.cross_alpha  # research: D40 / D41
            x_mid, x_out = (d_mid, d_out) if extra is None else (d_mid + alpha * extra[i], d_out + alpha * extra[i])
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
            alphas = torch.tensor([s.cross_alpha_layers[i] if s.cross_alpha_layers else s.cross_alpha for i in range(L)],
                                  device=self.device)
            scalars += alphas * self.path_scalar_credit(inputs, extra, x0, outs)
        self.measured = dict(chain=chain, g_top=g_top, targets=targets)
        q_targets, key_errors = targets, self._input_key_errors
        self.internals(gradients, q_targets, b, t, value_targets or None, key_errors, raw_values)
        self._input_key_errors = None
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        if self.part:  # research (split): formed in full by every GPU of the sequence, kept by its first GPU (the
            # internals split or average their own: keys share dims, queries share draws, hidden units share units)
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight') or name in (
                        'transformer.wte.weight', 'resid_lambdas', 'x0_lambdas'):
                    gradients[name] = torch.zeros_like(gradients[name])
        if s.future_draws and s.future_window[0] <= getattr(self, 'progress', 0.0) < s.future_window[1]:
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


def future_resid_moves(raw, directions, step):
    """Research: central moves of a block's output stream itself (its MLP hidden units reach later blocks only through
    it, by the linear writer; their credit then takes the one ReLU^2 step of hidden_local_credit)."""
    delta = step[:, None]*directions
    return torch.cat((delta, -delta), 1)


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
compiled_resid = torch.compile(future_resid_moves, dynamic=False)
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
                    if family == "resid":
                        yield family, layer, f"resid:{layer}"
                        continue
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
        elif family == "resid":
            response = partial(compiled_resid, raw[sl])
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
        if split > 1:  # research: each GPU of the sequence takes a share of the draws, decoded at the full budget
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
            if family == "resid":  # block `layer`'s output stream: input + attention + MLP writers
                pre = f"transformer.h.{layer}."
                raw = (d.blocks[layer][self.sequence].float() + d.outputs[pre+"attn.c_proj"][self.sequence].float()
                       + d.outputs[pre+"mlp.c_proj"][self.sequence].float())
            else:
                raw = d.outputs[name][self.sequence].float()
            t, width = raw.shape
            if budgets[-1] > width:
                raise ValueError(f"{name}: budget {budgets[-1]} exceeds the {width}-direction cycle")
            gen = torch.Generator(raw.device).manual_seed(self.seed+7919*(layer+2)+101*("hidden query wte resid".split().index(family)))
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
        corrections = future.estimate((s.future_draws,), families=tuple(s.future_families))[s.future_draws]
        for site in [k for k in corrections if k.startswith('resid:')]:
            layer = int(site.split(':')[1])
            fc = f'transformer.h.{layer}.mlp.c_fc'
            u = d.outputs[fc][sequence].float()
            steps = s.fc_sigma*u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            hidden = hidden_local_credit(u, steps, corrections.pop(site)[0],
                                         e.model.transformer.h[layer].mlp.c_proj.weight.float())
            gradients[fc + '.weight'].addmm_(hidden.T, d.inputs[fc][sequence])
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


def build_model(device, settings):
    return make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)


def make_optimizer(model, recipe, steps):
    """ANVIL II on block matrices, Adam elsewhere."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
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


def holdout_loss(model, rows, rank, world):
    """Research: mean next-token loss over the held-out training sequences, split across ranks."""
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in rows[rank::world].split(8):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            total[0] += loss.double()
            total[1] += batch[:, 1:].numel()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])  # one matrix of block i (attn.c_q, ..., mlp.c_proj)
    return 've_gates' if 've_gate' in name else name


SOUP_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)  # research: weight of the newest snapshot
SOUP_RHOS = (0.0, 0.5, 1.0)  # research: how far a candidate moves outside the high-curvature subspace
SOUP_K, SOUP_TOKENS = 64, 512  # research: input-covariance directions per linear layer; frequent-token rows per table


@torch.no_grad()
def curvature_bases(model, rows, rank, world, k=SOUP_K):
    """Research: per linear layer, the top-k eigenvectors of its input covariance E[x x^T], and the token order of
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
    """Research: the part of a weight move inside its high-curvature subspace: the top input-covariance directions of a
    linear layer, the rows of the most frequent tokens of an embedding table; other tensors whole."""
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:SOUP_TOKENS]] = 1
        return move * mask
    return move


SOUP_COEFS = (0.75, 1.0, 1.25)  # research: multiples of each move in the joint search


@torch.no_grad()
def joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load,
                sweeps=2, shortlist=8):
    """Research: the joint search over keys (rho, j, c) = final + c * move(rho, j) per part (None = the final weights):
    the best single key for every part, then coordinate sweeps (all keys per part, then each part's best `shortlist`)."""
    parts = sorted({merge_part(n) for n, _ in named})
    members = {p: [i for i, (n, _) in enumerate(named) if merge_part(n) == p] for p in parts}
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in SOUP_COEFS]
    cache = {}

    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]

    def state(choice):
        return [value(i, choice[merge_part(n)]) for i, (n, _) in enumerate(named)]
    best_key, best = None, plain
    for key in keys[1:]:
        load([value(i, key) for i in range(len(params))])
        loss = holdout_loss(model, rows, rank, world)
        if loss < best:
            best_key, best = key, loss
    choice = {p: best_key for p in parts}
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
                load(state(trial))
                loss = holdout_loss(model, rows, rank, world)
                tried.append((loss, key))
                if loss < best:
                    best, choice = loss, trial
            if sweep == 0:
                ranked[p] = [k for _, k in sorted(tried, key=lambda x: x[0])[:shortlist]]
    load(state(choice))
    return dict(holdout_plain=plain, holdout_soup=best,
                soup_choices={p: (None if k is None else [k[0], merge_label(k[1]), k[2]]) for p, k in choice.items()})


MERGE_EXTRA = []  # research: --merge-tails: labels of the tail averages appended to the merge's candidates


def merge_label(j):
    return SOUP_DECAYS[j] if j < len(SOUP_DECAYS) else MERGE_EXTRA[j - len(SOUP_DECAYS)]


@torch.no_grad()
def soup_parts(model, emas, rows, rank, world, fit_rows=None, joint=False):
    """Research: per part (each block matrix, each table, the scalars), greedily on the held-out loss, the candidate
    move toward an exponential mean of the late snapshots (one per decay in SOUP_DECAYS) taken in full inside the part's
    high-curvature subspace and a fraction rho (SOUP_RHOS) outside it (curvature_bases, from forward passes on
    `fit_rows`; rho = 1 is the plain move), then a blend coefficient for the chosen move. Every rank takes the same
    decisions (the losses are all-reduced)."""
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, fit_rows, rank, world) if fit_rows is not None else ({}, None)
    rhos = SOUP_RHOS if fit_rows is not None else (1.0,)
    candidates = [(rho, j) for rho in rhos for j in range(len(emas))]

    def move(i, rho, j):
        full = emas[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)

    def load(ws):
        for q, w in zip(params, ws):
            q.copy_(w.to(q.dtype))
    current = list(final)
    load(current)
    best = plain = holdout_loss(model, rows, rank, world)
    if joint:  # research: one shared candidate for every part, then coordinate sweeps (research/merge_lab/combined.py)
        return joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load)
    chosen = {}
    for part in sorted({merge_part(n) for n, _ in named}):
        members = [i for i, (n, _) in enumerate(named) if merge_part(n) == part]
        chosen[part] = None
        for rho, j in candidates:
            trial = list(current)
            for i in members:
                trial[i] = final[i] + move(i, rho, j)
            load(trial)
            loss = holdout_loss(model, rows, rank, world)
            if loss < best:
                best, current, chosen[part] = loss, trial, (rho, j)
    blends = {}  # then per chosen part, a multiple of its chosen move
    for part, choice in chosen.items():
        if choice is None:
            continue
        members = [i for i, (n, _) in enumerate(named) if merge_part(n) == part]
        moves = {i: move(i, *choice) for i in members}
        for a in (0.25, 0.5, 0.75, 1.25):
            trial = list(current)
            for i in members:
                trial[i] = final[i] + a * moves[i]
            load(trial)
            loss = holdout_loss(model, rows, rank, world)
            if loss < best:
                best, current, blends[part] = loss, trial, a
    load(current)
    return dict(holdout_plain=plain, holdout_soup=best, soup_blends=blends,
                soup_choices={part: (None if c is None else [c[0], merge_label(c[1])]) for part, c in chosen.items()})


def train_stages(args, settings, device, rank, world):
    """Boost -> own-query Cut; each row is consumed exactly once, including across resume."""
    if world not in (1, 2, 4, 8):
        raise ValueError('Stage training supports 1, 2, 4 or 8 GPUs')
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    # research: the data order (and so the held-out rows) stays the seed-0 order for every --seed
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(1))]
    holdout_rows = None
    if args.holdout:  # research: these sequences feed no update (merge candidates are scored on them)
        holdout_rows, training = training[-args.holdout:], training[:-args.holdout]
    if args.train_rows:  # research (30M track 2): one shorter epoch on the first N rows, the schedule fit to it
        training = training[:args.train_rows]
    emas = None  # research: exponential means of the late snapshots, one per SOUP_DECAYS entry
    schedule = stage_schedule(len(training), settings)
    steps = min(args.steps or len(schedule), len(schedule))
    # research: --tail-rates f: EMA at rate 1 / round(f * updates) from --tail-from; --tail-powers gamma@start: power EMA
    tails = [dict(key=f'ema{f}', rate=1 / round(float(f) * len(schedule)), gamma=None, start=args.tail_from, t=0,
                  ema=None) for f in args.tail_rates.split(',') if f]
    tails += [dict(key=f'pow{g}@{a}', rate=None, gamma=float(g), start=float(a), t=0, ema=None)
              for g, a in (spec.split('@') for spec in args.tail_powers.split(',') if spec)]
    stop = min(steps, args.stop_after_step or steps)
    split = max(1, world // settings.boost_batch)
    groups = ([torch.distributed.new_group(list(range(i, i+split))) for i in range(0, world, split)]
              if split > 1 else [None])
    group = groups[rank // split] if split > 1 else None
    cut_split = max(1, world // settings.cut_batch)  # research: GPUs per sequence in Cut
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
                       'independent_future_hidden_query_embedding_probes'],
            resume=str(args.resume.resolve()) if args.resume else None, torch_version=torch.__version__,
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
        model = build_model(device, settings)
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
    model = build_model(device, settings)
    optimizer, estimator = make_optimizer(model, settings, phases[phase]), estimator_for(model, phase)
    if args.resume:
        completed, previous_seconds = restore_training_state(args.resume, model, optimizer, estimator, recipe=recipe_state)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, paused, elapsed = time.perf_counter(), 0., previous_seconds
    try:
        for step in range(completed+1, stop+1):
            current, phase_step, begin, end, horizon = schedule[step-1]
            if current != phase:
                estimator.close()
                next_optimizer = make_optimizer(model, settings, horizon)
                handoff_optimizer(optimizer, next_optimizer, phase_step)
                optimizer, estimator, phase = next_optimizer, estimator_for(model, current), current
            batch = rank_slice(training[begin:end], phase)
            set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
            estimator.progress = begin / len(training)  # research: future_window
            x, y = batch[:, :-1], batch[:, 1:]
            loss = estimator.step(x.contiguous(), y.contiguous(), optimizer)
            with torch.no_grad():  # research: tail averages (track 2 #15) and power EMAs, one fp32 copy each
                for tail in tails:
                    if estimator.progress < tail['start']:
                        continue
                    tail['t'] += 1
                    if tail['ema'] is None:
                        tail['ema'] = [q.detach().float().clone() for q in model.parameters()]
                        continue
                    # fixed rate, or power EMA (weights ~ t^gamma over its window; gamma 0 = uniform average)
                    a = tail['rate'] if tail['rate'] else 1 - (1 - 1 / tail['t']) ** (tail['gamma'] + 1)
                    for e, q in zip(tail['ema'], model.parameters()):
                        e.lerp_(q.detach().float(), a)
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
            if args.save_weights_every and step % args.save_weights_every == 0 and rank == 0:  # research: analysis
                torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()},
                           args.output/f'weights_{step}.pt')
            if args.snap_every and (step % args.snap_every == 0 or step == stop) and begin / len(training) >= args.snap_from:
                if rank == 0 and args.dump_snaps:
                    (args.output/'snaps').mkdir(exist_ok=True)  # research: research/soup_lab.py
                    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()},
                               args.output/'snaps'/f'step{step:05d}.pt')
                if args.soup_parts:
                    with torch.no_grad():
                        if emas is None:
                            emas = [[q.detach().float().clone() for q in model.parameters()] for _ in SOUP_DECAYS]
                        else:
                            for a, ema in zip(SOUP_DECAYS, emas):
                                for e, q in zip(ema, model.parameters()):
                                    e.lerp_(q.detach().float(), a)
            if (args.checkpoint_every and (step % args.checkpoint_every == 0 or step == stop)) or step == stop < steps:
                pause = time.perf_counter()
                save_training_state(args.output/'checkpoint.pt', model, optimizer, estimator, step=step,
                                    train_seconds=elapsed, recipe=recipe_state,
                                    progress=dict(phase=phase, phase_step=phase_step, seqs=end, epoch_updates=len(schedule)))
                if args.keep_checkpoints and rank == 0:  # research: keep every training state (branches resume from them)
                    os.link(args.output/'checkpoint.pt', args.output/f'checkpoint_{step}.pt')
                paused += time.perf_counter()-pause
        if stop < steps:
            if rank == 0:
                (args.output/'segment.json').write_text(json.dumps(dict(step=stop, total_steps=steps, seqs=end,
                    train_seconds=elapsed, checkpoint=str((args.output/'checkpoint.pt').resolve())), indent=2)+'\n')
            return
        extra = {}
        if any(tail['ema'] is not None for tail in tails):  # research: plain + mix * (average - plain), untimed
            params, plain_params = list(model.parameters()), [q.detach().clone() for q in model.parameters()]
            grid = {}
            with torch.no_grad():
                for tail in tails:
                    if tail['ema'] is None:
                        continue
                    for mix in args.tail_mixes.split(','):
                        for q, p0, e in zip(params, plain_params, tail['ema']):
                            q.copy_((p0.float() + float(mix) * (e - p0.float())).to(q.dtype))
                        key = f"{tail['key']}_mix{mix}"
                        grid[key] = dict(val=evaluate(model.state_dict(), args.data_dir),
                                         holdout=holdout_loss(model, holdout_rows, rank, world)
                                         if holdout_rows is not None else None)
                for q, p0 in zip(params, plain_params):
                    q.copy_(p0)
            extra['tail_grid'] = grid
            wanted = args.merge_tails.split(',') if args.merge_tails else []
            extra_means = [tail['ema'] for key in wanted for tail in tails if tail['key'] == key]
            if len(extra_means) != len(wanted):
                raise ValueError(f'--merge-tails names a tail average that was not kept: {wanted}')
            MERGE_EXTRA[:] = wanted
            if emas:  # research: the chosen tail averages join the merge's candidates
                emas = emas + extra_means
            tails = []
        if args.soup_parts and emas:  # the end merge, per part, scored on the held-out rows; timed with the training
            plain = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            extra['val_loss_plain'] = evaluate(plain, args.data_dir)  # untimed, like every validation
            merge_start = time.perf_counter()  # the held-out rows feed only this merge (curvature bases + choices)
            extra.update(soup_parts(model, emas, holdout_rows, rank, world,
                                    fit_rows=holdout_rows if args.soup_subspace else None, joint=args.soup_joint))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
            elapsed += time.perf_counter() - merge_start
            if rank == 0:
                torch.save(plain, args.output/'final_plain.pt')
            emas = None
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


# The run's settings: every field this record sets away from the Settings defaults.
RUN_SETTINGS = {'cut_batch': 4, 'anvil_lr': 0.026, 'lr': 0.0021, 'betas': [0.5, 0.98], 'warmdown': 1.0, 'final': 0.3,
                'head_std': 0.02, 'future_draws': 128, 'boost_head': 'avg', 'future_onesided': True, 'switch': 0.0668,
                'anvil_wd': 0.75, 'readout': True, 'readout_k_blocks': [0, 1, 2, 3], 'readout_k_receivers': 16,
                'readout_gate': True, 'readout_hi_out3': True, 'readout_hi_blocks': [4, 5, 6], 'readout_in_roots': [],
                'readout_hi_top_root': True, 'readout_shared': True, 'head_edit': True, 'head_edit_start': 50,
                'head_edit_every': 25, 'head_edit_stop': 450, 'head_edit_per_block': 2, 'head_edit_nxfer': 5,
                'head_edit_betas': [0.5, 1.0, 1.5], 'head_edit_deltas': [1, 2, 3], 'head_edit_protect': 200, 'distill': True,
                'distill_lam': 0.3, 'distill_beta': 0.99, 'distill_start': 0.1, 'twin': True, 'twin_kc': [5, 7],
                'twin_transport': 'identity', 'twin_consumers': 'linear', 'speed_exact': True, 'speed_store': True,
                'speed_lite8': True, 'readout_q_full': [], 'readout_fc_cycle': [128, 128, 128, 128],
                'readout_hi_fc_cycle': [128, 128, 128], 'readout_skip_blocks': [5, 6], 'readout_k_select': 'bound',
                'transit': True, 'transit_store': 'fp32', 'transit_base_block': 0, 'transit_base_mid': True,
                'transit_roots': [0, 1, 2, 3, 4, 5, 6, 7], 'query_central': False, 'query_sigma': 0.1,
                'boost_emb_block0': True, 'transit_inroots': True, 'readout_shared_pairs': True,
                'plant': {'c': 0.0, 'prev': [[4, 0]]}, 'anvil_head_polar': ['q', 'k'], 'adam_scalar_mult': 60.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)  # research: init and probe randomness only
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/val'))
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=406)
    parser.add_argument('--save-weights-every', type=int, default=0)  # research: weights every N updates
    parser.add_argument('--holdout', type=int, default=32)  # the last 32 training sequences feed no update (merge)
    parser.add_argument('--train-rows', type=int, default=0)  # research: train on the first N rows only (track 2)
    parser.add_argument('--keep-checkpoints', action='store_true')  # research: checkpoint_<step>.pt next to checkpoint.pt
    parser.add_argument('--snap-every', type=int, default=5)  # the merge's weight means: every 5 updates
    parser.add_argument('--snap-from', type=float, default=0.6)
    parser.add_argument('--dump-snaps', action='store_true')  # research: also save the snapshots under snaps/
    parser.add_argument('--soup-parts', action=argparse.BooleanOptionalAction, default=True)  # per-part end merge
    parser.add_argument('--soup-subspace', action=argparse.BooleanOptionalAction, default=True)  # by curvature subspace
    parser.add_argument('--soup-joint', action=argparse.BooleanOptionalAction, default=True)  # joint start + sweeps
    parser.add_argument('--tail-rates', default='')  # research: e.g. '0.1,0.25,0.5' -> EMA rate 1/round(f*N)
    parser.add_argument('--tail-mixes', default='0.4,0.7,1.0')  # research: final move toward each tail average
    parser.add_argument('--tail-from', type=float, default=0.6)
    parser.add_argument('--merge-tails', default='')  # research: tail-average keys (e.g. pow12.67@0) the merge may pick
    parser.add_argument('--tail-powers', default='')  # research: e.g. '0@0.6,4@0' -> power EMA gamma from that fraction
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--steps', type=int, help='Diagnostic prefix; preserves the full-epoch LR schedule.')
    # Execution chunks only: the default command reproduces the logged algorithm and budgets.
    parser.add_argument('--chain-tokens', type=int, default=512)
    parser.add_argument('--key-chunk', type=int, default=32)
    parser.add_argument('--key-path-tokens', type=int, default=64)
    parser.add_argument('--future-tokens', type=int, default=512)
    parser.add_argument('--future-chunk', type=int, default=512)
    parser.add_argument('--settings-json', default='', help='Overrides of the run\'s Settings fields (RUN_SETTINGS).')
    args = parser.parse_args()
    global SEED
    SEED = args.seed
    for key in ('steps', 'stop_after_step'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(key + ' must be positive')
    if min(args.checkpoint_every, args.eval_every) < 0:
        parser.error('Logging intervals must be nonnegative')
    overrides = RUN_SETTINGS | (json.loads(args.settings_json) if args.settings_json else {})
    overrides = {k: tuple(v) if k in ('betas', 'future_window', 'future_families', 'cross_alpha_layers') else v
                 for k, v in overrides.items()}
    settings = Settings(chain_tokens=args.chain_tokens, key_chunk=args.key_chunk,
                        key_path_tokens=args.key_path_tokens, future_tokens=args.future_tokens,
                        future_chunk=args.future_chunk, **overrides)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    try:
        train_stages(args, settings, device, rank, world)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


# ======================================== module onesided
"""One-sided future probes. The record measures each future direction d as a +-step pair; here the hidden / query / wte
families measure only +step d and decode s(+) / step d (the record's own one-sided decode, LocalExperiment(central=False)).
The directions, draws, budget and two-GPU split are the record's; signs are uniform, so the expected credit is the same
with half the experiments. Each trial is one forward evaluation of the moved source scored at a measured hub.
"""
import dataclasses
from dataclasses import dataclass
from functools import partial
import sys

import torch
import torch.nn.functional as F

import recipe  # noqa: E402,F401  (the 'recipe' section: one recipe module per process, by this name)
R = sys.modules['recipe']
RECORD_SETTINGS, RECORD_ASDICT, RECORD_EXPERIMENT = R.Settings, R.asdict, R.FiniteFuture.experiment
#   arguments), the responses copied below, the one-sided decode, estimate's +-1 rows and budget division


# --------------------------------------------------------------------------------------------- the + halves (compiled)


def plus_hidden(raw, directions, step, weight):
    """The + half of R.signed_hidden_moves, its own ops: per unit even + d odd = relu^2(raw + step d) - relu^2(raw)."""
    base = raw.relu()
    plus, minus = (raw+step).relu(), (raw-step).relu()
    dp, dm = (plus-base)*(plus+base), (minus-base)*(minus+base)
    odd, even = (dp-dm)*.5, (dp+dm)*.5
    moved = F.linear(directions*odd[:, None], weight)
    center = F.linear(even, weight)[:, None]
    return center+moved


def plus_embedding(raw, directions, step):  # the + half of R.future_embedding_moves
    return R.norm_delta(raw[:, None], step[:, None]*directions)


def plus_query(raw, directions, step, cos, sin, keys, values, probabilities, weight, base):
    """The + half of R.streamed_query_moves: only the +step d queries go through the finite softmax response."""
    m, p, width = directions.shape
    heads, _, hd = keys.shape
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = R.rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = R.rotate((step[:, None]*directions).reshape(m, p, heads, hd), c, s)
    response = R.query_response(R.norm_delta(r0, dr), keys, values, probabilities, base)
    return F.linear(response.flatten(-2), weight)


def plus_query_mlp(raw, directions, step, cos, sin, keys, values, probabilities, weight, base, mid, hidden, w_fc, w_out):
    move = plus_query(raw, directions, step, cos, sin, keys, values, probabilities, weight, base)
    return R.mlp_output_moves(mid, move, hidden, w_fc, w_out)


PLUS = {R.compiled_hidden: torch.compile(plus_hidden, dynamic=False),  # the record's response function -> its + half
        R.compiled_embedding: torch.compile(plus_embedding, dynamic=False),
        R.compiled_query: torch.compile(plus_query, dynamic=False),
        R.query_mlp_moves: torch.compile(plus_query_mlp, dynamic=False)}


# ---------------------------------------------------------------------------------------------------------- the patch


def experiment(self, family, layer, raw, sl, receivers, start):
    """FiniteFuture.experiment: the record's; with future_onesided (hidden / query / wte) its response partial runs the + half
    of the same function on the same bound arguments, its scorer unchanged, decoded one-sided (score / step)."""
    trial = RECORD_EXPERIMENT(self, family, layer, raw, sl, receivers, start)
    if family == 'resid' or not getattr(self.e.s, 'future_onesided', False):
        return trial
    response = trial.response
    plus = PLUS.get(response.func)
    if plus is None:
        raise RuntimeError(f'onesided: {family} response {response.func!r} is not the record\'s (replaced by a wrapper?)')
    return R.LocalExperiment(partial(plus, *response.args, **response.keywords), trial.score, central=False)


@dataclass
class Settings(RECORD_SETTINGS):  # the record's fields plus one
    future_onesided: bool = False  # research (direction 4): the hidden / query / wte future probes measure each direction once
    #   (+step, moved minus clean) instead of as a +/- pair: the same directions, half the experiments

    def __post_init__(self):
        super().__post_init__()
        if type(self.future_onesided) is not bool:
            raise ValueError('future_onesided is true or false')


def asdict(settings):  # off, the recipe dict (config.json, resume, old checkpoints) is the record's
    state = RECORD_ASDICT(settings)
    if not getattr(settings, 'future_onesided', False):
        state.pop('future_onesided', None)
    return state


def install():
    """Patch the loaded record module in place (idempotent): Settings, its recipe dict, FiniteFuture.experiment."""
    foreign = (RECORD_SETTINGS.__module__ != R.__name__ or RECORD_EXPERIMENT.__module__ != R.__name__
               or RECORD_ASDICT is not dataclasses.asdict)
    if foreign or R.Settings not in (RECORD_SETTINGS, Settings) or R.asdict not in (RECORD_ASDICT, asdict) or \
            R.FiniteFuture.experiment not in (RECORD_EXPERIMENT, experiment):
        raise RuntimeError('onesided: another wrapper replaced Settings, asdict or FiniteFuture.experiment')
    R.Settings, R.asdict, R.FiniteFuture.experiment = Settings, asdict, experiment
    return R


# ------------------------------------------------------------------------------------------------------------- checks


#   untimed Cut step and the GPU work of its paired future probes (fused scorer .36 s, query kernel .045 s, responses and
#   scorer prologues), which bounds the paired loop there
rel = lambda u, v: float(f'{((u - v).norm() / v.norm().clamp_min(1e-30)).item():.3g}')


# ======================================== module fast_kernels
"""Faster Cut-step kernels: the record's custom ops get CUDA kernels with wider tiles (the same arithmetic per (query,
column) pair, only the order of the sums differs), the scorer's prologue fused, one-pass TF32 in the future and key
scorers. install() registers them and swaps recipe.receiver_scores.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


SHA1 = '5faeaca000e5'  # sha1 of the research file this section was copied from
PIN = '44d60a074acc'  # sha1 of the record pieces these copy or rely on (install() checks it)
# launch settings per mode, measured on H100 at the record shapes (bench): future / key / fused (BS sources, TP trials,
# warps, stages), query (BP probes, warps, stages); the record: future 1 x 32, key 64 columns, query 32, 1 stage
TILES = dict(fast=dict(future=(32, 2, 4, 2), key=(32, 2, 4, 2), query=(128, 4, 2), fused=(32, 2, 4, 2)),
             strict=dict(future=(32, 1, 4, 2), key=(32, 2, 4, 2), query=(32, 4, 2), fused=(32, 2, 4, 2)))
QUERY_PRECISION = 'tf32x3'  # the Q-response kernel's dots in every mode (see --tf32)
STATE = dict(mode=None, tiles=dict(TILES['fast']), precision='tf32x3', fused=False, installed=None, noted=False)


# --------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def future_kernel(Q, G, DK, DV, A, C, Out,
                  T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                  KH: tl.constexpr, KP: tl.constexpr, KM: tl.constexpr, KD: tl.constexpr,
                  VH: tl.constexpr, VP: tl.constexpr, VM: tl.constexpr, VD: tl.constexpr,
                  QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                  GH: tl.constexpr, GT: tl.constexpr, GD: tl.constexpr,
                  AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                  CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                  Start, Window: tl.constexpr, BS: tl.constexpr, TP: tl.constexpr, BQ: tl.constexpr,
                  PRECISION: tl.constexpr, EVEN: tl.constexpr):
    """The record's score_kernel; a program scores TP trials x BS sources. Every column adds the same BQ-query tiles
    in the same order. The record's mask (in range, query after the source, within its window) is applied on the
    tiles where it is false for some pair; on the others every query of the tile is after every source of the block
    and within every source's window, so the mask is true for all pairs and dropping it changes nothing (for any a).
    EVEN (T % BQ, P % TP, M % BS all 0): every load is in bounds, so none is masked. dk / dv [h, P, M, D] may have
    any strides (K*, V*), so receiver_changes' permuted views need no copy."""
    pblock, sblock, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    j = tl.arange(0, TP * BS)
    trial = pblock * TP + j // BS
    source = sblock * BS + j % BS
    ok = (trial < P) & (source < M)
    col = trial * M + source
    features = tl.arange(0, D)
    k_offsets = head * KH + trial[None, :] * KP + source[None, :] * KM + features[:, None] * KD
    v_offsets = head * VH + trial[None, :] * VP + source[None, :] * VM + features[:, None] * VD
    if EVEN:
        dk = tl.load(DK + k_offsets)
        dv = tl.load(DV + v_offsets)
    else:
        dk = tl.load(DK + k_offsets, ok[None, :], 0)
        dv = tl.load(DV + v_offsets, ok[None, :], 0)
    low = Start + sblock * BS  # the block's first and last source positions
    high = Start + tl.minimum(sblock * BS + BS, M) - 1
    first = low // BQ * BQ
    last = tl.minimum(T, high + Window + 1)
    result = tl.full((TP * BS,), 0, tl.float32)
    for start in range(first, last, BQ):
        query = start + tl.arange(0, BQ)
        if EVEN:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD)
            g = tl.load(G + head * GH + query[:, None] * GT + features[None, :] * GD)
        else:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD, query[:, None] < T, 0)
            g = tl.load(G + head * GH + query[:, None] * GT + features[None, :] * GD, query[:, None] < T, 0)
        ds = tl.dot(q, dk, input_precision=PRECISION) * (D ** -.5)
        vs = tl.dot(g, dv, input_precision=PRECISION)
        valid = (query[:, None] < T) & ok[None, :]
        if EVEN:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM)
        else:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
        changed = a * libdevice.expm1(ds)
        response = (changed * c + (a + changed) * vs) / (1 + changed)
        if EVEN:
            if (start > high) & (start + BQ - 1 - low <= Window):  # the mask is true for every pair of this tile
                result += tl.sum(response, 0)
            else:
                allowed = (query[:, None] > Start + source[None, :]) & (
                    query[:, None] - Start - source[None, :] <= Window)
                result += tl.sum(tl.where(allowed, response, 0), 0)
        else:
            allowed = valid & (query[:, None] > Start + source[None, :]) & (
                query[:, None] - Start - source[None, :] <= Window)
            result += tl.sum(tl.where(allowed, response, 0), 0)
    tl.store(Out + head * P * M + col, result, ok)


@triton.jit
def fused_kernel(Q, G, PJ, R0, CS, SN, GC, VE, A, C, Out,
                 T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                 CSM: tl.constexpr, CSF: tl.constexpr, SNM: tl.constexpr, SNF: tl.constexpr, GATED: tl.constexpr,
                 QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                 GH: tl.constexpr, GT: tl.constexpr, GD: tl.constexpr,
                 AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                 CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                 Start, Window: tl.constexpr, BS: tl.constexpr, TP: tl.constexpr, BQ: tl.constexpr,
                 PRECISION: tl.constexpr):
    """future_kernel (even shapes) with receiver_changes' last step in its prologue: from the projection PJ
    [M, P, 2, H, D] (F.linear(dn, wkv)), the rotated clean key R0 [M, H, D] and the rope rows CS / SN [M, D/2], each
    column's dk = norm_delta(R0, rotate(PJ[.., 0, ..])) and dv = PJ[.., 1, ..] (+ GC [M, P, H] * VE [M, H, D]), the
    record's formulas in fp32 (rounding may differ from Inductor's fused kernel in the last bit)."""
    pblock, sblock, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    j = tl.arange(0, TP * BS)
    trial = pblock * TP + j // BS
    source = sblock * BS + j % BS
    col = trial * M + source
    features = tl.arange(0, D)
    half = features % (D // 2)
    base = ((source[None, :] * P + trial[None, :]) * 2 * H + head) * D
    pk = tl.load(PJ + base + features[:, None])
    pk_other = tl.load(PJ + base + (features[:, None] + D // 2) % D)
    cs = tl.load(CS + source[None, :] * CSM + half[:, None] * CSF)
    sn = tl.load(SN + source[None, :] * SNM + half[:, None] * SNF)
    low_half = features[:, None] < D // 2
    dr = tl.where(low_half, pk * cs + pk_other * sn, -pk_other * sn + pk * cs)  # rotate: [x1 c + x2 s, -x1 s + x2 c]
    r0 = tl.load(R0 + (source[None, :] * H + head) * D + features[:, None])
    square = tl.sum(r0 * r0, 0)
    inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
    moved = libdevice.rsqrt((square + 2 * tl.sum(r0 * dr, 0) + tl.sum(dr * dr, 0)) / D + 1.1920928955078125e-07)
    dk = dr * moved[None, :] + r0 * (moved - inverse)[None, :]
    dv = tl.load(PJ + base + H * D + features[:, None])
    if GATED:
        gc = tl.load(GC + (source * P + trial) * H + head)
        dv = dv + gc[None, :] * tl.load(VE + (source[None, :] * H + head) * D + features[:, None])
    low = Start + sblock * BS
    high = Start + sblock * BS + BS - 1
    first = low // BQ * BQ
    last = tl.minimum(T, high + Window + 1)
    result = tl.full((TP * BS,), 0, tl.float32)
    for start in range(first, last, BQ):
        query = start + tl.arange(0, BQ)
        q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD)
        g = tl.load(G + head * GH + query[:, None] * GT + features[None, :] * GD)
        ds = tl.dot(q, dk, input_precision=PRECISION) * (D ** -.5)
        vs = tl.dot(g, dv, input_precision=PRECISION)
        a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM)
        c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM)
        changed = a * libdevice.expm1(ds)
        response = (changed * c + (a + changed) * vs) / (1 + changed)
        if (start > high) & (start + BQ - 1 - low <= Window):  # the record's mask is true for every pair here
            result += tl.sum(response, 0)
        else:
            allowed = (query[:, None] > Start + source[None, :]) & (query[:, None] - Start - source[None, :] <= Window)
            result += tl.sum(tl.where(allowed, response, 0), 0)
    tl.store(Out + head * P * M + col, result)


@triton.jit
def key_kernel(Q, DK, A, C, Out,
               H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               KH: tl.constexpr, KP: tl.constexpr, KM: tl.constexpr, KD: tl.constexpr,
               QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
               AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
               CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
               SourceStart, QueryStart, Window: tl.constexpr,
               BS: tl.constexpr, TP: tl.constexpr, BQ: tl.constexpr, PRECISION: tl.constexpr, EVEN: tl.constexpr):
    """The record's key_response; a program scores TP trials x BS sources over its own sources' query range (the
    record's kernel has no mask either: A, C are 0 outside each source's window). EVEN: no load is masked. dk
    [h, P, M, D] may have any strides (prepared_key_moves' permuted view needs no copy)."""
    pblock, sblock, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    j = tl.arange(0, TP * BS)
    trial = pblock * TP + j // BS
    source = sblock * BS + j % BS
    ok = (trial < P) & (source < M)
    col = trial * M + source
    features = tl.arange(0, D)
    k_offsets = head * KH + trial[None, :] * KP + source[None, :] * KM + features[:, None] * KD
    if EVEN:
        changed = tl.load(DK + k_offsets)
    else:
        changed = tl.load(DK + k_offsets, ok[None, :], 0)
    first = tl.maximum(0, SourceStart + sblock * BS - QueryStart)
    last = tl.minimum(T, SourceStart + tl.minimum(sblock * BS + BS, M) - 1 + Window + 1 - QueryStart)
    first = first // BQ * BQ
    result = tl.full((TP * BS,), 0, tl.float32)
    for start in range(first, last, BQ):
        query = start + tl.arange(0, BQ)
        if EVEN:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD)
        else:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD, query[:, None] < T, 0)
        delta = tl.dot(q, changed, input_precision=PRECISION) * (D ** -.5)
        valid = (query[:, None] < T) & ok[None, :]
        if EVEN:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM)
        else:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
        change = a * libdevice.expm1(delta)
        result += tl.sum(change / (1 + change) * c, 0)
    tl.store(Out + head * P * M + col, result, ok)


@triton.jit
def query_kernel(DQ, K, V, A, Base, Out,
                 M: tl.constexpr, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 T: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                 VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                 AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                 OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                 BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr):
    """The record's query_kernel (future Q family), the dots' precision a parameter."""
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    probe = tl.program_id(0) * BP + tl.arange(0, BP)
    feature = tl.arange(0, D)
    delta = tl.load(DQ + ((token * P + probe[:, None]) * H + head) * D + feature[None, :], probe[:, None] < P, 0)
    numerator = tl.full((BP, D), 0, tl.float32)
    partition = tl.full((BP,), 0, tl.float32)
    for start in range(0, T, BK):
        key = start + tl.arange(0, BK)
        a = tl.load(A + head * AH + token * AM + key * AT, key < T, 0)
        if tl.sum(a, 0) > 0:  # causal / window-masked tiles are exactly zero
            k = tl.load(K + head * KH + key[None, :] * KT + feature[:, None] * KD, key[None, :] < T, 0)
            v = tl.load(V + head * VH + key[:, None] * VT + feature[None, :] * VD, key[:, None] < T, 0)
            ds = tl.dot(delta, k, input_precision=PRECISION) * (D ** -.5)
            change = a[None, :] * libdevice.expm1(ds)
            partition += tl.sum(change, 1)
            numerator += tl.dot(change, v, input_precision=PRECISION)
    base = tl.load(Base + head * OH + token * OM + feature * OD)
    response = (numerator - partition[:, None] * base[None, :]) / (1 + partition[:, None])
    tl.store(Out + ((token * P + probe[:, None]) * H + head) * D + feature[None, :], response, probe[:, None] < P)


# ------------------------------------------------------------------------------------- the ops' CUDA implementations


def future_scores(q, target, dk, dv, a, contrast, source_start, window):
    h, p, m, d = dk.shape
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    bs, tp, warps, stages = STATE['tiles']['future']
    even = q.shape[1] % 64 == 0 and p % tp == 0 and m % bs == 0
    future_kernel[(triton.cdiv(p, tp), triton.cdiv(m, bs), h)](
        q, target, dk, dv, a, contrast, out, q.shape[1], m, p, d, *dk.stride(), *dv.stride(), *q.stride(),
        *target.stride(), *a.stride(), *contrast.stride(), source_start, window, bs, tp, 64, STATE['precision'], even,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out


def key_response(q, dk, A, C, source_start, query_start, window):
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    bs, tp, warps, stages = STATE['tiles']['key']
    even = q.shape[1] % 64 == 0 and probes % tp == 0 and sources % bs == 0
    key_kernel[(triton.cdiv(probes, tp), triton.cdiv(sources, bs), heads)](
        q, dk, A, C, out, heads, q.shape[1], sources, probes, width, *dk.stride(), *q.stride(), *A.stride(),
        *C.stride(), source_start, query_start, window, bs, tp, 64, STATE['precision'], even, num_warps=warps,
        num_stages=stages, enable_fp_fusion=False)
    return out


def query_response(dq, keys, values, probabilities, base):
    m, p, h, d = dq.shape
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    bp, warps, stages = STATE['tiles']['query']
    query_kernel[(triton.cdiv(p, bp), m * h)](
        dq.contiguous(), keys, values, probabilities, base, out, m, p, h, d, keys.shape[1], *keys.stride(),
        *values.stride(), *probabilities.stride(), *base.stride(), bp, 64, QUERY_PRECISION, num_warps=warps,
        num_stages=stages, enable_fp_fusion=False)
    return out


def receiver_scores_views(R):
    """R.receiver_scores with R.receiver_changes' last line returning the permuted [h, P, m, d] views instead of
    contiguous copies (the fast scorer reads any strides): one copy of dv (and a strided rewrite of dk) less per
    receiver call; bitwise the record's dk / dv at the record's shapes (Inductor picks the kernels that form them)."""
    norm_delta, rotate, fused_scores = R.norm_delta, R.rotate, R.fused_scores

    def receiver_changes(x, move, kraw, gate, ve, wkv, wg, cos, sin):  # copied from the record (pinned)
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
        return dk.permute(2, 1, 0, 3), dv.permute(2, 1, 0, 3)

    @torch.compile(dynamic=True)
    def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin,
                        q, target, a, contrast, start, window):
        dk, dv = receiver_changes(x, move*coefficient, kraw, gate, ve, wkv, wg, cos, sin)
        return fused_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T
    return receiver_scores


@torch.library.custom_op('fast_kernels::projected_scores', mutates_args=())
def projected_scores(q: torch.Tensor, target: torch.Tensor, projected: torch.Tensor, r0: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor, gate_change: Optional[torch.Tensor],
                     ve: Optional[torch.Tensor], a: torch.Tensor, contrast: torch.Tensor, source_start: int,
                     window: int) -> torch.Tensor:
    """fused_scores with dk / dv formed in the kernel (fused_kernel); other than even shapes: formed here, then the fast
    scorer."""
    m, p, _, h, d = projected.shape
    bs, tp, warps, stages = STATE['tiles']['fused']
    if q.shape[1] % 64 or p % tp or m % bs:
        R = STATE['installed']
        dk = R.norm_delta(r0[:, None], R.rotate(projected[:, :, 0], cos[:, None, None], sin[:, None, None]))
        dv = projected[:, :, 1] if gate_change is None else projected[:, :, 1] + gate_change[..., None] * ve[:, None]
        return future_scores(q, target, dk.permute(2, 1, 0, 3), dv.permute(2, 1, 0, 3), a, contrast, source_start,
                             window)
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    gated = gate_change is not None
    fused_kernel[(p // tp, m // bs, h)](
        q, target, projected.contiguous(), r0.contiguous(), cos, sin,
        gate_change.contiguous() if gated else out, ve.contiguous() if gated else out, a, contrast, out,
        q.shape[1], m, p, d, h, *cos.stride(), *sin.stride(), gated, *q.stride(), *target.stride(), *a.stride(),
        *contrast.stride(), source_start, window, bs, tp, 64, STATE['precision'], num_warps=warps, num_stages=stages,
        enable_fp_fusion=False)
    return out


@projected_scores.register_fake
def _(q, target, projected, r0, cos, sin, gate_change, ve, a, contrast, source_start, window):
    m, p, _, h, d = projected.shape
    return torch.empty((h, p, m), device=q.device, dtype=torch.float32)


def receiver_scores_fused(R):
    """R.receiver_scores with receiver_changes' rotate / norm_delta / gate step moved into the scorer's prologue
    (projected_scores): the projection, its TF32 GEMM and the gate's inputs are the record's code, unchanged."""
    norm_delta, rotate = R.norm_delta, R.rotate

    @torch.compile(dynamic=True)
    def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin,
                        q, target, a, contrast, start, window):
        move = move*coefficient
        m, p, width = move.shape
        heads, hd = kraw.shape[-2:]
        dn = norm_delta(x[:, None], move)
        projected = F.linear(dn, wkv).reshape(m, p, 2, heads, hd)
        r0 = rotate(kraw, cos[:, None], sin[:, None])
        gate_change = None
        if gate is not None:
            dg = F.linear(dn[..., :wg.shape[-1]], wg)
            gate_change = 2*((gate[:, None]+dg).sigmoid()-gate[:, None].sigmoid())
        return projected_scores(q, target, projected, r0, cos, sin, gate_change, ve, a, contrast, start,
                                window).sum(0).T
    return receiver_scores


def ops(R):  # (the record op, its fast CUDA implementation)
    return ((R.fused_scores, future_scores), (R.fused_key_response, key_response), (R.query_response, query_response))


def describe():
    """What install() set up (fast_kernels.json)."""
    mode = STATE['mode']
    return dict(mode=mode, tiles=None if mode in (None, 'record') else STATE['tiles'],
                precision=None if mode in (None, 'record') else dict(future=STATE['precision'], key=STATE['precision'],
                                                                    query=QUERY_PRECISION),
                fused=STATE['fused'], pin=PIN, file=str(Path(__file__).resolve()), sha1=SHA1,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                torch=torch.__version__, triton=triton.__version__)


def note_run():
    """In a torchrun process, rank 0 writes fast_kernels.json into the run's --output once the recipe has written
    config.json there (the recipe refuses a non-empty directory before that). Once per process."""
    if os.environ.get('RANK') != '0' or STATE['noted']:
        return
    STATE['noted'] = True
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=Path('logs/val'))  # the recipe's default
    output, begin = parser.parse_known_args(sys.argv[1:])[0].output, time.time()

    def write():
        config = output / 'config.json'
        for _ in range(14400):  # the recipe writes config.json right after its checks, before the compile warm-up
            if config.exists() and config.stat().st_mtime >= begin - 5:
                (output / 'fast_kernels.json').write_text(json.dumps(describe(), indent=1) + '\n')
                return
            time.sleep(0.25)
    threading.Thread(target=write, name='fast_kernels.json', daemon=True).start()


def install(R=None, tf32=False, only=None, fused=False, mode=None, note=True):
    """Register the kernels of `mode` on the record module's custom ops (process-wide; a later call switches modes):
    'fast' (default), 'strict' (the record's tiles), 'record' (the record's kernels and receiver_scores again).
    R.receiver_scores becomes the views (fused: the fused) version. tf32 / fused: rounding changes on top of fast.
    Other wrappers call this after their own install() (pins that read R.receiver_scores must see the record's);
    nothing else in the module is touched. only: a subset of the ops' names (checks). note: see note_run. Returns R."""
    R = sys.modules['recipe'] if R is None else R
    mode = mode or 'fast'
    if mode not in ('fast', 'strict', 'record') or (mode != 'fast' and (tf32 or fused)):
        raise ValueError(f'fast_kernels: mode {mode!r} (fast | strict | record); --tf32 / --fused go on top of fast only')
    if mode != 'record':
        device = int(os.environ.get('LOCAL_RANK', 0)) if torch.cuda.device_count() > 1 else 0
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError(f'fast_kernels: tuned for H100 (sm_90); {torch.cuda.get_device_name(device)} would not '
                               'fit its tiles (112 KB of shared memory): use the record entry')
    from torch._library.custom_ops import OPDEFS
    for op, impl in ops(R):
        if OPDEFS.get(op._qualname) is not op:  # a later load of the recipe re-defined the op under the same name
            raise RuntimeError(f'{op._qualname} belongs to another recipe module: install on the live one')
    if not hasattr(R, '_record_receiver_scores'):
        R._record_receiver_scores, R._fast_receiver_scores = R.receiver_scores, {}
    for op, impl in ops(R):
        if only is not None and impl.__name__ not in only:
            continue
        if mode != 'record':
            op.register_kernel('cuda', impl)
        elif 'cuda' in op._backend_fns:  # back to the record's own implementation
            op.register_kernel('cuda', op._init_fn)
    if mode == 'record':
        R.receiver_scores = R._record_receiver_scores
    elif only is None or 'future_scores' in only:  # receiver_scores is looked up by name in FiniteFuture.experiment
        kind = 'fused' if fused else 'views'
        if kind not in R._fast_receiver_scores:
            R._fast_receiver_scores[kind] = (receiver_scores_fused if fused else receiver_scores_views)(R)
        R.receiver_scores = R._fast_receiver_scores[kind]
    STATE.update(mode=mode, tiles=dict(TILES['strict' if mode == 'strict' else 'fast']),
                 precision='tf32' if tf32 else 'tf32x3', fused=fused, installed=R)
    if note:
        note_run()
    return R


# ------------------------------------------------------------------------------------------------- checks and timing


# ======================================== module fast_chain
"""Faster chain probes (Cut.chain_credits): the same experiments (every root, probe, block visit and head readout, the
same bases from the same generator, the same collectives in the same order) with all roots of a token chunk advancing
together and Triton kernels for the own query, the MLP route replay and the head.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

SHA1 = '3a171487dfac'  # sha1 of the research file this section was copied from
PIN = 'dd2cbf51a8b3'  # sha1 of the record code this replaces or reads (install() checks it; same in ~/noise30)
# the recipe's module functions the record's chain_credits calls (the fast path runs its own versions of them)
RELIED = ('chain_chunk', 'query_change_routed', 'norm_difference', 'routed_response', 'norm')
STATE = dict(installed=None, impl=dict(oq='triton', mlp='triton', head='triton'), chunk=None, pin=None, record=None,
             verified=None, calls={}, output=None, exact_blocks=None, exact_head=False, series_limit=0.5)
EPS = torch.finfo(torch.float32).eps


# ------------------------------------------------------------------------------------------------ record pieces


def unsupported(est):
    """Why the kernels do not cover est's chain (that call runs the record's chain_credits), or None. Covered: the
    record's settings and its research variants with own query, query_keys 64 / 128 / 256, a bf16 chain, width 512
    with head size 64, route_k a multiple of 256, a vocabulary a multiple of 128, at least 64 probes per GPU, the
    record's strong_keys / own_query_inputs and the record's module functions (as install() found them)."""
    s, cfg = est.s, est.model.config
    refs = STATE['record']
    width, probes = cfg.n_embd, est.width // est.split
    for ok, reason in (
            (s.own_query, 'own_query off'),
            (s.query_keys in (64, 128, 256), f'query_keys {s.query_keys} (the kernels: 64, 128 or 256)'),
            (est.capture.length >= s.query_keys, f'sequences shorter than query_keys {s.query_keys}'),
            (s.chain_precision == 'bf16', f'chain_precision {s.chain_precision}'),
            (width == 512 and width // cfg.n_head == 64,
             f'width {width}, head size {width // cfg.n_head} (the kernels: 512, 64)'),
            (s.route_k % MLP_UP[1] == 0, f'route_k {s.route_k} (the kernels: a multiple of {MLP_UP[1]})'),
            (cfg.vocab_size % HEAD[1] == 0, f'vocabulary {cfg.vocab_size} (the kernels: a multiple of {HEAD[1]})'),
            (probes >= OQ_PROJ[0], f'{probes} probes per GPU (proj_stats_kernel: at least {OQ_PROJ[0]} per token)'),
            (next(est.model.parameters()).is_cuda, 'not on CUDA'),
            (type(est).strong_keys is refs['strong_keys'], f"{type(est).__name__}.strong_keys is not the record's"),
            (type(est).own_query_inputs is refs['own_query_inputs'],
             f"{type(est).__name__}.own_query_inputs is not the record's")):
        if not ok:
            return reason
    R = STATE['installed']
    for name in RELIED:
        if getattr(R, name) is not refs[name]:
            return f'recipe.{name} was replaced after install'
    return None


def supported(est):
    return unsupported(est) is None


def unsupported_routes(routes):
    widths = sorted({r[0].shape[1] for r in routes})
    if any(k % MLP_UP[1] for k in widths):
        return f'MLP routes of {widths} units (the kernels: multiples of {MLP_UP[1]})'
    return None


def mismatch(model, mids, routes, width):
    """Why chain_credits' inputs break what the kernels rely on beyond the pin, or None (once per process, 8 tokens):
    the MLP replay reads the routes' clean pre-activations hs as norm(mid) W_fc^T on the routed units (Cut.estimate
    sets them so; the replay's identity N(mid + d) W - N(mid) W = inv' d W + (inv' / inv0 - 1) hs needs it)."""
    tok = torch.linspace(0, mids[0].shape[0] - 1, 8, device=mids[0].device).round().long()
    errors = []
    for b, (mid, (idx, hs)) in enumerate(zip(mids, routes)):
        ref = F.linear(F.rms_norm(mid[tok].float(), (width,)), model.transformer.h[b].mlp.c_fc.weight.float())
        ref = ref.gather(1, idx[tok])
        errors.append((hs[tok].float() - ref).abs().max() / ref.abs().max().clamp(min=1e-30))
    worst = torch.stack(errors).max().item()
    if worst <= 3e-3:  # the record's own TF32 GEMM vs this one: ~2e-6
        return None
    return f"the routes' clean pre-activations are not norm(mid) W_fc^T (relative difference {worst:.2e})"


def describe():
    """What install() set up and what the chain calls did (fast_chain.json)."""
    FK = sys.modules.get('fast_kernels')
    try:
        fk = FK.describe() if FK is not None and getattr(FK, 'STATE', {}).get('installed') is not None else None
    except Exception as error:  # fast_kernels changed its interface: say so, never stop a run for it
        fk = repr(error)
    return dict(installed=STATE['installed'] is not None, impl=STATE['impl'], chunk=STATE['chunk'], pin=PIN,
                file=str(Path(__file__).resolve()), sha1=SHA1, calls=dict(STATE['calls']), fast_kernels=fk,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, torch=torch.__version__,
                triton=triton.__version__)


def note(path):
    """Count the path a chain call took; the first time a path is taken rank 0 prints it; in a torchrun process
    rank 0 rewrites fast_chain.json then and every 50 calls (calls so far and the count of exact head blocks: one
    sync)."""
    calls = STATE['calls']
    new = path not in calls
    calls[path] = calls.get(path, 0) + 1
    if new and os.environ.get('RANK', '0') == '0':
        print('fast_chain: chain_credits runs ' + ('the fast kernels' if path == 'fast' else f'the record: {path}'),
              flush=True)
    output = STATE['output']
    if os.environ.get('RANK') == '0' and output is not None and (new or sum(calls.values()) % 50 == 0):
        if (output / 'config.json').exists():  # the recipe refuses a non-empty --output before it writes this
            info = describe()
            if STATE['exact_blocks'] is not None:
                info['exact_head_blocks'] = int(STATE['exact_blocks'].item())
            (output / 'fast_chain.json').write_text(json.dumps(info, indent=1) + '\n')


def query_routes(own, k, rest=False):
    """Cut.strong_keys (the record's formulas) returning the top-k key indices instead of gathered keys / values,
    the clean output o (all keys; the kernel forms the held rest o - sum_k p_k v_k itself), the clean rotated query
    projection r0 = rotate(norm(x) W_q^T) per head and its squared length. rest=True: also the record's rest (the
    torch reference path)."""
    pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
    length = len(pos)
    out = []
    for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
        heads, _, width = kk.shape
        c, s = cos[:, None], sin[:, None]
        r = (F.rms_norm(x, (x.shape[-1],)) @ wq.T).view(length, heads, width)
        r0 = torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                        r[..., :width // 2] * -s + r[..., width // 2:] * c], -1)
        q = F.rms_norm(r0, (width,))
        lag = pos[:, None] - pos[None]
        allowed = (lag >= 0) & (lag <= window)
        prob = torch.softmax((torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width ** -0.5)
                             .masked_fill(~allowed, float('-inf')), -1)
        p, idx = prob.topk(min(k, length), dim=-1)
        o = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
        item = dict(idx=idx.transpose(0, 1).to(torch.int32).contiguous(), p=p.transpose(0, 1).contiguous(),
                    o=o.transpose(0, 1).contiguous(), r0=r0.contiguous(), sq0=r0.square().sum(-1).contiguous(),
                    keys=kk.contiguous(), values=vv.contiguous())
        if rest:
            vtop = vv.unsqueeze(1).expand(-1, length, -1, -1).gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, width))
            item['rest'] = (o - torch.einsum('htk,htkd->htd', p, vtop.float())).transpose(0, 1).contiguous()
        del prob
        out.append(item)
    return out


# --------------------------------------------------------------------------------- reference (torch) block pieces


def oq_torch(R, delta, b, ctx):
    """The record's own-query step on a batch of rows (delta (m, n, W)), keys / values gathered here."""
    q = ctx['routes'][b]
    sl = ctx['local']
    idx = q['idx'][sl].long()  # (m, H, k)
    keys, values = q['keys'], q['values']  # (H, T, D)
    heads = keys.shape[0]
    hsel = torch.arange(heads, device=idx.device)[None, :, None]
    top_keys, top_values = keys[hsel, idx], values[hsel, idx]  # (m, H, k, D)
    own = ctx['own']
    return delta + R.query_change_routed(own[1][b][sl], delta, own[0][sl], own[2][b], top_keys, top_values,
                                         q['p'][sl], q['rest'][sl], own[5][b], own[6], own[7])


def mlp_torch(R, delta, b, ctx):
    """The record's routed MLP replay on a batch of rows."""
    mid, fc, wo, hs = ctx['mids'][b], ctx['fc'][b], ctx['wo'][b], ctx['hs'][b]
    dz = R.norm_difference(mid, delta)
    dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
    h = hs.unsqueeze(1)
    up, down = F.relu(h + dh), F.relu(h)
    return delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()


def head_torch(R, delta, ctx):
    """The record's head readout: (loss(moved) - loss(clean)) per row, before the division by the step."""
    xL, base, target_w, y, weight_t = ctx['xL'], ctx['base'], ctx['target_w'], ctx['y'], ctx['weight_t']
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return moved - clean


# ----------------------------------------------------------------------------------------------------- the driver


def chain_credits(self, e, x0, mids, outs, routes, y):
    """Cut.chain_credits with all roots of a token chunk advancing together (rows (root, probe) per token).

    Per token the rows are: root -1 (the embedding, P probes), then roots 0 .. L-1 (P probes, plus the 4 scalar
    probes on the pair's first GPU). At block b the rows of roots < b take the own-query step, root b's rows start,
    and the rows of roots <= b take the MLP step and then block b + 1's mixing scalars. The same experiments as the
    record, each row through the same sequence of steps; the bases are drawn in the record's order. A call the
    kernels do not cover runs the record's chain_credits (the reason is printed once and kept in fast_chain.json)."""
    R = STATE['installed']
    reason = unsupported(self) or unsupported_routes(routes)
    if reason is None:
        if STATE['verified'] is None:
            STATE['verified'] = mismatch(self.model, mids, routes, self.width) or 'ok'
        if STATE['verified'] != 'ok':
            reason = STATE['verified']
    if STATE['exact_blocks'] is None:
        STATE['exact_blocks'] = torch.zeros(1, device=self.device, dtype=torch.int32)
    note(reason or 'fast')
    if reason is not None:
        return R.Cut._record_chain_credits(self, e, x0, mids, outs, routes, y)
    impl = STATE['impl']
    model, s = self.model, self.s
    L = model.config.n_layer
    vocab = model.config.vocab_size
    W = self.width
    weight = model.lm_head.weight.float()[:vocab]
    lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
    mids = [z.float().contiguous() for z in mids]  # the kernels read rows of contiguous fp32 (no copy at the record's)
    xL = outs[-1].float().contiguous()
    base = xL @ weight.T
    target_w = weight[y.clamp(min=0)]
    n, h = e.shape[0], s.scalar_step
    lo, hi = self.share(W)
    P, scalars_here = hi - lo, self.part == 0
    bases, steps = [], []
    for site in range(-1, L):  # the record's draw order
        bases.append(self.basis()[lo:hi])
        probed = e if site < 0 else mids[site]
        steps.append(s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6))
    Ps = P + 4 * scalars_here  # rows per root s >= 0
    rows = P + L * Ps
    first = [0] + [P + site * Ps for site in range(L)]  # first row of each root (index site + 1)
    credit = [torch.empty(n, W, device=self.device) for _ in range(L + 1)]
    lam = [torch.zeros(2, device=self.device) for _ in range(L)]
    weight_t = weight.T.to(torch.bfloat16).contiguous()
    common = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1), lam_r=lam_r,
                  lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(), exact_head=STATE['exact_head'],
                  series_limit=STATE['series_limit'], exact_blocks=STATE['exact_blocks'])
    w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
    w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    w_q_t = [blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    w_proj_t = [blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    length = self.capture.length
    m_chunk = min(STATE['chunk'] or s.chain_tokens, length)  # default: the record's chain_tokens
    units = max(r[0].shape[1] for r in routes)
    buffers = dict(scratch=torch.empty(m_chunk * rows * max(W, units), device=self.device, dtype=torch.bfloat16),
                   scratch32=torch.empty(m_chunk * rows * W, device=self.device),
                   stats=torch.empty(m_chunk * rows * 2, device=self.device))
    delta_all = torch.empty(m_chunk * rows * W, device=self.device)
    idx32 = [r[0].to(torch.int32).contiguous() for r in routes]
    hs_all = [r[1].float().contiguous() for r in routes]
    for start in range(0, n, length):
        own = self.own_query_inputs(start // length)
        qroutes = query_routes(own, s.query_keys, rest=impl['oq'] == 'torch')
        inputs = [z.float().contiguous() for z in own[1]]
        seq = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=w_q_t, w_proj_t=w_proj_t, inputs=inputs,
                   sqx_seq=[z.square().sum(-1) for z in inputs])
        for t0 in range(start, start + length, m_chunk):
            t1 = min(t0 + m_chunk, start + length)
            sl, m = slice(t0, t1), t1 - t0
            local = slice(t0 - start, t1 - start)
            ctx = dict(common, **buffers, **seq, own=own, routes=qroutes, local=local, t0=t0 - start,
                       mids=[z[sl] for z in mids], hs=[z[sl] for z in hs_all], idx32=[z[sl] for z in idx32],
                       sqx_in=[z[local] for z in seq['sqx_seq']], xL=xL[sl], base=base[sl], target_w=target_w[sl],
                       y=y[sl], w_fc=w_fc, w_out_t=w_out_t)
            if impl['mlp'] == 'torch':
                ctx['fc'] = [w[r[0][sl]] for w, r in zip(w_fc, routes)]
                ctx['wo'] = [w[r[0][sl]] for w, r in zip(w_out_t, routes)]
            delta = delta_all[:m * rows * W].view(m, rows, W)
            dx0 = F.rms_norm(e[sl].unsqueeze(1) + steps[0][sl].unsqueeze(-1) * bases[0].unsqueeze(0), (W,)) \
                - x0[sl].unsqueeze(1)
            torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
            for b in range(L):
                n_oq = first[b + 1]  # rows of roots -1 .. b-1
                if impl['oq'] == 'torch':
                    delta[:, :n_oq] = oq_torch(R, delta[:, :n_oq], b, ctx)
                else:
                    oq_fast(delta, n_oq, b, ctx)
                r0 = n_oq  # root b starts at its block's mid stream
                torch.mul(steps[b + 1][sl].unsqueeze(-1), bases[b + 1].unsqueeze(0), out=delta[:, r0:r0 + P])
                if scalars_here:  # symmetric probes of both linear mixing scalars
                    prev = (x0 if b == 0 else outs[b - 1])[sl]
                    torch.mul(prev, h, out=delta[:, r0 + P])
                    torch.mul(x0[sl], h, out=delta[:, r0 + P + 1])
                    torch.mul(prev, -h, out=delta[:, r0 + P + 2])
                    torch.mul(x0[sl], -h, out=delta[:, r0 + P + 3])
                n_mlp = r0 + Ps
                if impl['mlp'] == 'torch':
                    delta[:, :n_mlp] = mlp_torch(R, delta[:, :n_mlp], b, ctx)
                    if b < L - 1:  # block b + 1's mixing scalars (the embedding root also re-enters through x0)
                        delta[:, :n_mlp] = lam_r[b + 1] * delta[:, :n_mlp]
                        delta[:, :P] = delta[:, :P] + lam_0[b + 1] * dx0
                else:
                    mlp_fast(delta, n_mlp, b, ctx, dx0 if b < L - 1 else None, P)
            change = head_torch(R, delta, ctx) if impl['head'] == 'torch' else head_fast(delta, ctx)
            change = change * (y[sl] >= 0).unsqueeze(1)
            for site in range(-1, L):
                r0 = first[site + 1]
                credit[site + 1][sl] = (change[:, r0:r0 + P] / steps[site + 1][sl]) @ bases[site + 1] / W
                if site >= 0 and scalars_here:  # (d(+h) + d(-h)) / 2 per scalar: the central difference over 2h
                    pair = change[:, r0 + P:r0 + P + 4] / h
                    lam[site] += (pair[:, :2] - pair[:, 2:]).sum(0) / 2
        del own, qroutes, seq
    del buffers, delta_all
    credits = {}
    for site in range(-1, L):  # the record's collective order
        credits['emb' if site < 0 else site] = self.group_sum(credit[site + 1])
        if site >= 0:
            credits['lam', site] = self.group_sum(lam[site])
    return credits


OQ_PROJ = (64, 256, 32, 4, 4)  # BM, BN, BK, warps, stages
OQ_ATTN = (64, 4, 1)  # BP, warps, stages
PROJ = (64, 128, 64, 4, 4)  # BM, BN, BK, warps, stages


@triton.jit
def proj_stats_kernel(D, X, SQX, WT, U, STATS, n, n_rows, D_ROWS, U_ROWS, m,
                      W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EPS: tl.constexpr):
    """u = d W^T (bf16 inputs, fp32 accumulation) for the n_rows first rows of every token (flattened rows: token
    i // n_rows, row i % n_rows), and per row inv' = rsqrt(|x + d|^2 / W + eps), coef = inv' / inv0 - 1 (x: the
    token's clean stream; from the same fp32 loads), so that N(x + d) W^T - N(x) W^T = inv' u + coef N(x) W^T.
    A block of BM rows spans at most two tokens (ta, tb): n_rows >= BM (oq_fast checks it)."""
    pn, pm = tl.program_id(0), tl.program_id(1)
    i = pm * BM + tl.arange(0, BM)
    ok = i < n
    tok, r = i // n_rows, i % n_rows
    ta = (pm * BM) // n_rows
    tb = tl.minimum((pm * BM + BM - 1) // n_rows, m - 1)
    first = (tok == ta)[:, None]
    drow = (tok * D_ROWS + r).to(tl.int64) * W
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    dotx = tl.zeros((BM,), tl.float32)
    sqd = tl.zeros((BM,), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(D + drow[:, None] + kk[None, :], ok[:, None], 0.)
        if pn == 0:
            x = tl.where(first, tl.load(X + ta * W + kk)[None, :], tl.load(X + tb * W + kk)[None, :])
            dotx += tl.sum(a * x, 1)
            sqd += tl.sum(a * a, 1)
        b = tl.load(WT + kk[:, None] * W + cols[None, :])
        acc = tl.dot(a.to(tl.bfloat16), b, acc)
    out = (tok * U_ROWS + r).to(tl.int64)
    tl.store(U + out[:, None] * W + cols[None, :], acc, ok[:, None])
    if pn == 0:
        sq = tl.where(tok == ta, tl.load(SQX + ta), tl.load(SQX + tb))
        inv0 = libdevice.rsqrt(sq / W + EPS)
        inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
        tl.store(STATS + out * 2, inv, ok)
        tl.store(STATS + out * 2 + 1, (inv - inv0) / inv0, ok)


@triton.jit
def oq_attn_kernel(U, STATS, R0, SQ0, CS, SN, KEYS, VALS, IDX, TP, O, Out, n_rows, U_ROWS, O_ROWS, t_start,
                   T: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, W: tl.constexpr, KK: tl.constexpr,
                   BP: tl.constexpr, EPS: tl.constexpr):
    """The own-query response of every row of one token in one head: from the projected move u (proj_stats_kernel),
    dr = inv' rotate(u) + coef r0, the per-head RMSNorm change dq = N(r0 + dr) - N(r0), then the record's routed
    response over the token's KK strongest clean keys (gathered once per program), the other keys' output held:
    with e = expm1(dq k / sqrt(DH)) and z = sum p e (= expm1(dlse)), weights p expm1(ds - dlse) = p (e - z) / (1 + z)
    and expm1(-dlse) = -z / (1 + z). bf16 out."""
    hh, t = tl.program_id(0), tl.program_id(1)
    pos = t_start + t
    base = pos * H + hh
    QH: tl.constexpr = DH // 2
    hd = tl.arange(0, QH)
    feat = tl.arange(0, DH)
    keys_n = tl.arange(0, KK)
    c = tl.load(CS + pos * QH + hd)[None, :]
    s = tl.load(SN + pos * QH + hd)[None, :]
    r0l = tl.load(R0 + base * DH + hd)[None, :]
    r0h = tl.load(R0 + base * DH + QH + hd)[None, :]
    sq0 = tl.load(SQ0 + base)
    qi0 = libdevice.rsqrt(sq0 / DH + EPS)
    kidx = tl.load(IDX + base * KK + keys_n).to(tl.int64)
    krow = (hh * T + kidx) * DH
    kl = tl.trans(tl.load(KEYS + krow[:, None] + hd[None, :]))
    kh = tl.trans(tl.load(KEYS + krow[:, None] + QH + hd[None, :]))
    v = tl.load(VALS + krow[:, None] + feat[None, :])
    p = tl.load(TP + base * KK + keys_n)
    rest = (tl.load(O + base * DH + feat) - tl.sum(p[:, None] * v.to(tl.float32), 0))[None, :]  # the held keys' output
    p = p[None, :]
    for r0 in range(0, n_rows, BP):
        rows = r0 + tl.arange(0, BP)
        ok = rows < n_rows
        row = (t * U_ROWS + rows).to(tl.int64)
        ub = U + row * W + hh * DH
        ul = tl.load(ub[:, None] + hd[None, :], ok[:, None], 0.)
        uh = tl.load(ub[:, None] + QH + hd[None, :], ok[:, None], 0.)
        inv = tl.load(STATS + row * 2, ok, 1.)[:, None]
        coef = tl.load(STATS + row * 2 + 1, ok, 0.)[:, None]
        drl = inv * (ul * c + uh * s) + coef * r0l
        drh = inv * (ul * -s + uh * c) + coef * r0h
        cross = tl.sum(r0l * drl, 1) + tl.sum(r0h * drh, 1)
        sqdr = tl.sum(drl * drl, 1) + tl.sum(drh * drh, 1)
        qi = libdevice.rsqrt((sq0 + 2 * cross + sqdr) / DH + EPS)[:, None]
        dql = (drl * qi + r0l * (qi - qi0)).to(tl.bfloat16)
        dqh = (drh * qi + r0h * (qi - qi0)).to(tl.bfloat16)
        ds = tl.dot(dqh, kh, tl.dot(dql, kl)) * (DH ** -.5)
        e = tl.exp(ds) - 1  # absolute error ~2^-22 (the record rounds ds itself to bf16 first: ~2^-9 relative)
        z = tl.sum(p * e, 1)[:, None]
        rz = 1 / (1 + z)
        weights = (p * (e - z) * rz).to(tl.bfloat16)
        moved = tl.dot(weights, v) - z * rz * rest
        tl.store(Out + (t * O_ROWS + rows).to(tl.int64)[:, None] * W + hh * DH + feat[None, :],
                 moved.to(tl.bfloat16), ok[:, None])


@triton.jit
def linear_residual_kernel(A, B, D, n, n_rows, A_ROWS, D_ROWS,
                           K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """d[i] += a[i] @ B for the n_rows first rows of every token (flattened: row i = token i // n_rows, row i % n_rows;
    A rows of A_ROWS per token, D rows of D_ROWS), bf16 inputs, fp32 accumulation, in place."""
    pn, pm = tl.program_id(0), tl.program_id(1)
    i = pm * BM + tl.arange(0, BM)
    ok = i < n
    tok, r = i // n_rows, i % n_rows
    arow = (tok * A_ROWS + r).to(tl.int64) * K
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + arow[:, None] + kk[None, :], ok[:, None], 0.)
        b = tl.load(B + kk[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    dptr = D + (tok * D_ROWS + r).to(tl.int64)[:, None] * N + cols[None, :]
    tl.store(dptr, tl.load(dptr, ok[:, None], 0.) + acc, ok[:, None])


def oq_fast(delta, n_oq, b, ctx):
    """delta[:, :n_oq] += the own-query response at block b: proj_stats_kernel (d W_q^T and the norm scalars),
    oq_attn_kernel (rotation, query norm, routed softmax response), then the c_proj GEMM and the add."""
    m, rows, W = delta.shape
    q = ctx['routes'][b]
    keys = q['keys']
    H, T, DH = keys.shape
    buf = ctx['scratch'][:m * rows * W].view(m, rows, W)
    u = ctx['scratch32'][:m * rows * W].view(m, rows, W)
    stats = ctx['stats'][:m * rows * 2]
    x = ctx['inputs'][b][ctx['local']]  # the block's clean input rows
    bm, bn, bk, warps, stages = OQ_PROJ
    if n_oq < bm:  # proj_stats_kernel reads the clean rows of at most two tokens per block (unsupported() keeps this)
        raise RuntimeError(f'fast_chain: {n_oq} own-query rows per token < proj_stats_kernel block {bm}')
    proj_stats_kernel[(W // bn, triton.cdiv(m * n_oq, bm))](
        delta, x, ctx['sqx_in'][b], ctx['w_q_t'][b], u, stats, m * n_oq, n_oq, rows, rows, m, W, bm, bn, bk, EPS,
        num_warps=warps, num_stages=stages)
    bp, warps, stages = OQ_ATTN
    oq_attn_kernel[(H, m)](
        u, stats, q['r0'], q['sq0'], ctx['cos'], ctx['sin'], keys, q['values'], q['idx'], q['p'], q['o'], buf,
        n_oq, rows, rows, ctx['t0'], T, H, DH, W, q['idx'].shape[-1], bp, EPS, num_warps=warps, num_stages=stages)
    bm, bn, bk, warps, stages = PROJ
    linear_residual_kernel[(W // bn, triton.cdiv(m * n_oq, bm))](
        buf, ctx['w_proj_t'][b], delta, m * n_oq, n_oq, rows, rows, W, W, bm, bn, bk,
        num_warps=warps, num_stages=stages)


MLP_UP = (128, 256, 64, 8, 3)  # BP, BN, BK, warps, stages
MLP_DOWN = (64, 128, 128, 4, 4)


@triton.jit
def mlp_up_kernel(D, X, IDX, WFC, HS, Out, n_rows, D_ROWS, O_ROWS,
                  W: tl.constexpr, K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  EPS: tl.constexpr):
    """The MLP route replay's first half for BP rows (probes) of one token and BN of its K routed units:
    dh = N(mid + d) W_fc^T - N(mid) W_fc^T on the units = inv' (d W_fc^T) + (inv' / inv0 - 1) h (h: the clean
    pre-activations), then relu(h + dh)^2 - relu(h)^2 (as (up - down)(up + down)), bf16 out. The units' weight rows
    are gathered on load; the row sums of d come from the same loads."""
    nb, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    units = nb * BN + tl.arange(0, BN)
    uidx = tl.load(IDX + t * K + units).to(tl.int64)
    acc = tl.zeros((BP, BN), tl.float32)
    dotx = tl.zeros((BP,), tl.float32)
    sqd = tl.zeros((BP,), tl.float32)
    sqx = tl.zeros((BK,), tl.float32)
    dbase = D + (t * D_ROWS + rows).to(tl.int64) * W
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
        x = tl.load(X + t * W + kk)
        dotx += tl.sum(d * x[None, :], 1)
        sqd += tl.sum(d * d, 1)
        sqx += x * x
        w = tl.load(WFC + uidx[None, :] * W + kk[:, None])
        acc = tl.dot(d.to(tl.bfloat16), w, acc)
    sq = tl.sum(sqx, 0)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    h = tl.load(HS + t * K + units)
    dh = inv[:, None] * acc + ((inv - inv0) / inv0)[:, None] * h[None, :]
    up = tl.maximum(h[None, :] + dh, 0.)
    down = tl.maximum(h, 0.)[None, :]
    r = (up - down) * (up + down)
    tl.store(Out + (t * O_ROWS + rows).to(tl.int64)[:, None] * K + units[None, :], r.to(tl.bfloat16), ok[:, None])


@triton.jit
def mlp_down_kernel(Rb, IDX, WOT, D, DX0, LAM, n_rows, D_ROWS, R_ROWS, n_x0,
                    W: tl.constexpr, K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SCALE: tl.constexpr, X0: tl.constexpr):
    """The replay's second half: d += r W_out^T on the routed units (rows gathered on load), then block b + 1's
    mixing scalars d = lam (d + change) (+ lam0 dx0 on the embedding root's rows), in place."""
    nb, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    cols = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BP, BN), tl.float32)
    rbase = Rb + (t * R_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, K, BK):
        uu = u0 + tl.arange(0, BK)
        a = tl.load(rbase[:, None] + uu[None, :], ok[:, None], 0.)
        uidx = tl.load(IDX + t * K + uu).to(tl.int64)
        w = tl.load(WOT + uidx[:, None] * W + cols[None, :])
        acc = tl.dot(a, w, acc)
    dptr = D + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if SCALE:
        new = tl.load(LAM) * new
        if X0:
            lam0 = tl.load(LAM + 1)
            mine = rows < n_x0
            x0d = tl.load(DX0 + (t * n_x0 + rows).to(tl.int64)[:, None] * W + cols[None, :], mine[:, None], 0.)
            new = tl.where(mine[:, None], new + lam0 * x0d, new)
    tl.store(dptr, new, ok[:, None])


def mlp_fast(delta, n_mlp, b, ctx, dx0, P):
    """delta[:, :n_mlp] += the routed MLP change at block b, then block b + 1's mixing scalars (dx0 None: last)."""
    m, rows, W = delta.shape
    idx, hs, mid = ctx['idx32'][b], ctx['hs'][b], ctx['mids'][b]
    K = idx.shape[1]
    buf = ctx['scratch'][:m * rows * K].view(m, rows, K)
    bp, bn, bk, warps, stages = MLP_UP
    mlp_up_kernel[(K // bn, triton.cdiv(n_mlp, bp), m)](
        delta, mid, idx, ctx['w_fc'][b], hs, buf, n_mlp, rows, rows, W, K, bp, bn, bk, EPS,
        num_warps=warps, num_stages=stages)
    bp, bn, bk, warps, stages = MLP_DOWN
    scale = dx0 is not None
    mlp_down_kernel[(W // bn, triton.cdiv(n_mlp, bp), m)](
        buf, idx, ctx['w_out_t'][b], delta, dx0 if scale else delta, ctx['lams'][min(b + 1, len(ctx['lams']) - 1)],
        n_mlp, rows, rows, P, W, K, bp, bn, bk, scale, scale, num_warps=warps, num_stages=stages)


HEAD_PREP = (32, 128, 4)  # BR, BK, warps
HEAD = (128, 128, 64, 8, 3)  # BM, BN, BK, warps, stages


@triton.jit
def head_prep_kernel(D, XL, TW, SQXL, BY, T0Y, MB, MW, A, STATS, n_rows, D_ROWS,
                     W: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, EPS: tl.constexpr):
    """Per row of one token: the bf16 copy of d (the logits GEMM's input, as the record's delta.to(bf16)), and from
    the fp32 d: inv' = rsqrt(|xL + d|^2 / W + eps), inv' - inv0, the target logit's capped change
    c(zy) - c(zy0) with zy = (base_y + d . w_y) inv' (fp32, as the record), c(z) = 15 tanh(z / 15), in difference
    form: 15 tanh(dz / 15) (1 - t0^2) / (1 + t0 tanh(dz / 15)), t0 = tanh(zy0 / 15); and a bound on every entry's
    |u| = |base (inv' - inv0) + (d_bf16 . w) inv'| / 15 <= (max|base| |inv' - inv0| + 1.01 |d| max|w| inv') / 15."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BR + tl.arange(0, BR)
    ok = rows < n_rows
    row = (t * D_ROWS + rows).to(tl.int64)
    dots = tl.zeros((BR,), tl.float32)
    sqd = tl.zeros((BR,), tl.float32)
    tdot = tl.zeros((BR,), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(D + row[:, None] * W + kk[None, :], ok[:, None], 0.)
        dots += tl.sum(d * tl.load(XL + t * W + kk)[None, :], 1)
        tdot += tl.sum(d * tl.load(TW + t * W + kk)[None, :], 1)
        sqd += tl.sum(d * d, 1)
        tl.store(A + row[:, None] * W + kk[None, :], d.to(tl.bfloat16), ok[:, None])
    sq = tl.load(SQXL + t)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dots + sqd) / W + EPS)
    dinv = inv - inv0
    by, t0 = tl.load(BY + t), tl.load(T0Y + t)
    dz = by * dinv + tdot * inv
    tb = libdevice.tanh(dz / 15)
    dc = 15 * tb * (1 - t0 * t0) / (1 + t0 * tb)
    bound = (tl.load(MB + t) * tl.abs(dinv) + 1.01 * libdevice.sqrt(sqd) * tl.load(MW) * inv) / 15
    tl.store(STATS + row * 4, inv, ok)
    tl.store(STATS + row * 4 + 1, dinv, ok)
    tl.store(STATS + row * 4 + 2, dc, ok)
    tl.store(STATS + row * 4 + 3, bound, ok)


@triton.jit
def head_kernel(A, WT, STATS, BASE, T0, P0, Part, n_rows, R,
                V: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                EXACT: tl.constexpr):
    """The logits change of BM rows of one token on BN vocabulary entries, delta_bf16 W_head^T with fp32 accumulation
    (the record rounds it to bf16), then each entry's capped logit change in difference form
    dc = 15 tanh(u) (1 - t0^2) / (1 + t0 tanh(u)), u = (base (inv' - inv0) + change inv') / 15, t0 = tanh(z0 / 15),
    and the row's partial sum over these entries of p0 expm1(dc) (p0: the clean capped softmax): the loss change is
    log1p(sum over all entries) - dc_target. tanh by its series (|u| < 0.5: head_prep_kernel bounds every row's |u|;
    measured |u| < 0.01) and expm1(dc) = exp(dc) - 1 (absolute error ~2^-22); head_fix_kernel redoes the row blocks
    whose bound reaches 0.5 with libdevice tanh / expm1. EXACT: libdevice everywhere (checks)."""
    pn, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_rows
    row = (t * R + rows).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.)
        b = tl.load(WT + kk[:, None] * V + cols[None, :])
        acc = tl.dot(a, b, acc)
    inv = tl.load(STATS + row * 4, ok, 1.)[:, None]
    dinv = tl.load(STATS + row * 4 + 1, ok, 0.)[:, None]
    base = tl.load(BASE + t * V + cols)[None, :]
    t0 = tl.load(T0 + t * V + cols)[None, :]
    p0 = tl.load(P0 + t * V + cols)[None, :]
    u = (base * dinv + acc * inv) * (1 / 15)
    if EXACT:
        th = libdevice.tanh(u)
        em = libdevice.expm1(15 * th * (1 - t0 * t0) / (1 + t0 * th))
    else:  # tanh by its series to u^15: relative error < 2e-8 for |u| < 0.5 (head_prep_kernel bounds |u| per row)
        u2 = u * u
        th = u * (1 + u2 * (-1 / 3 + u2 * (2 / 15 + u2 * (-17 / 315 + u2 * (62 / 2835 + u2 * (-1382 / 155925 + u2 * (
            21844 / 6081075 + u2 * (-929569 / 638512875))))))))
        em = tl.exp(tl.fdiv(15 * th * (1 - t0 * t0), 1 + t0 * th)) - 1
    part = tl.sum(p0 * em, 1)
    tl.store(Part + row * (V // BN) + pn, part, ok)


@triton.jit
def head_fix_kernel(A, WT, STATS, BASE, T0, P0, Part, Count, n_rows, R, limit,
                    V: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """After head_kernel, per token: every block of BM rows holding a row whose bound on |u| reaches limit (0.5, the
    tanh series' range) gets all its partial sums again with libdevice tanh / expm1 (head_kernel's EXACT arithmetic)
    and adds 1 to Count; the other blocks only read their bounds (never seen to trigger: largest bound 0.048)."""
    t = tl.program_id(0)
    for r0 in range(0, n_rows, BM):
        rows = r0 + tl.arange(0, BM)
        ok = rows < n_rows
        row = (t * R + rows).to(tl.int64)
        if tl.max(tl.load(STATS + row * 4 + 3, ok, 0.), 0) >= limit:
            tl.atomic_add(Count, 1)
            inv = tl.load(STATS + row * 4, ok, 1.)[:, None]
            dinv = tl.load(STATS + row * 4 + 1, ok, 0.)[:, None]
            for pn in range(0, V // BN):
                cols = pn * BN + tl.arange(0, BN)
                acc = tl.zeros((BM, BN), tl.float32)
                for k0 in range(0, W, BK):
                    kk = k0 + tl.arange(0, BK)
                    a = tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.)
                    b = tl.load(WT + kk[:, None] * V + cols[None, :])
                    acc = tl.dot(a, b, acc)
                base = tl.load(BASE + t * V + cols)[None, :]
                t0 = tl.load(T0 + t * V + cols)[None, :]
                p0 = tl.load(P0 + t * V + cols)[None, :]
                u = (base * dinv + acc * inv) * (1 / 15)
                th = libdevice.tanh(u)
                em = libdevice.expm1(15 * th * (1 - t0 * t0) / (1 + t0 * th))
                tl.store(Part + row * (V // BN) + pn, tl.sum(p0 * em, 1), ok)


def head_fast(delta, ctx):
    """(loss(moved) - loss(clean)) per row of delta (m, R, W): head_prep_kernel, head_kernel, then
    log1p(sum of the partials) - the target's capped logit change."""
    m, rows, W = delta.shape
    weight_t = ctx['weight_t']
    V = weight_t.shape[1]
    xL, base, y = ctx['xL'], ctx['base'], ctx['y']
    sq = xL.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / W + EPS)
    z0 = base * inv0
    t0 = torch.tanh(z0 / 15)
    p0 = torch.softmax(15 * t0, -1)
    target = y.clamp(min=0).unsqueeze(1)
    a = ctx['scratch'][:m * rows * W].view(m, rows, W)
    stats = ctx['scratch32'][:m * rows * 4]
    br, bk, warps = HEAD_PREP
    head_prep_kernel[(triton.cdiv(rows, br), m)](
        delta, xL, ctx['target_w'], sq, base.gather(1, target).contiguous(), t0.gather(1, target).contiguous(),
        base.abs().amax(-1).contiguous(), ctx['head_row_norm'], a, stats, rows, rows, W, br, bk, EPS, num_warps=warps)
    stats = stats.view(m, rows, 4)
    bm, bn, bk, warps, stages = HEAD
    part = torch.empty(m * rows, V // bn, device=delta.device)
    head_kernel[(V // bn, triton.cdiv(rows, bm), m)](
        a, weight_t, stats, base, t0, p0, part, rows, rows, V, W, bm, bn, bk, ctx['exact_head'],
        num_warps=warps, num_stages=stages)
    if not ctx['exact_head']:  # row blocks outside the tanh series' range (if any): libdevice instead
        head_fix_kernel[(m,)](a, weight_t, stats, base, t0, p0, part, ctx['exact_blocks'], rows, rows,
                              ctx['series_limit'], V, W, bm, bn, bk, num_warps=warps, num_stages=1)
    return torch.log1p(part.sum(-1)).view(m, rows) - stats[..., 2]


# ---------------------------------------------------------------------------------------------------------- install


def install(R=None, impl='triton', chunk=None, check_pin=True):
    """Swap Cut.chain_credits of the loaded recipe module (process-wide; idempotent). impl: 'triton' (fast),
    'torch' (the record's block functions on the batched rows: a bookkeeping reference) or a dict per piece
    (oq / mlp / head: 'triton' | 'torch'). Returns R."""
    impl = dict.fromkeys(('oq', 'mlp', 'head'), impl) if isinstance(impl, str) else dict(impl)
    R = sys.modules['recipe'] if R is None else R
    if getattr(R.chain_chunk, '_torchdynamo_orig_callable', R.chain_chunk) is not R.chain_chunk_eager:
        raise RuntimeError("recipe.chain_chunk is not the record's compiled chain_chunk_eager (replaced by a wrapper?): "
                           "fast_chain would not run the record chain's experiments")
    if not hasattr(R.Cut, '_record_chain_credits'):
        R.Cut._record_chain_credits = R.Cut.chain_credits
    R.Cut.chain_credits = chain_credits
    STATE.update(installed=R, impl=impl, chunk=chunk, pin=PIN, verified=None,
                 record=dict({name: getattr(R, name) for name in RELIED}, strong_keys=R.Cut.strong_keys,
                             own_query_inputs=R.Cut.own_query_inputs))
    if os.environ.get('RANK') is not None:  # torchrun: fast_chain.json goes into the run's --output
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', type=Path, default=Path('logs/val'))  # the recipe's default
        STATE['output'] = parser.parse_known_args(sys.argv[1:])[0].output
    return R


# ------------------------------------------------------------------------------------------------- checks and timing


# ======================================== module fast_moments
"""Moment-table scoring. A key move at source t changes later queries' attention in one column; the receivers s > t are
summed through the second-order expansion in the logit change, which leaves per-(t, head) tables shared by every
experiment of t (s = t exact). Used by the input-K / raw-K experiments ('key') and the future receivers ('future').
A running float64 check every 50 calls switches the process to the exact path if the expansion drifts.
"""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


FILE = Path(__file__).resolve()  # (run as a script, __main__ loses __file__ once its body returns: atexit runs later)
SHA1 = '5c673d28cfab'  # sha1 of the research file this section was copied from
PIN = 'b16253ae8a6e'  # sha1 of the record code the key part copies or relies on (pin_of); install() refuses another
FUTURE_PIN = 'cd9d537a5acc'  # the same for the future part (future_pin_of: the record and fast_kernels pieces)
PARTS = ('key', 'future')  # the parts this file implements (FAST_MOMENTS=key or key,future)
CHECK_EVERY, FALLBACK_REL, NOTE_EVERY = 50, 1e-3, 50
FUTURE_STRIDE, FUTURE_CHECK_TOKENS = 11, 16  # the future running check's call rotation and its float64 source tokens
EPS = torch.finfo(torch.float32).eps
BQ, BP, BPF = 64, 64, 64  # builder query tile; input-K directions per program; future trials per program
# n: key_input_credits calls of this process (the running check's clock); calls: per path ('moments', 'fallback',
# 'record: <why>'); unsupported: per reason; previous: the resumed segment's fast_moments.json (fallback, checks, calls)
STATE = dict(installed=None, parts=(), n=0, calls={}, checks=[], fallback=False, output=None, record=None, shared=None,
             unsupported={}, previous=None, check_every=CHECK_EVERY, fallback_rel=FALLBACK_REL, exit_hook=False,
             future_calls={}, future_unsupported={}, prepares=0, future_check=None, future_counts={}, future_last={},
             previous_scorer=None, future_compiled=None)
# future: future_calls / future_unsupported as calls / unsupported, per receiver call (R.receiver_scores); prepares:
# FiniteFuture.prepare calls (one per sequence and Cut step: the future running check's clock); future_check: k while the
# current sequence is the k-th checked one, else None; future_counts / future_last: per receiver block, the calls the
# moment path covers in the current / previous sequence (the checked call's ordinal / N_j); previous_scorer: the
# R.receiver_scores installed before (fast_kernels' fused), run where the moment path does not apply
LOOKUP = {}  # id(prepare's ctx['q']) -> weakref to its block's Tables (the scorer is called with q = ctx['q'])


# --------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def build_kernel(Q, A, C, U, M1, T, Window, QH, QT, AH, AS, AT, CH, CS, CT,
                 H: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr):
    """Sources t0 = 2 program_id(0) and t0 + 1, head h: u = sum a c q_s and M1 = sum a c (1/2 - a) q_s q_s^T over the
    queries t < s <= t + Window (s < T), the two sharing each q tile (per source the same tiles in the same order: a
    zero tile more at most). Q [H, T, D]; A, C [H, T(query), T(source)] with any strides; U [T, H, D] fp32, M1 [T, H, D,
    D] bf16."""
    t0, h = tl.program_id(0) * 2, tl.program_id(1)
    d = tl.arange(0, D)
    u0 = tl.zeros((D,), tl.float32)
    u1 = tl.zeros((D,), tl.float32)
    acc0 = tl.zeros((D, D), tl.float32)
    acc1 = tl.zeros((D, D), tl.float32)
    lo = t0 + 1
    hi0 = tl.minimum(T, t0 + Window + 1)
    hi1 = tl.minimum(T, t0 + Window + 2)
    for s0 in range(lo // BQ * BQ, hi1, BQ):
        s = s0 + tl.arange(0, BQ)
        ok0 = (s >= lo) & (s < hi0)
        ok1 = (s >= lo + 1) & (s < hi1)
        q = tl.load(Q + h * QH + s[:, None] * QT + d[None, :], (ok0 | ok1)[:, None], 0.)
        a0 = tl.load(A + h * AH + s * AS + t0 * AT, ok0, 0.)
        c0 = tl.load(C + h * CH + s * CS + t0 * CT, ok0, 0.)
        a1 = tl.load(A + h * AH + s * AS + (t0 + 1) * AT, ok1, 0.)
        c1 = tl.load(C + h * CH + s * CS + (t0 + 1) * CT, ok1, 0.)
        qb = q.to(tl.bfloat16)
        ac0 = a0 * c0
        ac1 = a1 * c1
        u0 += tl.sum(q * ac0[:, None], 0)
        u1 += tl.sum(q * ac1[:, None], 0)
        acc0 = tl.dot(tl.trans((q * (ac0 * (0.5 - a0))[:, None]).to(tl.bfloat16)), qb, acc0)
        acc1 = tl.dot(tl.trans((q * (ac1 * (0.5 - a1))[:, None]).to(tl.bfloat16)), qb, acc1)
    row = t0 * H + h
    tl.store(U + row * D + d, u0)
    tl.store(M1 + row * D * D + d[:, None] * D + d[None, :], acc0.to(tl.bfloat16))
    row = (t0 + 1) * H + h
    tl.store(U + row * D + d, u1, t0 + 1 < T)
    tl.store(M1 + row * D * D + d[:, None] * D + d[None, :], acc1.to(tl.bfloat16), t0 + 1 < T)


@triton.jit
def inputk_kernel(PX, PR, DOTS, SQ, STEPS, RSQ, KRAW, CS, SN, U, M1, Q, A, C, Out, P, KH, KT, QH, QT, AH, AD, CH, CD,
                  H: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr, BP: tl.constexpr):
    """One (source t, BP directions of this GPU's P): the + and - experiments' scores summed over heads, Out [T, 2P]
    (+ then -). The move is prepared_key_moves' finite identity, formed here: PX [T, WIDTH] = x W_k^T, PR [P, WIDTH] =
    rows W_k^T, DOTS [T, P] = x . rows, SQ [T] = |x|^2, STEPS [T], RSQ [P] = |row|^2; per head the rotation (CS / SN
    [T, D/2], linear, so x W and h W are rotated apart) and the key norm difference against r0 = rot(KRAW[h, t]). Score:
    the tables (queries s > t) + the exact s = t term (Q [H, T, D] = qp, A / C diagonals at AD / CD strides)."""
    t, pb = tl.program_id(0), tl.program_id(1)
    p = pb * BP + tl.arange(0, BP)
    pm = p < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    cs = tl.load(CS + t * (D // 2) + half)
    sn = tl.load(SN + t * (D // 2) + half)
    sq = tl.load(SQ + t)
    step = tl.load(STEPS + t)
    dots = tl.load(DOTS + t * P + p, pm, 0.)
    rsq = step * step * tl.load(RSQ + p, pm, 0.)
    inv0 = libdevice.rsqrt(sq / WIDTH + 1.1920928955078125e-07)
    inv_up = libdevice.rsqrt((sq + rsq + 2 * step * dots) / WIDTH + 1.1920928955078125e-07)
    inv_dn = libdevice.rsqrt((sq + rsq - 2 * step * dots) / WIDTH + 1.1920928955078125e-07)
    # inv' - inv0 without cancellation: -(S' - S0) (inv0 inv')^2 / (inv0 + inv'), S = |.|^2 / WIDTH + eps
    d_up = -((rsq + 2 * step * dots) / WIDTH) * (inv0 * inv_up) * (inv0 * inv_up) / (inv0 + inv_up)
    d_dn = -((rsq - 2 * step * dots) / WIDTH) * (inv0 * inv_dn) * (inv0 * inv_dn) / (inv0 + inv_dn)
    up = tl.zeros((BP,), tl.float32)
    dn = tl.zeros((BP,), tl.float32)
    for h in range(H):
        px = tl.load(PX + t * WIDTH + h * D + f)
        pxo = tl.load(PX + t * WIDTH + h * D + other)
        rpx = tl.where(low, px * cs + pxo * sn, -pxo * sn + px * cs)
        pr = tl.load(PR + p[:, None] * WIDTH + h * D + f[None, :], pm[:, None], 0.)
        pro = tl.load(PR + p[:, None] * WIDTH + h * D + other[None, :], pm[:, None], 0.)
        rpr = tl.where(low[None, :], pr * cs[None, :] + pro * sn[None, :], -pro * sn[None, :] + pr * cs[None, :])
        kr = tl.load(KRAW + h * KH + t * KT + f)
        kro = tl.load(KRAW + h * KH + t * KT + other)
        r0 = tl.where(low, kr * cs + kro * sn, -kro * sn + kr * cs)
        r0sq = tl.sum(r0 * r0, 0)
        k0 = libdevice.rsqrt(r0sq / D + 1.1920928955078125e-07)
        u = tl.load(U + (t * H + h) * D + f)
        m1 = tl.load(M1 + (t * H + h) * D * D + f[:, None] * D + f[None, :])
        qs = tl.load(Q + h * QH + t * QT + f)
        a_self = tl.load(A + h * AH + t * AD)
        c_self = tl.load(C + h * CH + t * CD)
        for sign in tl.static_range(2):
            inv = inv_up if sign == 0 else inv_dn
            dinv = d_up if sign == 0 else d_dn
            dr = dinv[:, None] * rpx[None, :] + ((1. - 2. * sign) * (step * inv))[:, None] * rpr
            grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
            k1 = libdevice.rsqrt(r0sq / D + grow + 1.1920928955078125e-07)
            dk = dr * k1[:, None] - r0[None, :] * (grow * (k0 * k1) * (k0 * k1) / (k0 + k1))[:, None]
            y = tl.dot(dk.to(tl.bfloat16), m1)
            change = a_self * libdevice.expm1(tl.sum(dk * qs[None, :], 1) * (D ** -.5))
            score = (tl.sum(dk * u[None, :], 1) * (D ** -.5) + tl.sum(y * dk, 1) * (1. / D)
                     + change / (1 + change) * c_self)
            if sign == 0:
                up += score
            else:
                dn += score
    tl.store(Out + t * 2 * P + p, up, pm)
    tl.store(Out + t * 2 * P + P + p, dn, pm)


@triton.jit
def rawk_kernel(DK, STEPS, U, M1, Q, A, C, Out, T0, Lo, J, KH, KJ, KM, KD, SH, ST, QH, QT, AH, AD, CH, CD, OH, OT,
                H: tl.constexpr, D: tl.constexpr, BJ: tl.constexpr):
    """One (source t = T0 + program_id(0), head h): its raw-K experiments j < J (this GPU's dims Lo + j), the moves
    DK [H, J, M, D] (any strides) being R.raw_key_moves of the KEY_BUCKET from T0 (the record's own moves), scored by the
    tables + the exact s = t term and divided by the step: Out[h, t, Lo + j] (strides OH, OT)."""
    m, h = tl.program_id(0), tl.program_id(1)
    t = T0 + m
    j = tl.arange(0, BJ)
    jm = j < J
    f = tl.arange(0, D)
    dk = tl.load(DK + h * KH + j[:, None] * KJ + m * KM + f[None, :] * KD, jm[:, None], 0.)
    u = tl.load(U + (t * H + h) * D + f)
    m1 = tl.load(M1 + (t * H + h) * D * D + f[:, None] * D + f[None, :])
    qs = tl.load(Q + h * QH + t * QT + f)
    a_self = tl.load(A + h * AH + t * AD)
    c_self = tl.load(C + h * CH + t * CD)
    step = tl.load(STEPS + h * SH + t * ST)
    y = tl.dot(dk.to(tl.bfloat16), m1)
    change = a_self * libdevice.expm1(tl.sum(dk * qs[None, :], 1) * (D ** -.5))
    score = tl.sum(dk * u[None, :], 1) * (D ** -.5) + tl.sum(y * dk, 1) * (1. / D) + change / (1 + change) * c_self
    tl.store(Out + h * OH + t * OT + Lo + j, score / step, jm)


@triton.jit
def exact_kernel(Q, DK, A, C, Out, Xmax, T, M, P, SourceStart, QueryStart, Window,
                 QH, QT, KH, KP, KM, KD, AH, AT, AM, CH, CT, CM, D: tl.constexpr, BC: tl.constexpr, BQ: tl.constexpr):
    """The record's key_response (finite_attention_kernels: tf32x3 dots, the same query range and order per tile of BC
    columns) that also returns, per column, max |x| over the pairs inside the window (query >= source, lag <= Window).
    Q [H, T(queries from QueryStart), D]; DK [H, P, M, D] any strides; A, C [H, T, M]. The running check's exact path."""
    head = tl.program_id(1)
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    ok = col < P * M
    trial = col // M
    source = col % M
    f = tl.arange(0, D)
    changed = tl.load(DK + head * KH + trial[None, :] * KP + source[None, :] * KM + f[:, None] * KD, ok[None, :], 0)
    first = tl.maximum(0, SourceStart + tl.min(source, 0) - QueryStart)
    last = tl.minimum(T, SourceStart + tl.max(source, 0) + Window + 1 - QueryStart)
    first = first // BQ * BQ
    result = tl.full((BC,), 0, tl.float32)
    largest = tl.full((BC,), 0, tl.float32)
    for start in range(first, last, BQ):
        query = start + tl.arange(0, BQ)
        q = tl.load(Q + head * QH + query[:, None] * QT + f[None, :], query[:, None] < T, 0)
        delta = tl.dot(q, changed, input_precision='tf32x3') * (D ** -.5)
        valid = (query[:, None] < T) & ok[None, :]
        a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
        c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
        change = a * libdevice.expm1(delta)
        result += tl.sum(change / (1 + change) * c, 0)
        lag = (QueryStart + query)[:, None] - (SourceStart + source)[None, :]
        inside = valid & (lag >= 0) & (lag <= Window)
        largest = tl.maximum(largest, tl.max(tl.where(inside, tl.abs(delta), 0.), 0))
    tl.store(Out + head * P * M + col, result, ok)
    tl.store(Xmax + head * P * M + col, largest, ok)


@triton.jit
def future_build_kernel(Q, G, A, C, U, WV, M1, M2, T, Window, QH, QT, GH, GT, AH, AS, AT, CH, CS, CT,
                        H: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr):
    """Future receivers (FAST_MOMENTS future), source t = program_id(0), head h, over the queries t < s <= t + Window
    (s < T; the record's future mask): u = sum a c q_s, w = sum a g_s (fp32), M1 = sum a c (1/2 - a) q_s q_s^T, M2 =
    sum a (1 - a) q_s g_s^T (bf16 tensor cores, fp32 accumulation, stored bf16). Q, G [H, T, D] (prepare's q and
    target: strides QH / QT, GH / GT, the last 1); A, C [H, T(query), T(source)] (prepare's a and contrast) any
    strides. U, WV [T, H, D]; M1, M2 [T, H, D, D]."""
    t, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    u = tl.zeros((D,), tl.float32)
    w = tl.zeros((D,), tl.float32)
    acc1 = tl.zeros((D, D), tl.float32)
    acc2 = tl.zeros((D, D), tl.float32)
    lo = t + 1
    hi = tl.minimum(T, t + Window + 1)
    for s0 in range(lo // BQ * BQ, hi, BQ):
        s = s0 + tl.arange(0, BQ)
        ok = (s >= lo) & (s < hi)
        q = tl.load(Q + h * QH + s[:, None] * QT + d[None, :], ok[:, None], 0.)
        g = tl.load(G + h * GH + s[:, None] * GT + d[None, :], ok[:, None], 0.)
        a = tl.load(A + h * AH + s * AS + t * AT, ok, 0.)
        c = tl.load(C + h * CH + s * CS + t * CT, ok, 0.)
        ac = a * c
        u += tl.sum(q * ac[:, None], 0)
        w += tl.sum(g * a[:, None], 0)
        acc1 = tl.dot(tl.trans((q * (ac * (0.5 - a))[:, None]).to(tl.bfloat16)), q.to(tl.bfloat16), acc1)
        acc2 = tl.dot(tl.trans((q * (a * (1. - a))[:, None]).to(tl.bfloat16)), g.to(tl.bfloat16), acc2)
    row = t * H + h
    tl.store(U + row * D + d, u)
    tl.store(WV + row * D + d, w)
    tl.store(M1 + row * D * D + d[:, None] * D + d[None, :], acc1.to(tl.bfloat16))
    tl.store(M2 + row * D * D + d[:, None] * D + d[None, :], acc2.to(tl.bfloat16))


@triton.jit
def future_kernel(PJ, R0, CS, SN, GC, VE, U, WV, M1, M2, Out, Start, P, CSM, CSF, SNM, SNF,
                  H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, BP: tl.constexpr):
    """One (source t of the token chunk, BP of the P trials): the future receiver score summed over heads, Out [m, P].
    Prologue from fast_kernels.fused_kernel (fp32, the record's formulas): from the projection PJ [m, P, 2, H, D]
    (F.linear(dn, wkv)), the rotated clean key R0 [m, H, D] and the rope rows CS / SN [m, D/2] (strides), dk =
    norm_delta(R0, rotate(PJ[.., 0, ..])), dv = PJ[.., 1, ..] (+ GC [m, P, H] * VE [m, H, D]); the key norm's inverse
    difference written without cancellation (as the input-K kernel does: fused_kernel's moved - inverse loses up to ~3
    digits on small moves, 1.7e-4 - 3.5e-4 of a receiver's scores at blocks 3 / 7, the record's too). Score (the record's
    exact single-column update expanded to second order in x = q . dk / 8 and dv): u . dk / 8 + w . dv + dk^T M1 dk /
    64 + dk^T M2 dv / 8, the tables at row Start + t (prepare's queries after t within the window)."""
    t, pb = tl.program_id(0), tl.program_id(1)
    p = pb * BP + tl.arange(0, BP)
    pm = p < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    cs = tl.load(CS + t * CSM + half * CSF)
    sn = tl.load(SN + t * SNM + half * SNF)
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        base = ((t * P + p[:, None]) * 2 * H + h) * D
        pk = tl.load(PJ + base + f[None, :], pm[:, None], 0.)
        pk_other = tl.load(PJ + base + other[None, :], pm[:, None], 0.)
        dr = tl.where(low[None, :], pk * cs[None, :] + pk_other * sn[None, :], -pk_other * sn[None, :] + pk * cs[None, :])
        r0 = tl.load(R0 + (t * H + h) * D + f)
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        # norm_delta(r0, dr) = dr moved + r0 (moved - inverse), the difference without cancellation:
        # moved - inverse = -grow (inverse moved)^2 / (inverse + moved)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        dv = tl.load(PJ + base + H * D + f[None, :], pm[:, None], 0.)
        if GATED:
            gc = tl.load(GC + (t * P + p) * H + h, pm, 0.)
            dv = dv + gc[:, None] * tl.load(VE + (t * H + h) * D + f)[None, :]
        row = (Start + t) * H + h
        u = tl.load(U + row * D + f)
        w = tl.load(WV + row * D + f)
        m1 = tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])
        m2 = tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])
        dkb = dk.to(tl.bfloat16)
        y1 = tl.dot(dkb, m1)
        y2 = tl.dot(dkb, m2)
        total += (tl.sum(dk * u[None, :], 1) * (D ** -.5) + tl.sum(dv * w[None, :], 1)
                  + tl.sum(y1 * dk, 1) * (1. / D) + tl.sum(y2 * dv, 1) * (D ** -.5))
    tl.store(Out + t * P + p, total, pm)


# ----------------------------------------------------------------------------------------------------- launchers


def build_tables(qp, A, C, window):
    """Per (source t, head h) over the queries t < s <= t + window: U [T, H, D] fp32, M1 [T, H, D, D] bf16."""
    h, t, d = qp.shape
    U = torch.empty((t, h, d), device=qp.device, dtype=torch.float32)
    M1 = torch.empty((t, h, d, d), device=qp.device, dtype=torch.bfloat16)
    build_kernel[(triton.cdiv(t, 2), h)](qp, A, C, U, M1, t, window, qp.stride(0), qp.stride(1), *A.stride(),
                                         *C.stride(), h, d, BQ, num_warps=4, num_stages=2)
    return U, M1


def inputk_scores(xin, steps, rows, projected_x, projected_rows, raw, cos, sin, qp, A, C, tables):
    """[T, 2P] (+ then -) input-K experiment scores of one sequence and block, summed over heads (record:
    fused_key_response on prepared_key_moves, per head). dots = x . rows as prepared_key_moves forms them."""
    U, M1 = tables
    t, width = xin.shape
    p = rows.shape[0]
    h, _, d = raw.shape
    dots = (xin @ rows.T).contiguous()
    out = torch.empty((t, 2 * p), device=xin.device, dtype=torch.float32)
    inputk_kernel[(t, triton.cdiv(p, BP))](
        projected_x.contiguous(), projected_rows.contiguous(), dots, xin.square().sum(-1), steps.reshape(-1).contiguous(),
        rows.square().sum(-1).contiguous(), raw, cos.contiguous(), sin.contiguous(), U, M1, qp, A, C, out, p,
        raw.stride(0), raw.stride(1), qp.stride(0), qp.stride(1), A.stride(0), A.stride(1) + A.stride(2), C.stride(0),
        C.stride(1) + C.stride(2), h, d, width, BP, num_warps=4, num_stages=1)
    return out


def rawk_credit(kp, raw, steps, cos, sin, qp, A, C, tables, lo, hi):
    """Boost.key_errors_from_context's result [h, T, hd] (this GPU's dims lo:hi; zeros elsewhere): the record's moves,
    R.raw_key_moves per KEY_BUCKET with the record's arguments (the same compiled graph on the same inputs: bitwise its
    moves), scored by the tables."""
    R = STATE['installed'] or sys.modules['recipe']
    U, M1 = tables
    h, t, d = raw.shape
    result = torch.zeros_like(raw)
    if hi <= lo:
        return result
    eye = torch.eye(d, device=raw.device)
    for t0 in range(0, t, R.KEY_BUCKET):
        source = slice(t0, min(t0 + R.KEY_BUCKET, t))
        moves = R.raw_key_moves(kp[:, source], raw[:, source], steps[:, source], cos[source], sin[source], eye[lo:hi])
        rawk_kernel[(moves.shape[2], h)](
            moves, steps, U, M1, qp, A, C, result, t0, lo, hi - lo, *moves.stride(), steps.stride(0), steps.stride(1),
            qp.stride(0), qp.stride(1), A.stride(0), A.stride(1) + A.stride(2), C.stride(0), C.stride(1) + C.stride(2),
            result.stride(0), result.stride(1), h, d, max(16, triton.next_power_of_2(hi - lo)), num_warps=4,
            num_stages=1)
    return result


def build_future_tables(q, g, a, contrast, window):
    """Per (source t, head h) over the queries t < s <= t + window, from prepare's q / target / a / contrast of one
    block: U, W [T, H, D] fp32, M1, M2 [T, H, D, D] bf16 (future_build_kernel; a / contrast read through transposed
    copies, so each source's column is contiguous)."""
    h, t, d = q.shape
    at, ct = a.transpose(1, 2).contiguous().transpose(1, 2), contrast.transpose(1, 2).contiguous().transpose(1, 2)
    U = torch.empty((t, h, d), device=q.device, dtype=torch.float32)
    W = torch.empty((t, h, d), device=q.device, dtype=torch.float32)
    M1 = torch.empty((t, h, d, d), device=q.device, dtype=torch.bfloat16)
    M2 = torch.empty((t, h, d, d), device=q.device, dtype=torch.bfloat16)
    future_build_kernel[(t, h)](q, g, at, ct, U, W, M1, M2, t, window, q.stride(0), q.stride(1), g.stride(0),
                                g.stride(1), *at.stride(), *ct.stride(), h, d, BQ, num_warps=4, num_stages=2)
    return U, W, M1, M2


def launch_future(projected, r0, cos, sin, gate_change, ve, U, W, M1, M2, start):
    """[m, P] future receiver scores summed over heads (the record: fused_scores(...).sum(0).T) from the moment tables
    of the receiver block (future_kernel), launched directly (no custom op: a receiver call's CPU cost is what bounds
    the one-sided future loop)."""
    m, p, _, h, d = projected.shape
    out = torch.empty((m, p), device=projected.device, dtype=torch.float32)
    gated = gate_change is not None
    future_kernel[(m, triton.cdiv(p, BPF))](
        projected.contiguous(), r0.contiguous(), cos, sin, gate_change.contiguous() if gated else out,
        ve.contiguous() if gated else out, U, W, M1, M2, out, start, p, *cos.stride(), *sin.stride(), h, d, gated,
        BPF, num_warps=4, num_stages=2)
    return out


def future_prologue(R):
    """The future scorer's compiled part: fast_kernels.receiver_scores_fused's code (copied: the move's coefficient,
    norm_delta, the K / V projection, the clean rotated key, the gate step) up to what its scorer op takes, (projected,
    r0, gate_change); launch_future scores them from the tables. Checks build their float64 reference from the same
    function's outputs, so it starts from the scorer's own fp32 inputs."""
    norm_delta, rotate = R.norm_delta, R.rotate

    @torch.compile(dynamic=True)
    def prologue(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin):
        move = move*coefficient
        m, p, width = move.shape
        heads, hd = kraw.shape[-2:]
        dn = norm_delta(x[:, None], move)
        projected = F.linear(dn, wkv).reshape(m, p, 2, heads, hd)
        r0 = rotate(kraw, cos[:, None], sin[:, None])
        gate_change = None
        if gate is not None:
            dg = F.linear(dn[..., :wg.shape[-1]], wg)
            gate_change = 2*((gate[:, None]+dg).sigmoid()-gate[:, None].sigmoid())
        return projected, r0, gate_change
    return prologue


def exact_scores(q, dk, A, C, source_start, query_start, window):
    """(scores [h, P, m], max |x| [h, P, m]) of the record's key_response arithmetic (tf32x3), exact_kernel."""
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    largest = torch.empty_like(out)
    exact_kernel[(triton.cdiv(probes * sources, 64), heads)](
        q, dk, A, C, out, largest, q.shape[1], sources, probes, source_start, query_start, window, q.stride(0),
        q.stride(1), *dk.stride(), *A.stride(), *C.stride(), width, 64, 64, num_warps=4, enable_fp_fusion=False)
    return out, largest


# ------------------------------------------------------------------------------------------ one block, three ways


def block_draws(est, targets, block):
    """Block `block`'s input-K rows and total as key_input_credits draws them (a private copy of the generator, the
    blocks before it drawn in order), this GPU's share."""
    generator = torch.Generator(est.device).set_state(est.generator.get_state())
    R = STATE['installed'] or sys.modules['recipe']
    frame = R.hadamard(est.width, est.device, torch.float32)
    for i in targets:
        indices = torch.randperm(est.width, device=est.device, generator=generator)[:est.s.key_path_draws]
        signs = torch.randint(0, 2, (est.width,), device=est.device, generator=generator) * 2 - 1
        if i == block:
            rows = frame[indices] * signs
            r0, r1 = est.share(len(rows))
            return rows[r0:r1], len(rows)
    raise KeyError(block)


def block_inputs(est, i, r, target, rows, projected_rows=None):
    """The record's per-(block, sequence) inputs of key_input_credits (its formulas)."""
    d, s = est.capture, est.s
    context = est.key_context(i, r, target)
    xin = d.blocks[i][r].float()
    w_key = est.model.transformer.h[i].attn.c_k.weight.float()
    steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return dict(context=context, xin=xin, steps=steps, projected_x=xin @ w_key.T,
                projected_rows=rows @ w_key.T if projected_rows is None else projected_rows)


def record_scorer(kind):
    """(q, dk, A, C, source_start, query_start, window) -> (scores, max |x| or None): 'today' = the op as registered
    (fast_kernels' key kernel, one-pass TF32 in the full stack), 'record' = the record's kernel (tf32x3), 'exact' =
    exact_kernel (the record's arithmetic, with max |x|)."""
    R = STATE['installed'] or sys.modules['recipe']
    if kind == 'exact':
        return exact_scores
    fn = R.fused_key_response if kind == 'today' else R.fused_key_response._init_fn
    return lambda *args: (fn(*args), None)


def record_block(est, i, inputs, rows, total_rows, kind='exact'):
    """One sequence and block by the record's path (key_input_credits' and key_errors_from_context's CUDA branches,
    copied: R.prepared_key_moves / R.raw_key_moves) with the scorer `kind`: (input credit [T, W] (this GPU's share,
    before the pair's sum), raw credit [h, T, hd], max |x| input-K, max |x| raw-K)."""
    R = STATE['installed'] or sys.modules['recipe']
    s, scorer = est.s, record_scorer(kind)
    qp, kp, raw, cos, sin, A, C = inputs['context']
    xin, steps = inputs['xin'], inputs['steps']
    length, window = xin.shape[0], est.model.window_sizes[i][0]
    estimate = torch.zeros(length, est.width, device=est.device)
    big = torch.zeros((), device=est.device)
    for t0 in range(0, length, s.key_path_tokens):
        t1 = min(t0 + s.key_path_tokens, length)
        sl = slice(t0, t1)
        q1 = min(length, t1 + window)
        qlength = min(length, -(-(q1 - t0) // R.KEY_BUCKET) * R.KEY_BUCKET)
        qs = slice(max(0, q1 - qlength), q1)
        dk = R.prepared_key_moves(xin[sl], steps[sl], rows, inputs['projected_x'][sl], inputs['projected_rows'],
                                  raw[:, sl], cos[sl], sin[sl])
        scores, largest = scorer(qp[:, qs], dk, A[:, qs, sl], C[:, qs, sl], t0, qs.start, window)
        estimate[sl] = R.decode_key_pairs(scores, steps[sl], rows) * (len(rows) / total_rows)
        if largest is not None:
            big = torch.maximum(big, largest.max())
    h, t, hd = raw.shape
    lo, hi = est.share(hd)
    result = torch.zeros_like(raw)
    eye = torch.eye(hd, device=est.device)
    rsteps = s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    big_raw = torch.zeros((), device=est.device)
    for t0 in range(0, t, R.KEY_BUCKET):
        t1 = min(t0 + R.KEY_BUCKET, t)
        q1 = min(t, t1 + window)
        source, query = slice(t0, t1), slice(t0, q1)
        dk = R.raw_key_moves(kp[:, source], raw[:, source], rsteps[:, source], cos[source], sin[source], eye[lo:hi])
        scored, largest = scorer(qp[:, query], dk, A[:, query, source], C[:, query, source], t0, t0, window)
        result[:, source, lo:hi] = scored.transpose(1, 2) / rsteps[:, source]
        if largest is not None:
            big_raw = torch.maximum(big_raw, largest.max())
    return estimate, result, big.item(), big_raw.item()


def moment_block(est, i, inputs, rows, total_rows, tables=None):
    """One sequence and block by the moment tables: (input credit [T, W], raw credit [h, T, hd]) as record_block."""
    R = STATE['installed'] or sys.modules['recipe']
    qp, kp, raw, cos, sin, A, C = inputs['context']
    window = est.model.window_sizes[i][0]
    tables = build_tables(qp, A, C, window) if tables is None else tables
    h, t, hd = raw.shape
    lo, hi = est.share(hd)
    rsteps = est.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    raw_credit = rawk_credit(kp, raw, rsteps, cos, sin, qp, A, C, tables, lo, hi)
    if not len(rows):  # no draws on this GPU (key_path_draws < GPUs per sequence): the record decodes nothing too
        return torch.zeros(inputs['xin'].shape[0], est.width, device=est.device), raw_credit
    scores = inputk_scores(inputs['xin'], inputs['steps'], rows, inputs['projected_x'], inputs['projected_rows'], raw,
                           cos, sin, qp, A, C, tables)
    credit = R.decode_key_pairs(scores.T.unsqueeze(0), inputs['steps'], rows) * (len(rows) / total_rows)
    return credit, raw_credit


# ------------------------------------------------------------------------------------------- the replaced method


def key_input_credits(self, targets):
    """Cut.key_input_credits (copied, pinned) with every sequence and block's input-K and raw-K experiments scored from
    the moment tables (moment_block); the private generator copy, its draws in order, the GPU's share, the decode and
    the pair's sum (group_sum) are the record's. Settings the kernels do not cover, a subclass's key_context /
    key_errors_from_context, a recipe function replaced after install, or a running-check fallback: the record's. Every
    call is counted by its path (note: fast_moments.json)."""
    record = STATE['record']['key_input_credits']
    n = STATE['n']
    STATE['n'] += 1
    why = unsupported(self)
    if why is not None:
        note(f'record: {why}')
        return record(self, targets)
    every = STATE['check_every']
    if every and not STATE['fallback'] and n % every == 1 % every:
        running_check(self, targets, n)
    if STATE['fallback']:
        note('fallback')
        return record(self, targets)
    note('moments')
    R = STATE['installed']
    d, s = self.capture, self.s
    generator = torch.Generator(self.device).set_state(self.generator.get_state())
    frame = R.hadamard(self.width, self.device, torch.float32)
    credits = {}
    raw_errors = {}
    for i, target in targets.items():
        indices = torch.randperm(self.width, device=self.device, generator=generator)[:s.key_path_draws]
        signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator)*2-1
        rows = frame[indices] * signs
        total_rows = len(rows)
        r0, r1 = self.share(total_rows)  # research: this GPU's share of the draws
        rows = rows[r0:r1]
        estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
        raw_credit = torch.empty_like(estimate)
        projected_rows = rows @ self.model.transformer.h[i].attn.c_k.weight.float().T
        for r in range(d.batch):
            inputs = block_inputs(self, i, r, target, rows, projected_rows)
            estimate[r], raw = moment_block(self, i, inputs, rows, total_rows)
            raw_credit[r] = raw.transpose(0, 1).reshape(d.length, self.width)
            del inputs
        credits[i] = self.group_sum(estimate.flatten(0, 1))
        raw_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
    self._input_key_errors = raw_errors
    return credits


def unsupported(est):
    """Why the moment path does not cover est's input-K call (the record's then runs), or None."""
    R, refs = STATE['installed'], STATE['record']
    cfg = est.model.config
    for ok, reason in (
            (next(est.model.parameters()).is_cuda, 'not on CUDA'),
            (cfg.n_embd // cfg.n_head == 64, f'head size {cfg.n_embd // cfg.n_head} (the kernels: 64)'),
            (cfg.n_embd == cfg.n_head * 64 and cfg.n_head == cfg.n_kv_head, 'K heads differ from the query heads'),
            (type(est).key_context is refs['key_context'], f"{type(est).__name__}.key_context is not the record's"),
            (type(est).key_errors_from_context is refs['key_errors_from_context'],
             f"{type(est).__name__}.key_errors_from_context is not the record's")):
        if not ok:
            return reason
    for name in RELIED:
        if getattr(R, name) is not refs[name]:
            return f'recipe.{name} was replaced after install'
    return None


def running_check(est, targets, n):
    """One block (in turn) of this call also scored exactly (record_block: the record's moves, raw-K R.raw_key_moves and
    input-K R.prepared_key_moves, scored by the record's tf32x3 arithmetic): relative L2 of its input-K and raw-K credits
    against the moment ones and max |x|, the max over all ranks (one all_reduce on every rank: they all make this call),
    written to fast_moments.json at once; above FALLBACK_REL: fallback."""
    L = est.model.config.n_layer
    block = (n // max(1, STATE['check_every'])) % L
    rows, total = block_draws(est, targets, block)
    inputs = block_inputs(est, block, 0, targets[block], rows)
    got_in, got_raw = moment_block(est, block, inputs, rows, total)
    ref_in, ref_raw, x_in, x_raw = record_block(est, block, inputs, rows, total, 'exact')

    def rel(a, b):
        den = b.double().norm()
        num = (a.double() - b.double()).norm()
        return torch.where(den > 0, num / den.clamp_min(1e-300), torch.where(num > 0, torch.inf, 0.))
    stats = torch.stack([rel(got_in, ref_in), rel(got_raw, ref_raw),
                         torch.tensor(x_in, device=est.device, dtype=torch.float64),
                         torch.tensor(x_raw, device=est.device, dtype=torch.float64)])
    stats = torch.nan_to_num(stats, nan=torch.inf)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    rel_in, rel_raw, x_in, x_raw = stats.tolist()
    ok = max(rel_in, rel_raw) <= STATE['fallback_rel']
    STATE['checks'].append(dict(call=n, block=block, rel_input=float(f'{rel_in:.3e}'), rel_raw=float(f'{rel_raw:.3e}'),
                                max_x_input=round(x_in, 5), max_x_raw=round(x_raw, 6), ok=ok))
    if STATE['shared'] is None:
        STATE['shared'] = shared_quantities(est, targets)
    if not ok:
        STATE['fallback'] = True
        if os.environ.get('RANK', '0') == '0':
            print(f'fast_moments: running check failed at call {n} (block {block}: input-K {rel_in:.2e}, raw-K '
                  f"{rel_raw:.2e} > {STATE['fallback_rel']}): the record from now on", flush=True)
    note(force=True)


def prepare_context(est, i, sequence, target):
    """FiniteFuture.prepare's q / a / contrast for block i of one sequence (its formulas: attention_state; the contrast
    masked to the window as key_context's C is), in key_context's layout [H, T, D] / [H, T(query), T(source)]."""
    R, d, model = STATE['installed'] or sys.modules['recipe'], est.capture, est.model
    t, heads = d.length, model.config.n_head
    cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
    block = model.transformer.h[i]
    prefix = f'transformer.h.{i}.attn.'
    qraw, kraw, vraw = [d.outputs[prefix + n][sequence:sequence + 1].float() for n in ('c_q', 'c_k', 'c_v')]
    v = vraw.reshape(1, t, heads, -1)
    if block.attn.ve_gate is not None:
        gate = d.outputs[prefix + 've_gate'][sequence].float()
        ve = d.outputs[f'value_embeds.{i}'][sequence].float().reshape(t, heads, -1)
        v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
    values = v.transpose(1, 2)[0]
    q, _, a, o = (z[0] for z in R.attention_state(qraw, kraw, values[None], cos, sin, heads, model.window_sizes[i][0]))
    hub = target[sequence].reshape(t, heads, -1).transpose(0, 1)
    contrast = (hub @ values.transpose(-1, -2)) - (hub * o).sum(-1, keepdim=True)
    pos = torch.arange(t, device=est.device)
    lag = pos[:, None] - pos[None, :]
    return q, a, contrast * ((lag >= 0) & (lag <= model.window_sizes[i][0]))


def shared_quantities(est, targets, sequence=0):
    """prepare's q / a / contrast next to key_context's qp / A / C, per block of one sequence: max |difference| / max
    |key_context's|. They agree to TF32 rounding (the logits' TF32 GEMM rounds a last-bit-different q differently), not
    bitwise: the key part builds its tables from key_context's (the record's own inputs to these experiments)."""
    out = {}
    for i in range(est.model.config.n_layer):
        qp, _, _, _, _, A, C = est.key_context(i, sequence, targets[i])
        q, a, contrast = prepare_context(est, i, sequence, targets[i])
        rel = lambda u, w: float(f'{((u - w).abs().max() / w.abs().max().clamp_min(1e-30)).item():.2e}')
        out[i] = dict(q=rel(q, qp), a=rel(a, A), contrast=rel(contrast, C))
    return out


# --------------------------------------------------------------------------------------------- the future receivers


class Tables:
    """One block's future moment tables (U, W, M1, M2), built from its prepare context on first use and freed with it:
    ctx['moments'] holds this, LOOKUP maps id(ctx['q']) to a weak reference (the record's experiment binds the
    scorer's q = ctx['q']; holding q here keeps that id from being reused while this lives)."""
    __slots__ = ('q', 'target', 'a', 'contrast', 'window', 'block', 'built', '__weakref__')

    def __init__(self, ctx, block):
        self.q, self.target, self.a, self.contrast = ctx['q'], ctx['target'], ctx['a'], ctx['contrast']
        self.window, self.block, self.built = ctx['window'], block, None

    def get(self):
        if self.built is None:
            self.built = build_future_tables(self.q, self.target, self.a, self.contrast, self.window)
        return self.built


@torch.no_grad()
def prepare(self, sequence=0):
    """FiniteFuture.prepare: the record's, called as is; with the future part each block's context gets its Tables
    (ctx['moments']: built on first use from this context's q / target / a / contrast, freed with the contexts).
    Counts the prepares (one per sequence) and starts a new sequence's per-block call counts (future_counts; the
    previous sequence's become future_last, the N_j of the running check); the first real Cut step's sequence (prepare
    1) and every CHECK_EVERY-th after it are checked: future_check = k (n // CHECK_EVERY), and receiver_scores checks
    one call per receiver block."""
    out = STATE['record']['prepare'](self, sequence)
    if 'future' in STATE['parts'] and self.contexts is not None:
        n = STATE['prepares']
        STATE['prepares'] += 1
        for key in [k for k, r in LOOKUP.items() if r() is None]:
            del LOOKUP[key]
        STATE['future_last'], STATE['future_counts'], STATE['future_check'] = STATE['future_counts'], {}, None
        if not STATE['fallback']:
            for j, ctx in enumerate(self.contexts):
                holder = Tables(ctx, j)
                ctx['moments'] = holder
                LOOKUP[id(ctx['q'])] = weakref.ref(holder)
            every = STATE['check_every']
            if every and n % every == 1 % every:
                STATE['future_check'] = n // every
    return out


def future_unsupported(holder, move, q, kraw, start, window):
    """Why the moment path does not cover this receiver call (the scorer installed before runs), or None."""
    if holder is None or holder.q is not q:
        return 'no moment tables for this receiver (its context is not from the patched FiniteFuture.prepare)'
    for ok, reason in ((q.is_cuda, 'not on CUDA'), (q.shape[-1] == 64, f'head size {q.shape[-1]} (the kernels: 64)'),
                       (kraw.shape[-2] == q.shape[0] and kraw.shape[-1] == q.shape[-1], 'K heads differ from Q heads'),
                       (holder.window == window, 'the window differs from the tables\''),
                       (0 <= start and start + move.shape[0] <= q.shape[1], 'sources outside the tables')):
        if not ok:
            return reason
    return None


def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin, q, target, a, contrast, start, window):
    """R.receiver_scores (bound by FiniteFuture.experiment per receiver block and token chunk, looked up when the
    experiment is built: onesided.py's experiment calls the record's and keeps its scorer) with the future part: the
    receiver block's moment tables (through q, prepare's ctx['q']) score every trial (future_prologue, then
    launch_future); a call the moment path does not cover, or after a fallback, runs the scorer installed before
    (fast_kernels' fused). Counted by path. In a checked sequence (future_check = k) each receiver block j's
    (FUTURE_STRIDE k mod N_j)-th covered call (N_j: its covered calls in the previous sequence; none known: its first)
    is also checked (future_running_check)."""
    previous = STATE['previous_scorer']
    args = (move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin, q, target, a, contrast, start, window)
    if STATE['fallback']:
        note_future('fallback')
        return previous(*args)
    ref = LOOKUP.get(id(q))
    holder = ref() if ref is not None else None
    why = future_unsupported(holder, move, q, kraw, start, window)
    if why is not None:
        note_future(f'record: {why}')
        return previous(*args)
    block = holder.block
    ordinal = STATE['future_counts'].get(block, 0)  # this call's place among the block's covered calls of the sequence
    STATE['future_counts'][block] = ordinal + 1
    U, W, M1, M2 = holder.get()
    projected, r0, gate_change = STATE['future_compiled'](move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin)
    scores = launch_future(projected, r0, cos, sin, gate_change, ve, U, W, M1, M2, start)
    k = STATE['future_check']
    if k is not None:
        calls = STATE['future_last'].get(block, 0)
        if ordinal == ((FUTURE_STRIDE * k) % calls if calls else 0):
            if not future_running_check(scores, args, (projected, r0, gate_change), block, ordinal, calls):
                note_future('fallback')
                return previous(*args)
    note_future('moments')
    return scores


def score_columns64(q, target, a_cols, c_cols, dk, dv, src, window):
    """joint_scores' exact single-column update in float64 for the sources src ([n] absolute positions; a_cols / c_cols
    [H, T, n]: their columns of prepare's a / contrast; dk, dv [n, P, H, D]) over the later queries in the window,
    t < s <= t + window: ([n, P] summed over heads and queries, max |x| over them as a 0-dim tensor)."""
    hd = dk.shape[-1]
    ds = torch.einsum('hsd,nphd->nphs', q.double(), dk) * hd ** -.5
    vs = torch.einsum('hsd,nphd->nphs', target.double(), dv)
    a = a_cols.double().permute(2, 0, 1)[:, None]
    cc = c_cols.double().permute(2, 0, 1)[:, None]
    changed = a * torch.expm1(ds)
    response = (changed * cc + (a + changed) * vs) / (1 + changed)
    pos = torch.arange(q.shape[1], device=dk.device)[None]
    allowed = ((pos > src[:, None]) & (pos - src[:, None] <= window))[:, None, None]
    return (response * allowed).sum((2, 3)), (ds.abs() * allowed).amax()


def future_reference64(args, prologue, local, chunk=4):
    """The running check's reference for one receiver call: float64 scores [n, P] (summed over heads) of the chunk's
    source tokens `local` (positions in the chunk, [n]) and max |x|, from the scorer's own fp32 prologue outputs
    (projected, r0, gate_change): rotation, key norm_delta against r0 and the gate's value-embedding term in float64
    (moves_from_projection64's formulas), then score_columns64; `chunk` tokens at a time (~0.6 GB at 128 trials)."""
    move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin, q, target, a, contrast, start, window = args
    projected, r0, gate_change = prologue
    out, big = [], torch.zeros((), device=projected.device, dtype=torch.float64)
    for k in range(0, len(local), chunk):
        li = local[k:k + chunk]
        c, sn = cos[li].double()[:, None, None], sin[li].double()[:, None, None]
        dk = norm_delta64(r0[li].double()[:, None], rot64(projected[li, :, 0].double(), c, sn))
        dv = projected[li, :, 1].double()
        if gate_change is not None:
            dv = dv + gate_change[li].double()[..., None] * ve[li].double()[:, None]
        score, largest = score_columns64(q, target, a[:, :, li], contrast[:, :, li], dk, dv, start + li, window)
        out.append(score)
        big = torch.maximum(big, largest)
    return torch.cat(out), big


def future_running_check(scores, args, prologue, block, ordinal, calls):
    """One covered receiver call of a checked sequence (receiver block `block`, its `ordinal`-th covered call of the
    sequence, `calls` = N_j) also scored in float64 on FUTURE_CHECK_TOKENS source tokens of its chunk (edge_tokens: the
    chunk's edges and evenly spaced others) from the scorer's own fp32 prologue outputs (future_reference64): relative
    L2 of the moment scores of those tokens (all trials) and max |x|, the max over all ranks (one all_reduce on every
    rank: each makes this same call), written to fast_moments.json at once; above FALLBACK_REL: fallback (both parts).
    True if ok."""
    m, trials = scores.shape
    local = torch.as_tensor(edge_tokens(m, FUTURE_CHECK_TOKENS), device=scores.device)
    ref, xmax = future_reference64(args, prologue, local)
    den = ref.norm()
    num = (scores[local].double() - ref).norm()
    rel = torch.where(den > 0, num / den.clamp_min(1e-300), torch.where(num > 0, torch.inf, 0.))
    stats = torch.nan_to_num(torch.stack([rel, xmax]), nan=torch.inf)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    rel, xmax = stats.tolist()
    ok = rel <= STATE['fallback_rel']
    STATE['checks'].append(dict(part='future', call=STATE['prepares'] - 1, event=STATE['future_check'], block=block,
                                ordinal=ordinal, of=calls, start=int(args[14]), chunk=int(m), tokens=int(len(local)),
                                trials=int(trials), rel=float(f'{rel:.3e}'), max_x=round(xmax, 5), ok=ok))
    if not ok:
        STATE['fallback'] = True
        if os.environ.get('RANK', '0') == '0':
            print(f"fast_moments: future running check failed at prepare {STATE['prepares'] - 1} (receiver block {block},"
                  f" call {ordinal}: {rel:.2e} > {STATE['fallback_rel']} from float64): the scorer before it from now on",
                  flush=True)
    note(force=True)
    return ok


def note_future(path):
    """Count a receiver call by its path ('moments', 'fallback', 'record: <why>'); a new path is printed (rank 0) and
    written to fast_moments.json at once."""
    calls = STATE['future_calls']
    new = path not in calls
    calls[path] = calls.get(path, 0) + 1
    if path.startswith('record: '):
        why = path[len('record: '):]
        STATE['future_unsupported'][why] = STATE['future_unsupported'].get(why, 0) + 1
    if new:
        if os.environ.get('RANK', '0') == '0':
            print('fast_moments: future receivers run ' + dict(moments='the moment tables', fallback='the scorer before '
                  '(a running check fell back)').get(path, path.replace('record: ', 'the scorer before: ')), flush=True)
        note(force=True)


# --------------------------------------------------------------------------------------------- install, notes


RELIED = ('prepared_key_moves', 'raw_key_moves', 'decode_key_pairs', 'decode_local_response', 'hadamard',
          'fused_key_response', 'norm_difference', 'attention_state')


def parse_parts(spec):
    parts = tuple(p for p in (spec or '').replace(' ', '').split(',') if p)
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f'FAST_MOMENTS: {unknown} not implemented here (this file: {PARTS})')
    return tuple(p for p in PARTS if p in parts)  # canonical order: 'key,future'


def describe():
    """What install() set up and what the calls did (fast_moments.json)."""
    FK, FC = sys.modules.get('fast_kernels'), sys.modules.get('fast_chain')
    stack = {}
    for name, mod in (('fast_kernels', FK), ('fast_chain', FC)):
        try:
            stack[name] = mod.describe() if mod is not None and mod.STATE.get('installed') is not None else None
        except Exception as error:  # never stop a run for a note
            stack[name] = repr(error)
    previous = STATE['previous_scorer']
    R = STATE['installed']
    fast = getattr(R, '_fast_receiver_scores', {}) if R is not None else {}
    scorer = None if previous is None else next((f'fast_kernels {k}' for k, v in fast.items() if v is previous),
                                                'the record' if previous is getattr(R, '_record_receiver_scores', None)
                                                else repr(previous))
    return dict(parts=','.join(STATE['parts']), installed=STATE['installed'] is not None, pin=PIN,
                future_pin=FUTURE_PIN if 'future' in STATE['parts'] else None,
                file=str(FILE), sha1=SHA1, calls=dict(STATE['calls']),
                calls_total=sum(STATE['calls'].values()), future_calls=dict(STATE['future_calls']),
                future_calls_total=sum(STATE['future_calls'].values()), prepares=STATE['prepares'],
                future_previous_scorer=scorer, check_every=STATE['check_every'],
                future_check=dict(stride=FUTURE_STRIDE, tokens=FUTURE_CHECK_TOKENS, blocks='all, one call each',
                                  reference='float64 from the scorer prologue outputs')
                if 'future' in STATE['parts'] else None,
                fallback_rel=STATE['fallback_rel'], fallback=STATE['fallback'], checks=STATE['checks'],
                unsupported=dict(STATE['unsupported']), future_unsupported=dict(STATE['future_unsupported']),
                previous=STATE['previous'], shared=STATE['shared'],
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, torch=torch.__version__,
                triton=triton.__version__, **stack)


def note(path=None, force=False):
    """Count a key_input_credits call by its path ('moments', 'fallback', 'record: <why>'; None: count nothing). The
    first time a path is taken rank 0 prints it; rank 0 of a torchrun process rewrites <output>/fast_moments.json then,
    when forced (each running check, a new receiver path, exit) and every NOTE_EVERY calls (as fast_chain.note), once
    the recipe has written config.json there (it refuses a non-empty --output before that)."""
    calls = STATE['calls']
    new = path is not None and path not in calls
    if path is not None:
        calls[path] = calls.get(path, 0) + 1
        if path.startswith('record: '):
            why = path[len('record: '):]
            STATE['unsupported'][why] = STATE['unsupported'].get(why, 0) + 1
    if new and os.environ.get('RANK', '0') == '0':
        print('fast_moments: key_input_credits runs ' + dict(moments='the moment tables', fallback='the record (a '
              'running check fell back: checks, previous)').get(path, path.replace('record: ', 'the record: ')), flush=True)
    output = STATE['output']
    if os.environ.get('RANK') != '0' or output is None or not (output / 'config.json').exists():
        return
    if new or force or (path is not None and sum(calls.values()) % NOTE_EVERY == 0):
        part = output / 'fast_moments.json.part'
        part.write_text(json.dumps(describe(), indent=1) + '\n')
        os.replace(part, output / 'fast_moments.json')  # whole files only (a resumed segment reads it)


def exit_note():
    """atexit: the final counts (never raises)."""
    try:
        note(force=True)
    except Exception as error:
        print(f'fast_moments: fast_moments.json not written at exit: {error!r}', flush=True)


def resumed_from(argv):
    """The previous segment's fast_moments.json (next to --resume's checkpoint), as 'previous': its file, sha1, fallback,
    checks and calls; None without --resume."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resume', type=Path)
    resume = parser.parse_known_args(argv)[0].resume
    if resume is None:
        return None
    path = resume.parent / 'fast_moments.json'
    if not path.exists():
        return dict(file=str(path), missing=True, fallback=False)
    try:
        prev = json.loads(path.read_text())
    except ValueError as error:  # unreadable: assume the worst (the record from here on, the run not adopted)
        return dict(file=str(path), unreadable=repr(error), fallback=True)
    return dict(file=str(path), sha1=prev.get('sha1'), fallback=bool(prev.get('fallback')),
                checks=prev.get('checks', []), calls=prev.get('calls'), future_calls=prev.get('future_calls'))


def install(R=None, parts=None, check_pin=True):
    """Install the parts on the loaded recipe module (process-wide; idempotent; a later call switches parts), after
    research/fast_kernels.py and research/fast_chain.py. parts: 'key' (Cut.key_input_credits), 'future'
    (FiniteFuture.prepare gets the tables, R.receiver_scores the moment scorer; the scorer installed before is kept as
    the fallback); default $FAST_MOMENTS; none = nothing is replaced (the current stack, bitwise). In a torchrun
    process: fast_moments.json goes into --output (see note; at exit too), and with --resume the previous segment's
    fallback stays on. Returns R."""
    R = sys.modules['recipe'] if R is None else R
    parts = parse_parts('key,future' if parts is None else ','.join(parts)
                        if isinstance(parts, (tuple, list)) else parts)
    uninstall(R)
    STATE.update(installed=None, parts=())
    if not parts:
        return R
    if 'future' in parts:
        FK = sys.modules.get('fast_kernels')
        if FK is None or FK.STATE.get('installed') is not R:
            raise RuntimeError('fast_moments future: install fast_kernels (and fast_chain) on this recipe first')
    if not hasattr(R.Cut, '_record_key_input_credits'):
        R.Cut._record_key_input_credits = R.Cut.key_input_credits
    if not hasattr(R.FiniteFuture, '_record_prepare'):
        R.FiniteFuture._record_prepare = R.FiniteFuture.prepare
    if 'key' in parts:
        R.Cut.key_input_credits = key_input_credits
    if 'future' in parts:
        STATE['previous_scorer'] = R.receiver_scores
        if STATE['future_compiled'] is None:
            STATE['future_compiled'] = future_prologue(R)
        R.FiniteFuture.prepare = prepare
        R.receiver_scores = receiver_scores
    STATE.update(installed=R, parts=parts, record=dict(
        {name: getattr(R, name) for name in RELIED}, key_input_credits=R.Cut._record_key_input_credits,
        key_context=R.Boost.key_context, key_errors_from_context=R.Boost.key_errors_from_context,
        prepare=R.FiniteFuture._record_prepare))
    if os.environ.get('RANK') is not None:  # torchrun: fast_moments.json goes into the run's --output
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', type=Path, default=Path('logs/val'))  # the recipe's default
        STATE['output'] = parser.parse_known_args(sys.argv[1:])[0].output
        STATE['previous'] = resumed_from(sys.argv[1:])
        if STATE['previous'] is not None and STATE['previous']['fallback']:
            STATE['fallback'] = True  # a fallback in an earlier segment: the record for the rest of the run too
        if not STATE['exit_hook']:
            atexit.register(exit_note)
            STATE['exit_hook'] = True
    return R


def uninstall(R):
    """The record's key_input_credits and prepare, and the scorer installed before the future part, back."""
    if hasattr(R.Cut, '_record_key_input_credits'):
        R.Cut.key_input_credits = R.Cut._record_key_input_credits
    if hasattr(R.FiniteFuture, '_record_prepare'):
        R.FiniteFuture.prepare = R.FiniteFuture._record_prepare
    if R.receiver_scores is receiver_scores and STATE['previous_scorer'] is not None:
        R.receiver_scores = STATE['previous_scorer']


# ------------------------------------------------------------------------------------------------- checks and timing


def edge_tokens(T, n):
    """n distinct source positions: the edges 0, 1, 63, 64, 1023, 1024, T - 1 (those < T) and evenly spaced others."""
    edges = sorted({t for t in (0, 1, 63, 64, 1023, 1024, T - 1) if t < T})
    others = [t for t in torch.linspace(0, T - 1, 4 * n).round().long().tolist() if t not in edges]
    others = sorted(set(others))
    k = max(0, min(n - len(edges), len(others)))
    pick = [others[round(j * (len(others) - 1) / max(1, k - 1))] for j in range(k)] if k else []
    return sorted(set(edges) | set(pick))


def rot64(v, c, sn):
    half = v.shape[-1] // 2
    return torch.cat((v[..., :half] * c + v[..., half:] * sn, -v[..., :half] * sn + v[..., half:] * c), -1)


def norm_delta64(x, delta):  # the record's norm_delta in float64 (its constant: the fp32 eps of training)
    width = x.shape[-1]
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + EPS)
    moved = torch.rsqrt((square + 2 * (x * delta).sum(-1, keepdim=True) + delta.square().sum(-1, keepdim=True)) / width
                        + EPS)
    return delta * moved + x * (moved - inverse)


# (source family, layer, receiver block, token chunk start): blocks 0 (short window, no VE), 1 (short, VE), 3 (long,
# VE), 7 (the last); the first and last chunks (windows cut at 0 / T) and two inside


#   128 trials) on a clean H100, the yardstick of how much a shared GPU stretches kernels


# ======================================== module fast_chain2
"""fast_chain2's kernels (exact MLP unit pruning, row statistics from partial sums), used by fast_readout's pass; the
chain2 chain itself is not installed in this run.
"""
import argparse
import atexit
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


FILE = Path(__file__).resolve()
import fast_chain as FC  # noqa: E402  (the module fast_chain.install() set up; its kernels are reused)

SHA1 = '0e1a3544723b'  # sha1 of the research file this section was copied from
FC_PIN = '1f078c8d9e14'  # sha1 of the fast_chain code this copies or calls (fc_pin(); install() refuses another)
MARGIN = 1.02  # prune bound safety factor on A and C (needs > 1 + 2^-8 + fp32 rounding)
CHECK_EVERY, CHECK_TOKENS, NOTE_EVERY = 50, 32, 50
EPS = torch.finfo(torch.float32).eps
# launch tiles (swept on real chain inputs, GPU 7; prep's is fast_chain's linear_residual tile with 3 stages)
PREP = (64, 128, 64, 4, 3)  # prep_kernel: BM, BN (columns), BK, warps, stages
UP = (64, 128, 64, 4, 4)  # mlp_up_live_kernel: BP (the prune row block), BN (unit tile), BK, warps, stages
DOWN = (64, 128, 128, 4, 3)  # mlp_down_live_kernel: BP (divides UP's), BN (columns), BK (unit tile: divides UP's), ...
# impl (checks / timing): oq 'part' (proj_kernel + fast_chain's oq_attn_kernel) | 'fc2' (fast_chain's projection and
# attention kernels) | 'fc' (fast_chain's oq_fast, c_proj included); mlp 'live' (pruned) | 'full' (no pruning)
DEFAULT_IMPL = dict(oq='part', mlp='live')
STATE = dict(installed=None, impl=dict(DEFAULT_IMPL), chunk=None, calls={}, output=None, checks=[],
             fallback=False, n=0, cut_sum=None, cut_rows=None, check_now=None, exit_hook=False, inject=None,
             fc_chain=None, pin=None)


# ----------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def proj_kernel(D, PSI, SQX, WT, U, STATS, n, n_rows, D_ROWS, U_ROWS, m,
                W: tl.constexpr, NCH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                EPS: tl.constexpr):
    """fast_chain's proj_stats_kernel without the row statistics in its main loop: u = d W_q^T (bf16 inputs, fp32
    accumulation) for the n_rows first rows of every token (flattened rows: token i // n_rows); the pn == 0 programs
    write inv' = rsqrt(|x + d|^2 / W + eps) and coef = (inv' - inv0) / inv0 (fast_chain's formulas) from the row's
    partial sums d . x, d . d (PSI, NCH column blocks: written by the previous block's mlp_down_live_kernel), x the block's
    clean input. A block of BM rows spans at most two tokens (n_rows >= BM)."""
    pn, pm = tl.program_id(0), tl.program_id(1)
    i = pm * BM + tl.arange(0, BM)
    ok = i < n
    tok, r = i // n_rows, i % n_rows
    drow = (tok * D_ROWS + r).to(tl.int64) * W
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(D + drow[:, None] + kk[None, :], ok[:, None], 0.)
        acc = tl.dot(a.to(tl.bfloat16), tl.load(WT + kk[:, None] * W + cols[None, :]), acc)
    out = (tok * U_ROWS + r).to(tl.int64)
    tl.store(U + out[:, None] * W + cols[None, :], acc, ok[:, None])
    if pn == 0:
        ta = (pm * BM) // n_rows
        tb = tl.minimum((pm * BM + BM - 1) // n_rows, m - 1)
        pidx = ((tok * D_ROWS + r).to(tl.int64)[:, None] * NCH + tl.arange(0, NCH)[None, :]) * 2
        dotx = tl.sum(tl.load(PSI + pidx, ok[:, None], 0.), 1)
        sqd = tl.sum(tl.load(PSI + pidx + 1, ok[:, None], 0.), 1)
        sq = tl.where(tok == ta, tl.load(SQX + ta), tl.load(SQX + tb))
        inv0 = libdevice.rsqrt(sq / W + EPS)
        inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
        tl.store(STATS + out * 2, inv, ok)
        tl.store(STATS + out * 2 + 1, (inv - inv0) / inv0, ok)


@triton.jit
def prep_kernel(A, B, D, MID, PS, STEPS, BASIS, PREV, X0R, h, n_oq, n_mlp, n_probe, A_ROWS, D_ROWS,
                W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CPROJ: tl.constexpr,
                ROOT: tl.constexpr):
    """BM rows x BN columns of one token at block b, every row < n_mlp: rows < n_oq (CPROJ) d += a W_proj^T
    (fast_chain's linear_residual_kernel tile and arithmetic: bf16 inputs, fp32 accumulation); (ROOT) root b's rows,
    formed here as the record's driver forms them: rows [n_oq, n_oq + n_probe) step_t * basis[p] and the scalar probes
    after them +h prev, +h x0, -h prev, -h x0 (prev: block b's clean input stream); d stored; and the partial sums
    d . mid, d . d on these columns (PS, fp32, per (row, column block)) for the MLP's RMSNorm."""
    nb, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_mlp
    oq = rows < n_oq
    cols = nb * BN + tl.arange(0, BN)
    dptr = D + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    if CPROJ:
        if pb * BM < n_oq:
            arow = (t * A_ROWS + rows).to(tl.int64) * W
            for k0 in range(0, W, BK):
                kk = k0 + tl.arange(0, BK)
                a = tl.load(A + arow[:, None] + kk[None, :], oq[:, None], 0.)
                b = tl.load(B + kk[:, None] * W + cols[None, :])
                acc = tl.dot(a, b, acc)
    if ROOT:
        new = tl.load(dptr, oq[:, None], 0.) + acc
        if pb * BM + BM > n_oq:
            j = rows - n_oq
            probe = (j >= 0) & (j < n_probe)
            fresh = tl.load(STEPS + t) * tl.load(BASIS + tl.maximum(j, 0)[:, None] * W + cols[None, :],
                            probe[:, None], 0.)
            k = j - n_probe
            scal = (k >= 0) & ok
            src = tl.where((k % 2 == 0)[:, None], tl.load(PREV + t * W + cols)[None, :],
                           tl.load(X0R + t * W + cols)[None, :])
            fresh = tl.where(scal[:, None], src * tl.where(k < 2, h, -h)[:, None], fresh)
            new = tl.where(oq[:, None], new, fresh)
        tl.store(dptr, new, ok[:, None])
    else:
        new = tl.load(dptr, ok[:, None], 0.) + acc
        if CPROJ:
            tl.store(dptr, new, oq[:, None])
    x = tl.load(MID + t * W + cols)
    prow = ((t * D_ROWS + rows).to(tl.int64) * (W // BN) + nb) * 2
    tl.store(PS + prow, tl.sum(new * x[None, :], 1), ok)
    tl.store(PS + prow + 1, tl.sum(new * new, 1), ok)


@triton.jit
def mlp_up_live_kernel(D, PS, MID, SC, IDX, WFC, HS, CUT, Out, n_rows, D_ROWS, O_ROWS, n_rb,
                       W: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, NCH: tl.constexpr, BP: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, EPS: tl.constexpr, MARGIN: tl.constexpr,
                       PRUNE: tl.constexpr, FULL: tl.constexpr):
    """fast_chain's mlp_up_kernel (fp32 moves read and rounded to bf16 for the GEMM) on a row block's live unit prefix.
    The row's RMSNorm scalars from prep_kernel's partial sums: inv' = rsqrt(|mid + d|^2 / W + eps), coef =
    (inv' - inv0) / inv0 (fast_chain's formulas); PRUNE: the
    row block's prune point (CUT) = 1 + the last sorted unit with s >= -A' / (1 - C') (module docstring), and only
    the unit tiles before it are formed; FULL: all K units (nothing stored in CUT). dh = inv' (d W_fc^T) + coef h on
    the units, then relu(h + dh)^2 - relu(h)^2 as (up - down)(up + down), bf16 out."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    prow = (t * D_ROWS + rows).to(tl.int64) * NCH
    ch = tl.arange(0, NCH)
    pidx = (prow[:, None] + ch[None, :]) * 2
    dotx = tl.sum(tl.load(PS + pidx, ok[:, None], 0.), 1)
    sqd = tl.sum(tl.load(PS + pidx + 1, ok[:, None], 0.), 1)
    xm = tl.load(MID + t * W + tl.arange(0, W))
    sq = tl.sum(xm * xm, 0)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    coef = (inv - inv0) / inv0  # fast_chain's formula
    n_units = K
    if PRUNE:
        a_max = tl.max(tl.where(ok, inv * libdevice.sqrt(sqd), 0.), 0) * MARGIN
        c_max = tl.max(tl.where(ok, tl.abs(coef), 0.), 0) * MARGIN
        thr = -a_max / (1 - c_max)
        upos = tl.arange(0, KP)
        s = tl.load(SC + t * K + upos, upos < K, float('-inf'))
        n_units = tl.max(tl.where(s >= thr, upos + 1, 0), 0)
        n_units = tl.where((c_max < 0.5) & (a_max == a_max), n_units, K)  # (non-finite bound: every unit)
    if not FULL:
        tl.store(CUT + t * n_rb + pb, n_units)
    if FULL:
        n_units = K
    inv = inv[:, None]
    coef = coef[:, None]
    dbase = D + (t * D_ROWS + rows).to(tl.int64) * W
    obase = Out + (t * O_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, n_units, BN):
        units = u0 + tl.arange(0, BN)
        uidx = tl.load(IDX + t * K + units).to(tl.int64)
        acc = tl.zeros((BP, BN), tl.float32)
        for k0 in range(0, W, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
            w = tl.load(WFC + uidx[None, :] * W + kk[:, None])
            acc = tl.dot(a.to(tl.bfloat16), w, acc)
        h = tl.load(HS + t * K + units)
        dh = inv * acc + coef * h[None, :]
        up = tl.maximum(h[None, :] + dh, 0.)
        down = tl.maximum(h, 0.)[None, :]
        tl.store(obase[:, None] + units[None, :], ((up - down) * (up + down)).to(tl.bfloat16), ok[:, None])


@triton.jit
def mlp_down_live_kernel(Rb, IDX, WOT, D, DX0, LAM, CUT, XN, PSI, A16, XL, TW, n_rows, D_ROWS, R_ROWS, n_x0, n_rb,
                         W: tl.constexpr, K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         RB: tl.constexpr, SCALE: tl.constexpr, X0: tl.constexpr, FULL: tl.constexpr,
                         NEXT: tl.constexpr, HEADP: tl.constexpr):
    """fast_chain's mlp_down_kernel over the row block's live unit prefix (CUT of its RB-row prune block, rounded up to
    BK; every unit there was formed by mlp_up_live_kernel, the rest are exact zeros): d += r W_out^T, then block b + 1's
    mixing scalars d = lam (d + change) (+ lam0 dx0 on the embedding root's rows), in place; NEXT: the new rows'
    partial sums d . x and d . d on these columns (PSI) for block b + 1's input RMSNorm (x = its clean input, XN).
    HEADP (last block): the head's inputs instead of the fp32 moves (read by nothing else): their bf16 copy (A16, the
    logits GEMM's input, as fast_chain's head_prep_kernel) and the partial sums d . xL, d . w_y, d . d (PSI, 3 per
    column block)."""
    nb, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    cols = nb * BN + tl.arange(0, BN)
    if FULL:
        n_units = K
    else:
        n_units = tl.load(CUT + t * n_rb + (pb * BP) // RB)
    acc = tl.zeros((BP, BN), tl.float32)
    rbase = Rb + (t * R_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, n_units, BK):
        uu = u0 + tl.arange(0, BK)
        a = tl.load(rbase[:, None] + uu[None, :], ok[:, None], 0.)
        uidx = tl.load(IDX + t * K + uu).to(tl.int64)
        w = tl.load(WOT + uidx[:, None] * W + cols[None, :])
        acc = tl.dot(a, w, acc)
    dptr = D + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if SCALE:
        new = tl.load(LAM) * new
        if X0:
            lam0 = tl.load(LAM + 1)
            mine = rows < n_x0
            x0d = tl.load(DX0 + (t * n_x0 + rows).to(tl.int64)[:, None] * W + cols[None, :], mine[:, None], 0.)
            new = tl.where(mine[:, None], new + lam0 * x0d, new)
    if HEADP:
        tl.store(A16 + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :], new.to(tl.bfloat16), ok[:, None])
        prow = ((t * D_ROWS + rows).to(tl.int64) * (W // BN) + nb) * 3
        tl.store(PSI + prow, tl.sum(new * tl.load(XL + t * W + cols)[None, :], 1), ok)
        tl.store(PSI + prow + 1, tl.sum(new * tl.load(TW + t * W + cols)[None, :], 1), ok)
        tl.store(PSI + prow + 2, tl.sum(new * new, 1), ok)
    else:
        tl.store(dptr, new, ok[:, None])
        if NEXT:
            x = tl.load(XN + t * W + cols)
            prow = ((t * D_ROWS + rows).to(tl.int64) * (W // BN) + nb) * 2
            tl.store(PSI + prow, tl.sum(new * x[None, :], 1), ok)
            tl.store(PSI + prow + 1, tl.sum(new * new, 1), ok)


@triton.jit
def head_stats_kernel(PSI, SQXL, BY, T0Y, MB, MW, STATS, n_rows, D_ROWS,
                      W: tl.constexpr, NCH: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr):
    """fast_chain's head_prep_kernel arithmetic from the last block's partial sums (mlp_down_live_kernel HEADP) instead
    of a pass over the moves: per row inv' = rsqrt(|xL + d|^2 / W + eps), inv' - inv0, the target's capped logit change
    in difference form, and the bound on |u| (head_fix_kernel's trigger)."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BR + tl.arange(0, BR)
    ok = rows < n_rows
    row = (t * D_ROWS + rows).to(tl.int64)
    pidx = (row[:, None] * NCH + tl.arange(0, NCH)[None, :]) * 3
    dots = tl.sum(tl.load(PSI + pidx, ok[:, None], 0.), 1)
    tdot = tl.sum(tl.load(PSI + pidx + 1, ok[:, None], 0.), 1)
    sqd = tl.sum(tl.load(PSI + pidx + 2, ok[:, None], 0.), 1)
    sq = tl.load(SQXL + t)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dots + sqd) / W + EPS)
    dinv = inv - inv0
    by, t0 = tl.load(BY + t), tl.load(T0Y + t)
    dz = by * dinv + tdot * inv
    tb = libdevice.tanh(dz / 15)
    dc = 15 * tb * (1 - t0 * t0) / (1 + t0 * tb)
    bound = (tl.load(MB + t) * tl.abs(dinv) + 1.01 * libdevice.sqrt(sqd) * tl.load(MW) * inv) / 15
    tl.store(STATS + row * 4, inv, ok)
    tl.store(STATS + row * 4 + 1, dinv, ok)
    tl.store(STATS + row * 4 + 2, dc, ok)
    tl.store(STATS + row * 4 + 3, bound, ok)


# ------------------------------------------------------------------------------------------------------ block steps


def oq_step(delta, n_oq, b, ctx):
    """The own-query response of rows < n_oq at block b into ctx['ybuf'] (bf16, before c_proj; prep_kernel adds it).
    impl oq 'part': proj_kernel (row scalars from the partial sums ctx['psi']) and fast_chain's oq_attn_kernel; 'fc2':
    fast_chain's proj_stats_kernel and oq_attn_kernel (its oq_fast without the c_proj launch); 'fc': fast_chain's
    oq_fast itself (c_proj and the add included)."""
    mode = STATE['impl']['oq']
    if mode == 'fc':
        return FC.oq_fast(delta, n_oq, b, ctx)
    m, rows, W = delta.shape
    q = ctx['routes'][b]
    keys = q['keys']
    H, T, DH = keys.shape
    out = ctx['ybuf'][:m * rows * W].view(m, rows, W)
    u = ctx['scratch32'][:m * rows * W].view(m, rows, W)
    stats = ctx['stats'][:m * rows * 2]
    bm, bn, bk, warps, stages = FC.OQ_PROJ
    if n_oq < bm:
        raise RuntimeError(f'fast_chain2: {n_oq} own-query rows per token < projection block {bm}')
    if mode == 'fc2':  # fast_chain's oq_fast, its first launch (copied; fc_pin)
        FC.proj_stats_kernel[(W // bn, triton.cdiv(m * n_oq, bm))](
            delta, ctx['inputs'][b][ctx['local']], ctx['sqx_in'][b], ctx['w_q_t'][b], u, stats, m * n_oq, n_oq, rows,
            rows, m, W, bm, bn, bk, EPS, num_warps=warps, num_stages=stages)
    else:
        proj_kernel[(W // bn, triton.cdiv(m * n_oq, bm))](
            delta, ctx['psi'], ctx['sqx_in'][b], ctx['w_q_t'][b], u, stats, m * n_oq, n_oq, rows, rows, m, W,
            W // DOWN[1], bm, bn, bk, EPS, num_warps=warps, num_stages=stages)
    bp, warps, stages = FC.OQ_ATTN
    FC.oq_attn_kernel[(H, m)](
        u, stats, q['r0'], q['sq0'], ctx['cos'], ctx['sin'], keys, q['values'], q['idx'], q['p'], q['o'], out,
        n_oq, rows, rows, ctx['t0'], T, H, DH, W, q['idx'].shape[-1], bp, EPS, num_warps=warps, num_stages=stages)


def mlp_step(delta, n_oq, n_mlp, b, ctx, dx0, P, root):
    """Block b after the own-query step: prep_kernel (c_proj + residual of rows < n_oq unless impl oq 'fc'; root b's
    rows formed (root: their steps, basis, prev, x0, h); the MLP's partial sums), the pruned route replay
    (mlp_up_live_kernel; mlp_down_live_kernel, which also applies block b + 1's mixing scalars (dx0 None: last block)
    and writes block b + 1's input partial sums). impl mlp 'full': the same kernels without the bound."""
    m, rows, W = delta.shape
    impl = STATE['impl']
    idx, hs, sc, mid = ctx['idx32'][b], ctx['hs'][b], ctx['sc'][b], ctx['mids'][b]
    K = idx.shape[1]
    nxt = b + 1 < len(ctx['inputs'])
    xn = ctx['inputs'][b + 1][ctx['local']] if nxt else delta
    scale = dx0 is not None
    prune = impl['mlp'] == 'live' and not STATE['fallback']
    rbuf = ctx['ybuf'][:m * rows * K].view(m, rows, K)
    bm, bn, bk, warps, stages = PREP
    ps = ctx['ps'][:m * rows * (W // bn) * 2]
    steps, basis, prev, x0r, h = root
    prep_kernel[(W // bn, triton.cdiv(n_mlp, bm), m)](
        ctx['ybuf'], ctx['w_proj_t'][b], delta, mid, ps, steps, basis, prev, x0r, h, n_oq, n_mlp, P, rows, rows, W, bm,
        bn, bk, impl['oq'] != 'fc', True, num_warps=warps, num_stages=stages)
    ubp, ubn, ubk, warps, stages = UP
    n_rb = triton.cdiv(n_mlp, ubp)
    cut = ctx['cut'][:m * n_rb]
    up_args = (delta, ps, mid, sc, idx, ctx['w_fc'][b], hs, cut, rbuf, n_mlp, rows, rows, n_rb, W, K,
               triton.next_power_of_2(K), W // bn, ubp, ubn, ubk, EPS, MARGIN)
    mlp_up_live_kernel[(n_rb, m)](*up_args, prune, not prune, num_warps=warps, num_stages=stages)
    check = STATE['check_now']
    if check is not None and check[0] == b and ctx['t0'] == 0 and prune:
        running_check(up_args, cut, rbuf, n_mlp, n_rb, b, check[1])
    if STATE['cut_sum'] is not None and prune:
        STATE['cut_sum'][b] += cut.sum(dtype=torch.float64) / K * ubp
        STATE['cut_rows'][b] += n_rb * m * ubp
    dbp, dbn, dbk, warps, stages = DOWN
    headp = not nxt  # the last block: the head's inputs instead of the fp32 moves (head_step reads them)
    mlp_down_live_kernel[(W // dbn, triton.cdiv(n_mlp, dbp), m)](
        rbuf, idx, ctx['w_out_t'][b], delta, dx0 if scale else delta, ctx['lams'][min(b + 1, len(ctx['lams']) - 1)],
        cut, xn, ctx['hps'] if headp else ctx['psi'], ctx['scratch'], ctx['xL'], ctx['target_w'], n_mlp, rows, rows,
        P, n_rb, W, K, dbp, dbn, dbk, ubp, scale, scale, not prune, nxt, headp, num_warps=warps, num_stages=stages)


def head_step(m, rows, W, ctx):
    """(loss(moved) - loss(clean)) per row: fast_chain's head_fast with its head_prep_kernel replaced by
    head_stats_kernel on the last block's partial sums (the bf16 copy of the moves is already in ctx['scratch']);
    fast_chain's head_kernel and head_fix_kernel as they are (copied driver; fc_pin)."""
    weight_t = ctx['weight_t']
    V = weight_t.shape[1]
    xL, base, y = ctx['xL'], ctx['base'], ctx['y']
    sq = xL.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / W + EPS)
    z0 = base * inv0
    t0 = torch.tanh(z0 / 15)
    p0 = torch.softmax(15 * t0, -1)
    target = y.clamp(min=0).unsqueeze(1)
    a = ctx['scratch'][:m * rows * W].view(m, rows, W)
    stats = ctx['scratch32'][:m * rows * 4]
    br = 64
    head_stats_kernel[(triton.cdiv(rows, br), m)](
        ctx['hps'], sq, base.gather(1, target).contiguous(), t0.gather(1, target).contiguous(),
        base.abs().amax(-1).contiguous(), ctx['head_row_norm'], stats, rows, rows, W, W // DOWN[1], br, EPS,
        num_warps=4)
    stats = stats.view(m, rows, 4)
    bm, bn, bk, warps, stages = FC.HEAD
    part = torch.empty(m * rows, V // bn, device=a.device)
    FC.head_kernel[(V // bn, triton.cdiv(rows, bm), m)](
        a, weight_t, stats, base, t0, p0, part, rows, rows, V, W, bm, bn, bk, ctx['exact_head'],
        num_warps=warps, num_stages=stages)
    if not ctx['exact_head']:  # row blocks outside the tanh series' range (if any): libdevice instead
        FC.head_fix_kernel[(m,)](a, weight_t, stats, base, t0, p0, part, ctx['exact_blocks'], rows, rows,
                                 ctx['series_limit'], V, W, bm, bn, bk, num_warps=warps, num_stages=1)
    return torch.log1p(part.sum(-1)).view(m, rows) - stats[..., 2]


def running_check(up_args, cut, rbuf, n_mlp, n_rb, b, offset):
    """Replay CHECK_TOKENS tokens of this chunk's block b unpruned (mlp_up_live_kernel FULL) next to the pruned replay:
    units at or beyond a row block's prune point must be exactly 0 (the bound's claim) and every unit the pruned replay
    formed must equal the unpruned one bitwise."""
    m, rows, K = rbuf.shape
    T = min(CHECK_TOKENS, m)
    t0 = offset % max(1, m - T + 1)
    ubp, ubn, ubk, warps, stages = UP
    full = torch.empty(T, rows, K, device=rbuf.device, dtype=torch.bfloat16)
    delta, ps, mid, sc, idx, w_fc, hs = up_args[:7]
    nch = up_args[16]
    sl = slice(t0, t0 + T)
    mlp_up_live_kernel[(n_rb, T)](
        delta[sl], ps.view(m, rows * nch * 2)[sl], mid[sl], sc[sl], idx[sl], w_fc, hs[sl], cut, full, n_mlp, rows,
        rows, n_rb, *up_args[13:], False, True, num_warps=warps, num_stages=stages)
    if STATE['inject'] is not None:  # selftest: a unit beyond the prune point that changes
        full[0, 0, K - 1] = 1.0
    cuts = cut.view(m, n_rb)[sl].repeat_interleave(ubp, dim=1)[:, :n_mlp].long()  # (T, n_mlp)
    pos = torch.arange(K, device=rbuf.device)
    formed = pos[None, None] < (-(-cuts // ubn) * ubn)[..., None]
    dead = pos[None, None] >= cuts[..., None]
    fr, pr = full[:, :n_mlp], rbuf[sl, :n_mlp]
    dead_nonzero = int(((fr != 0) & dead).sum().item())
    mismatch = int(((fr != pr) & formed).sum().item())
    item = dict(call=STATE['n'], block=b, tokens=[int(t0), int(t0 + T)], rows=int(n_mlp),
                live_fraction=round(cuts.float().mean().item() / K, 4), dead_nonzero=dead_nonzero,
                formed_mismatch=mismatch, ok=dead_nonzero == 0 and mismatch == 0)
    STATE['checks'].append(item)
    if not item['ok']:
        STATE['fallback'] = True
        rank = os.environ.get('RANK', '0')
        if STATE['output'] is not None and (STATE['output'] / 'config.json').exists():
            (STATE['output'] / f'fast_chain2_violation_rank{rank}.json').write_text(json.dumps(item) + '\n')
        print(f'fast_chain2 rank {rank}: running check failed ({item}); pruning off from now on', flush=True)


def sort_routes(w_fc, routes):
    """Per block: each token's routed units ordered by s = h / |W_fc[u]| (bf16 weights; active or NaN: +inf),
    descending, by fp16 keys (stable): (idx int32, h fp32, s fp32), each (n, K) contiguous. Any order is exact (a row
    block's prune point is one past the LAST unit it may change); a good order only makes the live prefix short."""
    idx32, hs_all, sc_all = [], [], []
    for w, (idx, hs) in zip(w_fc, routes):
        wn = w.float().norm(dim=1)
        hs = hs.float()
        s = torch.where((hs >= 0) | hs.isnan(), torch.full_like(hs, float('inf')), hs / wn[idx])
        order = s.half().argsort(dim=1, descending=True, stable=True)
        idx32.append(idx.gather(1, order).to(torch.int32).contiguous())
        hs_all.append(hs.gather(1, order).contiguous())
        sc_all.append(s.gather(1, order).contiguous())
    return idx32, hs_all, sc_all


def why_not(est, routes):
    """Why this file's kernels do not cover est's chain (that call runs fast_chain's chain_credits), or None."""
    R = STATE['installed']
    if FC.STATE['installed'] is not R:
        return 'fast_chain is not installed on this recipe'
    if any(v != 'triton' for v in FC.STATE['impl'].values()):
        return f"fast_chain impl {FC.STATE['impl']}"
    reason = FC.unsupported(est) or FC.unsupported_routes(routes)
    if reason is not None:
        return 'fast_chain: ' + reason
    widths = sorted({r[0].shape[1] for r in routes})
    if len(widths) != 1 or widths[0] % UP[1] or widths[0] % DOWN[2] or widths[0] > 4096:
        return f'MLP routes of {widths} units (the kernels: one width, a multiple of {UP[1]}, at most 4096)'
    width = est.model.config.n_embd
    if UP[0] % DOWN[0] or UP[1] % DOWN[2] or width % PREP[1] or width % DOWN[1]:
        return f'tiles PREP {PREP} / UP {UP} / DOWN {DOWN}'
    return None


def chain_credits(self, e, x0, mids, outs, routes, y):
    """fast_chain's chain_credits (all roots of a token chunk together, rows (root, probe) per token, the record's draw
    order and collectives) with this file's block steps: oq_step, then root b's rows, then mlp_step; the head is
    fast_chain's head_fast. A call these kernels do not cover runs fast_chain's chain_credits."""
    STATE['n'] += 1
    reason = why_not(self, routes)
    if reason is None:
        if FC.STATE['verified'] is None:
            FC.STATE['verified'] = FC.mismatch(self.model, mids, routes, self.width) or 'ok'
        if FC.STATE['verified'] != 'ok':
            reason = 'fast_chain: ' + FC.STATE['verified']
    path = reason or 'fast2'
    if reason is not None:
        note(path)
        return STATE['fc_chain'](self, e, x0, mids, outs, routes, y)
    if FC.STATE['exact_blocks'] is None:
        FC.STATE['exact_blocks'] = torch.zeros(1, device=self.device, dtype=torch.int32)
    if STATE['cut_sum'] is None:
        L0 = self.model.config.n_layer
        STATE['cut_sum'] = torch.zeros(L0, device=self.device, dtype=torch.float64)
        STATE['cut_rows'] = [0] * L0
    n_call = STATE['n']
    checking = (n_call == 1 or n_call % CHECK_EVERY == 0) and STATE['impl']['mlp'] == 'live' and not STATE['fallback']
    STATE['check_now'] = ((n_call // CHECK_EVERY) % self.model.config.n_layer, 97 * (n_call // CHECK_EVERY)) \
        if checking else None
    model, s = self.model, self.s
    L = model.config.n_layer
    vocab = model.config.vocab_size
    W = self.width
    weight = model.lm_head.weight.float()[:vocab]
    lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
    mids = [z.float().contiguous() for z in mids]
    xL = outs[-1].float().contiguous()
    base = xL @ weight.T
    target_w = weight[y.clamp(min=0)]
    n, h = e.shape[0], s.scalar_step
    lo, hi = self.share(W)
    P, scalars_here = hi - lo, self.part == 0
    bases, steps = [], []
    for site in range(-1, L):  # the record's draw order
        bases.append(self.basis()[lo:hi])
        probed = e if site < 0 else mids[site]
        steps.append(s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6))
    Ps = P + 4 * scalars_here
    rows = P + L * Ps
    first = [0] + [P + site * Ps for site in range(L)]
    credit = [torch.empty(n, W, device=self.device) for _ in range(L + 1)]
    lam = [torch.zeros(2, device=self.device) for _ in range(L)]
    weight_t = weight.T.to(torch.bfloat16).contiguous()
    common = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1), lam_r=lam_r,
                  lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(), exact_head=FC.STATE['exact_head'],
                  series_limit=FC.STATE['series_limit'], exact_blocks=FC.STATE['exact_blocks'])
    w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
    w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    w_q_t = [blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    w_proj_t = [blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
    idx32, hs_all, sc_all = sort_routes(w_fc, routes)
    length = self.capture.length
    m_chunk = min(STATE['chunk'] or s.chain_tokens, length)  # default: the record's chain_tokens
    units = max(r[0].shape[1] for r in routes)
    nch = W // DOWN[1]
    buffers = dict(scratch=torch.empty(m_chunk * rows * W, device=self.device, dtype=torch.bfloat16),  # head_fast
                   ybuf=torch.empty(m_chunk * rows * max(W, units), device=self.device, dtype=torch.bfloat16),
                   scratch32=torch.empty(m_chunk * rows * W, device=self.device),
                   stats=torch.empty(m_chunk * rows * 2, device=self.device),
                   ps=torch.empty(m_chunk * rows * (W // PREP[1]) * 2, device=self.device),
                   psi=torch.empty(m_chunk * rows * nch * 2, device=self.device),
                   hps=torch.empty(m_chunk * rows * nch * 3, device=self.device),
                   cut=torch.empty(m_chunk * triton.cdiv(rows, UP[0]), device=self.device, dtype=torch.int32))
    delta_all = torch.empty(m_chunk * rows * W, device=self.device)
    for start in range(0, n, length):
        own = self.own_query_inputs(start // length)
        qroutes = FC.query_routes(own, s.query_keys)
        inputs = [z.float().contiguous() for z in own[1]]
        seq = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=w_q_t, w_proj_t=w_proj_t, inputs=inputs,
                   sqx_seq=[z.square().sum(-1) for z in inputs])
        for t0 in range(start, start + length, m_chunk):
            t1 = min(t0 + m_chunk, start + length)
            sl, m = slice(t0, t1), t1 - t0
            local = slice(t0 - start, t1 - start)
            ctx = dict(common, **buffers, **seq, own=own, routes=qroutes, local=local, t0=t0 - start,
                       mids=[z[sl] for z in mids], hs=[z[sl] for z in hs_all], idx32=[z[sl] for z in idx32],
                       sc=[z[sl] for z in sc_all],
                       sqx_in=[z[local] for z in seq['sqx_seq']], xL=xL[sl], base=base[sl], target_w=target_w[sl],
                       y=y[sl], w_fc=w_fc, w_out_t=w_out_t)
            delta = delta_all[:m * rows * W].view(m, rows, W)
            dx0 = F.rms_norm(e[sl].unsqueeze(1) + steps[0][sl].unsqueeze(-1) * bases[0].unsqueeze(0), (W,)) \
                - x0[sl].unsqueeze(1)
            torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
            if STATE['impl']['oq'] == 'part':  # block 0's input partial sums (the embedding root's rows)
                psi = ctx['psi'][:m * rows * nch * 2].view(m, rows, nch, 2)
                psi[:, :P].zero_()
                psi[:, :P, 0, 0] = torch.einsum('mpd,md->mp', delta[:, :P], ctx['inputs'][0][local])
                psi[:, :P, 0, 1] = delta[:, :P].square().sum(-1)
            for b in range(L):
                n_oq = first[b + 1]  # rows of roots -1 .. b-1
                oq_step(delta, n_oq, b, ctx)
                # root b starts at its block's mid stream (rows n_oq ..), then (first GPU) the symmetric probes of
                # both linear mixing scalars: prep_kernel forms these rows as the record's driver does
                root = (steps[b + 1][sl], bases[b + 1], (x0 if b == 0 else outs[b - 1])[sl], x0[sl], h)
                mlp_step(delta, n_oq, n_oq + Ps, b, ctx, dx0 if b < L - 1 else None, P, root)
            change = head_step(m, rows, W, ctx)
            change = change * (y[sl] >= 0).unsqueeze(1)
            for site in range(-1, L):
                r0 = first[site + 1]
                credit[site + 1][sl] = (change[:, r0:r0 + P] / steps[site + 1][sl]) @ bases[site + 1] / W
                if site >= 0 and scalars_here:  # (d(+h) + d(-h)) / 2 per scalar: the central difference over 2h
                    pair = change[:, r0 + P:r0 + P + 4] / h
                    lam[site] += (pair[:, :2] - pair[:, 2:]).sum(0) / 2
        del own, qroutes, seq
    del buffers, delta_all
    STATE['check_now'] = None
    note(path, force=checking)
    credits = {}
    for site in range(-1, L):  # the record's collective order
        credits['emb' if site < 0 else site] = self.group_sum(credit[site + 1])
        if site >= 0:
            credits['lam', site] = self.group_sum(lam[site])
    return credits


# --------------------------------------------------------------------------------------------- install / bookkeeping


def fc_pin():
    """sha1 of the fast_chain code this file copies (the driver, the oq / MLP kernels and their tiles) or calls (query
    routes, the head, the support checks) and of fast_chain's own record pin."""
    pieces = (FC.chain_credits, FC.query_routes, FC.oq_fast, FC.proj_stats_kernel.fn, FC.oq_attn_kernel.fn,
              FC.linear_residual_kernel.fn, FC.mlp_fast, FC.mlp_up_kernel.fn, FC.mlp_down_kernel.fn, FC.head_fast,
              FC.head_prep_kernel.fn, FC.head_kernel.fn, FC.head_fix_kernel.fn, FC.unsupported,
              FC.unsupported_routes, FC.mismatch, FC.install)
    text = ''.join(map(inspect.getsource, pieces)) + repr(
        (FC.PIN, FC.RELIED, FC.OQ_PROJ, FC.OQ_ATTN, FC.PROJ, FC.MLP_UP, FC.MLP_DOWN, FC.HEAD_PREP, FC.HEAD))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def describe():
    try:
        fc = FC.describe()
    except Exception as error:  # never stop a run for the bookkeeping
        fc = repr(error)
    out = dict(installed=STATE['installed'] is not None, file=str(FILE), sha1=SHA1, fc_pin=STATE['pin'] or FC_PIN,
               impl=dict(STATE['impl']), chunk=STATE['chunk'], margin=MARGIN, calls=dict(STATE['calls']),
               fallback=STATE['fallback'], checks=STATE['checks'][-40:],
               checks_total=len(STATE['checks']), checks_failed=sum(not c['ok'] for c in STATE['checks']),
               fast_chain=fc, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               torch=torch.__version__, triton=triton.__version__)
    if STATE['cut_sum'] is not None:
        sums = STATE['cut_sum'].tolist()
        out['live_unit_fraction'] = {str(b): round(sums[b] / r, 4) if r else None
                                     for b, r in enumerate(STATE['cut_rows'])}
    return out


def note(path, force=False):
    """Count the path a chain call took; rank 0 prints a path the first time and, in a torchrun process, rewrites
    fast_chain2.json then, after each running check (force) and every NOTE_EVERY calls."""
    calls = STATE['calls']
    new = path not in calls
    calls[path] = calls.get(path, 0) + 1
    if new and os.environ.get('RANK', '0') == '0':
        print('fast_chain2: chain_credits runs ' + ('the chain2 kernels' if path == 'fast2' else f'fast_chain: {path}'),
              flush=True)
    if new or force or sum(calls.values()) % NOTE_EVERY == 0:
        write_note()


def write_note():
    output = STATE['output']
    if os.environ.get('RANK') == '0' and output is not None and (output / 'config.json').exists():
        (output / 'fast_chain2.json').write_text(json.dumps(describe(), indent=1) + '\n')


def install(R=None, enabled=None, check_pin=True):
    """Swap Cut.chain_credits (fast_chain's, which must be installed on R first) for this file's; process-wide,
    idempotent. enabled: default $FAST_CHAIN2 ('' / '0' / unset = nothing is replaced: the current stack, bitwise).
    Returns R."""
    R = sys.modules['recipe'] if R is None else R
    if enabled is None:
        enabled = False
    uninstall(R)
    STATE.update(installed=None)
    if not enabled:
        return R
    if FC.STATE['installed'] is not R or R.Cut.chain_credits is not FC.chain_credits:
        raise RuntimeError('fast_chain2: install fast_chain on this recipe first (its chain_credits is the fallback)')
    STATE['fc_chain'] = R.Cut.chain_credits
    R.Cut.chain_credits = chain_credits
    STATE.update(installed=R, pin=fc_pin())
    if os.environ.get('RANK') is not None:  # torchrun: fast_chain2.json goes into the run's --output
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', type=Path, default=Path('logs/val'))  # the recipe's default
        STATE['output'] = parser.parse_known_args(sys.argv[1:])[0].output
        if not STATE['exit_hook']:
            atexit.register(write_note)
            STATE['exit_hook'] = True
    return R


def uninstall(R):
    if R.Cut.chain_credits is chain_credits and STATE['fc_chain'] is not None:
        R.Cut.chain_credits = STATE['fc_chain']


# ------------------------------------------------------------------------------------------------------------ entries


# ------------------------------------------------------------------------------------------------- checks and timing


# ======================================== module readout
"""The true-path readout of blocks 0-3 and wte. The chain moves each token's stream at a root along a complete cycle;
those moved streams are pushed up the token's real path (own query on the other tokens' held keys / values, routed MLP,
the token's own key / value: the self-edge), and at every later block j the change of the token's key / value / gate
is turned into the exact single-column softmax change of every receiver s >= t in the window, scored against block j's
measured hub chain_j W_proj. Decoded over the cycle: R_r, the cross-token credit at root r along the true path, which
replaces the identity transport for the linear consumers (attn.c_proj, c_v / value embeddings, mixing scalars,
mlp.c_proj, wte). The q and c_fc source experiments are pushed and read out the same way. Never a scoring target.
"""
import dataclasses
from dataclasses import dataclass
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import onesided  # noqa: E402  (loads research/pr31/run_recipe.py as module 'recipe' unless a wrapper already has)
R = sys.modules['recipe']
import fast_chain as FC  # noqa: E402  (its kernels only; installing it is the speed stack's job)
import fast_moments as FM  # noqa: E402  (its future tables / prologue only)

BLOCKS = (0, 1, 2, 3)  # the corrected blocks (= the future probes' source blocks)
RECORD_SETTINGS, RECORD_ASDICT, CUT_ESTIMATE = onesided.Settings, onesided.asdict, R.Cut.estimate
EPS = torch.finfo(torch.float32).eps
STATS = dict(calls=0, every=200, keep=False, seconds=[])  # keep: checks read the last estimate's pieces


# ------------------------------------------------------------------------------------------------- the scorer kernel


BPR = 64  # rows per program


@triton.jit
def readout_kernel(PJ, R0, CS, SN, GC, VE, U, WV, M1, M2, QD, GD, AD, CD, VT, OT, Out, SE, Start, P,
                   CSM, CSF, SNM, SNF, H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, DIAG: tl.constexpr,
                   SELF: tl.constexpr, BP: tl.constexpr):
    """One (token t of the chunk, BP of its P rows): the readout score summed over heads, Out [m, P], and (SELF) the
    token's own attention output change per head SE [m, P, H * D] (bf16). fast_moments.future_kernel (prologue from the
    projected K / V change PJ [m, P, 2, H, D], the rotated clean raw key R0 [m, H, D], rope rows CS / SN, the gate change
    GC [m, P, H] times the value embedding VE [m, H, D]; receivers t < s <= t + window by the moment tables U, WV, M1, M2
    at row Start + t), plus the receiver s = t exactly: x = q_t . dk / sqrt(D), a = A[t, t], change of t's output
    (a (e^x - 1) (v_t - o_t) + a e^x dv) / (1 + a (e^x - 1)) (QD = q_t, VT = v_t, OT = o_t [T, H, D]; AD [T, H]); its
    score against the hub GD = target_t [T, H, D] is (a (e^x - 1) c + a e^x g . dv) / (1 + a (e^x - 1)), c = CD [T, H]."""
    t, pb = tl.program_id(0), tl.program_id(1)
    p = pb * BP + tl.arange(0, BP)
    pm = p < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    cs = tl.load(CS + t * CSM + half * CSF)
    sn = tl.load(SN + t * SNM + half * SNF)
    total = tl.zeros((BP,), tl.float32)
    tok = Start + t
    for h in range(H):
        base = ((t * P + p[:, None]) * 2 * H + h) * D
        pk = tl.load(PJ + base + f[None, :], pm[:, None], 0.)
        pk_other = tl.load(PJ + base + other[None, :], pm[:, None], 0.)
        dr = tl.where(low[None, :], pk * cs[None, :] + pk_other * sn[None, :], -pk_other * sn[None, :] + pk * cs[None, :])
        r0 = tl.load(R0 + (t * H + h) * D + f)
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        dv = tl.load(PJ + base + H * D + f[None, :], pm[:, None], 0.)
        if GATED:
            gc = tl.load(GC + (t * P + p) * H + h, pm, 0.)
            dv = dv + gc[:, None] * tl.load(VE + (t * H + h) * D + f)[None, :]
        row = tok * H + h
        if DIAG or SELF:  # first (dk / dv die sooner): the receiver s = t, exactly
            x = tl.sum(dk * tl.load(QD + row * D + f)[None, :], 1) * (D ** -.5)
            a = tl.load(AD + row)
            e = libdevice.expm1(x)
            gain = a * e / (1. + a * e)
            vgain = (a + a * e) / (1. + a * e)
            if DIAG:
                total += gain * tl.load(CD + row) + vgain * tl.sum(dv * tl.load(GD + row * D + f)[None, :], 1)
        dkb = dk.to(tl.bfloat16)
        total += (tl.sum(dk * tl.load(U + row * D + f)[None, :], 1) * (D ** -.5)
                  + tl.sum(dv * tl.load(WV + row * D + f)[None, :], 1)
                  + tl.sum(tl.dot(dkb, tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])) * dk, 1) * (1. / D)
                  + tl.sum(tl.dot(dkb, tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])) * dv, 1) * (D ** -.5))
        if SELF:  # last: the token's own output change (bf16)
            vo = tl.load(VT + row * D + f) - tl.load(OT + row * D + f)
            tl.store(SE + (t * P + p[:, None]).to(tl.int64) * (H * D) + h * D + f[None, :],
                     (gain[:, None] * vo[None, :] + vgain[:, None] * dv).to(tl.bfloat16), pm[:, None])
    tl.store(Out + t * P + p, total, pm)


@triton.jit
def stats_kernel(Dm, X, SQX, DB, ST, Start, n, D_ROWS, W: tl.constexpr, BP: tl.constexpr, EPS: tl.constexpr):
    """Stage 1 of the two-stage scorer: per row of the first n rows of every token of the delta Dm [m, D_ROWS, W], its
    bf16 copy DB [m, n, W] (the projection GEMM's input) and ST [m, n, 2] = (inv', inv' / inv0 - 1) of block j's input
    RMSNorm at x + d (X [T, W], SQX = |x|^2, token Start + t)."""
    pb, t = tl.program_id(0), tl.program_id(1)
    tok = Start + t
    p = pb * BP + tl.arange(0, BP)
    pm = p < n
    k = tl.arange(0, W)
    d = tl.load(Dm + (t * D_ROWS + p).to(tl.int64)[:, None] * W + k[None, :], pm[:, None], 0.)
    dotx = tl.sum(d * tl.load(X + tok * W + k)[None, :], 1)
    sqd = tl.sum(d * d, 1)
    tl.store(DB + (t * n + p).to(tl.int64)[:, None] * W + k[None, :], d.to(tl.bfloat16), pm[:, None])
    sq = tl.load(SQX + tok)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    tl.store(ST + (t * n + p) * 2, inv, pm)
    tl.store(ST + (t * n + p) * 2 + 1, (inv - inv0) / inv0, pm)


@triton.jit
def staged_kernel(UP, ST, R0, VC, GZ, VE, CS, SN, U, WV, M1, M2, QD, GD, AD, CD, VT, OT, Out, SE, Start, n, SE_ROWS,
                  W: tl.constexpr, H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, DIAG: tl.constexpr,
                  SELF: tl.constexpr, BP: tl.constexpr, NC: tl.constexpr):
    """Stage 3: readout_kernel's arithmetic from the bf16 projection of the raw moves UP [m, n, NC] (columns: d W_k^T,
    d W_v^T, then d[:32] W_g^T padded to 16) and the row statistics ST: the projected change of the normed input is
    inv' (d W^T) + (inv' / inv0 - 1) (clean projection: R0 rotated raw key [T, H, D], VC raw value [T, W], GZ gate
    logits [T, H])."""
    pb, t = tl.program_id(0), tl.program_id(1)  # row blocks fastest: a token's tables stay in L2
    tok = Start + t
    p = pb * BP + tl.arange(0, BP)
    pm = p < n
    f = tl.arange(0, D)
    other = (f + D // 2) % D
    half = f % (D // 2)
    cfull = tl.load(CS + tok * (D // 2) + half)
    ssig = tl.load(SN + tok * (D // 2) + half) * tl.where(f < D // 2, 1., -1.)
    inv = tl.load(ST + (t * n + p) * 2, pm, 1.)
    coef = tl.load(ST + (t * n + p) * 2 + 1, pm, 0.)
    urow = (t * n + p).to(tl.int64) * NC
    g16 = tl.arange(0, 16)
    if GATED:
        ug = tl.load(UP + urow[:, None] + 2 * W + g16[None, :], pm[:, None], 0.).to(tl.float32)
        gz = tl.load(GZ + tok * H + g16, g16 < H, 0.)
        gch = 2 * (tl.sigmoid(gz[None, :] + inv[:, None] * ug + coef[:, None] * gz[None, :]) - tl.sigmoid(gz)[None, :])
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        uk = tl.load(UP + urow[:, None] + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        uks = tl.load(UP + urow[:, None] + h * D + other[None, :], pm[:, None], 0.).to(tl.float32)
        uv = tl.load(UP + urow[:, None] + W + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        row = tok * H + h
        r0 = tl.load(R0 + row * D + f)
        dr = inv[:, None] * (uk * cfull[None, :] + uks * ssig[None, :]) + coef[:, None] * r0[None, :]
        dv = inv[:, None] * uv + coef[:, None] * tl.load(VC + tok * W + h * D + f)[None, :]
        if GATED:
            gc = tl.sum(gch * (g16 == h).to(tl.float32)[None, :], 1)
            dv = dv + gc[:, None] * tl.load(VE + row * D + f)[None, :]
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        if DIAG or SELF:  # the receiver s = t, exactly
            x = tl.sum(dk * tl.load(QD + row * D + f)[None, :], 1) * (D ** -.5)
            a = tl.load(AD + row)
            e = libdevice.expm1(x)
            gain = a * e / (1. + a * e)
            vgain = (a + a * e) / (1. + a * e)
            if DIAG:
                total += gain * tl.load(CD + row) + vgain * tl.sum(dv * tl.load(GD + row * D + f)[None, :], 1)
            if SELF:
                vo = tl.load(VT + row * D + f) - tl.load(OT + row * D + f)
                tl.store(SE + (t * SE_ROWS + p[:, None]).to(tl.int64) * W + h * D + f[None, :],
                         (gain[:, None] * vo[None, :] + vgain[:, None] * dv).to(tl.bfloat16), pm[:, None])
        dkb = dk.to(tl.bfloat16)
        total += (tl.sum(dk * tl.load(U + row * D + f)[None, :], 1) * (D ** -.5)
                  + tl.sum(dv * tl.load(WV + row * D + f)[None, :], 1)
                  + tl.sum(tl.dot(dkb, tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])) * dk, 1) * (1. / D)
                  + tl.sum(tl.dot(dkb, tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])) * dv, 1) * (D ** -.5))
    tl.store(Out + t * n + p, total, pm)


PROLOGUE = {}
IMPL = dict(score='staged')  # 'staged' | 'torch' (checks: fast_moments' compiled prologue + readout_kernel)
PROFILE = {}  # checks: PROFILE['on'] = True accumulates synchronized seconds per phase of the pass in PROFILE['t']


def tick(name):
    if PROFILE.get('on'):
        torch.cuda.synchronize()
        now = time.perf_counter()
        last = PROFILE.get('last', now)
        PROFILE.setdefault('t', {})
        if name is not None:
            PROFILE['t'][name] = PROFILE['t'].get(name, 0.) + now - last
        PROFILE['last'] = now


def prologue():
    if 'f' not in PROLOGUE:
        PROLOGUE['f'] = FM.future_prologue(R)
    return PROLOGUE['f']


# ------------------------------------------------------------------------------------------------------- the settings


@dataclass
class Settings(RECORD_SETTINGS):  # the record's fields and onesided's plus the method's
    readout: bool = False  # research (D114): the chain's true-path readout (blocks 0-3 + wte) and the q / c_fc source
    #   experiments replace the identity transport and FiniteFuture; false = the current stack
    readout_sources: bool = True  # q / c_fc source experiments (false: c_q / c_fc keep only the record's own-path credit)
    readout_q_keys: int = 16  # q sources of blocks 1-3: per head the subspace moving its this many strongest keys' logits
    readout_q_full: tuple = (0,)  # blocks whose q source runs the complete 512-direction cycle instead
    readout_fc_cycle: tuple = (512, 128, 128, 128)  # c_fc source cycle per block 0-3 (cycle - 1 strongest active units)
    readout_self: bool = True  # the readout pass carries each move through the token's own key / value too (self-edge)
    readout_chunk: int = 512  # tokens per readout chunk (memory: ~12 MB per token at 2,048 rows)
    readout_sigma: float = 0.01  # q / c_fc source steps (relative RMS of the raw query / hidden pre-activations)

    def __post_init__(self):
        super().__post_init__()
        if type(self.readout) is not bool:
            raise ValueError('readout is true or false')
        if not self.readout:
            return
        self.readout_q_full = tuple(self.readout_q_full)
        self.readout_fc_cycle = tuple(self.readout_fc_cycle)
        cycles = self.readout_fc_cycle
        heads, width = self.model.n_head, self.model.n_embd
        if (self.estimator != 'cut' or not (self.value_path and self.key_path_draws and self.gate_path)
                or self.cross_alpha != 1 or self.cross_alpha_layers
                or self.future_layers != len(BLOCKS) or self.model.n_layer <= len(BLOCKS) or not self.own_query
                or self.chain_precision != 'bf16' or width != 512 or width // heads != 64
                or type(self.readout_sources) is not bool or type(self.readout_self) is not bool
                or type(self.readout_q_keys) is not int or not 1 <= self.readout_q_keys <= 64
                or (heads * self.readout_q_keys) & (heads * self.readout_q_keys - 1)
                or not set(self.readout_q_full) <= set(BLOCKS) or len(cycles) != len(BLOCKS)
                or any(type(c) is not int or c < 32 or c & (c - 1) or c > 4 * width for c in cycles)
                or type(self.readout_chunk) is not int or self.readout_chunk < 16
                or not 0 < self.readout_sigma < float('inf')):
            raise ValueError('readout: Cut with value / input-K / gate paths, cross_alpha 1, no per-block alphas, '
                             'future_layers 4, own query, bf16 chain, width 512 with 8 heads of 64; '
                             'n_head * readout_q_keys a power of two, readout_q_full within blocks 0-3, four c_fc cycles '
                             '(powers of two, 32..2048), readout_chunk >= 16, finite positive readout_sigma')


def asdict(settings):  # off: the recipe dict (config.json, resume) is the current stack's
    state = RECORD_ASDICT(settings)
    if not getattr(settings, 'readout', False):
        for key in [k for k in state if k.startswith('readout')]:
            state.pop(key)
    return state


# ------------------------------------------------------------------------------------------------------- the method


def orthonormal(B):
    """Rows of B [N, K, D] -> orthonormal rows spanning the same space (modified Gram-Schmidt, twice per row; a row
    already in the span of the earlier ones becomes zero, so its code slot moves nothing and decodes nothing)."""
    Q = torch.zeros_like(B)
    scale = B.norm(dim=-1).amax(-1, keepdim=True).clamp_min(1e-30)
    for k in range(B.shape[1]):
        v = B[:, k]
        for _ in range(2 if k else 0):
            v = v - torch.einsum('nk,nkd->nd', torch.einsum('nkd,nd->nk', Q[:, :k], v), Q[:, :k])
        norm = v.norm(dim=-1, keepdim=True)
        Q[:, k] = torch.where(norm > 1e-4 * scale, v / norm.clamp_min(1e-30), torch.zeros_like(v))
    return Q


def layout(s, P, P_q, P_f):
    """Rows per token on this GPU, in the order they start: emb | root 0, q 0, fc 0 | root 1, q 1, fc 1 | ... | fc 3.
    Returns the groups {name: slice}, in_rows[b] (rows moved at block b's input) and mid_rows[b] (rows through block
    b's MLP)."""
    groups, at = {'emb': slice(0, P)}, P
    in_rows, mid_rows = [], []
    for b in range(s.model.n_layer):
        in_rows.append(at)
        if b in BLOCKS:
            groups[f'r{b}'] = slice(at, at + P)
            groups[f'q{b}'] = slice(at + P, at + P + P_q[b])
            at += P + P_q[b]
            mid_rows.append(at)
            groups[f'f{b}'] = slice(at, at + P_f[b])
            at += P_f[b]
        else:
            mid_rows.append(at)
    return groups, in_rows, mid_rows, at


class Method:
    """One Cut estimate with the readout: the record's Cut.estimate runs as is with two instance hooks (removed after the
    call): chain_credits (the installed chain, its bases recorded) and path_credits (the record's measured input credits,
    then the readout pass; the returned extra carries the readout credit of blocks 0-3 in place of the identity transport,
    which routes it to attn.c_proj, V / VE and the mixing scalars), FiniteFuture off for the call; then mlp.c_proj of
    blocks 0-3 and wte (a pair's first GPU) and the sources' c_q / c_fc credit (every GPU, its share of the cycles)."""

    def __init__(self, est):
        self.e = est

    def estimate(self, x, y):
        e, s = self.e, self.e.s
        e.s = dataclasses.replace(s, future_draws=0)  # FiniteFuture off for this call (replaced by readout + sources)
        hooks = dict(chain_credits=self.chain_credits, path_credits=self.path_credits)
        e.__dict__.update(hooks)
        try:
            gradients = CUT_ESTIMATE(e, x, y)
        finally:
            e.s = s
            for name in hooks:
                e.__dict__.pop(name, None)
        self.finish(gradients)
        self.note()
        return gradients

    # --------------------------------------------------------------------------------------------------- hooks

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        """The installed chain (fast_chain's or the record's), as is; its bases recorded (the same draws)."""
        est, bases = self.e, []
        plain = type(est).basis

        def basis():
            H = plain(est)
            bases.append(H)
            return H
        est.basis = basis
        try:
            chain = type(est).chain_credits(est, emb, x0, mids, outs, routes, y)
        finally:
            est.__dict__.pop('basis', None)
        if len(bases) != est.model.config.n_layer + 1:
            raise RuntimeError(f'readout: the chain drew {len(bases)} bases, expected one per root')
        self.chain_args, self.bases = (emb, x0, mids, outs, routes, y), bases
        return chain

    def path_credits(self, chain, targets, raw_values=None):
        """The record's measured input credits and identity extra, then the readout pass; extra_i of blocks 0-3 becomes
        the readout credit R_i (the record then forms x_mid = chain_i + R_i: attn.c_proj and the V target, and the mixing
        scalars' input-path credit inputs_i + R_i)."""
        est = self.e
        extra, inputs = type(est).path_credits(est, chain, targets, raw_values)
        start = time.perf_counter()
        self.chain, self.extra, self.inputs = chain, extra, inputs
        if STATS['keep']:  # checks: the measured hubs too
            self.targets = targets
        self.run(chain, targets)
        out = dict(extra)
        for i in BLOCKS:
            out[i] = self.X[i]
        STATS['seconds'].append(time.perf_counter() - start)
        return out, inputs

    # ------------------------------------------------------------------------------------------------ finishing

    def finish(self, gradients):
        e, d = self.e, self.e.capture
        lam_r = e.model.resid_lambdas.float()
        torch.randint(2 ** 31, (), device=e.device,  # add_future_gradients' one draw: the next steps' record draws as before
                      generator=d.shared_generator if e.split > 1 else d.generator)
        if not e.part:  # formed in full by a pair's first GPU (the record's rule; the other keeps zeros)
            for i in BLOCKS:
                p = f'transformer.h.{i}.'
                cross = lam_r[i + 1] * (self.inputs[i + 1] + self.X[i + 1]) if i + 1 in BLOCKS else self.extra[i]
                x_out = lam_r[i + 1] * self.chain[i + 1] + cross
                gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            gradients['transformer.wte.weight'] += d.reassemble(
                'transformer.wte', self.X['emb'].view(d.batch, d.length, e.width))
        for i, credit in self.q_credit.items():  # every GPU: its share of the cycles (the pair's sum is the decode)
            p = f'transformer.h.{i}.attn.c_q'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))
        for i, credit in self.fc_credit.items():
            p = f'transformer.h.{i}.mlp.c_fc'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))

    def note(self):
        STATS['calls'] += 1
        n = STATS['calls']
        if os.environ.get('RANK', '0') != '0' or (n != 1 and n % STATS['every']):
            return
        cos = lambda a, c: ((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-30)).item()
        rows = {i: dict(cos=round(cos(self.X[i], self.extra[i]), 3),
                        ratio=round((self.X[i].norm() / self.extra[i].norm().clamp_min(1e-30)).item(), 3)) for i in BLOCKS}
        sec = STATS['seconds'][-50:]
        print(f'readout estimate {n}: readout vs identity extra per block {json.dumps(rows)}; |R_emb| / |chain_emb| '
              f"{(self.X['emb'].norm() / self.chain['emb'].norm().clamp_min(1e-30)).item():.3f}; "
              f'pass {sum(sec) / len(sec):.3f} s', flush=True)

    # ------------------------------------------------------------------------------------------------- the pass

    def contexts(self, seq, targets):
        """Per block j of sequence seq: FiniteFuture.prepare's clean quantities (the record's formulas) and from them the
        moment tables (fast_moments) and the s = t pieces; for blocks 0-3 also what the q sources need."""
        e = self.e
        d, model = e.capture, e.model
        T, H = d.length, model.config.n_head
        D = e.width // H
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        ctxs = []
        for j, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{j}.attn.'
            qraw, kraw, vraw = [d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k', 'c_v')]
            v = vraw.reshape(1, T, H, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix + 've_gate'][seq].float()
                ve = d.outputs[f'value_embeds.{j}'][seq].float().reshape(T, H, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
            values = v.transpose(1, 2)[0]
            window = model.window_sizes[j][0]
            qp, kp, a, out = R.attention_state(qraw, kraw, values[None], cos, sin, H, window)
            q, k, a, out = qp[0], kp[0], a[0], out[0]
            target = targets[j][seq].reshape(T, H, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2)) - (target * out).sum(-1, keepdim=True)
            U, W, M1, M2 = FM.build_future_tables(q, target, a, contrast, window)
            kr = kraw[0].reshape(T, H, D)
            ctx = dict(x=d.blocks[j][seq].float().contiguous(), kraw=kr.contiguous(), gate=gate,
                       R0=R.rotate(kr, cos[0, :, 0][:, None], sin[0, :, 0][:, None]).contiguous(),
                       VC=vraw[0].contiguous(), GZ=None if gate is None else gate.contiguous(),
                       ve=None if ve is None else ve.contiguous(), wg=wg,
                       wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(), U=U, W=W, M1=M1, M2=M2,
                       QD=q.transpose(0, 1).contiguous(), GD=target.transpose(0, 1).contiguous(),
                       AD=a.diagonal(0, 1, 2).T.contiguous(), CD=contrast.diagonal(0, 1, 2).T.contiguous(),
                       VT=values.transpose(0, 1).contiguous(), OT=out.transpose(0, 1).contiguous())
            if j in BLOCKS and e.s.readout_sources:
                ctx.update(qraw=qraw[0], k=k, values=values, a=a, out=out)
            ctx['sqx'] = ctx['x'].square().sum(-1).contiguous()
            ctxs.append(ctx)
            del contrast
        return ctxs

    def q_basis(self, ctx, keys):
        """Per (token, head) an orthonormal basis [T, H, D, keys] of the raw-query subspace that moves the logits of the
        head's `keys` strongest clean keys: R_t^T (I - r r^T) k_s (rotation back, the query norm's projection)."""
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        cs, sn = e.model.cos[0, :T, 0].float(), e.model.sin[0, :T, 0].float()
        top = ctx['a'].topk(keys, dim=-1).indices  # [H, T, keys]
        kk = ctx['k'][torch.arange(H, device=top.device)[:, None, None], top]  # [H, T, keys, D]
        r = R.rotate(ctx['qraw'].reshape(T, H, D), cs[:, None], sn[:, None]).transpose(0, 1)  # [H, T, D]
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        proj = kk - (kk * r[:, :, None]).sum(-1, keepdim=True) * r[:, :, None]
        back = R.rotate(proj, cs[None, :, None], -sn[None, :, None])  # R^T
        return orthonormal(back.permute(1, 0, 2, 3).reshape(T * H, keys, D)).reshape(T, H, keys, D).transpose(2, 3)

    def fc_slots(self, b, seq, cycle):
        """The c_fc source's units per token: the cycle - 1 strongest active units of block b (relu^2(h) |W_out col|, the
        record's route ranking); slot k + 1 of the cycle is unit idx[:, k] (slot 0, the all-ones Hadamard column, unused);
        valid: the unit is active."""
        d, model = self.e.capture, self.e.model
        h = d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float()
        w_out = model.transformer.h[b].mlp.c_proj.weight.float()
        rank, idx = (F.relu(h).square() * w_out.norm(dim=0)).topk(cycle - 1, dim=-1)
        return idx, h.gather(1, idx), rank > 0

    @staticmethod
    def draw_code(gen, T, n, device):
        """A random-sign complete Hadamard cycle of size n per token: (row permutation [n], column signs [T, n])."""
        perm = torch.randperm(n, device=device, generator=gen)
        signs = 2 * torch.randint(0, 2, (T, n), device=device, generator=gen).float() - 1
        return perm, signs

    @staticmethod
    def code_rows(code, n, lo, hi, local, device):
        """This GPU's rows lo:hi of the cycle for the tokens `local`: [m, hi - lo, n]."""
        perm, signs = code
        return R.hadamard(n, device, torch.float32)[perm[lo:hi]][None] * signs[local][:, None]

    def unsupported(self):
        """Why the fast_chain kernels the pass reuses cannot run this setting (None: they can)."""
        e, s, cfg = self.e, self.e.s, self.e.model.config
        for ok, reason in ((s.query_keys in (64, 128, 256), f'query_keys {s.query_keys}'),
                           (e.capture.length >= s.query_keys, 'sequences shorter than query_keys'),
                           (s.route_k % FC.MLP_UP[1] == 0, f'route_k {s.route_k}'),
                           (e.width // e.split >= FC.OQ_PROJ[0], f'{e.width // e.split} probes per GPU'),
                           (cfg.n_embd == 512 and cfg.n_embd // cfg.n_head == 64, 'width / head size'),
                           (next(e.model.parameters()).is_cuda, 'not on CUDA')):
            if not ok:
                return reason
        return None

    def run(self, chain, targets):
        e, s = self.e, self.e.s
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'readout: the fast_chain kernels do not cover this setting ({reason})')
        emb, x0, mids, outs, routes, y = self.chain_args
        model, d = e.model, e.capture
        L, W = model.config.n_layer, e.width
        H = model.config.n_head
        D = W // H
        T, nseq = d.length, d.batch
        dev = e.device
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P = hi - lo
        bases = [Hb[lo:hi].contiguous() for Hb in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb] + list(mids)]
        sources = s.readout_sources
        q_cycle = {b: W if b in s.readout_q_full else H * s.readout_q_keys for b in BLOCKS}
        f_cycle = dict(zip(BLOCKS, s.readout_fc_cycle))
        q_share = {b: e.share(q_cycle[b]) for b in BLOCKS}
        f_share = {b: e.share(f_cycle[b]) for b in BLOCKS}
        P_q = {b: q_share[b][1] - q_share[b][0] if sources else 0 for b in BLOCKS}
        P_f = {b: f_share[b][1] - f_share[b][0] if sources else 0 for b in BLOCKS}
        groups, in_rows, mid_rows, rows = layout(s, P, P_q, P_f)
        # the sources' directions: a seed from a private copy of the generator add_future_gradients draws from (the pair's
        # shared one: both GPUs alike); the record's draws stay as they are (finish() advances it once, as FiniteFuture did)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        w_q_t = [blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        w_proj_t = [blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        w_proj = [blk.attn.c_proj.weight.float() for blk in model.transformer.h]
        wkt = [blk.attn.c_k.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        wvt = [blk.attn.c_v.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        wgt = [None if blk.attn.ve_gate is None else F.pad(blk.attn.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16)
               .contiguous() for blk in model.transformer.h]
        NC = 2 * W + 16  # the staged scorer's projection: d W_k^T | d W_v^T | d[:32] W_g^T (padded to 16)
        wcat = [torch.cat([wkt[j], wvt[j], torch.zeros(W, 16, device=dev, dtype=torch.bfloat16) if wgt[j] is None
                           else F.pad(wgt[j], (0, 0, 0, W - 32))], 1).contiguous() for j in range(L)]
        mids = [z.float().contiguous() for z in mids]
        idx32 = [r[0].to(torch.int32).contiguous() for r in routes]
        hs_all = [r[1].float().contiguous() for r in routes]
        units = max(r[0].shape[1] for r in routes)
        m_chunk = min(s.readout_chunk, T)
        buffers = dict(scratch=torch.empty(m_chunk * rows * max(W, units), device=dev, dtype=torch.bfloat16),
                       scratch32=torch.empty(m_chunk * rows * W, device=dev),
                       stats=torch.empty(m_chunk * rows * 2, device=dev))
        delta_all = torch.empty(m_chunk * rows * W, device=dev)
        se_all = torch.empty(m_chunk * rows * W, device=dev, dtype=torch.bfloat16) if s.readout_self else None
        db_all = torch.empty(m_chunk * rows * W, device=dev, dtype=torch.bfloat16)  # staged scorer buffers
        st_all = torch.empty(m_chunk * rows * 2, device=dev)
        up_all = torch.empty(m_chunk * rows * NC, device=dev, dtype=torch.bfloat16)
        common = dict(lam_r=lam_r, lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(), w_fc=w_fc,
                      w_out_t=w_out_t)
        score = torch.zeros(nseq * T, rows, device=dev)
        keep = STATS['keep']
        per_block = torch.zeros(L, nseq * T, rows, device=dev) if keep else None
        q_credit = {b: torch.zeros(nseq * T, W, device=dev) for b in BLOCKS} if sources else {}
        fc_credit = {b: torch.zeros(nseq * T, 4 * W, device=dev) for b in BLOCKS} if sources else {}
        cs_all, sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()
        pro = prologue()
        one = torch.ones((), device=dev)
        tick(None)
        for seq in range(nseq):
            ctxs = self.contexts(seq, targets)
            tick('contexts')
            own = e.own_query_inputs(seq)
            qroutes = FC.query_routes(own, s.query_keys)
            inputs = [z.float().contiguous() for z in own[1]]
            seqd = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=w_q_t, w_proj_t=w_proj_t, inputs=inputs,
                        sqx_seq=[z.square().sum(-1) for z in inputs])
            src = {}
            if sources:
                gen = torch.Generator(dev).manual_seed(seed + 104729 * seq)
                for b in BLOCKS:
                    c = ctxs[b]
                    basis = None if q_cycle[b] == W else self.q_basis(c, s.readout_q_keys)
                    qstep = s.readout_sigma * c['qraw'].square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    src[f'q{b}'] = (self.draw_code(gen, T, q_cycle[b], dev), basis, qstep)
                    hfull = d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float()
                    fstep = s.readout_sigma * hfull.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    src[f'f{b}'] = (self.draw_code(gen, T, f_cycle[b], dev), fstep) + self.fc_slots(b, seq, f_cycle[b])
                tick('source_setup')
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m = t1 - t0
                sl = slice(seq * T + t0, seq * T + t1)
                local = slice(t0, t1)
                ctx = dict(common, **buffers, **seqd, own=own, routes=qroutes, local=local, t0=t0,
                           mids=[z[sl] for z in mids], hs=[z[sl] for z in hs_all], idx32=[z[sl] for z in idx32],
                           sqx_in=[z[local] for z in seqd['sqx_seq']])
                delta = delta_all[:m * rows * W].view(m, rows, W)
                dirs = {}
                if sources:  # this chunk's source directions: q [m, P_q, W] (|d|^2 = 512), c_fc codes [m, P_f, cycle]
                    for b in BLOCKS:
                        code, basis, qstep = src[f'q{b}']
                        rows_q = self.code_rows(code, q_cycle[b], *q_share[b], local, dev)
                        if basis is not None:  # 2 x blockdiag(basis_h) code: sum_d dir dir^T = 512 x the projector
                            rows_q = 2 * torch.einsum('thdk,tphk->tphd', basis[local],
                                                      rows_q.view(m, P_q[b], H, s.readout_q_keys)).reshape(m, P_q[b], W)
                        dirs[f'q{b}'] = rows_q
                        dirs[f'f{b}'] = self.code_rows(src[f'f{b}'][0], f_cycle[b], *f_share[b], local, dev)
                tick('dirs')
                dx0 = F.rms_norm(emb[sl].unsqueeze(1) + steps[0][sl].unsqueeze(-1) * bases[0].unsqueeze(0), (W,)) \
                    - x0[sl].unsqueeze(1)
                torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                for b in range(L):
                    n = in_rows[b]
                    c = ctxs[b]
                    out = torch.empty(m, n, device=dev)
                    use_self = s.readout_self and b < L - 1
                    gated = c['gate'] is not None
                    if IMPL['score'] == 'staged':  # bf16 copy + row statistics, one GEMM, the scorer
                        se = se_all if use_self else out
                        dummy = out
                        stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], db_all, st_all, t0, n, rows, W, 16,
                                                              EPS, num_warps=4)
                        up = up_all[:m * n * NC].view(m * n, NC)
                        torch.mm(db_all[:m * n * W].view(m * n, W), wcat[b], out=up)
                        staged_kernel[(triton.cdiv(n, 64), m)](
                            up, st_all, c['R0'], c['VC'], c['GZ'] if gated else dummy, c['ve'] if gated else dummy,
                            cs_all, sn_all, c['U'], c['W'], c['M1'], c['M2'], c['QD'], c['GD'], c['AD'], c['CD'], c['VT'],
                            c['OT'], out, se, t0, n, rows, W, H, D, gated, True, use_self, 64, NC, num_warps=4,
                            num_stages=2)
                    else:  # the reference: fast_moments' compiled prologue, then readout_kernel
                        cs, sn = cs_all[local], sn_all[local]
                        projected, r0, gate_change = pro(delta[:, :n], one, c['x'][local], c['kraw'][local],
                                                         None if c['gate'] is None else c['gate'][local],
                                                         None if c['ve'] is None else c['ve'][local], c['wkv'], c['wg'],
                                                         cs, sn)
                        se = buffers['scratch'][:m * n * W].view(m, n, W) if use_self else out
                        readout_kernel[(m, triton.cdiv(n, BPR))](
                            projected.contiguous(), r0.contiguous(), cs, sn, gate_change.contiguous() if gated else out,
                            c['ve'][local] if gated else out, c['U'], c['W'], c['M1'], c['M2'], c['QD'], c['GD'],
                            c['AD'], c['CD'], c['VT'], c['OT'], out, se, t0, n, *cs.stride(), *sn.stride(), H, D, gated,
                            True, use_self, BPR, num_warps=4, num_stages=2)
                        del projected, r0, gate_change
                        if use_self:  # into the fused path's layout (rows per token = rows)
                            se_all[:m * rows * W].view(m, rows, W)[:, :n] = se
                    tick('score')
                    score[sl, :n] += out
                    if keep:
                        per_block[b, sl, :n] = out
                    if b == L - 1:
                        break
                    tick('misc')
                    FC.oq_fast(delta, n, b, ctx)  # the own query on the other tokens' frozen keys / values
                    tick('oq')
                    if use_self:  # the self-edge (the token's own moved key / value seen by its own query), c_proj added
                        bm, bn, bk, warps, stages = FC.PROJ
                        FC.linear_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
                            se_all, w_proj_t[b], delta, m * n, n, rows, rows, W, W, bm, bn, bk, num_warps=warps,
                            num_stages=stages)
                    tick('self_edge')
                    if b in BLOCKS:
                        torch.mul(steps[b + 1][sl].unsqueeze(-1), bases[b + 1].unsqueeze(0), out=delta[:, groups[f'r{b}']])
                        if P_q[b]:
                            delta[:, groups[f'q{b}']] = self.q_moves(ctxs[b], b, dirs[f'q{b}'], src[f'q{b}'][2][local],
                                                                     local)
                    tick('q_moves')
                    FC.mlp_fast(delta, mid_rows[b], b, ctx, dx0, P)  # block b's routed MLP, block b + 1's mixing
                    tick('mlp')
                    if b in BLOCKS and P_f[b]:
                        _, fstep, idx, hval, valid = src[f'f{b}']
                        delta[:, groups[f'f{b}']] = lam_r[b + 1] * self.fc_moves(
                            b, dirs[f'f{b}'], fstep[local], idx[local], hval[local], valid[local])
                tick('fc_moves')
                for b in BLOCKS if sources else ():  # this chunk's tokens are complete: decode their sources
                    _, _, qstep = src[f'q{b}']
                    q_credit[b][sl] = torch.einsum('tp,tpw->tw', score[sl, groups[f'q{b}']] / qstep[local],
                                                   dirs[f'q{b}']) / W  # sum_d dir dir^T = 512 x (identity | projector)
                    _, fstep, idx, hval, valid = src[f'f{b}']
                    slot = torch.einsum('tp,tpk->tk', score[sl, groups[f'f{b}']] / fstep[local],
                                        dirs[f'f{b}'][:, :, 1:]) / f_cycle[b]
                    fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                del dx0, dirs
                tick('decode')
            del ctxs, own, qroutes, seqd, src
        del buffers, delta_all, db_all, st_all, up_all, se_all
        X = torch.stack([(score[:, groups[g]] / steps[k]) @ bases[k] / W
                         for k, g in enumerate(['emb'] + [f'r{i}' for i in BLOCKS])])
        X = e.group_sum(X)  # the pair's halves of every cycle
        self.X = {'emb': X[0], **{i: X[1 + k] for k, i in enumerate(BLOCKS)}}
        self.q_credit, self.fc_credit = q_credit, fc_credit
        if keep:  # checks: the per-block readout of each root (this GPU's share) and the layout
            self.per_block = {(g, b): (per_block[b][:, groups[g]] / steps[k]) @ bases[k] / W
                              for k, g in enumerate(['emb'] + [f'r{i}' for i in BLOCKS]) for b in range(L)}
            self.groups, self.score, self.rows = groups, score, rows

    def q_moves(self, c, b, dirs, qstep, local):
        """W_proj (Attn(q + step d) - Attn(q)) of the token's own query at block b (exact over all keys, the record's
        query_response; every key / value clean), [m, P_q, W]."""
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        m, p, _ = dirs.shape
        cs, sn = e.model.cos[0, local, 0].float(), e.model.sin[0, local, 0].float()
        c_, s_ = cs[:, None, None], sn[:, None, None]
        r0 = R.rotate(c['qraw'][local].reshape(m, 1, H, D), c_, s_)
        dr = R.rotate((qstep[:, None] * dirs).reshape(m, p, H, D), c_, s_)
        response = R.query_response(R.norm_delta(r0, dr).contiguous(), c['k'], c['values'], c['a'][:, local],
                                    c['out'][:, local])
        return F.linear(response.flatten(-2), e.model.transformer.h[b].attn.c_proj.weight.float())

    def fc_moves(self, b, code, fstep, idx, hval, valid):
        """The block-b output change W_out (relu^2(h + step d) - relu^2(h)) on the token's slot units, [m, P_f, W]."""
        w_sel = self.e.model.transformer.h[b].mlp.c_proj.weight.float().T[idx]  # [m, slots, W]
        move = fstep[:, None] * code[:, :, 1:] * valid[:, None]
        up, down = F.relu(hval[:, None] + move), F.relu(hval)[:, None]
        return torch.bmm((up - down) * (up + down), w_sel)


def cut_estimate(self, x, y):  # replaces Cut.estimate: off = the record's, called as is
    if not getattr(self.s, 'readout', False):
        return CUT_ESTIMATE(self, x, y)
    method = Method(self)
    gradients = method.estimate(x, y)
    if STATS['keep']:
        self.readout_method = method
    return gradients


def install():
    """onesided's patch, then this one on the loaded recipe module (idempotent): Settings, its recipe dict and
    Cut.estimate. The speed stack goes on after (it touches none of these)."""
    if R.Settings is not Settings:
        onesided.install()
    if R.Settings not in (RECORD_SETTINGS, Settings) or R.asdict not in (RECORD_ASDICT, asdict) or \
            R.Cut.estimate not in (CUT_ESTIMATE, cut_estimate):
        raise RuntimeError('readout: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Settings, R.asdict, R.Cut.estimate = Settings, asdict, cut_estimate
    return R


# ======================================== module readout_kv
"""Receiver experiments for c_k (blocks 0-3) and ve_gate: per key token and head its strongest receivers; raw-key moves
along the dual basis of their logit pull-backs (each moves one receiver's logit to first order) and gate-logit moves,
each turned into the receiver's exact single-column output change and pushed up the receiver's real path.
"""
from dataclasses import dataclass
import json
import math
import os

import torch
import torch.nn.functional as F
import triton

import readout as RO  # noqa: E402  (loads onesided and research/pr31/run_recipe.py as module 'recipe')
R, FC = RO.R, RO.FC

BLOCKS = RO.BLOCKS
EPS = RO.EPS
K_SIGMA = .01  # raw-key step: |delta_r| = K_SIGMA |k_raw_t,h| (L2 over the head)
GATE_SIGMA = .01  # gate-logit step
K_TAU = .1  # guard: share of |p_r| a receiver's pull-back must keep outside the stronger accepted receivers' span
FIELDS = ('readout_k_blocks', 'readout_k_receivers', 'readout_gate', 'readout_k_cap')
STATS = dict(calls=0, every=200, keep=False, seconds=[])  # keep: checks read the last estimate's pieces


# ------------------------------------------------------------------------------------------------------- the settings


@dataclass
class Settings(RO.Settings):  # readout's fields plus E1's
    readout_k_blocks: tuple = (0, 1, 2, 3)  # blocks whose c_k (and ve_gate, if any) take the receiver experiments
    readout_k_receivers: int = 8  # R: receivers per key token and head (0: off, readout.py as is)
    readout_gate: bool = True  # the gate experiment on the chosen blocks that have value embeddings
    readout_k_cap: float = 1.5  # rows per receiver token and group: this x the mean load 8 R (rounded up to 16, >= 64)

    def __post_init__(self):
        super().__post_init__()
        if not self.readout:
            return
        self.readout_k_blocks = tuple(self.readout_k_blocks)
        blocks = self.readout_k_blocks
        if (any(type(b) is not int for b in blocks) or len(set(blocks)) < len(blocks) or not set(blocks) <= set(BLOCKS)
                or type(self.readout_k_receivers) is not int or not 0 <= self.readout_k_receivers <= 32
                or type(self.readout_gate) is not bool or type(self.readout_k_cap) not in (int, float)
                or not 0 < self.readout_k_cap <= 8):
            raise ValueError('readout_kv: readout_k_blocks distinct within 0-3, readout_k_receivers 0..32, readout_gate '
                             'true / false, 0 < readout_k_cap <= 8')


def kv_on(s):
    return bool(getattr(s, 'readout', False) and getattr(s, 'readout_k_receivers', 0) and getattr(s, 'readout_k_blocks', ()))


def asdict(settings):  # E1 off: readout.py's recipe dict (config.json, resume)
    state = RO.asdict(settings)
    if not kv_on(settings):
        for key in FIELDS:
            state.pop(key, None)
    return state


def group_rows(s, heads):
    """Rows per receiver token of one group: readout_k_cap x the mean load heads x R, rounded up to 16, at least 64
    (fast_chain's own-query kernel takes >= 64 rows per token)."""
    return max(64, 16 * math.ceil(s.readout_k_cap * heads * s.readout_k_receivers / 16))


def groups_of(s, model, split):
    """The (block, 'k' | 'g') groups in insertion order and the part of a pair that runs each: heaviest first (rows
    inserted at block b pass blocks b + 1 .. L - 1) onto the least loaded part, ties to the lower part."""
    L = model.config.n_layer
    groups = []
    for b in sorted(s.readout_k_blocks):
        groups.append((b, 'k'))
        if s.readout_gate and model.transformer.h[b].attn.ve_gate is not None:
            groups.append((b, 'g'))
    load, owner = [0] * split, {}
    for g in sorted(groups):  # (block, kind): smaller block = heavier; 'g' before 'k'
        part = min(range(split), key=lambda q: (load[q], q))
        owner[g] = part
        load[part] += L - 1 - g[0]
    return groups, owner


# ------------------------------------------------------------------------------------------------ the experiments' math


def rot(x, cs, sn, inverse=False):
    """RoPE of each token t on x [H, T, R, D] (cs / sn [T, D / 2]); inverse: its transpose R_t^T."""
    c, s = cs[None, :, None], sn[None, :, None]
    return R.rotate(x, c, -s if inverse else s)


def norm_change(u, dr):
    """N(u + dr) - N(u) of the per-head RMSNorm (fp32 eps, as the record's key experiments), in the scorer kernels'
    cancellation-free form."""
    D = u.shape[-1]
    square = u.square().sum(-1, keepdim=True)
    grow = (2 * (u * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / D
    inverse = torch.rsqrt(square / D + EPS)
    moved = torch.rsqrt(square / D + grow + EPS)
    return dr * moved - u * (grow * (inverse * moved).square() / (inverse + moved))


def guard(P, ok, tau):
    """Receivers in attention order: accept r iff valid and its pull-back keeps >= tau |p_r| outside the span of the
    accepted stronger ones (modified Gram-Schmidt, twice). P [H, T, R, D], ok [H, T, R] -> accepted [H, T, R]."""
    Q = torch.zeros_like(P)
    acc = torch.zeros(P.shape[:3], dtype=torch.bool, device=P.device)
    norms = P.norm(dim=-1)
    for r in range(P.shape[2]):
        v = P[:, :, r]
        for _ in range(2 if r else 0):
            v = v - torch.einsum('htk,htkd->htd', torch.einsum('htkd,htd->htk', Q[:, :, :r], v), Q[:, :, :r])
        rn = v.norm(dim=-1)
        good = ok[:, :, r] & (rn >= tau * norms[:, :, r]) & (norms[:, :, r] > 0)
        Q[:, :, r] = torch.where(good[..., None], v / rn.clamp_min(1e-30)[..., None], torch.zeros_like(v))
        acc[:, :, r] = good
    return acc


def receiver_slots(idx, vals, ok, C, T):
    """Pairs (h, t, r) [H, T, R] (receiver idx, clean attention vals, valid ok) -> the slot table [T, C] of each receiver
    token (flat pair index (h T + t) R + r, -1 empty; at most C per receiver, the strongest attention first) and the kept
    mask [H, T, R]. No host synchronization."""
    n = idx.numel()
    rec = idx.reshape(-1)
    valid = ok.reshape(-1)
    order = torch.argsort(torch.where(valid, vals.reshape(-1), -1.), descending=True, stable=True)
    order = order[torch.argsort(rec[order], stable=True)]  # by receiver, the strongest first
    rs = rec[order]
    counts = torch.zeros(T, dtype=torch.long, device=idx.device).scatter_add_(0, rs, torch.ones_like(rs))
    rank = torch.arange(n, device=idx.device) - (torch.cumsum(counts, 0) - counts)[rs]
    keep = (rank < C) & valid[order]
    table = torch.full((T * C + 1,), -1, dtype=torch.long, device=idx.device)  # the last entry: a sink for the rest
    table.scatter_(0, torch.where(keep, rs * C + rank, T * C), torch.where(keep, order, -1))
    kept = torch.zeros(n, dtype=torch.bool, device=idx.device).scatter_(0, order, keep)
    return table[:T * C].view(T, C), kept.view_as(ok)


# ------------------------------------------------------------------------------------------------------- the method


class Method(RO.Method):
    """readout.py's Method as is, plus the receiver rows: run() prepares this GPU's groups, then readout's pass runs
    unchanged; readout calls contexts(seq) once per sequence and gets the same contexts back after this GPU's receiver
    rows of that sequence have run in their own pass (own buffers); finish() adds the decoded credit to c_k / ve_gate."""

    def run(self, chain, targets):
        self.kv_setup()
        self.kv_live = True
        try:
            super().run(chain, targets)
        finally:
            self.kv_live = False
            self.kv_buffers = None

    def contexts(self, seq, targets):
        ctxs = super().contexts(seq, targets)
        if getattr(self, 'kv_live', False) and self.kv_mine:
            RO.tick('contexts')
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.kv_sequence(seq, ctxs)
            end.record()
            self.kv_events.append((start, end))
            RO.tick('kv')
        return ctxs

    # -------------------------------------------------------------------------------------------------- setup

    def kv_setup(self):
        e, s = self.e, self.e.s
        model, d = e.model, e.capture
        L, W, H = model.config.n_layer, e.width, model.config.n_head
        dev = e.device
        groups, owner = groups_of(s, model, e.split)
        mine = [g for g in groups if owner[g] == e.part]
        C = group_rows(s, H)
        self.kv_groups, self.kv_owner, self.kv_mine, self.kv_C = groups, owner, mine, C
        self.kv_first = {g: k * C for k, g in enumerate(mine)}  # each group's first row
        self.kv_rows = len(mine) * C
        self.kv_in = [sum(C for g in mine if g[0] < b) for b in range(L)]  # rows at block b's input
        self.kv_mid = [sum(C for g in mine if g[0] <= b) for b in range(L)]  # rows through block b's MLP
        n = d.batch * d.length
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, kind in mine if kind == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, kind in mine if kind == 'g'}
        self.kv_events, self.kv_counts, self.kv_keep, self.kv_added = [], {}, {}, {}
        if not mine:
            return
        emb, x0, mids, outs, routes, y = self.chain_args
        blocks = model.transformer.h
        self.kv_w = dict(w_fc=[blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in blocks],
                         w_out_t=[blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in blocks],
                         w_q_t=[blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in blocks],
                         w_proj_t=[blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in blocks])
        NC = 2 * W + 16  # the staged scorer's projection: d W_k^T | d W_v^T | d[:32] W_g^T (padded to 16), as readout's
        wgt = [None if blk.attn.ve_gate is None else F.pad(blk.attn.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16)
               .contiguous() for blk in blocks]
        self.kv_wcat = [torch.cat([blk.attn.c_k.weight.to(torch.bfloat16).T.contiguous(),
                                   blk.attn.c_v.weight.to(torch.bfloat16).T.contiguous(),
                                   torch.zeros(W, 16, device=dev, dtype=torch.bfloat16) if wgt[j] is None
                                   else F.pad(wgt[j], (0, 0, 0, W - 32))], 1).contiguous() for j, blk in enumerate(blocks)]
        self.kv_wproj = {b: blocks[b].attn.c_proj.weight.float() for b in {g[0] for g in mine}}
        self.kv_mids = [z.float().contiguous() for z in mids]
        self.kv_idx32 = [r[0].to(torch.int32).contiguous() for r in routes]
        self.kv_hs = [r[1].float().contiguous() for r in routes]
        units = max(r[0].shape[1] for r in routes)
        m, rows = min(s.readout_chunk, d.length), self.kv_rows
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        self.kv_common = dict(lam_r=lam_r, lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(),
                              w_fc=self.kv_w['w_fc'], w_out_t=self.kv_w['w_out_t'])
        self.kv_buffers = dict(scratch=torch.empty(m * rows * max(W, units), device=dev, dtype=torch.bfloat16),
                               scratch32=torch.empty(m * rows * W, device=dev), stats=torch.empty(m * rows * 2, device=dev),
                               delta=torch.empty(m * rows * W, device=dev),
                               se=torch.empty(m * rows * W, device=dev, dtype=torch.bfloat16),
                               db=torch.empty(m * rows * W, device=dev, dtype=torch.bfloat16),
                               st=torch.empty(m * rows * 2, device=dev),
                               up=torch.empty(m * rows * NC, device=dev, dtype=torch.bfloat16))

    # ------------------------------------------------------------------------------------------ one sequence

    def clean_attention(self, b, seq, c):
        """Block b's clean attention [H, T(query), T(key)] (contexts keeps it when the sources are on)."""
        if 'a' in c:
            return c['a']
        e = self.e
        d, model = e.capture, e.model
        T, H = d.length, model.config.n_head
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        prefix = f'transformer.h.{b}.attn.'
        qraw, kraw = (d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k'))
        return R.attention_state(qraw, kraw, c['VT'].transpose(0, 1)[None], cos, sin, H, model.window_sizes[b][0])[2][0]

    def plan(self, b, seq, c, kinds, cs, sn):
        """Block b's experiments: per (head h, key t) the top-R receivers, the receiver slot table, and per pair the
        coefficient of its receiver's output change (k: the exact single-column gain along v_t - o_s; gate: A_st (2 sig(z +
        step) - 2 sig(z)) along ve_t)."""
        e, s = self.e, self.e.s
        H = e.model.config.n_head
        T, D = e.capture.length, e.width // H
        Rr, C = s.readout_k_receivers, self.kv_C
        A = self.clean_attention(b, seq, c)
        top = A.transpose(1, 2).topk(Rr, dim=-1)  # per (h, key t): its strongest receivers s, in attention order
        vals, idx = top.values.contiguous(), top.indices.contiguous()  # [H, T, R]
        ok = vals > 0
        table, kept = receiver_slots(idx, vals, ok, C, T)
        plans = {}
        if 'k' in kinds:
            hix = torch.arange(H, device=A.device)[:, None, None]
            qs = c['QD'].transpose(0, 1)[hix, idx]  # the receivers' clean (rotated, normed) queries [H, T, R, D]
            u = c['R0'].transpose(0, 1)[:, :, None]  # the rotated raw key [H, T, 1, D]
            ms = u.square().mean(-1, keepdim=True) + EPS
            jq = (qs - u * (u * qs).sum(-1, keepdim=True) / (D * ms)) / ms.sqrt()  # the key norm's Jacobian (symmetric)
            P = rot(jq, cs, sn, inverse=True) / math.sqrt(D)  # p_s = d logit_st / d k_raw_t
            acc = guard(P, kept, K_TAU)
            Pm = (P * acc[..., None]).double()
            G = torch.einsum('htrd,htsd->htrs', Pm, Pm) + torch.diag_embed((~acc).double())
            dual = torch.einsum('htrj,htjd->htrd', torch.linalg.inv_ex(G)[0], Pm).float()  # p_s . d_r = [s = r]
            dn = dual.norm(dim=-1)
            step = K_SIGMA * c['kraw'].transpose(0, 1).norm(dim=-1).clamp_min(1e-12)[:, :, None]  # [H, T, 1]
            x_lin = torch.where(acc, step / dn.clamp_min(1e-30), torch.zeros_like(dn))  # first-order logit change of r
            dk = norm_change(u, rot(x_lin[..., None] * dual, cs, sn))  # each move's exact key change (RoPE + QK-norm)
            x = (qs * dk).sum(-1) / math.sqrt(D)  # receiver r's actual logit change
            ex = torch.expm1(x)
            gain = torch.where(acc, vals * ex / (1 + vals * ex), torch.zeros_like(x))  # exact single-column update
            plans[b, 'k'] = dict(table=table, live=acc, coef=gain, P=P, x_lin=x_lin)
            self.kv_counts[b, 'k'] = torch.stack([ok.sum(), kept.sum(), acc.sum()])
        if 'g' in kinds:
            z = c['gate']  # [T, H]
            dvs = 2 * (torch.sigmoid(z + GATE_SIGMA) - torch.sigmoid(z))
            coef = torch.where(kept, vals * dvs.T[:, :, None], torch.zeros_like(vals))
            plans[b, 'g'] = dict(table=table, live=kept, coef=coef)
            self.kv_counts[b, 'g'] = torch.stack([ok.sum(), kept.sum(), kept.sum()])
        return plans

    def moves(self, g, p, c, t0, t1):
        """The mid_b moves of group g's rows for the receiver tokens t0:t1, [m, C, W] (zero in empty slots): the receiver's
        output change of head h through c_proj."""
        b, kind = g
        H = self.e.model.config.n_head
        T, W = self.e.capture.length, self.e.width
        D, Rr = W // H, self.e.s.readout_k_receivers
        tbl = p['table'][t0:t1]
        m, C = tbl.shape
        f = tbl.clamp_min(0)
        live = (tbl >= 0) & p['live'].reshape(-1)[f]
        coef = torch.where(live, p['coef'].reshape(-1)[f], torch.zeros((), device=f.device))
        h, t = f // (T * Rr), (f // Rr) % T
        if kind == 'k':  # receiver s's output moves along v_t - o_s
            s_ = torch.arange(t0, t1, device=f.device)[:, None].expand(m, C)
            vec = c['VT'][t, h] - c['OT'][s_, h]
        else:  # along the value change: the value embedding ve_t
            vec = c['ve'][t, h]
        head = torch.zeros(m, C, H, D, device=f.device)
        head.scatter_(2, h[:, :, None, None].expand(m, C, 1, D), (coef[..., None] * vec)[:, :, None])
        return F.linear(head.view(m, C, W), self.kv_wproj[b])

    def kv_sequence(self, seq, ctxs):
        """This GPU's receiver rows of one sequence: the experiments, the push and scoring with readout's kernels (the
        rows of a group start at its block's mid stream, as readout's root rows), the decode."""
        e, s = self.e, self.e.s
        model, d = e.model, e.capture
        L, W, H = model.config.n_layer, e.width, model.config.n_head
        D, T, dev = W // H, d.length, e.device
        cs_all, sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()
        plans = {}
        for b in sorted({g[0] for g in self.kv_mine}):
            plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in self.kv_mine if bb == b], cs_all, sn_all))
        RO.tick('kv_plan')
        own = e.own_query_inputs(seq)
        qroutes = FC.query_routes(own, s.query_keys)
        inputs = [z.float().contiguous() for z in own[1]]
        seqd = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=self.kv_w['w_q_t'],
                    w_proj_t=self.kv_w['w_proj_t'], inputs=inputs, sqx_seq=[z.square().sum(-1) for z in inputs])
        buf = self.kv_buffers
        rows, C = self.kv_rows, self.kv_C
        NC = 2 * W + 16
        score = torch.zeros(T, rows, device=dev)
        m_chunk = min(s.readout_chunk, T)
        first = min(g[0] for g in self.kv_mine)
        RO.tick('kv_routes')
        for t0 in range(0, T, m_chunk):
            t1 = min(t0 + m_chunk, T)
            m = t1 - t0
            sl, local = slice(seq * T + t0, seq * T + t1), slice(t0, t1)
            ctx = dict(self.kv_common, scratch=buf['scratch'], scratch32=buf['scratch32'], stats=buf['stats'], **seqd,
                       own=own, routes=qroutes, local=local, t0=t0, mids=[z[sl] for z in self.kv_mids],
                       hs=[z[sl] for z in self.kv_hs], idx32=[z[sl] for z in self.kv_idx32],
                       sqx_in=[z[local] for z in seqd['sqx_seq']])
            delta = buf['delta'][:m * rows * W].view(m, rows, W)
            for b in range(first, L):
                n, c = self.kv_in[b], ctxs[b]
                use_self = s.readout_self and b < L - 1
                if n:  # readout's staged scorer on the rows at block b's input
                    gated = c['gate'] is not None
                    out = torch.empty(m, n, device=dev)
                    se = buf['se'] if use_self else out
                    RO.stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], buf['db'], buf['st'], t0, n, rows,
                                                             W, 16, EPS, num_warps=4)
                    up = buf['up'][:m * n * NC].view(m * n, NC)
                    torch.mm(buf['db'][:m * n * W].view(m * n, W), self.kv_wcat[b], out=up)
                    RO.staged_kernel[(triton.cdiv(n, 64), m)](
                        up, buf['st'], c['R0'], c['VC'], c['GZ'] if gated else out, c['ve'] if gated else out, cs_all,
                        sn_all, c['U'], c['W'], c['M1'], c['M2'], c['QD'], c['GD'], c['AD'], c['CD'], c['VT'], c['OT'],
                        out, se, t0, n, rows, W, H, D, gated, True, use_self, 64, NC, num_warps=4, num_stages=2)
                    score[local, :n] += out
                    RO.tick('kv_score')
                if b == L - 1:
                    break
                if n:
                    FC.oq_fast(delta, n, b, ctx)  # the own query on the other tokens' frozen keys / values
                    RO.tick('kv_oq')
                    if use_self:  # the self-edge (the token's own moved key / value seen by its own query), c_proj added
                        bm, bn, bk, warps, stages = FC.PROJ
                        FC.linear_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
                            buf['se'], self.kv_w['w_proj_t'][b], delta, m * n, n, rows, rows, W, W, bm, bn, bk,
                            num_warps=warps, num_stages=stages)
                RO.tick('kv_self_edge')
                for g in self.kv_mine:  # this block's receiver moves start at its mid stream
                    if g[0] == b:
                        delta[:, self.kv_first[g]:self.kv_first[g] + C] = self.moves(g, plans[g], c, t0, t1)
                RO.tick('kv_moves')
                if self.kv_mid[b]:  # block b's routed MLP, then block b + 1's mixing scalars (no embedding rows)
                    FC.mlp_fast(delta, self.kv_mid[b], b, ctx, delta, 0)
                RO.tick('kv_mlp')
        Rr = s.readout_k_receivers
        for g in self.kv_mine:  # decode
            b, kind = g
            p = plans[g]
            f = p['table'].clamp_min(0)
            live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
            got = score[:, self.kv_first[g]:self.kv_first[g] + C]
            reading = torch.zeros(H * T * Rr + 1, device=dev)  # the last entry: a sink for the empty slots
            reading.scatter_(0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, got, 0.).reshape(-1))
            reading = reading[:-1].view(H, T, Rr)
            out = slice(seq * T, (seq + 1) * T)
            if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
            else:
                self.g_credit[b][out] = reading.sum(-1).T / GATE_SIGMA
            if STATS['keep']:
                self.kv_keep[seq, g] = dict(p, reading=reading)
        del plans, own, qroutes, seqd, score

    # ------------------------------------------------------------------------------------------------ finishing

    def finish(self, gradients):
        super().finish(gradients)
        d = self.e.capture
        n = STATS['calls'] + 1
        report = os.environ.get('RANK', '0') == '0' and (n == 1 or n % STATS['every'] == 0)
        for kind, credits, site in (('k', self.k_credit, 'attn.c_k'), ('g', self.g_credit, 'attn.ve_gate')):
            for b, credit in credits.items():
                p = f'transformer.h.{b}.{site}'
                g = gradients[p + '.weight']
                added = credit.T @ d.inputs[p].flatten(0, 1)  # the same arithmetic whether or not this call reports
                if report:
                    self.kv_added[b, kind] = (added.norm() / g.norm().clamp_min(1e-30)).item()
                g.add_(added)

    def note(self):
        super().note()
        STATS['calls'] += 1
        STATS['seconds'].append(self.kv_events)
        del STATS['seconds'][:-50]
        n = STATS['calls']
        if os.environ.get('RANK', '0') != '0' or (n != 1 and n % STATS['every']):
            return
        sec = pass_seconds()
        info = {}
        for (b, kind), cnt in self.kv_counts.items():
            valid, kept, live = cnt.tolist()
            info[f'{kind}{b}'] = dict(kept=round(kept / max(valid, 1), 4), live=round(live / max(valid, 1), 4),
                                      added_vs_record=round(self.kv_added.get((b, kind), 0.), 3))
        print(f'readout_kv estimate {n}: groups here {["%s%d" % (k, b) for b, k in self.kv_mine]}, {self.kv_rows} rows per '
              f'token (C {self.kv_C}); {json.dumps(info)}; pass {sum(sec) / max(len(sec), 1):.3f} s', flush=True)


def pass_seconds():
    """Seconds of the receiver pass per estimate (CUDA events; synchronizes), the last 50 estimates."""
    torch.cuda.synchronize()
    return [sum(a.elapsed_time(b) for a, b in events) / 1e3 for events in STATS['seconds']]


def cut_estimate(self, x, y):  # replaces Cut.estimate: E1 off = readout.py's, called as is
    if not kv_on(self.s):
        return RO.cut_estimate(self, x, y)
    method = Method(self)
    gradients = method.estimate(x, y)
    if STATS['keep'] or RO.STATS['keep']:
        self.readout_method = method
    return gradients


def install():
    """readout.py's patch (and onesided's), then this one (idempotent): Settings, its recipe dict and Cut.estimate. The
    speed stack goes on after (it touches none of these)."""
    if R.Settings is not Settings:
        RO.install()
        if R.Settings is not RO.Settings or R.asdict is not RO.asdict or R.Cut.estimate is not RO.cut_estimate:
            raise RuntimeError('readout_kv: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Settings, R.asdict, R.Cut.estimate = Settings, asdict, cut_estimate
    return R


# ======================================== module fast_readout
"""readout / readout_kv at rounding level, faster: routes kept from the chain, the own query from the scorer's bf16
copy, the self-edge and own-query c_proj in one pass, fused q-source and c_fc kernels, exact MLP pruning.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import readout_kv as KV  # noqa: E402  (loads readout, onesided and research/pr31/run_recipe.py as module 'recipe')
import fast_kernels as FK  # noqa: E402  (its query kernel's precision / tiles only; installing it is the stack's job)
import fast_chain2 as FC2  # noqa: E402  (its pruned MLP kernels and route order only; nothing installed)
RO, R, FC = KV.RO, KV.R, KV.FC

BLOCKS, EPS = RO.BLOCKS, RO.EPS
IMPL = dict(routes=True, oq='fused', qsrc='fused', fcrelu='fused', mlp='prune')  # checks: each change can be switched back


UP = (128, 256, 64, 8, 4)  # mlp_up_prune_kernel: rows, units per tile, K step, warps, stages (rows / K step / warps: fast_chain's)
QSRC_TILES = {64: (64, 4, 2), 128: (128, 8, 2)}  # probes per program (by probes per token), warps, stages (bitwise alike)


# ----------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def dual_residual_kernel(A, A2, B, D, n, n_rows, A_ROWS, A2_ROWS, D_ROWS,
                         K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         TWO: tl.constexpr):
    """d[i] = (d[i] + a[i] @ B) + a2[i] @ B for the n_rows first rows of every token (flattened: row i = token i // n_rows,
    row i % n_rows; A rows of A_ROWS per token, A2 of A2_ROWS, D of D_ROWS), in place: fast_chain's linear_residual_kernel
    twice in one pass (each product in its own fp32 accumulator, the same tiles, then the two adds in readout's order):
    bitwise readout's two launches, one read and write of d instead of two."""
    pn, pm = tl.program_id(0), tl.program_id(1)
    i = pm * BM + tl.arange(0, BM)
    ok = i < n
    tok, r = i // n_rows, i % n_rows
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    arow = (tok * A_ROWS + r).to(tl.int64) * K
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + arow[:, None] + kk[None, :], ok[:, None], 0.)
        b = tl.load(B + kk[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    dptr = D + (tok * D_ROWS + r).to(tl.int64)[:, None] * N + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if TWO:
        acc2 = tl.zeros((BM, BN), tl.float32)
        arow2 = (tok * A2_ROWS + r).to(tl.int64) * K
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A2 + arow2[:, None] + kk[None, :], ok[:, None], 0.)
            b = tl.load(B + kk[:, None] * N + cols[None, :])
            acc2 = tl.dot(a, b, acc2)
        new = new + acc2
    tl.store(dptr, new, ok[:, None])


@triton.jit
def mlp_up_prune_kernel(D, X, IDX, WFC, HS, SC, CUT, Out, n_rows, D_ROWS, O_ROWS, n_rb,
                        W: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, EPS: tl.constexpr, MARGIN: tl.constexpr):
    """fast_chain's mlp_up_kernel for BP rows of one token over its routed units in fast_chain2's order (SC: their s,
    descending), stopping at the row block's prune point (fast_chain2's bound: every later unit's change is exactly 0 for
    every row of the block; stored in CUT for the down kernel). The row statistics are fast_chain's in-loop sums (the same
    tiles, in a loop of their own) and every formed unit is fast_chain's arithmetic: relu(h + dh)^2 - relu(h)^2, bf16."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    dbase = D + (t * D_ROWS + rows).to(tl.int64) * W
    dotx = tl.zeros((BP,), tl.float32)
    sqd = tl.zeros((BP,), tl.float32)
    sqx = tl.zeros((BK,), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
        x = tl.load(X + t * W + kk)
        dotx += tl.sum(d * x[None, :], 1)
        sqd += tl.sum(d * d, 1)
        sqx += x * x
    sq = tl.sum(sqx, 0)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    coef = (inv - inv0) / inv0
    a_max = tl.max(tl.where(ok, inv * libdevice.sqrt(sqd), 0.), 0) * MARGIN
    c_max = tl.max(tl.where(ok, tl.abs(coef), 0.), 0) * MARGIN
    thr = -a_max / (1 - c_max)
    upos = tl.arange(0, KP)
    s = tl.load(SC + t * K + upos, upos < K, float('-inf'))
    n_units = tl.max(tl.where(s >= thr, upos + 1, 0), 0)
    n_units = tl.where((c_max < 0.5) & (a_max == a_max), n_units, K)  # (a non-finite bound: every unit)
    tl.store(CUT + t * n_rb + pb, n_units)
    obase = Out + (t * O_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, n_units, BN):
        units = u0 + tl.arange(0, BN)
        uidx = tl.load(IDX + t * K + units).to(tl.int64)
        acc = tl.zeros((BP, BN), tl.float32)
        for k0 in range(0, W, BK):
            kk = k0 + tl.arange(0, BK)
            d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
            w = tl.load(WFC + uidx[None, :] * W + kk[:, None])
            acc = tl.dot(d.to(tl.bfloat16), w, acc)
        h = tl.load(HS + t * K + units)
        dh = inv[:, None] * acc + coef[:, None] * h[None, :]
        up = tl.maximum(h[None, :] + dh, 0.)
        down = tl.maximum(h, 0.)[None, :]
        tl.store(obase[:, None] + units[None, :], ((up - down) * (up + down)).to(tl.bfloat16), ok[:, None])


@triton.jit
def tsum64(e, BP: tl.constexpr):
    """Row sums of e [BP, 64] in torch's order for a contiguous last dimension of 64 fp32 values (its reduce kernel: lane
    x < 32 holds e[x] + e[x + 32], then the warp's halving tree x + 16, 8, 4, 2, 1): bitwise torch's .sum(-1)."""
    s = tl.sum(tl.reshape(e, (BP, 2, 32)), 1)
    s = tl.sum(tl.reshape(s, (BP, 2, 16)), 1)
    s = tl.sum(tl.reshape(s, (BP, 2, 8)), 1)
    s = tl.sum(tl.reshape(s, (BP, 2, 4)), 1)
    s = tl.sum(tl.reshape(s, (BP, 2, 2)), 1)
    return tl.sum(s, 1)


@triton.jit
def qsrc_kernel(DIRS, QRAW, QSTEP, CS, SN, K, V, A, Base, Out, t0,
                P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
                KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, EPS: tl.constexpr):
    """readout's q move of BP probes of one token in one head, then fast_kernels' query_kernel on it: r0 = rotate(q_raw),
    dr = rotate(step d) (the token's direction rows DIRS [m, P, H * D], its step QSTEP [m]), dq = norm_delta(r0, dr) (the
    record's formula: dr inv' + r0 (inv' - inv), per-head RMSNorm, fp32 eps), then the exact softmax response over every
    key (the record's streamed form, PRECISION dots). Launch with enable_fp_fusion=False (as the query kernel): the
    rotation's products and sums round as torch's."""
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    probe = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = probe < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    pos = t0 + token
    cs = tl.load(CS + pos * (D // 2) + half)
    sn = tl.load(SN + pos * (D // 2) + half)
    qrow = QRAW + (pos * H + head) * D
    x = tl.load(qrow + f)
    xo = tl.load(qrow + other)
    r0 = tl.where(low, x * cs + xo * sn, -xo * sn + x * cs)
    step = tl.load(QSTEP + token)
    drow = DIRS + ((token * P + probe) * H + head).to(tl.int64) * D
    dd = tl.load(drow[:, None] + f[None, :], pm[:, None], 0.) * step
    ddo = tl.load(drow[:, None] + other[None, :], pm[:, None], 0.) * step
    dr = tl.where(low[None, :], dd * cs[None, :] + ddo * sn[None, :], -ddo * sn[None, :] + dd * cs[None, :])
    square = tsum64((r0 * r0)[None, :], 1)  # [1]
    cross = tsum64(r0[None, :] * dr, BP)
    sqd = tsum64(dr * dr, BP)
    inverse = libdevice.rsqrt(square / D + EPS)
    moved = libdevice.rsqrt((square + 2 * cross + sqd) / D + EPS)
    delta = dr * moved[:, None] + r0[None, :] * (moved - inverse)[:, None]
    numerator = tl.full((BP, D), 0, tl.float32)
    partition = tl.full((BP,), 0, tl.float32)
    for start in range(0, T, BK):
        key = start + tl.arange(0, BK)
        a = tl.load(A + head * AH + token * AM + key * AT, key < T, 0)
        if tl.sum(a, 0) > 0:  # causal / window-masked tiles are exactly zero
            k = tl.load(K + head * KH + key[None, :] * KT + f[:, None] * KD, key[None, :] < T, 0)
            v = tl.load(V + head * VH + key[:, None] * VT + f[None, :] * VD, key[:, None] < T, 0)
            ds = tl.dot(delta, k, input_precision=PRECISION) * (D ** -.5)
            change = a[None, :] * libdevice.expm1(ds)
            partition += tl.sum(change, 1)
            numerator += tl.dot(change, v, input_precision=PRECISION)
    base = tl.load(Base + head * OH + token * OM + f * OD)
    response = (numerator - partition[:, None] * base[None, :]) / (1 + partition[:, None])
    tl.store(Out + ((token * P + probe[:, None]) * H + head).to(tl.int64) * D + f[None, :], response, pm[:, None])


@triton.jit
def fc_relu_kernel(DIRS, FSTEP, HV, VALID, Out, P, NS, CYC, BP: tl.constexpr, BK: tl.constexpr):
    """readout's fc_moves before its GEMM, one pass: r[t, p, k] = (up - down)(up + down) with up = relu(h + move), down =
    relu(h), move = (step_t code[t, p, k + 1]) valid[t, k] (DIRS [m, P, CYC] the code rows; HV / VALID [m, NS]). Launch
    with enable_fp_fusion=False: each op rounds as torch's separate kernels do (bitwise)."""
    pb, kb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = pb * BP + tl.arange(0, BP)
    k = kb * BK + tl.arange(0, BK)
    mask = (p < P)[:, None] & (k < NS)[None, :]
    row = (t * P + p).to(tl.int64)
    code = tl.load(DIRS + row[:, None] * CYC + 1 + k[None, :], mask, 0.)
    valid = tl.load(VALID + t * NS + k, k < NS, 0).to(tl.float32)
    h = tl.load(HV + t * NS + k, k < NS, 0.)
    move = tl.load(FSTEP + t) * code * valid[None, :]
    up = tl.maximum(h[None, :] + move, 0.)
    down = tl.maximum(h, 0.)[None, :]
    tl.store(Out + row[:, None] * NS + k[None, :], (up - down) * (up + down), mask)


# ------------------------------------------------------------------------------------------------------ block pieces


def own_query_step(delta, n, b, ctx, db, st, se):
    """delta[:, :n] += c_proj(the own-query response) (then += c_proj(se), the self-edge, if se is not None) at block b:
    the projection d W_q^T from the scorer's bf16 copy db (rows packed n per token) by cuBLAS (fp32 out), fast_chain's
    oq_attn_kernel on it with the scorer's row statistics st, then both c_proj products in one pass."""
    m, rows, W = delta.shape
    q = ctx['routes'][b]
    keys = q['keys']
    H, T, DH = keys.shape
    u = ctx['scratch32'][:m * n * W].view(m * n, W)
    torch.mm(db[:m * n * W].view(m * n, W), ctx['w_q_t'][b], out_dtype=torch.float32, out=u)
    buf = ctx['ybuf'][:m * n * W]
    bp, warps, stages = FC.OQ_ATTN
    FC.oq_attn_kernel[(H, m)](
        u, st, q['r0'], q['sq0'], ctx['cos'], ctx['sin'], keys, q['values'], q['idx'], q['p'], q['o'], buf,
        n, n, n, ctx['t0'], T, H, DH, W, q['idx'].shape[-1], bp, EPS, num_warps=warps, num_stages=stages)
    bm, bn, bk, warps, stages = FC.PROJ
    two = se is not None
    dual_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
        buf, se if two else buf, ctx['w_proj_t'][b], delta, m * n, n, n, rows, rows, W, W, bm, bn, bk, two,
        num_warps=warps, num_stages=stages)


def score_block(delta, n, rows, b, c, t0, bufs, wcat, cs_all, sn_all, use_self, H, D):
    """readout's staged scorer at block b (as readout.py launches it): stats_kernel (bf16 copy + row statistics), the
    projection GEMM, staged_kernel; returns the scores [m, n] (the self-edge output into bufs['se'])."""
    m, _, W = delta.shape
    NC = wcat.shape[1]
    out = torch.empty(m, n, device=delta.device)
    gated = c['gate'] is not None
    se = bufs['se'] if use_self else out
    RO.stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], bufs['db'], bufs['st'], t0, n, rows, W, 16, EPS,
                                             num_warps=4)
    up = bufs['up'][:m * n * NC].view(m * n, NC)
    torch.mm(bufs['db'][:m * n * W].view(m * n, W), wcat, out=up)
    RO.staged_kernel[(triton.cdiv(n, 64), m)](
        up, bufs['st'], c['R0'], c['VC'], c['GZ'] if gated else out, c['ve'] if gated else out, cs_all, sn_all, c['U'],
        c['W'], c['M1'], c['M2'], c['QD'], c['GD'], c['AD'], c['CD'], c['VT'], c['OT'], out, se, t0, n, rows, W, H, D,
        gated, True, use_self, 64, NC, num_warps=4, num_stages=2)
    return out


def push_block(delta, n, b, ctx, bufs, use_self):
    """The own query (and self-edge) of the n rows at block b's input: the fused step, or readout.py's launches."""
    if IMPL['oq'] == 'fused':
        own_query_step(delta, n, b, ctx, bufs['db'], bufs['st'], bufs['se'] if use_self else None)
        return
    m, rows, W = delta.shape
    FC.oq_fast(delta, n, b, ctx)
    if use_self:
        bm, bn, bk, warps, stages = FC.PROJ
        FC.linear_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
            bufs['se'], ctx['w_proj_t'][b], delta, m * n, n, rows, rows, W, W, bm, bn, bk, num_warps=warps,
            num_stages=stages)


def mlp_step(delta, n_mlp, b, ctx, dx0, P):
    """Block b's routed MLP on rows < n_mlp, then block b + 1's mixing scalars (dx0: the embedding rows' x0 change, the
    first P rows; P = 0: none). 'prune': fast_chain2's exact unit pruning (each token's routed units in fast_chain2's
    order; a row block's prune point from its rows' bound; skipped terms are exact zeros) on fast_chain's arithmetic
    (mlp_up_prune_kernel, then fast_chain2's mlp_down_live_kernel = fast_chain's down tile over the live prefix);
    'plain': fast_chain's mlp_fast."""
    if IMPL['mlp'] != 'prune':
        return FC.mlp_fast(delta, n_mlp, b, ctx, dx0, P)
    m, rows, W = delta.shape
    idx, hs, sc, mid = ctx['idx_s'][b], ctx['hs_s'][b], ctx['sc_s'][b], ctx['mids'][b]
    K = idx.shape[1]
    rbuf = ctx['scratch'][:m * rows * K].view(m, rows, K)
    ubp, ubn, ubk, warps, stages = UP
    n_rb = triton.cdiv(n_mlp, ubp)
    cut = ctx['cut'][:m * n_rb]
    mlp_up_prune_kernel[(n_rb, m)](
        delta, mid, idx, ctx['w_fc'][b], hs, sc, cut, rbuf, n_mlp, rows, rows, n_rb, W, K, triton.next_power_of_2(K),
        ubp, ubn, ubk, EPS, FC2.MARGIN, num_warps=warps, num_stages=stages)
    dbp, dbn, dbk, warps, stages = FC2.DOWN
    scale = dx0 is not None
    FC2.mlp_down_live_kernel[(W // dbn, triton.cdiv(n_mlp, dbp), m)](
        rbuf, idx, ctx['w_out_t'][b], delta, dx0 if scale else delta, ctx['lams'][min(b + 1, len(ctx['lams']) - 1)],
        cut, delta, delta, delta, delta, delta, n_mlp, rows, rows, P, n_rb, W, K, dbp, dbn, dbk, ubp, scale, scale,
        False, False, False, num_warps=warps, num_stages=stages)


def prune_buffers(m, rows, dev):
    """The pruned MLP's prune points (per token and row block)."""
    return dict(cut=torch.empty(m * triton.cdiv(rows, UP[0]), device=dev, dtype=torch.int32))


def sorted_slices(sorted_routes, sl):
    idx, hs, sc = sorted_routes
    return dict(idx_s=[z[sl] for z in idx], hs_s=[z[sl] for z in hs], sc_s=[z[sl] for z in sc])


# ------------------------------------------------------------------------------------------------------- the method


class FastMethod(RO.Method):
    """readout.py's Method with the faster pass (run), q_basis and q_moves; the chain call keeps its own-query inputs and
    routes per sequence for the pass."""

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        est = self.e
        cache = self.route_cache = {}
        if not IMPL['routes']:
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        plain_own, plain_routes = type(est).own_query_inputs, FC.query_routes

        def own_query_inputs(sequence=0):  # instance attribute: fast_chain checks the class's
            own = plain_own(est, sequence)
            cache[sequence] = dict(own=own)
            return own

        def query_routes(own, k, rest=False):
            out = plain_routes(own, k, rest)
            for entry in cache.values():
                if entry['own'] is own:
                    entry['routes'] = (k, out)
            return out
        est.own_query_inputs = own_query_inputs
        FC.query_routes = query_routes
        try:
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        finally:
            est.__dict__.pop('own_query_inputs', None)
            FC.query_routes = plain_routes

    def routes_of(self, seq):
        """The chain's own-query inputs and routes of sequence seq (formed here if the chain did not)."""
        entry = getattr(self, 'route_cache', {}).get(seq)
        k = self.e.s.query_keys
        if entry is not None and entry.get('routes', (None,))[0] == k:
            return entry['own'], entry['routes'][1]
        own = self.e.own_query_inputs(seq)
        return own, FC.query_routes(own, k)

    def run(self, chain, targets):
        e, s = self.e, self.e.s
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'fast_readout: the fast_chain kernels do not cover this setting ({reason})')
        emb, x0, mids, outs, routes, y = self.chain_args
        model, d = e.model, e.capture
        L, W = model.config.n_layer, e.width
        H = model.config.n_head
        D = W // H
        T, nseq = d.length, d.batch
        dev = e.device
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P = hi - lo
        bases = [Hb[lo:hi].contiguous() for Hb in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb] + list(mids)]
        sources = s.readout_sources
        q_cycle = {b: W if b in s.readout_q_full else H * s.readout_q_keys for b in BLOCKS}
        f_cycle = dict(zip(BLOCKS, s.readout_fc_cycle))
        q_share = {b: e.share(q_cycle[b]) for b in BLOCKS}
        f_share = {b: e.share(f_cycle[b]) for b in BLOCKS}
        P_q = {b: q_share[b][1] - q_share[b][0] if sources else 0 for b in BLOCKS}
        P_f = {b: f_share[b][1] - f_share[b][0] if sources else 0 for b in BLOCKS}
        groups, in_rows, mid_rows, rows = RO.layout(s, P, P_q, P_f)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        w_q_t = [blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        w_proj_t = [blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        wkt = [blk.attn.c_k.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        wvt = [blk.attn.c_v.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        wgt = [None if blk.attn.ve_gate is None else F.pad(blk.attn.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16)
               .contiguous() for blk in model.transformer.h]
        wcat = [torch.cat([wkt[j], wvt[j], torch.zeros(W, 16, device=dev, dtype=torch.bfloat16) if wgt[j] is None
                           else F.pad(wgt[j], (0, 0, 0, W - 32))], 1).contiguous() for j in range(L)]
        NC = 2 * W + 16
        mids = [z.float().contiguous() for z in mids]
        idx32 = [r[0].to(torch.int32).contiguous() for r in routes]
        hs_all = [r[1].float().contiguous() for r in routes]
        units = max(r[0].shape[1] for r in routes)
        m_chunk = min(s.readout_chunk, T)
        prune = IMPL['mlp'] == 'prune'
        self.sorted_routes = FC2.sort_routes(w_fc, routes) if prune else None  # (idx, h, s) per block, fast_chain2's order
        scratch = torch.empty(m_chunk * rows * max(W, units), device=dev, dtype=torch.bfloat16)
        buffers = dict(scratch=scratch, ybuf=scratch, scratch32=torch.empty(m_chunk * rows * W, device=dev),
                       stats=torch.empty(m_chunk * rows * 2, device=dev))
        if prune:
            buffers.update(prune_buffers(m_chunk, rows, dev))
        delta_all = torch.empty(m_chunk * rows * W, device=dev)
        bufs = dict(se=torch.empty(m_chunk * rows * W, device=dev, dtype=torch.bfloat16) if s.readout_self else None,
                    db=torch.empty(m_chunk * rows * W, device=dev, dtype=torch.bfloat16),
                    st=torch.empty(m_chunk * rows * 2, device=dev),
                    up=torch.empty(m_chunk * rows * NC, device=dev, dtype=torch.bfloat16))
        common = dict(lam_r=lam_r, lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(), w_fc=w_fc,
                      w_out_t=w_out_t)
        score = torch.zeros(nseq * T, rows, device=dev)
        keep = RO.STATS['keep']
        per_block = torch.zeros(L, nseq * T, rows, device=dev) if keep else None
        q_credit = {b: torch.zeros(nseq * T, W, device=dev) for b in BLOCKS} if sources else {}
        fc_credit = {b: torch.zeros(nseq * T, 4 * W, device=dev) for b in BLOCKS} if sources else {}
        cs_all, sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()
        self.cs_all, self.sn_all = cs_all, sn_all
        RO.tick(None)
        for seq in range(nseq):
            ctxs = self.contexts(seq, targets)
            RO.tick('contexts')
            own, qroutes = self.routes_of(seq)
            inputs = [z.float().contiguous() for z in own[1]]
            seqd = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=w_q_t, w_proj_t=w_proj_t, inputs=inputs,
                        sqx_seq=[z.square().sum(-1) for z in inputs])
            src = {}
            if sources:
                gen = torch.Generator(dev).manual_seed(seed + 104729 * seq)
                for b in BLOCKS:
                    c = ctxs[b]
                    basis = None if q_cycle[b] == W else self.q_basis(c, s.readout_q_keys)
                    qstep = s.readout_sigma * c['qraw'].square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    src[f'q{b}'] = (self.draw_code(gen, T, q_cycle[b], dev), basis, qstep)
                    hfull = d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float()
                    fstep = s.readout_sigma * hfull.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    src[f'f{b}'] = (self.draw_code(gen, T, f_cycle[b], dev), fstep) + self.fc_slots(b, seq, f_cycle[b])
                RO.tick('source_setup')
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m = t1 - t0
                sl = slice(seq * T + t0, seq * T + t1)
                local = slice(t0, t1)
                ctx = dict(common, **buffers, **seqd, own=own, routes=qroutes, local=local, t0=t0,
                           mids=[z[sl] for z in mids], hs=[z[sl] for z in hs_all], idx32=[z[sl] for z in idx32],
                           sqx_in=[z[local] for z in seqd['sqx_seq']])
                if prune:
                    ctx.update(sorted_slices(self.sorted_routes, sl))
                delta = delta_all[:m * rows * W].view(m, rows, W)
                dirs = {}
                if sources:
                    for b in BLOCKS:
                        code, basis, qstep = src[f'q{b}']
                        rows_q = self.code_rows(code, q_cycle[b], *q_share[b], local, dev)
                        if basis is not None:
                            rows_q = 2 * torch.einsum('thdk,tphk->tphd', basis[local],
                                                      rows_q.view(m, P_q[b], H, s.readout_q_keys)).reshape(m, P_q[b], W)
                        dirs[f'q{b}'] = rows_q
                        dirs[f'f{b}'] = self.code_rows(src[f'f{b}'][0], f_cycle[b], *f_share[b], local, dev)
                RO.tick('dirs')
                dx0 = F.rms_norm(emb[sl].unsqueeze(1) + steps[0][sl].unsqueeze(-1) * bases[0].unsqueeze(0), (W,)) \
                    - x0[sl].unsqueeze(1)
                torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                for b in range(L):
                    n = in_rows[b]
                    use_self = s.readout_self and b < L - 1
                    out = score_block(delta, n, rows, b, ctxs[b], t0, bufs, wcat[b], cs_all, sn_all, use_self, H, D)
                    RO.tick('score')
                    score[sl, :n] += out
                    if keep:
                        per_block[b, sl, :n] = out
                    if b == L - 1:
                        break
                    push_block(delta, n, b, ctx, bufs, use_self)
                    RO.tick('oq')
                    if b in BLOCKS:
                        torch.mul(steps[b + 1][sl].unsqueeze(-1), bases[b + 1].unsqueeze(0), out=delta[:, groups[f'r{b}']])
                        if P_q[b]:
                            delta[:, groups[f'q{b}']] = self.q_moves(ctxs[b], b, dirs[f'q{b}'], src[f'q{b}'][2][local],
                                                                     local)
                    RO.tick('q_moves')
                    mlp_step(delta, mid_rows[b], b, ctx, dx0, P)
                    RO.tick('mlp')
                    if b in BLOCKS and P_f[b]:
                        _, fstep, idx, hval, valid = src[f'f{b}']
                        moves = self.fc_moves(b, dirs[f'f{b}'], fstep[local], idx[local], hval[local], valid[local])
                        torch.mul(moves, lam_r[b + 1], out=delta[:, groups[f'f{b}']])
                        del moves
                    RO.tick('fc_moves')
                for b in BLOCKS if sources else ():
                    _, _, qstep = src[f'q{b}']
                    q_credit[b][sl] = torch.einsum('tp,tpw->tw', score[sl, groups[f'q{b}']] / qstep[local],
                                                   dirs[f'q{b}']) / W
                    _, fstep, idx, hval, valid = src[f'f{b}']
                    slot = torch.einsum('tp,tpk->tk', score[sl, groups[f'f{b}']] / fstep[local],
                                        dirs[f'f{b}'][:, :, 1:]) / f_cycle[b]
                    fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                del dx0, dirs
                RO.tick('decode')
            del ctxs, own, qroutes, seqd, src
            getattr(self, 'route_cache', {}).pop(seq, None)
        del buffers, delta_all, bufs, scratch
        X = torch.stack([(score[:, groups[g]] / steps[k]) @ bases[k] / W
                         for k, g in enumerate(['emb'] + [f'r{i}' for i in BLOCKS])])
        X = e.group_sum(X)
        self.X = {'emb': X[0], **{i: X[1 + k] for k, i in enumerate(BLOCKS)}}
        self.q_credit, self.fc_credit = q_credit, fc_credit
        if keep:
            self.per_block = {(g, b): (per_block[b][:, groups[g]] / steps[k]) @ bases[k] / W
                              for k, g in enumerate(['emb'] + [f'r{i}' for i in BLOCKS]) for b in range(L)}
            self.groups, self.score, self.rows = groups, score, rows

    def fc_moves(self, b, code, fstep, idx, hval, valid):
        """readout's fc_moves with its elementwise steps in one kernel (bitwise; the gathered GEMM is readout's)."""
        if IMPL['fcrelu'] != 'fused':
            return super().fc_moves(b, code, fstep, idx, hval, valid)
        w_sel = self.e.model.transformer.h[b].mlp.c_proj.weight.float().T[idx]  # [m, slots, W] (readout's)
        m, P, cyc = code.shape
        ns = idx.shape[1]
        r = torch.empty(m, P, ns, device=code.device)
        fc_relu_kernel[(triton.cdiv(P, 32), triton.cdiv(ns, 128), m)](
            code.contiguous(), fstep.contiguous(), hval.contiguous(), valid.contiguous(), r, P, ns, cyc, 32, 128,
            num_warps=4, enable_fp_fusion=False)
        return torch.bmm(r, w_sel)

    def q_moves(self, c, b, dirs, qstep, local):
        if IMPL['qsrc'] != 'fused':
            return super().q_moves(c, b, dirs, qstep, local)
        e = self.e
        H = e.model.config.n_head
        D = e.width // H
        m, p, _ = dirs.shape
        keys, values, a, base = c['k'], c['values'], c['a'][:, local], c['out'][:, local]
        out = torch.empty(m, p, H, D, device=dirs.device)
        bp, warps, stages = QSRC_TILES[64 if p <= 64 else 128]
        qsrc_kernel[(triton.cdiv(p, bp), m * H)](
            dirs.contiguous(), c['qraw'].contiguous(), qstep.contiguous(), self.cs_all, self.sn_all, keys, values, a, base,
            out, local.start, p, H, D, keys.shape[1], *keys.stride(), *values.stride(), *a.stride(), *base.stride(), bp,
            64, FK.QUERY_PRECISION, EPS, num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        return F.linear(out.flatten(-2), e.model.transformer.h[b].attn.c_proj.weight.float())


class FastKVMethod(KV.Method, FastMethod):
    """readout_kv.py's Method on the faster pass (MRO: KV.Method.run / contexts / finish, then FastMethod's pass); its
    receiver rows run with the chain's routes and the fused own-query step."""

    def kv_sequence(self, seq, ctxs):
        e, s = self.e, self.e.s
        model, d = e.model, e.capture
        L, W, H = model.config.n_layer, e.width, model.config.n_head
        D, T, dev = W // H, d.length, e.device
        cs_all, sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()
        plans = {}
        for b in sorted({g[0] for g in self.kv_mine}):
            plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in self.kv_mine if bb == b], cs_all, sn_all))
        RO.tick('kv_plan')
        own, qroutes = self.routes_of(seq)
        inputs = [z.float().contiguous() for z in own[1]]
        seqd = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=self.kv_w['w_q_t'],
                    w_proj_t=self.kv_w['w_proj_t'], inputs=inputs, sqx_seq=[z.square().sum(-1) for z in inputs])
        buf = self.kv_buffers
        rows, C = self.kv_rows, self.kv_C
        score = torch.zeros(T, rows, device=dev)
        m_chunk = min(s.readout_chunk, T)
        first = min(g[0] for g in self.kv_mine)
        bufs = dict(se=buf['se'], db=buf['db'], st=buf['st'], up=buf['up'])
        prune = IMPL['mlp'] == 'prune'
        if prune and 'cut' not in buf:
            buf.update(prune_buffers(m_chunk, rows, dev))
        RO.tick('kv_routes')
        for t0 in range(0, T, m_chunk):
            t1 = min(t0 + m_chunk, T)
            m = t1 - t0
            sl, local = slice(seq * T + t0, seq * T + t1), slice(t0, t1)
            ctx = dict(self.kv_common, scratch=buf['scratch'], ybuf=buf['scratch'], scratch32=buf['scratch32'],
                       stats=buf['stats'], **seqd, own=own, routes=qroutes, local=local, t0=t0,
                       mids=[z[sl] for z in self.kv_mids], hs=[z[sl] for z in self.kv_hs],
                       idx32=[z[sl] for z in self.kv_idx32], sqx_in=[z[local] for z in seqd['sqx_seq']])
            if prune:
                ctx.update(cut=buf['cut'], **sorted_slices(self.sorted_routes, sl))
            delta = buf['delta'][:m * rows * W].view(m, rows, W)
            for b in range(first, L):
                n, c = self.kv_in[b], ctxs[b]
                use_self = s.readout_self and b < L - 1
                if n:
                    out = score_block(delta, n, rows, b, c, t0, bufs, self.kv_wcat[b], cs_all, sn_all, use_self, H, D)
                    score[local, :n] += out
                    RO.tick('kv_score')
                if b == L - 1:
                    break
                if n:
                    push_block(delta, n, b, ctx, bufs, use_self)
                    RO.tick('kv_oq')
                for g in self.kv_mine:
                    if g[0] == b:
                        delta[:, self.kv_first[g]:self.kv_first[g] + C] = self.moves(g, plans[g], c, t0, t1)
                RO.tick('kv_moves')
                if self.kv_mid[b]:
                    mlp_step(delta, self.kv_mid[b], b, ctx, delta, 0)
                RO.tick('kv_mlp')
        Rr = s.readout_k_receivers
        for g in self.kv_mine:
            b, kind = g
            p = plans[g]
            f = p['table'].clamp_min(0)
            live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
            got = score[:, self.kv_first[g]:self.kv_first[g] + C]
            reading = torch.zeros(H * T * Rr + 1, device=dev)
            reading.scatter_(0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, got, 0.).reshape(-1))
            reading = reading[:-1].view(H, T, Rr)
            out = slice(seq * T, (seq + 1) * T)
            if kind == 'k':
                c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
            else:
                self.g_credit[b][out] = reading.sum(-1).T / KV.GATE_SIGMA
            if KV.STATS['keep']:
                self.kv_keep[seq, g] = dict(p, reading=reading)
        del plans, own, qroutes, seqd, score


# ------------------------------------------------------------------------------------------------------- the settings


@dataclass
class Settings(KV.Settings):  # readout.py's and readout_kv.py's fields; E1 off unless asked
    readout_k_receivers: int = 0  # E1's R (readout_kv.py's default is 8): 0 = readout.py's method


def cut_estimate(self, x, y):  # replaces Cut.estimate: readout off = the record's (through readout.py's), called as is
    if not getattr(self.s, 'readout', False):
        return RO.cut_estimate(self, x, y)
    method = (FastKVMethod if KV.kv_on(self.s) else FastMethod)(self)
    gradients = method.estimate(x, y)
    if RO.STATS['keep'] or KV.STATS['keep']:
        self.readout_method = method
    return gradients


def install():
    """readout_kv.py's patch (with readout.py's and onesided's), then this one (idempotent): Settings and Cut.estimate
    (the recipe dict stays readout_kv.py's asdict). The speed stack goes on after (it touches none of these)."""
    if R.Settings is not Settings:
        KV.install()
        if R.Settings is not KV.Settings or R.asdict is not KV.asdict or R.Cut.estimate is not KV.cut_estimate:
            raise RuntimeError('fast_readout: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Settings, R.Cut.estimate = Settings, cut_estimate
    return R


# ======================================== module readout_hi
"""The readout carried past block 3: a root at mid4 for mlp.c_proj 3, the readout package (roots, sources, receivers)
on blocks 4-6, and the top root's readout.
"""
from dataclasses import dataclass
import json
import os

import torch
import torch.nn.functional as F

import fast_readout as FR  # noqa: E402  (loads readout_kv, readout, onesided and research/pr31/run_recipe.py as 'recipe')
KV, RO, R, FC = FR.KV, FR.RO, FR.R, FR.FC

LO = RO.BLOCKS  # E1's blocks (0-3)
FIRST = len(LO)  # 4: the first block past E1's
HI_SEED = 7_777_777  # the hi pass's per-sequence generator: readout's seed + this + 104729 x sequence (readout's: no offset)
FIELDS = ('readout_hi_blocks', 'readout_hi_out3', 'readout_in_roots', 'readout_hi_fc_cycle', 'readout_hi_top_root')
STATS = dict(calls=0, every=200, keep=False, seconds=[])  # keep: checks read the last estimate's pieces


# ------------------------------------------------------------------------------------------------------- the settings


@dataclass
class Settings(FR.Settings):  # readout.py's, readout_kv.py's and fast_readout.py's fields plus E3's
    readout_hi_blocks: tuple = (4, 5)  # E3b: blocks past 3 with E1's package (a run from 4; () = none)
    readout_hi_out3: bool = True  # E3a: mlp.c_proj 3 takes lam_4 (inputs_4 + R_4) (a root at mid4 when 4 is no hi block)
    readout_in_roots: tuple = ()  # blocks j (>= 5, j - 1 a hi block) whose input own-query path goes into out_{j-1}
    readout_hi_fc_cycle: tuple = (512, 128, 128)  # c_fc source cycle of blocks 4, 5, 6 when they are hi blocks
    readout_hi_top_root: bool = True  # E3a's rule one block up: mlp.c_proj of the top hi block b takes lam_{b+1} (inputs_{b+1}
    #   + R_{b+1}) from a readout root at mid_{b+1} (no root when b + 1 is the last block: there R = 0, the identity is exact)

    def __post_init__(self):
        super().__post_init__()
        if not self.readout:
            return
        self.readout_hi_blocks = tuple(self.readout_hi_blocks)
        self.readout_in_roots = tuple(self.readout_in_roots)
        self.readout_hi_fc_cycle = tuple(self.readout_hi_fc_cycle)
        L, W = self.model.n_layer, self.model.n_embd
        hb, ir, cyc = self.readout_hi_blocks, self.readout_in_roots, self.readout_hi_fc_cycle
        if (any(type(b) is not int for b in hb + ir) or hb != tuple(range(FIRST, FIRST + len(hb)))
                or (hb and hb[-1] > L - 2) or type(self.readout_hi_out3) is not bool
                or type(self.readout_hi_top_root) is not bool
                or len(set(ir)) < len(ir) or not set(ir) <= set(range(FIRST + 1, L)) or not {j - 1 for j in ir} <= set(hb)
                or len(cyc) != L - 1 - FIRST
                or any(type(c) is not int or c < 32 or c & (c - 1) or c > 4 * W for c in cyc)):
            raise ValueError(f'readout_hi: readout_hi_blocks a run of blocks from {FIRST} up to n_layer - 2 (or []), '
                             f'readout_hi_out3 / readout_hi_top_root true / false, readout_in_roots distinct blocks '
                             f'{FIRST + 1}..n_layer - 1 each with block j - 1 a hi block (so both halves of block j - 1\'s '
                             f'MLP take the path), readout_hi_fc_cycle one power of two 32..{4 * W} per block '
                             f'{FIRST}..n_layer - 2')


def hi_on(s):
    return bool(getattr(s, 'readout', False) and (getattr(s, 'readout_hi_blocks', ()) or getattr(s, 'readout_hi_out3', False)
                                                  or getattr(s, 'readout_in_roots', ())))


def asdict(settings):  # E3 off: E1's (readout_kv.py's) recipe dict (config.json, resume)
    state = KV.asdict(settings)
    if not hi_on(settings):
        for key in FIELDS:
            state.pop(key, None)
    return state


def balance(groups, split, L):
    """readout_kv.groups_of's split of receiver groups over a pair: heaviest first (rows inserted at block b pass blocks
    b + 1 .. L - 1) onto the least loaded part, ties to the lower part."""
    load, owner = [0] * split, {}
    for g in sorted(groups):
        part = min(range(split), key=lambda q: (load[q], q))
        owner[g] = part
        load[part] += L - 1 - g[0]
    return owner


# ------------------------------------------------------------------------------------------------------- the method


class HiMixin:
    """fast_readout's Method (E1's FastKVMethod or the readout's FastMethod) plus the hi pass: run() plans it, the readout
    pass runs as is and calls contexts(seq) once per sequence, which returns the same contexts after this GPU's hi rows of
    that sequence have run (their own buffers); path_credits() hands the hi roots to the record in place of extra_b;
    finish() rebuilds the writers it changes and adds the sources' and receivers' credit."""

    def run(self, chain, targets):
        self.hi_plan()
        self.hi_live = True
        try:
            super().run(chain, targets)
        finally:
            self.hi_live = False
            self.hi_buf = None
        self.hi_decode()

    def contexts(self, seq, targets):
        ctxs = super().contexts(seq, targets)  # readout's contexts (E1's receiver rows of this sequence have run)
        if getattr(self, 'hi_live', False) and self.hi_work:
            RO.tick('contexts')
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.hi_contexts(seq, ctxs)
            self.hi_sequence(seq, ctxs)
            end.record()
            self.hi_events.append((start, end))
            RO.tick('hi')
        return ctxs

    def path_credits(self, chain, targets, raw_values=None):
        """E1's (the readout pass, the hi pass inside it); extra_b of the hi blocks becomes R_b."""
        out, inputs = super().path_credits(chain, targets, raw_values)
        for b in self.hi_blocks:
            out[b] = self.X[b]
        return out, inputs

    # ------------------------------------------------------------------------------------------------- the plan

    def hi_plan(self):
        e, s = self.e, self.e.s
        model, d = e.model, e.capture
        L, W, H = model.config.n_layer, e.width, model.config.n_head
        dev = e.device
        hb = tuple(s.readout_hi_blocks)
        top = {hb[-1] + 1} if hb and s.readout_hi_top_root and hb[-1] + 1 <= L - 2 else set()  # E3a's rule one block up
        roots = sorted(set(hb) | ({FIRST} if s.readout_hi_out3 else set()) | top)
        in_roots = tuple(sorted(s.readout_in_roots))
        lo, hi = e.share(W)
        P = hi - lo
        src = s.readout_sources
        q_cycle = {b: H * s.readout_q_keys for b in hb}
        f_cycle = {b: s.readout_hi_fc_cycle[b - FIRST] for b in hb}
        q_share = {b: e.share(q_cycle[b]) for b in hb}
        f_share = {b: e.share(f_cycle[b]) for b in hb}
        P_q = {b: q_share[b][1] - q_share[b][0] if src else 0 for b in hb}
        P_f = {b: f_share[b][1] - f_share[b][0] if src else 0 for b in hb}
        groups = []
        if isinstance(self, KV.Method) and s.readout_k_receivers:  # E1 on: its receiver experiments on the hi blocks
            for b in hb:
                groups.append((b, 'k'))
                if s.readout_gate and model.transformer.h[b].attn.ve_gate is not None:
                    groups.append((b, 'g'))
        owner = balance(groups, e.split, L)
        mine = [g for g in groups if owner[g] == e.part]
        C = KV.group_rows(s, H) if groups else 0
        slots, in_rows, mid_rows, at = {}, [0] * L, [0] * L, 0
        for b in range(L):
            in_rows[b] = at
            if FIRST <= b < L - 1:
                if b in roots:
                    slots[f'r{b}'], at = slice(at, at + P), at + P
                if b in hb and P_q[b]:
                    slots[f'q{b}'], at = slice(at, at + P_q[b]), at + P_q[b]
                for g in mine:
                    if g[0] == b:
                        slots[g], at = slice(at, at + C), at + C
            mid_rows[b] = at
            if b in hb and P_f[b]:
                slots[f'f{b}'], at = slice(at, at + P_f[b]), at + P_f[b]
        self.hi_blocks, self.hi_roots, self.hi_in = hb, roots, in_roots
        self.hi_mproj = sorted(({FIRST - 1} if s.readout_hi_out3 else set()) | set(hb) | {j - 1 for j in in_roots})
        self.hi_cfg = dict(P=P, lo=lo, hi=hi, q_cycle=q_cycle, f_cycle=f_cycle, q_share=q_share, f_share=f_share, P_q=P_q,
                           P_f=P_f, groups=groups, owner=owner, mine=mine, C=C, slots=slots, in_rows=in_rows,
                           mid_rows=mid_rows, rows=at)
        self.hi_work = bool(at or in_roots)
        n = d.batch * d.length
        self.hi_score = torch.zeros(n, at, device=dev)
        self.hi_pb = torch.zeros(L, n, at, device=dev) if STATS['keep'] else None  # checks: per-block scores
        self.hi_q = {b: torch.zeros(n, W, device=dev) for b in hb if P_q[b]}
        self.hi_fc = {b: torch.zeros(n, 4 * W, device=dev) for b in hb if P_f[b]}
        self.hi_k = {b: torch.zeros(n, W, device=dev) for b, kind in mine if kind == 'k'}
        self.hi_g = {b: torch.zeros(n, H, device=dev) for b, kind in mine if kind == 'g'}
        self.hi_Q = {j: torch.zeros(n, W, device=dev) for j in in_roots}  # this GPU's share of each in-root's decode
        self.hi_events, self.hi_keep = [], {}
        emb, x0, mids, outs, routes, y = self.chain_args
        # the roots: the chain's bases and steps of mid_b (readout.py's arithmetic for its roots)
        self.hi_bases = {b: self.bases[b + 1][lo:hi].contiguous() for b in roots}
        self.hi_steps = {b: s.chain_sigma * mids[b].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for b in roots}
        # readout's source seed (a private copy of the generator add_future_gradients draws from: both GPUs of a pair alike)
        self.hi_seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        if not self.hi_work:
            return
        blocks = model.transformer.h
        NC = 2 * W + 16  # the staged scorer's projection: d W_k^T | d W_v^T | d[:32] W_g^T (padded to 16), as readout's
        wgt = [None if blk.attn.ve_gate is None else F.pad(blk.attn.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16)
               .contiguous() for blk in blocks]
        self.hi_w = dict(w_fc=[blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in blocks],
                         w_out_t=[blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in blocks],
                         w_q_t=[blk.attn.c_q.weight.to(torch.bfloat16).T.contiguous() for blk in blocks],
                         w_proj_t=[blk.attn.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in blocks],
                         wcat=[torch.cat([blk.attn.c_k.weight.to(torch.bfloat16).T.contiguous(),
                                          blk.attn.c_v.weight.to(torch.bfloat16).T.contiguous(),
                                          torch.zeros(W, 16, device=dev, dtype=torch.bfloat16) if wgt[j] is None
                                          else F.pad(wgt[j], (0, 0, 0, W - 32))], 1).contiguous()
                               for j, blk in enumerate(blocks)],
                         wproj={b: blocks[b].attn.c_proj.weight.float() for b in {g[0] for g in mine}},
                         wq={j: blocks[j].attn.c_q.weight.float() for j in in_roots})
        self.hi_mids = [z.float().contiguous() for z in mids]
        self.hi_idx32 = [r[0].to(torch.int32).contiguous() for r in routes]
        self.hi_hs = [r[1].float().contiguous() for r in routes]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        self.hi_common = dict(lam_r=lam_r, lam_0=lam_0, lams=torch.stack([lam_r, lam_0], 1).contiguous(),
                              w_fc=self.hi_w['w_fc'], w_out_t=self.hi_w['w_out_t'])
        m, rows = min(s.readout_chunk, d.length), max(at, 1)
        units = max(r[0].shape[1] for r in routes)
        self.hi_buf = dict(scratch=torch.empty(m * rows * max(W, units), device=dev, dtype=torch.bfloat16),
                           scratch32=torch.empty(m * rows * W, device=dev), stats=torch.empty(m * rows * 2, device=dev),
                           delta=torch.empty(m * rows * W, device=dev),
                           se=torch.empty(m * rows * W, device=dev, dtype=torch.bfloat16),
                           db=torch.empty(m * rows * W, device=dev, dtype=torch.bfloat16),
                           st=torch.empty(m * rows * 2, device=dev),
                           up=torch.empty(m * rows * NC, device=dev, dtype=torch.bfloat16))
        if FR.IMPL['mlp'] == 'prune':
            self.hi_buf.update(FR.prune_buffers(m, rows, dev))

    # ------------------------------------------------------------------------------------------ one sequence

    def hi_contexts(self, seq, ctxs):
        """The pieces the q sources and in-roots need at their blocks (readout's contexts keep them for blocks 0-3): the
        captured raw query, the clean rotated / normed keys, the values (gated value embedding included), the clean
        attention and outputs (readout's formula, R.attention_state)."""
        e = self.e
        d, model = e.capture, e.model
        T, H = d.length, model.config.n_head
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        need = (set(self.hi_blocks) if e.s.readout_sources else set()) | set(self.hi_in)
        for j in sorted(need):
            c = ctxs[j]
            if 'qraw' in c:
                continue
            prefix = f'transformer.h.{j}.attn.'
            qraw, kraw = (d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k'))
            values = c['VT'].transpose(0, 1)  # [H, T, D]: readout's values (the gated value embedding included)
            qp, kp, a, out = R.attention_state(qraw, kraw, values[None], cos, sin, H, model.window_sizes[j][0])
            c.update(qraw=qraw[0], k=kp[0], values=values, a=a[0], out=out[0])

    def oq_scores(self, c, j, moves, credit, local):
        """Scores of block-j input moves [m, p, W] through the token's own query only: <W_proj (Attn(q + dq) - Attn(q)),
        chain_j> with dq = (norm(x + move) - norm(x)) W_q^T (the record's RMSNorm change), the exact softmax over every
        key (every key / value clean), credit = the measured mid_j credit of these tokens [m, W]. [m, p]"""
        x = c['x'][local]
        dq = torch.matmul(R.norm_delta(x[:, None], moves), self.hi_w['wq'][j].T)
        mid = self.q_moves(c, j, dq.contiguous(), torch.ones(x.shape[0], device=x.device), local)
        return torch.einsum('tpw,tw->tp', mid, credit)

    def hi_moves(self, g, p, c, t0, t1):
        """readout_kv.Method.moves (E1's receiver moves of group g at the receivers' mid_b), its arithmetic, with this
        pass's W_proj."""
        b, kind = g
        H = self.e.model.config.n_head
        T, W = self.e.capture.length, self.e.width
        D, Rr = W // H, self.e.s.readout_k_receivers
        tbl = p['table'][t0:t1]
        m, C = tbl.shape
        f = tbl.clamp_min(0)
        live = (tbl >= 0) & p['live'].reshape(-1)[f]
        coef = torch.where(live, p['coef'].reshape(-1)[f], torch.zeros((), device=f.device))
        h, t = f // (T * Rr), (f // Rr) % T
        if kind == 'k':
            s_ = torch.arange(t0, t1, device=f.device)[:, None].expand(m, C)
            vec = c['VT'][t, h] - c['OT'][s_, h]
        else:
            vec = c['ve'][t, h]
        head = torch.zeros(m, C, H, D, device=f.device)
        head.scatter_(2, h[:, :, None, None].expand(m, C, 1, D), (coef[..., None] * vec)[:, :, None])
        return F.linear(head.view(m, C, W), self.hi_w['wproj'][b])

    def hi_sequence(self, seq, ctxs):
        """This GPU's hi rows of one sequence: the experiments, the push and scoring with readout's kernels (fast_readout's
        launches), the sources' / receivers' / in-roots' decode into this GPU's share."""
        e, s = self.e, self.e.s
        model, d = e.model, e.capture
        L, W, H = model.config.n_layer, e.width, model.config.n_head
        D, T, dev = W // H, d.length, e.device
        cf = self.hi_cfg
        slots, rows, C = cf['slots'], cf['rows'], cf['C']
        lam_r = self.hi_common['lam_r'] if rows else model.resid_lambdas.float()
        cs_all, sn_all = self.cs_all, self.sn_all
        gen = torch.Generator(dev).manual_seed(self.hi_seed + HI_SEED + 104729 * seq)
        src = {}
        for b in self.hi_blocks:  # readout's source draws, in its order (q then c_fc per block), this pass's generator
            if not (cf['P_q'][b] or cf['P_f'][b]):
                continue
            c = ctxs[b]
            basis = self.q_basis(c, s.readout_q_keys)
            qstep = s.readout_sigma * c['qraw'].square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            src[f'q{b}'] = (self.draw_code(gen, T, cf['q_cycle'][b], dev), basis, qstep)
            hfull = d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float()
            fstep = s.readout_sigma * hfull.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            src[f'f{b}'] = (self.draw_code(gen, T, cf['f_cycle'][b], dev), fstep) + self.fc_slots(b, seq, cf['f_cycle'][b])
        for j in self.hi_in:  # the in-roots' complete cycles of block j's input
            xin = ctxs[j]['x']
            src[f'in{j}'] = (self.draw_code(gen, T, W, dev),
                             s.readout_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
        RO.tick('hi_setup')
        plans = {}
        for b in sorted({g[0] for g in cf['mine']}):
            plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in cf['mine'] if bb == b], cs_all, sn_all))
        RO.tick('hi_plan')
        m_chunk = min(s.readout_chunk, T)
        if rows:
            own, qroutes = self.routes_of(seq)
            inputs = [z.float().contiguous() for z in own[1]]
            seqd = dict(cos=own[6].contiguous(), sin=own[7].contiguous(), w_q_t=self.hi_w['w_q_t'],
                        w_proj_t=self.hi_w['w_proj_t'], inputs=inputs, sqx_seq=[z.square().sum(-1) for z in inputs])
            buf = self.hi_buf
            bufs = dict(se=buf['se'], db=buf['db'], st=buf['st'], up=buf['up'])
            prune = FR.IMPL['mlp'] == 'prune'
        fc_in = {j: f'f{j - 1}' for j in self.hi_in if f'f{j - 1}' in slots}  # c_fc sources scored on the in-root path
        for t0 in range(0, T, m_chunk):
            t1 = min(t0 + m_chunk, T)
            m = t1 - t0
            sl, local = slice(seq * T + t0, seq * T + t1), slice(t0, t1)
            dirs = {}
            if rows:
                ctx = dict(self.hi_common, scratch=buf['scratch'], ybuf=buf['scratch'], scratch32=buf['scratch32'],
                           stats=buf['stats'], **seqd, own=own, routes=qroutes, local=local, t0=t0,
                           mids=[z[sl] for z in self.hi_mids], hs=[z[sl] for z in self.hi_hs],
                           idx32=[z[sl] for z in self.hi_idx32], sqx_in=[z[local] for z in seqd['sqx_seq']])
                if prune:
                    ctx.update(cut=buf['cut'], **FR.sorted_slices(self.sorted_routes, sl))
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                for b in self.hi_blocks:  # this chunk's source directions (readout's construction)
                    if f'q{b}' in slots:
                        code, basis, _ = src[f'q{b}']
                        rows_q = self.code_rows(code, cf['q_cycle'][b], *cf['q_share'][b], local, dev)
                        dirs[f'q{b}'] = 2 * torch.einsum('thdk,tphk->tphd', basis[local], rows_q.view(
                            m, cf['P_q'][b], H, s.readout_q_keys)).reshape(m, cf['P_q'][b], W)
                    if f'f{b}' in slots:
                        dirs[f'f{b}'] = self.code_rows(src[f'f{b}'][0], cf['f_cycle'][b], *cf['f_share'][b], local, dev)
                RO.tick('hi_dirs')
                for b in range(FIRST, L):
                    n = cf['in_rows'][b]
                    c = ctxs[b]
                    use_self = s.readout_self and b < L - 1
                    if n:
                        out = FR.score_block(delta, n, rows, b, c, t0, bufs, self.hi_w['wcat'][b], cs_all, sn_all,
                                             use_self, H, D)
                        self.hi_score[sl, :n] += out
                        if self.hi_pb is not None:
                            self.hi_pb[b, sl, :n] = out
                        RO.tick('hi_score')
                    if n and b in fc_in:  # block b - 1's c_fc source rows at x_b (before the push; the last block too):
                        g = slots[fc_in[b]]  # the in-root's own-query path
                        self.hi_score[sl, g] += self.oq_scores(c, b, delta[:, g], self.chain[b][sl], local)
                    if b == L - 1:
                        break
                    if n:
                        FR.push_block(delta, n, b, ctx, bufs, use_self)  # own query (+ self-edge), c_proj added
                        RO.tick('hi_oq')
                    if f'r{b}' in slots:  # the chain's mid_b root moves
                        torch.mul(self.hi_steps[b][sl].unsqueeze(-1), self.hi_bases[b].unsqueeze(0),
                                  out=delta[:, slots[f'r{b}']])
                    if f'q{b}' in slots:
                        delta[:, slots[f'q{b}']] = self.q_moves(c, b, dirs[f'q{b}'], src[f'q{b}'][2][local], local)
                    for g in cf['mine']:
                        if g[0] == b:
                            delta[:, slots[g]] = self.hi_moves(g, plans[g], c, t0, t1)
                    RO.tick('hi_moves')
                    if cf['mid_rows'][b]:
                        FR.mlp_step(delta, cf['mid_rows'][b], b, ctx, delta, 0)  # block b's MLP, block b + 1's mixing
                        RO.tick('hi_mlp')
                    if f'f{b}' in slots:
                        _, fstep, idx, hval, valid = src[f'f{b}']
                        moves = self.fc_moves(b, dirs[f'f{b}'], fstep[local], idx[local], hval[local], valid[local])
                        torch.mul(moves, lam_r[b + 1], out=delta[:, slots[f'f{b}']])
                        del moves
                    RO.tick('hi_fc_moves')
                for b in self.hi_blocks:  # this chunk's tokens are complete: decode their sources (readout's decode)
                    if f'q{b}' in slots:
                        qstep = src[f'q{b}'][2]
                        self.hi_q[b][sl] = torch.einsum('tp,tpw->tw', self.hi_score[sl, slots[f'q{b}']] / qstep[local],
                                                        dirs[f'q{b}']) / W
                    if f'f{b}' in slots:
                        _, fstep, idx, hval, valid = src[f'f{b}']
                        slot = torch.einsum('tp,tpk->tk', self.hi_score[sl, slots[f'f{b}']] / fstep[local],
                                            dirs[f'f{b}'][:, :, 1:]) / cf['f_cycle'][b]
                        self.hi_fc[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                RO.tick('hi_decode')
            for j in self.hi_in:  # the in-roots: this GPU's rows of the cycle, decoded into its share
                code, istep = src[f'in{j}']
                rows_in = self.code_rows(code, W, cf['lo'], cf['hi'], local, dev)  # [m, P, W]
                sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                self.hi_Q[j][sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
                RO.tick('hi_in')
            del dirs
        Rr = s.readout_k_receivers
        out = slice(seq * T, (seq + 1) * T)
        for g in cf['mine']:  # decode (readout_kv's)
            b, kind = g
            p = plans[g]
            f = p['table'].clamp_min(0)
            live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
            got = self.hi_score[out, slots[g]]
            reading = torch.zeros(H * T * Rr + 1, device=dev)  # the last entry: a sink for the empty slots
            reading.scatter_(0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, got, 0.).reshape(-1))
            reading = reading[:-1].view(H, T, Rr)
            if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                self.hi_k[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
            else:
                self.hi_g[b][out] = reading.sum(-1).T / KV.GATE_SIGMA
            if STATS['keep']:
                self.hi_keep[seq, g] = dict(p, reading=reading)
        del plans, src

    def hi_decode(self):
        """The roots' and in-roots' decode, the pair's halves of every cycle summed (one group_sum)."""
        e, W = self.e, self.e.width
        slots = self.hi_cfg['slots']
        parts = [(self.hi_score[:, slots[f'r{b}']] / self.hi_steps[b]) @ self.hi_bases[b] / W for b in self.hi_roots]
        parts += [self.hi_Q[j] for j in self.hi_in]
        if parts:
            X = e.group_sum(torch.stack(parts))
            for k, b in enumerate(self.hi_roots):
                self.X[b] = X[k]
            self.hi_Q = {j: X[len(self.hi_roots) + k] for k, j in enumerate(self.hi_in)}
        if self.hi_pb is not None:  # checks: the per-block readout of each root (this GPU's share)
            self.hi_per_block = {(b, j): (self.hi_pb[j][:, slots[f'r{b}']] / self.hi_steps[b]) @ self.hi_bases[b] / W
                                 for b in self.hi_roots for j in range(b + 1, len(self.hi_pb))}
            self.hi_pb = None
        if not STATS['keep']:
            self.hi_score = None

    # ------------------------------------------------------------------------------------------------ finishing

    def finish(self, gradients):
        super().finish(gradients)  # E1's: mlp.c_proj 0-3 (out_3 = extra_3), wte, the 0-3 sources and receivers
        e, d = self.e, self.e.capture
        lam_r = e.model.resid_lambdas.float()
        if not e.part:  # the writers and scalars E3 changes, formed in full by a pair's first GPU (the record's rule)
            emb, x0, mids, outs, routes, y = self.chain_args
            for j, Q in self.hi_Q.items():  # an in-root's query path at x_j: the mixing scalars of block j (linear)
                gradients['resid_lambdas'][j] += (Q * (x0 if j == 0 else outs[j - 1])).sum()
                gradients['x0_lambdas'][j] += (Q * x0).sum()
            for i in self.hi_mproj:
                p = f'transformer.h.{i}.'
                cross = lam_r[i + 1] * (self.inputs[i + 1] + self.X[i + 1]) if i + 1 in self.hi_roots else self.extra[i]
                x_out = lam_r[i + 1] * self.chain[i + 1] + cross
                if i + 1 in self.hi_Q:
                    x_out = x_out + lam_r[i + 1] * self.hi_Q[i + 1]
                gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
        for b, credit in self.hi_q.items():  # every GPU: its share of the cycles (the pair's sum is the decode)
            p = f'transformer.h.{b}.attn.c_q'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))
        for b, credit in self.hi_fc.items():
            p = f'transformer.h.{b}.mlp.c_fc'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))
        for credits, site in ((self.hi_k, 'attn.c_k'), (self.hi_g, 'attn.ve_gate')):  # every GPU: its own groups
            for b, credit in credits.items():
                p = f'transformer.h.{b}.{site}'
                gradients[p + '.weight'].add_(credit.T @ d.inputs[p].flatten(0, 1))

    def note(self):
        super().note()
        STATS['calls'] += 1
        STATS['seconds'].append(self.hi_events)
        del STATS['seconds'][:-50]
        n = STATS['calls']
        if os.environ.get('RANK', '0') != '0' or (n != 1 and n % STATS['every']):
            return
        torch.cuda.synchronize()
        sec = [sum(a.elapsed_time(b) for a, b in ev) / 1e3 for ev in STATS['seconds']]
        cos = lambda a, c: ((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-30)).item()
        roots = {b: dict(cos_identity=round(cos(self.X[b], self.extra[b]), 3),
                         ratio=round((self.X[b].norm() / self.extra[b].norm().clamp_min(1e-30)).item(), 3))
                 for b in self.hi_roots}
        ins = {j: round((self.hi_Q[j].norm() / self.chain[j].norm().clamp_min(1e-30)).item(), 3) for j in self.hi_in}
        print(f'readout_hi estimate {n}: hi blocks {list(self.hi_blocks)}, roots vs identity extra {json.dumps(roots)}, '
              f'|Q_j| / |chain_j| {json.dumps(ins)}, writers rebuilt {self.hi_mproj}, {self.hi_cfg["rows"]} hi rows per token '
              f'(groups here {["%s%d" % (k, b) for b, k in self.hi_cfg["mine"]]}); hi pass '
              f'{sum(sec) / max(len(sec), 1):.3f} s', flush=True)


class HiKVMethod(HiMixin, FR.FastKVMethod):
    """E3 on E1 (readout + receivers)."""


class HiMethod(HiMixin, FR.FastMethod):
    """E3 on the readout alone (readout_k_receivers 0)."""


def cut_estimate(self, x, y):  # replaces Cut.estimate: E3 off = fast_readout's (E1 / readout / the record), called as is
    if not hi_on(self.s):
        return FR.cut_estimate(self, x, y)
    method = (HiKVMethod if KV.kv_on(self.s) else HiMethod)(self)
    gradients = method.estimate(x, y)
    if STATS['keep'] or RO.STATS['keep'] or KV.STATS['keep']:
        self.readout_method = method
    return gradients


def install():
    """fast_readout.py's patch (with readout_kv.py's, readout.py's and onesided's), then this one (idempotent): Settings,
    the recipe dict and Cut.estimate. The speed stack goes on after (it touches none of these)."""
    if R.Settings is not Settings:
        FR.install()
        if R.Settings is not FR.Settings or R.asdict is not KV.asdict or R.Cut.estimate is not FR.cut_estimate:
            raise RuntimeError('readout_hi: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Settings, R.asdict, R.Cut.estimate = Settings, asdict, cut_estimate
    return R


# ======================================== module shared_push
"""The chain shares its pushes with the readout roots: the chain's experiments run with the self-edge at every block,
roots from the top down, so each root's rows are scored for the readout on the way up and no row is pushed twice.
"""
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
import triton

import readout_hi as HI  # noqa: E402  (loads fast_readout, readout_kv, readout, onesided and the recipe module)
FR, KV, RO, R, FC = HI.FR, HI.KV, HI.RO, HI.R, HI.FC
FM, FC2 = RO.FM, FR.FC2

EPS, BLOCKS, FIRST = RO.EPS, RO.BLOCKS, HI.FIRST
STATS = dict(calls=0, every=200, keep=False, seconds=[])


@dataclass
class Settings(HI.Settings):  # readout_hi's fields and the switch
    readout_shared: bool = False  # stage 3.1: the self-edged chain scores the readout roots (needs readout_self)

    def __post_init__(self):
        super().__post_init__()
        if type(self.readout_shared) is not bool or self.readout_shared and not (self.readout and self.readout_self):
            raise ValueError('readout_shared: true / false; true needs readout and readout_self')


def on(s):
    return bool(getattr(s, 'readout', False) and getattr(s, 'readout_shared', False))


def asdict(settings):  # off: readout_hi's recipe dict
    state = HI.asdict(settings)
    if not on(settings):
        state.pop('readout_shared', None)
    return state


class Shared:
    """readout_hi's Method (first in the MRO) with the self-edged chain (chain_credits) and the readout pass without its
    roots (run). The record's estimate around them, path_credits and finish (writers, sources, receivers, in-roots) are
    readout_hi's / fast_readout's / readout's as they are."""
    SELF_EDGE = True  # checks: False = fast_chain's experiments (no self-edge) on these kernels

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        """The self-edged chain (its credits as the record's chain_credits returns them); the readout roots' R_r in X."""
        e, s = self.e, self.e.s
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'shared_push: the fast_chain kernels do not cover this setting ({reason})')
        model, d, dev = e.model, e.capture, e.device
        L, W, T, n = model.config.n_layer, e.width, d.length, emb.shape[0]
        self.sp_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.sp_events[0].record()
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P, scalars, h = hi - lo, e.part == 0, s.scalar_step
        self.chain_args = (emb, x0, mids, outs, routes, y)
        self.bases = [e.basis() for _ in range(L + 1)]  # the record's draws in its order: emb, mid0, mid1, ...
        bases = [B[lo:hi].contiguous() for B in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb, *mids]]
        self.prepare(mids, routes)
        head = model.lm_head.weight.float()[:model.config.vocab_size]
        xL = outs[-1].float().contiguous()
        base, target_w, weight_t = xL @ head.T, head[y.clamp(min=0)], head.T.to(torch.bfloat16).contiguous()
        if FC.STATE['exact_blocks'] is None:
            FC.STATE['exact_blocks'] = torch.zeros(1, device=dev, dtype=torch.int32)
        hx = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1),
                  **{k: FC.STATE[k] for k in ('exact_head', 'series_limit', 'exact_blocks')})
        credit, score = torch.empty(L + 1, n, W, device=dev), torch.zeros(L + 1, n, P, device=dev)
        lam = torch.zeros(L, 2, device=dev)
        m_chunk, k4 = min(s.readout_chunk, T), 4 * scalars  # the scalar probes per root (a pair's first GPU)
        buf = self.buffers(m_chunk, P + L * k4)
        self.route_cache, self.ctx_cache = {}, {}
        for seq in range(d.batch):
            own = e.own_query_inputs(seq)
            self.route_cache[seq] = dict(own=own, routes=(s.query_keys, FC.query_routes(own, s.query_keys)))
            ctxs = [self.clean(seq, j) for j in range(L)]
            for r in range(L - 1, -2, -1):  # root r: mid_r, from the top down; r = -1: the embedding with every root's
                k = P + L * k4 if r < 0 else P  # scalar probes (each from its mid stream: P rows per phase elsewhere)
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1) | hx | dict(xL=xL[sl], base=base[sl], target_w=target_w[sl],
                                                                       y=y[sl])
                    delta, dx0 = buf['delta'][:(t1 - t0) * k * W].view(t1 - t0, k, W), None
                    if r < 0:
                        dx0 = F.rms_norm(emb[sl, None] + steps[0][sl, None] * bases[0], (W,)) - x0[sl, None]
                        torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                    else:
                        torch.mul(steps[r + 1][sl, None], bases[r + 1], out=delta)
                        self.mlp(x, delta, P, r)
                    for b in range(r + 1, L):
                        n_b = P + b * k4 if r < 0 else P
                        score[r + 1, sl] += self.score(x, delta, n_b, b, self.SELF_EDGE)[:, :P]
                        self.push(x, delta, n_b, b)
                        if r < 0 and k4:  # the symmetric probes of block b's two mixing scalars start at mid_b
                            prev = (x0 if b == 0 else outs[b - 1])[sl]
                            delta[:, n_b:n_b + 4] = torch.stack([prev, x0[sl], -prev, -x0[sl]], 1) * h
                        self.mlp(x, delta, n_b + k4 * (r < 0), b, dx0, P)
                    change = FC.head_fast(delta, x) * (y[sl] >= 0).unsqueeze(1)
                    credit[r + 1, sl] = (change[:, :P] / steps[r + 1][sl]) @ bases[r + 1] / W
                    if r < 0 and k4:  # the central difference over 2h per scalar
                        pair = change[:, P:].view(t1 - t0, L, 4) / h
                        lam += (pair[..., :2] - pair[..., 2:]).sum(0) / 2
                tok = e.group_sum(credit[r + 1, seq * T:(seq + 1) * T])  # the pair's halves: root r is complete
                if r >= 0:  # its hub's moment tables, read by every root below it
                    self.tables(ctxs[r], r, tok @ model.transformer.h[r].attn.c_proj.weight.float())
            self.ctx_cache[seq] = ctxs
        X = e.group_sum(torch.stack([(score[k] / steps[k]) @ bases[k] / W for k in range(L + 1)]))
        self.X = {'emb': X[0], **dict(enumerate(X[1:]))}
        e.group_sum(lam)
        self.sp_events[1].record()
        return {'emb': credit[0], **dict(enumerate(credit[1:])), **{('lam', r): lam[r] for r in range(L)}}

    def run(self, chain, targets):
        """The readout pass without the roots: E1's q / c_fc sources and c_k / ve_gate receivers (blocks 0-3), E3's on
        the hi blocks, the in-roots; readout's draws, layouts, moves and decodes (readout.py, readout_kv.py, readout_hi.py),
        the chain's contexts and kernels; the receiver s = t for the c_fc rows at their first block only."""
        e, s = self.e, self.e.s
        model, d, dev = e.model, e.capture, e.device
        L, W, H, T = model.config.n_layer, e.width, model.config.n_head, d.length
        self.sp_events[2].record()
        lam_r, Rr = model.resid_lambdas.float(), s.readout_k_receivers
        hb, ins = tuple(s.readout_hi_blocks), tuple(sorted(s.readout_in_roots))
        top = {hb[-1] + 1} if hb and s.readout_hi_top_root and hb[-1] + 1 <= L - 2 else set()
        self.hi_blocks, self.hi_in, self.hi_q, self.hi_fc, self.hi_k, self.hi_g = hb, ins, {}, {}, {}, {}
        self.hi_roots = sorted(set(hb) | ({FIRST} if s.readout_hi_out3 else set()) | top)
        self.hi_mproj = sorted(({FIRST - 1} if s.readout_hi_out3 else set()) | set(hb) | {j - 1 for j in ins})
        self.hi_w = dict(wq={j: model.transformer.h[j].attn.c_q.weight.float() for j in ins})
        src_b = BLOCKS + hb if s.readout_sources else ()
        cyc = {('q', b): W if b in BLOCKS and b in s.readout_q_full else H * s.readout_q_keys for b in src_b}
        cyc |= {('f', b): (s.readout_fc_cycle + s.readout_hi_fc_cycle)[b] for b in src_b}
        share = {key: e.share(c) for key, c in cyc.items()}
        mine1 = mine2 = []  # this GPU's receiver groups: E1's (blocks 0-3) and E3's (hi blocks), split over the pair as there
        if KV.kv_on(s):
            g1, own1 = KV.groups_of(s, model, e.split)
            g2 = [(b, k) for b in hb for k in 'kg' if k == 'k' or s.readout_gate and model.transformer.h[b].attn.ve_gate
                  is not None]
            own2 = HI.balance(g2, e.split, L)
            mine1, mine2 = [g for g in g1 if own1[g] == e.part], [g for g in g2 if own2[g] == e.part]
        mine = mine1 + mine2
        self.kv_C, self.kv_counts, self.kv_added = KV.group_rows(s, H), {}, {}
        self.kv_wproj = {b: model.transformer.h[b].attn.c_proj.weight.float() for b, _ in mine}
        size = {key: b - a for key, (a, b) in share.items()} | {g: self.kv_C for g in mine}
        slots, at, in_rows, mid_rows = {}, 0, [], []  # rows per token, insertion order: q_b, groups of b | MLP b | f_b
        for b in range(L):
            in_rows.append(at)
            for key in [('q', b), *[g for g in mine if g[0] == b], 'mlp', ('f', b)]:
                if key == 'mlp':
                    mid_rows.append(at)
                elif key in size:
                    slots[key], at = slice(at, at + size[key]), at + size[key]
        rows, n = at, d.batch * T
        # readout_kv's / readout_hi's notes: their groups and rows in this pass (their own passes are gone: 0 s)
        self.kv_mine, self.kv_rows, self.kv_events, self.hi_events = mine1, len(mine1) * self.kv_C, [], []
        self.hi_cfg = dict(rows=sum((in_rows + [rows])[b + 1] - in_rows[b] for b in hb), mine=mine2)
        self.q_credit = {b: torch.zeros(n, W, device=dev) for k, b in cyc if k == 'q'}
        self.fc_credit = {b: torch.zeros(n, 4 * W, device=dev) for k, b in cyc if k == 'f'}
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, k in mine if k == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, k in mine if k == 'g'}
        Q, score = torch.zeros(len(ins), n, W, device=dev), torch.zeros(n, rows, device=dev)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(  # readout's seed
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        lo, hi = e.share(W)
        fc_in = {j: slots['f', j - 1] for j in ins if ('f', j - 1) in slots}  # c_fc rows also along the in-root's path
        rms = lambda z: s.readout_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        m_chunk = min(s.readout_chunk, T)
        buf = self.buffers(m_chunk, max(rows, 1))
        for seq in range(d.batch):
            ctxs, src, plans = self.ctx_cache.pop(seq), {}, {}
            gens = [torch.Generator(dev).manual_seed(seed + off + 104729 * seq) for off in (0, HI.HI_SEED)]
            for b in src_b:  # readout's draws in its order (block b's q code, then its c_fc code; hi blocks: 2nd generator)
                g, c = gens[b >= FIRST], ctxs[b]
                basis = self.q_basis(c, s.readout_q_keys) if b >= FIRST or cyc['q', b] < W else None
                src['q', b] = (self.draw_code(g, T, cyc['q', b], dev), basis, rms(c['qraw']))
                src['f', b] = (self.draw_code(g, T, cyc['f', b], dev),
                               rms(d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float())) + self.fc_slots(b, seq, cyc['f', b])
            for j in ins:
                src['in', j] = (self.draw_code(gens[1], T, W, dev), rms(ctxs[j]['x']))
            for b in sorted({g[0] for g in mine}):
                plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in mine if bb == b], self.cs_all, self.sn_all))
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m, sl, local = t1 - t0, slice(seq * T + t0, seq * T + t1), slice(t0, t1)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                dirs = {key: self.code_rows(src[key][0], cyc[key], *share[key], local, dev) for key in cyc}
                for key in [key for key in cyc if key[0] == 'q' and src[key][1] is not None]:  # 2 blockdiag(basis_h) code
                    dirs[key] = 2 * torch.einsum('thdk,tphk->tphd', src[key][1][local], dirs[key].view(
                        m, size[key], H, s.readout_q_keys)).reshape(m, size[key], W)
                for b in range(L):
                    c, k, fb = ctxs[b], in_rows[b], slots.get(('f', b - 1))
                    if k:
                        score[sl, :k] += self.score(x, delta, k, b, b < L - 1 or fb is not None)
                    if fb is not None:  # block b - 1's c_fc rows: the receiver s = t at their first block, from the self-edge
                        se = x['se'][:m * rows * W].view(m, rows, W)[:, fb]
                        score[sl, fb] += (se.float() * c['target'][local, None]).sum(-1)
                    if k and b in fc_in:  # block b - 1's c_fc rows at x_b, before the push (the last block too)
                        score[sl, fc_in[b]] += self.oq_scores(c, b, delta[:, fc_in[b]], self.chain[b][sl], local)
                    if b == L - 1:
                        break
                    if k:
                        self.push(x, delta, k, b)
                    if ('q', b) in slots:
                        delta[:, slots['q', b]] = self.q_moves(c, b, dirs['q', b], src['q', b][2][local], local)
                    for g in mine:
                        if g[0] == b:
                            delta[:, slots[g]] = self.moves(g, plans[g], c, t0, t1)
                    if mid_rows[b]:
                        self.mlp(x, delta, mid_rows[b], b)
                    if ('f', b) in slots:
                        _, fstep, idx, hval, valid = src['f', b]
                        torch.mul(self.fc_moves(b, dirs['f', b], fstep[local], idx[local], hval[local], valid[local]),
                                  lam_r[b + 1], out=delta[:, slots['f', b]])
                for (kind, b), got in ((key, score[sl, slots[key]]) for key in cyc):  # the sources' decode (readout's)
                    if kind == 'q':
                        self.q_credit[b][sl] = torch.einsum('tp,tpw->tw', got / src[kind, b][2][local], dirs[kind, b]) / W
                    else:
                        _, fstep, idx, hval, valid = src[kind, b]
                        slot = torch.einsum('tp,tpk->tk', got / fstep[local], dirs[kind, b][:, :, 1:]) / cyc[kind, b]
                        self.fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                for i, j in enumerate(ins):  # the in-roots: this GPU's rows of their cycles, decoded into its share
                    code, istep = src['in', j]
                    rows_in = self.code_rows(code, W, lo, hi, local, dev)
                    sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                    Q[i, sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
            out = slice(seq * T, (seq + 1) * T)
            for g in mine:  # the receivers' decode (readout_kv's)
                b, kind = g
                p = plans[g]
                f = p['table'].clamp_min(0)
                live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
                reading = torch.zeros(H * T * Rr + 1, device=dev).scatter_(  # the last entry: a sink for empty slots
                    0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, score[out, slots[g]], 0.).reshape(-1))
                reading = reading[:-1].view(H, T, Rr)
                if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                    c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                    self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
                else:
                    self.g_credit[b][out] = reading.sum(-1).T / KV.GATE_SIGMA
            self.route_cache.pop(seq, None)
        self.hi_Q = dict(zip(ins, e.group_sum(Q))) if ins else {}
        self.sp_events[3].record()

    # ------------------------------------------------------------------------------------------------------ pieces

    def prepare(self, mids, routes):
        """Per estimate: fast_chain's / fast_readout's bf16 weights and the scorer's projection d W_k^T | d W_v^T |
        d[:32] W_g^T (padded to 16), the routes in fast_chain2's order for the pruned MLP."""
        model, W, T = self.e.model, self.e.width, self.e.capture.length
        H, blocks = model.config.n_head, model.transformer.h
        bt = lambda w: w.to(torch.bfloat16).T.contiguous()
        gate = lambda a: torch.zeros(W, 16, device=self.e.device, dtype=torch.bfloat16) if a.ve_gate is None else \
            F.pad(F.pad(a.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16), (0, 0, 0, W - 32))
        w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in blocks]
        self.sp_w = dict(w_fc=w_fc, w_out_t=[bt(blk.mlp.c_proj.weight) for blk in blocks],
                         w_q_t=[bt(blk.attn.c_q.weight) for blk in blocks],
                         w_proj_t=[bt(blk.attn.c_proj.weight) for blk in blocks],
                         wcat=[torch.cat([bt(b.attn.c_k.weight), bt(b.attn.c_v.weight), gate(b.attn)], 1).contiguous()
                               for b in blocks],
                         lams=torch.stack([model.resid_lambdas.float(), model.x0_lambdas.float()], 1).contiguous(),
                         mids=[z.float().contiguous() for z in mids], idx32=[r[0].to(torch.int32).contiguous() for r in routes],
                         hs=[r[1].float().contiguous() for r in routes], units=max(r[0].shape[1] for r in routes))
        self.sorted_routes = FC2.sort_routes(w_fc, routes) if FR.IMPL['mlp'] == 'prune' else None
        self.cs_all, self.sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()

    def clean(self, seq, j):
        """readout.Method.contexts' quantities of block j that do not read the hub (its formulas), with the clean
        attention state every consumer reads (raw query, keys, values, attention, outputs)."""
        d, model = self.e.capture, self.e.model
        T, H = d.length, model.config.n_head
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        attn, prefix = model.transformer.h[j].attn, f'transformer.h.{j}.attn.'
        qraw, kraw, vraw = [d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k', 'c_v')]
        v, gate, ve = vraw.reshape(1, T, H, -1), None, None
        if attn.ve_gate is not None:
            gate = d.outputs[prefix + 've_gate'][seq].float()
            ve = d.outputs[f'value_embeds.{j}'][seq].float().reshape(T, H, -1)
            v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
        values = v.transpose(1, 2)[0]
        q, k, a, out = (z[0] for z in R.attention_state(qraw, kraw, values[None], cos, sin, H, model.window_sizes[j][0]))
        kr, x = kraw[0].reshape(T, H, -1), d.blocks[j][seq].float().contiguous()
        return dict(x=x, sqx=x.square().sum(-1).contiguous(), kraw=kr.contiguous(), gate=gate,
                    R0=R.rotate(kr, cos[0, :, 0][:, None], sin[0, :, 0][:, None]).contiguous(), VC=vraw[0].contiguous(),
                    GZ=None if gate is None else gate.contiguous(), ve=None if ve is None else ve.contiguous(),
                    QD=q.transpose(0, 1).contiguous(), AD=a.diagonal(0, 1, 2).T.contiguous(),
                    VT=values.transpose(0, 1).contiguous(), OT=out.transpose(0, 1).contiguous(),
                    qraw=qraw[0], q=q, k=k, values=values, a=a, out=out)

    def tables(self, c, j, target):
        """readout.Method.contexts' hub part: block j's moment tables of the receivers s > t from target = chain_j W_proj
        (kept too: the c_fc rows' s = t)."""
        T, H = self.e.capture.length, self.e.model.config.n_head
        g = target.reshape(T, H, -1).transpose(0, 1)
        contrast = (g @ c['values'].transpose(-1, -2)) - (g * c['out']).sum(-1, keepdim=True)
        c['U'], c['W'], c['M1'], c['M2'] = FM.build_future_tables(c['q'], g, c['a'], contrast,
                                                                  self.e.model.window_sizes[j][0])
        c['target'] = target

    def buffers(self, m, rows):
        W, dev = self.e.width, self.e.device
        bf = lambda k: torch.empty(m * rows * k, device=dev, dtype=torch.bfloat16)
        return dict(delta=torch.empty(m * rows * W, device=dev), scratch32=torch.empty(m * rows * W, device=dev),
                    scratch=bf(max(W, self.sp_w['units'])), db=bf(W), se=bf(W), up=bf(2 * W + 16),
                    st=torch.empty(m * rows * 2, device=dev), **FR.prune_buffers(m, rows, dev))

    def chunk(self, buf, ctxs, seq, t0, t1):
        """fast_readout's block-step context for tokens t0:t1 of sequence seq."""
        sl, w = slice(seq * self.e.capture.length + t0, seq * self.e.capture.length + t1), self.sp_w
        own, routes = self.route_cache[seq]['own'], self.route_cache[seq]['routes'][1]
        x = dict(buf, ybuf=buf['scratch'], ctxs=ctxs, own=own, routes=routes, local=slice(t0, t1), t0=t0,
                 cos=own[6].contiguous(), sin=own[7].contiguous(), mids=[z[sl] for z in w['mids']],
                 **{k: w[k] for k in ('w_q_t', 'w_proj_t', 'w_fc', 'w_out_t', 'lams')})
        if self.sorted_routes is None:
            return x | dict(idx32=[z[sl] for z in w['idx32']], hs=[z[sl] for z in w['hs']])
        return x | FR.sorted_slices(self.sorted_routes, sl)

    def score(self, x, delta, n, b, self_edge):
        """readout's staged scorer at block b on rows < n: the receivers s > t only -> [m, n]; self_edge: also each row's
        own output change (bf16, x['se']: push() adds it, run() scores the c_fc rows' s = t from it)."""
        m, rows, W = delta.shape
        c, H, nc = x['ctxs'][b], self.e.model.config.n_head, self.sp_w['wcat'][b].shape[1]
        out = torch.empty(m, n, device=delta.device)
        gated = c['gate'] is not None
        RO.stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], x['db'], x['st'], x['t0'], n, rows, W, 16, EPS,
                                                 num_warps=4)
        up = x['up'][:m * n * nc].view(m * n, nc)
        torch.mm(x['db'][:m * n * W].view(m * n, W), self.sp_w['wcat'][b], out=up)
        RO.staged_kernel[(triton.cdiv(n, 64), m)](
            up, x['st'], c['R0'], c['VC'], c['GZ'] if gated else out, c['ve'] if gated else out, self.cs_all, self.sn_all,
            c['U'], c['W'], c['M1'], c['M2'], c['QD'], out, c['AD'], out, c['VT'], c['OT'], out, x['se'], x['t0'], n, rows,
            W, H, W // H, gated, False, self_edge, 64, nc, num_warps=4, num_stages=2)
        return out

    def push(self, x, delta, n, b):
        """Rows < n through block b's attention: the own query and the self-edge just scored, both through c_proj."""
        FR.own_query_step(delta, n, b, x, x['db'], x['st'], x['se'] if self.SELF_EDGE else None)

    def mlp(self, x, delta, n, b, dx0=None, P=0):
        """Block b's routed MLP on rows < n, then block b + 1's mixing scalars (dx0: the embedding rows' x0 change on
        the first P rows); none after the last block."""
        last = b == self.e.model.config.n_layer - 1
        FR.mlp_step(delta, n, b, x, None if last else delta if dx0 is None else dx0, 0 if dx0 is None else P)

    def note(self):
        super().note()  # readout's, readout_kv's and readout_hi's lines (and report cadences) as in E3's runs
        STATS['calls'] += 1
        STATS['seconds'] = STATS['seconds'][-49:] + [self.sp_events]
        if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % STATS['every'] == 0):
            torch.cuda.synchronize()
            sec = lambda i: sum(ev[i].elapsed_time(ev[i + 1]) for ev in STATS['seconds']) / len(STATS['seconds']) / 1e3
            print(f"shared_push estimate {STATS['calls']}: self-edged chain with the readout roots {sec(0):.3f} s, readout "
                  f"pass without them (the receivers' and hi rows inside) {sec(2):.3f} s", flush=True)


def shared(base):  # this file's Method on a readout_hi-style Method class
    return type(f'Shared{base.__name__}', (Shared, base), {'__doc__': f'{base.__name__} with the shared push.'})


SharedKVMethod, SharedMethod = shared(HI.HiKVMethod), shared(HI.HiMethod)  # E3 on E1 / on the readout alone
CLASSES = {True: SharedKVMethod, False: SharedMethod}  # by readout_kv.kv_on(settings)


def cut_estimate(self, x, y):  # replaces Cut.estimate: readout_shared off = readout_hi's, called as is
    if not on(self.s):
        return HI.cut_estimate(self, x, y)
    method = CLASSES[KV.kv_on(self.s)](self)
    gradients = method.estimate(x, y)
    if STATS['keep'] or RO.STATS['keep'] or KV.STATS['keep'] or HI.STATS['keep']:
        self.readout_method = method
    return gradients


def install():
    """readout_hi.py's patch (and everything under it), then this one (idempotent): Settings, the recipe dict and
    Cut.estimate. The speed stack goes on after (it touches none of these)."""
    if R.Settings is not Settings:
        HI.install()
        if R.Settings is not HI.Settings or R.asdict is not HI.asdict or R.Cut.estimate is not HI.cut_estimate:
            raise RuntimeError('shared_push: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Settings, R.asdict, R.Cut.estimate = Settings, asdict, cut_estimate
    return R


# ------------------------------------------------------------------------------------------------------------ checks


# ======================================== module fast_e3
"""The readout stack faster (exact or rounding level; the same measurements): in-root and q-source scores in contrast form
in one kernel each, the shared Methods (this run's stack: RUN_STACK below).
"""
import collections
import dataclasses
import json
import os
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import readout_hi as HI  # noqa: E402  (loads fast_readout, readout_kv, readout, onesided and the recipe module)
FR, KV, RO, R, FC = HI.FR, HI.KV, HI.RO, HI.R, HI.FC
import fast_chain2 as FC2  # noqa: E402  (installed by install_chain2 when IMPL['chain2'])
FM = RO.FM  # fast_moments (its moment path: the key part's kernels, decode and bookkeeping)

EPS = RO.EPS
LO, FIRST = HI.LO, HI.FIRST
IMPL_ON = dict(chain2=True, inroot='tf32r', qsrc='tf32r', ctx='shared', keypath='shared', qbasis='kernel',
               selfedge='one', balance='plain', fccycle='sized')
IMPL_OFF = dict(chain2=False, inroot='plain', qsrc='plain', ctx='plain', keypath='plain', qbasis='plain',
                selfedge='plain', balance='plain', fccycle='plain')
CHOICES = dict(chain2=(False, True), inroot=('plain', 'tf32x3', 'tf32', 'tf32r'),
               qsrc=('plain', 'tf32x3', 'tf32', 'tf32r', 'tf32m'), ctx=('plain', 'shared'), keypath=('plain', 'shared'),
               qbasis=('plain', 'kernel'), selfedge=('plain', 'one'), balance=('plain', 'b1', 'b2', 'b3'),
               fccycle=('plain', 'sized'))
FC_ENERGY = .999  # item 1.6: a hi block's c_fc cycle = the smallest power of two (>= 32, <= its configured cycle) whose
#   top cycle - 1 slots (readout's rank relu(h)^2 |W_out col|) hold this share of the relu(h)^2 |W_out col|^2 energy
# item 1.7: receiver groups per part of a pair (E1's blocks 0-3, E3's hi blocks); 'plain' = readout_kv.groups_of's /
# readout_hi.balance's rows-x-blocks split. A group's credit goes into the gradient of the GPU that runs it (no collective),
# so only which part holds c_k / ve_gate of a block changes (the pair's sum is the same up to rounding).
BALANCE = dict(b1=({(0, 'k'): 0, (3, 'g'): 0, (3, 'k'): 0, (1, 'g'): 1, (1, 'k'): 1, (2, 'k'): 1},
                   {(4, 'k'): 0, (5, 'g'): 1, (5, 'k'): 1, (6, 'k'): 1}),
               b2=(None, {(4, 'k'): 0, (5, 'g'): 1, (5, 'k'): 1, (6, 'k'): 1}),
               b3=({(0, 'k'): 0, (3, 'g'): 0, (3, 'k'): 0, (1, 'g'): 1, (1, 'k'): 1, (2, 'k'): 1}, None))
IMPL = dict(IMPL_ON)
INROOT_TILES = {64: (64, 4, 3), 256: (128, 4, 3)}  # probes per token -> probes per program, warps, stages (swept)
QSRC2_TILES = {64: (32, 4, 2), 256: (32, 4, 3)}
STATS = dict(calls=0)


def parse_impl(spec, base=None):
    """'name=value,...' -> an IMPL dict on top of base (default IMPL_ON); 'off' / 'on' / '' as shorthands."""
    out = dict(IMPL_ON if base is None else base)
    for item in filter(None, (spec or '').split(',')):
        if item in ('on', 'off'):
            out.update(IMPL_ON if item == 'on' else IMPL_OFF)
            continue
        name, _, value = item.partition('=')
        if name not in CHOICES:
            raise ValueError(f'fast_e3: unknown IMPL item {name!r} (known: {sorted(CHOICES)})')
        value = {'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}.get(value, value) \
            if isinstance(CHOICES[name][0], bool) else value
        if value not in CHOICES[name]:
            raise ValueError(f'fast_e3: IMPL {name}={value!r} not in {CHOICES[name]}')
        out[name] = value
    return out


IMPL.update(parse_impl('chain2=False,ctx=plain,keypath=plain,selfedge=plain,fccycle=plain'))  # the run's stack


# ----------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def round_tf32(x):
    """x rounded to the nearest TF32 value (ties away from zero: the sign-magnitude bits + half an ulp, low 13 bits
    cleared). The tensor cores truncate fp32 operands to TF32 (a bias of ~2^-11 per operand); a dot of rounded operands
    is unbiased."""
    b = x.to(tl.uint32, bitcast=True)
    return ((b + 0x1000) & 0xFFFFE000).to(tl.float32, bitcast=True)


@triton.jit
def inroot_kernel(DQ, QRAW, CS, SN, K, V, A, Base, TG, Out, t0, window, P,
                  H: tl.constexpr, D: tl.constexpr,
                  KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                  VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                  AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                  OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                  BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, EPS: tl.constexpr,
                  ROUND: tl.constexpr):
    """Item 1.2: the in-root score of BP raw-query moves of one token, summed over the heads, in contrast form. Per head:
    qsrc_kernel's prologue (r0 = rotate(q_raw), dr = rotate(dq), the per-head RMSNorm change delta = dr inv' + r0
    (inv' - inv)), then over the key tiles of the window (from max(0, t - window) to t): dl = delta . k / sqrt(D)
    (PRECISION dot), e = exp(dl) - 1, Z += sum a e, S += sum a e c with the contrast c_s = target . (v_s - o_t) formed
    from the V tile in fp32 (TG [m, H * D]: target = chain_j W_proj of the token); score += S / (1 + Z). Out [m, P]."""
    pb, token = tl.program_id(0), tl.program_id(1)
    probe = pb * BP + tl.arange(0, BP)
    pm = probe < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    pos = t0 + token
    cs = tl.load(CS + pos * (D // 2) + half)
    sn = tl.load(SN + pos * (D // 2) + half)
    lo = (tl.maximum(pos - window, 0) // BK) * BK
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        qrow = QRAW + (pos * H + h) * D
        x = tl.load(qrow + f)
        xo = tl.load(qrow + other)
        r0 = tl.where(low, x * cs + xo * sn, -xo * sn + x * cs)
        drow = DQ + ((token * P + probe) * H + h).to(tl.int64) * D
        dd = tl.load(drow[:, None] + f[None, :], pm[:, None], 0.)
        ddo = tl.load(drow[:, None] + other[None, :], pm[:, None], 0.)
        dr = tl.where(low[None, :], dd * cs[None, :] + ddo * sn[None, :], -ddo * sn[None, :] + dd * cs[None, :])
        square = tl.sum(r0 * r0, 0)
        cross = tl.sum(r0[None, :] * dr, 1)
        sqd = tl.sum(dr * dr, 1)
        inverse = libdevice.rsqrt(square / D + EPS)
        moved = libdevice.rsqrt((square + 2 * cross + sqd) / D + EPS)
        delta = (dr * moved[:, None] + r0[None, :] * (moved - inverse)[:, None]) * (D ** -.5)
        if ROUND:
            delta = round_tf32(delta)
        tg = tl.load(TG + (token * H + h) * D + f)
        base = tl.load(Base + h * OH + token * OM + f * OD)
        z = tl.zeros((BP,), tl.float32)
        s = tl.zeros((BP,), tl.float32)
        for start in range(lo, pos + 1, BK):
            key = start + tl.arange(0, BK)
            km = key <= pos
            a = tl.load(A + h * AH + token * AM + key * AT, km, 0.)
            k = tl.load(K + h * KH + key[None, :] * KT + f[:, None] * KD, km[None, :], 0.)
            v = tl.load(V + h * VH + key[:, None] * VT + f[None, :] * VD, km[:, None], 0.)
            c = tl.sum((v - base[None, :]) * tg[None, :], 1)
            if ROUND:
                k = round_tf32(k)
            ae = a[None, :] * (tl.exp(tl.dot(delta, k, input_precision=PRECISION)) - 1)
            z += tl.sum(ae, 1)
            s += tl.sum(ae * c[None, :], 1)
        total += s / (1 + z)
    tl.store(Out + token * P + probe, total, pm)


@triton.jit
def qsrc2_kernel(DIRS, QRAW, QSTEP, CS, SN, K, V, A, Base, Out, t0, window, P,
                 H: tl.constexpr, D: tl.constexpr,
                 KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                 VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                 AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                 OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                 BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, EPS: tl.constexpr,
                 ROUND: tl.constexpr, QKP: tl.constexpr):
    """Item 1.3: fast_readout's qsrc_kernel (readout's q move of BP probes of one token in one head, then the exact
    softmax response over every key) in centered form: numerator += (a e) (v - o) over the key tiles of the window (from
    max(0, t - window) to t), e = exp(dl) - 1, response = numerator / (1 + partition): the same algebra as (numerator -
    partition o) / (1 + partition) without its cancellation, so the dots can run one-pass TF32 (PRECISION)."""
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    probe = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = probe < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    pos = t0 + token
    cs = tl.load(CS + pos * (D // 2) + half)
    sn = tl.load(SN + pos * (D // 2) + half)
    qrow = QRAW + (pos * H + head) * D
    x = tl.load(qrow + f)
    xo = tl.load(qrow + other)
    r0 = tl.where(low, x * cs + xo * sn, -xo * sn + x * cs)
    step = tl.load(QSTEP + token)
    drow = DIRS + ((token * P + probe) * H + head).to(tl.int64) * D
    dd = tl.load(drow[:, None] + f[None, :], pm[:, None], 0.) * step
    ddo = tl.load(drow[:, None] + other[None, :], pm[:, None], 0.) * step
    dr = tl.where(low[None, :], dd * cs[None, :] + ddo * sn[None, :], -ddo * sn[None, :] + dd * cs[None, :])
    square = tl.sum(r0 * r0, 0)
    cross = tl.sum(r0[None, :] * dr, 1)
    sqd = tl.sum(dr * dr, 1)
    inverse = libdevice.rsqrt(square / D + EPS)
    moved = libdevice.rsqrt((square + 2 * cross + sqd) / D + EPS)
    delta = (dr * moved[:, None] + r0[None, :] * (moved - inverse)[:, None]) * (D ** -.5)
    if ROUND and QKP == 'tf32':
        delta = round_tf32(delta)
    base = tl.load(Base + head * OH + token * OM + f * OD)
    numerator = tl.zeros((BP, D), tl.float32)
    partition = tl.zeros((BP,), tl.float32)
    lo = (tl.maximum(pos - window, 0) // BK) * BK
    for start in range(lo, pos + 1, BK):
        key = start + tl.arange(0, BK)
        km = key <= pos
        a = tl.load(A + head * AH + token * AM + key * AT, km, 0.)
        k = tl.load(K + head * KH + key[None, :] * KT + f[:, None] * KD, km[None, :], 0.)
        v = tl.load(V + head * VH + key[:, None] * VT + f[None, :] * VD, km[:, None], 0.)
        vo = v - base[None, :]
        if ROUND:
            vo = round_tf32(vo)
            if QKP == 'tf32':
                k = round_tf32(k)
        change = a[None, :] * (tl.exp(tl.dot(delta, k, input_precision=QKP)) - 1)
        partition += tl.sum(change, 1)
        if ROUND:
            numerator = tl.dot(round_tf32(change), vo, numerator, input_precision=PRECISION)
        else:
            numerator = tl.dot(change, vo, numerator, input_precision=PRECISION)
    response = numerator / (1 + partition[:, None])
    tl.store(Out + ((token * P + probe[:, None]) * H + head).to(tl.int64) * D + f[None, :], response, pm[:, None])


@triton.jit
def oq_attn_se_kernel(U, STATS, R0, SQ0, CS, SN, KEYS, VALS, IDX, TP, O, SE, Out, n_rows, U_ROWS, O_ROWS, SE_ROWS,
                      t_start, T: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, W: tl.constexpr, KK: tl.constexpr,
                      BP: tl.constexpr, EPS: tl.constexpr):
    """Item 1.5c: fast_chain's oq_attn_kernel (copied) with the self-edge output change SE (bf16, the scorer's) added to
    the own-query response before the bf16 store, so ONE c_proj product carries both (7D): (o + se)_bf16 W_proj^T instead
    of o_bf16 W_proj^T + se_bf16 W_proj^T."""
    hh, t = tl.program_id(0), tl.program_id(1)
    pos = t_start + t
    base = pos * H + hh
    QH: tl.constexpr = DH // 2
    hd = tl.arange(0, QH)
    feat = tl.arange(0, DH)
    keys_n = tl.arange(0, KK)
    c = tl.load(CS + pos * QH + hd)[None, :]
    s = tl.load(SN + pos * QH + hd)[None, :]
    r0l = tl.load(R0 + base * DH + hd)[None, :]
    r0h = tl.load(R0 + base * DH + QH + hd)[None, :]
    sq0 = tl.load(SQ0 + base)
    qi0 = libdevice.rsqrt(sq0 / DH + EPS)
    kidx = tl.load(IDX + base * KK + keys_n).to(tl.int64)
    krow = (hh * T + kidx) * DH
    kl = tl.trans(tl.load(KEYS + krow[:, None] + hd[None, :]))
    kh = tl.trans(tl.load(KEYS + krow[:, None] + QH + hd[None, :]))
    v = tl.load(VALS + krow[:, None] + feat[None, :])
    p = tl.load(TP + base * KK + keys_n)
    rest = (tl.load(O + base * DH + feat) - tl.sum(p[:, None] * v.to(tl.float32), 0))[None, :]
    p = p[None, :]
    for r0 in range(0, n_rows, BP):
        rows = r0 + tl.arange(0, BP)
        ok = rows < n_rows
        row = (t * U_ROWS + rows).to(tl.int64)
        ub = U + row * W + hh * DH
        ul = tl.load(ub[:, None] + hd[None, :], ok[:, None], 0.)
        uh = tl.load(ub[:, None] + QH + hd[None, :], ok[:, None], 0.)
        inv = tl.load(STATS + row * 2, ok, 1.)[:, None]
        coef = tl.load(STATS + row * 2 + 1, ok, 0.)[:, None]
        drl = inv * (ul * c + uh * s) + coef * r0l
        drh = inv * (ul * -s + uh * c) + coef * r0h
        cross = tl.sum(r0l * drl, 1) + tl.sum(r0h * drh, 1)
        sqdr = tl.sum(drl * drl, 1) + tl.sum(drh * drh, 1)
        qi = libdevice.rsqrt((sq0 + 2 * cross + sqdr) / DH + EPS)[:, None]
        dql = (drl * qi + r0l * (qi - qi0)).to(tl.bfloat16)
        dqh = (drh * qi + r0h * (qi - qi0)).to(tl.bfloat16)
        ds = tl.dot(dqh, kh, tl.dot(dql, kl)) * (DH ** -.5)
        e = tl.exp(ds) - 1
        z = tl.sum(p * e, 1)[:, None]
        rz = 1 / (1 + z)
        weights = (p * (e - z) * rz).to(tl.bfloat16)
        moved = tl.dot(weights, v) - z * rz * rest
        moved += tl.load(SE + (t * SE_ROWS + rows).to(tl.int64)[:, None] * W + hh * DH + feat[None, :], ok[:, None],
                         0.).to(tl.float32)
        tl.store(Out + (t * O_ROWS + rows).to(tl.int64)[:, None] * W + hh * DH + feat[None, :],
                 moved.to(tl.bfloat16), ok[:, None])


def inroot_scores(dq, qraw, keys, values, a, base, target, t0, cs, sn, window, precision, tiles=None):
    """inroot_kernel on dq [m, p, W] (the raw-query moves of tokens t0 .. t0 + m), a / base the attention rows / outputs
    of these tokens ([H, m, T] / [H, m, D] views), target [m, W]: the in-root scores [m, p]."""
    m, p, _ = dq.shape
    H, T, D = keys.shape
    out = torch.empty(m, p, device=dq.device)
    bp, warps, stages = tiles or INROOT_TILES.get(p, INROOT_TILES[64])
    inroot_kernel[(triton.cdiv(p, bp), m)](
        dq, qraw, cs, sn, keys, values, a, base, target, out, t0, window, p, H, D, *keys.stride(), *values.stride(),
        *a.stride(), *base.stride(), bp, 64, 'tf32' if precision == 'tf32r' else precision, EPS, precision == 'tf32r',
        num_warps=warps, num_stages=stages)
    return out


def qsrc2_response(dirs, qraw, qstep, keys, values, a, base, t0, cs, sn, window, precision, tiles=None):
    """qsrc2_kernel: the q moves' attention responses [m, p, H, D] (fast_readout's q_moves before c_proj)."""
    m, p, _ = dirs.shape
    H, T, D = keys.shape
    out = torch.empty(m, p, H, D, device=dirs.device)
    bp, warps, stages = tiles or QSRC2_TILES[64 if p <= 64 else 256]
    qsrc2_kernel[(triton.cdiv(p, bp), m * H)](
        dirs, qraw, qstep, cs, sn, keys, values, a, base, out, t0, window, p, H, D, *keys.stride(), *values.stride(),
        *a.stride(), *base.stride(), bp, 64, 'tf32' if precision in ('tf32r', 'tf32m') else precision, EPS,
        precision in ('tf32r', 'tf32m'), 'tf32x3' if precision == 'tf32m' else 'tf32' if precision == 'tf32r' else precision,
        num_warps=warps, num_stages=stages)
    return out


# ------------------------------------------------------------------------------------------------------ block pieces


PLAIN = dict(push_block=FR.push_block)  # fast_readout's, as checked


def push_block(delta, n, b, ctx, bufs, use_self):
    """fast_readout's push_block (own query + self-edge + c_proj, in place) with item 1.5c: the self-edge added to the
    own-query response before ONE c_proj product (oq_attn_se_kernel, then fast_chain's linear_residual_kernel)."""
    if IMPL['selfedge'] != 'one' or not use_self or FR.IMPL['oq'] != 'fused':
        return PLAIN['push_block'](delta, n, b, ctx, bufs, use_self)
    m, rows, W = delta.shape
    q = ctx['routes'][b]
    keys = q['keys']
    H, T, DH = keys.shape
    u = ctx['scratch32'][:m * n * W].view(m * n, W)
    torch.mm(bufs['db'][:m * n * W].view(m * n, W), ctx['w_q_t'][b], out_dtype=torch.float32, out=u)  # as fast_readout
    buf = ctx['ybuf'][:m * n * W]
    bp, warps, stages = FC.OQ_ATTN
    oq_attn_se_kernel[(H, m)](
        u, bufs['st'], q['r0'], q['sq0'], ctx['cos'], ctx['sin'], keys, q['values'], q['idx'], q['p'], q['o'], bufs['se'],
        buf, n, n, n, rows, ctx['t0'], T, H, DH, W, q['idx'].shape[-1], bp, EPS, num_warps=warps, num_stages=stages)
    bm, bn, bk, warps, stages = FC.PROJ
    FC.linear_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
        buf, ctx['w_proj_t'][b], delta, m * n, n, n, rows, W, W, bm, bn, bk, num_warps=warps, num_stages=stages)


PLAIN.update(groups_of=KV.groups_of, balance=HI.balance)


def groups_of(s, model, split):
    """readout_kv.groups_of with item 1.7's assignment (a pair, the table covering every group), else as there."""
    groups, owner = PLAIN['groups_of'](s, model, split)
    table = BALANCE.get(IMPL['balance'], (None, None))[0]
    if split == 2 and table is not None and set(groups) <= set(table):
        owner = {g: table[g] for g in groups}
    return groups, owner


def balance(groups, split, L):
    """readout_hi.balance with item 1.7's assignment for the hi blocks' groups (a pair, covered), else as there."""
    table = BALANCE.get(IMPL['balance'], (None, None))[1]
    if split == 2 and table is not None and groups and set(groups) <= set(table):
        return {g: table[g] for g in groups}
    return PLAIN['balance'](groups, split, L)


class patched:
    """fast_readout's push_block (and readout_kv's / readout_hi's group assignment) replaced by this file's while an
    estimate runs (every pass calls them through their module: fast_readout's run / kv_sequence, readout_kv's kv_setup,
    readout_hi's hi_plan / hi_sequence); restored after, also on errors."""

    def __enter__(self):
        FR.push_block, KV.groups_of, HI.balance = push_block, groups_of, balance

    def __exit__(self, *exc):
        FR.push_block, KV.groups_of, HI.balance = PLAIN['push_block'], PLAIN['groups_of'], PLAIN['balance']


@triton.jit
def mgs_rows_kernel(B, Q, N, K: tl.constexpr, KP: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """Item 1.5a: readout.orthonormal for BR rows n of B [N, K, D] per program ([BR, K, D] tiles): modified Gram-Schmidt,
    twice per vector after the first, in fp32 (FMA sums instead of the TF32 einsums), the same threshold (a vector whose
    remainder is <= 1e-4 x the largest input norm of its row becomes zero)."""
    pid = tl.program_id(0)
    n = pid * BR + tl.arange(0, BR)
    kk = tl.arange(0, KP)
    f = tl.arange(0, D)
    m3 = (n < N)[:, None, None] & (kk < K)[None, :, None]
    off = (n.to(tl.int64)[:, None, None] * K + kk[None, :, None]) * D + f[None, None, :]
    rows = tl.load(B + off, m3, 0.)
    scale = tl.maximum(tl.max(tl.sqrt(tl.sum(rows * rows, 2)), 1), 1e-30)  # [BR]
    q = tl.zeros((BR, KP, D), tl.float32)
    for k in tl.static_range(K):
        sel = (kk == k)[None, :, None]
        v = tl.sum(tl.where(sel, rows, 0.), 1)  # [BR, D]
        if k > 0:
            for _ in tl.static_range(2):
                c = tl.sum(q * v[:, None, :], 2)  # [BR, KP]
                v = v - tl.sum(c[:, :, None] * q, 1)
        nrm = tl.sqrt(tl.sum(v * v, 1))  # [BR]
        qk = tl.where((nrm > 1e-4 * scale)[:, None], v / tl.maximum(nrm, 1e-30)[:, None], 0.)
        q = tl.where(sel, qk[:, None, :], q)
    tl.store(Q + off, q, m3)


MGS_TILE = dict(rows=8, warps=4)  # swept (profiler, shared GPU 7)


def orthonormal(B):
    """readout.orthonormal by mgs_rows_kernel (item 1.5a)."""
    N, K, D = B.shape
    B = B.contiguous()
    Q = torch.empty_like(B)
    br = MGS_TILE['rows']
    mgs_rows_kernel[(triton.cdiv(N, br),)](B, Q, N, K, triton.next_power_of_2(K), D, br, num_warps=MGS_TILE['warps'])
    return Q


# ------------------------------------------------------------------------------------------------------- the methods


def sized_cycles(est):
    """Item 1.6: per hi block with a c_fc source, (configured cycle, sized cycle, energy share dropped vs the configured
    cycle) from this step's capture (all its sequences): the smallest power of two n in [32, configured] whose top n - 1
    slots by readout's rank (relu(h)^2 |W_out col|) hold >= FC_ENERGY of the relu(h)^2 |W_out col|^2 energy (else the
    configured cycle); the max over the pair (one all_reduce if the pair is distributed); one host read for all blocks."""
    s, model, d = est.s, est.model, est.capture
    blocks = list(s.readout_hi_blocks)
    cfgs = [s.readout_hi_fc_cycle[b - FIRST] for b in blocks]
    picks, drops = [], []
    for b, cfg in zip(blocks, cfgs):
        h = d.outputs[f'transformer.h.{b}.mlp.c_fc'].float().flatten(0, 1)
        wn = model.transformer.h[b].mlp.c_proj.weight.float().norm(dim=0)
        r2 = F.relu(h).square()
        energy = r2 * wn.square()
        top = (r2 * wn).topk(cfg - 1, dim=-1).indices
        kept = energy.gather(1, top).sum(0).cumsum(0) / energy.sum().clamp_min(1e-30)  # share held by the top k slots
        sizes = [c for c in (32, 64, 128, 256, 512, 1024, 2048) if c <= cfg]
        ok = torch.stack([kept[c - 2] >= FC_ENERGY for c in sizes])
        first = torch.where(ok.any(), ok.int().argmax(), torch.tensor(len(sizes) - 1, device=ok.device))
        n = torch.tensor(sizes, device=ok.device)[first]
        picks.append(n)
        drops.append(kept[-1] - kept[n - 2])
    stats = torch.stack([torch.stack(picks).double(), -torch.stack(drops).double()])
    if est.split > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX, group=est.group)  # larger n, smaller drop
    n_list, drop_list = stats.tolist()
    return {b: (cfg, int(n), max(0., -dr)) for b, cfg, n, dr in zip(blocks, cfgs, n_list, drop_list)}


class KeyPath:
    """Item 1.4 (key path): Cut.key_input_credits as fast_moments' moment path runs it (moment_block: its kernels and
    decode; the record's draws from the private generator copy, this GPU's share, the pair's sum per block in block order)
    with each sequence's tables (U, M1) and s = t diagonals (A, C) taken from that sequence's readout contexts, which
    readout builds anyway (future_build_kernel forms U / M1 with build_kernel's arithmetic, from prepare's q / a /
    contrast: the same quantities as key_context's up to TF32 rounding of the logits / contrast GEMMs); run per sequence
    inside the readout pass instead of before it. key_context's dense softmax / contrast and build_kernel are gone;
    qp / kp / raw keep key_context's formulas. fast_moments' bookkeeping stays (its call counter, 'moments' notes, the
    50-call running check: one block of sequence 0 scored exactly by record_block on key_context's inputs and the
    record's moves, against the credit this path formed from the reused inputs; above FALLBACK_REL: fallback)."""

    def __init__(self, method, targets):
        est = method.e
        self.est, self.targets = est, targets
        d, s = est.capture, est.s
        self.n = FM.STATE['n']
        FM.STATE['n'] += 1
        every = FM.STATE['check_every']
        due = bool(every) and self.n % every == 1 % every
        self.check_block = (self.n // max(1, every)) % est.model.config.n_layer if due else None
        FM.note('moments')
        generator = torch.Generator(est.device).set_state(est.generator.get_state())  # the record's private copy
        frame = R.hadamard(est.width, est.device, torch.float32)
        self.rows, self.total, self.prows = {}, {}, {}
        for i in targets:  # Cut.key_input_credits' draws, in its order
            indices = torch.randperm(est.width, device=est.device, generator=generator)[:s.key_path_draws]
            signs = torch.randint(0, 2, (est.width,), device=est.device, generator=generator) * 2 - 1
            rows = frame[indices] * signs
            self.total[i] = len(rows)
            r0, r1 = est.share(len(rows))
            self.rows[i] = rows[r0:r1]
            self.prows[i] = self.rows[i] @ est.model.transformer.h[i].attn.c_k.weight.float().T
        self.estimate = {i: torch.zeros(d.batch, d.length, est.width, device=est.device) for i in targets}
        self.raw = {i: torch.empty(d.batch, d.length, est.width, device=est.device) for i in targets}
        self.check = None

    @staticmethod
    def supported(est):
        return (FM.STATE['installed'] is R and 'key' in FM.STATE['parts'] and R.Cut.key_input_credits is FM.key_input_credits
                and not FM.STATE['fallback'] and FM.unsupported(est) is None and bool(est.s.key_path_draws))

    def sequence(self, seq, ctxs):
        """This sequence's input-K and raw-K experiments of every block, scored from its readout contexts."""
        est = self.est
        d, s, model = est.capture, est.s, est.model
        T, H = d.length, model.config.n_head
        D = est.width // H
        cos, sin = model.cos[0, :T, 0].float(), model.sin[0, :T, 0].float()
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)  # key_context's
        for i in self.targets:
            c = ctxs[i]
            qraw = d.outputs[f'transformer.h.{i}.attn.c_q'][seq].float().view(T, H, D)
            qp = unit(R.rotate(qraw, cos[:, None], sin[:, None])).transpose(0, 1)  # key_context's qp
            kp = unit(c['R0']).transpose(0, 1)  # key_context's kp (R0 = the rotated raw key, the same products)
            A = torch.as_strided(c['AD'], (H, T, T), (1, H, 0))  # the kernels read only the diagonal (s = t)
            C = torch.as_strided(c['CD'], (H, T, T), (1, H, 0))
            xin = d.blocks[i][seq].float()
            inputs = dict(context=(qp, kp, c['kraw'].transpose(0, 1), cos, sin, A, C), xin=xin,
                          steps=s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6),
                          projected_x=xin @ model.transformer.h[i].attn.c_k.weight.float().T,
                          projected_rows=self.prows[i])
            got, raw = FM.moment_block(est, i, inputs, self.rows[i], self.total[i], tables=(c['U'], c['M1']))
            self.estimate[i][seq] = got
            self.raw[i][seq] = raw.transpose(0, 1).reshape(T, est.width)
            if seq == 0 and i == self.check_block:
                self.check = (i, got.clone(), raw.clone())

    def finish(self):
        """The pair's sums (block order) and the raw-K errors for the internals (Cut.key_input_credits' outputs)."""
        est = self.est
        if self.check is not None:
            self.running_check()
        credits = {i: est.group_sum(self.estimate[i].flatten(0, 1)) for i in self.targets}
        est._input_key_errors = {f'transformer.h.{i}.attn.c_k': self.raw[i] for i in self.targets}
        return credits

    def running_check(self):
        est = self.est
        i, got_in, got_raw = self.check
        inputs = FM.block_inputs(est, i, 0, self.targets[i], self.rows[i], self.prows[i])
        ref_in, ref_raw, x_in, x_raw = FM.record_block(est, i, inputs, self.rows[i], self.total[i], 'exact')

        def rel(a, b):
            den, num = b.double().norm(), (a.double() - b.double()).norm()
            return torch.where(den > 0, num / den.clamp_min(1e-300), torch.where(num > 0, torch.inf, 0.))
        stats = torch.stack([rel(got_in, ref_in), rel(got_raw, ref_raw),
                             torch.tensor(x_in, device=est.device, dtype=torch.float64),
                             torch.tensor(x_raw, device=est.device, dtype=torch.float64)])
        stats = torch.nan_to_num(stats, nan=torch.inf)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        rel_in, rel_raw, x_in, x_raw = stats.tolist()
        ok = max(rel_in, rel_raw) <= FM.STATE['fallback_rel']
        FM.STATE['checks'].append(dict(call=self.n, block=i, rel_input=float(f'{rel_in:.3e}'),
                                       rel_raw=float(f'{rel_raw:.3e}'), max_x_input=round(x_in, 5),
                                       max_x_raw=round(x_raw, 6), ok=ok, inputs='fast_e3: readout contexts'))
        if FM.STATE['shared'] is None:
            FM.STATE['shared'] = FM.shared_quantities(est, self.targets)
        if not ok:
            FM.STATE['fallback'] = True
            if os.environ.get('RANK', '0') == '0':
                print(f'fast_e3: key path running check failed at call {self.n} (block {i}: input-K {rel_in:.2e}, raw-K '
                      f"{rel_raw:.2e} > {FM.STATE['fallback_rel']}): fast_moments falls back to the record", flush=True)
        FM.note(force=True)


class ContextBase(RO.Method):
    """readout.py's contexts (sits right above RO.Method in the MRO), keeping the hi pass's pieces (item 1.4)."""

    def contexts(self, seq, targets):
        ctxs = self.build_contexts(seq, targets)
        kp = getattr(self, 'kp', None)
        if kp is not None:  # item 1.4: this sequence's key path from these contexts
            kp.sequence(seq, ctxs)
            RO.tick('key_path')
        return ctxs

    def build_contexts(self, seq, targets):
        keep = set(getattr(self, 'hi_need', ()))
        if IMPL['ctx'] != 'shared' or not keep:
            return super().contexts(seq, targets)
        e = self.e
        d, model = e.capture, e.model
        T, H = d.length, model.config.n_head
        D = e.width // H
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        ctxs = []
        for j, block in enumerate(model.transformer.h):  # readout.Method.contexts, copied (keep: the hi blocks' pieces)
            prefix = f'transformer.h.{j}.attn.'
            qraw, kraw, vraw = [d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k', 'c_v')]
            v = vraw.reshape(1, T, H, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix + 've_gate'][seq].float()
                ve = d.outputs[f'value_embeds.{j}'][seq].float().reshape(T, H, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
            values = v.transpose(1, 2)[0]
            window = model.window_sizes[j][0]
            qp, kp, a, out = R.attention_state(qraw, kraw, values[None], cos, sin, H, window)
            q, k, a, out = qp[0], kp[0], a[0], out[0]
            target = targets[j][seq].reshape(T, H, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2)) - (target * out).sum(-1, keepdim=True)
            U, W, M1, M2 = RO.FM.build_future_tables(q, target, a, contrast, window)
            kr = kraw[0].reshape(T, H, D)
            ctx = dict(x=d.blocks[j][seq].float().contiguous(), kraw=kr.contiguous(), gate=gate,
                       R0=R.rotate(kr, cos[0, :, 0][:, None], sin[0, :, 0][:, None]).contiguous(),
                       VC=vraw[0].contiguous(), GZ=None if gate is None else gate.contiguous(),
                       ve=None if ve is None else ve.contiguous(), wg=wg,
                       wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(), U=U, W=W, M1=M1, M2=M2,
                       QD=q.transpose(0, 1).contiguous(), GD=target.transpose(0, 1).contiguous(),
                       AD=a.diagonal(0, 1, 2).T.contiguous(), CD=contrast.diagonal(0, 1, 2).T.contiguous(),
                       VT=values.transpose(0, 1).contiguous(), OT=out.transpose(0, 1).contiguous())
            if (j in RO.BLOCKS and e.s.readout_sources) or j in keep:
                ctx.update(qraw=qraw[0], k=k, values=values, a=a, out=out)
            ctx['sqx'] = ctx['x'].square().sum(-1).contiguous()
            ctxs.append(ctx)
            del contrast
        return ctxs


class E3Mixin:
    """The changes on top of readout_hi's HiMixin / fast_readout's Methods (first in the MRO)."""
    KEYPATH = True  # item 1.4's key path runs inside ContextBase.contexts (False: a pass that never calls it)

    def estimate(self, x, y):
        with patched():
            return super().estimate(x, y)

    def path_credits(self, chain, targets, raw_values=None):
        """readout's (and readout_hi's) path_credits with item 1.4's key path: Cut.path_credits (copied: the value path,
        the input-K credit, the gate path, the identity extra) with the input-K experiments run per sequence inside the
        readout pass on its contexts (KeyPath), then readout's / readout_hi's replacement of extra by the roots."""
        est = self.e
        if IMPL['keypath'] != 'shared' or not self.KEYPATH or not KeyPath.supported(est):
            return super().path_credits(chain, targets, raw_values)
        s, L = est.s, est.model.config.n_layer
        inputs = {i: torch.zeros_like(chain[i]) for i in range(L)}
        if s.value_path:
            _, inputs = est.value_path_credit(chain, return_inputs=True, raw_values=raw_values)
        gate = est.gate_input_credits(targets, raw_values) if s.gate_path else {}  # private generator copy: order-free
        start = time.perf_counter()
        self.chain = chain
        if RO.STATS['keep']:
            self.targets = targets
        self.kp = KeyPath(self, targets)
        try:
            self.run(chain, targets)
            keys = self.kp.finish()
        finally:
            self.kp = None
        inputs = {i: inputs[i] + keys[i] for i in range(L)}
        for i, credit in gate.items():
            inputs[i] = inputs[i] + credit
        extra = {L - 1: torch.zeros_like(chain[0])}
        for i in reversed(range(1, L)):
            extra[i - 1] = est.model.resid_lambdas[i].float() * (inputs[i] + extra[i])
        self.extra, self.inputs = extra, inputs
        out = dict(extra)
        for i in RO.BLOCKS:  # readout's roots
            out[i] = self.X[i]
        for b in getattr(self, 'hi_blocks', ()):  # readout_hi's
            out[b] = self.X[b]
        RO.STATS['seconds'].append(time.perf_counter() - start)
        return out, inputs

    def q_basis(self, ctx, keys):
        """readout's q_basis with its Gram-Schmidt in one kernel (item 1.5a); the rest is readout's."""
        if IMPL['qbasis'] != 'kernel':
            return super().q_basis(ctx, keys)
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        cs, sn = e.model.cos[0, :T, 0].float(), e.model.sin[0, :T, 0].float()
        top = ctx['a'].topk(keys, dim=-1).indices
        kk = ctx['k'][torch.arange(H, device=top.device)[:, None, None], top]
        r = R.rotate(ctx['qraw'].reshape(T, H, D), cs[:, None], sn[:, None]).transpose(0, 1)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        proj = kk - (kk * r[:, :, None]).sum(-1, keepdim=True) * r[:, :, None]
        back = R.rotate(proj, cs[None, :, None], -sn[None, :, None])
        return orthonormal(back.permute(1, 0, 2, 3).reshape(T * H, keys, D)).reshape(T, H, keys, D).transpose(2, 3)

    def hi_plan(self):
        e = self.e
        self.fc_sizes = None
        if IMPL['fccycle'] == 'sized' and e.s.readout_hi_blocks and e.s.readout_sources:
            self.fc_sizes = sized_cycles(e)
            e.s = dataclasses.replace(e.s, readout_hi_fc_cycle=tuple(
                self.fc_sizes.get(b, (c, c, 0.))[1] for b, c in enumerate(e.s.readout_hi_fc_cycle, FIRST)))
            STATS['fc_sizes'] = {b: v[1:] for b, v in self.fc_sizes.items()}
        super().hi_plan()  # (readout_hi reads the sized cycles; readout.Method.estimate restores the settings after)
        self.hi_need = sorted((set(self.hi_blocks) if e.s.readout_sources else set()) | set(self.hi_in))

    def hi_sequence(self, seq, ctxs):
        """readout_hi's, with item 1.6's draws: every code is drawn at its configured size, in readout_hi's order, and a
        sized-down c_fc cycle keeps the first n of its permutation's values < n and the first n sign columns (a uniform
        random cycle of n), so the pass's generator, and every other source's draws, stay as today's."""
        if not self.fc_sizes:
            return super().hi_sequence(seq, ctxs)
        cf, W = self.hi_cfg, self.e.width
        plan = []
        for b in self.hi_blocks:
            if cf['P_q'][b] or cf['P_f'][b]:
                plan += [(cf['q_cycle'][b], cf['q_cycle'][b]), (self.fc_sizes.get(b, (cf['f_cycle'][b],))[0], cf['f_cycle'][b])]
        plan += [(W, W) for _ in self.hi_in]
        self.draw_plan = collections.deque(plan)
        try:
            super().hi_sequence(seq, ctxs)
            if self.draw_plan:
                raise RuntimeError(f'fast_e3: hi_sequence drew fewer codes than planned ({list(self.draw_plan)})')
        finally:
            self.draw_plan = None

    def draw_code(self, gen, T, n, device):
        plan = getattr(self, 'draw_plan', None)
        if not plan:
            return RO.Method.draw_code(gen, T, n, device)
        full, want = plan.popleft()
        if want != n:
            raise RuntimeError(f'fast_e3: hi_sequence drew a cycle of {n}, planned {want}')
        perm, signs = RO.Method.draw_code(gen, T, full, device)
        if full == n:
            return perm, signs
        keep = torch.argsort((perm >= n).to(torch.int8), stable=True)[:n]  # perm's values < n, in order (no sync)
        return perm[keep], signs[:, :n].contiguous()

    def oq_scores(self, c, j, moves, credit, local):
        """Item 1.2: readout_hi's oq_scores (dq = (norm(x + move) - norm(x)) W_q^T as there), the scores from
        inroot_kernel in contrast form."""
        if IMPL['inroot'] == 'plain':
            return super().oq_scores(c, j, moves, credit, local)
        x = c['x'][local]
        dq = torch.matmul(R.norm_delta(x[:, None], moves), self.hi_w['wq'][j].T)
        wproj = self.e.model.transformer.h[j].attn.c_proj.weight
        target = (credit.double() @ wproj.double()).float()
        return inroot_scores(dq.contiguous(), c['qraw'], c['k'], c['values'], c['a'][:, local], c['out'][:, local],
                             target.contiguous(), local.start, self.cs_all, self.sn_all,
                             self.e.model.window_sizes[j][0], IMPL['inroot'])

    def q_moves(self, c, b, dirs, qstep, local):
        """Item 1.3: fast_readout's q_moves with the centered one-pass query kernel (qsrc2_kernel)."""
        if IMPL['qsrc'] == 'plain':
            return super().q_moves(c, b, dirs, qstep, local)
        out = qsrc2_response(dirs.contiguous(), c['qraw'].contiguous(), qstep.contiguous(), c['k'], c['values'],
                             c['a'][:, local], c['out'][:, local], local.start, self.cs_all, self.sn_all,
                             self.e.model.window_sizes[b][0], IMPL['qsrc'])
        return F.linear(out.flatten(-2), self.e.model.transformer.h[b].attn.c_proj.weight.float())


class E3KVMethod(E3Mixin, HI.HiKVMethod, ContextBase):
    """E3 on E1 (readout + receivers) with the changes."""


class E3Method(E3Mixin, HI.HiMethod, ContextBase):
    """E3 on the readout alone (readout_k_receivers 0)."""


class E1KVMethod(E3Mixin, FR.FastKVMethod, ContextBase):
    """E1 (E3 off) with the changes that reach it."""


class E1Method(E3Mixin, FR.FastMethod, ContextBase):
    """The readout alone (E3 and E1 off) with the changes that reach it."""


SHARED = True  # stage 3.1 on these Methods (install(shared=True))
SP, SHARED_CLASSES = None, {}  # shared_push and its Shared on E3KVMethod / E3Method (by readout_kv.kv_on)


def method_class(s):
    if getattr(s, 'readout_shared', False):
        return SHARED_CLASSES[KV.kv_on(s)]
    if HI.hi_on(s):
        return E3KVMethod if KV.kv_on(s) else E3Method
    return E1KVMethod if KV.kv_on(s) else E1Method


def cut_estimate(self, x, y):  # replaces Cut.estimate: readout off = the record's (through readout.py's), called as is
    if not getattr(self.s, 'readout', False):
        return RO.cut_estimate(self, x, y)
    method = method_class(self.s)(self)
    gradients = method.estimate(x, y)
    STATS['calls'] += 1
    if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % 200 == 0):
        print(f"fast_e3 estimate {STATS['calls']}: impl {json.dumps(IMPL)}; c_fc cycles (configured, sized, energy dropped) "
              f"{json.dumps(STATS.get('fc_sizes'))}", flush=True)
    if HI.STATS['keep'] or RO.STATS['keep'] or KV.STATS['keep']:
        self.readout_method = method
    return gradients


def install_chain2(R_, enabled=None):
    """fast_chain2's chain on the recipe (item 1.1; needs fast_chain installed first: the speed stack's job)."""
    enabled = IMPL['chain2'] if enabled is None else enabled
    if enabled and FC.STATE['installed'] is not R_:
        raise RuntimeError('fast_e3: chain2 needs fast_chain installed on this recipe (the speed stack)')
    return FC2.install(R_, enabled=bool(enabled))


def install(shared=None):
    """readout_hi.py's patch (with fast_readout's, readout_kv's, readout's and onesided's), then Cut.estimate (the
    recipe dict and Settings stay readout_hi's). shared (default $FAST_E3_SHARED): shared_push's patch instead (its
    Settings / recipe dict: "readout_shared"), its Shared on this file's Methods. The speed stack (and fast_chain2:
    install_chain2) goes on after."""
    global SP
    shared = SHARED if shared is None else shared
    if shared:
        import shared_push as SP  # noqa: F811
        SP.install()
        mk = lambda base: type(f'Shared{base.__name__}', (SP.Shared, base), dict(KEYPATH=False))
        SHARED_CLASSES.update({True: mk(E3KVMethod), False: mk(E3Method)})
        want, ok = (SP.Settings, SP.asdict), (SP.cut_estimate,)
    else:
        if R.Settings is not HI.Settings:
            HI.install()
        want, ok = (HI.Settings, HI.asdict), (HI.cut_estimate, cut_estimate)
    if (R.Settings, R.asdict) != want or R.Cut.estimate not in ok:
        raise RuntimeError('fast_e3: another wrapper replaced Settings, asdict or Cut.estimate')
    R.Cut.estimate = cut_estimate
    return R


# ------------------------------------------------------------------------------------------------------------ checks


# --------------------------------------------------------------------------------------------- kernels vs float64


# ======================================== module s2_head
"""The chain head's soft-capped log-sum-exp change to second order in the normed-stream change (per-token gradient and
curvature tables) instead of the [rows x vocab] logits; the RMSNorm change and the target's capped logit change stay
exact. A float64 running check falls back to the exact head above its tolerance.
"""
import argparse
import atexit
import functools
import json
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import fast_chain as FC  # noqa: E402
import fast_chain2 as FC2  # noqa: E402

SHA1 = '6710573c2a57'  # sha1 of the research file this section was copied from
EPS = torch.finfo(torch.float32).eps
LIMIT, CHECK_EVERY, CHECK_TOKENS, FALLBACK_REL = 0.1, 50, 4, 5e-4
SCALE_S, SCALE_H = 4096., 1024.
TABLE = (128, 64, 8, 3)  # table_kernel: B, BK, warps, stages
EVAL = (64, 128, 64, 4, 3)  # eval_kernel: BM, BN, BK, warps, stages
ORIG = dict(step=FC2.head_step, fast=FC.head_fast)
STATE = dict(on=False, n=0, fallback=False, checks=[], fixed=None, cache={}, output=None, hook=False, pairs={},
             every=CHECK_EVERY, fallback_rel=FALLBACK_REL, limit=LIMIT)


@triton.jit
def table_kernel(WT, WF, SW, G, Out, PI, PJ, V, W: tl.constexpr, B: tl.constexpr, BK: tl.constexpr,
                 SCALE_S: tl.constexpr, SCALE_H: tl.constexpr):
    """Token t, tile pair (PI[p], PJ[p]) (i <= j): (W^T diag(s_t) W - g_t g_t^T) SCALE_H in fp16, stored and mirrored."""
    p, t = tl.program_id(0), tl.program_id(1)
    bi, bj = tl.load(PI + p), tl.load(PJ + p)
    ri, rj = bi * B + tl.arange(0, B), bj * B + tl.arange(0, B)
    acc = tl.zeros((B, B), tl.float32)
    for v0 in range(0, V, BK):
        vv = v0 + tl.arange(0, BK)
        s = tl.load(SW + t * V + vv) * SCALE_S
        a = (tl.load(WT + ri[:, None] * V + vv[None, :]).to(tl.float32) * s[None, :]).to(tl.float16)
        acc = tl.dot(a, tl.load(WF + vv[:, None] * W + rj[None, :]), acc)
    acc = acc * (1 / SCALE_S)
    acc = tl.where(bi == bj, (acc + tl.trans(acc)) * 0.5, acc)  # diagonal tiles exactly symmetric: the two stores agree
    h = ((acc - tl.load(G + t * W + ri)[:, None] * tl.load(G + t * W + rj)[None, :]) * SCALE_H).to(tl.float16)
    out = Out + t.to(tl.int64) * W * W
    tl.store(out + ri[:, None] * W + rj[None, :], h)
    tl.store(out + rj[None, :] * W + ri[:, None], h)


@triton.jit
def eval_kernel(A, H, G, HB, TS, STATS, Out, n_rows, R, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, SCALE_H: tl.constexpr):
    """BM rows of token t: dinv ab + inv g.d + (dinv^2 sbb + 2 dinv inv hb.d + inv^2 d^T H d) / 2 - dc."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_rows
    row = (t * R + rows).to(tl.int64)
    ht = H + t.to(tl.int64) * W * W
    q = tl.zeros((BM,), tl.float32)
    gd = tl.zeros((BM,), tl.float32)
    hd = tl.zeros((BM,), tl.float32)
    for n0 in range(0, W, BN):
        cols = n0 + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, W, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.).to(tl.float16)
            acc = tl.dot(a, tl.load(ht + kk[:, None] * W + cols[None, :]), acc)
        d = tl.load(A + row[:, None] * W + cols[None, :], ok[:, None], 0.).to(tl.float32)
        q += tl.sum(acc * d, 1)
        gd += tl.sum(d * tl.load(G + t * W + cols)[None, :], 1)
        hd += tl.sum(d * tl.load(HB + t * W + cols)[None, :], 1)
    inv, dinv = tl.load(STATS + row * 4, ok, 1.), tl.load(STATS + row * 4 + 1, ok, 0.)
    ab, sbb = tl.load(TS + 2 * t), tl.load(TS + 2 * t + 1)
    lin = dinv * ab + inv * gd
    sec = dinv * dinv * sbb + 2 * dinv * inv * hd + inv * inv * q * (1 / SCALE_H)
    tl.store(Out + row, lin + 0.5 * sec - tl.load(STATS + row * 4 + 2, ok, 0.), ok)


@triton.jit
def fix_kernel(A, WT, STATS, BASE, T0, P0, Out, Count, n_rows, R, limit, V: tl.constexpr, W: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """A block of BM rows of token t with a bound >= limit: today's head arithmetic (head_fix_kernel's), Out overwritten."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_rows
    row = (t * R + rows).to(tl.int64)
    if tl.max(tl.load(STATS + row * 4 + 3, ok, 0.), 0) >= limit:
        tl.atomic_add(Count, 1)
        inv = tl.load(STATS + row * 4, ok, 1.)[:, None]
        dinv = tl.load(STATS + row * 4 + 1, ok, 0.)[:, None]
        part = tl.zeros((BM,), tl.float32)
        for v0 in range(0, V, BN):
            cols = v0 + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, W, BK):
                kk = k0 + tl.arange(0, BK)
                acc = tl.dot(tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.),
                             tl.load(WT + kk[:, None] * V + cols[None, :]), acc)
            t0 = tl.load(T0 + t * V + cols)[None, :]
            th = libdevice.tanh((tl.load(BASE + t * V + cols)[None, :] * dinv + acc * inv) * (1 / 15))
            part += tl.sum(tl.load(P0 + t * V + cols)[None, :] * libdevice.expm1(15 * th * (1 - t0 * t0) / (1 + t0 * th)), 1)
        tl.store(Out + row, libdevice.log1p(part) - tl.load(STATS + row * 4 + 2, ok, 0.), ok)


def pairs(device, n):
    if (device, n) not in STATE['pairs']:
        ij = [(i, j) for i in range(n) for j in range(i, n)]
        STATE['pairs'][device, n] = [torch.tensor(v, device=device, dtype=torch.int32) for v in zip(*ij)]
    return STATE['pairs'][device, n]


def tables(ctx, t0, p0, cache):
    """(H fp16 [m, W, W], g [m, W], hb [m, W], (ab, sbb) [m, 2]) of the call's tokens; cached per chain call if asked."""
    weight_t, base, c = ctx['weight_t'], ctx['base'], STATE['cache']
    key = ctx['xL'].data_ptr()
    if cache:
        if c.get('weight_t') is not weight_t:  # a new chain call (the cache holds the old weight_t: no id reuse)
            c.clear()
            c['weight_t'] = weight_t
        if key in c:
            return c[key][1:]
    W, (m, V) = weight_t.shape[0], base.shape
    cp = 1 - t0 * t0
    aw = p0 * cp
    sw = (aw * (cp - t0 * (2 / 15))).contiguous()
    gh = torch.stack([aw, sw * base], 1).double() @ weight_t.double().T  # [m, 2, W]
    ab = (aw.double() * base).sum(-1)
    g = gh[:, 0]
    hb = (gh[:, 1] - ab[:, None] * g).float().contiguous()
    ts = torch.stack([ab, (sw.double() * base.double().square()).sum(-1) - ab * ab], 1).float().contiguous()
    g = g.float().contiguous()
    B, BK, warps, stages = TABLE
    pi, pj = pairs(base.device, W // B)
    H = torch.empty(m, W, W, device=base.device, dtype=torch.float16)
    table_kernel[(len(pi), m)](weight_t, weight_t.T.half().contiguous(), sw, g, H, pi, pj, V, W, B, BK, SCALE_S, SCALE_H,
                               num_warps=warps, num_stages=stages)
    if cache:
        c[key] = (ctx['xL'], H, g, hb, ts)  # the xL view keeps its storage (the key) alive
    return H, g, hb, ts


def expand(a, stats, ctx, t0, p0, cache):
    """change [m, rows] from the bf16 rows a [m, rows, W] and the stats [m, rows, 4] (inv', dinv, dc_target, bound)."""
    m, rows, W = a.shape
    H, g, hb, ts = tables(ctx, t0, p0, cache)
    out = torch.empty(m, rows, device=a.device)
    bm, bn, bk, warps, stages = EVAL
    eval_kernel[(triton.cdiv(rows, bm), m)](a, H, g, hb, ts, stats, out, rows, rows, W, bm, bn, bk, SCALE_H,
                                            num_warps=warps, num_stages=stages)
    if STATE['fixed'] is None:
        STATE['fixed'] = torch.zeros(1, device=a.device, dtype=torch.int32)
    fix_kernel[(triton.cdiv(rows, 64), m)](a, ctx['weight_t'], stats, ctx['base'], t0, p0, out, STATE['fixed'], rows,
                                           rows, STATE['limit'], t0.shape[1], W, 64, 128, 64, num_warps=8, num_stages=1)
    STATE['n'] += 1
    if STATE['every'] and (STATE['n'] == 1 or STATE['n'] % STATE['every'] == 0):
        running_check(a, stats, ctx, out)
    return out


def exact64(a, stats, ctx, toks, order2=False):
    """Float64 change [k, rows] of tokens toks: the exact log-sum-exp of the same dz (same rows, stats, base, W);
    order2: its second-order expansion in float64 (the truncation alone)."""
    W = a.shape[-1]
    wt = ctx['weight_t'].double()
    cap = lambda z: 15 * torch.tanh(z / 15)
    out = []
    for t in toks:
        st = stats[t].double()
        base = ctx['base'][t].double()
        z0 = base * torch.rsqrt(ctx['xL'][t].square().sum() / W + EPS).double()
        dz = base * st[:, 1:2] + (a[t].double() @ wt) * st[:, :1]
        if order2:
            t0 = torch.tanh(z0 / 15)
            cp = 1 - t0 * t0
            aw = torch.softmax(15 * t0, -1) * cp
            lin = dz @ aw
            out.append(lin + (dz.square() @ (aw * (cp - t0 * 2 / 15)) - lin * lin) / 2 - st[:, 2])
            continue
        out.append(torch.logsumexp(cap(z0 + dz), -1) - torch.logsumexp(cap(z0), -1) - st[:, 2])
    return torch.stack(out)


def group_rel(got, ref, group=64):
    """Max over row groups (all tokens pooled) of |got - ref| / |ref|, and the overall relative L2."""
    k, rows = ref.shape
    pad = -rows % group
    e = torch.nn.functional.pad((got.double() - ref), (0, pad)).view(k, -1, group).square().sum((0, 2))
    r = torch.nn.functional.pad(ref, (0, pad)).view(k, -1, group).square().sum((0, 2))
    return (e / r.clamp_min(1e-300)).sqrt().max().item(), ((got.double() - ref).norm() / ref.norm()).item()


def running_check(a, stats, ctx, out):
    m = a.shape[0]
    k = min(CHECK_TOKENS, m)
    s0 = (97 * (STATE['n'] // STATE['every'])) % (m - k + 1)
    toks = range(s0, s0 + k)
    rel, overall = group_rel(out[s0:s0 + k], exact64(a, stats, ctx, toks))
    ok = rel <= STATE['fallback_rel']
    STATE['checks'].append(dict(call=STATE['n'], tokens=[s0, s0 + k], rel_group_max=float(f'{rel:.3e}'),
                                rel=float(f'{overall:.3e}'), ok=ok))
    if not ok:
        STATE['fallback'] = True
        print(f"s2_head rank {os.environ.get('RANK', '0')}: running check failed ({STATE['checks'][-1]}); today's head "
              'from now on', flush=True)
    note()


def note():
    if STATE['output'] is None or os.environ.get('RANK', '0') != '0' or not STATE['output'].is_dir():
        return
    fixed = int(STATE['fixed'].item()) if STATE['fixed'] is not None else 0
    (STATE['output'] / 's2_head.json').write_text(json.dumps(dict(
        sha1=SHA1, on=STATE['on'], calls=STATE['n'], fallback=STATE['fallback'], fixed_blocks=fixed, limit=STATE['limit'],
        fallback_rel=STATE['fallback_rel'], checks=STATE['checks'][-40:]), indent=1) + '\n')


def usable(ctx, W):
    return not STATE['fallback'] and not ctx['exact_head'] and W % TABLE[0] == 0 and ctx['weight_t'].shape[1] % 128 == 0


def clean(ctx, W):
    """Today's clean quantities: |xL|^2, t0, p0 and the target / bound inputs of the stats kernels."""
    xL, base, y = ctx['xL'], ctx['base'], ctx['y']
    sq = xL.square().sum(-1, keepdim=True)
    t0 = torch.tanh(base * torch.rsqrt(sq / W + EPS) / 15)
    p0 = torch.softmax(15 * t0, -1)
    target = y.clamp(min=0).unsqueeze(1)
    return sq, t0, p0, (base.gather(1, target).contiguous(), t0.gather(1, target).contiguous(),
                        base.abs().amax(-1).contiguous(), ctx['head_row_norm'])


@functools.wraps(ORIG['step'])
def head_step(m, rows, W, ctx):  # fast_chain2.head_step: its head_stats_kernel, then the expansion
    if not usable(ctx, W):
        return ORIG['step'](m, rows, W, ctx)
    sq, t0, p0, tgt = clean(ctx, W)
    stats = ctx['scratch32'][:m * rows * 4]
    FC2.head_stats_kernel[(triton.cdiv(rows, 64), m)](ctx['hps'], sq, *tgt, stats, rows, rows, W, W // FC2.DOWN[1], 64,
                                                      EPS, num_warps=4)
    return expand(ctx['scratch'][:m * rows * W].view(m, rows, W), stats.view(m, rows, 4), ctx, t0, p0, False)


@functools.wraps(ORIG['fast'])
def head_fast(delta, ctx):  # fast_chain.head_fast: its head_prep_kernel, then the expansion (tables cached per chain call)
    m, rows, W = delta.shape
    if not usable(ctx, W):
        return ORIG['fast'](delta, ctx)
    sq, t0, p0, tgt = clean(ctx, W)
    a = ctx['scratch'][:m * rows * W].view(m, rows, W)
    stats = ctx['scratch32'][:m * rows * 4]
    br, bk, warps = FC.HEAD_PREP
    FC.head_prep_kernel[(triton.cdiv(rows, br), m)](delta, ctx['xL'], ctx['target_w'], sq, *tgt, a, stats, rows, rows,
                                                    W, br, bk, EPS, num_warps=warps)
    return expand(a, stats.view(m, rows, 4), ctx, t0, p0, True)


def install(enabled=None):
    """S2_HEAD (default) on: both heads replaced; off: today's heads (nothing of this file runs). Idempotent."""
    enabled = False if enabled is None else enabled
    FC2.head_step, FC.head_fast = (head_step, head_fast) if enabled else (ORIG['step'], ORIG['fast'])
    STATE['on'] = bool(enabled)
    STATE['cache'].clear()
    if enabled and os.environ.get('RANK') is not None and not STATE['hook']:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', type=Path, default=Path('logs/val'))
        STATE['output'] = parser.parse_known_args(sys.argv[1:])[0].output
        atexit.register(note)
        STATE['hook'] = True
        if os.environ.get('RANK') == '0':
            print(f's2_head: the chain head runs the second-order tables (sha1 {SHA1}, limit {LIMIT}, check every '
                  f'{CHECK_EVERY} calls, fallback above {FALLBACK_REL})', flush=True)


# ------------------------------------------------------------------------------------------------------------ check


# ======================================== module s2_skipslots
"""The shared pass skips empty receiver slots: two row kernels walk each token's list of live rows (the same arithmetic
for a live row; a skipped row is a zero the decode drops).
"""
import contextlib
import inspect
import linecache
import textwrap

import torch
import torch.nn.functional as F
import triton

import fast_e3 as FE  # noqa: E402  (readout_hi, fast_readout, readout_kv, readout, the recipe module)
import shared_push as SP  # noqa: E402
FR, KV, RO, R = FE.FR, FE.KV, FE.RO, FE.R

ON = True
STATS = dict(keep=False)  # checks: the receiver slots' live share


# ------------------------------------------------------------------------------------------- the live-list kernels


def listed(kernel, old, new):
    """kernel with its row lines `old` replaced by `new` and (LROW, LCNT, LSTRIDE) in front of its arguments: LROW [m,
    LSTRIDE] each token's live rows in order, LCNT [m, LSTRIDE] its live rows below each row. Compiled as its own kernel."""
    fn = kernel.fn
    src = textwrap.dedent(inspect.getsource(fn))
    if src.count(old) != 1:
        raise RuntimeError(f's2_skipslots: {fn.__name__} has not the rows it replaces')
    name = fn.__name__ + '_live'
    src = src[src.index('def '):].replace(old, new).replace(f'def {fn.__name__}(', f'def {name}(LROW, LCNT, LSTRIDE, ')
    file = f'<s2_skipslots {name}>'
    linecache.cache[file] = (len(src), None, src.splitlines(True), file)  # triton reads the source back
    scope = dict(fn.__globals__)
    exec(compile(src, file, 'exec'), scope)
    return triton.jit(scope[name])


def take(pos, var, mask, n, t, step):
    """step rows from position pos of token t's live list (live_n: its live rows below n; a program past it returns).
    The list and the count load side by side (a row read waiting on the count cost ~5% of a tile)."""
    return (f'    {var} = {pos} + tl.arange(0, {step})\n    live_row = tl.load(LROW + {t} * LSTRIDE + {var}, {var} < LSTRIDE, 0)\n'
            f'    live_n = tl.load(LCNT + {t} * LSTRIDE + {n})\n    if {pos} >= live_n:\n        return\n'
            f'    {mask} = {var} < live_n\n    {var} = live_row')


KERNELS = {  # (module, name) the stack's launches look up -> the live-list kernel
    (RO, 'staged_kernel'): listed(RO.staged_kernel, '    p = pb * BP + tl.arange(0, BP)\n    pm = p < n',
                                  take('pb * BP', 'p', 'pm', 'n', 't', 'BP')),
    (FR, 'dual_residual_kernel'): listed(  # per token (program_id(2)) instead of the flattened rows
        FR.dual_residual_kernel, '    pn, pm = tl.program_id(0), tl.program_id(1)\n    i = pm * BM + tl.arange(0, BM)\n'
        '    ok = i < n\n    tok, r = i // n_rows, i % n_rows',
        '    pn, pm, tok = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n' + take('pm * BM', 'r', 'ok', 'n_rows', 'tok', 'BM')),
}


class Launch:
    """kernel[grid](*args) as the stack launches it, run as the live-list kernel; the residual's flattened grid (columns,
    rows of m x n) becomes (columns, rows of n, tokens)."""

    def __init__(self, kernel, lists, per_token):
        self.kernel, self.per_token, (self.rows, self.cnt) = kernel, per_token, lists

    def __getitem__(self, grid):
        def launch(*a, **kw):  # dual_residual_kernel: a[5] rows per token, a[11] BM
            g = (grid[0], triton.cdiv(a[5], a[11]), self.cnt.shape[0]) if self.per_token else grid
            return self.kernel[g](self.rows, self.cnt, self.cnt.shape[1], *a, **kw)
        return launch


@contextlib.contextmanager
def swapped(lists):
    """The row kernels the stack's launches name (module attributes, looked up at each call) as Launches."""
    plain = {key: getattr(*key) for key in KERNELS}
    for (mod, name), kernel in KERNELS.items():
        setattr(mod, name, Launch(kernel, lists, name == 'dual_residual_kernel'))
    try:
        yield
    finally:
        for (mod, name), kernel in plain.items():
            setattr(mod, name, kernel)


# ------------------------------------------------------------------------------------------------------- the mixin


class SkipSlots:
    """First in the MRO of fast_e3's shared Methods: per chunk, the receiver groups' live slots (taken as their moves are
    formed) make the tokens' live-row lists, and score / push run the live-list kernels. The chain (no receiver rows) and
    the switch off run the stack's kernels."""

    def chunk(self, buf, ctxs, seq, t0, t1):  # every chunk of the chain and of the pass starts here
        self.sk_live, self.sk_lists, self.sk_new = None, None, []
        return super().chunk(buf, ctxs, seq, t0, t1)

    def moves(self, g, p, c, t0, t1):
        if ON:  # the decode's live mask of these slots [m, C]
            tbl = p['table'][t0:t1]
            self.sk_new.append((tbl >= 0) & p['live'].reshape(-1)[tbl.clamp_min(0)])
        return super().moves(g, p, c, t0, t1)

    def mlp(self, x, delta, n, b, dx0=None, P=0):
        if getattr(self, 'sk_new', None):  # block b's groups, formed just now: their rows end at n (..., groups of b | MLP b)
            self.lists(delta, n)
        return super().mlp(x, delta, n, b, dx0, P)

    def score(self, *a):
        return self.listed(super().score, *a)

    def push(self, *a):
        return self.listed(super().push, *a)

    def listed(self, step, *a):
        if getattr(self, 'sk_lists', None) is None:
            return step(*a)
        with swapped(self.sk_lists):
            return step(*a)

    def lists(self, delta, n):
        """Each token's live rows in order, then its dead ones (LROW), and its live rows below each row (LCNT); both
        [m, rows + 1] (one stride). No host synchronization."""
        m, rows = delta.shape[:2]
        if self.sk_live is None:
            self.sk_live = torch.ones(m, rows, dtype=torch.bool, device=delta.device)
        new = torch.cat(self.sk_new, 1)
        self.sk_live[:, n - new.shape[1]:n], self.sk_new = new, []
        if STATS['keep']:
            STATS['live'] = STATS.get('live', 0) + new.sum()
            STATS['slots'] = STATS.get('slots', 0) + new.numel()
        cnt = F.pad(self.sk_live.cumsum(1, dtype=torch.int32), (1, 0))
        below, p = cnt[:, :-1], torch.arange(rows, device=delta.device, dtype=torch.int32)
        at = torch.where(self.sk_live, below, cnt[:, -1:] + p - below)  # a permutation: live rows first, in order
        self.sk_lists = (torch.zeros_like(cnt).scatter_(1, at.long(), p.expand(m, rows)), cnt)


# -------------------------------------------------------------------------------------------------------- install


def install(shared=None):
    """fast_e3's patch (shared: its default, FAST_E3_SHARED), then this mixin first in its shared Methods."""
    FE.install(shared)
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, SkipSlots):
            FE.SHARED_CLASSES[kv] = type(f'SkipSlots{cls.__name__}', (SkipSlots, cls), {})
    return R


# --------------------------------------------------------------------------------------------------------- check


# ======================================== module s2_plan
"""The receiver plan (readout_kv.Method.plan: top receivers, guard, dual basis, key changes, gains) in one kernel and the
chain's routes (masked softmax, top-k, bf16 cast) in one fused kernel; swapped in while a readout estimate runs.
"""
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import readout_kv as KV  # noqa: E402  (loads readout, fast_chain, onesided and the recipe module)
RO, FC = KV.RO, KV.FC

SWITCH = ''
PLAIN = dict(plan=KV.Method.plan, routes=FC.query_routes, estimate=RO.Method.estimate)
BT, PLAN_WARPS = 4, 2  # plan_kernel: keys per program, warps (swept: 0.11 ms per block)
STATE = dict(noted=False)


def parts(switch):
    return {'': (), '0': (), '1': ('plan', 'routes'), 'plan': ('plan',), 'routes': ('routes',)}[switch]


# ----------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def plan_kernel(IDX, VALS, KEPT, QD, R0, KRAW, CS, SN, P, LIVE, COEF, XLIN, T,
                H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, TAU: tl.constexpr,
                SIGMA: tl.constexpr, EPS: tl.constexpr, SQD: tl.constexpr):
    """readout_kv.Method.plan's key experiments for keys t (BT per program) of head h, R receivers each (vectors as halves
    [BT, R, D / 2]): the pull-backs p_r (key-norm Jacobian, inverse RoPE), the guard (Gram-Schmidt twice in receiver order;
    accepted iff kept and >= TAU |p_r| outside the stronger accepted ones' span), the dual basis L^-T Q, x_lin = step /
    |d_r|, the move's exact key change (RoPE + QK-norm), the receiver's logit change x and gain a (e^x - 1) / (1 +
    a (e^x - 1)). Out: P [H, T, R, D], live, gain and x_lin [H, T, R]."""
    pt, h = tl.program_id(0), tl.program_id(1)
    HD: tl.constexpr = D // 2
    t = pt * BT + tl.arange(0, BT)
    r = tl.arange(0, R)[None, :]
    d = tl.arange(0, HD)[None, None, :]
    on_t = (t < T)[:, None]
    htr = (h * T + t)[:, None] * R + r  # [BT, R]
    rec = tl.load(IDX + htr, on_t, 0)
    val = tl.load(VALS + htr, on_t, 0.)
    kept = tl.load(KEPT + htr, on_t, 0) != 0
    qo = (rec * H + h)[:, :, None] * D + d
    qlo, qhi = tl.load(QD + qo, on_t[:, :, None], 0.), tl.load(QD + qo + HD, on_t[:, :, None], 0.)
    uo = (t * H + h)[:, None, None] * D + d
    ulo, uhi = tl.load(R0 + uo, on_t[:, :, None], 0.), tl.load(R0 + uo + HD, on_t[:, :, None], 0.)  # [BT, 1, HD]
    klo, khi = tl.load(KRAW + uo, on_t[:, :, None], 0.), tl.load(KRAW + uo + HD, on_t[:, :, None], 0.)
    co = t[:, None, None] * HD + d
    c, s = tl.load(CS + co, on_t[:, :, None], 1.), tl.load(SN + co, on_t[:, :, None], 0.)
    sq = tl.sum(ulo * ulo, 2) + tl.sum(uhi * uhi, 2)  # [BT, 1]
    ms = (sq / D + EPS)[:, :, None]
    uq = (tl.sum(ulo * qlo, 2) + tl.sum(uhi * qhi, 2))[:, :, None]
    jlo, jhi = (qlo - ulo * uq / (D * ms)) / tl.sqrt(ms), (qhi - uhi * uq / (D * ms)) / tl.sqrt(ms)
    plo, phi = (jlo * c - jhi * s) / SQD, (jlo * s + jhi * c) / SQD  # p_r = d logit / d k_raw
    nrm = tl.sqrt(tl.sum(plo * plo, 2) + tl.sum(phi * phi, 2))  # [BT, R]
    Qlo, Qhi = tl.zeros([BT, R, HD], tl.float32), tl.zeros([BT, R, HD], tl.float32)
    L = tl.zeros([BT, R, R], tl.float32)  # P_acc = L Q, row i: the accepted receiver i's coefficients
    acc = tl.zeros([BT, R], tl.int32)
    for i in tl.static_range(R):
        on = r == i
        vlo, vhi = tl.sum(tl.where(on[:, :, None], plo, 0.), 1), tl.sum(tl.where(on[:, :, None], phi, 0.), 1)
        row = tl.zeros([BT, R], tl.float32)
        if i > 0:
            for _ in tl.static_range(2):
                cf = tl.sum(Qlo * vlo[:, None, :], 2) + tl.sum(Qhi * vhi[:, None, :], 2)
                vlo, vhi = vlo - tl.sum(cf[:, :, None] * Qlo, 1), vhi - tl.sum(cf[:, :, None] * Qhi, 1)
                row += cf
        rn = tl.sqrt(tl.sum(vlo * vlo, 1) + tl.sum(vhi * vhi, 1))  # [BT]
        ni = tl.sum(tl.where(on, nrm, 0.), 1)
        good = (tl.sum(tl.where(on & kept, 1, 0), 1) > 0) & (rn >= TAU * ni) & (ni > 0)
        put = on[:, :, None] & good[:, None, None]
        div = tl.maximum(rn, 1e-30)[:, None]
        Qlo, Qhi = tl.where(put, (vlo / div)[:, None, :], Qlo), tl.where(put, (vhi / div)[:, None, :], Qhi)
        L = tl.where(put, tl.where(on, rn[:, None], row)[:, None, :], L)
        acc = tl.where(on & good[:, None], 1, acc)
    Dlo, Dhi = tl.zeros([BT, R, HD], tl.float32), tl.zeros([BT, R, HD], tl.float32)
    for j in tl.static_range(R):  # d_i = (q_i - sum_{k > i} L_ki d_k) / L_ii, from the last receiver down
        on = r == R - 1 - j
        col = tl.sum(tl.where(r[:, None, :] == R - 1 - j, L, 0.), 2)  # L[k, i] (k = i: d_i is still zero)
        good = tl.sum(tl.where(on, acc, 0), 1) > 0
        diag = tl.where(good, tl.sum(tl.where(on, col, 0.), 1), 1.)[:, None]
        dlo = (tl.sum(tl.where(on[:, :, None], Qlo, 0.), 1) - tl.sum(col[:, :, None] * Dlo, 1)) / diag
        dhi = (tl.sum(tl.where(on[:, :, None], Qhi, 0.), 1) - tl.sum(col[:, :, None] * Dhi, 1)) / diag
        put = on[:, :, None] & good[:, None, None]
        Dlo, Dhi = tl.where(put, dlo[:, None, :], Dlo), tl.where(put, dhi[:, None, :], Dhi)
    live = acc > 0
    step = SIGMA * tl.maximum(tl.sqrt(tl.sum(klo * klo, 2) + tl.sum(khi * khi, 2)), 1e-12)  # [BT, 1]
    dn = tl.sqrt(tl.sum(Dlo * Dlo, 2) + tl.sum(Dhi * Dhi, 2))
    xlin = tl.where(live, step / tl.maximum(dn, 1e-30), 0.)  # the first-order logit change of receiver r
    mlo, mhi = xlin[:, :, None] * Dlo, xlin[:, :, None] * Dhi
    rlo, rhi = mlo * c + mhi * s, mhi * c - mlo * s  # RoPE at t
    grow = (2 * (tl.sum(ulo * rlo, 2) + tl.sum(uhi * rhi, 2)) + tl.sum(rlo * rlo, 2) + tl.sum(rhi * rhi, 2)) / D
    inverse = libdevice.rsqrt(sq / D + EPS)
    moved = libdevice.rsqrt(sq / D + grow + EPS)
    shrink = (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, :, None]
    dklo, dkhi = rlo * moved[:, :, None] - ulo * shrink, rhi * moved[:, :, None] - uhi * shrink  # QK-norm change
    x = (tl.sum(qlo * dklo, 2) + tl.sum(qhi * dkhi, 2)) / SQD
    ex = libdevice.expm1(x)
    po = htr[:, :, None] * D + d
    tl.store(P + po, plo, on_t[:, :, None])
    tl.store(P + po + HD, phi, on_t[:, :, None])
    tl.store(LIVE + htr, live, on_t)
    tl.store(COEF + htr, tl.where(live, val * ex / (1 + val * ex), 0.), on_t)
    tl.store(XLIN + htr, xlin, on_t)


@triton.jit
def routes_kernel(LG, PROB, IDX, PV, T, window, SCALE: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                  BN: tl.constexpr):
    """Query t of head h: the masked softmax of its bf16 logits row (keys t - window .. t), the probabilities as bf16
    (PROB), and the K largest (IDX int32 / PV fp32 at [t, h, :], in key order): threshold = the K-th largest 16-bit key
    of the bf16 logits (monotone in the logit), found by bisection; ties at it by key index."""
    t, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, BN)
    row = (h * T + t).to(tl.int64) * T
    allowed = (j <= t) & (j >= t - window)
    lg = tl.load(LG + row + j, j < T, 0.)
    bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key = tl.where(allowed, tl.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000), tl.where(j < T, 0, -1))
    z = tl.where(allowed, lg.to(tl.float32) * SCALE, float('-inf'))
    e = libdevice.exp(z - tl.max(z, 0))
    p = e / tl.sum(e, 0)
    tl.store(PROB + row + j, p.to(tl.bfloat16), j < T)
    lo, hi = 0, 65536
    for _ in tl.static_range(16):
        mid = (lo + hi) // 2
        big = tl.sum((key >= mid).to(tl.int32), 0) >= K
        lo, hi = tl.where(big, mid, lo), tl.where(big, hi, mid)
    eq = key == lo
    sel = (key > lo) | (eq & (tl.cumsum(eq.to(tl.int32), 0) <= K - tl.sum((key > lo).to(tl.int32), 0)))
    out = (t * H + h).to(tl.int64) * K + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(IDX + out, j, sel)
    tl.store(PV + out, p, sel)


# ------------------------------------------------------------------------------------------------------- the patches


def plan(self, b, seq, c, kinds, cs, sn):
    """readout_kv.Method.plan with the key experiments in plan_kernel (its top-R receivers, slot table and gate
    coefficients as there)."""
    e, s = self.e, self.e.s
    H, T = e.model.config.n_head, e.capture.length
    D, Rr, C = e.width // H, s.readout_k_receivers, self.kv_C
    if 'k' not in kinds or Rr & (Rr - 1) or D & (D - 1):  # the kernel: R and D powers of two
        return PLAIN['plan'](self, b, seq, c, kinds, cs, sn)
    A = self.clean_attention(b, seq, c)
    top = A.transpose(1, 2).topk(Rr, dim=-1)  # per (h, key t): its strongest receivers s, in attention order
    vals, idx = top.values.contiguous(), top.indices.contiguous()  # [H, T, R]
    ok = vals > 0
    table, kept = KV.receiver_slots(idx, vals, ok, C, T)
    P = torch.empty(H, T, Rr, D, device=A.device)
    live, coef, x_lin = torch.empty_like(ok), torch.empty_like(vals), torch.empty_like(vals)
    plan_kernel[(triton.cdiv(T, BT), H)](idx, vals, kept, c['QD'], c['R0'], c['kraw'], cs, sn, P, live, coef, x_lin, T, H,
                                         D, Rr, BT, KV.K_TAU, KV.K_SIGMA, KV.EPS, math.sqrt(D), num_warps=PLAN_WARPS)
    plans = {(b, 'k'): dict(table=table, live=live, coef=coef, P=P, x_lin=x_lin)}
    self.kv_counts[b, 'k'] = torch.stack([ok.sum(), kept.sum(), live.sum()])
    if 'g' in kinds:  # readout_kv's
        z = c['gate']
        dvs = 2 * (torch.sigmoid(z + KV.GATE_SIGMA) - torch.sigmoid(z))
        coef = torch.where(kept, vals * dvs.T[:, :, None], torch.zeros_like(vals))
        plans[b, 'g'] = dict(table=table, live=kept, coef=coef)
        self.kv_counts[b, 'g'] = torch.stack([ok.sum(), kept.sum(), kept.sum()])
    return plans


def query_routes(own, k, rest=False):
    """fast_chain.query_routes (its query, logits and output GEMMs as there) with routes_kernel for the masked softmax,
    the top-k and the bf16 probabilities. Positions are 0..T-1 (Cut.own_query_inputs')."""
    if rest:
        return PLAIN['routes'](own, k, rest)
    pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
    length = len(pos)
    K = min(k, length)
    out = []
    for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
        heads, _, width = kk.shape
        c, s = cos[:, None], sin[:, None]
        r = (F.rms_norm(x, (x.shape[-1],)) @ wq.T).view(length, heads, width)
        r0 = torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                        r[..., :width // 2] * -s + r[..., width // 2:] * c], -1)
        q = F.rms_norm(r0, (width,))
        logits = torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk)
        prob = torch.empty_like(logits)
        idx = torch.empty(length, heads, K, device=x.device, dtype=torch.int32)
        p = torch.empty(length, heads, K, device=x.device)
        routes_kernel[(length, heads)](logits, prob, idx, p, length, window, width ** -0.5, K, heads,
                                       triton.next_power_of_2(length), num_warps=4)
        o = torch.einsum('hts,hsd->htd', prob, vv).float()
        out.append(dict(idx=idx, p=p, o=o.transpose(0, 1).contiguous(), r0=r0.contiguous(),
                        sq0=r0.square().sum(-1).contiguous(), keys=kk.contiguous(), values=vv.contiguous()))
        del logits, prob
    return out


def estimate(self, x, y):
    """readout.Method.estimate (as installed before) with this file's plan / routes for the call."""
    on, saved = parts(SWITCH), (KV.Method.plan, FC.query_routes)
    if on and not STATE['noted'] and os.environ.get('RANK', '0') == '0':
        print(f's2_plan: this estimate runs {" and ".join(on)} from s2_plan.py', flush=True)
        STATE['noted'] = True
    if 'plan' in on:
        KV.Method.plan = plan
    if 'routes' in on:
        FC.query_routes = query_routes
    try:
        return PLAIN['estimate'](self, x, y)
    finally:
        KV.Method.plan, FC.query_routes = saved


def install(switch=None):
    """Wrap readout.Method.estimate (idempotent); switch: S2_PLAN's values (default: the environment's)."""
    global SWITCH
    SWITCH = SWITCH if switch is None else switch
    parts(SWITCH)  # refuse an unknown value
    if RO.Method.estimate is not estimate:
        PLAIN['estimate'] = RO.Method.estimate
        RO.Method.estimate = estimate


# ------------------------------------------------------------------------------------------------------------- check


# ======================================== module s2_momentgemm
"""fast_moments' moment tables built from banded GEMMs and the input-K scorer as GEMMs (exact algebra, rounding level)."""
import contextlib

import torch
import triton
import triton.language as tl

import fast_moments as FM  # noqa: E402

PLAIN = {n: getattr(FM, n) for n in ('build_future_tables', 'build_tables', 'inputk_scores')}
ALL = ('tables', 'inputk')
EPS = FM.EPS
BT = 512  # sources per banded bmm (swept: 128 / 256 / 512 -> 1.51 / 1.18 / 1.12 ms per S block, shared GPU)
PARTS = ()


@contextlib.contextmanager
def ieee():  # fp32 GEMMs without TF32 (the legacy API: the recipe sets 'high' with it)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


# ------------------------------------------------------------------------------------------------------------ tables


@torch.compile(dynamic=False)
def _weights(a, c, window: int, future: bool):
    """a, c [H, T(query s), T(source t)] -> a c (fp32), a c (1/2 - a) (bf16) [, a (fp32), a (1 - a) (bf16)] on the pairs
    t < s <= t + window, 0 elsewhere."""
    pos = torch.arange(a.shape[-1], device=a.device)
    lag = pos[:, None] - pos[None, :]
    a = torch.where((lag > 0) & (lag <= window), a, 0.)
    ac = a * c
    w = (ac, (ac * (0.5 - a)).bfloat16())
    return w + (a, (a * (1 - a)).bfloat16()) if future else w


@triton.jit
def outer_kernel(X, Y, Out, N, XB, XN, YB, YN, D: tl.constexpr, BN: tl.constexpr):
    b, n = tl.program_id(0), tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, D * D)
    ok = (n < N)[:, None]
    x = tl.load(X + b * XB + n[:, None] * XN + (k // D)[None, :], ok)
    y = tl.load(Y + b * YB + n[:, None] * YN + (k % D)[None, :], ok)
    tl.store(Out + (b * N + n[:, None]) * D * D + k[None, :], (x * y).to(tl.bfloat16), ok)


def _outer(x, y):
    """vec(x_n y_n^T) [B, N, D * D] bf16 of x, y [B, N, D] (unit last stride; the fp32 product rounded once)."""
    B, N, D = x.shape
    out = torch.empty((B, N, D * D), device=x.device, dtype=torch.bfloat16)
    outer_kernel[(B, triton.cdiv(N, 4))](x, y, out, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), D, 4)
    return out


def tables(q, a, c, window, g=None):
    """U [T, H, D] fp32, M1 [T, H, D, D] bf16 (with g: U, W, M1, M2) of the queries t < s <= t + window; q, g [H, T, D]."""
    H, T, D = q.shape
    w = _weights(a, c, window, g is not None)
    lin, quad = [], []
    for y, wl, wq in [(q, w[0], w[1])] + ([(g, w[2], w[3])] if g is not None else []):
        V = q.new_empty((T, H, D))
        with ieee():
            torch.bmm(wl.mT, y, out=V.transpose(0, 1))
        M = torch.empty((T, H, D, D), device=q.device, dtype=torch.bfloat16)
        qy, Mv = _outer(q, y), M.view(T, H, D * D)
        for t0 in range(0, T, BT):  # sources t0:t1 read the queries t0:s1 (s = t0 has weight 0)
            t1 = min(T, t0 + BT)
            s1 = min(T, t1 + window)
            torch.bmm(wq[:, t0:s1, t0:t1].mT, qy[:, t0:s1], out=Mv[t0:t1].transpose(0, 1))
        lin.append(V)
        quad.append(M)
    return (*lin, *quad)


def build_future_tables(q, g, a, contrast, window):
    return tables(q, a, contrast, window, g)


def build_tables(qp, A, C, window):
    return tables(qp, A, C, window)


# ------------------------------------------------------------------------------------------------------------ input-K


@triton.jit
def prep_kernel(M1, U, Q, PX, KR, CS, SN, AD, CD, Mt, V, S, T, QH, QT, KH, KT, AH, AT, CH, CT,
                H: tl.constexpr, D: tl.constexpr):
    """Per (t, h): Mt = R_t^T M1 R_t (bf16), V[h, :, t] = (lam kr, lam px, R^T u, R^T q, Ms px, Ms kr) (Ms = Mt + Mt^T,
    lam = cos^2 + sin^2) and the scalars S[:, h, t] (_epilogue's names). R^T v = c v + s' v[other] (s' = -sin on the
    first half, +sin on the second: the transpose of the kernels' rotation)."""
    t, h = tl.program_id(0), tl.program_id(1)
    f = tl.arange(0, D)
    o = (f + D // 2) % D
    c = tl.load(CS + t * (D // 2) + f % (D // 2))
    s = tl.load(SN + t * (D // 2) + f % (D // 2)) * tl.where(f < D // 2, -1., 1.)
    row = t * H + h
    mb = M1 + row * D * D
    mt = (c[:, None] * (c[None, :] * tl.load(mb + f[:, None] * D + f[None, :]).to(tl.float32)
                        + s[None, :] * tl.load(mb + f[:, None] * D + o[None, :]).to(tl.float32))
          + s[:, None] * (c[None, :] * tl.load(mb + o[:, None] * D + f[None, :]).to(tl.float32)
                          + s[None, :] * tl.load(mb + o[:, None] * D + o[None, :]).to(tl.float32)))
    tl.store(Mt + row * D * D + f[:, None] * D + f[None, :], mt.to(tl.bfloat16))
    px, kr = tl.load(PX + row * D + f), tl.load(KR + h * KH + t * KT + f)
    ru = c * tl.load(U + row * D + f) + s * tl.load(U + row * D + o)
    rq = c * tl.load(Q + h * QH + t * QT + f) + s * tl.load(Q + h * QH + t * QT + o)
    lam = c * c + s * s
    vx = tl.sum(mt * px[None, :], 1) + tl.sum(mt * px[:, None], 0)
    v0 = tl.sum(mt * kr[None, :], 1) + tl.sum(mt * kr[:, None], 0)
    vb = V + (h * 6 * T + t) * D + f
    tl.store(vb, lam * kr)
    tl.store(vb + T * D, lam * px)
    tl.store(vb + 2 * T * D, ru)
    tl.store(vb + 3 * T * D, rq)
    tl.store(vb + 4 * T * D, vx)
    tl.store(vb + 5 * T * D, v0)
    sb, HT = S + h * T + t, H * T
    tl.store(sb, tl.sum(kr * lam * px))
    tl.store(sb + HT, tl.sum(px * lam * px))
    tl.store(sb + 2 * HT, tl.sum(px * ru))
    tl.store(sb + 3 * HT, tl.sum(kr * ru))
    tl.store(sb + 4 * HT, tl.sum(px * rq))
    tl.store(sb + 5 * HT, tl.sum(kr * rq))
    tl.store(sb + 6 * HT, tl.sum(px * vx) / 2)
    tl.store(sb + 7 * HT, tl.sum(kr * v0) / 2)
    tl.store(sb + 8 * HT, tl.sum(px * v0))
    tl.store(sb + 9 * HT, tl.sum(kr * lam * kr))
    tl.store(sb + 10 * HT, tl.load(AD + h * AH + t * AT))
    tl.store(sb + 11 * HT, tl.load(CD + h * CH + t * CT))


def _prepare(M1, U, qp, px, kraw, cos, sin, a_self, c_self):
    """prep_kernel's Mt [T, H, D * D] bf16, V [H, 6, T, D], S [12, H, 1, T]; px [T, H, D] contiguous; qp, kraw [H, T, D]
    (unit last stride); a_self, c_self [H, T]."""
    T, H, D = px.shape
    Mt = torch.empty((T, H, D * D), device=px.device, dtype=torch.bfloat16)
    V = px.new_empty((H, 6, T, D))
    S = px.new_empty((12, H, 1, T))
    prep_kernel[(T, H)](M1, U, qp, px, kraw, cos, sin, a_self, c_self, Mt, V, S, T, qp.stride(0), qp.stride(1),
                        kraw.stride(0), kraw.stride(1), *a_self.stride(), *c_self.stride(), H, D, num_warps=4)
    return Mt, V, S


@torch.compile(dynamic=False)
def _epilogue(G, Q, L, S, sq, step, dots, rsq, width: int):
    """[T, 2P] (+ then -) scores summed over heads, inputk_kernel's formulas on the expanded pieces: G [H, P, 6, T] =
    pr . (lam kr, lam px, R^T u, R^T q, Ms px, Ms kr), Q = rpr^T M1 rpr, L = |rpr|^2 [H, P, T]; S [12, H, 1, T]; sq,
    step [T]; dots [T, P] (x . rows); rsq [P] (|row|^2)."""
    D = width // G.shape[0]
    r0r, xr, ur, qr, vxr, v0r = G.unbind(2)
    r0x, xx, xu, r0u, xq, r0q, qx, q0, x0, r0sq, a_self, c_self = S.unbind(0)
    k0 = torch.rsqrt(r0sq / D + EPS)
    inv0 = torch.rsqrt(sq / width + EPS)
    rs = step * step * rsq[:, None]
    out = []
    for sign in (1., -1.):
        e = rs + 2 * sign * step * dots.T
        inv = torch.rsqrt((sq + rs + 2 * sign * step * dots.T) / width + EPS)
        dinv = -(e / width) * (inv0 * inv) * (inv0 * inv) / (inv0 + inv)
        b = sign * step * inv
        grow = (2 * (dinv * r0x + b * r0r) + dinv * dinv * xx + b * b * L + 2 * dinv * b * xr) / D
        k1 = torch.rsqrt(r0sq / D + grow + EPS)
        A, B, C = k1 * dinv, k1 * b, -grow * (k0 * k1) * (k0 * k1) / (k0 + k1)
        quad = A * A * qx + B * B * Q + C * C * q0 + A * B * vxr + A * C * x0 + B * C * v0r
        change = a_self * torch.expm1((A * xq + B * qr + C * r0q) * D ** -.5)
        out.append(((A * xu + B * ur + C * r0u) * D ** -.5 + quad / D + change / (1 + change) * c_self).sum(0))
    return torch.cat(out).T


def inputk_scores(xin, steps, rows, projected_x, projected_rows, raw, cos, sin, qp, A, C, tables):
    """fast_moments.inputk_scores' result [T, 2P] from GEMMs (same arguments)."""
    U, M1 = tables
    T, width = xin.shape
    H, _, D = raw.shape
    P = rows.shape[0]
    dots = xin @ rows.T  # as fast_moments forms them
    pr = projected_rows.reshape(P, H, D).transpose(0, 1)
    Mt, V, S = _prepare(M1.contiguous(), U.contiguous(), qp, projected_x.reshape(T, H, D).contiguous(), raw,
                        cos.contiguous(), sin.contiguous(), A.diagonal(0, 1, 2), C.diagonal(0, 1, 2))
    with ieee():
        G = torch.bmm(pr, V.flatten(1, 2).mT).view(H, P, 6, T)
        L = (projected_rows.square().reshape(P * H, 2, D // 2).sum(1) @ (cos * cos + sin * sin).T).view(P, H, T).transpose(0, 1)
    Q = torch.bmm(_outer(pr, pr), Mt.permute(1, 2, 0), out_dtype=torch.float32)
    return _epilogue(G, Q, L, S, xin.square().sum(-1), steps.reshape(-1), dots, rows.square().sum(-1), width)


# ------------------------------------------------------------------------------------------------------------ switch


def install(parts=None):
    """fast_moments' functions of these parts replaced ('tables', 'inputk'; default $S2_MOMENTGEMM), the others back."""
    global PARTS
    spec = '' if parts is None else ','.join(parts)
    spec = {'': '', '0': '', 'off': '', '1': ','.join(ALL), 'on': ','.join(ALL)}.get(spec.strip(), spec)
    want = set(filter(None, spec.split(',')))
    if want - set(ALL):
        raise ValueError(f's2_momentgemm: unknown parts {sorted(want - set(ALL))} (known: {ALL})')
    PARTS = tuple(p for p in ALL if p in want)
    new = dict(build_future_tables=build_future_tables, build_tables=build_tables, inputk_scores=inputk_scores)
    for name, plain in PLAIN.items():
        setattr(FM, name, new[name] if ('inputk' if name == 'inputk_scores' else 'tables') in PARTS else plain)
    return PARTS


install()


# ------------------------------------------------------------------------------------------------------------ checks


def rel(a, b):
    """|a - b| / |b| in float64, in pieces (the tables are 67M elements)."""
    num = den = 0.
    for x, y in zip(a.flatten().split(1 << 22), b.flatten().split(1 << 22)):
        x, y = x.double(), y.double()
        num, den = num + (x - y).square().sum().item(), den + y.square().sum().item()
    return float(f'{(num / max(den, 1e-300)) ** .5:.3e}')


# ======================================== module fast_s2
"""The stage-2 speed items on the shared readout stack: s2_head, s2_momentgemm, s2_skipslots, s2_plan (all items on)."""
import importlib

import fast_e3 as FE  # noqa: E402

ITEMS = dict(head=('s2_head', '6710573c2a57'), momentgemm=('s2_momentgemm', '0d068ccb275d'),
             skipslots=('s2_skipslots', 'beeb782d749b'), plan=('s2_plan', '7d03fcbce1c8'))  # module, sha1 as checked


def parse(spec):
    spec = spec.strip()
    on = tuple(ITEMS) if spec in ('', '1', 'all') else () if spec == '0' else tuple(filter(None, spec.split(',')))
    if set(on) - set(ITEMS):
        raise ValueError(f'fast_s2: unknown items {sorted(set(on) - set(ITEMS))} (known: {list(ITEMS)})')
    return on


def install(on=None, shared=None):
    """The items in `on` (default $FAST_S2) on, the others off; then fast_e3's patch (shared: its default, FAST_E3_SHARED),
    with skipslots' mixin on its shared Methods if on."""
    on = parse('1') if on is None else tuple(on)  # all items
    mod = {k: importlib.import_module(m) for k, (m, _) in ITEMS.items()}
    mod['momentgemm'].install(mod['momentgemm'].ALL if 'momentgemm' in on else ())
    mod['plan'].install('1' if 'plan' in on else '0')
    mod['head'].install('head' in on)
    mod['skipslots'].ON = 'skipslots' in on
    mod['skipslots'].install(shared) if 'skipslots' in on else FE.install(shared)
    return on


# ------------------------------------------------------------------------------------------------------------ checks


# ======================================== module head_edit
"""Early head transplants. After the update of every 25th step in [50, 450] (Boost and Cut alike), on the update's own
rows: P = the first half proposes and ranks, Q = the second half confirms. Dormancy per head = the loss with the head's
output replaced by its token mean; for the blocks with the most dormant head, candidates (clone a strong head with a
RoPE shift, split it, or transfer a head from another block) are scored by one forward on P each, and the best is kept
iff Q's loss drops too. Weights and optimizer moments move together. Every decision is a forward loss comparison.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

import fast_s2 as S2  # noqa: E402
import shared_push as SP  # noqa: E402
FE, R = S2.FE, S2.FE.R

H, D = 8, 64
THETA = 1.0 / 10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
STEP = {}  # the estimators' step under the patch


@dataclass
class Settings(SP.Settings):  # shared_push's fields and the head edits
    head_edit: bool = False
    head_edit_start: int = 50
    head_edit_every: int = 25
    head_edit_stop: int = 600
    head_edit_per_block: int = 2
    head_edit_nxfer: int = 5
    head_edit_betas: tuple = (0.5, 1.0, 1.5)
    head_edit_deltas: tuple = (1, 2, 3)
    head_edit_protect: int = 200

    def __post_init__(self):
        super().__post_init__()
        self.head_edit_betas, self.head_edit_deltas = tuple(self.head_edit_betas), tuple(self.head_edit_deltas)
        if type(self.head_edit) is not bool or self.head_edit and (
                self.boost_batch != self.cut_batch or self.cut_batch % 2 or self.head_edit_nxfer < 0
                or min(self.head_edit_start, self.head_edit_every, self.head_edit_per_block) < 1
                or not (self.head_edit_betas and self.head_edit_deltas)):
            raise ValueError('head_edit: true / false; on needs boost_batch = cut_batch (even), start / every / per_block '
                             '>= 1, nxfer >= 0, some betas and deltas')


def asdict(settings):  # off: shared_push's recipe dict
    state = SP.asdict(settings)
    if not getattr(settings, 'head_edit', False):
        for key in [k for k in state if k.startswith('head_edit')]:
            state.pop(key)
    return state


def rope_shift(w, delta):
    """Rows of one head's c_k (or their moments): the rotated key at position t is the original key at t + delta (exact:
    RoPE and QK-norm commute with it)."""
    c, s = (f(delta * THETA).to(w.device)[:, None] for f in (torch.cos, torch.sin))
    a, b = w[:32].double(), w[32:].double()
    return torch.cat([c * a + s * b, -s * a + c * b]).to(w.dtype)


class Editor:
    """The head edits of one model (kept on it, so the protect window crosses the Boost -> Cut handoff)."""

    def __init__(self, model, s):
        self.model, self.s = model, s
        self.ve = {int(k): v for k, v in model.value_embeds.items()}
        self.edited, self.target, self.means, self.record = {}, None, {}, False

    def hook(self, i):  # c_proj's input = the heads' outputs side by side
        def fn(mod, args):
            if self.record:
                self.means[i] = args[0].float().mean((0, 1))
            if self.target is None or self.target[0] != i:
                return None
            y, sl = args[0].clone(), slice(self.target[1] * D, (self.target[1] + 1) * D)
            y[..., sl] = self.means[i][sl].to(y.dtype)
            return (y,)
        return fn

    def loss(self, x, y):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model(x, y)

    def slots(self, b, h):  # (name, weight, dim, slice) of head h of block b
        attn, sl = self.model.transformer.h[b].attn, slice(h * D, (h + 1) * D)
        out = [('q', attn.c_q.weight, 0, sl), ('k', attn.c_k.weight, 0, sl), ('v', attn.c_v.weight, 0, sl),
               ('o', attn.c_proj.weight, 1, sl)]
        if b in self.ve:
            out += [('ve', self.ve[b].weight, 1, sl), ('gate', attn.ve_gate.weight, 0, slice(h, h + 1))]
        return out

    @staticmethod
    def states(opts, p, dim, sl):  # (key, view) of p's optimizer tensors laid out like p along dim
        return [(k, t.narrow(dim, sl.start, sl.stop - sl.start)) for o in opts for k, t in o.state.get(p, {}).items()
                if torch.is_tensor(t) and t.dim() == p.dim() and t.shape[dim] == p.shape[dim]]

    def get(self, opts, b, h):
        return {n: (p.narrow(dim, sl.start, sl.stop - sl.start).clone(),
                    {k: v.clone() for k, v in self.states(opts, p, dim, sl)}) for n, p, dim, sl in self.slots(b, h)}

    def put(self, opts, b, h, head, states=True):
        for n, p, dim, sl in self.slots(b, h):
            if n in head:
                w, st = head[n]
                p.narrow(dim, sl.start, sl.stop - sl.start).copy_(w.to(p.dtype))
                for k, v in self.states(opts, p, dim, sl) if states else ():
                    if k in st and st[k].shape == v.shape:
                        v.copy_(st[k].to(v.dtype))

    @staticmethod
    def transform(head, delta=0, beta=1.0, drop_ve=False):
        """A copy of a head (weights, moments): keys RoPE-shifted by delta, output x beta, value-embedding columns off."""
        out = {}
        for n, (w, st) in head.items():
            if n == 'k' and delta:
                w, st = rope_shift(w, delta), {k: rope_shift(t, delta) if t.shape == w.shape else t for k, t in st.items()}
            if n == 'o' and beta != 1.0:
                w, st = w * beta, {k: t * beta if k != 'energy' else t for k, t in st.items()}
            out[n] = (w, st)
        if drop_ve and 've' in out:
            out['ve'] = (out['ve'][0] * 0, {k: t * 0 for k, t in out['ve'][1].items()})
        return out

    def candidates(self, opts, b, orig_d, orig_s, ranked):  # [(tag, new s or None, new d)]
        betas, deltas, T = self.s.head_edit_betas, self.s.head_edit_deltas, self.transform
        out = [(f'clone_d{dl}_b{bt}', None, T(orig_s, dl, bt)) for dl in deltas for bt in betas]
        half = T(orig_s, 0, 0.5)
        out += [(f'split_d{dl}', half, T(orig_s, dl, 0.5)) for dl in deltas]
        for _, b2, h2 in [r for r in ranked if r[1] != b][:self.s.head_edit_nxfer]:
            src = self.get(opts, b2, h2)
            if b in self.ve and b2 not in self.ve:
                src['ve'], src['gate'] = orig_d['ve'], orig_d['gate']
                src = T(src, drop_ve=True)
            elif b2 in self.ve and b not in self.ve:
                del src['ve'], src['gate']
            out += [(f'xfer_{b2}.{h2}_d0_b{bt}', None, T(src, 0, bt)) for bt in betas]
        return out

    @torch.no_grad()
    def __call__(self, opts, x, y, step, rank=0, world=1):
        """One edit step on the update's rows x, y (the same on every rank) -> its log: (step, block, d, s, tag,
        P change, Q change, kept) per tried slot."""
        cfg, half = self.s, len(x) // 2
        P, Q = (x[:half], y[:half]), (x[half:], y[half:])

        def shared(n, fn):  # fn(i) for i < n, each on one rank; every value on every rank
            out = torch.zeros(n, dtype=torch.float64, device=x.device)
            for i in range(rank, n, world):
                out[i] = fn(i)
            if world > 1:
                dist.all_reduce(out)
            return out.tolist()

        def first(i):  # P's loss (rank 0's), Q's loss, then the 64 mean-ablations on P
            if i < 2:
                return base if i == 0 else self.loss(*Q)
            self.target = divmod(i - 2, H)
            loss, self.target = self.loss(*P), None
            return loss

        hooks = [blk.attn.c_proj.register_forward_pre_hook(self.hook(i)) for i, blk in enumerate(self.model.transformer.h)]
        try:
            self.record = True
            base = self.loss(*P)  # on every rank: its token means feed its ablations
            self.record = False
            v = shared(2 + H * H, first)
            base_p, base_q = v[:2]
            abl = torch.tensor([u - base_p for u in v[2:]]).view(H, H)  # float32, as the simulator keeps it
            ranked = sorted(((float(abl[b, h]), b, h) for b in range(H) for h in range(H)), reverse=True)
            tried, log = set(), []
            for b in sorted(range(H), key=lambda b: float(abl[b].min())) * cfg.head_edit_per_block:
                free = [h for h in range(H) if step - self.edited.get((b, h), -10 ** 9) > cfg.head_edit_protect
                        and (b, h) not in tried]
                if len(free) < 2:
                    continue
                d = min(free, key=lambda h: float(abl[b, h]))
                tried.add((b, d))
                s = max(range(H), key=lambda h: float(abl[b, h]))
                if s == d:
                    continue
                orig_d, orig_s = self.get(opts, b, d), self.get(opts, b, s)
                cands = self.candidates(opts, b, orig_d, orig_s, ranked)

                def trial(i):
                    _, new_s, new_d = cands[i]
                    if new_s is not None:
                        self.put(opts, b, s, new_s, states=False)
                    self.put(opts, b, d, new_d, states=False)
                    loss = self.loss(*P)
                    self.put(opts, b, d, orig_d, states=False)
                    if new_s is not None:
                        self.put(opts, b, s, orig_s, states=False)
                    return loss
                lp = shared(len(cands), trial)
                i = min(range(len(cands)), key=lp.__getitem__)
                if not lp[i] < base_p:
                    log.append((step, b, d, s, 'none', 0.0, 0.0, False))
                    continue
                tag, new_s, new_d = cands[i]
                if new_s is not None:
                    self.put(opts, b, s, new_s, states=False)
                self.put(opts, b, d, new_d, states=False)
                lq = shared(1, lambda _: self.loss(*Q))[0]
                keep = lq < base_q
                log.append((step, b, d, s, tag, round(lp[i] - base_p, 5), round(lq - base_q, 5), keep))
                if keep:  # the weights again, now with their moments
                    if new_s is not None:
                        self.put(opts, b, s, new_s)
                    self.put(opts, b, d, new_d)
                    self.edited[(b, d)] = step
                    base_p, base_q = lp[i], lq
                else:
                    self.put(opts, b, d, orig_d, states=False)
                    if new_s is not None:
                        self.put(opts, b, s, orig_s, states=False)
        finally:
            for hk in hooks:
                hk.remove()
            self.target, self.record = None, False
        return log


def rows(est, x, y):
    """The update's rows on every rank (a rank holds those of its sequence, rank // split)."""
    if est.world == 1:
        return x, y
    per = x.shape[0]
    full = torch.zeros(2, est.world // est.split * per, *x.shape[1:], dtype=x.dtype, device=x.device)
    if est.part == 0:
        at = est.rank // est.split * per
        full[:, at:at + per] = torch.stack([x, y])
    dist.all_reduce(full)
    return full[0], full[1]


def edit(est, x, y, t, optimizer):
    start = time.perf_counter()
    ed = getattr(est.model, '_head_edit', None)
    if ed is None:
        ed = est.model._head_edit = Editor(est.model, est.s)
    log = ed(optimizer.optimizers, *rows(est, x, y), t, est.rank, est.world)
    torch.cuda.synchronize()
    if est.rank == 0:
        kept = [e for e in log if e[-1]]
        kinds = {k: sum(e[4].startswith(k) for e in kept) for k in ('xfer', 'clone', 'split')}
        line = dict(step=t, phase='cut' if isinstance(est, R.Cut) else 'boost', seconds=round(time.perf_counter() - start, 2),
                    tried=len(log), kept=len(kept), **kinds)
        print('head_edit', json.dumps(line), flush=True)
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument('--output', type=Path)
        out = ap.parse_known_args(sys.argv[1:])[0].output
        if out is not None and out.is_dir():
            with open(out / 'head_edit.jsonl', 'a') as f:
                f.write(json.dumps(dict(line, edits=log)) + '\n')


def step(self, x, y, optimizer):
    loss = STEP['step'](self, x, y, optimizer)
    s = self.s
    if getattr(s, 'head_edit', False):
        t = optimizer.optimizers[0].t  # ANVIL's clock after the update: this update's number
        if s.head_edit_start <= t <= s.head_edit_stop and t % s.head_edit_every == 0:
            edit(self, x, y, t, optimizer)
    return loss


def install(on=None):
    """fast_s2's install (items: $FAST_S2, on fast_e3 + shared_push), then this file's Settings / recipe dict and the
    step patch (idempotent). The speed stack goes on after (it touches none of these)."""
    if R.Settings is Settings:
        R.Settings, R.asdict = SP.Settings, SP.asdict
    on = S2.install(on, shared=True)
    if (R.Settings, R.asdict) != (SP.Settings, SP.asdict) or 'step' in vars(R.Cut):
        raise RuntimeError("head_edit: another wrapper replaced Settings, asdict or the estimators' step")
    R.Settings, R.asdict = Settings, asdict
    if 'step' not in STEP:
        STEP['step'], R.Boost.step = R.Boost.step, step
    return on


# ======================================== module mom_distill
"""Momentum distillation. The teacher is an EMA of the model's weights (beta 0.99, from the first update). From 10% of
the epoch (Cut), each update runs one teacher forward on its rows and every per-token loss the estimate measures
becomes the cross-entropy against (1 - lam) one-hot(y) + lam softmax(teacher) (lam 0.3): the head's coordinate
differences, the top stream's head probe and the chain head's readout.
"""
import argparse
import copy
from dataclasses import dataclass
import functools
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

import fast_s2 as S2  # noqa: E402
import shared_push as SP  # noqa: E402
FE, R, FC = S2.FE, S2.FE.R, SP.FC

EPS = R.EPS
LOG_EVERY = 200
REQUIRED = ('head_error', 'probe', 'head')  # the sites every distilled estimate must run
PLAIN = dict(head_error=R.Capture.head_error, probe=R.Boost.probe, cycle=R.cycle_derivatives)
STEP = {}  # the estimators' step under the patch
ACTIVE = None  # during a distilled estimate: dict(est, q [b, t, V], y, lam, calls)


@dataclass
class Settings(SP.Settings):  # shared_push's fields and the distillation
    distill: bool = False
    distill_lam: float = 0.3
    distill_beta: float = 0.99
    distill_start: float = 0.1
    distill_until: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if type(self.distill) is not bool or self.distill and not (
                0 <= self.distill_lam <= 1 and 0 < self.distill_beta < 1 and 0 <= self.distill_start < self.distill_until):
            raise ValueError('distill: true / false; on needs 0 <= lam <= 1 (0: the target unmixed), 0 < beta < 1, '
                             '0 <= start < until')


def asdict(settings):  # off: shared_push's recipe dict
    state = SP.asdict(settings)
    if not getattr(settings, 'distill', False):
        for key in [k for k in state if k.startswith('distill')]:
            state.pop(key)
    return state


class Teacher:
    """The EMA of the model's weights and its forward (kept on the model, so it crosses the Boost -> Cut handoff)."""

    def __init__(self, model, beta):
        self.beta, self.net, self.seen = beta, copy.deepcopy(model), 0
        for module in self.net.modules():  # the estimators' capture hooks stay on the model
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()

    @torch.no_grad()
    def update(self, model):
        for t, p in zip(self.net.parameters(), model.parameters()):
            t.lerp_(p.to(t.dtype), 1 - self.beta)

    @torch.no_grad()
    def probs(self, x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            return torch.softmax(self.net(x).float(), -1)


# ------------------------------------------------------------------------------------------- the mixed-target losses


@torch.compile
def coordinate_differences(raw, y, sigma, q, lam):
    """The record's coordinate_differences with the mixed target: logit v's target term is ((1 - lam) 1(v = y) + lam q_v)
    times its capped change (c(raw_v + sigma) - c(raw_v - sigma)). raw, q [..., V]; y [...]."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(1 - lam) * (up - down).gather(-1, target)) - lam * q * (up - down)
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, idx, steps, sq, dots, dir_sq, width, Y, Q, lam):
    """The record's cycle_derivatives with the mixed target: the label's capped logit x (1 - lam) and lam sum_v q_v c(z_v)
    from the capped logits the log-sum-exp reads, clean and moved; idx: the rows' tokens (the probe slices its y by rows),
    Y / Q every token's label / q."""
    y, q = Y[idx], Q[idx]
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    c0 = 15 * torch.tanh(base * inv0 / 15)
    clean = torch.exp(c0).sum(-1, keepdim=True).log() - (1 - lam) * 15 * torch.tanh(base_y * inv0 / 15) - lam * (
        q * c0).sum(-1, keepdim=True)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    c = 15 * torch.tanh(z / 15)
    moved = torch.exp(c).sum(-1).log() - (1 - lam) * 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15) - lam * (
        q.unsqueeze(1) * c).sum(-1)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


def head_rows(xL, a):
    """Per token of the chain's top stream: W^T (q c') and (q c') . base, with base = xL W^T and c' = 1 - t0^2 at the
    clean capped logits the head reads."""
    model = a['est'].model
    weight = model.lm_head.weight.float()[:model.config.vocab_size]
    base = xL @ weight.T
    t0 = torch.tanh(base * torch.rsqrt(xL.square().sum(-1, keepdim=True) / xL.shape[-1] + EPS) / 15)
    w = a['q'].view(-1, base.shape[-1]) * (1 - t0 * t0)
    return w @ weight, (w * base).sum(-1)


# ------------------------------------------------------------------------------------------------------ the patches


def head_error(self, sigma=0.1):  # Capture.head_error under the patch
    a = ACTIVE
    if a is None or a['est'].capture is not self:
        return PLAIN['head_error'](self, sigma)
    a['calls']['head_error'] += 1
    V = self.model.config.vocab_size
    error = coordinate_differences(self.outputs['lm_head'][..., :V], self.y, sigma, a['q'], a['lam'])
    return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - V))


def probe(self, x, y, mix=None, probed=None):  # Boost.probe under the patch (in Cut: the top stream's g_top)
    a = ACTIVE
    if a is None or a['est'] is not self or mix is not None or probed is not None:
        return PLAIN['probe'](self, x, y, mix, probed)
    q = a['q'].view(-1, a['q'].shape[-1])
    if x.shape[0] != len(q) or y.shape != (len(q),):
        raise RuntimeError(f'mom_distill: a probe on {tuple(x.shape)} rows, the distilled batch has {len(q)} tokens')
    a['calls']['probe'] += 1
    R.cycle_derivatives = functools.partial(cycle_derivatives, Y=y, Q=q, lam=a['lam'])
    try:  # the record's probe as is (draws, chunks, decode); its rows carry their token index in place of the label
        return PLAIN['probe'](self, x, torch.arange(len(y), device=y.device), mix, probed)
    finally:
        R.cycle_derivatives = PLAIN['cycle']


def wrap_head(inner):
    """fast_chain.head_fast under the patch, around the installed head (s2_head's or fast_chain's): both leave the row
    stats (inv', dinv, dc_y, bound) in ctx['scratch32'] and read the chain's top stream as ctx['xL'], a view of its rows."""
    @functools.wraps(inner)  # pins over fast_chain's source still see fast_chain's code (as s2_head's wrapper)
    def head_fast(delta, ctx):
        out = inner(delta, ctx)
        a = ACTIVE
        if a is None:
            return out
        full, (m, rows, W) = ctx['xL']._base, delta.shape
        if full is None or full.shape[0] != a['q'][..., 0].numel():
            raise RuntimeError("mom_distill: the chain head's xL is not a view of the distilled batch's top stream")
        a['calls']['head'] += 1
        if a.get('xL') is not full:  # the first head call of this estimate's chain
            a['xL'], a['rows'] = full, head_rows(full, a)
        t0 = (ctx['xL'].storage_offset() - full.storage_offset()) // W
        u, ub = (z[t0:t0 + m] for z in a['rows'])
        st = ctx['scratch32'][:m * rows * 4].view(m, rows, 4)
        dq = st[..., 1] * ub[:, None] + st[..., 0] * torch.bmm(delta, u.unsqueeze(-1)).squeeze(-1)
        return out + a['lam'] * (st[..., 2] - dq)
    head_fast.mom_distill = inner
    return head_fast


@torch.no_grad()
def report(est, a, t, loss):
    """Losses on this update's rows, averaged over the ranks (every rank calls it at the same updates)."""
    y, q, ok = a['y'], a['q'], (a['y'] >= 0).float()
    V = q.shape[-1]
    logp = torch.log_softmax(15 * torch.tanh(est.capture.outputs['lm_head'][..., :V].float() / 15), -1)
    vals = torch.stack([(-q.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log() * ok).sum(),
                        (-(q * logp).sum(-1) * ok).sum(), ok.sum()]).double()
    if est.world > 1:
        dist.all_reduce(vals)
    teacher_ce, kd = (vals[:2] / vals[2]).tolist()
    lam = a['lam']
    line = dict(step=t, progress=round(est.progress, 4), teacher_ce=round(teacher_ce, 4), student_ce=round(loss, 4),
                kd=round(kd, 4), mixed=round((1 - lam) * loss + lam * kd, 4))
    if est.rank == 0:
        print('mom_distill', json.dumps(line), flush=True)
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument('--output', type=Path)
        out = ap.parse_known_args(sys.argv[1:])[0].output
        if out is not None and out.is_dir():
            with open(out / 'mom_distill.jsonl', 'a') as f:
                f.write(json.dumps(line) + '\n')


def step(self, x, y, optimizer):
    s = self.s
    if not getattr(s, 'distill', False):
        return STEP['step'](self, x, y, optimizer)
    global ACTIVE
    teacher = getattr(self.model, '_distill', None)
    progress = getattr(self, 'progress', None)  # None: the recipe's compile step on random tokens
    if teacher is None:  # the model's first update: the copy is its weights before it
        if progress not in (None, 0.0):  # the teacher is not in the checkpoint: a resumed run would restart it mid-run
            raise RuntimeError('mom_distill: the teacher starts at the first update; a distilled run cannot resume')
        teacher = self.model._distill = Teacher(self.model, s.distill_beta)
    if isinstance(self, R.Cut) and (progress is None or s.distill_start <= progress < s.distill_until):
        ACTIVE = dict(est=self, q=teacher.probs(x), y=y, lam=s.distill_lam, calls=dict.fromkeys(REQUIRED, 0))
    try:
        loss = STEP['step'](self, x, y, optimizer)
        if ACTIVE is not None:
            if not all(ACTIVE['calls'].values()):
                raise RuntimeError(f"mom_distill: a distilled estimate ran a site without the mixed target {ACTIVE['calls']}")
            if progress is not None:
                t = optimizer.optimizers[0].t  # ANVIL's clock after the update: this update's number
                if teacher.seen == 0 or t % LOG_EVERY == 0:
                    report(self, ACTIVE, t, loss)
                teacher.seen += 1
    finally:
        ACTIVE = None
    teacher.update(self.model)
    return loss


def install(on=None):
    """fast_s2's install (items: $FAST_S2, on fast_e3 + shared_push), then this file's Settings / recipe dict, the step
    patch and the three loss sites (idempotent). The speed stack goes on after (it touches none of these)."""
    if R.Settings is Settings:
        R.Settings, R.asdict = SP.Settings, SP.asdict
    on = S2.install(on, shared=True)
    if (R.Settings, R.asdict) != (SP.Settings, SP.asdict) or 'step' in vars(R.Cut) or 'probe' in vars(R.Cut) or \
            R.Capture.head_error not in (PLAIN['head_error'], head_error) or R.Boost.probe not in (PLAIN['probe'], probe):
        raise RuntimeError("mom_distill: another wrapper replaced Settings, asdict, the estimators' step or a loss site")
    R.Settings, R.asdict = Settings, asdict
    if 'step' not in STEP:
        STEP['step'], R.Boost.step = R.Boost.step, step
    R.Capture.head_error, R.Boost.probe = head_error, probe
    if not hasattr(FC.head_fast, 'mom_distill'):  # s2_head's install (inside fast_s2's) sets FC.head_fast again
        FC.head_fast = wrap_head(FC.head_fast)
    return on


# ======================================== module combo_he_md
"""Head transplants + momentum distillation in one Settings / recipe dict; the estimators' step chained
mom_distill.step -> head_edit.step -> the stack's step.
"""
from dataclasses import dataclass

import head_edit as HE  # noqa: E402
import mom_distill as MD  # noqa: E402
S2, SP, FE, R, FC = MD.S2, MD.SP, MD.FE, MD.R, MD.FC


@dataclass
class Settings(HE.Settings, MD.Settings):  # shared_push's fields, the head edits and the distillation (both checks run)
    pass


def asdict(settings):  # off: shared_push's recipe dict; each layer's keys only when it is on
    state = SP.asdict(settings)
    for prefix in ('head_edit', 'distill'):
        if not getattr(settings, prefix, False):
            for key in [k for k in state if k.startswith(prefix)]:
                state.pop(key)
    return state


def install(on=None):
    """fast_s2's install (items: $FAST_S2, on fast_e3 + shared_push), then both layers' patches (idempotent; refuses a
    process where either layer was installed alone). The speed stack goes on after (it touches none of these)."""
    if (HE.STEP or MD.STEP) and (R.Boost.step, MD.STEP.get('step'), 'step' in HE.STEP) != (MD.step, HE.step, True):
        raise RuntimeError('combo_he_md: head_edit.py or mom_distill.py was installed alone in this process')
    if R.Settings is Settings:
        R.Settings, R.asdict = SP.Settings, SP.asdict
    on = S2.install(on, shared=True)
    if (R.Settings, R.asdict) != (SP.Settings, SP.asdict) or 'step' in vars(R.Cut) or 'probe' in vars(R.Cut) or \
            R.Capture.head_error not in (MD.PLAIN['head_error'], MD.head_error) or \
            R.Boost.probe not in (MD.PLAIN['probe'], MD.probe):
        raise RuntimeError("combo_he_md: another wrapper replaced Settings, asdict, the estimators' step or a loss site")
    R.Settings, R.asdict = Settings, asdict
    if not HE.STEP:
        HE.STEP['step'], MD.STEP['step'], R.Boost.step = R.Boost.step, HE.step, MD.step
    R.Capture.head_error, R.Boost.probe = MD.head_error, MD.probe
    if not hasattr(FC.head_fast, 'mom_distill'):  # s2_head's install (inside fast_s2's) sets FC.head_fast again
        FC.head_fast = MD.wrap_head(FC.head_fast)
    return on


# ======================================== module twin_rows
"""Next-token twin rows (the injection kernel): token t's moved key / value at a block give receiver t + 1's exact
single-column attention change, which a twin row in t + 1's slot carries forward.
"""
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

import fast_s2 as S2  # noqa: E402
import shared_push as SP  # noqa: E402
FE = S2.FE
R, FC, RO = FE.R, SP.FC, SP.RO

STATS = dict(calls=0, every=200)
IMPL = 'kernel'  # 'torch': the injection in torch arithmetic (the reference, checks)


@dataclass
class Settings(SP.Settings):  # shared_push's fields and the twin rows
    twin: bool = False
    twin_cap1: int = 6
    twin_cap2: int = 7

    def __post_init__(self):
        super().__post_init__()
        if type(self.twin) is not bool or self.twin and not (getattr(self, 'readout_shared', False) and
                                                             0 <= self.twin_cap1 < self.twin_cap2 <= 7):
            raise ValueError('twin: true / false; true needs readout_shared and 0 <= twin_cap1 < twin_cap2 <= 7')


def asdict(settings):  # off: shared_push's recipe dict
    state = SP.asdict(settings)
    if not getattr(settings, 'twin', False):
        for key in [k for k in state if k.startswith('twin')]:
            state.pop(key)
    return state


@triton.jit
def twin_kernel(UP, ST, R0, VC, GZ, VE, CS, SN, QD, AS, VT, OT, Out, Start, T, n, OUT_ROWS,
                W: tl.constexpr, H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, BP: tl.constexpr,
                NC: tl.constexpr):
    """readout's staged kernel's dk / dv of each row (token Start + t; UP / ST as it reads them) and, in place of its
    s = t update, receiver t + 1's: x = dk . q_{t+1} / sqrt(D), a = AS[t] = A[t+1, t], out = gain (v_t - o_{t+1}) +
    vgain dv per head -> Out [m, OUT_ROWS, W] bf16 (zero for the sequence's last token)."""
    pb, t = tl.program_id(0), tl.program_id(1)
    tok = Start + t
    nxt = tl.minimum(tok + 1, T - 1)
    p = pb * BP + tl.arange(0, BP)
    pm = p < n
    f = tl.arange(0, D)
    other = (f + D // 2) % D
    half = f % (D // 2)
    cfull = tl.load(CS + tok * (D // 2) + half)
    ssig = tl.load(SN + tok * (D // 2) + half) * tl.where(f < D // 2, 1., -1.)
    inv = tl.load(ST + (t * n + p) * 2, pm, 1.)
    coef = tl.load(ST + (t * n + p) * 2 + 1, pm, 0.)
    urow = (t * n + p).to(tl.int64) * NC
    g16 = tl.arange(0, 16)
    if GATED:
        ug = tl.load(UP + urow[:, None] + 2 * W + g16[None, :], pm[:, None], 0.).to(tl.float32)
        gz = tl.load(GZ + tok * H + g16, g16 < H, 0.)
        gch = 2 * (tl.sigmoid(gz[None, :] + inv[:, None] * ug + coef[:, None] * gz[None, :]) - tl.sigmoid(gz)[None, :])
    for h in range(H):
        uk = tl.load(UP + urow[:, None] + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        uks = tl.load(UP + urow[:, None] + h * D + other[None, :], pm[:, None], 0.).to(tl.float32)
        uv = tl.load(UP + urow[:, None] + W + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        row = tok * H + h
        r0 = tl.load(R0 + row * D + f)
        dr = inv[:, None] * (uk * cfull[None, :] + uks * ssig[None, :]) + coef[:, None] * r0[None, :]
        dv = inv[:, None] * uv + coef[:, None] * tl.load(VC + tok * W + h * D + f)[None, :]
        if GATED:
            gc = tl.sum(gch * (g16 == h).to(tl.float32)[None, :], 1)
            dv = dv + gc[:, None] * tl.load(VE + row * D + f)[None, :]
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        nrow = nxt * H + h
        x = tl.sum(dk * tl.load(QD + nrow * D + f)[None, :], 1) * (D ** -.5)
        a = tl.load(AS + row)
        e = libdevice.expm1(x)
        gain = a * e / (1. + a * e)
        vgain = (a + a * e) / (1. + a * e)
        vo = tl.load(VT + row * D + f) - tl.load(OT + nrow * D + f)
        tl.store(Out + (t * OUT_ROWS + p[:, None]).to(tl.int64) * W + h * D + f[None, :],
                 (gain[:, None] * vo[None, :] + vgain[:, None] * dv).to(tl.bfloat16), pm[:, None])


def sub_diagonal(c, T):
    """AS [T, H]: receiver t+1's clean attention to key t (0 for the last token), cached in the block context."""
    if 'AS' not in c:
        a = c['a']  # [H, T, T]
        c['AS'] = F.pad(a.diagonal(-1, 1, 2), (0, 1)).T.contiguous()
    return c['AS']


def inject_kernel(c, up, st, n, P, cs, sn, w_proj_t, t0, T, out):
    """twin_kernel on the first P of n scored rows per token (up [m * n, NC] bf16, st [m * n * 2]) -> [m, P, W] fp32
    (out: a bf16 [m * P * W] scratch)."""
    m = up.shape[0] // n
    H, D = c['QD'].shape[1], c['QD'].shape[2]
    W, nc = H * D, up.shape[1]
    gated = c['gate'] is not None
    o = out[:m * P * W].view(m, P, W)
    twin_kernel[(triton.cdiv(P, 64), m)](up, st, c['R0'], c['VC'], c['GZ'] if gated else o, c['ve'] if gated else o,
                                         cs, sn, c['QD'], sub_diagonal(c, T), c['VT'], c['OT'], o, t0, T, n, P, W, H, D,
                                         gated, 64, nc, num_warps=4, num_stages=2)
    return torch.mm(o.view(m * P, W), w_proj_t, out_dtype=torch.float32).view(m, P, W)


def inject(c, up, st, cs, sn, w_proj, t0, T):
    """Receiver t+1's exact attention change at this block from the moved key / value of token t, through c_proj.
    c: block b's clean context (shared_push.Shared.clean); up [m, P, NC] the scorer's projection of the raw moves (d W_k^T |
    d W_v^T | d[:32] W_g^T); st [m, P, 2] = (inv', inv' / inv0 - 1); cs / sn [T, D/2]; w_proj [W, W]; tokens t0 .. t0+m-1.
    Returns [m, P, W] fp32 (token t's row = the change of t + 1's mid stream; zero for the sequence's last token)."""
    m, P, _ = up.shape
    H, D = c['QD'].shape[1], c['QD'].shape[2]
    W = H * D
    tok = torch.arange(t0, t0 + m, device=up.device)
    nxt = (tok + 1).clamp(max=T - 1)
    up = up.float()
    inv, coef = st[..., 0, None, None], st[..., 1, None, None]  # [m, P, 1, 1]
    cfull = torch.cat([cs[tok], cs[tok]], -1)[:, None, None]  # [m, 1, 1, D] (the staged kernel's cfull / ssig)
    ssig = torch.cat([sn[tok], -sn[tok]], -1)[:, None, None]
    uk = up[..., :W].view(m, P, H, D)
    uks = torch.cat([uk[..., D // 2:], uk[..., :D // 2]], -1)  # uk at (f + D/2) % D
    uv = up[..., W:2 * W].view(m, P, H, D)
    r0 = c['R0'][tok][:, None]  # [m, 1, H, D]
    dr = inv * (uk * cfull + uks * ssig) + coef * r0
    dv = inv * uv + coef * c['VC'][tok].view(m, 1, H, D)
    if c['gate'] is not None:
        gz = c['GZ'][tok][:, None]  # [m, 1, H]
        ug = up[..., 2 * W:2 * W + H]
        gch = 2 * (torch.sigmoid(gz + st[..., 0, None] * ug + st[..., 1, None] * gz) - torch.sigmoid(gz))
        dv = dv + gch[..., None] * c['ve'][tok][:, None]
    square = r0.square().sum(-1)  # [m, 1, H]
    grow = (2 * (r0 * dr).sum(-1) + dr.square().sum(-1)) / D
    inverse = torch.rsqrt(square / D + 1.1920928955078125e-07)
    moved = torch.rsqrt(square / D + grow + 1.1920928955078125e-07)
    dk = dr * moved[..., None] - r0 * (grow * (inverse * moved) ** 2 / (inverse + moved))[..., None]
    x = (dk * c['QD'][nxt][:, None]).sum(-1) * D ** -0.5  # [m, P, H]
    a = c['a'][:, nxt, tok].T[:, None]  # [m, 1, H]: receiver t+1's clean attention to key t
    e = torch.expm1(x)
    gain, vgain = a * e / (1 + a * e), (a + a * e) / (1 + a * e)
    vo = (c['VT'][tok] - c['OT'][nxt])[:, None]  # [m, 1, H, D]
    out = gain[..., None] * vo + vgain[..., None] * dv
    out = out * (tok + 1 < T).view(m, 1, 1, 1)
    return out.reshape(m * P, W).matmul(w_proj.T).view(m, P, W)


class Twin:
    """First in the MRO of fast_e3's shared Methods: the chain with the next-token twin rows (module docstring); the
    pass, finish and every other piece are the stack's."""

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        e, s = self.e, self.e.s
        if not getattr(s, 'twin', False):
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'twin_rows: the fast_chain kernels do not cover this setting ({reason})')
        model, d, dev = e.model, e.capture, e.device
        L, W, T, n = model.config.n_layer, e.width, d.length, emb.shape[0]
        cap1, cap2 = min(s.twin_cap1, L - 2), min(s.twin_cap2, L - 1)
        self.sp_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.sp_events[0].record()
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P, scalars, h = hi - lo, e.part == 0, s.scalar_step
        self.chain_args = (emb, x0, mids, outs, routes, y)
        self.bases = [e.basis() for _ in range(L + 1)]  # the record's draws in its order: emb, mid0, mid1, ...
        bases = [B[lo:hi].contiguous() for B in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb, *mids]]
        self.prepare(mids, routes)
        head = model.lm_head.weight.float()[:model.config.vocab_size]
        xL = outs[-1].float().contiguous()
        base, target_w, weight_t = xL @ head.T, head[y.clamp(min=0)], head.T.to(torch.bfloat16).contiguous()
        if FC.STATE['exact_blocks'] is None:
            FC.STATE['exact_blocks'] = torch.zeros(1, device=dev, dtype=torch.int32)
        hx = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1),
                  **{k: FC.STATE[k] for k in ('exact_head', 'series_limit', 'exact_blocks')})
        credit, score = torch.empty(L + 1, n, W, device=dev), torch.zeros(L + 1, n, P, device=dev)
        lam = torch.zeros(L, 2, device=dev)
        m_chunk, k4 = min(s.readout_chunk, T), 4 * scalars  # the scalar probes per root (a pair's first GPU)
        buf = self.buffers(m_chunk, P + L * k4)
        twin_buf = torch.empty(m_chunk * P * W, device=dev)
        inj_buf = torch.empty(m_chunk * P * W, device=dev, dtype=torch.bfloat16)
        carry = torch.zeros(L, P, W, device=dev)
        w_proj = [blk.attn.c_proj.weight.float() for blk in model.transformer.h]
        self.route_cache, self.ctx_cache = {}, {}
        for seq in range(d.batch):
            own = e.own_query_inputs(seq)
            self.route_cache[seq] = dict(own=own, routes=(s.query_keys, FC.query_routes(own, s.query_keys)))
            ctxs = [self.clean(seq, j) for j in range(L)]
            for r in range(L - 1, -2, -1):  # root r: mid_r, from the top down; r = -1: the embedding (+ scalar probes)
                k = P + L * k4 if r < 0 else P
                carry.zero_()  # token 0 of a sequence has no previous token
                last = min(cap2, L - 1)  # the twins' last scored block
                twins = r + 1 <= cap1  # a first hop at r + 1 .. cap1 and a second hop after it
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    m = t1 - t0
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1) | hx | dict(xL=xL[sl], base=base[sl], target_w=target_w[sl],
                                                                       y=y[sl])
                    delta, dx0 = buf['delta'][:m * k * W].view(m, k, W), None
                    twin = twin_buf[:m * P * W].view(m, P, W).zero_()
                    if r < 0:
                        dx0 = F.rms_norm(emb[sl, None] + steps[0][sl, None] * bases[0], (W,)) - x0[sl, None]
                        torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                    else:
                        torch.mul(steps[r + 1][sl, None], bases[r + 1], out=delta)
                        self.mlp(x, delta, P, r)
                    for b in range(r + 1, L):
                        n_b = P + b * k4 if r < 0 else P
                        score[r + 1, sl] += self.score(x, delta, n_b, b, self.SELF_EDGE)[:, :P]
                        inj = None
                        if twins and b <= cap1:  # the chain rows' first hop into t + 1, from the scorer's buffers
                            nc = self.sp_w['wcat'][b].shape[1]
                            if IMPL == 'torch':  # the reference arithmetic (checks)
                                up = x['up'][:m * n_b * nc].view(m, n_b, nc)[:, :P]
                                st = x['st'][:m * n_b * 2].view(m, n_b, 2)[:, :P]
                                inj = inject(ctxs[b], up, st, self.cs_all, self.sn_all, w_proj[b], t0, T)
                            else:
                                inj = inject_kernel(ctxs[b], x['up'][:m * n_b * nc].view(m * n_b, nc), x['st'], n_b, P,
                                                    self.cs_all, self.sn_all, self.sp_w['w_proj_t'][b], t0, T, inj_buf)
                        self.push(x, delta, n_b, b)
                        if twins and r + 2 <= b <= last:  # the twin rows' second hop: receivers u > t + 1, chain hubs
                            got = self.score(x, twin, P, b, self.SELF_EDGE)[:, :P]
                            if t0 > 0:  # slot i holds key t0 + i - 1
                                score[r + 1, seq * T + t0 - 1:seq * T + t1 - 1] += got
                            else:
                                score[r + 1, seq * T:seq * T + t1 - 1] += got[1:]
                            if b < last:
                                self.push(x, twin, P, b)
                        if inj is not None:  # t + 1's mid_b gains t's first hop
                            twin[1:] += inj[:-1]
                            twin[0] += carry[b]
                            carry[b] = inj[-1]
                        if r < 0 and k4:  # the symmetric probes of block b's two mixing scalars start at mid_b
                            prev = (x0 if b == 0 else outs[b - 1])[sl]
                            delta[:, n_b:n_b + 4] = torch.stack([prev, x0[sl], -prev, -x0[sl]], 1) * h
                        self.mlp(x, delta, n_b + k4 * (r < 0), b, dx0, P)
                        if twins and b < last:
                            self.mlp(x, twin, P, b)
                    change = FC.head_fast(delta, x) * (y[sl] >= 0).unsqueeze(1)
                    credit[r + 1, sl] = (change[:, :P] / steps[r + 1][sl]) @ bases[r + 1] / W
                    if r < 0 and k4:  # the central difference over 2h per scalar
                        pair = change[:, P:].view(t1 - t0, L, 4) / h
                        lam += (pair[..., :2] - pair[..., 2:]).sum(0) / 2
                tok = e.group_sum(credit[r + 1, seq * T:(seq + 1) * T])  # the pair's halves: root r is complete
                if r >= 0:  # its hub's moment tables (chain hubs only), read by every root below it
                    self.tables(ctxs[r], r, tok @ model.transformer.h[r].attn.c_proj.weight.float())
            self.ctx_cache[seq] = ctxs
        X = e.group_sum(torch.stack([(score[k] / steps[k]) @ bases[k] / W for k in range(L + 1)]))
        self.X = {'emb': X[0], **dict(enumerate(X[1:]))}
        e.group_sum(lam)
        self.sp_events[1].record()
        STATS['calls'] += 1
        if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % STATS['every'] == 0):
            print(f"twin_rows estimate {STATS['calls']}: next-token twin rows on (first hops <= block {cap1}, second "
                  f"hops <= block {cap2})", flush=True)
        return {'emb': credit[0], **dict(enumerate(credit[1:])), **{('lam', r): lam[r] for r in range(L)}}


def install(on=None):
    """fast_s2's install (items: $FAST_S2, on fast_e3 + shared_push), then this file's Settings / recipe dict and the
    Twin mixin first in fast_e3's shared Methods (idempotent). The speed stack goes on after."""
    if R.Settings is Settings:
        R.Settings, R.asdict = SP.Settings, SP.asdict
    on = S2.install(on, shared=True)
    if (R.Settings, R.asdict) != (SP.Settings, SP.asdict):
        raise RuntimeError('twin_rows: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Twin):
            FE.SHARED_CLASSES[kv] = type(f'Twin{cls.__name__}', (Twin, cls), {})
    return on


# ======================================== module combo_twin
"""Two-hop credit by next-token twin rows. The readout scores a receiver's attention change at its own-path hub, so credit
crossing two attention edges (t -> t + 1 at one block, then t + 1's key / value -> u at a later one) is dropped. Each
experiment's row gets a twin row in token t + 1's slot, run by the stack's own scorer / push / MLP kernels on t + 1's
contexts. In this run (K-composition: twin_kc [5, 7], identity transport, linear consumers) the first hops run below
block 7 and the second hop is scored only at blocks 5 and 7 through t + 1's key, against the receivers' chain hubs; the
twin's scores join its row's before the decode, for the chain roots and the block-input cycle x_1. Each twin score is its
row's forward experiment scored once at a measured hub, decoded linearly into the row's own consumers.
"""
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
import triton

import combo_he_md as CB  # noqa: E402
import twin_rows as TW  # noqa: E402  (twin_kernel / inject only)
S2, SP, FE, R, FC = CB.S2, CB.SP, CB.FE, CB.R, CB.FC
RO = SP.RO

STATS = dict(calls=0, every=200)


@dataclass
class Settings(CB.Settings):  # combo_he_md's fields (shared_push's, head edits, distillation) and the twin rows
    twin: bool = False
    twin_consumers: str = 'every'
    twin_cap1: int = 6
    twin_cap2: int = 7
    twin_inputs: tuple = (1, 2, 3, 4, 5, 6)
    twin_kc: tuple = ()
    twin_transport: str = 'real'

    def __post_init__(self):
        super().__post_init__()
        self.twin_inputs, self.twin_kc = tuple(self.twin_inputs), tuple(self.twin_kc)
        if type(self.twin) is not bool or self.twin and not (
                getattr(self, 'readout_shared', False) and self.twin_consumers in ('linear', 'every')
                and -1 <= self.twin_cap1 < self.twin_cap2 <= 7 and set(self.twin_inputs) <= set(range(1, 7))
                and set(self.twin_kc) <= set(range(1, 8)) and self.twin_transport in ('real', 'identity')
                and (self.twin_transport == 'real' or self.twin_kc)):
            raise ValueError('twin: true / false; true needs readout_shared, twin_consumers linear / every, -1 <= twin_cap1 '
                             '< twin_cap2 <= 7, twin_inputs in 1..6, twin_kc in 1..7, twin_transport real / identity '
                             '(identity: twin_kc only)')


def asdict(settings):  # off: combo_he_md's recipe dict
    state = CB.asdict(settings)
    if not getattr(settings, 'twin', False):
        for key in [k for k in state if k.startswith('twin')]:
            state.pop(key)
    return state


def plan(s, L):
    """(first-hop blocks, last scored block, kc blocks) of the settings."""
    kc = tuple(sorted(s.twin_kc))
    last = max(kc) if kc else min(s.twin_cap2, L - 1)
    return [b for b in range(min(s.twin_cap1, last - 1, L - 2) + 1)], last, kc


class Rows:
    """The twin rows of one experiment's rows (`rows` per token; slot i of a chunk holds key t0 + i - 1's twins): first
    hops in (inject, land), second hops scored (second) and carried (mlp), scores out (collect). The stack's dense kernels
    (shared_push.Shared's score / push / mlp: never the receivers' live lists, which belong to the token's own rows)."""

    def __init__(self, ro, m_chunk, rows):
        e, s = ro.e, ro.e.s
        self.ro, self.L, self.W, self.T = ro, e.model.config.n_layer, e.width, e.capture.length
        self.first, self.last, self.kc = plan(s, self.L)
        self.real = s.twin_transport == 'real'
        dev, W = e.device, self.W
        self.buf = torch.empty(m_chunk * rows * W, device=dev)
        self.inj32 = torch.empty(m_chunk * rows * W, device=dev)
        self.inj16 = torch.empty(m_chunk * rows * W, device=dev, dtype=torch.bfloat16)
        self.carry = torch.zeros(self.L, rows, W, device=dev)
        self.sc = torch.zeros(m_chunk, rows, device=dev)
        self.lam = e.model.resid_lambdas.float()
        if self.kc:  # the key-only scorer: the projection without its V / gate columns, a zero raw value
            self.wk = [torch.cat([w[:, :W], torch.zeros_like(w[:, W:])], 1) for w in ro.sp_w['wcat']]
            self.vc0 = torch.zeros(self.T, W, device=dev)

    def phase(self):  # a new row set: nothing carried, nothing live
        self.carry.zero_()

    def chunk(self, m, t0):
        self.m, self.t0, self.k_live, self.inj = m, t0, 0, None
        self.twin = self.buf[:m * self.carry.shape[1] * self.W].view(m, -1, self.W)
        self.twin.zero_()
        self.sc[:m].zero_()

    def inject(self, x, n, k, b):
        """After the rows' score at block b (x['up'] / x['st'] theirs, n rows per token): receiver t + 1's change from rows
        < k of token t, through c_proj (held for land)."""
        if b not in self.first or not k:
            return
        ro, m, W = self.ro, self.m, self.W
        c, nc = x['ctxs'][b], ro.sp_w['wcat'][b].shape[1]
        up = x['up'][:m * n * nc].view(m * n, nc)
        if TW.IMPL == 'torch':  # the reference arithmetic (checks)
            inj = TW.inject(c, up.view(m, n, nc)[:, :k], x['st'][:m * n * 2].view(m, n, 2)[:, :k], ro.cs_all, ro.sn_all,
                            ro.e.model.transformer.h[b].attn.c_proj.weight.float(), self.t0, self.T)
        else:
            H, D = c['QD'].shape[1], c['QD'].shape[2]
            o = self.inj16[:m * k * W].view(m, k, W)
            gated = c['gate'] is not None
            TW.twin_kernel[(triton.cdiv(k, 64), m)](
                up, x['st'], c['R0'], c['VC'], c['GZ'] if gated else o, c['ve'] if gated else o, ro.cs_all, ro.sn_all,
                c['QD'], TW.sub_diagonal(c, self.T), c['VT'], c['OT'], o, self.t0, self.T, n, k, W, H, D, gated, 64, nc,
                num_warps=4, num_stages=2)
            inj = self.inj32[:m * k * W].view(m, k, W)
            torch.mm(o.view(m * k, W), ro.sp_w['w_proj_t'][b], out_dtype=torch.float32, out=inj.view(m * k, W))
        self.inj = (b, inj)

    def land(self, b):
        """t + 1's mid_b gains t's first hop (the chunk's last token: into the carry, the next chunk's first slot)."""
        if self.inj is None or self.inj[0] != b:
            return
        inj, k = self.inj[1], self.inj[1].shape[1]
        self.twin[1:, :k] += inj[:-1]
        self.twin[0, :k] += self.carry[b, :k]
        self.carry[b, :k] = inj[-1]
        self.k_live, self.inj = max(self.k_live, k), None

    def second(self, x, b):
        """After the rows' push at block b: the twins live before block b scored at its receivers u > t + 1 (all of K / V
        / gate; kc: the key alone at a kc block), then pushed (real transport, below the last scored block)."""
        k, ro = self.k_live, self.ro
        if not k or b > self.last:
            return
        push = self.real and b < self.last
        if not self.kc:
            self.sc[:self.m, :k] += SP.Shared.score(ro, x, self.twin, k, b, ro.SELF_EDGE)
        else:
            if b in self.kc:
                self.sc[:self.m, :k] += self.key_score(x, k, b)
            if push:  # the push reads the scorer's statistics and the self edge of these rows
                SP.Shared.score(ro, x, self.twin, k, b, ro.SELF_EDGE)
        if push:
            SP.Shared.push(ro, x, self.twin, k, b)

    def mlp(self, x, b):
        """Block b's routed MLP and block b + 1's mixing scalars on the live twins (identity: the scalar alone)."""
        if self.k_live and b < self.last:
            if self.real:
                SP.Shared.mlp(self.ro, x, self.twin, self.k_live, b)
            else:
                self.twin[:, :self.k_live] *= self.lam[b + 1]

    def key_score(self, x, n, b):
        """readout's staged scorer at block b on rows < n with only the key moved (value / gate held; receivers s > t)."""
        ro, d, m, W = self.ro, self.twin, self.m, self.W
        rows = d.shape[1]
        c, H, nc = x['ctxs'][b], ro.e.model.config.n_head, self.wk[b].shape[1]
        out = torch.empty(m, n, device=d.device)
        RO.stats_kernel[(triton.cdiv(n, 16), m)](d, c['x'], c['sqx'], x['db'], x['st'], x['t0'], n, rows, W, 16, RO.EPS,
                                                 num_warps=4)
        up = x['up'][:m * n * nc].view(m * n, nc)
        torch.mm(x['db'][:m * n * W].view(m * n, W), self.wk[b], out=up)
        RO.staged_kernel[(triton.cdiv(n, 64), m)](
            up, x['st'], c['R0'], self.vc0, out, out, ro.cs_all, ro.sn_all, c['U'], c['W'], c['M1'], c['M2'], c['QD'], out,
            c['AD'], out, c['VT'], c['OT'], out, x['se'], x['t0'], n, rows, W, H, W // H, False, False, False, 64, nc,
            num_warps=4, num_stages=2)
        return out

    def collect(self, score, base):
        """The chunk's twin scores into score [tokens, rows] (slot i -> token base + t0 + i - 1)."""
        k, m, t0 = self.k_live, self.m, self.t0
        if k:
            lo = base + t0 - (1 if t0 else 0)
            score[lo:base + t0 + m - 1, :k] += self.sc[0 if t0 else 1:m, :k]


def on(s):
    return bool(getattr(s, 'twin', False))


class Twin:
    """First in the MRO of fast_e3's shared Methods (on combo_he_md's stack): shared_push's chain and pass with the twin
    rows (copies of Shared.chain_credits / Shared.run, every twin line marked "twin"), the input twins inside the chain, and
    I2 into the record's inputs (path_credits). Twin off: the stack's methods, as they are."""

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        e, s = self.e, self.e.s
        if not on(s):
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'combo_twin: the fast_chain kernels do not cover this setting ({reason})')
        model, d, dev = e.model, e.capture, e.device
        L, W, T, n = model.config.n_layer, e.width, d.length, emb.shape[0]
        self.sp_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.sp_events[0].record()
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P, scalars, h = hi - lo, e.part == 0, s.scalar_step
        self.chain_args = (emb, x0, mids, outs, routes, y)
        self.bases = [e.basis() for _ in range(L + 1)]  # the record's draws in its order: emb, mid0, mid1, ...
        bases = [B[lo:hi].contiguous() for B in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb, *mids]]
        self.prepare(mids, routes)
        head = model.lm_head.weight.float()[:model.config.vocab_size]
        xL = outs[-1].float().contiguous()
        base, target_w, weight_t = xL @ head.T, head[y.clamp(min=0)], head.T.to(torch.bfloat16).contiguous()
        if FC.STATE['exact_blocks'] is None:
            FC.STATE['exact_blocks'] = torch.zeros(1, device=dev, dtype=torch.int32)
        hx = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1),
                  **{k: FC.STATE[k] for k in ('exact_head', 'series_limit', 'exact_blocks')})
        credit, score = torch.empty(L + 1, n, W, device=dev), torch.zeros(L + 1, n, P, device=dev)
        lam = torch.zeros(L, 2, device=dev)
        m_chunk, k4 = min(s.readout_chunk, T), 4 * scalars  # the scalar probes per root (a pair's first GPU)
        buf = self.buffers(m_chunk, P + L * k4)
        tw = Rows(self, m_chunk, P)  # twin
        ins = [i for i in s.twin_inputs if i in tw.first]  # twin: the input twins (x_i's first hop at block i)
        isc = torch.zeros(len(ins), n, P, device=dev)
        isteps = {i: s.chain_sigma * d.blocks[i].float().flatten(0, 1).square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                  for i in ins}
        self.route_cache, self.ctx_cache = {}, {}
        for seq in range(d.batch):
            own = e.own_query_inputs(seq)
            self.route_cache[seq] = dict(own=own, routes=(s.query_keys, FC.query_routes(own, s.query_keys)))
            ctxs = [self.clean(seq, j) for j in range(L)]
            for r in range(L - 1, -2, -1):  # root r: mid_r, from the top down; r = -1: the embedding (+ scalar probes)
                k = P + L * k4 if r < 0 else P
                tw.phase()  # twin
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1) | hx | dict(xL=xL[sl], base=base[sl], target_w=target_w[sl],
                                                                       y=y[sl])
                    delta, dx0 = buf['delta'][:(t1 - t0) * k * W].view(t1 - t0, k, W), None
                    tw.chunk(t1 - t0, t0)  # twin
                    if r < 0:
                        dx0 = F.rms_norm(emb[sl, None] + steps[0][sl, None] * bases[0], (W,)) - x0[sl, None]
                        torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                    else:
                        torch.mul(steps[r + 1][sl, None], bases[r + 1], out=delta)
                        self.mlp(x, delta, P, r)
                    for b in range(r + 1, L):
                        n_b = P + b * k4 if r < 0 else P
                        score[r + 1, sl] += self.score(x, delta, n_b, b, self.SELF_EDGE)[:, :P]
                        tw.inject(x, n_b, P, b)  # twin: the rows' first hop at block b, from the scorer's projection
                        self.push(x, delta, n_b, b)
                        tw.second(x, b)  # twin: the twins' second hop at block b
                        tw.land(b)  # twin
                        if r < 0 and k4:  # the symmetric probes of block b's two mixing scalars start at mid_b
                            prev = (x0 if b == 0 else outs[b - 1])[sl]
                            delta[:, n_b:n_b + 4] = torch.stack([prev, x0[sl], -prev, -x0[sl]], 1) * h
                        self.mlp(x, delta, n_b + k4 * (r < 0), b, dx0, P)
                        tw.mlp(x, b)  # twin
                    change = FC.head_fast(delta, x) * (y[sl] >= 0).unsqueeze(1)
                    credit[r + 1, sl] = (change[:, :P] / steps[r + 1][sl]) @ bases[r + 1] / W
                    tw.collect(score[r + 1], seq * T)  # twin
                    if r < 0 and k4:  # the central difference over 2h per scalar
                        pair = change[:, P:].view(t1 - t0, L, 4) / h
                        lam += (pair[..., :2] - pair[..., 2:]).sum(0) / 2
                tok = e.group_sum(credit[r + 1, seq * T:(seq + 1) * T])  # the pair's halves: root r is complete
                if r >= 0:  # its hub's moment tables, read by every root below it
                    self.tables(ctxs[r], r, tok @ model.transformer.h[r].attn.c_proj.weight.float())
            for j, i in enumerate(ins):  # twin: x_i's cycle (root i's basis), its first hop at block i, the twins after it
                tw.phase()
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1)
                    delta = buf['delta'][:(t1 - t0) * P * W].view(t1 - t0, P, W)
                    torch.mul(isteps[i][sl, None], bases[i + 1], out=delta)
                    tw.chunk(t1 - t0, t0)
                    self.project(x, delta, P, i)
                    tw.inject(x, P, P, i)
                    tw.land(i)
                    tw.mlp(x, i)
                    for b in range(i + 1, L):
                        tw.second(x, b)
                        tw.mlp(x, b)
                    tw.collect(isc[j], seq * T)
            self.ctx_cache[seq] = ctxs
        X = e.group_sum(torch.stack([(score[k] / steps[k]) @ bases[k] / W for k in range(L + 1)]))
        self.X = {'emb': X[0], **dict(enumerate(X[1:]))}
        self.I2 = dict(zip(ins, e.group_sum(torch.stack([(isc[j] / isteps[i]) @ bases[i + 1] / W for j, i in
                                                         enumerate(ins)])))) if ins else {}  # twin
        e.group_sum(lam)
        self.sp_events[1].record()
        STATS['calls'] += 1
        if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % STATS['every'] == 0):
            first, last, kc = plan(s, L)
            print(f"combo_twin estimate {STATS['calls']}: twin rows on (consumers {s.twin_consumers}, first hops at blocks "
                  f"{first}, second hops <= block {last}{f' at the keys of {list(kc)}' if kc else ''}, transport "
                  f"{s.twin_transport}, input twins {ins})", flush=True)
        return {'emb': credit[0], **dict(enumerate(credit[1:])), **{('lam', r): lam[r] for r in range(L)}}

    def path_credits(self, chain, targets, raw_values=None):
        """The stack's (the record's inputs, the pass, extra -> R); twin: inputs_i += I2_i in the same dict (the record's
        scalars and the writers' mlp.c_proj read it)."""
        out, inputs = super().path_credits(chain, targets, raw_values)
        for i, credit in (getattr(self, 'I2', None) or {}).items() if on(self.e.s) else ():
            inputs[i] = inputs[i] + credit
        return out, inputs

    def project(self, x, delta, n, b):
        """The scorer's first two launches at block b on rows < n (bf16 copy, row statistics, d W_k^T | d W_v^T | gate)."""
        m, rows, W = delta.shape
        c, nc = x['ctxs'][b], self.sp_w['wcat'][b].shape[1]
        RO.stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], x['db'], x['st'], x['t0'], n, rows, W, 16,
                                                 RO.EPS, num_warps=4)
        torch.mm(x['db'][:m * n * W].view(m * n, W), self.sp_w['wcat'][b], out=x['up'][:m * n * nc].view(m * n, nc))

    def run(self, chain, targets):
        """shared_push's pass; twin_consumers 'every': each row's twin (decoded with the row)."""
        e, s = self.e, self.e.s
        if not (on(s) and s.twin_consumers == 'every'):
            return super().run(chain, targets)
        model, d, dev = e.model, e.capture, e.device
        L, W, H, T = model.config.n_layer, e.width, model.config.n_head, d.length
        self.sp_events[2].record()
        lam_r, Rr = model.resid_lambdas.float(), s.readout_k_receivers
        hb, ins = tuple(s.readout_hi_blocks), tuple(sorted(s.readout_in_roots))
        top = {hb[-1] + 1} if hb and s.readout_hi_top_root and hb[-1] + 1 <= L - 2 else set()
        self.hi_blocks, self.hi_in, self.hi_q, self.hi_fc, self.hi_k, self.hi_g = hb, ins, {}, {}, {}, {}
        self.hi_roots = sorted(set(hb) | ({SP.FIRST} if s.readout_hi_out3 else set()) | top)
        self.hi_mproj = sorted(({SP.FIRST - 1} if s.readout_hi_out3 else set()) | set(hb) | {j - 1 for j in ins})
        self.hi_w = dict(wq={j: model.transformer.h[j].attn.c_q.weight.float() for j in ins})
        src_b = SP.BLOCKS + hb if s.readout_sources else ()
        cyc = {('q', b): W if b in SP.BLOCKS and b in s.readout_q_full else H * s.readout_q_keys for b in src_b}
        cyc |= {('f', b): (s.readout_fc_cycle + s.readout_hi_fc_cycle)[b] for b in src_b}
        share = {key: e.share(c) for key, c in cyc.items()}
        mine1 = mine2 = []  # this GPU's receiver groups: E1's (blocks 0-3) and E3's (hi blocks), split over the pair as there
        if SP.KV.kv_on(s):
            g1, own1 = SP.KV.groups_of(s, model, e.split)
            g2 = [(b, k) for b in hb for k in 'kg' if k == 'k' or s.readout_gate and model.transformer.h[b].attn.ve_gate
                  is not None]
            own2 = SP.HI.balance(g2, e.split, L)
            mine1, mine2 = [g for g in g1 if own1[g] == e.part], [g for g in g2 if own2[g] == e.part]
        mine = mine1 + mine2
        self.kv_C, self.kv_counts, self.kv_added = SP.KV.group_rows(s, H), {}, {}
        self.kv_wproj = {b: model.transformer.h[b].attn.c_proj.weight.float() for b, _ in mine}
        size = {key: b - a for key, (a, b) in share.items()} | {g: self.kv_C for g in mine}
        slots, at, in_rows, mid_rows = {}, 0, [], []  # rows per token, insertion order: q_b, groups of b | MLP b | f_b
        for b in range(L):
            in_rows.append(at)
            for key in [('q', b), *[g for g in mine if g[0] == b], 'mlp', ('f', b)]:
                if key == 'mlp':
                    mid_rows.append(at)
                elif key in size:
                    slots[key], at = slice(at, at + size[key]), at + size[key]
        rows, n = at, d.batch * T
        # readout_kv's / readout_hi's notes: their groups and rows in this pass (their own passes are gone: 0 s)
        self.kv_mine, self.kv_rows, self.kv_events, self.hi_events = mine1, len(mine1) * self.kv_C, [], []
        self.hi_cfg = dict(rows=sum((in_rows + [rows])[b + 1] - in_rows[b] for b in hb), mine=mine2)
        self.q_credit = {b: torch.zeros(n, W, device=dev) for k, b in cyc if k == 'q'}
        self.fc_credit = {b: torch.zeros(n, 4 * W, device=dev) for k, b in cyc if k == 'f'}
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, k in mine if k == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, k in mine if k == 'g'}
        Q, score = torch.zeros(len(ins), n, W, device=dev), torch.zeros(n, rows, device=dev)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(  # readout's seed
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        lo, hi = e.share(W)
        fc_in = {j: slots['f', j - 1] for j in ins if ('f', j - 1) in slots}  # c_fc rows also along the in-root's path
        rms = lambda z: s.readout_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        m_chunk = min(s.readout_chunk, T)
        buf = self.buffers(m_chunk, max(rows, 1))
        tw = Rows(self, m_chunk, max(rows, 1))  # twin
        for seq in range(d.batch):
            ctxs, src, plans = self.ctx_cache.pop(seq), {}, {}
            gens = [torch.Generator(dev).manual_seed(seed + off + 104729 * seq) for off in (0, SP.HI.HI_SEED)]
            for b in src_b:  # readout's draws in its order (block b's q code, then its c_fc code; hi blocks: 2nd generator)
                g, c = gens[b >= SP.FIRST], ctxs[b]
                basis = self.q_basis(c, s.readout_q_keys) if b >= SP.FIRST or cyc['q', b] < W else None
                src['q', b] = (self.draw_code(g, T, cyc['q', b], dev), basis, rms(c['qraw']))
                src['f', b] = (self.draw_code(g, T, cyc['f', b], dev),
                               rms(d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float())) + self.fc_slots(b, seq, cyc['f', b])
            for j in ins:
                src['in', j] = (self.draw_code(gens[1], T, W, dev), rms(ctxs[j]['x']))
            for b in sorted({g[0] for g in mine}):
                plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in mine if bb == b], self.cs_all, self.sn_all))
            tw.phase()  # twin
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m, sl, local = t1 - t0, slice(seq * T + t0, seq * T + t1), slice(t0, t1)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                tw.chunk(m, t0)  # twin
                dirs = {key: self.code_rows(src[key][0], cyc[key], *share[key], local, dev) for key in cyc}
                for key in [key for key in cyc if key[0] == 'q' and src[key][1] is not None]:  # 2 blockdiag(basis_h) code
                    dirs[key] = 2 * torch.einsum('thdk,tphk->tphd', src[key][1][local], dirs[key].view(
                        m, size[key], H, s.readout_q_keys)).reshape(m, size[key], W)
                for b in range(L):
                    c, k, fb = ctxs[b], in_rows[b], slots.get(('f', b - 1))
                    if k:
                        score[sl, :k] += self.score(x, delta, k, b, b < L - 1 or fb is not None)
                        tw.inject(x, k, k, b)  # twin: every scored row's first hop at block b
                    if fb is not None:  # block b - 1's c_fc rows: the receiver s = t at their first block, from the self-edge
                        se = x['se'][:m * rows * W].view(m, rows, W)[:, fb]
                        score[sl, fb] += (se.float() * c['target'][local, None]).sum(-1)
                    if k and b in fc_in:  # block b - 1's c_fc rows at x_b, before the push (the last block too)
                        score[sl, fc_in[b]] += self.oq_scores(c, b, delta[:, fc_in[b]], self.chain[b][sl], local)
                    if b == L - 1:
                        tw.second(x, b)  # twin: the twins' second hop at the last block (after the rows' scores)
                        break
                    if k:
                        self.push(x, delta, k, b)
                    tw.second(x, b)  # twin
                    tw.land(b)  # twin
                    if ('q', b) in slots:
                        delta[:, slots['q', b]] = self.q_moves(c, b, dirs['q', b], src['q', b][2][local], local)
                    for g in mine:
                        if g[0] == b:
                            delta[:, slots[g]] = self.moves(g, plans[g], c, t0, t1)
                    if mid_rows[b]:
                        self.mlp(x, delta, mid_rows[b], b)
                    tw.mlp(x, b)  # twin
                    if ('f', b) in slots:
                        _, fstep, idx, hval, valid = src['f', b]
                        torch.mul(self.fc_moves(b, dirs['f', b], fstep[local], idx[local], hval[local], valid[local]),
                                  lam_r[b + 1], out=delta[:, slots['f', b]])
                tw.collect(score, seq * T)  # twin: each row's experiment takes its twin's scores
                for (kind, b), got in ((key, score[sl, slots[key]]) for key in cyc):  # the sources' decode (readout's)
                    if kind == 'q':
                        self.q_credit[b][sl] = torch.einsum('tp,tpw->tw', got / src[kind, b][2][local], dirs[kind, b]) / W
                    else:
                        _, fstep, idx, hval, valid = src[kind, b]
                        slot = torch.einsum('tp,tpk->tk', got / fstep[local], dirs[kind, b][:, :, 1:]) / cyc[kind, b]
                        self.fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                for i, j in enumerate(ins):  # the in-roots: this GPU's rows of their cycles, decoded into its share
                    code, istep = src['in', j]
                    rows_in = self.code_rows(code, W, lo, hi, local, dev)
                    sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                    Q[i, sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
            out = slice(seq * T, (seq + 1) * T)
            for g in mine:  # the receivers' decode (readout_kv's)
                b, kind = g
                p = plans[g]
                f = p['table'].clamp_min(0)
                live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
                reading = torch.zeros(H * T * Rr + 1, device=dev).scatter_(  # the last entry: a sink for empty slots
                    0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, score[out, slots[g]], 0.).reshape(-1))
                reading = reading[:-1].view(H, T, Rr)
                if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                    c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                    self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
                else:
                    self.g_credit[b][out] = reading.sum(-1).T / SP.KV.GATE_SIGMA
            self.route_cache.pop(seq, None)
        self.hi_Q = dict(zip(ins, e.group_sum(Q))) if ins else {}
        self.sp_events[3].record()


def install(on=None):
    """combo_he_md's install (fast_s2's items once, both layers' patches; idempotent), then this file's Settings / recipe
    dict and the Twin mixin first in fast_e3's shared Methods. The speed stack goes on after."""
    if R.Settings is Settings:
        R.Settings, R.asdict = CB.Settings, CB.asdict
    on = CB.install(on)
    if (R.Settings, R.asdict) != (CB.Settings, CB.asdict):
        raise RuntimeError('combo_twin: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Twin):
            FE.SHARED_CLASSES[kv] = type(f'Twin{cls.__name__}', (Twin, cls), {})
    return on


# ======================================== module speed_stack
"""Speed levers: the same quantities computed once (c_fc source cycles sized to the update's active units, the input-K
tables and attention maps read from the chain's contexts, q-source head patterns once per sign) and lossless storage
(bf16 / fp8-with-scale row copies where the math stays fp32).
"""
import contextlib
from dataclasses import dataclass
import functools
import inspect
import linecache
import math
import os
import textwrap

import torch
import torch.nn.functional as F
import triton

import combo_twin as CT  # noqa: E402
import s2_plan as S2P  # noqa: E402
CB, SP, FE, R, FC, RO = CT.CB, CT.SP, CT.FE, CT.R, CT.FC, CT.RO
FR, KV, FM, MD = SP.FR, SP.KV, SP.FM, CB.MD

PLAIN = dict(block_inputs=FM.block_inputs, moment_block=FM.moment_block, attention_maps=R.Boost.attention_maps)
STATS = dict(calls=0, every=200)
EXACT = dict(fc=True, tables=True, maps=True, q=True)  # speed_exact's items (checks: one at a time)
F8MAX = 448.  # e4m3


@dataclass
class Settings(CT.Settings):  # combo_twin's fields and the levers
    speed_exact: bool = False
    speed_store: bool = False
    speed_lite8: bool = False
    readout_skip_blocks: tuple = ()
    readout_k_select: str = 'attention'

    def __post_init__(self):
        super().__post_init__()
        self.readout_skip_blocks = tuple(self.readout_skip_blocks)
        flags = (self.speed_exact, self.speed_store, self.speed_lite8)
        used = any(flags) or self.readout_skip_blocks or self.readout_k_select != 'attention'
        if (any(type(f) is not bool for f in flags) or self.speed_lite8 and not self.speed_store
                or used and not (SP.on(self) and (not CT.on(self) or self.twin_consumers == 'linear'))
                or self.readout_k_select not in ('attention', 'bound')
                or not set(self.readout_skip_blocks) <= set(range(self.model.n_layer - 1))):
            raise ValueError('speed_stack: speed_exact / speed_store / speed_lite8 true / false (lite8 needs store), '
                             'readout_skip_blocks within 0..n_layer - 2, readout_k_select attention / bound; any of '
                             'them needs the shared stack (readout, readout_shared) and twin rows off or linear')


def asdict(settings):  # off: combo_twin's recipe dict
    state = CT.asdict(settings)
    for key in ('speed_exact', 'speed_store', 'speed_lite8', 'readout_skip_blocks', 'readout_k_select'):
        if state.get(key) in (False, (), [], 'attention'):
            state.pop(key, None)
    return state


# ----------------------------------------------------------------------------------------------------------- kernels


def derive(kernel, name, edits):
    """The stack's Triton kernel with each (old, new) edit made once (checked), compiled as its own kernel `name`."""
    fn = kernel.fn
    src = textwrap.dedent(inspect.getsource(fn))
    src = src[src.index('def '):]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f'speed_stack: {fn.__name__} is not the kernel this file edits ({old.strip()[:60]!r})')
        src = src.replace(old, new)
    src = src.replace(f'def {fn.__name__}(', f'def {name}(', 1)
    file = f'<speed_stack {name}>'
    linecache.cache[file] = (len(src), None, src.splitlines(True), file)  # triton reads the source back
    scope = dict(fn.__globals__)
    exec(compile(src, file, 'exec'), scope)
    return triton.jit(scope[name])


F32 = '.to(tl.float32)'
LOAD_D = 'k[None, :], pm[:, None], 0.)\n'  # stats_kernel's load of a move row
STATS_X = derive(RO.stats_kernel, 'stats_kernel_x', [(LOAD_D, LOAD_D[:-1] + F32 + '\n')])  # bf16 moves read as fp32
STATS8 = derive(RO.stats_kernel, 'stats8_kernel', [  # the row copy in fp8 with its scale (SA) instead of bf16
    (LOAD_D, LOAD_D[:-1] + F32 + '\n'), ('Dm, X, SQX, DB, ST,', 'Dm, X, SQX, DB, SA, ST,'),
    ('    tl.store(DB + (t * n + p).to(tl.int64)[:, None] * W + k[None, :], d.to(tl.bfloat16), pm[:, None])\n',
     '    sa = tl.maximum(tl.max(tl.abs(d), 1), 1e-30) / 448.\n'
     '    tl.store(DB + (t * n + p).to(tl.int64)[:, None] * W + k[None, :], (d / sa[:, None]).to(tl.float8e4nv), '
     'pm[:, None])\n    tl.store(SA + t * n + p, sa, pm)\n')])
UP_X = derive(FR.mlp_up_prune_kernel, 'mlp_up_prune_kernel_x', [
    (f'{s}d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)\n{s}{n}',
     f'{s}d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.){F32}\n{s}{n}')
    for s, n in ((' ' * 8, 'x = '), (' ' * 12, 'w = '))])
HEAD_X = derive(FC.head_prep_kernel, 'head_prep_kernel_x', [
    ('d = tl.load(D + row[:, None] * W + kk[None, :], ok[:, None], 0.)\n',
     f'd = tl.load(D + row[:, None] * W + kk[None, :], ok[:, None], 0.){F32}\n')])
OQ_SE = derive(FC.oq_attn_kernel, 'oq_attn_se_kernel', [  # bf16 projections; FUSE: + the row's self-edge (SE, bf16)
    ('IDX, TP, O, Out, n_rows, U_ROWS, O_ROWS, t_start,', 'IDX, TP, O, Out, SE, n_rows, U_ROWS, O_ROWS, SE_ROWS, t_start,'),
    ('BP: tl.constexpr, EPS: tl.constexpr):', 'BP: tl.constexpr, EPS: tl.constexpr, FUSE: tl.constexpr):'),
    ('ul = tl.load(ub[:, None] + hd[None, :], ok[:, None], 0.)', 'ul = tl.load(ub[:, None] + hd[None, :], ok[:, None], 0.)' + F32),
    ('uh = tl.load(ub[:, None] + QH + hd[None, :], ok[:, None], 0.)',
     'uh = tl.load(ub[:, None] + QH + hd[None, :], ok[:, None], 0.)' + F32),
    ('        tl.store(Out + (t * O_ROWS + rows)',
     '        if FUSE:  # c_proj then runs once on the sum (an exact identity)\n'
     '            moved += tl.load(SE + (t * SE_ROWS + rows).to(tl.int64)[:, None] * W + hh * DH + feat[None, :], ok[:, None],'
     f' 0.){F32}\n        tl.store(Out + (t * O_ROWS + rows)')])


def smm(a8, b8t, sa, sb, out):
    """a8 @ b8t, an fp8 e4m3 GEMM (cuBLAS scaled mm: per-row x per-column scales, fast accumulation) into out (bf16)."""
    if STATS.get('smm_out', True):
        try:
            return torch._scaled_mm(a8, b8t, scale_a=sa, scale_b=sb, out_dtype=out.dtype, use_fast_accum=True, out=out)
        except (TypeError, RuntimeError):  # a torch without the out= overload: write the result into the buffer
            STATS['smm_out'] = False
    return out.copy_(torch._scaled_mm(a8, b8t, scale_a=sa, scale_b=sb, out_dtype=out.dtype, use_fast_accum=True))


def q8(w):
    """[N, K] -> (e4m3 [N, K], fp32 scale [N]): one scale per output channel (row), round to nearest."""
    w = w.float()
    s = w.abs().amax(1).clamp_min(1e-30) / F8MAX
    return (w / s[:, None]).to(torch.float8_e4m3fn).contiguous(), s.contiguous()


def fp8_weights(model):
    """Per estimate: e4m3 copies of the scorer's [W_k; W_v; W_g (16 rows, padded)] (shared_push.prepare's wcat, by
    output channel) and of W_q, with their scales."""
    W, H = model.config.n_embd, model.config.n_head
    out = dict(cat=[], cat_s=[], q=[], q_s=[])
    for blk in model.transformer.h:
        a = blk.attn
        gate = torch.zeros(16, W, device=a.c_q.weight.device)
        if a.ve_gate is not None:
            gate[:H, :32] = a.ve_gate.weight.float()
        for key, w in (('cat', torch.cat([a.c_k.weight.float(), a.c_v.weight.float(), gate])), ('q', a.c_q.weight)):
            w8, s = q8(w)
            out[key].append(w8)
            out[key + '_s'].append(s)
    return out


def head16(plain):
    """mom_distill.wrap_head's head for moves stored in bf16: the installed head as is, then the distilled target's term
    with its dot of the moves and the teacher's head row in bf16 with an fp32 result (its torch.bmm needs one dtype)."""
    inner = getattr(plain, 'mom_distill', None)

    @functools.wraps(plain)
    def head_fast(delta, ctx):
        a = MD.ACTIVE
        if inner is None or a is None or delta.dtype == torch.float32:
            return plain(delta, ctx)
        out = inner(delta, ctx)
        full, (m, rows, W) = ctx['xL']._base, delta.shape
        if full is None or full.shape[0] != a['q'][..., 0].numel():
            raise RuntimeError("speed_stack: the chain head's xL is not a view of the distilled batch's top stream")
        a['calls']['head'] += 1
        if a.get('xL') is not full:  # the first head call of this estimate's chain
            a['xL'], a['rows'] = full, MD.head_rows(full, a)
        t0 = (ctx['xL'].storage_offset() - full.storage_offset()) // W
        u, ub = (z[t0:t0 + m] for z in a['rows'])
        st = ctx['scratch32'][:m * rows * 4].view(m, rows, 4)
        du = torch.bmm(delta, u.unsqueeze(-1).to(delta.dtype), out_dtype=torch.float32).squeeze(-1)
        return out + a['lam'] * (st[..., 2] - (st[..., 1] * ub[:, None] + st[..., 0] * du))
    head_fast.mom_distill = inner
    return head_fast


# ------------------------------------------------------------------------------------- speed_exact: tables and maps


def reused(est, i, r, item='tables'):
    """Block i's chain context of sequence r when speed_exact's item reuses it (tables: its hub's tables built and the
    contrast kept)."""
    m = getattr(est, '_speed', None)
    cache = getattr(m, 'ctx_cache', None) if m is not None and m.e.s.speed_exact and EXACT[item] else None
    c = cache[r][i] if cache and r in cache else None
    return c if c is not None and (item != 'tables' or 'contrast' in c) else None


def block_inputs(est, i, r, target, rows, projected_rows=None):
    """fast_moments.block_inputs with key_context's (qp, kp, raw, cos, sin, A, C) and the moment tables (U, M1) read from
    the chain's context of block i (the same clean attention state, contrast and tables of the hub chain_i W_proj)."""
    c = reused(est, i, r)
    if c is None:
        return PLAIN['block_inputs'](est, i, r, target, rows, projected_rows)
    d, s, model = est.capture, est.s, est.model
    T = d.length
    pos = torch.arange(T, device=est.device)
    lag = pos[:, None] - pos[None, :]
    allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()  # key_context's mask of C
    cos, sin = model.cos[0, :T, 0].float(), model.sin[0, :T, 0].float()
    context = (c['q'], c['k'], c['kraw'].transpose(0, 1), cos, sin, c['a'], c['contrast'] * allowed)
    xin = d.blocks[i][r].float()  # fast_moments.block_inputs' lines
    w_key = model.transformer.h[i].attn.c_k.weight.float()
    steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return dict(context=context, xin=xin, steps=steps, projected_x=xin @ w_key.T,
                projected_rows=rows @ w_key.T if projected_rows is None else projected_rows, tables=(c['U'], c['M1']))


def moment_block(est, i, inputs, rows, total_rows, tables=None):
    return PLAIN['moment_block'](est, i, inputs, rows, total_rows, inputs.get('tables') if tables is None else tables)


def attention_maps(est, i):
    """Boost.attention_maps; speed_exact: block i's clean attention from the chain's contexts (every sequence)."""
    if i not in est.maps and reused(est, i, 0, 'maps') is not None and all(r in est._speed.ctx_cache
                                                                          for r in range(est.capture.batch)):
        est.maps[i] = torch.stack([est._speed.ctx_cache[r][i]['a'] for r in range(est.capture.batch)]).to(torch.bfloat16)
    return PLAIN['attention_maps'](est, i)


@contextlib.contextmanager
def swapped(s):
    """The stack's kernels and functions the settings replace, for one estimate (restored after, also on errors)."""
    new = []
    if s.speed_store:
        new += [(RO, 'stats_kernel', STATS_X), (FR, 'mlp_up_prune_kernel', UP_X), (FC, 'head_prep_kernel', HEAD_X),
                (FC, 'head_fast', head16(FC.head_fast))]
    if s.speed_exact:
        new += [(FM, 'block_inputs', block_inputs), (FM, 'moment_block', moment_block),
                (R.Boost, 'attention_maps', attention_maps)]
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in new]
    for owner, name, value in new:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)


# ------------------------------------------------------------------------------------------------------- the mixin


class Speed:
    """First in the MRO of fast_e3's shared Methods (on combo_twin's): the levers, each off = the stack's method."""

    def estimate(self, x, y):
        self.e._speed = self
        try:
            with swapped(self.e.s):
                return super().estimate(x, y)
        finally:
            self.e.__dict__.pop('_speed', None)

    def live(self, step, *a):  # s2_skipslots' live-row kernels for the receiver rows (when it is installed)
        listed = getattr(self, 'listed', None)
        return step(*a) if listed is None else listed(step, *a)

    def prepare(self, mids, routes):
        super().prepare(mids, routes)
        self.sp_w['w8'] = fp8_weights(self.e.model) if self.e.s.speed_lite8 else None

    def buffers(self, m, rows):
        """shared_push's buffers; speed_store: the moves and the own-query projection in bf16; lite8: the fp8 row copy."""
        s = self.e.s
        if not s.speed_store:
            return super().buffers(m, rows)
        W, dev = self.e.width, self.e.device
        bf = lambda k: torch.empty(m * rows * k, device=dev, dtype=torch.bfloat16)
        out = dict(delta=bf(W), scratch32=torch.empty(m * rows * W, device=dev),
                   scratch=bf(max(W, self.sp_w['units'])), db=bf(W), se=bf(W), up=bf(2 * W + 16),
                   st=torch.empty(m * rows * 2, device=dev), u16=bf(W), **FR.prune_buffers(m, rows, dev))
        if s.speed_lite8:
            out.update(db8=torch.empty(m * rows * W, device=dev, dtype=torch.float8_e4m3fn),
                       sa=torch.empty(m * rows, device=dev))
        return out

    def score(self, x, delta, n, b, self_edge):
        if not self.e.s.speed_lite8:
            return super().score(x, delta, n, b, self_edge)
        return self.live(self.score8, x, delta, n, b, self_edge)

    def score8(self, x, delta, n, b, self_edge):
        """shared_push.Shared.score with the projection d [W_k | W_v | W_g]^T an fp8 GEMM of the rows' fp8 copy."""
        m, rows, W = delta.shape
        c, H, nc = x['ctxs'][b], self.e.model.config.n_head, self.sp_w['wcat'][b].shape[1]
        out = torch.empty(m, n, device=delta.device)
        gated = c['gate'] is not None
        STATS8[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], x['db8'], x['sa'], x['st'], x['t0'], n, rows, W, 16,
                                        RO.EPS, num_warps=4)
        up = x['up'][:m * n * nc].view(m * n, nc)
        w8 = self.sp_w['w8']
        smm(x['db8'][:m * n * W].view(m * n, W), w8['cat'][b].T, x['sa'][:m * n].view(m * n, 1),
            w8['cat_s'][b].view(1, nc), up)
        RO.staged_kernel[(triton.cdiv(n, 64), m)](
            up, x['st'], c['R0'], c['VC'], c['GZ'] if gated else out, c['ve'] if gated else out, self.cs_all, self.sn_all,
            c['U'], c['W'], c['M1'], c['M2'], c['QD'], out, c['AD'], out, c['VT'], c['OT'], out, x['se'], x['t0'], n, rows,
            W, H, W // H, gated, False, self_edge, 64, nc, num_warps=4, num_stages=2)
        return out

    def push(self, x, delta, n, b):
        if not self.e.s.speed_store:
            return super().push(x, delta, n, b)
        return self.live(self.push16, x, delta, n, b)

    def push16(self, x, delta, n, b):
        """fast_readout.own_query_step with d W_q^T stored bf16 (lite8: the fp8 GEMM), the self-edge added to the
        response inside the own-query kernel and ONE c_proj product."""
        m, rows, W = delta.shape
        q = x['routes'][b]
        keys = q['keys']
        H, T, DH = keys.shape
        u = x['u16'][:m * n * W].view(m * n, W)
        if self.e.s.speed_lite8:
            w8 = self.sp_w['w8']
            smm(x['db8'][:m * n * W].view(m * n, W), w8['q'][b].T, x['sa'][:m * n].view(m * n, 1),
                w8['q_s'][b].view(1, W), u)
        else:
            torch.mm(x['db'][:m * n * W].view(m * n, W), x['w_q_t'][b], out=u)
        out = x['ybuf'][:m * n * W]
        bp, warps, stages = FC.OQ_ATTN
        OQ_SE[(H, m)](u, x['st'], q['r0'], q['sq0'], x['cos'], x['sin'], keys, q['values'], q['idx'], q['p'], q['o'],
                      out, x['se'], n, n, n, rows, x['t0'], T, H, DH, W, q['idx'].shape[-1], bp, FR.EPS, self.SELF_EDGE,
                      num_warps=warps, num_stages=stages)
        bm, bn, bk, warps, stages = FC.PROJ
        FR.dual_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](
            out, out, x['w_proj_t'][b], delta, m * n, n, n, rows, rows, W, W, bm, bn, bk, False, num_warps=warps,
            num_stages=stages)

    def tables(self, c, j, target):
        """shared_push.Shared.tables; speed_exact: the contrast kept for the input-K / raw-K experiments."""
        if not (self.e.s.speed_exact and EXACT['tables']):
            return super().tables(c, j, target)
        T, H = self.e.capture.length, self.e.model.config.n_head
        g = target.reshape(T, H, -1).transpose(0, 1)
        contrast = (g @ c['values'].transpose(-1, -2)) - (g * c['out']).sum(-1, keepdim=True)
        c['U'], c['W'], c['M1'], c['M2'] = FM.build_future_tables(c['q'], g, c['a'], contrast,
                                                                  self.e.model.window_sizes[j][0])
        c['target'], c['contrast'] = target, contrast

    def plan(self, b, seq, c, kinds, cs, sn):
        """readout_k_select 'bound': s2_plan.plan with the receivers of each (head, key t) the top R by a_st |v_t - o_s|
        |(W_proj^T X_b(s))_h| and the cap ranked by it; the experiments, slot table and gate coefficients as there."""
        e, s = self.e, self.e.s
        if s.readout_k_select != 'bound':
            return super().plan(b, seq, c, kinds, cs, sn)
        H, T = e.model.config.n_head, e.capture.length
        D, Rr, C = e.width // H, s.readout_k_receivers, self.kv_C
        A = self.clean_attention(b, seq, c)  # [H, T (receiver s), T (key t)]
        g = (self.X[b][seq * T:(seq + 1) * T] @ e.model.transformer.h[b].attn.c_proj.weight.float()).view(T, H, D)
        v, o = c['values'], c['out']  # [H, T, D]
        dist = ((v * v).sum(-1)[:, None, :] - 2 * (o @ v.transpose(1, 2)) + (o * o).sum(-1)[:, :, None]).clamp_min(0).sqrt()
        top = (A * dist * g.norm(dim=-1).T[:, :, None]).transpose(1, 2).topk(Rr, dim=-1)
        idx, bound = top.indices.contiguous(), top.values.contiguous()  # [H, T, R], the largest bound first
        del dist, top
        vals = A.transpose(1, 2).gather(-1, idx).contiguous()  # their attention (the exact single-column gains read it)
        ok = vals > 0
        table, kept = KV.receiver_slots(idx, bound, ok, C, T)
        plans = {}
        if 'k' in kinds:  # s2_plan's
            P = torch.empty(H, T, Rr, D, device=A.device)
            live, coef, x_lin = torch.empty_like(ok), torch.empty_like(vals), torch.empty_like(vals)
            S2P.plan_kernel[(triton.cdiv(T, S2P.BT), H)](
                idx, vals, kept, c['QD'], c['R0'], c['kraw'], cs, sn, P, live, coef, x_lin, T, H, D, Rr, S2P.BT, KV.K_TAU,
                KV.K_SIGMA, KV.EPS, math.sqrt(D), num_warps=S2P.PLAN_WARPS)
            plans[b, 'k'] = dict(table=table, live=live, coef=coef, P=P, x_lin=x_lin)
            self.kv_counts[b, 'k'] = torch.stack([ok.sum(), kept.sum(), live.sum()])
        if 'g' in kinds:  # readout_kv's
            z = c['gate']
            dvs = 2 * (torch.sigmoid(z + KV.GATE_SIGMA) - torch.sigmoid(z))
            coef = torch.where(kept, vals * dvs.T[:, :, None], torch.zeros_like(vals))
            plans[b, 'g'] = dict(table=table, live=kept, coef=coef)
            self.kv_counts[b, 'g'] = torch.stack([ok.sum(), kept.sum(), kept.sum()])
        return plans

    def q_moves_dedup(self, c, b, dirs, src, n, share, local):
        """q_moves of this GPU's rows lo:hi of the q cycle (size n = H kq), each head's response formed once per distinct
        (head pattern, sign): H_n[rho, (h, k)] = H_H[rho // kq, h] H_kq[rho % kq, k], so row rho's head h is unique row
        u = 2 (rho % kq - l0) + (H_H[rho // kq, h] < 0); each unique row's head direction is taken from a row that has it
        in run()'s dirs (bitwise the rows' own directions), its response gathered back to every row."""
        (perm, _), qstep = src[0], src[2][local]
        (lo, hi), e = share, self.e
        m, P, W = dirs.shape
        dev, H = dirs.device, e.model.config.n_head
        kq, D = n // H, W // H
        rho = perm[lo:hi]
        nl = kq * P // n if e.split > 1 else kq  # the per-head indices on this GPU (run()'s split)
        l0 = (rho[:1] % kq) // nl * nl  # [1] on the device: this GPU's first per-head index
        u = 2 * (rho % kq - l0)[:, None] + (R.hadamard(H, dev, torch.float32)[rho // kq] < 0).long()  # [P, H]
        row = torch.zeros(2 * nl, H, dtype=torch.long, device=dev).scatter_(  # a row with each (unique row, head)
            0, u, torch.arange(P, device=dev)[:, None].expand(P, H).contiguous())
        U = dirs.view(m, P, H, D).gather(1, row[None, :, :, None].expand(m, -1, -1, D)).reshape(m, 2 * nl, W)
        out = FE.qsrc2_response(U.contiguous(), c['qraw'].contiguous(), qstep.contiguous(), c['k'], c['values'],
                                c['a'][:, local], c['out'][:, local], local.start, self.cs_all, self.sn_all,
                                e.model.window_sizes[b][0], FE.IMPL['qsrc'])  # [m, 2 nl, H, D]
        out = out.gather(1, u[None, :, :, None].expand(m, -1, -1, D))
        return F.linear(out.flatten(-2), e.model.transformer.h[b].attn.c_proj.weight.float())

    def run(self, chain, targets):
        """shared_push.Shared.run (copied) with readout_skip_blocks, speed_exact's c_fc cycles / q split / q dedup, and
        the in-root path's c_fc rows read as fp32 (speed_store); every change is marked 'speed'."""
        e, s = self.e, self.e.s
        if not (s.speed_exact or s.speed_store or s.readout_skip_blocks):
            return super().run(chain, targets)
        model, d, dev = e.model, e.capture, e.device
        L, W, H, T = model.config.n_layer, e.width, model.config.n_head, d.length
        self.sp_events[2].record()
        lam_r, Rr = model.resid_lambdas.float(), s.readout_k_receivers
        hb, ins = tuple(s.readout_hi_blocks), tuple(sorted(s.readout_in_roots))
        skip = set(s.readout_skip_blocks)  # speed: blocks without pass rows
        top = {hb[-1] + 1} if hb and s.readout_hi_top_root and hb[-1] + 1 <= L - 2 else set()
        self.hi_blocks, self.hi_in, self.hi_q, self.hi_fc, self.hi_k, self.hi_g = hb, ins, {}, {}, {}, {}
        self.hi_roots = sorted(set(hb) | ({SP.FIRST} if s.readout_hi_out3 else set()) | top)
        self.hi_mproj = sorted(({SP.FIRST - 1} if s.readout_hi_out3 else set()) | set(hb) | {j - 1 for j in ins})
        self.hi_w = dict(wq={j: model.transformer.h[j].attn.c_q.weight.float() for j in ins})
        src_b = tuple(b for b in SP.BLOCKS + hb if b not in skip) if s.readout_sources else ()  # speed: skip
        cyc = {('q', b): W if b in SP.BLOCKS and b in s.readout_q_full else H * s.readout_q_keys for b in src_b}
        cyc |= {('f', b): (s.readout_fc_cycle + s.readout_hi_fc_cycle)[b] for b in src_b}
        f_cfg = {b: cyc['f', b] for b in src_b}  # speed: the configured c_fc cycles (their draws keep these sizes)
        if s.speed_exact and EXACT['fc'] and src_b:  # speed: the smallest complete cycle covering every token's active units
            act = torch.stack([(d.outputs[f'transformer.h.{b}.mlp.c_fc'] > 0).sum(-1).amax() for b in src_b]).tolist()
            for b, n_act in zip(src_b, act):
                cyc['f', b] = min(f_cfg[b], max(32, 1 << int(n_act).bit_length()))  # >= n_act + 1 slots
        self.sp_fc = {b: cyc['f', b] for b in src_b}  # speed: notes
        share = {key: e.share(c) for key, c in cyc.items()}
        mine1 = mine2 = []  # this GPU's receiver groups: E1's (blocks 0-3) and E3's (hi blocks), split over the pair as there
        if KV.kv_on(s):
            g1, own1 = KV.groups_of(s, model, e.split)
            g2 = [(b, k) for b in hb if b not in skip for k in 'kg'  # speed: skip
                  if k == 'k' or s.readout_gate and model.transformer.h[b].attn.ve_gate is not None]
            own2 = SP.HI.balance(g2, e.split, L)
            mine1 = [g for g in g1 if own1[g] == e.part and g[0] not in skip]  # speed: skip
            mine2 = [g for g in g2 if own2[g] == e.part]
        mine = mine1 + mine2
        self.kv_C, self.kv_counts, self.kv_added = KV.group_rows(s, H), {}, {}
        self.kv_wproj = {b: model.transformer.h[b].attn.c_proj.weight.float() for b, _ in mine}
        size = {key: b - a for key, (a, b) in share.items()} | {g: self.kv_C for g in mine}
        slots, at, in_rows, mid_rows = {}, 0, [], []  # rows per token, insertion order: q_b, groups of b | MLP b | f_b
        for b in range(L):
            in_rows.append(at)
            for key in [('q', b), *[g for g in mine if g[0] == b], 'mlp', ('f', b)]:
                if key == 'mlp':
                    mid_rows.append(at)
                elif key in size:
                    slots[key], at = slice(at, at + size[key]), at + size[key]
        rows, n = at, d.batch * T
        # readout_kv's / readout_hi's notes: their groups and rows in this pass (their own passes are gone: 0 s)
        self.kv_mine, self.kv_rows, self.kv_events, self.hi_events = mine1, len(mine1) * self.kv_C, [], []
        self.hi_cfg = dict(rows=sum((in_rows + [rows])[b + 1] - in_rows[b] for b in hb), mine=mine2)
        self.q_credit = {b: torch.zeros(n, W, device=dev) for k, b in cyc if k == 'q'}
        self.fc_credit = {b: torch.zeros(n, 4 * W, device=dev) for k, b in cyc if k == 'f'}
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, k in mine if k == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, k in mine if k == 'g'}
        Q, score = torch.zeros(len(ins), n, W, device=dev), torch.zeros(n, rows, device=dev)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(  # readout's seed
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        lo, hi = e.share(W)
        fc_in = {j: slots['f', j - 1] for j in ins if ('f', j - 1) in slots}  # c_fc rows also along the in-root's path
        rms = lambda z: s.readout_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        m_chunk = min(s.readout_chunk, T)
        buf = self.buffers(m_chunk, max(rows, 1))
        dedup = s.speed_exact and EXACT['q'] and FE.IMPL['qsrc'] != 'plain'  # speed: q responses once per head experiment
        for seq in range(d.batch):
            ctxs, src, plans = self.ctx_cache.pop(seq), {}, {}
            for c in ctxs:  # speed: the key path's contrast is not read past it
                c.pop('contrast', None)
            gens = [torch.Generator(dev).manual_seed(seed + off + 104729 * seq) for off in (0, SP.HI.HI_SEED)]
            for b in src_b:  # readout's draws in its order (block b's q code, then its c_fc code; hi blocks: 2nd generator)
                g, c = gens[b >= SP.FIRST], ctxs[b]
                basis = self.q_basis(c, s.readout_q_keys) if b >= SP.FIRST or cyc['q', b] < W else None
                code = self.draw_code(g, T, cyc['q', b], dev)
                if dedup and e.split > 1:  # speed: the pair splits the cycle by the per-head index (stably)
                    kq = cyc['q', b] // H
                    code = (code[0][torch.argsort(((code[0] % kq) * e.split // kq).to(torch.int32), stable=True)], code[1])
                src['q', b] = (code, basis, rms(c['qraw']))
                code = self.draw_code(g, T, f_cfg[b], dev)  # speed: drawn at the configured size
                if cyc['f', b] < f_cfg[b]:  # speed: its values < n, in order (a uniform random complete cycle of n)
                    keep = torch.argsort((code[0] >= cyc['f', b]).to(torch.int32), stable=True)[:cyc['f', b]]
                    code = (code[0][keep].contiguous(), code[1][:, :cyc['f', b]].contiguous())
                src['f', b] = (code, rms(d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float())) + self.fc_slots(
                    b, seq, cyc['f', b])
            for j in ins:
                src['in', j] = (self.draw_code(gens[1], T, W, dev), rms(ctxs[j]['x']))
            for b in sorted({g[0] for g in mine}):
                plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in mine if bb == b], self.cs_all, self.sn_all))
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m, sl, local = t1 - t0, slice(seq * T + t0, seq * T + t1), slice(t0, t1)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                dirs = {key: self.code_rows(src[key][0], cyc[key], *share[key], local, dev) for key in cyc}
                for key in [key for key in cyc if key[0] == 'q' and src[key][1] is not None]:  # 2 blockdiag(basis_h) code
                    dirs[key] = 2 * torch.einsum('thdk,tphk->tphd', src[key][1][local], dirs[key].view(
                        m, size[key], H, s.readout_q_keys)).reshape(m, size[key], W)
                for b in range(L):
                    c, k, fb = ctxs[b], in_rows[b], slots.get(('f', b - 1))
                    if k:
                        score[sl, :k] += self.score(x, delta, k, b, b < L - 1 or fb is not None)
                    if fb is not None:  # block b - 1's c_fc rows: the receiver s = t at their first block, from the self-edge
                        se = x['se'][:m * rows * W].view(m, rows, W)[:, fb]
                        score[sl, fb] += (se.float() * c['target'][local, None]).sum(-1)
                    if k and b in fc_in:  # block b - 1's c_fc rows at x_b, before the push (the last block too)
                        score[sl, fc_in[b]] += self.oq_scores(c, b, delta[:, fc_in[b]].float(), self.chain[b][sl],
                                                              local)  # speed: .float() (bf16 moves)
                    if b == L - 1:
                        break
                    if k:
                        self.push(x, delta, k, b)
                    if ('q', b) in slots:
                        if dedup:  # speed
                            delta[:, slots['q', b]] = self.q_moves_dedup(c, b, dirs['q', b], src['q', b], cyc['q', b],
                                                                         share['q', b], local)
                        else:
                            delta[:, slots['q', b]] = self.q_moves(c, b, dirs['q', b], src['q', b][2][local], local)
                    for g in mine:
                        if g[0] == b:
                            delta[:, slots[g]] = self.moves(g, plans[g], c, t0, t1)
                    if mid_rows[b]:
                        self.mlp(x, delta, mid_rows[b], b)
                    if ('f', b) in slots:
                        _, fstep, idx, hval, valid = src['f', b]
                        torch.mul(self.fc_moves(b, dirs['f', b], fstep[local], idx[local], hval[local], valid[local]),
                                  lam_r[b + 1], out=delta[:, slots['f', b]])
                for (kind, b), got in ((key, score[sl, slots[key]]) for key in cyc):  # the sources' decode (readout's)
                    if kind == 'q':
                        self.q_credit[b][sl] = torch.einsum('tp,tpw->tw', got / src[kind, b][2][local], dirs[kind, b]) / W
                    else:
                        _, fstep, idx, hval, valid = src[kind, b]
                        slot = torch.einsum('tp,tpk->tk', got / fstep[local], dirs[kind, b][:, :, 1:]) / cyc[kind, b]
                        self.fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                for i, j in enumerate(ins):  # the in-roots: this GPU's rows of their cycles, decoded into its share
                    code, istep = src['in', j]
                    rows_in = self.code_rows(code, W, lo, hi, local, dev)
                    sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                    Q[i, sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
            out = slice(seq * T, (seq + 1) * T)
            for g in mine:  # the receivers' decode (readout_kv's)
                b, kind = g
                p = plans[g]
                f = p['table'].clamp_min(0)
                live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
                reading = torch.zeros(H * T * Rr + 1, device=dev).scatter_(  # the last entry: a sink for empty slots
                    0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, score[out, slots[g]], 0.).reshape(-1))
                reading = reading[:-1].view(H, T, Rr)
                if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                    c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                    self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
                else:
                    self.g_credit[b][out] = reading.sum(-1).T / KV.GATE_SIGMA
            self.route_cache.pop(seq, None)
        self.hi_Q = dict(zip(ins, e.group_sum(Q))) if ins else {}
        self.sp_events[3].record()

    def note(self):
        super().note()
        STATS['calls'] += 1
        n, s = STATS['calls'], self.e.s
        if os.environ.get('RANK', '0') == '0' and (n == 1 or n % STATS['every'] == 0):
            print(f"speed_stack estimate {n}: exact {s.speed_exact} (c_fc cycles used {getattr(self, 'sp_fc', None)}), "
                  f'store {s.speed_store}, lite8 {s.speed_lite8}, pass: q_full {list(s.readout_q_full)}, c_fc '
                  f'{list(s.readout_fc_cycle + s.readout_hi_fc_cycle)}, skip {list(s.readout_skip_blocks)}, receivers '
                  f'{s.readout_k_receivers} by {s.readout_k_select}', flush=True)


# -------------------------------------------------------------------------------------------------------- install


def install(on=None):
    """combo_twin's install (combo_he_md's, fast_s2's items; idempotent), then this file's Settings / recipe dict and the
    Speed mixin first in fast_e3's shared Methods. The speed stack goes on after."""
    if R.Settings is Settings:
        R.Settings, R.asdict = CT.Settings, CT.asdict
    on = CT.install(on)
    if (R.Settings, R.asdict) != (CT.Settings, CT.asdict):
        raise RuntimeError('speed_stack: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Speed):
            FE.SHARED_CLASSES[kv] = type(f'Speed{cls.__name__}', (Speed, cls), {})
    return on


# ======================================== module transit_base
"""Transit decode. The mid roots take their chain credit and readout from one stream-only complete cycle instead of their
own cycles: the base moves are pushed to the loss once with the self-edge, recording every row's displacement at each
root; going down, root b's credit solves D_b g = s per token (D_b the recorded displacements, s the measured loss
changes and readout scores), with a few directly measured moves kept at their measurements. Only displacements the
pushes produced, only the stream basis; every number is a loss change or a hub score of a forward experiment.
"""
import contextlib
from dataclasses import dataclass
import math
import os
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import speed_stack as SS  # noqa: E402
import s2_momentgemm as S2M  # noqa: E402  (ieee)
CT, SP, FE, R, FC, RO = SS.CT, SS.SP, SS.FE, SS.R, SS.FC, SS.RO

STATS = dict(calls=0, every=200)
TRANSIT_SEED = 3_141_592  # the fixed start block of the weak-direction search (no record draw)
SE_CUT = ('        dkb = dk.to(tl.bfloat16)\n        total += (tl.sum(dk * tl.load(U + row * D + f)[None, :], 1) * (D ** -.5)\n'
          '                  + tl.sum(dv * tl.load(WV + row * D + f)[None, :], 1)\n'
          '                  + tl.sum(tl.dot(dkb, tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])) * dk, 1) * (1. / D)\n'
          '                  + tl.sum(tl.dot(dkb, tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])) * dv, 1) * (D ** -.5))\n'
          '    tl.store(Out + t * n + p, total, pm)\n')
SE_ONLY = SS.derive(RO.staged_kernel, 'staged_se_only_kernel', [(SE_CUT, '')])  # the self-edge (SELF) alone, no tables


@dataclass
class Settings(SS.Settings):  # speed_stack's fields and transit's
    transit: bool = False
    transit_roots: tuple = (1, 2, 3, 4, 5, 6, 7)  # mid roots decoded from the base (the others keep their cycles)
    transit_sigma: float = 0.001  # the base's one-sided step: this x the RMS of block 1's input, per token
    transit_direct: int = 16  # direct moves per transit root along its weakest-transmitted directions
    transit_dsigma: float = 0.001  # the direct moves' one-sided step (x the RMS of mid_b)
    transit_iters: int = 1  # subspace-iteration sweeps per root (the first root: max(iters + 1, 2) from a random block)
    transit_spare: int = 8  # spare columns of the search beyond transit_direct
    transit_store: str = 'fp32'  # 'stack' | 'bf16' | 'fp32' (module docstring)

    def __post_init__(self):
        super().__post_init__()
        self.transit_roots = tuple(self.transit_roots)
        if type(self.transit) is not bool or self.transit and not (
                SP.on(self) and CT.on(self) and self.twin_kc and self.twin_transport == 'identity'
                and self.twin_consumers == 'linear' and self.model.n_embd % 128 == 0
                and set(self.transit_roots) <= set(range(1, self.model.n_layer)) and 0 < self.transit_sigma < 1
                and 0 <= self.transit_direct and 0 < self.transit_dsigma < 1 and 0 <= self.transit_iters
                and 0 <= self.transit_spare and self.transit_direct + self.transit_spare <= 62
                and self.transit_store in ('stack', 'bf16', 'fp32')):
            raise ValueError('transit_base: transit true / false; true needs the shared stack, twin rows in kc mode '
                             '(identity transport, linear consumers), transit_roots within 1..n_layer - 1, steps in (0, 1), '
                             'transit_direct + transit_spare <= 62, transit_store stack / bf16 / fp32')


def asdict(settings):  # off: speed_stack's recipe dict
    state = SS.asdict(settings)
    if not getattr(settings, 'transit', False):
        for key in [k for k in state if k.startswith('transit')]:
            state.pop(key)
    return state


def on(s):
    return bool(getattr(s, 'transit', False))


# ------------------------------------------------------------------------------------------------------- the mixin


class Transit:
    """First in the MRO of fast_e3's shared Methods (on speed_stack's): combo_twin's chain (a copy, every transit line
    marked) with the roots of transit_roots decoded from the base. Transit off: the stack's chain, as it is."""

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        e, s = self.e, self.e.s
        if not on(s):
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        reason = self.unsupported()
        if reason is not None:
            raise RuntimeError(f'transit_base: the fast_chain kernels do not cover this setting ({reason})')
        model, d, dev = e.model, e.capture, e.device
        L, W, T, n = model.config.n_layer, e.width, d.length, emb.shape[0]
        self.sp_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.sp_events[0].record()
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P, scalars, h = hi - lo, e.part == 0, s.scalar_step
        self.chain_args = (emb, x0, mids, outs, routes, y)
        self.bases = [e.basis() for _ in range(L + 1)]  # the record's draws in its order (transit: mid1's is the base's)
        bases = [B[lo:hi].contiguous() for B in self.bases]
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb, *mids]]
        self.prepare(mids, routes)
        head = model.lm_head.weight.float()[:model.config.vocab_size]
        xL = outs[-1].float().contiguous()
        base, target_w, weight_t = xL @ head.T, head[y.clamp(min=0)], head.T.to(torch.bfloat16).contiguous()
        if FC.STATE['exact_blocks'] is None:
            FC.STATE['exact_blocks'] = torch.zeros(1, device=dev, dtype=torch.int32)
        hx = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1),
                  **{k: FC.STATE[k] for k in ('exact_head', 'series_limit', 'exact_blocks')})
        credit, score = torch.zeros(L + 1, n, W, device=dev), torch.zeros(L + 1, n, P, device=dev)  # transit: halves
        Xt = torch.zeros(L + 1, n, W, device=dev)  # transit: the decoded readouts (this GPU's tokens)
        lam = torch.zeros(L, 2, device=dev)
        m_chunk, k4 = min(s.readout_chunk, T), 4 * scalars  # the scalar probes per root (a pair's first GPU)
        roots = sorted(set(s.transit_roots))  # transit
        buf = self.buffers(m_chunk, max(P + L * k4, W if s.transit_store != 'fp32' else 0))  # transit: the base's rows
        tw = CT.Rows(self, m_chunk, P)  # twin
        ins = [i for i in s.twin_inputs if i in tw.first]  # twin: the input twins (x_i's first hop at block i)
        isc = torch.zeros(len(ins), n, P, device=dev)
        isteps = {i: s.chain_sigma * d.blocks[i].float().flatten(0, 1).square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                  for i in ins}
        tb0, tb1 = e.share(T)  # transit: this GPU's tokens of every sequence (the base, the direct moves, the solves)
        with self.transit_mode():  # transit: the per-estimate pieces of transit's experiments
            self.tx = dict(buf=buf if s.transit_store != 'fp32' else self.buffers(m_chunk, W), rows=CT.Rows(self, m_chunk, W),
                           y=y, head=lambda x, sl: x | hx | dict(xL=xL[sl], base=base[sl], target_w=target_w[sl], y=y[sl]),
                           chunks=[(t0, min(t0 + m_chunk, tb1)) for t0 in range(tb0, tb1, m_chunk)], tb=(tb0, tb1),
                           scale={(b, k): lam_r[b + 1:k + 1].prod() for b in tw.first for k in tw.kc if k > b})
        self.route_cache, self.ctx_cache = {}, {}
        for seq in range(d.batch):
            own = e.own_query_inputs(seq)
            self.route_cache[seq] = dict(own=own, routes=(s.query_keys, FC.query_routes(own, s.query_keys)))
            ctxs = [self.clean(seq, j) for j in range(L)]
            tr = self.transit_base(ctxs, seq, self.bases[2], roots) if roots else None  # transit
            for r in range(L - 1, -2, -1):  # root r: mid_r, from the top down; r = -1: the embedding (+ scalar probes)
                if tr is not None and roots[0] <= r < L - 1:  # transit: block r + 1's hub is measured
                    self.transit_score(tr, ctxs, seq, r + 1)
                if r in roots:  # transit: root r from the base
                    mine = slice(seq * T + tb0, seq * T + tb1)
                    credit[r + 1, mine], Xt[r + 1, mine] = self.transit_root(tr, ctxs, seq, r, steps[r + 1])
                else:
                    k = P + L * k4 if r < 0 else P
                    tw.phase()  # twin
                    for t0 in range(0, T, m_chunk):
                        t1 = min(t0 + m_chunk, T)
                        sl = slice(seq * T + t0, seq * T + t1)
                        x = self.tx['head'](self.chunk(buf, ctxs, seq, t0, t1), sl)
                        delta, dx0 = buf['delta'][:(t1 - t0) * k * W].view(t1 - t0, k, W), None
                        tw.chunk(t1 - t0, t0)  # twin
                        if r < 0:
                            dx0 = F.rms_norm(emb[sl, None] + steps[0][sl, None] * bases[0], (W,)) - x0[sl, None]
                            torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                        else:
                            torch.mul(steps[r + 1][sl, None], bases[r + 1], out=delta)
                            self.mlp(x, delta, P, r)
                        for b in range(r + 1, L):
                            n_b = P + b * k4 if r < 0 else P
                            score[r + 1, sl] += self.score(x, delta, n_b, b, self.SELF_EDGE)[:, :P]
                            tw.inject(x, n_b, P, b)  # twin: the rows' first hop at block b, from the scorer's projection
                            self.push(x, delta, n_b, b)
                            tw.second(x, b)  # twin: the twins' second hop at block b
                            tw.land(b)  # twin
                            if r < 0 and k4:  # the symmetric probes of block b's two mixing scalars start at mid_b
                                prev = (x0 if b == 0 else outs[b - 1])[sl]
                                delta[:, n_b:n_b + 4] = torch.stack([prev, x0[sl], -prev, -x0[sl]], 1) * h
                            self.mlp(x, delta, n_b + k4 * (r < 0), b, dx0, P)
                            tw.mlp(x, b)  # twin
                        change = FC.head_fast(delta, x) * (y[sl] >= 0).unsqueeze(1)
                        credit[r + 1, sl] = (change[:, :P] / steps[r + 1][sl]) @ bases[r + 1] / W
                        tw.collect(score[r + 1], seq * T)  # twin
                        if r < 0 and k4:  # the central difference over 2h per scalar
                            pair = change[:, P:].view(t1 - t0, L, 4) / h
                            lam += (pair[..., :2] - pair[..., 2:]).sum(0) / 2
                tok = e.group_sum(credit[r + 1, seq * T:(seq + 1) * T])  # the pair's halves: root r is complete
                if r >= 0:  # its hub's moment tables, read by every root below it
                    self.tables(ctxs[r], r, tok @ model.transformer.h[r].attn.c_proj.weight.float())
            del tr  # transit
            for j, i in enumerate(ins):  # twin: x_i's cycle (root i's basis), its first hop at block i, the twins after it
                tw.phase()
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1)
                    delta = buf['delta'][:(t1 - t0) * P * W].view(t1 - t0, P, W)
                    torch.mul(isteps[i][sl, None], bases[i + 1], out=delta)
                    tw.chunk(t1 - t0, t0)
                    self.project(x, delta, P, i)
                    tw.inject(x, P, P, i)
                    tw.land(i)
                    tw.mlp(x, i)
                    for b in range(i + 1, L):
                        tw.second(x, b)
                        tw.mlp(x, b)
                    tw.collect(isc[j], seq * T)
            self.ctx_cache[seq] = ctxs
        self.tx = None  # transit
        X = e.group_sum(torch.stack([Xt[k] if k - 1 in roots else (score[k] / steps[k]) @ bases[k] / W  # transit: Xt
                                     for k in range(L + 1)]))
        self.X = {'emb': X[0], **dict(enumerate(X[1:]))}
        self.I2 = dict(zip(ins, e.group_sum(torch.stack([(isc[j] / isteps[i]) @ bases[i + 1] / W for j, i in
                                                         enumerate(ins)])))) if ins else {}  # twin
        e.group_sum(lam)
        self.sp_events[1].record()
        STATS['calls'] += 1
        if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % STATS['every'] == 0):
            print(f'transit_base estimate {STATS["calls"]}: roots {roots} from the block-1 base (sigma {s.transit_sigma}, '
                  f'{s.transit_direct} direct moves at {s.transit_dsigma}, {s.transit_iters} sweeps, store '
                  f'{s.transit_store}), the others by their cycles; twins at the keys of {list(tw.kc)}', flush=True)
        return {'emb': credit[0], **dict(enumerate(credit[1:])), **{('lam', r): lam[r] for r in range(L)}}

    @contextlib.contextmanager
    def transit_mode(self):
        """transit's experiments: transit_store 'bf16' runs them with speed_lite8 off, 'fp32' also speed_store off."""
        s = self.e.s
        flags = s.speed_store, s.speed_lite8
        s.speed_lite8 = s.speed_lite8 and s.transit_store == 'stack'
        s.speed_store = s.speed_store and s.transit_store != 'fp32'
        try:
            yield
        finally:
            s.speed_store, s.speed_lite8 = flags

    def transit_base(self, ctxs, seq, Hb, roots):
        """The stream-only complete cycle at block 1's input for this GPU's tokens of sequence seq (module docstring):
        returns the per-sequence state (the steps, the loss changes per unit step, the transposed unit-step rows at the
        transit roots' mid_b, the raw rows at block inputs lowest + 1 .. L - 1, the readout sums and the twin sums)."""
        e, s, tx = self.e, self.e.s, self.tx
        L, W, T, dev = e.model.config.n_layer, e.width, e.capture.length, e.device
        (tb0, tb1), buf, S0 = tx['tb'], tx['buf'], seq * T
        nb, lowest = tb1 - tb0, roots[0]
        xin = e.capture.blocks[1][seq].float()
        bstep = s.transit_sigma * xin[tb0:tb1].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        tr = dict(bstep=bstep, yv=torch.empty(nb, W, device=dev), c=torch.zeros(nb, W, device=dev), prev=None, live=set(),
                  dm={b: torch.empty(nb, W, W, device=dev, dtype=buf['delta'].dtype) for b in roots},
                  din={b: torch.empty(nb, W, W, device=dev, dtype=buf['delta'].dtype) for b in range(lowest + 1, L)},
                  twin={k: torch.zeros(nb, W, W, device=dev) for k in tx['rows'].kc if k > lowest + 1})
        with self.transit_mode():
            for t0, t1 in tx['chunks']:
                m, sl, loc = t1 - t0, slice(S0 + t0, S0 + t1), slice(t0 - tb0, t1 - tb0)
                x = tx['head'](self.chunk(buf, ctxs, seq, t0, t1), sl)
                delta = buf['delta'][:m * W * W].view(m, W, W)
                torch.mul(bstep[loc, None], Hb, out=delta)
                for b in range(1, L):
                    if b in tr['din']:
                        tr['din'][b][loc] = delta
                    self.transit_se(x, delta, b)
                    self.push(x, delta, W, b)
                    if b in tr['dm']:
                        transit_tdiv_kernel[(W // 64, W // 64, m)](delta, tr['dm'][b][loc], W, 64, num_warps=4)
                    self.mlp(x, delta, W, b)
                tr['yv'][loc] = FC.head_fast(delta, x) * (tx['y'][sl] >= 0).unsqueeze(1) / bstep[loc]
        return tr

    def transit_se(self, x, delta, b):
        """score()'s self-edge alone at block b (no hub yet): x['se'] and what push() reads (row statistics, the
        projection's input copy: Speed.score8's fp8 one under lite8, else Shared.score's bf16 one)."""
        m, rows, W = delta.shape
        c, H, nc = x['ctxs'][b], self.e.model.config.n_head, self.sp_w['wcat'][b].shape[1]
        up = x['up'][:m * rows * nc].view(m * rows, nc)
        if self.e.s.speed_lite8:
            SS.STATS8[(triton.cdiv(rows, 16), m)](delta, c['x'], c['sqx'], x['db8'], x['sa'], x['st'], x['t0'], rows, rows,
                                                  W, 16, RO.EPS, num_warps=4)
            w8 = self.sp_w['w8']
            SS.smm(x['db8'][:m * rows * W].view(m * rows, W), w8['cat'][b].T, x['sa'][:m * rows].view(m * rows, 1),
                   w8['cat_s'][b].view(1, nc), up)
        else:
            RO.stats_kernel[(triton.cdiv(rows, 16), m)](delta, c['x'], c['sqx'], x['db'], x['st'], x['t0'], rows, rows, W,
                                                        16, RO.EPS, num_warps=4)
            torch.mm(x['db'][:m * rows * W].view(m * rows, W), self.sp_w['wcat'][b], out=up)
        z, gated = x['st'], c['gate'] is not None  # z: the pointers the cut lines read
        SE_ONLY[(triton.cdiv(rows, 64), m)](
            up, x['st'], c['R0'], c['VC'], c['GZ'] if gated else z, c['ve'] if gated else z, self.cs_all, self.sn_all, z, z,
            z, z, c['QD'], z, c['AD'], z, c['VT'], c['OT'], z, x['se'], x['t0'], rows, rows, W, H, W // H, gated, False,
            True, 64, nc, num_warps=4, num_stages=2)

    def transit_score(self, tr, ctxs, seq, b):
        """Block b's hub is measured: the base rows' readout scores there (receivers s > t, from their stored block-b
        input) per unit step into the readout sums, and their twins' first hop at block b into the twin sums of the kc
        blocks above (carried there by the residual scalars: combo_twin's identity transport)."""
        tx, W = self.tx, self.e.width
        din, buf, rw, tb0 = tr['din'].pop(b), tx['buf'], tx['rows'], tx['tb'][0]
        kc = [k for k in tr['twin'] if k > b] if b in rw.first else []
        with self.transit_mode():
            for t0, t1 in tx['chunks']:
                m, loc = t1 - t0, slice(t0 - tb0, t1 - tb0)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * W * W].view(m, W, W)
                delta.copy_(din[loc])
                tr['c'][loc] += self.score(x, delta, W, b, False) / tr['bstep'][loc]
                if kc:
                    rw.m, rw.t0 = m, t0
                    rw.inject(x, W, W, b)
                    for k in kc:
                        tr['twin'][k][loc].addcmul_(rw.inj[1], tx['scale'][b, k])
                    rw.inj = None
        tr['live'].update(kc)

    def transit_twins(self, ctxs, seq, sums, rows):
        """The twins' second hop (combo_twin's kc mode): token t's twin sum at each kc block sits in slot t + 1 and is
        key-scored at the receivers u > t + 1 against that block's hub (the sequence's last token has none) -> [nb, rows]."""
        tx, T = self.tx, self.e.capture.length
        (tb0, tb1), rw = tx['tb'], tx['rows']
        out = torch.zeros(tb1 - tb0, rows, device=self.e.device)
        with self.transit_mode():
            for k, z in sums.items():
                for t0, t1 in tx['chunks']:
                    t1 = min(t1, T - 1)
                    if t1 > t0:
                        loc = slice(t0 - tb0, t1 - tb0)
                        rw.twin, rw.m = z[loc], t1 - t0
                        out[loc] += rw.key_score(self.chunk(tx['buf'], ctxs, seq, t0 + 1, t1 + 1), rows, k)
        rw.twin = None
        return out

    def transit_root(self, tr, ctxs, seq, r, dstep):
        """Root mid_r from the base (module docstring) -> (credit, readout) of this GPU's tokens, [nb, W] each."""
        e, s, tx = self.e, self.e.s, self.tx
        L, W, K, dev = e.model.config.n_layer, e.width, s.transit_direct, e.device
        (tb0, tb1), S0 = tx['tb'], seq * e.capture.length
        nb = tb1 - tb0
        live = {k: tr['twin'][k] for k in sorted(tr['live'])}  # the twin sums with first hops above r
        rhs = tr['c'] + self.transit_twins(ctxs, seq, live, W) / tr['bstep'] if live else tr['c']
        Q, iters = None, s.transit_iters
        if K:
            Q = transit_x0(r, W, K + s.transit_spare, dev).expand(nb, W, K + s.transit_spare)
            if tr['prev'] is None:  # the first root: one more sweep from the random block, at least 2
                iters = max(iters + 1, 2)
            else:  # the root above's directions + the spare columns
                with S2M.ieee():
                    Q = transit_orth_t(torch.cat([tr['prev'], Q[..., K:]], -1))
        with S2M.ieee():
            sol, U, Zu, lamu = transit_fast(tr['dm'].pop(r), torch.stack([tr['yv'], rhs], -1) * tr['bstep'][..., None],
                                            Q, K, iters)  # D g = bstep s: the same g, U and Zu / lam
        g, xr = sol[..., 0], (sol[..., 1] if r < L - 1 else torch.zeros_like(sol[..., 1]))
        if not K:
            return g, xr
        tr['prev'] = U
        dst = dstep[S0 + tb0:S0 + tb1] * math.sqrt(W) * (s.transit_dsigma / s.chain_sigma)  # [nb, 1]: a move's length
        meas, xm = self.transit_direct(ctxs, seq, r, U, dst)
        with S2M.ieee():
            return transit_keep(g, U, Zu, lamu, meas), (transit_keep(xr, U, Zu, lamu, xm) if r < L - 1 else xr)

    def transit_direct(self, ctxs, seq, r, U, dst):
        """Moves of mid_r along U [nb, W, K] (one-sided, length dst [nb, 1]) pushed and scored like a root's rows (the
        self-edge, the receivers at every later hub), with their twins -> (loss changes, readout scores) per unit length,
        [nb, K] each."""
        e, tx = self.e, self.tx
        L, W, K, T, dev = e.model.config.n_layer, e.width, U.shape[-1], e.capture.length, e.device
        (tb0, tb1), buf, rw, S0 = tx['tb'], tx['buf'], tx['rows'], seq * T
        meas, xm = torch.empty(tb1 - tb0, K, device=dev), torch.zeros(tb1 - tb0, K, device=dev)
        sums = {k: torch.zeros(tb1 - tb0, K, W, device=dev) for k in rw.kc if k > r + 1 and r + 1 in rw.first}
        with self.transit_mode():
            for t0, t1 in tx['chunks']:
                m, sl, loc = t1 - t0, slice(S0 + t0, S0 + t1), slice(t0 - tb0, t1 - tb0)
                x = tx['head'](self.chunk(buf, ctxs, seq, t0, t1), sl)
                delta = buf['delta'][:m * K * W].view(m, K, W)
                torch.mul(dst[loc, None], U[loc].transpose(1, 2), out=delta)
                self.mlp(x, delta, K, r)
                for b in range(r + 1, L):
                    xm[loc] += self.score(x, delta, K, b, True)
                    kc = [k for k in sums if k > b] if b in rw.first else []
                    if kc:  # twin: the rows' first hop at block b
                        rw.m, rw.t0 = m, t0
                        rw.inject(x, K, K, b)
                        for k in kc:
                            sums[k][loc].addcmul_(rw.inj[1], tx['scale'][b, k])
                        rw.inj = None
                    self.push(x, delta, K, b)
                    self.mlp(x, delta, K, b)
                meas[loc] = FC.head_fast(delta, x) * (tx['y'][sl] >= 0).unsqueeze(1)
        if sums:
            xm += self.transit_twins(ctxs, seq, sums, K)
        return meas / dst, xm / dst


# ---------------------------------------------------------------------- the solver (rec_transit_final.py, names kept)


def transit_x0(r, W, c, dev):
    """Root r's fixed start block of the search (no record draw), made once per root and size. Its QR is the process's
    first CUDA linalg call, where torch dlopens libtorch_cuda_linalg.so lazily (that failed once, transiently, in the
    checks: No such file or directory): retried."""
    key = (r, W, c, dev)
    if key not in TRANSIT_X0:
        z = torch.randn(W, c, device=dev, generator=torch.Generator(dev).manual_seed(TRANSIT_SEED + 7919 * r))
        for i in range(6):
            try:
                TRANSIT_X0[key] = torch.linalg.qr(z)[0]
                break
            except RuntimeError as exc:
                if 'dlopen' not in str(exc) or i == 5:
                    raise
                time.sleep(2)
    return TRANSIT_X0[key]


def transit_keep(g, U, Zu, lam, meas):
    """The constrained least squares: g (= M^-1 y) moved to U^T g = meas with the least change of |M g - y|:
    g - (M^T M)^-1 U (U^T (M^T M)^-1 U)^-1 (U^T g - meas)."""
    ok = lam > lam.amax(-1, keepdim=True) * 1e-12
    r = torch.where(ok, ((U.transpose(1, 2) @ g.unsqueeze(-1))[..., 0] - meas) / lam.clamp_min(1e-30), 0.)
    return g - (Zu @ r.unsqueeze(-1))[..., 0]


TRANSIT_TG, TRANSIT_TC = 128, 64  # transit_solver 'fast': Gram tile, Cholesky / substitution tile


@triton.jit
def transit_split16(x):
    """x fp32 -> fp16 (hi, lo) with x 2^s = hi + lo (~22 bits), 2^s putting max |x| near 2^14; and 2^-s."""
    sc = tl.math.exp2(14. - tl.math.ceil(tl.math.log2(tl.maximum(tl.max(tl.max(tl.abs(x), 1), 0), 1e-30))))
    xs = x * sc
    hi = xs.to(tl.float16)
    return hi, (xs - hi.to(tl.float32)).to(tl.float16), 1. / sc


@triton.jit
def transit_tdiv_kernel(D, O, W: tl.constexpr, BT: tl.constexpr):
    """O[t] = D[t]^T: a token's displacement rows stored transposed (coordinates x moves; transit_base: raw, in the
    moves' dtype, the step folded into the right-hand sides), so the Gram contracts over the contiguous index (K-major
    tensor-core operands)."""
    rb, cb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r, c = rb * BT + tl.arange(0, BT), cb * BT + tl.arange(0, BT)
    x = tl.load(D + t.to(tl.int64) * W * W + r[:, None] * W + c[None, :])
    tl.store(O + t.to(tl.int64) * W * W + c[None, :] * W + r[:, None], x)


@triton.jit
def transit_gramt_kernel(MT, Y, G, R, TRP, TI, TJ, n: tl.constexpr, NY: tl.constexpr, T: tl.constexpr,
                         BK: tl.constexpr, NBG: tl.constexpr, FP16: tl.constexpr, HI: tl.constexpr):
    """transit_gram_kernel from the transposed rows MT (= M^T, coordinates x moves): tile (i, j) of MT MT^T with
    K-major operands; FP16: hi/lo fp16 split of power-of-2 scaled tiles (3 products), else 3xTF32; HI (transit_base:
    bf16 rows, whose scaled tiles fp16 holds exactly: lo ~ 0) the hi product alone. The diagonal tiles also R = MT Y and
    their trace."""
    t, b = tl.program_id(0), tl.program_id(1)
    i, j = tl.load(TI + t), tl.load(TJ + t)
    rk, rt, ry = tl.arange(0, BK), tl.arange(0, T), tl.arange(0, 16)
    mb = MT + b.to(tl.int64) * n * n
    acc = tl.zeros((T, T), tl.float32)
    for k0 in range(0, n, BK):
        a = tl.load(mb + (i * T + rt)[:, None] * n + (k0 + rk)[None, :]).to(tl.float32)  # (transit_base: bf16 rows)
        c = tl.load(mb + (j * T + rt)[:, None] * n + (k0 + rk)[None, :]).to(tl.float32)
        if HI:
            ah, _, sa = transit_split16(a)
            ch, _, sc = transit_split16(c)
            acc += tl.dot(ah, tl.trans(ch)) * (sa * sc)
        elif FP16:
            ah, al, sa = transit_split16(a)
            ch, cl, sc = transit_split16(c)
            p = tl.dot(ah, tl.trans(ch))
            p = tl.dot(ah, tl.trans(cl), p)
            acc += tl.dot(al, tl.trans(ch), p) * (sa * sc)
        else:
            acc = tl.dot(a, tl.trans(c), acc, input_precision='tf32x3')
    tl.store(G + b.to(tl.int64) * n * n + (i * T + rt)[:, None] * n + (j * T + rt)[None, :], acc)
    if i == j:
        yb = Y + b.to(tl.int64) * n * NY
        accr = tl.zeros((T, 16), tl.float32)
        for k0 in range(0, n, BK):
            a = tl.load(mb + (i * T + rt)[:, None] * n + (k0 + rk)[None, :]).to(tl.float32)
            yk = tl.load(yb + (k0 + rk)[:, None] * NY + ry[None, :], mask=ry[None, :] < NY, other=0.)
            accr = tl.dot(a, yk, accr, input_precision='tf32x3')
        tl.store(R + b.to(tl.int64) * n * NY + (i * T + rt)[:, None] * NY + ry[None, :], accr, mask=ry[None, :] < NY)
        tl.store(TRP + b * NBG + i, tl.sum(tl.sum(tl.where(rt[:, None] == rt[None, :], acc, 0.), 1), 0))


@triton.jit
def transit_mtile(mb, i, j, n: tl.constexpr, T: tl.constexpr):
    """A token's Gram tile (i, j) recomputed from its transposed rows MT (the refactoring pass: G's lower tiles already
    hold L)."""
    rt, rk = tl.arange(0, T), tl.arange(0, 32)
    acc = tl.zeros((T, T), tl.float32)
    for k0 in range(0, n, 32):
        a = tl.load(mb + (i * T + rt)[:, None] * n + (k0 + rk)[None, :]).to(tl.float32)
        acc = tl.dot(a, tl.trans(tl.load(mb + (j * T + rt)[:, None] * n + (k0 + rk)[None, :]).to(tl.float32)), acc,
                     input_precision='tf32x3')
    return acc


@triton.jit
def transit_refactor_kernel(G, TRP, MT, DI, INFO, ridge, n: tl.constexpr, T: tl.constexpr, NB: tl.constexpr,
                            NBG: tl.constexpr):
    """transit_factor's retry for the tokens whose first factorization met a pivot <= 0 (INFO 1; the others return at
    once): G + ridge mean(diag G) I = L L^T with the Gram tiles recomputed from the transposed rows MT, left-looking
    by T-blocks (one register tile at a time: a diagonal block's columns go to L as they are made, its inverse is built
    from L's rows read back); DI = the inverse diagonal blocks; INFO 2: still not positive."""
    b = tl.program_id(0)
    gb = G + b.to(tl.int64) * n * n
    mb = MT + b.to(tl.int64) * n * n
    db = DI + b.to(tl.int64) * NB * T * T
    rt = tl.arange(0, T)
    rows, cols = rt[:, None], rt[None, :]
    if tl.load(INFO + b) != 0:
        lam = ridge * tl.sum(tl.load(TRP + b * NBG + tl.arange(0, NBG)), 0) / n
        bad = tl.full([], 0, tl.int32)
        for j in range(NB):
            A = transit_mtile(mb, j, j, n, T)
            A = tl.where(rows == cols, A + lam, A)
            for k in range(j):
                Ljk = tl.load(gb + (j * T + rows) * n + k * T + cols)
                A -= tl.dot(Ljk, tl.trans(Ljk), input_precision='tf32x3')
            ld = gb + (j * T + rt) * n + j * T  # column c of L's diagonal block: ld + c
            for c in range(T):
                col = tl.sum(tl.where(cols == c, A, 0.), 1)
                d = tl.sum(tl.where(rt == c, col, 0.), 0)
                bad = tl.where(d > 0., bad, 1)
                lc = tl.where(rt >= c, col / tl.sqrt_rn(tl.maximum(d, 1e-30)), 0.)
                A -= lc[:, None] * lc[None, :]
                tl.store(ld + c, lc)
            tl.debug_barrier()
            X = tl.zeros((T, T), tl.float32)
            for r in range(T):
                lrow = tl.load(gb + (j * T + r) * n + j * T + rt)  # L's row r (zeros above the diagonal)
                v = tl.sum(lrow[:, None] * X, 0)
                drr = tl.sum(tl.where(rt == r, lrow, 0.), 0)
                X = tl.where(rows == r, ((rt == r).to(tl.float32) - v)[None, :] / drr, X)
            tl.store(db + j * T * T + rows * T + cols, X)
            tl.debug_barrier()
            for i in range(j + 1, NB):
                A2 = transit_mtile(mb, i, j, n, T)
                for k in range(j):
                    Lik = tl.load(gb + (i * T + rows) * n + k * T + cols)
                    Ljk = tl.load(gb + (j * T + rows) * n + k * T + cols)
                    A2 -= tl.dot(Lik, tl.trans(Ljk), input_precision='tf32x3')
                Xt = tl.load(db + j * T * T + cols * T + rows)  # DI_j^T, reloaded (not live across the loop)
                tl.store(gb + (i * T + rows) * n + j * T + cols, tl.dot(A2, Xt, input_precision='tf32x3'))
            tl.debug_barrier()
        tl.store(INFO + b, bad + 1)


@triton.jit
def transit_chol_inv32(A, bad):
    """Cholesky factor L and its inverse X of a 32 x 32 SPD register tile (fp32; one pass over the columns: L's row c
    is complete after column c, so X's row c follows). Pivots <= 0 set bad."""
    rc = tl.arange(0, 32)
    rows, cols = rc[:, None], rc[None, :]
    Lt = tl.zeros((32, 32), tl.float32)
    X = tl.zeros((32, 32), tl.float32)
    for c in range(32):
        col = tl.sum(tl.where(cols == c, A, 0.), 1)
        d = tl.sum(tl.where(rc == c, col, 0.), 0)
        bad = tl.where(d > 0., bad, 1)
        sd = tl.sqrt_rn(tl.maximum(d, 1e-30))
        lc = tl.where(rc >= c, col / sd, 0.)
        A -= lc[:, None] * lc[None, :]
        Lt = tl.where(cols == c, lc[:, None], Lt)
        lrow = tl.sum(tl.where(rows == c, Lt, 0.), 0)
        v = tl.sum(lrow[:, None] * X, 0)
        X = tl.where(rows == c, ((rc == c).to(tl.float32) - v)[None, :] / sd, X)
    return Lt, X, bad


@triton.jit
def transit_diag_kernel(G, TRP, DI, INFO, ridge, j: tl.constexpr, n: tl.constexpr, NB: tl.constexpr,
                        NBG: tl.constexpr):
    """Token b: the diagonal 64-block j of G (already updated by the earlier panels) + ridge mean(diag G) I = L L^T as
    2 x 2 blocks of 32 (one warp: every reduction stays in the warp); DI_j = L^-1 -> DI; a pivot <= 0 -> INFO 1."""
    b = tl.program_id(0)
    r = tl.arange(0, 32)
    rows, cols = r[:, None], r[None, :]
    lam = ridge * tl.sum(tl.load(TRP + b * NBG + tl.arange(0, NBG)), 0) / n
    gb = G + b.to(tl.int64) * n * n + (j * 64) * n + j * 64
    A11 = tl.load(gb + rows * n + cols)
    A21 = tl.load(gb + (32 + rows) * n + cols)
    A22 = tl.load(gb + (32 + rows) * n + 32 + cols)
    A11 = tl.where(rows == cols, A11 + lam, A11)
    A22 = tl.where(rows == cols, A22 + lam, A22)
    bad = tl.load(INFO + b)
    L11, X11, bad = transit_chol_inv32(A11, bad)
    L21 = tl.dot(A21, tl.trans(X11), input_precision='tf32x3')
    S = A22 - tl.dot(L21, tl.trans(L21), input_precision='tf32x3')
    L22, X22, bad = transit_chol_inv32(S, bad)
    X21 = -tl.dot(tl.dot(X22, L21, input_precision='tf32x3'), X11, input_precision='tf32x3')
    db = DI + b.to(tl.int64) * NB * 4096 + j * 4096
    tl.store(db + rows * 64 + cols, X11)
    tl.store(db + rows * 64 + 32 + cols, tl.zeros((32, 32), tl.float32))
    tl.store(db + (32 + rows) * 64 + cols, X21)
    tl.store(db + (32 + rows) * 64 + 32 + cols, X22)
    tl.store(INFO + b, bad)


@triton.jit
def transit_panel_kernel(G, DI, j: tl.constexpr, n: tl.constexpr, NB: tl.constexpr):
    """Program (p, b): row block i = j + 1 + p of block column j: L_ij = (G_ij - sum_k<j L_ik L_jk^T) DI_j^T, then the
    right-looking update of row i's diagonal tile G_ii -= L_ij L_ij^T (it is factored at step i)."""
    p, b = tl.program_id(0), tl.program_id(1)
    i = j + 1 + p
    gb = G + b.to(tl.int64) * n * n
    rt = tl.arange(0, 64)
    rows, cols = rt[:, None], rt[None, :]
    A2 = tl.load(gb + (i * 64 + rows) * n + j * 64 + cols)
    for k in range(j):
        Lik = tl.load(gb + (i * 64 + rows) * n + k * 64 + cols)
        Ljk = tl.load(gb + (j * 64 + rows) * n + k * 64 + cols)
        A2 -= tl.dot(Lik, tl.trans(Ljk), input_precision='tf32x3')
    Xt = tl.load(DI + b.to(tl.int64) * NB * 4096 + j * 4096 + cols * 64 + rows)
    Lij = tl.dot(A2, Xt, input_precision='tf32x3')
    tl.store(gb + (i * 64 + rows) * n + j * 64 + cols, Lij)
    gii = gb + (i * 64 + rows) * n + i * 64 + cols
    tl.store(gii, tl.load(gii) - tl.dot(Lij, tl.trans(Lij), input_precision='tf32x3'))


@triton.jit
def transit_solve_kernel(L, DI, X, n: tl.constexpr, T: tl.constexpr, NB: tl.constexpr, R: tl.constexpr):
    """Token b: X [n, R] <- L^-T L^-1 X in place (blocked substitution; diagonal blocks by their inverses DI)."""
    b = tl.program_id(0)
    lb = L + b.to(tl.int64) * n * n
    db = DI + b.to(tl.int64) * NB * T * T
    xb = X + b.to(tl.int64) * n * R
    rt, rr = tl.arange(0, T), tl.arange(0, R)
    rows, cols = rt[:, None], rt[None, :]
    for i in range(NB):
        acc = tl.load(xb + (i * T + rows) * R + rr[None, :])
        for k in range(i):
            Lik = tl.load(lb + (i * T + rows) * n + k * T + cols)
            acc -= tl.dot(Lik, tl.load(xb + (k * T + rows) * R + rr[None, :]), input_precision='tf32x3')
        Di = tl.load(db + i * T * T + rows * T + cols)
        tl.store(xb + (i * T + rows) * R + rr[None, :], tl.dot(Di, acc, input_precision='tf32x3'))
        tl.debug_barrier()
    for ii in range(NB):
        i = NB - 1 - ii
        acc = tl.load(xb + (i * T + rows) * R + rr[None, :])
        for kk in range(ii):
            k = NB - 1 - kk
            Lki = tl.load(lb + (k * T + rows) * n + i * T + cols)
            acc -= tl.dot(tl.trans(Lki), tl.load(xb + (k * T + rows) * R + rr[None, :]), input_precision='tf32x3')
        Di = tl.load(db + i * T * T + rows * T + cols)
        tl.store(xb + (i * T + rows) * R + rr[None, :], tl.dot(tl.trans(Di), acc, input_precision='tf32x3'))
        tl.debug_barrier()


@triton.jit
def transit_chol_inv_dead(A, CP: tl.constexpr, rel):
    """Inverse lower Cholesky factor of a CP x CP PSD register tile (fp64) with dependent columns zeroed: a column
    whose pivot is <= rel x its original diagonal (or not positive) is dead (unit pivot, no update; its row of the
    inverse zeroed: the orthonormalized column becomes 0, as transit_orth's clamped directions)."""
    rc = tl.arange(0, CP)
    rows, cols = rc[:, None], rc[None, :]
    d0 = tl.sum(tl.where(rows == cols, A, 0.), 1)
    Lt = tl.zeros((CP, CP), tl.float64)
    dead = tl.zeros([CP], tl.int32)
    for c in range(CP):
        col = tl.sum(tl.where(cols == c, A, 0.), 1)
        d = tl.sum(tl.where(rc == c, col, 0.), 0)
        dc = tl.sum(tl.where(rc == c, d0, 0.), 0)
        deadc = tl.where(d <= rel * dc, 1, 0)
        dead = tl.where(rc == c, deadc, dead)
        bad = deadc > 0
        sd = tl.sqrt(tl.where(bad, 1., d))
        lc = tl.where(bad, (rc == c).to(tl.float64), tl.where(rc >= c, col / sd, 0.))
        A -= tl.where(bad, 0., lc[:, None] * lc[None, :])
        Lt = tl.where(cols == c, lc[:, None], Lt)
    X = tl.zeros((CP, CP), tl.float64)
    for r in range(CP):
        lrow = tl.sum(tl.where(rows == r, Lt, 0.), 0)
        v = tl.sum(lrow[:, None] * X, 0)
        drr = tl.sum(tl.where(rc == r, lrow, 0.), 0)
        X = tl.where(rows == r, ((rc == r).to(tl.float64) - v)[None, :] / drr, X)
    return tl.where(dead[:, None] > 0, 0., X)


@triton.jit
def transit_orth_kernel(Z, Q, n: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, RB: tl.constexpr):
    """Token b: Q [n, CP] = transit_orth of Z [n, CP] (first C columns; the rest zero) by CholeskyQR2: the Gram in
    3xTF32 (as transit_orth's fp32 Gram), fp64 pivots; a column whose pivot is <= 1e-6 of its own squared norm is
    numerically dependent and becomes 0 (transit_orth: its clamped directions)."""
    b = tl.program_id(0)
    zb = Z + b.to(tl.int64) * n * CP
    qb = Q + b.to(tl.int64) * n * CP
    rc, rr = tl.arange(0, CP), tl.arange(0, RB)
    rows, cols = rc[:, None], rc[None, :]
    for p in tl.static_range(2):
        src = zb if p == 0 else qb
        A = tl.zeros((CP, CP), tl.float32)
        for r0 in range(0, n, RB):
            zt = tl.load(src + (r0 + rr)[:, None] * CP + rc[None, :])
            A = tl.dot(tl.trans(zt), zt, A, input_precision='tf32x3')
        A = A.to(tl.float64)
        A = (A + tl.trans(A)) * .5
        A = tl.where((rows == cols) & (rows >= C), 1., A)
        LiT = tl.trans(transit_chol_inv_dead(A, CP, 1e-6)).to(tl.float32)
        tl.debug_barrier()
        for r0 in range(0, n, RB):
            zt = tl.load(src + (r0 + rr)[:, None] * CP + rc[None, :])
            tl.store(qb + (r0 + rr)[:, None] * CP + rc[None, :], tl.dot(zt, LiT, input_precision='tf32x3'))
        tl.debug_barrier()


def transit_pad(Z, CP):
    out = torch.zeros(*Z.shape[:-1], CP, device=Z.device)
    out[..., :Z.shape[-1]] = Z
    return out


def transit_orth_t(Z):
    """transit_orth by transit_orth_kernel (CholeskyQR2): Z [m, W, c] -> [m, W, c]."""
    m, n, c = Z.shape
    CP = max(16, 1 << (c - 1).bit_length())
    Zp = transit_pad(Z, CP)
    Q = torch.empty_like(Zp)
    transit_orth_kernel[(m,)](Zp, Q, n, c, CP, 32, num_warps=1, num_stages=1)
    return Q[..., :c]


TRANSIT_TILES, TRANSIT_X0 = {}, {}


def transit_fast_gram(MT, Y, fp16=True):
    """The transposed rows MT [m, W, W] (= M^T), Y [m, W, NY] -> G (lower 128-tiles of M^T M; fp16-split or
    3xTF32 products), R = M^T Y, the trace parts [m, W / 128]."""
    m, n, _ = MT.shape
    NY, nbg, dev = Y.shape[-1], n // TRANSIT_TG, MT.device
    if (nbg, dev) not in TRANSIT_TILES:
        ij = [(i, j) for i in range(nbg) for j in range(i + 1)]
        TRANSIT_TILES[nbg, dev] = [torch.tensor(v, dtype=torch.int32, device=dev) for v in zip(*ij)]
    TI, TJ = TRANSIT_TILES[nbg, dev]
    G, R, TRP = torch.empty(m, n, n, device=dev), torch.empty(m, n, NY, device=dev), torch.empty(m, nbg, device=dev)
    transit_gramt_kernel[(len(TI), m)](MT, Y.contiguous(), G, R, TRP, TI, TJ, n, NY, TRANSIT_TG, 32, nbg, fp16,
                                       MT.dtype == torch.bfloat16, num_warps=8, num_stages=3)
    return G, R, TRP


def transit_fast_chol(G, TRP, MT):
    """In place: G's lower off-diagonal tiles <- L of G + 1e-7 mean(diag G) I and DI = the inverse diagonal blocks, by
    block columns (a 1-warp program per token factors the diagonal block, then one program per tile below makes L_ij
    and updates row i's diagonal tile); tokens with a pivot <= 0 are refactored with + 1e-4 (transit_factor's retry).
    -> DI, INFO (0: first pass, 1: refactored)."""
    m, n, _ = G.shape
    NB, nbg = n // TRANSIT_TC, n // TRANSIT_TG
    DI = torch.empty(m, NB, TRANSIT_TC, TRANSIT_TC, device=G.device)
    INFO = torch.zeros(m, dtype=torch.int32, device=G.device)
    for j in range(NB):
        transit_diag_kernel[(m,)](G, TRP, DI, INFO, 1e-7, j, n, NB, nbg, num_warps=1, num_stages=1)
        if j + 1 < NB:
            transit_panel_kernel[(NB - 1 - j, m)](G, DI, j, n, NB, num_warps=4, num_stages=1)
    transit_refactor_kernel[(m,)](G, TRP, MT, DI, INFO, 1e-7 + 1e-4, n, TRANSIT_TC, NB, nbg, num_warps=4,
                                  num_stages=1)
    return DI, INFO


def transit_fast_solve(L, DI, B):
    """(L L^T)^-1 B for B [m, W, c] (padded to 16 / 32 / 64 columns)."""
    m, n, c = B.shape
    R = max(16, 1 << (c - 1).bit_length())
    X = torch.zeros(m, n, R, device=B.device)
    X[..., :c] = B
    transit_solve_kernel[(m,)](L, DI, X, n, TRANSIT_TC, n // TRANSIT_TC, R, num_warps=4, num_stages=1)
    return X[..., :c]


def transit_ritz(Q, Z, k):
    """transit_weak's Rayleigh-Ritz step on span(Q) (Q orthonormal, Z = (M^T M)^-1 Q): the k largest."""
    B = (Q.transpose(1, 2) @ Z).double()
    B = (B + B.transpose(1, 2)) / 2
    B = B + B.diagonal(dim1=1, dim2=2).abs().mean(-1)[:, None, None] * 1e-12 * torch.diag(
        torch.linspace(1, 2, B.shape[-1], device=B.device, dtype=B.dtype))
    ev, V = torch.linalg.eigh(B)
    V, ev = V[..., -k:].float(), ev[..., -k:].float()
    return Q @ V, Z @ V, ev


def transit_fast(MT, Y, Q, k, iters, fp16=True):
    """transit_factor 'chol' + transit_msolve + transit_weak in fused kernels, from the transposed rows MT: the
    least-squares solutions (M^T M)^-1 M^T Y [m, W, NY] and (k > 0) the k weakest directions U, Zu = (M^T M)^-1 U, lam
    by iters sweeps of subspace iteration from Q [m, W, c] (orthonormal columns; the first application rides with the
    right-hand sides; transit_orth_t between sweeps, transit_ritz at the end)."""
    NY = Y.shape[-1]
    G, R, TRP = transit_fast_gram(MT, Y, fp16)
    DI, INFO = transit_fast_chol(G, TRP, MT)
    sol = transit_fast_solve(G, DI, R if not k else torch.cat([R, Q], -1))
    if not k:
        return sol, None, None, None
    Z = sol[..., NY:]
    for _ in range(iters):
        Q = transit_orth_t(Z)
        Z = transit_fast_solve(G, DI, Q)
    return (sol[..., :NY],) + transit_ritz(Q, Z, k)


# -------------------------------------------------------------------------------------------------------- install


def install(on_items=None):
    """speed_stack's install (combo_twin's, combo_he_md's, fast_s2's items; idempotent), then this file's Settings /
    recipe dict and the Transit mixin first in fast_e3's shared Methods. The speed stack goes on after."""
    if R.Settings is Settings:
        R.Settings, R.asdict = SS.Settings, SS.asdict
    on_items = SS.install(on_items)
    if (R.Settings, R.asdict) != (SS.Settings, SS.asdict):
        raise RuntimeError('transit_base: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Transit):
            FE.SHARED_CLASSES[kv] = type(f'Transit{cls.__name__}', (Transit, cls), {})
    return on_items


# ======================================== module transit_in0
"""The transit base cycle at block 0's mid stream (x0's injections held): it is a complete mid0 cycle, so mid0 is a transit
root decoded by its own solve, and its own cycle goes.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import transit_base as TB  # noqa: E402
SS, CT, R, FE, FC, S2M = TB.SS, TB.CT, TB.R, TB.FE, TB.FC, TB.S2M


@dataclass
class Settings(TB.Settings):  # transit_base's fields and the base's block
    transit_base_block: int = 1
    transit_base_mid: bool = False
    transit_base_whiten: bool = False
    transit_whiten_mode: str = 'lu'  # 'lu' (MLP 0 identity push + batched solve) | 'wood' (closed form, routed units)
    transit_whiten_k: int = 256  # 'wood': MLP 0's top routed units per token used (the active ones come first)
    transit_whiten_eps: float = 0.01  # 'wood': Tikhonov floor of the k x k core (bounds the moves where J0 is near-singular)

    def __post_init__(self):
        roots = tuple(self.transit_roots)
        if self.transit_base_block not in (0, 1) or type(self.transit_base_mid) is not bool or (
                self.transit_base_mid and self.transit_base_block != 0) or type(self.transit_base_whiten) is not bool or (
                self.transit_base_whiten and not self.transit_base_mid) or self.transit_whiten_mode not in ('lu', 'wood') or (
                not 1 <= self.transit_whiten_k <= 4 * self.model.n_embd) or not 0 <= self.transit_whiten_eps < 1:
            raise ValueError('transit_in0: transit_base_block 0 | 1; transit_base_mid only with block 0; '
                             'transit_base_whiten only with transit_base_mid; transit_whiten_mode lu | wood, k, eps')
        if self.transit_base_block == 0:  # transit_base's checks on the roots above block 0, then root 0 allowed
            self.transit_roots = tuple(r for r in roots if r >= 1) or (1,)
            super().__post_init__()
            self.transit_roots = roots
            if not (set(roots) <= set(range(0, self.model.n_layer)) and roots):
                raise ValueError('transit_in0: transit_roots within 0..n_layer - 1')
        else:
            super().__post_init__()
            if 0 in roots:
                raise ValueError('transit_in0: root 0 needs transit_base_block 0')


def asdict(settings):  # transit off: transit_base's (= speed_stack's) recipe dict
    return TB.asdict(settings)


class Transit0:
    """First in the MRO (on transit_base's Transit): the base cycle at block transit_base_block's input."""

    def transit_base(self, ctxs, seq, Hb, roots):
        e, s, tx = self.e, self.e.s, self.tx
        b0 = getattr(s, 'transit_base_block', 1)
        if b0 == 1:
            return super().transit_base(ctxs, seq, Hb, roots)
        L, W, T, dev = e.model.config.n_layer, e.width, e.capture.length, e.device
        (tb0, tb1), buf, S0 = tx['tb'], tx['buf'], seq * T
        nb, lowest = tb1 - tb0, roots[0]
        mid = getattr(s, 'transit_base_mid', False)
        whiten = mid and getattr(s, 'transit_base_whiten', False)
        if mid:  # the base moves mid0 (the stream after attention 0)
            xin = self.chain_args[2][b0][seq * T:(seq + 1) * T].float()
        else:  # in0: block 0's input (after its mixing scalars)
            xin = e.capture.blocks[b0][seq].float()
        bstep = s.transit_sigma * xin[tb0:tb1].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        tr = dict(bstep=bstep, yv=torch.empty(nb, W, device=dev), c=torch.zeros(nb, W, device=dev), prev=None, live=set(),
                  dm={b: torch.empty(nb, W, W, device=dev, dtype=buf['delta'].dtype) for b in roots},
                  din={b: torch.empty(nb, W, W, device=dev, dtype=buf['delta'].dtype) for b in range(lowest + 1, L)},
                  twin={k: torch.zeros(nb, W, W, device=dev) for k in tx['rows'].kc if k > lowest + 1})
        if mid and not whiten:
            tr['Hb'] = Hb  # the base root's own decode (transit_root below)
        with self.transit_mode():
            for t0, t1 in tx['chunks']:
                m, sl, loc = t1 - t0, slice(S0 + t0, S0 + t1), slice(t0 - tb0, t1 - tb0)
                x = tx['head'](self.chunk(buf, ctxs, seq, t0, t1), sl)
                delta = buf['delta'][:m * W * W].view(m, W, W)
                torch.mul(bstep[loc, None], Hb, out=delta)
                if whiten and s.transit_whiten_mode == 'wood':  # closed form from MLP 0's routed units (no push)
                    g = slice(S0 + t0, S0 + t1)
                    blk0 = e.model.transformer.h[0].mlp
                    with S2M.ieee():
                        delta.copy_(wood_moves(Hb, self.sp_w['mids'][0][g], self.sp_w['idx32'][0][g], self.sp_w['hs'][0][g],
                                               blk0.c_fc.weight, blk0.c_proj.weight, float(e.model.resid_lambdas[1]),
                                               s.transit_whiten_k, s.transit_whiten_eps) * bstep[loc, None])
                elif whiten:  # moves M (x bstep) with M J0^T = H: J0^T from MLP 0's identity push, then a batched solve
                    d0 = (torch.eye(W, device=dev, dtype=delta.dtype).expand(m, W, W) * bstep[loc, None]).contiguous()
                    self.mlp(x, d0, W, b0)  # rows: the unit moves' displacements at in1 (x bstep)
                    delta.copy_(torch.linalg.solve(d0 / bstep[loc, None], delta, left=False))
                    del d0
                for b in range(b0, L):
                    if b in tr['din']:
                        tr['din'][b][loc] = delta
                    if not (mid and b == b0):  # a mid0 base starts after attention 0
                        self.transit_se(x, delta, b)
                        self.push(x, delta, W, b)
                    if b in tr['dm']:
                        TB.transit_tdiv_kernel[(W // 64, W // 64, m)](delta, tr['dm'][b][loc], W, 64, num_warps=4)
                    self.mlp(x, delta, W, b)
                tr['yv'][loc] = FC.head_fast(delta, x) * (tx['y'][sl] >= 0).unsqueeze(1) / bstep[loc]
        return tr


    def transit_root(self, tr, ctxs, seq, r, dstep):
        s = self.e.s
        if not ('Hb' in tr and r == getattr(s, 'transit_base_block', 1)):
            return super().transit_root(tr, ctxs, seq, r, dstep)
        # the mid0 base's own root: its moves ARE a complete cycle (D = bstep Hb, Hb Hb^T = W I), so g = s Hb / W, the
        # mid0 cycle's own decode; no solve and no direct moves (D is perfectly conditioned)
        W, L = self.e.width, self.e.model.config.n_layer
        tr['dm'].pop(r)
        live = {k: tr['twin'][k] for k in sorted(tr['live'])}
        rhs = tr['c'] + self.transit_twins(ctxs, seq, live, W) / tr['bstep'] if live else tr['c']
        Hb = tr['Hb'].float()
        with S2M.ieee():
            g = tr['yv'] @ Hb / W
            xr = rhs @ Hb / W if r < L - 1 else torch.zeros_like(g)
        return g, xr


def wood_moves(Hb, x, idx, a, w_fc, w_proj, lam1, k, eps):
    """Moves M (rows) with M J0^T ~ Hb per token: J0^T = lam1 (I + U^T V), U [m, k, W] = MLP 0's routed units' input rows
    through the norm at mid0 (x [m, W]), V [m, k, W] = 2 relu(a_u) x the units' output rows; Woodbury
    (I + U^T V)^-1 = I - U^T (I + V U^T)^-1 V with the k x k core inverted as (C^T C + eps^2 I)^-1 C^T (bounded moves)."""
    m, W = x.shape
    idx, a = idx[:, :k].long(), a[:, :k].float()
    r = (x.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).sqrt()          # [m, 1]
    xh = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    U = w_fc[idx].float()                                                                        # [m, k, W]
    U = (U - (U @ xh.unsqueeze(-1)) * xh.unsqueeze(1)) / r.unsqueeze(-1)
    V = w_proj.T[idx].float() * (2 * F.relu(a)).unsqueeze(-1)                                  # [m, k, W]
    C = torch.eye(U.shape[1], device=x.device) + V @ U.transpose(1, 2)                        # [m, k, k]
    G = C.transpose(1, 2) @ C
    G.diagonal(dim1=1, dim2=2).add_(eps * eps)
    L, info = torch.linalg.cholesky_ex(G)
    bad = info != 0                                                                             # unwhitened (H) there
    if bad.any():
        L[bad] = torch.eye(G.shape[-1], device=x.device)
        V = V.masked_fill(bad[:, None, None], 0.)
    HU = Hb @ U.transpose(1, 2)                                                                 # [m, W, k]
    Y = torch.cholesky_solve(HU.transpose(1, 2), L).transpose(1, 2)                             # HU G^-1
    return (Hb - (Y @ C.transpose(1, 2)) @ V) / lam1


def install(on_items=None):
    """transit_base's install, then this file's Settings / recipe dict and the Transit0 mixin first in fast_e3's shared
    Methods (the speed stack goes on after, as for transit_base)."""
    if R.Settings is Settings:
        R.Settings, R.asdict = TB.Settings, TB.asdict
    on_items = TB.install(on_items)
    if (R.Settings, R.asdict) != (TB.Settings, TB.asdict):
        raise RuntimeError('transit_in0: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Transit0):
            FE.SHARED_CLASSES[kv] = type(f'Transit0{cls.__name__}', (Transit0, cls), {})
    return on_items


# ======================================== module transit_gap
"""(1) Boost: the embedding's credit adds block 0's receivers (each embedding move also changes the token's block-0 key /
value; the later queries are scored by #31's receiver scorer against Boost's measured block-0 hub, decoded by the same
cycle). (2) In-roots: the block-input credit in_j (j = 2..7) is decoded by its own solve from what root j - 1 already
measures (the base rows' displacements at in_j, the same right-hand sides); it feeds only linear consumers.
"""
from dataclasses import dataclass
import math
import os
import sys

import torch
import torch.nn.functional as F

import transit_in0 as T0  # noqa: E402
TB = T0.TB
SS, CT, R, FE, FC, S2M = TB.SS, TB.CT, TB.R, TB.FE, TB.FC, TB.S2M

STATS = dict(calls=0, boost=0, every=200)


@dataclass
class Settings(T0.Settings):  # transit_in0's fields and the two switches
    boost_emb_block0: bool = False
    transit_inroots: bool = False

    def __post_init__(self):
        if type(self.boost_emb_block0) is not bool or type(self.transit_inroots) is not bool:
            raise ValueError('transit_gap: boost_emb_block0 / transit_inroots true / false')
        if self.transit_inroots:
            if not self.transit or tuple(self.readout_in_roots):
                raise ValueError('transit_gap: transit_inroots needs transit true and readout_in_roots [] (the in-roots '
                                 'replace the readout\'s own-query in-roots)')
            self.twin_inputs = tuple(i for i in self.twin_inputs if i not in in_set(self))  # the input twins they replace
        super().__post_init__()


def in_set(s):
    """The block inputs j decoded from the base: j = r + 1 for every transit root r >= 1 below the top."""
    return {r + 1 for r in s.transit_roots if 1 <= r < s.model.n_layer - 1} if getattr(s, 'transit_inroots', False) else set()


def asdict(settings):  # both off: transit_in0's recipe dict
    state = T0.asdict(settings)
    for key in ('boost_emb_block0', 'transit_inroots'):
        if not state.get(key, True):
            state.pop(key)
    return state


class Gap:
    """First in the MRO (on transit_in0's Transit0): the in-roots inside the chain, their consumers in finish."""

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        ins = in_set(self.e.s)
        if not ins:
            return super().chain_credits(emb, x0, mids, outs, routes, y)
        n, W = emb.shape[0], self.e.width
        self.gap_in = {j: torch.zeros(2, n, W, device=self.e.device) for j in sorted(ins)}  # (own, cross), my tokens
        out = super().chain_credits(emb, x0, mids, outs, routes, y)
        G = self.e.group_sum(torch.stack(list(self.gap_in.values())))  # the pair's halves
        self.gap_in = dict(zip(self.gap_in, G))
        STATS['calls'] += 1
        if os.environ.get('RANK', '0') == '0' and (STATS['calls'] == 1 or STATS['calls'] % STATS['every'] == 0):
            rat = {j: round(float(g[0].norm() / out[j].norm().clamp_min(1e-30)), 3) for j, g in self.gap_in.items()}
            print(f'transit_gap estimate {STATS["calls"]}: in-roots {sorted(ins)} decoded from the base (|g_in| / '
                  f'|chain_j| {rat}), input twins {list(self.e.s.twin_inputs)}', flush=True)
        return out

    def transit_score(self, tr, ctxs, seq, b):
        """transit_base's; in_b's displacements (which it drops) kept transposed for root b - 1's in-root (no extra peak:
        the transposed copy replaces the raw rows)."""
        if b not in getattr(self, 'gap_in', {}):
            return super().transit_score(tr, ctxs, seq, b)
        din, W = tr['din'][b], self.e.width
        super().transit_score(tr, ctxs, seq, b)
        MT = torch.empty_like(din)
        TB.transit_tdiv_kernel[(W // 64, W // 64, din.shape[0])](din, MT, W, 64, num_warps=4)
        tr['din_in'] = (b, MT)

    def transit_root(self, tr, ctxs, seq, r, dstep):
        """transit_base's transit_root (a copy, every gap line marked) + in_{r+1}'s solve on the same right-hand sides."""
        j, MT = tr.pop('din_in', (None, None))  # gap
        if j != r + 1 or j not in getattr(self, 'gap_in', {}):  # gap
            return super().transit_root(tr, ctxs, seq, r, dstep)
        e, s, tx = self.e, self.e.s, self.tx
        L, W, K, dev = e.model.config.n_layer, e.width, s.transit_direct, e.device
        (tb0, tb1), S0 = tx['tb'], seq * e.capture.length
        nb = tb1 - tb0
        live = {k: tr['twin'][k] for k in sorted(tr['live'])}
        rhs = tr['c'] + self.transit_twins(ctxs, seq, live, W) / tr['bstep'] if live else tr['c']
        Y = torch.stack([tr['yv'], rhs], -1) * tr['bstep'][..., None]  # gap: shared with in_j
        Q, iters = None, s.transit_iters
        if K:
            Q = TB.transit_x0(r, W, K + s.transit_spare, dev).expand(nb, W, K + s.transit_spare)
            if tr['prev'] is None:
                iters = max(iters + 1, 2)
            else:
                with S2M.ieee():
                    Q = TB.transit_orth_t(torch.cat([tr['prev'], Q[..., K:]], -1))
        with S2M.ieee():
            sol, U, Zu, lamu = TB.transit_fast(tr['dm'].pop(r), Y, Q, K, iters)
        g, xr = sol[..., 0], sol[..., 1]  # gap: r < L - 1 here
        mine = slice(S0 + tb0, S0 + tb1)  # gap
        if not K:
            with S2M.ieee():  # gap
                self.gap_in[j][:, mine] = inroot_solve(MT, Y).permute(2, 0, 1)
            return g, xr
        tr['prev'] = U
        dst = dstep[S0 + tb0:S0 + tb1] * math.sqrt(W) * (s.transit_dsigma / s.chain_sigma)
        meas, xm, V = self.transit_direct(ctxs, seq, r, U, dst, keep_in=True)  # gap: + their displacements at in_j
        with S2M.ieee():
            self.gap_in[j][:, mine] = inroot_solve(MT, Y, V, torch.stack([meas, xm], -1)).permute(2, 0, 1)  # gap
            return TB.transit_keep(g, U, Zu, lamu, meas), TB.transit_keep(xr, U, Zu, lamu, xm)

    def transit_direct(self, ctxs, seq, r, U, dst, keep_in=False):
        """transit_base's transit_direct (a copy, every gap line marked); keep_in: also the moves' displacements at in_{r+1}
        per unit length, V [nb, W, K]."""
        if not keep_in:  # gap
            return super().transit_direct(ctxs, seq, r, U, dst)
        e, tx = self.e, self.tx
        L, W, K, T, dev = e.model.config.n_layer, e.width, U.shape[-1], e.capture.length, e.device
        (tb0, tb1), buf, rw, S0 = tx['tb'], tx['buf'], tx['rows'], seq * T
        meas, xm = torch.empty(tb1 - tb0, K, device=dev), torch.zeros(tb1 - tb0, K, device=dev)
        V = torch.empty(tb1 - tb0, W, K, device=dev)  # gap
        sums = {k: torch.zeros(tb1 - tb0, K, W, device=dev) for k in rw.kc if k > r + 1 and r + 1 in rw.first}
        with self.transit_mode():
            for t0, t1 in tx['chunks']:
                m, sl, loc = t1 - t0, slice(S0 + t0, S0 + t1), slice(t0 - tb0, t1 - tb0)
                x = tx['head'](self.chunk(buf, ctxs, seq, t0, t1), sl)
                delta = buf['delta'][:m * K * W].view(m, K, W)
                torch.mul(dst[loc, None], U[loc].transpose(1, 2), out=delta)
                self.mlp(x, delta, K, r)
                V[loc] = delta.transpose(1, 2) / dst[loc, None]  # gap
                for b in range(r + 1, L):
                    xm[loc] += self.score(x, delta, K, b, True)
                    kc = [k for k in sums if k > b] if b in rw.first else []
                    if kc:
                        rw.m, rw.t0 = m, t0
                        rw.inject(x, K, K, b)
                        for k in kc:
                            sums[k][loc].addcmul_(rw.inj[1], tx['scale'][b, k])
                        rw.inj = None
                    self.push(x, delta, K, b)
                    self.mlp(x, delta, K, b)
                meas[loc] = FC.head_fast(delta, x) * (tx['y'][sl] >= 0).unsqueeze(1)
        if sums:
            xm += self.transit_twins(ctxs, seq, sums, K)
        return meas / dst, xm / dst, V

    def finish(self, gradients):
        """The stack's, then the in-roots' consumers (module docstring): mlp.c_proj_{j-1} and block j's scalars (formed in
        full by a pair's first GPU, the record's rule), mlp.c_fc_{j-1}'s own part (every GPU: its share of the units)."""
        super().finish(gradients)
        G = getattr(self, 'gap_in', None)
        if not G:
            return
        e, d = self.e, self.e.capture
        lam_r = e.model.resid_lambdas.float()
        emb, x0, mids, outs, routes, y = self.chain_args
        lo, hi = e.share(4 * e.width)
        for j, (g, xr) in G.items():
            p = f'transformer.h.{j - 1}.mlp.'
            if not e.part:
                cin = g + xr
                gradients[p + 'c_proj.weight'] = (lam_r[j] * cin).T @ d.inputs[p + 'c_proj'].flatten(0, 1)
                gradients['resid_lambdas'][j] = (cin * outs[j - 1]).sum()
                gradients['x0_lambdas'][j] = (cin * x0).sum()
            u = d.outputs[p + 'c_fc'].float().flatten(0, 1)  # the hidden rule is linear in its credit: add the change
            steps = e.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            w = e.model.transformer.h[j - 1].mlp.c_proj.weight.float()
            dh = R.hidden_local_credit(u[:, lo:hi], steps, lam_r[j] * (g - self.chain[j]), w[:, lo:hi])
            gradients[p + 'c_fc.weight'][lo:hi] += dh.T @ d.inputs[p + 'c_fc'].flatten(0, 1)
        self.gap_in = None


def inroot_solve(MT, Y, V=None, meas=None):
    """in_j's decode: MT [nb, W, W] (the base's raw displacement rows, transposed: coordinates x moves), Y [nb, W, c] (raw
    changes) -> least squares (D^T D)^-1 D^T Y [nb, W, c] by transit's fused Gram / Cholesky / substitution; V [nb, W, K]
    (measured directions, unit length) with meas [nb, K, c]: moved to V^T g = meas with the least change of |D g - Y|
    (transit_keep with the general K x K Gram V^T (D^T D)^-1 V: the directions are not the Gram's Ritz vectors here)."""
    c = Y.shape[-1]
    G, Rr, TRP = TB.transit_fast_gram(MT, Y)
    DI, _ = TB.transit_fast_chol(G, TRP, MT)  # (its retry pass rereads the rows of tokens with a bad pivot)
    sol = TB.transit_fast_solve(G, DI, Rr if V is None else torch.cat([Rr, V], -1))
    return sol if V is None else keep_general(sol[..., :c], V, sol[..., c:], meas)


def keep_general(g, V, Zv, meas):
    """g - Zv (V^T Zv)^-1 (V^T g - meas): g [nb, W, c], V / Zv [nb, W, K], meas [nb, K, c]; the K x K Gram floored at 1e-10
    of its trace (a direction the solve already pins, or a repeated one, adds nothing)."""
    C = V.transpose(1, 2) @ Zv
    C = (C + C.transpose(1, 2)) / 2
    C.diagonal(dim1=1, dim2=2).add_(C.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True).clamp_min(1e-30) * 1e-10)
    L, info = torch.linalg.cholesky_ex(C)
    r = V.transpose(1, 2) @ g - meas
    corr = Zv @ torch.cholesky_solve(r, L)
    return torch.where((info == 0)[:, None, None], g - corr, g)


# ------------------------------------------------------------------------------------- Boost: block 0's receivers


def boost_internals(fn):
    def internals(self, gradients, targets, *a, **k):
        if getattr(self.s, 'boost_emb_block0', False) and not isinstance(self, R.Cut):
            self._gap_hub0 = targets[0]  # Boost's measured block-0 hub (g_mid0 W_proj), read by embedding_block0 next
        return fn(self, gradients, targets, *a, **k)
    internals.gap_wrapped = fn
    return internals


def boost_embedding(fn):
    def embedding_block0(self, e, y, x):
        hub = self.__dict__.pop('_gap_hub0', None)
        if hub is None:
            return fn(self, e, y, x)
        drawn, prior, src = [], self.__dict__.get('basis'), self.basis
        self.basis = lambda: drawn.append(src()) or drawn[-1]  # the record's draw, kept (no extra draw)
        try:
            own = fn(self, e, y, x)
        finally:
            self.__dict__.pop('basis')
            if prior is not None:
                self.basis = prior
        STATS['boost'] += 1
        if os.environ.get('RANK', '0') == '0' and STATS['boost'] == 1:
            print('transit_gap: Boost embedding credit + block 0\'s receivers (boost_emb_block0)', flush=True)
        return own + block0_receivers(self, e, hub, drawn[0])
    embedding_block0.gap_wrapped = fn
    return embedding_block0


def scorer():
    """The future receiver scorer below fast_moments' moment path (which serves only FiniteFuture's contexts)."""
    FM = sys.modules.get('fast_moments')
    prev = FM.STATE.get('previous_scorer') if FM is not None and isinstance(getattr(FM, 'STATE', None), dict) else None
    return prev or R.receiver_scores


def block0_receivers(est, e, hub, H):
    """Each embedding move's block-0 key / value change scored at the receivers s > t (window) against the hub -> this
    pair's decode [b t, W] (embedding_block0's moves: one-sided, steps sigma x RMS(e), rows H[lo:hi])."""
    model, d, s = est.model, est.capture, est.s
    b, t = d.batch, d.length
    Wd, h = est.width, model.config.n_head
    hd = Wd // h
    e = e.float().view(b, t, Wd)
    attn = model.transformer.h[0].attn
    mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
    window = model.window_sizes[0][0]
    cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
    c1, s1 = cos[0, :, 0], sin[0, :, 0]
    wkv = torch.cat((attn.c_k.weight, attn.c_v.weight)).float()
    steps = s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    lo, hi = est.share(Wd)
    rows = H[lo:hi]
    score = scorer() if e.is_cuda else cpu_scores
    derivative = torch.zeros(b, t, Wd, device=e.device)
    for r in range(b):
        qraw, kraw, vraw = (d.outputs[f'transformer.h.0.attn.c_{n}'][r:r + 1].float() for n in 'qkv')
        values = vraw.view(1, t, h, hd).transpose(1, 2)[0]
        q, _, a, out = R.attention_state(qraw, kraw, values[None], cos, sin, h, window)
        q, a, out = q[0], a[0], out[0]
        target = hub[r].reshape(t, h, hd).transpose(0, 1).float()
        contrast = target @ values.transpose(-1, -2) - (target * out).sum(-1, keepdim=True)
        xin, kr = d.blocks[0][r].float(), kraw[0].view(t, h, hd)
        for t0 in range(0, t, s.future_tokens):
            sl = slice(t0, min(t0 + s.future_tokens, t))
            move = R.norm_delta(e[r, sl, None], steps[r, sl, None] * rows[None])  # x0' - x0 per row
            derivative[r, sl, lo:hi] = score(move, mix, xin[sl], kr[sl], None, None, wkv, None, c1[sl], s1[sl], q, target,
                                             a[:, :, sl], contrast[:, :, sl], t0, window) / steps[r, sl]
    return est.group_sum((derivative @ H / Wd).reshape(b * t, Wd))


def cpu_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin, q, target, a, contrast, start, window):
    """The record's receiver_scores without its compiled kernel (CPU checks): receiver_changes + joint_scores."""
    dk, dv = R.receiver_changes(x, move * coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return R.joint_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T


def install(on_items=None):
    """transit_in0's install, then this file's Settings / recipe dict, the Gap mixin first in fast_e3's shared Methods and
    the two Boost wrappers (they act only with boost_emb_block0)."""
    if R.Settings is Settings:
        R.Settings, R.asdict = T0.Settings, T0.asdict
    on_items = T0.install(on_items)
    if (R.Settings, R.asdict) != (T0.Settings, T0.asdict):
        raise RuntimeError('transit_gap: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Gap):
            FE.SHARED_CLASSES[kv] = type(f'Gap{cls.__name__}', (Gap, cls), {})
    if not hasattr(R.Boost.internals, 'gap_wrapped'):
        R.Boost.internals = boost_internals(R.Boost.internals)
        R.Boost.embedding_block0 = boost_embedding(R.Boost.embedding_block0)
    return on_items


# ======================================== module transit_gap2
"""Boost's block-0 receivers for wte are each GPU's share of the cycle (Boost.step's all-reduce adds the pair once)."""

import transit_gap as TG  # noqa: E402
R, FE, T0, TB = TG.R, TG.FE, TG.T0, TG.TB

PAIR_SUM = TG.block0_receivers  # transit_gap's (ends with the pair's group_sum)


class Share:
    """est as seen by transit_gap's block0_receivers: every attribute its own, group_sum the identity."""

    def __init__(self, est):
        self.est = est

    def __getattr__(self, name):
        return getattr(self.est, name)

    @staticmethod
    def group_sum(tensor):
        return tensor


def block0_receivers(est, e, hub, H):
    """This GPU's share of block 0's receivers decode [b t, W] (rows H[lo:hi]); Boost.step's all-reduce adds the pair."""
    return PAIR_SUM(Share(est), e, hub, H)


def patch():
    TG.block0_receivers = block0_receivers


def install(on_items=None):
    """transit_gap's install, then its Boost wrapper's receivers -> this file's (each GPU's share)."""
    on_items = TG.install(on_items)
    patch()
    return on_items


# ======================================== module shared_pairs
"""Shared logit pairs. c_q and c_k of a block meet at the same attention logits, so one experiment per measured (head h,
key t, receiver s) pair feeds both (blocks 0-4): each receiver takes its top-16 keys per head by a forward bound; the
row moves query s's logit on key t, takes its exact single-column softmax change through c_proj, and is pushed and
scored like the receiver rows; the decoded logit credit goes to c_k and c_q through their pull-backs (RoPE + QK-norm).
Plus one q-only tail row per receiver and head along the top singular direction of the tail keys' map.
"""
import contextlib
from dataclasses import dataclass
import math
import os

import torch
import torch.nn.functional as F

import transit_gap as TG  # noqa: E402
TB, T0, SS, R, FE = TG.TB, TG.T0, TG.SS, TG.R, TG.FE
SP, KV, S2P = SS.SP, SS.KV, SS.S2P

KEYS, ITERS = 16, 3  # pairs per (receiver, head); power iterations of the tail direction (pass sim: 2 ~ 10)
STATS = dict(calls=0, every=200, keep=False)


@dataclass
class Settings(TG.Settings):  # transit_gap's fields and the switch
    readout_shared_pairs: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.readout_shared_pairs) is not bool or self.readout_shared_pairs and not (
                KV.kv_on(self) and self.readout_sources and (self.speed_exact or self.speed_store or self.readout_skip_blocks)):
            raise ValueError('shared_pairs: readout_shared_pairs true / false; true needs the receivers, the sources and '
                             'speed_stack\'s pass (speed_exact / speed_store / readout_skip_blocks)')


def asdict(settings):  # off: transit_gap's recipe dict
    state = TG.asdict(settings)
    if not state.get('readout_shared_pairs', True):
        state.pop('readout_shared_pairs')
    return state


@contextlib.contextmanager
def fp32():  # full-precision matmuls (the experiments' responses; the directions may use TF32)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


def jn(u, z):
    """The per-head RMSNorm's Jacobian at u applied to z (symmetric): (z - u (u.z) / (D ms)) / sqrt(ms), ms = |u|^2 / D + eps."""
    D = u.shape[-1]
    ms = u.square().sum(-1, keepdim=True) / D + KV.EPS
    return (z - u * (u * z).sum(-1, keepdim=True) / (D * ms)) * ms.rsqrt()


def rope(x, cs, sn, inverse=False):  # x [..., T, D] with cs / sn [T, D / 2] (broadcast over the leading dims)
    return R.rotate(x, cs, -sn if inverse else sn)


class Pairs:
    """First in the MRO (on transit_gap's Gap): the pass with shared logit pairs and tail rows; off = the stack's run."""

    def run(self, chain, targets):
        e, s = self.e, self.e.s
        if not s.readout_shared_pairs:
            return super().run(chain, targets)
        model, d, dev = e.model, e.capture, e.device
        L, W, H, T = model.config.n_layer, e.width, model.config.n_head, d.length
        self.sp_events[2].record()
        lam_r = model.resid_lambdas.float()
        hb, ins = tuple(s.readout_hi_blocks), tuple(sorted(s.readout_in_roots))
        skip = set(s.readout_skip_blocks)
        top = {hb[-1] + 1} if hb and s.readout_hi_top_root and hb[-1] + 1 <= L - 2 else set()
        self.hi_blocks, self.hi_in, self.hi_q, self.hi_fc, self.hi_k, self.hi_g = hb, ins, {}, {}, {}, {}
        self.hi_roots = sorted(set(hb) | ({SP.FIRST} if s.readout_hi_out3 else set()) | top)
        self.hi_mproj = sorted(({SP.FIRST - 1} if s.readout_hi_out3 else set()) | set(hb) | {j - 1 for j in ins})
        self.hi_w = dict(wq={j: model.transformer.h[j].attn.c_q.weight.float() for j in ins})
        src_b = tuple(b for b in SP.BLOCKS + hb if b not in skip) if s.readout_sources else ()
        g1, own1 = KV.groups_of(s, model, e.split)
        g2 = [(b, k) for b in hb if b not in skip for k in 'kg'
              if k == 'k' or s.readout_gate and model.transformer.h[b].attn.ve_gate is not None]
        own2 = SP.HI.balance(g2, e.split, L)
        kb = {b for b, k in g1 + g2 if k == 'k' and b not in skip}
        shared = tuple(b for b in src_b if b in kb)  # pairs: blocks whose q rows and k groups the pairs replace
        cyc = {('q', b): W if b in SP.BLOCKS and b in s.readout_q_full else H * s.readout_q_keys for b in src_b}
        cyc |= {('f', b): (s.readout_fc_cycle + s.readout_hi_fc_cycle)[b] for b in src_b}
        f_cfg = {b: cyc['f', b] for b in src_b}
        if s.speed_exact and SS.EXACT['fc'] and src_b:
            act = torch.stack([(d.outputs[f'transformer.h.{b}.mlp.c_fc'] > 0).sum(-1).amax() for b in src_b]).tolist()
            for b, n_act in zip(src_b, act):
                cyc['f', b] = min(f_cfg[b], max(32, 1 << int(n_act).bit_length()))
        self.sp_fc = {b: cyc['f', b] for b in src_b}
        q_cfg = {b: cyc.pop(('q', b)) for b in shared}  # pairs: no q source rows there (their codes are still drawn)
        share = {key: e.share(c) for key, c in cyc.items()}
        mine1 = [g for g in g1 if own1[g] == e.part and g[0] not in skip and not (g[1] == 'k' and g[0] in shared)]  # pairs
        mine2 = [g for g in g2 if own2[g] == e.part and not (g[1] == 'k' and g[0] in shared)]  # pairs
        mine = mine1 + mine2
        if H % e.split:
            raise ValueError('shared_pairs: the heads must split evenly over a pair')
        hs, h0 = H // e.split, e.part * (H // e.split)  # pairs: this GPU's heads (a pair splits them)
        heads = slice(h0, h0 + hs)
        self.kv_C, self.kv_counts, self.kv_added = KV.group_rows(s, H), {}, {}
        self.kv_wproj = {b: model.transformer.h[b].attn.c_proj.weight.float() for b, _ in mine}
        size = {key: b - a for key, (a, b) in share.items()} | {g: self.kv_C for g in mine}
        size |= {key: n for b in shared for key, n in ((('t', b), hs), (('p', b), hs * KEYS))}  # pairs
        slots, at, in_rows, mid_rows = {}, 0, [], []  # rows per token: q_b, tail_b, pairs_b, groups of b | MLP b | f_b
        for b in range(L):
            in_rows.append(at)
            for key in [('q', b), ('t', b), ('p', b), *[g for g in mine if g[0] == b], 'mlp', ('f', b)]:  # pairs
                if key == 'mlp':
                    mid_rows.append(at)
                elif key in size:
                    slots[key], at = slice(at, at + size[key]), at + size[key]
        rows, n = at, d.batch * T
        self.kv_mine, self.kv_rows, self.kv_events, self.hi_events = mine1, len(mine1) * self.kv_C, [], []
        self.hi_cfg = dict(rows=sum((in_rows + [rows])[b + 1] - in_rows[b] for b in hb), mine=mine2)
        self.q_credit = {b: torch.zeros(n, W, device=dev) for k, b in list(cyc) + [('q', b) for b in shared] if k == 'q'}
        self.fc_credit = {b: torch.zeros(n, 4 * W, device=dev) for k, b in cyc if k == 'f'}
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, k in mine + [(b, 'k') for b in shared] if k == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, k in mine if k == 'g'}
        self.sl_heads, self.sl_rows, self.sl_shared = (h0, h0 + hs), rows, shared
        Q, score = torch.zeros(len(ins), n, W, device=dev), torch.zeros(n, rows, device=dev)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())
        lo, hi = e.share(W)
        fc_in = {j: slots['f', j - 1] for j in ins if ('f', j - 1) in slots}
        rms = lambda z: s.readout_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        m_chunk = min(s.readout_chunk, T)
        buf = self.buffers(m_chunk, max(rows, 1))
        dedup = s.speed_exact and SS.EXACT['q'] and FE.IMPL['qsrc'] != 'plain'
        for seq in range(d.batch):
            ctxs, src, plans, pp = self.ctx_cache.pop(seq), {}, {}, {}
            for c in ctxs:
                c.pop('contrast', None)
            gens = [torch.Generator(dev).manual_seed(seed + off + 104729 * seq) for off in (0, SP.HI.HI_SEED)]
            for b in src_b:
                g, c = gens[b >= SP.FIRST], ctxs[b]
                if b in shared:  # pairs: the q code drawn as before (the generator's stream unchanged), not used
                    self.draw_code(g, T, q_cfg[b], dev)
                else:
                    basis = self.q_basis(c, s.readout_q_keys) if b >= SP.FIRST or cyc['q', b] < W else None
                    code = self.draw_code(g, T, cyc['q', b], dev)
                    if dedup and e.split > 1:
                        kq = cyc['q', b] // H
                        code = (code[0][torch.argsort(((code[0] % kq) * e.split // kq).to(torch.int32), stable=True)], code[1])
                    src['q', b] = (code, basis, rms(c['qraw']))
                code = self.draw_code(g, T, f_cfg[b], dev)
                if cyc['f', b] < f_cfg[b]:
                    keep = torch.argsort((code[0] >= cyc['f', b]).to(torch.int32), stable=True)[:cyc['f', b]]
                    code = (code[0][keep].contiguous(), code[1][:, :cyc['f', b]].contiguous())
                src['f', b] = (code, rms(d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float())) + self.fc_slots(
                    b, seq, cyc['f', b])
            for j in ins:
                src['in', j] = (self.draw_code(gens[1], T, W, dev), rms(ctxs[j]['x']))
            for b in sorted({g[0] for g in mine}):
                plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in mine if bb == b], self.cs_all, self.sn_all))
            for b in shared:  # pairs
                pp[b] = self.pair_plan(b, seq, ctxs[b], heads, rms)
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m, sl, local = t1 - t0, slice(seq * T + t0, seq * T + t1), slice(t0, t1)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                dirs = {key: self.code_rows(src[key][0], cyc[key], *share[key], local, dev) for key in cyc}
                for key in [key for key in cyc if key[0] == 'q' and src[key][1] is not None]:
                    dirs[key] = 2 * torch.einsum('thdk,tphk->tphd', src[key][1][local], dirs[key].view(
                        m, size[key], H, s.readout_q_keys)).reshape(m, size[key], W)
                tails = {}
                for b in range(L):
                    c, k, fb = ctxs[b], in_rows[b], slots.get(('f', b - 1))
                    if k:
                        score[sl, :k] += self.score(x, delta, k, b, b < L - 1 or fb is not None)
                    if fb is not None:
                        se = x['se'][:m * rows * W].view(m, rows, W)[:, fb]
                        score[sl, fb] += (se.float() * c['target'][local, None]).sum(-1)
                    if k and b in fc_in:
                        score[sl, fc_in[b]] += self.oq_scores(c, b, delta[:, fc_in[b]].float(), self.chain[b][sl], local)
                    if b == L - 1:
                        break
                    if k:
                        self.push(x, delta, k, b)
                    if ('q', b) in slots:
                        if dedup:
                            delta[:, slots['q', b]] = self.q_moves_dedup(c, b, dirs['q', b], src['q', b], cyc['q', b],
                                                                         share['q', b], local)
                        else:
                            delta[:, slots['q', b]] = self.q_moves(c, b, dirs['q', b], src['q', b][2][local], local)
                    if b in pp:  # pairs: the tail rows, then the pair rows (mid_b moves)
                        tails[b], delta[:, slots['t', b]] = self.tail_moves(b, pp[b], local)
                        delta[:, slots['p', b]] = self.pair_moves(b, pp[b], local)
                    for g in mine:
                        if g[0] == b:
                            delta[:, slots[g]] = self.moves(g, plans[g], c, t0, t1)
                    if mid_rows[b]:
                        self.mlp(x, delta, mid_rows[b], b)
                    if ('f', b) in slots:
                        _, fstep, idx, hval, valid = src['f', b]
                        torch.mul(self.fc_moves(b, dirs['f', b], fstep[local], idx[local], hval[local], valid[local]),
                                  lam_r[b + 1], out=delta[:, slots['f', b]])
                for (kind, b), got in ((key, score[sl, slots[key]]) for key in cyc):
                    if kind == 'q':
                        self.q_credit[b][sl] = torch.einsum('tp,tpw->tw', got / src[kind, b][2][local], dirs[kind, b]) / W
                    else:
                        _, fstep, idx, hval, valid = src[kind, b]
                        slot = torch.einsum('tp,tpk->tk', got / fstep[local], dirs[kind, b][:, :, 1:]) / cyc[kind, b]
                        self.fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                for b, p in pp.items():  # pairs: the decode (c_q of this chunk's receivers, c_k of their keys)
                    self.pair_decode(b, p, local, seq * T, score[sl, slots['p', b]], score[sl, slots['t', b]], tails[b], heads)
                for i, j in enumerate(ins):
                    code, istep = src['in', j]
                    rows_in = self.code_rows(code, W, lo, hi, local, dev)
                    sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                    Q[i, sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
            out = slice(seq * T, (seq + 1) * T)
            for g in mine:  # the receivers' decode (readout_kv's): ve_gate here (and c_k of blocks without pairs)
                b, kind = g
                p = plans[g]
                Rr = p['coef'].shape[-1]
                f = p['table'].clamp_min(0)
                live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
                reading = torch.zeros(H * T * Rr + 1, device=dev).scatter_(
                    0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, score[out, slots[g]], 0.).reshape(-1))
                reading = reading[:-1].view(H, T, Rr)
                if kind == 'k':
                    c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                    self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
                else:
                    self.g_credit[b][out] = reading.sum(-1).T / KV.GATE_SIGMA
            self.route_cache.pop(seq, None)
        self.hi_Q = dict(zip(ins, e.group_sum(Q))) if ins else {}
        self.sp_events[3].record()

    # ------------------------------------------------------------------------------------------------ the experiments

    def pair_plan(self, b, seq, c, heads, rms):
        """Block b, this GPU's heads: per receiver s its top-KEYS keys by the bound, their clean attention, the logit
        step x, the gains, both pull-backs of the logit (key side at t, query side at s; raw spaces), and the tail rows."""
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        cs, sn = self.cs_all, self.sn_all
        A = self.clean_attention(b, seq, c)[heads]  # [h, s, t]
        g = (self.X[b][seq * T:(seq + 1) * T] @ e.model.transformer.h[b].attn.c_proj.weight.float()).view(T, H, D)[:, heads]
        v, o = c['VT'][:, heads].transpose(0, 1), c['OT'][:, heads].transpose(0, 1)  # [h, T, D]
        dist = ((v * v).sum(-1)[:, None, :] - 2 * (o @ v.transpose(1, 2)) + (o * o).sum(-1)[:, :, None]).clamp_min(0).sqrt()
        idx = (A * dist * g.norm(dim=-1).T[:, :, None]).topk(min(KEYS, T), dim=-1).indices  # [h, s, K] by the bound
        del dist
        a = A.gather(-1, idx)
        valid = a > 0
        meas = torch.zeros_like(A, dtype=torch.bool).scatter_(-1, idx, valid)
        tail = (A > 0) & ~meas  # the q tail: live keys without a pair
        kr = c['R0'][:, heads].transpose(0, 1)  # [h, T, D] rotated raw keys
        kh = F.rms_norm(kr, (D,), eps=KV.EPS)  # normed rotated keys
        qraw = e.capture.outputs[f'transformer.h.{b}.attn.c_q'][seq].float()  # [T, W]
        rq = rope(qraw.view(T, H, D)[:, heads].transpose(0, 1), cs, sn)  # [h, T, D] rotated raw queries
        qh = c['QD'][:, heads].transpose(0, 1)  # [h, T, D] normed rotated queries
        hh = torch.arange(idx.shape[0], device=idx.device)[:, None, None]
        ku, kn = kr[hh, idx], kh[hh, idx]  # [h, s, K, D] at the keys
        csk, snk = cs[idx], sn[idx]
        pk = rope(jn(ku, qh[:, :, None]), csk, snk, inverse=True) / math.sqrt(D)  # d logit / d k_raw_t
        pq = rope(jn(rq[:, :, None], kn), cs[None, :, None], sn[None, :, None], inverse=True) / math.sqrt(D)  # / d q_raw_s
        knorm = c['kraw'][:, heads].transpose(0, 1).norm(dim=-1)  # [h, T]
        x = torch.where(valid, KV.K_SIGMA * knorm.gather(1, idx.flatten(1)).view_as(a) * pk.norm(dim=-1), 0.)
        ex = torch.expm1(x)
        wp = e.model.transformer.h[b].attn.c_proj.weight.float().view(-1, H, D)[:, heads].permute(1, 0, 2)  # [h, W, D]
        self.kv_counts[b, 'p'] = torch.stack([valid.sum(), meas.sum(), tail.sum()])
        coef = torch.where(valid, a * ex / (1 + a * ex), 0.)
        step = rms(qraw)  # [T, 1] the q rows' step
        u, dout = self.tail_rows(A * tail, kh, v, o, rq, step)
        return dict(idx=idx, valid=valid, inv=torch.where(valid, a / coef.clamp_min(1e-30), 0.), coef=coef, pk=pk, pq=pq,
                    wp=wp, vw=torch.einsum('htd,hwd->htw', v, wp), ow=torch.einsum('htd,hwd->htw', o, wp), step=step,
                    u=rope(u, cs, sn, inverse=True), dout=dout)

    @staticmethod
    def tail_rows(w, kh, v, o, rq, step):
        """The tail rows of every receiver (w = a on its tail keys, 0 elsewhere; [h, T, T]): direction u [h, T, D] = the
        top left singular vector of the raw tail map J^T M, M = sum_tail a_st k_t (v_t - o_s)^T, in the rotated frame (J^T
        = R^T P / sqrt(ms), P the QK-norm's tangent projector: P M's vector rotated back; ITERS power iterations, TF32),
        and the exact response of head h's output to the raw query moved by step along it (fp32): the measured keys'
        logits held, the tail keys' exact softmax, dout = sum_t a_t (e^d_t - 1) (v_t - o_s) / (1 + sum a (e^d - 1))."""
        r = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        tangent = lambda y: y - r * (r * y).sum(-1, keepdim=True)
        unit = lambda y: y / y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        u = unit(tangent(w @ kh))  # start: the tail's weighted key sum
        for _ in range(ITERS):  # (P M)(P M)^T u
            z = w * (u @ kh.transpose(1, 2))
            y = z @ v - o * z.sum(-1, keepdim=True)
            u = unit(tangent((w * (y @ v.transpose(1, 2) - (o * y).sum(-1, keepdim=True))) @ kh))
        with fp32():
            dq = KV.norm_change(rq, step.T[:, :, None] * u)  # N(r + dr) - N(r), dr = R_s step u_raw = step u
            ex = torch.expm1((dq @ kh.transpose(1, 2)) / math.sqrt(kh.shape[-1])) * w
            z = ex.sum(-1, keepdim=True)
            dout = (ex @ v - o * z) / (1 + z)  # [h, T, D]
        return u, dout

    def pair_moves(self, b, p, local):
        """[m, h K, W]: receiver s's output change of head h along v_t - o_s (exact single-column gain), through c_proj."""
        idx, coef = p['idx'][:, local], p['coef'][:, local]  # [h, m, K]
        hh = torch.arange(idx.shape[0], device=idx.device)[:, None, None]
        mv = coef[..., None] * (p['vw'][hh, idx] - p['ow'][:, local, None])  # [h, m, K, W]
        return mv.permute(1, 0, 2, 3).flatten(1, 2)

    def tail_moves(self, b, p, local):
        """The tail rows of receivers `local` (pair_plan's): (u, step) for the decode and the moves [m, h, W]."""
        return (p['u'][:, local], p['step'][local]), torch.einsum('hmd,hwd->mhw', p['dout'][:, local], p['wp'])

    def pair_decode(self, b, p, local, base, got, got_t, tail, heads):
        """reading a / gain = c_st (the row moved s's output by gain (v_t - o_s), as a logit change gain / a would to first
        order) -> c_q of the receivers (their pull-backs) and c_k of the keys (scattered); the tail rows' reading / step along
        u -> c_q."""
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        hs, m = p['idx'].shape[0], got.shape[0]
        c = got.reshape(m, hs, KEYS).transpose(0, 1) * p['inv'][:, local]
        u, step = tail
        qc = torch.einsum('hsk,hskd->shd', c, p['pq'][:, local]) + (u * (got_t.T / step.T)[:, :, None]).transpose(0, 1)
        self.q_credit[b][base + local.start:base + local.stop].view(m, H, D)[:, heads] = qc
        kc = torch.zeros(hs * T, D, device=got.device)
        hh = torch.arange(hs, device=got.device)[:, None, None] * T
        kc.index_add_(0, (hh + p['idx'][:, local]).flatten(), (c[..., None] * p['pk'][:, local]).flatten(0, 2))
        self.k_credit[b][base:base + T].view(T, H, D)[:, heads] += kc.view(hs, T, D).transpose(0, 1)

    def note(self):
        super().note()
        STATS['calls'] += 1
        n, s = STATS['calls'], self.e.s
        if s.readout_shared_pairs and os.environ.get('RANK', '0') == '0' and (n == 1 or n % STATS['every'] == 0):
            cnt = {f'p{b}': [int(z) for z in v.tolist()] for (b, k), v in self.kv_counts.items() if k == 'p'}
            print(f'shared_pairs estimate {n}: blocks {list(self.sl_shared)}, heads here {list(self.sl_heads)}, {KEYS} '
                  f'pairs + 1 tail row per head, rows per token {self.sl_rows}; pairs valid / measured / tail keys {cnt}',
                  flush=True)


def install(on_items=None):
    """transit_gap's install, then this file's Settings / recipe dict and the Pairs mixin first in fast_e3's shared Methods."""
    if R.Settings is Settings:
        R.Settings, R.asdict = TG.Settings, TG.asdict
    on_items = TG.install(on_items)
    if (R.Settings, R.asdict) != (TG.Settings, TG.asdict):
        raise RuntimeError('shared_pairs: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict = Settings, asdict
    for kv, cls in FE.SHARED_CLASSES.items():
        if not issubclass(cls, Pairs):
            FE.SHARED_CLASSES[kv] = type(f'Pairs{cls.__name__}', (Pairs, cls), {})
    return on_items


# ======================================== module shared_pairs2
"""shared_pairs on transit_gap2 (one install)."""

import shared_pairs as SPR  # noqa: E402
import transit_gap2 as TG2  # noqa: E402
TG, R, FE = TG2.TG, TG2.R, TG2.FE


def install(on_items=None):
    on_items = SPR.install(on_items)
    TG2.patch()
    if TG.block0_receivers is not TG2.block0_receivers or R.Settings is not SPR.Settings or not all(
            issubclass(c, SPR.Pairs) and issubclass(c, TG.Gap) for c in FE.SHARED_CLASSES.values()):
        raise RuntimeError('shared_pairs2: install incomplete')
    return on_items


# ======================================== module plant_init
"""Planted previous-token heads at init: the listed head's c_q / c_k rows become the content-free RoPE lag-1 pair applied
to the block's mean input direction at init (seed 1000 + SEED), broadcast from rank 0. Init only, before the clock.
"""
from dataclasses import dataclass
import functools
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is train/val/run.py)
import transit_base as TB  # noqa: E402
R, FE = TB.R, TB.FE

PREV_PROFILE = [0.233, 0.138, 0.094, 0.076, 0.065, 0.047, 0.05, 0.049, 0.043, 0.021, 0.025, 0.041]  # w1_ideas'
PLAIN = dict(build_model=R.build_model)


@dataclass
class Settings(TB.Settings):  # transit_base's fields and the plant
    plant: dict = None

    def __post_init__(self):
        super().__post_init__()
        if self.plant is None:
            return
        p, L, H = self.plant, self.model.n_layer, self.model.n_head
        try:
            ok = set(p) <= {'c', 'prev'} and 0 <= float(p.get('c', .15)) < 1 and all(
                len(i) in (2, 3) and 0 <= int(i[0]) < L and 0 <= int(i[1]) < H and (len(i) == 2 or int(i[2]) >= 1) for i in p['prev'])
        except (TypeError, KeyError, ValueError):
            ok = False
        if not ok or len({(int(i[0]), int(i[1])) for i in p['prev']}) != len(p['prev']):
            raise ValueError('plant_init: plant = {"c": [0, 1), "prev": [[block, head(, lag >= 1)], ...]} (distinct heads)')
        self.plant = dict(c=float(p.get('c', .15)), prev=[[int(v) for v in i] for i in p['prev']])


def asdict(settings):  # unset: transit_base's recipe dict
    state = TB.asdict(settings)
    if getattr(settings, 'plant', None) is None:
        state.pop('plant', None)
    return state


def rope_qk(hd, g, lag=1):
    """q*, k* (|q*| = |k*| = 8) whose RoPE + qk-norm logits are 8 x sum_i p_i cos(theta_i (t - s - lag)) (w1_ideas)."""
    half = hd // 2
    prof = torch.zeros(half)
    prof[:len(PREV_PROFILE)] = torch.tensor(PREV_PROFILE)
    prof[len(PREV_PROFILE):] = (1 - sum(PREV_PROFILE)) / (half - len(PREV_PROFILE))
    amp = (prof / prof.sum() * hd).sqrt()
    theta = 1.0 / 10000 ** (torch.arange(0, hd, 2).float() / hd) * lag
    phi = torch.rand(half, generator=g) * 2 * math.pi
    q = torch.zeros(hd)
    q[:half], q[half:] = amp * torch.cos(phi), amp * torch.sin(phi)
    k = torch.zeros(hd)
    k[:half] = torch.cos(theta) * q[:half] + torch.sin(theta) * q[half:]
    k[half:] = -torch.sin(theta) * q[:half] + torch.cos(theta) * q[half:]
    return q, k


@functools.cache
def holdout_rows(n=4, T=1024):
    """The first n held-out rows (the recipe's seed-1 order, last 32 rows), first T tokens."""
    from utils.prepare_data import TRAIN_ROWS, load_rows
    tr = load_rows(ROOT / 'data' / 'train.pt', TRAIN_ROWS)
    tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(1))]
    return tr[-32:][:n, :T].contiguous()


@torch.no_grad()
def attn_inputs(model, rows):
    """norm(x) entering each block's attention on the rows (list over blocks, [N, d] float)."""
    caps = {}
    hooks = [blk.attn.register_forward_pre_hook(lambda mod, inp, i=i: caps.__setitem__(i, inp[0].float().reshape(-1, inp[0].shape[-1])))
             for i, blk in enumerate(model.transformer.h)]
    try:
        dev = model.lm_head.weight.device
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            model(rows.to(dev))
    finally:
        for h in hooks:
            h.remove()
    return [caps[i] for i in range(len(model.transformer.h))]


@torch.no_grad()
def plant(model, spec, seed, verbose=True):
    """w1_ideas' plant_multi (dir 'mean', b0u) for spec = {"c", "prev"}; returns the names of the tensors it wrote."""
    g = torch.Generator(device='cpu').manual_seed(1000 + int(seed))
    c = float(spec.get('c', 0.15))
    w = model.transformer.wte.weight
    u = torch.randn(w.shape[1], generator=g)
    u = u / u.norm()
    names = []
    if c > 0:
        wf = w.float()
        w.copy_((wf + c * wf.norm(dim=1, keepdim=True) * u.to(w.device)).to(w.dtype))
        names.append('transformer.wte.weight')
    rows = holdout_rows()
    for item in sorted(spec['prev']):
        b, h, lag = int(item[0]), int(item[1]), int(item[2]) if len(item) > 2 else 1
        attn = model.transformer.h[b].attn
        X = attn_inputs(model, rows)[b]
        if b == 0 and c > 0:
            d = u.clone()
        else:
            d = X.mean(0).cpu()
            d = d / d.norm()
        proj = (X.cpu() @ d) / X.norm(dim=1).cpu()
        q, k = rope_qk(attn.head_dim, g, lag)
        sl = slice(h * attn.head_dim, (h + 1) * attn.head_dim)
        for lin, vec in ((attn.c_q, q), (attn.c_k, k)):
            lin.weight[sl] = torch.outer(vec, d).to(device=lin.weight.device, dtype=lin.weight.dtype)
        if verbose:
            print(f'[plant] block {b} head {h} lag {lag}: input share along d min {float(proj.min()):.3f} mean '
                  f'{float(proj.mean()):.3f} neg {float((proj < 0).float().mean()):.4f}', flush=True)
        names += [f'transformer.h.{b}.attn.c_q.weight', f'transformer.h.{b}.attn.c_k.weight']
    return names


def build_model(device, settings):
    """transit_base's build_model, then the plant (if set); several ranks: the planted tensors from rank 0."""
    model = PLAIN['build_model'](device, settings)
    if getattr(settings, 'plant', None) is None:
        return model
    dist = torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    rank0 = not dist or torch.distributed.get_rank() == 0
    names = plant(model, settings.plant, R.SEED, verbose=rank0)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in names if dist else ():
            torch.distributed.broadcast(params[name].data, 0)
    if rank0:
        print(f'plant_init: planted {settings.plant} at seed {R.SEED} ({len(set(names))} tensors)', flush=True)
    return model


def install(on_items=None):
    """transit_base's install (speed_stack's, combo_twin's, ...; idempotent), then this file's Settings / recipe dict and
    build_model. The speed stack goes on after."""
    if R.build_model not in (PLAIN['build_model'], build_model):
        raise RuntimeError('plant_init: build_model replaced')
    if R.Settings is Settings:
        R.Settings, R.asdict = TB.Settings, TB.asdict
    on_items = TB.install(on_items)
    if (R.Settings, R.asdict) != (TB.Settings, TB.asdict):
        raise RuntimeError('plant_init: another wrapper replaced Settings or asdict')
    R.Settings, R.asdict, R.build_model = Settings, asdict, build_model
    return on_items


# ======================================== module hq_plant
"""Settings for the plant (here {"c": 0, "prev": [[4, 0]]}: block 4, head 0, lag 1) and build_model with the plant."""
import argparse
from dataclasses import dataclass
import sys

import torch

import shared_pairs2 as SP2  # noqa: E402
import head_edit as HE  # noqa: E402
import plant_init as PI  # noqa: E402  (library use only: plant(), holdout_rows(); its install is not called)
SPR, R = SP2.SPR, SP2.R

PLAIN = dict(build_model=R.build_model, Editor=HE.Editor, rows=HE.rows)
HOLD = {}  # device -> the 32 held-out rows


@dataclass
class Settings(SPR.Settings):  # shared_pairs' fields and the two levers
    head_edit_qtest: str = 'batch'
    plant: dict = None

    def __post_init__(self):
        super().__post_init__()
        if self.head_edit_qtest not in ('batch', 'hold2') or self.head_edit_qtest == 'hold2' and not self.head_edit:
            raise ValueError('hq_plant: head_edit_qtest "batch" / "hold2" (hold2 needs head_edit)')
        if self.plant is not None:
            _plant_check(self)


def _plant_check(s):  # plant_init.Settings' check of the plant field (same message, same normal form)
    p, L, H = s.plant, s.model.n_layer, s.model.n_head
    try:
        ok = set(p) <= {'c', 'prev'} and 0 <= float(p.get('c', .15)) < 1 and all(
            len(i) in (2, 3) and 0 <= int(i[0]) < L and 0 <= int(i[1]) < H and (len(i) == 2 or int(i[2]) >= 1) for i in p['prev'])
    except (TypeError, KeyError, ValueError):
        ok = False
    if not ok or len({(int(i[0]), int(i[1])) for i in p['prev']}) != len(p['prev']):
        raise ValueError('plant_init: plant = {"c": [0, 1), "prev": [[block, head(, lag >= 1)], ...]} (distinct heads)')
    s.plant = dict(c=float(p.get('c', .15)), prev=[[int(v) for v in i] for i in p['prev']])


def asdict(settings):  # unset: shared_pairs' recipe dict
    state = SPR.asdict(settings)
    if state.get('head_edit_qtest', 'batch') == 'batch':
        state.pop('head_edit_qtest', None)
    if state.get('plant') is None:
        state.pop('plant', None)
    return state


class HoldEditor(PLAIN['Editor']):
    def loss(self, x, y):  # jump_ideas' _hloss: chunks of 8 rows, row-weighted mean (record: the last chunk's token means)
        tot = 0.0
        for xs, ys in zip(x.split(8), y.split(8)):
            tot += float(PLAIN['Editor'].loss(self, xs, ys)) * xs.shape[0]
        return tot / x.shape[0]


def editor(model, s):
    return (HoldEditor if getattr(s, 'head_edit_qtest', 'batch') == 'hold2' else PLAIN['Editor'])(model, s)


def rows(est, x, y):
    if getattr(est.s, 'head_edit_qtest', 'batch') != 'hold2':
        return PLAIN['rows'](est, x, y)
    dev = x.device
    if dev not in HOLD:
        HOLD[dev] = HOLD['cpu_rows'].to(dev)
    h = HOLD[dev]
    return h[:, :-1].contiguous(), h[:, 1:].contiguous()


def build_model(device, settings):
    model = PLAIN['build_model'](device, settings)
    if getattr(settings, 'head_edit_qtest', 'batch') == 'hold2' and 'cpu_rows' not in HOLD:
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument('--holdout', type=int, default=0)
        n = ap.parse_known_args(sys.argv[1:])[0].holdout
        if n != 32:
            raise ValueError(f'hq_plant: hold2 needs --holdout 32 (got {n})')
        HOLD['cpu_rows'] = PI.holdout_rows(32, settings.model.sequence_len + 1)
    if getattr(settings, 'plant', None) is None:
        return model
    dist = torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    rank0 = not dist or torch.distributed.get_rank() == 0
    names = PI.plant(model, settings.plant, R.SEED, verbose=rank0)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in names if dist else ():
            torch.distributed.broadcast(params[name].data, 0)
    if rank0:
        print(f'hq_plant: planted {settings.plant} at seed {R.SEED} ({len(set(names))} tensors)', flush=True)
    return model


def install(on_items=None):
    """shared_pairs2's install (idempotent), then this file's Settings / recipe dict, build_model and the two names in
    head_edit that its edit() looks up."""
    if R.build_model not in (PLAIN['build_model'], build_model):
        raise RuntimeError('hq_plant: build_model replaced')
    if R.Settings is Settings:
        R.Settings, R.asdict = SPR.Settings, SPR.asdict
    on_items = SP2.install(on_items)
    if (R.Settings, R.asdict) != (SPR.Settings, SPR.asdict) or HE.Editor not in (PLAIN['Editor'], editor) \
            or HE.rows not in (PLAIN['rows'], rows):
        raise RuntimeError('hq_plant: another wrapper replaced Settings, asdict or head_edit\'s Editor / rows')
    R.Settings, R.asdict, R.build_model = Settings, asdict, build_model
    HE.Editor, HE.rows = editor, rows
    return on_items


# ======================================== module polarfar
"""Per-head polar: ANVIL's polar step for c_q and c_k is taken per attention head (each head's 64 rows of the momentum
matrix orthogonalized on their own, then ANVIL's lane energy, cautious decay and step as before). Only the optimizer's
map from the forward-estimated credit to the step changes. (far_mu, an attention-distance penalty, is off: 0.)
"""
from dataclasses import dataclass
import math
import os

import torch
import torch.nn.functional as F

import shared_pairs2 as SP2  # noqa: E402
SPR, TG, R, FE = SP2.SPR, SP2.TG, SP2.R, SP2.FE
from utils.model import norm  # noqa: E402

STATE = dict(n=0, every=25, last=None)
BASE = {}


@dataclass
class Settings(SPR.Settings):  # shared_pairs' fields and the two switches
    anvil_head_polar: tuple = ()  # subset of ('q', 'k'): one polar per attention head for those matrices
    far_mu: float = 0.0  # far-attention-mass penalty weight (0 = off)
    far_window: int = 256
    far_until: float = 0.0833  # progress (rows before the update / epoch rows) < far_until; .0833 = updates 1-305
    far_queries: int = 64

    def __post_init__(self):
        super().__post_init__()
        self.anvil_head_polar = tuple(self.anvil_head_polar)
        if (not set(self.anvil_head_polar) <= {'q', 'k'} or len(set(self.anvil_head_polar)) != len(self.anvil_head_polar)
                or self.far_mu < 0 or self.far_window < 1 or self.far_queries < 1 or not 0 <= self.far_until <= 1):
            raise ValueError('polarfar: anvil_head_polar a subset of [q, k]; far_mu >= 0, far_window / far_queries >= 1, '
                             'far_until in [0, 1]')


def asdict(settings):  # both off: shared_pairs' recipe dict
    state = SPR.asdict(settings)
    if not state.get('anvil_head_polar', ()):
        state.pop('anvil_head_polar', None)
    else:
        state['anvil_head_polar'] = list(state['anvil_head_polar'])
    if not state.get('far_mu', 0.0):
        for key in ('far_mu', 'far_window', 'far_until', 'far_queries'):
            state.pop(key, None)
    return state


# ------------------------------------------------------------------------------------------------ per-head polar


def head_polar(M, nh, margin):
    """[B, r, c] momentum stack -> each matrix's nh row blocks orthogonalized on their own (bc.polar on every [r / nh, c]
    slice: scaled by margin x the slice norm, the record's maps in bf16)."""
    B, r, c = M.shape
    x = M.reshape(B * nh, r // nh, c)
    if x.size(1) > x.size(2):
        raise ValueError('head_polar: slices must be wide')
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    return R.newton_schulz(x).float().reshape(B, r, c)


def anvil_update_heads(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes, ratio,
                       margin, nh):
    """R.anvil_update with head_polar in place of polar_batch (eager)."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = head_polar(g.lerp(weight * fast + (1 - weight) * slow, beta), nh, margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class HeadAnvil(R.BatchedAnvil):
    """BatchedAnvil whose head matrices (ids in self.heads) form their own stacks and take anvil_update_heads."""

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
                by_shape.setdefault((tuple(p.shape), id(p) in self.heads), []).append(p)
            for (shape, heads), params in by_shape.items():
                stack = self.stacked_state(shape, params)
                g, w = torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params])
                lanes, ratio = 1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5
                if heads:
                    new = anvil_update_heads(g, w, stack['fast'], stack['slow'], stack['energy'], beta, fast_beta,
                                             self.slow_beta, weight, self.lane_beta, lr, wd, lanes, ratio, self.polar_margin,
                                             self.n_head)
                else:
                    new = R.anvil_update(g, w, stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                                         value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                                         lanes, ratio, self.polar_margin)
                torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.t += 1


def make_optimizer(model, recipe, steps):
    optimizer = BASE['make_optimizer'](model, recipe, steps)
    mats = getattr(recipe, 'anvil_head_polar', ())
    if mats:
        anvil = optimizer.optimizers[0]
        if type(anvil) is not R.BatchedAnvil:
            raise RuntimeError(f'polarfar: ANVIL is {type(anvil).__name__}, not BatchedAnvil')
        anvil.__class__ = HeadAnvil
        anvil.n_head = model.config.n_head
        anvil.heads = {id(getattr(b.attn, f'c_{m}').weight) for b in model.transformer.h for m in mats}
        if os.environ.get('RANK', '0') == '0' and not STATE.get('said'):
            STATE['said'] = True
            print(f'polarfar head polar: ANVIL per head for {list(mats)} ({len(anvil.heads)} matrices, '
                  f'{anvil.n_head} heads each)', flush=True)
    return optimizer


# ------------------------------------------------------------------------------------------- far-attention credit


def far_queries(T, W, n, seed, device):
    """The sim's draw (far.py _far_mass): n query positions in [W + 1, T) from a generator seeded `seed`."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(W + 1, T, (n,), generator=g, device=device)


def block_far_credit(a_in, q_raw, k_raw, cos, sin, H, W, window, idx):
    """One block: mean far mass P over (rows, heads, queries idx) and its closed-form local credit for c_q / c_k.
    a_in [B, T, C] the attention input (normed stream), q_raw / k_raw = a_in W^T; cos / sin [T, D / 2]."""
    B, T, C = a_in.shape
    D = C // H
    n = idx.numel()
    a_in, q_raw, k_raw = a_in.float(), q_raw.float().view(B, T, H, D), k_raw.float().view(B, T, H, D)
    cs, sn = cos[None, :, None], sin[None, :, None]
    qr, kr = R.rotate(q_raw, cs, sn), R.rotate(k_raw, cs, sn)  # the model's apply_rotary_emb
    qn, kn = F.rms_norm(qr, (D,), eps=SPR.KV.EPS), F.rms_norm(kr, (D,), eps=SPR.KV.EPS)
    qs = qn[:, idx]  # [B, n, H, D]
    dist = idx[:, None] - torch.arange(T, device=idx.device)[None, :]
    live = (dist >= 0) & (dist <= window)
    a = (torch.einsum('bnhd,bthd->bhnt', qs, kn) / math.sqrt(D)).masked_fill(~live, float('-inf')).softmax(-1)
    far = (dist > W).float()
    Fm = (a * far).sum(-1, keepdim=True)  # [B, H, n, 1]
    c = a * (far - Fm) / (B * H * n)  # dP / d logit
    dq = SPR.jn(qr[:, idx], torch.einsum('bhnt,bthd->bnhd', c, kn) / math.sqrt(D))  # through the QK-norm
    dk = SPR.jn(kr, torch.einsum('bhnt,bnhd->bthd', c, qs) / math.sqrt(D))
    dq = R.rotate(dq, cos[idx][None, :, None], -sin[idx][None, :, None])  # inverse RoPE
    dk = R.rotate(dk, cs, -sn)
    gq = dq.reshape(B * n, C).T @ a_in[:, idx].reshape(B * n, C)
    gk = dk.reshape(B * T, C).T @ a_in.reshape(B * T, C)
    return Fm.mean(), gq, gk


@torch.no_grad()
def far_credit(model, x, W, nq, seed):
    """{param name: d (mean over blocks of the mean far mass) / dW} for c_q / c_k of every block whose window exceeds W,
    each block's input held (local); the stream from one no-grad forward of the rows x (as the model computes it)."""
    T = x.shape[1]
    H = model.config.n_head
    cos, sin = model.cos[0, :T, 0].float(), model.sin[0, :T, 0].float()
    rot = (model.cos[:, :T], model.sin[:, :T])
    active = [i for i, w in enumerate(model.window_sizes) if w[0] > W]
    out, masses = {}, []
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
        h = norm(model.transformer.wte(x))
        x0 = h
        for i, block in enumerate(model.transformer.h):
            h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
            if i in active:
                a_in = norm(h)
                idx = far_queries(T, W, nq, seed + i, x.device)
                q_raw, k_raw = block.attn.c_q(a_in), block.attn.c_k(a_in)  # as the model computes them (bf16 on GPU)
                with torch.autocast(x.device.type, enabled=False):
                    P, gq, gk = block_far_credit(a_in, q_raw, k_raw, cos, sin, H, W, model.window_sizes[i][0], idx)
                masses.append(P)
                out[f'transformer.h.{i}.attn.c_q.weight'], out[f'transformer.h.{i}.attn.c_k.weight'] = gq, gk
            if i < len(model.transformer.h) - 1:
                ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
                h = block(h, ve, rot, model.window_sizes[i])
    return {k: v / len(active) for k, v in out.items()}, (torch.stack(masses).mean() if masses else None)


def add_far(est, gradients, x, y):
    """gradients (the estimate's: summed-loss credit, each GPU of a pair its part) += mu x valid / split x d mean P."""
    s = est.s
    credit, mass = far_credit(est.model, x, s.far_window, s.far_queries, 1009 * STATE['n'])
    scale = s.far_mu * float((y != -1).sum()) / est.split
    for name, g in credit.items():
        gradients[name] = gradients[name].float() + scale * g
    STATE['last'] = mass
    if os.environ.get('RANK', '0') == '0' and (STATE['n'] == 1 or STATE['n'] % STATE['every'] == 0):
        print(f'polarfar far: update {STATE["n"]} mean far mass {float(mass):.4f} (rows here {x.shape[0]}, blocks '
              f'{len(credit) // 2}, mu {s.far_mu}, W {s.far_window})', flush=True)
    return gradients


def step(self, x, y, optimizer):
    """Outermost estimator step: while the penalty is on, this update's first estimate gets the far credit."""
    s, progress = self.s, getattr(self, 'progress', None)  # None: the recipe's compile step on random tokens
    if not getattr(s, 'far_mu', 0.0) or progress is None or progress >= s.far_until:
        return BASE['step'](self, x, y, optimizer)
    STATE['n'] += 1
    done = []

    def estimate(xx, yy):
        gradients = type(self).estimate(self, xx, yy)
        if not done:
            done.append(True)
            add_far(self, gradients, xx, yy)
        return gradients
    self.estimate = estimate
    try:
        loss = BASE['step'](self, x, y, optimizer)
    finally:
        del self.__dict__['estimate']
    if not done:
        raise RuntimeError('polarfar: the step ran no estimate')
    return loss


def install(on_items=None):
    """shared_pairs2's install, then this file's Settings / recipe dict, make_optimizer and the outermost step wrapper."""
    on_items = SP2.install(on_items)
    if (R.Settings, R.asdict) != (SPR.Settings, SPR.asdict) or 'step' in vars(R.Cut):
        raise RuntimeError('polarfar: another wrapper replaced Settings / asdict, or Cut has its own step')
    R.Settings, R.asdict = Settings, asdict
    if 'step' not in BASE:
        BASE['step'], R.Boost.step = R.Boost.step, step
        BASE['make_optimizer'], R.make_optimizer = R.make_optimizer, make_optimizer
    return on_items


# ======================================== module plant_polar
"""The planted head and the per-head polar in one install, plus the mixing scalars' Adam rate (x60; the record's x30)."""
from dataclasses import dataclass
import os

import torch

import hq_plant as HQ  # noqa: E402
import polarfar as PF  # noqa: E402
SP2, SPR, R = HQ.SP2, HQ.SPR, HQ.R

PF_KEYS = ('anvil_head_polar', 'far_mu', 'far_window', 'far_until', 'far_queries')
BASE = {}


@dataclass
class Settings(HQ.Settings, PF.Settings):  # shared_pairs' fields + both modules' switches + the scalar rate
    adam_scalar_mult: float = 30.0

    def __post_init__(self):
        super().__post_init__()  # HQ -> PF -> shared_pairs (cooperative)
        if not self.adam_scalar_mult > 0:
            raise ValueError('plant_polar: adam_scalar_mult > 0')


def asdict(settings):
    state = HQ.asdict(settings)
    pf = PF.asdict(settings)
    for key in PF_KEYS:
        if key in pf:
            state[key] = pf[key]
        else:
            state.pop(key, None)
    if state.get('adam_scalar_mult', 30.0) == 30.0:
        state.pop('adam_scalar_mult', None)
    return state


def make_optimizer(model, recipe, steps):
    optimizer = BASE['make_optimizer'](model, recipe, steps)  # polarfar's (head polar) on the record's
    mult = float(getattr(recipe, 'adam_scalar_mult', 30.0))
    if mult == 30.0:
        return optimizer
    adam = optimizer.optimizers[1]
    lam = {id(p) for n, p in model.named_parameters() if 'lambdas' in n}
    groups = [g for g in adam.param_groups if any(id(p) in lam for p in g['params'])]
    if len(groups) != 1 or {id(p) for p in groups[0]['params']} != lam or 'initial_lr' in groups[0] \
            or abs(groups[0]['lr'] - recipe.lr * 30) > 1e-12:
        raise RuntimeError('plant_polar: the lambdas Adam group is not the record\'s x30 group')
    groups[0]['lr'] = recipe.lr * mult  # before the first set_rate, which records initial_lr from it
    if os.environ.get('RANK', '0') == '0' and not BASE.get('said'):
        BASE['said'] = True
        print(f'plant_polar: mixing scalars Adam rate x{mult:g} ({len(lam)} tensors)', flush=True)
    return optimizer


def install(on_items=None):
    if 'make_optimizer' in BASE:
        return on_items
    on_items = HQ.install(on_items)  # build_model, head_edit's Editor / rows (first: polarfar's Boost.step wrapper must be
    R.Settings, R.asdict = SPR.Settings, SPR.asdict  # outermost, else the stack's re-install refuses); reset for polarfar's check
    on_items = PF.install(on_items)  # Settings / asdict, make_optimizer, Boost.step
    R.Settings, R.asdict = Settings, asdict
    BASE['make_optimizer'], R.make_optimizer = R.make_optimizer, make_optimizer
    return on_items
