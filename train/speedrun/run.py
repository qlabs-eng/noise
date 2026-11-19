"""Track 2: an identity-path phase hands off to Cut; an in-run merge of a weight average, chosen part by part, ends it.

Builds on qlabs-eng/noise#56 (itself on #52, #51 and #44); the kernels it calls live next to this file: #56's, most
of them edited (visit_chain*.py, visit_gluon_root.py, query_factored_gluon.py, moment_kernels.py, key_gluon.py,
value_gluon.py, id_kernels.py, window_topk.py), and new ones for the frames (fwht.py), the roots (factor_inline.py,
root_pps.py, selected_*.py), attention (fixed_attention.py, attention_prepare.py, query_context.py,
query_sampling.py, pps_budget.py, frozen_attention.py, coordinate_gate.py), the MLPs and the descending pass
(cut_mlp.py, mlp_pack.py, route_prepare.py, descending_fused.py), the identity phase (identity_vocab.py,
local_head_prepare.py, protected_wire.py), the lm_head credit (head_response.py) and the update (qk_sphere.py,
gradient_pack.py, cut_update_graph.py).

1. Identity phase: #56's 315 updates of 8 rows, with its rate decay over the last 150 to 0.3x. Differences: the
   readout credit's degree-3 expansion keeps its first-order sum and its norm-change sums exact over the vocabulary
   and takes the five higher-order sums over one logit in four (weighted by 4; the pick a hash of update and group,
   shifted by the rank, so every logit is picked by two of the 8 ranks); each GPU scores all eight heads of its own
   row (#56: one head of every row, by two all-to-all exchanges); from update 195 on, no query credit is computed
   and no c_q is stepped; Adam's first-moment decay is 0 for lm_head; the flat gradient is all-reduced in fp32 for
   lm_head and the mixing scalars and in fp16 at scale 32 for the rest. Block 1's planted query / key pairs use lags
   1, 1, 1, 1, 2, 2, 4, 8 over the heads (#56: 1), and each lm_head row starts with the same share of its energy in
   the 32 gate channels as that token's embedding row. As in #56: each attn.c_proj starts as the norm-matched
   transpose of its c_v, each wte row with 1/256 of its energy in the gate channels; for the first 195 updates wte
   rows keep that split, the value embeddings' vocabulary-common Adam motion is amplified 4x at fixed energy, and
   each VE layer's centered Adam motion is joined by its partner layer's (1-5, 3-7). #56's pooling of the VE gates'
   credit is off.
2. Cut phase: 754 updates of 16 rows over the remaining rows (#56: 152 of 8, then 678 of 16), at 1.3x both
   optimizers' rates. An update gives each GPU two rows, estimated together by one replay of a two-row estimate
   graph (the second row drawing from a partner Cut's generators). Rows are taken one per difficulty stratum per
   8 rows (strata by the mean add-one-smoothed unigram surprisal of a row's targets, counted over the Cut rows inside
   the timer); in the first half of the plan each stratum's rows come in ascending surprisal. The estimator is
   #56's, with these differences:
   - both MLP roots are replayed on every update (#56: root 4 on every second update over 10-65% of the Cut). Root 6
     is measured first, on every token, around the readout credit times the residual scalars above it. Root 4 is
     measured on a fixed count of each row's tokens, 768 of 2048, drawn by systematic sampling with inclusion
     probabilities 40% uniform and 60% proportional to the norm of its control, root 6's decoded credit times the
     two scalars between them. A drawn token's credit is the control plus its measured deviation over its inclusion
     probability; the others keep the control. The replays are #56's (full 512-direction signed Hadamard frames,
     routed MLPs, block 5's attention over 16 strong keys for root 4, diagonal readout moments); root 6's runs in
     one kernel that composes blocks 6 and 7's routed weights per token and rounds the moved stream to bf16 once, at
     the end (#56 rounded each block's change, each sum and the scaled stream), and the per-query part of root 4's
     attention preparation runs for its drawn tokens only;
   - query credits (the degree-2 moment model's response to a finite step of each raw coordinate through the norm
     and RoPE) are measured on 24 of every 64 tokens per head, drawn with inclusion probabilities of at least 1/32
     that grow with the square root of the variance of the head's value scores under the query's clean attention,
     and divided by that probability. Key credits read bf16 operands, and the same scan returns the raw value
     credits. Block 0's c_q and c_k get no credit and no update in the Cut;
   - the value gate's input credit comes from finite moves of the block input along each of the 512 coordinate axes
     through the norm and the gate (#56: a random signed Hadamard frame, decoded);
   - the Cut's products with its signed Hadamard frames are fast Walsh-Hadamard transforms, except the factored
     query replay's projection of W_q; the lm_head credit is kept in fp16 at scale 64; the MLP credits, route
     selection and the descending value chain are fused kernels; the clean forward's attention is one tiled
     online-softmax kernel with the model's window mask (#56: flex attention).
   The ANVIL step orthogonalizes each head's c_q and c_k rows together with four Newton-Schulz maps and a terminal
   polynomial (#56: six maps), and then removes from each head's c_q or c_k change the part along that head's
   current rows (the q and k norms make the model independent of their scale). The mixing scalars' Adam rates are
   split 1 : 4 between resid_lambdas and x0_lambdas at the same root-mean-square. The estimate packs its gradients
   straight into the bf16 wire; the MLP matrices' slice is packed first and its reduce-scatter starts under the
   estimate's tail; each collective group has a communicator of its own; Adam's step is a captured graph. As in
   #56: weight decay zero after the first 275/1043 of the updates, warmdown 0.55, rank-major shards, NCCL LL128.
3. One weight average (decay 0.02, every 5 steps of 8 rows over the last 70% of the Cut rows) is merged part by
   part inside the run, as in #56 but with 11 parts: each block with its value-embedding table, the two mixing
   scalar vectors together, wte, and lm_head. Each part's coefficient a in w + a (avg - w) comes from a parabola
   through three trial values on 24 of the 64 held-out rows per rank, and the joint move is shrunk by the best of
   three factors on all 64. The merge reads only held-out rows: no row it reads feeds an update.

Run on one 8xH100 node, seed 0; measured results are in train/speedrun/log/.
As in #56, train.pt is read (with its pinned-hash check) and split by seeded permutations, and the identity batches
are put on the GPUs, before the timer; apart from the organizers' pinned-hash check, the row order is the only
computation on the rows' contents before the first update, and it is timed (on a host thread while the identity
graphs replay). The random-token warmup (both identity graphs, a throwaway Cut, the capture of the Cut's two-row
estimate graph, the ANVIL unit graphs, the Adam graph, the merge's forwards and the merge scorer) is discarded or
reset before timed training starts from seed 0, and the collectives' communicators are opened with zeros; final
validation is untimed. metrics.jsonl writes the Cut records in one flush after the last Cut update, so they share
its time.

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
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.model import GPT, GPTConfig, apply_rotary_emb, has_ve, norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visit_gluon_root  # noqa: E402  (next to this file)
import visit_chain3  # noqa: E402
import query_factored_gluon  # noqa: E402
import moment_kernels  # noqa: E402
import key_gluon  # noqa: E402
import id_kernels  # noqa: E402
import value_gluon  # noqa: E402
import attention_prepare  # noqa: E402
import fixed_attention  # noqa: E402
import coordinate_gate  # noqa: E402
import cut_mlp  # noqa: E402
import cut_update_graph  # noqa: E402
import descending_fused  # noqa: E402
import frozen_attention  # noqa: E402
import gradient_pack  # noqa: E402
import head_response  # noqa: E402
import identity_vocab  # noqa: E402
import local_head_prepare  # noqa: E402
import mlp_pack  # noqa: E402
import protected_wire  # noqa: E402
import qk_sphere  # noqa: E402
import factor_inline  # noqa: E402
import root_pps  # noqa: E402
import route_prepare  # noqa: E402
import selected_attention  # noqa: E402
import selected_payload  # noqa: E402
import query_context  # noqa: E402
import query_sampling  # noqa: E402
import fwht  # noqa: E402
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
DESCENDING_BT = 4  # tokens per program of the descending chain's kernel
FROZEN_QK_PREFIX = 1  # the Cut leaves the queries and keys of this many leading blocks as the identity phase left them
IDENTITY_FROZEN_QUERIES = 8
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
    wide_from: float = 0.0  # fraction of the Cut's cut_batch-row steps after which updates take wide_rows
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
    telemetry_batch: int = 1024
    fc_sigma: float = 0.05
    route_k_late: int = 64  # this many routes from block route_split on
    route_split: int = 2
    query_moment_sigma: float = 0.003
    query_sample_keep: int = 24
    query_sample_group: int = 64
    chain_sigma: float = 0.01
    chain_depth_roots: tuple = (4, 6)
    chain_token_fraction: tuple = (0.375, 1.0)  # per root: share of a row's tokens replayed (root 4: 768 of 2,048)
    chain_pps_floor: float = 0.4  # share of each token's draw probability that is uniform
    chain_carry: bool = True  # a root's control: the next deeper root's credit, carried down
    joint_root_groups: tuple = ((4,), (6,))
    route_k: int = 128
    chain_tokens: int = 2048
    query_replay_long_layers: tuple = (5,)
    gate_path_sigma: float = 0.01
    query_keys: int = 16
    key_sigma: float = 0.003
    emb_scale: float = 750.0
    lambda_scale: float = 15.0
    resid_rate_gain: float = 0.3429971702850177
    x0_rate_gain: float = 1.3719886811400708
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


def query_change_routed(x, pos, w_q, top_keys, top_values, top_p, cos, sin, query_inv, factored):
    """The replayed attention's key weights for a root's moves, from the root visit's routed factors (the query
    change itself is never formed). Returns the response's two factors, (weights, the gathered value tables), which
    the next visit multiplies."""
    m = x.shape[0]
    _, heads, count, width = top_keys.shape
    assert heads * count == 128 and isinstance(top_values, tuple)
    c, s = cos[pos][:, None], sin[pos][:, None]
    q0 = (norm(x) @ w_q.T).view(m, heads, width)
    r0 = torch.cat([q0[..., :width // 2] * c + q0[..., width // 2:] * s,
      q0[..., :width // 2] * -s + q0[..., width // 2:] * c], -1)
    inv0, inv = query_inv
    change, steps, scale, wo, routes, basis = factored
    at, f = query_factored_gluon.query_factors_table(wo, routes, basis, w_q)
    weights = query_factored_gluon.fused_query_replay_factored_rot(
        change, steps, scale, at, f, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_p, routes)
    return weights, top_values


@torch.compile(dynamic=False)
def norm_input_credit(x, g, sigma, H):
    width = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    step = sigma * (sq / width).sqrt().clamp(min=1e-6)
    xh, gh, gx = fwht.frame(x, H), fwht.frame(g, H), (g * x).sum(-1, keepdim=True)
    inv = torch.rsqrt(sq / width + EPS)
    inv_up = torch.rsqrt((sq + 2 * step * xh + step.square() * width) / width + EPS)
    inv_down = torch.rsqrt((sq - 2 * step * xh + step.square() * width) / width + EPS)
    up = step * gh * inv_up + gx * (inv_up - inv)
    down = -step * gh * inv_down + gx * (inv_down - inv)
    return fwht.frame((up - down) / (2 * step), H, reverse=True) / width


def root_norm_inverses(x, steps, basis):
    d = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    dots = steps * fwht.frame(x, basis)
    moved_sq = steps.square() * basis.square().sum(-1).unsqueeze(0)
    return torch.rsqrt(sq / d + EPS), torch.rsqrt((sq + 2 * dots + moved_sq) / d + EPS)


def query_inverses_from_output_partials(next_x, partials):
    stats = partials.sum(-1)
    square = next_x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(square / next_x.shape[-1] + EPS)
    inv = torch.rsqrt(
      (square + 2 * stats[:, :, 0] + stats[:, :, 1]) / next_x.shape[-1] + EPS
    )
    return inv0, inv


def chain_chunk_eager(site, root_moves, mids, xL, idx, hs, lam_r, w_fc, w_out_t, base,
    target_logit_w, y, own, moments, root_projection):
    """Replay one root: its signed frame moves enter at site and go through the routed MLPs above it to the readout.
    Root 6 (own is None): blocks 6 and 7 in one kernel (factor_inline). Root 4: its visit with the routed change,
    block 5's attention replayed on own's strong keys from those factors, then blocks 5 to 7 in one kernel
    (visit_chain3). Returns each token's moved-minus-clean loss for every move, from diagonal readout moments."""
    L = len(mids)
    assert site in (L - 4, L - 2) and all(idx[b].shape[1] == 64 for b in range(site, L))
    tail = (mids[L - 1], w_fc[L - 1], w_out_t[L - 1], idx[L - 1], hs[L - 1], xL, target_logit_w,
            moments[2], moments[3], moments[4])
    inv0, inv = root_norm_inverses(mids[site], *root_moves)
    if site == L - 2:
        terminal_stats = factor_inline.root_stats(*root_moves, root_projection, w_out_t[site], idx[site],
            hs[site], inv0, inv, lam_r[site + 1], *tail)
    else:
        b, m = site + 1, L - 2  # the block whose attention is replayed, and the one between it and the last
        pos, inputs, w_q, keys, values, prob, cos_sin = own
        delta, change, partials = visit_gluon_root.fused_root_visit_change(*root_moves, root_projection,
            w_out_t[site], idx[site], hs[site], inv0, inv, lam_r[b], inputs[b])
        factored = (change, root_moves[0], lam_r[b], w_out_t[site], idx[site], root_moves[1])
        query_inv = query_inverses_from_output_partials(inputs[b], partials)
        add_weights, (table, correction, gathered_rows) = query_change_routed(
            inputs[b], pos, w_q[b], keys[b], values[b], prob[b], cos_sin[0], cos_sin[1], query_inv, factored)
        terminal_stats = visit_chain3.fused_resp_visit_readout_fast(delta.to(torch.bfloat16), mids[b], w_fc[b],
            w_out_t[b], idx[b], hs[b], lam_r[b + 1], add_weights, table, mids[m], w_fc[m], w_out_t[m], idx[m],
            hs[m], lam_r[m + 1], *tail, gathered_rows=gathered_rows, correction=correction)
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


@torch.compile(dynamic=False, options={'combo_kernels': True})
def capture_forward(model, idx, targets, weights=None):
    """GPT.forward (utils/model.py) under bf16 autocast, its attention by fixed_attention's kernel and its relu^2 by
    cut_mlp's, returning the tensors Capture's hooks record: each Linear's input (fp32) and output, the embeddings'
    outputs, each block's input, clean mid and output streams, the pre-activation RMS and the per-token loss.
    weights: the Linears' bf16 weights by module name (cast once by the caller), else autocast casts them."""
    inputs, outputs, blocks = {}, {}, {}
    BF16_KINDS = ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        b, t = idx.shape
        cos, sin = model.cos[:, :t], model.sin[:, :t]
        def linear(name, module, z):
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
            y = fixed_attention.attention(q, k, v, model.window_sizes[i][0]).contiguous().view(b, t, -1)
            x = x + linear(p + 'attn.c_proj', attn.c_proj, y)
            outputs[p + 'mlp.clean_mid'] = blocks[i].float() + outputs[p + 'attn.c_proj'].float()
            hid = linear(p + 'mlp.c_fc', block.mlp.c_fc, norm(x))
            act, rms = cut_mlp.activation_rms(hid)
            outputs[p + 'mlp.fc_rms'] = rms
            x = x + linear(p + 'mlp.c_proj', block.mlp.c_proj, act)
            outputs[p + 'mlp.clean_out'] = outputs[p + 'mlp.clean_mid'] + outputs[p + 'mlp.c_proj'].float()
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
    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        if site == 'lm_head':
            return head_response.contract_fp16(error, x, 64.0)
        return torch.einsum('btd,bti->di', error, x)


@torch.compile(dynamic=False)
def chain_transposes(weights):
    """Each weight transposed, contiguous."""
    return [w.T.contiguous() for w in weights]


@torch.compile(dynamic=False)
def root_projections_of(fc_weights, frames):
    """Each root's bf16 c_fc rows projected on its frame (fp32, contiguous)."""
    return [fwht.frame(w, frame) for w, frame in zip(fc_weights, frames)]


@torch.compile(dynamic=False)
def route_norms(fc_weights, proj_weights, first: int, chain_sigma: float):
    radius = torch.stack([fc_weights[i].float().norm(dim=1) for i in range(first, len(fc_weights))])
    out_norm = torch.stack([proj_weights[i].float().norm(dim=0) for i in range(first, len(fc_weights))])
    return radius, out_norm


@torch.compile(dynamic=False)
def weight_credits_bf16(chains, g_top, lam_r, extras, attn_proj_in, mlp_proj_in, fc_outs, fc_in, fc_out_weights,
                        sigma: float, fc_rmss):
    """The blocks' attn.c_proj, mlp.c_proj and mlp.c_fc weight credits from the chain's block credits, the GEMMs on bf16
    operands (fp32 accumulation and output)."""
    L = len(chains)
    bf = torch.bfloat16
    x_mid, x_out, packed = mlp_pack.pack_credits(chains, g_top, lam_r, extras)
    hidden = cut_mlp.hidden_credits_rms(fc_outs, packed,
                                        [w.to(bf) for w in fc_out_weights], fc_rmss, sigma)
    if mlp_proj_in is None:  # each block's relu^2 (bf16, as the forward's c_proj GEMM input)
        # formed in the slope pass that reads the same c_fc output, then one mm per block (bitwise with the bmm)
        mlp_proj = [torch.mm(x_out[i].T, F.relu(fc_outs[i].flatten(0, 1)).square().to(bf), out_dtype=torch.float32)
                    for i in range(L)]
    elif isinstance(mlp_proj_in, list):
        mlp_proj = [torch.mm(x_out[i].T, act.flatten(0, 1), out_dtype=torch.float32)
                    for i, act in enumerate(mlp_proj_in)]
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
    C = fwht.frame(weight, H, bf16=True, transpose=True)
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
    mu1_h, nu_h = fwht.frame(first, H, bf16=True).split(n)
    moments = (b1, b2, mu1, nu, curvature)
    inv = torch.rsqrt((sq + 2 * steps * fwht.frame(x, H, bf16=True) + steps.square() * width) / width + EPS)
    a = inv - inv0
    lin = a * b1.unsqueeze(1) + inv * steps * mu1_h
    change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * steps * nu_h
              + inv.square() * steps.square() * quad - 0.5 * lin.square())
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    zy = (base_y + steps * C.T[target]) * inv
    derivative = (change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(base_y * inv0 / 15))) / steps
    return fwht.frame(derivative * (y >= 0).unsqueeze(1), H, reverse=True, bf16=True) / width, moments


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
        residual = residual - step * fwht.frame(control, frame)
    return [control + boost * (fwht.frame(residual / step, frame, reverse=True) / frame.shape[0])
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
    """A block's RoPE'd unit queries (tokens, heads, head_dim) in one compiled graph (selected_attention.prepare)."""
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


class Cut:
    def __init__(self, model, settings, seed, rank, world):
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model, seed + 1000 * rank)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * rank)
        self.move_generator = torch.Generator(self.device).manual_seed(seed + 23 + 1000 * rank)
        self.sign_generator = torch.Generator(self.device).manual_seed(seed + 29 + 1000 * rank)
        self.query_generator = torch.Generator(self.device).manual_seed(seed + 41 + 1000 * rank)
        self.partner = None  # a second Cut on the same model whose graph estimates a wide update's row 2
        self._wide_graph = None
        self._telemetry_health = []
        self._query_moment_errors = {}
        self._side_stream = torch.cuda.Stream()
        self._sources = [self]  # each estimated row's draw source (row 2 draws from the partner's generators)
    def basis(self, row=0):
        generator = self._sources[row].generator
        signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator).float() * 2 - 1
        return hadamard(self.width, self.device, torch.float32) * signs
    def batched_key_contexts(self, targets, on_window):
        """The key contexts (lse and center) and the query moments' inputs of every layer, the layers of one window size
        stacked along the head axis so that each kernel launches once per window size, each window size on a stream
        of its own. on_window(item) is called with each window's item (window, layers, kp, k_raw, lse, center) as soon
        as it is queued on its stream, and self._context_ready[window] is recorded after it. Returns the items; the
        query moments' inputs are left in self._moment_inputs."""
        d, model, s = self.capture, self.model, self.s
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        windows = {}
        for i in range(L):
            windows.setdefault(model.window_sizes[i][0], []).append(i)
        work = []
        self._query_strength = {}
        self._moment_inputs = {}  # (read by batched_key_scores_by_token)
        rows = d.batch  # (several rows: each layer's rows stacked along the head axis after it, layer-major)
        items = list(windows.items())
        origin = torch.cuda.current_stream()
        if not hasattr(self, '_context_streams'):
            self._context_streams = {window: torch.cuda.Stream() for window, _ in items}
            self._context_ready = {window: torch.cuda.Event() for window, _ in items}
        for stream in self._context_streams.values():
            stream.wait_stream(origin)
            cos.record_stream(stream)
            sin.record_stream(stream)
        for window, layers in items:
            stream = self._context_streams[window]
            with torch.cuda.stream(stream):
                pairs = [(i, r) for i in layers for r in range(rows)]
                pick = lambda name: [d.outputs[f'transformer.h.{i}.attn.{name}'][r] for i, r in pairs]
                ves = [d.outputs[f'value_embeds.{i}'][r] if has_ve(i, L) else None for i, r in pairs]
                gates = [d.outputs[f'transformer.h.{i}.attn.ve_gate'][r] if has_ve(i, L) else None for i, r in pairs]
                (kp, k_raw), self._moment_inputs[window] = attention_prepare.prepare(
                    pick('c_q'), pick('c_k'), pick('c_v'), ves, gates, [targets[i][r] for i, r in pairs], cos, sin, h,
                    s.query_moment_sigma)
                # the lse / center pass on the query moments' bf16 q / k / values / hub
                lse, center, strength = query_context.context(*self._moment_inputs[window][:4], window)
                self._query_strength[window] = strength
                work.append((window, layers, kp, k_raw, lse, center))
                on_window(work[-1])
                for z in (*work[-1][2:], *self._moment_inputs[window], strength):
                    z.record_stream(origin)
                self._context_ready[window].record(stream)
        return work
    def batched_key_scores_by_token(self, work, qkv, maps, seeds):
        """The query moments of batched_key_contexts' items, each layer's query credits written into columns
        [0, width) of the (layers, tokens, 3 width) buffer qkv (window group g's layers at maps[g]; seeds: one draw per
        (layer, row) for the sampled queries). Returns each layer's key-credit columns [width, 2 width) as (tokens,
        width) views (filled by key_gluon.key_values_into; zero for the frozen blocks); the query credits go to
        self._query_moment_errors as (rows, tokens, width) views."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys, w = {}, self.width
        rows = d.batch  # (qkv's (layer, row) stretches as (layers x rows, tokens, 3 width), maps' indices into it)
        by_row = qkv.view(-1, t, qkv.shape[-1])
        for g, (window, layers, kp, k_raw, lse, center) in enumerate(work):
            q16, k16, v16, hub16, q_raw, sq, steps = self._moment_inputs[window]
            frozen_rows = sum(i < FROZEN_QK_PREFIX for i in layers) * rows
            cut = frozen_rows * h
            active_map = maps[g][frozen_rows:]
            q16, k16, v16, hub16, kp, k_raw, lse, center = (
                z[cut:] for z in (q16, k16, v16, hub16, kp, k_raw, lse, center))
            q_raw, sq, steps = q_raw[:, cut:], sq[cut:], steps[cut:]
            query_sampling.sample_query_scores_into(
                q16, k16, v16, hub16, q_raw, sq, kp, cos[:, :hd // 2].contiguous(), sin[:, :hd // 2].contiguous(),
                lse, center, steps, by_row[:, :, :w], seeds, self._query_strength[window][cut:], active_map, window,
                s.query_sample_keep, s.query_sample_group)
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
        # every layer's query, key and value weight credits from one batched GEMM of the side-by-side errors
        weight = qkv_weight_credits_bf16(self._qkv, d.inputs['stack:attn.c_q:bf16'])
        for i in targets:
            p = f'transformer.h.{i}.attn.'
            for k, name in enumerate(('c_q', 'c_k', 'c_v')):
                gradients[p + name + '.weight'] = weight[i, k * self.width:(k + 1) * self.width]
        for site, error in errors.items():
            if not site.endswith(('attn.c_q', 'attn.c_k', 'attn.c_v')):
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
        last = {id(q) for k in anvil.rank_groups[-1] for q in anvil.batches[k][0]}
        positions = [n for n, (_, q) in enumerate(self._flat_order) if id(q) in last]
        assert positions == list(range(positions[0], positions[0] + len(positions)))
        self._early_range = (positions[0], positions[-1] + 1)  # the last comm group's parameters
        self._early_wires = getattr(self, '_early_wires', {})
        self._reduced_wires = getattr(self, '_reduced_wires', set())  # packed without the early slice
        lo, hi = self._early_range
        self._wire_order = self._flat_order[:lo] + self._flat_order[hi:]
    def packs_gradients(self):
        return getattr(self, '_packs', False)
    def step(self, x, y, optimizer):
        """One eager update from this GPU's row: the warm-up's (the timed updates are wide_step's)."""
        gradients = self.estimate(x, y)
        return self.update(gradients, y, optimizer, lambda: torch.where(y != -1, self.capture.clean_loss, 0).sum())
    def wide_step(self, x, y, optimizer):
        """A Cut update from this GPU's two rows: one replay of the two-row estimate graph (capture_pair_graph), which
        packs the gradients averaged over the rows, then the update. Row 2 draws from the partner's generators, set
        to this Cut's states one row's draws on, and this Cut's take theirs after the replay."""
        partner = self.partner
        if self._wide_graph is None or not self.packs_gradients() or partner is None:
            raise ValueError('wide updates need the captured estimate graph, packed gradients and the partner Cut')
        if x.shape[0] != 2:
            raise ValueError('a wide update estimates two rows per GPU')
        for mine, theirs, step in zip(self.graph_generators(), partner.graph_generators(), self._row_offsets):
            theirs.set_state(mine.get_state())
            theirs.set_offset(mine.get_offset() + step)
        # the rows' copies run on a side stream once the previous replay is done with its inputs (not after the
        # previous update's step, which the main stream is still running)
        main = torch.cuda.current_stream()
        if not hasattr(self, '_stage_stream'):  # (first use: after everything the main stream has queued)
            self._stage_stream, self._replay_done = torch.cuda.Stream(priority=PREFETCH_PRIORITY), torch.cuda.Event()
            self._stage_stream.wait_stream(main)
        self._stage_stream.wait_event(self._replay_done)
        with torch.cuda.stream(self._stage_stream):
            self._wide_x.copy_(x)
            self._wide_y.copy_(y)
        # the replay, after the copies: the graph waits for the main stream's queue so far through self._step_done,
        # and the main stream for the graph
        self._step_done.record()
        with torch.cuda.stream(self._stage_stream):
            self._wide_graph.replay()
        main.wait_stream(self._stage_stream)
        self._replay_done.record()
        # the update's early part, right after the replay is launched and before the host waits for it (the
        # generators' hand-off reads their device state)
        self._early = self.early_update(optimizer, self._packed)
        for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
            mine.set_state(theirs.get_state())
        losses = self._wide_loss  # (read when the update asks, before the next replay)
        return self.update(self._wide_gradients, y, optimizer, lambda: torch.where(y != -1, losses, 0).sum())
    def capture_pair_graph(self, x, y):
        """Capture the estimate of two rows (x, y: (2, T)) as one graph, row 2 drawing its frames and signs
        from the partner's generators. Measures each generator's draws per row."""
        partner = self.partner
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
                    self.estimate(self._wide_x, self._wide_y)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            restore_rng()
            graph = torch.cuda.CUDAGraph()
            for generator in generators:
                graph.register_generator_state(generator)
            torch._C._cuda_clearCublasWorkspaces()
            self._step_done = torch.cuda.Event(external=True)  # the graph's first node waits for its last record
            self._step_done.record()
            self._early_ready = torch.cuda.Event(external=True)  # recorded inside the graph
            self._early_ready.record()
            with torch.cuda.graph(graph):
                torch.cuda.current_stream().wait_event(self._step_done)
                self._wide_gradients = self.estimate(self._wide_x, self._wide_y)
        finally:
            self._sources = [self]
        self._wide_graph, self._wide_loss = graph, self.capture.clean_loss
        restore_rng()
        before = [g.get_offset() for g in generators]
        graph.replay()
        torch.cuda.synchronize()
        steps = [g.get_offset() - b for g, b in zip(generators, before)]
        half = len(steps) // 2
        if steps[:half] != steps[half:]:
            raise ValueError(f'the pair graph draws unevenly per row: {steps}')
        self._row_offsets = steps[:half]
        restore_rng()
    def early_update(self, optimizer, packed):
        """ANVIL's last comm group ahead of the rest of the update: once the estimate graph has recorded
        self._early_ready (its MLP credits are packed), the group's reduce-scatter runs on a stream of its own,
        under the estimate's tail. Returns that stream, or None (no captured graphs)."""
        anvil = optimizer.optimizers[0]
        if not (packed is not None and hasattr(self, '_early_ready') and getattr(anvil, '_unit_graphs', None)):
            return None
        wire = self._early_wires.get(packed[1].data_ptr())
        if wire is None:
            return None
        stream = self._early_stream = getattr(self, '_early_stream', None) or torch.cuda.Stream()
        stream.wait_event(self._early_ready)
        if getattr(self, '_means_done', None) is not None:  # (the averages read the weights this writes)
            stream.wait_event(self._means_done)
        with torch.cuda.stream(stream):
            anvil.early_step(wire)
        return stream
    def update(self, gradients, y, optimizer, loss_sum):
        early, self._early = getattr(self, '_early', None), None
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
        if packed is not None:  # the estimate already packed the bf16 wire; this fp32 buffer is the eager step's
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
        apart = None  # the early slice's own wire, when the estimate packed it only there
        if packed is not None and wire.data_ptr() in getattr(self, '_reduced_wires', ()):
            apart = self._early_wires[wire.data_ptr()]
        for g in reversed(range(len(anvil.rank_groups))):  # the last unit (the merged polar) first
            last = g == len(anvil.rank_groups) - 1
            if early is not None and last:  # (under way: early_update)
                works[g] = anvil.early_work
                continue
            works[g] = torch.distributed.reduce_scatter_tensor(
                anvil.rank_receives[g],
                apart if last and apart is not None else wire[starts[g]:starts[g] + len(anvil.rank_stores[g])],
                group=cut_group(g), async_op=True)
        anvil.shard_works = [works[anvil.rank_group_of[k]] for k in range(len(sizes))]
        anvil.shard_grads = list(anvil.shard_buffers)
        rest = torch.distributed.all_reduce(wire[sum(sizes) - (0 if apart is None else apart.numel()):],
                                            group=cut_group('rest'), async_op=True)
        anvil_ids = {id(q) for batch in anvil.batches for q in batch[0]}

        def finish_rest():  # between ANVIL's and the other optimizers' steps
            rest.wait()
            # bf16 params' grads are views of the reduced wire (what .to(bf16) of its fp32 copy gives),
            # fp32 params' slices copied once; the finite check reads the bf16 wire
            main = torch.cuda.current_stream()
            check = self._finite_stream = getattr(self, '_finite_stream', None) or torch.cuda.Stream()
            check.wait_stream(main)
            with torch.cuda.stream(check):
                health = torch.isfinite(torch.sum(wire, dtype=torch.float32) + (
                    0 if apart is None else torch.sum(apart, dtype=torch.float32)))
            health.record_stream(main)
            self._telemetry_health.append(health)
            updater = getattr(optimizer.optimizers[1], '_captured_update', None)
            if updater is not None:
                updater.current_wire = wire
                return
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
        if early is not None:
            torch.cuda.current_stream().wait_stream(early)
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
    def fused_descending_key_values(self, chain, targets, x0, outs, after_extra=None):
        """The attention credits below the chain: the query moments on a side stream per window, the key credits with
        the raw value credits, the value gates' input credits and the descending value chain batched, and the terminal
        value credits after it, in the layout their consumers read. Returns its outputs and the path scalars."""
        d, s, model = self.capture, self.s, self.model
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        errors, raw_values, terminal_values = {}, {}, {}
        main, side = torch.cuda.current_stream(), self._side_stream
        # every layer's query, key and value credits by token, side by side: one batched reassembly GEMM
        rows = d.batch  # (several rows: (layer, row) pairs in place of layers, row-major within a layer)
        qkv = torch.empty(L, rows * t, 3 * self.width, device=self.device)
        qkv[:FROZEN_QK_PREFIX, :, :2 * self.width].zero_()
        keys, maps = {}, []
        query_seeds = None
        def launch_window(item):  # a window's query moments on the side stream as soon as its context is queued
            nonlocal query_seeds
            window, layers, *_ = item
            mapping = self.layer_map([i * rows + r for i in layers for r in range(rows)])
            maps.append(mapping)
            side.wait_stream(torch.cuda.current_stream())
            mapping.record_stream(main)
            for z in (qkv, mapping, *item[2:], *self._moment_inputs[window], self._query_strength[window]):
                z.record_stream(side)
            with torch.cuda.stream(side):
                if query_seeds is None:
                    query_seeds = torch.stack([
                        torch.randint(0, 2**31 - 1, (L,), dtype=torch.int32, device=self.device,
                                      generator=self._sources[r].query_generator)
                        for r in range(rows)], 1).flatten()
                keys.update(self.batched_key_scores_by_token([item], qkv, [mapping], query_seeds))
        work = self.batched_key_contexts(targets, launch_window)
        raws, place, raw_groups = {}, {}, []
        for g, (window, layers, kp, k_raw, lse, center) in enumerate(work):
            main.wait_event(self._context_ready[window])
            q16, k16, v16, hub16, *_ = self._moment_inputs[window]
            cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
            key_out = qkv.view(L * rows, t, 3 * self.width)[:, :, self.width:2 * self.width]
            frozen_rows = sum(i < FROZEN_QK_PREFIX for i in layers) * rows
            cut = frozen_rows * h
            if frozen_rows:
                frozen_raw = frozen_attention.raw_values(q16[:cut], k16[:cut], lse[:cut], hub16[:cut], window, h)
                active_raw = key_gluon.key_values_into(
                    q16[cut:], k16[cut:], k_raw[cut:], lse[cut:], v16[cut:], hub16[cut:], center[cut:], cos, sin,
                    window, s.key_sigma, h, key_out, maps[g][frozen_rows:])
                raw = torch.cat((frozen_raw, active_raw), 0)
            else:
                raw = key_gluon.key_values_into(q16, k16, k_raw, lse, v16, hub16, center, cos, sin,
                                                window, s.key_sigma, h, key_out, maps[g])
            raw_groups.append(raw)
            for j, i in enumerate(layers):
                raws[i], place[i] = raw[j] if rows == 1 else raw[j * rows:(j + 1) * rows].flatten(0, 1), (g, j)
        ve_layers = [i for i in range(L) if has_ve(i, L)]
        if rows > 1:  # each (layer, row) apart, then a layer's rows joined
            pairs = [(i, r) for i in ve_layers for r in range(rows)]
            responses = coordinate_gate.gate_inputs_direct(
                [raws[i][r * t:(r + 1) * t] for i, r in pairs], [d.outputs[f'value_embeds.{i}'][r] for i, r in pairs],
                [d.blocks[i][r:r + 1] for i, r in pairs], [model.transformer.h[i].attn.ve_gate.weight for i, r in pairs],
                s.gate_path_sigma)
            gates = {i: torch.cat(responses[k * rows:(k + 1) * rows]) for k, i in enumerate(ve_layers)}
        else:
            gates = dict(zip(ve_layers, coordinate_gate.gate_inputs_direct(
                [raws[i] for i in ve_layers], [d.outputs[f'value_embeds.{i}'][0] for i in ve_layers],
                [d.blocks[i] for i in ve_layers], [model.transformer.h[i].attn.ve_gate.weight for i in ve_layers],
                s.gate_path_sigma)))
        c_vs = [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)]
        moved = [None] * L
        for (window, layers, *_), raw in zip(work, raw_groups):
            v = torch.bmm(raw.view(len(layers), rows * t, -1), torch.stack([c_vs[i] for i in layers]))
            for j, i in enumerate(layers):
                moved[i] = v[j]
        c_vs = moved
        extra_list, path = descending_fused.descending(
            [gates.get(i) for i in range(L)], [d.blocks[i] for i in range(L)], c_vs, model.resid_lambdas.float(),
            x0, outs, bt=DESCENDING_BT)
        if after_extra is not None:  # (what needs only `extra`: the MLP weight credits)
            after_extra(extra_list)
        corrected = [torch.empty(len(item[1]) * rows, t, self.width, device=self.device) for item in work]
        for g, (window, layers, *_) in enumerate(work):
            torch.bmm(stacked_sums([chain[i] for i in layers], [extra_list[i] for i in layers]),
                      torch.stack([model.transformer.h[i].attn.c_proj.weight.float() for i in layers]),
                      out=corrected[g].view(len(layers), rows * t, -1))
        w = self.width
        by_row = qkv.view(L * rows, t, 3 * w)
        for g, (window, layers, kp, k_raw, lse, center) in enumerate(work):
            value_gluon.value_pass_gluon(*self._moment_inputs[window][:2], lse, corrected[g], window,
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
        return errors, raw_values, terminal_values, dict(enumerate(extra_list)), path
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
        rows = self.capture.batch
        frames = [self.basis(r) for r in range(rows)]
        if rows > 1:
            derivative, self._head_moments = probe_moment_terms_rows(
                x, y, self.model.lm_head.weight.float(), frames)
        else:
            derivative, self._head_moments = probe_moment_terms(
                x, y, self.model.lm_head.weight.float(), frames[0])
        return derivative
    @torch.no_grad()
    def graph_generators(self):
        generators = [self.generator, self.move_generator, self.capture.shared_generator, self.capture.generator]
        return generators + [self.sign_generator, self.query_generator]
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
        groups = s.joint_root_groups
        for group in groups:
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
        replay_layers = lambda group: tuple(b for b in range(group[0] + 1, L) if b in s.query_replay_long_layers)
        active_query_layers = tuple(sorted({b for group in groups for b in replay_layers(group)}))
        raw_owns = [self.own_query_inputs(active_query_layers, r) for r in range(rows)]
        # Root 6 has no attention replay; root 4 prepares its query rows once its tokens are selected.
        assert groups == [(4,), (6,)] or set(groups) == {(4,), (6,)}
        chunks = [slice(t0, t0 + m) for t0 in range(0, n, m)]
        main = torch.cuda.current_stream()
        lanes = [None] + [self.pair_lane(r) for r in range(1, rows)] if rows > 1 else [None] * rows
        for lane in lanes[1:]:  # each further row's replays on its own stream, joined below
            lane.wait_stream(main)
        for sl in chunks:
            row = 0 if rows == 1 else sl.start // t  # (whole-row chunks: the row's own queries, keys and frames)
            with contextlib.nullcontext() if lanes[row] is None else torch.cuda.stream(lanes[row]):
                for group in sorted(groups, key=lambda g: -g[0]):  # deepest first: its credit can be the next control
                    site, = group
                    H = by_site[site][0][row]
                    deeper = [g[0] for g in groups if g[0] > site]
                    if s.chain_carry and deeper:
                        control = lam_r[site + 1:min(deeper) + 1].prod() * by_site[min(deeper)][2][sl]
                    else:
                        control = controls[site][sl]
                    fraction = s.chain_token_fraction[s.chain_depth_roots.index(site)]
                    if fraction >= 1:  # every token, no query rows
                        assert site == 6
                        take, mine, boost = sl, None, 1.0
                        steps = signed_steps[site][take]
                        picked_mids = [None if b < site else z[take] for b, z in enumerate(mids)]
                        picked_xL = xL[take]
                        picked_routes = [None if r[0] is None or b < site else r[0][take] for b, r in enumerate(routes)]
                        picked_clean = [None if r[1] is None or b < site else r[1][take] for b, r in enumerate(routes)]
                        picked_base, picked_target, picked_y = base[take], target_w[take], y[take]
                        picked_moments = tuple(z[take] for z in moments)
                    else:  # a fixed count of the row's tokens, drawn in proportion to the control's norm
                        tokens = sl.stop - sl.start
                        count = max(64, round(fraction * tokens / 64) * 64)
                        size = control.norm(dim=-1).double()
                        offset = torch.rand(1, device=self.device, dtype=torch.float64,
                                            generator=self._sources[row].move_generator)
                        inside, boost = root_pps.plan(size, offset, count, s.chain_pps_floor)
                        take = inside + sl.start
                        by_site[site][2][sl] = control
                        assert site == 4
                        mine = selected_attention.prepare(sys.modules[__name__], raw_owns[row], inside, s.query_keys)
                        (picked_mids, picked_xL, picked_routes, picked_clean, steps,
                         picked_base, picked_target, picked_y, picked_moments,
                         control) = selected_payload.selected_payload(
                            mids, xL, routes, signed_steps[site], base, target_w, y, moments, control,
                            inside, sl.start, tokens, site)
                    loss_change = chain_chunk(
                        site, (steps, H), picked_mids, picked_xL, picked_routes, picked_clean, lam_r, w_fc, w_out_t,
                        picked_base, picked_target, picked_y, mine,
                        moments=picked_moments,
                        root_projection=root_projections[site][row],
                    )
                    by_site[site][2][take] = group_credits(loss_change, [steps], [control], [H], boost)[0]
        for lane in lanes[1:]:
            main.wait_stream(lane)
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
        head = head_response.response(d.outputs['lm_head'], d.y, d.clean_loss, 0.1, fp16=True, scale=64.0)
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        first = min(group[0] for group in self.s.joint_root_groups)
        mids = [d.outputs[f'transformer.h.{i}.mlp.clean_mid'].float().flatten(0, 1) for i in range(L)]
        hpres = [d.outputs[f'transformer.h.{i}.mlp.c_fc'].flatten(0, 1) for i in range(L)]
        outs = [d.outputs[f'transformer.h.{i}.mlp.clean_out'].float().flatten(0, 1) for i in range(L)]
        radius, out_norm = route_norms(
            [blk.mlp.c_fc.weight for blk in model.transformer.h],
            [blk.mlp.c_proj.weight for blk in model.transformer.h], first, self.s.chain_sigma)
        count = self.s.route_k
        if first >= self.s.route_split:
            count = self.s.route_k_late  # every scored block keeps only its strongest route_k_late
        tops, clean = route_prepare.prepare(hpres[first:], radius, out_norm, count, scale=self.s.chain_sigma)
        routes = [(None, None)] * first + [
            (top, h) if i < self.s.route_split else
            (top[:, :self.s.route_k_late].contiguous(), h[:, :self.s.route_k_late].contiguous())
            for i, (top, h) in enumerate(zip(tops.unbind(0), clean.unbind(0)), first)]
        g_top = self.probe_moments(outs[-1], yf)
        chain = self.chain_credits(e, x0, mids, outs, routes, yf, g_top)
        return self.chain_gradients(gradients, chain, g_top, x0, outs, y)
    def pack_early(self, gradients, y):
        """ANVIL's last comm group's gradients (the merged polar's) packed apart, bit for bit their slice of
        the full wire, and the event the update's early part waits for (recorded in the graph)."""
        lo, hi = self._early_range
        self._early_wire = gradient_pack.pack([gradients[name] for name, _ in self._flat_order[lo:hi]],
                                               (y != -1).sum() * self.world)[1]
        if hasattr(self, '_early_ready'):
            self._early_ready.record()
    def chain_gradients(self, gradients, chain, g_top, x0, outs, y):
        """The estimate's weight gradients from the chain's block credits (into gradients; packed)."""
        d, model = self.capture, self.model
        L = model.config.n_layer
        b, t = y.shape
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        def mlp_credits(extra, into):
            attn_proj, mlp_proj, fc = weight_credits_bf16(
                [chain[i] for i in range(L)], g_top, lam_r, [extra[i] for i in range(L)],
                d.inputs['stack:attn.c_proj'],
                [d.inputs[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)],
                [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], d.inputs['stack:mlp.c_fc'],
                [self._bf16_weights[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)], self.s.fc_sigma,
                [d.outputs[f'transformer.h.{i}.mlp.fc_rms'] for i in range(L)])
            for i in range(L):
                p = f'transformer.h.{i}.'
                into[p + 'attn.c_proj.weight'], into[p + 'mlp.c_proj.weight'] = attn_proj[i], mlp_proj[i]
                into[p + 'mlp.c_fc.weight'] = fc[i]
        early = self.packs_gradients()

        def mlp_first(extra):  # inside the descending pass, ahead of the terminal value credits
            mlp_credits(extra, gradients)
            self.pack_early(gradients, y)
        key_errors, raw_values, terminal_values, extra, path = self.fused_descending_key_values(
            chain, targets, x0, outs, mlp_first if early else None)
        scalars = torch.empty(2, L, device=self.device)
        if not early:
            mlp_credits(extra, gradients)
        for i in range(L):
            scalars[:, i] = chain['lam', i]
        scalars += path
        self.internals(gradients, targets, b, t, key_errors, raw_values, terminal_values)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.packs_gradients():  # the bf16 wire is packed here, inside the estimate graph
            divisor, capturing = (y != -1).sum() * self.world, torch.cuda.is_current_stream_capturing()
            every = [gradients[name] for name, _ in self._flat_order]
            lo, hi = self._early_range
            rest = every[:lo] + every[hi:]
            if capturing:  # the early slice is not packed again: Cut.update reads it from its own wire
                self._packed = gradient_pack.pack(rest, divisor)
                self._reduced_wires.add(self._packed[1].data_ptr())
                self._early_wires[self._packed[1].data_ptr()] = self._early_wire
            else:  # an eager estimate packs the full wire (and compiles the captured one's pack ahead of the capture)
                self._packed = gradient_pack.pack(every, divisor)
                gradient_pack.pack(rest, divisor)
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


def cut_terminal_polar(x):
    dtype = x.dtype
    eye = torch.eye(x.size(1), device=x.device, dtype=dtype).expand(x.size(0), -1, -1)
    r = x @ x.transpose(1, 2)
    t = (r.float() / 0.51005 - eye.float()).to(dtype)
    t2 = t @ t
    part = (0.36933162752120996 * eye.float()
            - 1.3501374753347521 * t.float()
            + 1.0673457849172854 * t2.float()).to(dtype)
    high = t2 @ part
    q = (1.4015680434370221 * eye.float()
         - 0.46675738077666473 * t.float() + high.float()).to(dtype)
    return q @ x


def newton_schulz(x):
    x = x.bfloat16()
    for a, b, c in MAPS[:4]:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return cut_terminal_polar(x)


def gram_newton_schulz(x, restarts=(2,)):
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    R, Q = x @ x.transpose(1, 2), eye
    for t, (a, b, c) in enumerate(MAPS[:4]):
        if t in restarts:
            x = Q @ x
            R, Q = x @ x.transpose(1, 2), eye
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return cut_terminal_polar(Q @ x)


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


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
    ratio: float, margin: float, heads: int = 1):
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin, heads)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    proposed = w - lr * ratio * (update + decay)
    return qk_sphere.constrain(w, proposed, heads)


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
        gained = update * energy.clamp_min(1e-10).rsqrt()
        update = gained * (update.norm(dim=(1, 2), keepdim=True)
                           / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
        decay = (slow * w >= 0) * w * (wd * lr)
        news.append(w - lr * ratio * (update + decay))
    return news
NCCL_ENV = {'NCCL_PROTO': 'LL128'}


def nccl_options():
    """The collectives' internal stream at high priority (their CTAs take free SMs first)."""
    options = torch.distributed.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    return options
CUT_GROUPS = {}


def cut_group(name):
    """The Cut update's collectives `name` on a communicator of their own, so that they do not queue behind the
    update's other collectives (as the identity phase's all-gathers). Made and exercised before the timer."""
    if name not in CUT_GROUPS:
        group = dist.new_group(pg_options=nccl_options())
        probe = torch.zeros(64, device='cuda', dtype=torch.bfloat16)
        dist.all_reduce(probe, group=group)
        torch.cuda.synchronize()
        CUT_GROUPS[name] = group
    return CUT_GROUPS[name]


PREFETCH_PRIORITY = -1  # the staging stream's priority (high: its kernels take free SMs first)


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
            owned = self.batches[k][0][sl]
            frozen = [id(parameter) in self.frozen_ids for parameter in owned]
            if any(frozen):
                assert all(frozen) and self.batch_heads[k] > 1
                news = [w]
            else:
                news = [anvil_update(self.shard_grads[k].float(), w.float(), st['fast'][sl], st['slow'][sl],
                                     st['energy'][sl], *scalars, self.batches[k][2], self.batches[k][3],
                                     self.polar_margin,
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
        self.early_work = None
        self.shard_grads = None
        self.t += 1

    @torch.no_grad()
    def early_step(self, wire):
        """The last comm group's (the merged polar's) reduce-scatter from `wire` ahead of the others', on the
        current stream (Cut.update takes self.early_work as that group's shard work)."""
        units, g = self.shard_units(), len(self.rank_groups) - 1
        assert set(units[-1]) == set(self.rank_groups[g])
        self.early_work = torch.distributed.reduce_scatter_tensor(self.rank_receives[g], wire, group=cut_group(g),
                                                                  async_op=True)

    def rank_gather(self, g):
        work = dist.all_gather_into_tensor(self.rank_stores[g], self.rank_sends[g], group=cut_group(g),
                                           async_op=True)
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
    # model. The seed and the init draws set the default generators' states.
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
        updater = getattr(self, '_captured_update', None)
        if updater is not None:
            return updater.graphs[updater.current_wire.data_ptr()].replay()
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
        else recipe.lambda_scale * recipe.resid_rate_gain if name == 'resid_lambdas'
        else recipe.lambda_scale * recipe.x0_rate_gain if name == 'x0_lambdas'
        else recipe.lambda_scale if 'lambdas' in name else 1)
        groups.setdefault(scale, []).append(parameter)
    adam = ExactAdamW(
      [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
      betas=recipe.betas, eps=1e-8, weight_decay=0.0, fused=True,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
      polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    anvil.frozen_ids = {id(getattr(block.attn, f'c_{kind}').weight)
                        for block in model.transformer.h[:FROZEN_QK_PREFIX] for kind in HEAD_POLAR}
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
    """ANVIL's per-update scalars for these updates: each the mean of its steps' rows, the rate scaled."""
    anvil = optimizer.optimizers[0]
    table, rows = anvil.scalar_schedule, []
    for group in updates:
        row = table[group[0] - 1:group[-1]].mean(0)
        row[5] *= settings.wide_anvil_scale
        rows.append(row)
    anvil.scalar_schedule = torch.stack(rows)


def merge_part(name):
    """The merge part a parameter belongs to: its block (with the block's value embedding), the two mixing scalars
    together, or itself."""
    if name.startswith('transformer.h.'):
        return name.split('.')[2] + '.block'
    if name.startswith('value_embeds.'):
        return name.split('.')[1] + '.block'
    if name in ('resid_lambdas', 'x0_lambdas'):
        return 'mix_lambdas'
    return name


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
    return dict(quad_laps=dict(zip(('setup', 'center', 'probes', 'shrink'), laps)), merge_parts=len(parts),
        quad_shrink=QUAD_SHRINKS[best], quad_holdout=dict(zip(map(str, QUAD_SHRINKS), losses)),
                holdout_merged=losses[best],
                quad_vertices={p: round(QUAD_C0 + d, 4) for p, d in zip(parts, shift)})


F32, BF16 = torch.float32, torch.bfloat16
BATCH, SEQ = 8, 2048
ID_SEQ_LEN = 1024  # identity sequence length: each 2049-token row trains as two 1024-token sequences
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8
LR_WTE, LR_VE, LR_LAMBDAS = 1.8, 1.8, 0.006
ANVIL_LR = 0.04
ANVIL_DECAY = 0.09  # identity-phase ANVIL decay (lr x wd, wd 2.25)
GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1
ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240)  # #44's ANVIL momentum warmup: the first 240 of 1194 updates (see anvil_betas)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_RHO, START_CONTENT = 1, 0.9, 0.25
START_LAGS = (1, 1, 1, 1, 2, 2, 4, 8)


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


SHAPING_UPDATES = 195  # the three update-shaping mechanisms below act on identity updates 0..194
SHAPING_ON = True  # (set while each graph is traced: False drops the mechanisms and freezes the queries)
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


VE_PARTNER_GAIN = 0.5
VE_PARTNERS = (2, 3, 0, 1)  # each VE layer's partner by index in VE_BLOCKS (1, 3, 5, 7): 1<->5, 3<->7


def shape_values_closed(deltas, frames, step_t):
    """Each value-embedding table's vocabulary-common Adam motion amplified VE_COMMON_GAIN-fold at fixed energy, then
    mixed with its partner table's motion (VE_PARTNERS: the layer with the same SSSL window size), mapped through the
    partner's c_proj frame and back through its own, at fixed centered energy; in closed form. Completed Adam motion
    only: no transformed tensor returns to a nonlinear probe.
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


@torch.no_grad()
def initialize_head_sector_energy(head, embedding):
    norm2 = head.float().square().sum(dim=0, keepdim=True)
    e = embedding.float().T
    fraction = e[:GATE_CHANNELS].square().sum(dim=0, keepdim=True) / e.square().sum(dim=0, keepdim=True)
    inside = head[:GATE_CHANNELS].float()
    outside = head[GATE_CHANNELS:].float()
    head[:GATE_CHANNELS].copy_(inside * torch.sqrt(norm2 * fraction / inside.square().sum(dim=0, keepdim=True)))
    head[GATE_CHANNELS:].copy_(outside * torch.sqrt(norm2 * (1 - fraction) / outside.square().sum(dim=0, keepdim=True)))
    actual_norm2 = head.float().square().sum(dim=0, keepdim=True)
    actual_fraction = head[:GATE_CHANNELS].float().square().sum(dim=0, keepdim=True) / actual_norm2
    assert torch.allclose(actual_norm2, norm2, rtol=2e-6, atol=1e-10)
    assert torch.allclose(actual_fraction, fraction, rtol=2e-6, atol=1e-9)
    assert head.untyped_storage().data_ptr() != embedding.untyped_storage().data_ptr()


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
        initialize_head_sector_energy(self.lm_head, self.wte)
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
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAGS[h], START_RHO, generator)):
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


IDENTITY_VOCAB_STRIDE = 4


REPEAT, REPEAT_K = HD // 16, HD // 32


def identity_frozen_queries():
    """The blocks whose queries get no identity update: none while the shaping mechanisms act, then the first
    IDENTITY_FROZEN_QUERIES."""
    return 0 if SHAPING_ON else IDENTITY_FROZEN_QUERIES


def id_head_scores(got, rope32, back, b, t, prepared):
    q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = prepared
    frozen = identity_frozen_queries()
    if frozen:
        back[:, :frozen, 0].zero_()
    if frozen < L:
        first = frozen * b
        moment_kernels._moment_query_scores(q1[first:], k1[first:], values[first:], hub[first:],
            got[:, frozen:], sq_q[first:], None, *rope32, lse[first:], hub_out[first:], steps_q[first:],
            back[:, frozen:, 0], REPEAT, pair_ops=tuple(p[first:] for p in k1f))
    moment_kernels.moment_key_scores(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k,
                                     back[:, :, 1], back[:, :, 2], REPEAT_K)


def compact_head_inputs(cache, targets_o):
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])
    return ([mine(cache[f'h{i}.q']) for i in range(L)],
            [mine(cache[f'h{i}.k']) for i in range(L)],
            [mine(cache[f'h{i}.values']) for i in range(L)],
            [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)],
            [targets_o[i] for i in range(L)], [cache[f'h{i}.lse'] for i in range(L)])


def compact_head_scores(inputs, rope32, back, b, t):
    prepared, raw = local_head_prepare.prepare(*inputs, *rope32, b, t, H, KEY_SIGMA, NORM_EPS)
    id_head_scores(raw, rope32, back, b, t, prepared)
    return prepared, raw


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


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


# identity_grad's collectives: the wide gradients' reduce-scatter is issued as soon as they are complete (the
# collectives' stream idles then), and their ANVIL update and all-gather run on a side stream from there
# (the step's last read of wide_bf, the hidden credits, precedes the fork). The ANVIL all-gathers use a second
# communicator, so they do not queue behind the flat all-reduce.
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
        gather = dist.all_gather_into_tensor(model.wide_bf, new, group=model.gather_group, async_op=True)
    return gather, new, side


def identity_grad(model, idx, targets, rows_total):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens, all eight heads' queries, keys and values of its own row. The ANVIL matrices' gradients stay this
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
    credit_top = identity_vocab.credit(sys.modules[__name__], model, stream(f'h{L - 1}.stream'), y,
                                       IDENTITY_VOCAB_STRIDE)
    cred, credb = block_credits(credit_top, model.lam_r)
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))
    assert world == H
    coords = coordinate_change(cache['lm_head'], y)
    prevs = torch.stack([x.reshape(-1, x.shape[-1])
                         for x in [cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)]]).float()
    n1s, os_, n2s, acts = ([mine(cache[f'h{i}.{key}']) for i in range(L)]
                           for key in ('in.attn', 'in.attn.c_proj', 'in.mlp.c_fc', 'in.mlp.c_proj'))
    hidden = lambda: id_kernels.hidden_credits_fused([cache[f'h{i}.mlp.c_fc'] for i in range(L)],
                                                     cache.get('fc_steps'), credb, model.wide_bf[L:])
    wide = lambda: (id_kernels.at_b_layers(n2s, list(hidden()), gwide[:L]),
                    id_kernels.at_b_layers(list(credb), acts, gwide[L:]))
    wide()
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    works = {}
    works['wide'] = dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True)
    works['wide_update'] = early_wide_update(model, works['wide'], n)
    # each rank scores the heads of its own rows, where they were computed (no exchange between ranks)
    got2 = torch.empty(H, L, 3, m, HD, device=idx.device, dtype=BF16)
    compact_head_scores(compact_head_inputs(cache, targets_o), rope32, got2, rows_total, t)
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    id_kernels.at_b_layers(os_, list(credb), gsq[3 * L:])
    id_kernels.square_credit_grads(n1s, got2, gsq[:3 * L], frozen_queries=identity_frozen_queries())
    vcredits = [got2[:, i, 2].permute(1, 0, 2) for i in VE_BLOCKS]
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        gate, rows = value_embed_credits(mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(gate)
        model.grad[f've{i}'].zero_().index_add_(0, tokens, rows)
    works['flat'] = protected_wire.reduce(model)
    return loss, works, n


def anvil_betas(t):
    # ANVIL_RUN's momentum warmup (0.85 -> 0.93 over 240 of 1194 updates) scaled to the 1831-update identity
    # schedule lasts 368 updates, so all 315 identity updates are on the ramp.
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
            frozen = max(0, min(sl.stop, identity_frozen_queries()) - sl.start) if name == 'square' else 0
            if frozen == 0:
                new = anvil_math(w[sl], model.owned[name], n, st['fast'][sl], st['slow'][sl], st['energy'][sl],
                                 model.step_t, model.rates[name][sl])
            elif frozen == sl.stop - sl.start:
                new = w[sl].to(BF16)
            else:
                active = slice(sl.start + frozen, sl.stop)
                changed = anvil_math(w[active], model.owned[name][frozen:], n, st['fast'][active],
                    st['slow'][active], st['energy'][active], model.step_t, model.rates[name][active])
                new = torch.cat((w[sl][:frozen].to(BF16), changed))
            gathers.append((dist.all_gather_into_tensor(w_bf, new, group=model.gather_group, async_op=True), new))
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
    c2 = 1 - BETA2 ** t
    value_deltas = []
    scale = id_rate_scale(step_t)
    for index, (p, g, m, v, lr) in enumerate(zip(ps, gs, ms, vs, lrs)):
        beta1 = 0.0 if index == 1 else BETA1
        c1 = 1 - beta1 ** t
        lr = lr * scale
        g = g / n
        if p.dtype == BF16:
            g = g.to(BF16).float()
        m_new = beta1 * m + (1 - beta1) * g
        v_new = BETA2 * v + (1 - BETA2) * g.square()
        if 8 <= index < 12:  # the value-embedding tables
            delta = lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)
            value_deltas.append(delta)
        else:
            p.copy_((p.float() - lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)
    transferred = shape_values_closed(torch.stack(value_deltas), frames, step_t) if SHAPING_ON else value_deltas
    for i in range(4):
        ps[8 + i].copy_((ps[8 + i].float() - transferred[i]).to(ps[8 + i].dtype))


def adam_update(model, works, n, frames):
    works['flat'].wait()
    ps, gs, lrs = zip(*model.adam)
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
    def run(self):
        for i in range(IDENTITY_UPDATES):
            (self.plain_graph if self.plain_graph is not None and i >= SHAPING_UPDATES else self.graph).replay()
        self.model.gather_fp32()
    def close(self):
        """Drops the graphs and the model; their blocks stay in the caching allocator for the Cut (no cudaFree in the
        timer)."""
        self.graph = self.plain_graph = None
        self.model = None


CURRICULUM_PREFIX = 0.5


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
    prefix = round(CURRICULUM_PREFIX * len(plan))
    selected = plan[:prefix].copy()
    for column in range(batch):
        ranks = np.argsort(score[selected[:, column]], kind='stable')
        plan[:prefix, column] = selected[ranks, column]
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
    horizon = len(train_indices) // settings.cut_batch  # every remaining row: 14648 - 8*315 - 64 = 8*1508
    schedule = cut_updates(horizon, settings)  # one ANVIL schedule row per update
    if any(len(group) != 2 for group in schedule):
        raise ValueError('every Cut update takes two rows per GPU (wide_rows = 2 cut_batch from the first update on)')
    stop = min(max(args.steps or horizon, schedule[0][-1]), horizon)
    updates = [group for group in schedule if group[-1] <= stop]
    stop = updates[-1][-1]
    train_count = stop * settings.cut_batch
    selected_train_indices = train_indices[:train_count]
    unused_indices = train_indices[train_count:]
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
                    upstream='qlabs-eng/noise@b0a8675ea3aa1cd0ccb3661b409a978d3d6e89e2',
                    torch_version=torch.__version__,
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
    # seed below, and the timed Cut captures its estimate graph on two of the same rows.
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
    del model, optimizer, estimator, inp, target, batch
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(device, SEED, settings.model)
    optimizer, estimator = (
        make_optimizer(model, settings, horizon),
        Cut(model, settings, SEED, rank, world),
    )
    widen_anvil_schedule(optimizer, schedule, settings)
    estimator.bind_flat_layout(optimizer)  # (the graph packs the gradients in step's layout)
    estimator.partner = Cut(model, settings, SEED, rank, world)
    pair_rows = noise[:2].to(device)
    estimator.capture_pair_graph(pair_rows[:, :-1].contiguous(), pair_rows[:, 1:].contiguous())
    del pair_rows
    optimizer.optimizers[0].capture_shard_graphs(torch.bfloat16)
    updater = cut_update_graph.AdamGraphs(estimator, optimizer, schedule, settings, horizon)
    updater.capture([estimator._packed])
    optimizer.optimizers[1]._captured_update = updater
    log = (args.output / 'metrics.jsonl').open('w') if rank == 0 else None
    scorer = prepare_resident_merge_score(model, settings, rank, world)
    del noise
    # Untimed, as in #44 (qlabs-eng/noise@3343973b, train_stages): read train.pt, split it by the seeded permutations
    # above and put the identity phase's batches on the GPU. No split or order here depends on the rows' contents;
    # load_rows only verifies their pinned hash.
    all_rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    cut_rows = all_rows.index_select(0, selected_train_indices)
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
    for group_name in (*range(len(optimizer.optimizers[0].rank_groups)), 'rest',):
        cut_group(group_name)
    torch.cuda.synchronize()
    begin_training()
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    def prepare_rows():
        """The timed host setup of the Cut rows (no CUDA calls): their order, the ordered rows (into the pinned staging
        buffer), the disjointness check and row_usage.json."""
        order = torch.from_numpy(difficulty_order(cut_rows, settings))  # timed: it reads the rows' tokens
        used = selected_train_indices[order]
        unused = unused_indices
        rows = torch.index_select(cut_rows, 0, order, out=staged_rows)
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
            batch = rank_slice(training[begin:end])
            # views of the device rows, copied into the graphs' inputs on the staging stream
            rows_in, rows_out = (batch[:, :-1], batch[:, 1:])
            loss = estimator.wide_step(rows_in, rows_out, optimizer)
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
        if log is not None:
            log.close()


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--no-ckpt', action='store_true')
    parser.add_argument('--steps', type=int,
                        help='8-row Cut steps to run after the identity phase (two per update); the schedule is kept.')
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
    if torch.cuda.is_initialized():
        raise RuntimeError('NCCL_ENV must be set before CUDA starts')
    os.environ.update({key: str(value) for key, value in NCCL_ENV.items()})  # before NCCL initializes
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
