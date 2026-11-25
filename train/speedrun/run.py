"""Forward-only Noise training on eight H100s: track 2, the speedrun to validation loss 3.5.

Builds on qlabs-eng/noise#59 (the upstream commit in config.json) and keeps its three parts.

Identity phase: 315 updates of eight rows. The first 195 split each row into two 1,024-token
contexts. The last 120 use one 2,048-token context, and their batches are reordered within windows
of six updates.

Cut: 758 updates of 16 rows, two per GPU. Root 6 (block 6) is replayed on every position and root 4
(block 4) on 544 positions per row. Those are drawn systematically along a Morton order of each
position's input and target token, with a mass that is 25% uniform and 75% proportional to the norm
of root 6's credit carried down to it, and weighted by the inverse inclusion probability. Three
second-order sums of the readout expansion use 512 of the 4,096 classes. Query credits are measured
on 20, 28 or 32 of every 64 positions, with #59's query and key measurement. The weight gradients of
the blocks' matrices are batched cuBLAS products (cublasGemmBatchedEx through ctypes) written into
the bf16 reduce-scatter buffer.

End: every tensor is blended with a weighted average of 48 weight snapshots taken from Cut update
230 on. The blend's coefficients and the snapshot schedule are constants of the code, and the blend
reads no row.

All 14,648 training rows feed exactly one update. Before the timer the script builds both models and
optimizers from seed 0, captures its CUDA graphs and compiles its kernels on random tokens, restores
the state those passes changed, runs each Cut collective and the default group's all-gather once, on
zeros or on the seed-0 weights, and makes one 4 GiB allocation so that the caching allocator holds
free blocks. It also reads train.pt, splits it by the seeded permutations and puts the Identity
batches on the GPUs. No statistic of a training row is computed before the timer, and validation
runs after it.

    uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
import contextlib
from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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
import receiver_tiles  # noqa: E402
import token_value  # noqa: E402
import root_carry  # noqa: E402
import initial_hub  # noqa: E402
import mlp_replicas  # noqa: E402
import attention_wire  # noqa: E402
import id_kernels  # noqa: E402
import identity_lexical  # noqa: E402
import identity_spacing  # noqa: E402
import lexical_peers  # noqa: E402
import attention_prepare  # noqa: E402
import fixed_attention  # noqa: E402
import gate_head_dot  # noqa: E402
import coordinate_gate  # noqa: E402
import cut_mlp  # noqa: E402
import cut_update_graph  # noqa: E402
import frozen_attention  # noqa: E402
import clean_anchor  # noqa: E402
import cap_lookup  # noqa: E402
import identity_head_product  # noqa: E402
import identity_vocab  # noqa: E402
import local_head_prepare  # noqa: E402
import scan_operands  # noqa: E402
import value_row_owners  # noqa: E402
import paired_walsh  # noqa: E402
import terminal_panels  # noqa: E402
import replay_anchor  # noqa: E402
import pair_sampling  # noqa: E402
import route_prepare  # noqa: E402
import query_bank  # noqa: E402
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
    """Call after the random-token warmup, with every training state built from or restored to its seed-0 start."""
    _boundary(BEGIN_MARKER)


def end_training() -> None:
    """Call after the last update and the weight-average blend."""
    _boundary(END_MARKER)


SEED = 0
ROOT4_POSITIONS = 544
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
    batch_history_sign: int = -1
    average_knots: int = 48
    late_cut_decay_multiplier: float = 0.0  # ANVIL weight-decay factor after the first 275/1043 of Cut updates
    cut_batch: int = 8
    wide_rows: int = 16  # rows per Cut update from wide_from on (two per GPU, by one two-row estimate replay)
    wide_from: float = 0.0  # fraction of the Cut's cut_batch-row steps after which updates take wide_rows
    wide_lr_scale: float = 1.2  # AdamW's rates in the wide updates, relative to the schedule
    wide_anvil_scale: float = 1.2  # ANVIL's rate in the wide updates, relative to its schedule
    lr: float = 0.0028
    betas: tuple = (0.4, 0.98)
    warmdown: float = 0.55
    final: float = 0.35
    anvil_lr: float = 0.036
    anvil_wd: float = 2.25
    fast_weight: float = 1.0
    polar_margin: float = 1.1
    fc_sigma: float = 0.05
    route_k_late: int = 64  # routes kept per token in each scored block
    query_moment_sigma: float = 0.003
    depth_query_low: int = 20
    depth_query_high: int = 28
    depth_query_panel: int = 1024
    query_sample_group: int = 64
    chain_sigma: float = 0.014
    chain_depth_roots: tuple = (4, 6)
    chain_pps_floor: float = 0.25  # share of each token's draw probability that is uniform
    joint_root_groups: tuple = ((4,), (6,))
    chain_tokens: int = 2048
    query_replay_long_layers: tuple = (5,)
    gate_path_sigma: float = 0.01
    query_keys: int = 16
    key_sigma: float = 0.0045
    emb_scale: float = 600.0
    lambda_scale: float = 15.0
    resid_rate_gain: float = 0.3429971702850177
    x0_rate_gain: float = 1.3719886811400708


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


def query_change_routed(pos, top_keys, top_values, top_p, cos, sin, query_inv, factored, prepared):
    """The replayed attention's key weights for a root's moves, from the root visit's routed factors (the query
    change itself is never formed). Returns the response's two factors, (weights, the gathered value tables), which
    the next visit multiplies."""
    inv0, inv = query_inv
    change, steps, scale, routes = factored
    at, f, r0 = prepared
    weights = query_factored_gluon.fused_query_replay_factored_rot(
        change, steps, scale, at, f, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_p, routes)
    return weights, top_values


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
    target_logit_w, y, own, moments, root_projection, terminal_offsets, weight_maps):
    """Replay one root: its signed frame moves enter at site and go through the routed MLPs above it to the readout.
    Root 6 (own is None): blocks 6 and 7 in one kernel (terminal_panels: mean curvature plus a sampled remainder).
    Root 4: its visit with the routed change, block 5's attention replayed on own's strong keys from those factors,
    then blocks 5 to 7 in one kernel (visit_chain3: diagonal moments). Returns each token's moved-minus-clean losses."""
    L = len(mids)
    assert site in (L - 4, L - 2) and all(idx[b].shape[1] == 64 for b in range(site, L))
    tail = (mids[L - 1], w_fc[L - 1], w_out_t[L - 1], idx[L - 1], hs[L - 1], xL, target_logit_w,
            moments[2], moments[3], moments[4])
    inv0, inv = root_norm_inverses(mids[site], *root_moves)
    if site == L - 2:
        meta = replay_anchor.context(base, y, *moments[:2])
        return terminal_panels.root_stats(*root_moves, root_projection, w_out_t[site], idx[site],
            hs[site], inv0, inv, lam_r[site + 1], *tail, terminal_offsets, meta, weight_maps)
    else:
        b, m = site + 1, L - 2  # the block whose attention is replayed, and the one between it and the last
        pos, inputs, keys, values, prob, cos_sin = own[:6]
        delta, change, partials = visit_gluon_root.fused_root_visit_change(*root_moves, root_projection,
            w_out_t[site], idx[site], hs[site], inv0, inv, lam_r[b], inputs[b])
        factored = (change, root_moves[0], lam_r[b], idx[site])
        query_inv = query_inverses_from_output_partials(inputs[b], partials)
        add_weights, payload = query_change_routed(
            pos, keys[b], values[b], prob[b], cos_sin[0], cos_sin[1], query_inv, factored, own[6])
        table, correction, gathered = payload
        terminal_stats = visit_chain3.fused_resp_visit_readout_fast(delta.to(torch.bfloat16), mids[b], w_fc[b],
            w_out_t[b], idx[b], hs[b], lam_r[b + 1], add_weights, table, mids[m], w_fc[m], w_out_t[m], idx[m],
            hs[m], lam_r[m + 1], *tail, gathered, correction)
    return replay_anchor.loss(terminal_stats, base, y, *moments[:2])
chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)


@torch.compile(dynamic=False, options={'combo_kernels': True})
def capture_forward(model, idx, weights):
    """GPT.forward (utils/model.py) under bf16 autocast, its attention by fixed_attention's kernel and its relu^2 by
    cut_mlp's, returning what the estimate reads: each Linear's input (bf16; fp32 for ve_gate and lm_head) and output,
    the embeddings' outputs, each block's input, clean mid and output streams and the pre-activation RMS.
    weights: the Linears' bf16 weights by module name, cast or published by the caller."""
    inputs, outputs, blocks = {}, {}, {}
    BF16_KINDS = ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        b, t = idx.shape
        cos, sin = model.cos[:, :t], model.sin[:, :t]
        def linear(name, z):
            inputs[name] = z.to(torch.bfloat16) if name.endswith(BF16_KINDS) else z.float()
            out = F.linear(z, weights[name])
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
            inputs[p + 'attn.c_q'] = xn.float()
            qkv = F.linear(xn, weights[p + 'attn.qkv'])
            c, ck = attn.c_q.weight.shape[0], attn.c_k.weight.shape[0]
            q, k, v = qkv[..., :c], qkv[..., c:c + ck], qkv[..., c + ck:]
            outputs[p + 'attn.c_q'], outputs[p + 'attn.c_k'], outputs[p + 'attn.c_v'] = q, k, v
            q, k, v = q.view(b, t, h, hd), k.view(b, t, attn.n_kv_head, hd), v.view(b, t, attn.n_kv_head, hd)
            if ve is not None:
                gate = 2 * torch.sigmoid(linear(p + 'attn.ve_gate', xn[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve.view(b, t, attn.n_kv_head, hd)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            y = fixed_attention.attention(q, k, v, model.window_sizes[i][0]).contiguous().view(b, t, -1)
            x = x + linear(p + 'attn.c_proj', y)
            outputs[p + 'mlp.clean_mid'] = blocks[i].float() + outputs[p + 'attn.c_proj'].float()
            hid = linear(p + 'mlp.c_fc', norm(x))
            act, rms = cut_mlp.activation_rms(hid)
            outputs[p + 'mlp.fc_rms'] = rms
            x = x + linear(p + 'mlp.c_proj', act)
            outputs[p + 'mlp.clean_out'] = outputs[p + 'mlp.clean_mid'] + outputs[p + 'mlp.c_proj'].float()
        linear('lm_head', norm(x))
    L = len(model.transformer.h)
    for kind in ('attn.c_q', 'attn.c_proj', 'mlp.c_fc'):
        for i in range(L):
            name = f'transformer.h.{i}.{kind}'
            inputs[name] = inputs[name].to(torch.bfloat16)
    return inputs, outputs, blocks


@torch.compile(dynamic=False)
def contract_fp16(error, inputs, scale=64.0):
    return torch.mm(error.flatten(0, 1).T, inputs.flatten(0, 1).to(torch.float16),
                    out_dtype=torch.float32) / scale


class Capture:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.weights = None  # (the Linears' bf16 weights, set by Cut.estimate)
    def capture(self, x, y):
        """One compiled forward that returns the Linears' inputs and outputs and the blocks' inputs."""
        self.y = y
        self.batch, self.length = x.shape
        self.inputs, self.outputs, self.blocks = capture_forward(self.model, x, self.weights)
    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        if site == 'lm_head':
            return contract_fp16(error, x, 64.0)
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
def group_credits(loss_change, steps, controls, frames, boost: float):
    """Each root's credit from its replay group's loss changes: the identity-path predictions of every root removed,
    the remainder read along the root's frame and added back to its own prediction."""
    residual = loss_change
    for step, control, frame in zip(steps, controls, frames):
        residual = residual - step * fwht.frame(control, frame)
    return [control + boost * (fwht.frame(residual / step, frame, reverse=True) / frame.shape[0])
            for step, control, frame in zip(steps, controls, frames)]


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


def master_view(weights):
    first = weights[0]
    offsets = [w.storage_offset() for w in weights]
    stride = offsets[1] - offsets[0]
    return first.as_strided((8, 512, 512), (stride, 512, 1))


class RowGenerators:
    """One estimated row's generators: frame signs, root 4's positions, step signs, query seeds, readout classes and
    root 6's remainder coordinates. The Cut is its first row's set; its partner is the second row's."""
    def __init__(self, device, seed, rank):
        self.generator = torch.Generator(device).manual_seed(seed + 17 + 1000 * rank)
        self.move_generator = torch.Generator(device).manual_seed(seed + 23 + 1000 * rank)
        self.sign_generator = torch.Generator(device).manual_seed(seed + 29 + 1000 * rank)
        self.query_generator = torch.Generator(device).manual_seed(seed + 41 + 1000 * rank)
        self.vocab_generator = torch.Generator(device).manual_seed(seed + 53 + 1000 * rank)
        self.terminal_generator = torch.Generator(device).manual_seed(seed + 107 + 1000 * rank)
    def graph_generators(self):
        return [self.generator, self.move_generator, self.sign_generator, self.query_generator,
                self.vocab_generator, self.terminal_generator]


class Cut(RowGenerators):
    def __init__(self, model, settings, seed, rank, world, publication, cap_tables):
        self.mlp_publication = publication
        self.cap_tables = cap_tables  # (the Identity phase's)
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model)
        self.device = self.capture.device
        self.width = model.config.n_embd
        super().__init__(self.device, seed, rank)
        self.partner = RowGenerators(self.device, seed, rank)
        self._wide_graph = None
        self._query_moment_errors = {}
        self._side_stream = torch.cuda.Stream()
        self.route_pair_stream = torch.cuda.Stream()
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
                operands = [targets[i] for i in layers]
                hub = initial_hub.window(operands, self._terminal_writers, layers)
                (kp, k_raw), self._moment_inputs[window] = attention_prepare.prepare(
                    pick('c_q'), pick('c_k'), pick('c_v'), ves, gates, cos, sin, h,
                    s.query_moment_sigma, hub=hub)
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
        [0, width) of the (layers, tokens, 2 width) buffer qkv (window group g's layers at maps[g]; seeds: one draw per
        (layer, row) for the sampled queries). Returns each layer's key-credit columns [width, 2 width) as (tokens,
        width) views (filled by receiver_tiles.key_values_into; zero for the frozen blocks); the query credits go to
        self._query_moment_errors as (rows, tokens, width) views."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys, w = {}, self.width
        rows = d.batch  # (qkv's (layer, row) stretches as (layers x rows, tokens, 2 width), maps' indices into it)
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
                s.query_sample_group,
                draw_labels=d.inputs['transformer.wte'],
                depth=(rows, s.depth_query_low, s.depth_query_high, s.depth_query_panel))
            for i in layers:
                keys[i] = qkv[i, :, w:2 * w]
                self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = qkv[i, :, :w].view(rows, t, w)
        return keys
    def internals(self, gradients, targets, key_errors, terminal_values, divisor):
        d = self.capture
        errors = self._query_moment_errors
        errors.update(key_errors)
        for i in targets:
            errors.update(self.exact_values(i, terminal_values[i]))
        L = self.model.config.n_layer
        # every layer's weight credits from two batched GEMMs: query and key side by side, then value and writer
        layer_inputs = [d.inputs[f'transformer.h.{i}.attn.c_q'] for i in range(L)]
        self._attention_buffers = self.attention_layout.allocate(self._qkv)
        writers = (self._writer_credit, [d.inputs[f'transformer.h.{i}.attn.c_proj'] for i in range(L)],
                   self.attention_layout.writers)
        attention_wire.write_attention(self._qkv, self._value_credit, layer_inputs, divisor,
                                       self._attention_buffers, self.attention_layout.readers, writers)
        for site, error in errors.items():
            if not site.endswith(('attn.c_q', 'attn.c_k', 'attn.c_v')):
                gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gradients[f'value_embeds.{i}.weight'] = self._terminal_lexical[i]
    def bind_flat_layout(self, optimizer):
        """The update's flat gradient layout (ANVIL's shape groups first, in its batch order). Bound before the estimate
        graph's capture, it lets the estimate pack its gradients there."""
        params = list(self.model.named_parameters())
        anvil = optimizer.optimizers[0]
        anvil_ids = ([id(q) for q in anvil.rank_major_order()])
        rank_of = {pid: k for k, pid in enumerate(anvil_ids)}
        self._flat_order = sorted(params, key=lambda item: rank_of.get(id(item[1]), len(anvil_ids)))
        last = {id(q) for k in anvil.rank_groups[-1] for q in anvil.batches[k][0]}
        positions = [n for n, (_, q) in enumerate(self._flat_order) if id(q) in last]
        self._early_range = (positions[0], positions[-1] + 1)  # the last comm group's parameters
        lo, hi = self._early_range
        self._wire_order = self._flat_order[:lo] + self._flat_order[hi:]
        self.attention_layout = attention_wire.Layout(self._wire_order)
    def wide_step(self, x, y, order, optimizer, loss_sum):
        """A Cut update from this GPU's two rows: one replay of the two-row estimate graph (capture_pair_graph), which
        packs the gradients averaged over the rows, then the update. Row 2 draws from the partner's generators, set
        to this Cut's states one row's draws on, and this Cut's take theirs after the replay. order is the rows'
        position order for root 4's draw (pair_sampling.transition_orders). loss_sum (a 0-d device tensor)
        receives the sum of the replay's clean token losses."""
        partner = self.partner
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
            self._wide_order.copy_(order)
        # the replay, after the copies: the graph waits for the main stream's queue so far through self._step_done,
        # and the main stream for the graph
        self._step_done.record()
        with torch.cuda.stream(self._stage_stream):
            self._wide_graph.replay()
        main.wait_stream(self._stage_stream)
        self._replay_done.record()
        # the update's early part, right after the replay is launched and before the host waits for it (the
        # generators' hand-off reads only their host-side Philox seed and offset)
        self._early = self.early_update(optimizer)
        for mine, theirs in zip(self.graph_generators(), partner.graph_generators()):
            mine.set_state(theirs.get_state())
        self.update(optimizer)
        # on the replay's stream, so after this replay and before the next; the main stream does not wait for it
        with torch.cuda.stream(self._stage_stream):
            torch.sum(self._wide_loss, dim=(0, 1), out=loss_sum)
    def capture_pair_graph(self, x, y):
        """Capture the estimate of two rows (x, y: (2, T)) as one graph, row 2 drawing its frames and signs
        from the partner's generators. Measures each generator's draws per row."""
        partner = self.partner
        self._wide_x, self._wide_y = x.clone(), y.clone()
        self._wide_order = pair_sampling.transition_orders(x.unsqueeze(0), y.unsqueeze(0))[0]
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
                self.estimate(self._wide_x, self._wide_y)
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
    def early_update(self, optimizer):
        """ANVIL's last comm group ahead of the rest of the update: once the estimate graph has recorded
        self._early_ready (its MLP credits are packed), the group's reduce-scatter runs on a stream of its own,
        under the estimate's tail. Returns that stream."""
        anvil = optimizer.optimizers[0]
        stream = self._early_stream = getattr(self, '_early_stream', None) or torch.cuda.Stream()
        stream.wait_event(self._early_ready)
        if getattr(self, '_means_done', None) is not None:  # (the averages read the weights this writes)
            stream.wait_event(self._means_done)
        with torch.cuda.stream(stream):
            anvil.early_step(self._early_wire)
        return stream
    def update(self, optimizer):
        early, self._early = self._early, None
        if getattr(self, '_means_done', None) is not None:  # the averages read the weights this step writes
            torch.cuda.current_stream().wait_event(self._means_done)
            self._means_done = None
        optimizer.zero_grad(set_to_none=True)
        anvil = optimizer.optimizers[0]
        sizes = [sum(q.numel() for q in batch[0]) for batch in anvil.batches]
        wire = self._packed  # the bf16 gradient wire
        # ANVIL's shards first, each batch's update waits only for its own shard, and the rest's
        # all-reduce runs under ANVIL's update
        anvil.receive_rank_major(wire.dtype)
        starts = [sum(len(anvil.rank_stores[h]) for h in range(g)) for g in range(len(anvil.rank_groups))]
        works = {}
        for g in reversed(range(len(anvil.rank_groups))):  # the last unit (the merged polar) first
            if g == len(anvil.rank_groups) - 1:  # (under way: early_update, from the early slice's own wire)
                works[g] = anvil.early_work
                continue
            works[g] = torch.distributed.reduce_scatter_tensor(
                anvil.rank_receives[g], wire[starts[g]:starts[g] + len(anvil.rank_stores[g])],
                group=cut_group(g), async_op=True)
        anvil.shard_works = [works[anvil.rank_group_of[k]] for k in range(len(sizes))]
        anvil.shard_grads = list(anvil.shard_buffers)
        rest = torch.distributed.all_reduce(wire[sum(sizes) - len(anvil.rank_stores[-1]):],
                                            group=cut_group('rest'), async_op=True)

        def finish_rest():  # between ANVIL's and the other optimizers' steps
            rest.wait()
            optimizer.optimizers[1]._captured_update.current_wire = wire
        optimizer.step(between=finish_rest)
        torch.cuda.current_stream().wait_stream(early)
        for work in anvil.gather_works:
            work.wait()
        anvil.gather_works = []
    def fused_descending_key_values(self, chain, g_top, targets, x0, outs, after_extra):
        """The attention credits below the chain: the query moments on a side stream per window, the key credits with
        the raw value credits, the value gates' input credits and the descending value chain batched, and the terminal
        value credits after it, in the layout their consumers read. Returns its outputs and the path scalars."""
        d, s, model = self.capture, self.s, self.model
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        errors, terminal_values = {}, {}
        main, side = torch.cuda.current_stream(), self._side_stream
        # every layer's query and key credits by token, side by side (the value credits have their own buffer)
        rows = d.batch  # (several rows: (layer, row) pairs in place of layers, row-major within a layer)
        stored_width = 2 * self.width
        qkv = torch.empty((L, rows * t, stored_width), device=self.device, dtype=torch.bfloat16)
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
        key_scan = receiver_tiles.key_values_into
        gate_args = gate_head_dot.arguments(d)
        self._gate_dots = gate_args[1]
        packed = torch.empty((L * rows, t, self.width), device=self.device, dtype=torch.float32)
        key_options = dict(gate_args=gate_args, packed=packed, batch_rows=rows)
        for g, (window, layers, kp, k_raw, lse, center) in enumerate(work):
            main.wait_event(self._context_ready[window])
            q16, k16, v16, hub16, *_ = self._moment_inputs[window]
            cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
            key_out = qkv.view(L * rows, t, stored_width)[:, :, self.width:2 * self.width]
            frozen_rows = sum(i < FROZEN_QK_PREFIX for i in layers) * rows
            cut = frozen_rows * h
            if frozen_rows:
                frozen_attention.raw_values(q16[:cut], k16[:cut], lse[:cut], hub16[:cut], window, h,
                                            out=packed[:frozen_rows])
                key_scan(
                    q16[cut:], k16[cut:], k_raw[cut:], lse[cut:], v16[cut:], hub16[cut:], center[cut:], cos, sin,
                    window, s.key_sigma, h, key_out, maps[g][frozen_rows:], **key_options)
            else:
                key_scan(q16, k16, k_raw, lse, v16, hub16, center, cos, sin,
                         window, s.key_sigma, h, key_out, maps[g], **key_options)
        ve_layers = [i for i in range(L) if has_ve(i, L)]
        gates = dict(zip(ve_layers, coordinate_gate.gate_inputs_direct(
            [d.blocks[i] for i in ve_layers],
            [model.transformer.h[i].attn.ve_gate.weight for i in ve_layers], s.gate_path_sigma, dots=self._gate_dots)))
        c_vs = torch.bmm(packed.view(L, rows * t, self.width), self._reader_master_view).unbind(0)
        args = ([gates.get(i) for i in range(L)], [d.blocks[i] for i in range(L)], c_vs,
                model.resid_lambdas.float(), x0, outs)
        prepared, path = scan_operands.run(*args, [chain[i] for i in range(L)], g_top, DESCENDING_BT)
        after_extra(prepared)
        projected = torch.bmm(prepared[1], self._terminal_writers)
        all_sources = projected.view(L * rows, t, self.width)
        w = self.width
        ids = d.inputs['transformer.wte']
        gates = [d.outputs[f'transformer.h.{i}.attn.ve_gate'] for i in ve_layers]
        lexical = token_value.lexical_storage(ids, gates, model.config.vocab_size, w)
        self._terminal_lexical = dict(zip(ve_layers, lexical[2].unbind(0)))
        value_source = torch.empty((L * rows, t, w), device=self.device, dtype=torch.bfloat16)
        for g, (window, layers, kp, k_raw, lse, center) in enumerate(work):
            args = (*self._moment_inputs[window][:2], lse, all_sources, window, value_source, maps[g])
            token_value.transport(*args, lexical=lexical)
            for i in layers:
                terminal_values[i] = value_source[i * rows:(i + 1) * rows].view(rows, t, h, hd)
        self._value_credit = value_source.reshape(L, rows * t, w)
        self._qkv = qkv
        main.wait_stream(side)
        for i in range(L):
            keys[i].record_stream(main)
            errors[f'transformer.h.{i}.attn.c_k'] = keys[i].view(rows, t, w)
            self._query_moment_errors[f'transformer.h.{i}.attn.c_q'].record_stream(main)
        return errors, terminal_values, path
    def layer_map(self, layers):
        """The layers' indices as an int32 device tensor (made once, outside graph capture)."""
        if not hasattr(self, '_layer_maps'):
            self._layer_maps = {}
        key = tuple(layers)
        if key not in self._layer_maps:
            self._layer_maps[key] = torch.tensor(key, device=self.device, dtype=torch.int32)
        return self._layer_maps[key]
    def exact_values(self, i, value_credit):
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate']
            errors[prefix + 've_gate'] = gate_head_dot.parameter_response(z, self._gate_dots[i // 2])
        return errors
    def probe_moments(self, x, y):
        rows = self.capture.batch
        frames = [self.basis(r) for r in range(rows)]
        tail_draws = [torch.randint(240, (self.model.config.vocab_size // 16,),
                                    device=self.device, generator=self._sources[r].vocab_generator)
                      for r in range(rows)]
        cap = self.capture
        derivative, self._head_moments = clean_anchor.readout(
            x, y, self.model.lm_head.weight.float(), frames, tail_draws, EPS,
            cap.outputs['lm_head'].reshape(-1, self.model.config.vocab_size))
        return derivative
    def chain_credits(self, e, mids, outs, routes, y, g_top):
        model, s = self.model, self.s
        L = model.config.n_layer
        weight = model.lm_head.weight.float()
        names = [f'transformer.h.{i}.mlp.' for i in range(L)]
        w_fc = [self._bf16_weights[p + 'c_fc'] for p in names]
        w_out_t = chain_transposes([self._bf16_weights[p + 'c_proj'] for p in names])
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        normalized = self.capture.inputs['lm_head']
        full_clean = self.capture.outputs['lm_head'].reshape(-1, weight.shape[0])
        base, target_w = replay_anchor.indexed_tokens(xL, normalized, weight, y, full_clean)
        moments = self._head_moments
        n, m = e.shape[0], s.chain_tokens
        rows = self.capture.batch  # (each row draws from its own generators)
        t = n // rows
        by_site = {}
        for site in s.chain_depth_roots:
            frames = []
            for r in range(rows):
                frame = self.basis(r)
                if site == 6 and r == 1:
                    frame = frames[0]  # root 6's rows share one frame; the unused draw keeps their generators in step
                frames.append(frame)
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
                sign = (torch.cat(signs) * 2 - 1).float()
                signed_steps[site] = steps * sign
                if site == 6:  # (root 4's control is root 6's credit, carried down)
                    controls[site] = g_top * lam_r[site + 1:].prod()
        roots = [group[0] for group in groups]
        separate = [r for r in roots if r != 6]
        projected = root_projections_of([w_fc[r] for r in separate for _ in range(rows)],
                                        [frame for r in separate for frame in by_site[r][0]])
        root_projections = {r: projected[k * rows:(k + 1) * rows] for k, r in enumerate(separate)}
        packed = paired_walsh.packed_weights(w_fc[6], w_fc[7], w_out_t[6], w_out_t[7], by_site[6][0][0])
        weight_maps = [packed] * rows
        root_projections[6] = [maps[0] for maps in weight_maps]
        replay_layers = lambda group: tuple(b for b in range(group[0] + 1, L) if b in s.query_replay_long_layers)
        active_query_layers = tuple(sorted({b for group in groups for b in replay_layers(group)}))
        raw_owns = [self.own_query_inputs(active_query_layers, r) for r in range(rows)]
        current_query = raw_owns[0][2][5]
        shared_product, query_frames = query_bank.project_weights(w_out_t[4], by_site[4][0], current_query)
        # Root 6 has no attention replay; root 4 prepares its query rows once its tokens are selected.
        chunks = [slice(t0, t0 + m) for t0 in range(0, n, m)]
        main = torch.cuda.current_stream()
        lanes = [None] + [self.pair_lane(r) for r in range(1, rows)]
        for lane in lanes[1:]:  # each further row's replays on its own stream, joined below
            lane.wait_stream(main)
        def draw_positions(row, control):
            size = control.norm(dim=-1).double()
            offset = torch.rand(1, device=self.device, dtype=torch.float64,
                                generator=self._sources[row].move_generator)
            return pair_sampling.plan(size, offset, ROOT4_POSITIONS, s.chain_pps_floor, self._wide_order[row])

        for group in sorted(groups, key=lambda g: -g[0]):
            site, = group
            plans, paired_routes = [], None
            if site == 4:
                deeper = min(g[0] for g in groups if g[0] > site)
                for row, sl in enumerate(chunks):
                    with contextlib.nullcontext() if lanes[row] is None else torch.cuda.stream(lanes[row]):
                        control = lam_r[site + 1:deeper + 1].prod() * by_site[deeper][2][sl]
                        plans.append((*draw_positions(row, control), control))
                route_stream = self.route_pair_stream
                route_stream.wait_stream(main)
                route_stream.wait_stream(lanes[1])
                preload, radii, magnitudes, k = self._deferred_inputs
                with torch.cuda.stream(route_stream):
                    paired_routes = route_prepare.prepare(preload, radii, magnitudes, k, scale=s.chain_sigma,
                        row_ids=plans[0][0], partner_ids=plans[1][0], row_length=t)
                    for indices, _, _ in plans:
                        indices.record_stream(route_stream)
            for sl in chunks:
                row = sl.start // t
                with contextlib.nullcontext() if lanes[row] is None else torch.cuda.stream(lanes[row]):
                    H = by_site[site][0][row]
                    if site == 6:  # every token, no query rows
                        control = controls[site][sl]
                        take, mine, boost = sl, None, 1.0
                        steps = signed_steps[site][take]
                        picked_mids = [None if b < site else z[take] for b, z in enumerate(mids)]
                        picked_xL = xL[take]
                        picked_routes = [None if r[0] is None or b < site else r[0][take] for b, r in enumerate(routes)]
                        picked_clean = [None if r[1] is None or b < site else r[1][take] for b, r in enumerate(routes)]
                        picked_base, picked_target, picked_y = base[take], target_w[take], y[take]
                        picked_moments = tuple(z[take] for z in moments)
                    else:  # a fixed count of the row's tokens, drawn in proportion to the control's norm
                        inside, boost, control = plans[row]
                        tokens = sl.stop - sl.start
                        take = inside + sl.start
                        by_site[site][2][sl] = control
                        mine = query_bank.prepare_selected(raw_owns[row], inside, s.query_keys,
                                                           shared_product, query_frames[row])
                        (picked_mids, picked_xL, picked_routes, picked_clean, steps,
                         picked_base, picked_target, picked_y, picked_moments,
                         control) = selected_payload.selected_payload(
                            mids, xL, routes, signed_steps[site], base, target_w, y, moments, control,
                            inside, sl.start, tokens, site)
                        row_stream = torch.cuda.current_stream()
                        row_stream.wait_stream(self.route_pair_stream)
                        for tensor in paired_routes:
                            tensor.record_stream(row_stream)
                        count = inside.numel()
                        span = slice(row * count, (row + 1) * count)
                        ri, rc = [v[:, span] for v in paired_routes]
                        picked_routes[4:6] = ri.unbind(0)
                        picked_clean[4:6] = rc.unbind(0)
                    terminal_offsets = None
                    if site == 6:
                        terminal_offsets = torch.randint(256, (steps.shape[0], 8), device=self.device,
                            generator=self._sources[row].terminal_generator, dtype=torch.int32)
                    loss_change = chain_chunk(
                        site, (steps, H), picked_mids, picked_xL, picked_routes, picked_clean, lam_r, w_fc, w_out_t,
                        picked_base, picked_target, picked_y, mine,
                        moments=picked_moments,
                        root_projection=root_projections[site][row], terminal_offsets=terminal_offsets,
                        weight_maps=weight_maps[row] if site == 6 else None,
                    )
                    by_site[site][2][take] = group_credits(loss_change, [steps], [control], [H], boost)[0]
        for lane in lanes[1:]:
            main.wait_stream(lane)
        measured_sites = {site for group in groups for site in group}
        credits = {site: credit for site, (_, _, credit) in by_site.items() if site in measured_sites}
        roots = [credits[4]] * 5 + [credits[6]] * 2 + [g_top]
        out = dict(enumerate(roots))
        embedding, operands = root_carry.embed(roots, lam_r, lam_0, e, s.chain_sigma)
        out['operands'] = operands
        out['emb'] = embedding
        return out
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
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)
    @torch.no_grad()
    def cast_weights(self):
        """Every Linear's bf16 weight in persistent buffers: the MLPs' are the publication's replicas, the others
        one multi-tensor copy (the capture forward and chain_credits read them instead of casting their own)."""
        if not hasattr(self, '_bf16_weights'):
            linears = {n: m.weight for n, m in self.capture.modules.items() if isinstance(m, nn.Linear)}
            self._bf16_weights = {}
            depth, width = self.model.config.n_layer, self.model.config.n_embd
            self._terminal_writers = torch.empty((depth, width, width), device=self.device, dtype=torch.bfloat16)
            for i in range(depth):
                self._bf16_weights[f'transformer.h.{i}.attn.c_proj'] = self._terminal_writers[i]
            self._reader_master_view = master_view([b.attn.c_v.weight for b in self.model.transformer.h])
            for i in range(self.model.config.n_layer):
                p = f'transformer.h.{i}.attn.'
                parts = [linears[p + n] for n in ('c_q', 'c_k', 'c_v')]
                cat = torch.empty((sum(w.shape[0] for w in parts), parts[0].shape[1]), device=parts[0].device,
                                  dtype=torch.bfloat16)
                self._bf16_weights[p + 'qkv'] = cat
                for n, rows in zip(('c_q', 'c_k', 'c_v'), cat.split([w.shape[0] for w in parts])):
                    self._bf16_weights[p + n] = rows
            replicas = self.mlp_publication.weights
            self._bf16_weights.update(replicas)
            for n, w in linears.items():
                if n not in self._bf16_weights:
                    self._bf16_weights[n] = torch.empty_like(w, dtype=torch.bfloat16)
            copied = [n for n in linears if n not in replicas]
            self._bf16_targets = [self._bf16_weights[n] for n in copied]
            self._fp32_weights = [linears[n] for n in copied]
        torch._foreach_copy_(self._bf16_targets, self._fp32_weights)
        self.capture.weights = self._bf16_weights
    def estimate(self, x, y):
        d, model = self.capture, self.model
        self.cast_weights()
        d.capture(x, y)
        L = model.config.n_layer
        yf = y.flatten()
        gradients = {}
        d.clean_loss, head = cap_lookup.run(d.outputs['lm_head'], d.y, self.cap_tables)
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        first = min(group[0] for group in self.s.joint_root_groups)
        mids = [d.outputs[f'transformer.h.{i}.mlp.clean_mid'].float().flatten(0, 1) for i in range(L)]
        hpres = [d.outputs[f'transformer.h.{i}.mlp.c_fc'].flatten(0, 1) for i in range(L)]
        outs = [d.outputs[f'transformer.h.{i}.mlp.clean_out'].float().flatten(0, 1) for i in range(L)]
        radius, out_norm = self.mlp_publication.route_norms(first)
        count = self.s.route_k_late  # every scored block keeps only its strongest route_k_late
        prepare_from = 6
        self._deferred_inputs = (hpres[first:prepare_from], radius[:prepare_from - first],
                                 out_norm[:prepare_from - first], count)
        offset = prepare_from - first
        tops, clean = route_prepare.prepare(hpres[prepare_from:], radius[offset:], out_norm[offset:],
                                            count, scale=self.s.chain_sigma)
        routes = [(None, None)] * prepare_from + list(zip(tops.unbind(0), clean.unbind(0)))
        g_top = self.probe_moments(outs[-1], yf)
        chain = self.chain_credits(e, mids, outs, routes, yf, g_top)
        return self.chain_gradients(gradients, chain, g_top, x0, outs, y)
    def chain_gradients(self, gradients, chain, g_top, x0, outs, y):
        """The estimate's weight gradients from the chain's block credits (into gradients; packed)."""
        d, model = self.capture, self.model
        L = model.config.n_layer
        b, t = y.shape
        targets = {i: chain['operands'][i].view(b, t, self.width) for i in range(L)}
        divisor = (y != -1).sum() * self.world  # every gradient on the wire is divided by it

        def mlp_first(prepared):  # inside the descending pass, ahead of the terminal value credits
            hidden_targets, attn_proj, mlp_proj = prepared
            fc = cut_mlp.hidden_credits_rms(
                [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], hidden_targets,
                [self._bf16_weights[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)],
                [d.outputs[f'transformer.h.{i}.mlp.fc_rms'] for i in range(L)], self.s.fc_sigma)
            self._early_wire = attention_wire.full_mlp_products(
                mlp_proj, [d.inputs[f'transformer.h.{i}.mlp.c_proj'] for i in range(L)],
                fc, [d.inputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], divisor)
            self._writer_credit = attn_proj
            if hasattr(self, '_early_ready'):
                self._early_ready.record()
        key_errors, terminal_values, scalars = self.fused_descending_key_values(
            chain, g_top, targets, x0, outs, mlp_first)
        self.internals(gradients, targets, key_errors, terminal_values, divisor)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        self._packed = self._attention_buffers  # the bf16 wire is packed here, inside the estimate graph
        self.attention_layout.pack_rest(gradients, divisor, self._packed)
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
# together (their own ANVIL batch, interleaving q and k per block)
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


def constrain(old, proposed, heads: int):
    """old, proposed: (batch, rows, columns) weights before and after an update. Removes from each head's part of the
    update (rows / heads consecutive output rows of one matrix) its component along that head's current weights, so
    the update is tangent to the sphere of that head's weights. heads == 1: proposed, unchanged."""
    if heads == 1:
        return proposed
    shape = old.shape
    w = old.reshape(shape[0], heads, -1)
    p = proposed.reshape_as(w)
    power = w.square().sum(-1, keepdim=True)
    delta = p - w
    radial = (delta * w).sum(-1, keepdim=True) / power.clamp_min(1e-30)
    result = w + delta - radial * w
    return result.reshape(shape)


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
    return constrain(w, proposed, heads)


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


def warm_cut_update(anvil, wire):
    """First uses of the Cut update before the timer, on zeroed scratch of its sizes: publish()'s compiled norms under
    deterministic algorithms (as Optimizers.step sets them), and each Cut group's collectives at the sizes and dtypes
    of Cut.update, early_step and rank_gather."""
    publication = anvil.mlp_publication
    norms = torch.zeros_like(publication.send)[publication.chunk:].view(torch.float32).view_as(publication.send_norms)
    flag = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    mlp_replicas._pack_norms(publication.reader, publication.writer, norms)
    torch.use_deterministic_algorithms(flag[0], warn_only=flag[1])
    for g in reversed(range(len(anvil.rank_groups))):
        shard, store, send = anvil.rank_receives[g], anvil.rank_stores[g], anvil.rank_sends[g]  # shapes only
        dist.reduce_scatter_tensor(torch.zeros_like(shard), torch.zeros_like(store, dtype=shard.dtype),
                                   group=cut_group(g))
        dist.all_gather_into_tensor(torch.zeros_like(store), torch.zeros_like(send), group=cut_group(g))
    rest = torch.zeros_like(wire[sum(len(store) for store in anvil.rank_stores[:-1]):])
    dist.all_reduce(rest, group=cut_group('rest'))
    torch.cuda.synchronize()


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
            lo, hi, cool = .85, .90, self.total - self.cooldown
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
            ratio = max(1, shape[0] / shape[1]) ** .5
            world, rank = dist.get_world_size(), dist.get_rank()
            per = len(params) // world
            owned = params[rank * per:(rank + 1) * per]
            step_ratios = [ratio * (0.5 if id(p) in self.half_step_writers else 1.0) for p in owned]
            if len(set(step_ratios)) == 1:
                ratio = step_ratios[0]
            else:
                ratio = torch.tensor(step_ratios, device=device, dtype=torch.float32).view(-1, 1, 1)
            self.batches.append((params, state, lanes, ratio))
        if dist.is_initialized():
            self.store_rank_major()

    def store_rank_major(self):
        """Each comm group's weights (each shard unit's batches) moved into one buffer, rank r's chunk holding its shard
        rows of the group's batches (for the record's model: block r's matrices), so one all-gather fills a group."""
        world = dist.get_world_size()
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
        (the polar and state update; the collectives stay eager), the scalars read from a static row. The state
        changes of the eager pass and of the capture are undone."""
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
        """Per unit, wait its reduce-scatter and run its graph; the last unit (the merged polar) on this stream, the
        others on a side stream at the same time. Then the all-gathers (one per unit's group, the last unit's first,
        each issued from the stream that ran it)."""
        units = self.shard_units()
        graphs = self._unit_graphs
        waited = set()
        def run(n):
            for k in units[n]:
                work = self.shard_works[k] if getattr(self, 'shard_works', None) else None
                if work is not None and id(work) not in waited:
                    work.wait()
                    waited.add(id(work))
            graphs[units[n]][0].replay()
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
        g = len(self.rank_groups) - 1
        self.early_work = torch.distributed.reduce_scatter_tensor(self.rank_receives[g], wire, group=cut_group(g),
                                                                  async_op=True)

    def rank_gather(self, g):
        publication = self.mlp_publication
        if g == publication.group_index:
            work = publication.publish()
            self.gather_works = getattr(self, 'gather_works', []) + [work]
            if not publication.full:
                return
        work = dist.all_gather_into_tensor(self.rank_stores[g], self.rank_sends[g], group=cut_group(g),
                                           async_op=True)
        if work is not None:
            self.gather_works = getattr(self, 'gather_works', []) + [work]

    @torch.no_grad()
    def step(self):
        """This rank's shard rows of every batch, from the gradients' reduce-scatter (Cut.update)."""
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
    # Only the random-token captures read these init values: the identity handoff overwrites every weight. The seed
    # and the init draws set the default generators' states.
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


class ExactAdamW(torch.optim.Optimizer):
    """AdamW(fused=True) with weight_decay 0, numerically the same (bf16 tensors bitwise; fp32 ones up to the fp64
    moments' last-bit rounding). Its step replays cut_update_graph's captured graph, which keeps the moments
    (exp_avg, exp_avg_sq) in this optimizer's state."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        updater = self._captured_update
        return updater.graphs[updater.current_wire.data_ptr()].replay()


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
      betas=recipe.betas, eps=1e-8, weight_decay=0.0,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
      polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    anvil.half_step_writers = {id(writer.weight) for block in model.transformer.h[:2]
                               for writer in (block.attn.c_proj, block.mlp.c_proj)}
    anvil.frozen_ids = {id(getattr(block.attn, f'c_{kind}').weight)
                        for block in model.transformer.h[:FROZEN_QK_PREFIX] for kind in HEAD_POLAR}
    anvil.head_ids = {id(getattr(block.attn, f'c_{kind}').weight): model.config.n_head
                      for block in model.transformer.h for kind in HEAD_POLAR}
    # ANVIL weight decay is scaled by late_cut_decay_multiplier (zero) after the first 275/1043 of the Cut updates.
    late_from = round(275 / 1043 * steps)
    decay_factors = tuple(recipe.late_cut_decay_multiplier if t >= late_from else 1.0 for t in range(steps))
    anvil.prepare_scalar_schedule(recipe.warmdown, recipe.final, decay_factors)
    mlp_group = cut_group(len(anvil.rank_groups) - 1)
    anvil.mlp_publication = mlp_replicas.MLPPublication(anvil, model, mlp_group)
    anvil.mlp_publication.initialize()
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


MERGE_FROM, MERGE_EVERY = 0.3, 5
MERGE_DECAY = 0.02


def average_plan(updates, knot_count):
    """The Cut updates after which a weight snapshot joins the running average, each with its lerp weight: an EMA
    of decay MERGE_DECAY over the events (every MERGE_EVERY steps from MERGE_FROM of the Cut on, and the last update),
    its weights moved onto knot_count of them by linear interpolation in the update index."""
    last = updates[-1][-1]
    events = np.asarray([
        step for step, group in enumerate(updates, 1)
        if (group[0] - 1) / last >= MERGE_FROM
        and (step == len(updates) or any(k % MERGE_EVERY == 0 for k in group))
    ], dtype=np.int64)

    decay = MERGE_DECAY
    tail = (1.0 - decay) ** np.arange(len(events) - 1, -1, -1, dtype=np.float64)
    native_mass = decay * tail
    native_mass[0] = tail[0]
    # For a random-walk trajectory, local interpolation variance scales
    # with squared sampling weight times the cube of the interval length.
    dense_tail = min(32, len(events) - 2, knot_count - 2)
    coarse_stop = len(events) - dense_tail
    coarse_events = events[:coarse_stop]
    density = tail[:coarse_stop] ** (2.0 / 3.0)
    area = (density[1:] + density[:-1]) * np.diff(coarse_events) * 0.5
    integral = np.r_[0.0, np.cumsum(area)]
    targets = np.linspace(0.0, integral[-1], knot_count - dense_tail)
    boundary = (integral[:-1] + integral[1:]) * 0.5
    coarse_knots = coarse_events[np.unique(np.searchsorted(boundary, targets))]
    knots = np.concatenate((coarse_knots, events[coarse_stop:]))

    hi = np.clip(np.searchsorted(knots, events), 1, len(knots) - 1)
    lo = hi - 1
    right = (events - knots[lo]) / (knots[hi] - knots[lo])
    mass = np.bincount(lo, weights=native_mass * (1.0 - right), minlength=len(knots))
    mass += np.bincount(hi, weights=native_mass * right, minlength=len(knots))
    return {int(step): float(alpha) for step, alpha in zip(knots, mass / np.cumsum(mass))}


@torch.no_grad()
def blend_average(model, averages):
    last_block = model.config.n_layer - 1
    for (name, parameter), average in zip(model.named_parameters(), averages, strict=True):
        # Block-owned tensors have their block index in the parameter path.
        block = next((int(part) for part in name.split('.') if part.isdecimal()), last_block / 2)
        weight = 1.0 - 0.5 * block / last_block
        parameter.copy_(torch.lerp(parameter.float(), average, weight))


F32, BF16 = torch.float32, torch.bfloat16
BATCH, SEQ = 8, 2048
ID_SEQ_LEN = 2048  # the context of Identity updates 196-315 (updates 1-195: two 1,024-token contexts per row)
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8
LR_WTE, LR_VE, LR_LAMBDAS = 1.8, 1.8, 0.006
ANVIL_LR = 0.04
ANVIL_DECAY = 0.09  # identity-phase ANVIL decay (lr x wd, wd 2.25)
GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
SIGMA, FC_SIGMA, KEY_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1
ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240)  # #44's ANVIL momentum warmup: the first 240 of 1194 updates (see anvil_betas)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_RHO, START_CONTENT = 1, 0.9, 0.25
START_LAGS = (1, 1, 1, 1, 2, 2, 4, 8)
CARRIER_SIGN = -1.0


def id_has_ve(layer):
    return layer % 2 == (L - 1) % 2


VE_BLOCKS = tuple(i for i in range(L) if id_has_ve(i))


def id_norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)


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


SHAPING_UPDATES = 195  # three update-shaping mechanisms act on identity updates 0..194 (two in value_row_owners)
SHAPING_ON = True  # (set while each graph is traced; False: no shaping, no query update, one 2,048-token context)
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


@torch.no_grad()
def initialize_head_sector_energy(head, embedding):
    norm2 = head.float().square().sum(dim=0, keepdim=True)
    e = embedding.float().T
    fraction = e[:GATE_CHANNELS].square().sum(dim=0, keepdim=True) / e.square().sum(dim=0, keepdim=True)
    inside = head[:GATE_CHANNELS].float()
    outside = head[GATE_CHANNELS:].float()
    head[:GATE_CHANNELS].copy_(inside * torch.sqrt(norm2 * fraction / inside.square().sum(dim=0, keepdim=True)))
    head[GATE_CHANNELS:].copy_(outside * torch.sqrt(norm2 * (1 - fraction) / outside.square().sum(dim=0, keepdim=True)))


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
        short_window = SEQ // 2
        native = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, 1024, 1024, device=device)
        causal = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, ID_SEQ_LEN, ID_SEQ_LEN, device=device)
        short = create_block_mask(lambda b, h, qi, ki: (qi >= ki) & (qi - ki <= short_window),
                                  None, None, ID_SEQ_LEN, ID_SEQ_LEN, device=device)
        self.masks = {1024: (native,) * L, ID_SEQ_LEN: (short, short, short, causal) * 2}
        self.rope = {t: rotary(t, device) for t in (1024, ID_SEQ_LEN)}
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
def lag_carrier(model, mean):
    reader = model.w(0, 'mlp.c_fc').double()
    writer = model.w(0, 'mlp.c_proj').double() * model.lam_r[1].double()
    variance = reader.new_full((D,), D * (1 - WTE_GATE_ENERGY) / (D - GATE_CHANNELS))
    variance[:GATE_CHANNELS] = D * WTE_GATE_ENERGY / GATE_CHANNELS
    projected = variance[:, None] * reader
    gram = reader.T @ projected
    powers = gram.diag()
    scale = torch.sqrt(powers)
    correlation = (gram / (scale[:, None] * scale[None, :])).clamp(-1.0, 1.0)
    angular = (3 * correlation * (1 - correlation.square()).clamp_min(0).sqrt()
               + (math.pi - torch.acos(correlation)) * (1 + 2 * correlation.square()))
    hidden_cov = (angular / (2 * math.pi) - 0.25) * (powers[:, None] * powers[None, :])
    skip = model.lam_r[1] * (model.lam_r[0] + model.lam_0[0]) + model.lam_0[1]
    cross = (projected * (scale * math.sqrt(2 / math.pi))) @ writer
    cov = torch.diag(variance * skip.square()) + writer.T @ (hidden_cov @ writer) + skip * (cross + cross.T)
    cov = (cov + cov.T) * 0.5
    regularized = cov.clone()
    regularized.diagonal().add_(0.15 * cov.diagonal().mean())
    mean = mean.double()
    original = mean / (mean @ mean)
    direction = torch.linalg.solve(regularized, mean)
    shift = direction / (direction @ mean) - original
    shift = shift * (original.norm() / shift.norm().clamp_min(1e-20)).clamp(max=1.0)
    selected = original + CARRIER_SIGN * shift
    return selected.float()


@torch.no_grad()
def structured_start(model):
    generator = torch.Generator(device='cpu').manual_seed(SEED + 31)
    mean = shared_mean(model, START_BLOCK)
    carrier = lag_carrier(model, mean)
    for h in range(H):
        cols = slice(h * HD, (h + 1) * HD)
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAGS[h], START_RHO, generator)):
            w = model.w(START_BLOCK, kind)
            w[:, cols] = START_CONTENT * w[:, cols] + torch.outer(carrier, v.to(mean.device))


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


def forward_impl(model, idx, cos, sin):
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
        o, aux = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.masks[t][i],
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
    return cache


forward = torch.compile(forward_impl, dynamic=False, options={'combo_kernels': True})


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


def identity_frozen_queries():
    """The blocks whose queries get no identity update: none while the shaping mechanisms act, then the first
    IDENTITY_FROZEN_QUERIES."""
    return 0 if SHAPING_ON else IDENTITY_FROZEN_QUERIES


def id_head_scores(got, rope32, back, b, prepared, lexical):
    q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = prepared
    frozen = identity_frozen_queries()
    if frozen:
        back[:, :frozen, 0].zero_()
    if frozen < L:
        first = frozen * b
        moment_kernels._moment_query_scores(q1[first:], k1[first:], values[first:], hub[first:],
            got[:, frozen:], sq_q[first:], tuple(p[first:] for p in k1f), *rope32, lse[first:], hub_out[first:],
            steps_q[first:], back[:, frozen:, 0])
    moment_kernels.moment_key_scores(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k,
                                     back[:, :, 1], back[:, :, 2], short_window=SEQ // 2, lexical=lexical)


def compact_head_inputs(cache, targets_o):
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])
    return ([mine(cache[f'h{i}.q']) for i in range(L)],
            [mine(cache[f'h{i}.k']) for i in range(L)],
            [mine(cache[f'h{i}.values']) for i in range(L)],
            [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)],
            [targets_o[i] for i in range(L)], [cache[f'h{i}.lse'] for i in range(L)])


def compact_head_scores(inputs, rope32, back, b, t, lexical):
    prepared, raw = local_head_prepare.prepare(*inputs, *rope32, b, t, H, KEY_SIGMA, NORM_EPS)
    id_head_scores(raw, rope32, back, b, prepared, lexical)


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
def value_gate_credit(z, ve, vcredit, n1):
    z = z.float()
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    read = (vcredit.float() * ve.float().view(-1, H, HD)).sum(-1)
    return n1[:, :GATE_CHANNELS].float().T @ (slope * read)


# The Identity MLP products go to their owners by an NCCL all-to-all; _sum_local adds the eight senders' chunks.
@triton.jit
def _sum_local(Inbox, Output, SHARD: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for sender in tl.static_range(8):
        total += tl.load(Inbox + sender * SHARD + i).to(tl.float32)
    tl.store(Output + i, total.to(tl.bfloat16))


class WideExchange:
    def __init__(self, model):
        self.group = model.gather_group
        self.products = model.grad['wide']
        self.inbox = torch.empty_like(self.products).view(8, -1)
        self.work = None

    def write(self, left, right, first):
        id_kernels.at_b_layers(left, right, self.products[first:first + 8])
        if first == 8:
            # Each destination owns two full matrices. Receive chunks in sender-rank order.
            self.work = dist.all_to_all_single(self.inbox, self.products.view_as(self.inbox),
                                               group=self.group, async_op=True)

    def reduce(self, output):
        self.work.wait()
        self.work = None
        shard = self.inbox.shape[1]
        _sum_local[(shard // 1024,)](self.inbox, output, shard, 1024, num_warps=4)


# identity_grad's collectives: the wide gradients' all-to-all is issued as soon as they are complete (the
# collectives' stream idles then), and their ANVIL update and all-gather run on a side stream from there
# (the step's last read of wide_bf, the hidden credits, precedes the fork). The ANVIL all-gathers use a second
# communicator, so they do not queue behind the flat all-reduce.
def early_wide_update(model, n):
    """The wide matrices' ANVIL update and all-gather on a side stream once their all-to-all
    lands. Returns (gather work, its input, the side stream): train_step waits for the gather and joins the stream."""
    if not hasattr(model, 'wide_stream'):
        model.wide_stream = torch.cuda.Stream()
    side = model.wide_stream
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        model.wide_exchange.reduce(model.owned['wide'])
        sl, st = model.shard['wide'], model.anvil['wide']
        new = anvil_math(model.wide[sl], model.owned['wide'], n, st['fast'][sl], st['slow'][sl], st['energy'][sl],
                         model.step_t, model.rates['wide'][sl])
        gather = dist.all_gather_into_tensor(model.wide_bf, new, group=model.gather_group, async_op=True)
    return gather, new, side


def identity_grad(model, idx, targets, rows_total):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens, all eight heads' queries, keys and values of its own row. The square matrices' gradients are
    reduce-scattered, the MLP matrices' sent to their owners by an all-to-all, the value tables' rows to theirs, and
    the rest all-reduced. Returns the clean loss of its tokens, the collectives in flight and the token count."""
    t = idx.shape[1]
    n, m = rows_total * t, idx.numel()
    rope32 = model.rope[t]
    cache = forward(model, idx, *rope32)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1), idx.reshape(-1)
    gsq = model.grad['square']
    credit_top = identity_vocab.credit(sys.modules[__name__], model, stream(f'h{L - 1}.stream'), y,
                                       IDENTITY_VOCAB_STRIDE)
    cred, credb = block_credits(credit_top, model.lam_r)
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))
    loss, head_data = identity_head_product.prepare(cache['lm_head'], y, model.head_tables)
    prevs = torch.stack([x.reshape(-1, x.shape[-1])
                         for x in [cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)]]).float()
    n1s, os_, n2s, acts = ([mine(cache[f'h{i}.{key}']) for i in range(L)]
                           for key in ('in.attn', 'in.attn.c_proj', 'in.mlp.c_fc', 'in.mlp.c_proj'))
    hidden = lambda: id_kernels.hidden_credits_fused([cache[f'h{i}.mlp.c_fc'] for i in range(L)],
                                                     cache.get('fc_steps'), credb, model.wide_bf[L:])
    exchange = model.wide_exchange
    exchange.write(n2s, list(hidden()), 0)
    exchange.write(list(credb), acts, 8)
    identity_head_product.staged_product(cache['in.lm_head'], head_data, model.grad['lm_head'])
    works = {'wide_update': early_wide_update(model, n)}
    # each rank scores the heads of its own rows, where they were computed (no exchange between ranks)
    got2 = torch.empty(H, L, 3, m, HD, device=idx.device, dtype=BF16)
    lexical, boundaries = identity_lexical.prepare(model, [mine(cache[f'h{i}.ve_gate']) for i in VE_BLOCKS])
    compact_head_scores(compact_head_inputs(cache, targets_o), rope32, got2, rows_total, t, lexical)
    works['transfer_frames'] = model.square_bf[3 * L + 1:4 * L:2].float() if SHAPING_ON else None
    main = torch.cuda.current_stream()
    side = model.owner_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        model.lexical_peers.exchange(model.lexical_sums, boundaries, model.step_t, model.owner_in)
        works['published'] = value_row_owners.advance(model, n, works['transfer_frames'], sys.modules[__name__])
    boundaries.record_stream(side)
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    id_kernels.at_b_layers(os_, list(credb), gsq[3 * L:])
    id_kernels.square_credit_grads(n1s, got2, gsq[:3 * L], frozen_queries=identity_frozen_queries())
    vcredits = [got2[:, i, 2].permute(1, 0, 2) for i in VE_BLOCKS]
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        args = (mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(value_gate_credit(*args))
    works['flat'] = value_row_owners.reduce(model)
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
    if not hasattr(model, 'side_stream'):
        model.side_stream = torch.cuda.Stream()
    side = model.side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        works['square'].wait()
        sl, st = model.shard['square'], model.anvil['square']
        frozen = max(0, min(sl.stop, identity_frozen_queries()) - sl.start)
        if frozen == 0:
            new = anvil_math(model.square[sl], model.owned['square'], n, st['fast'][sl], st['slow'][sl], st['energy'][sl],
                             model.step_t, model.rates['square'][sl])
        else:
            new = model.square[sl].to(BF16)
        gathers.append((dist.all_gather_into_tensor(model.square_bf, new, group=model.gather_group, async_op=True), new))
    main.wait_stream(side)
    gather, new, wide_side = works['wide_update']
    gathers.append((gather, new))
    main.wait_stream(wide_side)
    return gathers


@torch.compile(dynamic=False)
def adam_math(ps, gs, n: int, ms, vs, lrs: tuple, step_t):
    t = step_t + 1
    c2 = 1 - BETA2 ** t
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
        p.copy_((p.float() - lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)


def train_step(model):
    world, rank = model.world, model.rank
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]
    if SHAPING_ON:
        inputs = batch[:, :-1].reshape(-1, 1024)
        targets = batch[:, 1:].reshape(-1, 1024)
    else:
        inputs, targets = batch[:, :ID_SEQ_LEN], batch[:, 1:ID_SEQ_LEN + 1]
    rows_total = inputs.shape[0]
    per = rows_total // world
    idx, tgt = inputs[rank * per:(rank + 1) * per], targets[rank * per:(rank + 1) * per]
    loss, works, n = identity_grad(model, idx, tgt, rows_total)
    gathers = id_anvil_update(model, works, n)
    value_row_owners.update(model, works, n, sys.modules[__name__])
    constrain_wte_gate_energy_compiled(model.wte, model.step_t)
    for work, _ in gathers:
        work.wait()
    model.refresh_bf16()
    model.loss.copy_(loss.view(1))
    model.step_t.add_(1)


class IdentityPhase:
    """The identity-path phase: compiled and captured as two CUDA graphs (updates 1-195, 196-315) on random tokens,
    then reset to its fresh state. bind_rows() puts its batches on the GPU before the timer; run() replays them."""
    def __init__(self, device, rank, world):
        self.order = np.random.default_rng((SEED + 1, 0)).permutation(TRAIN_ROWS)
        model = self.model = Model(device, rank, world)
        model.head_tables = cap_lookup.allocate(device)
        cap_lookup.fill(model.head_tables)
        snapshot = [x.clone() for x in model.tensors()]
        global SHAPING_ON
        group = dist.new_group(pg_options=nccl_options())
        value_row_owners.bind(model, group)
        model.wide_exchange = WideExchange(model)
        identity_lexical.allocate(model)
        self.spacing_tables = identity_spacing.tables(device)
        identity_spacing.arrange(model.rows.clone(), *self.spacing_tables)
        model.lexical_peers = lexical_peers.Exchange(model)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                train_step(model)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            train_step(model)
        self.plain_graph = None
        SHAPING_ON = False
        try:
            with torch.cuda.stream(side):
                for _ in range(2):
                    train_step(model)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
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
        """Put the identity batches on the GPU: batch i is rows order[BATCH * i:BATCH * (i + 1)], sorted by row index
        (the captured step reads model.rows[step_t]; run() reorders batches 195 on within windows of six)."""
        batches = rows[np.sort(self.order[:IDENTITY_UPDATES * BATCH].reshape(IDENTITY_UPDATES, BATCH), axis=1)]
        with torch.no_grad():
            self.model.rows.copy_(torch.from_numpy(batches))
    def run(self):
        identity_spacing.arrange(self.model.rows, *self.spacing_tables)
        identity_lexical.bind_training_plan(self.model)
        self.model.lexical_peers.bind(self.model.lexical_plan[1])
        for i in range(IDENTITY_UPDATES):
            if i >= SHAPING_UPDATES:
                self.plain_graph.replay()
            else:
                self.graph.replay()
        self.model.gather_fp32()
    def close(self):
        """Drops the graphs and the model (called after the timer, before validation)."""
        self.graph = self.plain_graph = None
        self.model = None


CURRICULUM_PREFIX = 0.5


def batch_history_order(token_ids, order, settings):
    batches = order.reshape(-1, settings.wide_rows)
    n_vocab = settings.model.vocab_size
    hist = np.empty((len(batches), n_vocab), dtype=np.float64)
    for i, members in enumerate(batches):
        hist[i] = np.bincount(token_ids[members].reshape(-1), minlength=n_vocab)
    inverse_scale = 1.0 / np.sqrt(np.maximum(hist.mean(axis=0), 1.0))
    basic_prefix = round(CURRICULUM_PREFIX * len(order) / settings.cut_batch)
    prefix_rows = basic_prefix * settings.cut_batch
    mixed_start, spill = divmod(prefix_rows, settings.wide_rows)
    spans = ((0, mixed_start), (mixed_start + int(spill != 0), len(batches)))
    permutation = np.arange(len(batches))
    for lo, hi in spans:
        for begin in range(lo, hi, 32):
            end = min(begin + 32, hi)
            counts = hist[begin:end]
            directions = (counts - counts.mean(axis=0)) * inverse_scale
            lengths = np.sqrt(np.einsum('ij,ij->i', directions, directions))
            directions /= np.maximum(lengths[:, None], 1e-12)
            gram = np.einsum('iv,jv->ij', directions, directions)
            ordered_history = gram[0].copy()
            available = np.ones(len(counts), dtype=bool)
            available[0] = False
            for pos in range(1, len(counts)):
                preference = settings.batch_history_sign * ordered_history
                chosen = int(np.argmax(np.where(available, preference, -np.inf)))
                available[chosen] = False
                permutation[begin + pos] = begin + chosen
                ordered_history = 0.75 * ordered_history + gram[chosen]
    if not np.array_equal(np.sort(permutation), np.arange(len(batches))):
        raise RuntimeError('Cut optimizer batch permutation is incomplete')
    return batches[permutation].reshape(-1)


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


def metrics_records(updates, settings, identity_loss, identity_seconds, cut_losses, cut_peaks, cut_seconds):
    """metrics.jsonl in #59's format: the Identity phase's record, then one record per Cut update (updates: cut_updates'
    tuples). Each loss is the mean over the update's tokens on all ranks; cut_peaks are bytes; every Cut record takes
    cut_seconds."""
    rows, length = IDENTITY_UPDATES * BATCH, settings.model.sequence_len
    records = [dict(step=IDENTITY_UPDATES, phase='identity', phase_step=IDENTITY_UPDATES, seqs=rows,
                    tokens=rows * length, train_loss=identity_loss, train_seconds=round(identity_seconds, 2))]
    for step, (group, loss, peak) in enumerate(zip(updates, cut_losses, cut_peaks, strict=True), 1):
        seqs = rows + group[-1] * settings.cut_batch
        records.append(dict(step=step + IDENTITY_UPDATES, phase='cut', phase_step=step, seqs=seqs, tokens=seqs * length,
                            train_loss=loss, peak_cuda_gb=round(peak / 1e9, 3), train_seconds=round(cut_seconds, 2)))
    return records


def train_stages(args, settings, device, rank, world):
    if world != 8:
        raise ValueError('This speedrun recipe requires eight GPUs')
    identity = IdentityPhase(device, rank, world)
    # The two phases draw disjoint rows from the seeded permutation.
    identity_indices = torch.from_numpy(identity.order[:IDENTITY_UPDATES * BATCH])
    keep = torch.ones(TRAIN_ROWS, dtype=torch.bool)
    keep[identity_indices] = False
    indices = torch.arange(TRAIN_ROWS)[keep]
    indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(SEED + 1))]
    # The last 64 rows of the seeded permutation, which #59 held out, are trained as the four Cut updates after
    # prefix_rows, where the weight average starts.
    pool_indices, extra_indices = indices[:-64], indices[-64:]
    averages = None
    horizon = len(indices) // settings.cut_batch
    updates = cut_updates(horizon, settings)  # one ANVIL schedule row per update
    train_count = horizon * settings.cut_batch
    prefix_rows = int(MERGE_FROM * (len(pool_indices) // settings.wide_rows)) * settings.wide_rows
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(
            json.dumps(
                dict(
                    script=__file__, world=world, seed=SEED, recipe=asdict(settings),
                    steps=len(updates) + IDENTITY_UPDATES,
                    epoch_updates=len(updates) + IDENTITY_UPDATES,
                    train_rows=train_count + IDENTITY_UPDATES * BATCH,
                    upstream='qlabs-eng/noise@de9732b9882fdc21ff9be566f6f577f37be500a1',
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

    # The Cut captures its estimate graph on two rows of random tokens.
    noise = torch.randint(0, settings.model.vocab_size, (settings.cut_batch, settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(SEED))
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(device, SEED, settings.model)
    optimizer = make_optimizer(model, settings, horizon)
    publication = optimizer.optimizers[0].mlp_publication
    estimator = Cut(model, settings, SEED, rank, world, publication, identity.model.head_tables)
    widen_anvil_schedule(optimizer, updates, settings)
    estimator.bind_flat_layout(optimizer)  # (the graph packs the gradients in the update's layout)
    pair_rows = noise[:2].to(device)
    estimator.capture_pair_graph(pair_rows[:, :-1].contiguous(), pair_rows[:, 1:].contiguous())
    del pair_rows
    random_pairs = noise[:2].repeat(world, 1).to(device).view(1, world, 2, noise.size(1))[:, rank]
    # (compiles the kernel for the timed call's strides)
    pair_sampling.transition_orders(random_pairs[:, :, :-1], random_pairs[:, :, 1:])
    del random_pairs
    optimizer.optimizers[0].capture_shard_graphs(torch.bfloat16)
    updater = cut_update_graph.AdamGraphs(estimator, optimizer, updates, settings, horizon)
    updater.capture([estimator._packed])
    optimizer.optimizers[1]._captured_update = updater
    del noise
    # Untimed, as in #44 (qlabs-eng/noise@3343973b, train_stages): read train.pt, split it by the seeded permutations
    # above and put the identity phase's batches on the GPU. No split or order here depends on the rows' contents;
    # load_rows only verifies their pinned hash.
    all_rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    cut_rows = all_rows.index_select(0, pool_indices)
    extra_rows = all_rows.index_select(0, extra_indices)
    identity.bind_rows(all_rows.numpy())
    del all_rows
    staged_rows = torch.empty((train_count, cut_rows.shape[1]), dtype=cut_rows.dtype, pin_memory=True)
    device_rows = torch.empty(staged_rows.shape, dtype=staged_rows.dtype, device=device)
    copy_stream = torch.cuda.Stream()
    torch.empty(4 * 2 ** 30, dtype=torch.uint8, device=device)  # Prime the allocator for timed temporary buffers.
    warm_cut_update(optimizer.optimizers[0], estimator._packed)  # (here: its scratch is cut from cached blocks)
    # The default group's all-gather, which the Identity hand-off uses (gather_fp32), once on zeros.
    dist.all_gather_into_tensor(torch.empty((2 * world, 2), device=device), torch.zeros((2, 2), device=device))
    # For metrics.jsonl: this rank's sum of each Cut update's clean token losses (wide_step) and three timestamps of
    # the main stream, read after the timer, and rank 0's peak allocated bytes, read as each update is issued.
    cut_loss_sums, cut_peaks = torch.full((len(updates),), float('nan'), device=device), []
    start_event, identity_end, cut_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.freeze()
    torch.cuda.synchronize()
    begin_training()
    start = time.perf_counter()
    start_event.record()
    cap_lookup.fill(identity.model.head_tables)  # (the cap tables both phases read, computed inside the timer)
    def prepare_rows():
        order = torch.from_numpy(batch_history_order(cut_rows.numpy()[:, 1:], difficulty_order(cut_rows, settings),
                                                     settings))
        resume = prefix_rows + len(extra_rows)
        torch.index_select(cut_rows, 0, order[:prefix_rows], out=staged_rows[:prefix_rows])
        staged_rows[prefix_rows:resume].copy_(extra_rows)
        torch.index_select(cut_rows, 0, order[prefix_rows:], out=staged_rows[resume:])
        return order

    with ThreadPoolExecutor(max_workers=1) as pool:
        plan = pool.submit(prepare_rows)
        identity.run()
        order = plan.result()
    with torch.cuda.stream(copy_stream):
        training = device_rows.copy_(staged_rows, non_blocking=True)
        rank_rows = training.view(len(updates), world, 2, training.size(1))[:, rank]
        orders = pair_sampling.transition_orders(rank_rows[:, :, :-1], rank_rows[:, :, 1:])
    torch.cuda.current_stream().wait_stream(copy_stream)
    with torch.no_grad():
        weights = model.state_dict()
        for name, value in identity.model.state_dict().items():
            weights[name].copy_(value)
    identity_end.record()
    publication.initialize()
    average_steps = average_plan(updates, settings.average_knots)
    means_stream = torch.cuda.Stream()
    for step, group in enumerate(updates, 1):
        begin, end = (group[0] - 1) * settings.cut_batch, group[-1] * settings.cut_batch
        batch = rank_slice(training[begin:end])
        # views of the device rows, copied into the graphs' inputs on the staging stream
        rows_in, rows_out = (batch[:, :-1], batch[:, 1:])
        publication.full = step in average_steps or step == len(updates)
        estimator.wide_step(rows_in, rows_out, orders[step - 1], optimizer, cut_loss_sums[step - 1])
        if rank == 0:
            cut_peaks.append(torch.cuda.memory_stats_as_nested_dict()['allocated_bytes']['all']['peak'])
        if step in average_steps:
            if averages is None:
                averages = [p.detach().float().clone() for p in model.parameters()]
            else:
                means_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(means_stream):
                    current = [p.detach().float() for p in model.parameters()]
                    torch._foreach_lerp_(averages, current, average_steps[step])
                    estimator._means_done = torch.cuda.Event()
                    estimator._means_done.record()
                del current
    cut_end.record()
    torch.cuda.current_stream().wait_stream(means_stream)
    blend_average(model, averages)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    end_training()

    # The Cut rows in the order they were trained.
    used = pool_indices[order]
    used = torch.cat((used[:prefix_rows], extra_indices, used[prefix_rows:]))
    identity_loss = average_ranks(identity.model.loss.clone())
    # Every Cut update's clean loss, from the sums wide_step took (NaN where none was written).
    if not (torch.isfinite(identity_loss).all() and torch.isfinite(cut_loss_sums).all()):
        raise FloatingPointError('Non-finite training loss')
    cut_losses = average_ranks(cut_loss_sums / estimator._wide_loss.numel())
    identity.close()
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    result = dict(
        val_loss=evaluate(state, args.data_dir), train_seconds=round(elapsed, 2),
        steps=len(updates) + IDENTITY_UPDATES, identity_steps=IDENTITY_UPDATES, cut_steps=len(updates),
        seqs=train_count + IDENTITY_UPDATES * BATCH,
        tokens=(train_count + IDENTITY_UPDATES * BATCH) * settings.model.sequence_len,
        peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), average_samples=len(average_steps),
    )
    if rank == 0:
        all_used = torch.cat((identity_indices, used))
        assert all_used.unique().numel() == len(all_used) == result['seqs'] == TRAIN_ROWS
        metrics = metrics_records(updates, settings, identity_loss.item(),
                                  start_event.elapsed_time(identity_end) / 1000,
                                  cut_losses.tolist(), cut_peaks, start_event.elapsed_time(cut_end) / 1000)
        (args.output / 'metrics.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in metrics))
        if not args.no_ckpt:
            torch.save(state, args.output / 'final.pt')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--no-ckpt', action='store_true')
    args = parser.parse_args()
    settings = Settings()
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
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
