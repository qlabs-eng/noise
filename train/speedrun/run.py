"""Track 2: the shortest training time on 8xH100 to reach validation loss TARGET, without backpropagation.

The track 1 record (Dust with ANVIL II on the block matrices and Adam elsewhere) at a 5k population.

Three changes to Dust's estimator (the optimizer is the ANVIL II record's; its Adam rate is doubled):
- The head's output error by central differences along every logit of every token (coordinate_differences):
  a token's loss depends on its own logits alone, so all of them are probed at once on the cached logits, with
  no draws and no sampling variance.
- Orthogonal draws: a token's draws over the step are rows of a sign-flipped Hadamard matrix, orthogonal across
  chunks and GPUs, so K draws recover K directions of the credit exactly.
- The MLP hidden layer is jittered only at live units (pre-activation > 0): a dead unit's credit is zero.
- An associative memory of credit: each direct site and attention output keeps a ridge regression M from the
  head's output error at a token (the realized error, which the coordinate head gives exactly) to the site's
  own output error, fitted on past steps' estimates. Its prediction mu = alpha M c is a control variate: the
  draws estimate only delta - mu (the predicted part of each reward is subtracted) and mu is added back, so the
  estimate keeps Dust's expectation for any mu while its variance falls by what the memory predicts.

Launch: torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.eval import evaluate
from utils.model import GPT, GPTConfig, attention, apply_rotary_emb, has_ve, norm
from utils.prepare_data import TRAIN_ROWS, load_rows

BATCH = 8  # sequences per update; 610 updates per epoch
SEED = 0
TARGET = 4.2  # validation loss to reach; the run counts only if its final weights reach it


@dataclass
class Recipe:
    """Draws per update, summed over GPUs; lists are per block."""
    writers: list[int]      # attn and mlp c_proj
    hidden: list[int]       # mlp.c_fc
    hubs: list[int]         # attention output
    embedding: int
    local: dict[str, int]   # attention internals
    nrep: int               # draws per batched forward
    lr: float               # Adam; embeddings 300x, mixing scalars 30x
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.0   # final fraction of the epoch with linear lr decay
    gamma: float = 0.98     # attention credit decay per token
    first_gamma: float = 0.3  # reward-to-go decay for block 0
    anvil_lr: float = 0.023
    anvil_wd: float = 2.25
    memory_beta: float = 0.9  # decay of the memory's running sums of estimates and of the key covariance
    mesa_ridge: float = 0.1   # ridge of the memory's read-out, in units of the key covariance's mean variance
    alpha_beta: float = 0.8   # running average over steps of the prior's scale fit


def recipe(scale):
    """The tuned 16k Adam cell, with every draw count scaled."""
    count = lambda k: max(64, round(k * scale / 64) * 64)
    return Recipe(
        writers=[count(256)] * 8, hidden=[count(512)] * 8, hubs=[count(512)] * 8,
        embedding=count(2048),
        local=dict(q=count(384), v=count(384), gate=count(128), k=count(1664), ve=count(4608)),
        nrep=16, lr=0.003, warmdown=0.3,
    )


RECIPE = recipe(0.3125)


def chunk_size(draws, cap):
    """Largest divisor of draws up to cap."""
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def reward_to_go(reward, gamma):
    """Discounted sum of future rewards per token."""
    lags = math.ceil(math.log(1e-6) / math.log(gamma))
    kernel = gamma ** torch.arange(lags, device=reward.device, dtype=reward.dtype)
    padded = F.pad(reward.flatten(0, 1).unsqueeze(1), (0, lags - 1))
    return F.conv1d(padded, kernel.view(1, 1, -1)).view_as(reward)


@torch.compile
def head_loss(x, weight, targets, vocab):
    """Final norm, head, soft cap and per-token cross-entropy."""
    logits = F.linear(norm(x), weight)[..., :vocab].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='none')


SITES = {'o': 'o@{}', 'attn': 'transformer.h.{}.attn.c_proj', 'fc': 'transformer.h.{}.mlp.c_fc',
         'mlp': 'transformer.h.{}.mlp.c_proj'}
torch._dynamo.config.cache_size_limit = 64


@torch.compile
def block_forward(x, w, ve, cos, sin, window, o=None, attn=None, fc=None, mlp=None):
    """Block.forward with optional jitter at o, attn, fc or mlp."""
    add = lambda t, jitter: t if jitter is None else t + jitter
    B, T, C = x.shape
    h = norm(x)
    dim = 2 * cos.shape[-1]
    q, k, v = (F.linear(h, w[name]).view(B, T, -1, dim) for name in ('q', 'k', 'v'))
    if ve is not None:
        gate = 2 * torch.sigmoid(F.linear(h[..., :w['gate'].shape[1]], w['gate']))
        v = v + gate.unsqueeze(-1) * ve.view_as(v)
    q, k = norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))
    y = add(attention(q, k, v, window).reshape(B, T, C), o)
    x = x + add(F.linear(y, w['proj']), attn)
    hidden = add(F.linear(norm(x), w['fc']), fc)
    return x + add(F.linear(F.relu(hidden).square(), w['out']), mlp)


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


HADAMARDS = {}


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


@torch.compile(fullgraph=True, dynamic=False)
def construct_probes(hadamard, patterns, offset, index, signs, fp32):
    """Fuse the deterministic lookup, sign products, and optional output conversion."""
    width = hadamard.shape[0]
    rows = (offset.unsqueeze(0) + index[:, None, None]) % width
    probes = hadamard[rows] * patterns[index // width][:, None, None, :] * signs
    return probes.float() if fp32 else probes


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


class DUST:
    """Forward-only gradient estimator: caches activations, then reruns from the jittered block."""

    def __init__(self, model, recipe, seed=0, rank=0, world=1):
        self.model, self.recipe, self.world, self.rank = model, recipe, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)  # the same on every GPU
        self.frames = {}
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.weights = [dict(q=b.attn.c_q.weight, k=b.attn.c_k.weight, v=b.attn.c_v.weight,
                             gate=b.attn.ve_gate.weight if b.attn.ve_gate is not None else None,
                             proj=b.attn.c_proj.weight, fc=b.mlp.c_fc.weight, out=b.mlp.c_proj.weight)
                        for b in model.transformer.h]
        self.capturing = False
        # The credit memory: per site its running sums S (estimates x keys) and the prior's scale alpha with its
        # fit statistics; the key covariance P, shared by every site since the key is the head's error at a token.
        self.memory, self.alpha, self.fits, self.calibration, self.estimates, self.shared = {}, {}, {}, {}, {}, {}
        self.prior_sites = [n for n in self.modules if n.endswith(('attn.c_proj', 'mlp.c_proj', 'mlp.c_fc'))]
        self.prior_sites += ['transformer.wte'] + [f'o@{layer}' for layer in range(model.config.n_layer)]

        counts = recipe.writers + recipe.hidden + recipe.hubs + [recipe.embedding] + list(recipe.local.values())
        if any(k % world for k in counts) or 8 % world:
            raise ValueError('Every draw count must split evenly across the GPUs.')

        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(layer)))

    def autocast(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.device.type == 'cuda' else nullcontext()

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

    def noise(self, *shape, dtype=None):
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype, generator=self.generator)

    def signs(self, *shape, shared=False):
        """Random +-1 in the draw dtype, from the GPU's own generator or the one shared by all GPUs."""
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws start..start + n of this step's world * draws at `site`. Draw i at a token is row (offset + i)
        mod d of the Hadamard matrix under the sign pattern of cycle i // d: the patterns (one per cycle of d
        draws, drawn up front in the same order on every GPU) and the tokens' row offsets are shared by the
        GPUs, so a token's draws are orthogonal across chunks and GPUs within each cycle, and the independent
        offsets leave the tokens' errors uncorrelated. Each draw's own sign flip keeps distinct draws
        uncorrelated, so the chunk centering shrinks the estimate uniformly, as with Gaussian draws. E[a a^T] = I."""
        if site not in self.frames:  # this step's frame at the site
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        matrix = hadamard(d, self.device, self.dtype)
        signs = self.signs(n, b, t, 1)  # Keep random sampling outside the compiled helper.
        if dtype not in (None, self.dtype, torch.float32):
            raise ValueError(f'Unsupported output dtype for the 5k recipe: {dtype}')
        return construct_probes(matrix, patterns, offset, index, signs, dtype == torch.float32)

    def losses(self, x, y):
        with self.autocast():
            return self.model(x, y, loss_reduction='none').view_as(y).float()

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()  # this step's draw frames
        self.capturing = True
        try:
            self.clean_loss = self.losses(x, y)
        finally:
            self.capturing = False
        with self.autocast():
            self.x0 = norm(self.outputs['transformer.wte'])

    def rerun(self, n, entry, jitter):
        """Per-token losses of n jittered draws, rerun from block entry (None: from the embedding)."""
        model, length = self.model, self.length
        cos, sin = model.cos[:, :length], model.sin[:, :length]
        with self.autocast():
            if entry is None:
                embedding = self.outputs['transformer.wte'].repeat(n, 1, 1)
                x = x0 = norm(embedding + jitter['transformer.wte'])
            else:
                x, x0 = self.blocks[entry].repeat(n, 1, 1), self.x0.repeat(n, 1, 1)
            for i in range(entry or 0, model.config.n_layer):
                if entry is None or i > entry:
                    x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                ve = self.outputs.get(f'value_embeds.{i}')
                sites = {key: jitter.get(site.format(i)) for key, site in SITES.items()} if i == entry else {}
                x = block_forward(x, self.weights[i], None if ve is None else ve.repeat(n, 1, 1), cos, sin,
                                  model.window_sizes[i][0], **sites)
            # The final norm, head and loss in float32. A draw moves a token's loss by about 1e-2, while bfloat16
            # logits (soft-capped at 15) round by up to about 6e-2, differently in every draw: in the deeper blocks
            # the rounding was most of the reward's variance. The clean loss's own rounding is common to all draws.
            with torch.autocast(self.device.type, enabled=False):
                weight = model.lm_head.weight.float()
                loss = head_loss(x.float(), weight, self.y.repeat(n, 1), model.config.vocab_size)
        return loss.view(n, self.batch, length)

    def memory_matrix(self, site):
        """The site's memory M = S (P + ridge)^-1, the ridge regression of past estimates on past keys; None
        before any write. One Cholesky factorization per step serves every site, since P is shared."""
        memory = self.memory.get(site)
        if memory is None:
            return None
        P = self.shared['P']
        ridge = self.recipe.mesa_ridge * P.diagonal().mean()
        if self.shared.get('chol_of') is not P:
            self.shared['chol'] = torch.linalg.cholesky(P + ridge * torch.eye(len(P), device=P.device))
            self.shared['chol_of'] = P
        return torch.cholesky_solve(memory['S'].T, self.shared['chol']).T

    def retrieve(self, sites):
        """Each site's retrieval M c from the memory as it stood before this step's estimates, c the head's
        output error at each token (zero where the target is ignored)."""
        retrieved = {}
        for site in sites:
            m = self.memory_matrix(site)
            if m is not None:
                retrieved[site] = F.linear(self.head_credit, m.float())
        return retrieved

    def remember(self, valid_tokens):
        """Write this step's GPU-averaged estimates into each site's memory: S = EMA(sum_t delta_t c_t^T / tokens),
        and once per step P = EMA(sum_t c_t c_t^T / tokens), the keys' covariance."""
        r, c = self.recipe, self.head_credit
        C = average_ranks(c.flatten(0, 1).T @ c.flatten(0, 1)) / valid_tokens
        self.shared['P'] = r.memory_beta * self.shared.get('P', torch.zeros_like(C)) + (1 - r.memory_beta) * C
        for site in self.prior_sites:
            G = average_ranks(torch.einsum('btd,btk->dk', self.estimates[site], c)) / valid_tokens
            memory = self.memory.setdefault(site, {})
            memory['S'] = r.memory_beta * memory.get('S', torch.zeros_like(G)) + (1 - r.memory_beta) * G

    def calibrate(self):
        """Average the fits over GPUs and set each site's alpha for the next step: alpha = <r, estimate> / |r|^2,
        with the estimate undone of centering's 1 - 1/n shrinkage, is unbiased for the variance-optimal scale,
        since the estimate's noise is independent of the retrieval r; numerator and denominator are averaged
        over steps (alpha_beta) separately."""
        sites = list(self.fits)
        if not sites:
            return
        fits = average_ranks(torch.stack([self.fits[s] for s in sites])).tolist()
        for site, (dot, retrieved) in zip(sites, fits):
            if retrieved > 0:
                beta = self.recipe.alpha_beta
                old_dot, old_retrieved = self.calibration.get(site, (dot, retrieved))
                self.calibration[site] = beta * old_dot + (1 - beta) * dot, beta * old_retrieved + (1 - beta) * retrieved
                self.alpha[site] = self.calibration[site][0] / self.calibration[site][1]
        self.fits.clear()

    def direct_errors(self, sites, draws, sigma, entry=None, gamma=0.0, retrieved=None):
        """Output errors from one-sided loss drops, centered per chunk. With retrievals from the memory, the
        draws estimate the residual delta - mu around the prior mu = alpha M c, and the prior is added back."""
        retrieved = retrieved or {}
        priors = {s: self.alpha[s] * r for s, r in retrieved.items() if s in self.alpha}
        n, b, t = chunk_size(draws, self.recipe.nrep), self.batch, self.length
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(b, t, width(s), device=self.device) for s in sites}
        # the MLP hidden layer is jittered only at live units: a dead unit's credit is zero (the ReLU is flat),
        # so its share of the jitter would only add noise to the estimate at that unit
        live = {s: (self.outputs[s] > 0).to(self.dtype) for s in sites if s.endswith('mlp.c_fc')}
        for s, mask in live.items():  # the prior is masked like the draws: a dead unit's credit is zero
            if s in priors:
                priors[s] = priors[s] * mask.float()

        for chunk in range(draws // n):
            noise = {s: self.orthogonal_noise(s, draws, chunk, n, b, t, width(s)).flatten(0, 1) for s in sites}
            for s, mask in live.items():
                noise[s] = (noise[s].view(n, b, t, -1) * mask).flatten(0, 1)
            perturbed_loss = self.rerun(n, entry, {s: sigma * a for s, a in noise.items()})

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            if gamma:
                reward = reward_to_go(reward, gamma)
            if priors:
                # The loss drop the prior predicts is -sigma a.mu, summed over the jittered sites; adding back its
                # negation leaves the reward of the residual. Centered like the reward it corrects.
                predicted = sigma * sum(torch.einsum('nbtd,btd->nbt', noise[s].view(n, b, t, -1), mu.to(self.dtype))
                                        for s, mu in priors.items()).float()
                reward = reward + (predicted - predicted.mean(0, keepdim=True) if n > 1 else predicted)
            for site, a in noise.items():
                errors[site] -= torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma

        # Centering over n draws scales the estimate's mean by 1 - 1/n; the prior is scaled to match, so the
        # estimate has Dust's expectation for any mu.
        kept = 1 - 1 / n if n > 1 else 1.0
        errors = {s: error / draws + kept * priors.get(s, 0) for s, error in errors.items()}
        self.estimates.update({s: e for s, e in errors.items() if s in self.prior_sites})  # for the memory's writes
        for site, r in retrieved.items():  # least-squares fit on the retrieval, for the next steps' alpha
            target = errors[site] / kept  # unshrunk: the reward correction should cancel sigma a.delta itself
            self.fits[site] = torch.stack([(r * target).sum(), (r * r).sum()])
        return errors

    def head_error(self, sigma=0.1):
        """The head's output error by a central difference of every token's loss along each of its own logits.

        A token's loss depends on its own logits alone, and moving one raw logit changes the partition function
        by one term, so every coordinate of every token is probed at once, elementwise on the cached logits:
        no draws and no sampling variance, only the O(sigma^2) error of the difference.
        """
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def local_errors(self, kind, draws, targets, sigma=0.1):
        """Attention-internal errors, scored against the attention-output targets."""
        cfg = self.model.config
        n = chunk_size(draws, self.recipe.nrep)
        b, t, h, d = self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        def repeat(x):
            return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]
            credit = ((lag >= 0) & (lag <= window)).float() * self.recipe.gamma ** lag.clamp(min=0)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for chunk in range(draws // n):
            for i in layers:
                a = self.orthogonal_noise(sites[i], draws, chunk, n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if kind in ('q', 'k'):
                    raw = q if kind == 'q' else k
                    perturbed = processed(
                        (raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d)
                    )
                    qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
                    delta = (
                        attention(qa, ka, repeat(values.to(self.dtype)), window).view(n, b, t, h, d)
                        - baseline
                    )
                else:
                    if kind == 'gate':
                        delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                        delta_value = delta_gate.unsqueeze(-1) * ve
                        divisor = 2 * sigma
                    else:
                        delta_value = sigma * a.view(n, b, t, h, d)
                        if kind == 've':
                            delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                    delta = attention(
                        repeat(qp), repeat(kp),
                        delta_value.reshape(n * b, t, h, d).to(self.dtype), window,
                    ).view(n, b, t, h, d)

                score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}

    def reassemble(self, site, error):
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def scalar_gradients(self, draws, sigma=0.03):
        """Two-sided ES for the residual mixing scalars."""
        parameters = [self.model.resid_lambdas, self.model.x0_lambdas]
        original = [p.clone() for p in parameters]
        estimate = torch.zeros(2, self.model.config.n_layer, device=self.device)
        try:
            for _ in range(draws):
                a = self.noise(2, self.model.config.n_layer, dtype=torch.float32)
                losses = []
                for sign in (1, -1):
                    for p, clean, direction in zip(parameters, original, a):
                        p.copy_(clean + sign * sigma * direction)
                    losses.append(self.losses(self.x, self.y))
                difference = (losses[0] - losses[1])[self.y != -1].sum() / (2 * sigma)
                estimate += difference * a / draws
        finally:
            for p, clean in zip(parameters, original):
                p.copy_(clean)
        return dict(zip(('resid_lambdas', 'x0_lambdas'), estimate))

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """Estimate every gradient, then take one optimizer step."""
        self.capture(x, y)
        r, world, layers = self.recipe, self.world, self.model.config.n_layer
        gradients = {}
        # The head first: its output error at each token is the memory's key
        head = self.head_error()
        self.head_credit = head[..., :self.model.config.vocab_size]
        gradients['lm_head.weight'] = self.reassemble('lm_head', head)
        for layer in range(layers):
            prefix = f'transformer.h.{layer}.'
            gamma = r.first_gamma if layer == 0 else 0.0
            groups = [
                ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                ([prefix + 'mlp.c_fc'], r.hidden[layer]),
            ]
            for sites, count in groups:
                retrieved = self.retrieve(sites)
                for site, error in self.direct_errors(sites, count // world, 0.4, layer, gamma, retrieved).items():
                    gradients[site + '.weight'] = self.reassemble(site, error)

        site = 'transformer.wte'
        error = self.direct_errors([site], r.embedding // world, 0.8, retrieved=self.retrieve([site]))[site]
        gradients[site + '.weight'] = self.reassemble(site, error)

        targets = {}
        for layer in range(layers):
            site = f'o@{layer}'
            sigma = 0.4 if layer < layers // 2 else 0.8
            gamma = r.first_gamma if layer == 0 else 0.0
            error = self.direct_errors([site], r.hubs[layer] // world, sigma, layer, gamma, self.retrieve([site]))[site]
            targets[layer] = average_ranks(error)

        for kind, count in r.local.items():
            for site, error in self.local_errors(kind, count // world, targets).items():
                gradients[site + '.weight'] = self.reassemble(site, error)
        gradients.update(self.scalar_gradients(8 // world))
        self.calibrate()  # the next step's alpha

        valid_tokens = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        self.remember(valid_tokens)  # after this step's estimates, so a retrieval never sees its own step
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()


def make_model(device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(GPTConfig())
    model.to_empty(device=device)
    model.init_weights()
    # output projections: uniform with std 1.5 / sqrt(fan_in) instead of zeros
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
# PR #360's total, warmup, cooldown and engage steps; scaled to the run length
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


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


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
    return Optimizers(Anvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps), adam)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--scale', type=float, help='Scale every draw count of the 16k Adam cell instead.')
    parser.add_argument('--warmdown', type=float, help='Override the recipe\'s warmdown fraction.')
    parser.add_argument('--lr', type=float, help='Override the recipe\'s Adam learning rate.')
    parser.add_argument('--anvil-lr', type=float, help='Override the recipe\'s ANVIL II learning rate.')
    parser.add_argument('--steps', type=int, help='Stop after this many updates (smoke tests).')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many updates.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    settings = recipe(args.scale) if args.scale else RECIPE
    if args.warmdown is not None:
        settings.warmdown = args.warmdown
    if args.lr is not None:
        settings.lr = args.lr
    if args.anvil_lr is not None:
        settings.anvil_lr = args.anvil_lr

    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps = min(len(training) // BATCH, args.steps or len(training))
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, batch=BATCH, steps=steps, seed=SEED,
            target=TARGET, recipe=asdict(settings), torch_version=torch.__version__,
        ), indent=2) + '\n')

    def init():
        model = make_model(device, SEED)
        return model, make_optimizer(model, settings, steps), DUST(model, settings, SEED, rank, world)

    def set_rate(optimizer, step):
        left = (steps - step + 1) / steps
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / settings.warmdown if settings.warmdown else 1.0)

    # untimed warmup on random tokens, then rebuild from the seed
    model, optimizer, estimator = init()
    noise = torch.randint(0, 4096, (BATCH, 2049), generator=torch.Generator().manual_seed(SEED)).to(device)
    estimator.step(noise[:, :-1].contiguous(), noise[:, 1:].contiguous(), optimizer)
    estimator.close()
    model, optimizer, estimator = init()

    if rank == 0:
        print(f'speedrun: {steps} steps, recipe={asdict(settings)}, GPUs={world}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        batch = training[(step - 1) * BATCH:step * BATCH].to(device)
        set_rate(optimizer, step)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and step % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record['val_loss'] = evaluate(model.state_dict(), args.data_dir)
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')

    estimator.close()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if rank == 0:
        torch.save(state, args.output / 'final.pt')
    val_loss = evaluate(state, args.data_dir)
    if rank == 0:
        result = dict(val_loss=val_loss, train_seconds=round(elapsed, 2), steps=step,
                      tokens=step * BATCH * 2048, reached_target=val_loss <= TARGET)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
