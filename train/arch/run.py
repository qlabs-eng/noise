"""Track 3: Boost -> Cut with a quarter-turn context tap and exact future-token probes, merged at the end.

The architecture (train/arch/model.py) adds a quarter turn of the previous token's embedding to each token's:
z_t = E[x_t] + 0.5 * R(E[x_(t-1)]), with R[a, b] = [-b, a] on the two halves of the embedding channels. The
first position of every row has no predecessor. Measured embedding credit passes through the transpose of this
linear lookup map, credit_t + 0.5 * R.T(credit_(t+1)), before the table update; nothing is differentiated.

Training follows the native 30M record: 20% Boost and 80% Cut at global batch four, with two GPUs sharing each
sequence's probes on an eight-GPU node, the record's ANVIL/Adam recipe and warmdown. The shuffled epoch
reserves its final 32 training rows solely for the end-only trajectory merge; the other 14,616 rows each feed
one update. Validation never selects weights.

Future-token probes move one source at a time (the hidden units and queries of layers 0-3, and the embedding)
and score its actual forward responses at later, original measured attention hubs over complete cycles, with
nothing sampled: coordinate cycles for hidden units and complete Hadamard cycles for queries and the
embedding. The value term is exact at every receiver; the key term is evaluated exactly at each head's 16
strongest receivers. Corrected linear credits never become another nonlinear probe's target.
A --steps diagnostic prefix is not a completed benchmark epoch.

    uv run torchrun --standalone --nproc_per_node=8 train/arch/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import cache, partial
from importlib import import_module
from math import gcd
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch._dynamo import config as dynamo_config
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Track 3: the model is this folder's model.py; utils/eval.py scores it with --model.
MODEL_FILE = Path(__file__).resolve().with_name('model.py')
from model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask
from utils.eval import evaluate
from utils.prepare_data import TRAIN_ROWS, load_rows

if TYPE_CHECKING:
    from collections import defaultdict
    from collections.abc import Callable, Iterable, Iterator
    from typing import TextIO

    from torch.nn.attention.flex_attention import BlockMask
    from torch.utils.hooks import RemovableHandle

SEED = 0
UPSTREAM_REVISION = 'qlabs-eng/noise@b6405a155a38371f994a8f3915ea5d8a87235315'
_FIXED_HEAD_STD = 0.001
_MAX_RELAY_RECIPIENTS = 32
EPS = torch.finfo(torch.float32).eps
dynamo_config.cache_size_limit = 64


# Typed views of the consumed Torch 2.10 interfaces; they never wrap or change a tensor operation.
_Kernel = TypeVar('_Kernel', bound='Callable[..., object]')


class CollectiveOperations(Protocol):
    """Describe the synchronous collectives used by the trainers."""

    def all_reduce(
        self,
        tensor: Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Reduce a tensor in place with the default synchronous behavior."""
        ...

    def all_gather(self, tensors: list[Tensor], tensor: Tensor) -> None:
        """Gather into the supplied tensor list synchronously."""
        ...

    def new_group(self, ranks: list[int]) -> dist.ProcessGroup | int | None:
        """Create a process group, retaining the non-member sentinel."""
        ...

    def barrier(self) -> None:
        """Wait for the default process group synchronously."""
        ...


class LinearAlgebra(Protocol):
    """Describe the default dense solves used by the tiny proposal fit."""

    def eigvalsh(self, matrix: Tensor) -> Tensor:
        """Compute Hermitian eigenvalues with Torch's default lower triangle."""
        ...

    def solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        """Solve the default left-sided linear system."""
        ...


class TorchOperations(Protocol):
    """Describe consumed module-level APIs whose upstream types are incomplete."""

    distributed: CollectiveOperations
    linalg: LinearAlgebra

    @overload
    def compile(self, model: _Kernel, *, dynamic: bool | None = None) -> _Kernel: ...

    @overload
    def compile(
        self,
        model: None = None,
        *,
        dynamic: bool | None = None,
    ) -> Callable[[_Kernel], _Kernel]: ...

    def manual_seed(self, seed: int) -> torch.Generator:
        """Seed Torch using its unchanged global-generator implementation."""
        ...


class TensorOperations(Protocol):
    """Describe the exact norm and split overloads consumed by this repository."""

    def norm(
        self,
        p: float | Literal['fro', 'nuc'] | None = 'fro',
        dim: int | tuple[int, ...] | None = None,
        *,
        keepdim: bool = False,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Retain Tensor.norm's dispatch, dtype and default Frobenius semantics."""
        ...

    def split(self, split_size: int | list[int], dim: int = 0) -> tuple[Tensor, ...]:
        """Split by a chunk size or explicit sizes without copying tensor data."""
        ...


class NamedModules(Protocol):
    """Describe the no-argument module traversal used during training setup."""

    def named_modules(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield the original named module traversal in registration order."""
        ...


torch_operations = cast('TorchOperations', torch)
# Pinned Torch 2.10 exposes no public equivalent preserving this batched in-place copy.
foreach_copy = torch._foreach_copy_  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]


@dataclass
class Settings:
    """Own-query/value-path credit, full hidden FD, and optional independent input-K probes."""

    model: GPTConfig = field(default_factory=GPTConfig)
    estimator: str = 'cut'
    boost_hub: str = field(default='projected', init=False)  # persisted algorithm version, not an ablation flag
    mlp_output_credit: str = field(default='next_mid', init=False)
    switch: float = 0.2
    boost_batch: int = 4
    cut_batch: int = 4
    projection_std: float = 1.5
    lr: float = 0.0021
    betas: tuple[float, float] = (0.5, 0.98)
    warmdown: float = 0.4
    final: float = 0.3  # end-merge trajectories retain this fraction of each learning rate
    head_std: float = 0.02
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
    chunk: int = 2**27
    chain_sigma: float = 0.01
    route_k: int = 512
    chain_tokens: int = 512
    chain_precision: str = 'bf16'
    own_query: bool = True  # PR #19's measured own-query paths in later attention blocks
    value_path: bool = True  # PR #23: one RMSNorm step, then only linear writers/V/VE
    path_scalars: bool = True  # also contract input-path credit into the linear mixing coefficients
    key_path_draws: int = 512  # independent input-K central pairs per token
    key_path_sigma: float = 0.01  # relative input RMS step
    key_path_tokens: int = 64  # source-token chunk; every allowed query contributes
    gate_path: bool = True  # independent full-cycle input probes through the VE gate
    gate_path_sigma: float = 0.01  # relative input RMS step
    future_draws: int = 128  # independent Q/WTE probes; zero disables all future corrections
    future_hidden_draws: int = 128  # independent hidden-source directions, up to the full hidden width
    future_layers: int = 4  # first blocks whose hidden and Q outputs receive this correction
    future_replay_depth: int = 0  # intervening source-token blocks; zero preserves residual-identity transport
    future_residual_correction: bool = False
    future_relay_tokens: int = 0  # selected recipients per first attention hop; zero disables two-hop credit
    merge_normalized: bool = False
    future_tokens: int = 128
    future_chunk: int = 512
    future_sigma: float = 0.01  # source-relative RMS; Q includes its own complete MLP response
    future_exact: tuple[str, ...] = ()  # families ('hidden', 'query', 'wte') measured over complete cycles
    future_receivers: int = 16  # exact families: receivers per head whose key term is evaluated exactly
    query_keys: int = 256  # moving key logits per head; 0 reruns all allowed logits
    key_sigma: float = 0.003
    key_chunk: int = 32

    def __post_init__(self) -> None:
        """Validate the accepted stage configuration before allocating state."""
        width, hidden = self.model.n_embd, 4 * self.model.n_embd
        if self.estimator not in ('boost', 'cut') or not 0 <= self.switch <= 1:
            message = 'Invalid stage estimator or Boost fraction'
            raise ValueError(message)
        if not 0 <= self.fast_weight <= 1:
            message = 'ANVIL fast weight must be in [0,1]'
            raise ValueError(message)
        if width < 1 or width & (width - 1):
            message = 'Full cycles require power-of-two model width'
            raise ValueError(message)
        if not 1 <= self.route_k <= hidden or self.chain_precision not in ('bf16', 'fp32'):
            message = 'Invalid sparse route budget or chain precision'
            raise ValueError(message)
        if self.query_draws != width // self.model.n_head:
            message = 'Queries use one complete cycle per head'
            raise ValueError(message)
        self._validate_query_and_gate()
        self._validate_future()
        self._validate_relays()
        self._validate_probe_sizes(width)

    def _validate_query_and_gate(self) -> None:
        """Validate the independent query and gate experiment controls."""
        if not 0 < self.query_sigma < float('inf'):
            message = 'Query step must be finite and positive'
            raise ValueError(message)
        if not 0 < self.gate_path_sigma < float('inf') or (self.gate_path and self.estimator != 'cut'):
            message = 'Gate input probes require Cut and a finite positive step'
            raise ValueError(message)

    def _validate_future(self) -> None:
        """Validate source coverage, independent budgets, and execution tiles."""
        width = self.model.n_embd
        if (
            type(self.future_draws) is not int
            or not 0 <= self.future_draws <= width
            or (self.future_draws and self.estimator != 'cut')
        ):
            message = 'Future probes require Cut and a budget no greater than model width'
            raise ValueError(message)
        if type(self.future_hidden_draws) is not int or not 1 <= self.future_hidden_draws <= 4 * width:
            message = 'Future hidden probes require an integer budget within the hidden width'
            raise ValueError(message)
        if type(self.future_replay_depth) is not int or not 0 <= self.future_replay_depth < self.model.n_layer:
            message = 'Future replay depth must be an integer below the number of blocks'
            raise ValueError(message)
        if self.future_residual_correction and (not self.future_draws or not self.future_replay_depth):
            message = 'Residual correction requires enabled future probes and positive source replay depth'
            raise ValueError(message)
        if (
            type(self.future_layers) is not int
            or self.future_layers < 1
            or (self.future_draws and self.future_layers > self.model.n_layer)
            or type(self.future_tokens) is not int
            or self.future_tokens < 1
            or type(self.future_chunk) is not int
            or self.future_chunk < 1
            or not 0 < self.future_sigma < float('inf')
        ):
            message = 'Invalid future source layers, batching limits or finite step'
            raise ValueError(message)
        if type(self.future_receivers) is not int or self.future_receivers < 1:
            message = 'Exact future probes need a positive integer receiver count'
            raise ValueError(message)
        if not set(self.future_exact) <= {'hidden', 'query', 'wte'} or (
            self.future_exact
            and (
                not self.future_draws
                or self.future_replay_depth
                or self.future_residual_correction
                or self.future_relay_tokens
            )
        ):
            message = 'Exact future families need enabled identity-transport future probes'
            raise ValueError(message)

    def _validate_relays(self) -> None:
        """Bound two-hop execution tiles without changing the sampled direction budget."""
        if type(self.future_relay_tokens) is not int or not 0 <= self.future_relay_tokens <= _MAX_RELAY_RECIPIENTS:
            message = 'Future relay recipients must be an integer between zero and 32'
            raise ValueError(message)
        if self.future_relay_tokens:
            if not self.future_draws or not self.future_replay_depth:
                message = 'Future relays require enabled future probes and positive source replay depth'
                raise ValueError(message)
            self.future_tokens = min(self.future_tokens, 4)
            self.future_chunk = min(self.future_chunk, 16)

    def _validate_probe_sizes(self, width: int) -> None:
        """Validate the remaining full-cycle and batching controls."""
        query_keys = cast('object', self.query_keys)
        if not isinstance(query_keys, int) or query_keys < 0:
            message = 'Own-query key count must be a nonnegative integer'
            raise ValueError(message)
        if type(self.key_path_draws) is not int or not 0 <= self.key_path_draws <= width:
            message = 'Input-K pairs must be an integer between zero and model width'
            raise ValueError(message)
        if (
            type(self.key_path_tokens) is not int
            or self.key_path_tokens < 1
            or not 0 < self.key_path_sigma < float('inf')
        ):
            message = 'Input-K chunk and finite step must be positive'
            raise ValueError(message)
        if any(
            v <= 0
            for v in (
                self.nrep,
                self.emb_chunk,
                self.chain_tokens,
                self.key_chunk,
                self.chunk,
                self.sigma,
                self.fc_sigma,
                self.chain_sigma,
                self.key_sigma,
                self.scalar_step,
            )
        ):
            message = 'Probe sizes and batching limits must be positive'
            raise ValueError(message)

    @property
    def chain_dtype(self) -> torch.dtype:
        """Return the replay precision selected by the accepted configuration."""
        return torch.bfloat16 if self.chain_precision == 'bf16' else torch.float32


# Real Torch/Triton objects are cast only at incomplete upstream typing boundaries.
OwnQuery: TypeAlias = tuple[
    Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor], Tensor, Tensor, list[int]
]
OwnRouted: TypeAlias = tuple[
    Literal['routed'],
    Tensor,
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    tuple[Tensor, Tensor],
]
OwnWindow: TypeAlias = tuple[
    Literal['window'],
    Tensor,
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    Tensor,
    Tensor,
    list[int],
]
OwnChain: TypeAlias = OwnRouted | OwnWindow

HADAMARDS: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
_ROUTED_WIDTHS = (32, 64)
_ROUTED_KEY_LIMIT = 256
_MATRIX_RANK = 2


class _FlexLSE(Protocol):
    def __call__(
        self, query: Tensor, key: Tensor, value: Tensor, *, block_mask: BlockMask, return_lse: Literal[True]
    ) -> tuple[Tensor, Tensor]: ...


class _FlexGQA(Protocol):
    def __call__(
        self, query: Tensor, key: Tensor, value: Tensor, *, block_mask: BlockMask, enable_gqa: Literal[True]
    ) -> Tensor: ...


_flex_lse = cast('_FlexLSE', flex)
_flex_gqa = cast('_FlexGQA', flex)
_OpParameters = ParamSpec('_OpParameters')


class _CustomTensorOp(Protocol[_OpParameters]):
    def __call__(self, *args: _OpParameters.args, **kwargs: _OpParameters.kwargs) -> Tensor: ...

    def register_fake(self, function: Callable[_OpParameters, Tensor]) -> Callable[_OpParameters, Tensor]:
        """Register the real custom operator's fake-tensor implementation."""
        ...


class _CustomOpFactory(Protocol):
    def __call__(
        self, name: str, *, mutates_args: tuple[()]
    ) -> Callable[[Callable[_OpParameters, Tensor]], _CustomTensorOp[_OpParameters]]: ...


_custom_op = cast('_CustomOpFactory', torch.library.custom_op)


if TYPE_CHECKING:

    class _TritonTensor(Protocol):
        """Describe the consumed tensor/pointer operations in Triton's AST interpreter."""

        @property
        def dtype(self) -> object:
            """Expose the opaque real Triton dtype, passed unchanged to tensor.to."""
            ...

        def to(self, dtype: object) -> _TritonTensor:
            """Convert a DSL tensor using the original Triton dtype."""
            ...

        def __getitem__(self, index: slice | tuple[slice | None, ...] | None) -> _TritonTensor: ...
        def __add__(self, other: _TritonTensor | float) -> _TritonTensor: ...
        def __radd__(self, other: float) -> _TritonTensor: ...
        def __sub__(self, other: _TritonTensor | float) -> _TritonTensor: ...
        def __rsub__(self, other: float) -> _TritonTensor: ...
        def __mul__(self, other: _TritonTensor | float) -> _TritonTensor: ...
        def __rmul__(self, other: float) -> _TritonTensor: ...
        def __truediv__(self, other: _TritonTensor | float) -> _TritonTensor: ...
        def __rtruediv__(self, other: float) -> _TritonTensor: ...
        def __floordiv__(self, other: int) -> _TritonTensor: ...
        def __mod__(self, other: int) -> _TritonTensor: ...
        def __neg__(self) -> _TritonTensor: ...
        def __lt__(self, other: _TritonTensor | int) -> _TritonTensor: ...
        def __le__(self, other: _TritonTensor | int) -> _TritonTensor: ...
        def __gt__(self, other: _TritonTensor | int) -> _TritonTensor: ...
        def __ge__(self, other: _TritonTensor | int) -> _TritonTensor: ...
        def __and__(self, other: _TritonTensor) -> _TritonTensor: ...
else:
    # Triton visits annotation ASTs as well as __annotations__; erase this host-only type.
    _TritonTensor = None


class _TritonLanguage(Protocol):
    """Describe only the actual language functions used by finite-forward kernels."""

    constexpr: type[object]
    float32: object

    def program_id(self, axis: int) -> _TritonTensor: ...
    def arange(self, start: int, end: int) -> _TritonTensor: ...
    def load(
        self, pointer: _TritonTensor, mask: _TritonTensor | None = None, other: int | None = None
    ) -> _TritonTensor: ...
    def store(self, pointer: _TritonTensor, value: _TritonTensor, mask: _TritonTensor) -> None: ...
    def dot(
        self, left: _TritonTensor, right: _TritonTensor, *, input_precision: str | None = None
    ) -> _TritonTensor: ...
    def trans(self, value: _TritonTensor) -> _TritonTensor: ...
    def sum(self, value: _TritonTensor, axis: int) -> _TritonTensor: ...
    def min(self, value: _TritonTensor, axis: int) -> _TritonTensor: ...
    def max(self, value: _TritonTensor, axis: int) -> _TritonTensor: ...
    def minimum(self, left: int | _TritonTensor, right: int | _TritonTensor) -> _TritonTensor: ...
    def maximum(self, left: int | _TritonTensor, right: int | _TritonTensor) -> _TritonTensor: ...
    def full(self, shape: tuple[int, ...], value: int, dtype: object) -> _TritonTensor: ...
    def where(self, condition: _TritonTensor, value: _TritonTensor, other: int) -> _TritonTensor: ...


class _TritonLibdevice(Protocol):
    def exp(self, value: _TritonTensor) -> _TritonTensor: ...
    def expm1(self, value: _TritonTensor) -> _TritonTensor: ...
    def log1p(self, value: _TritonTensor) -> _TritonTensor: ...


class _TritonLaunch(Protocol):
    def __call__(
        self, *args: Tensor | int, num_warps: int, num_stages: int = 3, enable_fp_fusion: bool = True
    ) -> object: ...


class _TritonKernel(Protocol):
    def __getitem__(self, grid: tuple[int, int]) -> _TritonLaunch: ...


class _TritonRuntime(Protocol):
    def jit(self, function: Callable[..., None]) -> _TritonKernel: ...
    def cdiv(self, numerator: int, denominator: int) -> int: ...
    def next_power_of_2(self, value: int) -> int: ...


class _TritonBuiltins(Protocol):
    def range(self, start: int | _TritonTensor, stop: int | _TritonTensor, step: int) -> Iterable[_TritonTensor]: ...


# Triton accepts real module globals and recognizes this range by identity. Its
# interpreter accepts scalar tensor bounds that Python's host type signature cannot express.
_triton_builtins = cast('_TritonBuiltins', import_module('builtins'))


@cache
def _triton_runtime() -> _TritonRuntime:
    return cast('_TritonRuntime', import_module('triton'))


def _jit_kernel(function: Callable[..., None], language: _TritonLanguage) -> _TritonKernel:
    # Only int-annotated arguments are constexpr. Runtime scalar/pointer arguments
    # remain unannotated for JIT specialization, as in the published kernels.
    function.__annotations__ = {
        name: language.constexpr
        for name, annotation in function.__annotations__.items()
        if annotation is int or annotation == 'int'
    }
    return _triton_runtime().jit(function)


@torch_operations.compile
def coordinate_differences(raw: Tensor, y: Tensor, sigma: float) -> Tensor:
    """Evaluate coordinate differences with the published finite-forward arithmetic.

    Finite differences of each token's loss along each of its own logits: for every token and every logit j, the
    loss with raw logit j moved up by sigma minus the loss with it moved down by sigma, over 2 sigma.

    Each of the two is a full forward evaluation of the soft-capped cross-entropy at the moved logit, written in
    closed form: moving one logit changes the partition function Z = sum_k exp(z_k) by one term, so the moved
    loss differs from the clean loss by log(Z - exp(z_j) + exp(z_j')) - log Z = log1p(exp(z_j) / Z * expm1(z_j'
    - z_j)) in the partition and by -(z_j' - z_j) at the target logit. Nothing is differentiated: exp(z_j) / Z
    are the clean forward's probabilities, used only to evaluate the perturbed losses. Zero at ignored tokens (y
    < 0).
    """

    def cap(v: Tensor) -> Tensor:
        """Evaluate cap with the published finite-forward arithmetic."""
        return 15 * torch.tanh(v / 15)

    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)  # exp(z_j) / Z of the clean forward

    def moved_loss(new: Tensor) -> Tensor:
        """Evaluate moved loss with the published finite-forward arithmetic."""
        return torch.log1p(weight * torch.expm1(new - z))

    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))  # the target logit's own term
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Evaluate hadamard with the published finite-forward arithmetic.

    The Sylvester Hadamard matrix of order d (a power of two), entries +-1, built once per process.
    """
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            message = f'Orthogonal draws need a power-of-two width, not {d}.'
            raise ValueError(message)
        h = torch.ones(1, 1, device=device)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HADAMARDS[d, device, dtype] = h.to(dtype)  # +-1 is exact in bfloat16
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor: Tensor) -> Tensor:
    """Evaluate average ranks with the published finite-forward arithmetic."""
    if torch.distributed.is_initialized():
        torch_operations.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


def attention_state(
    qraw: Tensor,
    kraw: Tensor,
    values: Tensor,
    cos: Tensor,
    sin: Tensor,
    nhead: int,
    window: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate attention state with the published finite-forward arithmetic.

    Dense clean forward context for independently scored local attention trials.
    """

    def heads(raw: Tensor) -> Tensor:
        """Evaluate heads with the published finite-forward arithmetic."""
        shaped = raw.reshape(*raw.shape[:2], nhead, -1)
        return norm(apply_rotary_emb(shaped, cos, sin)).transpose(1, 2)

    q, k = heads(qraw), heads(kraw)
    pos = torch.arange(qraw.shape[1], device=qraw.device)
    lag = pos[:, None] - pos[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).masked_fill(~allowed, -torch.inf).softmax(-1)
    return q, k, a, a @ values


def chunk_size(draws: int, cap: int) -> int:
    """Evaluate chunk size with the published finite-forward arithmetic.

    Largest divisor of draws up to cap.
    """
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def frozen_attention(
    q: Tensor,
    k_own: Tensor,
    v_own: Tensor,
    k_clean: Tensor,
    v_clean: Tensor,
    window: int,
) -> Tensor:
    """Evaluate frozen attention with the published finite-forward arithmetic.

    The attention each token would see if only its own state moved: every query attends to the other positions'
    clean keys and values and to its own moved ones, by swapping its own clean key and value through the log-
    sum-exp of one attention over the clean keys and values. Tensors are [batch, token, head, dim].
    """
    if not q.is_cuda:
        t = q.shape[1]
        lag = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
        logits = q.transpose(1, 2) @ k_clean.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** 0.5
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), -torch.inf)
        own = (q * k_own).sum(-1).transpose(1, 2) / q.shape[-1] ** 0.5
        logits = logits.diagonal_scatter(own, dim1=-2, dim2=-1)
        weights = logits.softmax(-1)
        out = (weights @ v_clean.transpose(1, 2)).transpose(1, 2)
        return out + weights.diagonal(dim1=-2, dim2=-1).transpose(1, 2).unsqueeze(-1) * (v_own - v_clean)
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = _flex_lse(
        q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask, return_lse=True
    )
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)


frozen_attention_compiled = torch_operations.compile(frozen_attention)


@torch_operations.compile(dynamic=False)
def cycle_derivatives(
    base: Tensor,
    c: Tensor,
    y: Tensor,
    steps: Tensor,
    sq: Tensor,
    dots: Tensor,
    dir_sq: Tensor,
    width: int,
) -> Tensor:
    """Evaluate cycle derivatives with the published finite-forward arithmetic.

    Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x
    of the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq
    (P,) |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every
    probe.
    """
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = c.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base_y * inv0 / 15
    )
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * c) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch_operations.compile(dynamic=False)
def moved_heads(
    raw: Tensor,
    a: Tensor,
    sigma: float,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Evaluate moved heads with the published finite-forward arithmetic."""
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch_operations.compile(dynamic=False)
def query_probe_heads(
    raw: Tensor,
    a: Tensor,
    sigma: float,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Evaluate query probe heads with the published finite-forward arithmetic.

    Preserve the original finite norm calculation, then group probe heads.
    """
    moved = moved_heads(raw, a, sigma, cos, sin)
    batch, length, heads, width = raw.shape
    if raw.is_cuda and batch > 1:
        return (
            moved.view(len(a), batch, length, heads, width)
            .permute(1, 3, 0, 2, 4)
            .reshape(batch, heads * len(a), length, width)
        )
    return moved


def query_probe_attention(
    moved: Tensor,
    keys: Tensor,
    values: Tensor,
    window: int,
    count: int,
    batch: int,
) -> Tensor:
    """Evaluate query probe attention with the published finite-forward arithmetic.

    All probes share clean K/V; group probe heads instead of copying K/V.
    """
    _, length, heads, width = keys.shape
    if not moved.is_cuda or batch == 1:

        def repeat(z: Tensor) -> Tensor:
            """Evaluate repeat with the published finite-forward arithmetic."""
            return z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count * batch, length, heads, width)

        return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)
    out = _flex_gqa(
        moved,
        keys.transpose(1, 2),
        values.transpose(1, 2),
        block_mask=window_mask(window, length, moved.device),
        enable_gqa=True,
    )
    return out.view(batch, heads, count, length, width).permute(2, 0, 3, 1, 4)


@torch_operations.compile(dynamic=False)
def head_scores(out: Tensor, baseline: Tensor, hub: Tensor) -> Tensor:
    """Evaluate head scores with the published finite-forward arithmetic.

    Score either moved-minus-clean or plus-minus-minus by ordinary broadcasting.
    """
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch_operations.compile(dynamic=False)
def code_estimate(credited: Tensor, a: Tensor) -> Tensor:
    """Evaluate code estimate with the published finite-forward arithmetic."""
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores: Tensor, directions: Tensor) -> Tensor:
    """Evaluate decode local scores with the published finite-forward arithmetic.

    Sum scalar probe scores times directions; the caller owns budget normalization.

    scores [..., P], directions [..., P, D] (or shared [P, D]). Keep this reduction inside the caller's compiled
    kernel, without creating a materialized score-times-direction tensor.
    """
    if directions.ndim == _MATRIX_RANK:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch_operations.compile(dynamic=False)
def decode_local_response(
    scores: Tensor,
    steps: Tensor,
    directions: Tensor,
    *,
    central: bool = True,
) -> Tensor:
    """Evaluate decode local response with the published finite-forward arithmetic.

    Decode independent trials; one-sided scores must already subtract clean.
    """
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

    response: Callable[[Tensor, Tensor], Tensor]
    score: Callable[[Tensor], Tensor]
    central: bool = field(default=True, kw_only=True)

    def __call__(self, rows: Tensor, steps: Tensor) -> Tensor:
        """Decode the prepared experiment against its fixed scoring target."""
        return decode_local_response(self.score(self.response(rows, steps)), steps, rows, central=self.central)


@torch_operations.compile(dynamic=False)
def hidden_local_credit(
    u: Tensor,
    steps: Tensor,
    g: Tensor,
    out_weight: Tensor,
) -> Tensor:
    """Evaluate hidden local credit with the published finite-forward arithmetic.

    Collapse the full Hadamard central-difference cycle for the separable hidden objective.

    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's scalar finite difference. Keep
    that finite step, including ReLU crossings; contract in FP32 instead of rounding every direction's projected
    response to BF16. u and out_weight contain all units (or a GPU's contiguous slice).
    """
    up, down = functional.relu(u + steps), functional.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope


@torch_operations.compile(dynamic=False)
def embedding_moves(
    e: Tensor,
    steps: Tensor,
    rows: Tensor,
    sign: float,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate embedding moves with the published finite-forward arithmetic."""
    n, (b, t, width) = rows.shape[0], e.shape
    hd = 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + sign * steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)

    def heads(z: Tensor) -> Tensor:
        """Evaluate heads with the published finite-forward arithmetic."""
        return z.view(n * b, t, -1, hd)

    q = norm(apply_rotary_emb(heads(functional.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(functional.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(functional.linear(x0, wv))


@torch_operations.compile(dynamic=False)
def readout_losses(x: Tensor, weight: Tensor, y: Tensor) -> Tensor:
    """Evaluate readout losses with the published finite-forward arithmetic."""
    logits = functional.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return functional.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)


@torch_operations.compile(dynamic=False)
def stage_readout(
    x0: Tensor,
    out: Tensor,
    mix: Tensor,
    wproj: Tensor,
    weight: Tensor,
    y: Tensor,
) -> Tensor:
    """Evaluate stage readout with the published finite-forward arithmetic."""
    xmid = mix * x0 + functional.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)


def norm_difference(x: Tensor, delta: Tensor) -> Tensor:
    """Evaluate norm difference with the published finite-forward arithmetic.

    norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves.
    """
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / x.shape[-1] + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch_operations.compile(dynamic=False)
def norm_move(x: Tensor, v: Tensor, sigma: float = 0.01) -> Tensor:
    """Evaluate norm move with the published finite-forward arithmetic.

    PR #23's central forward RMSNorm response along v.

    RMSNorm's Jacobian is symmetric: this finite directional response tends to its credit pullback. This is one
    nonlinear step, not an analytic VJP. The tiny floor also defines a finite response for a zero input or
    credit.
    """
    s = (
        sigma
        * (
            x.square().mean(-1, keepdim=True).clamp_min(1e-30) / v.square().mean(-1, keepdim=True).clamp_min(1e-30)
        ).sqrt()
    )
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch_operations.compile(dynamic=False)
def prepared_gate_input_response(
    x: Tensor,
    weight: Tensor,
    kappa: Tensor,
    rows: Tensor,
    projected_rows: Tensor,
    row_square: Tensor,
    sigma: float,
) -> Tensor:
    """Evaluate prepared gate input response with the published finite-forward arithmetic.

    Token chunks share the direction projections and squared lengths.
    """
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ rows.T) / x.shape[-1]
    direction_square = row_square.unsqueeze(0)
    base = functional.linear(x[:, : weight.shape[1]], weight).unsqueeze(1)
    direction = projected_rows.unsqueeze(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * direction_square + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * direction_square + EPS)
    zp = (base + steps.unsqueeze(-1) * direction) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * direction) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
    return decode_local_scores(response, rows) / len(rows)


def query_change(
    x: Tensor,
    delta: Tensor,
    pos: Tensor,
    rel: Tensor,
    w_q: Tensor,
    keys: Tensor,
    values: Tensor,
    w_proj_t: Tensor,
    cos: Tensor,
    sin: Tensor,
    window: int,
) -> Tensor:
    """Evaluate query change with the published finite-forward arithmetic.

    PR #19's finite own-query response; all keys and values stay at their clean values.

    Carry finite changes through both RMSNorms and softmax. The difference form preserves small probe responses
    without replacing them by a local derivative.
    """
    m, probes, _ = delta.shape
    heads, length, width = keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]

    def rotate(u: Tensor) -> Tensor:
        """Evaluate rotate with the published finite-forward arithmetic."""
        return torch.cat(
            [u[..., : width // 2] * c + u[..., width // 2 :] * s, u[..., : width // 2] * -s + u[..., width // 2 :] * c],
            -1,
        )

    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    q0, dq = r0 * inv0, dr * inv + r0 * (inv - inv0)
    lag = rel[:, None] - torch.arange(length, device=rel.device)[None]
    allowed = ((lag >= 0) & (lag <= window))[:, None, None]
    scores = torch.einsum('mphd,htd->mpht', q0.to(keys.dtype), keys).float() * width**-0.5
    p0 = torch.softmax(scores.masked_fill(~allowed, float('-inf')), -1)
    ds = torch.einsum('mphd,htd->mpht', dq.to(keys.dtype), keys).float() * width**-0.5
    dlse = torch.log1p((p0 * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mpht,htd->mphd', (p0 * torch.expm1(ds - dlse)).to(values.dtype), values).float()
    return dout.flatten(-2) @ w_proj_t


@cache
def finite_attention_kernels() -> tuple[_TritonKernel, _TritonKernel]:
    """Evaluate finite attention kernels with the published finite-forward arithmetic.

    Finite-response reductions. Neither kernel differentiates an attention operation.
    """
    tl = cast('_TritonLanguage', import_module('triton.language'))
    libdevice = cast('_TritonLibdevice', import_module('triton.language.extra.cuda.libdevice'))

    def routed(
        query_ptr: _TritonTensor,
        key_ptr: _TritonTensor,
        value_ptr: _TritonTensor,
        probability_ptr: _TritonTensor,
        rest_ptr: _TritonTensor,
        out_ptr: _TritonTensor,
        probe_count: int,
        head_count: int,
        key_count: int,
        d: int,
        bp: int,
        bk: int,
    ) -> None:
        """Evaluate the original fused Triton tile without changing its arithmetic."""
        token, head = tl.program_id(1) // head_count, tl.program_id(1) % head_count
        probes = tl.program_id(0) * bp + tl.arange(0, bp)
        keys, features = tl.arange(0, bk), tl.arange(0, d)
        q = tl.load(
            query_ptr + ((token * probe_count + probes[:, None]) * head_count + head) * d + features[None, :],
            probes[:, None] < probe_count,
            0,
        )
        offsets = ((token * head_count + head) * key_count + keys[:, None]) * d + features[None, :]
        k = tl.load(key_ptr + offsets, keys[:, None] < key_count, 0)
        v = tl.load(value_ptr + offsets, keys[:, None] < key_count, 0)
        p = tl.load(probability_ptr + (token * head_count + head) * key_count + keys, keys < key_count, 0)
        # Match both BF16 matmul output roundings of the original implementation.
        ds = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * (d**-0.5)
        dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
        weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(rest_ptr + (token * head_count + head) * d + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(
            out_ptr + ((token * probe_count + probes[:, None]) * head_count + head) * d + features[None, :],
            moved,
            probes[:, None] < probe_count,
        )

    def key_response(
        query_ptr: _TritonTensor,
        dk_ptr: _TritonTensor,
        attention_ptr: _TritonTensor,
        contrast_ptr: _TritonTensor,
        out_ptr: _TritonTensor,
        _h: int,
        t: int,
        m: int,
        p: int,
        d: int,
        qh: int,
        qt: int,
        qd: int,
        ah: int,
        at: int,
        am: int,
        ch: int,
        ct: int,
        cm: int,
        source_start: _TritonTensor,
        query_start: _TritonTensor,
        window: int,
        bc: int,
        bq: int,
    ) -> None:
        """Evaluate the original fused Triton tile without changing its arithmetic."""
        head = tl.program_id(1)
        col = tl.program_id(0) * bc + tl.arange(0, bc)
        source = col % m
        features = tl.arange(0, d)
        changed = tl.load(dk_ptr + (head * p * m + col[None, :]) * d + features[:, None], col[None, :] < p * m, 0)
        first = tl.maximum(0, source_start + tl.min(source, 0) - query_start)
        last = tl.minimum(t, source_start + tl.max(source, 0) + window + 1 - query_start)
        first = first // bq * bq
        result = tl.full((bc,), 0, tl.float32)
        for start in _triton_builtins.range(first, last, bq):
            query = start + tl.arange(0, bq)
            q = tl.load(query_ptr + head * qh + query[:, None] * qt + features[None, :] * qd, query[:, None] < t, 0)
            delta = tl.dot(q, changed, input_precision='tf32x3') * (d**-0.5)
            valid = (query[:, None] < t) & (col[None, :] < p * m)
            a = tl.load(attention_ptr + head * ah + query[:, None] * at + source[None, :] * am, valid, 0)
            c = tl.load(contrast_ptr + head * ch + query[:, None] * ct + source[None, :] * cm, valid, 0)
            change = a * libdevice.expm1(delta)
            result += tl.sum(change / (1 + change) * c, 0)
        tl.store(out_ptr + head * p * m + col, result, col < p * m)

    return _jit_kernel(routed, tl), _jit_kernel(key_response, tl)


@_custom_op('noise_record::routed_response', mutates_args=())
def routed_response(
    dq: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    rest: Tensor,
) -> Tensor:
    """Evaluate routed response with the published finite-forward arithmetic."""
    triton = _triton_runtime()
    m, probes, heads, width = dq.shape
    count = keys.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device, dtype=torch.float32)
    finite_attention_kernels()[0][(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(),
        keys.contiguous(),
        values.contiguous(),
        probabilities.contiguous(),
        rest.contiguous(),
        out,
        probes,
        heads,
        count,
        width,
        32,
        max(16, triton.next_power_of_2(count)),
        num_warps=4,
    )
    return out


@routed_response.register_fake
def _fake_routed_response(
    dq: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    rest: Tensor,
) -> Tensor:
    """Evaluate fake routed response with the published finite-forward arithmetic."""
    del keys, values, probabilities, rest
    return torch.empty(dq.shape, device=dq.device, dtype=torch.float32)


@_custom_op('noise_record::key_response', mutates_args=())
def fused_key_response(
    q: Tensor,
    dk: Tensor,
    a: Tensor,
    c: Tensor,
    source_start: int,
    query_start: int,
    window: int,
) -> Tensor:
    """Evaluate fused key response with the published finite-forward arithmetic."""
    triton = _triton_runtime()
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    finite_attention_kernels()[1][(triton.cdiv(probes * sources, 64), heads)](
        q,
        dk.contiguous(),
        a,
        c,
        out,
        heads,
        q.shape[1],
        sources,
        probes,
        width,
        *q.stride(),
        *a.stride(),
        *c.stride(),
        source_start,
        query_start,
        window,
        64,
        64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@fused_key_response.register_fake
def _fake_key_response(
    q: Tensor,
    dk: Tensor,
    a: Tensor,
    c: Tensor,
    source_start: int,
    query_start: int,
    window: int,
) -> Tensor:
    """Evaluate fake key response with the published finite-forward arithmetic."""
    del a, c, source_start, query_start, window
    return torch.empty(dk.shape[:3], device=q.device, dtype=torch.float32)


def query_change_routed(
    x: Tensor,
    delta: Tensor,
    pos: Tensor,
    w_q: Tensor,
    top_keys: Tensor,
    top_values: Tensor,
    top_p: Tensor,
    rest: Tensor,
    w_proj_t: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Evaluate query change routed with the published finite-forward arithmetic.

    Move only the selected key logits; retain the exact rescaling of the held output.
    """
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]

    def rotate(u: Tensor) -> Tensor:
        """Evaluate rotate with the published finite-forward arithmetic."""
        return torch.cat(
            [u[..., : width // 2] * c + u[..., width // 2 :] * s, u[..., : width // 2] * -s + u[..., width // 2 :] * c],
            -1,
        )

    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    if (
        dq.is_cuda
        and top_keys.dtype == torch.bfloat16
        and width in _ROUTED_WIDTHS
        and top_keys.shape[2] <= _ROUTED_KEY_LIMIT
    ):
        dout = routed_response(dq.to(top_keys.dtype), top_keys, top_values, top_p, rest)
        return dout.flatten(-2) @ w_proj_t
    ds = torch.einsum('mphd,mhkd->mphk', dq.to(top_keys.dtype), top_keys).float() * width**-0.5
    p = top_p.unsqueeze(1)
    dlse = torch.log1p((p * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mphk,mhkd->mphd', (p * torch.expm1(ds - dlse)).to(top_values.dtype), top_values).float()
    dout = dout + torch.expm1(-dlse) * rest.unsqueeze(1)
    return dout.flatten(-2) @ w_proj_t


def chain_chunk_eager(
    site: int,
    move: Tensor,
    scale: Tensor,
    e: Tensor,
    x0: Tensor,
    mids: list[Tensor],
    top_stream: Tensor,
    idx: list[Tensor],
    hs: list[Tensor],
    lam_r: Tensor,
    lam_0: Tensor,
    w_fc: list[Tensor],
    w_out_t: list[Tensor],
    weight_t: Tensor,
    base: Tensor,
    target_logit_w: Tensor,
    y: Tensor,
    own: OwnChain | None = None,
    *,
    from_output: bool = False,
    packed: bool = False,
) -> Tensor:
    """Evaluate chain chunk eager with the published finite-forward arithmetic.

    Forward differences of each token's loss along every move of one site, K/V held, MLPs restricted to each
    token's top-k units. site: -1 for the raw embedding, else the block whose mid stream moves. move (m, P, W);
    scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block (m, W); idx, hs per
    block (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block (H, W);
    weight_t (W, V); base (m, V) clean top-stream logits before the norm; target_logit_w (m, W). own optionally
    supplies PR #19's own-query responses in every entered attention. from_output starts at the block's MLP
    output instead: skip its MLP, then enter the next block through BOTH its residual scaling and its own-query
    response. Returns (m, P).
    """
    layer_count = len(mids)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site + int(from_output)
    for b in range(first, layer_count):
        if b > first or from_output:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        if own is not None and (site < 0 or b > first or from_output):
            if own[0] == 'routed':
                _, pos, inputs, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos_sin = own
                delta = delta + query_change_routed(
                    inputs[b],
                    delta,
                    pos,
                    w_q[b],
                    top_keys[b],
                    top_values[b],
                    top_p[b],
                    rest[b],
                    w_proj_t[b],
                    cos_sin[0],
                    cos_sin[1],
                )
            else:
                _, pos, rels, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
                delta = delta + query_change(
                    inputs[b], delta, pos, rels[b], w_q[b], keys[b], values[b], w_proj_t[b], cos, sin, windows[b]
                )
        dz = norm_difference(mids[b], delta)
        fc = w_fc[b] if packed else w_fc[b][idx[b]]
        wo = w_out_t[b] if packed else w_out_t[b][idx[b]]
        dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = functional.relu(h + dh), functional.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
    sq = top_stream.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', top_stream, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / top_stream.shape[-1] + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / top_stream.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15
    )
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


chain_chunk = torch_operations.compile(chain_chunk_eager, dynamic=False)
chain_chunk_own = torch_operations.compile(chain_chunk_eager, dynamic=True)
KEY_BUCKET = 256


@torch_operations.compile(dynamic=False)
def key_scores(
    qp: Tensor,
    kp: Tensor,
    k_raw: Tensor,
    steps: Tensor,
    cos: Tensor,
    sin: Tensor,
    a: Tensor,
    c: Tensor,
    basis: Tensor,
) -> Tensor:
    """Evaluate key scores with the published finite-forward arithmetic.

    Single-key experiments, each evaluated exactly: the raw key at position t moves by steps_t e_j and nothing
    else moves, so each query's attention changes in one term, o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'),
    a' = a exp(dl). Scored against each query's target (C[s, t] = hub_s . (v_t - o_s)): one local step, as the
    record's coordinate head scores single-logit moves. The processed key's change is formed exactly before the
    logit product. qp, kp, k_raw (h, T, d); steps (h, T, 1); cos, sin (T, d/2); A, C (h, T, T), A zero outside
    the window; basis (J, d). Returns (h, T, J).
    """
    d = k_raw.shape[-1]
    moved = k_raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    x1, x2 = moved[..., : d // 2], moved[..., d // 2 :]
    moved = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    dk = moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)
    dl = torch.einsum('hsd,hjtd->hjst', qp, dk) / d**0.5
    change = a[:, None] * torch.expm1(dl)
    return (change / (1 + change) * c[:, None]).sum(2).transpose(1, 2) / steps


@torch_operations.compile(dynamic=False)
def raw_key_moves(
    kp: Tensor,
    raw: Tensor,
    steps: Tensor,
    cos: Tensor,
    sin: Tensor,
    basis: Tensor,
) -> Tensor:
    """Evaluate raw key moves with the published finite-forward arithmetic."""
    moved = raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    half = raw.shape[-1] // 2
    moved = torch.cat(
        (moved[..., :half] * cos + moved[..., half:] * sin, -moved[..., :half] * sin + moved[..., half:] * cos), -1
    )
    return moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)


@torch_operations.compile(dynamic=False)
def prepared_key_moves(
    x: Tensor,
    steps: Tensor,
    rows: Tensor,
    projected_x: Tensor,
    projected_rows: Tensor,
    k_raw: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Evaluate prepared key moves with the published finite-forward arithmetic.

    Same central experiments, reusing the linear projections of x and each direction.

    N(x +/- s*h) W - N(x) W =     (inv_plusminus - inv0) * (x W) +/- s*inv_plusminus * (h W). This is a finite
    forward identity, including both RMSNorm responses. Positive and negative trials share one batched query/key
    product.
    """
    m, probes, width = x.shape[0], rows.shape[0], x.shape[-1]
    heads, _, dh = k_raw.shape
    square = x.square().sum(-1, keepdim=True)
    dots = x @ rows.T
    moved_square = square + steps.square() * rows.square().sum(-1)
    inv0 = torch.rsqrt(square / width + EPS)
    inv_up = torch.rsqrt((moved_square + 2 * steps * dots) / width + EPS)
    inv_down = torch.rsqrt((moved_square - 2 * steps * dots) / width + EPS)
    up = (inv_up - inv0).unsqueeze(-1) * projected_x.unsqueeze(1) + (steps * inv_up).unsqueeze(
        -1
    ) * projected_rows.unsqueeze(0)
    down = (inv_down - inv0).unsqueeze(-1) * projected_x.unsqueeze(1) - (steps * inv_down).unsqueeze(
        -1
    ) * projected_rows.unsqueeze(0)
    raw_change = torch.cat((up, down), dim=1).reshape(m, 2 * probes, heads, dh).permute(2, 1, 0, 3)

    def rotate(z: Tensor) -> Tensor:
        """Evaluate rotate with the published finite-forward arithmetic."""
        return torch.cat(
            [z[..., : dh // 2] * cos + z[..., dh // 2 :] * sin, -z[..., : dh // 2] * sin + z[..., dh // 2 :] * cos], -1
        )

    r0 = rotate(k_raw)
    dr = rotate(raw_change)
    dk = norm_difference(r0.reshape(heads * m, dh), dr.permute(0, 2, 1, 3).reshape(heads * m, 2 * probes, dh))
    return dk.reshape(heads, m, 2 * probes, dh).permute(0, 2, 1, 3)


@torch_operations.compile(dynamic=False)
def key_input_prepared_derivatives(
    x: Tensor,
    steps: Tensor,
    rows: Tensor,
    projected_x: Tensor,
    projected_rows: Tensor,
    k_raw: Tensor,
    cos: Tensor,
    sin: Tensor,
    qp: Tensor,
    a: Tensor,
    c: Tensor,
) -> Tensor:
    """Evaluate key input prepared derivatives with the published finite-forward arithmetic."""
    dk = prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin)
    probes, dh = rows.shape[0], k_raw.shape[-1]
    dl = torch.einsum('hsd,hpmd->hpsm', qp, dk) / dh**0.5
    change = a[:, None] * torch.expm1(dl)
    scores = (change / (1 + change) * c[:, None]).sum((0, 2)).T
    return (scores[:, :probes] - scores[:, probes:]) / (2 * steps)


@torch_operations.compile(dynamic=False)
def decode_key_pairs(scores: Tensor, steps: Tensor, rows: Tensor) -> Tensor:
    """Evaluate decode key pairs with the published finite-forward arithmetic."""
    return decode_local_response(scores.sum(0).T, steps, rows) / len(rows)


SCORE_COLUMNS = 32  # Smaller tiles avoid shared-memory/occupancy pressure on RTX 5090.


def norm_delta(x: Tensor, delta: Tensor) -> Tensor:
    """Evaluate norm delta with the published finite-forward arithmetic."""
    width = x.shape[-1]
    eps = torch.finfo(x.dtype).eps
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + eps)
    moved_inverse = torch.rsqrt(
        (square + 2 * (x * delta).sum(-1, keepdim=True) + delta.square().sum(-1, keepdim=True)) / width + eps
    )
    return delta * moved_inverse + x * (moved_inverse - inverse)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Evaluate rotate with the published finite-forward arithmetic."""
    half = x.shape[-1] // 2
    return torch.cat((x[..., :half] * cos + x[..., half:] * sin, -x[..., :half] * sin + x[..., half:] * cos), -1)


def receiver_changes(
    x: Tensor,
    move: Tensor,
    kraw: Tensor,
    gate: Tensor | None,
    ve: Tensor | None,
    wkv: Tensor,
    wg: Tensor | None,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate receiver changes with the published finite-forward arithmetic.

    Finite K and effective-V changes of each independent source token.
    """
    m, p, _width = move.shape
    heads, hd = kraw.shape[-2:]
    dn = norm_delta(x[:, None], move)
    projected = functional.linear(dn, wkv).reshape(m, p, 2, heads, hd)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(kraw[:, None], c, s)
    dr = rotate(projected[:, :, 0], c, s)
    dk = norm_delta(r0, dr)
    dv = projected[:, :, 1]
    if gate is not None:
        gate_weight = cast('Tensor', wg)
        value_embedding = cast('Tensor', ve)
        dg = functional.linear(dn[..., : gate_weight.shape[-1]], gate_weight)
        gate_change = 2 * ((gate[:, None] + dg).sigmoid() - gate[:, None].sigmoid())
        dv = dv + gate_change[..., None] * value_embedding[:, None]
    return dk.permute(2, 1, 0, 3).contiguous(), dv.permute(2, 1, 0, 3).contiguous()


def joint_scores(
    q: Tensor,
    target: Tensor,
    dk: Tensor,
    dv: Tensor,
    a: Tensor,
    contrast: Tensor,
    source_start: int,
    window: int,
) -> Tensor:
    """Evaluate joint scores with the published finite-forward arithmetic.

    Exact single-column softmax update; output shape H,P,M.
    """
    dh = q.shape[-1]
    ds = torch.einsum('hsd,hpmd->hpsm', q, dk) * dh**-0.5
    changed = a[:, None] * torch.expm1(ds)
    value_score = torch.einsum('hsd,hpmd->hpsm', target, dv)
    scores = (changed * contrast[:, None] + (a[:, None] + changed) * value_score) / (1 + changed)
    s = torch.arange(q.shape[1], device=q.device)[:, None]
    t = source_start + torch.arange(dk.shape[2], device=q.device)[None]
    allowed = (s > t) & (s - t <= window)
    return (scores * allowed[None, None]).sum(2)


@cache
def score_kernel() -> _TritonKernel:
    """Evaluate score kernel with the published finite-forward arithmetic."""
    tl = cast('_TritonLanguage', import_module('triton.language'))
    libdevice = cast('_TritonLibdevice', import_module('triton.language.extra.cuda.libdevice'))

    def kernel(
        query_ptr: _TritonTensor,
        target_ptr: _TritonTensor,
        dk_ptr: _TritonTensor,
        dv_ptr: _TritonTensor,
        attention_ptr: _TritonTensor,
        contrast_ptr: _TritonTensor,
        out_ptr: _TritonTensor,
        t: int,
        m: int,
        p: int,
        d: int,
        qh: int,
        qt: int,
        qd: int,
        gh: int,
        gt: int,
        gd: int,
        ah: int,
        at: int,
        am: int,
        ch: int,
        ct: int,
        cm: int,
        source_start: _TritonTensor,
        window: int,
        bc: int,
        bq: int,
    ) -> None:
        """Evaluate the original fused Triton tile without changing its arithmetic."""
        head = tl.program_id(1)
        col = tl.program_id(0) * bc + tl.arange(0, bc)
        source = col % m
        features = tl.arange(0, d)
        offsets = (head * p * m + col[None, :]) * d + features[:, None]
        dk = tl.load(dk_ptr + offsets, col[None, :] < p * m, 0)
        dv = tl.load(dv_ptr + offsets, col[None, :] < p * m, 0)
        first = (source_start + tl.min(source, 0)) // bq * bq
        last = tl.minimum(t, source_start + tl.max(source, 0) + window + 1)
        result = tl.full((bc,), 0, tl.float32)
        for start in _triton_builtins.range(first, last, bq):
            query = start + tl.arange(0, bq)
            q = tl.load(query_ptr + head * qh + query[:, None] * qt + features[None, :] * qd, query[:, None] < t, 0)
            g = tl.load(target_ptr + head * gh + query[:, None] * gt + features[None, :] * gd, query[:, None] < t, 0)
            ds = tl.dot(q, dk, input_precision='tf32x3') * (d**-0.5)
            vs = tl.dot(g, dv, input_precision='tf32x3')
            valid = (query[:, None] < t) & (col[None, :] < p * m)
            a = tl.load(attention_ptr + head * ah + query[:, None] * at + source[None, :] * am, valid, 0)
            c = tl.load(contrast_ptr + head * ch + query[:, None] * ct + source[None, :] * cm, valid, 0)
            changed = a * libdevice.expm1(ds)
            allowed = (query[:, None] > source_start + source[None, :]) & (
                query[:, None] - source_start - source[None, :] <= window
            )
            response = (changed * c + (a + changed) * vs) / (1 + changed)
            result += tl.sum(tl.where(valid & allowed, response, 0), 0)
        tl.store(out_ptr + head * p * m + col, result, col < p * m)

    return _jit_kernel(kernel, tl)


@_custom_op('noise_record::future_scores', mutates_args=())
def fused_scores(
    q: Tensor,
    target: Tensor,
    dk: Tensor,
    dv: Tensor,
    a: Tensor,
    contrast: Tensor,
    source_start: int,
    window: int,
) -> Tensor:
    """Evaluate fused scores with the published finite-forward arithmetic."""
    if not q.is_cuda:
        return joint_scores(q, target, dk, dv, a, contrast, source_start, window)
    triton = _triton_runtime()
    h, p, m, d = dk.shape
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    score_kernel()[(triton.cdiv(p * m, SCORE_COLUMNS), h)](
        q,
        target,
        dk.contiguous(),
        dv.contiguous(),
        a,
        contrast,
        out,
        q.shape[1],
        m,
        p,
        d,
        *q.stride(),
        *target.stride(),
        *a.stride(),
        *contrast.stride(),
        source_start,
        window,
        SCORE_COLUMNS,
        64,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@fused_scores.register_fake
def _fake_fused_scores(
    q: Tensor,
    target: Tensor,
    dk: Tensor,
    dv: Tensor,
    a: Tensor,
    contrast: Tensor,
    source_start: int,
    window: int,
) -> Tensor:
    """Evaluate fake fused scores with the published finite-forward arithmetic."""
    del target, dv, a, contrast, source_start, window
    return torch.empty(dk.shape[:-1], device=q.device, dtype=torch.float32 if q.is_cuda else q.dtype)


def signed_hidden_moves(
    raw: Tensor,
    directions: Tensor,
    step: Tensor,
    weight: Tensor,
) -> Tensor:
    """Evaluate signed hidden moves with the published finite-forward arithmetic.

    Exact paired squared-ReLU responses for sign directions, with one large GEMM.

    The even component is independent of each +/-1 direction. This identity includes threshold crossings; it is
    not a derivative approximation.
    """
    base = raw.relu()
    plus, minus = (raw + step).relu(), (raw - step).relu()
    dp, dm = (plus - base) * (plus + base), (minus - base) * (minus + base)
    odd, even = (dp - dm) * 0.5, (dp + dm) * 0.5
    moved = functional.linear(directions * odd[:, None], weight)
    center = functional.linear(even, weight)[:, None]
    return torch.cat((center + moved, center - moved), 1)


def future_embedding_moves(raw: Tensor, directions: Tensor, step: Tensor) -> Tensor:
    """Evaluate future embedding moves with the published finite-forward arithmetic."""
    delta = step[:, None] * directions
    return torch.cat((norm_delta(raw[:, None], delta), norm_delta(raw[:, None], -delta)), 1)


@cache
def query_kernel() -> _TritonKernel:
    """Evaluate query kernel with the published finite-forward arithmetic."""
    tl = cast('_TritonLanguage', import_module('triton.language'))
    libdevice = cast('_TritonLibdevice', import_module('triton.language.extra.cuda.libdevice'))

    def kernel(
        dq_ptr: _TritonTensor,
        key_ptr: _TritonTensor,
        value_ptr: _TritonTensor,
        attention_ptr: _TritonTensor,
        base_ptr: _TritonTensor,
        out_ptr: _TritonTensor,
        _m: int,
        p: int,
        h: int,
        d: int,
        t: int,
        kh: int,
        kt: int,
        kd: int,
        vh: int,
        vt: int,
        vd: int,
        ah: int,
        am: int,
        at: int,
        oh: int,
        om: int,
        od: int,
        bp: int,
        bk: int,
    ) -> None:
        """Evaluate the original fused Triton tile without changing its arithmetic."""
        token, head = tl.program_id(1) // h, tl.program_id(1) % h
        probe = tl.program_id(0) * bp + tl.arange(0, bp)
        feature = tl.arange(0, d)
        delta = tl.load(
            dq_ptr + ((token * p + probe[:, None]) * h + head) * d + feature[None, :], probe[:, None] < p, 0
        )
        numerator = tl.full((bp, d), 0, tl.float32)
        partition = tl.full((bp,), 0, tl.float32)
        for start in _triton_builtins.range(0, t, bk):
            key = start + tl.arange(0, bk)
            a = tl.load(attention_ptr + head * ah + token * am + key * at, key < t, 0)
            # Causal/window-masked tiles are exactly zero, not a top-k truncation.
            if tl.sum(a, 0) > 0:
                k = tl.load(key_ptr + head * kh + key[None, :] * kt + feature[:, None] * kd, key[None, :] < t, 0)
                v = tl.load(value_ptr + head * vh + key[:, None] * vt + feature[None, :] * vd, key[:, None] < t, 0)
                ds = tl.dot(delta, k, input_precision='tf32x3') * (d**-0.5)
                change = a[None, :] * libdevice.expm1(ds)
                partition += tl.sum(change, 1)
                numerator += tl.dot(change, v, input_precision='tf32x3')
        base = tl.load(base_ptr + head * oh + token * om + feature * od)
        response = (numerator - partition[:, None] * base[None, :]) / (1 + partition[:, None])
        tl.store(
            out_ptr + ((token * p + probe[:, None]) * h + head) * d + feature[None, :], response, probe[:, None] < p
        )

    return _jit_kernel(kernel, tl)


@_custom_op('noise_record::future_query_response', mutates_args=())
def query_response(
    dq: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    base: Tensor,
) -> Tensor:
    """Evaluate query response with the published finite-forward arithmetic.

    Stream the exact finite softmax response without a token/probe/key tensor.
    """
    if not dq.is_cuda:
        ds = torch.einsum('mphd,hsd->mphs', dq, keys) * dq.shape[-1] ** -0.5
        changed = probabilities.permute(1, 0, 2)[:, None] * torch.expm1(ds)
        partition = changed.sum(-1, keepdim=True)
        numerator = torch.einsum('mphs,hsd->mphd', changed, values)
        return (numerator - partition * base.transpose(0, 1)[:, None]) / (1 + partition)
    triton = _triton_runtime()
    m, p, h, d = dq.shape
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    query_kernel()[(triton.cdiv(p, 32), m * h)](
        dq.contiguous(),
        keys,
        values,
        probabilities,
        base,
        out,
        m,
        p,
        h,
        d,
        keys.shape[1],
        *keys.stride(),
        *values.stride(),
        *probabilities.stride(),
        *base.stride(),
        32,
        64,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@query_response.register_fake
def _fake_query_response(
    dq: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    base: Tensor,
) -> Tensor:
    """Evaluate fake query response with the published finite-forward arithmetic."""
    del keys, values, probabilities, base
    return torch.empty_like(dq, memory_format=torch.contiguous_format)


@cache
def joint_own_kernel() -> _TritonKernel:
    """Stream the exact joint Q/K/V response of each source token to its own move."""
    tl = cast('_TritonLanguage', import_module('triton.language'))
    libdevice = cast('_TritonLibdevice', import_module('triton.language.extra.cuda.libdevice'))

    def kernel(
        dq_ptr: _TritonTensor,
        dk_ptr: _TritonTensor,
        dv_ptr: _TritonTensor,
        query_ptr: _TritonTensor,
        key_ptr: _TritonTensor,
        value_ptr: _TritonTensor,
        attention_ptr: _TritonTensor,
        base_ptr: _TritonTensor,
        out_ptr: _TritonTensor,
        m: int,
        p: int,
        h: int,
        d: int,
        t: int,
        dqm: int,
        dqp: int,
        dqh: int,
        dqd: int,
        qh: int,
        qm: int,
        qd: int,
        kh: int,
        kt: int,
        kd: int,
        vh: int,
        vt: int,
        vd: int,
        ah: int,
        am: int,
        at: int,
        oh: int,
        om: int,
        od: int,
        source_start: _TritonTensor,
        bp: int,
        bk: int,
    ) -> None:
        """Accumulate probe/key tiles without materializing the full response logits."""
        token, head = tl.program_id(1) // h, tl.program_id(1) % h
        probe = tl.program_id(0) * bp + tl.arange(0, bp)
        feature = tl.arange(0, d)
        delta = tl.load(
            dq_ptr + token * dqm + probe[:, None] * dqp + head * dqh + feature[None, :] * dqd,
            probe[:, None] < p,
            0,
        )
        query = tl.load(query_ptr + head * qh + token * qm + feature * qd)
        # receiver_changes supplies contiguous [H,P,M,D], not [M,P,H,D].
        source_offset = ((head * p + probe[:, None]) * m + token) * d + feature[None, :]
        delta_key = tl.load(dk_ptr + source_offset, probe[:, None] < p, 0)
        # This self-key correction retains the joint dq*dk term.
        self_change = tl.sum((query[None, :] + delta) * delta_key, 1) * (d**-0.5)
        numerator = tl.full((bp, d), 0, tl.float32)
        changed_partition = tl.full((bp,), 0, tl.float32)
        partition = tl.full((bp,), 0, tl.float32)
        self_weight = tl.full((bp,), 0, tl.float32)
        for tile_start in _triton_builtins.range(0, t, bk):
            key = tile_start + tl.arange(0, bk)
            a = tl.load(attention_ptr + head * ah + token * am + key * at, key < t, 0)
            # The measured probabilities already encode causal/window support.
            if tl.sum(a, 0) > 0:
                k = tl.load(
                    key_ptr + head * kh + key[None, :] * kt + feature[:, None] * kd,
                    key[None, :] < t,
                    0,
                )
                v = tl.load(
                    value_ptr + head * vh + key[:, None] * vt + feature[None, :] * vd,
                    key[:, None] < t,
                    0,
                )
                is_self = (key >= source_start + token) & (key <= source_start + token)
                ds = tl.dot(delta, k, input_precision='tf32x3') * (d**-0.5)
                ds += tl.where(is_self[None, :], self_change[:, None], 0)
                # Zero-support logits must not introduce 0*inf into either sum.
                ds = tl.where(a[None, :] > 0, ds, 0)
                changed = a[None, :] * libdevice.expm1(ds)
                new_weight = a[None, :] * libdevice.exp(ds)
                changed_partition += tl.sum(changed, 1)
                partition += tl.sum(new_weight, 1)
                numerator += tl.dot(changed, v, input_precision='tf32x3')
                self_weight += tl.sum(tl.where(is_self[None, :], new_weight, 0), 1)
        base = tl.load(base_ptr + head * oh + token * om + feature * od)
        delta_value = tl.load(dv_ptr + source_offset, probe[:, None] < p, 0)
        response = (
            numerator - changed_partition[:, None] * base[None, :] + self_weight[:, None] * delta_value
        ) / partition[:, None]
        tl.store(
            out_ptr + ((token * p + probe[:, None]) * h + head) * d + feature[None, :],
            response,
            probe[:, None] < p,
        )

    return _jit_kernel(kernel, tl)


@_custom_op('noise_record::future_joint_own_response', mutates_args=())
def joint_own_response(
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    base: Tensor,
    start: int,
) -> Tensor:
    """Return the finite own-attention change, including joint self Q/K/V terms.

    dq/output are [M,P,H,D]; dk/dv are native contiguous [H,P,M,D]. Clean
    query/base are [H,M,D], keys/values [H,T,D], and probabilities [H,M,T].
    Only source start+m changes: ds=dq*K/sqrt(D), with (Q+dq)*dk/sqrt(D)
    added at self. For c=A*expm1(ds), w=A*exp(ds), the output change is
    (sum(c*V)-sum(c)*O+w_self*dv)/sum(w), with a positive direct denominator.
    """
    m, p, h, d = dq.shape
    if not dq.is_cuda:
        ds = torch.einsum('mphd,htd->mpht', dq, keys) * d**-0.5
        source = (start + torch.arange(m, device=dq.device))[:, None, None, None].expand(m, p, h, 1)
        self_change = ((query.transpose(0, 1)[:, None] + dq) * dk.permute(2, 1, 0, 3)).sum(-1) * d**-0.5
        ds.scatter_add_(-1, source, self_change[..., None])
        a = probabilities.permute(1, 0, 2)[:, None]
        ds = torch.where(a > 0, ds, 0)
        changed = a * torch.expm1(ds)
        new_weight = a * torch.exp(ds)
        numerator = torch.einsum('mpht,htd->mphd', changed, values)
        numerator = numerator - changed.sum(-1, keepdim=True) * base.transpose(0, 1)[:, None]
        numerator = numerator + new_weight.gather(-1, source) * dv.permute(2, 1, 0, 3)
        return numerator / new_weight.sum(-1, keepdim=True)
    triton = _triton_runtime()
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    joint_own_kernel()[(triton.cdiv(p, 32), m * h)](
        dq,
        dk,
        dv,
        query,
        keys,
        values,
        probabilities,
        base,
        out,
        m,
        p,
        h,
        d,
        keys.shape[1],
        *dq.stride(),
        *query.stride(),
        *keys.stride(),
        *values.stride(),
        *probabilities.stride(),
        *base.stride(),
        start,
        32,
        64,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@joint_own_response.register_fake
def _fake_joint_own_response(
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    base: Tensor,
    start: int,
) -> Tensor:
    """Expose the contiguous [M,P,H,D] output without running attention arithmetic."""
    del dk, dv, query, keys, values, probabilities, base, start
    return torch.empty_like(dq, memory_format=torch.contiguous_format)


def streamed_query_moves(
    raw: Tensor,
    directions: Tensor,
    step: Tensor,
    cos: Tensor,
    sin: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    weight: Tensor,
    base: Tensor,
) -> Tensor:
    """Evaluate streamed query moves with the published finite-forward arithmetic."""
    m, p, _width = directions.shape
    heads, _, hd = keys.shape
    delta = torch.cat((step[:, None] * directions, -step[:, None] * directions), 1)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = rotate(delta.reshape(m, 2 * p, heads, hd), c, s)
    response = query_response(norm_delta(r0, dr), keys, values, probabilities, base)
    return functional.linear(response.flatten(-2), weight)


def mlp_output_moves(
    mid: Tensor,
    move: Tensor,
    hidden: Tensor,
    w_fc: Tensor,
    w_out: Tensor,
) -> Tensor:
    """Evaluate mlp output moves with the published finite-forward arithmetic.

    Carry each signed Q trial through its own block's complete MLP.

    Anchor at the captured hidden pre-activation, exactly as receiver_changes anchors K/V. This uses finite
    forward responses, including ReLU crossings. The incoming attention/residual move remains part of the block
    output.
    """
    dh = functional.linear(norm_delta(mid[:, None], move), w_fc)
    base = hidden[:, None].relu()
    moved = (hidden[:, None] + dh).relu()
    return move + functional.linear((moved - base) * (moved + base), w_out)


compiled_hidden = torch_operations.compile(signed_hidden_moves, dynamic=False)
compiled_embedding = torch_operations.compile(future_embedding_moves, dynamic=False)
compiled_query = torch_operations.compile(streamed_query_moves, dynamic=False)


@torch_operations.compile(dynamic=True)
def receiver_scores(
    move: Tensor,
    coefficient: Tensor,
    x: Tensor,
    kraw: Tensor,
    gate: Tensor | None,
    ve: Tensor | None,
    wkv: Tensor,
    wg: Tensor | None,
    cos: Tensor,
    sin: Tensor,
    q: Tensor,
    target: Tensor,
    a: Tensor,
    contrast: Tensor,
    start: int,
    window: int,
) -> Tensor:
    """Evaluate receiver scores with the published finite-forward arithmetic."""
    dk, dv = receiver_changes(x, move * coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return fused_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T


@torch_operations.compile(dynamic=False)
def query_mlp_moves(
    raw: Tensor,
    directions: Tensor,
    step: Tensor,
    cos: Tensor,
    sin: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    weight: Tensor,
    base: Tensor,
    mid: Tensor,
    hidden: Tensor,
    w_fc: Tensor,
    w_out: Tensor,
) -> Tensor:
    """Evaluate query mlp moves with the published finite-forward arithmetic."""
    move = streamed_query_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight, base)
    return mlp_output_moves(mid, move, hidden, w_fc, w_out)


KeyContext = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
ChainCredits = dict[int | str | tuple[str, int], Tensor]


class MeasuredCredits(TypedDict):
    """Keep the raw measured targets distinct from corrected writer signals."""

    chain: ChainCredits
    g_top: Tensor
    targets: dict[int, Tensor]


class FutureContext(TypedDict):
    """The clean single-sequence receiver state used by finite source probes."""

    x: Tensor
    q: Tensor
    qraw: Tensor | None
    k: Tensor
    kraw: Tensor
    values: Tensor
    a: Tensor
    out: Tensor
    contrast: Tensor
    target: Tensor
    gate: Tensor | None
    ve: Tensor | None
    wg: Tensor | None
    wkv: Tensor
    window: int
    source_mlp: tuple[Tensor, Tensor, Tensor, Tensor] | None


def replay_block_moves(
    move: Tensor,
    x: Tensor,
    qraw: Tensor,
    wq: Tensor,
    dk: Tensor,
    dv: Tensor,
    cos: Tensor,
    sin: Tensor,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    probabilities: Tensor,
    base: Tensor,
    wproj: Tensor,
    mid: Tensor,
    hidden: Tensor,
    w_fc: Tensor,
    w_out: Tensor,
    start: int,
) -> Tensor:
    """Advance one changed source token through its joint attention and complete MLP."""
    tokens, probes, _width = move.shape
    heads, _, head_width = keys.shape
    normalized_move = norm_delta(x[:, None], move)
    raw_move = functional.linear(normalized_move, wq).reshape(tokens, probes, heads, head_width)
    c, s = cos[:, None, None], sin[:, None, None]
    clean = rotate(qraw.reshape(tokens, 1, heads, head_width), c, s)
    dq = norm_delta(clean, rotate(raw_move, c, s))
    attention_move = joint_own_response(dq, dk, dv, query, keys, values, probabilities, base, start)
    mid_move = move + functional.linear(attention_move.flatten(-2), wproj)
    return mlp_output_moves(mid, mid_move, hidden, w_fc, w_out)


@torch_operations.compile(dynamic=True)
def replay_receiver(
    move: Tensor,
    context: FutureContext,
    cos: Tensor,
    sin: Tensor,
    wq: Tensor,
    wproj: Tensor,
    start: int,
    *,
    advance: bool,
    relay: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Score raw future hubs and optionally advance only the source's own state."""
    sl = slice(start, start + move.shape[0])
    gate, ve = context['gate'], context['ve']
    dk, dv = receiver_changes(
        context['x'][sl],
        move,
        context['kraw'][sl],
        None if gate is None else gate[sl],
        None if ve is None else ve[sl],
        context['wkv'],
        context['wg'],
        cos,
        sin,
    )
    score = (
        fused_scores(
            context['q'],
            context['target'],
            dk,
            dv,
            context['a'][:, :, sl],
            context['contrast'][:, :, sl],
            start,
            context['window'],
        )
        .sum(0)
        .T
    )
    if relay is not None:
        score = score + relay(dk, dv)
    if advance:
        mid, hidden, w_fc, w_out = cast('tuple[Tensor, Tensor, Tensor, Tensor]', context['source_mlp'])
        qraw = cast('Tensor', context['qraw'])
        move = replay_block_moves(
            move,
            context['x'][sl],
            qraw[sl],
            wq,
            dk,
            dv,
            cos,
            sin,
            context['q'][:, sl],
            context['k'],
            context['values'],
            context['a'][:, sl],
            context['out'][:, sl],
            wproj,
            mid[sl],
            hidden[sl],
            w_fc,
            w_out,
            start,
        )
    return score, move


@torch_operations.compile(dynamic=True)
def relay_recipient_moves(
    dk: Tensor,
    dv: Tensor,
    query: Tensor,
    probabilities: Tensor,
    value_contrast: Tensor,
    wproj: Tensor,
    mid: Tensor,
    hidden: Tensor,
    w_fc: Tensor,
    w_out: Tensor,
) -> Tensor:
    """Replay a single changed memory column into selected recipients and their complete MLPs."""
    logits = torch.einsum('hrd,hpd->hpr', query, dk) * dk.shape[-1] ** -0.5
    changed = probabilities[:, None] * torch.expm1(logits)
    output = (
        changed[..., None] * value_contrast[:, None] + (probabilities[:, None] + changed)[..., None] * dv[:, :, None]
    ) / (1 + changed)[..., None]
    move = functional.linear(output.permute(2, 1, 0, 3).flatten(-2), wproj)
    return mlp_output_moves(mid, move, hidden, w_fc, w_out)


@runtime_checkable
class LinearEmbeddingCredit(Protocol):
    """Describe a linear embedding map's transpose, never a nonlinear backward."""

    def linear_embedding_credit(self, credit: Tensor) -> Tensor:
        """Map measured output credit to the original table lookup positions."""
        ...


def linear_embedding_credit(module: nn.Embedding, credit: Tensor) -> Tensor:
    """Apply an optional linear transpose exactly once before table reconstruction.

    Loss responses are measured by the estimator's forward probes. This hook
    only transports those measurements through a linear embedding map; it does
    not differentiate a nonlinear stage or create a new nonlinear probe target.
    """
    return module.linear_embedding_credit(credit) if isinstance(module, LinearEmbeddingCredit) else credit


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    x: Tensor
    y: Tensor
    x0: Tensor
    batch: int
    length: int
    clean_loss: Tensor

    def __init__(self, model: GPT, seed: int = 0, rank: int = 0, world: int = 1) -> None:
        """Install capture hooks and initialize the independent source RNG streams."""
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames: dict[str, tuple[Tensor, Tensor]] = {}
        self.modules = {
            n: m for n, m in cast('NamedModules', model).named_modules() if isinstance(m, (nn.Linear, nn.Embedding))
        }
        self.inputs: dict[str, Tensor] = {}
        self.outputs: dict[str, Tensor] = {}
        self.blocks: dict[int, Tensor] = {}
        self.capturing = False
        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(layer)))

    def output_hook(self, name: str) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        """Build the activation hook for one linear or embedding module."""

        def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output

        return hook

    def block_hook(self, layer: int) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        """Build the clean residual-input hook for one transformer block."""

        def hook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
            if self.capturing:
                self.blocks[layer] = args[0]

        return hook

    def close(self) -> None:
        """Remove the hooks installed by this capture."""
        for hook in self.hooks:
            hook.remove()

    def signs(self, *shape: int, shared: bool = False) -> Tensor:
        """Draw signed coordinates from the selected unchanged generator."""
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(
        self, site: str, draws: int, chunk: int, n: int, b: int, t: int, d: int, dtype: torch.dtype | None = None
    ) -> Tensor:
        """Draw a slice of the sign-flipped Hadamard cycles shared across ranks.

        Chunks and GPUs use disjoint rows within each cycle.
        """
        if site not in self.frames:
            cycles = -(-self.world * draws // d)
            self.frames[site] = (
                self.signs(cycles, d, shared=True),
                torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator),
            )
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x: Tensor, y: Tensor) -> None:
        """Cache one clean forward pass and its normalized embedding stream."""
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
                self.clean_loss = cast('Tensor', self.model(x, y, loss_reduction='none')).view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma: float = 0.1) -> Tensor:
        """Measure coordinate readout differences at the final logits."""
        raw = self.outputs['lm_head'][..., : self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return functional.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site: str, error: Tensor) -> Tensor:
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            error = linear_embedding_credit(module, error)
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)


class Boost:
    """Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it."""

    def __init__(
        self,
        model: GPT,
        settings: Settings,
        seed: int,
        rank: int,
        world: int,
        split: int = 1,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Configure sequence-group partitioning and the accepted probe RNGs."""
        if model.config.n_head != model.config.n_kv_head or model.config.n_layer % 2:
            message = 'Stage estimators require equal Q/KV heads and an even number of blocks'
            raise ValueError(message)
        if any(k % split for k in (model.config.n_embd, settings.query_draws)):
            message = 'Readout and query cycles must divide across the sequence group'
            raise ValueError(message)
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.shared_generator = self.capture.shared_generator
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)
        self.maps: dict[int, Tensor] = {}

    def close(self) -> None:
        """Release capture hooks without modifying model state."""
        self.capture.close()

    def share(self, total: int) -> tuple[int, int]:
        """Return this sequence-group rank's contiguous cycle interval."""
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor: Tensor) -> Tensor:
        """Sum a partitioned estimate over its sequence group in place."""
        if self.split > 1:
            torch_operations.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def signs(self, d: int) -> Tensor:
        """Draw one independent signed basis column pattern."""
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self) -> Tensor:
        """Construct the next randomized full-width Hadamard cycle."""
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def probe(self, x: Tensor, y: Tensor, mix: Tensor | None = None, probed: Tensor | None = None) -> Tensor:
        """Measure each token's readout credit along a full cycle in the probed space.

        GPUs sharing a sequence take disjoint slices and add their contributions.
        """
        weight = self.model.lm_head.weight.float()[: self.model.config.vocab_size]
        basis = self.basis()
        stream_dirs = basis.T if mix is None else mix.float() @ basis.T
        contrast = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], contrast.shape[1]
        torch.randint(0, self.width, (n, 1), device=self.device, generator=self.generator)  # each token's
        # first row (used only by partial cycles; drawn either way so the random stream is the same)
        credit = torch.zeros(n, self.width, device=self.device)
        lo, hi = self.share(self.width)
        contrast, stream_dirs, dir_sq, basis_slice = (
            contrast[lo:hi].contiguous(),
            stream_dirs[:, lo:hi],
            dir_sq[lo:hi],
            basis[lo:hi],
        )
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0 : t0 + per], y[t0 : t0 + per]
            derivative = cycle_derivatives(
                xs @ weight.T,
                contrast,
                ys,
                steps[t0 : t0 + per],
                xs.square().sum(-1, keepdim=True),
                xs @ stream_dirs,
                dir_sq,
                self.width,
            )
            credit[t0 : t0 + per] = derivative @ basis_slice / self.width
        return self.group_sum(credit)

    def embedding_block0(self, e: Tensor, y: Tensor, x: Tensor) -> Tensor:
        """Measure embedding credit through each token's own first-stage path.

        Each query sees the other positions' clean keys and values and its own moved ones.
        """
        model = self.model
        cfg = model.config
        attn = model.transformer.h[0].attn
        b, t = x.shape
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        window = model.window_sizes[0][0]
        mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
        weight = model.lm_head.weight.float()[: cfg.vocab_size]
        weights = {
            'q': attn.c_q.weight.float(),
            'k': attn.c_k.weight.float(),
            'v': attn.c_v.weight.float(),
            'proj': attn.c_proj.weight.float(),
        }
        e = e.float().view(b, t, self.width)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(functional.linear(x0, weights['k']).view(b, t, h, hd), cos, sin))
        v_clean = functional.linear(x0, weights['v']).view(b, t, h, hd)
        basis = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, self.width, device=self.device)
        n = chunk_size(self.width // self.split, self.s.emb_chunk)
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(functional.linear(x0, weights['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + functional.linear(out0.reshape(b, t, self.width).float(), weights['proj'])
        clean = readout_losses(xmid0.reshape(-1, self.width), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(self.width)
        for j0 in range(lo, hi, n):
            rows = basis[j0 : j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, weights['q'], weights['k'], weights['v'], cos, sin)

            def rep(z: Tensor) -> Tensor:
                return z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)

            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, weights['proj'], weight, yf).view(n, b, t)
            derivative[..., j0 : j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ basis / self.width).reshape(b * t, self.width)

    def attention_maps(self, i: int) -> Tensor:
        """Clean attention A[b, h, query, source]; no matched-filter matrix is needed."""
        if i in self.maps:
            return self.maps[i]
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (
            norm(apply_rotary_emb(d.outputs[prefix + name].float().view(b, t, h, hd), cos, sin))
            for name in ('c_q', 'c_k')
        )
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        window = model.window_sizes[i][0]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        probabilities = logits.softmax(-1)
        del logits
        self.maps[i] = probabilities.to(torch.bfloat16)
        return self.maps[i]

    def query_errors(
        self, draws: int, targets: dict[int, Tensor], layers: Iterable[int] | None = None
    ) -> dict[str, Tensor]:
        """Score a complete per-head query cycle against each query's own output target."""
        d, model = self.capture, self.model
        cfg = model.config
        layer_count, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        n = chunk = chunk_size(draws, self.s.nrep)

        def processed(raw: Tensor) -> Tensor:
            return norm(apply_rotary_emb(raw, cos, sin))

        dt = d.dtype
        errors: dict[str, Tensor] = {}
        for i in range(layer_count) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, layer_count) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, layer_count) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(cast('Tensor', z))).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = (
                None if self.s.query_central else attention(processed(q_raw).to(dt), kp.to(dt), values.to(dt), window)
            )
            fixed, vals = kp.to(dt), values.to(dt)
            raw = q_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = query_probe_heads(raw, a, sigma, cos, sin)
                out = query_probe_attention(moved, fixed, vals, window, n, b)
                if self.s.query_central:
                    minus = query_probe_attention(
                        query_probe_heads(raw, a, -sigma, cos, sin), fixed, vals, window, n, b
                    )
                    score = head_scores(out, minus, target)
                else:
                    score = head_scores(out, cast('Tensor', baseline), target)
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma * (2 if self.s.query_central else 1))).reshape(b, t, cfg.n_embd)
        return errors

    def key_context(self, i: int, sequence: int, target: Tensor) -> KeyContext:
        """Clean FP32 single-key score context, shared by raw-K and input-K experiments."""
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()

        def rot(z: Tensor) -> Tensor:
            return torch.cat(
                [z[..., : hd // 2] * cos + z[..., hd // 2 :] * sin, -z[..., : hd // 2] * sin + z[..., hd // 2 :] * cos],
                -1,
            )

        def unit(z: Tensor) -> Tensor:
            return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)

        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
        prefix = f'transformer.h.{i}.attn.'

        def heads(name: str) -> Tensor:
            return d.outputs[prefix + name][sequence].float().view(t, h, hd).transpose(0, 1)

        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][sequence].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        logits = (qp @ kp.transpose(1, 2)) / hd**0.5
        probabilities = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
        hub = target[sequence].view(t, h, hd).transpose(0, 1)
        contrast = (hub @ values.transpose(1, 2) - (hub * (probabilities @ values)).sum(-1, keepdim=True)) * allowed
        return qp, kp, k_raw, cos, sin, probabilities, contrast

    def exact_key_errors(self, targets: dict[int, Tensor], layers: Iterable[int] | None = None) -> dict[str, Tensor]:
        """Measure single-key finite responses without interference between positions.

        GPUs sharing a sequence split the directions; one-sided finite-step error remains.
        """
        d, model = self.capture, self.model
        cfg = model.config
        layer_count, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors: dict[str, Tensor] = {}
        for i in range(layer_count) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                credit_all[r] = self.key_errors_from_context(i, self.key_context(i, r, targets[i]))
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def key_errors_from_context(self, layer: int, context: KeyContext) -> Tensor:
        """Raw-K finite experiments, sharing the input-K context and skipping masked query blocks."""
        qp, kp, raw, cos, sin, probabilities, contrast = context
        _, t, hd = raw.shape
        lo, hi = self.share(hd)
        result = torch.zeros_like(raw)
        eye = torch.eye(hd, device=self.device)
        steps = self.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        for t0 in range(0, t, KEY_BUCKET):
            t1 = min(t0 + KEY_BUCKET, t)
            q1 = min(t, t1 + self.model.window_sizes[layer][0])
            source, query = slice(t0, t1), slice(t0, q1)
            if raw.is_cuda and hd in (32, 64):
                moves = raw_key_moves(
                    kp[:, source], raw[:, source], steps[:, source], cos[source], sin[source], eye[lo:hi]
                )
                scored = fused_key_response(
                    qp[:, query],
                    moves,
                    probabilities[:, query, source],
                    contrast[:, query, source],
                    t0,
                    t0,
                    self.model.window_sizes[layer][0],
                )
                result[:, source, lo:hi] = scored.transpose(1, 2) / steps[:, source]
                continue
            for j0 in range(lo, hi, self.s.key_chunk):
                j1 = min(j0 + self.s.key_chunk, hi)
                result[:, source, j0:j1] = key_scores(
                    qp[:, query],
                    kp[:, source],
                    raw[:, source],
                    steps[:, source],
                    cos[source],
                    sin[source],
                    probabilities[:, query, source],
                    contrast[:, query, source],
                    eye[j0:j1],
                )
        return result

    def value_credit(self, i: int, target: Tensor) -> Tensor:
        """Fixed-attention value credit; all query tokens contribute, with no discount."""
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        probabilities = self.attention_maps(i)  # A[b, h, query, source]
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return (probabilities.float().transpose(-1, -2) @ hub).transpose(1, 2)

    def exact_values(
        self, i: int, target: Tensor, value_target: Tensor | None = None, raw_credit: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Measure local value and gate responses with clean attention probabilities.

        A value move at t reaches query s by A[s, t], giving summed value credit.
        The gate uses the record's central forward sigmoid difference.
        """
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

    def hidden_credit(self, u: Tensor, g: Tensor, out_weight: Tensor) -> Tensor:
        """All hidden units' local central differences, partitioned across a sequence's GPUs."""
        # Preserve full-state checkpoint RNG advancement from the retired hidden cycle.
        self.signs(256)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        lo, hi = self.share(u.shape[-1])
        local = hidden_local_credit(u[:, lo:hi], steps, g, out_weight[:, lo:hi].float())
        return local if self.split == 1 else functional.pad(local, (lo, u.shape[-1] - hi))

    def stage_losses(
        self, i: int, lam_r: Tensor, lam_0: Tensor, x_prev: Tensor, x0: Tensor, ve: Tensor | None, y: Tensor
    ) -> Tensor:
        """Evaluate one stage with independently moved linear mixing scalars."""
        model, block = self.model, self.model.transformer.h[i]
        length = x_prev.shape[1]
        cos_sin = (model.cos[:, :length], model.sin[:, :length])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = (
            xin
            + block.attn.forward(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        )
        weight = model.lm_head.weight.float()[: model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def internals(
        self,
        gradients: dict[str, Tensor],
        targets: dict[int, Tensor],
        b: int,
        t: int,
        value_targets: dict[int, Tensor] | None = None,
        key_errors: dict[str, Tensor] | None = None,
        raw_values: dict[int, Tensor] | None = None,
    ) -> None:
        """Q/K/gate use measured targets; only linear V/VE paths accept corrected targets."""
        d, s = self.capture, self.s
        errors = self.query_errors(s.query_draws // self.split, targets)
        errors = {site: error / self.split for site, error in errors.items()}
        errors.update(self.exact_key_errors(targets) if key_errors is None else key_errors)
        for i, target in targets.items():
            value_target = None if value_targets is None else value_targets[i]
            raw = None if raw_values is None else raw_values[i]
            errors.update(
                {site: error / self.split for site, error in self.exact_values(i, target, value_target, raw).items()}
            )
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Assemble stagewise measured finite-response parameter credits."""
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        d.capture(x, y)
        layers = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients: dict[str, Tensor] = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        weight = model.lm_head.weight.float()[: model.config.vocab_size]
        x0 = d.x0
        targets, streams = self._boost_stage_credits(gradients, weight, yf, b, t)
        self.internals(gradients, targets, b, t)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, self.width))
        scalars = torch.zeros(2, layers, device=self.device)
        step = s.scalar_step
        for i in range(self.part, layers, self.split):
            x_prev = x0 if i == 0 else streams[i - 1][2]
            ve = cast('Tensor', model.value_embeds[str(i)](x)) if str(i) in model.value_embeds else None
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

    def _boost_stage_credits(
        self, gradients: dict[str, Tensor], weight: Tensor, yf: Tensor, b: int, t: int
    ) -> tuple[dict[int, Tensor], list[tuple[Tensor, Tensor, Tensor]]]:
        """Measure each stage against its own readout while preserving the shared RNG sequence."""
        d, model = self.capture, self.model
        layers = model.config.n_layer
        targets: dict[int, Tensor] = {}
        streams: list[tuple[Tensor, Tensor, Tensor]] = []
        for i in range(layers):
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
            if i < layers - 1:
                for stream in (xm, xo):
                    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + EPS)
                    error = coordinate_differences(normed @ weight.T, yf, 0.1)
                    gradients['lm_head.weight'][: weight.shape[0]] += error.T @ normed
        return targets, streams

    def step(self, x: Tensor, y: Tensor, optimizer: Optimizers) -> float:
        """Normalize and reduce credits before applying one optimizer update."""
        gradients = self.estimate(x, y)
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        sequences = self.world // self.split
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if self.world > 1:
            torch_operations.distributed.all_reduce(flat)
        flat /= sequences
        if not bool(torch.isfinite(flat).all()):
            message = 'Non-finite estimate'
            raise FloatingPointError(message)
        offset = 0
        for _, p in params:
            p.grad = flat[offset : offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        self.maps.clear()
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()


class Cut(Boost):
    """Own-query forward loss probes, optionally augmented by PR #23's value path."""

    _input_key_errors: dict[str, Tensor] | None = None
    measured: MeasuredCredits | None = None

    def __init__(
        self,
        model: GPT,
        settings: Settings,
        seed: int,
        rank: int,
        world: int,
        split: int = 1,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Share each sequence's cycles and draws across its probe ranks."""
        super().__init__(model, settings, seed, rank, world, split=split, group=group)

    @overload
    def value_path_credit(
        self, chain: ChainCredits, *, return_inputs: Literal[False] = False, raw_values: dict[int, Tensor] | None = None
    ) -> dict[int, Tensor]: ...

    @overload
    def value_path_credit(
        self, chain: ChainCredits, *, return_inputs: Literal[True], raw_values: dict[int, Tensor] | None = None
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]: ...

    def value_path_credit(
        self, chain: ChainCredits, *, return_inputs: bool = False, raw_values: dict[int, Tensor] | None = None
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Independent measured mid credits -> V -> one RMSNorm step -> residual identities.

        Never feed accumulated extra credit into another norm or local nonlinear
        experiment. Reuse this capture's attention maps later for V/gate credit.
        """
        model, d = self.model, self.capture
        layers = model.config.n_layer
        extra = {layers - 1: torch.zeros_like(chain[0])}
        inputs: dict[int, Tensor] = {}
        for i in reversed(range(0 if return_inputs else 1, layers)):
            attn = model.transformer.h[i].attn
            ev = (
                self.value_credit(i, chain[i] @ attn.c_proj.weight.float()) if raw_values is None else raw_values[i]
            ).reshape(-1, self.width)
            vp = norm_move(d.blocks[i].float().flatten(0, 1), ev @ attn.c_v.weight.float())
            inputs[i] = vp
            if i:
                extra[i - 1] = model.resid_lambdas[i].float() * (vp + extra[i])
        return (extra, inputs) if return_inputs else extra

    def key_input_credits(self, targets: dict[int, Tensor]) -> dict[int, Tensor]:
        """Independent central probes at each block input, moving only its K branch.

        Use a private copy of the saved probe RNG so these optional draws neither
        shift Q/hidden probes nor require another checkpoint RNG stream.
        Partial Hadamard cycles use randomized rows and column signs.
        """
        d, s = self.capture, self.s
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        input_credit: dict[int, Tensor] = {}
        raw_errors: dict[str, Tensor] = {}
        for i, target in targets.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)[: s.key_path_draws]
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            rows = frame[indices] * signs
            total_rows = len(rows)
            r0, r1 = self.share(total_rows)
            rows = rows[r0:r1]
            estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
            raw_credit = torch.empty_like(estimate)
            w_key = self.model.transformer.h[i].attn.c_k.weight.float()
            projected_rows = rows @ w_key.T
            for r in range(d.batch):
                context = self.key_context(i, r, target)
                qp, _, raw, cos, sin, probabilities, contrast = context
                raw_credit[r] = self.key_errors_from_context(i, context).transpose(0, 1).reshape(d.length, self.width)
                xin = d.blocks[i][r].float()
                projected_x = xin @ w_key.T
                steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                for t0 in range(0, d.length, s.key_path_tokens):
                    t1 = min(t0 + s.key_path_tokens, d.length)
                    sl = slice(t0, t1)
                    # Queries outside this union of source windows have exactly
                    # zero response. Round the retained range to kernel buckets.
                    q1 = min(d.length, t1 + self.model.window_sizes[i][0])
                    qlength = min(d.length, -(-(q1 - t0) // KEY_BUCKET) * KEY_BUCKET)
                    qs = slice(max(0, q1 - qlength), q1)
                    if xin.is_cuda and raw.shape[-1] in (32, 64):
                        # All directions share a source tile. The fused scorer
                        # reduces query tiles without writing H*P*Q*M intermediates.
                        moves = prepared_key_moves(
                            xin[sl], steps[sl], rows, projected_x[sl], projected_rows, raw[:, sl], cos[sl], sin[sl]
                        )
                        scores = fused_key_response(
                            qp[:, qs],
                            moves,
                            probabilities[:, qs, sl],
                            contrast[:, qs, sl],
                            t0,
                            qs.start,
                            self.model.window_sizes[i][0],
                        )
                        estimate[r, sl] = decode_key_pairs(scores, steps[sl], rows) * (len(rows) / total_rows)
                        continue
                    derivatives = torch.empty(len(xin[sl]), len(rows), device=self.device)
                    for j0 in range(0, len(rows), s.key_chunk):
                        js = slice(j0, min(j0 + s.key_chunk, len(rows)))
                        derivatives[:, js] = key_input_prepared_derivatives(
                            xin[sl],
                            steps[sl],
                            rows[js],
                            projected_x[sl],
                            projected_rows[js],
                            raw[:, sl],
                            cos[sl],
                            sin[sl],
                            qp[:, qs],
                            probabilities[:, qs, sl],
                            contrast[:, qs, sl],
                        )
                    estimate[r, sl] = derivatives @ rows / total_rows
            input_credit[i] = self.group_sum(estimate.flatten(0, 1))
            raw_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
        self._input_key_errors = raw_errors
        return input_credit

    def path_credits(
        self, chain: ChainCredits, targets: dict[int, Tensor], raw_values: dict[int, Tensor] | None = None
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Sum independent input responses, then transport only along residual identities."""
        layers = self.model.config.n_layer
        inputs = {i: torch.zeros_like(chain[i]) for i in range(layers)}
        if self.s.value_path:
            _, inputs = self.value_path_credit(chain, return_inputs=True, raw_values=raw_values)
        if self.s.key_path_draws:
            keys = self.key_input_credits(targets)
            inputs = {i: inputs[i] + keys[i] for i in range(layers)}
        if self.s.gate_path:
            for i, credit in self.gate_input_credits(targets, raw_values).items():
                inputs[i] = inputs[i] + credit
        extra = {layers - 1: torch.zeros_like(chain[0])}
        for i in reversed(range(1, layers)):
            extra[i - 1] = self.model.resid_lambdas[i].float() * (inputs[i] + extra[i])
        return extra, inputs

    def gate_input_credits(
        self, targets: dict[int, Tensor], raw_values: dict[int, Tensor] | None = None
    ) -> dict[int, Tensor]:
        """Independent gate-input experiments from raw hubs; only linear consumers follow."""
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        gate_credit: dict[int, Tensor] = {}
        for i, target in targets.items():
            # Advance for every layer to retain the tested private pilot bases.
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            if not has_ve(i, self.model.config.n_layer):
                continue
            rows = frame[indices] * signs
            value = self.value_credit(i, target) if raw_values is None else raw_values[i]
            value = value.reshape(-1, heads, self.width // heads)
            ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
            kappa = (value * ve).sum(-1)
            xin = d.blocks[i].float().flatten(0, 1)
            weight = cast('nn.Linear', self.model.transformer.h[i].attn.ve_gate).weight.float()
            projected_rows = functional.linear(rows[:, : weight.shape[1]], weight)
            row_square = rows.square().mean(-1)
            result = torch.empty_like(xin)
            for t0 in range(0, len(xin), s.chain_tokens):
                sl = slice(t0, t0 + s.chain_tokens)
                result[sl] = prepared_gate_input_response(
                    xin[sl], weight, kappa[sl], rows, projected_rows, row_square, s.gate_path_sigma
                )
            gate_credit[i] = result
        return gate_credit

    @staticmethod
    def path_scalar_credit(
        inputs: dict[int, Tensor], extra: dict[int, Tensor], x0: Tensor, outs: list[Tensor]
    ) -> Tensor:
        """Linear mixing coefficients see input responses plus later identity-path credit."""
        scalar_terms: list[Tensor] = []
        for i in range(len(outs)):
            credit = inputs[i] + extra[i]
            previous = x0 if i == 0 else outs[i - 1]
            scalar_terms.append(torch.stack([(credit * previous).sum(), (credit * x0).sum()]))
        return torch.stack(scalar_terms, dim=1)

    def chain_credits(
        self,
        e: Tensor,
        x0: Tensor,
        mids: list[Tensor],
        outs: list[Tensor],
        routes: list[tuple[Tensor, Tensor]],
        y: Tensor,
    ) -> ChainCredits:
        """Measure embedding and mid-stream credits from loss.

        Each root uses a full orthogonal cycle. Mixing scalars retain their
        central mid-stream probes. MLP outputs reuse the next mid credit;
        the final output uses the existing head probe.
        """
        model, s = self.model, self.s
        vocab = model.config.vocab_size
        weight = model.lm_head.weight.float()[:vocab]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        top_stream = outs[-1]
        base = top_stream @ weight.T
        target_w = weight[y.clamp(min=0)]
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        # Draw bases in the original site order. Replay one sequence at a time
        # so routed K/V storage does not grow with the per-GPU batch size.
        probes = self._chain_probe_frames(e, mids)
        count, scalars_here = self.width // self.split, self.part == 0
        length = self.capture.length if s.own_query else n
        for start in range(0, n, length):
            own: OwnQuery | OwnRouted | None = None
            if s.own_query:
                own = self.own_query_inputs(start // length)
                if s.query_keys:
                    own = self.strong_keys(own, s.query_keys)
            own_sl: OwnChain | None = None
            for t0 in range(start, start + length, m):
                t1 = min(t0 + m, start + length)
                sl = slice(t0, t1)
                own_sl = self._chain_context_slice(own, t0 - start, t1 - start)
                kernel = chain_chunk_own if own_sl is not None and own_sl[0] == 'window' else chain_chunk
                # All roots use the same selected units. Gather once per token block,
                # not again for each of the 44 layer visits of the nine roots.
                selected_fc = [w[r[0][sl]] for w, r in zip(w_fc, routes, strict=False)]
                selected_out = [w[r[0][sl]] for w, r in zip(w_out_t, routes, strict=False)]
                for site, basis, steps, credit, lam in probes:
                    move = steps[sl].unsqueeze(-1) * basis.unsqueeze(0)
                    scale = steps[sl].expand(-1, count)
                    if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat(
                            [move, h * prev, h * x0[sl].unsqueeze(1), -h * prev, -h * x0[sl].unsqueeze(1)], 1
                        )
                        scale = torch.cat(
                            [scale, torch.full_like(scale[:, :2], h), torch.full_like(scale[:, :2], -h)], 1
                        )
                    derivative = kernel(
                        site,
                        move,
                        scale,
                        e[sl],
                        x0[sl],
                        [z[sl] for z in mids],
                        top_stream[sl],
                        [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes],
                        lam_r,
                        lam_0,
                        selected_fc,
                        selected_out,
                        weight_t,
                        base[sl],
                        target_w[sl],
                        y[sl],
                        own_sl,
                        packed=True,
                    )
                    credit[sl] = derivative[:, :count] @ basis / self.width
                    if site >= 0 and scalars_here:
                        lam.add_((derivative[:, count : count + 2] + derivative[:, count + 2 :]).sum(0) / 2)
                del selected_fc, selected_out
            # Drop views as well as the owning cache before constructing the
            # next sequence's routed keys and values.
            del own_sl, own
        measured_credit: ChainCredits = {}
        for site, _, _, credit, lam in probes:
            measured_credit['emb' if site < 0 else site] = self.group_sum(credit)
            if site >= 0:
                measured_credit['lam', site] = self.group_sum(lam)
        return measured_credit

    def _chain_probe_frames(self, e: Tensor, mids: list[Tensor]) -> list[tuple[int, Tensor, Tensor, Tensor, Tensor]]:
        """Draw all root bases before replay, retaining the source-site RNG chronology."""
        probes: list[tuple[int, Tensor, Tensor, Tensor, Tensor]] = []
        lo, hi = self.share(self.width)
        for site in range(-1, self.model.config.n_layer):
            basis = self.basis()[lo:hi]
            probed = e if site < 0 else mids[site]
            steps = self.s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            credit = torch.empty(e.shape[0], self.width, device=self.device)
            lam = torch.zeros(2, device=self.device)
            probes.append((site, basis, steps, credit, lam))
        return probes

    def _chain_context_slice(self, own: OwnQuery | OwnRouted | None, begin: int, end: int) -> OwnChain | None:
        """Slice routed roots or construct the exact full-context window for one replay tile."""
        if own is None:
            return None
        if isinstance(own[0], str):
            routed = cast('OwnRouted', own)
            local = slice(begin, end)
            return (
                'routed',
                routed[1][local],
                [z[local] for z in routed[2]],
                routed[3],
                [z[local] for z in routed[4]],
                [z[local] for z in routed[5]],
                [z[local] for z in routed[6]],
                [z[local] for z in routed[7]],
                routed[8],
                routed[9],
            )
        return self.own_query_window(cast('OwnQuery', own), begin, end)

    def own_query_inputs(self, sequence: int = 0) -> OwnQuery:
        """Clean keys/values for one sequence; preserve upstream math and token positions."""
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs: list[Tensor] = []
        w_q: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        w_proj_t: list[Tensor] = []
        windows: list[int] = []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            k = d.outputs[prefix + 'c_k'][sequence].float().view(length, heads, width)
            k1, k2 = k[..., : width // 2], k[..., width // 2 :]
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
    def strong_keys(own: OwnQuery, k: int) -> OwnRouted:
        """Only top-k logits move; keep the remaining output for exact renormalization."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        length = len(pos)
        top_keys: list[Tensor] = []
        top_values: list[Tensor] = []
        top_p: list[Tensor] = []
        rests: list[Tensor] = []
        for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows, strict=False):
            heads, _, width = kk.shape
            c, s = cos[:, None], sin[:, None]
            r = (norm(x) @ wq.T).view(length, heads, width)
            q = norm(
                torch.cat(
                    [
                        r[..., : width // 2] * c + r[..., width // 2 :] * s,
                        r[..., : width // 2] * -s + r[..., width // 2 :] * c,
                    ],
                    -1,
                )
            )
            lag = pos[:, None] - pos[None]
            allowed = (lag >= 0) & (lag <= window)
            prob = torch.softmax(
                (torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width**-0.5).masked_fill(
                    ~allowed, float('-inf')
                ),
                -1,
            )
            p, idx = prob.topk(min(k, length), dim=-1)
            ktop = Cut._gather_strong_keys(kk, idx, length, width)
            vtop = Cut._gather_strong_keys(vv, idx, length, width)
            out = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
            rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
            top_keys.append(ktop.transpose(0, 1).contiguous())
            top_values.append(vtop.transpose(0, 1).contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
            rests.append(rest.transpose(0, 1).contiguous())
        return ('routed', pos, inputs, w_q, top_keys, top_values, top_p, rests, w_proj_t, (cos, sin))

    @staticmethod
    def _gather_strong_keys(z: Tensor, indices: Tensor, length: int, width: int) -> Tensor:
        """Gather the selected per-query keys or values without copying the expanded view."""
        return z.unsqueeze(1).expand(-1, length, -1, -1).gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, width))

    @staticmethod
    def own_query_window(own: OwnQuery, t0: int, t1: int) -> OwnWindow:
        """Slice clean context windows, with the same 256-token buckets as PR #19."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        total = keys[0].shape[1]
        rels: list[Tensor] = []
        ks: list[Tensor] = []
        vs: list[Tensor] = []
        for k, v, window in zip(keys, values, windows, strict=False):
            need = t1 - max(0, t0 - window)
            length = min(total, -(-need // KEY_BUCKET) * KEY_BUCKET)
            start = max(0, t1 - length)
            rels.append(pos[t0:t1] - start)
            ks.append(k[:, start : start + length])
            vs.append(v[:, start : start + length])
        return ('window', pos[t0:t1], rels, [z[t0:t1] for z in inputs], w_q, ks, vs, w_proj_t, cos, sin, windows)

    def select_mlp_routes(self, layer: int, activation: Tensor) -> Tensor:
        """Select the fixed replay units; experiments may override only this ranking."""
        weight = self.model.transformer.h[layer].mlp.c_proj.weight.float()
        return (activation * cast('TensorOperations', weight).norm(dim=0)).topk(self.s.route_k, dim=-1).indices

    @torch.no_grad()
    def estimate(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Assemble raw local probes, corrected writers, and source-only future credit."""
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        self._input_key_errors = None
        self.measured = None  # Last step's diagnostic credits are not inputs to this estimate.
        d.capture(x, y)
        b, t = x.shape
        yf = y.flatten()
        gradients: dict[str, Tensor] = {}
        head = d.head_error()  # coordinate differences at the final readout
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = self._clean_mlp_states()
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        chain = self.chain_credits(e, x0, mids, outs, routes, yf)
        lam_r = model.resid_lambdas.float()
        targets = {
            i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            for i, block in enumerate(model.transformer.h)
        }
        # Reuse only directly measured hub credit, scoped to this capture.
        # Corrected targets still take a separate A.T @ target in exact_values.
        raw_values = {i: self.value_credit(i, target) for i, target in targets.items()}
        extra, inputs = (
            self.path_credits(chain, targets, raw_values)
            if s.value_path or s.key_path_draws or s.gate_path
            else (None, None)
        )
        value_targets, scalars = self._writer_credits(gradients, chain, g_top, extra, lam_r, b, t)
        if s.path_scalars and extra is not None:
            scalars += self.path_scalar_credit(cast('dict[int, Tensor]', inputs), extra, x0, outs)
        self.measured = MeasuredCredits(chain=chain, g_top=g_top, targets=targets)
        self.internals(gradients, targets, b, t, value_targets or None, self._input_key_errors, raw_values)
        self._input_key_errors = None
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        if self.part:
            # These full estimates are already shared; internals partition or average their own.
            for name, gradient in gradients.items():
                if name in (
                    'lm_head.weight',
                    'transformer.wte.weight',
                    'resid_lambdas',
                    'x0_lambdas',
                ) or name.endswith('c_proj.weight'):
                    gradients[name] = torch.zeros_like(gradient)
        if s.future_draws:
            # Use this capture's original measured hub targets, never corrected
            # targets. Only the source parameter gradients receive the addition.
            add_future_gradients(self, gradients)
        return gradients

    def _clean_mlp_states(self) -> tuple[list[Tensor], list[Tensor], list[tuple[Tensor, Tensor]]]:
        """Reconstruct the clean complete MLP and select the fixed replay routes."""
        d, model = self.capture, self.model
        mids: list[Tensor] = []
        outs: list[Tensor] = []
        routes: list[tuple[Tensor, Tensor]] = []
        for i in range(model.config.n_layer):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = functional.linear(norm(mid), mlp.c_fc.weight.float())
            r = functional.relu(hpre).square()
            out = mid + functional.linear(r, mlp.c_proj.weight.float())
            top = self.select_mlp_routes(i, r)
            mids.append(mid)
            outs.append(out)
            routes.append((top, hpre.gather(1, top)))
        return mids, outs, routes

    def _writer_credits(
        self,
        gradients: dict[str, Tensor],
        chain: ChainCredits,
        g_top: Tensor,
        extra: dict[int, Tensor] | None,
        lam_r: Tensor,
        b: int,
        t: int,
    ) -> tuple[dict[int, Tensor], Tensor]:
        """Apply corrected credits only to linear writers, retaining raw hidden targets."""
        d, model = self.capture, self.model
        layers = model.config.n_layer
        value_targets: dict[int, Tensor] = {}
        scalars = torch.zeros(2, layers, device=self.device)
        for i in range(layers):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            d_out = g_top if i == layers - 1 else lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = (d_mid, d_out) if extra is None else (d_mid + extra[i], d_out + extra[i])
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            if extra is not None:
                value_targets[i] = (x_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            # Keep mixing probes on their directly measured chain targets.
            scalars[:, i] = chain['lam', i]
        return value_targets, scalars


# -------------------------------------------------------------------------------------------------- future probes


_FUTURE_FAMILY_NAMES: tuple[str, ...] = ('hidden', 'query', 'wte')


def _score_difference(
    richer: Callable[[Tensor], Tensor], baseline: Callable[[Tensor], Tensor]
) -> Callable[[Tensor], Tensor]:
    """Pair both forward scorers on the identical finite source response."""

    def score(move: Tensor) -> Tensor:
        return richer(move) - baseline(move)

    return score


class FiniteFuture:
    """Measure source-local responses at strictly future raw attention hubs."""

    def __init__(
        self,
        estimator: Cut,
        source_layers: int = 4,
        tokens: int = 64,
        chunk: int = 512,
        sigma: float = 0.01,
        seed: int = 9181,
        *,
        query_mlp: bool = True,
        replay_depth: int = 0,
        residual_correction: bool = False,
        relay_tokens: int = 0,
        timing: bool = True,
    ) -> None:
        """Configure independent finite source probes and receiver execution tiles."""
        self.e = estimator
        self.source_layers, self.tokens, self.chunk = source_layers, tokens, chunk
        self.sigma, self.seed = sigma, seed
        self.query_mlp = query_mlp
        self.replay_depth = replay_depth
        self.residual_correction = residual_correction
        self.relay_tokens = relay_tokens
        if relay_tokens:
            self.tokens, self.chunk = min(tokens, 4), min(chunk, 16)
        self.timing = timing
        self.contexts: list[FutureContext] | None = None
        if tokens <= 0 or chunk <= 0 or sigma <= 0:
            message = 'tokens, chunk and sigma must be positive'
            raise ValueError(message)
        if type(replay_depth) is not int or not 0 <= replay_depth < estimator.model.config.n_layer:
            message = 'Replay depth must be an integer below the number of blocks'
            raise ValueError(message)
        if residual_correction and not replay_depth:
            message = 'Residual correction requires positive source replay depth'
            raise ValueError(message)
        if (
            type(relay_tokens) is not int
            or not 0 <= relay_tokens <= _MAX_RELAY_RECIPIENTS
            or (relay_tokens and not replay_depth)
        ):
            message = 'Relay recipients require an integer in [0,32] and positive source replay depth'
            raise ValueError(message)

    @torch.no_grad()
    def prepare(self, sequence: int = 0) -> FiniteFuture:
        """Cache clean receiver states for exactly one captured sequence."""
        e = self.e
        d, model = e.capture, e.model
        t, heads = d.length, model.config.n_head
        measured = cast('MeasuredCredits', e.measured)
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        contexts: list[FutureContext] = []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            qraw, kraw, vraw = [d.outputs[prefix + n][sequence : sequence + 1].float() for n in ('c_q', 'c_k', 'c_v')]
            v = vraw.reshape(1, t, heads, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix + 've_gate'][sequence].float()
                ve = d.outputs[f'value_embeds.{i}'][sequence].float().reshape(t, heads, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
            values = v.transpose(1, 2)[0]
            qp, kp, a, out = attention_state(qraw, kraw, values[None], cos, sin, heads, model.window_sizes[i][0])
            q, k, a, out = qp[0], kp[0], a[0], out[0]
            target = measured['targets'][i][sequence].reshape(t, heads, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2)) - (target * out).sum(-1, keepdim=True)
            needs_replay = i < model.config.n_layer - 1 and (
                self.relay_tokens > 0 or (self.replay_depth > 0 and i < self.source_layers + self.replay_depth)
            )
            contexts.append(
                FutureContext(
                    x=d.blocks[i][sequence].float(),
                    q=q,
                    qraw=qraw[0] if needs_replay else None,
                    k=k,
                    kraw=kraw[0].reshape(t, heads, -1),
                    values=values,
                    a=a,
                    out=out,
                    contrast=contrast,
                    target=target,
                    gate=gate,
                    ve=ve,
                    wg=wg,
                    wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(),
                    window=model.window_sizes[i][0],
                    source_mlp=None,
                )
            )
            if (self.query_mlp and i < self.source_layers) or needs_replay:
                contexts[-1]['source_mlp'] = (
                    d.blocks[i][sequence].float() + d.outputs[prefix + 'c_proj'][sequence].float(),
                    d.outputs[f'transformer.h.{i}.mlp.c_fc'][sequence].float(),
                    block.mlp.c_fc.weight.float(),
                    block.mlp.c_proj.weight.float(),
                )
        self.contexts, self.sequence = contexts, sequence
        self.cos, self.sin = cos[0, :, 0], sin[0, :, 0]
        return self

    def sources(self, families: Iterable[str]) -> Iterator[tuple[str, int, str]]:
        """Yield source sites in the original family and layer order."""
        for family in families:
            if family == 'wte':
                yield family, -1, 'transformer.wte'
            else:
                for layer in range(self.source_layers):
                    suffix = 'mlp.c_fc' if family == 'hidden' else 'attn.c_q'
                    yield family, layer, f'transformer.h.{layer}.{suffix}'

    def experiment(
        self, family: str, layer: int, raw: Tensor, sl: slice, receivers: list[tuple[int, Tensor]], start: int
    ) -> LocalExperiment:
        """Bind clean tensors once per token chunk, outside the direction loop."""
        model = self.e.model
        contexts = cast('list[FutureContext]', self.contexts)
        response: Callable[[Tensor, Tensor], Tensor]
        if family == 'hidden':
            response = partial(compiled_hidden, raw[sl], weight=model.transformer.h[layer].mlp.c_proj.weight.float())
        elif family == 'wte':
            response = partial(compiled_embedding, raw[sl])
        else:
            ctx = contexts[layer]
            weight = model.transformer.h[layer].attn.c_proj.weight.float()
            if self.query_mlp:
                mid, hidden, w_fc, w_out = cast('tuple[Tensor, Tensor, Tensor, Tensor]', ctx['source_mlp'])
                response = partial(
                    query_mlp_moves,
                    raw[sl],
                    cos=self.cos[sl],
                    sin=self.sin[sl],
                    keys=ctx['k'],
                    values=ctx['values'],
                    probabilities=ctx['a'][:, sl],
                    base=ctx['out'][:, sl],
                    weight=weight,
                    mid=mid[sl],
                    hidden=hidden[sl],
                    w_fc=w_fc,
                    w_out=w_out,
                )
            else:
                response = partial(
                    compiled_query,
                    raw[sl],
                    cos=self.cos[sl],
                    sin=self.sin[sl],
                    keys=ctx['k'],
                    values=ctx['values'],
                    probabilities=ctx['a'][:, sl],
                    base=ctx['out'][:, sl],
                    weight=weight,
                )
        if self.replay_depth:
            return LocalExperiment(response, self._replay_scorer(family, sl, receivers, start))
        scorers: list[Callable[[Tensor], Tensor]] = []
        for j, coefficient in receivers:
            ctx = contexts[j]
            gate, ve = ctx['gate'], ctx['ve']
            scorers.append(
                partial(
                    receiver_scores,
                    coefficient=coefficient,
                    x=ctx['x'][sl],
                    kraw=ctx['kraw'][sl],
                    gate=None if gate is None else gate[sl],
                    ve=None if ve is None else ve[sl],
                    wkv=ctx['wkv'],
                    wg=ctx['wg'],
                    cos=self.cos[sl],
                    sin=self.sin[sl],
                    q=ctx['q'],
                    target=ctx['target'],
                    a=ctx['a'][:, :, sl],
                    contrast=ctx['contrast'][:, :, sl],
                    start=start,
                    window=ctx['window'],
                )
            )

        def score(move: Tensor) -> Tensor:
            if not scorers:
                return move.new_zeros(move.shape[:2])
            total = scorers[0](move)
            for receiver in scorers[1:]:
                total = total + receiver(move)
            return total

        return LocalExperiment(response, score)

    def _replay_scorer(
        self, family: str, sl: slice, receivers: list[tuple[int, Tensor]], start: int
    ) -> Callable[[Tensor], Tensor]:
        """Keep the original embedding skip while evolving the selected source-token prefix."""
        model = self.e.model
        contexts = cast('list[FutureContext]', self.contexts)
        trials: list[tuple[int, Tensor, Callable[[Tensor], tuple[Tensor, Tensor]]]] = []
        for offset, (layer, residual) in enumerate(receivers):
            block = model.transformer.h[layer]
            trials.append(
                (
                    layer,
                    residual,
                    partial(
                        replay_receiver,
                        context=contexts[layer],
                        cos=self.cos[sl],
                        sin=self.sin[sl],
                        wq=block.attn.c_q.weight.float(),
                        wproj=block.attn.c_proj.weight.float(),
                        start=start,
                        advance=offset < self.replay_depth and layer < len(contexts) - 1,
                        relay=(
                            self._relay_scorer(layer, start, contexts[layer]['x'][sl].shape[0])
                            if self.relay_tokens and layer < len(contexts) - 1
                            else None
                        ),
                    ),
                )
            )
        lam0 = model.x0_lambdas.float()

        def score(move: Tensor) -> Tensor:
            root, evolved = move, move
            total: Tensor | None = None
            for layer, residual, trial in trials:
                incoming = residual * evolved
                if family == 'wte':
                    incoming = incoming + lam0[layer] * root
                credit, evolved = trial(incoming)
                total = credit if total is None else total + credit
            return move.new_zeros(move.shape[:2]) if total is None else total

        return score

    def _relay_scorer(self, layer: int, start: int, count: int) -> Callable[[Tensor, Tensor], Tensor]:
        """Add only a second strictly-future transfer, never a reconstructed credit as a nonlinear input."""
        contexts = cast('list[FutureContext]', self.contexts)
        context, block = contexts[layer], self.e.model.transformer.h[layer]
        length = context['x'].shape[0]
        mid, hidden, w_fc, w_out = cast('tuple[Tensor, Tensor, Tensor, Tensor]', context['source_mlp'])
        strength = context['target'].square().sum((0, 2)).sqrt()
        positions = torch.arange(length, device=mid.device)
        trials: list[tuple[int, Callable[[Tensor, Tensor], Tensor], list[Callable[[Tensor], Tensor]]]] = []
        for offset in range(count):
            source = start + offset
            recipients = min(self.relay_tokens, length - source - 1, context['window'])
            if not recipients:
                continue
            allowed = (positions > source) & (positions - source <= context['window'])
            importance = (context['a'][:, :, source].mean(0) * strength).masked_fill(~allowed, -1)
            selected = importance.topk(recipients).indices.sort().values
            response = partial(
                relay_recipient_moves,
                query=context['q'][:, selected],
                probabilities=context['a'][:, selected, source],
                value_contrast=context['values'][:, source, None] - context['out'][:, selected],
                wproj=block.attn.c_proj.weight.float(),
                mid=mid[selected],
                hidden=hidden[selected],
                w_fc=w_fc,
                w_out=w_out,
            )
            trials.append((offset, response, self._relay_receivers(layer, selected)))

        def score(dk: Tensor, dv: Tensor) -> Tensor:
            total = dk.new_zeros((dk.shape[2], dk.shape[1]))
            for offset, response, receivers in trials:
                moved = response(dk[:, :, offset], dv[:, :, offset])
                for receiver in receivers:
                    total[offset] += receiver(moved).sum(0)
            return total

        return score

    def _relay_receivers(self, layer: int, selected: Tensor) -> list[Callable[[Tensor], Tensor]]:
        """Score sparse recipient memories after frozen residual transport, preserving their real positions."""
        contexts = cast('list[FutureContext]', self.contexts)
        length = contexts[0]['x'].shape[0]
        queries = torch.arange(length, device=selected.device)[:, None]
        coefficient = torch.ones((), device=selected.device)
        scorers: list[Callable[[Tensor], Tensor]] = []
        for receiver in range(layer + 1, len(contexts)):
            context = contexts[receiver]
            coefficient = coefficient * self.e.model.resid_lambdas[receiver].float()
            allowed = (queries > selected[None]) & (queries - selected[None] <= context['window'])
            gate, ve = context['gate'], context['ve']
            # Sorted unique selected[i] >= i. Pre-mask the REAL positions, then
            # the existing kernel's relaxed start=0/window=length cannot drop a valid edge.
            scorers.append(
                partial(
                    receiver_scores,
                    coefficient=coefficient,
                    x=context['x'][selected],
                    kraw=context['kraw'][selected],
                    gate=None if gate is None else gate[selected],
                    ve=None if ve is None else ve[selected],
                    wkv=context['wkv'],
                    wg=context['wg'],
                    cos=self.cos[selected],
                    sin=self.sin[selected],
                    q=context['q'],
                    target=context['target'],
                    a=context['a'][:, :, selected] * allowed[None],
                    contrast=context['contrast'][:, :, selected],
                    start=0,
                    window=length,
                )
            )
        return scorers

    def _probe_span(self, budgets: list[int]) -> range:
        """Partition one shared budget, keeping its full normalization on every rank."""
        if not budgets or budgets[0] <= 0:
            message = 'Probe budgets must be positive'
            raise ValueError(message)
        if self.e.split > 1:
            if len(budgets) != 1 or budgets[0] % self.e.split:
                message = 'Split future probes need one budget divisible by the GPUs of a sequence'
                raise ValueError(message)
            share = budgets[0] // self.e.split
            return range(self.e.part * share, (self.e.part + 1) * share, gcd(self.chunk, share))
        return range(0, max(budgets), gcd(self.chunk, *budgets))

    @torch.no_grad()
    def estimate(
        self, budgets: Iterable[int] = (512,), families: Iterable[str] = ('hidden', 'query', 'wte')
    ) -> dict[int, dict[str, Tensor]]:
        """Measure replay directly, or add sampled replay-minus-identity to a full cheap cycle."""
        budgets = sorted(set(budgets))
        if not self.residual_correction:
            return self._estimate(budgets, families)
        self._probe_span(budgets)
        identity = FiniteFuture(
            self.e,
            source_layers=self.source_layers,
            tokens=self.tokens,
            chunk=self.chunk,
            sigma=self.sigma,
            seed=self.seed,
            query_mlp=self.query_mlp,
            timing=self.timing,
        )
        identity.contexts, identity.sequence = self.contexts, self.sequence
        identity.cos, identity.sin = self.cos, self.sin
        result: dict[int, dict[str, Tensor]] = {budget: {} for budget in budgets}
        timings: dict[str, float] = {}
        for family in families:
            width = self.e.model.config.n_embd * (4 if family == 'hidden' else 1)
            if budgets[-1] > width:
                message = f'{family}: budget exceeds its {width}-direction cycle'
                raise ValueError(message)
            partial_budgets = [budget for budget in budgets if budget < width]
            if partial_budgets:
                baseline = identity._estimate([width], (family,))[width]
                residuals = self._estimate(partial_budgets, (family,), reference=identity)
                for name in baseline:
                    timings[name] = identity.timings[name] + self.timings[name]
                for budget in partial_budgets:
                    result[budget].update({name: baseline[name] + credit for name, credit in residuals[budget].items()})
            # A complete richer cycle already has no subsampling variance.
            if width in result:
                result[width].update(self._estimate([width], (family,))[width])
                for name, elapsed in self.timings.items():
                    timings[name] = timings.get(name, 0.0) + elapsed
        self.timings = timings
        return result

    def _estimate(
        self, budgets: list[int], families: Iterable[str], *, reference: FiniteFuture | None = None
    ) -> dict[int, dict[str, Tensor]]:
        """Decode shared frames with independent normalization for each distributed budget."""
        contexts = cast('list[FutureContext]', self.contexts)
        e, model, d = self.e, self.e.model, self.e.capture
        span = self._probe_span(budgets)
        stops = {span.stop: budgets[0]} if e.split > 1 else {p: p for p in budgets}
        result: dict[int, dict[str, Tensor]] = {p: {} for p in budgets}
        timings: dict[str, float] = {}
        lam, lam0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        for family, layer, name in self.sources(families):
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw = d.outputs[name][self.sequence].float()
            t, width = raw.shape
            if budgets[-1] > width:
                message = f'{name}: budget {budgets[-1]} exceeds the {width}-direction cycle'
                raise ValueError(message)
            gen = torch.Generator(raw.device).manual_seed(
                self.seed + 7919 * (layer + 2) + 101 * _FUTURE_FAMILY_NAMES.index(family)
            )
            # Per-token random signs/offsets prevent a shared low-rank projection
            # from persisting in the batch's parameter gradient.
            frame = hadamard(width, raw.device, torch.float32)
            permutation = torch.randperm(width, device=raw.device, generator=gen)
            offsets = torch.randint(width, (t,), device=raw.device, generator=gen)
            signs = 2 * torch.randint(2, (t, width), device=raw.device, generator=gen) - 1
            steps = self.sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            prefix_credit = {p: torch.zeros_like(raw) for p in budgets}
            receivers = self._source_receivers(family, layer, raw, lam, lam0, len(contexts))
            reference_receivers = (
                None
                if reference is None
                else FiniteFuture._source_receivers(reference, family, layer, raw, lam, lam0, len(contexts))
            )
            for t0 in range(0, t, self.tokens):
                sl = slice(t0, min(t0 + self.tokens, t))
                trial = self.experiment(family, layer, raw, sl, receivers, t0)
                if reference is not None and reference_receivers is not None:
                    baseline_trial = reference.experiment(family, layer, raw, sl, reference_receivers, t0)
                    trial = LocalExperiment(trial.response, _score_difference(trial.score, baseline_trial.score))
                accumulator = torch.zeros_like(raw[sl])
                for p0 in span:
                    p1 = p0 + span.step
                    index = (offsets[sl, None] + permutation[None, p0:p1]) % width
                    rows = frame[index] * signs[sl, None]
                    accumulator += trial(rows, steps[sl])
                    if p1 in stops:
                        budget = stops[p1]
                        prefix_credit[budget][sl] = accumulator / budget
            for p in budgets:
                result[p][name] = prefix_credit[p][None]
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            timings[name] = time.perf_counter() - start
        self.timings = timings
        return result

    def _source_receivers(
        self, family: str, layer: int, raw: Tensor, lam: Tensor, lam0: Tensor, count: int
    ) -> list[tuple[int, Tensor]]:
        """Collapse frozen transport only; nonlinear replay needs each block's multiplier."""
        if self.replay_depth:
            first = 0 if family == 'wte' else layer + 1
            return [(j, lam[j]) for j in range(first, count)]
        return identity_receivers(family, layer, raw, lam, lam0, count)


def identity_receivers(
    family: str, layer: int, raw: Tensor, lam: Tensor, lam0: Tensor, count: int
) -> list[tuple[int, Tensor]]:
    """Receiver blocks and residual-identity multipliers of one source's move."""
    receivers: list[tuple[int, Tensor]] = []
    coefficient = torch.ones((), device=raw.device)
    if family == 'wte':
        for j in range(count):
            coefficient = lam[j] * coefficient + lam0[j]
            receivers.append((j, coefficient))
    else:
        for j in range(layer + 1, count):
            coefficient = coefficient * lam[j]
            receivers.append((j, coefficient))
    return receivers


ReceiverTables: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
UnitFactors: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor | None]
ReceiverState: TypeAlias = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    tuple[Tensor, Tensor, Tensor, Tensor] | None,
]


class ExactFuture:
    """#31's future-token estimand over complete cycles at every source layer, with nothing sampled.

    Sources, residual-identity transport and measured receiver hubs are FiniteFuture's. Hidden units use the
    coordinate cycle: each move is a scaled c_proj column, so a receiver's finite RMSNorm response depends on
    it only through x.c and |c|^2, and its finite K/V/gate changes are exact sums of precomputed projections.
    Query and embedding sources use complete Hadamard cycles. Every receiver's value term is exact; the
    single-column softmax update of the key term is evaluated exactly for each head's `receivers` strongest
    receivers and omitted elsewhere. Every score is a forward difference at a measured hub, as in #31.
    """

    def __init__(self, future: FiniteFuture, receivers: int, *, timing: bool = False) -> None:
        """Reuse a prepared FiniteFuture's clean receiver contexts and source responses."""
        if receivers < 1 or future.replay_depth or future.residual_correction or future.relay_tokens:
            message = 'Exact future probes need identity transport and at least one receiver'
            raise ValueError(message)
        self.f, self.e, self.receivers, self.timing = future, future.e, receivers, timing
        self.timings: dict[str, float] = {}
        self.table_cache: dict[tuple[int, int], ReceiverTables] = {}
        self.factor_cache: dict[tuple[int, int], UnitFactors] = {}
        self.state_cache: dict[tuple[int, int], ReceiverState] = {}

    @torch.no_grad()
    def estimate(self, families: Iterable[str] = ('hidden', 'query', 'wte')) -> dict[str, Tensor]:
        """Decode each source site's complete central cycle; GPUs sharing a sequence split its directions."""
        f, e = self.f, self.e
        contexts = cast('list[FutureContext]', f.contexts)
        model, d = e.model, e.capture
        lam, lam0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        result: dict[str, Tensor] = {}
        for family, layer, name in f.sources(families):
            raw = d.outputs[name][f.sequence].float()
            receivers = identity_receivers(family, layer, raw, lam, lam0, len(contexts))
            if not receivers:
                continue
            if self.timing and raw.is_cuda:
                torch.cuda.synchronize()
            began = time.perf_counter()
            steps = f.sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            credit = torch.zeros_like(raw)
            for t0 in range(0, raw.shape[0], f.tokens):
                sl = slice(t0, min(t0 + f.tokens, raw.shape[0]))
                bound = [(j, contexts[j], coefficient, self.tables(j, contexts[j], sl)) for j, coefficient in receivers]
                if family == 'hidden':
                    self.hidden(layer, raw, steps, sl, bound, credit)
                else:
                    self.cycle(family, layer, raw, steps, sl, bound, credit)
            result[name] = credit[None]
            if self.timing and raw.is_cuda:
                torch.cuda.synchronize()
            self.timings[family] = self.timings.get(family, 0.0) + time.perf_counter() - began
        return result

    def tables(self, layer: int, ctx: FutureContext, sl: slice) -> ReceiverTables:
        """Strongest allowed receivers per head and source, and the exact value pullback of all others."""
        key = (layer, cast('int', sl.start))
        if key in self.table_cache:
            return self.table_cache[key]
        a = ctx['a'][:, :, sl]
        length = a.shape[1]
        s = torch.arange(length, device=a.device)[:, None]
        t = cast('int', sl.start) + torch.arange(a.shape[2], device=a.device)[None]
        a = a * ((s > t) & (s - t <= ctx['window']))
        top_a, top_s = a.topk(min(self.receivers, length), dim=1)
        heads, count, sources = top_s.shape
        width = ctx['q'].shape[-1]
        index = top_s.reshape(heads, count * sources, 1).expand(-1, -1, width)
        q_top = ctx['q'].gather(1, index).view(heads, count, sources, width)
        target_top = ctx['target'].gather(1, index).view(heads, count, sources, width)
        contrast_top = ctx['contrast'][:, :, sl].gather(1, top_s)
        rest = torch.einsum('hsm,hsd->hmd', a, ctx['target']) - torch.einsum('hkm,hkmd->hmd', top_a, target_top)
        self.table_cache[key] = (top_a, q_top, target_top, contrast_top, rest)
        return self.table_cache[key]

    def factors(self, source: int, receiver: int, ctx: FutureContext, c_proj: Tensor) -> UnitFactors:
        """Per-head raw K/V projections, squared K norms and gate projections of every c_proj column."""
        key = (source, receiver)
        if key not in self.factor_cache:
            heads, hd = ctx['kraw'].shape[-2:]
            projected = ctx['wkv'] @ c_proj
            keys = projected[: heads * hd].view(heads, hd, -1)
            wg = ctx['wg']
            self.factor_cache[key] = (
                keys,
                projected[heads * hd :].view(heads, hd, -1),
                keys.square().sum(1),
                None if wg is None else wg @ c_proj[: wg.shape[1]],
            )
        return self.factor_cache[key]

    def receiver_state(self, layer: int, ctx: FutureContext, tables: ReceiverTables, sl: slice) -> ReceiverState:
        """Source-token receiver quantities shared by every hidden source of one receiver block and chunk.

        Rotary is orthogonal at each source position, so key-norm products use unrotated keys and every
        receiver query is rotated back to the source position once.
        """
        key = (layer, cast('int', sl.start))
        if key in self.state_cache:
            return self.state_cache[key]
        top_a, q_top, target_top, contrast_top, rest = tables
        heads, hd = ctx['kraw'].shape[-2:]
        x = ctx['x'][sl]
        m = x.shape[0]
        projected = functional.linear(x, ctx['wkv'])
        pk, pv = projected[:, : heads * hd].view(m, heads, hd), projected[:, heads * hd :].view(m, heads, hd)
        kraw = ctx['kraw'][sl]
        back = rotate(q_top, self.f.cos[sl][None, None], -self.f.sin[sl][None, None])
        ve, wg = ctx['ve'], ctx['wg']
        gate = None
        if ve is not None and wg is not None:
            ve_sl = ve[sl]
            gate = (
                cast('Tensor', ctx['gate'])[sl],
                functional.linear(x[:, : wg.shape[1]], wg),
                torch.einsum('hkmd,mhd->mhk', target_top, ve_sl),
                torch.einsum('hmd,mhd->mh', rest, ve_sl),
            )
        square = x.square().sum(-1, keepdim=True)
        state: ReceiverState = (
            x,
            square,
            torch.rsqrt(square / x.shape[-1] + torch.finfo(x.dtype).eps),
            kraw,
            pk,
            back,
            target_top,
            rest,
            kraw.square().sum(-1),
            (kraw * pk).sum(-1),
            pk.square().sum(-1),
            torch.einsum('hkmd,mhd->mhk', back, pk),
            torch.einsum('hkmd,mhd->mhk', back, kraw),
            torch.einsum('hkmd,mhd->mhk', target_top, pv),
            torch.einsum('hmd,mhd->m', rest, pv),
            top_a.permute(2, 0, 1),
            contrast_top.permute(2, 0, 1),
            gate,
        )
        self.state_cache[key] = state
        return state

    def score(self, ctx: FutureContext, tables: ReceiverTables, sl: slice, proj: Tensor, dg: Tensor | None) -> Tensor:
        """Hub scores of finite K/V(/gate) changes (m, moves, 2W) at one receiver block, as in joint_scores."""
        top_a, q_top, target_top, contrast_top, rest = tables
        heads, hd = ctx['kraw'].shape[-2:]
        m, moves = proj.shape[:2]
        c, s = self.f.cos[sl][:, None, None], self.f.sin[sl][:, None, None]
        r0 = rotate(ctx['kraw'][sl][:, None], c, s)
        dk = norm_delta(r0, rotate(proj[..., : heads * hd].reshape(m, moves, heads, hd), c, s))
        dv = proj[..., heads * hd :].reshape(m, moves, heads, hd)
        if dg is not None:
            gate = cast('Tensor', ctx['gate'])[sl][:, None]
            change = 2 * ((gate + dg).sigmoid() - gate.sigmoid())
            dv = dv + change[..., None] * cast('Tensor', ctx['ve'])[sl][:, None]
        weight = top_a.permute(2, 0, 1)[:, None]
        changed = weight * torch.expm1(torch.einsum('hkmd,mphd->mphk', q_top, dk) * hd**-0.5)
        value = torch.einsum('hkmd,mphd->mphk', target_top, dv)
        contrast = contrast_top.permute(2, 0, 1)[:, None]
        top = ((changed * contrast + (weight + changed) * value) / (1 + changed)).sum((-2, -1))
        return top + torch.einsum('hmd,mphd->mp', rest, dv)

    def hidden(
        self,
        layer: int,
        raw: Tensor,
        steps: Tensor,
        sl: slice,
        bound: list[tuple[int, FutureContext, Tensor, ReceiverTables]],
        credit: Tensor,
    ) -> None:
        """Coordinate cycle of squared-ReLU units, scored in closed form.

        A unit's move is a multiple of its c_proj column, so the receiver's finite input RMSNorm, K/V/gate
        changes, finite per-head QK-norm and single-column softmax update are closed expressions in fixed
        dot products: each move costs scalar work per head and receiver, not width-sized tensors.
        """
        c_proj = self.e.model.transformer.h[layer].mlp.c_proj.weight.float()
        u, step = raw[sl], steps[sl]
        base = u.relu().square()
        moves = torch.stack(((u + step).relu().square() - base, (u - step).relu().square() - base), 1)
        column_square = c_proj.square().sum(0)
        lo, hi = self.e.share(u.shape[-1])
        total = torch.zeros_like(moves[:, :, lo:hi])
        for j, ctx, coefficient, tables in bound:
            state = self.receiver_state(j, ctx, tables, sl)
            x, square, inverse = state[0], state[1], state[2]
            keys, values, key_square, gate_columns = self.factors(layer, j, ctx, c_proj)
            dots = x @ c_proj
            width = x.shape[-1]
            for u0 in range(lo, hi, self.f.chunk):
                u1 = min(u0 + self.f.chunk, hi)
                alpha = coefficient * moves[:, :, u0:u1]
                moved = torch.rsqrt(
                    (square[..., None] + 2 * alpha * dots[:, None, u0:u1] + alpha.square() * column_square[u0:u1])
                    / width
                    + torch.finfo(alpha.dtype).eps
                )
                total[:, :, u0 - lo : u1 - lo] += self.unit_scores(
                    state,
                    keys[..., u0:u1],
                    values[..., u0:u1],
                    key_square[:, u0:u1],
                    None if gate_columns is None else gate_columns[:, u0:u1],
                    alpha * moved,
                    moved - inverse[..., None],
                )
        credit[sl, lo:hi] = (total[:, 0] - total[:, 1]) / (2 * step)

    @staticmethod
    def unit_scores(
        state: ReceiverState,
        keys: Tensor,
        values: Tensor,
        key_square: Tensor,
        gate_columns: Tensor | None,
        scale: Tensor,
        shift: Tensor,
    ) -> Tensor:
        """Joint-score sum of moves scale * column + shift * x for units (m, signs, U); exact algebra."""
        _, _, _, kraw, pk, back, target_top, rest, k0, kpx, pp, qpx, q0, tpv, rpv, weight, contrast, gate = state
        hd = keys.shape[1]
        eps = torch.finfo(scale.dtype).eps
        s, t = scale[..., None], shift[..., None]
        k_r0 = torch.einsum('mhd,hdu->muh', kraw, keys)[:, None]
        k_px = torch.einsum('mhd,hdu->muh', pk, keys)[:, None]
        dr_square = s.square() * key_square.T + 2 * s * t * k_px + t.square() * pp[:, None, None]
        r0_dr = s * k_r0 + t * kpx[:, None, None]
        clean = k0[:, None, None]
        moved = torch.rsqrt((clean + 2 * r0_dr + dr_square) / hd + eps)
        inverse = torch.rsqrt(clean / hd + eps)
        q_dr = s[..., None] * torch.einsum('hkmd,hdu->muhk', back, keys)[:, None] + t[..., None] * qpx[:, None, None]
        logits = (moved[..., None] * q_dr + (moved - inverse)[..., None] * q0[:, None, None]) * hd**-0.5
        a = weight[:, None, None]
        changed = a * torch.expm1(logits)
        value = (
            s[..., None] * torch.einsum('hkmd,hdu->muhk', target_top, values)[:, None]
            + t[..., None] * tpv[:, None, None]
        )
        others = scale * torch.einsum('hmd,hdu->mu', rest, values)[:, None] + shift * rpv[:, None, None]
        if gate is not None and gate_columns is not None:
            clean_gate, gate_x, tve, rve = gate
            g = clean_gate[:, None, None]
            change = 2 * ((g + s * gate_columns.T + t * gate_x[:, None, None]).sigmoid() - g.sigmoid())
            value = value + change[..., None] * tve[:, None, None]
            others = others + (change * rve[:, None, None]).sum(-1)
        top = ((changed * contrast[:, None, None] + (a + changed) * value) / (1 + changed)).sum((-2, -1))
        return top + others

    def cycle(
        self,
        family: str,
        layer: int,
        raw: Tensor,
        steps: Tensor,
        sl: slice,
        bound: list[tuple[int, FutureContext, Tensor, ReceiverTables]],
        credit: Tensor,
    ) -> None:
        """Complete Hadamard cycle of query or embedding moves, each finitely normed at every receiver."""
        width = raw.shape[-1]
        frame = hadamard(width, raw.device, torch.float32)
        response = self.f.experiment(family, layer, raw, sl, [], cast('int', sl.start)).response
        lo, hi = self.e.share(width)
        for p0 in range(lo, hi, self.f.chunk):
            rows = frame[p0 : min(p0 + self.f.chunk, hi)]
            moves = response(rows[None].expand(len(raw[sl]), -1, -1), steps[sl])
            total = moves.new_zeros(moves.shape[:2])
            for _, ctx, coefficient, tables in bound:
                dn = norm_delta(ctx['x'][sl][:, None], coefficient * moves)
                wg = ctx['wg']
                dg = None if wg is None else functional.linear(dn[..., : wg.shape[1]], wg)
                total += self.score(ctx, tables, sl, functional.linear(dn, ctx['wkv']), dg)
            count = len(rows)
            credit[sl] += (total[:, :count] - total[:, count:]) / (2 * steps[sl]) @ rows / width


@torch.no_grad()
def add_future_gradients(estimator: Cut, gradients: dict[str, Tensor]) -> None:
    """Add independent forward responses, releasing dense contexts per sequence.

    Draw a fresh frame seed from the sequence-shared RNG when probes are split.
    Partial cycles must not persist in one subspace across training. Never
    overwrite the original measured hub targets.
    """
    e, s, d = estimator, estimator.s, estimator.capture
    e.maps.clear()
    generator = d.shared_generator if e.split > 1 else d.generator
    seed = int(torch.randint(2**31, (), device=e.device, generator=generator).item())
    future = FiniteFuture(
        e,
        source_layers=s.future_layers,
        tokens=s.future_tokens,
        chunk=s.future_chunk,
        sigma=s.future_sigma,
        seed=seed,
        query_mlp=True,
        replay_depth=s.future_replay_depth,
        residual_correction=s.future_residual_correction,
        relay_tokens=s.future_relay_tokens,
        timing=False,
    )
    for sequence in range(d.batch):
        future.seed = seed + 104729 * sequence
        future.prepare(sequence)
        if s.future_exact:
            corrections = ExactFuture(future, s.future_receivers).estimate(s.future_exact)
            if 'hidden' not in s.future_exact:
                hidden = future.estimate((s.future_hidden_draws,), families=('hidden',))
                corrections.update(hidden[s.future_hidden_draws])
            sampled = tuple(family for family in ('query', 'wte') if family not in s.future_exact)
            if sampled:
                corrections.update(future.estimate((s.future_draws,), families=sampled)[s.future_draws])
        elif s.future_hidden_draws == s.future_draws:
            corrections = future.estimate((s.future_draws,))[s.future_draws]
        else:
            # Family-local seeds preserve the original Q/WTE frames and source targets.
            corrections = future.estimate((s.future_hidden_draws,), families=('hidden',))[s.future_hidden_draws]
            corrections.update(future.estimate((s.future_draws,), families=('query', 'wte'))[s.future_draws])
        for site, measured_credit in corrections.items():
            credit = measured_credit
            inputs = d.inputs[site][sequence]
            gradient = gradients[site + '.weight']
            if 'mlp.c_fc' in site:
                raw = d.outputs[site][sequence].float()
                step = s.future_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                # Coordinates that are inactive in every signed trial have no
                # response; remove their partial-cycle decoding cross-talk.
                credit = credit * (raw + step > 0)[None]
            if site == 'transformer.wte':
                credit = linear_embedding_credit(cast('nn.Embedding', d.modules[site]), credit)
                gradient.index_add_(0, inputs.flatten(), credit[0].flatten(0, -2))
            else:
                gradient.addmm_(credit[0].T, inputs)
        future.contexts = None
        del corrections


ANVIL_MATRIX_DIMENSIONS = 2
MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)
Phase: TypeAlias = Literal['boost', 'cut']
Stage: TypeAlias = tuple[Phase, int, int, int, int]


class AnvilStack(TypedDict):
    """Batched moving averages with per-parameter state views."""

    fast: Tensor
    slow: Tensor
    energy: Tensor


class AnvilState(TypedDict, total=False):
    """Lazily allocated ANVIL state for one matrix."""

    fast: Tensor
    slow: Tensor
    energy: Tensor


class _GroupRequired(TypedDict):
    params: list[Tensor]
    lr: float
    weight_decay: float


class OptimizerGroup(_GroupRequired, total=False):
    """Optimizer fields consumed by the shared learning-rate schedule."""

    initial_lr: float


def newton_schulz(x: Tensor) -> Tensor:
    """Apply the record's six quintic maps to square matrices in bfloat16."""
    x = x.bfloat16()
    for a, b, c in MAPS:
        gram = x @ x.transpose(1, 2)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x


def gram_newton_schulz(x: Tensor, restarts: tuple[int, ...] = (2, 4)) -> Tensor:
    """Apply the same maps in float16, rebuilding the Gram matrix at the declared restarts."""
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    gram, transform = x @ x.transpose(1, 2), eye
    for index, (a, b, c) in enumerate(MAPS):
        if index in restarts:
            x = transform @ x
            gram, transform = x @ x.transpose(1, 2), eye
        term = b * gram + c * gram @ gram
        transform = transform @ term + a * transform
        gram_term = gram @ term + a * gram
        gram = term @ gram_term + a * gram_term
    return transform @ x


def polar_batch(g: Tensor, margin: float = 1.1) -> Tensor:
    """Orthogonalize a same-shape stack without changing the native precision branches."""
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * cast('TensorOperations', x).norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch_operations.compile(dynamic=False)
def anvil_update(
    g: Tensor,
    w: Tensor,
    fast: Tensor,
    slow: Tensor,
    energy: Tensor,
    beta: Tensor,
    fast_beta: Tensor,
    slow_beta: Tensor,
    weight: Tensor,
    lane_beta: Tensor,
    lr: Tensor,
    wd: Tensor,
    lanes: int,
    ratio: float,
    margin: float,
) -> Tensor:
    """Advance the three moving averages and compute one batched ANVIL II update."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (
        cast('TensorOperations', update).norm(dim=(1, 2), keepdim=True)
        / cast('TensorOperations', gained).norm(dim=(1, 2), keepdim=True).clamp_min(1e-10)
    )
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class Anvil(torch.optim.Optimizer):
    """Run the accepted batched ANVIL II algorithm with per-parameter handoff state."""

    state: defaultdict[Tensor, AnvilState]
    stacks: dict[tuple[Tensor, ...], AnvilStack]

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        weight_decay: float,
        steps: int,
        lane_beta: float = 0.9,
        beta_min: float = 0.85,
        beta_max: float = 0.93,
        slow_beta: float = 0.98,
        fast_beta: float = 0.85,
        fast_weight: float = 0.4385,
        polar_margin: float = 1.1,
    ) -> None:
        """Initialize update-clock moments and the phase-relative rail schedule."""
        super().__init__(params, {'lr': lr, 'weight_decay': weight_decay})
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.polar_margin = polar_margin
        self.t, self.stacks = 0, {}

    def rail_beta(self) -> float:
        """Compute the native warmup, plateau and cooldown momentum coefficient."""
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Discard stacked views before restoring per-parameter optimizer state."""
        self.stacks.clear()
        super().load_state_dict(state_dict)

    def stacked_state(self, shape: tuple[int, ...], params: list[Tensor]) -> AnvilStack:
        """Reuse same-parameter stacks, preserving all moments across the phase handoff."""
        key = tuple(params)
        cached = self.stacks.get(key)
        if cached is not None and all(self.state[p].get('fast') is not None for p in params):
            return cached
        lanes = 1 if shape[0] >= shape[1] else 0
        weights = torch.stack([p.float() for p in params])

        def gathered(name: Literal['fast', 'slow', 'energy'], like: Tensor) -> Tensor:
            held = [self.state[p].get(name) for p in params]
            return (
                torch.stack(cast('list[Tensor]', held)).clone()
                if all(value is not None for value in held)
                else torch.zeros_like(like)
            )

        stack: AnvilStack = {
            'fast': gathered('fast', weights),
            'slow': gathered('slow', weights),
            'energy': gathered('energy', weights.mean(lanes + 1, keepdim=True)),
        }
        for index, parameter in enumerate(params):
            self.state[parameter].update(
                fast=stack['fast'][index], slow=stack['slow'][index], energy=stack['energy'][index]
            )
        self.stacks[key] = stack
        return stack

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> NoReturn: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Apply measured gradients without invoking an automatic-differentiation closure."""
        if closure is not None:
            message = 'ANVIL requires measured gradients, not optimizer closures.'
            raise ValueError(message)
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        groups = cast('list[OptimizerGroup]', self.param_groups)
        device = groups[0]['params'][0].device

        def value(number: float) -> Tensor:
            return torch.tensor(float(number), device=device)

        for group in groups:
            lr, wd = group['lr'], group['weight_decay']
            by_shape: dict[tuple[int, ...], list[Tensor]] = {}
            for parameter in group['params']:
                by_shape.setdefault(tuple(parameter.shape), []).append(parameter)
            for shape, params in by_shape.items():
                stack = self.stacked_state(shape, params)
                updated = anvil_update(
                    torch.stack([cast('Tensor', p.grad).float() for p in params]),
                    torch.stack([p.float() for p in params]),
                    stack['fast'],
                    stack['slow'],
                    stack['energy'],
                    value(beta),
                    value(fast_beta),
                    value(self.slow_beta),
                    value(weight),
                    value(self.lane_beta),
                    value(lr),
                    value(wd),
                    1 if shape[0] >= shape[1] else 0,
                    max(1, shape[0] / shape[1]) ** 0.5,
                    self.polar_margin,
                )
                foreach_copy(params, list(updated.to(params[0].dtype).unbind(0)))
        self.t += 1


class _MeasuredOptimizerStep(Protocol):
    def step(self) -> None:
        """Advance an optimizer without a loss closure."""
        ...


class Optimizers:
    """Advance the disjoint matrix and Adam states with identical rank-replica arithmetic."""

    def __init__(self, anvil: Anvil, adam: torch.optim.AdamW) -> None:
        """Keep both native optimizers and their mutable parameter groups."""
        self.optimizers = (anvil, adam)
        self.param_groups = cast(
            'list[OptimizerGroup]', [g for optimizer in self.optimizers for g in optimizer.param_groups]
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear both groups without allocating replacement gradients."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Require deterministic optimizer arithmetic, restoring the caller's mode afterward."""
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(mode=True)
            for optimizer in self.optimizers:
                cast('_MeasuredOptimizerStep', optimizer).step()
        finally:
            torch.use_deterministic_algorithms(mode=enabled, warn_only=warn_only)


def make_model(
    device: torch.device,
    seed: int,
    config: GPTConfig,
    projection_std: float,
    head_std: float,
) -> GPT:
    """Use the model's initializer with the declared writer and readout scales."""
    torch_operations.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in cast('NamedModules', model).named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = projection_std * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        if head_std != _FIXED_HEAD_STD:
            nn.init.normal_(model.lm_head.weight, std=head_std)
    return model.requires_grad_(requires_grad=False)


def make_optimizer(model: GPT, recipe: Settings, steps: int) -> Optimizers:
    """Apply ANVIL II to block matrices and the accepted Adam rates elsewhere."""
    matrices: list[Tensor] = []
    groups: dict[int, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == ANVIL_MATRIX_DIMENSIONS and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{'params': parameters, 'lr': recipe.lr * scale} for scale, parameters in groups.items()],
        betas=recipe.betas,
        eps=1e-8,
        weight_decay=0.0,
    )
    anvil = Anvil(
        matrices,
        recipe.anvil_lr,
        recipe.anvil_wd,
        steps,
        polar_margin=recipe.polar_margin,
        fast_weight=recipe.fast_weight,
    )
    return Optimizers(anvil, adam)


def set_rate(optimizer: Optimizers, step: int, steps: int, warmdown: float, final: float) -> None:
    """Set the one-based native warmdown to the declared final rate fraction."""
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        initial_lr = group.setdefault('initial_lr', group['lr'])
        group['lr'] = initial_lr * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


def stage_schedule(rows: int, settings: Settings) -> list[Stage]:
    """Cover disjoint data rows once with separate Boost and Cut scheduler clocks."""
    boost_batch, cut_batch = settings.boost_batch, settings.cut_batch
    if rows % cut_batch or cut_batch % boost_batch:
        message = 'Data rows must divide into complete Cut batches, and Cut batch must divide by Boost batch'
        raise ValueError(message)
    boost_horizon, cut_horizon = rows // boost_batch, rows // cut_batch
    alignment = cut_batch // boost_batch
    boost = (
        boost_horizon
        if settings.estimator == 'boost'
        else round(settings.switch * boost_horizon) // alignment * alignment
    )
    batches: list[Stage] = [
        ('boost', index, (index - 1) * boost_batch, index * boost_batch, boost_horizon) for index in range(1, boost + 1)
    ]
    if settings.estimator != 'boost':
        batches.extend(
            ('cut', index, (index - 1) * cut_batch, index * cut_batch, cut_horizon)
            for index in range(boost * boost_batch // cut_batch + 1, cut_horizon + 1)
        )
    return batches


def handoff_optimizer(old: Optimizers, new: Optimizers, phase_step: int) -> None:
    """Carry all matrix/Adam moments while moving only the matrix scheduler clock."""
    new.optimizers[0].state.update(old.optimizers[0].state)
    new.optimizers[1].state.update(old.optimizers[1].state)
    new.optimizers[0].t = phase_step - 1


# End merge: choose PR40's end-only trajectory merge using held-out training forwards only.
# First score each candidate applied globally, then refine the winning state with one full coordinate
# sweep and one sweep over each part's ordered best eight. No validation rows, parameter gradients or
# training-time proposals enter here.
HOLDOUT = 32
MERGE_FROM = 0.6
MERGE_EVERY = 5
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)
MERGE_RHOS = (0.0, 0.5, 1.0)
MERGE_K = 64
MERGE_TOKENS = 512
MERGE_COEFS = (0.75, 1.0, 1.25)

MergeKey = tuple[float, int, float] | None


class _MergeChoiceKinds(TypedDict, total=False):
    """Distinguish finite-window candidates only when the optional search is enabled."""

    merge_choice_kinds: dict[str, Literal['initialized_ema', 'normalized', 'no_move']]


class MergeResult(_MergeChoiceKinds):
    """Report losses and the selected rho, mean decay and coefficient per part."""

    holdout_plain: float
    holdout_merged: float
    merge_choices: dict[str, tuple[float, float, float] | None]


@dataclass
class MergeTrajectory:
    """Hold initialized EMAs and the optional finite-window reconstruction inputs."""

    means: list[list[Tensor]]
    first_snapshot: list[Tensor] | None = None
    snapshot_count: int = 0

    @torch.no_grad()
    def observe(self, model: nn.Module, *, normalized: bool) -> None:
        """Append one eligible training snapshot without changing PR40's EMA arithmetic."""
        if not self.means:
            self.means = [
                [parameter.detach().float().clone() for parameter in model.parameters()] for _ in MERGE_DECAYS
            ]
        else:
            for decay, mean in zip(MERGE_DECAYS, self.means, strict=True):
                for average, parameter in zip(mean, model.parameters(), strict=True):
                    average.lerp_(parameter.detach().float(), decay)
        if normalized:
            if self.first_snapshot is None:
                self.first_snapshot = [parameter.detach().float().clone() for parameter in model.parameters()]
            self.snapshot_count += 1


class _Eigenvectors(Protocol):
    """Type Torch's default Hermitian eigendecomposition at its external boundary."""

    def eigh(self, matrix: Tensor) -> tuple[Tensor, Tensor]: ...


class _Broadcast(Protocol):
    """Type the synchronous default-group tensor broadcast used by the merge."""

    def broadcast(self, tensor: Tensor, src: int) -> None: ...


@torch.no_grad()
def holdout_loss(model: nn.Module, rows: Tensor, rank: int, world: int) -> float:
    """Average next-token loss on the supplied training rows across ranks."""
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for host_batch in cast('TensorOperations', rows[rank::world]).split(8):
        batch = host_batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
            loss = cast('Tensor', model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum'))
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    if torch.distributed.is_initialized():
        torch_operations.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def merge_part(name: str) -> str:
    """Separate block matrices and tables while grouping all VE gates together."""
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])
    return 've_gates' if 've_gate' in name else name


def _covariance_hook(
    key: str, accumulated: dict[str, Tensor]
) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
    def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
        inputs_flat = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
        covariance = (inputs_flat.T @ inputs_flat).double()
        previous = accumulated.get(key)
        accumulated[key] = covariance if previous is None else previous + covariance

    return hook


@torch.no_grad()
def curvature_bases(
    model: GPT, rows: Tensor, rank: int, world: int, k: int = MERGE_K
) -> tuple[dict[str, Tensor], Tensor]:
    """Broadcast top input-covariance directions and the held-out token frequency order.

    The fixed model uses 64 directions; small CPU diagnostics can explicitly
    request a smaller k. Hooks are removed even when a held-out forward fails.
    """
    device = next(model.parameters()).device
    layers = [
        (name + '.weight', module)
        for name, module in cast('NamedModules', model).named_modules()
        if isinstance(module, nn.Linear) and 've_gate' not in name
    ]
    bases = {key: torch.zeros(module.weight.shape[1], k, device=device) for key, module in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        accumulated: dict[str, Tensor] = {}
        hooks: list[RemovableHandle] = []
        counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)
        try:
            for key, module in layers:
                hooks.append(module.register_forward_hook(_covariance_hook(key, accumulated)))
            for host_batch in cast('TensorOperations', rows).split(4):
                batch = host_batch.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                    model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
                counts += torch.bincount(batch[:, :-1].reshape(-1).long(), minlength=len(counts)).double()
        finally:
            for hook in hooks:
                hook.remove()
        for key, _ in layers:
            bases[key] = (
                cast('_Eigenvectors', torch.linalg).eigh(accumulated[key])[1].flip(1)[:, :k].float().contiguous()
            )
        order = torch.argsort(counts, descending=True)
    if world > 1:
        for key, _ in layers:
            cast('_Broadcast', torch.distributed).broadcast(bases[key], 0)
        cast('_Broadcast', torch.distributed).broadcast(order, 0)
    return bases, order


def subspace_move(name: str, move: Tensor, bases: dict[str, Tensor], order: Tensor) -> Tensor:
    """Project linear moves on their input axis and table moves on frequent rows."""
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


@dataclass
class MergeState:
    """Keep the immutable final weights and candidate arithmetic for one end search."""

    model: nn.Module
    named: list[tuple[str, nn.Parameter]]
    final: list[Tensor]
    means: list[list[Tensor]]
    bases: dict[str, Tensor]
    order: Tensor
    rows: Tensor
    rank: int
    world: int
    cache: dict[tuple[int, float, int], Tensor] = field(default_factory=dict[tuple[int, float, int], Tensor])
    normalized_from: tuple[int, ...] = ()

    def add_normalized(self, first_snapshot: list[Tensor] | None, snapshot_count: int) -> None:
        """Append the two slow finite-window means without changing initialized EMAs.

        Snapshot count includes the first snapshot that initialized every EMA.
        One snapshot has no distinct finite-window candidate, so it is skipped.
        """
        if snapshot_count < 0 or (first_snapshot is None) != (snapshot_count == 0):
            message = 'Normalized merge requires both a first snapshot and a positive snapshot count'
            raise ValueError(message)
        if first_snapshot is None or snapshot_count == 1:
            return
        if len(self.means) != len(MERGE_DECAYS) or len(first_snapshot) != len(self.final):
            message = 'Normalized merge requires all EMA decays and one first snapshot tensor per parameter'
            raise ValueError(message)
        selected = (MERGE_DECAYS.index(0.02), MERGE_DECAYS.index(0.01))
        normalized: list[list[Tensor]] = []
        for mean_index in selected:
            mean = self.means[mean_index]
            if len(mean) != len(self.final):
                message = 'Normalized merge EMA tensor count does not match the model'
                raise ValueError(message)
            log_tail = snapshot_count * math.log1p(-MERGE_DECAYS[mean_index])
            denominator = -math.expm1(log_tail)
            tensors: list[Tensor] = []
            for initialized, first, final in zip(mean, first_snapshot, self.final, strict=True):
                if initialized.shape != final.shape or first.shape != final.shape:
                    message = 'Normalized merge snapshot and EMA shapes must match model parameters'
                    raise ValueError(message)
                # Center on the first snapshot to avoid cancellation of its common offset.
                tensors.append((initialized - first).div_(denominator).add_(first))
            normalized.append(tensors)
        self.means = [*self.means, *normalized]
        self.normalized_from = selected

    def value(self, index: int, key: MergeKey) -> Tensor:
        """Construct a candidate from the original final tensor, never the live state."""
        if key is None:
            return self.final[index]
        rho, mean_index, coefficient = key
        cache_key = (index, rho, mean_index)
        if cache_key not in self.cache:
            full = self.means[mean_index][index] - self.final[index]
            if rho == 1.0:
                move = full
            else:
                inside = subspace_move(self.named[index][0], full, self.bases, self.order)
                move = inside + rho * (full - inside)
            self.cache[cache_key] = move
        return self.final[index] + coefficient * self.cache[cache_key]

    def load(self, choice: dict[str, MergeKey]) -> None:
        """Load each part's candidate with the parameter's original storage dtype."""
        for index, (name, parameter) in enumerate(self.named):
            parameter.copy_(self.value(index, choice[merge_part(name)]).to(parameter.dtype))

    def loss(self) -> float:
        """Score the currently loaded state on held-out training rows only."""
        return holdout_loss(self.model, self.rows, self.rank, self.world)


def _coordinate_sweep(
    state: MergeState,
    choice: dict[str, MergeKey],
    best: float,
    candidates: dict[str, list[MergeKey]],
) -> tuple[dict[str, MergeKey], float, dict[str, list[MergeKey]]]:
    ranked: dict[str, list[MergeKey]] = {}
    for part, keys in candidates.items():
        tried: list[tuple[float, MergeKey]] = []
        for key in keys:
            if key == choice[part]:
                tried.append((best, key))
                continue
            trial = dict(choice)
            trial[part] = key
            state.load(trial)
            loss = state.loss()
            tried.append((loss, key))
            if loss < best:
                best, choice = loss, trial
        ranked[part] = [key for _, key in sorted(tried, key=lambda item: item[0])]
    return choice, best, ranked


@torch.no_grad()
def joint_parts(state: MergeState, *, sweeps: int = 2, shortlist: int = 8) -> MergeResult:
    """Select the best global candidate, then run full and shortlisted coordinate sweeps."""
    parts = sorted({merge_part(name) for name, _ in state.named})
    keys: list[MergeKey] = [None]
    original_count = len(state.means) - len(state.normalized_from)
    # Preserve the entire original candidate order, including strict-tie priority.
    for mean_indices in (range(original_count), range(original_count, len(state.means))):
        keys.extend(
            (rho, mean_index, coefficient)
            for rho in MERGE_RHOS
            for mean_index in mean_indices
            for coefficient in MERGE_COEFS
        )
    plain = state.loss()
    best_key: MergeKey = None
    best = plain
    for key in keys[1:]:
        state.load(dict.fromkeys(parts, key))
        loss = state.loss()
        if loss < best:
            best_key, best = key, loss
    choice = dict.fromkeys(parts, best_key)
    candidates = dict.fromkeys(parts, keys)
    for sweep in range(sweeps):
        choice, best, ranked = _coordinate_sweep(state, choice, best, candidates)
        if sweep == 0:
            candidates = {part: ranked[part][:shortlist] for part in parts}
    state.load(choice)
    decays = (*MERGE_DECAYS[:original_count], *(MERGE_DECAYS[index] for index in state.normalized_from))
    result: MergeResult = {
        'holdout_plain': plain,
        'holdout_merged': best,
        'merge_choices': {
            part: None if key is None else (key[0], decays[key[1]], key[2]) for part, key in choice.items()
        },
    }
    if state.normalized_from:
        result['merge_choice_kinds'] = {
            part: 'no_move' if key is None else 'normalized' if key[1] >= original_count else 'initialized_ema'
            for part, key in choice.items()
        }
    return result


@torch.no_grad()
def end_merge(model: GPT, trajectory: MergeTrajectory, rows: Tensor, rank: int, world: int) -> MergeResult:
    """Search initialized EMAs and optional normalized windows on held-out training rows.

    Supply the first EMA snapshot and total snapshot count together to append
    normalized .02/.01 candidates. Omission preserves PR40's original search.
    """
    named = list(model.named_parameters())
    final = [parameter.detach().float().clone() for _, parameter in named]
    bases, order = curvature_bases(model, rows, rank, world)
    state = MergeState(
        model=model,
        named=named,
        final=final,
        means=trajectory.means,
        bases=bases,
        order=order,
        rows=rows,
        rank=rank,
        world=world,
    )
    state.add_normalized(trajectory.first_snapshot, trajectory.snapshot_count)
    return joint_parts(state)


class _Arguments(argparse.Namespace):
    data_dir: Path
    output: Path
    steps: int | None
    chain_tokens: int
    key_chunk: int
    key_path_tokens: int
    future_tokens: int
    future_chunk: int
    future_layers: int
    future_draws: int
    future_hidden_draws: int | None
    merge_normalized: bool


@dataclass
class TrainingContext:
    """Own the fixed data schedule and the rank's immutable execution topology."""

    args: _Arguments
    settings: Settings
    device: torch.device
    rank: int
    world: int
    training: Tensor
    holdout_rows: Tensor
    schedule: list[Stage]
    split: int
    group: dist.ProcessGroup | None
    cut_split: int
    cut_group: dist.ProcessGroup | None

    def estimator(self, model: GPT, phase: Phase) -> Boost:
        """Create one fresh probe generator for the selected phase."""
        if phase == 'boost':
            return Boost(model, self.settings, SEED, self.rank, self.world, self.split, self.group)
        return Cut(model, self.settings, SEED, self.rank, self.world, self.cut_split, self.cut_group)

    def rank_slice(self, batch: Tensor, phase: Phase) -> Tensor:
        """Assign disjoint sequences to groups, sharing only each group's probe work."""
        group_size = self.split if phase == 'boost' else self.cut_split
        per = len(batch) // (self.world // group_size)
        first = (self.rank // group_size) * per
        return batch[first : first + per].to(self.device)

    def warmup(self, phases: dict[Phase, int]) -> None:
        """Compile each active phase on random tokens, discarding all warmed model/optimizer/RNG state."""
        noise = torch.randint(
            0,
            self.settings.model.vocab_size,
            (max(self.settings.boost_batch, self.settings.cut_batch), self.settings.model.sequence_len + 1),
            generator=torch.Generator().manual_seed(SEED),
        )
        for phase, horizon in phases.items():
            model = make_model(
                self.device,
                SEED,
                self.settings.model,
                self.settings.projection_std,
                self.settings.head_std,
            )
            optimizer, estimator = make_optimizer(model, self.settings, horizon), self.estimator(model, phase)
            size = self.settings.boost_batch if phase == 'boost' else self.settings.cut_batch
            batch = self.rank_slice(noise[:size], phase)
            try:
                estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            finally:
                estimator.close()
            del model, optimizer, estimator
        torch.cuda.empty_cache()

    def write_config(self, steps: int) -> None:
        """Record the actual recipe before training without overwriting any prior run."""
        if self.rank != 0:
            return
        output = self.args.output
        if output.exists() and any(output.iterdir()):
            message = f'Choose an empty output directory: {output}'
            raise FileExistsError(message)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            'script': __file__,
            'model_file': str(MODEL_FILE),
            'world': self.world,
            'seed': SEED,
            'recipe': asdict(self.settings),
            'steps': steps,
            'epoch_updates': len(self.schedule),
            'train_rows': len(self.training),
            'holdout_rows': len(self.holdout_rows),
            'diagnostic_prefix': steps < len(self.schedule),
            'upstream': UPSTREAM_REVISION,
            'additions': [
                'linear_path_scalars',
                'input_key_probes',
                'input_gate_probes',
                'independent_future_hidden_query_embedding_probes',
                'cut_batch4_split2',
                'recipe_11',
                'end_merge',
            ],
            'resume': None,
            'torch_version': torch.__version__,
        }
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')


def training_context(
    args: _Arguments, settings: Settings, device: torch.device, rank: int, world: int
) -> TrainingContext:
    """Reserve disjoint merge-only rows and establish both phases' probe-sharing groups."""
    if world not in (1, 2, 4, 8):
        message = 'Stage training supports 1, 2, 4 or 8 GPUs'
        raise ValueError(message)
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout_rows, training = training[-HOLDOUT :], training[: -HOLDOUT]
    schedule = stage_schedule(len(training), settings)
    split = max(1, world // settings.boost_batch)
    groups = (
        [torch_operations.distributed.new_group(list(range(index, index + split))) for index in range(0, world, split)]
        if split > 1
        else [None]
    )
    group = cast('dist.ProcessGroup | None', groups[rank // split] if split > 1 else None)
    cut_split = max(1, world // settings.cut_batch)
    cut_groups = (
        groups
        if cut_split == split
        else [
            torch_operations.distributed.new_group(list(range(index, index + cut_split)))
            for index in range(0, world, cut_split)
        ]
        if cut_split > 1
        else [None]
    )
    cut_group = cast('dist.ProcessGroup | None', cut_groups[rank // cut_split] if cut_split > 1 else None)
    return TrainingContext(
        args, settings, device, rank, world, training, holdout_rows, schedule, split, group, cut_split, cut_group
    )


def run_training(context: TrainingContext, phases: dict[Phase, int], steps: int, log: TextIO | None) -> None:
    """Execute fresh-state updates, tracking only the end merge's training trajectory."""
    settings = context.settings
    phase = context.schedule[0][0]
    model = make_model(
        context.device, SEED, settings.model, settings.projection_std, settings.head_std
    )
    optimizer = make_optimizer(model, settings, phases[phase])
    estimator = context.estimator(model, phase)
    trajectory = MergeTrajectory([])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, elapsed, end = time.perf_counter(), 0.0, 0
    try:
        for step in range(1, steps + 1):
            current, phase_step, begin, end, horizon = context.schedule[step - 1]
            if current != phase:
                estimator.close()
                next_optimizer = make_optimizer(model, settings, horizon)
                handoff_optimizer(optimizer, next_optimizer, phase_step)
                optimizer, estimator, phase = next_optimizer, context.estimator(model, current), current
            batch = context.rank_slice(context.training[begin:end], phase)
            set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            record = {
                'step': step,
                'phase': phase,
                'phase_step': phase_step,
                'seqs': end,
                'tokens': end * settings.model.sequence_len,
                'train_loss': loss,
                'train_seconds': round(elapsed, 2),
                'peak_cuda_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
            }
            if log is not None:
                line = json.dumps(record)
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
            if (step % MERGE_EVERY == 0 or step == steps) and begin / len(context.training) >= MERGE_FROM:
                trajectory.observe(model, normalized=settings.merge_normalized)
        finish_training(context, model, trajectory, steps, end, start, elapsed)
    finally:
        estimator.close()


def finish_training(
    context: TrainingContext,
    model: GPT,
    trajectory: MergeTrajectory,
    steps: int,
    end: int,
    start: float,
    elapsed: float,
) -> None:
    """Time the holdout-only end merge, excluding both final validation evaluations."""
    extra: dict[str, object] = {}
    if trajectory.means:
        plain = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        pause = time.perf_counter()
        extra['val_loss_plain'] = evaluate(plain, context.args.data_dir, model_file=MODEL_FILE)
        paused = time.perf_counter() - pause
        merge_start = time.perf_counter()
        extra.update(end_merge(model, trajectory, context.holdout_rows, context.rank, context.world))
        torch.cuda.synchronize()
        extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
        elapsed = time.perf_counter() - start - paused
        if context.rank == 0:
            torch.save(plain, context.args.output / 'final_plain.pt')
        trajectory.means.clear()
        trajectory.first_snapshot = None
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    result = {
        'val_loss': evaluate(state, context.args.data_dir, model_file=MODEL_FILE),
        'train_seconds': round(elapsed, 2),
        'steps': steps,
        'epoch_updates': len(context.schedule),
        'diagnostic_prefix': steps < len(context.schedule),
        'completed_epoch': steps == len(context.schedule),
        'seqs': end,
        'tokens': end * context.settings.model.sequence_len,
        'train_rows': len(context.training),
        'holdout_rows': len(context.holdout_rows),
        'upstream': UPSTREAM_REVISION,
        'estimator': context.settings.estimator,
        'model_file': str(MODEL_FILE),
        'peak_cuda_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
        **extra,
    }
    if context.rank == 0:
        torch.save(state, context.args.output / 'final.pt')
        (context.args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result), flush=True)


def train_stages(args: _Arguments, settings: Settings, device: torch.device, rank: int, world: int) -> None:
    """Train the fixed epoch from scratch, with warmup excluded from model/RNG state and timing."""
    context = training_context(args, settings, device, rank, world)
    steps = min(args.steps or len(context.schedule), len(context.schedule))
    context.write_config(steps)
    phases: dict[Phase, int] = {phase: horizon for phase, _, _, _, horizon in context.schedule}
    context.warmup(phases)
    log = (args.output / 'metrics.jsonl').open('a', encoding='utf-8') if rank == 0 else None
    try:
        run_training(context, phases, steps, log)
    finally:
        if log is not None:
            log.close()


def main() -> None:
    """Run the Track 3 record recipe; options change only execution chunks, budgets or the merge search."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/arch'))
    parser.add_argument('--steps', type=int, help='Diagnostic prefix; preserves the full-epoch phase/LR schedule.')
    parser.add_argument('--chain-tokens', type=int, default=512)
    parser.add_argument('--key-chunk', type=int, default=32)
    parser.add_argument('--key-path-tokens', type=int, default=64)
    parser.add_argument('--future-tokens', type=int, default=128)
    parser.add_argument('--future-chunk', type=int, default=512)
    parser.add_argument(
        '--future-draws',
        type=int,
        choices=(64, 128, 256, 512),
        default=128,
        help='Q/WTE source directions; also the hidden budget unless separately specified.',
    )
    parser.add_argument(
        '--future-layers', type=int, default=4, help='Hidden/Q source blocks; 4 is the accepted baseline.'
    )
    parser.add_argument(
        '--future-hidden-draws',
        type=int,
        choices=(64, 128, 256, 512, 1024, 2048),
        default=None,
        help='Independent hidden-source directions; defaults to --future-draws.',
    )
    parser.add_argument(
        '--merge-normalized', action='store_true', help='Add normalized slow trajectory averages at the end merge.'
    )
    args = parser.parse_args(namespace=_Arguments())
    if args.steps is not None and args.steps < 1:
        parser.error('steps must be positive')
    settings = Settings(
        chain_tokens=args.chain_tokens,
        key_chunk=args.key_chunk,
        key_path_tokens=args.key_path_tokens,
        future_tokens=args.future_tokens,
        future_chunk=args.future_chunk,
        future_layers=args.future_layers,
        future_draws=args.future_draws,
        future_hidden_draws=args.future_draws if args.future_hidden_draws is None else args.future_hidden_draws,
        # Every future-probe family is measured over complete cycles.
        future_exact=('hidden', 'query', 'wte'),
        merge_normalized=args.merge_normalized,
    )
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    try:
        train_stages(args, settings, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
