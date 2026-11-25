"""The Cut's bf16 gradient wire: q, k, v, c_proj and MLP weight gradients as scaled products, the rest by one kernel."""
from collections import defaultdict
import ctypes

import torch
import triton
import triton.language as tl

READERS = ('c_q', 'c_k', 'c_v')
_cublas = None


def _blas():
    """cublasGemmBatchedEx and the pointer-mode getter and setter of the cuBLAS library torch has loaded."""
    global _cublas
    if _cublas is None:
        library = ctypes.CDLL('libcublas.so.12')
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        gemm = library.cublasGemmBatchedEx
        gemm.argtypes = [ptr, integer, integer, integer, integer, integer,
                         ptr, ptr, integer, integer, ptr, integer, integer,
                         ptr, ptr, integer, integer, integer, integer, integer]
        get_mode = library.cublasGetPointerMode_v2
        get_mode.argtypes = [ptr, ctypes.POINTER(integer)]
        set_mode = library.cublasSetPointerMode_v2
        set_mode.argtypes = [ptr, integer]
        gemm.restype = get_mode.restype = set_mode.restype = integer
        _cublas = gemm, get_mode, set_mode
    return _cublas


# The Cut update's gradient wire for the tensors the products below do not write: each gradient divided by this rank's
# target count times the world size, cast to bf16 and laid at its offset, one launch per shape and layout.
@triton.jit
def _pack(Inputs, Wire, Divisor, Offsets: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(1)
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = at // C * S0 + at % C * S1
    divisor = tl.load(Divisor).to(tl.float32)
    for j in tl.static_range(GROUP):
        if group == j:
            value = tl.load(Inputs[j] + source, at < N, 0).to(tl.float32) / divisor
            tl.store(Wire + Offsets[j] + at, value.to(tl.bfloat16), at < N)


class Layout:
    def __init__(self, order):
        offsets = {}
        size = 0
        for name, parameter in order:
            offsets[name] = size
            size += parameter.numel()
        self.size = size
        names = [f'transformer.h.{layer}.attn.{kind}.weight'
                 for layer in range(8) for kind in READERS]
        self.readers = tuple(offsets[name] for name in names)
        writer_names = [f'transformer.h.{i}.attn.c_proj.weight' for i in range(8)]
        self.writers = tuple(offsets[name] for name in writer_names)
        names += writer_names
        self.rest = [(name, offset) for name, offset in offsets.items() if name not in names]

    def allocate(self, tensor):
        return torch.empty(self.size, device=tensor.device, dtype=torch.bfloat16)

    def pack_rest(self, grads, divisor, wire):
        groups = defaultdict(list)
        for name, offset in self.rest:
            gradient = grads[name]
            if gradient.ndim == 1:
                cols, row_stride, col_stride = gradient.numel(), 0, gradient.stride(0)
            else:
                cols, row_stride, col_stride = gradient.shape[1], *gradient.stride()
            groups[gradient.numel(), cols, row_stride, col_stride, gradient.dtype].append((gradient, offset))
        for (size, cols, row_stride, col_stride, _), entries in groups.items():
            tensors, offsets = zip(*entries)
            _pack[(triton.cdiv(size, 4096), len(entries))](
                tensors, wire, divisor, offsets, size, cols, row_stride, col_stride,
                len(entries), 4096, num_warps=4)


@triton.jit
def _addresses(Inputs, Credits, Destinations, Table, BATCH: tl.constexpr):
    for j in tl.static_range(BATCH):
        tl.store(Table + j, Inputs[j].to(tl.int64))
        tl.store(Table + BATCH + j, Credits[j].to(tl.int64))
        tl.store(Table + 2 * BATCH + j, Destinations[j].to(tl.int64))


def _products(inputs, credits, outputs, tokens, width, d, credit_stride, alpha, zero):
    """outputs[j] = bf16(alpha * credits[j]^T @ inputs[j]) for every j by one cublasGemmBatchedEx call: bf16
    operands and result (code 14), fp32 compute type (code 68), alpha and beta = 0 read on the device."""
    count = len(outputs)
    table = torch.empty(3 * count, device=alpha.device, dtype=torch.int64)
    _addresses[(1,)](tuple(inputs), tuple(credits), tuple(outputs), table, count, num_warps=1)
    gemm, get_mode, set_mode = _blas()
    handle = torch.cuda.current_blas_handle()
    original = ctypes.c_int()
    if get_mode(handle, ctypes.byref(original)) or set_mode(handle, 1):
        raise RuntimeError('cannot set the cuBLAS pointer mode')
    try:
        addr = table.data_ptr()
        status = gemm(handle, 0, 1, d, width, tokens,
                      alpha.data_ptr(), addr, 14, d, addr + 8 * count, 14, credit_stride,
                      zero.data_ptr(), addr + 16 * count, 14, d, count, 68, -1)
        if status:
            raise RuntimeError(f'cublasGemmBatchedEx returned status {status}')
    finally:
        if set_mode(handle, original.value):
            raise RuntimeError('cannot restore the cuBLAS pointer mode')


def write_attention(qk, value, inputs, divisor, wire, reader_offsets, writers):
    layers, tokens, double_width = qk.shape
    d = double_width // 2
    alpha = divisor.float().reciprocal()
    zero = torch.zeros_like(alpha)
    qk_dest = [wire.narrow(0, reader_offsets[3 * i], 2 * d * d) for i in range(layers)]
    _products(inputs, qk.unbind(0), qk_dest, tokens, 2 * d, d, 2 * d, alpha, zero)
    writer_credits, writer_inputs, writer_offsets = writers
    credits = [*value.unbind(0), *writer_credits.unbind(0)]
    destinations = [wire.narrow(0, reader_offsets[3 * i + 2], d * d) for i in range(layers)]
    destinations.extend(wire.narrow(0, off, d * d) for off in writer_offsets)
    _products([*inputs, *writer_inputs], credits, destinations, tokens, d, d, d, alpha, zero)


def full_mlp_products(writers, activations, readers, inputs, divisor):
    """Every layer's c_fc and c_proj gradients times 1 / divisor in one new bf16 wire, (layers, 2, -1), c_fc first."""
    layers, tokens, features = readers.shape
    width = writers.shape[-1]
    wire = readers.new_empty(layers * 2 * features * width)
    owner = wire.view(layers, 2, -1)
    alpha = divisor.float().reciprocal()
    zero = torch.zeros_like(alpha)
    _products([x.contiguous() for x in inputs], readers.unbind(), owner[:, 0].unbind(),
              tokens, features, width, features, alpha, zero)
    _products([x.contiguous() for x in activations], writers.unbind(), owner[:, 1].unbind(),
              tokens, width, features, width, alpha, zero)
    return wire
