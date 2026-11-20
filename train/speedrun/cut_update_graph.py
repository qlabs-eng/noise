"""The Cut update's AdamW part as captured CUDA graphs, one per packed gradient wire."""
import math
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _adam_graph(P, G, M, V, Table, Clock, N: tl.constexpr, C: tl.constexpr,
                COL: tl.constexpr, beta1: tl.constexpr, beta2: tl.constexpr,
                eps: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.load(Clock)
    step_size = tl.load(Table + t * C + COL)
    bc2_sqrt = tl.load(Table + t * C + COL + 1)
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


class AdamGraphs:
    def __init__(self, cut, optimizer, schedule, settings, horizon):
        self.cut, self.optimizer = cut, optimizer
        self.anvil, self.adam = optimizer.optimizers
        a = self.anvil
        self.device = a.rank_stores[0].device
        self.clock = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.health = torch.ones((), dtype=torch.bool, device=self.device)
        self.side = torch.cuda.Stream()  # (unused, like check: kept so that later streams get the same pool slots)
        self.check = torch.cuda.Stream()
        self.capture_stream = torch.cuda.Stream()
        self.graphs = {}
        self.offsets = {}
        offset = 0
        for _, p in getattr(cut, '_wire_order', cut._flat_order):  # (the wire's layout)
            self.offsets[id(p)] = offset
            offset += p.numel()
        a.receive_rank_major(torch.bfloat16)
        if not hasattr(a, '_scalars'):
            a._scalars = a.scalar_schedule[0].clone()
        self.adam_meta = []
        for gi, group in enumerate(self.adam.param_groups):
            for p in group['params']:
                state = self.adam.state[p]
                if not state:
                    state.update(step=torch.tensor(0.0),
                                 exp_avg=torch.zeros_like(p, memory_format=torch.preserve_format),
                                 exp_avg_sq=torch.zeros_like(p, memory_format=torch.preserve_format))
                assert state['step'].item() == 0
                self.adam_meta.append((p, state['exp_avg'], state['exp_avg_sq'], gi, group))
        rows = []
        for t, group_steps in enumerate(schedule, 1):
            step = sum(group_steps) / len(group_steps)
            factor = settings.final + (1 - settings.final) * min(1.0,
                ((horizon - step + 1) / horizon) / settings.warmdown)
            row = []
            for group in self.adam.param_groups:
                b1, b2 = group['betas']
                lr = group['lr'] * factor * settings.wide_lr_scale
                bc1 = float(np.float32(1 - b1 ** t))
                row += [float(np.float32(lr / bc1)), float(np.float32(math.sqrt(1 - b2 ** t)))]
            rows.append(row)
        self.table = torch.tensor(rows, dtype=torch.float32, device=self.device)

    def tensors(self):
        xs = list(self.cut.model.parameters())
        for _, state, *_ in self.anvil.batches:
            xs.extend(state.values())
        for _, m, v, _, _ in self.adam_meta:
            xs.extend((m, v))
        xs.extend((self.anvil._scalars, self.clock, self.health))
        return xs

    def snapshot(self):
        return [x.clone() for x in self.tensors()]

    def restore(self, values):
        for x, value in zip(self.tensors(), values):
            x.copy_(value)

    def capture(self, packed_buffers):
        saved = self.snapshot()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for packed in packed_buffers:
                wire = packed[1]
                if wire.data_ptr() in self.graphs:
                    continue
                original_wire = wire.clone()
                self.capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.capture_stream):
                    for _ in range(3):
                        self.restore(saved)
                        wire.copy_(original_wire)
                        self.body(wire)
                torch.cuda.current_stream().wait_stream(self.capture_stream)
                torch.cuda.synchronize()
                self.restore(saved)
                wire.copy_(original_wire)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.capture_stream):
                    self.body(wire)
                self.graphs[wire.data_ptr()] = graph
                self.restore(saved)
                wire.copy_(original_wire)
            torch.cuda.synchronize()
        finally:
            self.restore(saved)
            self.anvil.shard_grads = None
            torch.use_deterministic_algorithms(enabled, warn_only=warn)
        self.anvil.t = 0

    def body(self, wire):
        for p, m, v, gi, group in self.adam_meta:
            offset = self.offsets[id(p)]
            grad = wire[offset:offset + p.numel()]
            b1, b2 = group['betas']
            _adam_graph[(triton.cdiv(p.numel(), 2048),)](p, grad, m, v, self.table, self.clock,
                p.numel(), self.table.shape[1], 2 * gi, b1, b2, group['eps'], BLOCK=2048, num_warps=8)
        self.clock.add_(1)
