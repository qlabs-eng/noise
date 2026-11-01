"""Warp-specialized (Gluon, Hopper) value credits: the numbers of moment_kernels._value_credit_kernel (bitwise).

The Triton kernel runs all 8 warps of a CTA in lockstep over each 128-receiver block: fp32 query and hub tiles copied
into shared memory, the K = 64 tf32 logits GEMM, its wait, the probabilities, the hub's bf16 high and low parts formed
in registers and stored back (a CTA barrier), the two bf16 GEMMs, their wait. The tensor cores idle under the
elementwise work and the loads, and every block pays the causal / window mask.

Here CTAs (persistent: one per SM) take (row, 128-target tile) items with four partitions:
- two consumer warpgroups (64 targets each): per block one GEMM issue, the previous block's hi and lo GEMMs
  (m64n64k16, A from registers: the reference's accumulator chain) then this block's tf32 logits GEMM (m64n128k8 in
  two K = 32 halves, the keys held in registers: the reference's instruction and k-steps), one wait, then the
  probabilities (the reference's expression; the mask only on the diagonal and window-edge blocks, the only ones it
  cuts). With PINGPONG they issue in turns, each one's elementwise work under the other's GEMMs;
- a convert warpgroup: each fp32 hub block from shared memory into its bf16 high and low parts in the wgmma B layout
  (once per block for both consumers; by 32-row chunks, so that its few registers leave the consumers theirs);
- a load warp: TMA loads of the key tile and the rings of query tiles (with their lse) and fp32 hub tiles.
Persistent CTAs deal the items in a zigzag, heaviest tiles first as the reference's grid, in runs of consecutive tiles
of one row (neighbouring CTAs read the same receiver blocks steps apart: L2 hits instead of DRAM rereads).
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, tma, warpgroup_mma,
                                                              warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# query ring depth, hi / lo slots, fp32 hub slots, register budgets (the first consumer gets the rest), GEMM turns
CONFIG = dict(STAGES=2, SLOTS=2, HUB_SLOTS=2, CONSUMER_REGS=224, CONVERT_REGS=56, LOAD_REGS=24, PINGPONG=1)


@gluon.jit
def _item_count(ITEMS, PERSISTENT: gl.constexpr):
    """How many items this CTA takes (persistent: of the items dealt in a zigzag; else one)."""
    if PERSISTENT:
        p = gl.program_id(0)
        P = gl.num_programs(0)
        R = ITEMS // P
        return R + gl.where(gl.where(R % 2 == 0, p, P - 1 - p) < ITEMS % P, 1, 0)
    else:
        return 1


@gluon.jit
def _item(k, ROWS, T: gl.constexpr, WINDOW: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
          PERSISTENT: gl.constexpr, RUN: gl.constexpr):
    """This CTA's k-th item: its row, first target, first receiver block and block count. Items in the reference's
    order (heaviest tiles first) by runs of RUN consecutive tiles of one row."""
    w = gl.program_id(0)
    if PERSISTENT:
        P = gl.num_programs(0)
        w = k * P + gl.where(k % 2 == 0, w, P - 1 - w)
    u = w // RUN
    tile = (u // ROWS) * RUN + w % RUN
    first = tile * BT // BS
    last = gl.minimum((tile * BT + BT - 1 + WINDOW) // BS + 1, T // BS)
    return u % ROWS, tile * BT, first, last - first


# --- consumers -----------------------------------------------------------------------------------------------------

@gluon.jit
def _turn(pp, j, PART: gl.constexpr, PINGPONG: gl.constexpr):
    """Wait for this consumer's j-th GEMM turn (consumer 0 first)."""
    if PINGPONG:
        if PART == 0:
            mbarrier.wait(pp.index(0), (j & 1) ^ 1)
        else:
            mbarrier.wait(pp.index(1), j & 1)


@gluon.jit
def _pass(pp, PART: gl.constexpr, PINGPONG: gl.constexpr):
    if PINGPONG:
        mbarrier.arrive(pp.index(1 - PART))


@gluon.jit
def _logits(ka0, ka1, q_s, l_s, qready, lt, r, S: gl.constexpr, PART: gl.constexpr, mma: gl.constexpr):
    """Ring block r's logits GEMM issued (keys ka0 | ka1 in registers; the previous logits as its unread accumulator:
    no instruction writes the registers of an accumulator while other GEMMs run, else ptxas serializes every wgmma),
    and its lse loaded."""
    s = r % S
    mbarrier.wait(qready.index(s), (r // S) & 1)
    lt = warpgroup_mma(ka0, q_s.index(2 * s).permute((1, 0)), lt, use_acc=False, is_async=True)
    lt = warpgroup_mma(ka1, q_s.index(2 * s + 1).permute((1, 0)), lt, is_async=True)
    return lt, l_s.index(s).load(gl.SliceLayout(0, mma))


@gluon.jit
def _probabilities(lt, lse, it, t0, target, WINDOW: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
                   mma: gl.constexpr, opb: gl.constexpr):
    """Receiver block it's bf16 probabilities (the reference's expression; masked only where the mask cuts)."""
    receiver = it * BS + gl.arange(0, BS, gl.SliceLayout(0, mma))
    logits = lt * .125
    prob = gl.exp(logits - lse[None, :])
    if (it * BS < t0 + BT - 1) | (it * BS + BS - 1 - t0 > WINDOW):
        lag = receiver[None, :] - target[:, None]
        prob = gl.where((lag >= 0) & (lag <= WINDOW), prob, 0.)
    return gl.convert_layout(prob.to(gl.bfloat16), opb)


@gluon.jit
def _values(p, h_s, hready, acc, r, SH: gl.constexpr):
    """Ring block r's hi then lo GEMMs into the accumulator chain (the reference's order)."""
    h = r % SH
    mbarrier.wait(hready.index(h), (r // SH) & 1)
    acc = warpgroup_mma(p, h_s.index(2 * h), acc, is_async=True)
    return warpgroup_mma(p, h_s.index(2 * h + 1), acc, is_async=True)


@gluon.jit
def _consumer(k_s, q_s, l_s, h_s, kready, kfree, qready, qfree, hready, hfree, pp, out_ptr, out_map, OUT_LAYER,
              OUT_TOKEN, ITEMS, ROWS, T: gl.constexpr, WINDOW: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
              C: gl.constexpr, OUT_H: gl.constexpr, OUT_MAPPED: gl.constexpr, S: gl.constexpr, SH: gl.constexpr,
              PART: gl.constexpr, PINGPONG: gl.constexpr, PERSISTENT: gl.constexpr, RUN: gl.constexpr):
    """Targets [64 PART, 64 PART + 64) of each item's tile."""
    HALF: gl.constexpr = BT // 2
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BS, 8])
    mmab: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, C, 16])
    opb: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mmab, k_width=2)
    opk: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=1)
    lt = gl.zeros([HALF, BS], gl.float32, mma)
    g = 0  # ring blocks done
    j = 0  # GEMM turns done
    for k in range(_item_count(ITEMS, PERSISTENT)):
        head, t0, first, n = _item(k, ROWS, T, WINDOW, BT, BS, PERSISTENT, RUN)
        target = t0 + PART * HALF + gl.arange(0, HALF, gl.SliceLayout(1, mma))
        acc = gl.zeros([HALF, C], gl.float32, mmab)
        mbarrier.wait(kready.index(0), k & 1)
        ka0 = k_s.index(2 * PART).load(opk)
        ka1 = k_s.index(2 * PART + 1).load(opk)
        _turn(pp, j, PART, PINGPONG)
        lt, lse = _logits(ka0, ka1, q_s, l_s, qready, lt, g, S, PART, mma)
        _pass(pp, PART, PINGPONG)
        lt, ka0, ka1 = warpgroup_mma_wait(0, deps=[lt, ka0, ka1])
        mbarrier.arrive(kfree.index(0))  # (the key is in registers: its loads are done, the GEMM used them)
        p = _probabilities(lt, lse, first, t0, target, WINDOW, BT, BS, mma, opb)
        mbarrier.arrive(qfree.index(g % S))  # (only after the lse loads' use, see the converter)
        for i in range(1, n):
            _turn(pp, j + i, PART, PINGPONG)
            acc = _values(p, h_s, hready, acc, g + i - 1, SH)
            lt, lse = _logits(ka0, ka1, q_s, l_s, qready, lt, g + i, S, PART, mma)
            _pass(pp, PART, PINGPONG)
            lt, acc, p, ka0, ka1 = warpgroup_mma_wait(0, deps=[lt, acc, p, ka0, ka1])  # (p: read until here)
            mbarrier.arrive(hfree.index((g + i - 1) % SH))
            p = _probabilities(lt, lse, first + i, t0, target, WINDOW, BT, BS, mma, opb)
            mbarrier.arrive(qfree.index((g + i) % S))
        _turn(pp, j + n, PART, PINGPONG)
        acc = _values(p, h_s, hready, acc, g + n - 1, SH)
        _pass(pp, PART, PINGPONG)
        acc = warpgroup_mma_wait(0, deps=[acc])
        mbarrier.arrive(hfree.index((g + n - 1) % SH))
        layer = head // OUT_H
        if OUT_MAPPED:
            layer = gl.load(out_map + layer)
        start = layer.to(gl.int64) * OUT_LAYER + (head % OUT_H) * C
        rows = t0 + PART * HALF + gl.arange(0, HALF, gl.SliceLayout(1, mmab))
        channels = gl.arange(0, C, gl.SliceLayout(0, mmab))
        gl.store(out_ptr + start + rows[:, None] * OUT_TOKEN + channels[None, :], acc)
        g += n
        j += n + 1


# --- producers -----------------------------------------------------------------------------------------------------

@gluon.jit
def _converter(h32_s, h_s, h32ready, h32free, hready, hfree, ITEMS, ROWS, T: gl.constexpr, WINDOW: gl.constexpr,
               BT: gl.constexpr, BS: gl.constexpr, C: gl.constexpr, SH: gl.constexpr, SH32: gl.constexpr,
               PERSISTENT: gl.constexpr, RUN: gl.constexpr, CH: gl.constexpr = 32):
    lay: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])  # (unswizzled fp32 rows: no bank conflicts)
    g = 0
    for k in range(_item_count(ITEMS, PERSISTENT)):
        head, t0, first, n = _item(k, ROWS, T, WINDOW, BT, BS, PERSISTENT, RUN)
        for i in range(n):
            r = g + i
            f = r % SH32
            mbarrier.wait(h32ready.index(f), (r // SH32) & 1)
            h = r % SH
            mbarrier.wait(hfree.index(h), ((r // SH) & 1) ^ 1)
            for c in gl.static_range(BS // CH):
                hub = h32_s.index(f).slice(c * CH, CH).load(lay)
                hi = gl.inline_asm_elementwise('cvt.rn.bf16x2.f32 $0, $2, $1;', '=r,r,r', [hub], dtype=gl.bfloat16,
                                               is_pure=True, pack=2)  # (hub.to(bf16), as packed conversions)
                lo = (hub - hi.to(gl.float32)).to(gl.bfloat16)
                h_s.index(2 * h).slice(c * CH, CH).store(hi)
                h_s.index(2 * h + 1).slice(c * CH, CH).store(lo)
            fence_async_shared()
            gl.thread_barrier()
            mbarrier.arrive(hready.index(h))
            # the fp32 slot only now: its loads are surely done once the stores that use them are (a barrier does not
            # wait for loads in flight: released right after them, the next TMA load overwrote the slot under them)
            mbarrier.arrive(h32free.index(f))
        g += n


@gluon.jit
def _loader(q_desc, k_desc, l_desc, hub_desc, k_s, q_s, l_s, h32_s, kready, kfree, qready, qfree, h32ready, h32free,
            ITEMS, ROWS, T: gl.constexpr, WINDOW: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr, D: gl.constexpr,
            C: gl.constexpr, IN_H: gl.constexpr, S: gl.constexpr, SH32: gl.constexpr, PERSISTENT: gl.constexpr,
            RUN: gl.constexpr):
    g = 0
    for k in range(_item_count(ITEMS, PERSISTENT)):
        head, t0, first, n = _item(k, ROWS, T, WINDOW, BT, BS, PERSISTENT, RUN)
        base = head * T
        mbarrier.wait(kfree.index(0), (k & 1) ^ 1)
        mbarrier.expect(kready.index(0), BT * D * 4)
        for part in gl.static_range(2):
            for chunk in gl.static_range(2):
                tma.async_copy_global_to_shared(k_desc, [base + t0 + part * (BT // 2), chunk * (D // 2)],
                                                kready.index(0), k_s.index(2 * part + chunk))
        for i in range(n):
            r = g + i
            s = r % S
            mbarrier.wait(qfree.index(s), ((r // S) & 1) ^ 1)
            mbarrier.expect(qready.index(s), BS * D * 4 + BS * 4)
            for chunk in gl.static_range(2):
                tma.async_copy_global_to_shared(q_desc, [base + (first + i) * BS, chunk * (D // 2)],
                                                qready.index(s), q_s.index(2 * s + chunk))
            tma.async_copy_global_to_shared(l_desc, [base + (first + i) * BS], qready.index(s), l_s.index(s))
            f = r % SH32
            mbarrier.wait(h32free.index(f), ((r // SH32) & 1) ^ 1)
            mbarrier.expect(h32ready.index(f), BS * C * 4)
            tma.async_copy_global_to_shared(hub_desc, [(head // IN_H) * T + (first + i) * BS, (head % IN_H) * C],
                                            h32ready.index(f), h32_s.index(f))
        g += n


# --- kernel --------------------------------------------------------------------------------------------------------

@gluon.jit
def _value_credit_ws_kernel(q_desc, k_desc, l_desc, hub_desc, out_ptr, out_map, OUT_LAYER, OUT_TOKEN, ITEMS, ROWS,
                            T: gl.constexpr, WINDOW: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr,
                            C: gl.constexpr, D: gl.constexpr, IN_H: gl.constexpr, OUT_H: gl.constexpr,
                            OUT_MAPPED: gl.constexpr, S: gl.constexpr, SH: gl.constexpr, SH32: gl.constexpr,
                            REGS: gl.constexpr, CONVERT_REGS: gl.constexpr, LOAD_REGS: gl.constexpr,
                            PINGPONG: gl.constexpr, PERSISTENT: gl.constexpr, RUN: gl.constexpr):
    k_s = gl.allocate_shared_memory(gl.float32, [4, BT // 2, D // 2], k_desc.layout)  # (consumer, k chunk)
    q_s = gl.allocate_shared_memory(gl.float32, [2 * S, BS, D // 2], q_desc.layout)  # (stage, k chunk)
    l_s = gl.allocate_shared_memory(gl.float32, [S, BS], l_desc.layout)
    h32_s = gl.allocate_shared_memory(gl.float32, [SH32, BS, C], hub_desc.layout)
    h_s = gl.allocate_shared_memory(gl.bfloat16, [2 * SH, BS, C],
                                    gl.NVMMASharedLayout.get_default_for([BS, C], gl.bfloat16))  # (slot, hi / lo)
    blay: gl.constexpr = mbarrier.MBarrierLayout()
    kready = gl.allocate_shared_memory(gl.int64, [1, 1], blay)
    kfree = gl.allocate_shared_memory(gl.int64, [1, 1], blay)
    pp = gl.allocate_shared_memory(gl.int64, [2, 1], blay)
    qready = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    qfree = gl.allocate_shared_memory(gl.int64, [S, 1], blay)
    h32ready = gl.allocate_shared_memory(gl.int64, [SH32, 1], blay)
    h32free = gl.allocate_shared_memory(gl.int64, [SH32, 1], blay)
    hready = gl.allocate_shared_memory(gl.int64, [SH, 1], blay)
    hfree = gl.allocate_shared_memory(gl.int64, [SH, 1], blay)
    mbarrier.init(kready.index(0), count=1)
    mbarrier.init(kfree.index(0), count=2)
    for i in gl.static_range(2):
        mbarrier.init(pp.index(i), count=1)
    for i in gl.static_range(S):
        mbarrier.init(qready.index(i), count=1)
        mbarrier.init(qfree.index(i), count=2)
    for i in gl.static_range(SH32):
        mbarrier.init(h32ready.index(i), count=1)
        mbarrier.init(h32free.index(i), count=1)
    for i in gl.static_range(SH):
        mbarrier.init(hready.index(i), count=1)
        mbarrier.init(hfree.index(i), count=2)
    gl.warp_specialize([
        (_consumer, (k_s, q_s, l_s, h_s, kready, kfree, qready, qfree, hready, hfree, pp, out_ptr, out_map, OUT_LAYER,
                     OUT_TOKEN, ITEMS, ROWS, T, WINDOW, BT, BS, C, OUT_H, OUT_MAPPED, S, SH, 0, PINGPONG, PERSISTENT,
                     RUN)),
        (_consumer, (k_s, q_s, l_s, h_s, kready, kfree, qready, qfree, hready, hfree, pp, out_ptr, out_map, OUT_LAYER,
                     OUT_TOKEN, ITEMS, ROWS, T, WINDOW, BT, BS, C, OUT_H, OUT_MAPPED, S, SH, 1, PINGPONG, PERSISTENT,
                     RUN)),
        (_converter, (h32_s, h_s, h32ready, h32free, hready, hfree, ITEMS, ROWS, T, WINDOW, BT, BS, C, SH, SH32,
                      PERSISTENT, RUN)),
        (_loader, (q_desc, k_desc, l_desc, hub_desc, k_s, q_s, l_s, h32_s, kready, kfree, qready, qfree, h32ready,
                   h32free, ITEMS, ROWS, T, WINDOW, BT, BS, D, C, IN_H, S, SH32, PERSISTENT, RUN)),
    ], [4, 4, 1], [REGS, CONVERT_REGS, LOAD_REGS])


# --- host ----------------------------------------------------------------------------------------------------------

_LSE_LAYOUT = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=1)
_HUB_LAYOUT = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)
_SMS, _RUNS = {}, {}


def _desc(x, block, layout=None):
    return TensorDescriptor.from_tensor(x, block, layout or gl.NVMMASharedLayout.get_default_for(block, gl.float32))


def _run_length(rows, nt, window, ctas, bt=128, bs=128):
    """The longest run of tiles (8, 4, 2 or 1) whose zigzag deal keeps the most loaded CTA within 3% of the best
    balance (longer runs: fewer DRAM rereads; 96 rows, window 1024: 186 us with runs of 8, 209 us without)."""
    key = (rows, nt, window, ctas)
    if key not in _RUNS:
        def most(run):
            load = [0] * ctas
            for w in range(rows * nt):
                k, p = divmod(w, ctas)
                tile = w // run // rows * run + w % run
                n = min((tile * bt + bt - 1 + window) // bs + 1, nt * bt // bs) - tile * bt // bs
                load[p if k % 2 == 0 else ctas - 1 - p] += n
            return max(load)
        loads = {run: most(run) for run in (8, 4, 2, 1) if nt % run == 0}
        _RUNS[key] = next(run for run, m in loads.items() if m <= 1.03 * min(loads.values()))
    return _RUNS[key]


def value_pass_ws(q, k, lse, source, window, in_heads, out_heads, out, layers, persistent=True):
    """moment_kernels._value_pass (sum_s p * source) into out: q, k (rows, T, 64) fp32, lse (rows, T), source
    (rows // in_heads, T, in_heads * 64) contiguous, out (rows // out_heads, T, out_heads * 64) by token (last stride
    1; with layers, layer l's rows at out[layers[l]])."""
    h, t, d = q.shape
    width = source.shape[-1]
    channels = width // in_heads
    bt = bs = 128
    c = CONFIG
    assert d == 64 and channels == 64 and t % bt == 0 and q.is_contiguous() and k.is_contiguous()
    assert lse.is_contiguous() and source.is_contiguous() and source.shape[1] == t and out.stride(2) == 1
    if q.device.index not in _SMS:
        _SMS[q.device.index] = torch.cuda.get_device_properties(q.device).multi_processor_count
    items = h * (t // bt)
    ctas = min(items, _SMS[q.device.index]) if persistent else items
    _value_credit_ws_kernel[(ctas,)](
        _desc(q.view(h * t, d), [bs, d // 2]), _desc(k.view(h * t, d), [bt // 2, d // 2]),
        _desc(lse.view(h * t), [bs], _LSE_LAYOUT), _desc(source.view(-1, width), [bs, channels], _HUB_LAYOUT), out,
        out if layers is None else layers, out.stride(0), out.stride(1), items, h, T=t, WINDOW=window, BT=bt, BS=bs,
        C=channels, D=d, IN_H=in_heads, OUT_H=out_heads, OUT_MAPPED=layers is not None, S=c['STAGES'],
        SH=c['SLOTS'], SH32=c['HUB_SLOTS'], REGS=c['CONSUMER_REGS'], CONVERT_REGS=c['CONVERT_REGS'],
        LOAD_REGS=c['LOAD_REGS'], PINGPONG=c['PINGPONG'], PERSISTENT=persistent,
        RUN=_run_length(h, t // bt, window, ctas) if persistent else 1, num_warps=4)
    return out
