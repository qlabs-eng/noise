import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (async_copy as cp, fence_async_shared, mbarrier,
    warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from visit_chain import (_issue_rows, _reload, _tc_parts_issue, _bf16, _rbf16, _row_sum, _change, _readout_tc_done,
    _stats_store, split_parts)
import fwht

@triton.jit
def _base_products(WA, RA, X, U, SQ, R: tl.constexpr = 64, D: tl.constexpr = 512):
    t = tl.program_id(0)
    r = tl.arange(0, R)
    d = tl.arange(0, D)
    rows = tl.load(RA + t * R + r).to(tl.int32)
    x = tl.load(X + t * D + d)
    w = tl.load(WA + rows[:, None] * D + d[None, :]).to(tl.float32)
    tl.store(U + t * R + r, tl.sum(w * x[None, :], 1))
    tl.store(SQ + t, tl.sum(x * x, 0))


@gluon.jit
def _map_compose(Maps, Wsm, WA, FB, RA, RB, token):
    sl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    rr = gl.arange(0, 128, gl.SliceLayout(1, sl))
    cols = gl.arange(0, 128, gl.SliceLayout(0, sl))
    ra = gl.load(RA + token * 64 + rr % 64).to(gl.int32)
    rb = gl.load(RB + token * 64 + rr % 64).to(gl.int32)
    rows = gl.where(rr < 64, WA + ra * 512, FB + rb * 512)
    ml: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 128, 16])
    right = Maps._reinterpret(gl.bfloat16, [128, 128], gl.NVMMASharedLayout.get_default_for([128, 128], gl.bfloat16))
    acc = gl.zeros([64, 128], gl.float32, ml)
    for c in range(4):
        _issue_rows(right, rows, cols, c * 128)
        cp.wait_group(0)
        fence_async_shared()
        gl.thread_barrier()
        acc = warpgroup_mma(right.slice(0, 64), right.permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(0, deps=[acc])
        gl.thread_barrier()
    hi = _bf16(acc)
    lo = _bf16(acc - hi.to(gl.float32))
    Maps.index(0).store(hi)
    Maps.index(1).store(lo)
    fence_async_shared()
    gl.thread_barrier()

@gluon.jit
def _products(C, Maps, mma: gl.constexpr, result_layout: gl.constexpr):
    acc = warpgroup_mma(C, Maps.index(0), gl.zeros([64, 128], gl.float32, mma), use_acc=False)
    acc = warpgroup_mma(C, Maps.index(1), acc)
    gram, cross = gl.split(gl.permute(gl.reshape(acc, [64, 2, 64]), (0, 2, 1)))
    return gl.convert_layout(gram, result_layout), gl.convert_layout(cross, result_layout)

@gluon.jit
def _output(c, accr, s1, s5, desc, dbar, dsm, wsm, v3, Diag, ca, cb, zero, WA, WB, ra, rb, cols, move, nextmove,
    has_more, token, step, scale,
            BK: gl.constexpr, KC: gl.constexpr, BM: gl.constexpr, mma2: gl.constexpr, mmax: gl.constexpr):
    slot = c % 2
    mbarrier.wait(dbar.index(slot), (c // 2) & 1)
    cp.wait_group(0)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index(1), WB + rb * 512, cols, c * BK)
    first = warpgroup_mma(ca, wsm.index(0), zero, use_acc=False, is_async=True)
    first = warpgroup_mma_wait(0, deps=[first, ca])[0]
    cp.wait_group(0)
    fence_async_shared()
    gl.thread_barrier()
    if c + 1 < KC:
        _issue_rows(wsm.index(0), WA + ra * 512, cols, (c + 1) * BK)
    second = warpgroup_mma(cb, wsm.index(1), zero, use_acc=False, is_async=True)
    second = warpgroup_mma_wait(0, deps=[second, cb])[0]
    frame = dsm.index(slot).load(mma2).to(gl.float32)
    value = _bf16((step * frame + first) * scale + second)
    fence_async_shared()
    gl.thread_barrier()
    refill_move = gl.where(c + 2 < KC, move, nextmove)
    refill = (c + 2 < KC) | has_more
    _reload(desc, dbar, dsm, slot, (c + 2) % KC, refill_move, refill, BM * BK * 2, BK)
    vop = gl.convert_layout(value, gl.DotOperandLayout(operand_index=0, parent=mmax, k_width=2))
    accr = warpgroup_mma(vop, v3.index(c).permute((1, 0)), accr, is_async=True)
    accr, vop = warpgroup_mma_wait(0, deps=[accr, vop])
    v = value.to(gl.float32)
    ds = gl.arange(0, BK, gl.SliceLayout(0, mma2))
    diag = gl.load(Diag + token * 512 + c * BK + ds)
    s1 += _row_sum(v * v, BM, BK)
    s5 += _row_sum(v * v * diag[None, :], BM, BK)
    return accr, s1, s5

@gluon.jit
def _kernel(desc, WA, RA, CA, SCALE, WB, RB, CB, DIAG, OUT, S, PROJ, I0, I1, FB, HF, AH, XH, U, SQB, VP, TASKS,
            P: gl.constexpr, BM: gl.constexpr = 64, BK: gl.constexpr = 128, GROUP_PROBES: gl.constexpr = 8):
    KC: gl.constexpr = 512 // BK
    R: gl.constexpr = 64
    GROUPS: gl.constexpr = P // (BM * GROUP_PROBES)
    pid = gl.program_id(0)
    G = gl.num_programs(0)
    baseprobe = (pid % GROUPS) * GROUP_PROBES * BM
    dsm = gl.allocate_shared_memory(gl.bfloat16, [2, BM, BK], desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [2, R, BK], gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    maps = gl.allocate_shared_memory(gl.bfloat16, [2, R, 2 * R], gl.NVMMASharedLayout.get_default_for([R, 2 * R],
        gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for init_c in gl.static_range(2):
        mbarrier.init(dbar.index(init_c), count=1)
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=128)
    vl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    vbuf = gl.allocate_shared_memory(gl.bfloat16, [KC, 8, BK], vl)
    _tc_parts_issue(vbuf, VP, pid // GROUPS, True, 512, BK, 4)
    cp.mbarrier_arrive(vbar, increment_count=False)
    for load_c in gl.static_range(2):
        _reload(desc, dbar, dsm, load_c, load_c, baseprobe, True, BM * BK * 2, BK)
    sl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, sl))
    cols = gl.arange(0, BK, gl.SliceLayout(0, sl))
    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, R, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BK, 16])
    mmax: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16])
    ql: gl.constexpr = gl.BlockedLayout([4, 1], [16, 2], [1, 4], [0, 1])
    pr = gl.arange(0, BM, gl.SliceLayout(1, ql))
    qr = gl.arange(0, R, gl.SliceLayout(0, ql))
    zero = gl.zeros([BM, BK], gl.float32, mma2)
    scale = gl.load(SCALE)
    it = 0
    for task in range(pid, TASKS, G):
        token = task // GROUPS
        ntask = task + G
        has_next = ntask < TASKS
        ntoken = gl.where(has_next, ntask // GROUPS, token)
        _map_compose(maps, wsm, WA, FB, RA, RB, token)
        mbarrier.wait(vbar, it & 1)
        ra = gl.load(RA + token * R + rr).to(gl.int32)
        rb = gl.load(RB + token * R + rr).to(gl.int32)
        step = gl.load(S + token)
        for probe in range(GROUP_PROBES + (TASKS >> 31)):
            move = baseprobe + probe * BM
            row0 = token * P + move
            nextmove = gl.where(probe + 1 < GROUP_PROBES, move + BM, baseprobe)
            has_more = (probe + 1 < GROUP_PROBES) | has_next
            q = gl.load(PROJ + gl.load(RA + token * R + qr).to(gl.int32)[None, :] * P + (move + pr)[:, None])
            inv = gl.load(I1 + row0 + pr)
            inv0 = gl.load(I0 + token)
            h = gl.load(CA + token * R + qr)
            moved = _rbf16(q * step.to(gl.bfloat16).to(gl.float32))
            dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
            up = gl.maximum(h[None, :] + dh, 0)
            down = gl.maximum(h, 0)
            response = _bf16((up - down[None, :]) * (up + down[None, :]))
            c2 = gl.convert_layout(response, gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))
            cv = gl.convert_layout(response.to(gl.float32), mma1)
            gram, acc = _products(c2, maps, mma2, mma1)
            mr = gl.arange(0, BM, gl.SliceLayout(1, mma1))
            mc = gl.arange(0, R, gl.SliceLayout(0, mma1))
            ari = gl.load(RA + token * R + mc).to(gl.int32)
            j = gl.load(AH + ari[None, :] * P + (move + mr)[:, None])
            u = gl.load(U + token * R + mc)
            xh = gl.load(XH + token * P + move + mr)
            dot = scale * (step * xh + gl.sum(cv * u[None, :], 1))
            dsq = scale * scale * (step * step * 512 + 2 * step * gl.sum(cv * j, 1) + gl.sum(gram * cv, 1))
            bri = gl.load(RB + token * R + mc).to(gl.int32)
            hf = gl.load(HF + bri[None, :] * P + (move + mr)[:, None])
            acc = (acc + step * hf) * scale
            hb = gl.load(CB + token * R + mc)
            changeb = _change(acc, gl.load(SQB + token), dot, dsq, hb, 512, 1.1920928955078125e-7, mma1, mma2)
            gl.thread_barrier()
            fence_async_shared()
            _issue_rows(wsm.index(0), WA + ra * 512, cols, 0)
            accr = gl.zeros([BM, 8], gl.float32, mmax)
            s1 = _row_sum(zero, BM, BK)
            s5 = s1
            for out_c in range(KC + (TASKS >> 31)):
                accr, s1, s5 = _output(out_c, accr, s1, s5, desc, dbar, dsm, wsm, vbuf, DIAG, c2, changeb, zero, WA, WB,
                    ra, rb, cols, move, nextmove, has_more, token, step, scale, BK, KC, BM, mma2, mmax)
            s0, s1, s2, s3, s4, s5 = _readout_tc_done(accr, s1, s5, mma2, mmax)
            _stats_store(OUT, row0, s0, s1, s2, s3, s4, s5, BM, mma2)
        gl.thread_barrier()
        _tc_parts_issue(vbuf, VP, ntoken, has_next, 512, BK, 4)
        cp.mbarrier_arrive(vbar, increment_count=False)
        it += 1
    cp.wait_group(0)
    for end_c in gl.static_range(2):
        mbarrier.invalidate(dbar.index(end_c))
    mbarrier.invalidate(vbar)


def launch(a, b, vec, root, out, group_probes=8):
    h, _, _, wa, ra, ca, scale = a
    xb, fb, wb, rb, cb = b
    s, proj, i0, i1 = root
    n, p = s.shape[0], h.shape[0]
    assert p == 512 and group_probes in (1, 2, 8)
    hf = fwht.frame(fb, h)
    ah = fwht.frame(wa, h)
    xh = fwht.frame(xb, h)
    u = torch.empty((n, 64), device=h.device)
    sq = torch.empty((n,), device=h.device)
    _base_products[(n,)](wa, ra, xb, u, sq, num_warps=8)
    vp, _ = split_parts(list(vec[:4]), 2)
    desc = TensorDescriptor.from_tensor(h, [64, 128], gl.NVMMASharedLayout.get_default_for([64, 128], gl.bfloat16))
    groups = 8 // group_probes
    tasks = n * groups
    sms = torch.cuda.get_device_properties(h.device).multi_processor_count
    grid = min(tasks, 2 * sms)
    assert grid % groups == 0
    _kernel[(grid,)](desc, wa, ra, ca, scale, wb, rb, cb, vec[-1], out, s, proj, i0, i1, fb, hf, ah, xh, u, sq, vp,
        tasks, p, GROUP_PROBES=group_probes, num_warps=4)

@torch.library.custom_op('factor_inline::root_stats', mutates_args=())
def root_stats(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo_a: torch.Tensor,
               routes_a: torch.Tensor, clean_a: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor,
               scale_a: torch.Tensor,
               x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor, routes_b: torch.Tensor, clean_b: torch.Tensor,
               xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor,
               diag: torch.Tensor) -> torch.Tensor:
    f32 = lambda z: z.float().contiguous()
    n, p = steps.shape[0], basis.shape[0]
    out = torch.empty((n, p, 6), device=steps.device, dtype=torch.float32)
    a = (basis.to(torch.bfloat16).contiguous(), None, None, wo_a.contiguous(), routes_a.contiguous(), f32(clean_a),
        scale_a)
    b = (f32(x_b), fc_b.contiguous(), wo_b.contiguous(), routes_b.contiguous(), f32(clean_b))
    vec = tuple(f32(z) for z in (xl, target, mu, nu, diag))
    root = tuple(f32(z) for z in (steps, projection, inv0, inv))
    launch(a, b, vec, root, out)
    return out

@root_stats.register_fake
def _(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv, scale_a, x_b, fc_b, wo_b, routes_b, clean_b, xl,
    target, mu, nu, diag):
    return torch.empty((steps.shape[0], basis.shape[0], 6), device=steps.device, dtype=torch.float32)
