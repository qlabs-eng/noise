"""Root 6's replay, blocks 6 and 7 in one Gluon (Hopper) kernel: the readout's curvature term by its mean plus a
remainder sampled at 8 of the 512 coordinates on one position in four."""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, warpgroup_mma, warpgroup_mma_wait)
from visit_chain import _bf16, _rbf16, _change, split_parts
import fwht


TILE = 16
CORRECTION_GROUP = 4


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
def _products(C, Maps, mma: gl.constexpr, result_layout: gl.constexpr):
    acc = warpgroup_mma(C, Maps.index(0), gl.zeros([64, 128], gl.float32, mma), use_acc=False)
    acc = warpgroup_mma(C, Maps.index(1), acc)
    gram, cross = gl.split(gl.permute(gl.reshape(acc, [64, 2, 64]), (0, 2, 1)))
    return gl.convert_layout(gram, result_layout), gl.convert_layout(cross, result_layout)


@gluon.jit
def _cap(z):
    decay = gl.exp((-2.0 / 15.0) * gl.abs(z))
    magnitude = 15.0 * ((1.0 - decay) / (1.0 + decay))
    return gl.where(z < 0.0, -magnitude, magnitude)


# Loss readout for a single tile of moved forward evaluations.
@gluon.jit
def store(Out, Meta, token, dst, dots, dsq, target_dot, mu_dot, nu_dot, quad):
    coeff = Meta + token * 6
    sq = gl.load(coeff)
    base_y = gl.load(coeff + 1)
    b1 = gl.load(coeff + 2)
    b2 = gl.load(coeff + 3)
    valid = gl.load(coeff + 4)
    eps: gl.constexpr = 1.1920928955078125e-7
    inv = gl.rsqrt((sq + 2. * dots + dsq) * (1. / 512.) + eps)
    inv0 = gl.rsqrt(sq * (1. / 512.) + eps)
    a = inv - inv0
    lin = a * b1 + inv * mu_dot
    change = lin + (a * a) * b2 + (2. * a * inv) * nu_dot + (inv * inv) * quad - .5 * (lin * lin)
    clean_y = gl.load(coeff + 5)
    moved_y = clean_y + (a * base_y + inv * target_dot)
    target_change = _cap(moved_y) - _cap(clean_y)
    gl.store(Out + dst, (change - target_change) * valid)


@gluon.jit
def _store_linear_sample(buf, product, weight, routes, offsets, token, SAMPLES: gl.constexpr):
    ll: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    projected = gl.sum(gl.reshape(gl.convert_layout(product, ll), [64, 4, 2]), 2)
    hi = projected.to(gl.bfloat16)
    lo = (projected - hi.to(gl.float32)).to(gl.bfloat16)
    parts = gl.permute(gl.reshape(gl.join(hi, lo), [64, 8]), (1, 0))
    buf.slice(0, 8).store(parts)
    if SAMPLES:
        sl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        sector = gl.arange(0, 8, gl.SliceLayout(1, sl))
        r = gl.arange(0, 64, gl.SliceLayout(0, sl))
        coord = 64 * sector + (gl.load(offsets + token * 8 + sector) & 63)
        feature = gl.load(routes + token * 64 + r).to(gl.int32)
        buf.slice(8, 8).store(gl.load(weight + feature[None, :] * 512 + coord[:, None]))


@gluon.jit
def _compose_joint(WA, WB, FB, RA, RB, VP, offsets, token, maps, norm_maps, panels, SAMPLES: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    r = gl.arange(0, 64, gl.SliceLayout(1, layout))
    cols = gl.arange(0, 128, gl.SliceLayout(0, layout))
    stage_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 128], gl.bfloat16)
    first_weights = maps._reinterpret(gl.bfloat16, [128, 128], stage_layout)
    second_weights = norm_maps._reinterpret(gl.bfloat16, [128, 128], stage_layout)
    a = gl.load(RA + token * 64 + r).to(gl.int32)
    b = gl.load(RB + token * 64 + r).to(gl.int32)
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 128, 16])
    small: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16])
    pa = gl.zeros([64, 8], gl.float32, small)
    pb = gl.zeros([64, 8], gl.float32, small)
    vectors = gl.allocate_shared_memory(gl.bfloat16, [8, 128], gl.NVMMASharedLayout.get_default_for([8, 128], gl.bfloat16))
    vrows = gl.arange(0, 8, gl.SliceLayout(1, layout))
    first = gl.zeros([64, 128], gl.float32, mma)
    second = gl.zeros([64, 128], gl.float32, mma)
    for slab in range(4):
        offset = (slab * 128 + cols)[None, :]
        cp.async_copy_global_to_shared(second_weights.slice(0, 64), WB + b[:, None] * 512 + offset)
        cp.async_copy_global_to_shared(first_weights.slice(0, 64), WA + a[:, None] * 512 + offset)
        cp.async_copy_global_to_shared(first_weights.slice(64, 64), FB + b[:, None] * 512 + offset)
        cp.async_copy_global_to_shared(vectors,
            VP + token * 4096 + vrows[:, None] * 512 + offset)
        cp.commit_group()
        cp.wait_group(0)
        fence_async_shared()
        gl.thread_barrier()
        second_weights.slice(64, 64).store(first_weights.slice(0, 64).load(layout))
        fence_async_shared()
        gl.thread_barrier()
        first = warpgroup_mma(first_weights.slice(0, 64), first_weights.permute((1, 0)), first, is_async=True)
        second = warpgroup_mma(second_weights.slice(0, 64), second_weights.permute((1, 0)), second, is_async=True)
        pa = warpgroup_mma(first_weights.slice(0, 64), vectors.permute((1, 0)), pa, is_async=True)
        pb = warpgroup_mma(second_weights.slice(0, 64), vectors.permute((1, 0)), pb, is_async=True)
        first, second, pa, pb = warpgroup_mma_wait(0, deps=[first, second, pa, pb])
        gl.thread_barrier()
    _store_linear_sample(panels.index(0), pa, WA, RA, offsets, token, SAMPLES)
    _store_linear_sample(panels.index(1), pb, WB, RB, offsets, token, SAMPLES)
    first_hi = _bf16(first)
    second_hi = _bf16(second)
    maps.index(0).store(first_hi)
    maps.index(1).store(_bf16(first - first_hi.to(gl.float32)))
    norm_maps.index(0).store(second_hi)
    norm_maps.index(1).store(_bf16(second - second_hi.to(gl.float32)))
    fence_async_shared()
    gl.thread_barrier()


@gluon.jit
def _kernel(Frame, Offsets, WA, RA, CA, SCALE, WB, RB, CB, DIAG, OUT, S, PROJ, I0, I1, FB, HF, XH, U, SQB, MP,
            F0, F2, F3, F4, META, TASKS, TILE: gl.constexpr, P: gl.constexpr, CORRECTION: gl.constexpr,
            SUBSET: gl.constexpr):
    BM: gl.constexpr = 64
    BK: gl.constexpr = 128
    GROUP_PROBES: gl.constexpr = 8
    R: gl.constexpr = 64
    pid = gl.program_id(0)
    G = gl.num_programs(0)
    per_group: gl.constexpr = 1 if SUBSET == 1 else CORRECTION - 1
    map_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([64, 128], gl.bfloat16)
    maps = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 128], map_layout)
    norm_maps = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 128], map_layout)
    vl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([TILE, 64], gl.bfloat16)
    lbuf = gl.allocate_shared_memory(gl.bfloat16, [2, TILE, 64], vl)
    inclusion_probability: gl.constexpr = 1. / 64.
    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, R, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BK, 16])
    mmax: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, TILE, 16])
    ql: gl.constexpr = gl.BlockedLayout([4, 1], [16, 2], [1, 4], [0, 1])
    pr = gl.arange(0, BM, gl.SliceLayout(1, ql))
    qr = gl.arange(0, R, gl.SliceLayout(0, ql))
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    route7 = gl.arange(0, R, gl.SliceLayout(1, copy_layout))
    dir7 = gl.arange(0, BM, gl.SliceLayout(0, copy_layout))
    sm7 = gl.allocate_shared_memory(gl.int32, [R, BM], gl.SwizzledSharedLayout(4, 1, 8, [1, 0]))
    scale = gl.load(SCALE)
    for task in range(pid, TASKS, G):
        origin = (task // per_group) * CORRECTION
        pivot = (gl.load(Offsets + origin * 8) >> 6) & (CORRECTION - 1)
        if SUBSET == 1:
            token = origin + pivot
        else:
            token = origin + ((pivot + 1 + task % per_group) & (CORRECTION - 1))
        _compose_joint(WA, WB, FB, RA, RB, MP, Offsets, token, maps, norm_maps, lbuf, TILE == 16)
        step = gl.load(S + token)
        all_col = gl.arange(0, 512, gl.BlockedLayout([4], [32], [4], [0]))
        mean_diag = gl.sum(gl.load(DIAG + token * 512 + all_col), 0) / 512.
        if SUBSET == 1:
            correction_weight = CORRECTION
        else:
            correction_weight = 1
        src7 = gl.load(RB + token * R + route7).to(gl.int32)
        for probe in range(GROUP_PROBES + (TASKS >> 31)):
            move = probe * BM
            row0 = token * P + move
            cp.async_copy_global_to_shared(sm7, HF + (src7[:, None] * P + (move + dir7)[None, :]))
            cp.commit_group()
            feature = gl.load(RA + token * R + qr).to(gl.int32)
            bits = gl.load(PROJ + (feature[None, :] * P + (move + pr)[:, None])).to(gl.uint32, bitcast=True)
            q = bits.to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
            writer6_tile = (bits >> 16).to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
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
            j = gl.convert_layout(writer6_tile, mma1)
            u = gl.load(U + token * R + mc)
            xh = gl.load(XH + token * P + move + mr)
            dot = scale * (step * xh + gl.sum(cv * u[None, :], 1))
            dsq = scale * scale * (step * step * 512 + 2 * step * gl.sum(cv * j, 1) + gl.sum(gram * cv, 1))
            cp.wait_group(0)
            gl.thread_barrier()
            bits = sm7.permute((1, 0)).load(mma1).to(gl.uint32, bitcast=True)
            hf = bits.to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
            writer7_tile = (bits >> 16).to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
            acc = (acc + step * hf) * scale
            hb = gl.load(CB + token * R + mc)
            changeb = _change(acc, gl.load(SQB + token), dot, dsq, hb, 512, 1.1920928955078125e-7, mma1, mma2)
            bb, ba = _products(changeb, norm_maps, mma2, mma1)
            bcoef = gl.convert_layout(changeb, mma1).to(gl.float32)
            bh = writer7_tile
            last_norm = gl.sum(bcoef * bb, 1)
            incoming_last = scale * gl.sum(cv * ba + step * bcoef * bh, 1)
            terminal_norm = dsq + 2. * incoming_last + last_norm
            gl.thread_barrier()
            fence_async_shared()
            la = gl.convert_layout(c2, gl.DotOperandLayout(operand_index=0, parent=mmax, k_width=2))
            lb = gl.convert_layout(changeb, gl.DotOperandLayout(operand_index=0, parent=mmax, k_width=2))
            a4 = warpgroup_mma(la, lbuf.index(0).permute((1, 0)), gl.zeros([BM, TILE], gl.float32, mmax), use_acc=False)
            b4 = warpgroup_mma(lb, lbuf.index(1).permute((1, 0)), gl.zeros([BM, TILE], gl.float32, mmax), use_acc=False)
            accr = a4 * scale + b4
            rpos = gl.arange(0, BM, gl.SliceLayout(1, mmax))
            cols_out = gl.arange(0, TILE, gl.SliceLayout(0, mmax))
            row_layout: gl.constexpr = gl.SliceLayout(1, mma2)
            s0 = gl.convert_layout(gl.sum(gl.where((cols_out // 2 == 0)[None, :], accr, 0.), 1), row_layout)
            s2 = gl.convert_layout(gl.sum(gl.where((cols_out // 2 == 1)[None, :], accr, 0.), 1), row_layout)
            s3 = gl.convert_layout(gl.sum(gl.where((cols_out // 2 == 2)[None, :], accr, 0.), 1), row_layout)
            s4 = gl.convert_layout(gl.sum(gl.where((cols_out // 2 == 3)[None, :], accr, 0.), 1), row_layout)
            if TILE == 16:
                sector = gl.maximum(cols_out - 8, 0)
                coord = sector * 64 + (gl.load(Offsets + token * 8 + sector) & 63)
                direct = gl.load(Frame + (move + rpos[:, None]) * 512 + coord[None, :]).to(gl.float32)
                vec = accr + (scale * step) * direct
                diag = gl.load(DIAG + token * 512 + coord)
                total = gl.convert_layout(terminal_norm, gl.SliceLayout(1, mmax))
                centered = (diag - mean_diag)[None, :] * (vec * vec - total[:, None] / 512.)
                weighted = gl.where(cols_out[None, :] >= 8, centered, 0.)
                s5 = gl.convert_layout(gl.sum(weighted, 1), row_layout) * correction_weight
            else:
                s5 = gl.full((BM,), 0., gl.float32, row_layout)
            known = gl.convert_layout(terminal_norm, row_layout)
            dst = row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))
            amp = step * scale
            store(OUT, META, token, dst, s0 + amp * gl.load(F0 + dst), gl.maximum(known, 0.),
                  s2 + amp * gl.load(F2 + dst), s3 + amp * gl.load(F3 + dst),
                  s4 + amp * gl.load(F4 + dst), known * mean_diag + s5 * (1. / inclusion_probability))
        gl.thread_barrier()
    cp.wait_group(0)


def launch(a, b, vec, root, out, offsets, meta, weight_maps):
    h, wa, ra, ca, scale = a
    xb, fb, wb, rb, cb = b
    s, proj, i0, i1 = root
    n, p = s.shape[0], h.shape[0]
    hf = weight_maps[1]
    xh = fwht.frame(xb, h)
    u = torch.empty((n, 64), device=h.device)
    sq = torch.empty((n,), device=h.device)
    _base_products[(n,)](wa, ra, xb, u, sq, num_warps=8)
    vp, _ = split_parts(list(vec[:4]), 2)
    frame_vectors = [fwht.frame(v, h) for v in vec[:4]]
    sms = torch.cuda.get_device_properties(h.device).multi_processor_count
    for subset in (1, -1):
        if subset == 1:
            work = n // CORRECTION_GROUP
        else:
            work = (n // CORRECTION_GROUP) * (CORRECTION_GROUP - 1)
        grid = min(work, 2 * sms)
        width = 8 if subset == -1 else TILE
        _kernel[(grid,)](h, offsets, wa, ra, ca, scale, wb, rb, cb, vec[-1], out, s, proj, i0, i1,
                         fb, hf, xh, u, sq, vp, *frame_vectors, meta, work, width, p,
                         CORRECTION=CORRECTION_GROUP, SUBSET=subset, num_warps=4)


@torch.library.custom_op('terminal_panels::root_stats', mutates_args=())
def root_stats(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo_a: torch.Tensor,
               routes_a: torch.Tensor, clean_a: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor,
               scale_a: torch.Tensor,
               x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor, routes_b: torch.Tensor, clean_b: torch.Tensor,
               xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor,
               diag: torch.Tensor, offsets: torch.Tensor, meta: torch.Tensor,
               weight_maps: torch.Tensor) -> torch.Tensor:
    f32 = lambda z: z.float().contiguous()
    n, p = steps.shape[0], basis.shape[0]
    out = steps.new_empty((n, p))
    a = (basis.to(torch.bfloat16).contiguous(), wo_a.contiguous(), routes_a.contiguous(), f32(clean_a),
        scale_a)
    b = (f32(x_b), fc_b.contiguous(), wo_b.contiguous(), routes_b.contiguous(), f32(clean_b))
    vec = tuple(f32(z) for z in (xl, target, mu, nu, diag))
    root = (f32(steps), projection.contiguous(), f32(inv0), f32(inv))
    launch(a, b, vec, root, out, offsets, meta, weight_maps)
    return out


@root_stats.register_fake
def _(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv, scale_a, x_b, fc_b, wo_b, routes_b, clean_b, xl,
    target, mu, nu, diag, offsets, meta, weight_maps):
    n, p = steps.shape[0], basis.shape[0]
    return steps.new_empty((n, p))
