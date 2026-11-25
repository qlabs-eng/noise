"""The Cut's attention preparation: the layers' queries, keys and values of one window size from the captured
forward's outputs, laid out for the key and query kernels by one kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(Qs, Ks, Vs, VEs, Gates, Cos, Sin,
             KR, Q16, K16, LinearKey, V16, QR, Sq, Steps,
             QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
             VES: tl.constexpr, GS: tl.constexpr,
             HAVE_VE: tl.constexpr, GROUPS: tl.constexpr, T: tl.constexpr,
             H: tl.constexpr, D: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, SIGMA: tl.constexpr, BT: tl.constexpr):
    tt = tl.program_id(0) * BT + tl.arange(0, BT)
    gh = tl.program_id(1)
    group, head = gh // H, gh % H
    dd = tl.arange(0, D)
    other = (dd + D // 2) % D
    col = head * D + dd
    col2 = head * D + other
    q = tl.full((BT, D), 0, tl.float32)
    k = tl.full((BT, D), 0, tl.float32)
    qother = tl.full((BT, D), 0, tl.float32)
    kother = tl.full((BT, D), 0, tl.float32)
    v = tl.full((BT, D), 0, tl.float32)
    for j in tl.static_range(GROUPS):
        if group == j:
            q = tl.load(Qs[j] + tt[:, None] * QS[j][0] + col[None, :] * QS[j][1], tt[:, None] < T, 0).to(tl.float32)
            k = tl.load(Ks[j] + tt[:, None] * KS[j][0] + col[None, :] * KS[j][1], tt[:, None] < T, 0).to(tl.float32)
            qother = tl.load(Qs[j] + tt[:, None] * QS[j][0] + col2[None, :] * QS[j][1], tt[:, None] < T,
                0).to(tl.float32)
            kother = tl.load(Ks[j] + tt[:, None] * KS[j][0] + col2[None, :] * KS[j][1], tt[:, None] < T,
                0).to(tl.float32)
            v = tl.load(Vs[j] + tt[:, None] * VS[j][0] + col[None, :] * VS[j][1], tt[:, None] < T, 0).to(tl.float32)
            if HAVE_VE[j]:
                ve = tl.load(VEs[j] + tt[:, None] * VES[j][0] + col[None, :] * VES[j][1], tt[:, None] < T,
                    0).to(tl.float32)
                z = tl.load(Gates[j] + tt * GS[j][0] + head * GS[j][1], tt < T, 0).to(tl.float32)
                gate = 2.0 / (1.0 + tl.exp(-z))
                v = v + gate[:, None] * ve
    c = tl.load(Cos + tt[:, None] * C0 + (dd % (D // 2))[None, :] * C1, tt[:, None] < T, 0).to(tl.float32)
    s = tl.load(Sin + tt[:, None] * S0 + (dd % (D // 2))[None, :] * S1, tt[:, None] < T, 0).to(tl.float32)
    qr = q * c + tl.where(dd[None, :] < D // 2, qother, -qother) * s
    kr = k * c + tl.where(dd[None, :] < D // 2, kother, -kother) * s
    qi = tl.rsqrt(tl.sum(qr * qr, 1) / D + 1.1920928955078125e-7)
    ki = tl.rsqrt(tl.sum(kr * kr, 1) / D + 1.1920928955078125e-7)
    qp = qr * qi[:, None]
    kp = kr * ki[:, None]
    sq = tl.sum(q * q, 1)
    step = SIGMA * tl.maximum(tl.sqrt(sq / D), 1e-6)
    dest = (gh * T + tt[:, None]) * D + dd[None, :]
    mask = tt[:, None] < T
    tl.store(KR + dest, k, mask)
    tl.store(QR + dest, q, mask)
    tl.store(Q16 + dest, qp.to(tl.bfloat16), mask)
    physical_key = kp.to(tl.bfloat16)
    tl.store(K16 + dest, physical_key, mask)
    pair_col = 2 * (dd % (D // 2)) + dd // (D // 2)
    location = (gh * T + tt[:, None]) * D + pair_col[None, :]
    tl.store(LinearKey + location, (physical_key.to(tl.float32) * 0.125).to(tl.bfloat16), mask)
    tl.store(V16 + dest, v.to(tl.bfloat16), mask)
    tl.store(Sq + gh * T + tt, sq, tt < T)
    tl.store(Steps + gh * T + tt, step, tt < T)


def prepare(qs, ks, vs, ves, gates, cos, sin, heads: int, sigma: float, hub):
    groups, t, d = len(qs), qs[0].shape[0], qs[0].shape[1] // heads
    block_tokens = 4
    shape = (groups * heads, t, d)
    make = lambda dtype=torch.float32: torch.empty(shape, dtype=dtype, device=qs[0].device)
    kr, qr = make(), make()
    q16, k16, v16 = [make(torch.bfloat16) for _ in range(3)]
    linear_key = make(torch.bfloat16)
    sq = torch.empty(shape[:2], dtype=torch.float32, device=qs[0].device)
    steps = torch.empty_like(sq)
    # None entries are never loaded in the specialized branch.
    vep = tuple(q if ve is None else ve for q, ve in zip(qs, ves))
    gp = tuple(q if z is None else z for q, z in zip(qs, gates))
    strides = lambda items: tuple(tuple(v.stride()) for v in items)
    _prepare[(triton.cdiv(t, block_tokens), groups * heads)](
        tuple(qs), tuple(ks), tuple(vs), vep, gp, cos, sin,
        kr, q16, k16, linear_key, v16, qr, sq, steps,
        strides(qs), strides(ks), strides(vs), strides(vep), strides(gp),
        tuple(ve is not None for ve in ves), groups, t, heads, d,
        *cos.stride(), *sin.stride(), sigma, block_tokens,
        num_warps=4, enable_fp_fusion=False)
    return (linear_key, kr), (q16, k16, v16, hub, qr.unsqueeze(0), sq, steps)
