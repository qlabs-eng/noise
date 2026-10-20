"""Triton kernels for the PyTorch reference of PR #29: Triton versions of #29's Pallas kernels.

- `value_credit`: the values' exact credit sum_s A[s, t] hub_s.
- `moment_query_scores` / `moment_key_scores`: the exact single-coordinate query (key) experiments as attention-weighted
  moments on tensor cores, degree 2 in the logit change (#29's kernels/exact_moments.py).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _value_kernel(q_ptr, k_ptr, hub_ptr, lse_ptr, out_ptr, T, window, scale,
                  D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    bh, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    base = bh.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    acc = tl.zeros((BLOCK_T, D), tl.float32)
    first = t0 // BLOCK_S
    last = tl.minimum((t0 + BLOCK_T - 1 + window) // BLOCK_S + 1, T // BLOCK_S)
    for i in range(first, last):
        offs_s = i * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        lse = tl.load(lse_ptr + bh * T + offs_s)
        logits = tl.dot(q, tl.trans(k))
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp(logits * scale - lse[:, None]), 0.0)
        acc += tl.dot(tl.trans(a.to(tl.bfloat16)), hub)
    tl.store(out_ptr + base + offs_t[:, None] * D + offs_d[None, :], acc)


def value_credit(q, k, hub, lse, window, block_t=64, block_s=64, num_warps=4, num_stages=2):
    """sum_s A[s, t] hub_s [BH, T, D] f32: q, k, hub [BH, T, D] bf16, lse [BH, T] f32 (the clean attention's)."""
    bh, t, d = q.shape
    out = torch.empty((bh, t, d), device=q.device, dtype=torch.float32)
    _value_kernel[(bh, t // block_t)](q, k, hub, lse, out, t, window, d ** -0.5, D=d, BLOCK_T=block_t,
                                      BLOCK_S=block_s, num_warps=num_warps, num_stages=num_stages)
    return out


# moment kernels
# PR #29's exact query / key experiments as attention-weighted moments on tensor cores (degree 2 in the logit change):
# per (moved position, direction), dl = A L + Bu X + Bw Y; the response's polynomial in dl splits into products of a
# per-pair weight (a c L^m, a L^m) and per-(other position, direction) features ([X|Y], [X^2|Y^2], [XY|XY]); the sums
# over the other positions are [moved, other] x [other, 2J] products and the coefficients apply once at the end.


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, coef_ptr, feat_ptr, lse_ptr, hubout_ptr, out_ptr, T, window,
                           scale, REPEAT, D: tl.constexpr, W2: tl.constexpr, BLOCK_T: tl.constexpr,
                           BLOCK_S: tl.constexpr):
    i, s0 = tl.program_id(0), tl.program_id(1) * BLOCK_S
    row = i // REPEAT
    offs_s = s0 + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    offs_w = tl.arange(0, W2)
    base = row.to(tl.int64) * T * D
    q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    lse = tl.load(lse_ptr + row * T + offs_s)
    hub_out = tl.load(hubout_ptr + row * T + offs_s)
    fbase = feat_ptr + i.to(tl.int64) * 3 * T * W2  # features [BH R, 3, T, 2J]
    p00n = tl.zeros((BLOCK_S, W2), tl.float32)
    p01n = tl.zeros((BLOCK_S, W2), tl.float32)
    p02n = tl.zeros((BLOCK_S, W2), tl.float32)
    p10n = tl.zeros((BLOCK_S, W2), tl.float32)
    p00d = tl.zeros((BLOCK_S, W2), tl.float32)
    p01d = tl.zeros((BLOCK_S, W2), tl.float32)
    p02d = tl.zeros((BLOCK_S, W2), tl.float32)
    p10d = tl.zeros((BLOCK_S, W2), tl.float32)
    s1n = tl.zeros((BLOCK_S,), tl.float32)
    s2n = tl.zeros((BLOCK_S,), tl.float32)
    s1d = tl.zeros((BLOCK_S,), tl.float32)
    s2d = tl.zeros((BLOCK_S,), tl.float32)
    first = tl.maximum(s0 - window, 0) // BLOCK_T
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    for it in range(first, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale  # [BS, BT]
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
        c = tl.dot(hub, tl.trans(v)) - hub_out[:, None]
        f0 = tl.load(fbase + offs_t[:, None] * W2 + offs_w[None, :])
        f1 = tl.load(fbase + T * W2 + offs_t[:, None] * W2 + offs_w[None, :])
        f2 = tl.load(fbase + 2 * T * W2 + offs_t[:, None] * W2 + offs_w[None, :])
        wn = a * c
        wnl = wn * logits
        p00n += tl.dot(wn.to(tl.bfloat16), f0)
        p01n += tl.dot(wn.to(tl.bfloat16), f1)
        p02n += tl.dot(wn.to(tl.bfloat16), f2)
        p10n += tl.dot(wnl.to(tl.bfloat16), f0)
        s1n += tl.sum(wnl, axis=1)
        s2n += tl.sum(wnl * logits, axis=1)
        al = a * logits
        p00d += tl.dot(a.to(tl.bfloat16), f0)
        p01d += tl.dot(a.to(tl.bfloat16), f1)
        p02d += tl.dot(a.to(tl.bfloat16), f2)
        p10d += tl.dot(al.to(tl.bfloat16), f0)
        s1d += tl.sum(al, axis=1)
        s2d += tl.sum(al * logits, axis=1)
    cbase = coef_ptr + i.to(tl.int64) * 3 * T * W2  # coefficients [BH R, 3, T, 2J]: [A|A], P = [Bu|Bw], Q = [Bw|Bu]
    ca = tl.load(cbase + offs_s[:, None] * W2 + offs_w[None, :])
    cp = tl.load(cbase + T * W2 + offs_s[:, None] * W2 + offs_w[None, :])
    cq = tl.load(cbase + 2 * T * W2 + offs_s[:, None] * W2 + offs_w[None, :])
    g0, g1, g2 = cp, 0.5 * cp * cp, 0.5 * cp * cq
    h1, h2 = 0.5 * ca, 0.25 * ca * ca
    zn = p00n * g0 + p01n * g1 + p02n * g2 + p10n * (ca * g0) + s1n[:, None] * h1 + s2n[:, None] * h2
    zd = p00d * g0 + p01d * g1 + p02d * g2 + p10d * (ca * g0) + s1d[:, None] * h1 + s2d[:, None] * h2
    obase = out_ptr + i.to(tl.int64) * 2 * T * W2  # out [BH R, 2, T, 2J]
    tl.store(obase + offs_s[:, None] * W2 + offs_w[None, :], zn)
    tl.store(obase + T * W2 + offs_s[:, None] * W2 + offs_w[None, :], zd)


def moment_query_scores(q, k, v, hub, coef, feat, lse, hub_out, window, repeat, block_s=64, block_t=64, num_warps=8,
                        num_stages=2):
    """Query scores [BH R, T, J]: q, k, v, hub [BH, T, D] bf16; coef [BH R, 3, T, 2J] f32 (`moment_coefficients`);
    feat [BH R, 3, T, 2J] bf16 (`moment_features` of the key columns); lse, hub_out [BH, T] f32."""
    rows, _, t, w2 = coef.shape
    out = torch.empty((rows, 2, t, w2), device=q.device, dtype=torch.float32)
    _moment_queries_kernel[(rows, t // block_s)](q, k, v, hub, coef, feat, lse, hub_out, out, t, window,
                                                 q.shape[-1] ** -0.5, repeat, D=q.shape[-1], W2=w2, BLOCK_T=block_t,
                                                 BLOCK_S=block_s, num_warps=num_warps, num_stages=num_stages)
    pairs = out[..., :w2 // 2] + out[..., w2 // 2:]
    return pairs[:, 0] / (1.0 + pairs[:, 1])


@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, coef_ptr, feat_ptr, lse_ptr, hubout_ptr, out_ptr, T, window,
                        scale, REPEAT, D: tl.constexpr, W2: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    row = i // REPEAT
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_w = tl.arange(0, W2)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    fbase = feat_ptr + i.to(tl.int64) * 3 * T * W2
    p100 = tl.zeros((BLOCK_T, W2), tl.float32)
    p201 = tl.zeros((BLOCK_T, W2), tl.float32)
    p202 = tl.zeros((BLOCK_T, W2), tl.float32)
    p210 = tl.zeros((BLOCK_T, W2), tl.float32)
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    first = t0 // BLOCK_S
    last = tl.minimum((t0 + BLOCK_T - 1 + window) // BLOCK_S + 1, T // BLOCK_S)
    for it in range(first, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale  # [BT, BS]
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        f0 = tl.load(fbase + offs_s[:, None] * W2 + offs_w[None, :])
        f1 = tl.load(fbase + T * W2 + offs_s[:, None] * W2 + offs_w[None, :])
        f2 = tl.load(fbase + 2 * T * W2 + offs_s[:, None] * W2 + offs_w[None, :])
        w1 = a * c  # ch / (1 + ch) = a dl + (a / 2 - a^2) dl^2 + ...
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logits
        p100 += tl.dot(w1.to(tl.bfloat16), f0)
        p201 += tl.dot(w2.to(tl.bfloat16), f1)
        p202 += tl.dot(w2.to(tl.bfloat16), f2)
        p210 += tl.dot(w2l.to(tl.bfloat16), f0)
        s1 += tl.sum(w1 * logits, axis=1)
        s2 += tl.sum(w2l * logits, axis=1)
    cbase = coef_ptr + i.to(tl.int64) * 3 * T * W2
    ca = tl.load(cbase + offs_t[:, None] * W2 + offs_w[None, :])
    cp = tl.load(cbase + T * W2 + offs_t[:, None] * W2 + offs_w[None, :])
    cq = tl.load(cbase + 2 * T * W2 + offs_t[:, None] * W2 + offs_w[None, :])
    z = p100 * cp + p201 * (cp * cp) + p202 * (cp * cq) + p210 * (2.0 * ca * cp)
    z += s1[:, None] * (0.5 * ca) + s2[:, None] * (0.5 * ca * ca)
    tl.store(out_ptr + i.to(tl.int64) * T * W2 + offs_t[:, None] * W2 + offs_w[None, :], z)


def moment_key_scores(q, k, v, hub, coef, feat, lse, hub_out, window, repeat, block_t=64, block_s=64, num_warps=4,
                      num_stages=2):
    """Key scores [BH R, T, J]: coef [BH R, 3, T, 2J] of the raw keys; feat of the query columns; as above."""
    rows, _, t, w2 = coef.shape
    out = torch.empty((rows, t, w2), device=q.device, dtype=torch.float32)
    _moment_keys_kernel[(rows, t // block_t)](q, k, v, hub, coef, feat, lse, hub_out, out, t, window,
                                              q.shape[-1] ** -0.5, repeat, D=q.shape[-1], W2=w2, BLOCK_T=block_t,
                                              BLOCK_S=block_s, num_warps=num_warps, num_stages=num_stages)
    return out[..., :w2 // 2] + out[..., w2 // 2:]


def moment_features(cols, d):
    """[BH, 3, T, 2J] bf16 feature groups [X|Y], [X^2|Y^2], [XY|XY] from rotated columns [BH, 2, T, J] (f32)."""
    x, y = cols[:, 0] * d ** -0.5, cols[:, 1] * d ** -0.5
    return torch.stack([torch.cat([x, y], -1), torch.cat([x * x, y * y], -1), torch.cat([x * y, x * y], -1)],
                       1).to(torch.bfloat16)


def moment_coefficients(coef):
    """[BH, 3, T, 2J] f32: [A|A], [Bu|Bw], [Bw|Bu] from closed coefficients [BH, 3, T, J]."""
    a, bu, bw = coef[:, 0], coef[:, 1], coef[:, 2]
    return torch.stack([torch.cat([a, a], -1), torch.cat([bu, bw], -1), torch.cat([bw, bu], -1)], 1).float()
