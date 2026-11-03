"""The replay's projected top values, token-major (run.py project_routed_values_token_major): the per-(token, head)
projection of the centred top values through c_proj's head slice, by programs of TB tokens x BO output columns of one
head (the head's weight slice read once per program instead of once per token)."""
import torch
import triton
import triton.language as tl


# gathered_key_tables (n = 2048, 8 heads, 16 keys, 512 columns; visit-kernel/r20 bench_keys.py): one program per (head,
# 32-token block) looping over 128-column blocks: 0.046 ms vs 0.116 for strong_key_shifted (compiled) + gathered_tables
# (writes 81 MB: fill floor 0.028). Grid over column blocks too: 0.067; tf32x3 correction: 0.053.
FUSED_KEYS_CONFIG = dict(TB=32, BO=128, TS=16, num_warps=4, STAGES=3)


@triton.jit
def _gathered_key_tables_kernel(V, Origin, W, Idx, Table, Corr, Rows, Keys, Prob, AllKeys, Shifted, T: tl.constexpr,
                                H: tl.constexpr, K: tl.constexpr, D: tl.constexpr, O: tl.constexpr, TB: tl.constexpr,
                                BO: tl.constexpr, TS: tl.constexpr, STAGES: tl.constexpr):
    """_gathered_tables_kernel's outputs for TB tokens of one head (the token rows loaded once, the weight's column
    blocks pipelined; the correction by a tf32 dot), then those tokens' shifted top keys, TS at a time:
    bf16(keys[idx] - centroid), centroid = (all_keys - p . keys[idx]) / (1 - sum p) where that mass > 1e-5, else 0."""
    h, tb = tl.program_id(0), tl.program_id(1)
    tt, dd = tb * TB + tl.arange(0, TB), tl.arange(0, D)
    v = tl.load(V + (h * T + tt[:, None]) * D + dd[None, :])
    origin = tl.load(Origin + (h * T + tt[:, None]) * D + dd[None, :])
    for ob in tl.range(0, O // BO, num_stages=STAGES):
        oo = ob * BO + tl.arange(0, BO)
        w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
        tl.store(Table + (h * T + tt[:, None]) * O + oo[None, :], tl.dot(v, w.to(tl.bfloat16)).to(tl.bfloat16))
        neg = -tl.dot(origin, w, input_precision="tf32")
        hi = neg.to(tl.bfloat16)
        corr = Corr + (tt[:, None] * (2 * H) + h) * O + oo[None, :]
        tl.store(corr, hi)
        tl.store(corr + H * O, (neg - hi.to(tl.float32)).to(tl.bfloat16))
    kk = tl.arange(0, K)
    for sb in tl.range(0, TB // TS, num_stages=STAGES):
        ts = tb * TB + sb * TS + tl.arange(0, TS)
        hk = (h * T + ts[:, None]) * K + kk[None, :]  # (token, key)
        idx = tl.load(Idx + hk)
        tl.store(Rows + ts[:, None] * (H * K) + h * K + kk[None, :], (idx + h * T).to(tl.int32))
        top = tl.load(Keys + (h * T + idx)[:, :, None] * D + dd[None, None, :]).to(tl.float32)  # (token, key, d)
        p = tl.load(Prob + hk)
        tail = tl.load(AllKeys + (h * T + ts[:, None]) * D + dd[None, :]) - tl.sum(p[:, :, None] * top, 1)
        mass = tl.maximum(1 - tl.sum(p, 1), 0)[:, None]
        centroid = tl.where(mass > 1e-5, tail / tl.maximum(mass, 1e-5), 0)
        tl.store(Shifted + ((ts[:, None, None] * H + h) * K + kk[None, :, None]) * D + dd[None, None, :],
                 (top - centroid[:, None, :]).to(tl.bfloat16))


def gathered_key_tables(values, origin, indices, w_proj_t, keys, prob, all_keys):
    """gathered_tables(values, origin, indices, w_proj_t) (with a tf32 correction) and run.py's
    strong_key_shifted(keys, indices, prob, all_keys) (tokens, H, K, D) bf16 from one launch:
    (shifted, (table, correction, rows))."""
    heads, tokens, width = values.shape
    count, o = indices.shape[-1], w_proj_t.shape[1]
    cfg = dict(FUSED_KEYS_CONFIG)
    tb, bo, ts = cfg.pop('TB'), cfg.pop('BO'), cfg.pop('TS')
    assert tokens % tb == 0 and o % bo == 0 and tb % ts == 0 and indices.shape[:2] == (heads, tokens)
    assert keys.shape == values.shape and prob.shape == indices.shape and all_keys.shape == values.shape
    dev = values.device
    table = torch.empty((heads * tokens, o), device=dev, dtype=torch.bfloat16)
    corr = torch.empty((tokens, 2 * heads, o), device=dev, dtype=torch.bfloat16)
    rows = torch.empty((tokens, heads * count), device=dev, dtype=torch.int32)
    shifted = torch.empty((tokens, heads, count, width), device=dev, dtype=keys.dtype)
    _gathered_key_tables_kernel[(heads, tokens // tb)](values.contiguous(), origin.float().contiguous(),
      w_proj_t.float().contiguous(), indices.contiguous(), table, corr, rows, keys.contiguous(),
      prob.float().contiguous(), all_keys.float().contiguous(), shifted, T=tokens, H=heads, K=count, D=width, O=o,
      TB=tb, BO=bo, TS=ts, **cfg)
    return shifted, (table, corr, rows)
