"""Block 5's attention replay for root 4: block 4's MLP writer and each row's frame through block 5's query matrix,
and each row's selected queries with their strong keys and value tables."""
import torch
import triton
from torch.nn import functional as F
import bin_attention
import query_factored_gluon as qf


def project_weights(writer, frames, query):
    """The full product writer @ query^T and each row's frame @ query^T (bf16; query's RoPE halves interleaved)."""
    rows, width = writer.shape
    packed_query = qf.interleave_wq(query, width // 64)
    table = torch.empty((rows, width), device=writer.device, dtype=torch.bfloat16)
    every = torch.arange(rows, dtype=torch.int64, device=writer.device)
    cfg = qf.FACTOR_CONFIG
    qf._factor_at_kernel[(triton.cdiv(rows, cfg['BM']), width // cfg['BN'])](
        writer.contiguous(), every, packed_query, table, rows, width, width, 1, **cfg)
    sizes = [f.shape[0] for f in frames]
    projected = (torch.cat(frames, dim=0).float() @ packed_query.float().T).bfloat16()
    return table, projected.split(sizes)


@torch.compile(dynamic=False)
def rotated_queries(inputs, query, cosine, sine):
    width = 64
    heads = query.shape[0] // width
    normalized = F.rms_norm(inputs.float(), (inputs.shape[-1],))
    raw = (normalized @ query.float().T).unflatten(-1, (heads, width))
    a, b = raw.split(width // 2, dim=-1)
    c, s = cosine.unsqueeze(-2), sine.unsqueeze(-2)
    return torch.cat((a * c + b * s, b * c - a * s), dim=-1)


@torch.compile(dynamic=False)
def unit_query(rotated):
    return F.rms_norm(rotated, (rotated.shape[-1],))


def prepare_selected(raw_own, inside, count, product, frame):
    pos, inputs, wq, keys, values, wp, cos, sin, windows = raw_own
    site = 5
    selected = pos[inside]
    x = inputs[site][inside]
    reference = rotated_queries(x, wq[site], cos[selected], sin[selected])
    q = unit_query(reference)
    prob, ix, mean_value, mean_key = bin_attention.moments(q, keys[site], values[site], selected, windows[site], count)
    shifted, payload = bin_attention.tables(values[site], mean_value, ix, wp[site], keys[site], prob, mean_key)
    def layer_list(item):
        entries = [None] * len(inputs)
        entries[site] = item
        return entries
    return (selected, layer_list(x), layer_list(shifted), layer_list(payload),
            layer_list(prob.transpose(0, 1).contiguous()), (cos, sin), (product, frame, reference))
