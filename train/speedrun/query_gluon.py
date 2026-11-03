"""Gluon (Hopper) routed query replay: query_replay.fused_query_replay (fused_query_weights(delta @ w_q.T, ...)) with
the delta tile read once.

Persistent CTAs (one per SM, two warpgroups of 64 moves) walk tiles of BM = 128 moves of one token. The tile's delta
(BM x C bf16) comes in by TMA and stays in shared memory for all heads; w_q streams through a TMA ring, a pair of
heads (128 rows) by 64 columns per item; per head pair raw = bf16(delta @ w_q[pair]^T) accumulates on wgmma, then the
reference's replay epilogue runs on it in registers: w_q's rows are interleaved per head (a0 b0 a1 b1 ...) so each
RoPE pair sits in one thread; the key scores of both heads come from one wgmma against a block-diagonal key matrix
(exact: the off-diagonal zeros add nothing), its A operand the epilogue's dq in the accumulator layout.
"""
import torch

EPS = torch.finfo(torch.float32).eps


def interleave_wq(w_q, heads):
    """w_q's rows per head reordered a0 b0 a1 b1 ... (the RoPE halves interleaved), bf16."""
    c = w_q.shape[0]
    return w_q.to(torch.bfloat16).view(heads, 2, c // heads // 2, -1).transpose(1, 2).reshape(c, -1).contiguous()
