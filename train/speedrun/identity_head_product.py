"""The Identity phase's loss and lm_head gradient: every logit's finite-difference response times the head's input."""
import torch

import cap_lookup


def prepare(raw, y, tables):
    loss, response = cap_lookup.run(raw, y, tables)
    return loss.mean(), response


@torch.compile(dynamic=False)
def staged_product(hidden, response, out):
    out.copy_(torch.mm(hidden.to(torch.float16).T, response, out_dtype=torch.float32) * .015625)
