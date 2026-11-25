"""The Identity value tables' credits in token-id order: every update's plan and each step's operands."""
import torch


VOCAB = 4096
N = 2048
WIDTH = 512


def make_plan(rows):
    # Per update: each position's slot among its row's tokens sorted by id, and where each id's run starts.
    count = rows.shape[0]
    words, positions = rows.int().contiguous().sort(dim=1, stable=True)
    inverse = torch.empty_like(words)
    order = torch.arange(N, device=rows.device, dtype=torch.int32).expand(count, -1)
    inverse.scatter_(1, positions, order)
    keys = torch.arange(VOCAB + 1, dtype=torch.int32, device=rows.device).expand(count, -1).contiguous()
    starts = torch.searchsorted(words, keys, out_int32=True)
    return inverse, starts


def allocate(model):
    model.lexical_sums = model.flat_grad[model.owner_ve_start:][:N * 4 * WIDTH].view(N, 4, WIDTH)
    model.lexical_plan = make_plan(model.rows[:, model.rank, :N])


def bind_training_plan(model):
    for target, source in zip(model.lexical_plan, make_plan(model.rows[:, model.rank, :N])):
        target.copy_(source)


def prepare(model, gates):
    position, boundaries = (p.index_select(0, model.step_t.long().view(1))[0] for p in model.lexical_plan)
    return (torch.stack(gates), position, model.lexical_sums), boundaries
