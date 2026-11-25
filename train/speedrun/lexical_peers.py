"""The Identity value tables' credits sent to their owner ranks: each rank sums its positions' credits per token id
and an all-to-all delivers each vocabulary row to the rank that owns it."""
import torch
import torch.distributed as dist
import triton
import triton.language as tl


V = 4096
N = 2048
SHARD = 512
FEATURES = 2048
OWNERS = V // SHARD


@triton.jit
def _compact(Starts, Words, Sizes, V: tl.constexpr, N: tl.constexpr):
    step = tl.program_id(0)
    word = tl.arange(0, V)
    ptr = Starts + step * (V + 1) + word
    present = tl.load(ptr + 1) > tl.load(ptr)
    index = tl.cumsum(present.to(tl.int32), 0) - 1
    tl.store(Words + step * N + index, word, present)
    tl.store(Sizes + step, tl.sum(present.to(tl.int32), 0))


def make_plan(starts, group):
    steps = len(starts)
    words = torch.empty((steps, N), device=starts.device, dtype=torch.int32)
    sizes = torch.empty((steps,), device=starts.device, dtype=torch.int32)
    _compact[(steps,)](starts, words, sizes, V, N, num_warps=8)
    present = (starts[:, 1:] > starts[:, :-1]).view(steps, OWNERS, SHARD)
    send = present.permute(1, 0, 2).contiguous().to(torch.uint8)
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    bit = torch.arange(OWNERS, device=starts.device, dtype=torch.int32)[:, None, None]
    mask = (receive.int() << bit).sum(0).to(torch.uint8)
    return words, sizes, mask


@triton.jit
def _publish(Source, Starts, Words, Sizes, Step, Send, N: tl.constexpr, FEATURES: tl.constexpr,
             BLOCK: tl.constexpr, K: tl.constexpr):
    slot = tl.program_id(0)
    step = tl.load(Step).to(tl.int32)
    if slot < tl.load(Sizes + step):
        word = tl.load(Words + step * N + slot)
        first = tl.load(Starts + word)
        last = tl.load(Starts + word + 1)
        x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        rows = tl.arange(0, K)
        total = tl.full((BLOCK,), 0, tl.float32)
        for i in range(first, last, K):
            pos = i + rows
            values = tl.load(Source + pos[:, None] * FEATURES + x[None, :], pos[:, None] < last, 0.)
            total += tl.sum(values, 0)
        tl.store(Send + word * FEATURES + x, total * 32.)


@triton.jit
def _receive(Inbox, Masks, Step, Out, OWNERS: tl.constexpr, SHARD: tl.constexpr,
             FEATURES: tl.constexpr, BLOCK: tl.constexpr):
    word = tl.program_id(0)
    shift = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    step = tl.load(Step).to(tl.int32)
    mask = tl.load(Masks + step * SHARD + word).to(tl.int32)
    total = tl.full((BLOCK,), 0., tl.float32)
    for sender in tl.static_range(OWNERS):
        if (mask & (1 << sender)) != 0:
            address = (sender * SHARD + word) * FEATURES
            total += tl.load(Inbox + address + shift).to(tl.float32)
    tl.store(Out + word * FEATURES + shift, total)


class Exchange:
    def __init__(self, model):
        self.group = model.owner_group
        # The first axis names the destination on send and the sender on receive.
        self.send = torch.zeros((OWNERS, SHARD, FEATURES), device=model.device, dtype=torch.float16)
        self.inbox = torch.empty_like(self.send)
        self.plan = make_plan(model.lexical_plan[1], self.group)

    def bind(self, starts):
        for dest, source in zip(self.plan, make_plan(starts, self.group)):
            dest.copy_(source)
        self.send.zero_()

    def exchange(self, source, starts, step, out):
        words, sizes, masks = self.plan
        _publish[(N, 4)](source, starts, words, sizes, step, self.send, N, FEATURES, 512, 8, num_warps=4)
        dist.all_to_all_single(self.inbox, self.send, group=self.group)
        _receive[(SHARD, 4)](self.inbox, masks, step, out, OWNERS, SHARD, FEATURES, 512, num_warps=4)
