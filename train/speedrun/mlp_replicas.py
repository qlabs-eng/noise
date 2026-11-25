"""Publish dense MLP weights in the precision each consumer reads."""
import torch
import torch.distributed as dist


@torch.compile(dynamic=False)
def _pack_norms(reader, writer, out):
    out[0].copy_(reader.float().norm(dim=1))
    out[1].copy_(writer.float().norm(dim=0))


class MLPPublication:
    def __init__(self, anvil, model, group):
        self.group = group
        self.group_index = len(anvil.rank_groups) - 1
        rank, world = dist.get_rank(), dist.get_world_size()
        self.proposal = anvil.rank_sends[self.group_index]
        self.chunk = self.proposal.numel()
        self.master = anvil.rank_stores[self.group_index].view(world, self.chunk)[rank]
        layers = model.transformer.h
        hidden, width = layers[0].mlp.c_fc.weight.shape
        # Two FP32 vectors take four BF16 words per hidden feature. The all-gather copies their bits.
        self.send = self.proposal.new_empty((self.chunk + 4 * hidden,), dtype=torch.bfloat16)
        self.received = self.send.new_empty((world, self.send.numel()))
        self.send_norms = self.send[self.chunk:].view(torch.float32).view(2, hidden)
        self.norms = self.received[:, self.chunk:].view(torch.float32).view(world, 2, hidden)
        self.weights = {}
        local = {}
        for index in anvil.rank_groups[self.group_index]:
            parameters = anvil.batches[index][0]
            shape = parameters[0].shape
            kind = 'c_fc' if shape == (hidden, width) else 'c_proj'
            start = anvil.rank_offsets[index]
            stop = start + parameters[0].numel()
            local[kind] = self.master[start:stop].view(shape)
            for layer in range(len(parameters)):
                name = f'transformer.h.{layer}.mlp.{kind}'
                self.weights[name] = self.received[layer, start:stop].view(shape)
        self.reader, self.writer = local['c_fc'], local['c_proj']
        self.full = True

    def _gather(self, asynchronous):
        self.send[:self.chunk].copy_(self.master)
        _pack_norms(self.reader, self.writer, self.send_norms)
        return dist.all_gather_into_tensor(self.received.view(-1), self.send,
                                           group=self.group, async_op=asynchronous)

    @torch.no_grad()
    def initialize(self):
        self._gather(False)

    @torch.no_grad()
    def publish(self):
        self.master.copy_(self.proposal)
        return self._gather(True)

    def route_norms(self, first):
        return self.norms[first:, 0].contiguous(), self.norms[first:, 1].contiguous()
