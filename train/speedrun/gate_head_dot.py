"""Each head's raw value credit dotted with its token's value embedding, read by both Cut value-gate credits."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def emit(raw, sources, out, layer_row, head, rows, channels, T: gl.constexpr,
         ROWS: gl.constexpr):
    layer = layer_row // ROWS
    if layer % 2 == 1:
        slot = layer // 2
        source = sources[0]
        for i in gl.static_range(1, 4):
            source = gl.where(slot == i, sources[i], source)
        row = layer_row % ROWS * T + rows
        lexical = gl.load(source + row[:, None] * 512 + head * 64 + channels[None, :])
        projection = gl.sum(raw * lexical.to(gl.float32), axis=1)
        gl.store(out + (slot * ROWS * T + row) * 8 + head, projection)


@torch.compile(dynamic=False)
def parameter_response(logits, projection):
    z = logits.float()
    delta = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
    return delta * projection


def arguments(capture):
    ids = capture.inputs['transformer.wte']
    sources = tuple(capture.outputs[f'value_embeds.{i}'] for i in (1, 3, 5, 7))
    rows, t = ids.shape
    dots = torch.empty((4, rows, t, 8), device=ids.device, dtype=torch.float32)
    return sources, dots
