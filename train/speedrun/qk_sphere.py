"""The per-head tangent constraint of the c_q / c_k update."""


def constrain(old, proposed, heads: int):
    """old, proposed: (batch, rows, columns) weights before and after an update. Removes from each head's part of the
    update (rows / heads consecutive output rows of one matrix) its component along that head's current weights, so
    the update is tangent to the sphere of that head's weights. heads == 1: proposed, unchanged."""
    if heads == 1:
        return proposed
    shape = old.shape
    w = old.reshape(shape[0], heads, -1)
    p = proposed.reshape_as(w)
    power = w.square().sum(-1, keepdim=True)
    delta = p - w
    radial = (delta * w).sum(-1, keepdim=True) / power.clamp_min(1e-30)
    result = w + delta - radial * w
    return result.reshape(shape)
