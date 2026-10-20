"""Make GPU object transforms independent of candidate-list remainder size.

On this runtime, torch.einsum uses different float32 contraction paths for
batch sizes 2--5 and 1/6--8. A one-ULP vertex difference can move a silhouette
edge and violate strict rerender checks. Pad the last batch with repeats,
evaluate the same eight-slot kernel shape, then discard all padding outputs.
"""
import numpy as np
import torch
from occlusion_scene import Scene as OriginalScene


def padded(r, t, ids, size):
    n = len(ids)
    if not 0 < n <= size or len(r) != n or len(t) != n:
        raise ValueError("Expected a nonempty matching pose/frame chunk")
    extra = size-n
    return (np.concatenate([r, np.repeat(r[-1:], extra, axis=0)]),
            np.concatenate([t, np.repeat(t[-1:], extra, axis=0)]),
            np.concatenate([ids, np.repeat(ids[-1:], extra)]))


class Scene(OriginalScene):
    @torch.no_grad()
    def render(self, r, t, ids, visible=False):
        parts = []
        for start in range(0, len(ids), self.batch):
            rr, tt, ii = r[start:start+self.batch], t[start:start+self.batch], ids[start:start+self.batch]
            count = len(ii)
            rr, tt, ii = padded(rr, tt, ii, self.batch)
            parts.append(super().render(rr, tt, ii, visible=visible)[:count])
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def observe(self, r, t, ids=None):
        if ids is None:
            ids = np.arange(self.n)
        parts = []
        for start in range(0, len(ids), self.batch):
            rr, tt, ii = r[start:start+self.batch], t[start:start+self.batch], ids[start:start+self.batch]
            count = len(ii)
            rr, tt, ii = padded(rr, tt, ii, self.batch)
            # Pad the image/loss batch as well as the vertex contraction.
            parts.append(super().observe(rr, tt, ii)[:count])
        return np.concatenate(parts)
