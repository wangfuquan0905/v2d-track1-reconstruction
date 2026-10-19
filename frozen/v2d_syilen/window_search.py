"""Boundary-complete support for replacing a contiguous trajectory window."""
import numpy as np
from .window_objective import whole_window_score


def context_bounds(start, stop, length):
    if not 0 <= start < stop <= length:
        raise ValueError('Invalid half-open window')
    # An acceleration triple can include up to two fixed frames outside it.
    return max(0, start-2), min(length, stop+2)


def objective_sum(observations, rotations, translations, fps):
    return len(rotations)*whole_window_score(observations, rotations, translations, fps)['total']


def accepts(old, new):
    return bool(np.isfinite(new) and new < old-1e-6)
