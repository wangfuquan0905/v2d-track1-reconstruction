"""Observation-preserving candidates for the v4 occlusion experiment.

Public reference poses are never inputs. Keep the frozen v3 objective intact.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def bounded_anchor_reference(rotations, translations, reliable, fallback_r,
                             fallback_t, max_gap_frames):
    """Interpolate short bracketed gaps; never extend an endpoint into a prefix.

    Unsupported frames keep their original local prediction and have no anchor
    prior. A single anchor constrains only its own frame. Missing all anchors is
    an uncertainty condition, not permission to invent a constant trajectory.
    """
    if max_gap_frames < 1:
        raise ValueError('max_gap_frames must be positive')
    ids = np.flatnonzero(reliable)
    r = np.array(fallback_r, dtype=float, copy=True)
    t = np.array(fallback_t, dtype=float, copy=True)
    support = np.zeros(len(t), dtype=bool)
    distance = np.full(len(t), np.inf)
    for i in ids:
        r[i], t[i], support[i], distance[i] = rotations[i], translations[i], True, 0.
    for left, right in zip(ids[:-1], ids[1:]):
        if right-left > max_gap_frames:
            continue
        query = np.arange(left, right+1)
        fraction = (query-left)/(right-left)
        t[query] = (1-fraction[:, None])*translations[left]+fraction[:, None]*translations[right]
        r[query] = Slerp([left, right], Rotation.from_matrix(rotations[[left, right]]))(query).as_matrix()
        support[query] = True
        distance[query] = np.minimum(query-left, right-query)
    return r, t, distance, support


def pose_candidates(pred_r, pred_t, source_r, source_t, v1_r, v1_t, v2_r, v2_t, ref_r, ref_t):
    """Every hypothesis keeps its own translation, including partial frames."""
    return (np.stack([pred_r, source_r, v1_r, v2_r, ref_r], axis=1),
            np.stack([pred_t, source_t, v1_t, v2_t, ref_t], axis=1))


def evidence_limits(baseline_observations, observed, tolerance=.05):
    """Retain the best original/v1 censored silhouette within a fixed budget.

    This feasibility constraint is independent of visibility confidence. It
    protects actual observed pixels even when confidence is near zero. It
    cannot certify a mask fit when all original hypotheses already fail.
    """
    baseline = np.min(np.asarray(baseline_observations)[..., 0], axis=1)
    limit = np.minimum(1., baseline+tolerance)
    return np.where(observed, limit, np.inf)


def evidence_feasible(observations, limits):
    loss = np.asarray(observations)[..., 0]
    limit = np.asarray(limits)
    if loss.ndim == 2:
        limit = limit[:, None]
    return np.isfinite(loss) & (loss <= limit+1e-7)


def require_available_overlap(observations, limits, observed, minimum_iou=.1):
    """Do not choose zero overlap when any hypothesis explains visible pixels.

    If every candidate already misses the mask this cannot manufacture a
    measurement; the producer explicitly reports such unresolved frames.
    """
    best_iou=1-np.min(np.asarray(observations)[...,0],axis=1)
    required=np.minimum(minimum_iou,np.maximum(0.,best_iou))
    return np.minimum(limits,np.where(observed,1-required,np.inf))


def constrain_unary(unary, observations, limits):
    allowed = evidence_feasible(observations, limits)
    if not allowed.any(axis=1).all():
        raise ValueError('No observation-preserving candidate for at least one frame')
    return np.where(allowed, unary, np.inf), allowed
