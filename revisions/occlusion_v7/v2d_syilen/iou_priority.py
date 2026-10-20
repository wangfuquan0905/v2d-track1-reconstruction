"""IoU-first refinement policy with bounded appearance/acceleration auxiliaries.

This is a new experimental objective. Frozen v1 producers keep their original
objective and hashes. Every temporal contribution is local to a frame triple,
so a two-frame context still gives exact whole-clip acceptance deltas.
"""
import numpy as np
from scipy.spatial.transform import Rotation

COLOR_WEIGHT = .02
ANGULAR_WEIGHT = .005
LINEAR_WEIGHT = .005
MAX_FRAME_IOU_DROP = .01


def bounded(x):
    """Saturate a nonnegative residual below one, without a hard cutoff."""
    return x / (1 + x)


def observation_loss(observations):
    return observations[..., 0] + COLOR_WEIGHT * bounded(observations[..., 1])


def temporal_residuals(rotations, translations, fps):
    omega = Rotation.from_matrix(
        rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    ).as_rotvec() * fps
    angular = np.sum((np.diff(omega, axis=0) * fps / (2 * np.pi)) ** 2, axis=1)
    linear = np.sum((np.diff(translations, n=2, axis=0) * fps**2) ** 2, axis=1)
    return angular, linear


def score(observations, rotations, translations, fps):
    ob = np.asarray(observations, dtype=np.float64)
    n = len(ob)
    angular, linear = temporal_residuals(rotations, translations, fps)
    terms = {
        'object_silhouette': float(ob[:, 0].mean()),
        'object_color': float(ob[:, 1].mean()),
        'color_contribution': float(COLOR_WEIGHT * bounded(ob[:, 1]).mean()),
        'angular_contribution': float(ANGULAR_WEIGHT * bounded(angular).sum() / n),
        'linear_contribution': float(LINEAR_WEIGHT * bounded(linear).sum() / n),
        'angular_acceleration': float(angular.sum() / n),
        'linear_acceleration': float(linear.sum() / n),
    }
    terms['auxiliary'] = sum(terms[k] for k in
                             ['color_contribution', 'angular_contribution', 'linear_contribution'])
    terms['total'] = terms['object_silhouette'] + terms['auxiliary']
    return terms


def accepts(old, new, new_observations, best_silhouette):
    """Do not trade mean overlap for smoothness or hide a badly regressed frame.

    The per-frame floor is measured against the best overlap seen in the run,
    not the previous iteration, so repeated small regressions cannot accumulate.
    """
    ob = np.asarray(new_observations)
    return bool(
        np.isfinite(list(new.values())).all() and np.isfinite(ob).all()
        and new['total'] < old['total'] - 1e-9
        and new['object_silhouette'] <= old['object_silhouette']
        and np.all(ob[:, 0] <= best_silhouette + MAX_FRAME_IOU_DROP + 1e-7)
    )


def mask_box_translation(vertices, rotation, box, intrinsics, depth_factor=1.):
    """Image-derived translation proposal; final selection uses the real renderer.

    A visible mask box can be truncated by occlusion. This is only a search
    seed, not an asserted depth measurement or an extra observation loss.
    """
    local = np.asarray(vertices) @ rotation.T
    center = (local.min(0) + local.max(0)) / 2
    extent = np.maximum(np.ptp(local, axis=0), 1e-6)
    box = np.asarray(box, dtype=float)
    size = np.maximum(box[2:] - box[:2], 1)
    uv = (box[:2] + box[2:]) / 2
    focal, principal = np.asarray(intrinsics[:2]), np.asarray(intrinsics[2:])
    z = max(float(np.sqrt(np.prod(focal * extent[:2] / size))) * depth_factor,
            extent[2] / 2 + .02)
    translation = np.r_[(uv - principal) / focal * z, z] - center
    # Correct perspective-induced box center displacement while retaining depth.
    for _ in range(3):
        camera = local + translation
        projected = camera[:, :2] / camera[:, 2:] * focal + principal
        predicted_center = (projected.min(0) + projected.max(0)) / 2
        translation[:2] += (uv - predicted_center) / focal * z
    return translation
