"""Finite object-orientation search with the trajectory objective's acceleration."""
import itertools
import numpy as np


def cube_rotations():
    """24 proper axis permutations, including identity and 180-degree flips."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            r = np.eye(3)[list(permutation)] * np.asarray(signs)[:, None]
            if np.linalg.det(r) > .5:
                rotations.append(r)
    rotations.sort(key=lambda r: float(np.linalg.norm(r-np.eye(3))))
    return np.stack(rotations).astype(np.float32)


def acceleration_path(unary, rotations, translations, weight=.1):
    """Exact second-order DP; same per-frame normalization as continuous fit.

    Unary includes weighted image and penetration terms. Human acceleration
    is constant during this object-only candidate search and can be omitted.
    """
    n, k = unary.shape
    if n < 3:
        raise ValueError('Orientation trajectory search requires at least 3 frames')
    costs = unary[0, :, None] + unary[1, None, :]
    back = np.zeros((n, k, k), dtype=np.int16)
    for frame in range(2, n):
        dr = (rotations[frame-2, :, None, None] - 2*rotations[frame-1, None, :, None]
              + rotations[frame, None, None, :])/.05
        dt = (translations[frame-2, :, None, None] - 2*translations[frame-1, None, :, None]
              + translations[frame, None, None, :])/.005
        acceleration = (np.square(dr).mean((-2, -1)) + np.square(dt).mean(-1))*weight*n/(n-2)
        candidates = costs[:, :, None] + acceleration
        back[frame] = candidates.argmin(axis=0)
        costs = candidates.min(axis=0) + unary[frame, None, :]
    path = np.zeros(n, dtype=np.int64)
    path[-2], path[-1] = np.unravel_index(costs.argmin(), costs.shape)
    objective = float(costs[path[-2], path[-1]]/n)
    for frame in range(n-1, 1, -1):
        path[frame-2] = back[frame, path[frame-1], path[frame]]
    return path, objective


def so3_path(unary, rotations, translations, fps, velocity_weight=1., acceleration_weight=1.,
             velocity_scale=np.pi/2, acceleration_scale=2*np.pi, linear_weight=.1,
             linear_velocity_weight=0., linear_acceleration_weight=0., linear_velocity_scale=.5, linear_acceleration_scale=1.):
    """Second-order DP matching camera-frame SO(3) derivatives in the fitter."""
    from scipy.spatial.transform import Rotation
    n, k = unary.shape
    if n < 3:
        raise ValueError('SO(3) trajectory search requires at least 3 frames')
    def pair_velocity(frame):
        relative = rotations[frame, None] @ rotations[frame-1, :, None].swapaxes(-1, -2)
        return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(k, k, 3)*fps
    def pair_translation_cost(frame):
        velocity = (translations[frame, None]-translations[frame-1, :, None])*fps
        return linear_velocity_weight*n/(n-1)*np.square(velocity/linear_velocity_scale).sum(-1)
    previous = pair_velocity(1)
    costs = unary[0, :, None]+unary[1, None, :]
    costs += velocity_weight*n/(n-1)*np.square(previous/velocity_scale).sum(-1)
    costs += pair_translation_cost(1)
    back = np.zeros((n, k, k), dtype=np.int16)
    for frame in range(2, n):
        current = pair_velocity(frame)
        alpha = (current[None]-previous[:, :, None])*fps
        dt = (translations[frame-2, :, None, None]-2*translations[frame-1, None, :, None]
              +translations[frame, None, None, :])
        linear_cost = (linear_acceleration_weight*np.square(dt*fps**2/linear_acceleration_scale).sum(-1)
                       if linear_acceleration_weight > 0 else linear_weight*np.square(dt/.005).mean(-1))
        penalty = n/(n-2)*(acceleration_weight*np.square(alpha/acceleration_scale).sum(-1)
                           +linear_cost)
        candidates = costs[:, :, None]+penalty
        back[frame] = candidates.argmin(0)
        costs = (candidates.min(0)+unary[frame, None, :]
                 +velocity_weight*n/(n-1)*np.square(current/velocity_scale).sum(-1)+pair_translation_cost(frame))
        previous = current
    path = np.zeros(n, dtype=np.int64)
    path[-2], path[-1] = np.unravel_index(costs.argmin(), costs.shape)
    objective = float(costs[path[-2], path[-1]]/n)
    for frame in range(n-1, 1, -1):
        path[frame-2] = back[frame, path[frame-1], path[frame]]
    return path, objective
