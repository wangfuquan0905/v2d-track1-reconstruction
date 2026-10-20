"""Causal motion gates. Image evidence cannot override a motion rejection.

Coordinates are those of the input camera. Bounds are explicit experiment
assumptions, not estimates from GT. Large camera motion needs a world-frame
motion model instead. Partial masks never supply a depth innovation.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MotionConfig:
    max_speed_mps: float = 2.0
    max_angular_speed_degps: float = 180.0
    innovation_m: float = .04
    initial_radius_m: float = .02
    refinement_radius_m: float = .01
    velocity_samples: int = 5
    velocity_spread_mps: float = .15
    velocity_confidence: float = .5
    velocity_iou: float = .2
    max_prediction_seconds: float = .2
    reacquire_frames: int = 3
    minimum_fit_iou: float = .02

    def dictionary(self):
        return asdict(self)


def rotation_distance(a, b):
    return np.linalg.norm(Rotation.from_matrix(np.asarray(a) @ np.asarray(b).swapaxes(-1, -2)).as_rotvec(), axis=-1)


def candidate_gate(rotations, translations, predicted_r, predicted_t, previous_r,
                   previous_t, fps, depth_locked, config=MotionConfig()):
    r, t = np.asarray(rotations), np.asarray(translations)
    initial = previous_t is None
    radius = config.initial_radius_m if initial else config.innovation_m
    ok = np.isfinite(t).all(-1) & (t[..., 2] > 0)
    ok &= np.linalg.norm(t-predicted_t, axis=-1) <= radius+1e-9
    ok &= rotation_distance(r, predicted_r) <= np.deg2rad(config.max_angular_speed_degps)/fps+1e-9
    if depth_locked:
        ok &= np.abs(t[..., 2]-predicted_t[2]) <= 1e-9
    if not initial:
        ok &= np.linalg.norm(t-previous_t, axis=-1) <= config.max_speed_mps/fps+1e-9
        ok &= rotation_distance(r, previous_r) <= np.deg2rad(config.max_angular_speed_degps)/fps+1e-9
    return ok


class MotionState:
    """Short reliable velocity, otherwise hold; never use a future endpoint.

    Image-fit updates do not become trusted velocity samples during low
    confidence, missing observations, or the reacquisition quarantine.
    """
    def __init__(self, fps, config=MotionConfig()):
        if fps <= 0:
            raise ValueError('fps must be positive')
        self.fps, self.config = fps, config
        self.positions, self.reliable = [], []
        self.rotation = None
        self.velocity = np.zeros(3)
        self.prediction_age = config.max_prediction_seconds+1
        self.reacquiring = False
        self.reacquire_count = 0

    def predict(self, initial_r, initial_t, missing):
        if missing:
            self.reacquiring = True
            self.reacquire_count = 0
        if not self.positions:
            return initial_r.copy(), initial_t.copy(), 'initial_original'
        cfg = self.config
        k = cfg.velocity_samples
        fresh = len(self.positions) >= k+1 and all(self.reliable[-k-1:])
        if fresh:
            velocities = np.diff(np.asarray(self.positions[-k-1:]), axis=0)*self.fps
            velocity = np.median(velocities, axis=0)
            spread = np.max(np.linalg.norm(velocities-velocity, axis=-1))
            if spread <= cfg.velocity_spread_mps and np.linalg.norm(velocity) <= cfg.max_speed_mps:
                self.velocity = velocity
                self.prediction_age = 0.
            else:
                self.velocity = np.zeros(3)
                self.prediction_age = cfg.max_prediction_seconds+1
        self.prediction_age += 1/self.fps
        use_velocity = self.prediction_age <= cfg.max_prediction_seconds+1e-9
        shift = self.velocity/self.fps if use_velocity else np.zeros(3)
        mode = 'bounded_velocity' if use_velocity else 'hold'
        return self.rotation.copy(), self.positions[-1]+shift, mode

    def commit(self, rotation, translation, confidence, iou, observed, consistent_observation):
        cfg = self.config
        if self.reacquiring and observed:
            self.reacquire_count = self.reacquire_count+1 if consistent_observation else 0
            if self.reacquire_count >= cfg.reacquire_frames:
                self.reacquiring = False
        trusted = (observed and consistent_observation and not self.reacquiring
                   and confidence >= cfg.velocity_confidence and iou >= cfg.velocity_iou)
        self.rotation = np.array(rotation, copy=True)
        self.positions.append(np.array(translation, copy=True))
        self.reliable.append(bool(trusted))
        return bool(trusted)


def choose_local(observations, allowed, prediction_index=0, minimum_fit_iou=.02):
    """Visible evidence is optimized only inside the motion-feasible set.

    A good remote silhouette is never a reason to relax the motion gate.
    The fallback is the prediction, and an unresolved frame remains explicit.
    """
    ob = np.asarray(observations)
    valid = np.asarray(allowed, dtype=bool) & np.isfinite(ob).all(-1)
    if not valid[prediction_index]:
        raise ValueError('The motion prediction must remain feasible')
    cost = ob[:, 0]+.02*ob[:, 1]
    best = int(np.argmin(np.where(valid, cost, np.inf)))
    unresolved = bool(1-ob[best, 0] < minimum_fit_iou)
    return (prediction_index if unresolved else best), unresolved


def trajectory_feasible(rotations, translations, seed_r, seed_t, states, fps,
                        config=MotionConfig(), depth_locked=None):
    """Check actual adjacent poses, including window seams, before acceptance."""
    r, t = np.asarray(rotations), np.asarray(translations)
    if not (np.isfinite(r).all() and np.isfinite(t).all()):
        return False
    if np.any(np.linalg.norm(np.diff(t, axis=0), axis=-1) > config.max_speed_mps/fps+1e-8):
        return False
    if np.any(rotation_distance(r[1:], r[:-1]) > np.deg2rad(config.max_angular_speed_degps)/fps+1e-8):
        return False
    if np.any(np.linalg.norm(t-seed_t, axis=-1) > config.refinement_radius_m+1e-8):
        return False
    missing = np.asarray(states) == 'missing'
    frozen = missing | np.r_[missing[1:], False]
    locked = np.asarray(states) != 'clear' if depth_locked is None else np.asarray(depth_locked)
    if not np.allclose(t[locked, 2], seed_t[locked, 2], atol=1e-9, rtol=0):
        return False
    if not np.allclose(t[frozen], seed_t[frozen], atol=1e-9, rtol=0):
        return False
    if not np.allclose(r[frozen], seed_r[frozen], atol=1e-9, rtol=0):
        return False
    return True
