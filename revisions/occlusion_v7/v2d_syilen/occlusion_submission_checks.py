"""Independent NumPy-only motion audit, usable in the official packing runtime."""
import numpy as np


def check_motion(data, seed, fps, motion, decisions):
    r, t = np.asarray(data["rotation"]), np.asarray(data["translation"])
    sr, st = np.asarray(seed["rotation"]), np.asarray(seed["translation"])
    steps = np.linalg.norm(np.diff(t, axis=0), axis=-1)
    relative = r[1:] @ r[:-1].swapaxes(-1, -2)
    angles = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
    if np.any(steps > motion["max_speed_mps"]/fps+1e-8):
        raise ValueError("Translation speed exceeds motion gate")
    if np.any(angles > np.deg2rad(motion["max_angular_speed_degps"])/fps+1e-8):
        raise ValueError("Rotation speed exceeds motion gate")
    if np.any(np.linalg.norm(t-st, axis=-1) > motion["refinement_radius_m"]+1e-8):
        raise ValueError("Refinement exceeds seed trust region")
    locked = seed["depth_locked"].astype(bool)
    missing = seed["states"] == "missing"
    frozen = missing | np.r_[missing[1:], False]
    np.testing.assert_allclose(t[locked,2], st[locked,2], atol=1e-9, rtol=0)
    np.testing.assert_allclose(t[frozen], st[frozen], atol=1e-9, rtol=0)
    np.testing.assert_allclose(r[frozen], sr[frozen], atol=1e-9, rtol=0)
    if len(decisions) != len(t):
        raise ValueError("Missing per-frame motion decisions")
    for i in np.flatnonzero(missing):
        if i and decisions[i]["prediction"] == "hold":
            np.testing.assert_allclose(t[i], t[i-1], atol=1e-9, rtol=0)
        np.testing.assert_allclose(t[i], decisions[i]["prediction_translation_m"], atol=1e-9, rtol=0)
    return {"max_step_m": float(steps.max()), "max_angular_step_deg": float(np.rad2deg(angles.max())),
            "missing_frames": int(missing.sum()), "independent_motion_check": True}
