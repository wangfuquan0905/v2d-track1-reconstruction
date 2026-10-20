import numpy as np
import pytest
from v2d_syilen.occlusion_submission_checks import check_motion


def example():
    t = np.zeros((5,3))
    t[:,2] = 3.
    r = np.repeat(np.eye(3)[None],5,axis=0)
    data = {"translation": t.copy(), "rotation": r.copy()}
    seed = {"translation":t, "rotation":r, "states":np.array(["clear","partial","missing","missing","clear"]),
            "depth_locked":np.array([False,True,True,True,True])}
    decisions = [{"prediction":"hold","prediction_translation_m":row.tolist()} for row in t]
    motion = {"max_speed_mps":2., "max_angular_speed_degps":180., "refinement_radius_m":.01}
    return data,seed,30.,motion,decisions


def test_missing_hold_passes():
    assert check_motion(*example())["missing_frames"] == 2


def test_jump_is_rejected_even_with_matching_seed():
    args = list(example())
    args[0]["translation"][3,0] += 5
    args[1]["translation"][3,0] += 5
    with pytest.raises(ValueError,match="Translation speed"):
        check_motion(*args)


def test_partial_depth_cannot_change_inside_small_step_limit():
    args = list(example())
    args[0]["translation"][1,2] += .001
    with pytest.raises(AssertionError):
        check_motion(*args)


def test_missing_cannot_move_even_if_seed_was_changed():
    args = list(example())
    args[0]["translation"][3,0] += .001
    args[1]["translation"][3,0] += .001
    with pytest.raises(AssertionError):
        check_motion(*args)
