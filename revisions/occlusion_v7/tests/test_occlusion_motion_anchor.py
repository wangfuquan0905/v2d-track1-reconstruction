import numpy as np
from v2d_syilen.occlusion_motion_anchor import local_depth_anchor,choose_anchored


def test_shared_distant_error_cannot_become_an_anchor():
    assert local_depth_anchor(np.array([0,0,20.]),np.array([0,0,20.01]),np.array([0,0,3.]),True) is None


def test_nearby_disagreement_and_partial_observation_cannot_anchor_depth():
    p=np.array([0,0,3.])
    assert local_depth_anchor(p,p+[0,0,.15],p,True) is None
    assert local_depth_anchor(p,p,p,False) is None


def test_nearby_agreeing_current_poses_supply_clear_depth():
    p=np.array([0,0,3.])
    assert local_depth_anchor(p,p+[0,0,.02],p,True)==3.01


def test_tiny_silhouette_gain_does_not_buy_clear_depth_drift():
    ob=np.array([[.1,0],[.09,0]])
    t=np.array([[0,0,3.],[0,0,2.97]])
    choice,unresolved=choose_anchored(ob,[True,True],t,3.)
    assert choice==0 and not unresolved


def test_anchor_cannot_erase_positive_pixels_or_bypass_motion_gate():
    ob=np.array([[1.,0],[.2,0],[0.,0]])
    t=np.array([[0,0,3.],[.01,0,3.],[0,0,12.]])
    choice,unresolved=choose_anchored(ob,[True,True,False],t,3.)
    assert choice==1 and not unresolved
