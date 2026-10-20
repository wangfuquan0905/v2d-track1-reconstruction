import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from v2d_syilen.occlusion_candidates import (bounded_anchor_reference, pose_candidates,
    evidence_limits, evidence_feasible, constrain_unary, require_available_overlap)
from v2d_syilen.occlusion_objective import second_order_path


def test_distant_future_anchor_cannot_replace_the_prefix_or_suffix():
    n=400
    r=np.tile(np.eye(3),(n,1,1));t=np.tile([0.,1.,3.],(n,1))
    anchor=t.copy();anchor[334]=[0.,0.,5.]
    reliable=np.arange(n)==334
    rr,tt,d,supported=bounded_anchor_reference(r,anchor,reliable,r,t,60)
    np.testing.assert_array_equal(tt[~reliable],t[~reliable])
    np.testing.assert_array_equal(tt[334],anchor[334])
    assert supported.sum()==1 and np.isinf(d[0]) and np.isinf(d[-1])


def test_short_bracketed_gap_interpolates_but_long_gap_does_not():
    n=12
    r=Rotation.from_euler('z',np.arange(n)*5,degrees=True).as_matrix()
    t=np.arange(n)[:,None]*np.array([.1,0,.02])+[0,0,3]
    fallback=np.tile([9.,9.,9.],(n,1));reliable=np.isin(np.arange(n),[1,4,11])
    rr,tt,d,s=bounded_anchor_reference(r,t,reliable,np.tile(np.eye(3),(n,1,1)),fallback,3)
    np.testing.assert_allclose(tt[1:5],t[1:5]);np.testing.assert_allclose(rr[1:5],r[1:5],atol=1e-12)
    np.testing.assert_array_equal(tt[5:11],fallback[5:11]);assert not s[0]


def test_no_anchor_retains_original_predictions_and_no_anchor_prior():
    r=np.tile(np.eye(3),(4,1,1));t=np.arange(12).reshape(4,3)
    _,tt,d,s=bounded_anchor_reference(r,t,np.zeros(4,bool),r,t,2)
    np.testing.assert_array_equal(tt,t);assert not s.any() and np.isinf(d).all()


def test_original_translation_survives_a_conflicting_temporal_reference():
    r=np.tile(np.eye(3),(5,1,1));t=np.tile([0.,1.,3.],(5,1));ref=t+[0,-1,10]
    rr,tt=pose_candidates(r,t,r,t,r,t+.1,r,t+.2,r,ref)
    np.testing.assert_array_equal(tt[:,1],t)
    np.testing.assert_array_equal(tt[:,4],ref)
    assert not np.shares_memory(tt,t)


def test_partial_positive_evidence_cannot_be_bought_by_a_smooth_wrong_track():
    n=6;k=2
    r=np.broadcast_to(np.eye(3),(n,k,3,3)).copy()
    t=np.zeros((n,k,3));t[:,0,2]=3+np.arange(n)%2*.01;t[:,1,2]=8
    ob=np.zeros((n,k,2));ob[:,0,0]=.3;ob[:,1,0]=1
    limits=evidence_limits(ob[:,:1],np.ones(n,bool))
    # Even arbitrarily strong preference for the distant zero-overlap path
    # cannot erase existing observed silhouette evidence.
    unary=np.tile([10000.,0.],(n,1));unary,allowed=constrain_unary(unary,ob,limits)
    path,value=second_order_path(unary,r,t,30)
    assert np.isfinite(value);np.testing.assert_array_equal(path,np.zeros(n,int))
    assert not allowed[:,1].any()


def test_empty_masks_have_no_image_constraint_and_final_regression_is_rejected():
    base=np.array([[[.2,0]],[[0.,0]]]);limits=evidence_limits(base,[True,False])
    assert np.isinf(limits[1])
    np.testing.assert_array_equal(evidence_feasible([[.24,0],[0.,0]],limits),[True,True])
    np.testing.assert_array_equal(evidence_feasible([[1.,0],[0.,0]],limits),[False,True])
    with pytest.raises(ValueError):constrain_unary(np.zeros((2,1)),np.array([[[1.,0]],[[0.,0]]]),limits)


def test_failed_originals_cannot_hide_an_available_observed_object_candidate():
    ob=np.array([[[1.,0],[.6,0]],[[1.,0],[.98,0]],[[0.,0],[0.,0]]])
    observed=np.array([True,True,False])
    limits=evidence_limits(ob[:,:1],observed)
    limits=require_available_overlap(ob,limits,observed)
    _,allowed=constrain_unary(np.zeros((3,2)),ob,limits)
    np.testing.assert_array_equal(allowed[:2],[[False,True],[False,True]])
    assert allowed[2].all()
