import numpy as np
from scipy.spatial.transform import Rotation
from v2d_syilen.occlusion_motion import (MotionConfig, MotionState, candidate_gate,
    choose_local, trajectory_feasible)


def test_low_confidence_motion_does_not_extrapolate_through_three_missing_frames():
    state=MotionState(30);r=np.eye(3)
    for i in range(10):
        state.predict(r,np.array([0,0,3.]),False)
        state.commit(r,[.01*i,0,3.],.05,.8,True,True)
    previous=np.array(state.positions[-1])
    for i in range(3):
        pr,pt,mode=state.predict(r,np.array([0,0,12.]),True)
        assert mode=='hold'
        np.testing.assert_array_equal(pt,previous)
        state.commit(pr,pt,0.,0.,False,False)


def test_reliable_constant_velocity_is_used_briefly_then_stops():
    state=MotionState(30);r=np.eye(3)
    for i in range(10):
        state.predict(r,np.array([0,0,3.]),False)
        state.commit(r,[.01*i,0,3.],1.,.9,True,True)
    positions=[]
    for i in range(12):
        pr,pt,mode=state.predict(r,np.array([999,999,999.]),True)
        state.commit(pr,pt,0.,0.,False,False);positions.append(pt)
    np.testing.assert_allclose(np.array(positions)[:3,0],[.10,.11,.12])
    np.testing.assert_allclose(np.array(positions)[6:],np.tile(positions[5],(6,1)))
    assert mode=='hold'


def test_noisy_velocity_falls_back_to_hold():
    state=MotionState(30);r=np.eye(3)
    for x in [0,.01,0,.01,0,.01]:
        state.predict(r,np.array([0,0,3.]),False)
        state.commit(r,[x,0,3.],1.,.9,True,True)
    _,pt,mode=state.predict(r,np.array([0,0,12.]),True)
    np.testing.assert_allclose(pt,[.01,0,3.]);assert mode=='hold'


def test_far_perfect_iou_cannot_force_reacquisition_jump():
    r=np.tile(np.eye(3),(3,1,1));t=np.array([[0,0,3.],[.01,0,3.],[0,0,12.]])
    allowed=candidate_gate(r,t,r[0],t[0],r[0],t[0],30,True)
    index,unresolved=choose_local([[.99,0],[.6,0],[0,0]],allowed)
    assert index==1 and not unresolved and not allowed[2]


def test_unexplained_local_evidence_is_explicit_instead_of_relaxing_gate():
    index,unresolved=choose_local([[1.,0],[1.,0],[0.,0]],[True,True,False])
    assert index==0 and unresolved


def test_partial_depth_is_locked_but_actual_visible_pixels_can_move_xy():
    r=np.tile(np.eye(3),(3,1,1));t=np.array([[0,0,3.],[.02,0,3.],[0,0,3.01]])
    allowed=candidate_gate(r,t,r[0],t[0],r[0],t[0],30,True)
    index,_=choose_local([[.9,0],[.2,0],[.1,0]],allowed)
    np.testing.assert_array_equal(allowed,[True,True,False]);assert index==1


def test_reacquisition_needs_consecutive_consistent_observations():
    state=MotionState(30);r=np.eye(3);t=np.array([0,0,3.])
    state.predict(r,t,True);state.commit(r,t,0.,0.,False,False)
    for consistent in [True,False,True,True]:
        state.predict(r,t,False);state.commit(r,t,1.,.9,True,consistent)
        assert state.reacquiring
    state.predict(r,t,False);state.commit(r,t,1.,.9,True,True)
    assert not state.reacquiring


def test_actual_speed_bound_rejects_constant_high_speed_even_without_acceleration():
    t=np.array([[0,0,3.],[0,0,4.],[0,0,5.]])
    r=np.tile(np.eye(3),(3,1,1))
    assert not trajectory_feasible(r,t,r,t,np.array(['clear']*3),30)


def test_refinement_checks_seams_partial_depth_and_missing_predecessor():
    t=np.array([[0,0,3.],[.06,0,3.],[.06,0,3.],[.06,0,3.]])
    r=np.tile(np.eye(3),(4,1,1));states=np.array(['clear','partial','missing','partial'])
    assert trajectory_feasible(r,t,r,t,states,30)
    for index,offset in [(1,[.009,0,0]),(3,[0,0,.001]),(2,[.001,0,0])]:
        modified=t.copy();modified[index]+=offset
        assert not trajectory_feasible(r,modified,r,t,states,30)
    modified=r.copy();modified[3]=Rotation.from_euler('z',7,degrees=True).as_matrix()
    assert not trajectory_feasible(modified,t,r,t,states,30)


def test_first_pose_keeps_original_depth_and_rejects_remote_anchor():
    r=np.tile(np.eye(3),(2,1,1));t=np.array([[0,0,3.],[0,0,12.]])
    np.testing.assert_array_equal(candidate_gate(r,t,r[0],t[0],None,None,30,True),[True,False])
