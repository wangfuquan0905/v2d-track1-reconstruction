import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from v2d_syilen.occlusion_objective import (known_pixels,visibility_states,interpolate_anchors,
    prior_weights,score,second_order_path,pseudo_huber_squared,censored_terms)
from v2d_syilen.window_search import context_bounds


def test_occlusion_ignores_hidden_negatives_but_never_erases_positive_evidence():
    import torch
    a=np.zeros((16,16),dtype='float32');a[4:12,4:12]=1
    person=np.zeros_like(a);person[:,:9]=1
    target=a*(1-person)
    known=known_pixels(person,target,dilation=0)
    pred=torch.zeros(1,16,16,5);pred[0,...,1]=torch.from_numpy(a)
    rgb=torch.zeros(1,16,16,3)
    loss=censored_terms(pred,rgb,torch.from_numpy(target)[None],torch.from_numpy(known)[None])
    assert float(loss[0,0])==0
    # Even a wrong person mask overlapping the actual object cannot hide it.
    erased=censored_terms(torch.zeros_like(pred),rgb,torch.from_numpy(target)[None],torch.zeros(1,16,16))
    assert float(erased[0,0])==1
    missing=censored_terms(pred,rgb,torch.zeros(1,16,16),torch.ones(1,16,16))
    assert np.all(missing.numpy()==0)


def test_small_fully_visible_object_is_not_classified_as_occluded():
    a=np.zeros((7,32,32),dtype='float32');a[:,15:17,15:17]=1
    confidence,state,_,_=visibility_states(a,a,np.zeros_like(a),a)
    assert np.all(state=='clear') and np.all(confidence==1)


def test_missing_poses_interpolate_translation_and_rotation_from_visible_neighbors():
    r=Rotation.from_euler('z',[0,5,10,15,20],degrees=True).as_matrix()
    t=np.arange(5)[:,None]*np.array([.1,0,.02])+[0,0,3]
    corrupt_r=r.copy();corrupt_r[1:4]=np.eye(3)
    corrupt_t=t.copy();corrupt_t[1:4]=100
    rr,tt,d=interpolate_anchors(corrupt_r,corrupt_t,[True,False,False,False,True])
    np.testing.assert_allclose(rr,r,atol=1e-12);np.testing.assert_allclose(tt,t)
    np.testing.assert_array_equal(d,[0,1,2,1,0])


def test_large_jump_penalty_keeps_growing_instead_of_saturating():
    assert pseudo_huber_squared(10000**2)>9*pseudo_huber_squared(1000**2)


def test_discrete_joint_selection_matches_exhaustive_robust_objective():
    rng=np.random.default_rng(18);n,k=5,3
    r=Rotation.from_rotvec(rng.normal(size=(n*k,3))*.01).as_matrix().reshape(n,k,3,3)
    t=rng.normal(size=(n,k,3))*.002
    unary=rng.random((n,k))*.2
    path,value=second_order_path(unary,r,t,30)
    def evaluate(p):
        ids=np.arange(n)
        return score(np.stack([unary[ids,p],np.zeros(n)],-1),r[ids,p],t[ids,p],30,
                     np.ones(n),np.zeros((n,3)),np.zeros((n,3)))['total']
    brute=min(evaluate(p) for p in itertools.product(range(k),repeat=n))
    np.testing.assert_allclose(value,brute,atol=1e-12)
    np.testing.assert_allclose(evaluate(path),brute,atol=1e-12)


def test_window_delta_includes_cross_boundary_occlusion_and_temporal_terms():
    rng=np.random.default_rng(2);n=80
    r=Rotation.from_rotvec(rng.normal(size=(n,3))*.01).as_matrix();t=rng.normal(size=(n,3))*.01
    # Match the float32 observations/confidence returned by the real renderer.
    ref=t.copy();ob=rng.random((n,2)).astype('float32');confidence=rng.random(n).astype('float32')
    prior=prior_weights(confidence,np.zeros(n),30)
    nr=r.copy();nt=t.copy();no=ob.copy();start,stop=25,75;lo,hi=context_bounds(start,stop,n)
    nt[start:stop]+=.03;no[start:stop]*=.8
    def val(o,r,t,s):return len(r)*score(o,r,t,30,confidence[s],ref[s],prior[s])['total']
    np.testing.assert_allclose(val(no,nr,nt,slice(None))-val(ob,r,t,slice(None)),
       val(no[lo:hi],nr[lo:hi],nt[lo:hi],slice(lo,hi))-val(ob[lo:hi],r[lo:hi],t[lo:hi],slice(lo,hi)),atol=1e-10)
