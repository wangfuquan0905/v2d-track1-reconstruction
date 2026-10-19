import numpy as np
from scipy.spatial.transform import Rotation
from v2d_syilen.pruned_tracking import predict_pose,rotation_candidates,acceleration_cost,geometry_keep


def test_prediction_crosses_180_and_reaches_270_without_speed_penalty():
    angles=np.arange(0,271,3.)
    r=Rotation.from_rotvec(np.deg2rad(angles)[:,None]*np.array([0,1,0])).as_matrix()
    t=np.arange(len(r))[:,None]*np.array([.001,0,0])+np.array([0,0,2])
    for i in range(2,len(r)):
        pr,pt=predict_pose(r[:i],t[:i])
        np.testing.assert_allclose(pr,r[i],atol=1e-12)
        np.testing.assert_allclose(pt,t[i],atol=1e-12)
        np.testing.assert_allclose(acceleration_cost(r[:i],t[:i],pr[None],pt[None]),0,atol=1e-20)


def test_candidates_are_proper_rotations_and_include_predictor_and_previous():
    center=Rotation.from_rotvec([.5,-.7,2.]).as_matrix();previous=np.eye(3)
    for fine,expanded in [(False,False),(True,False),(True,True)]:
        r,radius=rotation_candidates(center,previous,fine,expanded)
        assert r.shape==(32,3,3) and radius.shape==(32,)
        np.testing.assert_allclose(r@r.transpose(0,2,1),np.broadcast_to(np.eye(3),r.shape),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-12)
        np.testing.assert_allclose(r[0],center);np.testing.assert_allclose(r[-1],previous)


def test_geometry_filter_keeps_partial_occlusion_and_rejects_offscreen():
    corners=np.array(np.meshgrid(*[[-.1,.1]]*3)).reshape(3,-1).T
    r=np.repeat(np.eye(3)[None],3,0);t=np.array([[0,0,2],[20,0,2],[0,0,-2]])
    keep=geometry_keep(r,t,corners,np.array([100,100,50,50]),[50,48,53,52],(100,100))
    np.testing.assert_array_equal(keep,[True,False,False])


def test_long_recursive_prediction_stays_rigid_with_float32_input():
    r=Rotation.from_rotvec([.5,-.3,2.]).as_matrix().astype('float32').astype(float)
    rotations=[r];translations=[np.array([0.,0.,2.])]
    for i in range(360):
        pr,pt=predict_pose(rotations,translations)
        np.testing.assert_allclose(pr@pr.T,np.eye(3),atol=1e-12)
        rotations.append(pr);translations.append(pt)
