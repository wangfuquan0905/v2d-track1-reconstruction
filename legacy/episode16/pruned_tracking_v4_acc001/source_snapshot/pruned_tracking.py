"""Derivative-free rigid-pose proposals and acceleration-only tracking costs."""
import numpy as np
from scipy.spatial.transform import Rotation


def predict_pose(rotations, translations):
    # Reproject saved float32 matrices before extrapolation. Repeated products
    # otherwise amplify tiny non-orthogonality into scale/shear exponentially.
    r, t = Rotation.from_matrix(rotations[-1]).as_matrix(), translations[-1]
    if len(rotations) < 2:
        return r.copy(), t.copy()
    previous = Rotation.from_matrix(rotations[-2]).as_matrix()
    return Rotation.from_matrix((r @ previous.T) @ r).as_matrix(), 2*t-translations[-2]


def rotation_candidates(center, previous, fine=False, expansion=False):
    # Camera-space LEFT updates; no absolute-angle clamp, no assumed upright axis.
    small, large = ((.10, .5) if fine else (.5, 2.))
    if expansion:
        small, large = 10., 45.
    vectors = [np.zeros(3)]
    for size in [small, large]:
        for axis in np.eye(3):
            vectors.extend([axis*np.deg2rad(size), -axis*np.deg2rad(size)])
    for i in range(18):
        z = 1-2*(i+.5)/18
        theta = np.pi*(3-np.sqrt(5))*i
        vectors.append(np.array([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])*np.deg2rad(large))
    r = Rotation.from_rotvec(np.array(vectors)).as_matrix() @ center
    return np.concatenate([r, previous[None]]), np.array([0]+[small]*6+[large]*24+[0])


def acceleration_cost(rotations, translations, candidate_r, candidate_t, fps=30., angular_scale=np.deg2rad(360), linear_scale=1., angular_weight=1., linear_weight=1.):
    if len(rotations) < 2:
        return np.zeros(len(candidate_r))
    previous_w = Rotation.from_matrix(rotations[-1]@rotations[-2].T).as_rotvec()*fps
    w = Rotation.from_matrix(candidate_r@rotations[-1].T).as_rotvec()*fps
    angular = np.sum(((w-previous_w)*fps/angular_scale)**2,axis=-1)
    linear = np.sum(((candidate_t-2*translations[-1]+translations[-2])*fps**2/linear_scale)**2,axis=-1)
    return angular_weight*angular+linear_weight*linear


def geometry_keep(rotations, translations, corners, K, observed_bbox, image_size):
    points = np.einsum('bij,vj->bvi',rotations,corners)+translations[:,None]
    positive = np.all(points[:,:,2]>.01,axis=1)
    z = points[:,:,2].clip(.01)
    uv = points[:,:,:2]/z[:,:,None]*K[:2]+K[2:]
    low, high = uv.min(1), uv.max(1)
    obslo, obshi = np.array(observed_bbox[:2]),np.array(observed_bbox[2:])
    observed_size = np.maximum(obshi-obslo,2.)
    # Loose screen overlap/size only: visible mask can be a small occluded fragment.
    overlap = np.all(high >= obslo-observed_size*.5,axis=1)&np.all(low <= obshi+observed_size*.5,axis=1)
    size = np.maximum(high-low,1.)
    plausible = np.all(size>=observed_size*.15,axis=1)&np.all(size<=np.array(image_size[::-1])*3,axis=1)
    return positive&overlap&plausible
