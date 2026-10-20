"""Censored silhouette observations and uncapped temporal regularization.

All visibility/uncertainty inputs are fixed before candidate optimization.
They cannot be improved by shrinking or hiding a candidate object.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import minimum_filter1d

ANGULAR_WEIGHT = .01
LINEAR_WEIGHT = .02
COLOR_WEIGHT = .02


def pseudo_huber_squared(squared_norm):
    """Smooth L1 of a vector norm; unbounded, with nonvanishing outlier force."""
    return np.sqrt(1 + squared_norm) - 1


def known_pixels(person, target, dilation=1):
    """Human-covered negatives are unknown; every observed object pixel is kept."""
    import cv2
    occluder = person.astype('uint8')
    if dilation:
        k = 2*dilation+1
        occluder = cv2.dilate(occluder, np.ones((k,k), 'uint8'))
    return np.maximum(1-occluder.astype('float32'), target.astype('float32'))


def visibility_states(amodal, scene_visible, person, target):
    area = amodal.sum((1,2)).clip(1)
    observed_occlusion = (amodal*person*(1-target)).sum((1,2))/area
    geometric_occlusion = np.maximum(amodal-scene_visible,0).sum((1,2))/area
    # Geometric disagreement remains diagnostic: do not let an inaccurate
    # predicted human erase visible object evidence. Confidence is conservative.
    confidence, states = visibility_from_fractions(observed_occlusion, geometric_occlusion, target.sum((1,2)))
    return confidence, states, observed_occlusion, geometric_occlusion


def visibility_from_fractions(observed_occlusion, geometric_occlusion, target_area):
    covered = np.maximum(observed_occlusion, geometric_occlusion)
    observed = target_area >= 1
    confidence = np.clip(1-covered, .05, 1)
    confidence[~observed] = 0
    confidence = minimum_filter1d(confidence, size=5, mode='nearest')
    confidence[observed] = np.maximum(confidence[observed], .05)
    confidence[~observed] = 0
    states = np.where(~observed, 'missing', np.where(confidence >= .9, 'clear', 'partial'))
    return confidence, states


def interpolate_anchors(rotations, translations, reliable):
    """Offline forward/backward initialization, never a reference measurement."""
    ids = np.flatnonzero(reliable)
    if len(ids) == 0:
        raise ValueError('No reliable visible anchor: cannot infer occluded pose')
    query = np.arange(len(rotations))
    t = np.stack([np.interp(query, ids, translations[ids,j]) for j in range(3)], -1)
    r = (np.repeat(rotations[ids[:1]], len(query), axis=0) if len(ids)==1 else
         Slerp(ids, Rotation.from_matrix(rotations[ids]))(np.clip(query,ids[0],ids[-1])).as_matrix())
    distance = np.min(abs(query[:,None]-ids[None]),axis=1)
    return r,t,distance


def prior_weights(confidence, distance, fps):
    # Longer missing intervals imply larger uncertainty, not an asserted pose.
    sigma = .15 + .10*distance/fps
    strength = .05*(1-confidence)/sigma**2
    return strength[:,None]*np.array([1.,1.,2.])[None]


def temporal_residuals(r,t,fps):
    omega=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()*fps
    angular=np.diff(omega,axis=0)*fps/(2*np.pi)
    linear=np.diff(t,n=2,axis=0)*fps**2
    return angular,linear


def score(ob,r,t,fps,confidence,reference_t,prior):
    # GPU observations are float32. Reduce in float64 so a local window delta
    # remains equal to its contribution to the whole-clip objective.
    ob=np.asarray(ob,dtype=np.float64)
    confidence=np.asarray(confidence,dtype=np.float64)
    n=len(r)
    angular,linear=temporal_residuals(r,t,fps)
    silhouette=float(np.sum(confidence*ob[:,0])/n)
    color=float(COLOR_WEIGHT*np.sum(confidence*ob[:,1])/n)
    a=float(ANGULAR_WEIGHT*pseudo_huber_squared(np.sum(angular**2,-1)).sum()/n)
    l=float(LINEAR_WEIGHT*pseudo_huber_squared(np.sum(linear**2,-1)).sum()/n)
    p=float(np.sum(prior*(t-reference_t)**2)/n)
    return {'silhouette':silhouette,'color':color,'angular':a,'linear':l,'occlusion_prior':p,
            'total':silhouette+color+a+l+p}


def second_order_path(unary, rotations, translations, fps):
    """Exact discrete joint trajectory selection under this same robust cost."""
    n,k=unary.shape
    if n<3:raise ValueError('Need at least three frames')
    costs=unary[0,:,None]+unary[1,None,:]
    back=np.zeros((n,k,k),dtype=np.int16)
    def velocity(i):
        relative=rotations[i,None]@rotations[i-1,:,None].swapaxes(-1,-2)
        return Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(k,k,3)*fps
    previous=velocity(1)
    for i in range(2,n):
        current=velocity(i)
        angular=(current[None]-previous[:,:,None])*fps/(2*np.pi)
        linear=(translations[i-2,:,None,None]-2*translations[i-1,None,:,None]+translations[i,None,None,:])*fps**2
        penalty=(ANGULAR_WEIGHT*pseudo_huber_squared(np.sum(angular**2,-1))+
                 LINEAR_WEIGHT*pseudo_huber_squared(np.sum(linear**2,-1)))
        candidates=costs[:,:,None]+penalty
        back[i]=candidates.argmin(0)
        costs=candidates.min(0)+unary[i,None,:]
        previous=current
    path=np.zeros(n,dtype=np.int64)
    path[-2],path[-1]=np.unravel_index(costs.argmin(),costs.shape)
    value=float(costs[path[-2],path[-1]]/n)
    for i in range(n-1,1,-1):path[i-2]=back[i,path[i-1],path[i]]
    return path,value


def censored_terms(rendered, rgb, target, known):
    """Torch terms on amodal rendering; observed positives cannot be censored."""
    import torch
    alpha=rendered[...,1]
    valid=torch.maximum(known,target)
    intersection=(alpha*target*valid).sum((1,2))
    union=((alpha+target-alpha*target)*valid).sum((1,2)).clamp_min(1)
    has_target=target.sum((1,2))>0
    silhouette=torch.where(has_target,1-intersection/union,torch.zeros_like(union))
    error=(rendered[...,2:]-rgb*target[...,None]).abs()*valid[...,None]
    color=error.sum((1,2,3))/(3*target.sum((1,2)).clamp_min(1))
    color=torch.where(has_target,color,torch.zeros_like(color))
    return torch.stack([silhouette,color],-1)
