"""Shared whole-window acceleration-only objective; no velocity magnitude term."""
import numpy as np
from scipy.spatial.transform import Rotation


def temporal_terms(rotations, translations, fps):
    n=len(rotations)
    omega=Rotation.from_matrix(rotations[1:]@rotations[:-1].transpose(0,2,1)).as_rotvec()*fps
    angular=np.diff(omega,axis=0)*fps
    linear=np.diff(translations,n=2,axis=0)*fps**2
    return {'angular_acceleration':float(np.sum((angular/(2*np.pi))**2)/n),
            'linear_acceleration':float(np.sum(linear**2)/n)}


def whole_window_score(observations, rotations, translations, fps, angular_weight=.01,linear_weight=.01):
    temporal=temporal_terms(rotations,translations,fps)
    terms={'object_silhouette':float(observations[:,0].mean()),'object_color':float(observations[:,1].mean()),**temporal}
    terms['total']=10*terms['object_silhouette']+20*terms['object_color']+angular_weight*temporal['angular_acceleration']+linear_weight*temporal['linear_acceleration']
    return terms


def preconditioners(n,fps,angular_weight,linear_weight):
    second=np.diff(np.eye(n),n=2,axis=0);curvature=second.T@second
    # Approximate temporal Hessians, plus numerical damping; these matrices
    # precondition updates and are NOT additional objective penalties.
    angular=np.eye(n)*(50/n)+2*angular_weight*fps**4/(n*(2*np.pi)**2)*curvature
    linear=np.eye(n)*(20000/n)+2*linear_weight*fps**4/n*curvature
    return angular,linear
