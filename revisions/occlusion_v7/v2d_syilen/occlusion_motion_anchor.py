"""Local 3D consensus for clear-frame depth; never a distant endpoint anchor."""
from dataclasses import dataclass
import numpy as np
from v2d_syilen.occlusion_motion import MotionConfig as BaseMotionConfig


@dataclass(frozen=True)
class MotionConfig(BaseMotionConfig):
    anchor_consensus_m: float = .08
    anchor_distance_m: float = .20
    anchor_sigma_m: float = .03
    anchor_weight: float = .1


def local_depth_anchor(source_t, stable_t, prediction, clear, config=MotionConfig()):
    """Two agreeing current-frame hypotheses, close to the existing trajectory.

    Agreement alone is insufficient: shared bad reconstructions far from the
    current pose are rejected. Partial and missing images cannot enable this.
    """
    anchor=(np.asarray(source_t)+np.asarray(stable_t))/2
    supported=bool(clear and np.isfinite(anchor).all()
                   and np.linalg.norm(source_t-stable_t)<=config.anchor_consensus_m
                   and np.linalg.norm(anchor-prediction)<=config.anchor_distance_m)
    return float(anchor[2]) if supported else None


def choose_anchored(observations, allowed, translations, anchor_z=None,
                    minimum_fit_iou=.02, config=MotionConfig()):
    ob=np.asarray(observations)
    valid=np.asarray(allowed,dtype=bool)&np.isfinite(ob).all(-1)
    if not valid[0]:raise ValueError('Prediction must remain feasible')
    cost=ob[:,0]+.02*ob[:,1]
    best_iou=1-np.min(ob[valid,0])
    if best_iou<minimum_fit_iou:return 0,True
    if anchor_z is not None:
        cost=cost+config.anchor_weight*((np.asarray(translations)[:,2]-anchor_z)/config.anchor_sigma_m)**2
    # Preserve visible evidence among the motion-compatible candidates only.
    valid &= ob[:,0] <= min(1.,1-best_iou+.05)+1e-7
    return int(np.argmin(np.where(valid,cost,np.inf))),False
