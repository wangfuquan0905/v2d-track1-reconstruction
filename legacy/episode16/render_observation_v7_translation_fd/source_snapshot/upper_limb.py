"""Differentiable wrist targets and upper-limb IK helpers (camera-space metres).

No detached hand meshes: IK always changes native MHR joint parameters, then
regenerates the whole mesh. Palm frames are anatomical frames from named joints,
not the unconverted native MHR joint rotations. This module does not import JAX.
"""
import torch
import torch.nn.functional as F


def upper_limb_controls(parameter_names, parameter_limits):
    """Map authoritative native model names to the body block (root occupies 6).

Clavicle, upper arm, elbow, forearm twist and wrist only; exclude fixed DOFs.
"""
    prefixes = tuple(f'{side}_{part}' for side in ('l','r') for part in
                     ('clavicle_', 'uparm_', 'elbow_', 'lowarm_', 'wrist_'))
    selected = [i for i, name in enumerate(parameter_names)
                if name.startswith(prefixes) and 6 <= i < 136
                and float(parameter_limits[i,1]-parameter_limits[i,0]) > 1e-6]
    if len(selected) != 18:
        raise ValueError(f'Expected 18 free arm controls for this native MHR model, found {len(selected)}')
    return [i-6 for i in selected], [parameter_names[i] for i in selected], parameter_limits[selected]


def palm_frames(joints, names):
    """Columns X=across palm, Y=towards middle finger, Z=palm normal; left/right."""
    lookup = {str(n):i for i,n in enumerate(names)}
    positions, frames = [], []
    for side in ('l','r'):
        wrist = joints[:, lookup[f'{side}_wrist']]
        y = F.normalize(joints[:, lookup[f'{side}_middle1']]-wrist,dim=-1)
        x = joints[:,lookup[f'{side}_index1']]-joints[:,lookup[f'{side}_pinky1']]
        x = F.normalize(x-(x*y).sum(-1,keepdim=True)*y,dim=-1)
        z = F.normalize(torch.cross(x,y,dim=-1),dim=-1)
        y = torch.cross(z,x,dim=-1)
        positions.append(wrist);frames.append(torch.stack((x,y,z),dim=-1))
    return torch.stack(positions,1), torch.stack(frames,1)


def project(points, intrinsics):
    z = points[...,2:].clamp_min(.05)
    return points[...,:2]/z*intrinsics[:2]+intrinsics[2:]


def second_difference(x):
    return x[2:]-2*x[1:-1]+x[:-2]


def box_sdf(points, half_size):
    delta = points.abs()-half_size
    return torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)


class MeshSDF:
    """Trilinear lookup of a precomputed signed distance grid; inside is negative."""
    def __init__(self, grid, lower, upper):
        self.grid = grid[None,None]  # D=Z, H=Y, W=X
        self.lower, self.upper = lower, upper

    def __call__(self, points):
        normalized = 2*(points-self.lower)/(self.upper-self.lower)-1
        clamped = normalized.clamp(-1,1)
        sampled = F.grid_sample(self.grid,clamped.reshape(1,1,1,-1,3),
                                align_corners=True,padding_mode='border',mode='bilinear').reshape(points.shape[:-1])
        outside = torch.linalg.vector_norm((normalized-clamped)*(self.upper-self.lower)/2,dim=-1)
        return sampled+outside


def capsule_samples(joints, edges, samples=7):
    start, end = joints[:,edges[:,0]], joints[:,edges[:,1]]
    t = torch.linspace(0,1,samples,device=joints.device,dtype=joints.dtype)
    return start[:,:,None]*(1-t[None,None,:,None])+end[:,:,None]*t[None,None,:,None]


def window_envelope(frames, active_start, active_end):
    """Cosine fades to zero at context endpoints; active interval has weight one."""
    import numpy as np
    frames = np.asarray(frames)
    weight = np.ones(len(frames))
    left, right = frames < active_start, frames > active_end
    if left.any():weight[left]=.5-.5*np.cos(np.pi*(frames[left]-frames[0])/(active_start-frames[0]))
    if right.any():weight[right]=.5-.5*np.cos(np.pi*(frames[-1]-frames[right])/(frames[-1]-active_end))
    return weight
