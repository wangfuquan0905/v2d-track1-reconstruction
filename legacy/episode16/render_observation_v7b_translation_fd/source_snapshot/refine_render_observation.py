"""Pilot: joint MHR/object trajectory fitting using rendered RGB, penetration,
and acceleration. This is direct trajectory optimization, NOT a SyILEN rollout.
No reference geometry or GT masks enter this program.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_cari4d_native import runtime_env
source, weights, environment = runtime_env(ROOT)
os.environ.update(environment)
sys.path[:0] = environment['PYTHONPATH'].split(os.pathsep)
import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from v2d_syilen.upper_limb import MeshSDF, upper_limb_controls, window_envelope
from v2d_syilen.render_observation import ObservationRenderer, observation_terms, translation_observation_gradient
from v2d_syilen.orientation_search import cube_rotations, acceleration_path, so3_path
from v2d_syilen.rotation_temporal import angular_derivatives


def main():
    import cv2
    import h5py
    import trimesh
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', default='form_hoi_full_000')
    p.add_argument('--start', type=int, default=90)
    p.add_argument('--end', type=int, default=120)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--render-supersample', type=int, default=1)
    p.add_argument('--translation-image-gradient', choices=['autograd', 'finite-difference'], default='autograd')
    p.add_argument('--translation-fd-step', type=float, default=.002)
    p.add_argument('--linear-velocity-weight', type=float, default=0.)
    p.add_argument('--linear-acceleration-weight', type=float, default=0.)
    p.add_argument('--linear-velocity-scale', type=float, default=.5, help='m/s')
    p.add_argument('--linear-acceleration-scale', type=float, default=1., help='m/s^2')
    p.add_argument('--lobs-weight', type=float, default=1., help='Multiplier for all image observation terms')
    p.add_argument('--penetration-weight', type=float, default=10., help='Weight of normalized penetration loss, used in both orientation search and joint fitting')
    p.add_argument('--loss-preset', choices=['full', 'object-observation-acc'], default='full', help='Object-only observation preset keeps object color, silhouette and existing trajectory acceleration')
    p.add_argument('--orientation-search', action='store_true', help='Search 24 proper rotations per frame with second-order temporal selection before joint fitting')
    p.add_argument('--rotation-temporal', choices=['matrix', 'so3'], default='matrix')
    p.add_argument('--angular-velocity-weight', type=float, default=1.)
    p.add_argument('--angular-acceleration-weight', type=float, default=1.)
    p.add_argument('--angular-velocity-scale-deg', type=float, default=90.)
    p.add_argument('--angular-acceleration-scale-deg', type=float, default=360.)
    p.add_argument('--include-first-orientations', action='store_true', help='Also offer the 24 first-frame rotations at every frame')
    p.add_argument('--baseline', type=Path, help='Explicit native/canonical baseline NPZ')
    p.add_argument('--resume-optimized', type=Path, help='Warm-start parameters from a compatible saved optimized NPZ (Adam state starts fresh)')
    p.add_argument('--bundle', type=Path, help='Corresponding bundle for human mesh topology')
    p.add_argument('--sdf', type=Path, help='Existing SDF cache of this exact object mesh')
    p.add_argument('--human-controls', choices=['upper', 'full'], default='upper')
    p.add_argument('--out', type=Path, default=ROOT/'outputs/render_observation_pilot_v1')
    a = p.parse_args()
    if not np.isfinite(a.lobs_weight) or a.lobs_weight <= 0:
        p.error('--lobs-weight must be finite and positive')
    if not np.isfinite(a.penetration_weight) or a.penetration_weight < 0:
        p.error('--penetration-weight must be finite and nonnegative')
    for value in [a.angular_velocity_weight, a.angular_acceleration_weight]:
        if not np.isfinite(value) or value < 0:
            p.error('Angular weights must be finite and nonnegative')
    for value in [a.linear_velocity_weight, a.linear_acceleration_weight]:
        if not np.isfinite(value) or value < 0:
            p.error('Linear weights must be finite and nonnegative')
    for value in [a.translation_fd_step, a.linear_velocity_scale, a.linear_acceleration_scale]:
        if not np.isfinite(value) or value <= 0:
            p.error('Finite-difference and linear normalization scales must be positive')
    for value in [a.angular_velocity_scale_deg, a.angular_acceleration_scale_deg]:
        if not np.isfinite(value) or value <= 0:
            p.error('Angular normalization scales must be finite and positive')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    root = ROOT/'outputs'/a.episode
    run = root/'cari4d'/a.episode
    baseline_path = a.baseline or root/'full_variants_v2/cari4d_full.npz'
    raw = np.load(baseline_path)
    count = len(raw['frames'])
    if not 0 <= a.start < a.end < count:
        raise ValueError('Requested interval is outside the input sequence')
    lo, hi = max(0, a.start-2), min(count, a.end+3)
    frames = raw['frames'][lo:hi]
    assert np.array_equal(frames, np.arange(lo, hi))
    n = len(frames)
    if a.orientation_search and (a.start != 0 or a.end != count-1):
        p.error('--orientation-search currently requires the complete sequence')
    device = 'cuda'
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    model = torch.jit.load(str(ROOT/'data/models/mhr_model.pt'), map_location=device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    arm, names, limits = upper_limb_controls(list(model.get_parameter_names()), model.get_parameter_limits())
    parameter_names = list(model.get_parameter_names())
    parameter_limits = model.get_parameter_limits().to(device)
    body_indices = ([i for i in range(6, 136) if float(parameter_limits[i, 1]-parameter_limits[i, 0])>1e-6]
                    if a.human_controls == 'full' else [i+6 for i in arm])
    indices = list(range(6)) + body_indices
    selector = torch.eye(136, device=device)[indices]
    pose, scales, shape = t(raw['pose'][lo:hi]), t(raw['scales']), t(raw['shape'])
    rotation_key = 'rotation' if 'rotation' in raw else 'object_rotation'
    translation_key = 'translation' if 'translation' in raw else 'object_translation'
    base_r, base_t = t(raw[rotation_key][lo:hi]), t(raw[translation_key][lo:hi])
    bundle_path = a.bundle or root/'full_refinement_v2/canonical_baseline/canonical.pth'
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    human_faces = torch.as_tensor(np.asarray(bundle['faces']), device=device, dtype=torch.int32)
    del bundle
    mesh_path = run/f'export/{a.episode}/object_mesh/output_aligned.glb'
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if mesh.visual.kind != 'vertex':
        raise ValueError('Pilot requires fixed per-vertex RGB; no silent material replacement')
    vertices, colors = t(np.asarray(mesh.vertices)), t(np.asarray(mesh.visual.vertex_colors)[:, :3]/255.)
    object_faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.int32)
    # Use an existing signed-distance cache of exactly this same mesh.
    import hashlib
    mesh_hash = hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()
    sdf_path = a.sdf or root/'full_refinement_v2/ik_001/object_sdf.npz'
    cached = np.load(sdf_path)
    assert str(cached['mesh_hash']) == mesh_hash
    sdf = MeshSDF(*(t(cached[k]) for k in ['grid', 'lower', 'upper']))
    with open(run/'depth/moge2/intrinsics.pkl', 'rb') as f:
        intrinsics = pickle.load(f)
    # Find a single clip-wide crop without retaining full-resolution masks.
    union = None
    with h5py.File(root/'segmentation/masks.h5', 'r') as f:
        fps = float(json.loads(f.attrs['provenance'])['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid source-video frame rate')
        group = f[a.episode]
        for frame in frames:
            visible = (group[f'{frame:06d}-k0.person_mask.png'][()].astype(bool) |
                       group[f'{frame:06d}-k0.obj_rend_mask.png'][()].astype(bool))
            union = visible if union is None else union | visible
    # One fixed crop from input segmentation across the whole clip.
    yy, xx = np.where(union)
    x0, y0 = max(0, int(xx.min())-50), max(0, int(yy.min())-50)
    x1, y1 = min(union.shape[1], int(xx.max())+51), min(union.shape[0], int(yy.max())+51)
    height = 384
    width = max(64, round((x1-x0)/(y1-y0)*height/8)*8)
    images, masks = [], []
    with h5py.File(root/'segmentation/masks.h5', 'r') as f:
        group = f[a.episode]
        for frame in frames:
            im = cv2.cvtColor(cv2.imread(str(root/f'segmentation/frames/{frame:06d}.jpg')), cv2.COLOR_BGR2RGB)
            m = np.stack([group[f'{frame:06d}-k0.person_mask.png'][()], group[f'{frame:06d}-k0.obj_rend_mask.png'][()]], -1).astype('float32')
            images.append(cv2.resize(im[y0:y1, x0:x1], (width, height)))
            masks.append(cv2.resize(m[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_NEAREST))
    images, masks = np.stack(images), np.stack(masks)
    rgb, observed = t(images/255.), t(masks)
    K = t([intrinsics['fx']*width/(x1-x0), intrinsics['fy']*height/(y1-y0),
           (intrinsics['cx']-x0)*width/(x1-x0), (intrinsics['cy']-y0)*height/(y1-y0)])
    renderer = ObservationRenderer(human_faces, object_faces, colors, K, height, width, supersample=a.render_supersample)
    envelope = t(window_envelope(frames, a.start, a.end))
    delta = torch.nn.Parameter(torch.zeros(n, len(indices), device=device))
    offset = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    rotvec = torch.nn.Parameter(torch.zeros(n, 3, device=device))
    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    # This native MHR model maps one root-translation parameter unit to
    # 10 cm of decoded geometry. Delta translation is in metres, so use 10,
    # not the 100 factor used to convert decoded mesh coordinates to metres.
    pose_units = t([10, 10, 10]+[1.]*(len(indices)-3))
    flip = t([1, -1, -1])
    controls = [delta, offset, rotvec, log_scale]
    optimizer = torch.optim.Adam([{'params': [delta], 'lr': .003},
                                  {'params': [offset], 'lr': .002},
                                  {'params': [rotvec], 'lr': .008},
                                  {'params': [log_scale], 'lr': .002}])
    def decode(ids):
        native = pose[ids] + ((delta[ids]*pose_units)*envelope[ids, None]) @ selector
        v, j = model(shape.expand(len(ids), -1).contiguous(), torch.cat([native, scales.expand(len(ids), -1)], -1), torch.zeros(len(ids), 72, device=device), True)
        r = axis_angle_to_matrix(rotvec[ids]*envelope[ids, None]) @ base_r[ids]
        trans = base_t[ids] + offset[ids]*envelope[ids, None]
        scale = log_scale.exp()
        return v*flip/100, j[..., :3]*flip/100, r, trans, scale, native
    joint_ids = [i for i, name in enumerate(model.get_joint_names()) if i and not any(x in name for x in ['_proc', 'twist'])]
    config = {'episode': a.episode, 'active_frames': [a.start, a.end], 'context_frames': [lo, hi-1],
        'steps': a.steps, 'render_size': [height, width], 'crop_xyxy': [x0, y0, x1, y1],
        'optimized': ['human root 6 DOF', f'{len(body_indices)} native pose DOF ({a.human_controls})', 'per-frame object SE(3)', 'shared object scale'],
        'human_parameter_names': [parameter_names[i] for i in indices],
        'root_translation_native_units_per_metre': 10,
        'fixed': ['human identity shape and skeletal scales', 'unselected/fixed-limit pose parameters', 'object geometry and colors', 'camera intrinsics'],
        'loss_weights': {'human_silhouette': 1, 'object_silhouette': 1, 'human_area': .25, 'object_area': .25, 'object_color': 2, 'penetration': 10, 'acceleration': .1},
        'normalization': {'penetration_m': .01, 'human_acc_m_per_frame2': .01, 'object_acc_m_per_frame2': .005, 'rotation_matrix_acc': .05},
        'reference_inputs': False, 'dynamics_rollout': False, 'baseline_path': str(baseline_path),
        'sdf_cache': str(sdf_path), 'sdf_accuracy_policy': 'reuse existing approximation; first-pass diagnostics, no accuracy gate',
        'notes': ['acceleration is second difference of actual trajectories, not error against GT', 'rotation regularity uses second difference of rotation matrices', 'no appearance/color loss for human', 'all human vertices queried for penetration; object-interior SDF only', 'colors have fixed unlit vertex appearance; illumination mismatch remains', 'direct trajectory trial; no SyILEN gradient or dynamics constraint']}
    config['lobs_weight'] = a.lobs_weight
    config['loss_weights']['penetration'] = a.penetration_weight
    config['object_color_formulation'] = 'full foreground RGB L1 against fixed observed mask, including missing pixels and spill'
    config['orientation_search'] = bool(a.orientation_search)
    for key in ['human_silhouette', 'object_silhouette', 'human_area', 'object_area', 'object_color']:
        config['loss_weights'][key] *= a.lobs_weight
    config['loss_preset'] = a.loss_preset
    if a.loss_preset == 'object-observation-acc':
        for key in ['human_silhouette', 'human_area', 'object_area', 'penetration']:
            config['loss_weights'][key] = 0.
        config['notes'].append('Only object color, object silhouette, and human/object trajectory acceleration drive optimization. Other terms are logged as diagnostics. Human motion remains variable and can affect object visibility.')
    config['rotation_temporal'] = a.rotation_temporal
    config['render_supersample'] = a.render_supersample
    config['translation_image_gradient'] = a.translation_image_gradient
    config['translation_fd_step_m'] = a.translation_fd_step
    if a.linear_velocity_weight > 0:
        config['loss_weights']['object_linear_velocity'] = a.linear_velocity_weight
    if a.linear_acceleration_weight > 0:
        config['loss_weights']['object_linear_acceleration'] = a.linear_acceleration_weight
    config['normalization'].update(object_velocity_m_per_s=a.linear_velocity_scale, object_acceleration_m_per_s2=a.linear_acceleration_scale)
    config['fps'] = fps
    config['include_first_orientations'] = a.include_first_orientations
    velocity_scale = np.deg2rad(a.angular_velocity_scale_deg)
    acceleration_scale = np.deg2rad(a.angular_acceleration_scale_deg)
    if a.rotation_temporal == 'so3':
        config['loss_weights'].update(angular_velocity=a.angular_velocity_weight, angular_acceleration=a.angular_acceleration_weight)
        config['normalization'].update(angular_velocity_rad_per_s=velocity_scale, angular_acceleration_rad_per_s2=acceleration_scale,
                                       human_linear_acc_m_per_s2=.01*fps**2, object_linear_acc_m_per_s2=.005*fps**2)
        config['notes'] = [note for note in config['notes'] if not note.startswith(('rotation regularity', 'Only object color'))]
        config['notes'].append('acceleration key contains only human/object linear acceleration; separate angular terms use Log(R_next R_prev^T)/dt in camera coordinates, then d(omega)/dt. Angular norms sum all three components.')
    (a.out/'config.json').write_text(json.dumps(config, indent=2))
    history = []
    best = None
    def pass_loss(backward=False, save_render=False, image_only=False):
        totals = {k: 0. for k in [*config['loss_weights'], 'total']}
        renders = []
        for begin in range(0, n, a.batch):
            end = min(n, begin+a.batch)
            left, right = max(0, begin-1), min(n, end+1)
            ids = torch.arange(left, right, device=device)
            v, joints, r, trans, scale, _ = decode(ids)
            sl = slice(begin-left, end-left)
            image_trans = trans[sl].detach() if a.translation_image_gradient == 'finite-difference' else trans[sl]
            obj = torch.einsum('bij,vj->bvi', r[sl], vertices*scale) + image_trans[:, None]
            rendered = renderer(v[sl], obj)
            terms = observation_terms(rendered, rgb[begin:end], observed[begin:end])
            terms = {k: val.sum()/n for k, val in terms.items()}
            local = torch.einsum('bvi,bij->bvj', v[sl]-trans[sl, None], r[sl])/scale
            penetration = F.relu(-sdf(local)*scale)
            terms['penetration'] = (penetration/.01).square().mean(1).sum()/n
            center_start, center_end = max(begin, 1), min(end, n-1)
            centers = torch.arange(center_start, center_end, device=device)-left
            def accel(x, norm):
                return ((x[centers+1]-2*x[centers]+x[centers-1])/norm).square().flatten(1).mean(1).sum()/(n-2)
            terms['acceleration'] = accel(joints[:, joint_ids], .01)
            edges = torch.arange(begin, min(end, n-1), device=device)-left
            if a.linear_acceleration_weight > 0:
                linear_alpha = (trans[centers+1]-2*trans[centers]+trans[centers-1])*fps**2
                terms['object_linear_acceleration'] = (linear_alpha/a.linear_acceleration_scale).square().sum()/(n-2)
            else:
                terms['acceleration'] += accel(trans, .005)
            if a.linear_velocity_weight > 0:
                linear_speed = (trans[edges+1]-trans[edges])*fps
                terms['object_linear_velocity'] = (linear_speed/a.linear_velocity_scale).square().sum()/(n-1)
            if a.rotation_temporal == 'so3':
                omega, alpha = angular_derivatives(r, fps)
                edges = torch.arange(begin, min(end, n-1), device=device)-left
                terms['angular_velocity'] = (omega[edges]/velocity_scale).square().sum(-1).sum()/(n-1)
                terms['angular_acceleration'] = (alpha[centers-1]/acceleration_scale).square().sum(-1).sum()/(n-2)
            else:
                terms['acceleration'] += accel(r, .05)
            loss = sum(terms[k]*config['loss_weights'][k] for k in terms)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite rendering loss')
            if backward:
                if image_only:
                    sum(terms[k]*config['loss_weights'][k] for k in ['human_silhouette', 'object_silhouette', 'human_area', 'object_area', 'object_color']).backward()
                else:
                    loss.backward()
                if a.translation_image_gradient == 'finite-difference':
                    numerical = translation_observation_gradient(renderer, v[sl], obj, rgb[begin:end], observed[begin:end], config['loss_weights'], a.translation_fd_step)
                    if offset.grad is None:
                        offset.grad = torch.zeros_like(offset)
                    offset.grad[begin:end].add_(numerical*envelope[begin:end, None]/n)
            for k, val in terms.items():
                totals[k] += float(val.detach())
            totals['total'] += float(loss.detach())
            if save_render:
                renders.append(rendered.detach().cpu().numpy())
        return totals, np.concatenate(renders) if renders else None
    with torch.no_grad():
        # Verify physical update units independently of baseline decode parity.
        probe_pose = pose[:1].expand(4, -1).clone()
        probe_pose[1:, :3] += torch.eye(3, device=device)*pose_units[:3]*.001
        probe_v, _ = model(shape.expand(4, -1).contiguous(),
                           torch.cat([probe_pose, scales.expand(4, -1)], -1),
                           torch.zeros(4, 72, device=device), True)
        measured = (probe_v[1:]-probe_v[:1])*flip/100
        expected = (torch.eye(3, device=device)*flip*.001)[:, None].expand_as(measured)
        torch.testing.assert_close(measured, expected, atol=2e-6, rtol=0)
        expected_vertices = raw['vertices'][lo:hi]
        for begin in range(0, n, a.batch):
            end = min(n, begin+a.batch)
            decoded = decode(torch.arange(begin, end, device=device))[0].cpu().numpy()
            np.testing.assert_allclose(decoded, expected_vertices[begin:end], atol=5e-5, rtol=0)
        initial, before_render = pass_loss(save_render=True)
    print('INITIAL', json.dumps(initial), flush=True)
    search_initial = initial
    if a.orientation_search:
        # Rotate around the fixed mesh bounds centre, preserving its camera
        # position. Mesh geometry/texture and human parameters stay unchanged.
        with torch.no_grad():
            candidates = t(cube_rotations())
            candidate_r = base_r[:, None] @ candidates[None]
            if a.include_first_orientations:
                candidate_r = torch.cat([candidate_r, candidate_r[:1].expand(n, -1, -1, -1)], dim=1)
            candidate_count = candidate_r.shape[1]
            center = (vertices.amin(0)+vertices.amax(0))/2
            candidate_t = (base_t[:, None] + (base_r @ center)[:, None]
                           - candidate_r @ center)
            flat_r, flat_t = candidate_r.reshape(-1, 3, 3), candidate_t.reshape(-1, 3)
            unary = np.zeros((n, candidate_count), dtype=np.float64)
            for begin in range(0, len(flat_r), a.batch):
                end = min(len(flat_r), begin+a.batch)
                frame_ids = np.arange(begin, end)//candidate_count
                hv = t(expected_vertices[frame_ids])
                obj = torch.einsum('bij,vj->bvi', flat_r[begin:end], vertices)+flat_t[begin:end, None]
                rendered = renderer(hv, obj)
                terms = observation_terms(rendered, rgb[frame_ids], observed[frame_ids])
                local = torch.einsum('bvi,bij->bvj', hv-flat_t[begin:end, None], flat_r[begin:end])
                terms['penetration'] = (F.relu(-sdf(local))/.01).square().mean(1)
                loss = sum(value*config['loss_weights'][key] for key, value in terms.items())
                if not torch.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite orientation candidate loss')
                unary.ravel()[begin:end] = loss.cpu().numpy()
                if begin % (candidate_count*32) == 0:
                    print('SEARCH', json.dumps({'frames_scored': begin//candidate_count, 'total': n}), flush=True)
            if a.rotation_temporal == 'so3':
                path, search_cost = so3_path(unary, candidate_r.cpu().numpy(), candidate_t.cpu().numpy(), fps,
                    a.angular_velocity_weight, a.angular_acceleration_weight, velocity_scale, acceleration_scale, config['loss_weights']['acceleration'],
                    a.linear_velocity_weight, a.linear_acceleration_weight, a.linear_velocity_scale, a.linear_acceleration_scale)
            else:
                path, search_cost = acceleration_path(unary, candidate_r.cpu().numpy(), candidate_t.cpu().numpy(), config['loss_weights']['acceleration'])
            base_r = candidate_r[torch.arange(n, device=device), torch.as_tensor(path, device=device)]
            base_t = candidate_t[torch.arange(n, device=device), torch.as_tensor(path, device=device)]
            search_initial, _ = pass_loss()
            # Search objective omits only the unchanged human acceleration.
            if search_initial['total'] > initial['total']+1e-3:
                raise RuntimeError('Orientation search worsened the exact objective')
            np.savez_compressed(a.out/'orientation_search.npz', candidates=candidates.cpu().numpy(),
                                unary=unary, selected=path, rotation=base_r.cpu().numpy(), translation=base_t.cpu().numpy())
            (a.out/'orientation_search.json').write_text(json.dumps({'candidates': candidate_count, 'includes_first_frame_candidates': a.include_first_orientations,
                'nonidentity_frames': int((path != 0).sum()), 'selection': 'second-order DP using the same image, penetration and acceleration objective',
                'objective_without_constant_human_acceleration': search_cost, 'after_search': search_initial}, indent=2))
            print('SEARCH_DONE', json.dumps(search_initial), flush=True)
    if a.resume_optimized:
        if not torch.all(envelope == 1):
            raise ValueError('Warm-start currently requires a full active interval')
        resumed = np.load(a.resume_optimized)
        np.testing.assert_array_equal(resumed['frames'], frames)
        np.testing.assert_allclose(resumed['shape'], raw['shape'], atol=1e-6)
        np.testing.assert_allclose(resumed['scales'], raw['scales'], atol=1e-6)
        with torch.no_grad():
            delta.copy_((t(resumed['pose'])-pose)[:, indices]/pose_units)
            offset.copy_(t(resumed['translation'])-base_t)
            rotvec.copy_(matrix_to_axis_angle(t(resumed['rotation'])@base_r.transpose(-1,-2)))
            log_scale.copy_(t(np.log(float(resumed['object_scale']))))
            resume_vertices = resumed['vertices']
            for begin in range(0,n,a.batch):
                end=min(n,begin+a.batch)
                decoded=decode(torch.arange(begin,end,device=device))
                np.testing.assert_allclose(decoded[0].cpu().numpy(),resume_vertices[begin:end],atol=5e-5,rtol=0)
                np.testing.assert_allclose(decoded[2].cpu().numpy(),resumed['rotation'][begin:end],atol=2e-5,rtol=0)
            search_initial,_=pass_loss()
        config['warm_start']={'path':str(a.resume_optimized),'optimizer_state':'fresh Adam','initial_loss':search_initial}
        (a.out/'config.json').write_text(json.dumps(config,indent=2))
        print('RESUMED',json.dumps(search_initial),flush=True)
    # Audit end-to-end image gradient, including through native MHR.
    optimizer.zero_grad()
    pass_loss(backward=True, image_only=True)
    grad_audit = {name: {'finite': bool(torch.isfinite(var.grad).all()), 'norm': float(var.grad.norm())} for name, var in zip(['human_pose', 'object_translation', 'object_rotation', 'object_scale'], controls)}
    if not all(x['finite'] and x['norm']>0 for x in grad_audit.values()):
        raise RuntimeError(f'Invalid differentiable rendering gradients: {grad_audit}')
    (a.out/'gradient_audit.json').write_text(json.dumps(grad_audit, indent=2))
    print('GRADIENT', json.dumps(grad_audit), flush=True)
    best = (search_initial['total'], [var.detach().clone() for var in controls], 0)
    for step in range(a.steps):
        optimizer.zero_grad()
        row, _ = pass_loss(backward=True)
        if row['total'] < best[0]:
            best = (row['total'], [var.detach().clone() for var in controls], step)
        torch.nn.utils.clip_grad_norm_(controls, 20)
        optimizer.step()
        with torch.no_grad():
            delta[:, :3].clamp_(-.15, .15)
            delta[:, 3:].clamp_(-.6, .6)
            # Preserve native joint limits without snapping preexisting violations.
            low = torch.minimum(parameter_limits[body_indices, 0][None], pose[:, body_indices])
            high = torch.maximum(parameter_limits[body_indices, 1][None], pose[:, body_indices])
            delta[:, 6:].copy_(torch.maximum(torch.minimum(delta[:, 6:], high-pose[:, body_indices]), low-pose[:, body_indices]))
            offset.clamp_(-.20, .20)
            rotvec.clamp_(-.8, .8)
            log_scale.clamp_(np.log(.85), np.log(1.15))
        row.update(step=step, elapsed_seconds=time.monotonic()-started)
        history.append(row)
        if step % 5 == 0:
            print('STEP', json.dumps(row), flush=True)
            (a.out/'history.json').write_text(json.dumps(history, indent=2))
    with torch.no_grad():
        final_row, _ = pass_loss()
        if final_row['total'] < best[0]:
            best = (final_row['total'], [var.detach().clone() for var in controls], a.steps)
        for var, value in zip(controls, best[1]):
            var.copy_(value)
        after, after_render = pass_loss(save_render=True)
        arrays = {key: [] for key in ['vertices', 'joints', 'rotation', 'translation', 'pose']}
        for begin in range(0, n, a.batch):
            v, j, r, trans, scale, native = decode(torch.arange(begin, min(n, begin+a.batch), device=device))
            for key, value in zip(arrays, [v, j, r, trans, native]):
                arrays[key].append(value.cpu().numpy())
        optimized_values = {key: np.concatenate(value) for key, value in arrays.items()}
        optimized_values['object_scale'] = float(scale)
        for name, values in [('baseline', {'vertices': raw['vertices'][lo:hi], 'joints': raw['joints'][lo:hi], 'rotation': raw[rotation_key][lo:hi], 'translation': raw[translation_key][lo:hi], 'pose': raw['pose'][lo:hi], 'object_scale': 1.}),
                             ('optimized', optimized_values)]:
            np.savez_compressed(a.out/f'{name}.npz', frames=frames, coordinate_frame=np.array('camera'), scales=raw['scales'], shape=raw['shape'], **values)
    import imageio.v2 as imageio
    def overlay(im, render):
        alpha = render[..., :2].sum(-1).clip(0, 1)
        color = render[..., 0:1]*np.array([.68, .78, .88]) + render[..., 2:]
        return np.uint8(np.clip(im*(1-.7*alpha[..., None])+.7*color*255, 0, 255))
    def strip(i):
        panels = [images[i], overlay(images[i], before_render[i]), overlay(images[i], after_render[i])]
        titles = ['RGB input', 'Full CARI4D', ('Object OBS + SO3' if a.rotation_temporal == 'so3' else 'Object OBS + ACC') if a.loss_preset == 'object-observation-acc' else 'Render + PEN + ACC']
        for index, title in enumerate(titles):
            panels[index] = cv2.copyMakeBorder(panels[index], 36, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(panels[index], f'{title} | {frames[i]}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate(panels, 1)
    with imageio.get_writer(str(a.out/'comparison.mp4'), fps=fps, codec='libx264', macro_block_size=None, ffmpeg_params=['-crf', '18']) as writer:
        for i in range(n):
            writer.append_data(strip(i))
    imageio.imwrite(a.out/'comparison.png', np.concatenate([strip(i) for i in [0, n//2, n-1]], 0))
    (a.out/'history.json').write_text(json.dumps(history, indent=2))
    report = {'status': 'complete', 'initial': initial, 'after_orientation_search': search_initial, 'optimized': after, 'best_step': best[2],
              'elapsed_seconds': time.monotonic()-started, 'object_scale': float(scale), 'ground_truth_used': False,
              'frames': n, 'scope': 'Direct-trajectory rendering experiment; no dynamics rollout, no claim of general improvement'}
    (a.out/'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
