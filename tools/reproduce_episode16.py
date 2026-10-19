"""Reconstruct the retained episode-16 warm-start chain in a separate workspace.

The source snapshots and scalar settings are historical; this driver is new.
No simulator is invoked. GPU reconstruction is not promised bitwise reproducible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline import read, write, digest, adapt


def make_diagnostic_sdf(work):
    """Rebuild the zero-weight historical diagnostic grid from the RGB mesh."""
    import numpy as np
    import torch
    import trimesh
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    root = work / 'outputs/episode_000016'
    mesh = trimesh.load(root / 'cari4d/episode_000016/export/episode_000016/object_mesh/output_aligned.glb', force='mesh', process=False)
    spec = read(REPO / 'provenance/episode16/sdf_spec.json')
    lower, upper = np.asarray(spec['lower']), np.asarray(spec['upper']); n = spec['resolution']
    z, y, x = np.meshgrid(np.linspace(lower[2], upper[2], n), np.linspace(lower[1], upper[1], n),
                          np.linspace(lower[0], upper[0], n), indexing='ij')
    points = torch.tensor(np.stack([x, y, z], -1).reshape(-1, 3), device='cuda', dtype=torch.float32)
    coordinates = points.cpu().numpy(); inside = np.zeros(len(points), bool)
    candidates = np.flatnonzero(np.all((coordinates >= mesh.bounds[0]) & (coordinates <= mesh.bounds[1]), -1))
    if mesh.is_watertight:
        for start in range(0, len(candidates), 4096):
            ids = candidates[start:start+4096]; inside[ids] = mesh.contains(coordinates[ids])
    else:
        tri = torch.as_tensor(mesh.vertices[mesh.faces], device='cuda', dtype=torch.float32)
        for start in range(0, len(candidates), 128):
            ids = candidates[start:start+128]; q = points[ids, None]
            a, b, c = (tri[None, :, i] - q for i in range(3))
            la, lb, lc = (torch.linalg.vector_norm(v, dim=-1) for v in (a, b, c))
            numerator = (a * torch.cross(b, c, dim=-1)).sum(-1)
            denominator = la*lb*lc + (a*b).sum(-1)*lc + (b*c).sum(-1)*la + (c*a).sum(-1)*lb
            inside[ids] = (torch.atan2(numerator, denominator).sum(-1).abs() > np.pi).cpu().numpy()
    triangles = torch.tensor(mesh.vertices[mesh.faces], device='cuda', dtype=torch.float32)[None].contiguous()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(points), 8192):
            d = point_to_mesh_distance(points[start:start+8192][None].contiguous(), triangles)[0].clamp_min(1e-12).sqrt()[0]
            chunks.append(torch.where(torch.as_tensor(inside[start:start+len(d)], device='cuda'), -d, d).cpu().numpy())
    path = root / 'syilen/upper_limb/object_sdf.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grid=np.concatenate(chunks).reshape(n, n, n), lower=lower.astype('f4'), upper=upper.astype('f4'),
                        mesh_hash=hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest(),
                        sign_revision='trimesh_contains_v1' if mesh.is_watertight else 'winding_number_v1')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workspace', type=Path, required=True); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(); work = a.workspace.absolute()
    chain = ['render_observation_v7_translation_fd', 'render_observation_v7b_translation_fd', 'render_observation_v8_acc_only']
    if a.dry_run:
        print(json.dumps({'episode': 16, 'warm_start_chain': chain, 'tracker': 'pruned_tracking_v4_acc001',
                          'final': 'baseline-preserving windows; canonical human and scale restored'}, indent=2)); return
    if not (work / 'materialization.json').is_file(): raise ValueError('Requires a materialized workspace')
    import numpy as np
    root = work / 'outputs/episode_000016'; runtime = read(work / 'runtime.local.json')
    python = str(work / '.venv-cari4d/bin/python'); statepath = root / 'legacy16_steps.json'
    state = read(statepath) if statepath.exists() else {}
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', V2D_RESOURCE_POLICY='unrestricted',
                                       PYTORCH_ALLOC_CONF='expandable_segments:True')
    def run(name, script, args, artifacts, object_env=False):
        command = [str(work / '.venv-sam3d-object/bin/python') if object_env else python, str(work/'scripts'/script), *map(str,args)]
        signature = digest(work/'scripts'/script)
        if name in state:
            old = state[name]
            if old['command'] != command or old['source_sha256'] != signature or any(digest(work/f) != h for f,h in old['outputs'].items()):
                raise ValueError('Historical replay stage changed: ' + name)
            return
        log = root/'legacy_logs'/(name+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
        print('LEGACY16', name, flush=True)
        with log.open('w') as f: subprocess.run(command, cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        state[name] = {'command':command,'source_sha256':signature,'outputs':{str(f.relative_to(work)):digest(f) for f in artifacts}}
        write(statepath,state)
    run('segment','segment_track1.py',['--video',root/'inputs/episode_000016.0.color.mp4','--out',root/'segmentation',
        '--prompts',root/'inputs/prompts.json','--checkpoint',runtime['sam2_checkpoint'],'--chunk-frames','96'],[root/'segmentation/masks.h5'])
    run('object','reconstruct_object.py',['--source',runtime['sam3d_objects'],'--image',root/'segmentation/frames/000000.jpg',
        '--mask',root/'segmentation/first_1_mask.png','--out',root/'object'],[root/'object/object_local.glb'],True)
    native=root/'cari4d/episode_000016'
    files={'depth':native/'depth/moge2/raw.npy','human_export':root/'human_export/episode_000016/object_mesh/output_aligned.glb',
           'human':native/'mhr_init/episode_000016.pkl','alignment':native/'depth/moge2/aligned.h5','scale':root/'object/object_metric_estimate.glb'}
    for stage,target in files.items():
        run(stage,'bootstrap_cari4d_native.py',[stage,'--episode','episode_000016','--frames','360','--batch-size','2'],[target])
    for stage,target in [('metric_export','export/episode_000016/object_mesh/output_aligned.glb'),
                         ('foundationpose','foundationpose/episode_000016.pkl'),('coconet','inference/coconet.pth'),('refinement','inference/refined.pth')]:
        run(stage,'run_cari4d_native.py',['--episode','episode_000016','--from-stage',stage,'--through-stage',stage],[native/target])
    base=root/'render_observation_v1/canonical_baseline'
    run('canonical','canonicalize_track1_body.py',['--bundle',native/'inference/refined.pth','--out',base],[base/'native.npz',base/'canonical.pth'])
    if not (root/'syilen/upper_limb/object_sdf.npz').exists(): make_diagnostic_sdf(work)
    originals = {}
    def restore():
        for path,data in originals.items(): path.write_bytes(data)
        originals.clear()
        for folder in [work/'scripts/__pycache__',work/'v2d_syilen/__pycache__']:
            if folder.exists(): shutil.rmtree(folder)
    def install(snapshot):
        restore()
        for f in snapshot.glob('*.py'):
            if f.name.startswith('test_'): continue
            folder = 'scripts' if f.name in ['track_object_pruned.py','refine_render_observation.py'] else 'v2d_syilen'
            target=work/folder/f.name; originals[target]=target.read_bytes()
            target.write_text(adapt(f.read_text(),work,runtime))
    try:
        previous=None
        for name in chain:
            install(REPO/'legacy/episode16'/name/'source_snapshot')
            cfg=read(REPO/'provenance/episode16'/name/'config.json'); weights=cfg['loss_weights']
            args=['--episode','episode_000016','--start','0','--end','359','--steps','120','--batch','4',
                  '--human-controls','full','--lobs-weight','10','--penetration-weight','0','--loss-preset','object-observation-acc',
                  '--orientation-search','--rotation-temporal','so3','--include-first-orientations',
                  '--angular-velocity-weight',str(weights['angular_velocity']),'--angular-acceleration-weight','1',
                  '--angular-velocity-scale-deg','90','--angular-acceleration-scale-deg','360',
                  '--render-supersample','4','--translation-image-gradient','finite-difference','--translation-fd-step','.002',
                  '--linear-velocity-weight',str(weights.get('object_linear_velocity',0)),'--linear-acceleration-weight','1',
                  '--linear-velocity-scale','.5','--linear-acceleration-scale','1','--baseline',base/'native.npz',
                  '--bundle',base/'canonical.pth','--sdf',root/'syilen/upper_limb/object_sdf.npz','--out',root/name]
            if previous: args += ['--resume-optimized',root/previous/'optimized.npz']
            run(name,'refine_render_observation.py',args,[root/name/'optimized.npz']); previous=name
        install(REPO/'legacy/episode16/pruned_tracking_v4_acc001/source_snapshot')
        tracked=root/'pruned_tracking_v4_acc001'
        run('historical_tracker','track_object_pruned.py',['--source',root/chain[-1],'--out',tracked,'--end','359',
            '--beam','4','--batch','4','--angular-acceleration-weight','.01','--linear-acceleration-weight','.01'],[tracked/'optimized.npz'])
    finally: restore()
    source=root/'safe_windows_input_v1'; alternate=root/'safe_windows_alternate_v1'
    source.mkdir(exist_ok=True); alternate.mkdir(exist_ok=True)
    with np.load(base/'native.npz') as z: data={k:z[k] for k in z.files}
    data['rotation']=data.pop('object_rotation'); data['translation']=data.pop('object_translation'); data['coordinate_frame']=np.array('camera')
    np.savez_compressed(source/'optimized.npz',**data)
    cfg=read(REPO/'provenance/episode16/safe_windows_input_v1/config.json')
    cfg=json.loads(adapt(json.dumps(cfg),work,runtime)); cfg['baseline_sha256']=digest(base/'native.npz'); write(source/'config.json',cfg)
    with np.load(tracked/'optimized.npz') as z:
        data['rotation']=z['rotation']; data['translation']=z['translation']
    np.savez_compressed(alternate/'optimized.npz',**data)
    write(alternate/'config.json',{'source':str(tracked/'optimized.npz'),'source_sha256':digest(tracked/'optimized.npz'),
        'adaptation':'Only rotations/translations retained; canonical human and scale restored'})
    run('safe_windows','refine_object_windows_safe.py',['--source',source,'--alternate',alternate,'--out',root/'safe_windows_v1',
        '--window','50','--stride','25','--steps','12','--batch','4'],[root/'safe_windows_v1/optimized.npz'])
    manifest={'input_sha256':{str(p):digest(p) for p in [source/'optimized.npz',source/'config.json',alternate/'optimized.npz',alternate/'config.json']},
              'source_sha256':read(root/'safe_windows_v1/config.json')['source_sha256'],'historical_replay':True}
    write(root/'safe_windows_v1/run_manifest.json',manifest)
    batch=work/'outputs/track1_safe_windows_v1/status.json'
    status=read(batch) if batch.exists() else {'episodes':{}}
    status['episodes']['episode_000016']={'status':'complete','historical_replay':True,'prediction_sha256':digest(root/'safe_windows_v1/optimized.npz')}
    write(batch,status)


if __name__=='__main__': main()
