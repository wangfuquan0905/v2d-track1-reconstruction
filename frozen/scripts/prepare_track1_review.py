"""Export completed test predictions for RGB/3D review; preserve camera and frame IDs."""
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track1_safe_windows_v1'
DEST=OUT/'review'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    subprocess.run([str(ROOT/'.venv-submission/bin/python'),str(ROOT/'scripts/prepare_track1_meshes.py')],check=True)
    inventory=json.loads((OUT/'inventory.json').read_text())
    protocol=json.loads((ROOT/'configs/track1_test_frames_v1.json').read_text())
    scored={r['episode_index']:r for r in protocol['episodes']}
    spec=np.load(ROOT/'third_party/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
    hands=np.unique(spec['vertex_indices']);DEST.mkdir(exist_ok=True);catalog=[]
    for ep in inventory:
        name=ep['episode'];root=ROOT/'outputs'/name;run=root/'safe_windows_v1'
        if not (run/'report.json').exists():continue
        cfg=json.loads((run/'config.json').read_text());report=json.loads((run/'report.json').read_text())
        if report['status']!='complete':continue
        base=Path(cfg['source']);base=base if base.is_absolute() else ROOT/base
        sources={'baseline':base/'optimized.npz','ours':run/'optimized.npz'}
        hashes={key:digest(path) for key,path in sources.items()}
        dest=DEST/name;dest.mkdir(exist_ok=True)
        manifest_path=dest/'manifest.json'
        budget=OUT/'submission_meshes'/(name+'.npz');mesh_hash=digest(budget)
        with np.load(budget) as m:ov,of=m['vertices'],m['faces']
        with np.load(sources['ours']) as raw:ov=ov*float(raw['object_scale'])
        (dest/'object.json').write_text(json.dumps({'vertices':ov.tolist(),'faces':of.tolist()}))
        catalog.append({'id':name,'label':f"{ep['episode_index']:02d} · {ep['object']}",'sequence':ep['sequence_id']})
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if {k:v['sha256'] for k,v in old['sources'].items()}==hashes:
                if old.get('mesh_budget_sha256')!=mesh_hash:
                    old['mesh_budget_sha256']=mesh_hash;manifest_path.write_text(json.dumps(old,ensure_ascii=False))
                print('REUSE_REVIEW',name,flush=True);continue
            raise ValueError('Review input changed: '+name)
        bundle=torch.load(cfg['bundle_path'],map_location='cpu',weights_only=False)
        faces=np.asarray(bundle['faces'],dtype='<u4');del bundle
        hand_faces=np.concatenate([spec['vertex_indices'][i][spec[f'hand_faces_{s}']] for i,s in enumerate(['left','right'])]).astype('<u4')
        faces[~np.all(np.isin(faces,hands),axis=1)].tofile(dest/'body_faces.bin');hand_faces.tofile(dest/'hand_faces.bin')
        with open(root/f'cari4d/{name}/depth/moge2/intrinsics.pkl','rb') as f:k=pickle.load(f)
        ids=scored[ep['episode_index']]['scored_frames'];groups=[]
        for x in ids:
            if not groups or x!=groups[-1][-1]+1:groups.append([x])
            else:groups[-1].append(x)
        manifest={'episode':name,'sequence':ep['sequence_id'],'fps':ep['fps'],'scored_frames':ids,
            'stretches':[[g[0],g[-1]] for g in groups],'width':ep['width'],'height':ep['height'],
            'intrinsics':{key:float(k[key]) for key in ['fx','fy','cx','cy']},
            'stages':{'baseline':'cari4d_full','ours':'safe_windows_v1'},'sources':{},'deltas':{},'scores':{},
            'training_objective':report,'reference_available':False,
            'mesh_budget_sha256':mesh_hash,
            'coordinate_frame':'original camera; no per-variant recentering or reference alignment',
            'object_mesh':'official submission-budget geometry; scale applied once'}
        for stage,path in sources.items():
            with np.load(path) as raw:
                vertices=raw['vertices'];frames=raw['frames'];np.testing.assert_array_equal(frames,np.arange(ep['frames']))
                rotation=raw['rotation'];translation=raw['translation']
                values=np.concatenate([vertices.reshape(len(frames),-1),rotation.reshape(len(frames),9),translation],axis=1).astype('<f4')
                if not np.isfinite(values).all():raise ValueError('Nonfinite review geometry')
                np.save(dest/(stage+'.npy'),values)
                manifest.update(frames=len(frames),vertex_count=vertices.shape[1],record_floats=values.shape[1])
                if stage=='baseline':br,bt=rotation.copy(),translation.copy()
                else:
                    dist=np.linalg.norm(translation-bt,axis=1)*1000
                    cos=(np.einsum('tij,tij->t',rotation,br)-1)/2
                    manifest['deltas'][stage]={'object_mm':dist.tolist(),'rotation_deg':np.degrees(np.arccos(np.clip(cos,-1,1))).tolist(),
                        'hand_mean_mm':[0.]*len(frames),'largest_object_shift_frame':ids[int(np.argmax(dist[ids]))]}
            manifest['sources'][stage]={'path':str(path),'sha256':hashes[stage]}
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False));print('PREPARED_REVIEW',name,flush=True)
    (DEST/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False))
    (DEST/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False))


if __name__=='__main__':main()
