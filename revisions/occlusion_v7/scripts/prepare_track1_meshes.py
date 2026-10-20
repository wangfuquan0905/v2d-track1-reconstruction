"""Export exactly the official submission-budget meshes in the packing environment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1'
KIT=ROOT/'third_party/v2d_submission_kit_20261018';sys.path.insert(0,str(KIT))
from v2dlb.mesh_budget import budget_mesh
from v2dlb.mesh_quality import report_mesh_quality


def main():
    versions={p:importlib.metadata.version(p) for p in ['numpy','trimesh','fast_simplification']}
    dest=OUT/'submission_meshes';dest.mkdir(exist_ok=True);meshes={};names={};rows=[]
    for ep in json.loads((OUT/'inventory.json').read_text()):
        root=ROOT/'outputs'/ep['episode'];report=root/'safe_windows_v1/report.json'
        if not report.exists() or json.loads(report.read_text()).get('status')!='complete':continue
        path=root/f"cari4d/{ep['episode']}/export/{ep['episode']}/object_mesh/output_aligned.glb"
        sha=hashlib.sha256(path.read_bytes()).hexdigest();cache=dest/(ep['episode']+'.npz');meta=dest/(ep['episode']+'.json')
        if meta.exists():
            prior=json.loads(meta.read_text())
            if prior['source_sha256']!=sha or prior['packages']!=versions:raise ValueError('Submission mesh source/runtime changed')
            with np.load(cache) as z:v,f=z['vertices'],z['faces']
        else:
            v,f=budget_mesh(path,4096,4096);np.savez_compressed(cache,vertices=v,faces=f)
            mesh=trimesh.Trimesh(v,f,process=True);mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices()
            prior={'episode':ep['episode'],'source_sha256':sha,'packages':versions,'source':str(path),
                'budget_vertices':len(v),'budget_faces':len(f),'surface_vertices':len(mesh.vertices),'surface_faces':len(mesh.faces),
                'watertight':bool(mesh.is_watertight),'extents_m':mesh.extents.tolist(),'area_m2':float(mesh.area),
                'budget_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
            meta.write_text(json.dumps(prior,indent=2))
        if not np.isfinite(v).all() or prior['area_m2']<=0:raise ValueError('Invalid mesh')
        key=(ep['episode_index'],0);meshes[key]=(v,f);names[key]=ep['object'];rows.append(prior)
    flags=report_mesh_quality(meshes,names)
    (OUT/'mesh_quality.json').write_text(json.dumps({'episodes':rows,'official_advisories':flags},indent=2))
    print('SUBMISSION_MESHES',len(rows))


if __name__=='__main__':main()
