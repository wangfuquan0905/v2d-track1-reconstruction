"""Validate completed safe-window predictions, stage episode files and optionally pack.

An incomplete run always stays incomplete. No placeholder predictions, fabricated
commit URLs, reference-label access, or competition upload are implemented here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v2d_syilen.submission_checks import check_episode,submission_arrays
OUT=ROOT/'outputs/track1_safe_windows_v1'
KIT=ROOT/'third_party/v2d_submission_kit_20261018'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit-url');a=p.parse_args()
    inventory=json.loads((OUT/'inventory.json').read_text())
    protocol=json.loads((ROOT/'configs/track1_test_frames_v1.json').read_text())
    sample=KIT/'data/track_1_sample_submission.parquet'
    if digest(sample)!=protocol['sample_sha256']:raise ValueError('Submission sample changed')
    scored={e['episode_index']:e['scored_frames'] for e in protocol['episodes']}
    dest=OUT/'submission_episodes';dest.mkdir(exist_ok=True)
    report={'status':'incomplete','method':'safe_windows_v1','expected_episodes':30,'expected_scored_frames':9877,
        'sample_sha256':digest(sample),'complete':[],'missing':[],'invalid':[],
        'reference_inputs':False,'uploaded':False,'commit_url':a.commit_url}
    for ep in inventory:
        name=ep['episode'];root=ROOT/'outputs'/name;run=root/'safe_windows_v1'
        if not (run/'report.json').exists():report['missing'].append(name);continue
        try:
            result=json.loads((run/'report.json').read_text());cfg=json.loads((run/'config.json').read_text())
            if result['status']!='complete':raise ValueError('Optimization incomplete')
            for path,sha in cfg['source_sha256'].items():
                if digest(path)!=sha:raise ValueError('Frozen optimization source changed: '+path)
            base=Path(cfg['source']);base=base if base.is_absolute() else ROOT/base
            with np.load(run/'optimized.npz') as src,np.load(base/'optimized.npz') as baseline:
                data={k:src[k] for k in src.files}
                checks=check_episode(data,ep['frames'],scored[ep['episode_index']],baseline=baseline)
            if not result.get('final_rerender_check') or result['after']['total']>result['baseline']['total']+1e-8:
                raise ValueError('Full objective verification failed')
            mesh=root/f'cari4d/{name}/export/{name}/object_mesh/output_aligned.glb'
            if not mesh.is_file():raise ValueError('Object mesh missing')
            target=dest/(name+'.npz');np.savez_compressed(target,**submission_arrays(data))
            link=dest/(name+'_object.glb')
            if link.exists() and link.resolve()!=mesh.resolve():raise ValueError('Wrong staged mesh')
            if not link.exists():link.symlink_to(mesh)
            report['complete'].append({'episode':name,'checks':checks,'prediction_sha256':digest(run/'optimized.npz'),
                'submission_npz_sha256':digest(target),'mesh_sha256':digest(mesh),
                'objective_before':result['baseline']['total'],'objective_after':result['after']['total'],
                'rollback':result['rolled_back_to_baseline'],'video':str(run/'comparison.mp4')})
        except Exception as exc:report['invalid'].append({'episode':name,'error':str(exc)})
    if not report['missing'] and not report['invalid'] and len(report['complete'])==30:
        media=subprocess.run([sys.executable,str(ROOT/'scripts/validate_track1_media.py')],cwd=ROOT)
        if media.returncode:
            report['invalid'].append({'episode':'review_media','error':'See media_validation.json'})
        else:report['status']='episodes_ready_pending_commit' if not a.commit_url else 'packing'
    (OUT/'submission_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'complete':len(report['complete']),
        'missing':report['missing'],'invalid':report['invalid']},indent=2),flush=True)
    if report['status']=='packing':
        output=OUT/'track1_safe_windows_v1.parquet'
        cmd=[str(ROOT/'.venv-submission/bin/python'),str(KIT/'eval_reconstruction.py'),
            '--episodes',str(dest),'--sample',str(sample),'--commit',a.commit_url,'--out',str(output)]
        with open(OUT/'packing.log','w') as log:subprocess.run(cmd,cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(ROOT/'.venv-submission/bin/python'),str(ROOT/'scripts/prepare_track1_meshes.py')],check=True)
        subprocess.run([str(ROOT/'.venv-submission/bin/python'),str(ROOT/'scripts/audit_track1_parquet.py'),str(output)],check=True)
        report.update(status='packed_pending_review',parquet=str(output),parquet_sha256=digest(output))
        (OUT/'submission_validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
