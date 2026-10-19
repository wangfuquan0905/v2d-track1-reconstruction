"""Resumable serial Track 1 RGB reconstruction and frozen safe_windows_v1 queue.

No reference data or Kaggle upload. Stage failures are recorded per episode;
other episodes continue. Completed artifacts are reused only after hash checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track1_safe_windows_v1'
PY=ROOT/'.venv-cari4d/bin/python'
OBJECT_PY=ROOT/'.venv-sam3d-object/bin/python'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['segment','full'],default='full')
    p.add_argument('--episodes',default='all',help='Comma-separated indices or all')
    a=p.parse_args();OUT.mkdir(exist_ok=True)
    lockfile=(OUT/'queue.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=json.loads((OUT/'inventory.json').read_text())
    selected=list(range(30)) if a.episodes=='all' else [int(x) for x in a.episodes.split(',')]
    entries={r['episode_index']:r for r in inventory}
    source_names=['scripts/segment_track1.py','scripts/reconstruct_object.py','scripts/bootstrap_cari4d_native.py',
        'scripts/run_cari4d_native.py','scripts/calibrate_object_scale.py','scripts/canonicalize_track1_body.py',
        'scripts/prepare_track1_optimizer_input.py','scripts/track_object_pruned.py','scripts/refine_object_windows_safe.py',
        'v2d_syilen/pruned_tracking.py','v2d_syilen/render_observation.py','v2d_syilen/rotation_temporal.py',
        'v2d_syilen/window_objective.py','v2d_syilen/window_search.py']
    frozen={name:digest(ROOT/name) for name in source_names}
    freeze_path=OUT/'implementation.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())['source_sha256']!=frozen:raise ValueError('Frozen implementation changed')
    else:
        write(freeze_path,{'method':'safe_windows_v1','source_sha256':frozen,
            'settings':{'window':50,'stride':25,'steps':12,'beam':4,'object_silhouette':10,'object_color':20,
                'angular_acceleration':.01,'linear_acceleration':.01,'velocity':0},
            'reference_inputs':False,'submission_authorized':False,
            'episode_16':'Reuse previously reviewed complete safe_windows_v1 after provenance validation'})
        for name in source_names:
            target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    state_path=OUT/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'episodes':{},'created':time.time()}
    state.update(status='running',phase=a.phase,started=time.time(),pid=os.getpid());write(state_path,state)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        CONDA_PREFIX='/home/lab-2/miniconda3/envs/RLSIM',V2D_RESOURCE_POLICY='unrestricted')
    def run(ep,name,cmd,artifacts,inputs=()):
        cmd=list(map(str,cmd));artifacts=list(map(Path,artifacts));inputs=list(map(Path,inputs))
        records=OUT/'stages'/ep;records.mkdir(parents=True,exist_ok=True);record=records/(name+'.json')
        script=Path(cmd[1]);script=script if script.is_absolute() else ROOT/script
        signature=digest(script)
        input_hashes={str(p):digest(p) for p in inputs}
        if record.exists():
            old=json.loads(record.read_text())
            if old.get('status')=='passed':
                if old['command']!=cmd or old['script_sha256']!=signature or old['input_sha256']!=input_hashes:
                    raise RuntimeError('Successful stage inputs changed: '+str(record))
                if 'postprocessing' in old:
                    post=old['postprocessing']
                    for path,h in {post['provenance']:post['sha256'],**post['source_sha256']}.items():
                        if digest(path)!=h:raise RuntimeError('Input correction provenance changed: '+path)
                if all(p.exists() and digest(p)==old['artifact_sha256'][str(p)] for p in artifacts):
                    print('REUSE',ep,name,flush=True);return
                raise RuntimeError('Successful stage output hash mismatch: '+str(record))
            if old.get('status') in ['failed','running']:
                stamp=str(int(time.time()));shutil.copy2(record,records/(name+'.previous_'+stamp+'.json'))
                old_log=records/(name+'.log')
                if old_log.exists():shutil.copy2(old_log,records/(name+'.previous_'+stamp+'.log'))
        for path,h in frozen.items():
            if digest(ROOT/path)!=h:raise RuntimeError('Frozen source changed: '+path)
        row={'status':'running','command':cmd,'script_sha256':signature,'input_sha256':input_hashes,'started':time.time()}
        write(record,row);state['episodes'][ep]={'status':'running','stage':name};write(state_path,state)
        print('START',ep,name,flush=True)
        try:
            if shutil.disk_usage(ROOT).free<15*1024**3:raise RuntimeError('Less than 15 GiB disk space remains')
            with open(records/(name+'.log'),'w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not all(p.is_file() for p in artifacts):raise RuntimeError('Stage missing required output')
            row.update(status='passed',artifact_sha256={str(p):digest(p) for p in artifacts},elapsed_seconds=time.time()-row['started'])
        except Exception as exc:
            row.update(status='failed',error=str(exc),elapsed_seconds=time.time()-row['started']);write(record,row);raise
        write(record,row);print('DONE',ep,name,round(row['elapsed_seconds'],1),flush=True)
    failed=[]
    for i in selected:
        ep=entries[i]['episode'];root=ROOT/'outputs'/ep;native=root/'cari4d'/ep;n=entries[i]['frames']
        try:
            if i==16:
                report=json.loads((root/'safe_windows_v1/report.json').read_text())
                manifest=json.loads((root/'safe_windows_v1/run_manifest.json').read_text())
                for path,h in {**manifest['input_sha256'],**manifest['source_sha256']}.items():
                    if digest(path)!=h:raise RuntimeError('Reviewed episode16 provenance changed: '+path)
                if report['status']!='complete' or report['frames']!=n:raise RuntimeError('Episode16 incomplete')
                state['episodes'][ep]={'status':'complete','reused_reviewed_result':True,'result':str(root/'safe_windows_v1/optimized.npz'),
                    'prediction_sha256':digest(root/'safe_windows_v1/optimized.npz')};write(state_path,state);continue
            video=root/'inputs'/(ep+'.0.color.mp4');prompts=root/'inputs/prompts.json'
            run(ep,'segment',[PY,'scripts/segment_track1.py','--video',video,'--out',root/'segmentation',
                '--prompts',prompts,'--checkpoint','/home/lab-2/Research/RLSIM/third_party/sam2/checkpoints/sam2.1_hiera_large.pt','--chunk-frames','96'],
                [root/'segmentation/segmentation_report.json',root/'segmentation/masks.h5'],[video,prompts])
            if a.phase=='segment':
                state['episodes'][ep]={'status':'segmented'};write(state_path,state);continue
            run(ep,'object',[OBJECT_PY,'scripts/reconstruct_object.py','--source','/home/lab-2/Research/RLSIM/sam-3d-objects',
                '--image',root/'segmentation/frames/000000.jpg','--mask',root/'segmentation/first_1_mask.png','--out',root/'object'],
                [root/'object/object_local.glb',root/'object/object_report.json'],[root/'segmentation/frames/000000.jpg',root/'segmentation/first_1_mask.png'])
            artifacts={
                'depth':[native/'depth/moge2/raw.npy',native/'depth/moge2/intrinsics.pkl',native/'depth/moge2/depth_report.json'],
                'human_export':[root/'human_export'/ep/'object_mesh/output_aligned.glb'],
                'human':[native/'mhr_init'/(ep+'.pkl')],
                'alignment':[native/'depth/moge2/aligned.h5'],
                'scale':[root/'object/object_metric_estimate.glb',root/'object/scale_report.json']}
            for stage in ['depth','human_export','human','alignment','scale']:
                command=[PY,'scripts/bootstrap_cari4d_native.py',stage,'--episode',ep,'--frames',str(n),'--batch-size','1']
                stage_inputs=[]
                if stage=='scale':
                    previous=OUT/'stages'/ep/'scale.json';previous_log=previous.with_suffix('.log')
                    old=json.loads(previous.read_text()) if previous.exists() else {}
                    registered=any('calibrate_object_scale_registered.py' in str(v) for v in old.get('command',[]))
                    layout_failure=old.get('status')=='failed' and previous_log.exists() and 'SAM3D layout and observed mask disagree' in previous_log.read_text()
                    if registered or layout_failure:
                        command=[PY,'scripts/calibrate_object_scale_registered.py','--episode',ep]
                        stage_inputs=[root/'object/object_local.glb',root/'object/object_report.json',
                            root/'segmentation/first_1_mask.png',native/'depth/moge2/intrinsics.pkl',native/'depth/moge2/aligned.h5']
                run(ep,stage,command,artifacts[stage],stage_inputs)
            for stage,files in [
                ('metric_export',[native/'export'/ep/'object_mesh/output_aligned.glb']),
                ('foundationpose',[native/'foundationpose'/(ep+'.pkl')]),
                ('coconet',[native/'inference/coconet.pth']),('refinement',[native/'inference/refined.pth'])]:
                run(ep,stage,[PY,'scripts/run_cari4d_native.py','--episode',ep,'--from-stage',stage,'--through-stage',stage],files)
            run(ep,'canonical',[PY,'scripts/canonicalize_track1_body.py','--bundle',native/'inference/refined.pth','--out',root/'test_canonical_v1'],
                [root/'test_canonical_v1/native.npz',root/'test_canonical_v1/canonical.pth',root/'test_canonical_v1/report.json'],[native/'inference/refined.pth'])
            run(ep,'optimizer_input',[PY,'scripts/prepare_track1_optimizer_input.py','--episode',ep],
                [root/'test_safe_input_v1/optimized.npz',root/'test_safe_input_v1/config.json'],[root/'test_canonical_v1/native.npz'])
            run(ep,'tracker',[PY,'scripts/track_object_pruned.py','--source',root/'test_safe_input_v1','--out',root/'test_tracker_v1',
                '--end',str(n-1),'--beam','4','--batch','4','--angular-acceleration-weight','.01','--linear-acceleration-weight','.01'],
                [root/'test_tracker_v1/optimized.npz',root/'test_tracker_v1/report.json'],[root/'test_safe_input_v1/optimized.npz',root/'test_safe_input_v1/config.json'])
            safe_batch='4'
            previous=OUT/'stages'/ep/'safe_windows.json';previous_log=previous.with_suffix('.log')
            old=json.loads(previous.read_text()) if previous.exists() else {}
            old_command=old.get('command',[])
            already_serial='--batch' in old_command and old_command[old_command.index('--batch')+1]=='1'
            already_cached=any('run_safe_windows_cached_topology.py' in v for v in old_command)
            rerender_failure=old.get('status')=='failed' and previous_log.exists() and 'np.testing.assert_allclose(final_ob,obs' in previous_log.read_text()
            if already_serial or rerender_failure:safe_batch='1'
            safe_script='scripts/run_safe_windows_cached_topology.py' if already_cached or rerender_failure else 'scripts/refine_object_windows_safe.py'
            safe_inputs=[root/'test_safe_input_v1/optimized.npz',root/'test_tracker_v1/optimized.npz']
            if safe_script.endswith('cached_topology.py'):safe_inputs.append(ROOT/'v2d_syilen/render_topology_cache.py')
            run(ep,'safe_windows',[PY,safe_script,'--source',root/'test_safe_input_v1','--alternate',root/'test_tracker_v1',
                '--out',root/'safe_windows_v1','--window','50','--stride','25','--steps','12','--batch',safe_batch],
                [root/'safe_windows_v1/optimized.npz',root/'safe_windows_v1/report.json',root/'safe_windows_v1/comparison.mp4',root/'safe_windows_v1/comparison.png'],
                safe_inputs)
            run(ep,'review_export',[PY,'scripts/prepare_track1_review.py'],
                [OUT/'review'/ep/'manifest.json'],[root/'safe_windows_v1/optimized.npz'])
            run(ep,'submission_validation',[PY,'scripts/prepare_track1_submission.py'],
                [OUT/'submission_episodes'/(ep+'.npz')],[root/'safe_windows_v1/optimized.npz'])
            state['episodes'][ep]={'status':'complete','result':str(root/'safe_windows_v1/optimized.npz'),
                'prediction_sha256':digest(root/'safe_windows_v1/optimized.npz')};write(state_path,state)
        except Exception as exc:
            failed.append(ep);state['episodes'].setdefault(ep,{}).update(status='failed',error=str(exc));write(state_path,state)
            print('FAILED',ep,str(exc),flush=True)
    state.update(status='finished_with_failures' if failed else 'phase_complete',failed=failed,finished=time.time());write(state_path,state)
    print('QUEUE_FINISHED',json.dumps({'failed':failed,'selected':selected,'phase':a.phase}),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
