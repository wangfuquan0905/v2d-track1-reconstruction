"""Repair observed SAM2 tracking loss using explicit RGB anchors, preserving history."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import h5py
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track1_safe_windows_v1'
CONFIG=ROOT/'configs/track1_mask_recovery_v1.json'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);a=p.parse_args()
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    patches=json.loads(CONFIG.read_text())['episodes'][a.episode]
    root=ROOT/'outputs'/a.episode/'segmentation';dest=root/'recovery_v1';dest.mkdir(exist_ok=True)
    success=dest/'provenance.json'
    if success.exists():
        done=json.loads(success.read_text())
        if done['patches']!=patches or any(digest(p)!=h for p,h in done['output_sha256'].items()):raise ValueError('Previously repaired masks changed')
        print('REUSE_RECOVERY',a.episode);return
    stage=OUT/'stages'/a.episode/'segment.json';initial=json.loads(stage.read_text())
    if initial['status']!='passed':raise ValueError('Initial segmentation incomplete')
    for path,h in initial['artifact_sha256'].items():
        if digest(path)!=h:raise ValueError('Initial segmentation changed')
    for filename in ['masks.h5','segmentation_report.json','masks_preview.mp4','first_1_mask.png']:
        backup=dest/('initial_'+filename)
        if not backup.exists():shutil.copy2(root/filename,backup)
    shutil.copy2(stage,dest/'initial_stage.json')
    temporary=root/'masks.recovery.h5';shutil.copy2(root/'masks.h5',temporary)
    report=json.loads((root/'segmentation_report.json').read_text());seq=report['prompts']['sequence']
    model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','/home/lab-2/Research/RLSIM/third_party/sam2/checkpoints/sam2.1_hiera_large.pt')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),h5py.File(temporary,'r+') as masks:
        for patch_id,patch in enumerate(patches):
            for direction,ids in [('backward',list(range(patch['anchor'],patch['start']-1,-1))),('forward',list(range(patch['anchor'],patch['end']+1)))]:
                carry=None
                for begin in range(0,len(ids),95):
                    chunk=ids[begin:begin+96]
                    if begin and len(chunk)==1:break
                    folder=dest/f'chunks/{patch_id}_{direction}_{begin}';folder.mkdir(parents=True,exist_ok=True)
                    for i,frame in enumerate(chunk):
                        link=folder/f'{i:06d}.jpg'
                        if not link.exists():os.link(root/'frames'/f'{frame:06d}.jpg',link)
                    state=model.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True)
                    if carry is None:
                        kwargs={'box':np.array(patch['box'],dtype=np.float32)}
                        if 'points' in patch:kwargs.update(points=np.array(patch['points'],dtype=np.float32),labels=np.array(patch['labels'],dtype=np.int32))
                        model.add_new_points_or_box(state,frame_idx=0,obj_id=1,**kwargs)
                    else:model.add_new_mask(state,frame_idx=0,obj_id=1,mask=carry)
                    for local,_,logits in model.propagate_in_video(state):
                        carry=(logits[0,0]>0).cpu().numpy();frame=chunk[local]
                        masks[f'{seq}/{frame:06d}-k0.obj_rend_mask.png'][...]=carry
                        report['stats'][frame]['area_pixels']['1']=int(carry.sum())
                    model.reset_state(state);del state;gc.collect();torch.cuda.empty_cache()
        masks.attrs['rgb_recovery_v1']=json.dumps(patches)
        if any(p['start']==0 for p in patches):
            cv2.imwrite(str(root/'first_1_mask.png'),masks[f'{seq}/000000-k0.obj_rend_mask.png'][:].astype('uint8')*255)
    report['rgb_recovery_v1']={'patches':patches,'initial_report':str(dest/'initial_segmentation_report.json'),'reference_inputs':False}
    temporary.replace(root/'masks.h5');write(root/'segmentation_report.json',report)
    source_hashes={str(Path(__file__).resolve()):digest(__file__)}
    outputs={str(root/f):digest(root/f) for f in ['masks.h5','segmentation_report.json','first_1_mask.png']}
    provenance={'episode':a.episode,'patches':patches,'source_sha256':source_hashes,'config_sha256_at_run':digest(CONFIG),'output_sha256':outputs,
        'initial_artifact_sha256':initial['artifact_sha256'],'created':time.time(),'reference_inputs':False}
    write(success,provenance)
    initial['initial_artifact_sha256']=initial['artifact_sha256'];initial['artifact_sha256']=outputs
    initial['postprocessing']={'provenance':str(success),'sha256':digest(success),'source_sha256':source_hashes}
    write(stage,initial)
    print('RECOVERY_COMPLETE',a.episode,flush=True)


if __name__=='__main__':main()
