"""Register SAM3D's layout in the inferred camera before the unchanged depth-scale gate.

Corrects focal/layout mismatch using only first-frame RGB segmentation. Rotation,
shape, topology and appearance stay fixed. A positive uniform scale and image-plane
translation are fitted; the original overlap and depth checks remain mandatory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import cv2
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',required=True)
    p.add_argument('--projection-only',action='store_true');a=p.parse_args()
    root=ROOT/'outputs'/a.episode;obj=root/'object';native=root/'cari4d'/a.episode
    layout=json.loads((obj/'object_report.json').read_text())
    original=trimesh.load(obj/'object_local.glb',force='mesh',process=False)
    q=np.array(layout['rotation'])[0];r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    rx=Rotation.from_euler('x',90,degrees=True).as_matrix();scale=np.array(layout['scale'])[0]
    translation=np.array(layout['translation'])[0];flip=np.array([-1.,-1.,1.]);centre=translation*flip
    camera=original.copy();camera.vertices=((original.vertices@rx.T*scale)@r+translation)*flip
    if np.min(camera.vertices[:,2])<=.01:raise ValueError('SAM3D layout has points behind the camera')
    with open(native/'depth/moge2/intrinsics.pkl','rb') as f:k=pickle.load(f)
    mask=cv2.imread(str(root/'segmentation/first_1_mask.png'),0)>0
    if mask.sum()<100:raise ValueError('Insufficient visible object pixels')
    h,w=mask.shape;scene=pyrender.Scene();node=scene.add(pyrender.Mesh.from_trimesh(camera,smooth=False))
    scene.add(pyrender.IntrinsicsCamera(k['fx'],k['fy'],k['cx'],k['cy'],znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]))
    renderer=pyrender.OffscreenRenderer(w,h);cache={};history=[]
    # p = log(uniform scale), camera-x translation, camera-y translation.
    def observe(params):
        key=tuple(np.asarray(params).tolist())
        if key in cache:return cache[key]
        factor=np.exp(params[0]);delta=np.array([params[1],params[2],0.])
        transform=np.eye(4);transform[:3,:3]*=factor;transform[:3,3]=(1-factor)*centre+delta
        scene.set_pose(node,transform)
        depth=renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY);visible=depth>0
        iou=float(np.sum(visible&mask)/max(1,np.sum(visible|mask)))
        yy,xx=np.where(visible)
        if not len(xx):return (-1.,None)
        bounds=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
        cache[key]=(iou,bounds);return cache[key]
    yy,xx=np.where(mask);target=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
    params=np.zeros(3);before,bounds=observe(params);best=params.copy();best_score=before
    try:
        if bounds is None:raise ValueError('SAM3D mesh outside camera')
        # Initialize from image bounding boxes, keeping SAM3D's depth and rotation.
        for _ in range(3):
            score,bounds=observe(params)
            ratios=(target[2:]-target[:2])/(bounds[2:]-bounds[:2])
            factor=float(np.sqrt(ratios.prod()))
            params[0]+=np.log(np.clip(factor,.5,2.))
            shift=(target[:2]+target[2:]-bounds[:2]-bounds[2:])/2
            params[1:]+=shift*centre[2]/np.array([k['fx'],k['fy']])
            score,_=observe(params)
            if score>best_score:best,best_score=params.copy(),score
        params=best.copy()
        for pixel_step,scale_step in [(16,.16),(8,.08),(4,.04),(2,.02),(1,.01),(.5,.005)]:
            steps=np.array([scale_step,pixel_step*centre[2]/k['fx'],pixel_step*centre[2]/k['fy']])
            for iteration in range(16):
                previous=best_score
                for axis in range(3):
                    for sign in [-1.,1.]:
                        trial=params.copy();trial[axis]+=sign*steps[axis]
                        if abs(trial[0])>np.log(4):continue
                        score,_=observe(trial)
                        if score>best_score+1e-7:params,best_score=trial,score
                history.append({'pixel_step':pixel_step,'iou':best_score,'params':params.tolist()})
                if best_score<=previous+1e-7:break
        after,bounds=observe(params)
        factor=float(np.exp(params[0]));delta=np.r_[params[1:],0.]
        transform=np.eye(4);transform[:3,:3]*=factor;transform[:3,3]=(1-factor)*centre+delta;scene.set_pose(node,transform)
        depth=renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:renderer.delete()
    dest=obj/'registered_layout_v1';dest.mkdir(exist_ok=True)
    proof={'method':'fixed SAM3D rotation and geometry; positive isotropic scale plus camera-plane translation fitted to first-frame RGB mask',
        'before_iou':before,'after_iou':after,'target_bbox':target.tolist(),'registered_bbox':bounds.tolist(),
        'layout_scale_factor':factor,'camera_translation_delta':delta.tolist(),'history':history,
        'original_report_sha256':hashlib.sha256((obj/'object_report.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_inputs':False}
    (dest/'registration.json').write_text(json.dumps(proof,indent=2))
    rgb=cv2.imread(str(root/'segmentation/frames/000000.jpg'));rgb[mask]=(.65*rgb[mask]+.35*np.array([30,240,30])).astype('uint8')
    contours,_=cv2.findContours((depth>0).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(0,0,255),2)
    cv2.imwrite(str(dest/'projection.jpg'),rgb)
    print('LAYOUT_REGISTRATION',json.dumps({k:v for k,v in proof.items() if k!='history'}),flush=True)
    if after<.4:raise ValueError(f'Registered layout still fails the original IoU gate: {after}')
    if a.projection_only:return
    registered=dict(layout);registered['scale']=(scale*factor)[None].tolist()
    registered['translation']=(translation+delta*flip)[None].tolist()
    registered['layout_registration']=str(dest/'registration.json')
    (dest/'object_report.json').write_text(json.dumps(registered,indent=2))
    link=dest/'object_local.glb'
    if not link.exists():link.symlink_to(obj/'object_local.glb')
    subprocess.run([sys.executable,str(ROOT/'scripts/calibrate_object_scale.py'),'--object-dir',str(dest),
        '--aligned-depth',str(native/'depth/moge2/aligned.h5'),'--intrinsics',str(native/'depth/moge2/intrinsics.pkl'),
        '--mask',str(root/'segmentation/first_1_mask.png')],cwd=ROOT,check=True)
    # Only promote calibration results after the unmodified overlap/depth gate passes.
    for name in ['object_metric_estimate.glb','scale_diagnostic.npz']:
        (obj/name).write_bytes((dest/name).read_bytes())
    report=json.loads((dest/'scale_report.json').read_text());report['layout_registration']=proof
    (obj/'scale_report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
