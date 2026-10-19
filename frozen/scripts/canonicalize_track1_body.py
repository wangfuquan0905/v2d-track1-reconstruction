"""Project CARI4D output to Track 1's shared identity and serialize native MHR.

Uses no reference data. Identity is the temporal median of RGB predictions,
expression is zero (not submitted). Native model parameters are decoded again
and checked against the CARI4D decoder before they may be evaluated.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def decode_native(model, pose, scales, shape, batch=4):
    device=next(model.parameters()).device
    flip=torch.tensor([1.,-1.,-1.],device=device)
    vertices=[];joints=[]
    with torch.no_grad():
        for start in range(0,len(pose),batch):
            p=torch.as_tensor(pose[start:start+batch],dtype=torch.float32,device=device)
            s=torch.as_tensor(scales,dtype=torch.float32,device=device).expand(len(p),-1)
            b=torch.as_tensor(shape,dtype=torch.float32,device=device).expand(len(p),-1)
            v,j=model(b.contiguous(),torch.cat([p,s],-1),torch.zeros(len(p),72,device=device),True)
            vertices.append((v*flip/100).cpu().numpy())
            joints.append((j[...,:3]*flip/100).cpu().numpy())
    return np.concatenate(vertices),np.concatenate(joints)


def main():
    from run_cari4d_native import runtime_env
    source,weights,env=runtime_env(ROOT)
    os.environ.update(env)
    for path in env['PYTHONPATH'].split(os.pathsep):sys.path.insert(0,path)
    from lib_mhr import MHRLayer,MHR_PARAM_DIMS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--already-canonical',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    bundle=torch.load(a.bundle,map_location='cpu',weights_only=False)
    revised=dict(bundle);revised['pr']={k:np.array(v,copy=True) for k,v in bundle['pr'].items()}
    block=revised['pr'];n=len(bundle['frames'])
    for key in ['mhr_scale','mhr_shape']:
        if a.already_canonical:
            if not np.allclose(block[key],block[key][:1],atol=1e-6,rtol=0):
                raise ValueError(f'{key} must be constant before native conversion')
        else:
            block[key][:]=np.median(block[key],axis=0)
    if a.already_canonical and np.any(block['mhr_face']):
        raise ValueError('Expression must be zero for the submission decoder')
    block['mhr_face'][:]=0
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=weights/'sam3d_body',device=a.device)
    native_params=[];max_error=0.
    with torch.no_grad():
        for start in range(0,n,4):
            params={k:torch.as_tensor(block[k][start:start+4],device=a.device,dtype=torch.float32) for k in MHR_PARAM_DIMS}
            backend=layer.backend
            context=backend._vertices_context(params,detach_fixed=True)
            trans,body,shape,scale=backend._mutable_vertices_inputs(params,context)
            vertices,joints,model_params=context.head.mhr_forward(global_trans=trans*context.flip,
                global_rot=context.global_rot,body_pose_params=body,hand_pose_params=context.hand,
                scale_params=scale,shape_params=shape,expr_params=torch.zeros_like(context.face),
                return_joint_coords=True,return_model_params=True)
            expected=layer.mhr_forward(params)
            max_error=max(max_error,float((vertices*context.flip-expected.vertices).abs().max()),
                          float((joints[:,1:]*context.flip-expected.joints[:,1:]).abs().max()))
            native_params.append(model_params.cpu().numpy())
    model_params=np.concatenate(native_params)
    if max_error>5e-5:raise ValueError(f'Coordinate conversion error: {max_error} m')
    if not np.allclose(model_params[:,136:],model_params[:1,136:],atol=1e-6,rtol=0):
        raise ValueError('Native scale parameters vary over episode')
    pose=model_params[:,:136].astype(np.float32);scales=model_params[0,136:].astype(np.float32)
    shape=block['mhr_shape'][0].astype(np.float32)
    model=torch.jit.load(str(ROOT/'data/models/mhr_model.pt'),map_location=a.device).eval()
    v,j=decode_native(model,pose,scales,shape)
    # Validate native codec against the source decoder at distributed frames.
    ids=np.unique(np.linspace(0,n-1,min(n,5)).astype(int))
    with torch.no_grad():
        expected=layer.mhr_forward({k:torch.as_tensor(block[k][ids],device=a.device,dtype=torch.float32) for k in MHR_PARAM_DIMS})
    vertex_error=float(np.max(np.abs(v[ids]-expected.vertices.cpu().numpy())))
    # Joint 0 is MHR's fixed dummy/world joint. CARI4D's external translation
    # moves that dummy; native root translation does not. It is not one of the
    # official 22 scored joints or our hand proxies. Keep native joint 0 in output.
    joint_error=float(np.max(np.abs(j[ids,1:]-expected.joints.cpu().numpy()[:,1:])))
    if max(vertex_error,joint_error)>5e-5:raise ValueError('Native MHR roundtrip failed')
    poses=np.asarray(block['pose_abs'])
    np.savez_compressed(a.out/'native.npz',pose=pose,scales=scales,shape=shape,
        object_rotation=poses[:,:3,:3],object_translation=poses[:,:3,3],object_scale=np.array(1.),
        frames=np.array([int(f) for f in bundle['frames']]),vertices=v,joints=j)
    torch.save(revised,a.out/'canonical.pth')
    (a.out/'report.json').write_text(json.dumps({'input_bundle':str(a.bundle),'frames':n,
        'identity':'shared temporal median of predicted compact shape/scale; zero expression',
        'native_shape_contract':{'pose':[n,136],'scales':[68],'shape':[45]},
        'roundtrip_max_vertex_error_m':vertex_error,'roundtrip_max_joint_error_m':joint_error,
        'joint_roundtrip_excludes':[0],
        'reference_inputs':False,'scope':'public native MHR decoder; private Kaggle asset equivalence unverified'},indent=2))


if __name__=='__main__':main()
