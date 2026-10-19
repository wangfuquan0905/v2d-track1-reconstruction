"""Baseline-preserving overlapping-window refinement with global objective acceptance.
Complete baseline and prior-tracker candidates use every frame. Overlapping
windows update all interior poses with boundary-complete acceptance line search.
No GT, speed penalty, fixed rotation range, dynamics rollout or human optimization.
"""
import argparse,json,pickle,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import refine_render_observation
import numpy as np,torch,cv2,h5py,trimesh
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix
from v2d_syilen.render_observation import ObservationRenderer,observation_terms
from v2d_syilen.rotation_temporal import angular_derivatives
from v2d_syilen.window_objective import whole_window_score,temporal_terms,preconditioners


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--alternate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--steps',type=int,default=12);p.add_argument('--window',type=int,default=50);p.add_argument('--stride',type=int,default=25);p.add_argument('--batch',type=int,default=4)
 p.add_argument('--angular-acceleration-weight',type=float,default=.01);p.add_argument('--linear-acceleration-weight',type=float,default=.01)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);started=time.monotonic()
 cfg=json.loads((a.source/'config.json').read_text());raw=np.load(a.source/'optimized.npz');data={k:raw[k] for k in raw.files}
 frames=data['frames'];n=len(frames);fps=cfg['fps'];episode=cfg['episode'];root=ROOT/'outputs'/episode;run=root/'cari4d'/episode
 if n<3 or not np.all(np.diff(frames)==1):raise ValueError('Need >=3 contiguous frames')
 aw,lw=a.angular_acceleration_weight,a.linear_acceleration_weight
 if any(not np.isfinite(v) or v<0 for v in [aw,lw]):raise ValueError('Weights must be nonnegative and finite')
 tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
 bundle=torch.load(Path(cfg['bundle_path']),map_location='cpu',weights_only=False)
 mesh=trimesh.load(run/f'export/{episode}/object_mesh/output_aligned.glb',force='mesh',process=False)
 vertices=tensor(np.asarray(mesh.vertices)*float(data['object_scale']))
 with open(run/'depth/moge2/intrinsics.pkl','rb') as f:k=pickle.load(f)
 x0,y0,x1,y1=cfg['crop_xyxy'];h,w=cfg['render_size'];K=np.array([k['fx']*w/(x1-x0),k['fy']*h/(y1-y0),(k['cx']-x0)*w/(x1-x0),(k['cy']-y0)*h/(y1-y0)])
 hf=torch.tensor(bundle['faces'],device='cuda',dtype=torch.int32);of=torch.tensor(mesh.faces,device='cuda',dtype=torch.int32);colors=tensor(mesh.visual.vertex_colors[:,:3]/255.)
 renderer=ObservationRenderer(hf,of,colors,tensor(K),h,w,supersample=4)
 images=[];masks=[]
 with h5py.File(root/'segmentation/masks.h5') as f:
  for frame in frames:
   im=cv2.cvtColor(cv2.imread(str(root/f'segmentation/frames/{frame:06d}.jpg')),cv2.COLOR_BGR2RGB);images.append(cv2.resize(im[y0:y1,x0:x1],(w,h)))
   m=np.stack([f[f'{episode}/{frame:06d}-k0.person_mask.png'][()],f[f'{episode}/{frame:06d}-k0.obj_rend_mask.png'][()]],-1).astype('float32')
   masks.append(cv2.resize(m[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_NEAREST))
 images=np.array(images);masks=np.array(masks);human=data['vertices']
 counts={'full':0,'coarse':0}
 @torch.no_grad()
 def observe(rs,ts,ids=None,return_images=False):
  if ids is None:ids=np.arange(n)
  parts=[];rendered_parts=[]
  for start in range(0,len(ids),a.batch):
   end=min(len(ids),start+a.batch);ii=ids[start:end]
   obj=torch.einsum('bij,vj->bvi',tensor(rs[start:end]),vertices)+tensor(ts[start:end,None])
   rendered=renderer(tensor(human[ii]).contiguous(),obj)
   terms=observation_terms(rendered,tensor(images[ii].astype('float32')/255.),tensor(masks[ii]))
   parts.append(np.stack([terms['object_silhouette'].cpu().numpy(),terms['object_color'].cpu().numpy()],-1))
   if return_images:rendered_parts.append(rendered.cpu().numpy())
  counts['full']+=len(ids)
  return np.concatenate(parts),np.concatenate(rendered_parts) if return_images else None
 from v2d_syilen.window_search import context_bounds,accepts
 if not 3 <= a.stride <= a.window or a.steps < 0:
  raise ValueError('Require 3 <= stride <= window and nonnegative steps')
 source_r=Rotation.from_matrix(data['rotation']).as_matrix();source_t=data['translation'].astype(float)
 with np.load(a.alternate/'optimized.npz') as alt:
  np.testing.assert_array_equal(alt['frames'],frames)
  for key in ['vertices','pose','object_scale']:
   np.testing.assert_array_equal(alt[key],data[key])
  alt_r=Rotation.from_matrix(alt['rotation']).as_matrix();alt_t=alt['translation'].astype(float)
 def score(ob,r,t):return whole_window_score(np.asarray(ob,dtype=np.float64),r,t,fps,aw,lw)
 baseline_ob,_=observe(source_r,source_t);alternate_ob,_=observe(alt_r,alt_t)
 before=score(baseline_ob,source_r,source_t);alt_terms=score(alternate_ob,alt_r,alt_t)
 r,t,obs=source_r.copy(),source_t.copy(),baseline_ob.copy()
 initial_choice='complete_baseline'
 if accepts(before['total'],alt_terms['total']):
  r,t,obs=alt_r.copy(),alt_t.copy(),alternate_ob.copy();initial_choice='complete_previous_tracker'
 config=dict(cfg);config.update(algorithm='baseline_preserving_overlapping_windows_v1',source=str(a.source),alternate=str(a.alternate),window=a.window,stride=a.stride,steps=a.steps,
  loss_weights={'object_color':20,'object_silhouette':10,'angular_acceleration':aw,'linear_acceleration':lw,'angular_velocity':0,'linear_velocity':0},
  acceptance='Exact change of whole-clip objective, including all triples crossing both window boundaries; full final rerender versus retained complete baseline',
  candidates=['current complete trajectory','complete baseline','previous complete tracker','baseline window with current exterior','previous tracker window with current exterior'],
  fixed=['human parameters/vertices','object scale','mesh and colors','camera'],reference_inputs=False,
  source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'v2d_syilen/window_search.py',ROOT/'v2d_syilen/window_objective.py',ROOT/'v2d_syilen/rotation_temporal.py',ROOT/'v2d_syilen/render_observation.py']})
 (a.out/'config.json').write_text(json.dumps(config,indent=2))
 snapshot=a.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for path in config['source_sha256']:(snapshot/Path(path).name).write_bytes(Path(path).read_bytes())
 print('INITIAL',json.dumps({'baseline':before,'previous_tracker':alt_terms,'selected':initial_choice}),flush=True)
 history=[];windows=[];starts=list(range(0,max(n-a.window+1,1),a.stride))
 if starts[-1]!=max(0,n-a.window):starts.append(max(0,n-a.window))
 for window_index,start in enumerate(starts):
  stop=min(start+a.window,n);lo,hi=context_bounds(start,stop,n);ids=np.arange(start,stop);core=slice(start-lo,stop-lo);m=hi-lo
  cr,ct,co=r[lo:hi].copy(),t[lo:hi].copy(),obs[lo:hi].copy()
  terms=score(co,cr,ct);window_before=terms['total'];seed_name='current';seed_scores={'current':window_before}
  for name,sr,st,so in [('baseline',source_r,source_t,baseline_ob),('previous_tracker',alt_r,alt_t,alternate_ob)]:
   nr,nt,no=cr.copy(),ct.copy(),co.copy();nr[core]=sr[ids];nt[core]=st[ids];no[core]=so[ids]
   candidate=score(no,nr,nt);seed_scores[name]=candidate['total']
   if accepts(terms['total'],candidate['total']):cr,ct,co,terms,seed_name=nr,nt,no,candidate,name
  r[lo:hi],t[lo:hi],obs[lo:hi]=cr,ct,co
  prec_r,prec_t=preconditioners(m,fps,aw,lw);prec_r=prec_r[core,core];prec_t=prec_t[core,core]
  for step in range(a.steps):
   gr=[];gt=[];eps_r=np.deg2rad(.25);eps_t=.001;k=len(ids)
   for axis in range(3):
    dr=np.zeros(3);dr[axis]=eps_r
    plus=Rotation.from_rotvec(dr).as_matrix()@cr[core];minus=Rotation.from_rotvec(-dr).as_matrix()@cr[core]
    ob,_=observe(np.concatenate([plus,minus]),np.concatenate([ct[core],ct[core]]),np.tile(ids,2));loss=10*ob[:,0]+20*ob[:,1]
    gr.append((loss[:k]-loss[k:])/(2*eps_r*m))
    dt=np.zeros(3);dt[axis]=eps_t
    ob,_=observe(np.concatenate([cr[core],cr[core]]),np.concatenate([ct[core]+dt,ct[core]-dt]),np.tile(ids,2));loss=10*ob[:,0]+20*ob[:,1]
    gt.append((loss[:k]-loss[k:])/(2*eps_t*m))
   rr=torch.tensor(cr,device='cuda',dtype=torch.float64);tt=torch.tensor(ct,device='cuda',dtype=torch.float64,requires_grad=True)
   q=torch.zeros(m,3,device='cuda',dtype=torch.float64,requires_grad=True)
   _,alpha=angular_derivatives(axis_angle_to_matrix(q)@rr,fps)
   lin=(tt[2:]-2*tt[1:-1]+tt[:-2])*fps**2
   temporal=(aw*(alpha/(2*np.pi)).square().sum()+lw*lin.square().sum())/m
   temp_r,temp_t=torch.autograd.grad(temporal,[q,tt])
   exact=temporal_terms(cr,ct,fps)
   assert abs(float(temporal.detach())-aw*exact['angular_acceleration']-lw*exact['linear_acceleration'])<1e-6
   grad_r=np.stack(gr,-1)+temp_r.cpu().numpy()[core];grad_t=np.stack(gt,-1)+temp_t.cpu().numpy()[core]
   delta_r=-np.linalg.solve(prec_r,grad_r);delta_t=-np.linalg.solve(prec_t,grad_t)
   delta_r*=min(1,np.deg2rad(3)/max(np.linalg.norm(delta_r,axis=1).max(),1e-12))
   delta_t*=min(1,.003/max(np.linalg.norm(delta_t,axis=1).max(),1e-12))
   accepted=False;old=terms['total'];global_before=score(obs,r,t)['total']
   for trial in range(7):
    factor=.5**trial;nr,nt,no=cr.copy(),ct.copy(),co.copy()
    nr[core]=Rotation.from_rotvec(delta_r*factor).as_matrix()@cr[core];nt[core]+=delta_t*factor
    no[core],_=observe(nr[core],nt[core],ids);candidate=score(no,nr,nt)
    if accepts(old,candidate['total']):
     cr,ct,co,terms=nr,nt,no,candidate;accepted=True;break
   r[lo:hi],t[lo:hi],obs[lo:hi]=cr,ct,co
   global_after=score(obs,r,t)['total']
   expected=global_before+(terms['total']-old)*m/n
   assert abs(global_after-expected)<1e-5 and global_after<=global_before+1e-8
   history.append({'window':window_index,'start':int(frames[start]),'stop_exclusive':int(frames[stop-1]+1),'step':step+1,'accepted':accepted,'factor':factor if accepted else 0,'local_before':old,'local_after':terms['total'],'global_before':global_before,'global_after':global_after})
   if not accepted:break
  record={'window':window_index,'start':start,'stop':stop,'seed':seed_name,'seed_scores':seed_scores,'local_before':window_before,'local_after':terms['total'],'global':score(obs,r,t),'elapsed_seconds':time.monotonic()-started}
  windows.append(record);print('WINDOW',json.dumps(record),flush=True)
  (a.out/'history.json').write_text(json.dumps(history,indent=2));(a.out/'windows.json').write_text(json.dumps(windows,indent=2))
  np.savez_compressed(a.out/'tracking_checkpoint.npz',frames=frames,rotation=r,translation=t)
 final_ob,_=observe(r,t);after=score(final_ob,r,t);cached=score(obs,r,t)
 np.testing.assert_allclose(final_ob,obs,atol=2e-6,rtol=1e-5)
 rollback=False
 if not np.isfinite(after['total']) or after['total']>before['total']:
  r,t,final_ob=source_r,source_t,baseline_ob;after=before;rollback=True
 result=data.copy();result.update(rotation=r,translation=t);np.savez_compressed(a.out/'optimized.npz',**result)
 (a.out/'per_frame_observations.json').write_text(json.dumps({'frames':frames.tolist(),'baseline':baseline_ob.tolist(),'previous_tracker':alternate_ob.tolist(),'optimized':final_ob.tolist(),'order':['silhouette','color']},indent=2))
 # Stream images; avoid retaining full-clip floating-point renders in host RAM.
 import imageio.v2 as imageio
 def overlay(im,render):
  alpha=render[...,:2].sum(-1).clip(0,1);color=render[...,0:1]*np.array([.68,.78,.88])+render[...,2:]
  return np.uint8(np.clip(im*(1-.7*alpha[...,None])+.7*color*255,0,255))
 sampled=[]
 with imageio.get_writer(str(a.out/'comparison.mp4'),fps=fps,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18']) as writer:
  for i,frame in enumerate(frames):
   _,renders=observe(np.stack([source_r[i],r[i]]),np.stack([source_t[i],t[i]]),np.array([i,i]),return_images=True)
   panels=[images[i],overlay(images[i],renders[0]),overlay(images[i],renders[1])]
   for j,label in enumerate(['RGB','Full CARI4D','Baseline + joint windows']):
    panels[j]=cv2.copyMakeBorder(panels[j],32,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(panels[j],f'{label} | {frame}',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
   strip=np.concatenate(panels,1);writer.append_data(strip)
   if i in [0,n//4,n//2,3*n//4,n-1]:sampled.append(strip)
 cv2.imwrite(str(a.out/'comparison.png'),cv2.cvtColor(np.concatenate(sampled),cv2.COLOR_RGB2BGR))
 report={'status':'complete','frames':n,'baseline':before,'previous_tracker':alt_terms,'after':after,'initial_choice':initial_choice,'rolled_back_to_baseline':rollback,'cached_final':cached,'final_rerender_check':True,'accepted_updates':sum(x['accepted'] for x in history),'windows':len(windows),'elapsed_seconds':time.monotonic()-started,'render_counts':counts,'reference_inputs':False,'scope':'Object R/t only; fixed human and scale; no SyILEN rollout; objective decrease does not imply GT improvement'}
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()
