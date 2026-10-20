"""Experimental IoU-first object recovery and overlapping-window refinement.
Bounded auxiliary losses, monotonic mean IoU, and per-frame overlap safeguards.
Writes a separate experiment; never changes the frozen v1 producer or submission.
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
from v2d_syilen.iou_priority import (score as priority_score, accepts, observation_loss,
 bounded, mask_box_translation, COLOR_WEIGHT, ANGULAR_WEIGHT, LINEAR_WEIGHT,
 MAX_FRAME_IOU_DROP)
from v2d_syilen.render_topology_cache import AntialiasTopologyCache
import nvdiffrast.torch as diffrast


def recover_from_masks(r,t,obs,source_r,masks,mesh_vertices,K,observe):
 """Search image-derived seeds where local finite differences cannot reacquire.

 No score changes, hidden reference poses, mesh rescaling, or human removal:
 every candidate is evaluated with the same full visible-object renderer.
 """
 ids=np.flatnonzero((obs[:,0]>.3)&(masks[...,1].sum((1,2))>=4))
 initial=obs.copy()
 if not len(ids):return r,t,obs,{'attempted_frames':0,'improved_frames':0}
 boxes=[]
 for i in ids:
  yy,xx=np.where(masks[i,...,1]>.5)
  boxes.append(np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]))
 good=np.flatnonzero(obs[:,0]<=.3)
 rr=[];tt=[];ii=[]
 perturb=Rotation.from_rotvec(np.concatenate([np.zeros((1,3)),np.eye(3)*np.deg2rad(30),-np.eye(3)*np.deg2rad(30)])).as_matrix()
 for i,box in zip(ids,boxes):
  # A nearby well-aligned pose provides orientation only; translation is
  # estimated afresh from this frame's segmentation, never extrapolated.
  nearby=int(good[np.argmin(abs(good-i))]) if len(good) else 0
  for base in [source_r[i],r[i],r[nearby]]:
   for rotation in perturb@base:
    for factor in [.7,1.,1.4]:
     rr.append(rotation);tt.append(mask_box_translation(mesh_vertices,rotation,box,K,factor));ii.append(i)
 rr=np.asarray(rr);tt=np.asarray(tt);ii=np.asarray(ii)
 proposal_ob,_=observe(rr,tt,ii)
 for i in ids:
  choices=np.flatnonzero(ii==i);j=choices[np.argmin(proposal_ob[choices,0])]
  if proposal_ob[j,0]<obs[i,0]:r[i],t[i],obs[i]=rr[j],tt[j],proposal_ob[j]
 print('REACQUIRE_SEEDS',json.dumps({'frames':len(ids),'mean_iou':float(1-obs[:,0].mean())}),flush=True)
 # Pixel-scale coordinate proposals can cross zero-overlap plateaus. Selection
 # is strictly per-frame IoU; smoothness is handled by the subsequent windows.
 for deg,pixels,fraction in [(8,4,.08),(4,2,.04),(2,1,.02),(1,.5,.01),(.5,.25,.005),(.25,.125,.0025)]:
  rr=[];tt=[];ii=[]
  for i in ids:
   for rotation in Rotation.from_rotvec(np.concatenate([np.zeros((1,3)),np.eye(3)*np.deg2rad(deg),-np.eye(3)*np.deg2rad(deg)])).as_matrix()@r[i]:
    rr.append(rotation);tt.append(t[i]);ii.append(i)
   for axis in range(3):
    for sign in [-1,1]:
     translation=t[i].copy()
     if axis<2:translation[axis]+=sign*pixels*max(t[i,2],.02)/K[axis]
     else:translation*=1+sign*fraction
     rr.append(r[i]);tt.append(translation);ii.append(i)
  rr=np.asarray(rr);tt=np.asarray(tt);ii=np.asarray(ii)
  ob,_=observe(rr,tt,ii)
  for j,i in enumerate(ids):
   choices=np.arange(j*13,(j+1)*13);best=choices[np.argmin(ob[choices,0])]
   if ob[best,0]<obs[i,0]:r[i],t[i],obs[i]=rr[best],tt[best],ob[best]
 print('REACQUIRE_DONE',json.dumps({'mean_iou':float(1-obs[:,0].mean())}),flush=True)
 return r,t,obs,{'attempted_frames':len(ids),'improved_frames':int(np.sum(obs[:,0]<initial[:,0]-1e-6)),
                 'mean_iou_before':float(1-initial[:,0].mean()),'mean_iou_after':float(1-obs[:,0].mean())}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--alternate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--steps',type=int,default=12);p.add_argument('--window',type=int,default=50);p.add_argument('--stride',type=int,default=25);p.add_argument('--batch',type=int,default=4)
 p.add_argument('--no-reacquire',action='store_true',help='Ablate mask-box seed recovery; keep the same IoU objective')
 p.add_argument('--resume-from',type=Path,help='Continue a completed window checkpoint into a separate new output directory')
 a=p.parse_args()
 if a.out.exists() and any(a.out.iterdir()):p.error('Output must be a new or empty experiment directory')
 a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);started=time.monotonic()
 cfg=json.loads((a.source/'config.json').read_text());raw=np.load(a.source/'optimized.npz');data={k:raw[k] for k in raw.files}
 frames=data['frames'];n=len(frames);fps=cfg['fps'];episode=cfg['episode'];root=ROOT/'outputs'/episode;run=root/'cari4d'/episode
 if n<3 or not np.all(np.diff(frames)==1):raise ValueError('Need >=3 contiguous frames')
 aw,lw=ANGULAR_WEIGHT,LINEAR_WEIGHT
 topology_cache=AntialiasTopologyCache(diffrast.antialias,diffrast.antialias_construct_topology_hash)
 diffrast.antialias=topology_cache
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
 from v2d_syilen.window_search import context_bounds
 if not 3 <= a.stride <= a.window or a.steps < 0:
  raise ValueError('Require 3 <= stride <= window and nonnegative steps')
 source_r=Rotation.from_matrix(data['rotation']).as_matrix();source_t=data['translation'].astype(float)
 with np.load(a.alternate/'optimized.npz') as alt:
  np.testing.assert_array_equal(alt['frames'],frames)
  for key in ['vertices','pose','object_scale']:
   np.testing.assert_array_equal(alt[key],data[key])
  alt_r=Rotation.from_matrix(alt['rotation']).as_matrix();alt_t=alt['translation'].astype(float)
 def score(ob,r,t):return priority_score(ob,r,t,fps)
 baseline_ob,_=observe(source_r,source_t);alternate_ob,_=observe(alt_r,alt_t)
 before=score(baseline_ob,source_r,source_t);alt_terms=score(alternate_ob,alt_r,alt_t)
 r,t,obs=source_r.copy(),source_t.copy(),baseline_ob.copy()
 choose=alternate_ob[:,0]<baseline_ob[:,0]
 r[choose],t[choose],obs[choose]=alt_r[choose],alt_t[choose],alternate_ob[choose]
 initial_choice='per_frame_best_IoU_of_source_and_alternate'
 config=dict(cfg);config.update(algorithm='iou_priority_windows_v2_experimental',source=str(a.source),alternate=str(a.alternate),window=a.window,stride=a.stride,steps=a.steps,
  loss_weights={'object_color_bounded':COLOR_WEIGHT,'object_silhouette':1,'angular_acceleration_bounded':aw,'linear_acceleration_bounded':lw},
  auxiliary_bound=COLOR_WEIGHT+aw+lw,bounded_transform='x / (1 + x), per frame or acceleration triple before averaging',
  acceptance='Exact whole-clip total decrease AND no mean IoU decrease; per-frame IoU may fall at most 0.01 below best seen; all boundary triples included',
  max_frame_iou_drop=MAX_FRAME_IOU_DROP,
  reacquire=not a.no_reacquire,
  candidates=['per-frame better overlap of both input trajectories','mask-box translation with source/current/nearby orientation proposals','pixel/depth/rotation coordinate proposals','joint overlapping-window updates'],
  fixed=['human parameters/vertices','object scale','mesh and colors','camera'],reference_inputs=False,
  source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'v2d_syilen/window_search.py',ROOT/'v2d_syilen/iou_priority.py',ROOT/'v2d_syilen/rotation_temporal.py',ROOT/'v2d_syilen/render_observation.py',ROOT/'v2d_syilen/render_topology_cache.py']},
  input_sha256={str(d/'optimized.npz'):hashlib.sha256((d/'optimized.npz').read_bytes()).hexdigest() for d in [a.source,a.alternate]})
 (a.out/'config.json').write_text(json.dumps(config,indent=2))
 snapshot=a.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for path in config['source_sha256']:(snapshot/Path(path).name).write_bytes(Path(path).read_bytes())
 print('INITIAL',json.dumps({'baseline':before,'previous_tracker':alt_terms,'selected':initial_choice}),flush=True)
 recovery={'attempted_frames':0,'improved_frames':0}
 if not a.no_reacquire and not a.resume_from:
  r,t,obs,recovery=recover_from_masks(r,t,obs,source_r,masks,vertices.cpu().numpy(),K,observe)
 seed_ob=obs.copy();seed_r=r.copy();seed_t=t.copy();best_silhouette=obs[:,0].copy()
 seed_terms=score(obs,r,t)
 history=[];windows=[];starts=list(range(0,max(n-a.window+1,1),a.stride))
 if starts[-1]!=max(0,n-a.window):starts.append(max(0,n-a.window))
 if a.resume_from:
  previous_config=json.loads((a.resume_from/'config.json').read_text())
  for key in ['input_sha256','window','stride','steps','loss_weights','reacquire']:
   if config[key]!=previous_config[key]:raise ValueError(f'Resume configuration differs: {key}')
  with np.load(a.resume_from/'recovery_checkpoint.npz') as seed:
   np.testing.assert_array_equal(seed['frames'],frames)
   seed_r,seed_t=seed['rotation'].copy(),seed['translation'].copy()
   seed_ob,_=observe(seed_r,seed_t)
   np.testing.assert_allclose(seed_ob,seed['observations'],atol=2e-6,rtol=1e-5)
  recovery={'attempted_frames':int(np.sum((obs[:,0]>.3)&(masks[...,1].sum((1,2))>=4))),
   'improved_frames':int(np.sum(seed_ob[:,0]<obs[:,0]-1e-6)),
   'mean_iou_before':float(1-obs[:,0].mean()),'mean_iou_after':float(1-seed_ob[:,0].mean())}
  seed_terms=score(seed_ob,seed_r,seed_t)
  with np.load(a.resume_from/'tracking_checkpoint.npz') as checkpoint:
   np.testing.assert_array_equal(checkpoint['frames'],frames)
   r,t=checkpoint['rotation'].copy(),checkpoint['translation'].copy()
   obs,_=observe(r,t)
   if 'best_silhouette' in checkpoint:
    best_silhouette=checkpoint['best_silhouette'].copy()
   else:
    # Older interrupted runs did not save the best-per-frame floor. This
    # conservative bound permits NO additional per-frame regression; it is
    # at least as strict as every possible floor consistent with that run.
    best_silhouette=np.minimum(seed_ob[:,0],obs[:,0]-MAX_FRAME_IOU_DROP)
  windows=json.loads((a.resume_from/'windows.json').read_text())
  history=[h for h in json.loads((a.resume_from/'history.json').read_text()) if h['window']<len(windows)]
  np.testing.assert_allclose(score(obs,r,t)['total'],windows[-1]['global']['total'],atol=2e-6)
  config['resumed_from']=str(a.resume_from)
  config['resume_checkpoint_sha256']=hashlib.sha256((a.resume_from/'tracking_checkpoint.npz').read_bytes()).hexdigest()
  config['resume_source_sha256']=previous_config['source_sha256']
  config['resume_completed_windows']=len(windows)
  (a.out/'resume_origin_config.json').write_text(json.dumps(previous_config,indent=2))
  (a.out/'config.json').write_text(json.dumps(config,indent=2))
  print('RESUME',json.dumps({'completed_windows':len(windows),'total_windows':len(starts),'mean_iou':float(1-obs[:,0].mean())}),flush=True)
 np.savez_compressed(a.out/'recovery_checkpoint.npz',frames=frames,rotation=seed_r,translation=seed_t,observations=seed_ob)
 completed_windows=len(windows)
 for window_index,start in enumerate(starts):
  if window_index<completed_windows:continue
  stop=min(start+a.window,n);lo,hi=context_bounds(start,stop,n);ids=np.arange(start,stop);core=slice(start-lo,stop-lo);m=hi-lo
  cr,ct,co=r[lo:hi].copy(),t[lo:hi].copy(),obs[lo:hi].copy()
  terms=score(co,cr,ct);window_before=terms['total'];seed_name='current';seed_scores={'current':window_before}
  r[lo:hi],t[lo:hi],obs[lo:hi]=cr,ct,co
  # Damping is numerical step conditioning, not an additional loss. Bounded
  # temporal terms are differentiated exactly below, not as unbounded squares.
  for step in range(a.steps):
   gr=[];gt=[];eps_r=np.deg2rad(.25);eps_t=.001;k=len(ids)
   for axis in range(3):
    dr=np.zeros(3);dr[axis]=eps_r
    plus=Rotation.from_rotvec(dr).as_matrix()@cr[core];minus=Rotation.from_rotvec(-dr).as_matrix()@cr[core]
    ob,_=observe(np.concatenate([plus,minus]),np.concatenate([ct[core],ct[core]]),np.tile(ids,2));loss=observation_loss(ob)
    gr.append((loss[:k]-loss[k:])/(2*eps_r*m))
    dt=np.zeros(3);dt[axis]=eps_t
    ob,_=observe(np.concatenate([cr[core],cr[core]]),np.concatenate([ct[core]+dt,ct[core]-dt]),np.tile(ids,2));loss=observation_loss(ob)
    gt.append((loss[:k]-loss[k:])/(2*eps_t*m))
   rr=torch.tensor(cr,device='cuda',dtype=torch.float64);tt=torch.tensor(ct,device='cuda',dtype=torch.float64,requires_grad=True)
   q=torch.zeros(m,3,device='cuda',dtype=torch.float64,requires_grad=True)
   _,alpha=angular_derivatives(axis_angle_to_matrix(q)@rr,fps)
   lin=(tt[2:]-2*tt[1:-1]+tt[:-2])*fps**2
   temporal=(aw*bounded((alpha/(2*np.pi)).square().sum(-1)).sum()+lw*bounded(lin.square().sum(-1)).sum())/m
   temp_r,temp_t=torch.autograd.grad(temporal,[q,tt])
   assert abs(float(temporal.detach())-terms['angular_contribution']-terms['linear_contribution'])<1e-6
   grad_r=np.stack(gr,-1)+temp_r.cpu().numpy()[core];grad_t=np.stack(gt,-1)+temp_t.cpu().numpy()[core]
   delta_r=-grad_r*m/5;delta_t=-grad_t*m/2000
   delta_r*=np.minimum(1,np.deg2rad(3)/np.maximum(np.linalg.norm(delta_r,axis=1),1e-12))[:,None]
   delta_t*=np.minimum(1,.01/np.maximum(np.linalg.norm(delta_t,axis=1),1e-12))[:,None]
   accepted=False;old=terms.copy();global_before=score(obs,r,t)
   for trial in range(7):
    factor=.5**trial;nr,nt,no=cr.copy(),ct.copy(),co.copy()
    nr[core]=Rotation.from_rotvec(delta_r*factor).as_matrix()@cr[core];nt[core]+=delta_t*factor
    no[core],_=observe(nr[core],nt[core],ids);candidate=score(no,nr,nt)
    if accepts(old,candidate,no,best_silhouette[lo:hi]):
     cr,ct,co,terms=nr,nt,no,candidate;accepted=True;break
   r[lo:hi],t[lo:hi],obs[lo:hi]=cr,ct,co
   best_silhouette[lo:hi]=np.minimum(best_silhouette[lo:hi],co[:,0])
   global_after=score(obs,r,t)
   expected=global_before['total']+(terms['total']-old['total'])*m/n
   assert abs(global_after['total']-expected)<1e-8 and global_after['total']<=global_before['total']+1e-10
   assert global_after['object_silhouette']<=global_before['object_silhouette']+1e-10
   history.append({'window':window_index,'start':int(frames[start]),'stop_exclusive':int(frames[stop-1]+1),'step':step+1,'accepted':accepted,'factor':factor if accepted else 0,'local_before':old,'local_after':terms.copy(),'global_before':global_before,'global_after':global_after})
   if not accepted:break
  record={'window':window_index,'start':start,'stop':stop,'seed':seed_name,'seed_scores':seed_scores,'local_before':window_before,'local_after':terms['total'],'global':score(obs,r,t),'elapsed_seconds':time.monotonic()-started}
  windows.append(record);print('WINDOW',json.dumps(record),flush=True)
  (a.out/'history.json').write_text(json.dumps(history,indent=2));(a.out/'windows.json').write_text(json.dumps(windows,indent=2))
  np.savez_compressed(a.out/'tracking_checkpoint.npz',frames=frames,rotation=r,translation=t,observations=obs,best_silhouette=best_silhouette)
 final_ob,_=observe(r,t);after=score(final_ob,r,t);cached=score(obs,r,t)
 np.testing.assert_allclose(final_ob,obs,atol=2e-6,rtol=1e-5)
 rollback=False
 if not np.isfinite(after['total']) or after['total']>seed_terms['total']+2e-6 or after['object_silhouette']>seed_terms['object_silhouette']+2e-6 or np.any(final_ob[:,0]>best_silhouette+MAX_FRAME_IOU_DROP+2e-6):
  r,t,final_ob=seed_r,seed_t,seed_ob;after=seed_terms;rollback=True
 result=data.copy();result.update(rotation=r,translation=t);np.savez_compressed(a.out/'optimized.npz',**result)
 (a.out/'per_frame_observations.json').write_text(json.dumps({'frames':frames.tolist(),'baseline':baseline_ob.tolist(),'previous_tracker':alternate_ob.tolist(),'recovered_seed':seed_ob.tolist(),'optimized':final_ob.tolist(),'order':['silhouette','color']},indent=2))
 # Stream images; avoid retaining full-clip floating-point renders in host RAM.
 import imageio.v2 as imageio
 sampled=[]
 with imageio.get_writer(str(a.out/'comparison.mp4'),fps=fps,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18']) as writer:
  for i,frame in enumerate(frames):
   _,renders=observe(np.stack([alt_r[i],r[i]]),np.stack([alt_t[i],t[i]]),np.array([i,i]),return_images=True)
   # Black background makes a missing prediction explicit; raw RGB underneath
   # an overlay could otherwise be mistaken for a successfully rendered bowl.
   def mask_panel(render):
    panel=np.zeros((h,w,3),dtype=np.float32);panel[...,1]=masks[i,...,1]
    panel[...,0]=render[...,1];panel[...,2]=render[...,1]
    return np.uint8(np.clip(panel*255,0,255))
   panels=[images[i],mask_panel(renders[0]),mask_panel(renders[1])]
   for j,label in enumerate(['RGB',f'v1 IoU {1-alternate_ob[i,0]:.3f}',f'IoU-first {1-final_ob[i,0]:.3f}']):
    panels[j]=cv2.copyMakeBorder(panels[j],32,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(panels[j],f'{label} | {frame}',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
   strip=np.concatenate(panels,1);writer.append_data(strip)
   if i in [0,n//2,3*n//4,n-1]:sampled.append(strip)
 cv2.imwrite(str(a.out/'comparison.png'),cv2.cvtColor(np.concatenate(sampled),cv2.COLOR_RGB2BGR))
 def overlap_summary(ob):
  iou=1-ob[:,0];tail=min(100,n)
  return {'mean_iou':float(iou.mean()),'median_iou':float(np.median(iou)),
   'frames_iou_below_0_1':int((iou<.1).sum()),'tail_frame_count':tail,'tail_mean_iou':float(iou[-tail:].mean()),'last_frame_iou':float(iou[-1])}
 report={'status':'complete','frames':n,'baseline':before,'previous_tracker':alt_terms,'recovered_seed':seed_terms,'after':after,'initial_choice':initial_choice,'rolled_back_to_recovered_seed':rollback,'recovery':recovery,'overlap':{'v1':overlap_summary(alternate_ob),'iou_priority':overlap_summary(final_ob)},'cached_final':cached,'final_rerender_check':True,'accepted_updates':sum(x['accepted'] for x in history),'windows':len(windows),'elapsed_seconds':time.monotonic()-started,'render_counts':counts,'reference_inputs':False,'scope':'Object R/t only; fixed human and scale; no SyILEN rollout; 2D overlap improvement does not imply 3D or Kaggle improvement'}
 if a.resume_from:report.update(resumed_from=str(a.resume_from),prior_completed_windows=completed_windows,render_counts_scope='this continuation only')
 (a.out/'topology_cache_runtime.json').write_text(json.dumps(topology_cache.report(),indent=2))
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()
