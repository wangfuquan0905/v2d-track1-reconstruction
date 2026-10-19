"""Forward beam tracking with projected geometry pruning and coarse-to-fine renders.
No GT; no speed penalties; no rotation autograd; fixed human/mesh/shared scale.
"""
import argparse,json,pickle,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import refine_render_observation  # Runtime library paths.
import cv2,h5py,numpy as np,torch,trimesh
from scipy.spatial.transform import Rotation
from v2d_syilen.render_observation import ObservationRenderer,observation_terms
from v2d_syilen.pruned_tracking import predict_pose,rotation_candidates,acceleration_cost,geometry_keep


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--bundle',type=Path,help='Matching canonical human mesh topology bundle')
 p.add_argument('--start',type=int,default=0,help='First source frame ID, inclusive')
 p.add_argument('--end',type=int,default=359);p.add_argument('--beam',type=int,default=3);p.add_argument('--batch',type=int,default=4)
 p.add_argument('--angular-acceleration-weight',type=float,default=1.)
 p.add_argument('--linear-acceleration-weight',type=float,default=1.)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);started=time.monotonic()
 if any(not np.isfinite(x) or x<0 for x in [a.angular_acceleration_weight,a.linear_acceleration_weight]):p.error('Acceleration weights must be finite and nonnegative')
 cfg=json.loads((a.source/'config.json').read_text());episode=cfg['episode'];root=ROOT/'outputs'/episode;run=root/'cari4d'/episode
 loaded=np.load(a.source/'optimized.npz');data={key:loaded[key] for key in loaded.files};fps=cfg['fps']
 source_frames=data['frames'];selected=np.flatnonzero((source_frames>=a.start)&(source_frames<=a.end))
 if len(selected)<3 or not np.array_equal(source_frames[selected],np.arange(a.start,a.end+1)):
  p.error('Requested inclusive interval must contain at least three contiguous source frames')
 for key in ['frames','rotation','translation','vertices','joints','pose']:
  data[key]=data[key][selected]
 frame_ids=data['frames'];n=len(frame_ids)
 tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
 bundle_path=a.bundle or Path(cfg.get('bundle_path',ROOT/'outputs/episode_000016/render_observation_v1/canonical_baseline/canonical.pth'))
 bundle=torch.load(bundle_path,map_location='cpu',weights_only=False)
 mesh=trimesh.load(run/f'export/{episode}/object_mesh/output_aligned.glb',force='mesh',process=False)
 vertices=np.asarray(mesh.vertices)*float(data['object_scale']);v=tensor(vertices)
 corners=np.array(np.meshgrid(*zip(vertices.min(0),vertices.max(0)),indexing='ij')).reshape(3,-1).T
 with open(run/'depth/moge2/intrinsics.pkl','rb') as f:k=pickle.load(f)
 x0,y0,x1,y1=cfg['crop_xyxy'];h,w=cfg['render_size'];K=np.array([k['fx']*w/(x1-x0),k['fy']*h/(y1-y0),(k['cx']-x0)*w/(x1-x0),(k['cy']-y0)*h/(y1-y0)])
 hf=torch.tensor(bundle['faces'],device='cuda',dtype=torch.int32);of=torch.tensor(mesh.faces,device='cuda',dtype=torch.int32);colors=tensor(mesh.visual.vertex_colors[:,:3]/255.)
 full=ObservationRenderer(hf,of,colors,tensor(K),h,w,supersample=4)
 coarse=ObservationRenderer(hf,of,colors,tensor(K/2),h//2,w//2,supersample=1)
 images=[];masks=[];boxes=[]
 with h5py.File(root/'segmentation/masks.h5') as f:
  for frame in frame_ids:
   im=cv2.cvtColor(cv2.imread(str(root/f'segmentation/frames/{frame:06d}.jpg')),cv2.COLOR_BGR2RGB)
   images.append(cv2.resize(im[y0:y1,x0:x1],(w,h)))
   m=np.stack([f[f'{episode}/{frame:06d}-k0.person_mask.png'][()],f[f'{episode}/{frame:06d}-k0.obj_rend_mask.png'][()]],-1).astype('float32')
   m=cv2.resize(m[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_NEAREST);masks.append(m)
   cc,labels,stats,_=cv2.connectedComponentsWithStats((m[...,1]>0).astype('uint8'))
   if cc>1:
    x,y,bw,bh=stats[1+stats[1:,cv2.CC_STAT_AREA].argmax(),:4];boxes.append([x,y,x+bw,y+bh])
   else: boxes.append([0,0,w,h])
 images=np.array(images);masks=np.array(masks);counts={'coarse':0,'full':0,'geometry_rejected':0,'expansions':0}
 config={'source':str(a.source),'episode':episode,'frames':n,'beam':a.beam,'first_frame':'previous corrected frame zero; no later object poses used as tracking inputs','fixed':['human vertices/parameters from source','shared object scale','mesh/colors/camera'], 'loss_weights':{'object_color':20,'object_silhouette':10,'angular_acceleration':1,'linear_acceleration':1,'angular_velocity':0,'linear_velocity':0},'normalization':{'angular_acceleration_deg_s2':360,'linear_acceleration_m_s2':1},'coarse_candidates':32,'fine_survivors':8,'fps':fps,'reference_inputs':False,'gradient_method':'actual candidate renders; no object image autograd','recovery':'silhouette loss > .55 or best local candidate on search boundary; larger rotation proposals','limitations':['Heuristic pruning, not guaranteed global optimum','Beam may merge similar paths','Fixed appearance/occlusion can bias RGB fit','Acceleration-only causal tracking can drift'],'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'v2d_syilen/pruned_tracking.py']}}
 config['loss_weights'].update(angular_acceleration=a.angular_acceleration_weight,linear_acceleration=a.linear_acceleration_weight)
 config['first_frame']=f'Source frame {int(frame_ids[0])} followed by image-only local registration before motion prediction; no earlier tracking history'
 config['frame_range_inclusive']=[int(frame_ids[0]),int(frame_ids[-1])]
 config['render_size']=cfg['render_size'];config['crop_xyxy']=cfg['crop_xyxy']
 config['bundle_path']=str(bundle_path)
 config['comparison_label']=cfg.get('comparison_label','Previous v8')
 config['candidate_retention']='Unrefined alternatives plus refined top two; endpoint diversity across beam'
 config['translation_proposals']='Constant-velocity prediction, local offsets, and observed mask-box displacement'
 (a.out/'config.json').write_text(json.dumps(config,indent=2))
 @torch.no_grad()
 def render_scores(frame,rs,ts,low=False,return_render=False):
  renderer=coarse if low else full;hh,ww=(h//2,w//2) if low else (h,w)
  rgb=cv2.resize(images[frame],(ww,hh))/255.;mask=cv2.resize(masks[frame],(ww,hh),interpolation=cv2.INTER_NEAREST)
  result=[];renders=[]
  for start in range(0,len(rs),a.batch):
   end=min(len(rs),start+a.batch);batch=end-start
   obj=torch.einsum('bij,vj->bvi',tensor(rs[start:end]),v)+tensor(ts[start:end,None])
   rendered=renderer(tensor(data['vertices'][frame:frame+1]).expand(batch,-1,-1).contiguous(),obj)
   terms=observation_terms(rendered,tensor(rgb)[None].expand(batch,-1,-1,-1),tensor(mask)[None].expand(batch,-1,-1,-1))
   result.append(np.stack([terms['object_silhouette'].cpu().numpy(),terms['object_color'].cpu().numpy()],-1))
   if return_render:renders.append(rendered.cpu().numpy())
  counts['coarse' if low else 'full']+=len(rs)
  return np.concatenate(result),np.concatenate(renders) if return_render else None
 def objective(state,rs,ts,observations):
  return 10*observations[:,0]+20*observations[:,1]+temporal_cost(state,rs,ts)
 def temporal_cost(state,rs,ts):
  return acceleration_cost(state['r'],state['t'],rs,ts,fps,angular_weight=a.angular_acceleration_weight,linear_weight=a.linear_acceleration_weight)
 first_r=Rotation.from_matrix(data['rotation'][0].astype(float)).as_matrix();first_t=data['translation'][0].astype(float)
 # Fit the first image before estimating motion: a constant first-frame pose
 # error must not be interpreted as an initial angular velocity.
 first_alignment=[]
 for deg,metres in [(2.,.002),(.5,.0005),(.1,.0001)]:
  for iteration in range(12):
   dirs=np.concatenate([np.zeros((1,3)),np.eye(3),-np.eye(3)])
   rs=Rotation.from_rotvec(dirs*np.deg2rad(deg)).as_matrix()@first_r;ts=np.repeat(first_t[None],7,0)
   ob,_=render_scores(0,rs,ts);scores=10*ob[:,0]+20*ob[:,1];j=scores.argmin();changed=int(j)!=0;first_r=rs[j]
   ts=first_t+dirs*metres;rs=np.repeat(first_r[None],7,0)
   ob,_=render_scores(0,rs,ts);scores=10*ob[:,0]+20*ob[:,1];j=scores.argmin();changed=changed or int(j)!=0;first_t=ts[j]
   first_alignment.append(float(scores[j]))
   if not changed:break
 (a.out/'first_frame_alignment.json').write_text(json.dumps({'loss_history':first_alignment,'purpose':'Remove static alignment residual before inferring inter-frame motion'},indent=2))
 score,_=render_scores(0,first_r[None],first_t[None]);first_loss=float(10*score[0,0]+20*score[0,1])
 beams=[{'r':[first_r],'t':[first_t],'cost':first_loss}];history=[]
 for frame in range(1,n):
  next_states=[];frame_expansions=0;frame_rejected=0
  for state in beams:
   predicted_r,predicted_t=predict_pose(state['r'],state['t'])
   # Refine translation near the constant-velocity prediction BEFORE orientation scoring.
   translation=predicted_t.copy()
   for step in [.002,.0005,.00005]:
    ts=translation+np.concatenate([np.zeros((1,3)),np.eye(3)*step,-np.eye(3)*step])
    oldbox,newbox=np.array(boxes[frame-1]),np.array(boxes[frame]);duv=(newbox[:2]+newbox[2:]-oldbox[:2]-oldbox[2:])/2
    image_t=state['t'][-1].copy();image_t[:2]+=duv*image_t[2]/K[:2]
    ts=np.concatenate([ts,image_t[None]])
    rs=np.repeat(predicted_r[None],len(ts),0);ob,_=render_scores(frame,rs,ts)
    translation=ts[np.argmin(objective(state,rs,ts,ob))]
   candidates=[]
   for expanded in [False,True]:
    if expanded:
     best=min(candidates,key=lambda x:x[0]);on_edge=best[4]>=((.5 if frame>1 else 2.)*.95)
     if best[3][0]<=.55 and not on_edge:break
     counts['expansions']+=1;frame_expansions+=1
    center=state['r'][-1] if expanded else predicted_r
    rs,radii=rotation_candidates(center,state['r'][-1],fine=frame>1,expansion=expanded)
    ts=np.repeat(translation[None],len(rs),0)
    keep=geometry_keep(rs,ts,corners,K,boxes[frame],(h,w));keep[0]=True;keep[-1]=True
    counts['geometry_rejected']+=int((~keep).sum());frame_rejected+=int((~keep).sum())
    rs,ts,radii=rs[keep],ts[keep],radii[keep]
    ob,_=render_scores(frame,rs,ts,low=True)
    temporal=temporal_cost(state,rs,ts)
    # Retain two image-only branches so temporal ranking does not discard every alternative.
    selected=list(dict.fromkeys([*np.argsort(10*ob[:,0]+temporal)[:5],*np.argsort(ob[:,0])[:2],len(rs)-1]))
    ob,_=render_scores(frame,rs[selected],ts[selected])
    totals=objective(state,rs[selected],ts[selected],ob)
    for j,index in enumerate(selected):candidates.append((float(totals[j]),rs[index],ts[index],ob[j],radii[index]))
   # Retain unrefined alternatives too, rather than collapsing every branch
   # to the same local optimum before future frames can disambiguate it.
   for cost,r,t,ob,_ in candidates:
    next_states.append({'r':state['r']+[r],'t':state['t']+[t],'cost':state['cost']+cost,'ob':ob.tolist()})
   # Two local refinements per parent: rotation AND translation use actual loss changes.
   for candidate in sorted(candidates,key=lambda x:x[0])[:2]:
    cost,r,t,ob,_=candidate
    for degrees in [.10,.025,.005]:
     updates=np.concatenate([np.zeros((1,3)),np.eye(3)*np.deg2rad(degrees),-np.eye(3)*np.deg2rad(degrees)])
     rs=Rotation.from_rotvec(updates).as_matrix()@r;ts=np.repeat(t[None],len(rs),0)
     obs,_=render_scores(frame,rs,ts);scores=objective(state,rs,ts,obs);j=scores.argmin();cost,r,t,ob=float(scores[j]),rs[j],ts[j],obs[j]
    ts=t+np.concatenate([np.zeros((1,3)),np.eye(3)*.00025,-np.eye(3)*.00025]);rs=np.repeat(r[None],len(ts),0)
    obs,_=render_scores(frame,rs,ts);scores=objective(state,rs,ts,obs);j=scores.argmin();cost,r,t,ob=float(scores[j]),rs[j],ts[j],obs[j]
    next_states.append({'r':state['r']+[r],'t':state['t']+[t],'cost':state['cost']+cost,'ob':ob.tolist()})
  # Keep distinct endpoints; otherwise identical candidates consume the entire beam.
  beams=[]
  for state in sorted(next_states,key=lambda x:x['cost']):
   duplicate=any(Rotation.from_matrix(state['r'][-1]@b['r'][-1].T).magnitude()<np.deg2rad(.25) and np.linalg.norm(state['t'][-1]-b['t'][-1])<.0002 for b in beams)
   if not duplicate:beams.append(state)
   if len(beams)>=a.beam:break
  if not beams:raise RuntimeError('No valid tracking hypothesis')
  best=beams[0];rotation_step=np.rad2deg(Rotation.from_matrix(best['r'][-1]@best['r'][-2].T).magnitude())
  row={'frame':int(frame_ids[frame]),'local_frame':frame,'beam_size':len(beams),'best_cumulative_cost':best['cost'],'object_silhouette_loss':best['ob'][0],'object_color_loss':best['ob'][1],'rotation_step_deg':float(rotation_step),'expansions':frame_expansions,'geometry_rejected':frame_rejected,'elapsed_seconds':time.monotonic()-started};history.append(row)
  if frame%10==0 or frame==n-1:
   print('TRACK',json.dumps(row),flush=True);(a.out/'history.json').write_text(json.dumps(history,indent=2))
   np.savez_compressed(a.out/'tracking_checkpoint.npz',frames=frame_ids[:frame+1],rotation=np.array(best['r']),translation=np.array(best['t']))
 best=min(beams,key=lambda s:s['cost']);rotations=np.array(best['r']);translations=np.array(best['t'])
 result={key:(value[:n] if key in ['frames','rotation','translation','vertices','joints','pose'] else value) for key,value in data.items()};result.update(rotation=rotations,translation=translations)
 np.savez_compressed(a.out/'optimized.npz',**result)
 import imageio.v2 as imageio
 sampled=[];all_ob=[];prior_ob=[]
 def overlay(im,r):
  alpha=r[...,:2].sum(-1).clip(0,1);color=r[...,0:1]*np.array([.68,.78,.88])+r[...,2:]
  return np.uint8(np.clip(im*(1-.7*alpha[...,None])+.7*color*255,0,255))
 with imageio.get_writer(str(a.out/'comparison.mp4'),fps=fps,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18']) as writer:
  for frame in range(n):
   rs=np.stack([data['rotation'][frame],rotations[frame]]);ts=np.stack([data['translation'][frame],translations[frame]])
   ob,renders=render_scores(frame,rs,ts,return_render=True);prior_ob.append(ob[0]);all_ob.append(ob[1])
   panels=[images[frame],overlay(images[frame],renders[0]),overlay(images[frame],renders[1])]
   for j,label in enumerate(['RGB',config['comparison_label'],'Pruned tracking']):
    panels[j]=cv2.copyMakeBorder(panels[j],32,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(panels[j],f'{label} | {frame_ids[frame]}',(6,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
   strip=np.concatenate(panels,1);writer.append_data(strip)
   if frame in [0,n//4,n//2,3*n//4,n-1]:sampled.append(strip)
 cv2.imwrite(str(a.out/'comparison.png'),cv2.cvtColor(np.concatenate(sampled),cv2.COLOR_RGB2BGR))
 def motion(r,t):
  increments=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec();rel=Rotation.from_matrix(r@r[0].T).magnitude()
  return {'rotation_path_length_deg':float(np.rad2deg(np.linalg.norm(increments,axis=1)).sum()),'max_orientation_difference_deg':float(np.rad2deg(rel).max()),'max_rotation_step_deg':float(np.rad2deg(np.linalg.norm(increments,axis=1)).max()),'summed_camera_rotation_components_deg':np.rad2deg(increments.sum(0)).tolist(),'angular_acceleration_rms_deg_s2':float(np.rad2deg(np.sqrt(np.mean(np.sum((np.diff(increments,axis=0)*fps**2)**2,axis=1))))),'linear_acceleration_rms_m_s2':float(np.sqrt(np.mean(np.sum((np.diff(t,n=2,axis=0)*fps**2)**2,axis=1))))}
 report={'status':'complete','frames':n,'elapsed_seconds':time.monotonic()-started,'render_counts':counts,'previous':{'mean_observation_terms':np.mean(prior_ob,0).tolist(),**motion(data['rotation'][:n].astype(float),data['translation'][:n].astype(float))},'tracked':{'mean_observation_terms':np.mean(all_ob,0).tolist(),**motion(rotations,translations)},'observation_order':['object_silhouette_loss','object_color_loss'],'ground_truth_used':False,'rotation_path_note':'Path length includes reversals and wobble; not proof of 270-degree upright rotation','scope':'Direct rigid-pose tracking; no SyILEN dynamics rollout; fixed source human and scale'}
 report['frame_range_inclusive']=[int(frame_ids[0]),int(frame_ids[-1])]
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
