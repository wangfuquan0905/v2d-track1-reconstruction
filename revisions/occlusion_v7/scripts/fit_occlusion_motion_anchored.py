"""V7: causal occlusion prediction and hard motion gates before image fitting.

Separate from the frozen v3 producer. Reference data are never loaded here.
"""
import argparse,json,hashlib,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from occlusion_scene import Scene
import numpy as np,torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix
from v2d_syilen.occlusion_objective import (interpolate_anchors,prior_weights,score,
    second_order_path,temporal_residuals,ANGULAR_WEIGHT,LINEAR_WEIGHT,COLOR_WEIGHT)
from v2d_syilen.iou_priority import mask_box_translation
from v2d_syilen.window_search import context_bounds
from v2d_syilen.rotation_temporal import angular_derivatives
from v2d_syilen.occlusion_candidates import (bounded_anchor_reference, pose_candidates,
    evidence_limits, evidence_feasible, constrain_unary, require_available_overlap)
from v2d_syilen.occlusion_motion import MotionConfig, MotionState, candidate_gate, choose_local, trajectory_feasible

from v2d_syilen.occlusion_motion_anchor import MotionConfig, local_depth_anchor, choose_anchored

MOTION=MotionConfig()


def save_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def observation_cost(ob,confidence):
    ob=np.asarray(ob,dtype=np.float64)
    return np.asarray(confidence,dtype=np.float64)*(ob[...,0]+COLOR_WEIGHT*ob[...,1])


def build_candidates(scene, confidence, states, source_r, source_t, v1, v2):
    """Causal, locally rendered candidates. No future pose/anchor access."""
    v1_r,v1_t=scene.poses(v1);v2_r,v2_t=scene.poses(v2)
    tracker=MotionState(scene.fps,MOTION)
    rr=[];tt=[];oo=[];records=[];locks=[]
    for i in range(scene.n):
        observed=bool(scene.target_area[i]>0)
        pr,pt,mode=tracker.predict(source_r[i],source_t[i],not observed)
        previous_r=rr[-1] if rr else None;previous_t=tt[-1] if tt else None
        locked=bool(states[i]!='clear' or tracker.reacquiring)
        locks.append(locked)
        anchor_z=local_depth_anchor(source_t[i],v1_t[i],pt,not locked,MOTION)
        ar=np.stack([pr,source_r[i],v1_r[i],v2_r[i]])
        at=np.stack([pt,source_t[i],v1_t[i],v2_t[i]])
        gate=lambda r,t:candidate_gate(r,t,pr,pt,previous_r,previous_t,scene.fps,locked,MOTION)
        # Keep original R,t pairs in the diagnostic candidate list; exclude
        # physically inconsistent ones before any silhouette comparison.
        allowed=gate(ar,at);initial_allowed=allowed.copy()
        if not observed:
            r,t=pr,pt;ob=np.zeros(2);unresolved=False;selected=0
        else:
            # Partial masks may supply lateral/rotation evidence, never depth.
            if locked:
                projected=at[1:].copy();projected[:,2]=pt[2]
                ar=np.concatenate([ar,ar[1:]]);at=np.concatenate([at,projected])
            if anchor_z is not None:
                guide=pt.copy();guide[2]+=np.clip(anchor_z-pt[2],-.03,.03)
                ar=np.concatenate([ar,[pr]]);at=np.concatenate([at,[guide]])
            allowed=gate(ar,at)
            ob_all=scene.observe(ar,at,np.full(len(at),i))
            selected,unresolved=choose_anchored(ob_all,allowed,at,anchor_z,minimum_fit_iou=MOTION.minimum_fit_iou,config=MOTION)
            r,t,ob=ar[selected],at[selected],ob_all[selected]
            for offset,angle in [(.02,3.),(.007,1.)]:
                xy=np.array([(x,y,0.) for x in [-offset,0.,offset] for y in [-offset,0.,offset]])
                if not locked:xy=np.concatenate([xy,[[0,0,-offset],[0,0,offset]]])
                rotvec=np.concatenate([np.eye(3),-np.eye(3)])*np.deg2rad(angle)
                local_r=np.concatenate([np.repeat(r[None],len(xy),axis=0),Rotation.from_rotvec(rotvec).as_matrix()@r])
                local_t=np.concatenate([t+xy,np.repeat(t[None],len(rotvec),axis=0)])
                ar=np.concatenate([[pr],[r],local_r]);at=np.concatenate([[pt],[t],local_t])
                allowed=gate(ar,at)
                # Render only feasible local poses; rejected poses cannot set
                # an image floor or drag the missing interval toward them.
                ob_all=np.full((len(at),2),np.inf)
                ob_all[allowed]=scene.observe(ar[allowed],at[allowed],np.full(allowed.sum(),i))
                selected,unresolved=choose_anchored(ob_all,allowed,at,anchor_z,minimum_fit_iou=MOTION.minimum_fit_iou,config=MOTION)
                r,t,ob=ar[selected],at[selected],ob_all[selected]
        iou=float(1-ob[0]) if observed else 0.
        consistent=bool(observed and not unresolved and iou>=MOTION.velocity_iou)
        trusted=tracker.commit(r,t,float(confidence[i]),iou,observed,consistent)
        rr.append(r.copy());tt.append(t.copy());oo.append(ob.copy())
        records.append({'frame':int(scene.frames[i]),'prediction':mode,'depth_locked':locked,
            'local_clear_depth_anchor_m':anchor_z,'trusted_velocity_sample':trusted,'reacquisition_pending':tracker.reacquiring,
            'reacquisition_consistent_count':tracker.reacquire_count,
            'input_candidates_motion_allowed':dict(zip(['prediction','original','v1','v2'],initial_allowed.tolist())),
            'unresolved_visible_evidence':unresolved,'censored_iou':iou if observed else None,
            'prediction_translation_m':pt.tolist(),'selected_translation_m':t.tolist()})
        if i%50==0 or i==scene.n-1:
            print('SEED',json.dumps({'frame':i,'state':str(states[i]),'mode':mode,'z':float(t[2]),'iou':iou,'unresolved':unresolved}),flush=True)
    r,t,ob=np.asarray(rr),np.asarray(tt),np.asarray(oo)
    assert trajectory_feasible(r,t,r,t,states,scene.fps,MOTION,np.asarray(locks))
    return r,t,ob,np.asarray(locks),records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','refine','finish'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--v2',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--window',type=int,default=50)
    p.add_argument('--stride',type=int,default=25);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--max-windows',type=int,default=1000)
    p.add_argument('--max-anchor-gap-seconds',type=float,default=2.)
    a=p.parse_args();started=time.monotonic()
    if a.stage=='prepare' and a.out.exists() and any(a.out.iterdir()):p.error('Prepare requires a new output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    scene=Scene(a.source,a.batch);n=scene.n;fps=scene.fps;frames=scene.frames
    if a.stage=='prepare':
        source_r,source_t=scene.poses(a.source)
        pred_t=gaussian_filter1d(source_t,2,axis=0,mode='nearest')
        pred_r=np.stack([Rotation.from_matrix(source_r[max(0,i-3):min(n,i+4)]).mean().as_matrix() for i in range(n)])
        confidence,states,observed,geometric=scene.visibility(pred_r,pred_t)
        config={**scene.cfg,'algorithm':'occlusion_motion_gated_v7','source':str(a.source),
                'v1':str(a.v1),'v2':str(a.v2),'window':a.window,'stride':a.stride,'steps':a.steps,
                'motion':MOTION.dictionary(),'reference_inputs':False,
                'clear_depth_anchor':'Only agreeing original/v1 current-frame positions within 0.2 m of prediction; soft depth reference prevents slow drift. Never enabled by partial masks.',
                'initialization':'Original first pose retained. Causal local search; no future anchor or neighboring-frame alternatives.',
                'missing':'No image update. Short robust constant velocity only from consecutive reliable observations, otherwise hold. Frozen through refinement.',
                'partial':'All visible positive pixels enter the censored objective. Lateral/rotation updates only; depth comes from motion prediction, never fragment size.',
                'reacquisition':'At least 3 consistent local observations; out-of-gate poses cannot impose an IoU floor. Unexplained evidence remains explicit.',
                'acceptance':'Hard speed and angular speed bounds across all adjacent pairs including window seams. Missing pose and partial depth locked. Refine within 1 cm of causal seed; local evidence budget 0.05.',
                'coordinate_frame':'input camera; assumes modest camera motion, no world-camera compensation',
                'fixed':['human geometry','object scale/mesh/colors','camera'],
                'input_sha256':{str(d/'optimized.npz'):hashlib.sha256((d/'optimized.npz').read_bytes()).hexdigest() for d in [a.source,a.v1,a.v2]}}
        sources=[Path(__file__),ROOT/'v2d_syilen/occlusion_motion.py',ROOT/'v2d_syilen/occlusion_motion_anchor.py',ROOT/'scripts/occlusion_scene.py',
                 ROOT/'v2d_syilen/occlusion_objective.py',ROOT/'v2d_syilen/occlusion_candidates.py',
                 ROOT/'v2d_syilen/render_observation.py',ROOT/'v2d_syilen/render_topology_cache.py',
                 ROOT/'v2d_syilen/rotation_temporal.py',ROOT/'v2d_syilen/window_search.py',ROOT/'v2d_syilen/iou_priority.py']
        config['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
        save_json(a.out/'config.json',config)
        snapshot=a.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for f in sources:(snapshot/f.name).write_bytes(f.read_bytes())
        save_json(a.out/'visibility.json',{'frames':frames.tolist(),'states':states.tolist(),'confidence':confidence.tolist(),
            'target_area':scene.target_area.tolist(),'observed_person_coverage':observed.tolist(),
            'predicted_geometric_occlusion':geometric.tolist(),'anchor_prior_supported':[False]*n})
        print('VISIBILITY',json.dumps({state:int(np.sum(states==state)) for state in ['clear','partial','missing']}),flush=True)
        r,t,obs,depth_locked,decisions=build_candidates(scene,confidence,states,source_r,source_t,a.v1,a.v2)
        # The causal path is the local trust region, not a distant pose anchor.
        ref_r,ref_t=r.copy(),t.copy();prior=np.zeros((n,3));supported=np.zeros(n,bool)
        for i,decision in enumerate(decisions):
            if decision['local_clear_depth_anchor_m'] is not None:
                ref_t[i,2]=decision['local_clear_depth_anchor_m']
                prior[i,2]=MOTION.anchor_weight/MOTION.anchor_sigma_m**2
                supported[i]=True
        vis=json.loads((a.out/'visibility.json').read_text());vis['anchor_prior_supported']=supported.tolist()
        save_json(a.out/'visibility.json',vis)
        limits=np.where(scene.target_area>0,np.minimum(1.,obs[:,0]+.05),np.inf)
        seed_score=score(obs,r,t,fps,confidence,ref_t,prior)
        np.savez_compressed(a.out/'seed.npz',frames=frames,rotation=r,translation=t,observations=obs,
            confidence=confidence,states=states,reference_rotation=ref_r,reference_translation=ref_t,prior=prior,
            depth_locked=depth_locked,evidence_limits=limits,anchor_prior_supported=supported)
        save_json(a.out/'motion_decisions.json',decisions)
        save_json(a.out/'prepare_report.json',{'status':'complete','score':seed_score,'counts':scene.counts,
                    'max_step_m':float(np.linalg.norm(np.diff(t,axis=0),axis=1).max()),
                    'unresolved_visible_frames':[x['frame'] for x in decisions if x['unresolved_visible_evidence']],
                    'elapsed_seconds':time.monotonic()-started})
        print('PREPARED',json.dumps(seed_score),flush=True);return
    config=json.loads((a.out/'config.json').read_text())
    for path,digest in config['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Producer changed: {path}')
    for path,digest in config['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Input changed: {path}')
    seed=np.load(a.out/'seed.npz');confidence=seed['confidence'];states=seed['states'];prior=seed['prior'];ref_t=seed['reference_translation']
    observed_frames=scene.target_area>0;seed_ob=seed['observations']
    limits=np.minimum(seed['evidence_limits'],np.where(observed_frames,seed_ob[:,0]+.05,np.inf))
    r,t,obs=seed['rotation'].copy(),seed['translation'].copy(),seed_ob.copy()
    history=[];windows=[]
    if (a.out/'checkpoint.npz').exists():
        ck=np.load(a.out/'checkpoint.npz');r,t,obs=ck['rotation'].copy(),ck['translation'].copy(),ck['observations'].copy()
        windows=json.loads((a.out/'windows.json').read_text());history=json.loads((a.out/'history.json').read_text())
        assert len(windows)==int(ck['completed_windows'])
    starts=list(range(0,max(n-config['window']+1,1),config['stride']))
    if starts[-1]!=max(0,n-config['window']):starts.append(max(0,n-config['window']))
    full_score=lambda ob,r,t:score(ob,r,t,fps,confidence,ref_t,prior)
    if a.stage=='refine':
        rerender=scene.observe(r,t);np.testing.assert_allclose(rerender,obs,atol=2e-6,rtol=1e-5)
        completed=len(windows)
        for wi in range(completed,min(len(starts),completed+a.max_windows)):
            start=starts[wi];stop=min(start+config['window'],n);lo,hi=context_bounds(start,stop,n)
            ids=np.arange(start,stop);core=slice(start-lo,stop-lo);m=hi-lo
            cr,ct,co=r[lo:hi].copy(),t[lo:hi].copy(),obs[lo:hi].copy()
            local=lambda ob,r,t:score(ob,r,t,fps,confidence[lo:hi],ref_t[lo:hi],prior[lo:hi])
            terms=local(co,cr,ct)
            for step in range(config['steps']):
                gr=[];gt=[];eps_r=np.deg2rad(.25);eps_t=.001;k=len(ids)
                for axis in range(3):
                    dr=np.eye(3)[axis]*eps_r;dt=np.eye(3)[axis]*eps_t
                    rr=np.concatenate([Rotation.from_rotvec(dr).as_matrix()@cr[core],Rotation.from_rotvec(-dr).as_matrix()@cr[core]])
                    ob=scene.observe(rr,np.concatenate([ct[core],ct[core]]),np.tile(ids,2))
                    loss=observation_cost(ob,np.tile(confidence[ids],2))
                    gr.append((loss[:k]-loss[k:])/(2*eps_r*m))
                    ob=scene.observe(np.concatenate([cr[core],cr[core]]),np.concatenate([ct[core]+dt,ct[core]-dt]),np.tile(ids,2))
                    loss=observation_cost(ob,np.tile(confidence[ids],2))
                    gt.append((loss[:k]-loss[k:])/(2*eps_t*m))
                rr=torch.tensor(cr,device='cuda',dtype=torch.float64)
                tt=torch.tensor(ct,device='cuda',dtype=torch.float64,requires_grad=True)
                q=torch.zeros(m,3,device='cuda',dtype=torch.float64,requires_grad=True)
                _,alpha=angular_derivatives(axis_angle_to_matrix(q)@rr,fps)
                lin=(tt[2:]-2*tt[1:-1]+tt[:-2])*fps**2
                temporal=(ANGULAR_WEIGHT*(torch.sqrt(1+(alpha/(2*np.pi)).square().sum(-1))-1).sum()+
                          LINEAR_WEIGHT*(torch.sqrt(1+lin.square().sum(-1))-1).sum())/m
                regularization=temporal+(torch.as_tensor(prior[lo:hi],device='cuda')*(tt-torch.as_tensor(ref_t[lo:hi],device='cuda')).square()).sum()/m
                temp_r,temp_t=torch.autograd.grad(regularization,[q,tt])
                np.testing.assert_allclose(float(regularization.detach()),terms['angular']+terms['linear']+terms['occlusion_prior'],atol=1e-7)
                grad_r=np.stack(gr,-1)+temp_r.cpu().numpy()[core];grad_t=np.stack(gt,-1)+temp_t.cpu().numpy()[core]
                ang,lin_np=temporal_residuals(cr,ct,fps);D=np.diff(np.eye(m),n=2,axis=0)
                ar=ANGULAR_WEIGHT*fps**4/(2*np.pi)**2*(D.T*(1/np.sqrt(1+np.sum(ang**2,-1))))@D
                at=LINEAR_WEIGHT*fps**4*(D.T*(1/np.sqrt(1+np.sum(lin_np**2,-1))))@D
                ar=(ar+5*np.eye(m))/m
                delta_r=-np.linalg.solve(ar[core,core],grad_r)
                delta_t=np.stack([-np.linalg.solve(((at+np.diag(2000+2*prior[lo:hi,j]))/m)[core,core],grad_t[:,j]) for j in range(3)],-1)
                delta_r*=min(1,np.deg2rad(3)/max(np.linalg.norm(delta_r,axis=1).max(),1e-12))
                delta_t*=min(1,.01/max(np.linalg.norm(delta_t,axis=1).max(),1e-12))
                delta_t[seed['depth_locked'][ids],2]=0.
                frozen_pose=(states=='missing') | np.r_[states[1:]=='missing',False]
                delta_t[frozen_pose[ids]]=0.
                delta_r[frozen_pose[ids]]=0.
                old=terms.copy();before=full_score(obs,r,t);accepted=False
                for trial in range(8):
                    factor=.5**trial;nr,nt,no=cr.copy(),ct.copy(),co.copy()
                    nr[core]=Rotation.from_rotvec(delta_r*factor).as_matrix()@cr[core];nt[core]+=factor*delta_t
                    if not trajectory_feasible(nr,nt,seed['rotation'][lo:hi],seed['translation'][lo:hi],
                            states[lo:hi],fps,MOTION,seed['depth_locked'][lo:hi]):continue
                    no[core]=scene.observe(nr[core],nt[core],ids);candidate=local(no,nr,nt)
                    observed_ids=observed_frames[lo:hi]
                    per_frame_ok=evidence_feasible(no,limits[lo:hi]).all()
                    mean_delta=float((no[:,0]-co[:,0])[observed_ids].sum())
                    observed_mean_ok=(not observed_frames.any() or obs[observed_frames,0].mean()+mean_delta/observed_frames.sum()<=seed_ob[observed_frames,0].mean()+.01)
                    if np.isfinite(candidate['total']) and candidate['total']<old['total']-1e-9 and per_frame_ok and observed_mean_ok:
                        cr,ct,co,terms=nr,nt,no,candidate;accepted=True;break
                r[lo:hi],t[lo:hi],obs[lo:hi]=cr,ct,co
                after=full_score(obs,r,t)
                expected=before['total']+(terms['total']-old['total'])*m/n
                assert abs(after['total']-expected)<1e-8 and after['total']<=before['total']+1e-10
                history.append({'window':wi,'step':step+1,'accepted':accepted,'before':before,'after':after})
                if not accepted:break
            row={'window':wi,'start':start,'stop':stop,'score':full_score(obs,r,t),'elapsed_seconds':time.monotonic()-started}
            windows.append(row);save_json(a.out/'windows.json',windows);save_json(a.out/'history.json',history)
            np.savez_compressed(a.out/'checkpoint.npz',frames=frames,rotation=r,translation=t,observations=obs,completed_windows=len(windows))
            print('WINDOW',json.dumps(row),flush=True)
        return
    if len(windows)!=len(starts):raise ValueError(f'Refinement incomplete: {len(windows)}/{len(starts)} windows')
    assert trajectory_feasible(r,t,seed['rotation'],seed['translation'],states,fps,MOTION,seed['depth_locked'])
    final_ob=scene.observe(r,t);np.testing.assert_allclose(final_ob,obs,atol=2e-6,rtol=1e-5)
    assert evidence_feasible(final_ob,limits).all(), 'Final observed silhouette regression'
    if observed_frames.any():assert final_ob[observed_frames,0].mean()<=seed_ob[observed_frames,0].mean()+.01+1e-7
    result=scene.data.copy();result.update(rotation=r,translation=t)
    np.savez_compressed(a.out/'optimized.npz',**result)
    variants={'source':scene.poses(a.source),'v1':scene.poses(a.v1),'v2':scene.poses(a.v2),'v7':(r,t)}
    metrics={};observations={};valid=scene.target_area>0
    for name,(rr,tt) in variants.items():
        raw=scene.visible_observations(rr,tt);censored=final_ob if name=='v7' else scene.observe(rr,tt)
        observations[name]={'visible':raw.tolist(),'censored':censored.tolist()}
        jump=np.linalg.norm(np.diff(tt,axis=0),axis=1);j=int(jump.argmax())
        metrics[name]={'visible_iou_nonempty':float((1-raw[valid,0]).mean()),
            'censored_iou_nonempty':float((1-censored[valid,0]).mean()),
            'visible_frames_below_0_1':int(np.sum((1-raw[:,0]<.1)&valid)),
            'max_step_m':float(jump[j]),'max_step_frames':frames[j:j+2].tolist(),
            'step_percentile_95_m':float(np.percentile(jump,95)),
            'image_observed_frames':int(valid.sum()),
            'depth_range_m':[float(tt[:,2].min()),float(tt[:,2].max())]}
        metrics[name]['by_visibility']={state:{'frames':int(np.sum(states==state)),
            'visible_iou':float((1-raw[states==state,0]).mean()) if np.any(states==state) else None,
            'censored_iou':float((1-censored[states==state,0]).mean()) if np.any(states==state) else None}
            for state in ['clear','partial']}
    save_json(a.out/'per_frame_observations.json',{'frames':frames.tolist(),'variants':observations,
        'target_observed':valid.tolist(),'order':['silhouette_loss','color_loss'],
        'note':'Visible IoU uses the original joint human/object z-buffer. Censored IoU is a separate occlusion-aware objective; never substitute one for the other in comparisons.'})
    report={'status':'complete','frames':n,'windows':len(windows),'final_rerender_check':True,
        'all_observed_frames_pass_evidence_guard':True,'hard_motion_guard_passed':True,
        'unresolved_motion_fit_frames':json.loads((a.out/'prepare_report.json').read_text())['unresolved_visible_frames'],'anchor_prior_supported_frames':int(seed['anchor_prior_supported'].sum()),
        'unexplained_observed_frames':frames[observed_frames&(final_ob[:,0]>=1-1e-6)].tolist(),
        'first_frame':{'translation_m':t[0].tolist(),'visible_iou':1-observations['v7']['visible'][0][0],
            'censored_iou':1-float(final_ob[0,0]),'anchor_prior_supported':bool(seed['anchor_prior_supported'][0])},
        'score_seed':full_score(seed_ob,seed['rotation'],seed['translation']),'score_final':full_score(final_ob,r,t),
        'state_counts':{s:int(np.sum(states==s)) for s in ['clear','partial','missing']},'metrics':metrics,
        'scope':'Experimental object pose only. Person/mesh/scale/camera fixed; no reference 3D or Kaggle scoring; no submission replacement.',
        'topology_cache':scene.cache.report()}
    save_json(a.out/'report.json',report);print('DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
