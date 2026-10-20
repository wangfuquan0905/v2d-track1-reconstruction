"""Shared local input/render adapter for occlusion experiments; no reference data."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import refine_render_observation  # CUDA/runtime setup, no fitting executed.
import json,pickle
import cv2,h5py,numpy as np,torch,trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
from v2d_syilen.render_observation import ObservationRenderer,observation_terms
from v2d_syilen.render_topology_cache import AntialiasTopologyCache
from v2d_syilen.occlusion_objective import known_pixels,censored_terms,visibility_from_fractions


class Scene:
    def __init__(self,source,batch=8):
        torch.set_num_threads(4)
        self.source=Path(source);self.batch=batch
        self.cfg=json.loads((self.source/'config.json').read_text())
        raw=np.load(self.source/'optimized.npz');self.data={k:raw[k] for k in raw.files}
        self.frames=self.data['frames'];self.n=len(self.frames);self.fps=self.cfg['fps']
        self.episode=self.cfg['episode'];self.root=ROOT/'outputs'/self.episode
        run=self.root/'cari4d'/self.episode
        mesh=trimesh.load(run/f'export/{self.episode}/object_mesh/output_aligned.glb',force='mesh',process=False)
        self.mesh_vertices=np.asarray(mesh.vertices)*float(self.data['object_scale'])
        self.tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
        self.v=self.tensor(self.mesh_vertices)
        bundle=torch.load(self.cfg['bundle_path'],map_location='cpu',weights_only=False)
        with open(run/'depth/moge2/intrinsics.pkl','rb') as f:k=pickle.load(f)
        x0,y0,x1,y1=self.cfg['crop_xyxy'];self.h,self.w=self.cfg['render_size']
        self.K=np.array([k['fx']*self.w/(x1-x0),k['fy']*self.h/(y1-y0),
                         (k['cx']-x0)*self.w/(x1-x0),(k['cy']-y0)*self.h/(y1-y0)])
        self.hf=torch.tensor(bundle['faces'],device='cuda',dtype=torch.int32)
        of=torch.tensor(mesh.faces,device='cuda',dtype=torch.int32);colors=self.tensor(mesh.visual.vertex_colors[:,:3]/255.)
        self.cache=AntialiasTopologyCache(dr.antialias,dr.antialias_construct_topology_hash);dr.antialias=self.cache
        self.visible=ObservationRenderer(self.hf,of,colors,self.tensor(self.K),self.h,self.w,supersample=4)
        self.amodal=ObservationRenderer(self.hf[:0],of,colors,self.tensor(self.K),self.h,self.w,supersample=4)
        images=[];masks=[];known=[]
        with h5py.File(self.root/'segmentation/masks.h5') as f:
            for frame in self.frames:
                im=cv2.cvtColor(cv2.imread(str(self.root/f'segmentation/frames/{frame:06d}.jpg')),cv2.COLOR_BGR2RGB)
                images.append(cv2.resize(im[y0:y1,x0:x1],(self.w,self.h)))
                m=np.stack([f[f'{self.episode}/{frame:06d}-k0.person_mask.png'][()],
                            f[f'{self.episode}/{frame:06d}-k0.obj_rend_mask.png'][()]],-1).astype('uint8')
                m=cv2.resize(m[y0:y1,x0:x1],(self.w,self.h),interpolation=cv2.INTER_NEAREST)
                masks.append(m);known.append(known_pixels(m[...,0],m[...,1]).astype('uint8'))
        self.images=np.array(images);self.masks=np.array(masks);self.known=np.array(known)
        self.human=self.data['vertices'];self.counts={'amodal':0,'visible':0}
        self.target_area=self.masks[...,1].sum((1,2))

    def poses(self,directory):
        with np.load(Path(directory)/'optimized.npz') as d:
            np.testing.assert_array_equal(d['frames'],self.frames)
            for key in ['vertices','pose','object_scale']:np.testing.assert_array_equal(d[key],self.data[key])
            return Rotation.from_matrix(d['rotation']).as_matrix(),d['translation'].astype(float)

    @torch.no_grad()
    def render(self,r,t,ids,visible=False):
        obj=torch.einsum('bij,vj->bvi',self.tensor(r),self.v)+self.tensor(t[:,None])
        renderer=self.visible if visible else self.amodal
        self.counts['visible' if visible else 'amodal']+=len(ids)
        return renderer(self.tensor(self.human[ids]).contiguous(),obj)

    @torch.no_grad()
    def observe(self,r,t,ids=None):
        if ids is None:ids=np.arange(self.n)
        parts=[]
        for start in range(0,len(ids),self.batch):
            ii=ids[start:start+self.batch]
            rendered=self.render(r[start:start+self.batch],t[start:start+self.batch],ii)
            terms=censored_terms(rendered,self.tensor(self.images[ii]/255.),
                                 self.tensor(self.masks[ii,...,1]),self.tensor(self.known[ii]))
            parts.append(terms.cpu().numpy())
        return np.concatenate(parts)

    @torch.no_grad()
    def visibility(self,r,t):
        observed=[];geometric=[]
        for start in range(0,self.n,self.batch):
            ids=np.arange(start,min(start+self.batch,self.n))
            a=self.render(r[ids],t[ids],ids)[...,1]
            v=self.render(r[ids],t[ids],ids,visible=True)[...,1]
            area=a.sum((1,2)).clamp_min(1)
            h=self.tensor(self.masks[ids,...,0]);m=self.tensor(self.masks[ids,...,1])
            observed.extend(((a*h*(1-m)).sum((1,2))/area).cpu().numpy())
            geometric.extend(((a-v).clamp_min(0).sum((1,2))/area).cpu().numpy())
        observed=np.array(observed);geometric=np.array(geometric)
        confidence,states=visibility_from_fractions(observed,geometric,self.target_area)
        return confidence,states,observed,geometric

    @torch.no_grad()
    def visible_observations(self,r,t):
        parts=[]
        for start in range(0,self.n,self.batch):
            ids=np.arange(start,min(start+self.batch,self.n))
            rendered=self.render(r[ids],t[ids],ids,visible=True)
            terms=observation_terms(rendered,self.tensor(self.images[ids]/255.),self.tensor(self.masks[ids]))
            parts.append(np.stack([terms['object_silhouette'].cpu().numpy(),terms['object_color'].cpu().numpy()],-1))
        return np.concatenate(parts)
