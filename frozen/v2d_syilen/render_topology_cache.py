"""Reuse nvdiffrast's documented topology hash for an unchanged triangle array.

The CUDA hash builder can pick different edge neighbours on nonmanifold meshes.
Constructing it once makes the observation function consistent within a run.
No mesh, loss, rasterizer, antialiasing rule, or validation tolerance is changed.
"""
import hashlib


class AntialiasTopologyCache:
    def __init__(self, original, construct):
        self.original=original;self.construct=construct;self.entries=[];self.calls=0

    def __call__(self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
        import torch
        self.calls+=1
        if topology_hash is None:
            for saved, topology_hash, _ in self.entries:
                if saved.device==tri.device and saved.shape==tri.shape and torch.equal(saved,tri):break
            else:
                saved=tri.detach().clone();topology_hash=self.construct(saved)
                sha=hashlib.sha256(saved.cpu().numpy().tobytes()).hexdigest()
                self.entries.append((saved,topology_hash,sha))
        return self.original(color,rast,pos,tri,topology_hash=topology_hash,pos_gradient_boost=pos_gradient_boost)

    def report(self):
        return {'method':'nvdiffrast antialias_construct_topology_hash reuse for identical triangle arrays',
            'antialias_calls':self.calls,'hash_constructions':len(self.entries),
            'triangle_sha256':[v[2] for v in self.entries],
            'loss_mesh_and_validation_tolerances_unchanged':True}
