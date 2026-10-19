"""Prepare fixed-human RGB optimizer inputs from canonical native MHR predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);a=p.parse_args()
    root=ROOT/'outputs'/a.episode;canonical=root/'test_canonical_v1';dest=root/'test_safe_input_v1';dest.mkdir(exist_ok=True)
    with np.load(canonical/'native.npz') as src: data={k:src[k] for k in src.files}
    frames=data['frames'];n=len(frames)
    np.testing.assert_array_equal(frames,np.arange(n))
    data['rotation']=data.pop('object_rotation');data['translation']=data.pop('object_translation')
    data['coordinate_frame']=np.array('camera')
    np.savez_compressed(dest/'optimized.npz',**data)
    union=None
    with h5py.File(root/'segmentation/masks.h5') as f:
        fps=json.loads(f.attrs['provenance'])['fps'];group=f[a.episode]
        for frame in frames:
            m=group[f'{frame:06d}-k0.person_mask.png'][()].astype(bool)|group[f'{frame:06d}-k0.obj_rend_mask.png'][()].astype(bool)
            union=m if union is None else union|m
    yy,xx=np.where(union)
    if not len(xx):raise ValueError('Empty foreground masks')
    x0,y0=max(0,int(xx.min())-50),max(0,int(yy.min())-50)
    x1,y1=min(union.shape[1],int(xx.max())+51),min(union.shape[0],int(yy.max())+51)
    h=384;w=max(64,round((x1-x0)/(y1-y0)*h/8)*8)
    config={'episode':a.episode,'fps':fps,'frames':n,'crop_xyxy':[x0,y0,x1,y1],
        'render_size':[h,w],'bundle_path':str(canonical/'canonical.pth'),
        'comparison_label':'Full CARI4D','reference_inputs':False,
        'preparation':'Canonical full CARI4D + masks from full monocular RGB; no reference labels',
        'baseline_sha256':hashlib.sha256((dest/'optimized.npz').read_bytes()).hexdigest()}
    (dest/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)


if __name__=='__main__':main()
