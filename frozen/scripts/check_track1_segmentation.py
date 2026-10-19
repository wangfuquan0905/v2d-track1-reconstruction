"""Create RGB-derived segmentation contact sheets and full-sequence area checks."""
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track1_safe_windows_v1'


def main():
    inventory=json.loads((OUT/'inventory.json').read_text())
    dest=OUT/'segmentation_review';dest.mkdir(exist_ok=True)
    checks=[]
    for group in range(5):
        canvas=Image.new('RGB',(1280,6*226),(25,29,29));draw=ImageDraw.Draw(canvas)
        for row,ep in enumerate(inventory[group*6:group*6+6]):
            name=ep['episode'];root=ROOT/'outputs'/name/'segmentation'
            path=root/'segmentation_report.json'
            if not path.exists():continue
            report=json.loads(path.read_text());seq=report['prompts']['sequence']
            areas=np.array([[s['area_pixels'][str(i)] for i in (0,1)] for s in report['stats']])
            if len(areas)!=ep['frames']:raise ValueError('Wrong segmentation length: '+name)
            frames=np.rint(np.linspace(0,ep['frames']-1,5)).astype(int)
            check={'episode':name,'frames':len(areas),'min_person_pixels':int(areas[:,0].min()),
                'min_object_pixels':int(areas[:,1].min()),'zero_person_frames':np.where(areas[:,0]==0)[0].tolist(),
                'zero_object_frames':np.where(areas[:,1]==0)[0].tolist(),'review_frames':frames.tolist()}
            checks.append(check)
            draw.text((5,row*226+2),f"{name} {ep['object']} | min area person={check['min_person_pixels']} object={check['min_object_pixels']}",fill='white')
            with h5py.File(root/'masks.h5','r') as masks:
                for col,frame in enumerate(frames):
                    rgb=np.array(Image.open(root/'frames'/f'{frame:06d}.jpg').convert('RGB'))
                    for suffix,color in [('person_mask.png',(30,190,240)),('obj_rend_mask.png',(60,220,40))]:
                        m=masks[f'{seq}/{frame:06d}-k0.{suffix}'][:].astype(bool)
                        rgb[m]=(.55*rgb[m]+.45*np.array(color)).astype('uint8')
                        contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rgb,contours,-1,color,3)
                    canvas.paste(Image.fromarray(rgb).resize((256,192)),(col*256,row*226+19))
                    draw.text((col*256+5,row*226+212),f"frame {frame} / {ep['frames']-1}",fill='white')
        canvas.save(dest/f'contact_{group}.jpg',quality=94)
    (dest/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'checked':len(checks),'zero_masks':[r for r in checks if r['zero_person_frames'] or r['zero_object_frames']]},indent=2))


if __name__=='__main__':main()
