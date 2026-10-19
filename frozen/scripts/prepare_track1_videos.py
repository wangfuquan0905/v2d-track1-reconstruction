"""Produce browser-playable H.264 mask videos, including corrected SAM2 masks."""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    for ep in json.loads((OUT/'inventory.json').read_text()):
        source=ROOT/'outputs'/ep['episode']/'segmentation';report=source/'segmentation_report.json'
        if not report.exists():continue
        target=OUT/'segmentation_review'/ep['episode'];target.mkdir(parents=True,exist_ok=True)
        hashes={str(p):digest(p) for p in [report,source/'masks.h5']};manifest=target/'video.json'
        if manifest.exists() and json.loads(manifest.read_text())['input_sha256']==hashes:
            print('REUSE_MASK_VIDEO',ep['episode'],flush=True);continue
        metadata=json.loads(report.read_text());temporary=target/'mask_encode.mp4';final=target/'masks_preview.mp4'
        if 'rgb_recovery_v1' in metadata:
            seq=metadata['prompts']['sequence'];raw=target/'corrected_raw.mp4'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),ep['fps'],(768,576))
            with h5py.File(source/'masks.h5','r') as masks:
                for frame in range(ep['frames']):
                    rgb=cv2.imread(str(source/'frames'/f'{frame:06d}.jpg'))
                    for suffix,color in [('person_mask.png',(240,190,30)),('obj_rend_mask.png',(40,220,60))]:
                        m=masks[f'{seq}/{frame:06d}-k0.{suffix}'][:].astype(bool)
                        rgb[m]=(.55*rgb[m]+.45*np.array(color)).astype('uint8')
                    cv2.putText(rgb,f"SAM2 {ep['episode']} frame {frame}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                    writer.write(cv2.resize(rgb,(768,576)))
            writer.release();video=raw
        else:video=source/'masks_preview.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=codec_name,nb_frames,r_frame_rate','-of','json',str(temporary)]))['streams'][0]
        if info['codec_name']!='h264' or int(info['nb_frames'])!=ep['frames']:raise ValueError('Video count mismatch')
        temporary.replace(final)
        manifest.write_text(json.dumps({'input_sha256':hashes,'output_sha256':digest(final),'stream':info},indent=2))
        print('MASK_VIDEO_READY',ep['episode'],flush=True)


if __name__=='__main__':main()
