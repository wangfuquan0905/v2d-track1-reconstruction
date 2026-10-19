"""Check that every complete prediction has matching full-length review artifacts."""
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1'


def main():
    rows=[];failures=[]
    for ep in json.loads((OUT/'inventory.json').read_text()):
        name=ep['episode'];root=ROOT/'outputs'/name;review=OUT/'review'/name
        try:
            result=json.loads((root/'safe_windows_v1/report.json').read_text())
            if result['status']!='complete':raise ValueError('Prediction incomplete')
            manifest=json.loads((review/'manifest.json').read_text())
            if manifest['frames']!=ep['frames']:raise ValueError('3D manifest frame count mismatch')
            for variant in ['baseline','ours']:
                data=np.load(review/(variant+'.npy'),mmap_mode='r')
                if data.shape!=(ep['frames'],manifest['record_floats']):raise ValueError('3D binary shape mismatch')
            streams=[]
            for video in [root/'safe_windows_v1/comparison.mp4',OUT/'segmentation_review'/name/'masks_preview.mp4']:
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                    'stream=codec_name,nb_frames,width,height,r_frame_rate','-of','json',str(video)]))['streams'][0]
                if probe['codec_name']!='h264' or int(probe['nb_frames'])!=ep['frames']:raise ValueError('Video codec/frame count mismatch')
                numerator,denominator=map(float,probe['r_frame_rate'].split('/'))
                if abs(numerator/denominator-ep['fps'])>1e-5:raise ValueError('Video FPS mismatch')
                streams.append(probe)
            with Image.open(root/'safe_windows_v1/comparison.png') as im:
                if im.size!=(streams[0]['width'],5*streams[0]['height']):raise ValueError('Keyframe sheet dimensions mismatch')
            rows.append({'episode':name,'frames':ep['frames'],'videos':streams,'3d':'complete','keyframes':5})
        except Exception as exc:failures.append({'episode':name,'error':str(exc)})
    report={'status':'passed' if len(rows)==30 and not failures else 'incomplete','complete':rows,'failures':failures}
    (OUT/'media_validation.json').write_text(json.dumps(report,indent=2));print('MEDIA_VALIDATION',report['status'],len(rows))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
