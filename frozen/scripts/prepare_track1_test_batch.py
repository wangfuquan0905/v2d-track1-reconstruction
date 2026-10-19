"""Download the pinned public RGB release and inventory all scored test episodes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/track1_safe_windows_v1'
DATA = ROOT / 'data/track1_release'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((OUT/'dataset_manifest.json').read_text())
    wanted = [r for r in manifest['files'] if r['path'].endswith('.mp4') or '/meta/' in r['path'] or r['path'].endswith('README.md')]
    def fetch(row):
        path = Path(hf_hub_download(manifest['repo'], row['path'], repo_type='dataset',
            revision=manifest['revision'], local_dir=DATA, token=False))
        sha = digest(path)
        if row['lfs'] and sha != row['lfs']['sha256']:
            raise ValueError('Download hash mismatch: '+row['path'])
        if path.stat().st_size != row['size']:
            raise ValueError('Download size mismatch: '+row['path'])
        print('DOWNLOADED', row['path'], flush=True)
        return {'path': str(path), 'sha256': sha, 'size': path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, wanted))
    (OUT/'downloads.json').write_text(json.dumps(downloaded, indent=2))
    meta = [json.loads(line) for line in (DATA/'track_1/meta/episodes_metadata.jsonl').read_text().splitlines()]
    scored = {e['episode_index']: e for e in json.loads((ROOT/'configs/track1_test_frames_v1.json').read_text())['episodes']}
    first = OUT/'first_frames';first.mkdir(exist_ok=True)
    inventory = []
    for row in meta:
        idx = row['episode_index'];name=f'episode_{idx:06d}'
        video=DATA/f'track_1/videos/chunk-000/observation.images.exo_camera/{name}.mp4'
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        ok, im=cap.read();cap.release()
        if not ok or max(scored[idx]['scored_frames'])>=count:raise ValueError(name+' invalid video coverage')
        cv2.imwrite(str(first/(name+'.jpg')), im, [cv2.IMWRITE_JPEG_QUALITY,95])
        inputs=ROOT/'outputs'/name/'inputs';inputs.mkdir(parents=True,exist_ok=True)
        target=inputs/(name+'.0.color.mp4')
        if target.exists():
            if digest(target)!=digest(video):raise ValueError(name+' existing input differs')
            if target.is_symlink():
                temporary=target.with_suffix('.hardlink.tmp');os.link(video,temporary);temporary.replace(target)
        else:os.link(video,target)
        inventory.append(dict(row,episode=name,video=str(video),frames=count,fps=fps,
            width=im.shape[1],height=im.shape[0],scored_count=scored[idx]['scored_count']))
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    for page in range(5):
        sheet=Image.new('RGB',(1536,2*620),(25,25,25));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(inventory[page*6:(page+1)*6]):
            im=Image.open(first/(row['episode']+'.jpg'));im.thumbnail((512,576))
            x=(j%3)*512;y=(j//3)*620
            sheet.paste(im,(x,y+40));draw.text((x+8,y+8),row['episode']+' '+row['object'],fill='white')
        sheet.save(first/f'contact_{page}.jpg',quality=94)
    print('READY',len(inventory),'episodes',sum(e['frames'] for e in inventory),'frames',flush=True)


if __name__=='__main__':main()
