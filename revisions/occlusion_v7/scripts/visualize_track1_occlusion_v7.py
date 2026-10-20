"""Full-frame v1/v7 test comparison. No hidden labels or accuracy claims."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from occlusion_scene import Scene
from run_track1_occlusion_v7 import OUT, read, write, digest
import cv2
import imageio.v2 as imageio
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=int, required=True)
    args = p.parse_args()
    ep = next(e for e in read(OUT / "run_manifest.json")["episodes"] if e["episode_index"] == args.episode)
    scene = Scene(ROOT / ep["source"])
    run = OUT / "episodes" / ep["episode"] / "occlusion"
    old_r, old_t = scene.poses(ROOT / ep["stable"])
    new_r, new_t = scene.poses(run)
    states = read(run / "visibility.json")["states"]
    observations = read(run / "per_frame_observations.json")["variants"]
    decisions = read(run / "motion_decisions.json")
    steps = np.r_[0., np.linalg.norm(np.diff(new_t, axis=0), axis=1)]
    snapshots = {0, scene.n//2, scene.n-1, int(steps.argmax())}
    missing = [i for i, s in enumerate(states) if s == "missing"]
    if missing:
        snapshots.update([missing[0], min(missing[-1]+1, scene.n-1)])
    w, h = 384, 384
    def text(im, value, x, y, size=.47):
        cv2.putText(im, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX, size, (240,240,240), 1, cv2.LINE_AA)
    def panel(rgb, label):
        a = np.zeros((h+36,w,3), np.uint8)
        ih, iw = rgb.shape[:2]
        scale = min(w/iw,h/ih)
        ww, hh = round(iw*scale), round(ih*scale)
        a[36+(h-hh)//2:36+(h-hh)//2+hh,(w-ww)//2:(w-ww)//2+ww] = cv2.resize(rgb,(ww,hh))
        text(a,label,8,24)
        return a
    def overlay(i, alpha):
        rgb = scene.images[i].astype(float)*.55
        rgb[...,1] += scene.masks[i,...,1]*90
        rgb[...,0] += alpha*90
        rgb[...,2] += alpha*90
        return np.clip(rgb,0,255).astype("uint8")
    path = run / "comparison.mp4"
    with imageio.get_writer(str(path), fps=scene.fps, codec="libx264", macro_block_size=None,
                           ffmpeg_params=["-crf","18"]) as writer:
        for i, frame in enumerate(scene.frames):
            if i % scene.batch == 0:
                ids = np.arange(i,min(i+scene.batch,scene.n))
                old = scene.render(old_r[ids],old_t[ids],ids,visible=True).cpu().numpy()[...,1]
                new = scene.render(new_r[ids],new_t[ids],ids,visible=True).cpu().numpy()[...,1]
            j = i % scene.batch
            iou = lambda v: f'{1-observations[v]["visible"][i][0]:.3f}' if scene.target_area[i] else "n/a"
            canvas = np.zeros((h+140,w*3,3),np.uint8)
            canvas[:h+36] = np.concatenate([
                panel(scene.images[i],f'RGB | frame {frame}'),
                panel(overlay(i,old[j]),f'Submitted v1 | visible IoU {iou("v1")}'),
                panel(overlay(i,new[j]),f'v7 | visible IoU {iou("v7")}')],axis=1)
            text(canvas,f'{ep["episode"]} | {ep["object"]} | {i/scene.fps:.2f}s | {states[i]}',10,h+60)
            text(canvas,f'Camera depth: v1 {old_t[i,2]:.3f} m | v7 {new_t[i,2]:.3f} m | v7 step {steps[i]*100:.2f} cm | {decisions[i]["prediction"]}',10,h+84)
            text(canvas,'Green = observed mask; magenta = rendered visible object. Missing mask: IoU n/a.',10,h+108)
            text(canvas,'Image fit and continuity diagnostics only; no reference 3D / Kaggle score.',10,h+130)
            writer.append_data(canvas)
            if i in snapshots:
                cv2.imwrite(str(run / f"frame_{int(frame):06d}.png"),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    cap = cv2.VideoCapture(str(path))
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != scene.n or fps != scene.fps:
        raise ValueError("Incomplete or mistimed video")
    write(run / "video_validation.json",{"decoded_frames":count,"fps":fps,"sha256":digest(path),
          "all_frames_included":True,"no_reference_inputs":True})
    print("VIDEO_COMPLETE", ep["episode"], count, flush=True)


if __name__ == "__main__":
    main()
