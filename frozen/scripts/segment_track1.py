"""Propagate explicit, image-grounded SAM2 boxes into CARI4D mask HDF5."""
import argparse
import json
import gc
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chunk-frames", type=int, default=96, help="Bound SAM2 video/state memory; carry the last predicted masks between chunks")
    a = p.parse_args()
    if a.chunk_frames < 3:
        p.error('--chunk-frames must be >=3')
    a.out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(a.prompts.read_text())
    seq = prompts["sequence"]
    frames_dir = a.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dest = frames_dir / f"{count:06d}.jpg"
        if not dest.exists():
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    cap.release()
    model = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", a.checkpoint)
    stats = []
    preview = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with h5py.File(a.out / "masks.h5", "w") as h:
            h.attrs["provenance"] = json.dumps({"video": str(a.video.resolve()), "prompts": prompts, "model": a.checkpoint, "fps": fps})
            def predictions():
                carry = None
                for begin in range(0, count, a.chunk_frames-1):
                    end = min(count, begin+a.chunk_frames)
                    if begin > 0 and end-begin == 1:
                        break
                    folder = a.out/'chunks'/f'{begin:06d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    for frame in range(begin, end):
                        link = folder/f'{frame-begin:06d}.jpg'
                        if not link.exists():
                            os.link(frames_dir/f'{frame:06d}.jpg', link)
                    state = model.init_state(str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
                    for obj in prompts['objects']:
                        if carry is None:
                            model.add_new_points_or_box(state, frame_idx=0, obj_id=obj['id'], box=np.array(obj['box'], dtype=np.float32))
                        else:
                            model.add_new_mask(state, frame_idx=0, obj_id=obj['id'], mask=carry[obj['id']])
                    for local_idx, ids, logits in model.propagate_in_video(state):
                        carry = {obj_id: (logits[k, 0]>0).cpu().numpy() for k, obj_id in enumerate(ids)}
                        if begin == 0 or local_idx > 0:
                            yield begin+local_idx, ids, logits
                    model.reset_state(state)
                    del state
                    gc.collect()
                    torch.cuda.empty_cache()
            for idx, ids, logits in predictions():
                frame = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"))
                overlay = frame.copy()
                areas = {}
                for k, obj_id in enumerate(ids):
                    mask = (logits[k, 0] > 0).cpu().numpy()
                    suffix = "person_mask.png" if obj_id == 0 else "obj_rend_mask.png"
                    h.create_dataset(f"{seq}/{idx:06d}-k0.{suffix}", data=mask, compression="gzip")
                    color = np.array((240, 190, 30) if obj_id == 0 else (40, 220, 60))
                    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
                    areas[str(obj_id)] = int(mask.sum())
                    if idx == 0:
                        cv2.imwrite(str(a.out / f"first_{obj_id}_mask.png"), mask.astype(np.uint8) * 255)
                stats.append({"frame": idx, "area_pixels": areas})
                cv2.putText(overlay, f"SAM2 {seq} frame {idx}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                overlay = cv2.resize(overlay, (768, 576))
                if preview is None:
                    preview = cv2.VideoWriter(str(a.out / "masks_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (768, 576))
                preview.write(overlay)
                if idx in (0, 71, 143, 215, 287, 359):
                    cv2.imwrite(str(a.out / f"preview_{idx:06d}.jpg"), overlay)
                if idx % 30 == 0:
                    print(f"MASK_FRAME {idx}/{count} areas={areas}", flush=True)
    if preview:
        preview.release()
    assert len(stats) == count, (len(stats), count)
    (a.out / "segmentation_report.json").write_text(json.dumps({"frames": count, "fps": fps, "prompts": prompts, "chunk_frames": a.chunk_frames, "chunk_handoff": "last predicted binary masks; one overlapping frame; no reference masks", "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
