"""Run official CARI4D stages using local paths and a calibrated candidate mesh.

The human/depth bootstrap is produced before this driver. It deliberately uses
a separate export directory: changing object scale cannot invalidate the RGB
and human mask inputs used to estimate human motion. All inputs are RGB-derived.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def runtime_env(root):
    modules = root / "third_party/video_to_data/reconstruction/modules"
    weights = root / "data/models/cari4d_runtime"
    source = modules / "v2d_cari4d/lib/cari4d"
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [source, modules / "v2d_sam3d_body/lib", modules / "v2d_foundation_pose/lib/FoundationPose"])),
               MHR_ASSETS_ROOT=str(weights / "sam3d_body"), SAM3D_BODY_ROOT=str(modules / "v2d_sam3d_body/lib"),
               TORCH_HOME=str(weights / "sam3d_body/torch_home"), HF_HOME=str(weights / "hf_home"),
               FOUNDATIONPOSE_WEIGHTS_DIR=str(weights / "foundationpose/nvlabs_pytorch"),
               CUDA_HOME="/usr/local/cuda-12.8", PYOPENGL_PLATFORM="egl", OPENCV_IO_ENABLE_OPENEXR="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
    return source, weights, env


def main():
    p = argparse.ArgumentParser()
    names = ["metric_export", "foundationpose", "coconet", "refinement", "render"]
    p.add_argument("--episode", default="episode_000016")
    p.add_argument("--from-stage", choices=names, default=names[0])
    p.add_argument("--through-stage", choices=names, default=names[-1])
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, weights, env = runtime_env(root)
    episode = root / "outputs" / a.episode
    run = episode / "cari4d" / a.episode
    export = run / "export" / a.episode
    raw = run / "depth/moge2"
    init = run / "mhr_init" / f"{a.episode}.pkl"
    fp = run / "foundationpose" / f"{a.episode}.pkl"
    inference = run / "inference"
    fp.parent.mkdir(parents=True, exist_ok=True)
    inference.mkdir(parents=True, exist_ok=True)
    video = episode / "inputs" / f"{a.episode}.0.color.mp4"
    model = weights / "cari4d/2026-08-25-09-35-57"
    mesh = export / "object_mesh/output_aligned.glb"
    before, after = inference / "coconet.pth", inference / "refined.pth"
    python = sys.executable
    commands = [
        [python, source / "prep/prepare_mhr_wild_export.py", "--video", video, "--mask-h5", episode / "segmentation/masks.h5", "--object-mesh", episode / "object/object_metric_estimate.glb", "--intrinsics-file", raw / "intrinsics.pkl", "--output-root", export.parent],
        [python, source / "prep/run_foundationpose_mhr_export.py", export, "--depth-root", raw / "aligned.h5", "--out-file", fp, "--camera-id", "0", "--iteration", "5", "--max-attempts", "5", "--register-first-then-track", "--no-first-usable-frame-gt-rotation-oracle", "--viz-file", fp.parent / "tracking_preview.mp4"],
        [python, source / "tools/run_mhr_wild_inference.py", export, "--depth-h5", raw / "aligned.h5", "--mhr-init", init, "--foundationpose-file", fp, "--config", model / "resolved_config.yaml", "--checkpoint", model / "step200000.pth", "--output", before, "--stride", "96", "--render-batch-size", "8", "--crop-workers", "4", "--crop-buffer-count", "2", "--input-cache", inference / "materialized_inputs.h5", "--device", "cuda", "--offline-supervision-contract"],
        [python, "-m", "learning.training.mhr_opt_refineout", "--bundle", before, "--object-mesh", mesh, "--out", after, "--mode", "smplh_parity", "--num-steps", "300", "--batch-size", "0", "--device", "cuda", "--postopt-checkpoint", inference / "refinement_checkpoint.pth", "--mhr-assets-root", weights / "sam3d_body", "--w-temporal", "100", "--w-human-pose-prior", "200", "--contact-activation-distance-m", "0.05", "--report-every", "10", "--diagnostics-every", "500"],
        [python, source / "tools/render_mhr_wild_inference.py", export, "--source-video", video, "--object-mesh", mesh, "--before-bundle", before, "--after-bundle", after, "--output", inference / "reconstruction.mp4", "--comparison-output", inference / "before_after.mp4", "--device", "cuda", "--batch-size", "4"],
    ]
    for idx in range(names.index(a.from_stage), names.index(a.through_stage) + 1):
        name = names[idx]
        command = list(map(str, commands[idx]))
        record = {"stage": name, "command": command, "status": "running", "started_unix": time.time()}
        status = run / f"native_{name}.json"
        status.write_text(json.dumps(record, indent=2))
        print(f"START {name}", flush=True)
        with (run / f"native_{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, status="passed" if result.returncode == 0 else "failed", elapsed_seconds=time.time() - record["started_unix"])
        status.write_text(json.dumps(record, indent=2))
        print(f"END {name}: {record['status']} ({record['elapsed_seconds']:.1f}s)", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
