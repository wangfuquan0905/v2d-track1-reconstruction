"""Anchor SAM3D Objects' layout to human-aligned monocular depth.

This is an estimated scale, not ground-truth metric calibration. Use only RGB
derived masks, SAM3D layout and the depth alignment output of the body model.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import h5py
import numpy as np
import pyrender
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object-dir", type=Path, required=True)
    p.add_argument("--aligned-depth", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    a = p.parse_args()
    layout = json.loads((a.object_dir / "object_report.json").read_text())
    mesh = trimesh.load(a.object_dir / "object_local.glb", force="mesh", process=False)
    q = np.asarray(layout["rotation"])[0]
    # SAM3D compose_transform uses PyTorch3D row-vector rotation. Its exported
    # GLB has an extra -90 degree X rotation; reverse it before applying layout.
    R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    Rx = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    scale = np.asarray(layout["scale"])[0]
    translation = np.asarray(layout["translation"])[0]
    verts_sam = (mesh.vertices @ Rx.T * scale) @ R + translation
    verts_cv = verts_sam * [-1, -1, 1]
    camera_mesh = mesh.copy()
    camera_mesh.vertices = verts_cv
    with open(a.intrinsics, "rb") as f:
        calibration = pickle.load(f)
    if isinstance(calibration, dict):
        print("CALIBRATION_KEYS", list(calibration), flush=True)
        K = np.array([[calibration["fx"], 0, calibration["cx"]],
                      [0, calibration["fy"], calibration["cy"]], [0, 0, 1]], dtype=float)
    else:
        K = np.asarray(calibration, dtype=float)
    mask = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE) > 0
    h, w = mask.shape
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(camera_mesh, smooth=False))
    scene.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=.01, zfar=100), pose=np.diag([1., -1., -1., 1.]))
    renderer = pyrender.OffscreenRenderer(w, h)
    try:
        depth_sam = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        renderer.delete()
    with h5py.File(a.aligned_depth, "r") as f:
        camera = next(iter(f["aligned"]))
        depth = cv2.imdecode(np.asarray(f[f"aligned/{camera}"][0]), cv2.IMREAD_UNCHANGED).astype(float) / 1000
        human_depth_scale = float(f[f"alignment/{camera}/scale"][0])
    overlap = mask & (depth_sam > 0) & (depth > 0)
    iou = float(np.sum(mask & (depth_sam > 0)) / np.sum(mask | (depth_sam > 0)))
    overlap = cv2.erode(overlap.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if overlap.sum() < 200 or iou < .4:
        raise ValueError(f"SAM3D layout and observed mask disagree: IoU={iou}, pixels={overlap.sum()}")
    ratios = depth[overlap] / depth_sam[overlap]
    ratio = float(np.median(ratios))
    metric = mesh.copy()
    metric.vertices = mesh.vertices * scale * ratio
    # Preserve the measured shape and local axes; reduce rasterization cost.
    original_vertices = metric.vertices.copy()
    original_colors = metric.visual.vertex_colors.copy()
    simplified = metric.simplify_quadric_decimation(face_count=20000)
    nearest = cKDTree(original_vertices).query(simplified.vertices)[1]
    simplified.visual.vertex_colors = original_colors[nearest]
    simplified.export(a.object_dir / "object_metric_estimate.glb")
    report = {"method": "median aligned-depth / SAM3D rendered-depth on eroded mask overlap", "metric_status": "monocular estimate anchored to inferred human size; no measured object dimensions",
              "sam_layout_mask_iou": iou, "overlap_pixels": int(overlap.sum()),
              "scene_depth_ratio": ratio, "ratio_p10_p90": np.percentile(ratios, [10, 90]).tolist(),
              "human_depth_scale_first_frame": human_depth_scale,
              "local_mesh_scale_to_m": (scale * ratio).tolist(),
              "mesh_extents_m": simplified.extents.tolist(), "faces": len(simplified.faces),
              "inputs": {"depth": str(a.aligned_depth.resolve()), "intrinsics": str(a.intrinsics.resolve()), "mask": str(a.mask.resolve())}}
    (a.object_dir / "scale_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(a.object_dir / "scale_diagnostic.npz", sam_depth=depth_sam, aligned_depth=depth, mask=mask, overlap=overlap)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
