"""Generate an object candidate from one RGB frame and its predicted mask.

The local mesh is unitless. Layout scale is a monocular estimate, retained
separately; later CARI4D input scaling must be explicitly verified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--low-vram", action="store_true", help="Disabled after host memory exhaustion; retained to reject old commands explicitly")
    a = p.parse_args()
    if a.low_vram:
        p.error("--low-vram is disabled: CPU offloading exhausted host resources during validation. Do not retry this command.")
    from resource_preflight import require_reconstruction_capacity
    require_reconstruction_capacity()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.source.resolve()), str(a.source.resolve() / "notebook")]
    from inference import Inference
    inference = Inference(str(a.source.resolve() / "checkpoints/hf/pipeline.yaml"), compile=False)
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    mask = np.asarray(Image.open(a.mask).convert("L")) > 0
    result = inference(rgb, mask, seed=42, with_mesh_postprocess=False,
                       with_texture_baking=False, with_layout_postprocess=True,
                       use_vertex_color=True)
    print("RESULT_KEYS", list(result), flush=True)
    mesh = result["glb"]
    mesh.export(a.out / "object_local.glb")
    mesh.export(a.out / "object_local.ply")
    report = {"source_rgb": str(a.image.resolve()), "source_mask": str(a.mask.resolve()),
              "seed": 42, "low_vram": a.low_vram, "mesh_units": "unitless model coordinates", "layout_scale_status": "monocular estimate, not independently metric calibrated"}
    for key in ("rotation", "translation", "scale"):
        value = result[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        report[key] = np.asarray(value).tolist()
    report["mesh_extents"] = mesh.extents.tolist()
    report["vertices"] = len(mesh.vertices)
    report["faces"] = len(mesh.faces)
    (a.out / "object_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
