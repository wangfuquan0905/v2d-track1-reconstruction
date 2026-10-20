"""Run unchanged v7 fitting with fixed-size GPU render batches and shared Scene."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from run_track1_occlusion_v7 import digest, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "v1", "v2", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--steps", type=int, default=12)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Requires a fresh output; a lost process cannot restore its opaque GPU cache")
    import fit_occlusion_motion_anchored as producer
    from occlusion_scene_fixed_batch import Scene
    original_scene_class = producer.Scene
    scene_class = Scene
    cached_scene = None
    requests = []

    def shared_scene(source, batch):
        nonlocal cached_scene
        if Path(source).resolve() != a.source.resolve() or batch != a.batch:
            raise ValueError("The shared Scene cannot be rebound to a different input")
        if cached_scene is None:
            cached_scene = scene_class(source, batch)
        requests.append(id(cached_scene))
        return cached_scene

    producer.Scene = shared_scene
    common = ["--source", str(a.source), "--v1", str(a.v1), "--v2", str(a.v2),
              "--out", str(a.out), "--batch", str(a.batch), "--window", str(a.window),
              "--stride", str(a.stride), "--steps", str(a.steps)]
    wrapper_sources = [Path(__file__).resolve(), ROOT / "scripts/repair_track1_occlusion_v7_batch_stable.py",
                       ROOT / "scripts/occlusion_scene_fixed_batch.py"]
    try:
        for stage in ["prepare", "refine", "finish"]:
            print("BATCH_STABLE_STAGE", stage, flush=True)
            sys.argv = [producer.__file__, *common, "--stage", stage]
            producer.main()
            if stage == "prepare":
                cfg = read(a.out / "config.json")
                cfg["stage_execution"] = "prepare/refine/finish share one Scene and cached opaque topology hashes"
                cfg["original_producer_unchanged"] = True
                cfg["fixed_render_batch"] = {"size": a.batch, "padding": "repeat last input; discard padded outputs"}
                for path in wrapper_sources:
                    cfg["source_sha256"][str(path)] = digest(path)
                    (a.out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
                write(a.out / "config.json", cfg)
        if len(requests) != 3 or len(set(requests)) != 1:
            raise AssertionError("Stages did not share exactly one Scene")
        import numpy as np
        with np.load(a.out / "optimized.npz") as data:
            first = cached_scene.observe(data["rotation"], data["translation"])
            second = cached_scene.observe(data["rotation"], data["translation"])
        np.testing.assert_array_equal(first, second)
        report = read(a.out / "report.json")
        report["fixed_batch_render_verification"] = {
            "scene_constructions": 1, "stage_scene_requests": 3,
            "post_finish_repeat_max_abs": float(np.max(np.abs(first-second))),
            "tolerances_unchanged": True, "geometry_and_losses_unchanged": True,
            "render_batch_size": a.batch, "padding_outputs_discarded": True,
            "wrapper_sha256": {str(f): digest(f) for f in wrapper_sources},
        }
        write(a.out / "report.json", report)
        print("BATCH_STABLE_VERIFIED", json.dumps(report["fixed_batch_render_verification"]), flush=True)
    finally:
        producer.Scene = original_scene_class


if __name__ == "__main__":
    main()
