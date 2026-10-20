"""Validate all v7 episodes, pack official rows and audit every roundtrip field."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from run_track1_occlusion_v7 import OUT, read, write, digest, verify, validate_result
from v2d_syilen.submission_checks import submission_arrays
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-url", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}", args.commit_url):
        raise ValueError("A published full producing commit URL is required")
    manifest = read(OUT / "run_manifest.json")
    kit = ROOT / "third_party/v2d_submission_kit_20261018"
    sample = kit / "data/track_1_sample_submission.parquet"
    if digest(sample) != manifest["sample_sha256"]:
        raise ValueError("Official sample changed")
    # Invalidate any stale delivery status before starting a new verification.
    write(OUT / "delivery_manifest.json", {"status": "validating", "kaggle_uploaded": False})
    dest = OUT / "submission_episodes"
    dest.mkdir(exist_ok=True)
    rows = []
    for ep in manifest["episodes"]:
        verify(manifest, ep)
        name = ep["episode"]
        run = OUT / "episodes" / name / "occlusion"
        checks = validate_result(ep, run)
        media = read(run / "video_validation.json")
        if media["decoded_frames"] != ep["frames"] or media["fps"] != ep["fps"] or digest(run / "comparison.mp4") != media["sha256"]:
            raise ValueError("Video verification failed: " + name)
        with np.load(run / "optimized.npz") as data:
            np.savez_compressed(dest / (name + ".npz"), **submission_arrays(data))
        mesh = ROOT / f"outputs/{name}/cari4d/{name}/export/{name}/object_mesh/output_aligned.glb"
        link = dest / (name + "_object.glb")
        if link.exists() and link.resolve() != mesh.resolve():
            raise ValueError("Wrong staged mesh: " + name)
        if not link.exists():
            link.symlink_to(mesh)
        rows.append({"episode": name, "checks": checks, "prediction_sha256": digest(run / "optimized.npz"),
                     "staged_sha256": digest(dest / (name + ".npz")), "mesh_sha256": digest(mesh)})
    if len(rows) != 30:
        raise ValueError("Incomplete competition coverage")
    # Reuse the exact official-budget/export auditor with a different output root.
    import prepare_track1_meshes
    prepare_track1_meshes.OUT = OUT
    prepare_track1_meshes.main()
    parquet = OUT / "track1_occlusion_v7.parquet"
    temporary = OUT / "track1_occlusion_v7.validating.parquet"
    command = [sys.executable, str(kit / "eval_reconstruction.py"), "--episodes", str(dest),
               "--sample", str(sample), "--commit", args.commit_url, "--out", str(temporary)]
    with (OUT / "packing.log").open("w") as log:
        subprocess.run(command, cwd=kit, stdout=log, stderr=subprocess.STDOUT, check=True)
    import audit_track1_parquet
    audit_track1_parquet.OUT = OUT
    argv = sys.argv
    try:
        sys.argv = ["audit_track1_parquet.py", str(temporary)]
        audit_track1_parquet.main()
    finally:
        sys.argv = argv
    audit = read(OUT / "packed_audit.json")
    if audit["code_commit_url"] != args.commit_url:
        raise ValueError("Producing commit mismatch")
    temporary.replace(parquet)
    audit["parquet"] = str(parquet)
    write(OUT / "packed_audit.json", audit)
    record = {
        "status": "ready_for_user_submission", "kaggle_uploaded": False,
        "method": "occlusion_motion_v7", "commit_url": args.commit_url,
        "run_manifest_sha256": digest(OUT / "run_manifest.json"),
        "parquet": str(parquet), "parquet_sha256": digest(parquet),
        "bytes": parquet.stat().st_size, "episodes": 30,
        "input_frames": sum(e["frames"] for e in manifest["episodes"]),
        "scored_frames": audit["scored_frames"], "rows": audit["rows"],
        "checks": rows, "generated_unix": time.time(),
        "known_limit": manifest["known_limit"],
    }
    write(OUT / "delivery_manifest.json", record)
    print("READY_FOR_USER_SUBMISSION", json.dumps({k: v for k, v in record.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
