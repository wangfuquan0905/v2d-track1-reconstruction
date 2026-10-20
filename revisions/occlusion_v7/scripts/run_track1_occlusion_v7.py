"""Frozen, resumable 30-episode v7 rerun. Produces local files; never uploads."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
OUT = ROOT / "outputs/track1_occlusion_v7"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_files():
    """Include the transitive local import closure, excluding third-party code."""
    pending = [Path("scripts") / name for name in [
        "run_track1_occlusion_v7.py", "pack_track1_occlusion_v7.py",
        "visualize_track1_occlusion_v7.py", "fit_occlusion_motion_anchored.py",
        "refine_object_iou_priority.py", "prepare_track1_meshes.py",
        "audit_track1_parquet.py"]]
    seen = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        for node in ast.walk(ast.parse((ROOT / relative).read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                parent = relative.parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                modules = [node.module] if node.module else [item.name for item in node.names]
                for module in modules:
                    candidate = parent / Path(*module.split(".")).with_suffix(".py")
                    if (ROOT / candidate).is_file() and candidate not in seen:
                        pending.append(candidate)
                continue
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [v.name for v in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                name = Path(*name.split(".")).with_suffix(".py")
                for candidate in [name, Path("scripts") / name]:
                    if (ROOT / candidate).is_file() and candidate not in seen:
                        pending.append(candidate)
    seen.add(Path("v2d_syilen/__init__.py"))
    return sorted(seen)


def check_hashes(hashes):
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input/source changed: " + path)


def freeze():
    inventory = read(ROOT / "outputs/track1_safe_windows_v1/inventory.json")
    if len(inventory) != 30 or {e["episode_index"] for e in inventory} != set(range(30)):
        raise ValueError("Expected all 30 official episodes")
    protocol = ROOT / "configs/track1_test_frames_v1.json"
    sample = ROOT / "third_party/v2d_submission_kit_20261018/data/track_1_sample_submission.parquet"
    if digest(sample) != read(protocol)["sample_sha256"]:
        raise ValueError("Official sample changed")
    episodes = []
    for ep in inventory:
        root = ROOT / "outputs" / ep["episode"]
        stable = root / "safe_windows_v1"
        cfg = read(stable / "config.json")
        source = (ROOT / cfg["source"]).resolve()
        if read(stable / "report.json")["status"] != "complete":
            raise ValueError("Baseline incomplete: " + ep["episode"])
        paths = [source / "optimized.npz", source / "config.json",
                 stable / "optimized.npz", stable / "config.json", stable / "report.json",
                 Path(cfg["bundle_path"]), root / "segmentation/masks.h5",
                 root / f"cari4d/{ep['episode']}/depth/moge2/intrinsics.pkl",
                 root / f"cari4d/{ep['episode']}/export/{ep['episode']}/object_mesh/output_aligned.glb"]
        paths += [root / f"segmentation/frames/{i:06d}.jpg" for i in range(ep["frames"])]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
        episodes.append({**ep, "source": str(source.relative_to(ROOT)),
                         "stable": str(stable.relative_to(ROOT)), "input_sha256": hashes,
                         "episode16_historical_reconstruction_reused": ep["episode_index"] == 16})
        print("FROZEN_INPUT", ep["episode"], ep["frames"], flush=True)
    from v2d_syilen.occlusion_motion_anchor import MotionConfig
    manifest = {
        "schema": "track1_occlusion_v7.v1", "episodes": episodes,
        "source_sha256": {str(p): digest(ROOT / p) for p in source_files()},
        "frame_protocol_sha256": digest(protocol), "sample_sha256": digest(sample),
        "settings": {"motion": MotionConfig().dictionary(), "window": 50, "stride": 25,
                     "steps": 12, "v7_batch": 8, "candidate_batch": 4},
        "candidate_policy": "Recompute IoU v2 for every episode, using original and stable v1. V7 gates every candidate by motion before accepting it.",
        "reference_inputs": False, "kaggle_upload": False,
        "known_limit": "Long partial occlusion can hold depth too conservatively; v7 development regressions are disclosed, not tuned on these test episodes.",
        "reconstruction_reuse": "All 30 monocular reconstructions and mask repairs reused. All 30 object trajectories rerun. Episode 16 retains the archived historical source chain.",
    }
    dest = OUT / "run_manifest.json"
    if dest.exists() and read(dest) != manifest:
        raise ValueError("Existing run differs; create a new version")
    write(dest, manifest)
    write(OUT / "inventory.json", inventory)
    print("FROZEN", digest(dest), flush=True)


def verify(manifest, ep=None):
    check_hashes(manifest["source_sha256"])
    if digest(ROOT / "configs/track1_test_frames_v1.json") != manifest["frame_protocol_sha256"]:
        raise ValueError("Frame protocol changed")
    if ep is not None:
        check_hashes(ep["input_sha256"])


def validate_result(ep, run):
    import numpy as np
    from v2d_syilen.submission_checks import check_episode
    from v2d_syilen.occlusion_submission_checks import check_motion
    report = read(run / "report.json")
    for key in ["final_rerender_check", "all_observed_frames_pass_evidence_guard", "hard_motion_guard_passed"]:
        if report.get(key) is not True:
            raise ValueError("Failed final guard: " + key)
    if report["status"] != "complete" or report["score_final"]["total"] > report["score_seed"]["total"] + 1e-8:
        raise ValueError("Incomplete/nonmonotone refinement")
    cfg = read(run / "config.json")
    check_hashes(cfg["source_sha256"])
    check_hashes(cfg["input_sha256"])
    protocol = read(ROOT / "configs/track1_test_frames_v1.json")
    scored = next(e["scored_frames"] for e in protocol["episodes"] if e["episode_index"] == ep["episode_index"])
    with np.load(run / "optimized.npz") as data, np.load(ROOT / ep["source"] / "optimized.npz") as base, np.load(run / "seed.npz") as seed:
        checks = check_episode(data, ep["frames"], scored, baseline=base)
        checks.update(check_motion(data, seed, ep["fps"], cfg["motion"],
                                   read(run / "motion_decisions.json")))
    return checks


def run_queue(selection, pack, commit_url):
    manifest = read(OUT / "run_manifest.json")
    verify(manifest)
    status_path = OUT / "status.json"
    status = read(status_path) if status_path.exists() else {"started_unix": time.time(), "episodes": {}}
    status.update(status="running", pid=os.getpid())
    write(status_path, status)

    def invoke(ep, stage, args):
        status.update(current_episode=ep["episode"], current_stage=stage, updated_unix=time.time())
        write(status_path, status)
        folder = OUT / "logs" / ep["episode"]
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, *map(str, args)]
        record = {"command": command, "started_unix": time.time(), "status": "running"}
        write(folder / (stage + ".json"), record)
        print("START", ep["episode"], stage, flush=True)
        with (folder / (stage + ".log")).open("a") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="passed" if result.returncode == 0 else "failed",
                      returncode=result.returncode, elapsed_seconds=time.time()-record["started_unix"])
        write(folder / (stage + ".json"), record)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {folder / (stage + '.log')}")
        print("DONE", ep["episode"], stage, flush=True)

    failures = []
    for i in selection:
        ep = next(e for e in manifest["episodes"] if e["episode_index"] == i)
        name = ep["episode"]
        try:
            verify(manifest, ep)
            dest = OUT / "episodes" / name
            candidate = dest / "iou_candidates"
            result = dest / "occlusion"
            source, stable = ROOT / ep["source"], ROOT / ep["stable"]
            settings = manifest["settings"]
            common = ["--window", settings["window"], "--stride", settings["stride"], "--steps", settings["steps"]]
            if not (candidate / "report.json").exists():
                extra = []
                if candidate.exists() and any(candidate.iterdir()):
                    archived = dest / ("iou_interrupted_" + str(time.time_ns()))
                    candidate.rename(archived)
                    if (archived / "tracking_checkpoint.npz").exists():
                        extra = ["--resume-from", archived]
                invoke(ep, "candidates", [ROOT / "scripts/refine_object_iou_priority.py",
                       "--source", source, "--alternate", stable, "--out", candidate,
                       "--batch", settings["candidate_batch"], *common, *extra])
            candidate_report = read(candidate / "report.json")
            if candidate_report["status"] != "complete" or not candidate_report["final_rerender_check"]:
                raise ValueError("Candidate generation incomplete")
            args = [ROOT / "scripts/fit_occlusion_motion_anchored.py", "--source", source,
                    "--v1", stable, "--v2", candidate, "--out", result,
                    "--batch", settings["v7_batch"], *common]
            if not (result / "seed.npz").exists():
                if result.exists() and any(result.iterdir()):
                    result.rename(dest / ("prepare_interrupted_" + str(time.time_ns())))
                invoke(ep, "prepare", [*args, "--stage", "prepare"])
            if not (result / "report.json").exists():
                invoke(ep, "refine", [*args, "--stage", "refine"])
                invoke(ep, "finish", [*args, "--stage", "finish"])
            checks = validate_result(ep, result)
            if not (result / "video_validation.json").exists():
                invoke(ep, "video", [ROOT / "scripts/visualize_track1_occlusion_v7.py", "--episode", i])
            verify(manifest, ep)
            report = read(result / "report.json")
            status["episodes"][name] = {"status": "complete", "checks": checks,
                "prediction_sha256": digest(result / "optimized.npz"), "metrics": report["metrics"],
                "state_counts": report["state_counts"],
                "unresolved_motion_fit_frames": report["unresolved_motion_fit_frames"]}
            write(status_path, status)
            print("EPISODE_COMPLETE", name, flush=True)
        except Exception as exc:
            failures.append(name)
            status["episodes"][name] = {"status": "failed", "error": str(exc)}
            write(status_path, status)
            print("FAILED", name, str(exc), flush=True)
    complete = sum(s["status"] == "complete" for s in status["episodes"].values())
    status.update(status="inference_complete" if complete == 30 else "failed" if failures else "partial",
                  complete=complete, failed=failures, updated_unix=time.time())
    write(status_path, status)
    if complete == 30 and pack:
        if not commit_url:
            commit_url = read(OUT / "code_commit.json")["url"]
        command = [str(ROOT / ".venv-submission/bin/python"), str(ROOT / "scripts/pack_track1_occlusion_v7.py"),
                   "--commit-url", commit_url]
        status["status"] = "packing"
        write(status_path, status)
        result = subprocess.run(command, cwd=ROOT)
        status.update(status="ready_for_user_submission" if result.returncode == 0 else "pack_failed",
                      updated_unix=time.time())
        write(status_path, status)
        if result.returncode:
            raise SystemExit(result.returncode)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "run"], required=True)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--commit-url")
    args = parser.parse_args()
    selection = [4, 16] + [i for i in range(30) if i not in (4, 16)] if args.episodes == "all" else [int(v) for v in args.episodes.split(",")]
    if not selection or len(set(selection)) != len(selection) or any(i not in range(30) for i in selection):
        parser.error("Expected unique episode IDs in 0..29")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        # A second invocation must fail, not wait and later duplicate the queue.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "freeze":
            freeze()
        else:
            run_queue(selection, args.pack, args.commit_url)


if __name__ == "__main__":
    main()
