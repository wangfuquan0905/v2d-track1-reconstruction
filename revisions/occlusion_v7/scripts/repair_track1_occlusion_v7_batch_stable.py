"""Repair render-batch-dependent v7 failures, preserving the 21 finished results."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from run_track1_occlusion_v7 import (OUT, read, write, digest, verify, check_hashes,
                                    validate_result, run_queue)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", action="store_true")
    p.add_argument("--commit-url")
    a = p.parse_args()
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(OUT / "run_manifest.json")
        verify(manifest)
        state_path = OUT / "status.json"
        state = read(state_path)
        plan_path = OUT / "batch_stable_repair_manifest.json"
        if not plan_path.exists():
            failed = [ep for ep, row in state["episodes"].items() if row["status"] == "failed"]
            if not failed:
                raise ValueError("No failed episodes to repair")
            # Exercise the largest reported mismatch first.
            failed.sort(key=lambda name: (name != "episode_000002", name))
            preserved = {str((OUT / "episodes" / name / "occlusion/optimized.npz").relative_to(ROOT)):
                         digest(OUT / "episodes" / name / "occlusion/optimized.npz")
                         for name, row in state["episodes"].items() if row["status"] == "complete"}
            sources = [Path(__file__).resolve(), ROOT / "scripts/fit_occlusion_v7_batch_stable.py",
                       ROOT / "scripts/occlusion_scene_fixed_batch.py"]
            write(plan_path, {
                "started_unix": time.time(), "episodes": failed,
                "original_manifest_sha256": digest(OUT / "run_manifest.json"),
                "preserved_completed_predictions": preserved,
                "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sources},
                "reason": "GPU einsum returns one-ULP-different transformed vertices for render batch sizes 2--5 versus 1/6--8; silhouette differences exceed strict rerender tolerance. Reusing a Scene alone did not fix it.",
                "change": "Pad render/loss chunks to eight slots and discard padded outputs; share Scene across v7 stages; rerun only failed trajectories.",
                "loss_motion_and_tolerances_unchanged": True, "reference_inputs": False,
            })
        plan = read(plan_path)
        check_hashes(plan["source_sha256"])
        check_hashes(plan["preserved_completed_predictions"])
        if digest(OUT / "run_manifest.json") != plan["original_manifest_sha256"]:
            raise ValueError("Original manifest changed")
        state.update(status="repairing", repair_manifest=str(plan_path))
        write(state_path, state)

        def invoke(ep, stage, args):
            state.update(current_episode=ep, current_stage=stage, updated_unix=time.time())
            write(state_path, state)
            logdir = OUT / "logs" / ep
            command = [sys.executable, *map(str, args)]
            record = {"command": command, "started_unix": time.time(), "status": "running"}
            write(logdir / (stage + ".json"), record)
            print("REPAIR_START", ep, stage, flush=True)
            with (logdir / (stage + ".log")).open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, status="passed" if result.returncode == 0 else "failed",
                          elapsed_seconds=time.time()-record["started_unix"])
            write(logdir / (stage + ".json"), record)
            if result.returncode:
                raise RuntimeError(f"{ep} {stage} failed; see {logdir / (stage + '.log')}")
            print("REPAIR_DONE", ep, stage, flush=True)

        failures = []
        for name in plan["episodes"]:
            ep = next(e for e in manifest["episodes"] if e["episode"] == name)
            dest = OUT / "episodes" / name
            run = dest / "occlusion"
            try:
                verify(manifest, ep)
                check_hashes(plan["source_sha256"])
                report = read(run / "report.json") if (run / "report.json").exists() else {}
                if not report.get("fixed_batch_render_verification"):
                    if run.exists():
                        archive = dest / ("occlusion_before_batch_stable_" + str(time.time_ns()))
                        run.rename(archive)
                        write(archive / "archive_reason.json", {
                            "reason": "Preserved failed/interrupted attempt before a complete fixed-batch rerun.",
                            "prior_seed_sha256": digest(archive / "seed.npz") if (archive / "seed.npz").exists() else None,
                        })
                    settings = manifest["settings"]
                    invoke(name, "batch_stable", [
                        ROOT / "scripts/fit_occlusion_v7_batch_stable.py",
                        "--source", ROOT / ep["source"], "--v1", ROOT / ep["stable"],
                        "--v2", dest / "iou_candidates", "--out", run,
                        "--batch", settings["v7_batch"], "--window", settings["window"],
                        "--stride", settings["stride"], "--steps", settings["steps"]])
                checks = validate_result(ep, run)
                if not (run / "video_validation.json").exists():
                    invoke(name, "repair_video", [ROOT / "scripts/visualize_track1_occlusion_v7.py",
                           "--episode", ep["episode_index"]])
                verify(manifest, ep)
                report = read(run / "report.json")
                state["episodes"][name] = {
                    "status": "complete", "checks": checks,
                    "prediction_sha256": digest(run / "optimized.npz"), "metrics": report["metrics"],
                    "state_counts": report["state_counts"],
                    "unresolved_motion_fit_frames": report["unresolved_motion_fit_frames"],
                    "stage_execution": "fixed_render_batches_shared_scene",
                }
                state["complete"] = sum(v["status"] == "complete" for v in state["episodes"].values())
                state["failed"] = [n for n, v in state["episodes"].items() if v["status"] == "failed"]
                write(state_path, state)
                print("REPAIR_EPISODE_COMPLETE", name, state["complete"], flush=True)
            except Exception as exc:
                failures.append(name)
                state["episodes"][name] = {"status": "failed", "error": str(exc)}
                write(state_path, state)
                print("REPAIR_FAILED", name, str(exc), flush=True)
        check_hashes(plan["preserved_completed_predictions"])
        state.update(status="repair_failed" if failures else "inference_complete",
                     complete=sum(v["status"] == "complete" for v in state["episodes"].values()),
                     failed=failures, updated_unix=time.time())
        write(state_path, state)
        if failures:
            raise SystemExit(1)
        # The original packer rechecks all 30 episodes and every official row.
        run_queue([], a.pack, a.commit_url)


if __name__ == "__main__":
    main()
