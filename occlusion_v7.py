"""Install, verify and run the v7 revision alongside the archived v1 pipeline."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

from pipeline import REPO, adapt, digest, read, write

OVERLAY = REPO / "revisions/occlusion_v7"


def content(relative, work):
    text = (OVERLAY / relative).read_text()
    runtime = work / "runtime.local.json"
    return adapt(text, work, read(runtime)) if runtime.exists() else text


def verify(work=None):
    manifest = read(OVERLAY / "source_manifest.json")["files"]
    for relative, expected in manifest.items():
        path = OVERLAY / relative
        if digest(path) != expected:
            raise ValueError("Archived v7 source changed: " + relative)
        if path.suffix == ".py":
            ast.parse(path.read_text())
        if work and (work / relative).read_text() != content(relative, work):
            raise ValueError("Runtime source differs: " + relative)
    print(json.dumps({"v7_source_verification": "passed", "files": len(manifest),
                      "workspace": str(work) if work else None}))


def install(work):
    verify()
    if not (work / ".venv-cari4d/bin/python").exists():
        raise ValueError("First materialize the reconstruction workspace described in README.md")
    manifest = read(OVERLAY / "source_manifest.json")["files"]
    planned = {}
    for relative in manifest:
        dest = work / relative
        text = content(relative, work)
        if dest.exists() and dest.read_text() != text:
            baseline = REPO / "frozen" / relative
            runtime = work / "runtime.local.json"
            old = baseline.read_text() if baseline.exists() else None
            if old is not None and runtime.exists():
                old = adapt(old, work, read(runtime))
            if old is None or dest.read_text() != old:
                raise ValueError("Refusing to overwrite an unrecognized local change: " + str(dest))
        planned[dest] = text
    for dest, text in planned.items():
        if not dest.exists() or dest.read_text() != text:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
    write(work / "occlusion_v7_materialization.json",
          {"source_repository": str(REPO),
           "source_sha256": {str(p.relative_to(work)): digest(p) for p in planned}})
    verify(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--workspace", type=Path)
    i = sub.add_parser("install")
    i.add_argument("--workspace", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--workspace", type=Path, required=True)
    r.add_argument("--episodes", default="all")
    r.add_argument("--pack", action="store_true")
    r.add_argument("--commit-url")
    f = sub.add_parser("repair")
    f.add_argument("--workspace", type=Path, required=True)
    f.add_argument("--pack", action="store_true")
    f.add_argument("--commit-url")
    k = sub.add_parser("pack")
    k.add_argument("--workspace", type=Path, required=True)
    k.add_argument("--commit-url", required=True)
    s = sub.add_parser("status")
    s.add_argument("--workspace", type=Path, required=True)
    args = p.parse_args()
    work = args.workspace.resolve() if getattr(args, "workspace", None) else None
    if args.command == "verify":
        verify(work)
    elif args.command == "install":
        install(work)
    elif args.command == "status":
        status = read(work / "outputs/track1_occlusion_v7/status.json")
        print(json.dumps({k:v for k,v in status.items() if k != "episodes"}, indent=2))
        print(json.dumps({ep:row["status"] for ep,row in status["episodes"].items()}, indent=2))
    else:
        verify(work)
        if args.command == "run":
            command = [str(work / ".venv-cari4d/bin/python"),
                       str(work / "scripts/run_track1_occlusion_v7.py")]
            subprocess.run([*command, "--stage", "freeze"], cwd=work, check=True)
            extra = ["--pack"] if args.pack else []
            if args.commit_url:
                extra += ["--commit-url", args.commit_url]
            subprocess.run([*command, "--stage", "run", "--episodes", args.episodes, *extra],
                           cwd=work, check=True)
        elif args.command == "repair":
            extra = ["--pack"] if args.pack else []
            if args.commit_url:
                extra += ["--commit-url", args.commit_url]
            subprocess.run([str(work / ".venv-cari4d/bin/python"),
                            str(work / "scripts/repair_track1_occlusion_v7_batch_stable.py"), *extra],
                           cwd=work, check=True)
        else:
            subprocess.run([str(work / ".venv-submission/bin/python"),
                            str(work / "scripts/pack_track1_occlusion_v7.py"),
                            "--commit-url", args.commit_url], cwd=work, check=True)


if __name__ == "__main__":
    main()
