# V2D Track 1 reconstruction

This branch adds the **v7 occlusion/motion rerun**. See
[OCCLUSION_V7.md](OCCLUSION_V7.md) for its changes, known regressions, single/all-30
commands and separate submission package. Its source lives in
`revisions/occlusion_v7/`; the original v1 archive below remains unchanged.

Code and provenance for the 30-episode `safe_windows_v1` reconstruction submission.
The submitted method combines public RGB reconstruction models with rigid-object
tracking and overlapping-window refinement. It does not run SyILEN dynamics.
The historical Python package name `v2d_syilen` is retained so producing source
hashes remain checkable. Simulation, robot policies and unrelated research
experiments are excluded.

The recorded run covers 16,563 input frames and all 9,877 official scored frames.
Objective improvements reported here are training losses, not competition scores.
Human reconstruction comes from CARI4D. The final optimizer keeps human parameters,
vertices, mesh, camera and object scale fixed, and adjusts only object rotation and
translation with silhouette, RGB, angular-acceleration and linear-acceleration losses.

## What is included

- `frozen/`: the necessary producing scripts, local modules, RGB prompts, repairs,
  mesh/frame contracts, review viewer and regression tests, copied without edits.
- `provenance/`: source/result hashes, original stage commands, validation reports,
  source versions and the exceptional episode-16 history. Absolute paths in these
  records describe the original run; they are not paths a new user must recreate.
- `dependency_patches/` and `dependencies.json`: upstream commits and local patches.
  Upstream repositories and model weights are fetched separately.
- `legacy/episode16/`: exact historical sources used by the retained warm-start chain.
- `pipeline.py`: portable setup, batch entry point, verification and packing.

## Verify the source archive

```bash
python pipeline.py verify
```

For access to the original producing workspace, also verify the full result lineage:

```bash
python pipeline.py verify --source-root /path/to/V2D --results-root /path/to/V2D
```

This checks all 30 predictions, segmentation masks, object meshes, staged arrays,
and the retained episode-16 intermediate hashes. The binary outputs are not stored
in this source repository.

## Prepare an independent runtime

Read [ENVIRONMENT.md](ENVIRONMENT.md) for the three environments, weight layout,
version pins and download entry points. Restore dependencies with:

```bash
python tools/restore_dependencies.py --dest /path/to/dependencies
```

The official kit has a fixed archive SHA-256. If its live download changes, supply
the matching archive with `--kit-archive`; the tool deliberately rejects a different
packer. This command fetches source and patches; it does not install CUDA packages.

Copy `runtime.example.json` to `runtime.local.json`, fill in absolute paths, then:

```bash
python pipeline.py materialize --workspace /path/to/new-workspace --runtime runtime.local.json
```

Materialization copies the frozen code, replaces only machine paths, records the
changes in `materialization.json`, and links your separate model/dependency/runtime
directories. Algorithm source in this repository and the original research results
are preserved. Do not use an existing research directory as the new workspace.

## Reconstruct one episode or all 30

```bash
# Show the stages without running inference.
python pipeline.py run --workspace /path/to/new-workspace --episodes 3 --dry-run

# Reconstruct one ordinary episode.
python pipeline.py run --workspace /path/to/new-workspace --episodes 3

# Reconstruct the complete submission, including episode 16's historical path.
python pipeline.py run --workspace /path/to/new-workspace --episodes all
```

The first run downloads the pinned public RGB release. The driver writes the recorded
RGB box prompts, segments, applies the recorded mask repairs, and invokes the frozen
reconstruction queue. Failed scale-overlap and final-rerender stages activate the
existing registration and topology-cache retries. Failures remain explicit after
three attempts; no placeholder prediction is generated. Partial runs produce a
partial validation report, not a valid full submission.

Episode 16 is intentionally different. It uses the earlier CARI4D reconstruction,
three render-optimization trials, a historical beam tracker, and an adaptation that
restores the canonical baseline human and object scale before the final windows.
See [EPISODE16.md](EPISODE16.md). Regenerating it from a generic batch baseline is not
the provenance of the originally submitted prediction.

## Tests and GPU verification

From the materialized workspace, using its reconstruction environment:

```bash
PYTHONPATH=.:scripts .venv-cari4d/bin/python -m pytest tests -q
```

The repository validation records describe what was actually checked. The portable
launcher is newly extracted; a clean-machine rerun of all pretrained models is not
claimed. GPU/library differences may change numerical predictions; source and original
result hashes identify this submission independently of rerun reproducibility.

## Pack the recorded submission

After publishing the producing source commit, use its full URL:

```bash
python pipeline.py pack --project-root /path/to/original/V2D \
  --commit-url https://github.com/OWNER/REPO/commit/FULL_40_CHARACTER_SHA
```

This command verifies the recorded sources and all result hashes before running the
official packer. It then checks every packed MHR parameter, object transform, scale,
frame index and budgeted mesh against the staged/reviewed data. The output is
`outputs/track1_safe_windows_v1/track1_safe_windows_v1.parquet` in the producing project.
For a newly generated run, use its own producing commit and the materialized
`scripts/prepare_track1_submission.py --commit-url ...` after inspecting its reports.

Packing does not submit to Kaggle. The official Track 1 format is one Parquet file
for all five metrics. This repository does not contain an upload command.

## Upstream sources

[V2D Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/),
[NVIDIA video_to_data](https://github.com/nvidia-isaac/video_to_data),
[SAM 2](https://github.com/facebookresearch/sam2),
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects),
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body),
[MoGe](https://github.com/microsoft/MoGe),
[FoundationPose](https://github.com/NVlabs/FoundationPose).
Each dependency and pretrained model retains its own license and access conditions.
See the notices copied in `notices/`; publication of this archive does not relicense
upstream code or models.
