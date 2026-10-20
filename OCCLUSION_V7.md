# Occlusion and motion revision (v7)

This branch reruns object poses for all 30 official test episodes. It reuses the
existing monocular reconstructions, repaired masks, calibrated meshes, cameras,
human geometry and scale. Episode 16 keeps the historical reconstruction lineage
in [EPISODE16.md](EPISODE16.md), but its final object trajectory is recomputed too.
The original `frozen/` archive and previous submission remain intact.

## Changes

Visible object pixels must still be explained even when the person overlaps them.
Unknown regions behind the person are censored in the fitting objective. The
reported joint-render visible IoU remains a separate diagnostic.

Missing observations use the last reliable position unless recent consecutive
observations support a consistent velocity. Trusted velocity extrapolation lasts
at most 0.2 seconds. Partial observations allow local lateral and rotational
updates; depth follows the motion prediction. Reappearance requires three
consistent local observations before depth is unlocked. Original first-frame
position remains a candidate, without a distant future-frame replacement.

Motion gates are checked before selecting image-fit candidates and at every
window boundary: 2 m/s translational speed, 180 degrees/s angular speed and a
1 cm refinement radius around the causal seed. Clear-frame depth uses a soft
reference only when the original and stable-v1 positions agree locally.
The exact settings and input hashes are in
`provenance/occlusion_v7/run_manifest.json`.

The IoU-only v2 candidates are recomputed for every episode, then screened by
v7's motion gates. Their unconstrained depths are not accepted merely for
achieving higher overlap. All v7 fitting settings match the completed development
version; this rerun does not tune against test leaderboard scores.

## Known limitations

The public FORM-HOI checks are development regressions against multi-view
reconstruction pseudo-GT, not a held-out benchmark or Kaggle scores. On the
cardboard scene, the three-frame missing interval became stationary and the
maximum adjacent position change fell from 3.884 m (v5) to 0.0366 m (v7).
Its mean object surface error fell from 132.35 to 14.48 cm.

The other two development scenes regressed from 4.86 to 5.49 cm and 7.27 to
11.91 cm. During long partial occlusion, depth can remain too conservative
and lag real motion. Camera motion is not separately compensated. This is
an experimental resubmission, with no claim that every scene or metric improves.
The underlying reports are archived under `provenance/occlusion_v7/`.

## Reproduce and run

First reconstruct the original inputs with `pipeline.py`, as described in the
main README, or use an existing completed producing workspace. Then install the
revision and verify it:

```bash
python occlusion_v7.py install --workspace /path/to/V2D
python occlusion_v7.py verify --workspace /path/to/V2D

# Single episode (the full manifest still freezes all 30 inputs).
python occlusion_v7.py run --workspace /path/to/V2D --episodes 4

# All 30; episode 04 and 16 run first.
python occlusion_v7.py run --workspace /path/to/V2D --episodes all
python occlusion_v7.py status --workspace /path/to/V2D
```

The launcher refuses unrecognized local changes. A process lock prevents two
queues from producing the same outputs. Interrupted candidate/prepare directories
are retained under a timestamped name; refinement resumes at completed windows.
Any failed episode remains failed, and prevents full submission packaging.

The first full queue completed 21 episodes; nine stopped before refinement when
a full-sequence rerender differed slightly from the preparation observations.
The maximum reported loss difference was 0.00003463. The GPU vertex contraction
returned results differing by one float32 unit (0.0000002384 m) for candidate
batch sizes 2--5 versus 1/6--8. This altered a few silhouette-edge pixels.
Keeping one Scene alone did not resolve the mismatch.

The repair pads render/loss batches to eight slots with repeated inputs and
discards padded outputs. Real poses, frame IDs, mesh geometry, loss definitions
and strict acceptance tolerances remain unchanged. It also keeps one Scene
through all three stages:

```bash
python occlusion_v7.py repair --workspace /path/to/V2D --pack \
  --commit-url https://github.com/OWNER/REPO/commit/FULL_40_CHARACTER_SHA
```

The repair keeps all finished predictions byte-for-byte, archives failed
attempts, and reruns only the failed v7 trajectories using the existing candidate
poses. It uses the unchanged frozen fitter and motion rules. Each repaired result
records its wrapper source hashes and checks another pair of identical renders
after finishing. The diagnostic in
`provenance/occlusion_v7/batch_stability_diagnostic.json` tests all requested
batch sizes 1--8 at two affected frames; fixed-batch observations are identical.
If interrupted, the affected episode's v7 stages restart together.

All new outputs are under `outputs/track1_occlusion_v7/`. Each episode contains
its IoU candidates, v7 source snapshot, visibility decisions, seed, window
checkpoints, final pose, report and a full comparison video against submitted v1.
Masks, video frames, source code and reused reconstruction inputs are hashed.

Run the archived regression tests with the reconstruction environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /path/to/V2D/.venv-cari4d/bin/python -m pytest \
  -o 'pythonpath=revisions/occlusion_v7 revisions/occlusion_v7/scripts' \
  revisions/occlusion_v7/tests -q
```

## Create the new submission

Publish the producing source commit, then supply its full URL:

```bash
python occlusion_v7.py pack --workspace /path/to/V2D \
  --commit-url https://github.com/OWNER/REPO/commit/FULL_40_CHARACTER_SHA
```

Alternatively add `--pack --commit-url ...` to the 30-episode run command.
The packer independently rechecks missing-frame holds, partial-depth locks,
speed bounds, native MHR parameters, fixed human/scale, final rerenders and all
video frame counts. It uses the pinned official packer and mesh-budget routine,
then compares every packed parameter and mesh with the staged arrays and checks
all 9,877 official scored frames. A temporary Parquet is promoted to the final
name only after the audit passes:

`outputs/track1_occlusion_v7/track1_occlusion_v7.parquet`

`delivery_manifest.json` records its SHA-256 and producing commit. No Kaggle
upload is performed. A running queue or partial report is not a finished
submission.
