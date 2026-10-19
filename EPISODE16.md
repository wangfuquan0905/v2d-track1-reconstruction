# Episode 16 lineage

This episode was completed before the other 29 episodes and reused only after the
input/source hashes in its `run_manifest.json` passed. Its final output hash is
`d7693ade2901f39be6daa373a6a62809949d2f4bc94bd16fb7245f68e861944f`.

1. Public RGB and the retained episode-16 box prompts produced SAM2 masks and a
   SAM 3D Objects mesh. Depth/body/scale, FoundationPose, CoCoNet and the recorded
   300-step CARI4D refinement produced `inference/refined.pth`.
2. Canonicalization produced the native shared-identity MHR baseline in
   `render_observation_v1/canonical_baseline/`.
3. `render_observation_v7_translation_fd` ran 120 steps with finite-difference
   translation image gradients, full human controls, object RGB/silhouette losses,
   acceleration, and nonzero linear/angular velocity penalties.
4. `render_observation_v7b_translation_fd` continued from that output for 120 steps.
5. `render_observation_v8_acc_only` continued for another 120 steps, removing the
   linear/angular velocity penalties. The earlier steps remain part of its history.
6. The historical `pruned_tracking_v4_acc001` used the v8 human/scale and image-guided
   frame-zero initialization, with beam 4 and acceleration weights 0.01/0.01.
7. `safe_windows_alternate_v1` retained only that tracker's rotation/translation.
   The canonical human and object scale 1.0 were restored from the full baseline.
8. The frozen 50-frame, stride-25, 12-step window optimizer selected/refined complete
   candidates and verified the final whole-clip objective by rerendering.

The retained `source_snapshot` files are necessary: the current tracker is a later
version. `provenance/episode16/` includes configuration, commands, hashes and reports;
`legacy/episode16/` includes the historical source snapshots and original upstream
patch. The new `tools/reproduce_episode16.py` driver executes this chain in a
materialized workspace, restores the normal frozen sources afterwards, and records
each replay stage separately.

Historical scripts load a geometric SDF cache located under a directory named
`syilen/upper_limb`. Its penetration loss has weight zero in the three retained
render trials. The driver rebuilds that diagnostic grid from the reconstructed mesh
using the recorded grid bounds; no SyILEN simulator or physics rollout is needed.
The original grid hash and dimensions are recorded in `sdf_spec.json`.

The replay driver is new. Its command construction and archived source hashes are
validated, but the entire legacy GPU chain has not been rerun as part of packaging.
The already completed prediction remains the submitted artifact, checked against
its original hashes; a fresh GPU replay is not claimed to be byte-identical.
