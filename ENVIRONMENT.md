# Runtime and model setup

The original run used Linux, one RTX 5090 and CUDA 12.8. Reconstruction and object
generation used separate Python 3.10.19 environments with a shared base installation;
packing used Python 3.13. `environments/*.json` records the installed package versions
in each environment. These are inventories, not a claim that a single pip command
recreates all CUDA extensions. `dependencies.json` records source revisions and patches.

Important reconstruction versions include torch 2.9.1, torchvision 0.24.1,
numpy 2.2.6, scipy 1.15.3, trimesh 4.9.0, nvdiffrast 0.3.4, pytorch3d 0.7.8,
kaolin 0.18.0, h5py 3.16.0 and OpenCV 4.12.0.88. The packing environment must match
its own inventory, particularly numpy, trimesh and fast-simplification: mesh budgeting
is verified against the exact geometry shown in the review viewer.

## Source dependencies

`tools/restore_dependencies.py` checks out each recorded commit and applies the
corresponding binary-safe patch. It fails before modifying an existing directory.
Use the upstream installation instructions at those commits to build CUDA extensions.
The V2D editable modules used here are `common`, `v2d_mesh`, `v2d_mv`, `v2d_moge`,
`v2d_sam3d_body`, `v2d_foundation_pose` and `v2d_cari4d` under
`video_to_data/reconstruction/modules/*/lib`. Install these in the reconstruction
environment; install the pinned SAM2 package there as well. SAM 3D Objects runs in
its own environment with its upstream requirements. Install `pytest` for regression
tests and system `ffmpeg`/`ffprobe` for review media.

For the optional 3D review viewer, run `npm ci --prefix debug_viewer` in the
materialized workspace. The retained lockfile pins Three.js 0.180.0.

The pinned additional Python dependencies are:

```text
MoGe:    https://github.com/microsoft/MoGe.git @ 925b8ed835a7a9cdb7578ba15c658a0afc969030
utils3d: https://github.com/EasternJournalist/utils3d.git @ 3fab839f0be9931dac7c8488eb0e1600c236e183
```

Kaolin and PyTorch3D source commits and any local patches are in `dependencies.json`.
CUDA extensions must be built for the target GPU architecture. The original environments
are recorded without bundling their binaries or unrelated research source trees.

For packing, create a separate Python 3.13 environment and install the unmodified
official kit's `requirements.txt`, matching `environments/submission.json`. Do not
replace its packer with the older kit from the research tree. The archive and every
kit file are hashed in `provenance/official_kit.json`.

## Pretrained models

`models.json` records names, sources, fixed revisions, filenames, byte counts and
SHA-256 hashes. No model weights are included. Use your own authorized Hugging Face
login for gated models; the downloader uses the standard local Hugging Face login
and does not embed or display any account credentials.

```bash
python tools/download_models.py \
  --models /path/to/models \
  --sam3d-objects /path/to/dependencies/sam3d_objects \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt
```

This downloads CARI4D, SAM 3D Body/MHR, SAM 3D Objects, MoGe 2 and SAM2 at the recorded
revisions/hashes. Small SAM3D YAML configuration overlays are retained in
`model_configs/`. The script creates the expected CARI4D model symlinks.

FoundationPose uses the upstream `nvlabs_pytorch` backend, not TensorRT. In the
reconstruction environment, use the pinned upstream downloader:

```bash
python -m v2d.foundation_pose.lib.download_weights \
  --output_dir /path/to/models/cari4d_runtime/foundationpose \
  --backend nvlabs_pytorch
```

This provisions the recorded `2023-10-28-18-33-37` refiner and
`2024-01-11-20-02-45` scorer. Validate them with the model tool's
`--only foundationpose --verify-only` and the same path arguments.

The CARI4D runtime also uses the upstream DINO Torch Hub caches. Restore these
repositories into `models/cari4d_runtime/sam3d_body/torch_home/hub/` using the exact
revisions in `models.json`, named `facebookresearch_dinov2_main` and
`facebookresearch_dinov3_main`. DINOv2 pretrained weights belong in the sibling
`checkpoints/` directory and are available from Meta's public DINOv2 release:

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

Validate these with `--only torch_hub --verify-only`. The DINOv3 backbone used by
SAM 3D Body is carried in its model checkpoint; retain its pinned source cache.

The resulting model layout is:

```text
models/
  mhr_model.pt -> sam-3d-body-dinov3/assets/mhr_model.pt
  cari4d_commercial/2026-08-25-09-35-57/{step200000.pth,resolved_config.yaml}
  sam-3d-body-dinov3/{model.ckpt,model_config.yaml,assets/mhr_model.pt}
  cari4d_runtime/
    cari4d -> ../cari4d_commercial
    foundationpose/nvlabs_pytorch/{2023-10-28-18-33-37,2024-01-11-20-02-45}/
    hf_home/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/<revision>/model.pt
    sam3d_body/
      checkpoints/sam-3d-body-dinov3 -> ../../../sam-3d-body-dinov3
      torch_home/hub/{facebookresearch_dinov2_main,facebookresearch_dinov3_main,checkpoints}/
```

Point `runtime.local.json` to these directories and the three environment executables.
The setup does not need the SyILEN simulator, robot policies, private reference labels
or a copy of the parent research project.
