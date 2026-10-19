#!/usr/bin/env bash
set -euo pipefail
cd /home/lab-2/Research/V2D
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-cari4d/bin/python scripts/refine_render_observation.py \
 --episode episode_000016 --start 0 --end 359 --steps 120 --batch 4 \
 --human-controls full --lobs-weight 10 --penetration-weight 0 --loss-preset object-observation-acc \
 --orientation-search --rotation-temporal so3 --include-first-orientations \
 --angular-velocity-weight 0 --angular-acceleration-weight 1 \
 --angular-velocity-scale-deg 90 --angular-acceleration-scale-deg 360 \
 --render-supersample 4 --translation-image-gradient finite-difference --translation-fd-step .002 \
 --linear-velocity-weight 0 --linear-acceleration-weight 1 \
 --linear-velocity-scale .5 --linear-acceleration-scale 1 \
 --baseline outputs/episode_000016/render_observation_v1/canonical_baseline/native.npz \
 --bundle outputs/episode_000016/render_observation_v1/canonical_baseline/canonical.pth \
 --sdf outputs/episode_000016/syilen/upper_limb/object_sdf.npz \
 --resume-optimized outputs/episode_000016/render_observation_v7b_translation_fd/optimized.npz \
 --out outputs/episode_000016/render_observation_v8_acc_only
