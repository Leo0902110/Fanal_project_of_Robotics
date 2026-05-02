#!/usr/bin/env bash
set -euo pipefail

SEED=42
MAX_STEPS=120
ROOT_DIR="results/local_mvp_rgbd"

mkdir -p "${ROOT_DIR}"

python main.py \
  --mode mvp \
  --scene clean \
  --policy scripted \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${ROOT_DIR}/clean"

python main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${ROOT_DIR}/pseudo_blur"

python main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --use-active-probe \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${ROOT_DIR}/active_probe"

python scripts/plot_results.py --results-dir "${ROOT_DIR}"
