#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-results/render_active_probe}"
MAX_STEPS="${MAX_STEPS:-120}"
SEED="${SEED:-42}"

echo "[Render 1/2] Check Python environment"
"${PYTHON_BIN}" scripts/check_pipeline_env.py

echo "[Render 2/2] Run ManiSkill RGBD active-probe video export"
"${PYTHON_BIN}" main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --use-active-probe \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --output-dir "${RESULT_DIR}"

echo "Render attempt complete. Confirm these fields in ${RESULT_DIR}/mvp_results.csv:"
echo "  env_backend=maniskill"
echo "  fallback_used=False"
echo "  video_path=${RESULT_DIR}/active_probe.mp4"
