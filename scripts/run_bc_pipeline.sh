#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEMO_DIR="${DEMO_DIR:-data/demos/pickcube_vtabr}"
RUN_DIR="${RUN_DIR:-runs/bc_vtabr}"
RESULT_DIR="${RESULT_DIR:-results/bc_vtabr}"
NUM_EPISODES="${NUM_EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-80}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
SEED="${SEED:-42}"

echo "[1/5] Environment check"
"${PYTHON_BIN}" scripts/check_pipeline_env.py

echo "[2/5] Collect demos"
rm -rf "${RUN_DIR}" "${RESULT_DIR}"
"${PYTHON_BIN}" scripts/collect_demos.py \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --scene pseudo_blur \
  --policy scripted \
  --use-active-probe \
  --clear-output-dir \
  --output-dir "${DEMO_DIR}"
"${PYTHON_BIN}" scripts/validate_outputs.py \
  manifest \
  --path "${DEMO_DIR}/manifest.json" \
  --min-mean-success 0.5 \
  --require-policy scripted

echo "[3/5] Train BC"
"${PYTHON_BIN}" scripts/train_bc.py \
  --demo-dir "${DEMO_DIR}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --seed "${SEED}"

echo "[4/5] Evaluate BC"
"${PYTHON_BIN}" scripts/evaluate_bc.py \
  --checkpoint "${RUN_DIR}/bc_policy.pt" \
  --num-episodes 5 \
  --max-steps "${MAX_STEPS}" \
  --scene pseudo_blur \
  --use-active-probe \
  --output-dir "${RESULT_DIR}/bc_eval"

echo "[5/5] Evaluate fallback"
"${PYTHON_BIN}" scripts/evaluate_fallback.py \
  --num-episodes 5 \
  --max-steps "${MAX_STEPS}" \
  --scene pseudo_blur \
  --use-active-probe \
  --output-dir "${RESULT_DIR}/fallback_eval"

echo "Pipeline complete."
