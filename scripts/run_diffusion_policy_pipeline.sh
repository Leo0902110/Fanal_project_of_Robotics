#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEMO_DIR="${DEMO_DIR:-data/demos/pickcube_dp}"
RUN_DIR="${RUN_DIR:-runs/dp_mvp}"
RESULT_DIR="${RESULT_DIR:-results/dp_mvp}"
NUM_EPISODES="${NUM_EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-120}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
HORIZON="${HORIZON:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
SEED="${SEED:-42}"

echo "[DP 1/5] Environment check"
"${PYTHON_BIN}" scripts/check_pipeline_env.py

echo "[DP 2/5] Collect successful scripted demos"
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
  --min-mean-success 0.8 \
  --require-policy scripted

echo "[DP 3/5] Train conditional Diffusion Policy"
"${PYTHON_BIN}" scripts/train_diffusion_policy.py \
  --demo-dir "${DEMO_DIR}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --horizon "${HORIZON}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --seed "${SEED}"

echo "[DP 4/5] Evaluate Diffusion Policy"
"${PYTHON_BIN}" scripts/evaluate_diffusion_policy.py \
  --checkpoint "${RUN_DIR}/diffusion_policy.pt" \
  --num-episodes 5 \
  --max-steps "${MAX_STEPS}" \
  --scene pseudo_blur \
  --use-active-probe \
  --output-dir "${RESULT_DIR}/diffusion_eval"

echo "[DP 5/5] Compare against sine fallback"
"${PYTHON_BIN}" scripts/evaluate_fallback.py \
  --num-episodes 5 \
  --max-steps "${MAX_STEPS}" \
  --scene pseudo_blur \
  --use-active-probe \
  --output-dir "${RESULT_DIR}/fallback_eval"

echo "Diffusion Policy pipeline complete."
echo "Key outputs:"
echo "  ${RUN_DIR}/diffusion_policy.pt"
echo "  ${RUN_DIR}/diffusion_metrics.json"
echo "  ${RESULT_DIR}/diffusion_eval/diffusion_eval_results.csv"
