#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-120}"
RUN_BC_PIPELINE="${RUN_BC_PIPELINE:-0}"
RESULT_ROOT="${RESULT_ROOT:-results/final_submission}"

echo "[Final 1/4] Environment check"
"${PYTHON_BIN}" scripts/check_pipeline_env.py

echo "[Final 2/4] Scripted MVP baselines"
"${PYTHON_BIN}" main.py \
  --mode mvp \
  --scene clean \
  --policy scripted \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${RESULT_ROOT}/clean"

"${PYTHON_BIN}" main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${RESULT_ROOT}/pseudo_blur"

"${PYTHON_BIN}" main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --use-active-probe \
  --obs-mode rgbd \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --no-video \
  --output-dir "${RESULT_ROOT}/active_probe"

echo "[Final 3/4] Validate active-probe result"
"${PYTHON_BIN}" scripts/validate_outputs.py \
  mvp-results \
  --path "${RESULT_ROOT}/active_probe/mvp_results.json" \
  --min-success 0.1 \
  --require-policy scripted

echo "[Final 4/4] Plot summary"
"${PYTHON_BIN}" scripts/plot_results.py \
  --results-dir "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/mvp_performance_chart.png" \
  --decision-output "${RESULT_ROOT}/mvp_decision_flow_chart.png"

if [[ "${RUN_BC_PIPELINE}" == "1" ]]; then
  echo "[Optional] Run BC pipeline"
  DEMO_DIR="${RESULT_ROOT}/demos" \
  RUN_DIR="${RESULT_ROOT}/bc_runs" \
  RESULT_DIR="${RESULT_ROOT}/bc_results" \
  MAX_STEPS="${MAX_STEPS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_bc_pipeline.sh
fi

echo "Final submission pipeline complete."
echo "Primary deliverables:"
echo "  ${RESULT_ROOT}/clean/mvp_results.csv"
echo "  ${RESULT_ROOT}/pseudo_blur/mvp_results.csv"
echo "  ${RESULT_ROOT}/active_probe/mvp_results.csv"
echo "  ${RESULT_ROOT}/mvp_performance_chart.png"
echo "  ${RESULT_ROOT}/mvp_decision_flow_chart.png"
