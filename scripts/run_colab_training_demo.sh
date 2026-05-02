#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_EPISODES="${NUM_EPISODES:-30}"
MAX_STEPS="${MAX_STEPS:-100}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-25}"
SEED="${SEED:-42}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"
RUN_MVP_BASELINES="${RUN_MVP_BASELINES:-1}"
RENDER_DEMO_VIDEO="${RENDER_DEMO_VIDEO:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

DEMO_DIR="${DEMO_DIR:-data/demos/colab_vtabr}"
RUN_DIR="${RUN_DIR:-runs/colab_vtabr}"
RESULT_DIR="${RESULT_DIR:-results/colab_vtabr}"

echo "[Colab 0/6] Python"
"${PYTHON_BIN}" --version

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  echo "[Colab 1/6] Install dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r requirements.txt
  "${PYTHON_BIN}" -m pip install -r requirements-A.txt
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
PY
else
  echo "[Colab 1/6] Skip install"
fi

echo "[Colab 2/6] Environment check"
"${PYTHON_BIN}" scripts/check_pipeline_env.py

if [[ "${RUN_SMOKE_FIRST}" == "1" ]]; then
  echo "[Colab 3/6] Smoke test"
  "${PYTHON_BIN}" main.py \
    --mode smoke \
    --obs-mode state \
    --max-steps 30 \
    --no-video \
    --output-dir "${RESULT_DIR}/smoke_state"
fi

if [[ "${RUN_MVP_BASELINES}" == "1" ]]; then
  echo "[Colab 4/6] MVP baselines"
  "${PYTHON_BIN}" main.py \
    --mode mvp \
    --obs-mode rgbd \
    --policy scripted \
    --max-steps "${MAX_STEPS}" \
    --no-video \
    --output-dir "${RESULT_DIR}/mvp_rgbd" || \
  "${PYTHON_BIN}" main.py \
    --mode mvp \
    --obs-mode state \
    --policy scripted \
    --max-steps "${MAX_STEPS}" \
    --no-video \
    --output-dir "${RESULT_DIR}/mvp_state"
  if [[ -f "${RESULT_DIR}/mvp_rgbd/mvp_results.json" ]]; then
    "${PYTHON_BIN}" scripts/validate_outputs.py \
      mvp-results \
      --path "${RESULT_DIR}/mvp_rgbd/mvp_results.json" \
      --min-success 0.1 \
      --require-policy scripted
  else
    "${PYTHON_BIN}" scripts/validate_outputs.py \
      mvp-results \
      --path "${RESULT_DIR}/mvp_state/mvp_results.json" \
      --min-success 0.1 \
      --require-policy scripted
  fi
fi

echo "[Colab 5/6] BC pipeline"
NUM_EPISODES="${NUM_EPISODES}" \
MAX_STEPS="${MAX_STEPS}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
SEED="${SEED}" \
DEMO_DIR="${DEMO_DIR}" \
RUN_DIR="${RUN_DIR}" \
RESULT_DIR="${RESULT_DIR}" \
bash scripts/run_bc_pipeline.sh

if [[ "${RENDER_DEMO_VIDEO}" == "1" ]]; then
  echo "[Colab 6/6] Render active-probe demo video"
  "${PYTHON_BIN}" main.py \
    --mode mvp \
    --scene pseudo_blur \
    --policy scripted \
    --use-active-probe \
    --obs-mode rgbd \
    --max-steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    --output-dir "${RESULT_DIR}/render_active_probe" || true
else
  echo "[Colab 6/6] Skip demo video render"
fi

echo "Colab training demo complete."
echo "Key outputs:"
echo "  ${DEMO_DIR}"
echo "  ${RUN_DIR}/bc_policy.pt"
echo "  ${RUN_DIR}/bc_metrics.json"
echo "  ${RESULT_DIR}/bc_eval/bc_eval_results.csv"
echo "  ${RESULT_DIR}/fallback_eval/fallback_eval_results.csv"
