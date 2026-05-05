#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-data/demos/pickcube_wristcam_hand_camera}"
RUN_DIR="${RUN_DIR:-runs/bc_wristcam_hand_camera}"
EVAL_DIR="${EVAL_DIR:-results/bc_wristcam_hand_camera_eval}"
NUM_EPISODES="${NUM_EPISODES:-12}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-80}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-25}"
SEED="${SEED:-9}"
SENSOR_WIDTH="${SENSOR_WIDTH:-384}"
SENSOR_HEIGHT="${SENSOR_HEIGHT:-384}"
PSEUDO_BLUR_PROFILE="${PSEUDO_BLUR_PROFILE:-dark}"
PSEUDO_BLUR_SEVERITY="${PSEUDO_BLUR_SEVERITY:-1.0}"
DEVICE="${DEVICE:-cuda}"

echo "[1/3] Collect wrist-camera demonstrations"
python scripts/collect_demos.py \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --obs-mode rgbd \
  --robot-uids panda_wristcam \
  --training-camera hand_camera \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --policy scripted \
  --use-active-probe \
  --clear-output-dir \
  --output-dir "${DEMO_DIR}"

echo "[2/3] Train BC on filtered hand-camera observations"
python scripts/train_bc.py \
  --demo-dir "${DEMO_DIR}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size 64 \
  --device "${DEVICE}"

echo "[3/3] Evaluate BC with the same panda_wristcam/hand_camera observation filter"
python scripts/evaluate_bc.py \
  --checkpoint "${RUN_DIR}/bc_policy.pt" \
  --num-episodes "${EVAL_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "$((SEED + 1000))" \
  --obs-mode rgbd \
  --robot-uids panda_wristcam \
  --training-camera hand_camera \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --use-active-probe \
  --grasp-assist \
  --output-dir "${EVAL_DIR}" \
  --device "${DEVICE}"

echo "Done."
echo "Demo manifest: ${DEMO_DIR}/manifest.json"
echo "BC metrics: ${RUN_DIR}/bc_metrics.json"
echo "BC eval CSV: ${EVAL_DIR}/bc_eval_results.csv"
