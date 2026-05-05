#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-data/demos/pickcube_wristcam_level2_camera_tactile}"
RUN_DIR="${RUN_DIR:-runs/bc_wristcam_level2_camera_tactile}"
EVAL_DIR="${EVAL_DIR:-results/bc_wristcam_level2_camera_tactile_eval}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-80}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-80}"
SEED="${SEED:-9}"
SENSOR_WIDTH="${SENSOR_WIDTH:-128}"
SENSOR_HEIGHT="${SENSOR_HEIGHT:-128}"
PSEUDO_BLUR_PROFILE="${PSEUDO_BLUR_PROFILE:-dark}"
PSEUDO_BLUR_SEVERITY="${PSEUDO_BLUR_SEVERITY:-1.0}"
DEVICE="${DEVICE:-cuda}"

echo "[Level 2 1/3] Collect camera-only hand-camera demonstrations"
python scripts/collect_demos.py \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --obs-mode rgbd \
  --robot-uids panda_wristcam \
  --training-camera hand_camera \
  --camera-only \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --policy scripted \
  --use-active-probe \
  --clear-output-dir \
  --output-dir "${DEMO_DIR}"

echo "[Level 2 2/3] Train BC on hand-camera RGB-D + tactile features"
python scripts/train_bc.py \
  --demo-dir "${DEMO_DIR}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size 256 \
  --hidden-dim 512 \
  --device "${DEVICE}" \
  --feature-set default \
  --log-every 10

echo "[Level 2 3/3] Evaluate without oracle geometry and without grasp assist"
python scripts/evaluate_bc.py \
  --checkpoint "${RUN_DIR}/bc_policy.pt" \
  --num-episodes "${EVAL_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "$((SEED + 1000))" \
  --obs-mode rgbd \
  --robot-uids panda_wristcam \
  --training-camera hand_camera \
  --camera-only \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --use-active-probe \
  --save-video \
  --camera hand_camera \
  --require-wrist \
  --video-fps 6 \
  --output-dir "${EVAL_DIR}" \
  --device "${DEVICE}"

echo "Done."
echo "Level 2 demos: ${DEMO_DIR}/manifest.json"
echo "Level 2 metrics: ${RUN_DIR}/bc_metrics.json"
echo "Level 2 eval CSV: ${EVAL_DIR}/bc_eval_results.csv"
