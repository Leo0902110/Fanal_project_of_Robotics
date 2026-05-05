#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-data/demos/pickcube_level2_vision_tactile}"
RUN_DIR="${RUN_DIR:-runs/vision_tactile_level2_bc}"
EVAL_DIR="${EVAL_DIR:-results/vision_tactile_level2_eval}"
NUM_EPISODES="${NUM_EPISODES:-30}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-80}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
SEED="${SEED:-9}"
SENSOR_WIDTH="${SENSOR_WIDTH:-128}"
SENSOR_HEIGHT="${SENSOR_HEIGHT:-128}"
PSEUDO_BLUR_PROFILE="${PSEUDO_BLUR_PROFILE:-dark}"
PSEUDO_BLUR_SEVERITY="${PSEUDO_BLUR_SEVERITY:-1.0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

echo "[Level 2 CNN 1/3] Collect hand-camera RGB-D + tactile demonstrations"
python scripts/collect_vision_tactile_demos.py \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --robot-uids panda_wristcam \
  --camera hand_camera \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --use-active-probe \
  --clear-output-dir \
  --output-dir "${DEMO_DIR}"

echo "[Level 2 CNN 2/3] Train CNN BC without oracle geometry"
python scripts/train_vision_tactile_bc.py \
  --demo-dir "${DEMO_DIR}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --device "${DEVICE}" \
  --amp \
  --log-every 10

echo "[Level 2 CNN 3/3] Evaluate without oracle geometry and without grasp assist"
python scripts/evaluate_vision_tactile_bc.py \
  --checkpoint "${RUN_DIR}/vision_tactile_bc.pt" \
  --num-episodes "${EVAL_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "$((SEED + 1000))" \
  --robot-uids panda_wristcam \
  --camera hand_camera \
  --sensor-width "${SENSOR_WIDTH}" \
  --sensor-height "${SENSOR_HEIGHT}" \
  --scene pseudo_blur \
  --pseudo-blur-profile "${PSEUDO_BLUR_PROFILE}" \
  --pseudo-blur-severity "${PSEUDO_BLUR_SEVERITY}" \
  --use-active-probe \
  --save-video \
  --video-fps 6 \
  --output-dir "${EVAL_DIR}" \
  --device "${DEVICE}"

echo "Done."
echo "Level 2 CNN demos: ${DEMO_DIR}/manifest.json"
echo "Level 2 CNN metrics: ${RUN_DIR}/vision_tactile_bc_metrics.json"
echo "Level 2 CNN eval CSV: ${EVAL_DIR}/vision_tactile_eval_results.csv"
