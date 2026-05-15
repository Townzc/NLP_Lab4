#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_assets
if [ ! -s "${MINDRECORD_FILE}" ]; then
  echo "MindRecord file is missing. Run scripts/02_prepare_data.sh first." >&2
  exit 1
fi

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SINK_SIZE="${SINK_SIZE:-5}"
LEARNING_RATE="${LEARNING_RATE:-1.e-4}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-FP32StateAdamWeightDecay}"
LR_SCHEDULE_TYPE="${LR_SCHEDULE_TYPE:-CosineWithWarmUpLR}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

python "${REPO_DIR}/src/patch_lora_config.py" \
  --source "${BASE_CONFIG}" \
  --output "${LORA_CONFIG}" \
  --checkpoint "${LLAMA_CKPT}" \
  --dataset "${MINDRECORD_FILE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --sink-size "${SINK_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --optimizer-type "${OPTIMIZER_TYPE}" \
  --lr-schedule-type "${LR_SCHEDULE_TYPE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --use-parallel False

copy_infer_templates

LOG_FILE="${RESULTS_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training with config ${LORA_CONFIG}"
echo "Log file: ${LOG_FILE}"
cd "${MF_DIR}/scripts"
bash run_standalone.sh "../configs/llama/${CONFIG_NAME}" 0 finetune 2>&1 | tee "${LOG_FILE}"
