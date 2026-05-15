#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_assets
copy_infer_templates

if [ -z "${LORA_CKPT:-}" ]; then
  CKPT_DIR="${MF_DIR}/scripts/mf_standalone/output/checkpoint/rank_0"
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
    exit 1
  fi
  LORA_CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -name 'llama_7b_lora_rank_0-*.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [ -z "${LORA_CKPT}" ] || [ ! -s "${LORA_CKPT}" ]; then
  echo "No LoRA checkpoint found. Run scripts/03_train_lora.sh first or set LORA_CKPT." >&2
  exit 1
fi

export LORA_CKPT
export NLP_LAB4_REPO_DIR="${REPO_DIR}"
cd "${MF_DIR}"
python lora_infer.py 2>&1 | tee "${RESULTS_DIR}/lora_infer.log"
