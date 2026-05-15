#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_assets
mkdir -p "${LAB_ALPACA_DIR}"

{
  echo "Cleaning Alpaca data"
  python "${REPO_DIR}/src/clean_alpaca_data.py" \
    --input "${RAW_ALPACA_JSON}" \
    --output "${CLEAN_ALPACA_JSON}"

  echo "Converting Alpaca JSON to conversation JSON"
  cd "${MF_DIR}/mindformers/tools/dataset_preprocess/llama"
  python alpaca_converter.py \
    --data_path "${CLEAN_ALPACA_JSON}" \
    --output_path "${CONVERSATION_JSON}"

  echo "Converting conversation JSON to MindRecord"
  rm -f "${MINDRECORD_FILE}" "${MINDRECORD_FILE}.db"
  python llama_preprocess_no_fschat.py \
    --dataset_type qa \
    --input_glob "${CONVERSATION_JSON}" \
    --model_file "${TOKENIZER_MODEL}" \
    --seq_length 2048 \
    --output_file "${MINDRECORD_FILE}"

  echo "MindRecord written to ${MINDRECORD_FILE}"
} 2>&1 | tee "${RESULTS_DIR}/preprocess.log"
