#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${WORK_DIR}"

if [ ! -d "${MF_DIR}" ]; then
  download_if_missing "${LAB_ZIP_URL}" "${WORK_DIR}/llama_lab.zip" 46745194
  unzip -t "${WORK_DIR}/llama_lab.zip" >/dev/null
  echo "Unzipping ${WORK_DIR}/llama_lab.zip to ${WORK_DIR}"
  unzip -q -o "${WORK_DIR}/llama_lab.zip" -d "${WORK_DIR}"
else
  echo "Found ${MF_DIR}"
fi

mkdir -p "${CHECKPOINT_DIR}"
download_if_missing "${ASSET_BASE_URL}/llama_7b.ckpt.lock" "${CHECKPOINT_DIR}/llama_7b.ckpt.lock" 0
download_if_missing "${ASSET_BASE_URL}/tokenizer.model.lock" "${CHECKPOINT_DIR}/tokenizer.model.lock" 0
download_if_missing "${ASSET_BASE_URL}/llama_7b.ckpt" "${LLAMA_CKPT}" 13476850247
download_if_missing "${ASSET_BASE_URL}/tokenizer.model" "${TOKENIZER_MODEL}" 534194

copy_infer_templates

echo "Lab package is ready at ${LAB_DIR}"
echo "Checkpoint: ${LLAMA_CKPT}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
