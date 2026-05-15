#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${WORK_DIR}"

if [ ! -d "${MF_DIR}" ]; then
  if [ -s "${REPO_DIR}/vendor/llama_lab_slim.zip" ]; then
    echo "Unzipping bundled llama_lab package from repository"
    unzip -q -o "${REPO_DIR}/vendor/llama_lab_slim.zip" -d "${WORK_DIR}"
  else
    download_if_missing "${LAB_ZIP_URL}" "${WORK_DIR}/llama_lab.zip" 46745194
    unzip -t "${WORK_DIR}/llama_lab.zip" >/dev/null
    echo "Unzipping ${WORK_DIR}/llama_lab.zip to ${WORK_DIR}"
    unzip -q -o "${WORK_DIR}/llama_lab.zip" -d "${WORK_DIR}"
  fi
else
  echo "Found ${MF_DIR}"
fi

mkdir -p "${CHECKPOINT_DIR}"
if [ -e "${REPO_DIR}/vendor/checkpoint_download/llama/llama_7b.ckpt.lock" ]; then
  cp "${REPO_DIR}/vendor/checkpoint_download/llama/llama_7b.ckpt.lock" "${CHECKPOINT_DIR}/llama_7b.ckpt.lock"
else
  : > "${CHECKPOINT_DIR}/llama_7b.ckpt.lock"
fi

if [ -e "${REPO_DIR}/vendor/checkpoint_download/llama/tokenizer.model.lock" ]; then
  cp "${REPO_DIR}/vendor/checkpoint_download/llama/tokenizer.model.lock" "${CHECKPOINT_DIR}/tokenizer.model.lock"
else
  : > "${CHECKPOINT_DIR}/tokenizer.model.lock"
fi

if [ ! -s "${TOKENIZER_MODEL}" ] && [ -s "${REPO_DIR}/vendor/checkpoint_download/llama/tokenizer.model" ]; then
  echo "Copying bundled tokenizer.model from repository"
  cp "${REPO_DIR}/vendor/checkpoint_download/llama/tokenizer.model" "${TOKENIZER_MODEL}"
fi

download_if_missing "${ASSET_BASE_URL}/llama_7b.ckpt" "${LLAMA_CKPT}" 13476850247 "${OBS_BASE_URI}/llama_7b.ckpt" "obs.cn-north-4.myhuaweicloud.com" \
  || download_if_missing "${MODELZOO_BASE_URL}/open_llama_7b.ckpt" "${LLAMA_CKPT}" 13476850247 "${MODELZOO_OBS_BASE_URI}/open_llama_7b.ckpt" "${MODELZOO_OBS_SERVER}"
download_if_missing "${ASSET_BASE_URL}/tokenizer.model" "${TOKENIZER_MODEL}" 534194 "${OBS_BASE_URI}/tokenizer.model"

copy_infer_templates

echo "Lab package is ready at ${LAB_DIR}"
echo "Checkpoint: ${LLAMA_CKPT}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
