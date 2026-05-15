#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MODELARTS_WORK_DIR:-/home/ma-user/work}"
LAB_DIR="${LLAMA_LAB_DIR:-${WORK_DIR}/llama_lab}"
MF_DIR="${LAB_DIR}/mindformers"
RESULTS_DIR="${REPO_DIR}/results"

LAB_ZIP_URL="https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/%E9%BB%84%E8%B4%BA%E9%98%B3%E8%BF%81%E7%A7%BB%E6%9D%90%E6%96%99/hhy123/llama_lab/llama_lab.zip"
ASSET_BASE_URL="https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/%E9%BB%84%E8%B4%BA%E9%98%B3%E8%BF%81%E7%A7%BB%E6%9D%90%E6%96%99/hhy123/llama_lab"

CHECKPOINT_DIR="${MF_DIR}/checkpoint_download/llama"
LLAMA_CKPT="${CHECKPOINT_DIR}/llama_7b.ckpt"
TOKENIZER_MODEL="${CHECKPOINT_DIR}/tokenizer.model"

RAW_ALPACA_JSON="${REPO_DIR}/data/campus_alpaca_data.json"
LAB_ALPACA_DIR="${LAB_DIR}/stanford_alpaca"
CLEAN_ALPACA_JSON="${LAB_ALPACA_DIR}/campus_alpaca_data_clean.json"
CONVERSATION_JSON="${LAB_ALPACA_DIR}/campus_alpaca_conversation.json"
MINDRECORD_FILE="${LAB_ALPACA_DIR}/campus-fastchat2048.mindrecord"

CONFIG_NAME="${CONFIG_NAME:-run_llama_7b_lora_campus.yaml}"
BASE_CONFIG="${MF_DIR}/configs/llama/run_llama_7b_lora.yaml"
LORA_CONFIG="${MF_DIR}/configs/llama/${CONFIG_NAME}"

mkdir -p "${RESULTS_DIR}"

download_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -s "${dest}" ]; then
    echo "Found ${dest}"
    return
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "Downloading ${url}"
  wget -c "${url}" -O "${dest}"
}

require_lab() {
  if [ ! -d "${MF_DIR}" ]; then
    echo "MindFormers lab package is missing. Run scripts/00_prepare_lab.sh first." >&2
    exit 1
  fi
}

require_assets() {
  require_lab
  if [ ! -s "${LLAMA_CKPT}" ] || [ ! -s "${TOKENIZER_MODEL}" ]; then
    echo "Llama checkpoint or tokenizer is missing. Run scripts/00_prepare_lab.sh first." >&2
    exit 1
  fi
}

copy_infer_templates() {
  require_lab
  cp "${REPO_DIR}/templates/pipeline_infer.py" "${MF_DIR}/pipeline_infer.py"
  cp "${REPO_DIR}/templates/lora_infer.py" "${MF_DIR}/lora_infer.py"
}
