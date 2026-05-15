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

MODELARTS_NO_PROXY_DEFAULT="a.test.com,127.0.0.1,2.2.2.2,localhost,localhost.localdomain"
export no_proxy="${no_proxy:-${MODELARTS_NO_PROXY_DEFAULT}}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"

if [ "${DISABLE_PROXY_DOWNLOAD:-0}" = "1" ]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  export no_proxy="*"
  export NO_PROXY="*"
fi

file_size() {
  if [ -e "$1" ]; then
    wc -c < "$1" | tr -d ' '
  else
    echo 0
  fi
}

has_expected_file() {
  local dest="$1"
  local min_bytes="${2:-1}"
  if [ "${min_bytes}" = "0" ]; then
    [ -e "${dest}" ]
    return
  fi
  [ -f "${dest}" ] && [ "$(file_size "${dest}")" -ge "${min_bytes}" ]
}

download_with_python() {
  local url="$1"
  local dest="$2"
  python - "$url" "$dest" <<'PY'
import sys
import urllib.request

url, dest = sys.argv[1], sys.argv[2]
print("Python urllib downloading {}".format(url), flush=True)
urllib.request.urlretrieve(url, dest)
PY
}

download_if_missing() {
  local url="$1"
  local dest="$2"
  local min_bytes="${3:-1}"

  if has_expected_file "${dest}" "${min_bytes}"; then
    echo "Found ${dest} ($(file_size "${dest}") bytes)"
    return
  fi

  mkdir -p "$(dirname "${dest}")"
  if [ -e "${dest}" ]; then
    echo "Resuming ${dest} ($(file_size "${dest}") bytes so far)"
  fi
  echo "Downloading ${url}"

  if command -v wget >/dev/null 2>&1; then
    wget \
      --continue \
      --tries="${DOWNLOAD_TRIES:-8}" \
      --connect-timeout="${DOWNLOAD_CONNECT_TIMEOUT:-30}" \
      --read-timeout="${DOWNLOAD_READ_TIMEOUT:-120}" \
      --waitretry="${DOWNLOAD_WAIT_RETRY:-10}" \
      --retry-connrefused \
      "${url}" \
      -O "${dest}" || true
  fi

  if ! has_expected_file "${dest}" "${min_bytes}" && command -v curl >/dev/null 2>&1; then
    curl \
      --location \
      --continue-at - \
      --retry "${DOWNLOAD_TRIES:-8}" \
      --retry-delay "${DOWNLOAD_WAIT_RETRY:-10}" \
      --connect-timeout "${DOWNLOAD_CONNECT_TIMEOUT:-30}" \
      --output "${dest}" \
      "${url}" || true
  fi

  if ! has_expected_file "${dest}" "${min_bytes}"; then
    download_with_python "${url}" "${dest}" || true
  fi

  if ! has_expected_file "${dest}" "${min_bytes}"; then
    cat >&2 <<EOF
Download failed or incomplete: ${dest}
Current size: $(file_size "${dest}") bytes, expected at least ${min_bytes} bytes.

If ModelArts keeps timing out on this OBS URL, try one of these:
  1. Re-run this script later; downloads resume from partial files.
  2. Run with direct download mode:
       DISABLE_PROXY_DOWNLOAD=1 bash scripts/00_prepare_lab.sh
  3. Download the file outside ModelArts and upload it to the exact path above,
     then re-run scripts/00_prepare_lab.sh.

The 7B checkpoint is about 12.6 GiB, so a flaky network can take several retries.
EOF
    exit 1
  fi
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
