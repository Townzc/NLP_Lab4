#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/00_prepare_lab.sh"
bash "${SCRIPT_DIR}/01_base_infer.sh"
bash "${SCRIPT_DIR}/02_prepare_data.sh"
bash "${SCRIPT_DIR}/03_train_lora.sh"
bash "${SCRIPT_DIR}/04_lora_infer.sh"
