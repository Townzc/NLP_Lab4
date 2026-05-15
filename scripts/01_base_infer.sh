#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_assets
copy_infer_templates

export NLP_LAB4_REPO_DIR="${REPO_DIR}"
cd "${MF_DIR}"
python pipeline_infer.py 2>&1 | tee "${RESULTS_DIR}/base_infer.log"
