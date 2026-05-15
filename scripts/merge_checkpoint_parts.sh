#!/usr/bin/env bash
set -euo pipefail

PART_DIR="${1:-/home/ma-user/work/upload_parts}"
OUT_FILE="${2:-/home/ma-user/work/llama_lab/mindformers/checkpoint_download/llama/llama_7b.ckpt}"
EXPECTED_SIZE="${EXPECTED_SIZE:-13476850247}"
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

if [ ! -d "${PART_DIR}" ]; then
  echo "Part directory not found: ${PART_DIR}" >&2
  exit 1
fi

mapfile -t PARTS < <(find "${PART_DIR}" -maxdepth 1 -type f -name 'llama_7b.ckpt.part*' | sort)
if [ "${#PARTS[@]}" -eq 0 ]; then
  echo "No part files found in ${PART_DIR}" >&2
  exit 1
fi

echo "Found ${#PARTS[@]} part files:"
for part in "${PARTS[@]}"; do
  ls -lh "${part}"
done

mkdir -p "$(dirname "${OUT_FILE}")"
rm -f "${OUT_FILE}"

echo "Merging to ${OUT_FILE}"
cat "${PARTS[@]}" > "${OUT_FILE}"

ACTUAL_SIZE="$(stat -c %s "${OUT_FILE}")"
echo "Merged size: ${ACTUAL_SIZE}"
if [ "${ACTUAL_SIZE}" != "${EXPECTED_SIZE}" ]; then
  echo "Size mismatch: expected ${EXPECTED_SIZE}, got ${ACTUAL_SIZE}" >&2
  exit 1
fi

ACTUAL_SHA256="$(sha256sum "${OUT_FILE}" | awk '{print $1}')"
echo "Merged sha256: ${ACTUAL_SHA256}"
if [ -n "${EXPECTED_SHA256}" ] && [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
  exit 1
fi

echo "Checkpoint merge completed: ${OUT_FILE}"

