#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-20260709}"
INPUT="${INPUT:-dataset_final/MSRS/training_seed20260620_levelbalanced_v2/msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset_final/MSRS/rl_level2_seed${SEED}}"
CHECK_PATHS="${CHECK_PATHS:-1}"
WRITE_ALL_LEVEL2="${WRITE_ALL_LEVEL2:-1}"

CHECK_ARGS=()
if [[ "${CHECK_PATHS}" == "1" ]]; then
  CHECK_ARGS=(--check_paths)
else
  CHECK_ARGS=(--no-check_paths)
fi

ALL_ARGS=()
if [[ "${WRITE_ALL_LEVEL2}" == "1" ]]; then
  ALL_ARGS=(--write_all_level2)
else
  ALL_ARGS=(--no-write_all_level2)
fi

python3 scripts/build_msrs_level2_rl_manifests.py \
  --input "${INPUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  "${CHECK_ARGS[@]}" \
  "${ALL_ARGS[@]}"
