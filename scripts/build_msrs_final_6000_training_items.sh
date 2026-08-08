#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-20260620}"
SAMPLES_PER_COMBO="${SAMPLES_PER_COMBO:-300}"
SUPPLEMENT="${SUPPLEMENT:-data/msrs_supplement_5combos_300_each_with_fusedgt_golden_qwen3vl_cot.json}"
COT_INPUT_GLOB="${COT_INPUT_GLOB:-data/*with_fusedgt_golden_qwen3vl_cot.json}"
OUTPUT="${OUTPUT:-data/msrs_raw_items_cot_final_6000_clean_seed${SEED}.json}"
REPORT="${REPORT:-${OUTPUT%.json}.report.json}"
BALANCE_BASE_IMAGES_BY_LEVEL="${BALANCE_BASE_IMAGES_BY_LEVEL:-0}"

if [[ ! -f "${SUPPLEMENT}" ]]; then
  printf '[ERROR] Supplement CoT is not ready: %s\n' "${SUPPLEMENT}" >&2
  exit 2
fi

python3 scripts/validate_msrs_qwen3vl_cot.py \
  --input "${SUPPLEMENT}" \
  --min_think_chars 100 \
  --check_paths

BUILD_ARGS=(
  --inputs
  "${COT_INPUT_GLOB}"
  "${SUPPLEMENT}"
  --output "${OUTPUT}"
  --samples_per_combo "${SAMPLES_PER_COMBO}"
  --min_items_per_combo "${SAMPLES_PER_COMBO}"
  --seed "${SEED}"
  --check_paths
  --strict_pairing
  --balance_base_images
  --output_schema clean_stage_cot
  --report "${REPORT}"
)
if [[ "${BALANCE_BASE_IMAGES_BY_LEVEL}" == "1" ]]; then
  BUILD_ARGS+=(--balance_base_images_by_level)
fi

python3 scripts/build_msrs_cot_training_items.py "${BUILD_ARGS[@]}"

expected=$((20 * SAMPLES_PER_COMBO))
actual="$(grep -c '^    "id":' "${OUTPUT}")"
if [[ "${actual}" -ne "${expected}" ]]; then
  printf '[ERROR] Expected %s final items, found %s\n' "${expected}" "${actual}" >&2
  exit 3
fi
printf '[DONE] final clean training JSON: %s items=%s\n' "${OUTPUT}" "${actual}"
