#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ITEMS="${RAW_ITEMS:-data/msrs_raw_items_cot_final_6000_clean_seed20260620.json}"
OUT_DIR="${OUT_DIR:-data/processed_mgpt2/msrs_ce_cot_final_6000_clean_480x640}"
TOKENIZER="${TOKENIZER:-pretrained/Lumina-mGPT-2.0-Omni}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_ITEMS="${EXPECTED_ITEMS:-6000}"
OUTPUT_PROTOCOL="${OUTPUT_PROTOCOL:-auto}"
TARGET_HEIGHT="${TARGET_HEIGHT:-480}"
TARGET_WIDTH="${TARGET_WIDTH:-640}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-28672}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PWD}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" imagefusion_r1/preprocess/pre_tokenize_mgpt2_ce.py \
  --in_filename "${RAW_ITEMS}" \
  --out_dir "${OUT_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --target_height "${TARGET_HEIGHT}" \
  --target_width "${TARGET_WIDTH}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --splits 1 \
  --rank 0 \
  --device "${DEVICE}" \
  --overwrite \
  --no_sanitize_cot \
  --output_protocol "${OUTPUT_PROTOCOL}"

record="${OUT_DIR}/0-of-1-record.jsonl"
filtered="${OUT_DIR}/0-of-1-record_len${MAX_SEQ_LEN}.jsonl"
total="$(wc -l < "${record}")"
kept="$(wc -l < "${filtered}")"
pkls="$(find "${OUT_DIR}/files" -maxdepth 1 -name '*.pkl' | wc -l)"
printf '[CHECK] expected=%s record=%s filtered=%s pkls=%s\n' \
  "${EXPECTED_ITEMS}" "${total}" "${kept}" "${pkls}"
if [[ "${total}" -ne "${EXPECTED_ITEMS}" || "${kept}" -ne "${EXPECTED_ITEMS}" || "${pkls}" -ne "${EXPECTED_ITEMS}" ]]; then
  printf '[ERROR] PKL output is incomplete or contains sequences over %s; do not train.\n' "${MAX_SEQ_LEN}" >&2
  exit 3
fi
printf '[DONE] all final PKLs passed count and sequence-length filtering checks.\n'
