#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  printf 'Usage: %s SOURCE_CHECKPOINT DESTINATION init|resume\n' "$0" >&2
  exit 2
fi

SOURCE_CHECKPOINT="$1"
DESTINATION="$2"
MODE="$3"

if [[ ! -d "${SOURCE_CHECKPOINT}" ]]; then
  printf '[ERROR] source checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${MODE}" != "init" && "${MODE}" != "resume" ]]; then
  printf '[ERROR] mode must be init or resume, got: %s\n' "${MODE}" >&2
  exit 2
fi
if [[ -e "${DESTINATION}" ]]; then
  printf '[ERROR] destination already exists; choose a new empty path: %s\n' "${DESTINATION}" >&2
  exit 2
fi

mkdir -p "${DESTINATION}"
if [[ "${MODE}" == "init" ]]; then
  rsync -a \
    --exclude='optimizer*' \
    --exclude='trainer_state.json' \
    --exclude='status.json' \
    --exclude='logs/' \
    "${SOURCE_CHECKPOINT}/" "${DESTINATION}/"
else
  rsync -a "${SOURCE_CHECKPOINT}/" "${DESTINATION}/"
fi

if ! find "${DESTINATION}" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
  printf '[ERROR] exported directory has no safetensors model shards: %s\n' "${DESTINATION}" >&2
  exit 3
fi

(
  cd "${DESTINATION}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
du -sh "${DESTINATION}"
printf '[DONE] exported mode=%s destination=%s checksums=%s/SHA256SUMS\n' \
  "${MODE}" "${DESTINATION}" "${DESTINATION}"

