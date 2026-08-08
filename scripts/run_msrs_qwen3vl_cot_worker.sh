#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-VL-32B-Thinking directory}"
MANIFEST="${MANIFEST:?Set MANIFEST to this worker's disjoint manifest JSON}"
OUTPUT_JSON="${OUTPUT_JSON:?Set OUTPUT_JSON to this worker's output JSON}"
LOG_PATH="${LOG_PATH:-outputs/logs/$(basename "${OUTPUT_JSON}" .json).log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
BSZ="${BSZ:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
KV_CACHE_MEMORY_GB="${KV_CACHE_MEMORY_GB:-8}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-100352}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "$(dirname "${LOG_PATH}")"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_TRTLLM_ATTENTION="${VLLM_USE_TRTLLM_ATTENTION:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"

exec > >(tee -a "${LOG_PATH}") 2>&1

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne "${TP_SIZE}" ]]; then
  printf '[ERROR] GPU count=%s but TP_SIZE=%s\n' "${#GPU_IDS[@]}" "${TP_SIZE}" >&2
  exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
  free_mb="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  printf '[GPU] id=%s free_mb=%s required=%s\n' "${gpu_id}" "${free_mb}" "${MIN_FREE_MB}"
  if [[ -z "${free_mb}" || "${free_mb}" -lt "${MIN_FREE_MB}" ]]; then
    printf '[ERROR] GPU %s is not sufficiently empty.\n' "${gpu_id}" >&2
    exit 3
  fi
done

"${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if not rows:
    raise SystemExit(f"empty manifest: {path}")
keys = (
    "visible_degraded_path", "infrared_degraded_path", "visible_clean_path",
    "infrared_clean_path", "fused_gt_path",
)
missing = [(row.get("id"), key, row.get(key)) for row in rows for key in keys if not Path(str(row.get(key, ""))).is_file()]
if missing:
    for item in missing[:10]:
        print("[MISSING_PATH]", item)
    raise SystemExit(f"manifest path preflight failed: {len(missing)} missing paths")
print(f"[MANIFEST] rows={len(rows)} path_preflight=ok")
PY

"${PYTHON_BIN}" -c \
  "import torch, qwen_vl_utils; from vllm import LLM, SamplingParams; print(f'[ENV] torch={torch.__version__} cuda={torch.version.cuda}'); print('[ENV] imports=ok')"

ARGS=(
  --input_json "${MANIFEST}"
  --output_json "${OUTPUT_JSON}"
  --model_path "${MODEL_PATH}"
  --tensor_parallel_size "${TP_SIZE}"
  --batch_size "${BSZ}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_tokens "${MAX_TOKENS}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --cpu_offload_gb "${CPU_OFFLOAD_GB}"
  --kv_cache_memory_gb "${KV_CACHE_MEMORY_GB}"
  --image_max_pixels "${IMAGE_MAX_PIXELS}"
  --temperature 0.2
  --top_p 0.8
  --top_k 20
  --retries 3
  --progress_every "${PROGRESS_EVERY}"
  --mm_encoder_attn_backend "${MM_ENCODER_ATTN_BACKEND}"
  --include_fused_gt
  --disable_vllm_tqdm
  --resume
)
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

printf '[RUN] manifest=%s output=%s gpus=%s bsz=%s max_samples=%s\n' \
  "${MANIFEST}" "${OUTPUT_JSON}" "${CUDA_VISIBLE_DEVICES}" "${BSZ}" "${MAX_SAMPLES}"
"${PYTHON_BIN}" scripts/generate_msrs_qwen3vl_cot.py "${ARGS[@]}"

"${PYTHON_BIN}" scripts/validate_msrs_qwen3vl_cot.py \
  --input "${OUTPUT_JSON}" --min_think_chars 100 --check_paths

printf '[DONE] worker output=%s\n' "${OUTPUT_JSON}"
