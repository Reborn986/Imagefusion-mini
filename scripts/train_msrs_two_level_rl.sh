#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MSRS_RUNTIME_CACHE_DIR="${MSRS_RUNTIME_CACHE_DIR:-/tmp/msrs_runtime_cache_${USER:-msrs}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MSRS_RUNTIME_CACHE_DIR}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MSRS_RUNTIME_CACHE_DIR}/xdg}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-lumina-mgpt2}"
if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  printf '[INFO] using the currently active Python environment (set CONDA_SH to activate another one)\n'
fi

REWARD_GPU="${REWARD_GPU:-0}"
POLICY_GPUS="${POLICY_GPUS:-1,2,3}"
NPROC="${NPROC:-3}"
MIN_FREE_REWARD_MB="${MIN_FREE_REWARD_MB:-30000}"
MIN_FREE_POLICY_MB="${MIN_FREE_POLICY_MB:-65000}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_CHECK_INTERVAL_SEC="${GPU_CHECK_INTERVAL_SEC:-60}"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REWARD_PYTHON_BIN="${REWARD_PYTHON_BIN:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-imagefusion_r1/rl/train_msrs_grpo_mgpt2.py}"

BASE_MODEL="${BASE_MODEL:-pretrained/Lumina-mGPT-2.0-Omni}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT must point to the SFT checkpoint used to initialize RL.}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/msrs_level1_noise_except_blur2_with_fusedgt_golden_qwen3vl_cot.json}"
PROTOCOL="${PROTOCOL:-full}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set.}"

QWEN_REWARD_MODEL="${QWEN_REWARD_MODEL:-models/Qwen3-VL-8B-Instruct}"
REWARD_HOST="${REWARD_HOST:-127.0.0.1}"
REWARD_PORT="${REWARD_PORT:-18080}"
REWARD_URL="http://${REWARD_HOST}:${REWARD_PORT}/score"
REWARD_TP="${REWARD_TP:-1}"
REWARD_MAX_MODEL_LEN="${REWARD_MAX_MODEL_LEN:-8192}"
REWARD_GPU_MEMORY_UTILIZATION="${REWARD_GPU_MEMORY_UTILIZATION:-0.85}"
REWARD_ENFORCE_EAGER="${REWARD_ENFORCE_EAGER:-1}"
REWARD_KV_CACHE_MEMORY_GB="${REWARD_KV_CACHE_MEMORY_GB:-0}"
LOAD_REWARD_ON_START="${LOAD_REWARD_ON_START:-1}"
WAIT_FOR_REWARD_READY_BEFORE_POLICY="${WAIT_FOR_REWARD_READY_BEFORE_POLICY:-1}"

NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24576}"
STOP_AFTER_IMAGES="${STOP_AFTER_IMAGES:-3}"
IMAGE_TOP_K="${IMAGE_TOP_K:-2000}"
TEXT_TOP_K="${TEXT_TOP_K:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
RL_EPOCHS="${RL_EPOCHS:-1}"
LR="${LR:-5e-7}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}"
GRAD_PRECISION="${GRAD_PRECISION:-fp32}"
KL_BETA="${KL_BETA:-0.0}"
NO_REFERENCE_KL="${NO_REFERENCE_KL:-1}"
CFG="${CFG:-3.0}"
ROLLOUT_USE_CACHE="${ROLLOUT_USE_CACHE:-1}"
PPO_CLIP_RANGE="${PPO_CLIP_RANGE:-0.2}"
LAMBDA_TEXT="${LAMBDA_TEXT:-0.1}"
LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.9}"
REFERENCE_REWARD_WEIGHT="${REFERENCE_REWARD_WEIGHT:-0.9}"
QWEN_REWARD_WEIGHT="${QWEN_REWARD_WEIGHT:-0.1}"
REFERENCE_PSNR_FLOOR="${REFERENCE_PSNR_FLOOR:-10.0}"
REFERENCE_PSNR_CEILING="${REFERENCE_PSNR_CEILING:-40.0}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOG_STEPS="${LOG_STEPS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TRAIN_SEED="${TRAIN_SEED:-20260709}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-1}"
# Three policy ranks share one serialized Qwen judge.  A 120s client timeout
# can expire for the third queued request even when every individual judge call
# is healthy, so keep this aligned with the trainer's safer default.
REWARD_TIMEOUT_SEC="${REWARD_TIMEOUT_SEC:-300}"
GENERATION_LOG_INTERVAL="${GENERATION_LOG_INTERVAL:-512}"
EMPTY_CUDA_CACHE_BETWEEN_PHASES="${EMPTY_CUDA_CACHE_BETWEEN_PHASES:-0}"
LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-${CE_LOSS_CHUNK_SIZE:-64}}"
REPLAY_MICRO_BATCH_SIZE="${REPLAY_MICRO_BATCH_SIZE:-1}"
FSDP_FULL_LOAD_ALL_RANKS="${FSDP_FULL_LOAD_ALL_RANKS:-1}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-shard_grad_op}"
MIN_GROUP_REWARD_STD="${MIN_GROUP_REWARD_STD:-0.0001}"
MAX_CONSECUTIVE_NO_SIGNAL_STEPS="${MAX_CONSECUTIVE_NO_SIGNAL_STEPS:-3}"
MAX_REPLAY_LOGPROB_ERROR="${MAX_REPLAY_LOGPROB_ERROR:-0.10}"
HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-60}"
# Ranks generate variable-length 25k-token trajectories independently.  A fast
# rank may enter the next collective long before a slow rank, so the process
# group timeout must be much longer than a single rollout.
DISTRIBUTED_TIMEOUT_SEC="${DISTRIBUTED_TIMEOUT_SEC:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  printf '[ERROR] Missing RL trainer entrypoint: %s\n' "${TRAIN_ENTRY}" >&2
  printf '[ERROR] Reward modules and launcher are ready, but the MSRS mGPT2 GRPO trainer is not implemented yet.\n' >&2
  printf '[ERROR] Do not use inference scripts as a substitute for RL training.\n' >&2
  exit 2
fi

if [[ ! -d "${INIT_CKPT}" ]]; then
  printf '[ERROR] INIT_CKPT not found: %s\n' "${INIT_CKPT}" >&2
  exit 2
fi

if [[ ! -d "${QWEN_REWARD_MODEL}" ]]; then
  printf '[ERROR] Qwen3-VL reward model not found: %s\n' "${QWEN_REWARD_MODEL}" >&2
  printf '[HINT] Set QWEN_REWARD_MODEL to the local private reward-model directory.\n' >&2
  exit 2
fi

if [[ -z "${RESUME_FROM_CHECKPOINT}" && -d "${OUTPUT_DIR}" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf '[ERROR] Refusing to mix a fresh RL run into non-empty OUTPUT_DIR: %s\n' "${OUTPUT_DIR}" >&2
  printf '[HINT] Set a new OUTPUT_DIR, or set RESUME_FROM_CHECKPOINT for an intentional resume.\n' >&2
  exit 2
fi

IFS=',' read -r -a POLICY_GPU_IDS <<< "${POLICY_GPUS}"
if [[ "${#POLICY_GPU_IDS[@]}" -ne "${NPROC}" ]]; then
  printf '[ERROR] POLICY_GPUS has %s GPUs but NPROC=%s\n' "${#POLICY_GPU_IDS[@]}" "${NPROC}" >&2
  exit 2
fi

if [[ "${NPROC}" -gt 1 && "${FSDP_SHARDING_STRATEGY}" != "shard_grad_op" ]]; then
  printf '[ERROR] Distributed autoregressive RL requires FSDP_SHARDING_STRATEGY=shard_grad_op; got %s\n' \
    "${FSDP_SHARDING_STRATEGY}" >&2
  exit 2
fi

free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

wait_for_gpu_set() {
  local label="$1"
  local required_mb="$2"
  shift 2
  local gpu_ids=("$@")
  if [[ "${WAIT_FOR_GPUS}" != "1" ]]; then
    return
  fi
  while true; do
    local ok=1
    for gpu_id in "${gpu_ids[@]}"; do
      local free
      free="$(free_mb "${gpu_id}")"
      printf '[GPU] %s gpu=%s free_mb=%s required_min_mb=%s\n' "${label}" "${gpu_id}" "${free}" "${required_mb}"
      if [[ -z "${free}" || "${free}" -lt "${required_mb}" ]]; then
        ok=0
      fi
    done
    if [[ "${ok}" == "1" ]]; then
      return
    fi
    printf '[WAIT] %s GPUs are not free yet; sleep %ss\n' "${label}" "${GPU_CHECK_INTERVAL_SEC}"
    sleep "${GPU_CHECK_INTERVAL_SEC}"
  done
}

mkdir -p "${OUTPUT_DIR}/logs"
wait_for_gpu_set "reward" "${MIN_FREE_REWARD_MB}" "${REWARD_GPU}"
wait_for_gpu_set "policy" "${MIN_FREE_POLICY_MB}" "${POLICY_GPU_IDS[@]}"

REWARD_LOG="${OUTPUT_DIR}/logs/qwen3vl_reward_server_gpu${REWARD_GPU}.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/msrs_two_level_rl_train.log"

LOAD_ARGS=()
if [[ "${LOAD_REWARD_ON_START}" == "1" ]]; then
  LOAD_ARGS=(--load-on-start)
fi
EAGER_ARGS=()
if [[ "${REWARD_ENFORCE_EAGER}" == "1" ]]; then
  EAGER_ARGS=(--enforce-eager)
else
  EAGER_ARGS=(--no-enforce-eager)
fi
KV_ARGS=()
if [[ "${REWARD_KV_CACHE_MEMORY_GB}" != "0" ]]; then
  KV_ARGS=(--kv-cache-memory-bytes "$((REWARD_KV_CACHE_MEMORY_GB * 1024 * 1024 * 1024))")
fi
REFERENCE_ARGS=()
if [[ "${NO_REFERENCE_KL}" == "1" ]]; then
  REFERENCE_ARGS=(--no_reference_kl)
fi
OPTIMIZER_OFFLOAD_ARGS=(--no-optimizer_cpu_offload)
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_OFFLOAD_ARGS=(--optimizer_cpu_offload)
fi

FSDP_LOAD_ARGS=()
if [[ "${FSDP_FULL_LOAD_ALL_RANKS}" == "1" ]]; then
  FSDP_LOAD_ARGS=(--fsdp_full_load_all_ranks)
fi
ROLLOUT_CACHE_ARGS=(--rollout_use_cache)
if [[ "${ROLLOUT_USE_CACHE}" == "0" ]]; then
  ROLLOUT_CACHE_ARGS=(--no-rollout_use_cache)
fi
EMPTY_CACHE_ARGS=(--no-empty_cuda_cache_between_phases)
if [[ "${EMPTY_CUDA_CACHE_BETWEEN_PHASES}" == "1" ]]; then
  EMPTY_CACHE_ARGS=(--empty_cuda_cache_between_phases)
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    printf '[ERROR] RESUME_FROM_CHECKPOINT not found: %s\n' "${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

printf '[RUN] protocol=%s\n' "${PROTOCOL}"
printf '[RUN] base_model=%s\n' "${BASE_MODEL}"
printf '[RUN] init_ckpt=%s\n' "${INIT_CKPT}"
printf '[RUN] resume_from_checkpoint=%s\n' "${RESUME_FROM_CHECKPOINT:-none}"
printf '[RUN] train_manifest=%s\n' "${TRAIN_MANIFEST}"
printf '[RUN] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[RUN] reward_gpu=%s qwen_model=%s reward_url=%s\n' "${REWARD_GPU}" "${QWEN_REWARD_MODEL}" "${REWARD_URL}"
printf '[RUN] policy_gpus=%s nproc=%s\n' "${POLICY_GPUS}" "${NPROC}"
printf '[RUN] reward_python=%s\n' "${REWARD_PYTHON_BIN}"
printf '[RUN] reward_enforce_eager=%s reward_kv_cache_gb=%s reward_batch_size=%s\n' \
  "${REWARD_ENFORCE_EAGER}" "${REWARD_KV_CACHE_MEMORY_GB}" "${REWARD_BATCH_SIZE}"
printf '[RUN] reward_timeout_sec=%s\n' "${REWARD_TIMEOUT_SEC}"
printf '[RUN] wait_for_reward_ready_before_policy=%s\n' "${WAIT_FOR_REWARD_READY_BEFORE_POLICY}"
printf '[RUN] no_reference_kl=%s kl_beta=%s\n' "${NO_REFERENCE_KL}" "${KL_BETA}"
printf '[RUN] optimizer_cpu_offload=%s\n' "${OPTIMIZER_CPU_OFFLOAD}"
printf '[RUN] grad_precision=%s\n' "${GRAD_PRECISION}"
printf '[RUN] cfg=%s ppo_clip_range=%s\n' "${CFG}" "${PPO_CLIP_RANGE}"
printf '[RUN] rollout_use_cache=%s\n' "${ROLLOUT_USE_CACHE}"
printf '[RUN] image_top_k=%s text_top_k=%s\n' "${IMAGE_TOP_K}" "${TEXT_TOP_K}"
printf '[RUN] reward_weights text=%s image=%s reference=%s qwen=%s\n' \
  "${LAMBDA_TEXT}" "${LAMBDA_IMAGE}" "${REFERENCE_REWARD_WEIGHT}" "${QWEN_REWARD_WEIGHT}"
printf '[RUN] max_new_tokens=%s stop_after_images=%s\n' "${MAX_NEW_TOKENS}" "${STOP_AFTER_IMAGES}"
printf '[RUN] train_seed=%s\n' "${TRAIN_SEED}"
printf '[RUN] generation_log_interval=%s\n' "${GENERATION_LOG_INTERVAL}"
printf '[RUN] empty_cuda_cache_between_phases=%s\n' "${EMPTY_CUDA_CACHE_BETWEEN_PHASES}"
printf '[RUN] logprob_chunk_size=%s fsdp_full_load_all_ranks=%s fsdp_sharding=%s\n' \
  "${LOGPROB_CHUNK_SIZE}" "${FSDP_FULL_LOAD_ALL_RANKS}" "${FSDP_SHARDING_STRATEGY}"
printf '[RUN] replay_micro_batch_size=%s\n' "${REPLAY_MICRO_BATCH_SIZE}"
printf '[RUN] heartbeat_interval_sec=%s distributed_timeout_sec=%s\n' \
  "${HEARTBEAT_INTERVAL_SEC}" "${DISTRIBUTED_TIMEOUT_SEC}"
printf '[RUN] cuda_device_order=%s\n' "${CUDA_DEVICE_ORDER}"
printf '[RUN] pytorch_cuda_alloc_conf=%s\n' "${PYTORCH_CUDA_ALLOC_CONF}"

CUDA_VISIBLE_DEVICES="${REWARD_GPU}" setsid "${REWARD_PYTHON_BIN}" scripts/serve_msrs_qwen3vl_reward.py \
  --model-path "${QWEN_REWARD_MODEL}" \
  --host "${REWARD_HOST}" \
  --port "${REWARD_PORT}" \
  --tensor-parallel-size "${REWARD_TP}" \
  --gpu-memory-utilization "${REWARD_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${REWARD_MAX_MODEL_LEN}" \
  "${EAGER_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  > "${REWARD_LOG}" 2>&1 &
REWARD_PID="$!"

cleanup() {
  if kill -0 "${REWARD_PID}" >/dev/null 2>&1; then
    kill -- "-${REWARD_PID}" >/dev/null 2>&1 || kill "${REWARD_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL -- "-${REWARD_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${WAIT_FOR_REWARD_READY_BEFORE_POLICY}" == "1" ]]; then
  printf '[RUN] waiting for reward server, log=%s\n' "${REWARD_LOG}"
  REWARD_READY=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${REWARD_PID}" >/dev/null 2>&1; then
      printf '[ERROR] reward server exited early. See %s\n' "${REWARD_LOG}" >&2
      exit 3
    fi
    if "${REWARD_PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${REWARD_HOST}:${REWARD_PORT}/health', timeout=2).read()" >/dev/null 2>&1; then
      REWARD_READY=1
      break
    fi
    sleep 5
  done
  if [[ "${REWARD_READY}" != "1" ]]; then
    printf '[ERROR] reward server did not become ready in time. See %s\n' "${REWARD_LOG}" >&2
    exit 3
  fi
else
  printf '[RUN] reward server is loading asynchronously; starting policy immediately, log=%s\n' "${REWARD_LOG}"
fi

printf '[RUN] starting MSRS two-level GRPO trainer, log=%s\n' "${TRAIN_LOG}"
TEE_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TEE_ARGS=(-a)
fi
CUDA_VISIBLE_DEVICES="${POLICY_GPUS}" \
MSRS_REWARD_SERVER_URL="${REWARD_URL}" \
"${TORCHRUN}" --nproc_per_node="${NPROC}" "${TRAIN_ENTRY}" \
  --base_model "${BASE_MODEL}" \
  --init_ckpt "${INIT_CKPT}" \
  "${RESUME_ARGS[@]}" \
  --train_manifest "${TRAIN_MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_server_url "${REWARD_URL}" \
  --num_generations "${NUM_GENERATIONS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --stop_after_images "${STOP_AFTER_IMAGES}" \
  --image_top_k "${IMAGE_TOP_K}" \
  --text_top_k "${TEXT_TOP_K}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${RL_EPOCHS}" \
  --lr "${LR}" \
  --grad_precision "${GRAD_PRECISION}" \
  "${OPTIMIZER_OFFLOAD_ARGS[@]}" \
  --kl_beta "${KL_BETA}" \
  "${REFERENCE_ARGS[@]}" \
  --cfg "${CFG}" \
  "${ROLLOUT_CACHE_ARGS[@]}" \
  --ppo_clip_range "${PPO_CLIP_RANGE}" \
  --lambda_text "${LAMBDA_TEXT}" \
  --lambda_image "${LAMBDA_IMAGE}" \
  --reference_reward_weight "${REFERENCE_REWARD_WEIGHT}" \
  --qwen_reward_weight "${QWEN_REWARD_WEIGHT}" \
  --reference_psnr_floor "${REFERENCE_PSNR_FLOOR}" \
  --reference_psnr_ceiling "${REFERENCE_PSNR_CEILING}" \
  --save_steps "${SAVE_STEPS}" \
  --log_steps "${LOG_STEPS}" \
  --reward_batch_size "${REWARD_BATCH_SIZE}" \
  --reward_timeout_sec "${REWARD_TIMEOUT_SEC}" \
  --generation_log_interval "${GENERATION_LOG_INTERVAL}" \
  "${EMPTY_CACHE_ARGS[@]}" \
  --logprob_chunk_size "${LOGPROB_CHUNK_SIZE}" \
  --replay_micro_batch_size "${REPLAY_MICRO_BATCH_SIZE}" \
  --fsdp_sharding_strategy "${FSDP_SHARDING_STRATEGY}" \
  --min_group_reward_std "${MIN_GROUP_REWARD_STD}" \
  --max_consecutive_no_signal_steps "${MAX_CONSECUTIVE_NO_SIGNAL_STEPS}" \
  --max_replay_logprob_error "${MAX_REPLAY_LOGPROB_ERROR}" \
  --heartbeat_interval_sec "${HEARTBEAT_INTERVAL_SEC}" \
  --distributed_timeout_sec "${DISTRIBUTED_TIMEOUT_SEC}" \
  "${FSDP_LOAD_ARGS[@]}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${TRAIN_SEED}" \
  2>&1 | tee "${TEE_ARGS[@]}" "${TRAIN_LOG}"
