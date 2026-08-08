#!/usr/bin/env bash
set -euo pipefail

RL_DATA_SEED="${RL_DATA_SEED:-20260709}"
RL_TIER="${RL_TIER:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_DEFAULT=""
EPOCHS_DEFAULT=1

case "${RL_TIER}" in
  preflight)
    SPLIT_NAME="preflight_9"
    GRAD_ACCUM_DEFAULT=1
    SAVE_STEPS_DEFAULT=0
    ;;
  pilot100)
    SPLIT_NAME="pilot100_100"
    GRAD_ACCUM_DEFAULT=2
    SAVE_STEPS_DEFAULT=10
    ;;
  pilot200)
    SPLIT_NAME="pilot200_200"
    GRAD_ACCUM_DEFAULT=2
    SAVE_STEPS_DEFAULT=20
    ;;
  extend200)
    SPLIT_NAME="pilot200_extra100_100"
    GRAD_ACCUM_DEFAULT=2
    SAVE_STEPS_DEFAULT=10
    EPOCHS_DEFAULT=2
    RESUME_DEFAULT="outputs/rl_msrs_level2_pilot100_full_epoch4_refcfg3_v4_seed${RL_DATA_SEED}/epoch0"
    ;;
  *)
    printf '[ERROR] RL_TIER must be preflight, pilot100, pilot200, or extend200; got %s\n' "${RL_TIER}" >&2
    exit 2
    ;;
esac

export REWARD_GPU="${REWARD_GPU:-2}"
export POLICY_GPUS="${POLICY_GPUS:-3,4,5}"
export NPROC="${NPROC:-3}"
export PROTOCOL="full"
export INIT_CKPT="${INIT_CKPT:-models/imagefusion-sft-epoch4}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-dataset_final/MSRS/rl_level2_seed${RL_DATA_SEED}/msrs_level2_rl_${SPLIT_NAME}_seed${RL_DATA_SEED}.json}"
export TRAIN_SEED="${TRAIN_SEED:-${RL_DATA_SEED}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl_msrs_level2_${RL_TIER}_full_epoch4_refcfg3_v4_seed${RL_DATA_SEED}}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME_DEFAULT}}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${GRAD_ACCUM_DEFAULT}}"
export RL_EPOCHS="${RL_EPOCHS:-${EPOCHS_DEFAULT}}"
export LR="${LR:-3e-7}"
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
export SAVE_STEPS="${SAVE_STEPS:-${SAVE_STEPS_DEFAULT}}"

if [[ "${RL_LOW_MEM:-0}" == "1" ]]; then
  export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-1}"
  export LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-8}"
  export REPLAY_MICRO_BATCH_SIZE="${REPLAY_MICRO_BATCH_SIZE:-1}"
  export FSDP_FULL_LOAD_ALL_RANKS="${FSDP_FULL_LOAD_ALL_RANKS:-0}"
  export POLICY_CUDA_MEMORY_LIMIT_GB="${POLICY_CUDA_MEMORY_LIMIT_GB:-55}"
  export ROLLOUT_USE_CACHE="${ROLLOUT_USE_CACHE:-0}"
  export EMPTY_CUDA_CACHE_BETWEEN_PHASES="${EMPTY_CUDA_CACHE_BETWEEN_PHASES:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32,expandable_segments:True}"
else
  export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-2}"
  export LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-16}"
  export REPLAY_MICRO_BATCH_SIZE="${REPLAY_MICRO_BATCH_SIZE:-1}"
  export FSDP_FULL_LOAD_ALL_RANKS="${FSDP_FULL_LOAD_ALL_RANKS:-1}"
  export ROLLOUT_USE_CACHE="${ROLLOUT_USE_CACHE:-1}"
  export EMPTY_CUDA_CACHE_BETWEEN_PHASES="${EMPTY_CUDA_CACHE_BETWEEN_PHASES:-0}"
fi

# The full-CoT checkpoint was validated with image CFG=3.  Text is decoded
# greedily to preserve its XML protocol; image codes remain sampled for GRPO.
export CFG="${CFG:-3.0}"
export TEXT_TOP_K="${TEXT_TOP_K:-1}"
export IMAGE_TOP_K="${IMAGE_TOP_K:-2000}"
export LAMBDA_TEXT="${LAMBDA_TEXT:-0.1}"
export LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.9}"
export REFERENCE_REWARD_WEIGHT="${REFERENCE_REWARD_WEIGHT:-0.9}"
export QWEN_REWARD_WEIGHT="${QWEN_REWARD_WEIGHT:-0.1}"

export WAIT_FOR_REWARD_READY_BEFORE_POLICY="${WAIT_FOR_REWARD_READY_BEFORE_POLICY:-0}"
export GPU_CHECK_INTERVAL_SEC="${GPU_CHECK_INTERVAL_SEC:-2}"
export GENERATION_LOG_INTERVAL="${GENERATION_LOG_INTERVAL:-512}"
export FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-shard_grad_op}"
export MAX_REPLAY_LOGPROB_ERROR="${MAX_REPLAY_LOGPROB_ERROR:-0.10}"
export MIN_GROUP_REWARD_STD="${MIN_GROUP_REWARD_STD:-0.0001}"
export MAX_CONSECUTIVE_NO_SIGNAL_STEPS="${MAX_CONSECUTIVE_NO_SIGNAL_STEPS:-3}"
export HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-60}"
export DISTRIBUTED_TIMEOUT_SEC="${DISTRIBUTED_TIMEOUT_SEC:-3600}"

cd "$(dirname "$0")/.."
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  printf '[ERROR] TRAIN_MANIFEST not found: %s\n' "${TRAIN_MANIFEST}" >&2
  printf '[HINT] Run: SEED=%s bash scripts/build_msrs_level2_rl_manifests.sh\n' "${RL_DATA_SEED}" >&2
  exit 2
fi
if [[ "${RL_TIER}" != "preflight" ]]; then
  PREFLIGHT_VALIDATION="${PREFLIGHT_VALIDATION:-outputs/rl_msrs_level2_preflight_full_epoch4_refcfg3_v4_seed${RL_DATA_SEED}/preflight_validation.json}"
  if [[ ! -f "${PREFLIGHT_VALIDATION}" ]] || ! "${PYTHON_BIN}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("passed") is True else 1)' \
    "${PREFLIGHT_VALIDATION}"; then
    printf '[ERROR] A passed RL preflight is required before %s: %s\n' \
      "${RL_TIER}" "${PREFLIGHT_VALIDATION}" >&2
    exit 2
  fi
fi

printf '[RL] tier=%s manifest=%s output=%s\n' "${RL_TIER}" "${TRAIN_MANIFEST}" "${OUTPUT_DIR}"
bash scripts/train_msrs_two_level_rl.sh

if [[ "${RL_TIER}" == "preflight" ]]; then
  printf '[RL] validating preflight output=%s\n' "${OUTPUT_DIR}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" \
    scripts/validate_msrs_rl_preflight.py --output_dir "${OUTPUT_DIR}"
fi
