#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PWD}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

NPROC="${NPROC:-4}"
MIN_FREE_MB="${MIN_FREE_MB:-65000}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
TORCHRUN="${TORCHRUN:-torchrun}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mgpt2_ce_cot_final_6000_levelbalanced_v2_480x640_5epoch}"
PRETRAINED="${PRETRAINED:-pretrained/Lumina-mGPT-2.0-Omni}"
DATA_CONFIG="${DATA_CONFIG:-configs/sft/mgpt2_ce_cot_final_6000_480x640.yaml}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-28672}"
CE_LOSS_CHUNK_SIZE="${CE_LOSS_CHUNK_SIZE:-512}"
GRAD_PRECISION="${GRAD_PRECISION:-fp32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-0.0}"
AUTO_RESUME="${AUTO_RESUME:-0}"
TARGET_PROTOCOL="${TARGET_PROTOCOL:-full}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne "${NPROC}" ]]; then
  printf '[ERROR] CUDA_VISIBLE_DEVICES has %s GPUs but NPROC=%s\n' \
    "${#GPU_IDS[@]}" "${NPROC}" >&2
  exit 2
fi

if [[ "${SKIP_GPU_CHECK}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '[ERROR] nvidia-smi not found; use SKIP_GPU_CHECK=1 only after checking GPUs manually.\n' >&2
    exit 2
  fi
  for local_rank in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[${local_rank}]}"
    free_mb="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    printf '[GPU] local_rank=%s physical_id=%s free_mb=%s required_min_mb=%s\n' \
      "${local_rank}" "${gpu_id}" "${free_mb}" "${MIN_FREE_MB}"
    if [[ -z "${free_mb}" || "${free_mb}" -lt "${MIN_FREE_MB}" ]]; then
      printf '[ERROR] GPU %s does not have enough free memory for SFT. Free memory on this GPU, choose another GPU set, or rerun with a lower MIN_FREE_MB only for an intentional test.\n' \
        "${gpu_id}" >&2
      exit 3
    fi
  done
fi

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ "${AUTO_RESUME}" == "1" ]]; then
  printf "[RUN] auto_resume=1\n"
else
  RESUME_ARGS=(--no_auto_resume)
  if compgen -G "${OUTPUT_DIR}/epoch*" > /dev/null; then
    printf '[ERROR] Fresh-start mode but %s already contains epoch checkpoints.\n' "${OUTPUT_DIR}" >&2
    printf '[ERROR] Use a new OUTPUT_DIR, or set AUTO_RESUME=1 only when you intentionally want to resume.\n' >&2
    exit 3
  fi
  printf "[RUN] auto_resume=0 fresh_start_from=%s\n" "${PRETRAINED}"
fi

printf "[RUN] nproc=%s gpus=%s batch=%s accum=%s epochs=%s max_seq_len=%s ce_loss_chunk_size=%s\n" \
  "${NPROC}" "${CUDA_VISIBLE_DEVICES}" "${BATCH_SIZE}" "${ACCUM_ITER}" \
  "${EPOCHS}" "${MAX_SEQ_LEN}" "${CE_LOSS_CHUNK_SIZE}"
printf "[RUN] data_config=%s\n" "${DATA_CONFIG}"
printf "[RUN] output_dir=%s\n" "${OUTPUT_DIR}"
printf "[RUN] PYTORCH_CUDA_ALLOC_CONF=%s\n" "${PYTORCH_CUDA_ALLOC_CONF}"
printf "[RUN] min_free_mb=%s grad_precision=%s\n" "${MIN_FREE_MB}" "${GRAD_PRECISION}"
printf "[RUN] target_protocol=%s\n" "${TARGET_PROTOCOL}"

"${TORCHRUN}" --nproc_per_node="${NPROC}" imagefusion_r1/trainers/finetune_solver_mgpt2_ce.py \
  --init_from "${PRETRAINED}" \
  --pretrained_name_or_path "${PRETRAINED}" \
  --data_config "${DATA_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_iter "${ACCUM_ITER}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs 0.0 \
  --lr "${LR}" \
  --min_lr "${MIN_LR}" \
  --wd 0.0 \
  --clip_grad 4.0 \
  --num_workers 0 \
  --precision bf16 \
  --grad_precision "${GRAD_PRECISION}" \
  --data_parallel fsdp \
  --checkpointing \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --ce_loss_chunk_size "${CE_LOSS_CHUNK_SIZE}" \
  --target_protocol "${TARGET_PROTOCOL}" \
  --z_loss_weight 0 \
  --ce_weight_infrared_image 1.0 \
  --ce_weight_visible_image 1.0 \
  --ce_weight_fused_image 1.0 \
  --ce_weight_text 1.0 \
  --ce_weight_image_structure 1.0 \
  --unmask_image_logits \
  --save_interval 1 \
  --save_iteration_interval 100000 \
  --ckpt_max_keep 0 \
  "${RESUME_ARGS[@]}"
