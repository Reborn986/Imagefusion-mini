from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import gc
import functools
import json
import math
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

try:
    from accelerate import init_empty_weights
except Exception:  # pragma: no cover - accelerate is available in the training env
    init_empty_weights = None

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
except Exception:  # pragma: no cover - optional on older torch builds
    CheckpointImpl = None
    apply_activation_checkpointing = None
    checkpoint_wrapper = None


REPO_ROOT = Path(__file__).resolve().parents[2]
LUMINA2_ROOT = REPO_ROOT / "third_party" / "lumina_mgpt_2"
LUMINA2_IMPL = LUMINA2_ROOT / "lumina_mgpt"

for import_path in (REPO_ROOT, LUMINA2_IMPL, LUMINA2_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

CACHE_ROOT = Path(os.environ.get("MSRS_RUNTIME_CACHE_DIR", f"/tmp/msrs_runtime_cache_{os.getuid()}"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
for cache_dir in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["XDG_CACHE_HOME"])):
    cache_dir.mkdir(parents=True, exist_ok=True)

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound  # noqa: E402

relax_huggingface_hub_upper_bound()

from data.convertsation import Conversation  # noqa: E402
from imagefusion_r1.inference.inference_solver_mgpt2_ce import (  # noqa: E402
    FixedGridMultiModalLogitsProcessor,
    ImageBlockCFGLogitsProcessor,
    InterleavedTopKLogitsProcessor,
    MSRSCEInferenceItemProcessor,
    infer_output_names_from_text,
)
from imagefusion_r1.rl.grpo_data import (  # noqa: E402
    MSRSRLSample,
    iter_distributed_batches,
    load_msrs_rl_samples,
)
from imagefusion_r1.rl.msrs_two_level_reward import (  # noqa: E402
    ImageLevelRewardResult,
    image_reward_from_qwen_scores,
    score_format_gate,
    score_two_level_reward,
)
from imagefusion_r1.rl.reference_image_reward import (  # noqa: E402
    ThreeImageReferenceReward,
    score_three_image_reference,
)
from model.chameleon import ChameleonForConditionalGeneration  # noqa: E402
from model.chameleon.configuration_chameleon import ChameleonConfig  # noqa: E402


IMAGE_PLACEHOLDER = "<|image|>"
IMAGE_TOKEN_START = 155000
IMAGE_TOKEN_END = 171383


def default_gpt_prefix(protocol: str) -> str:
    if protocol == "fused_only":
        return "<cot>\n"
    return "<infrared_cot>\n"


def rank0_print(*values: Any, rank: int = 0) -> None:
    if rank == 0:
        print(*values, flush=True)


def safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")
    return value[:160] or fallback


def dtype_from_precision(precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, FSDP) else model


def model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return fallback


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def empty_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda=unavailable"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"cuda_alloc={allocated:.2f}GiB cuda_reserved={reserved:.2f}GiB cuda_peak={peak:.2f}GiB"


@dataclass
class GeneratedCompletion:
    sample: MSRSRLSample
    generation_index: int
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logps: list[float]
    generated_text: str
    generated_images: list[Image.Image] = field(default_factory=list)
    generation_elapsed_sec: float = 0.0
    output_dir: Path | None = None
    fused_image_path: str = ""
    reward: float = 0.0
    reward_detail: Mapping[str, Any] = field(default_factory=dict)
    image_reward: ImageLevelRewardResult = field(
        default_factory=lambda: ImageLevelRewardResult(score=0.0, error="not_scored")
    )
    reference_reward: ThreeImageReferenceReward = field(
        default_factory=lambda: ThreeImageReferenceReward(score=0.0, error="not_scored")
    )


class MSRSGeneratedDecoder:
    def __init__(self, item_processor: MSRSCEInferenceItemProcessor) -> None:
        self.item_processor = item_processor

    def _show_images(self, batch: torch.Tensor) -> Image.Image:
        scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        return Image.fromarray(reshaped.numpy())

    def _grid_from_token_id(self, token_id: int) -> int:
        return int(self.item_processor.id2token(torch.tensor(token_id))[-5:-1]) - 8800

    def _decode_image_cache(self, cache: list[int]) -> Image.Image:
        if len(cache) < 2:
            raise ValueError("image cache is missing h/w grid tokens")

        h_grids = self._grid_from_token_id(cache[0])
        w_grids = self._grid_from_token_id(cache[1])
        h_latent = h_grids * 4
        w_latent = w_grids * 4
        new_line_id = self.item_processor.token2id(self.item_processor.new_line_token)
        image_tokens = [token_id for token_id in cache[2:] if token_id != new_line_id]

        expected = h_latent * w_latent
        if len(image_tokens) != expected:
            raise ValueError(f"bad image token count: got {len(image_tokens)}, expected {expected}")

        device = self.item_processor.vqgan.encoder.conv_in.weight.device
        codes = torch.tensor(image_tokens, dtype=torch.long, device=device) - IMAGE_TOKEN_START
        decoded = self.item_processor.vqgan.decode_code(codes, h_latent, w_latent)
        return self._show_images(decoded)

    def decode_ids(self, token_ids: list[int]) -> tuple[str, list[Image.Image]]:
        generated_images: list[Image.Image] = []
        text_token_ids: list[int] = []
        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        placeholder_image_id = self.item_processor.token2id(IMAGE_PLACEHOLDER)

        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            if token_id != image_start_id:
                text_token_ids.append(token_id)
                i += 1
                continue

            cache: list[int] = []
            found_end = False
            for j in range(i + 1, len(token_ids)):
                if token_ids[j] == image_end_id:
                    try:
                        generated_images.append(self._decode_image_cache(cache))
                        text_token_ids.append(placeholder_image_id)
                    except Exception as exc:
                        print(f"[WARN] image decode failed: {exc!r}", flush=True)
                    i = j + 1
                    found_end = True
                    break
                cache.append(token_ids[j])

            if not found_end:
                print("[WARN] image_start appeared without image_end", flush=True)
                break

        generated_text = self.item_processor.tokenizer.decode(text_token_ids)
        return generated_text, generated_images


class MSRSPolicySampler:
    def __init__(
        self,
        model: nn.Module,
        item_processor: MSRSCEInferenceItemProcessor,
        *,
        device: torch.device,
        precision: str,
        target_height: int,
        target_width: int,
        max_position_embeddings: int,
        rank: int,
        generation_log_interval: int,
    ) -> None:
        self.model = model
        self.item_processor = item_processor
        self.decoder = MSRSGeneratedDecoder(item_processor)
        self.device = device
        self.dtype = dtype_from_precision(precision)
        self.target_height = target_height
        self.target_width = target_width
        self.max_position_embeddings = max_position_embeddings
        self.rank = rank
        self.generation_log_interval = max(0, int(generation_log_interval))
        self.stop_token_id = item_processor.token2id(Conversation.sep_token)

    def create_logits_processor(
        self,
        *,
        image_top_k: int,
        text_top_k: int,
        stop_after_images: int,
        initial_image_end_count: int,
        cfg: float,
        use_cache: bool = True,
    ):
        module = unwrap_model(self.model)
        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        image_next_line_id = self.item_processor.token2id(self.item_processor.new_line_token)
        h_grids = self.target_height // self.item_processor.patch_size
        w_grids = self.target_width // self.item_processor.patch_size

        processors = []
        if float(cfg) != 1.0:
            processors.append(
                ImageBlockCFGLogitsProcessor(
                    guidance_scale=cfg,
                    model=self.model,
                    image_start_token_id=image_start_id,
                    image_end_token_id=image_end_id,
                    use_cache=use_cache,
                )
            )
        processors.extend(
            [
                FixedGridMultiModalLogitsProcessor(
                    item_processor=self.item_processor,
                    image_start_token_id=image_start_id,
                    image_end_token_id=image_end_id,
                    image_next_line_token_id=image_next_line_id,
                    voc_size=module.config.vocab_size,
                    h_grids=h_grids,
                    w_grids=w_grids,
                    stop_after_images=stop_after_images,
                    initial_image_end_count=initial_image_end_count,
                    stop_token_id=self.stop_token_id,
                ),
                InterleavedTopKLogitsProcessor(
                    image_top_k=image_top_k,
                    text_top_k=text_top_k,
                    image_start_token_id=image_start_id,
                    image_end_token_id=image_end_id,
                ),
            ]
        )
        return processors

    @staticmethod
    def _sample_token(scores: torch.Tensor, *, temperature: float, do_sample: bool) -> torch.Tensor:
        if not do_sample:
            return torch.argmax(scores, dim=-1, keepdim=True)

        temperature = max(float(temperature), 1e-6)
        probs = torch.softmax(scores / temperature, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        if bool((row_sums <= 0).any().item()):
            return torch.argmax(scores, dim=-1, keepdim=True)
        probs = probs / row_sums
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _next_token_logp(
        scores: torch.Tensor,
        next_token: torch.Tensor,
        *,
        temperature: float,
        do_sample: bool,
    ) -> torch.Tensor:
        if do_sample:
            scores = scores / max(float(temperature), 1e-6)
        log_probs = torch.log_softmax(scores, dim=-1)
        log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=-1e9)
        return torch.gather(log_probs, dim=-1, index=next_token).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        sample: MSRSRLSample,
        *,
        generation_index: int,
        seed: int,
        max_new_tokens: int,
        image_top_k: int,
        text_top_k: int,
        cfg: float,
        temperature: float,
        do_sample: bool,
        gpt_prefix: str,
        stop_after_images: int,
        use_cache: bool,
    ) -> GeneratedCompletion:
        started = time.time()
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()
        prompt_ids = self.item_processor.build_input_ids(
            infrared_path=sample.infrared_degraded_path,
            visible_path=sample.visible_degraded_path,
            gpt_prefix=gpt_prefix,
        )
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_ids = input_ids.clone()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        initial_image_end_count = int((input_ids[0] == image_end_id).sum().item())
        remaining_budget = self.max_position_embeddings - input_ids.shape[1]
        if remaining_budget <= 8:
            raise ValueError(
                "Prompt leaves no safe generation budget: "
                f"prompt_tokens={input_ids.shape[1]} "
                f"max_position_embeddings={self.max_position_embeddings}"
            )
        actual_max_new = min(int(max_new_tokens), remaining_budget - 8)
        processors = self.create_logits_processor(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            stop_after_images=stop_after_images,
            initial_image_end_count=initial_image_end_count,
            cfg=cfg,
            use_cache=use_cache,
        )
        print(
            f"[GEN][rank={self.rank}] configured sample={sample.id} gen={generation_index} "
            f"prompt_tokens={len(prompt_ids)} max_new_tokens={actual_max_new} "
            f"cfg={cfg} text_top_k={text_top_k} image_top_k={image_top_k} "
            f"use_cache={use_cache} {cuda_memory_summary(self.device)}",
            flush=True,
        )

        past_key_values = None
        step_input = input_ids
        old_logps: list[float] = []
        for _step in range(actual_max_new):
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                output = self.model(
                    input_ids=step_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            logits = output.logits[:, -1, :].float()
            past_key_values = output.past_key_values if use_cache else None
            scores = logits
            for processor in processors:
                scores = processor(generated_ids, scores)
            next_token = self._sample_token(scores, temperature=temperature, do_sample=do_sample)
            token_logp = self._next_token_logp(
                scores,
                next_token,
                temperature=temperature,
                do_sample=do_sample,
            )
            old_logps.append(float(token_logp.detach().cpu().item()))
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, dtype=torch.long)], dim=1)
            step_input = next_token if use_cache else generated_ids
            if int(next_token.item()) == self.stop_token_id:
                print(
                    f"[GEN][rank={self.rank}] sample={sample.id} gen={generation_index} "
                    f"stop_token at tokens={_step + 1}",
                    flush=True,
                )
                break
            if self.generation_log_interval and (_step + 1) % self.generation_log_interval == 0:
                image_end_count = int((generated_ids[0] == image_end_id).sum().item()) - initial_image_end_count
                elapsed = max(1e-6, time.time() - started)
                tokens_per_sec = (_step + 1) / elapsed
                remaining_tokens = max(0, actual_max_new - (_step + 1))
                eta_sec = remaining_tokens / max(tokens_per_sec, 1e-8)
                print(
                    f"[GEN][rank={self.rank}] sample={sample.id} gen={generation_index} "
                    f"tokens={_step + 1}/{actual_max_new} images_done={image_end_count}/{stop_after_images} "
                    f"elapsed_sec={elapsed:.1f} tok_per_sec={tokens_per_sec:.3f} "
                    f"max_token_eta_sec={eta_sec:.1f} {cuda_memory_summary(self.device)}",
                    flush=True,
                )

        completion_ids = generated_ids[0, len(prompt_ids) :].detach().cpu().tolist()
        decode_ids = completion_ids[:-1] if completion_ids and completion_ids[-1] == self.stop_token_id else completion_ids
        generated_text, generated_images = self.decoder.decode_ids(decode_ids)
        if gpt_prefix:
            generated_text = gpt_prefix + generated_text
        elapsed = time.time() - started
        stochastic_action_count = sum(abs(value) > 1e-8 for value in old_logps)
        stochastic_action_fraction = stochastic_action_count / max(1, len(old_logps))
        print(
            f"[GEN][rank={self.rank}] decoded sample={sample.id} gen={generation_index} "
            f"completion_tokens={len(completion_ids)} images={len(generated_images)} "
            f"nonzero_logp_tokens={stochastic_action_count} "
            f"nonzero_logp_fraction={stochastic_action_fraction:.4f} "
            f"elapsed_sec={elapsed:.1f} {cuda_memory_summary(self.device)}",
            flush=True,
        )
        return GeneratedCompletion(
            sample=sample,
            generation_index=generation_index,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            old_logps=old_logps,
            generated_text=generated_text,
            generated_images=generated_images,
            generation_elapsed_sec=elapsed,
        )


def load_policy_model(
    model_path: str,
    *,
    rank: int,
    max_position_embeddings: int,
    precision: str,
    meta_init: bool,
) -> ChameleonForConditionalGeneration:
    dtype = dtype_from_precision(precision)
    if meta_init:
        if init_empty_weights is None:
            raise RuntimeError("accelerate.init_empty_weights is required for non-rank0 FSDP init.")
        with init_empty_weights():
            config = ChameleonConfig.from_pretrained(
                model_path,
                max_position_embeddings=max_position_embeddings,
                mask_image_logits=False,
                torch_dtype=dtype,
            )
            model = ChameleonForConditionalGeneration(config)
    else:
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path,
            max_position_embeddings=max_position_embeddings,
            mask_image_logits=False,
            torch_dtype=dtype,
            device_map="cpu",
        )

    model.config.max_position_embeddings = max_position_embeddings
    model.config.mask_image_logits = False
    if hasattr(model.model, "vqmodel"):
        del model.model.vqmodel
    rank0_print("[LOAD] policy model initialized from", model_path, rank=rank)
    return model


def fsdp_wrap_modules(model: ChameleonForConditionalGeneration) -> list[nn.Module]:
    return [*list(model.model.layers), model.lm_head, model.model.embed_tokens]


def activation_checkpoint_modules(model: ChameleonForConditionalGeneration) -> list[nn.Module]:
    # The transformer blocks dominate activation memory.  lm_head is
    # checkpointed explicitly in small vocabulary-projection chunks during
    # GRPO replay, so wrapping it here as well would create nested checkpoints
    # around a child FSDP module.  Embeddings are small and likewise do not
    # benefit from activation checkpointing.
    return list(model.model.layers)


def apply_checkpointing(model: nn.Module, modules: Sequence[nn.Module]) -> None:
    if apply_activation_checkpointing is None or checkpoint_wrapper is None or CheckpointImpl is None:
        print("[WARN] torch activation checkpointing utilities not available; skip checkpointing.", flush=True)
        return
    wrapper = lambda module: checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=lambda submodule: submodule in modules)


def wrap_model_for_training(
    model: ChameleonForConditionalGeneration,
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    trainable: bool,
) -> nn.Module:
    print(
        f"[RL][rank={rank}] FSDP wrap start trainable={trainable} "
        f"world_size={world_size} device={device}",
        flush=True,
    )
    for param in model.parameters():
        param.requires_grad_(trainable)

    if world_size <= 1:
        wrapped_single = model.to(device)
        print(f"[RL][rank={rank}] single-GPU model move done trainable={trainable}", flush=True)
        return wrapped_single

    modules = fsdp_wrap_modules(model)
    checkpoint_modules = activation_checkpoint_modules(model)
    print(f"[RL][rank={rank}] FSDP target modules={len(modules)} trainable={trainable}", flush=True)
    full_load_all_ranks = bool(getattr(args, "fsdp_full_load_all_ranks", False))
    sync_module_states = not full_load_all_ranks
    param_init_fn = None if (rank == 0 or full_load_all_ranks) else lambda module: module.to_empty(device=device, recurse=False)
    print(
        f"[RL][rank={rank}] FSDP init mode "
        f"full_load_all_ranks={full_load_all_ranks} sync_module_states={sync_module_states}",
        flush=True,
    )
    strategy_name = str(getattr(args, "fsdp_sharding_strategy", "shard_grad_op")).lower()
    if world_size > 1 and strategy_name == "full_shard":
        raise ValueError(
            "Distributed autoregressive RL forbids FSDP FULL_SHARD: ranks may stop at "
            "different token counts and execute different numbers of image-CFG forwards, "
            "which makes per-forward parameter all-gathers unsafe. Use shard_grad_op."
        )
    sharding_strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }[strategy_name]
    print(
        f"[RL][rank={rank}] FSDP sharding_strategy={strategy_name}",
        flush=True,
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module in modules,
        ),
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(
            param_dtype=dtype_from_precision(args.precision),
            reduce_dtype=dtype_from_precision(args.grad_precision),
        ),
        device_id=device,
        sync_module_states=sync_module_states,
        limit_all_gathers=True,
        use_orig_params=True,
        param_init_fn=param_init_fn,
    )
    if trainable and args.checkpointing:
        print(f"[RL][rank={rank}] activation checkpointing start", flush=True)
        apply_checkpointing(wrapped, checkpoint_modules)
        print(f"[RL][rank={rank}] activation checkpointing done", flush=True)
    torch.cuda.synchronize(device)
    print(f"[RL][rank={rank}] FSDP wrap done trainable={trainable}", flush=True)
    return wrapped


class MSRSGRPOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process_started_at = time.time()
        self.phase = "distributed_setup"
        self.phase_started_at = self.process_started_at
        self.phase_detail = ""
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.rank, self.world_size, self.local_rank = self._setup_distributed()
        self.device = torch.device("cuda", self.local_rank)
        self._start_heartbeat()
        self._set_phase("output_setup")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "rl_samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.reward_jsonl = self.output_dir / "logs" / f"reward_details_rank{self.rank:05d}.jsonl"

        self._set_phase("manifest_load", str(args.train_manifest))
        self.samples = load_msrs_rl_samples(args.train_manifest, max_samples=args.max_samples)
        if len(self.samples) < self.world_size * args.per_device_batch_size:
            raise ValueError(
                "Not enough samples for one distributed batch: "
                f"samples={len(self.samples)} world_size={self.world_size} "
                f"per_device_batch_size={args.per_device_batch_size}"
            )
        if args.num_generations < 2:
            raise ValueError("GRPO needs num_generations >= 2 to form a within-prompt reward group.")

        tokenizer_path = args.tokenizer or args.base_model
        self._set_phase("item_processor_init", tokenizer_path)
        print(
            f"[RL][rank={self.rank}] item_processor init start tokenizer={tokenizer_path} "
            f"device=cuda:{self.local_rank}",
            flush=True,
        )
        self.item_processor = MSRSCEInferenceItemProcessor(
            tokenizer=tokenizer_path,
            target_size=max(args.target_height, args.target_width),
            target_height=args.target_height,
            target_width=args.target_width,
            device=f"cuda:{self.local_rank}",
        )
        print(f"[RL][rank={self.rank}] item_processor init done", flush=True)

        policy_checkpoint = args.resume_from_checkpoint or args.init_ckpt
        self._set_phase("policy_load", str(policy_checkpoint))
        print(
            f"[RL][rank={self.rank}] policy load start init_ckpt={policy_checkpoint} "
            f"meta_init={self.world_size > 1 and self.rank != 0 and not args.fsdp_full_load_all_ranks}",
            flush=True,
        )
        policy = load_policy_model(
            policy_checkpoint,
            rank=self.rank,
            max_position_embeddings=args.max_position_embeddings,
            precision=args.precision,
            meta_init=self.world_size > 1 and self.rank != 0 and not args.fsdp_full_load_all_ranks,
        )
        print(f"[RL][rank={self.rank}] policy load done", flush=True)
        self._set_phase("fsdp_wrap", "policy")
        self.policy_model = wrap_model_for_training(
            policy,
            args=args,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            trainable=True,
        )

        self.reference_model: nn.Module | None = None
        if args.kl_beta > 0 and not args.no_reference_kl:
            self._set_phase("reference_model_load", str(args.reference_model or args.init_ckpt))
            print(
                f"[RL][rank={self.rank}] reference load start "
                f"ckpt={args.reference_model or args.init_ckpt}",
                flush=True,
            )
            ref_model = load_policy_model(
                args.reference_model or args.init_ckpt,
                rank=self.rank,
                max_position_embeddings=args.max_position_embeddings,
                precision=args.precision,
                meta_init=self.world_size > 1 and self.rank != 0 and not args.fsdp_full_load_all_ranks,
            )
            print(f"[RL][rank={self.rank}] reference load done", flush=True)
            self.reference_model = wrap_model_for_training(
                ref_model,
                args=args,
                rank=self.rank,
                world_size=self.world_size,
                device=self.device,
                trainable=False,
            )
            self.reference_model.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        self.sampler = MSRSPolicySampler(
            self.policy_model,
            self.item_processor,
            device=self.device,
            precision=args.precision,
            target_height=args.target_height,
            target_width=args.target_width,
            max_position_embeddings=args.max_position_embeddings,
            rank=self.rank,
            generation_log_interval=args.generation_log_interval,
        )
        self.micro_step = 0
        self.update_step = 0
        self.grad_accum_count = 0
        self.consecutive_no_signal_steps = 0
        self.start_epoch = 0
        self.resume_batch_in_epoch = 0
        self.current_epoch = 0
        self.batch_in_epoch = -1
        if args.resume_from_checkpoint:
            self._set_phase("resume_state_load", str(args.resume_from_checkpoint))
            self._load_training_state(Path(args.resume_from_checkpoint))
        rank0_print(
            "[RL] trainer ready",
            f"samples={len(self.samples)}",
            f"world_size={self.world_size}",
            f"num_generations={args.num_generations}",
            f"reference_kl={self.reference_model is not None}",
            f"kl_beta={args.kl_beta if self.reference_model is not None else 0.0}",
            f"reward_batch_size={args.reward_batch_size}",
            rank=self.rank,
        )
        self._save_run_config()
        if int(args.text_top_k) == 1:
            rank0_print(
                "[WARN] TEXT_TOP_K=1 makes text-token sampling deterministic; "
                "text-token policy gradients are effectively zero after top-k filtering. "
                "R_text still ranks whole trajectories but mainly updates sampled image actions.",
                rank=self.rank,
            )
        self._set_phase("trainer_ready")

    def _set_phase(self, phase: str, detail: str = "") -> None:
        now = time.time()
        previous = getattr(self, "phase", "unknown")
        previous_started = getattr(self, "phase_started_at", now)
        self.phase = str(phase)
        self.phase_detail = str(detail)
        self.phase_started_at = now
        device = getattr(self, "device", torch.device("cpu"))
        rank = getattr(self, "rank", int(os.environ.get("RANK", "0")))
        print(
            f"[PHASE][rank={rank}] {previous}->{self.phase} "
            f"previous_elapsed_sec={now - previous_started:.2f} "
            f"detail={self.phase_detail or '-'} {cuda_memory_summary(device)}",
            flush=True,
        )

    def _start_heartbeat(self) -> None:
        interval = float(self.args.heartbeat_interval_sec)
        if self.rank != 0 or interval <= 0:
            return

        def run() -> None:
            while not self._heartbeat_stop.wait(interval):
                now = time.time()
                print(
                    f"[HEARTBEAT][rank=0] pid={os.getpid()} phase={self.phase} "
                    f"detail={self.phase_detail or '-'} "
                    f"phase_sec={now - self.phase_started_at:.1f} "
                    f"total_sec={now - self.process_started_at:.1f} "
                    f"micro_step={getattr(self, 'micro_step', 0)} "
                    f"update_step={getattr(self, 'update_step', 0)} "
                    f"{cuda_memory_summary(self.device)}",
                    flush=True,
                )

        self._heartbeat_thread = threading.Thread(
            target=run,
            name="msrs-grpo-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def _setup_distributed(self) -> tuple[int, int, int]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        print(
            f"[DIST][rank={rank}] setup start pid={os.getpid()} host={socket.gethostname()} "
            f"world_size={world_size} local_rank={local_rank} "
            f"master={os.environ.get('MASTER_ADDR', '?')}:{os.environ.get('MASTER_PORT', '?')}",
            flush=True,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("MSRS mGPT2 GRPO training requires CUDA.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(local_rank)
        print(
            f"[DIST][rank={rank}] cuda selected name={properties.name} "
            f"total_gib={properties.total_memory / 1024**3:.2f} {cuda_memory_summary(device)}",
            flush=True,
        )
        memory_limit_gb = os.environ.get("POLICY_CUDA_MEMORY_LIMIT_GB", "").strip()
        memory_fraction = os.environ.get("POLICY_CUDA_MEMORY_FRACTION", "").strip()
        if memory_limit_gb or memory_fraction:
            if memory_limit_gb:
                total_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
                fraction = float(memory_limit_gb) / total_gb
            else:
                fraction = float(memory_fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    "POLICY_CUDA_MEMORY_LIMIT_GB/POLICY_CUDA_MEMORY_FRACTION must resolve "
                    f"to a fraction in (0, 1], got {fraction:.4f}"
                )
            torch.cuda.set_per_process_memory_fraction(fraction, local_rank)
            print(
                f"[RL] rank={rank} local_rank={local_rank} cuda_memory_fraction={fraction:.4f}",
                flush=True,
            )
        if world_size > 1 and not dist.is_initialized():
            timeout_sec = max(1.0, float(self.args.distributed_timeout_sec))
            print(
                f"[DIST][rank={rank}] init_process_group start backend=nccl timeout_sec={timeout_sec:.0f}",
                flush=True,
            )
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=timeout_sec),
            )
            print(f"[DIST][rank={rank}] init_process_group done", flush=True)
        return rank, world_size, local_rank

    def _save_run_config(self) -> None:
        if self.rank != 0:
            return
        payload = vars(self.args).copy()
        payload.update({"world_size": self.world_size})
        (self.output_dir / "run_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _autocast(self):
        dtype = dtype_from_precision(self.args.precision)
        return torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)

    def _load_training_state(self, checkpoint_dir: Path) -> None:
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing trainer_state.json: {checkpoint_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        optimizer_path = checkpoint_dir / f"optimizer.{self.rank:05d}-of-{self.world_size:05d}.pth"
        if isinstance(self.policy_model, FSDP):
            optimizer_shards = sorted(checkpoint_dir.glob("optimizer.*-of-*.pth"))
            if len(optimizer_shards) != self.world_size:
                raise ValueError(
                    "FSDP resume requires the same world size as the saved optimizer: "
                    f"checkpoint_shards={len(optimizer_shards)} current_world_size={self.world_size}"
                )
            if not optimizer_path.is_file():
                raise FileNotFoundError(f"resume checkpoint is missing local optimizer state: {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
        else:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if not optimizer_path.is_file():
                raise FileNotFoundError(f"resume checkpoint is missing optimizer state: {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
        self.optimizer.load_state_dict(optimizer_state)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(self.args.lr)
            parameter_group["weight_decay"] = float(self.args.weight_decay)
        if self.args.optimizer_cpu_offload:
            self._move_optimizer_state(torch.device("cpu"))
        self.micro_step = int(state.get("micro_step", 0))
        self.update_step = int(state.get("update_step", 0))
        self.grad_accum_count = int(state.get("grad_accum_count", 0))
        saved_epoch = int(state.get("epoch", 0))
        if bool(state.get("epoch_complete", False)):
            self.start_epoch = saved_epoch + 1
            self.resume_batch_in_epoch = 0
        else:
            self.start_epoch = saved_epoch
            self.resume_batch_in_epoch = int(state.get("batch_in_epoch", -1)) + 1
        print(
            f"[RESUME][rank={self.rank}] checkpoint={checkpoint_dir} "
            f"micro_step={self.micro_step} update_step={self.update_step} "
            f"start_epoch={self.start_epoch} batch_in_epoch={self.resume_batch_in_epoch}",
            flush=True,
        )

    def save_completion_outputs(self, completion: GeneratedCompletion, *, epoch: int, local_index: int) -> None:
        sample_id = safe_name(completion.sample.id, f"sample_{local_index:04d}")
        save_dir = (
            self.output_dir
            / "rl_samples"
            / f"step{self.micro_step:06d}"
            / f"rank{self.rank:05d}_{local_index:02d}_{sample_id}_gen{completion.generation_index}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        completion.output_dir = save_dir
        (save_dir / "sample.json").write_text(
            json.dumps(completion.sample.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (save_dir / "generated_text.txt").write_text(completion.generated_text, encoding="utf-8")

        names = infer_output_names_from_text(completion.generated_text, len(completion.generated_images))
        for index, image in enumerate(completion.generated_images):
            out_path = save_dir / names[index]
            image.save(out_path)
            if names[index] == "fused_image.png":
                completion.fused_image_path = str(out_path)

        if not completion.fused_image_path and len(completion.generated_images) >= 3:
            fallback = save_dir / "fused_image.png"
            if not fallback.exists():
                completion.generated_images[2].save(fallback)
            completion.fused_image_path = str(fallback)

        status = {
            "epoch": epoch,
            "micro_step": self.micro_step,
            "rank": self.rank,
            "id": completion.sample.id,
            "generation_index": completion.generation_index,
            "num_generated_images": len(completion.generated_images),
            "completion_tokens": len(completion.completion_ids),
            "generation_elapsed_sec": completion.generation_elapsed_sec,
            "old_logps": {
                "count": len(completion.old_logps),
                "mean": (sum(completion.old_logps) / len(completion.old_logps)) if completion.old_logps else 0.0,
                "nonzero_count": sum(abs(value) > 1e-8 for value in completion.old_logps),
                "nonzero_fraction": (
                    sum(abs(value) > 1e-8 for value in completion.old_logps)
                    / max(1, len(completion.old_logps))
                ),
            },
            "fused_image_path": completion.fused_image_path,
        }
        (save_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _post_reward_server(self, items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        payload = json.dumps({"items": list(items)}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.args.reward_server_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        started = time.time()
        print(
            f"[REWARD][rank={self.rank}] POST start items={len(items)} "
            f"url={self.args.reward_server_url}",
            flush=True,
        )
        with urllib.request.urlopen(request, timeout=self.args.reward_timeout_sec) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if not isinstance(raw, Mapping) or raw.get("ok") is not True:
            raise RuntimeError(f"reward server returned non-ok response: {raw}")
        results = raw.get("results")
        if not isinstance(results, list):
            raise RuntimeError(f"reward server response missing results list: {raw}")
        print(
            f"[REWARD][rank={self.rank}] POST done items={len(items)} "
            f"results={len(results)} elapsed_sec={time.time() - started:.2f}",
            flush=True,
        )
        return results

    @staticmethod
    def _image_reward_from_server_result(raw: Mapping[str, Any]) -> ImageLevelRewardResult:
        if "score" in raw:
            try:
                return ImageLevelRewardResult(
                    score=float(raw.get("score", 0.0)),
                    artifact_suppression=float(raw.get("artifact_suppression", 0.0)),
                    visible_preservation=float(raw.get("visible_preservation", 0.0)),
                    infrared_preservation=float(raw.get("infrared_preservation", 0.0)),
                    fusion_naturalness=float(raw.get("fusion_naturalness", 0.0)),
                    semantic_consistency=float(raw.get("semantic_consistency", 0.0)),
                    overall=None if raw.get("overall") is None else float(raw.get("overall", 0.0)),
                    raw=raw.get("raw") if isinstance(raw.get("raw"), Mapping) else dict(raw),
                    error=str(raw.get("error", "")),
                )
            except (TypeError, ValueError):
                pass
        if isinstance(raw.get("raw"), Mapping):
            return image_reward_from_qwen_scores(raw["raw"])
        return image_reward_from_qwen_scores(raw)

    def score_image_rewards(self, completions: Sequence[GeneratedCompletion]) -> list[ImageLevelRewardResult]:
        rewards = []
        valid_indices = []
        for index, completion in enumerate(completions):
            gate = score_format_gate(
                {
                    "generated_text": completion.generated_text,
                    "generated_images": completion.generated_images,
                },
                protocol=self.args.protocol,
            )
            if not gate.ok:
                rewards.append(ImageLevelRewardResult(score=0.0, error="format_gate_failed"))
            elif not completion.fused_image_path or not Path(completion.fused_image_path).is_file():
                rewards.append(ImageLevelRewardResult(score=0.0, error="missing_fused_image"))
            else:
                rewards.append(ImageLevelRewardResult(score=0.0, error="pending_qwen_reward"))
                valid_indices.append(index)
        print(
            f"[REWARD][rank={self.rank}] image reward candidates="
            f"{len(valid_indices)}/{len(completions)} skip={self.args.skip_image_reward}",
            flush=True,
        )
        if not valid_indices or self.args.skip_image_reward:
            if self.args.skip_image_reward:
                return [ImageLevelRewardResult(score=0.0, error="image_reward_skipped") for _ in completions]
            return rewards

        request_items = []
        for index in valid_indices:
            completion = completions[index]
            row = completion.sample.to_dict()
            row.update(
                {
                    "sample_id": completion.sample.id,
                    "fused_image_path": completion.fused_image_path,
                }
            )
            request_items.append(row)

        batch_size = max(1, int(self.args.reward_batch_size))
        for start in range(0, len(request_items), batch_size):
            end = min(start + batch_size, len(request_items))
            chunk_items = request_items[start:end]
            chunk_indices = valid_indices[start:end]
            try:
                print(
                    f"[REWARD][rank={self.rank}] chunk start "
                    f"{start // batch_size + 1}/{math.ceil(len(request_items) / batch_size)} "
                    f"items={len(chunk_items)}",
                    flush=True,
                )
                raw_results = self._post_reward_server(chunk_items)
                if len(raw_results) != len(chunk_indices):
                    raise RuntimeError(
                        f"reward server returned {len(raw_results)} results for {len(chunk_indices)} items"
                    )
                for index, raw in zip(chunk_indices, raw_results):
                    if isinstance(raw, Mapping):
                        rewards[index] = self._image_reward_from_server_result(raw)
                    else:
                        rewards[index] = ImageLevelRewardResult(score=0.0, error=f"bad_result_type:{type(raw)}")
                print(
                    f"[REWARD][rank={self.rank}] chunk done "
                    f"{start // batch_size + 1}/{math.ceil(len(request_items) / batch_size)}",
                    flush=True,
                )
            except (urllib.error.URLError, TimeoutError, socket.timeout, RuntimeError, json.JSONDecodeError) as exc:
                message = repr(exc)
                print(
                    f"[REWARD][rank={self.rank}] chunk error "
                    f"{start // batch_size + 1}/{math.ceil(len(request_items) / batch_size)} "
                    f"{message}",
                    flush=True,
                )
                for index in chunk_indices:
                    rewards[index] = ImageLevelRewardResult(score=0.0, error=message)
        return rewards

    def score_completions(self, completions: Sequence[GeneratedCompletion]) -> torch.Tensor:
        print(
            f"[SCORE][rank={self.rank}] score start completions={len(completions)}",
            flush=True,
        )
        image_rewards = self.score_image_rewards(completions)
        reference_rewards = [
            score_three_image_reference(
                completion.generated_images,
                completion.sample.to_dict(),
                psnr_floor=self.args.reference_psnr_floor,
                psnr_ceiling=self.args.reference_psnr_ceiling,
            )
            for completion in completions
        ]
        reward_values = []
        rows = []
        for completion, image_reward, reference_reward in zip(
            completions,
            image_rewards,
            reference_rewards,
        ):
            result = score_two_level_reward(
                {
                    "generated_text": completion.generated_text,
                    "generated_images": completion.generated_images,
                },
                sample=completion.sample.to_dict(),
                image_reward=image_reward,
                reference_reward=reference_reward,
                protocol=self.args.protocol,
                lambda_text=self.args.lambda_text,
                lambda_image=self.args.lambda_image,
                reference_weight=self.args.reference_reward_weight,
                qwen_weight=self.args.qwen_reward_weight,
            )
            completion.reward = result.score
            completion.reward_detail = result.to_dict()
            completion.image_reward = image_reward
            completion.reference_reward = reference_reward
            reward_values.append(result.score)
            print(
                f"[SCORE_DETAIL][rank={self.rank}] id={completion.sample.id} "
                f"gen={completion.generation_index} gate={int(result.format_gate.ok)} "
                f"text={result.text_reward.score:.4f} "
                f"reference={result.reference_reward.score:.4f} "
                f"qwen={result.image_reward.score:.4f} "
                f"combined_image={result.combined_image_score:.4f} total={result.score:.4f} "
                f"qwen_error={result.image_reward.error or '-'} "
                f"reference_error={result.reference_reward.error or '-'}",
                flush=True,
            )
            rows.append(
                {
                    "micro_step": self.micro_step,
                    "update_step": self.update_step,
                    "rank": self.rank,
                    "id": completion.sample.id,
                    "generation_index": completion.generation_index,
                    "completion_tokens": len(completion.completion_ids),
                    "num_generated_images": len(completion.generated_images),
                    "output_dir": str(completion.output_dir or ""),
                    "fused_image_path": completion.fused_image_path,
                    "reward": result.to_dict(),
                }
            )

        with self.reward_jsonl.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if reward_values:
            reference_values = [reward.score for reward in reference_rewards]
            print(
                f"[SCORE][rank={self.rank}] score done "
                f"mean={sum(reward_values) / len(reward_values):.4f} "
                f"min={min(reward_values):.4f} max={max(reward_values):.4f} "
                f"reference_mean={sum(reference_values) / len(reference_values):.4f} "
                f"jsonl={self.reward_jsonl}",
                flush=True,
            )
        return torch.tensor(reward_values, dtype=torch.float32, device=self.device)

    @staticmethod
    def _is_external_reference_reward_error(reference_reward: ThreeImageReferenceReward) -> bool:
        error = str(reference_reward.error or "").strip()
        if not error:
            return False
        # Missing/extra/badly shaped generations are policy failures and must
        # remain in the group as zero reward.  Missing GT is a data failure and
        # must not create a spurious learning target.
        if "expected exactly 3 generated images" in error:
            return False
        modality_errors = [
            str(metrics.error or "").strip()
            for metrics in (
                reference_reward.infrared,
                reference_reward.visible,
                reference_reward.fused,
            )
            if str(metrics.error or "").strip()
        ]
        if modality_errors:
            return any(
                "reference reward forbids implicit resizing" not in modality_error
                for modality_error in modality_errors
            )
        return "reference reward forbids implicit resizing" not in error

    def _drop_external_reward_error_groups(
        self,
        completions: Sequence[GeneratedCompletion],
        rewards: torch.Tensor,
    ) -> tuple[list[GeneratedCompletion], torch.Tensor, int]:
        group_size = int(self.args.num_generations)
        if len(completions) % group_size != 0:
            raise ValueError(
                f"completion count {len(completions)} is not divisible by num_generations={group_size}"
            )

        kept_completions: list[GeneratedCompletion] = []
        kept_reward_tensors: list[torch.Tensor] = []
        skipped_groups = 0
        for start in range(0, len(completions), group_size):
            end = start + group_size
            group = list(completions[start:end])
            group_rewards = rewards[start:end]
            # Qwen is auxiliary and score_two_level_reward falls back to the
            # local three-GT metric for any judge error.  Only a local reference
            # data failure makes the group unsafe to learn from.
            external_errors = [
                str(completion.reference_reward.error)
                for completion in group
                if self._is_external_reference_reward_error(completion.reference_reward)
            ]
            if external_errors:
                skipped_groups += 1
                sample_id = group[0].sample.id if group else f"group_{start // group_size}"
                print(
                    f"[SCORE][rank={self.rank}] skip reward-error group "
                    f"sample={sample_id} errors={external_errors[:3]}",
                    flush=True,
                )
                continue
            kept_completions.extend(group)
            kept_reward_tensors.append(group_rewards)

        if kept_reward_tensors:
            kept_rewards = torch.cat(kept_reward_tensors, dim=0)
        else:
            kept_rewards = rewards.new_empty((0,))
        return kept_completions, kept_rewards, skipped_groups

    def _drop_zero_variance_groups(
        self,
        completions: Sequence[GeneratedCompletion],
        rewards: torch.Tensor,
    ) -> tuple[list[GeneratedCompletion], torch.Tensor, int]:
        group_size = int(self.args.num_generations)
        kept_completions: list[GeneratedCompletion] = []
        kept_rewards: list[torch.Tensor] = []
        skipped_groups = 0
        for start in range(0, len(completions), group_size):
            end = start + group_size
            group = list(completions[start:end])
            group_rewards = rewards[start:end]
            if len(group) != group_size:
                raise ValueError(f"incomplete GRPO reward group at offset {start}")
            std = float(group_rewards.std(unbiased=False).item())
            mean = float(group_rewards.mean().item())
            print(
                f"[GROUP][rank={self.rank}] sample={group[0].sample.id} "
                f"rewards={group_rewards.detach().cpu().tolist()} "
                f"mean={mean:.6f} std={std:.6f} "
                f"threshold={float(self.args.min_group_reward_std):.6f}",
                flush=True,
            )
            if not math.isfinite(std) or std <= float(self.args.min_group_reward_std):
                skipped_groups += 1
                print(
                    f"[SCORE][rank={self.rank}] skip zero-signal group "
                    f"sample={group[0].sample.id} std={std:.8f} "
                    f"rewards={group_rewards.detach().cpu().tolist()}",
                    flush=True,
                )
                continue
            kept_completions.extend(group)
            kept_rewards.append(group_rewards)
        return (
            kept_completions,
            torch.cat(kept_rewards, dim=0) if kept_rewards else rewards.new_empty((0,)),
            skipped_groups,
        )

    def _all_ranks_have_signal(self, local_has_signal: bool) -> bool:
        value = torch.tensor(1 if local_has_signal else 0, dtype=torch.long, device=self.device)
        if is_dist():
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def _record_no_signal_step(self) -> None:
        self.consecutive_no_signal_steps += 1
        self.micro_step += 1
        limit = int(self.args.max_consecutive_no_signal_steps)
        if limit > 0 and self.consecutive_no_signal_steps >= limit:
            raise RuntimeError(
                "GRPO stopped after consecutive no-signal distributed batches: "
                f"count={self.consecutive_no_signal_steps}, limit={limit}. "
                "Check strict format pass rate, three-image reference reward, and sampling diversity."
            )

    def _pad_prompt_completions(
        self,
        completions: Sequence[GeneratedCompletion],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs: list[list[int]] = []
        attn: list[list[int]] = []
        completion_masks: list[list[int]] = []
        old_logps: list[list[float]] = []
        for completion in completions:
            if len(completion.old_logps) != len(completion.completion_ids):
                raise ValueError(
                    "old_logps must align with completion_ids: "
                    f"id={completion.sample.id} gen={completion.generation_index} "
                    f"old_logps={len(completion.old_logps)} completion_ids={len(completion.completion_ids)}"
                )
            ids = completion.prompt_ids + completion.completion_ids
            mask = [0] * len(completion.prompt_ids) + [1] * len(completion.completion_ids)
            logp = [0.0] * len(completion.prompt_ids) + list(completion.old_logps)
            if len(ids) > self.args.max_position_embeddings:
                ids = ids[: self.args.max_position_embeddings]
                mask = mask[: self.args.max_position_embeddings]
                logp = logp[: self.args.max_position_embeddings]
            if not any(mask):
                ids = completion.prompt_ids + [self.sampler.stop_token_id]
                mask = [0] * len(completion.prompt_ids) + [1]
                logp = [0.0] * len(completion.prompt_ids) + [0.0]
            seqs.append(ids)
            attn.append([1] * len(ids))
            completion_masks.append(mask)
            old_logps.append(logp)

        max_len = max(len(ids) for ids in seqs)
        if is_dist():
            global_max_len = torch.tensor(max_len, dtype=torch.long, device=self.device)
            dist.all_reduce(global_max_len, op=dist.ReduceOp.MAX)
            max_len = int(global_max_len.item())
        pad_id = self.sampler.stop_token_id
        for ids, attention, mask, logp in zip(seqs, attn, completion_masks, old_logps):
            pad = max_len - len(ids)
            ids.extend([pad_id] * pad)
            attention.extend([0] * pad)
            mask.extend([0] * pad)
            logp.extend([0.0] * pad)

        return (
            torch.tensor(seqs, dtype=torch.long, device=self.device),
            torch.tensor(attn, dtype=torch.long, device=self.device),
            torch.tensor(completion_masks, dtype=torch.float32, device=self.device),
            torch.tensor(old_logps, dtype=torch.float32, device=self.device),
        )

    def _cfg_unconditional_hidden_states(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[tuple[int, int], tuple[int, int]]]:
        """Teacher-force the image-only CFG contexts used by rollout sampling.

        ``ImageBlockCFGLogitsProcessor`` drops all context before each generated
        image-start token.  Replaying those image blocks in one padded batch is
        mathematically equivalent to its cached token-by-token forward, but is
        orders of magnitude faster and keeps a valid autograd graph.
        """
        if float(self.args.cfg) == 1.0:
            return None, {}

        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        blocks: list[torch.Tensor] = []
        position_map: dict[tuple[int, int], tuple[int, int]] = {}
        for row in range(input_ids.shape[0]):
            valid_len = int(attention_mask[row].sum().item())
            ids = input_ids[row, :valid_len]
            completion_positions = torch.where(completion_mask[row, :valid_len] > 0)[0]
            completion_start = (
                int(completion_positions[0].item()) if completion_positions.numel() else valid_len
            )
            starts = [
                int(position)
                for position in torch.where(ids == image_start_id)[0].detach().cpu().tolist()
                if int(position) >= completion_start
            ]
            ends = torch.where(ids == image_end_id)[0].detach().cpu().tolist()
            end_cursor = 0
            for start in starts:
                while end_cursor < len(ends) and ends[end_cursor] <= start:
                    end_cursor += 1
                if end_cursor >= len(ends):
                    # A truncated/malformed completion can end inside an image
                    # block.  Rollout CFG was still active for that suffix, so
                    # replay it through the last valid token.  The strict
                    # format gate will assign the policy failure zero reward;
                    # replay must nevertheless remain distribution-identical.
                    end = valid_len
                else:
                    end = int(ends[end_cursor])
                    end_cursor += 1
                # Exclude image_end: the hidden state of the final image code
                # predicts image_end and therefore still uses guided logits.
                block = ids[start:end]
                if block.numel() < 3:
                    continue
                block_index = len(blocks)
                blocks.append(block)
                # CFG begins after image_start plus the two fixed grid tokens.
                for full_position in range(start + 2, end):
                    position_map[(row, full_position)] = (block_index, full_position - start)

        local_has_blocks = 1 if blocks else 0
        global_has_blocks = local_has_blocks
        if is_dist():
            has_blocks = torch.tensor(local_has_blocks, dtype=torch.long, device=self.device)
            dist.all_reduce(has_blocks, op=dist.ReduceOp.MAX)
            global_has_blocks = int(has_blocks.item())
        if not global_has_blocks:
            return None, {}

        if not blocks:
            # Every rank must execute the same FSDP forwards.  This dummy is
            # ignored locally but keeps collectives aligned with other ranks.
            blocks = [input_ids.new_tensor([self.sampler.stop_token_id])]
        max_block_len = max(int(block.numel()) for block in blocks)
        if is_dist():
            global_max_block_len = torch.tensor(max_block_len, dtype=torch.long, device=self.device)
            dist.all_reduce(global_max_block_len, op=dist.ReduceOp.MAX)
            max_block_len = int(global_max_block_len.item())
        block_ids = input_ids.new_full(
            (len(blocks), max_block_len),
            self.sampler.stop_token_id,
        )
        block_attention = attention_mask.new_zeros((len(blocks), max_block_len))
        for index, block in enumerate(blocks):
            length = int(block.numel())
            block_ids[index, :length] = block
            block_attention[index, :length] = 1

        with self._autocast():
            hidden_states = model(
                input_ids=block_ids,
                attention_mask=block_attention,
                use_cache=False,
                return_last_hidden_state=True,
            )
        return hidden_states, position_map

    def _processed_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            return input_ids.new_zeros((input_ids.shape[0], 0), dtype=torch.float32)

        target_ids = input_ids[:, 1:]
        processed_logps = torch.zeros(target_ids.shape, dtype=torch.float32, device=input_ids.device)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        shifted_completion_mask = completion_mask[:, 1:]
        do_sample = bool(self.args.do_sample)
        temperature = float(self.args.temperature)

        row_contexts = []
        for row in range(input_ids.shape[0]):
            completion_positions = torch.where(completion_mask[row] > 0)[0]
            if completion_positions.numel() == 0:
                row_contexts.append(None)
                continue
            prompt_len = int(completion_positions[0].item())
            initial_image_end_count = int((input_ids[row, :prompt_len] == image_end_id).sum().item())
            processors = self.sampler.create_logits_processor(
                image_top_k=self.args.image_top_k,
                text_top_k=self.args.text_top_k,
                stop_after_images=self.args.stop_after_images,
                initial_image_end_count=initial_image_end_count,
                cfg=1.0,
            )
            active_positions = torch.where((shifted_completion_mask[row] > 0) & (attention_mask[row, 1:] > 0))[0]
            row_contexts.append(
                {
                    "processors": processors,
                    "active_positions": active_positions.detach().cpu().tolist(),
                }
            )

        # Replay with dropout disabled, matching rollout's model.eval().  Eval
        # mode does not disable gradients.
        model.eval()
        with self._autocast():
            hidden_states = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_last_hidden_state=True,
            )
        cfg_hidden_states, cfg_position_map = self._cfg_unconditional_hidden_states(
            model,
            input_ids,
            attention_mask,
            completion_mask,
        )

        chunk_size = max(1, int(getattr(self.args, "logprob_chunk_size", 64)))
        effective_len = seq_len - 1
        num_chunks = math.ceil(effective_len / chunk_size)

        # Vocabulary projection is by far the largest replay tensor
        # (vocab~=171k).  Project only chunks containing generated actions, and
        # run the unconditional head only for chunks where image CFG was
        # actually active.  A single vector all-reduce makes those decisions
        # identical on every rank, preserving child-FSDP collective order even
        # when prompt/completion lengths differ.
        local_chunk_flags = torch.zeros((2, num_chunks), dtype=torch.long, device=self.device)
        active_entries_by_chunk: list[list[tuple[int, int]]] = [[] for _ in range(num_chunks)]
        for row, context in enumerate(row_contexts):
            if context is None:
                continue
            for position in context["active_positions"]:
                if 0 <= position < effective_len:
                    chunk_index = position // chunk_size
                    local_chunk_flags[0, chunk_index] = 1
                    active_entries_by_chunk[chunk_index].append((row, position))
        for row, position in cfg_position_map:
            if 0 <= position < effective_len and bool(shifted_completion_mask[row, position].item()):
                local_chunk_flags[1, position // chunk_size] = 1
        if is_dist():
            dist.all_reduce(local_chunk_flags, op=dist.ReduceOp.MAX)
        global_active_chunks = local_chunk_flags[0].detach().cpu().tolist()
        global_guided_chunks = local_chunk_flags[1].detach().cpu().tolist()

        lm_head = unwrap_model(model).lm_head
        for chunk_index, start in enumerate(range(0, effective_len, chunk_size)):
            end = min(effective_len, start + chunk_size)
            if not global_active_chunks[chunk_index]:
                continue
            conditional_chunk = hidden_states[:, start:end, :]
            cfg_chunk = torch.zeros_like(conditional_chunk)
            guided_mask = torch.zeros(
                (input_ids.shape[0], end - start),
                dtype=torch.bool,
                device=input_ids.device,
            )
            if cfg_hidden_states is not None:
                for row in range(input_ids.shape[0]):
                    for position in range(start, end):
                        mapped = cfg_position_map.get((row, position))
                        if mapped is None:
                            continue
                        block_index, block_position = mapped
                        cfg_chunk[row, position - start] = cfg_hidden_states[block_index, block_position]
                        guided_mask[row, position - start] = True

            use_cfg = bool(global_guided_chunks[chunk_index])
            if use_cfg and cfg_hidden_states is None:
                raise RuntimeError("global CFG replay chunk is active but unconditional hidden states are missing")

            def project_chunk_logps(
                conditional: torch.Tensor,
                unconditional: torch.Tensor,
                *,
                chunk_start: int = start,
                chunk_end: int = end,
                chunk_guided_mask: torch.Tensor = guided_mask,
                chunk_uses_cfg: bool = use_cfg,
                chunk_active_entries: tuple[tuple[int, int], ...] = tuple(
                    active_entries_by_chunk[chunk_index]
                ),
            ) -> torch.Tensor:
                with self._autocast():
                    conditional_logits = lm_head(conditional)
                    unconditional_logits = lm_head(unconditional) if chunk_uses_cfg else None
                # Keep a zero-valued differentiable dependency on every
                # lm_head call, even when this rank has no active completion
                # token in a globally padded tail chunk.  Without this anchor,
                # non-reentrant checkpointing may skip lm_head recomputation on
                # one rank while another rank enters its child-FSDP collective.
                result = conditional_logits[..., 0].float() * 0.0
                if unconditional_logits is not None:
                    result = result + unconditional_logits[..., 0].float() * 0.0
                for row, position in chunk_active_entries:
                    local_position = position - chunk_start
                    scores = conditional_logits[row : row + 1, local_position, :].float()
                    if chunk_uses_cfg and bool(chunk_guided_mask[row, local_position].item()):
                        uncond = unconditional_logits[row : row + 1, local_position, :].float()
                        cfg = float(self.args.cfg)
                        scores = cfg * (scores - uncond) + uncond
                    prefix = input_ids[row : row + 1, : position + 1]
                    for processor in row_contexts[row]["processors"]:
                        scores = processor(prefix, scores)
                    next_token = target_ids[row : row + 1, position : position + 1]
                    token_logp = self.sampler._next_token_logp(
                        scores,
                        next_token,
                        temperature=temperature,
                        do_sample=do_sample,
                    )
                    result[row, local_position] = token_logp.squeeze(0)
                return result

            chunk_logps = torch_checkpoint(
                project_chunk_logps,
                conditional_chunk,
                cfg_chunk,
                use_reentrant=False,
            )
            processed_logps[:, start:end] = chunk_logps
        return processed_logps

    def _compute_grpo_loss_with_advantages(
        self,
        completions: Sequence[GeneratedCompletion],
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        print(
            f"[GRPO][rank={self.rank}] loss start completions={len(completions)}",
            flush=True,
        )
        input_ids, attention_mask, completion_mask, old_logps = self._pad_prompt_completions(completions)
        shifted_completion_mask = completion_mask[:, 1:]
        shifted_old_logps = old_logps[:, 1:].detach()
        print(
            f"[GRPO][rank={self.rank}] batch tensors "
            f"shape={tuple(input_ids.shape)} completion_tokens={int(completion_mask.sum().item())}",
            flush=True,
        )
        self.policy_model.eval()
        per_token_logps = self._processed_per_token_logps(
            self.policy_model,
            input_ids,
            attention_mask,
            completion_mask,
        )
        print(f"[GRPO][rank={self.rank}] processed policy logps done", flush=True)

        valid_logp_mask = shifted_completion_mask > 0
        logp_delta = (per_token_logps.detach() - shifted_old_logps).abs()
        if bool(valid_logp_mask.any().item()):
            replay_logp_max_abs = float(logp_delta[valid_logp_mask].max().item())
            replay_logp_mean_abs = float(logp_delta[valid_logp_mask].mean().item())
        else:
            replay_logp_max_abs = 0.0
            replay_logp_mean_abs = 0.0
        global_replay_max = torch.tensor(
            replay_logp_max_abs,
            dtype=torch.float32,
            device=self.device,
        )
        global_replay_valid = torch.tensor(
            1 if math.isfinite(replay_logp_max_abs) else 0,
            dtype=torch.long,
            device=self.device,
        )
        if is_dist():
            dist.all_reduce(global_replay_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(global_replay_valid, op=dist.ReduceOp.MIN)
        replay_logp_max_abs = float(global_replay_max.item())
        tolerance = float(self.args.max_replay_logprob_error)
        if not bool(global_replay_valid.item()) or not math.isfinite(replay_logp_max_abs) or (
            tolerance > 0.0 and replay_logp_max_abs > tolerance
        ):
            raise RuntimeError(
                "Rollout/replay log-prob mismatch: "
                f"max_abs={replay_logp_max_abs:.6f}, mean_abs={replay_logp_mean_abs:.6f}, "
                f"tolerance={tolerance:.6f}, cfg={self.args.cfg}. "
                "Refusing an off-policy or numerically inconsistent update."
            )

        if self.reference_model is not None:
            self.reference_model.eval()
            with torch.no_grad():
                ref_per_token_logps = self._processed_per_token_logps(
                    self.reference_model,
                    input_ids,
                    attention_mask,
                    completion_mask,
                )
            print(f"[GRPO][rank={self.rank}] processed reference logps done", flush=True)
        else:
            ref_per_token_logps = per_token_logps.detach()

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
            ref_per_token_logps - per_token_logps
        ) - 1.0
        ratio = torch.exp((per_token_logps - shifted_old_logps).clamp(min=-20.0, max=20.0))
        clip_range = float(self.args.ppo_clip_range)
        clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        pg_unclipped = ratio * advantages.unsqueeze(1)
        pg_clipped = clipped_ratio * advantages.unsqueeze(1)
        pg_term = torch.minimum(pg_unclipped, pg_clipped)
        kl_beta = float(self.args.kl_beta) if self.reference_model is not None else 0.0
        per_token_loss = -(pg_term - kl_beta * per_token_kl)
        denom = shifted_completion_mask.sum(dim=1).clamp_min(1.0)
        loss = ((per_token_loss * shifted_completion_mask).sum(dim=1) / denom).mean()
        local_loss_valid = bool(loss.requires_grad) and bool(torch.isfinite(loss.detach()).item())
        global_loss_valid = torch.tensor(
            1 if local_loss_valid else 0,
            dtype=torch.long,
            device=self.device,
        )
        if is_dist():
            dist.all_reduce(global_loss_valid, op=dist.ReduceOp.MIN)
        if not bool(global_loss_valid.item()):
            raise RuntimeError(
                "Invalid GRPO loss on at least one rank: "
                f"local_requires_grad={loss.requires_grad}, local_value={loss.detach().item()}"
            )
        mean_kl = ((per_token_kl.detach() * shifted_completion_mask).sum(dim=1) / denom).mean()
        valid_tokens = shifted_completion_mask.sum().clamp_min(1.0)
        clip_fraction = (
            (((ratio - 1.0).abs() > clip_range).float() * shifted_completion_mask).sum() / valid_tokens
        )
        old_new_kl = (((shifted_old_logps - per_token_logps.detach()) * shifted_completion_mask).sum() / valid_tokens)
        metrics = {
            "loss": float(loss.detach().item()),
            "advantage_abs": float(advantages.abs().mean().item()),
            "completion_len": float(completion_mask.sum(dim=1).mean().item()),
            "kl": float(mean_kl.item()),
            "kl_beta": kl_beta,
            "old_new_kl": float(old_new_kl.item()),
            "replay_logp_max_abs": replay_logp_max_abs,
            "replay_logp_mean_abs": replay_logp_mean_abs,
            "clip_fraction": float(clip_fraction.item()),
            "ppo_clip_range": clip_range,
            "_valid_token_count": float(valid_tokens.item()),
        }
        print(
            f"[GRPO][rank={self.rank}] loss done "
            f"loss={metrics['loss']:.6f} advantage_abs={metrics['advantage_abs']:.4f}",
            flush=True,
        )
        return loss, metrics

    def _group_advantages(self, rewards: torch.Tensor) -> tuple[torch.Tensor, float]:
        grouped = rewards.view(-1, self.args.num_generations)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, unbiased=False, keepdim=True)
        advantages = ((grouped - mean) / (std + 1e-4)).view(-1).detach()
        return advantages, float(std.mean().item())

    def compute_grpo_loss(
        self,
        completions: Sequence[GeneratedCompletion],
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        """Compute a batched loss for diagnostics/small tests without backward."""
        advantages, reward_std = self._group_advantages(rewards)
        loss, metrics = self._compute_grpo_loss_with_advantages(completions, advantages)
        metrics = {key: value for key, value in metrics.items() if not key.startswith("_")}
        return loss, {
            **metrics,
            "reward": float(rewards.mean().item()),
            "reward_std": reward_std,
        }

    def backward_grpo_loss(
        self,
        completions: Sequence[GeneratedCompletion],
        rewards: torch.Tensor,
        *,
        accumulation_steps: int,
    ) -> Mapping[str, float]:
        """Replay/backward in small completion microbatches to cap peak memory.

        Advantages are still normalized over the complete prompt group.  Each
        microbatch loss is weighted by its fraction of completions, so the
        accumulated gradient is exactly the gradient of the original batch
        mean (and then divided by the outer gradient-accumulation factor).
        """
        total_completions = len(completions)
        if total_completions == 0:
            raise ValueError("cannot backward an empty GRPO completion batch")
        if rewards.numel() != total_completions:
            raise ValueError(
                f"rewards/completions mismatch: {rewards.numel()} vs {total_completions}"
            )

        count_bounds = torch.tensor(
            [total_completions, total_completions],
            dtype=torch.long,
            device=self.device,
        )
        if is_dist():
            minimum = count_bounds[:1].clone()
            maximum = count_bounds[1:].clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            if int(minimum.item()) != int(maximum.item()):
                raise RuntimeError(
                    "FSDP replay requires the same completion microbatch count on every rank: "
                    f"min={int(minimum.item())} max={int(maximum.item())}"
                )

        advantages, reward_std = self._group_advantages(rewards)
        micro_batch_size = max(1, int(self.args.replay_micro_batch_size))
        weighted_metrics: dict[str, float] = {}
        token_weighted_keys = {
            "old_new_kl",
            "clip_fraction",
            "replay_logp_mean_abs",
        }
        token_metric_sums = {key: 0.0 for key in token_weighted_keys}
        valid_token_total = 0.0
        replay_logp_max_abs = 0.0
        for start in range(0, total_completions, micro_batch_size):
            end = min(total_completions, start + micro_batch_size)
            fraction = (end - start) / total_completions
            print(
                f"[GRPO][rank={self.rank}] replay microbatch "
                f"{start // micro_batch_size + 1}/{math.ceil(total_completions / micro_batch_size)} "
                f"completions={end - start}",
                flush=True,
            )
            loss, metrics = self._compute_grpo_loss_with_advantages(
                completions[start:end],
                advantages[start:end],
            )
            scaled_loss = loss * (fraction / max(1, int(accumulation_steps)))
            print(f"[GRPO][rank={self.rank}] backward start microbatch={start}:{end}", flush=True)
            scaled_loss.backward()
            print(f"[GRPO][rank={self.rank}] backward done microbatch={start}:{end}", flush=True)
            valid_token_count = float(metrics.get("_valid_token_count", 0.0))
            valid_token_total += valid_token_count
            for key, value in metrics.items():
                if key == "replay_logp_max_abs":
                    replay_logp_max_abs = max(replay_logp_max_abs, float(value))
                elif key == "_valid_token_count":
                    continue
                elif key in token_weighted_keys:
                    token_metric_sums[key] += valid_token_count * float(value)
                else:
                    weighted_metrics[key] = weighted_metrics.get(key, 0.0) + fraction * float(value)

        for key, total in token_metric_sums.items():
            weighted_metrics[key] = total / max(1.0, valid_token_total)
        weighted_metrics["replay_logp_max_abs"] = replay_logp_max_abs
        weighted_metrics["reward"] = float(rewards.mean().item())
        weighted_metrics["reward_std"] = reward_std
        return weighted_metrics

    def clip_grad(self) -> float:
        if isinstance(self.policy_model, FSDP):
            value = self.policy_model.clip_grad_norm_(float(self.args.clip_grad))
            return float(value.item() if hasattr(value, "item") else value)
        value = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), float(self.args.clip_grad))
        return float(value.item() if hasattr(value, "item") else value)

    def _move_optimizer_state(self, device: torch.device) -> None:
        """Move Adam moments between CPU storage and the local policy GPU."""
        started = time.time()
        moved_bytes = 0
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if not torch.is_tensor(value) or value.device == device:
                    continue
                # Non-capturable Adam keeps its scalar step on CPU; only the
                # large moment tensors need to follow parameters to CUDA.
                if device.type == "cuda" and key == "step" and value.numel() == 1:
                    continue
                moved_bytes += value.numel() * value.element_size()
                state[key] = value.to(device=device, non_blocking=False)
        print(
            f"[OPT][rank={self.rank}] optimizer state move device={device} "
            f"gib={moved_bytes / 1024**3:.3f} elapsed_sec={time.time() - started:.2f}",
            flush=True,
        )

    def optimizer_step(self, *, accumulation_scale: float = 1.0) -> float:
        print(f"[OPT][rank={self.rank}] optimizer step start update_step={self.update_step}", flush=True)
        if float(accumulation_scale) != 1.0:
            for parameter in self.policy_model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(float(accumulation_scale))
        grad_norm = self.clip_grad()
        if not math.isfinite(grad_norm) or grad_norm <= float(self.args.min_grad_norm):
            raise RuntimeError(
                "Refusing optimizer.step with an invalid/zero gradient: "
                f"grad_norm={grad_norm}, min_grad_norm={self.args.min_grad_norm}"
            )
        if self.args.optimizer_cpu_offload:
            self._move_optimizer_state(self.device)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.args.optimizer_cpu_offload:
            self._move_optimizer_state(torch.device("cpu"))
        self.update_step += 1
        print(
            f"[OPT][rank={self.rank}] optimizer step done "
            f"update_step={self.update_step} grad_norm={grad_norm:.6f} "
            f"accumulation_scale={accumulation_scale:.4f}",
            flush=True,
        )
        return grad_norm

    def save_checkpoint(self, name: str, *, epoch_complete: bool = False) -> None:
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        rank0_print(f"[SAVE] checkpoint {save_dir}", rank=self.rank)

        if isinstance(self.policy_model, FSDP):
            with FSDP.state_dict_type(
                self.policy_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                state = self.policy_model.state_dict()
                if self.rank == 0:
                    state = {key: value.to(dtype_from_precision(self.args.precision)) for key, value in state.items()}
                    unwrap_model(self.policy_model).save_pretrained(
                        save_dir,
                        state_dict=state,
                        max_shard_size="10GB",
                    )
            with FSDP.state_dict_type(self.policy_model, StateDictType.LOCAL_STATE_DICT):
                torch.save(
                    self.optimizer.state_dict(),
                    save_dir / f"optimizer.{self.rank:05d}-of-{self.world_size:05d}.pth",
                )
        else:
            if self.rank == 0:
                unwrap_model(self.policy_model).save_pretrained(save_dir, max_shard_size="10GB")
                torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")

        if self.rank == 0:
            (save_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "micro_step": self.micro_step,
                        "update_step": self.update_step,
                        "grad_accum_count": self.grad_accum_count,
                        "epoch": self.current_epoch,
                        "batch_in_epoch": self.batch_in_epoch,
                        "epoch_complete": bool(epoch_complete),
                        "args": vars(self.args),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        if is_dist():
            dist.barrier()

    def train(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for epoch in range(self.start_epoch, int(self.args.epochs)):
            self.current_epoch = epoch
            self._set_phase("epoch_start", f"epoch={epoch}")
            rank0_print(f"[TRAIN] epoch {epoch} start", rank=self.rank)
            batches = iter_distributed_batches(
                self.samples,
                rank=self.rank,
                world_size=self.world_size,
                per_device_batch_size=self.args.per_device_batch_size,
                seed=self.args.seed,
                epoch=epoch,
            )
            for batch_index, local_batch in enumerate(batches):
                if epoch == self.start_epoch and batch_index < self.resume_batch_in_epoch:
                    continue
                self.batch_in_epoch = batch_index
                print(
                    f"[TRAIN][rank={self.rank}] micro_step={self.micro_step} "
                    f"epoch={epoch} local_batch={len(local_batch)} start",
                    flush=True,
                )
                completions: list[GeneratedCompletion] = []
                for local_index, sample in enumerate(local_batch):
                    for gen_index in range(self.args.num_generations):
                        seed = (
                            int(self.args.seed)
                            + epoch * 1_000_003
                            + self.micro_step * 10_007
                            + self.rank * 503
                            + gen_index
                        )
                        print(
                            f"[GEN][rank={self.rank}] start micro_step={self.micro_step} "
                            f"sample={sample.id} gen={gen_index} seed={seed}",
                            flush=True,
                        )
                        self._set_phase(
                            "rollout",
                            f"epoch={epoch} batch={batch_index} sample={sample.id} gen={gen_index}",
                        )
                        completion = self.sampler.generate(
                            sample,
                            generation_index=gen_index,
                            seed=seed,
                            max_new_tokens=self.args.max_new_tokens,
                            image_top_k=self.args.image_top_k,
                            text_top_k=self.args.text_top_k,
                            cfg=self.args.cfg,
                            temperature=self.args.temperature,
                            do_sample=self.args.do_sample,
                            gpt_prefix=self.args.gpt_prefix,
                            stop_after_images=self.args.stop_after_images,
                            use_cache=self.args.rollout_use_cache,
                        )
                        self.save_completion_outputs(completion, epoch=epoch, local_index=local_index)
                        print(
                            f"[GEN][rank={self.rank}] done sample={sample.id} gen={gen_index} "
                            f"tokens={len(completion.completion_ids)} images={len(completion.generated_images)} "
                            f"fused={completion.fused_image_path or 'missing'}",
                            flush=True,
                        )
                        completions.append(completion)
                        if self.args.empty_cuda_cache_between_phases:
                            empty_cuda_cache(self.device)

                self._set_phase("reward", f"epoch={epoch} batch={batch_index}")
                rewards = self.score_completions(completions)
                if self.args.empty_cuda_cache_between_phases:
                    empty_cuda_cache(self.device)
                self._set_phase("reward_filter", f"epoch={epoch} batch={batch_index}")
                completions, rewards, skipped_reward_groups = self._drop_external_reward_error_groups(
                    completions,
                    rewards,
                )
                completions, rewards, zero_variance_groups = self._drop_zero_variance_groups(
                    completions,
                    rewards,
                )
                if not self._all_ranks_have_signal(bool(completions)):
                    print(
                        f"[SCORE][rank={self.rank}] distributed batch skipped; "
                        f"micro_step={self.micro_step} reward_error_groups={skipped_reward_groups} "
                        f"zero_variance_groups={zero_variance_groups}",
                        flush=True,
                    )
                    self._record_no_signal_step()
                    continue
                self.consecutive_no_signal_steps = 0
                self._set_phase("replay_backward", f"epoch={epoch} batch={batch_index}")
                metrics = self.backward_grpo_loss(
                    completions,
                    rewards,
                    accumulation_steps=int(self.args.gradient_accumulation_steps),
                )
                if self.args.empty_cuda_cache_between_phases:
                    empty_cuda_cache(self.device)
                metrics = {
                    **metrics,
                    "skipped_reward_groups": float(skipped_reward_groups),
                    "zero_variance_groups": float(zero_variance_groups),
                    "max_cuda_memory_gib": float(
                        torch.cuda.max_memory_allocated(self.device) / 1024**3
                    ),
                }

                grad_norm = 0.0
                self.grad_accum_count += 1
                should_step = self.grad_accum_count % int(self.args.gradient_accumulation_steps) == 0
                if should_step:
                    self._set_phase("optimizer_step", f"next_update={self.update_step + 1}")
                    grad_norm = self.optimizer_step()

                self.micro_step += 1
                if self.micro_step % int(self.args.log_steps) == 0:
                    mean_metrics = {
                        key: (
                            all_reduce_max(value, self.device)
                            if key == "max_cuda_memory_gib"
                            else all_reduce_mean(value, self.device)
                        )
                        for key, value in {**metrics, "grad_norm": grad_norm}.items()
                    }
                    if self.rank == 0:
                        elapsed = time.time() - started
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "micro_step": self.micro_step,
                                    "update_step": self.update_step,
                                    "elapsed_sec": round(elapsed, 3),
                                    **mean_metrics,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

                if should_step and self.args.save_steps > 0 and self.update_step % int(self.args.save_steps) == 0:
                    self._set_phase("checkpoint", f"step={self.update_step}")
                    self.save_checkpoint(f"step{self.update_step:06d}")

            accumulation_steps = int(self.args.gradient_accumulation_steps)
            remainder = self.grad_accum_count % accumulation_steps
            if remainder:
                scale = accumulation_steps / remainder
                rank0_print(
                    f"[OPT] flush partial accumulation remainder={remainder}/{accumulation_steps} "
                    f"scale={scale:.4f}",
                    rank=self.rank,
                )
                self._set_phase("optimizer_flush", f"epoch={epoch} remainder={remainder}")
                self.optimizer_step(accumulation_scale=scale)
                self.grad_accum_count += accumulation_steps - remainder
            self._set_phase("epoch_checkpoint", f"epoch={epoch}")
            self.save_checkpoint(f"epoch{epoch}", epoch_complete=True)
        self._set_phase("training_done")
        rank0_print(f"[DONE] RL training finished in {time.time() - started:.1f}s", rank=self.rank)

    def close(self) -> None:
        self._set_phase("close_barrier")
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        self._set_phase("closed")
        self._stop_heartbeat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MSRS mGPT2 two-level GRPO trainer")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--reference_model", default="")
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--protocol", default="full", choices=("auto", "full", "fused_only"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_server_url", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=24576)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep Adam moments on CPU outside optimizer.step to reduce rollout/replay GPU memory.",
    )
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--no_reference_kl", action="store_true")
    parser.add_argument("--lambda_text", type=float, default=0.1)
    parser.add_argument("--lambda_image", type=float, default=0.9)
    parser.add_argument("--reference_reward_weight", type=float, default=0.9)
    parser.add_argument("--qwen_reward_weight", type=float, default=0.1)
    parser.add_argument("--reference_psnr_floor", type=float, default=10.0)
    parser.add_argument("--reference_psnr_ceiling", type=float, default=40.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--grad_precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--max_position_embeddings", type=int, default=28672)
    parser.add_argument("--image_top_k", type=int, default=2000)
    parser.add_argument(
        "--text_top_k",
        type=int,
        default=1,
        help="Keep CoT/tag decoding greedy while sampling image tokens for GRPO diversity.",
    )
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rollout_use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to recompute rollout context every token and trade speed for lower KV-cache memory.",
    )
    parser.add_argument("--ppo_clip_range", type=float, default=0.2)
    parser.add_argument(
        "--gpt_prefix",
        default=None,
        help="Defaults to <cot> for fused_only and <infrared_cot> otherwise.",
    )
    parser.add_argument("--stop_after_images", type=int, default=3)
    parser.add_argument("--clip_grad", type=float, default=4.0)
    parser.add_argument("--min_grad_norm", type=float, default=1e-8)
    parser.add_argument("--logprob_chunk_size", type=int, default=64)
    parser.add_argument(
        "--replay_micro_batch_size",
        type=int,
        default=1,
        help="Replay/backward this many completions at once; 1 is safest for 80GB GPUs.",
    )
    parser.add_argument(
        "--max_replay_logprob_error",
        type=float,
        default=0.10,
        help="Fail before backward when cached rollout and teacher-forced replay distributions disagree.",
    )
    parser.add_argument("--reward_timeout_sec", type=float, default=300.0)
    parser.add_argument("--reward_batch_size", type=int, default=1)
    parser.add_argument("--min_group_reward_std", type=float, default=1e-4)
    parser.add_argument("--max_consecutive_no_signal_steps", type=int, default=3)
    parser.add_argument(
        "--heartbeat_interval_sec",
        type=float,
        default=60.0,
        help="Rank-0 heartbeat interval; set <=0 to disable.",
    )
    parser.add_argument(
        "--distributed_timeout_sec",
        type=float,
        default=3600.0,
        help="NCCL process-group initialization/collective timeout.",
    )
    parser.add_argument("--generation_log_interval", type=int, default=512)
    parser.add_argument(
        "--empty_cuda_cache_between_phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Call gc.collect() and torch.cuda.empty_cache() between rollout/reward/replay phases.",
    )
    parser.add_argument("--fsdp_full_load_all_ranks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fsdp_sharding_strategy",
        choices=("full_shard", "shard_grad_op"),
        default="shard_grad_op",
        help=(
            "Use shard_grad_op for autoregressive RL rollouts so full parameters stay resident "
            "between token forwards. Distributed full_shard is rejected because divergent "
            "rollout/CFG forward counts can deadlock its per-forward collectives."
        ),
    )
    parser.add_argument("--skip_image_reward", action="store_true")
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_training_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if int(args.num_generations) < 2:
        errors.append("num_generations must be >= 2 for within-prompt GRPO")
    if int(args.max_new_tokens) <= 0:
        errors.append("max_new_tokens must be positive")
    if int(args.stop_after_images) != 3:
        errors.append("stop_after_images must be 3 because the policy/reward protocol is hard-wired to three outputs")
    if not bool(args.do_sample):
        errors.append("do_sample must be enabled; deterministic rollouts give each GRPO group zero learning signal")
    if int(args.image_top_k) < 2:
        errors.append("image_top_k must be >= 2 so image rollouts can differ within a GRPO group")
    if int(args.text_top_k) < 1:
        errors.append("text_top_k must be >= 1")
    if float(args.temperature) <= 0:
        errors.append("temperature must be positive")
    if int(args.per_device_batch_size) <= 0:
        errors.append("per_device_batch_size must be positive")
    if int(args.gradient_accumulation_steps) <= 0:
        errors.append("gradient_accumulation_steps must be positive")
    if int(args.epochs) <= 0:
        errors.append("epochs must be positive")
    if float(args.lr) <= 0:
        errors.append("lr must be positive")
    if float(args.kl_beta) < 0:
        errors.append("kl_beta must be non-negative")
    task_weights = (float(args.lambda_text), float(args.lambda_image))
    image_weights = (float(args.reference_reward_weight), float(args.qwen_reward_weight))
    if min(task_weights) < 0 or sum(task_weights) <= 0:
        errors.append("lambda_text/lambda_image must be non-negative with a positive sum")
    if min(image_weights) < 0 or sum(image_weights) <= 0:
        errors.append("reference_reward_weight/qwen_reward_weight must be non-negative with a positive sum")
    if bool(args.skip_image_reward) and float(args.reference_reward_weight) <= 0:
        errors.append("skip_image_reward requires reference_reward_weight > 0")
    if float(args.reward_timeout_sec) <= 0:
        errors.append("reward_timeout_sec must be positive")
    if float(args.distributed_timeout_sec) <= 0:
        errors.append("distributed_timeout_sec must be positive")
    if errors:
        raise ValueError("Invalid GRPO configuration:\n- " + "\n- ".join(errors))
    if float(args.kl_beta) > 0 and bool(args.no_reference_kl):
        print(
            "[WARN] kl_beta is positive but --no_reference_kl is set; KL will be disabled.",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.gpt_prefix is None:
        args.gpt_prefix = default_gpt_prefix(args.protocol)
    validate_training_args(args)
    print(
        f"[ARGS] protocol={args.protocol} gpt_prefix={args.gpt_prefix!r} "
        f"seed={args.seed} num_generations={args.num_generations} "
        f"text_top_k={args.text_top_k} image_top_k={args.image_top_k}",
        flush=True,
    )
    if float(args.cfg) < 1.0:
        raise ValueError(f"CFG must be >= 1.0, got {args.cfg}")
    trainer = None
    try:
        trainer = MSRSGRPOTrainer(args)
        trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
        elif is_dist():
            dist.destroy_process_group()
