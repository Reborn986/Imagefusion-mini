from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(os.environ.get("IMAGEFUSION_CACHE_DIR", f"/tmp/imagefusion_cache_{os.getuid()}"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
for cache_dir in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["XDG_CACHE_HOME"])):
    cache_dir.mkdir(parents=True, exist_ok=True)
LUMINA2_ROOT = REPO_ROOT / "third_party" / "lumina_mgpt_2"
LUMINA2_IMPL = LUMINA2_ROOT / "lumina_mgpt"
LUMINA2_GENERATE_EXAMPLES = LUMINA2_IMPL / "generate_examples"

for import_path in (REPO_ROOT, LUMINA2_IMPL, LUMINA2_ROOT, LUMINA2_GENERATE_EXAMPLES):
    import_path = str(import_path)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound

relax_huggingface_hub_upper_bound()

from PIL import Image
import torch
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


from data.item_processor import FlexARItemProcessor
from data.convertsation import Conversation
from imagefusion_r1.preprocess.pre_tokenize_mgpt2_ce import MSRSCEItemProcessor
from model.chameleon import ChameleonForConditionalGeneration


def log(*values):
    print(*values, flush=True)


IMAGE_PLACEHOLDER = "<|image|>"
TAG_OUTPUT_NAMES = {
    "clean_infrared_image": "infrared_restored.png",
    "clean_visible_image": "visible_restored.png",
    "clean_fused_image": "fused_image.png",
}
FALLBACK_OUTPUT_NAMES = [
    "infrared_restored.png",
    "visible_restored.png",
    "fused_image.png",
]


class MSRSCEInferenceItemProcessor(MSRSCEItemProcessor):
    def build_input_ids(self, infrared_path: str, visible_path: str, gpt_prefix: str = "") -> List[int]:
        human_value = self._build_human_value()
        item = {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": None},
            ],
            "image": [infrared_path, visible_path],
        }
        input_ids = FlexARItemProcessor.process_item(
            self,
            item,
            training_mode=False,
            out_flatten=True,
        )
        if gpt_prefix:
            input_ids.extend(self.tokenizer.encode_wo_prefix_space(gpt_prefix))
        return input_ids


class FixedGridMultiModalLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        item_processor: MSRSCEInferenceItemProcessor,
        image_start_token_id: int,
        image_end_token_id: int,
        image_next_line_token_id: int,
        voc_size: int,
        h_grids: int,
        w_grids: int,
        stop_after_images: int = 0,
        initial_image_end_count: int = 0,
        stop_token_id: int | None = None,
    ) -> None:
        self.item_processor = item_processor
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.h_grid_token_id = item_processor.token2id(item_processor.get_n_grids_token(h_grids))
        self.w_grid_token_id = item_processor.token2id(item_processor.get_n_grids_token(w_grids))
        self.h_latent_dim = h_grids * 4
        self.w_latent_dim = w_grids * 4
        self.voc_size = voc_size
        self.stop_after_images = max(0, int(stop_after_images))
        self.initial_image_end_count = max(0, int(initial_image_end_count))
        self.stop_token_id = stop_token_id
        self._image_open_tag_token_ids = tuple(
            tuple(item_processor.tokenizer.encode_wo_prefix_space(f"<{tag}>"))
            for tag in TAG_OUTPUT_NAMES
        )

        self._cached_device = None
        self._image_only_mask = None
        self._outside_image_forbidden_mask = None

    def _ensure_masks(self, device: torch.device) -> None:
        if self._cached_device == device:
            return

        vocab = torch.arange(self.voc_size, device=device)
        image_tokens = torch.arange(155000, 171383 + 1, device=device)
        self._image_only_mask = ~torch.isin(vocab, image_tokens)

        outside_forbidden = torch.cat(
            [
                image_tokens,
                torch.tensor(
                    [self.image_start_token_id, self.image_end_token_id, self.image_next_line_token_id],
                    device=device,
                ),
            ]
        )
        self._outside_image_forbidden_mask = torch.isin(vocab, outside_forbidden)
        self._cached_device = device

    @staticmethod
    def _force_token(scores: torch.FloatTensor, token_id: int) -> torch.FloatTensor:
        constrained = torch.full_like(scores, -math.inf)
        constrained[..., token_id] = 0
        return constrained

    def _ends_with_image_open_tag(self, input_ids: torch.LongTensor) -> bool:
        sequence = input_ids[0]
        for tag_ids in self._image_open_tag_token_ids:
            if not tag_ids or len(sequence) < len(tag_ids):
                continue
            suffix = torch.as_tensor(tag_ids, dtype=sequence.dtype, device=sequence.device)
            if bool(torch.equal(sequence[-len(tag_ids) :], suffix)):
                return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._ensure_masks(scores.device)

        num_image_start = int((input_ids[0] == self.image_start_token_id).sum().item())
        num_image_end = int((input_ids[0] == self.image_end_token_id).sum().item())

        generated_image_end = max(0, num_image_end - self.initial_image_end_count)
        if (
            self.stop_token_id is not None
            and self.stop_after_images > 0
            and generated_image_end >= self.stop_after_images
        ):
            return self._force_token(scores, self.stop_token_id)

        if num_image_start == num_image_end:
            outside_forbidden_mask = self._outside_image_forbidden_mask
            if self._ends_with_image_open_tag(input_ids):
                outside_forbidden_mask = outside_forbidden_mask.clone()
                outside_forbidden_mask[self.image_start_token_id] = False
            return torch.where(
                outside_forbidden_mask,
                torch.full_like(scores, -math.inf),
                scores,
            )

        if num_image_start != num_image_end + 1:
            return scores

        # Derive this from the prefix on every call.  Besides being cheap, this
        # makes the processor replay/checkpoint safe: backward recomputation may
        # visit sequence chunks in reverse order and must not depend on mutable
        # state left by a later image block.
        image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
        new_token_num = len(input_ids[0][image_start_token_id_index + 1 :])

        # Native AR decoding supplies [B, V] scores. Speculative Jacobi supplies
        # [B, draft_len, V]; constrain every draft position according to its
        # own fixed-grid offset.
        score_steps = scores.shape[-2] if scores.ndim >= 3 else 1
        step_scores = []
        for step in range(score_steps):
            current = scores[..., step, :] if scores.ndim >= 3 else scores
            position_after_start = new_token_num + step
            if position_after_start == 0:
                constrained = self._force_token(current, self.h_grid_token_id)
            elif position_after_start == 1:
                constrained = self._force_token(current, self.w_grid_token_id)
            else:
                next_pos = position_after_start - 1
                if next_pos == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    constrained = self._force_token(current, self.image_end_token_id)
                elif next_pos % (self.w_latent_dim + 1) == 0:
                    constrained = self._force_token(current, self.image_next_line_token_id)
                else:
                    constrained = torch.where(
                        self._image_only_mask,
                        torch.full_like(current, -math.inf),
                        current,
                    )
            step_scores.append(constrained)

        if scores.ndim >= 3:
            return torch.stack(step_scores, dim=-2)
        return step_scores[0]


class ImageBlockCFGLogitsProcessor(LogitsProcessor):
    """Apply classifier-free guidance only while decoding image-code blocks."""

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id: int,
        image_end_token_id: int,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
    ) -> None:
        self.guidance_scale = float(guidance_scale)
        self.model = model
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_start_token_id_index = None
        self.unconditional_context_backup = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.unconditional_context = None

    def _get_unconditional_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, self.image_start_token_id_index :]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"],
                    dtype=torch.long,
                )
            model_input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if self.unconditional_context["use_cache"]:
                model_input_ids = input_ids[:, -1:]
            else:
                model_input_ids = torch.cat(
                    [self.unconditional_context["input_ids"], input_ids[:, -1:]],
                    dim=1,
                )
            self.unconditional_context["input_ids"] = model_input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        output = self.model(
            model_input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = output.get("past_key_values", None)
        return output.logits[:, -1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.guidance_scale == 1.0:
            return scores

        num_image_start = int((input_ids[0] == self.image_start_token_id).sum().item())
        num_image_end = int((input_ids[0] == self.image_end_token_id).sum().item())

        if num_image_start == num_image_end:
            self.image_start_token_id_index = None
            self.unconditional_context = None
            return scores

        if num_image_start != num_image_end + 1:
            return scores

        if self.image_start_token_id_index is None:
            self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][
                -1
            ].item()

        new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
        if new_token_num < 2:
            return scores

        if self.unconditional_context is None:
            self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

        unconditional_logits = self._get_unconditional_logits(input_ids)
        return self.guidance_scale * (scores - unconditional_logits) + unconditional_logits


class InterleavedTopKLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id: int,
        image_end_token_id: int,
        filter_value: float = -math.inf,
    ) -> None:
        self.image_top_k = image_top_k
        self.text_top_k = text_top_k
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_image_start = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end = (input_ids[0] == self.image_end_token_id).sum()
        top_k = self.image_top_k if num_image_start == num_image_end + 1 else self.text_top_k
        top_k = min(max(1, top_k), scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        return scores.masked_fill(indices_to_remove, self.filter_value)


class ImageFusionMGPT2CESolver:
    def __init__(
        self,
        model_path: str,
        tokenizer: str,
        precision: str,
        target_height: int,
        target_width: int,
        max_position_embeddings: int,
        output_protocol: str = "auto",
        speculative_jacobi: bool = False,
        speculative_jacobi_window: int = 16,
        speculative_jacobi_seed: int | None = None,
    ) -> None:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.dtype = dtype_map[precision]
        self.target_height = target_height
        self.target_width = target_width

        self.model = ChameleonForConditionalGeneration.from_pretrained(
            model_path,
            max_position_embeddings=max_position_embeddings,
            mask_image_logits=False,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        if hasattr(self.model.model, "vqmodel"):
            del self.model.model.vqmodel
        self.model.eval()

        tokenizer_path = tokenizer if tokenizer else model_path
        self.item_processor = MSRSCEInferenceItemProcessor(
            tokenizer=tokenizer_path,
            target_size=max(target_height, target_width),
            target_height=target_height,
            target_width=target_width,
            device="cuda",
            output_protocol=output_protocol,
        )
        self.stop_token_id = self.item_processor.token2id(Conversation.sep_token)

        self.model.config.max_position_embeddings = max_position_embeddings
        self.speculative_jacobi = bool(speculative_jacobi)
        self.speculative_jacobi_window = int(speculative_jacobi_window)
        if self.speculative_jacobi_window < 2:
            raise ValueError("speculative_jacobi_window must be at least 2")
        if self.speculative_jacobi:
            self._enable_speculative_jacobi(
                window=self.speculative_jacobi_window,
                seed=speculative_jacobi_seed,
            )
        log("[LOAD] model_path =", model_path)
        log("[LOAD] tokenizer =", tokenizer_path)
        log("[LOAD] target_hw =", f"{target_height}x{target_width}")
        log("[LOAD] max_position_embeddings =", self.model.config.max_position_embeddings)
        log("[LOAD] output_protocol =", self.item_processor.output_protocol)
        log("[LOAD] stop_token =", Conversation.sep_token, self.stop_token_id)
        log("[LOAD] speculative_jacobi =", self.speculative_jacobi)
        if self.speculative_jacobi:
            log("[LOAD] speculative_jacobi_window =", self.speculative_jacobi_window)

    def _enable_speculative_jacobi(self, window: int, seed: int | None) -> None:
        """Install Lumina-mGPT2's inference-only SJD sampler on this model."""
        from jacobi_utils_static import renew_backbone, renew_sampler

        self.model.__class__ = renew_sampler(self.model.__class__)
        self.model._init_new_params(
            jacobi_loop_interval_l=0,
            jacobi_loop_interval_r=self.model.config.max_position_embeddings,
            max_num_new_tokens=window,
            guidance_scale=1.0,
            seed=seed,
            multi_token_init_scheme="random",
            do_cfg=False,
            prefix_token_sampler_scheme="speculative_jacobi",
            jacobi_image_block_mode=True,
        )
        self.model.model.__class__ = renew_backbone(self.model.model.__class__)
        self.model.model._init_new_params()

    def _input_device(self) -> torch.device:
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return torch.device("cuda:0")

    def create_logits_processor(
        self,
        image_top_k: int,
        text_top_k: int,
        stop_after_images: int,
        initial_image_end_count: int,
        cfg: float,
    ) -> LogitsProcessorList:
        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        image_next_line_id = self.item_processor.token2id(self.item_processor.new_line_token)
        h_grids = self.target_height // self.item_processor.patch_size
        w_grids = self.target_width // self.item_processor.patch_size

        processors = LogitsProcessorList()
        if float(cfg) != 1.0:
            processors.append(
                ImageBlockCFGLogitsProcessor(
                    guidance_scale=cfg,
                    model=self.model,
                    image_start_token_id=image_start_id,
                    image_end_token_id=image_end_id,
                    use_cache=True,
                )
            )
        processors.extend(
            [
                FixedGridMultiModalLogitsProcessor(
                    item_processor=self.item_processor,
                    image_start_token_id=image_start_id,
                    image_end_token_id=image_end_id,
                    image_next_line_token_id=image_next_line_id,
                    voc_size=self.model.config.vocab_size,
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

    @torch.no_grad()
    def generate(
        self,
        infrared_path: str,
        visible_path: str,
        max_gen_len: int,
        image_top_k: int,
        text_top_k: int,
        cfg: float,
        temperature: float,
        do_sample: bool,
        stream: bool,
        prefill_stage_tag: bool,
        stop_after_images: int,
        gpt_prefix: str | None = None,
    ):
        if self.speculative_jacobi and float(cfg) != 1.0:
            raise ValueError("speculative_jacobi currently requires cfg=1.0")
        if gpt_prefix is None:
            if self.item_processor.output_protocol == "no_cot":
                gpt_prefix = "<clean_infrared_image>" if prefill_stage_tag else ""
            elif self.item_processor.output_protocol == "fusion_cot_three_images":
                gpt_prefix = "<clean_infrared_image>" if prefill_stage_tag else ""
            else:
                gpt_prefix = "<infrared_cot>\n" if prefill_stage_tag else ""
        if self.item_processor.output_protocol == "no_cot":
            lowered_prefix = gpt_prefix.lower()
            forbidden = ("cot", "think", "answer", "understand")
            if any(fragment in lowered_prefix for fragment in forbidden):
                raise ValueError(
                    "output_protocol=no_cot forbids CoT-like gpt_prefix values; "
                    f"got {gpt_prefix!r}"
                )
        input_ids = self.item_processor.build_input_ids(
            infrared_path=infrared_path,
            visible_path=visible_path,
            gpt_prefix=gpt_prefix,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self._input_device()).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        prompt_len = input_ids.shape[1]
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        initial_image_end_count = int((input_ids[0] == image_end_id).sum().item())
        remaining_budget = self.model.config.max_position_embeddings - prompt_len
        max_new_tokens = min(max_gen_len, max(1, remaining_budget - 8))
        log("[PROMPT] token_len =", prompt_len)
        log("[PROMPT] initial_image_end_count =", initial_image_end_count)
        log("[PROMPT] stop_after_generated_images =", stop_after_images)
        log("[PROMPT] requested_max_gen_len =", max_gen_len)
        log("[PROMPT] actual_max_new_tokens =", max_new_tokens)
        log("[DECODE] do_sample =", do_sample)
        log("[DECODE] temperature =", temperature)
        log("[DECODE] cfg =", cfg)
        log("[DECODE] image_top_k =", image_top_k)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=self.model.config.max_position_embeddings,
            do_sample=do_sample,
            temperature=temperature,
            top_k=None,
            eos_token_id=self.stop_token_id,
            pad_token_id=self.stop_token_id,
            use_cache=True,
        )
        streamer = TextStreamer(self.item_processor.tokenizer, skip_prompt=True) if stream else None

        with torch.amp.autocast("cuda", dtype=self.dtype):
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                logits_processor=self.create_logits_processor(
                    image_top_k=image_top_k,
                    text_top_k=text_top_k,
                    stop_after_images=stop_after_images,
                    initial_image_end_count=initial_image_end_count,
                    cfg=cfg,
                ),
                streamer=streamer,
            )[0]

        new_ids = output_ids[prompt_len:].tolist()
        if new_ids and new_ids[-1] == self.stop_token_id:
            new_ids = new_ids[:-1]
        generated_text, generated_images = self.decode_ids(new_ids)
        if gpt_prefix:
            generated_text = gpt_prefix + generated_text
        return generated_text, generated_images

    def _encode_answer_text(self, text: str) -> List[int]:
        return self.item_processor.tokenizer.encode_wo_prefix_space(text)

    @staticmethod
    def _degradation_hint(path: str) -> str:
        parts = list(Path(path).parts)
        for level_name in ("level_1", "level_2"):
            if level_name in parts:
                index = parts.index(level_name)
                if index + 1 < len(parts):
                    return parts[index + 1].replace("_", " ")
        return "degradation"

    def _forced_stage_prefix(self, stage: str, infrared_path: str, visible_path: str) -> tuple[str, str]:
        if stage == "infrared":
            degradation = self._degradation_hint(infrared_path)
            image_tag = "clean_infrared_image"
            text = (
                "<infrared_cot>\n"
                "<think>\n"
                f"The degraded infrared image contains {degradation}. Restore the clean infrared image while preserving thermal structure and object positions.\n"
                "</think>\n"
                "<answer>\n"
                f"<infrared_degradation>{degradation}</infrared_degradation>\n"
                "<infrared_understand>Preserve thermal contrast, object locations, and scene structure; suppress infrared artifacts.</infrared_understand>\n"
                "<infrared_image>Generate the clean infrared restoration.</infrared_image>\n"
                "</answer>\n"
                "</infrared_cot>\n"
                "<clean_infrared_image>"
            )
            return text, image_tag

        if stage == "visible":
            degradation = self._degradation_hint(visible_path)
            image_tag = "clean_visible_image"
            text = (
                "<visible_cot>\n"
                "<think>\n"
                f"The degraded visible image contains {degradation}. Restore the clean visible image while preserving color, edges, and spatial layout.\n"
                "</think>\n"
                "<answer>\n"
                f"<visible_degradation>{degradation}</visible_degradation>\n"
                "<visible_understand>Preserve scene layout, color context, object boundaries, and semantic structure; suppress visible artifacts.</visible_understand>\n"
                "<visible_image>Generate the clean visible restoration.</visible_image>\n"
                "</answer>\n"
                "</visible_cot>\n"
                "<clean_visible_image>"
            )
            return text, image_tag

        if stage == "fused":
            image_tag = "clean_fused_image"
            text = (
                "<fused_cot>\n"
                "<think>\n"
                "Fuse the restored infrared and visible information. Preserve visible spatial/color detail and infrared thermal saliency while suppressing both degradations.\n"
                "</think>\n"
                "<answer>\n"
                "<fused_understand>Preserve visible structure and color together with infrared thermal contrast; emphasize salient objects and suppress artifacts.</fused_understand>\n"
                "<fused_image>Generate the clean fused image.</fused_image>\n"
                "</answer>\n"
                "</fused_cot>\n"
                "<clean_fused_image>"
            )
            return text, image_tag

        raise ValueError(f"unknown stage: {stage}")

    def _generate_forced_image_block(
        self,
        context_ids: List[int],
        max_gen_len: int,
        image_top_k: int,
        text_top_k: int,
        cfg: float,
        temperature: float,
        do_sample: bool,
    ):
        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)

        forced_context = [*context_ids, image_start_id]
        input_ids = torch.tensor(forced_context, dtype=torch.long, device=self._input_device()).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        initial_image_end_count = int((input_ids[0] == image_end_id).sum().item())
        prompt_len = input_ids.shape[1]
        remaining_budget = self.model.config.max_position_embeddings - prompt_len
        max_new_tokens = min(max_gen_len, max(1, remaining_budget - 8))

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=self.model.config.max_position_embeddings,
            do_sample=do_sample,
            temperature=temperature,
            top_k=None,
            eos_token_id=self.stop_token_id,
            pad_token_id=self.stop_token_id,
            use_cache=True,
        )

        with torch.amp.autocast("cuda", dtype=self.dtype):
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                logits_processor=self.create_logits_processor(
                    image_top_k=image_top_k,
                    text_top_k=text_top_k,
                    stop_after_images=1,
                    initial_image_end_count=initial_image_end_count,
                    cfg=cfg,
                ),
            )[0]

        new_ids = output_ids[prompt_len:].tolist()
        if new_ids and new_ids[-1] == self.stop_token_id:
            new_ids = new_ids[:-1]
        image_text, generated_images = self.decode_ids([image_start_id, *new_ids])
        if len(generated_images) != 1:
            raise RuntimeError(f"forced image stage decoded {len(generated_images)} images, expected 1")
        return [*forced_context, *new_ids], generated_images[0]

    @torch.no_grad()
    def generate_forced_image_stages(
        self,
        infrared_path: str,
        visible_path: str,
        max_gen_len: int,
        image_top_k: int,
        text_top_k: int,
        cfg: float,
        temperature: float,
        do_sample: bool,
    ):
        context_ids = self.item_processor.build_input_ids(
            infrared_path=infrared_path,
            visible_path=visible_path,
            gpt_prefix="",
        )
        generated_text_parts = []
        generated_images = []

        for stage in ("infrared", "visible", "fused"):
            stage_prefix, image_tag = self._forced_stage_prefix(
                stage=stage,
                infrared_path=infrared_path,
                visible_path=visible_path,
            )
            log("[FORCED_STAGE]", stage, "context_len_before=", len(context_ids))
            context_ids.extend(self._encode_answer_text(stage_prefix))
            context_ids, image = self._generate_forced_image_block(
                context_ids=context_ids,
                max_gen_len=max_gen_len,
                image_top_k=image_top_k,
                text_top_k=text_top_k,
                cfg=cfg,
                temperature=temperature,
                do_sample=do_sample,
            )
            close_text = f"</{image_tag}>\n"
            context_ids.extend(self._encode_answer_text(close_text))
            generated_text_parts.append(stage_prefix + IMAGE_PLACEHOLDER + close_text)
            generated_images.append(image)
            log("[FORCED_STAGE]", stage, "context_len_after=", len(context_ids))

        generated_text = "".join(generated_text_parts).rstrip()
        return generated_text, generated_images

    def _show_images(self, batch: torch.Tensor) -> Image.Image:
        scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        return Image.fromarray(reshaped.numpy())

    def _grid_from_token_id(self, token_id: int) -> int:
        return int(self.item_processor.id2token(torch.tensor(token_id))[-5:-1]) - 8800

    def _decode_image_cache(self, cache: List[int]) -> Image.Image:
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
        codes = torch.tensor(image_tokens, dtype=torch.long, device=device) - 155000
        decoded = self.item_processor.vqgan.decode_code(codes, h_latent, w_latent)
        return self._show_images(decoded)

    def decode_ids(self, token_ids: List[int]):
        generated_images = []
        text_token_ids = []
        image_start_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_id = self.item_processor.token2id(self.item_processor.image_end_token)
        placeholder_image_id = self.item_processor.token2id("<|image|>")

        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            if token_id != image_start_id:
                text_token_ids.append(token_id)
                i += 1
                continue

            cache = []
            found_end = False
            for j in range(i + 1, len(token_ids)):
                if token_ids[j] == image_end_id:
                    try:
                        generated_images.append(self._decode_image_cache(cache))
                        text_token_ids.append(placeholder_image_id)
                    except Exception as exc:
                        log("[WARN] image decode failed:", repr(exc))
                    i = j + 1
                    found_end = True
                    break
                cache.append(token_ids[j])

            if not found_end:
                log("[WARN] image_start appeared without image_end")
                break

        generated_text = self.item_processor.tokenizer.decode(text_token_ids)
        image_start_count = sum(1 for token_id in token_ids if token_id == image_start_id)
        image_end_count = sum(1 for token_id in token_ids if token_id == image_end_id)
        log("[RESULT] new_token_len =", len(token_ids))
        log("[RESULT] image_start_count =", image_start_count)
        log("[RESULT] image_end_count =", image_end_count)
        log("[RESULT] decoded_images =", len(generated_images))
        return generated_text, generated_images


def save_outputs(save_dir: Path, generated_text: str, generated_images: List[Image.Image]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    text_path = save_dir / "generated_text.txt"
    text_path.write_text(generated_text, encoding="utf-8")
    log("[SAVE]", text_path)

    names = infer_output_names_from_text(generated_text, len(generated_images))
    if len(generated_images) != 3:
        log("[WARN] expected 3 generated images, got", len(generated_images))

    for idx, image in enumerate(generated_images):
        name = names[idx]
        out_path = save_dir / name
        image.save(out_path)
        log("[SAVE]", out_path)

    manifest_path = save_dir / "outputs_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(f"num_generated_images={len(generated_images)}\n")
        f.write("generated_text_file=generated_text.txt\n")
        for idx, name in enumerate(names):
            f.write(f"{idx}: {name}\n")
    log("[SAVE]", manifest_path)


def infer_output_names_from_text(generated_text: str, image_count: int) -> List[str]:
    placeholder_matches = list(re.finditer(re.escape(IMAGE_PLACEHOLDER), generated_text))
    names: List[str | None] = [None] * image_count
    used = set()

    for idx, match in enumerate(placeholder_matches[:image_count]):
        for tag, filename in TAG_OUTPUT_NAMES.items():
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            open_pos = generated_text.rfind(open_tag, 0, match.start())
            close_pos = generated_text.find(close_tag, match.end())
            if open_pos >= 0 and close_pos >= match.end() and filename not in used:
                names[idx] = filename
                used.add(filename)
                break

    fallback_iter = iter(FALLBACK_OUTPUT_NAMES)
    extra_idx = 1
    resolved = []
    for name in names:
        while name is None:
            candidate = next(fallback_iter, None)
            if candidate is None:
                candidate = f"extra_image_{extra_idx}.png"
                extra_idx += 1
            if candidate not in used:
                name = candidate
                used.add(name)
        resolved.append(name)

    return resolved


def parse_args():
    parser = argparse.ArgumentParser("ImageFusion-R1 mGPT2 CE inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer", default="pretrained/Lumina-mGPT-2.0-Omni")
    parser.add_argument("--infrared_path", required=True)
    parser.add_argument("--visible_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--max_position_embeddings", type=int, default=28672)
    parser.add_argument("--max_gen_len", type=int, default=18000)
    parser.add_argument("--image_top_k", type=int, default=2000)
    parser.add_argument("--text_top_k", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--prefill_stage_tag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefill the first assistant tag before generation. Defaults to <infrared_cot> "
            "for CoT checkpoints and <clean_infrared_image> for no_cot checkpoints."
        ),
    )
    parser.add_argument(
        "--output_protocol",
        choices=("auto", "no_cot", "fused_only", "fusion_cot_three_images"),
        default="auto",
        help="Use no_cot for GT-only checkpoints trained without CoT text.",
    )
    parser.add_argument(
        "--gpt_prefix",
        default=None,
        help="Optional raw Assistant prefix. Overrides --prefill_stage_tag when set.",
    )
    parser.add_argument(
        "--stop_after_images",
        type=int,
        default=3,
        help="Force EOS after this many generated image blocks; 0 disables.",
    )
    parser.add_argument(
        "--force_image_stages",
        action="store_true",
        help="Debug mode: use fixed stage templates and force one image block for infrared, visible, and fused outputs.",
    )
    parser.add_argument(
        "--speculative_jacobi",
        action="store_true",
        help="Enable inference-only Speculative Jacobi decoding inside image-token blocks.",
    )
    parser.add_argument("--speculative_jacobi_window", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = ImageFusionMGPT2CESolver(
        model_path=args.model_path,
        tokenizer=args.tokenizer,
        precision=args.precision,
        target_height=args.target_height,
        target_width=args.target_width,
        max_position_embeddings=args.max_position_embeddings,
        output_protocol=args.output_protocol,
        speculative_jacobi=args.speculative_jacobi,
        speculative_jacobi_window=args.speculative_jacobi_window,
        speculative_jacobi_seed=None,
    )
    if args.force_image_stages:
        generated_text, generated_images = solver.generate_forced_image_stages(
            infrared_path=args.infrared_path,
            visible_path=args.visible_path,
            max_gen_len=args.max_gen_len,
            image_top_k=args.image_top_k,
            text_top_k=args.text_top_k,
            cfg=args.cfg,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
    else:
        generated_text, generated_images = solver.generate(
            infrared_path=args.infrared_path,
            visible_path=args.visible_path,
            max_gen_len=args.max_gen_len,
            image_top_k=args.image_top_k,
            text_top_k=args.text_top_k,
            cfg=args.cfg,
            temperature=args.temperature,
            do_sample=args.do_sample,
            stream=args.stream,
            prefill_stage_tag=args.prefill_stage_tag,
            stop_after_images=args.stop_after_images,
            gpt_prefix=args.gpt_prefix,
        )
    save_outputs(Path(args.save_dir), generated_text, generated_images)


if __name__ == "__main__":
    main()
