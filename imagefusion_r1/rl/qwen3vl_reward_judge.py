from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound
from imagefusion_r1.rl.msrs_two_level_reward import ImageLevelRewardResult, image_reward_from_qwen_scores


DEFAULT_QWEN3VL8B_PATH = "models/Qwen3-VL-8B-Instruct"

JUDGE_PROMPT = """
You are a strict visual expert for infrared-visible image fusion.

You are given three images in order:
1. degraded infrared image
2. degraded visible image
3. generated fused image

Expected infrared degradation label: {infrared_label}
Expected visible degradation label: {visible_label}

Judge only the generated fused image using the two degraded inputs and the labels as context.
Do not use or infer from any model-generated chain-of-thought text.

Score each item from 0 to 10:
- artifact_suppression: whether the fused image suppresses the expected visible and infrared artifacts.
- visible_preservation: whether visible structure, texture, edge detail, color/context, and layout are preserved.
- infrared_preservation: whether infrared thermal saliency, targets, contours, and object separability are preserved.
- fusion_naturalness: whether the image is clear, natural, coherent, and free of visual-token collapse.
- semantic_consistency: whether the fused image matches the input scene without hallucinated objects or structural drift.
- overall: your holistic quality score. It is for logging only.

Return only a valid JSON object with these keys:
{
  "artifact_suppression": 0,
  "visible_preservation": 0,
  "infrared_preservation": 0,
  "fusion_naturalness": 0,
  "semantic_consistency": 0,
  "overall": 0,
  "reason": "one short sentence"
}
""".strip()


@dataclass(frozen=True)
class JudgeInput:
    infrared_degraded_path: str
    visible_degraded_path: str
    fused_image_path: str
    infrared_label: str
    visible_label: str
    sample_id: str = ""


def extract_json_object(text: str) -> Mapping[str, Any]:
    value = str(text or "").strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", value, flags=re.S)
    if not match:
        raise ValueError(f"Qwen judge did not return a JSON object: {value[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, Mapping):
        raise ValueError("Qwen judge JSON is not an object.")
    return parsed


def build_judge_message(item: JudgeInput, image_max_pixels: int = 0, image_min_pixels: int = 0) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_options: dict[str, int] = {}
    if image_max_pixels > 0:
        image_options["max_pixels"] = image_max_pixels
    if image_min_pixels > 0:
        image_options["min_pixels"] = image_min_pixels

    for path in (item.infrared_degraded_path, item.visible_degraded_path, item.fused_image_path):
        payload: dict[str, Any] = {"type": "image", "image": path}
        payload.update(image_options)
        content.append(payload)

    content.append(
        {
            "type": "text",
            "text": JUDGE_PROMPT.format(
                infrared_label=item.infrared_label,
                visible_label=item.visible_label,
            ),
        }
    )
    return [{"role": "user", "content": content}]


class Qwen3VLImageRewardJudge:
    """Frozen Qwen3-VL visual expert for MSRS fused-image reward.

    This class lazy-loads vLLM and Qwen-VL utilities so text reward code can be
    imported without the reward model installed.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN3VL8B_PATH,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        image_max_pixels: int = 0,
        image_min_pixels: int = 0,
        enforce_eager: bool = True,
        kv_cache_memory_bytes: int = 0,
        disable_vllm_tqdm: bool = True,
    ) -> None:
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self.enforce_eager = enforce_eager
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.disable_vllm_tqdm = disable_vllm_tqdm

        self.processor = None
        self.llm = None
        self.sampling_params = None
        self.process_vision_info = None

    def load(self) -> None:
        if self.llm is not None:
            return
        started = time.time()
        print(
            f"[QWEN] model load start path={self.model_path} tp={self.tensor_parallel_size} "
            f"max_model_len={self.max_model_len} eager={self.enforce_eager}",
            flush=True,
        )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen3-VL reward model not found: {self.model_path}. "
                "Set --model-path or QWEN_REWARD_MODEL to its local private directory."
            )

        relax_huggingface_hub_upper_bound()
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        self.processor.tokenizer = tokenizer
        llm_kwargs: dict[str, Any] = {}
        if self.kv_cache_memory_bytes > 0:
            llm_kwargs["kv_cache_memory_bytes"] = self.kv_cache_memory_bytes

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 3, "video": 0},
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=self.enforce_eager,
            **llm_kwargs,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=384,
            stop_token_ids=[],
        )
        print(f"[QWEN] model load done elapsed_sec={time.time() - started:.2f}", flush=True)

    def _prepare_input(self, message: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.processor is not None
        assert self.process_vision_info is not None
        prompt = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, _video_inputs, video_kwargs = self.process_vision_info([message], return_video_kwargs=True)

        mm_data: dict[str, Any] = {}
        mm_kwargs: dict[str, Any] = {}
        if image_inputs:
            mm_data["image"] = image_inputs[0]
        if video_kwargs is not None:
            for key, value in video_kwargs.items():
                if value:
                    mm_kwargs[key] = value[0]

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": mm_kwargs,
        }

    def score_batch(self, items: Sequence[JudgeInput]) -> list[ImageLevelRewardResult]:
        started = time.time()
        self.load()
        assert self.llm is not None
        assert self.sampling_params is not None

        print(
            f"[QWEN] score batch start items={len(items)} "
            f"sample_ids={[item.sample_id for item in items]}",
            flush=True,
        )
        messages = [
            build_judge_message(
                item,
                image_max_pixels=self.image_max_pixels,
                image_min_pixels=self.image_min_pixels,
            )
            for item in items
        ]
        llm_inputs = [self._prepare_input(message) for message in messages]
        outputs = self.llm.generate(
            llm_inputs,
            sampling_params=self.sampling_params,
            use_tqdm=not self.disable_vllm_tqdm,
        )
        print(
            f"[QWEN] generate done items={len(items)} elapsed_sec={time.time() - started:.2f}",
            flush=True,
        )

        results: list[ImageLevelRewardResult] = []
        for item, output in zip(items, outputs):
            text = output.outputs[0].text.strip()
            try:
                raw = dict(extract_json_object(text))
                raw.setdefault("sample_id", item.sample_id)
                results.append(image_reward_from_qwen_scores(raw))
            except Exception as exc:
                results.append(ImageLevelRewardResult(score=0.0, error=repr(exc), raw={"raw_output": text}))
        errors = sum(bool(result.error) for result in results)
        print(
            f"[QWEN] score batch done items={len(results)} errors={errors} "
            f"scores={[round(result.score, 4) for result in results]} "
            f"elapsed_sec={time.time() - started:.2f}",
            flush=True,
        )
        return results

    def score_one(self, item: JudgeInput) -> ImageLevelRewardResult:
        return self.score_batch([item])[0]
