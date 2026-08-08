from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefusion_r1.preprocess.cot_sanitizer import sanitize_stage_cot  # noqa: E402


STAGE_TAGS: Mapping[str, Tuple[str, ...]] = {
    "infrared": ("infrared_degradation", "infrared_understand", "infrared_image"),
    "visible": ("visible_degradation", "visible_understand", "visible_image"),
    "fused": ("fused_understand", "fused_image"),
}
FINAL_ANSWER_TAGS: Tuple[str, ...] = (
    "visible_degradation",
    "visible_understand",
    "visible_image",
    "infrared_degradation",
    "infrared_understand",
    "infrared_image",
    "fused_understand",
    "fused_image",
)

DEFAULT_MODEL_PATH = os.environ.get("QWEN_COT_MODEL", "models/Qwen3-VL-32B-Thinking")
SANITIZE_THINK_CHARS = 2800
SANITIZE_FIELD_CHARS = 1000
KNOWN_DEGRADATIONS = ("stripe_noise", "blur2", "blur4", "haze", "noise", "rain")

COMMON_RULES = """
Use exactly this structure: <think>...</think><answer>required tags...</answer>.
The <think> section is a detailed dataset annotation rationale, not private reasoning.
Match this style: compare degraded and clean references, list concrete visual clues, explain damaged image properties, mention what remains preserved, and connect the modality to fusion.
For visible and infrared stages, write about 180-320 words in <think>; for fusion, write about 150-280 words.
Bullets or numbered observations are allowed only inside <think>.
Each answer tag should contain one rich but concise sentence.
Do not output JSON, markdown fences, extra tags, or explanations of the instructions.
Use the provided degradation name as the correct category, but cite concrete image evidence instead of merely repeating it.
""".strip()

VISIBLE_TEMPLATE = """
You are preparing concise training annotations for multimodal image restoration and fusion.

Images:
1. degraded visible image
2. clean visible reference

Correct visible degradation name: {visible_label}
Visible degradation components: {visible_components}

Compare the degraded image with the clean reference and write a compact visible-stage CoT.
In <think>, follow this order: compare degraded vs clean visible images, identify concrete visual clues, describe damaged image properties, explain why the evidence supports the given visible degradation, and state preserved information useful for fusion.
If the degradation has two components, explicitly discuss evidence for both components and how they jointly affect the visible image; do not treat the combined label as a single opaque word.

Required tags inside <answer>, in this exact order:
<visible_degradation>...</visible_degradation>
<visible_understand>...</visible_understand>
<visible_image>...</visible_image>

Tag meanings:
- <visible_degradation>: degradation category plus concrete visible evidence.
- <visible_understand>: damaged information, preserved information, and visible contribution to fusion.
- <visible_image>: ideal restored visible image description.

{common_rules}
""".strip()

INFRARED_TEMPLATE = """
You are preparing concise training annotations for multimodal image restoration and fusion.

Images:
1. degraded infrared image
2. clean infrared reference

Correct infrared degradation name: {infrared_label}
Infrared degradation components: {infrared_components}

Compare the degraded image with the clean reference and write a compact infrared-stage CoT.
In <think>, follow this order: compare degraded vs clean infrared images, identify concrete visual clues, describe damaged image properties, explain why the evidence supports the given infrared degradation, and state preserved thermal/structural information useful for fusion.
If the degradation has two components, explicitly discuss evidence for both components and how they jointly affect the infrared image; do not treat the combined label as a single opaque word.

Required tags inside <answer>, in this exact order:
<infrared_degradation>...</infrared_degradation>
<infrared_understand>...</infrared_understand>
<infrared_image>...</infrared_image>

Tag meanings:
- <infrared_degradation>: degradation category plus concrete infrared evidence.
- <infrared_understand>: damaged information, preserved information, and infrared contribution to fusion.
- <infrared_image>: ideal restored infrared image description.

{common_rules}
""".strip()

FUSION_TEMPLATE = """
You are preparing concise training annotations for multimodal image restoration and fusion.

Images:
1. degraded visible image
2. degraded infrared image
3. clean visible reference
4. clean infrared reference
{fused_gt_line}
{fused_gt_instruction}

Previous visible-stage annotation:
<visible_degradation>{visible_degradation}</visible_degradation>
<visible_understand>{visible_understand}</visible_understand>
<visible_image>{visible_image}</visible_image>
Visible degradation components: {visible_components}

Previous infrared-stage annotation:
<infrared_degradation>{infrared_degradation}</infrared_degradation>
<infrared_understand>{infrared_understand}</infrared_understand>
<infrared_image>{infrared_image}</infrared_image>
Infrared degradation components: {infrared_components}

Write a fusion-stage CoT that explains how to combine useful visible and infrared information.
In <think>, follow this style: summarize visible degradation and infrared degradation, then state what to preserve from visible, what to preserve from infrared, what artifacts to suppress, and what the fused result should emphasize.
When either modality has two degradation components, name both artifact types and explain how fusion should suppress each while preserving useful structure.

Required tags inside <answer>, in this exact order:
<fused_understand>...</fused_understand>
<fused_image>...</fused_image>

Tag meanings:
- <fused_understand>: preserve visible/infrared strengths, suppress degraded artifacts, and state the fusion emphasis.
- <fused_image>: ideal clean fused image description.

{common_rules}
""".strip()


def load_items(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_items(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".jsonl":
        with tmp_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(list(items), f, ensure_ascii=False, indent=2)
            f.write("\n")
    tmp_path.replace(path)


def extract_tag(text: object, tag: str) -> str:
    value = "" if text is None else str(text)
    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        flags=re.I | re.S,
    )
    matches = [match.strip() for match in pattern.findall(value) if match.strip()]
    return matches[-1] if matches else ""


def stage_valid(text: object, stage: str) -> bool:
    return all(extract_tag(text, tag) for tag in STAGE_TAGS[stage])


def stage_fields(text: object, stage: str) -> Dict[str, str]:
    sanitized = sanitize_stage_cot(
        text,
        stage,
        max_think_chars=SANITIZE_THINK_CHARS,
        max_field_chars=SANITIZE_FIELD_CHARS,
    )
    return {tag: extract_tag(sanitized, tag) for tag in STAGE_TAGS[stage]}


def stage_cot(text: object, stage: str) -> str:
    return sanitize_stage_cot(
        text,
        stage,
        max_think_chars=SANITIZE_THINK_CHARS,
        max_field_chars=SANITIZE_FIELD_CHARS,
    )


def label_from_path(path_value: object) -> str:
    if not path_value:
        return "unknown"
    path = Path(str(path_value))
    if len(path.parts) >= 2:
        return path.parent.name
    return "unknown"


def degradation_components(label: str) -> List[str]:
    parts: List[str] = []
    remaining = label
    while remaining:
        match = next(
            (
                name
                for name in sorted(KNOWN_DEGRADATIONS, key=len, reverse=True)
                if remaining == name or remaining.startswith(f"{name}_")
            ),
            None,
        )
        if match is None:
            parts.extend(part for part in remaining.split("_") if part)
            break
        parts.append(match)
        remaining = remaining[len(match) :].lstrip("_")
    return parts or [label]


def component_hint(label: str) -> str:
    components = degradation_components(label)
    if len(components) <= 1:
        return components[0]
    return " + ".join(components)


def sample_key(sample: Mapping[str, Any]) -> str:
    parts = [
        str(sample.get("id", "")),
        str(sample.get("infrared_degraded_path", "")),
        str(sample.get("visible_degraded_path", "")),
    ]
    return "||".join(parts)


def load_skip_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"skip ids JSON must be a list: {path}")
        return {str(item).strip() for item in payload if str(item).strip()}
    skip_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.split("#", 1)[0].strip()
            if value:
                skip_ids.add(value)
    return skip_ids


def build_message(
    image_paths: Iterable[str],
    text_prompt: str,
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        image_item: Dict[str, Any] = {"type": "image", "image": image_path}
        if image_max_pixels > 0:
            image_item["max_pixels"] = image_max_pixels
        if image_min_pixels > 0:
            image_item["min_pixels"] = image_min_pixels
        content.append(image_item)
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]


def build_visible_message(
    sample: Mapping[str, Any],
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> List[Dict[str, Any]]:
    label = str(sample.get("visible_label") or label_from_path(sample.get("visible_degraded_path")))
    prompt = VISIBLE_TEMPLATE.format(
        visible_label=label,
        visible_components=component_hint(label),
        common_rules=COMMON_RULES,
    )
    return build_message(
        [
            str(sample["visible_degraded_path"]),
            str(sample["visible_clean_path"]),
        ],
        prompt,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )


def build_infrared_message(
    sample: Mapping[str, Any],
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> List[Dict[str, Any]]:
    label = str(sample.get("infrared_label") or label_from_path(sample.get("infrared_degraded_path")))
    prompt = INFRARED_TEMPLATE.format(
        infrared_label=label,
        infrared_components=component_hint(label),
        common_rules=COMMON_RULES,
    )
    return build_message(
        [
            str(sample["infrared_degraded_path"]),
            str(sample["infrared_clean_path"]),
        ],
        prompt,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )


def build_fusion_message(
    sample: Mapping[str, Any],
    visible_fields: Mapping[str, str],
    infrared_fields: Mapping[str, str],
    include_fused_gt: bool,
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> List[Dict[str, Any]]:
    fused_gt_line = "5. clean fused reference" if include_fused_gt else ""
    fused_gt_instruction = (
        "Use the clean fused reference as the target appearance for the fusion-stage annotation."
        if include_fused_gt
        else "No clean fused reference is provided, so infer the target fused appearance from the clean visible and infrared references."
    )
    prompt = FUSION_TEMPLATE.format(
        fused_gt_line=fused_gt_line,
        fused_gt_instruction=fused_gt_instruction,
        visible_degradation=visible_fields["visible_degradation"],
        visible_understand=visible_fields["visible_understand"],
        visible_image=visible_fields["visible_image"],
        visible_components=component_hint(
            str(sample.get("visible_label") or label_from_path(sample.get("visible_degraded_path")))
        ),
        infrared_degradation=infrared_fields["infrared_degradation"],
        infrared_understand=infrared_fields["infrared_understand"],
        infrared_image=infrared_fields["infrared_image"],
        infrared_components=component_hint(
            str(sample.get("infrared_label") or label_from_path(sample.get("infrared_degraded_path")))
        ),
        common_rules=COMMON_RULES,
    )
    image_paths = [
        str(sample["visible_degraded_path"]),
        str(sample["infrared_degraded_path"]),
        str(sample["visible_clean_path"]),
        str(sample["infrared_clean_path"]),
    ]
    if include_fused_gt and sample.get("fused_gt_path"):
        image_paths.append(str(sample["fused_gt_path"]))
    return build_message(
        image_paths,
        prompt,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )


def prepare_llm_input(processor: Any, process_vision_info: Any, message: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _video_inputs, video_kwargs = process_vision_info([message], return_video_kwargs=True)

    mm_data: Dict[str, Any] = {}
    mm_kwargs: Dict[str, Any] = {}
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


def call_qwen_batch(
    llm: Any,
    processor: Any,
    process_vision_info: Any,
    messages: Sequence[List[Dict[str, Any]]],
    sampling_params: Any,
    use_tqdm: bool,
) -> List[str]:
    llm_inputs = [prepare_llm_input(processor, process_vision_info, message) for message in messages]
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)
    return [output.outputs[0].text.strip() for output in outputs]


def generate_stage_with_retry(
    llm: Any,
    processor: Any,
    process_vision_info: Any,
    sampling_params: Any,
    stage: str,
    samples: Sequence[Mapping[str, Any]],
    messages: Sequence[List[Dict[str, Any]]],
    retries: int,
    use_tqdm: bool,
) -> List[str]:
    del samples
    outputs = call_qwen_batch(llm, processor, process_vision_info, messages, sampling_params, use_tqdm)

    for _attempt in range(retries):
        bad_indices = [idx for idx, output in enumerate(outputs) if not stage_valid(output, stage)]
        if not bad_indices:
            break
        retry_messages = [messages[idx] for idx in bad_indices]
        retry_outputs = call_qwen_batch(
            llm,
            processor,
            process_vision_info,
            retry_messages,
            sampling_params,
            use_tqdm,
        )
        for idx, output in zip(bad_indices, retry_outputs):
            outputs[idx] = output
    return outputs


def flat_final_answer(visible_cot: str, infrared_cot: str, fused_cot: str) -> str:
    visible = stage_fields(visible_cot, "visible")
    infrared = stage_fields(infrared_cot, "infrared")
    fused = stage_fields(fused_cot, "fused")
    order = [
        ("visible_degradation", visible["visible_degradation"]),
        ("visible_understand", visible["visible_understand"]),
        ("visible_image", visible["visible_image"]),
        ("infrared_degradation", infrared["infrared_degradation"]),
        ("infrared_understand", infrared["infrared_understand"]),
        ("infrared_image", infrared["infrared_image"]),
        ("fused_understand", fused["fused_understand"]),
        ("fused_image", fused["fused_image"]),
    ]
    return "\n".join(f"<{tag}>{value}</{tag}>" for tag, value in order)


def result_from_outputs(
    sample: Mapping[str, Any],
    visible_raw: str,
    infrared_raw: str,
    fused_raw: str,
) -> Dict[str, Any]:
    visible_cot = stage_cot(visible_raw, "visible")
    infrared_cot = stage_cot(infrared_raw, "infrared")
    fused_cot = stage_cot(fused_raw, "fused")

    return {
        **dict(sample),
        "visible_label": str(sample.get("visible_label") or label_from_path(sample.get("visible_degraded_path"))),
        "infrared_label": str(sample.get("infrared_label") or label_from_path(sample.get("infrared_degraded_path"))),
        "infrared_cot": infrared_cot,
        "visible_cot": visible_cot,
        "fused_cot": fused_cot,
        "final_answer": flat_final_answer(visible_cot, infrared_cot, fused_cot),
        "cot_valid": {
            "visible": stage_valid(visible_cot, "visible"),
            "infrared": stage_valid(infrared_cot, "infrared"),
            "fused": stage_valid(fused_cot, "fused"),
        },
        "raw_qwen_outputs": {
            "visible": visible_raw,
            "infrared": infrared_raw,
            "fused": fused_raw,
        },
    }


def result_complete(row: Mapping[str, Any]) -> bool:
    if row.get("error"):
        return False
    final_answer = str(row.get("final_answer", ""))
    if re.search(r"<\s*/?\s*(think|answer)\s*>", final_answer, flags=re.I):
        return False
    for tag in FINAL_ANSWER_TAGS:
        value = extract_tag(final_answer, tag)
        if not value or re.search(r"<\s*/?\s*[^>]+>", value):
            return False
    cot_valid = row.get("cot_valid")
    if isinstance(cot_valid, Mapping):
        return all(bool(cot_valid.get(stage)) for stage in STAGE_TAGS)
    return all(row.get(key) for key in ["visible_cot", "infrared_cot", "fused_cot", "final_answer"])


def existing_results(path: Path, resume: bool) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not resume or not path.exists():
        return [], set()
    rows = load_items(path)
    refreshed = []
    for row in rows:
        raw_outputs = row.get("raw_qwen_outputs")
        if isinstance(raw_outputs, Mapping) and all(
            raw_outputs.get(stage) for stage in ("visible", "infrared", "fused")
        ):
            row = result_from_outputs(
                row,
                str(raw_outputs["visible"]),
                str(raw_outputs["infrared"]),
                str(raw_outputs["fused"]),
            )
        refreshed.append(row)
    completed = [row for row in refreshed if result_complete(row)]
    return completed, {sample_key(row) for row in completed}


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate concise MSRS paired CoT with Qwen3-VL.")
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--skip_ids",
        type=Path,
        default=None,
        help=(
            "Optional newline text file or JSON list of sample ids/sample keys to skip. "
            "Defaults to <output_json_stem>_skip_ids.txt when that file exists."
        ),
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--kv_cache_memory_gb", type=float, default=0.0)
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=0,
        help="Optional per-image max_pixels for Qwen-VL preprocessing. 0 keeps processor default.",
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=0,
        help="Optional per-image min_pixels for Qwen-VL preprocessing. 0 keeps processor default.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include_fused_gt", action="store_true")
    parser.add_argument("--no_enforce_eager", action="store_true")
    parser.add_argument("--disable_vllm_tqdm", action="store_true")
    parser.add_argument(
        "--mm_encoder_attn_backend",
        default=os.environ.get("MM_ENCODER_ATTN_BACKEND", ""),
        help="Optional multimodal encoder attention backend hint passed through environment.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print one concise progress line after this many generated items. 0 disables progress lines.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    print("[INFO] importing multiprocessing", flush=True)
    import multiprocessing as mp

    print("[INFO] importing torch/qwen_vl_utils/transformers/vllm", flush=True)
    import torch.multiprocessing as tmp
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams
    print("[INFO] runtime imports complete", flush=True)

    args = parse_args()
    if args.mm_encoder_attn_backend:
        os.environ.setdefault("MM_ENCODER_ATTN_BACKEND", args.mm_encoder_attn_backend)
        os.environ.setdefault("VLLM_MM_ENCODER_ATTN_BACKEND", args.mm_encoder_attn_backend)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    if tmp.get_start_method(allow_none=True) != "spawn":
        tmp.set_start_method("spawn", force=True)

    input_items = load_items(args.input_json)
    if args.max_samples > 0:
        input_items = input_items[: args.max_samples]

    results, done_keys = existing_results(args.output_json, args.resume)
    if args.resume and results:
        # Persist cleaner improvements immediately, even when every requested
        # sample is already complete and vLLM initialization can be skipped.
        save_items(args.output_json, results)
    default_skip_path = args.output_json.with_name(f"{args.output_json.stem}_skip_ids.txt")
    skip_path = args.skip_ids if args.skip_ids is not None else default_skip_path
    skip_ids = load_skip_ids(skip_path) if skip_path.exists() else set()
    pending = [
        item
        for item in input_items
        if sample_key(item) not in done_keys
        and sample_key(item) not in skip_ids
        and str(item.get("id", "")) not in skip_ids
    ]

    print(
        f"[INFO] input={len(input_items)} existing={len(results)} "
        f"skipped={len(skip_ids)} pending={len(pending)}"
    )
    if not pending:
        print(f"[DONE] no pending items, skip vLLM initialization output={args.output_json}", flush=True)
        return

    print(f"[INFO] loading processor/tokenizer model_path={args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    print(
        "[INFO] initializing vLLM "
        f"tp={args.tensor_parallel_size} max_model_len={args.max_model_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        kv_cache_memory_bytes=(
            int(args.kv_cache_memory_gb * (1024**3)) if args.kv_cache_memory_gb > 0 else None
        ),
        limit_mm_per_prompt={"image": 8, "video": 0},
        enforce_eager=not args.no_enforce_eager,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print("[INFO] vLLM initialized", flush=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    generated_this_run = 0
    chunks = list(chunked(pending, args.batch_size))
    progress_iter: Iterable[List[Dict[str, Any]]]
    if args.progress_every > 0:
        progress_iter = chunks
    else:
        progress_iter = tqdm(chunks, desc="Generating MSRS CoT")

    for batch in progress_iter:
        try:
            visible_messages = [
                build_visible_message(
                    sample,
                    image_max_pixels=args.image_max_pixels,
                    image_min_pixels=args.image_min_pixels,
                )
                for sample in batch
            ]
            visible_raw = generate_stage_with_retry(
                llm,
                processor,
                process_vision_info,
                sampling_params,
                "visible",
                batch,
                visible_messages,
                args.retries,
                not args.disable_vllm_tqdm,
            )
            visible_fields = [stage_fields(text, "visible") for text in visible_raw]

            infrared_messages = [
                build_infrared_message(
                    sample,
                    image_max_pixels=args.image_max_pixels,
                    image_min_pixels=args.image_min_pixels,
                )
                for sample in batch
            ]
            infrared_raw = generate_stage_with_retry(
                llm,
                processor,
                process_vision_info,
                sampling_params,
                "infrared",
                batch,
                infrared_messages,
                args.retries,
                not args.disable_vllm_tqdm,
            )
            infrared_fields = [stage_fields(text, "infrared") for text in infrared_raw]

            fusion_messages = [
                build_fusion_message(
                    sample,
                    vis,
                    ir,
                    include_fused_gt=args.include_fused_gt,
                    image_max_pixels=args.image_max_pixels,
                    image_min_pixels=args.image_min_pixels,
                )
                for sample, vis, ir in zip(batch, visible_fields, infrared_fields)
            ]
            fused_raw = generate_stage_with_retry(
                llm,
                processor,
                process_vision_info,
                sampling_params,
                "fused",
                batch,
                fusion_messages,
                args.retries,
                not args.disable_vllm_tqdm,
            )

            for sample, vis_text, ir_text, fused_text in zip(batch, visible_raw, infrared_raw, fused_raw):
                results.append(result_from_outputs(sample, vis_text, ir_text, fused_text))
                generated_this_run += 1
        except Exception as exc:
            for sample in batch:
                failed = dict(sample)
                failed["error"] = repr(exc)
                results.append(failed)
                generated_this_run += 1
            print(f"[ERROR] batch failed: {exc!r}", flush=True)

        save_items(args.output_json, results)
        if args.progress_every > 0 and (
            generated_this_run % args.progress_every == 0 or generated_this_run == len(pending)
        ):
            print(
                "[PROGRESS] "
                f"generated={generated_this_run} "
                f"done={len(results)}/{len(input_items)} "
                f"pending={max(len(input_items) - len(results), 0)} "
                f"output={args.output_json}",
                flush=True,
            )

    print(f"[DONE] saved={args.output_json} items={len(results)}")


if __name__ == "__main__":
    main()
