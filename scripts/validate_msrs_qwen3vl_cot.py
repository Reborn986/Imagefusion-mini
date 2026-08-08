from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


STAGE_TAGS: Mapping[str, Tuple[str, ...]] = {
    "visible": ("visible_degradation", "visible_understand", "visible_image"),
    "infrared": ("infrared_degradation", "infrared_understand", "infrared_image"),
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
PATH_KEYS: Tuple[str, ...] = (
    "visible_degraded_path",
    "infrared_degraded_path",
    "visible_clean_path",
    "infrared_clean_path",
    "fused_gt_path",
)


def load_items(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON at {path}, got {type(data).__name__}")
    return data


def extract_tag(text: object, tag: str) -> str:
    value = "" if text is None else str(text)
    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        flags=re.I | re.S,
    )
    matches = [match.strip() for match in pattern.findall(value) if match.strip()]
    return matches[-1] if matches else ""


def has_control_tags(text: object) -> bool:
    return bool(re.search(r"<\s*/?\s*(think|answer)\s*>", str(text or ""), flags=re.I))


def has_markup(text: object) -> bool:
    return bool(re.search(r"<\s*/?\s*[^>]+>", str(text or "")))


def final_answer_valid(text: object, min_field_chars: int) -> List[str]:
    value = "" if text is None else str(text)
    errors: List[str] = []
    if has_control_tags(value):
        errors.append("final_answer_has_think_or_answer_tag")

    pos = 0
    for tag in FINAL_ANSWER_TAGS:
        pattern = re.compile(
            rf"\s*<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
            flags=re.I | re.S,
        )
        match = pattern.match(value, pos)
        if not match:
            errors.append(f"final_answer_missing_or_out_of_order:{tag}")
            continue
        field = match.group(1).strip()
        if len(field) < min_field_chars:
            errors.append(f"final_answer_short_field:{tag}")
        if has_markup(field):
            errors.append(f"final_answer_nested_markup:{tag}")
        pos = match.end()

    if value[pos:].strip():
        errors.append("final_answer_extra_text")
    return errors


def stage_valid(
    item: Mapping[str, Any],
    stage: str,
    min_field_chars: int,
    min_think_chars: int,
) -> List[str]:
    key = f"{stage}_cot"
    text = str(item.get(key, ""))
    errors: List[str] = []
    if not text:
        return [f"missing:{key}"]
    think = extract_tag(text, "think")
    if not think:
        errors.append(f"{key}_missing_think")
    elif len(think) < min_think_chars:
        errors.append(f"{key}_short_think")
    if not extract_tag(text, "answer"):
        errors.append(f"{key}_missing_answer")
    for tag in STAGE_TAGS[stage]:
        field = extract_tag(text, tag)
        if len(field) < min_field_chars:
            errors.append(f"{key}_short_or_missing:{tag}")
        if has_control_tags(field):
            errors.append(f"{key}_control_tag_inside:{tag}")
    return errors


def item_errors(
    item: Mapping[str, Any],
    min_field_chars: int,
    min_think_chars: int,
    check_paths: bool,
) -> List[str]:
    errors: List[str] = []
    if item.get("error"):
        errors.append("has_error")

    for key in PATH_KEYS:
        path = str(item.get(key, ""))
        if not path:
            errors.append(f"missing_path:{key}")
        elif check_paths and not Path(path).exists():
            errors.append(f"path_not_found:{key}")

    cot_valid = item.get("cot_valid")
    if not isinstance(cot_valid, Mapping):
        errors.append("missing_cot_valid")
    else:
        for stage in STAGE_TAGS:
            if cot_valid.get(stage) is not True:
                errors.append(f"cot_valid_false:{stage}")

    for stage in STAGE_TAGS:
        errors.extend(stage_valid(item, stage, min_field_chars, min_think_chars))

    errors.extend(final_answer_valid(item.get("final_answer", ""), min_field_chars))
    return errors


def summarize_errors(error_rows: Iterable[Tuple[int, str, List[str]]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _idx, _sample_id, errors in error_rows:
        for error in errors:
            counts[error] = counts.get(error, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate generated MSRS Qwen3-VL CoT JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min_field_chars", type=int, default=8)
    parser.add_argument(
        "--min_think_chars",
        type=int,
        default=0,
        help="Optional minimum length for each stage <think>. Use e.g. 500 for golden-style long CoT.",
    )
    parser.add_argument("--check_paths", action="store_true")
    parser.add_argument("--max_examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.input)
    error_rows: List[Tuple[int, str, List[str]]] = []
    for idx, item in enumerate(items):
        errors = item_errors(item, args.min_field_chars, args.min_think_chars, args.check_paths)
        if errors:
            error_rows.append((idx, str(item.get("id", "")), errors))

    valid_count = len(items) - len(error_rows)
    print(f"[VALIDATE] input={args.input}")
    print(f"[VALIDATE] total={len(items)} valid={valid_count} invalid={len(error_rows)}")

    if error_rows:
        print("[VALIDATE] error_summary=" + json.dumps(summarize_errors(error_rows), ensure_ascii=False))
        for idx, sample_id, errors in error_rows[: args.max_examples]:
            print(f"[INVALID] idx={idx} id={sample_id} errors={errors}")
        raise SystemExit(1)

    print("[VALIDATE] all generated CoT items passed strict format checks")


if __name__ == "__main__":
    main()
