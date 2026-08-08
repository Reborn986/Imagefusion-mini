from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefusion_r1.preprocess.cot_sanitizer import sanitize_stage_cot


STAGES = ("infrared", "visible", "fused")
PATH_KEYS = (
    "visible_degraded_path",
    "infrared_degraded_path",
    "visible_clean_path",
    "infrared_clean_path",
    "fused_gt_path",
)
KNOWN_DEGRADATIONS = ("stripe_noise", "blur2", "blur4", "haze", "noise", "rain")


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    deduped = sorted({str(path): path for path in paths}.values())
    return deduped


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data)}")
    return data


def dump_items(items: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(items), f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(report), f, ensure_ascii=False, indent=2)
        f.write("\n")


def item_key(item: Dict[str, Any]) -> str:
    return "||".join(
        [
            str(item.get("id", "")),
            str(item.get("infrared_degraded_path", "")),
            str(item.get("visible_degraded_path", "")),
        ]
    )


def base_image_key(item: Mapping[str, Any]) -> str:
    value = str(item.get("base_image", "")).strip()
    if value:
        return value
    return Path(str(item.get("fused_gt_path", ""))).name


def pair_key(item: Mapping[str, Any]) -> str:
    """Identify the actual AR input pair, independent of old/new sample IDs."""
    return "||".join(
        [
            str(item.get("infrared_degraded_path", "")),
            str(item.get("visible_degraded_path", "")),
        ]
    )


def normalize_item_metadata(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Fill metadata omitted by the early demo JSON from its degraded paths."""
    normalized = dict(item)
    ir_path = Path(str(normalized.get("infrared_degraded_path", "")))
    vis_path = Path(str(normalized.get("visible_degraded_path", "")))
    fused_path = Path(str(normalized.get("fused_gt_path", "")))

    normalized.setdefault("infrared_label", ir_path.parent.name)
    normalized.setdefault("infrared_level", ir_path.parent.parent.name)
    normalized.setdefault("visible_label", vis_path.parent.name)
    normalized.setdefault("visible_level", vis_path.parent.parent.name)
    normalized.setdefault("base_image", fused_path.name or ir_path.name)
    return normalized


LEGACY_POLLUTION_RE = re.compile(
    r"(?:\n\s*){2,}(?:i need|let me|now,?\s+(?:writing|for)|check the requirements|the user said)",
    flags=re.I,
)


def sanitize_legacy_understand(value: object) -> str:
    """Trim response-format leakage found in a small subset of the demo records."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    tag_pos = text.find("<")
    if tag_pos >= 0:
        text = text[:tag_pos]
    pollution = LEGACY_POLLUTION_RE.search(text)
    if pollution:
        text = text[: pollution.start()]
    return text.strip().strip(':"\' ')


def normalize_legacy_understand(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    for stage in STAGES:
        key = f"{stage}_understand"
        if key in normalized:
            normalized[key] = sanitize_legacy_understand(normalized[key])
    return normalized


def extract_tag(text: object, tag: str) -> str:
    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        flags=re.I | re.S,
    )
    matches = [match.strip() for match in pattern.findall(str(text or "")) if match.strip()]
    return sanitize_legacy_understand(matches[-1]) if matches else ""


def understand_value(item: Mapping[str, Any], stage: str) -> str:
    legacy = sanitize_legacy_understand(item.get(f"{stage}_understand", ""))
    if legacy:
        return legacy

    tag = f"{stage}_understand"
    for source_key in (f"{stage}_cot", "final_answer", "final_cot"):
        value = extract_tag(item.get(source_key, ""), tag)
        if value:
            return value
    return ""


def to_legacy_understand_schema(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a golden CoT item to the compact schema used by the demo training set."""
    projected = {
        "id": str(item.get("id", "")),
        "visible_degraded_path": str(item["visible_degraded_path"]),
        "infrared_degraded_path": str(item["infrared_degraded_path"]),
        "visible_clean_path": str(item["visible_clean_path"]),
        "infrared_clean_path": str(item["infrared_clean_path"]),
        "fused_gt_path": str(item["fused_gt_path"]),
        "visible_understand": understand_value(item, "visible"),
        "infrared_understand": understand_value(item, "infrared"),
        "fused_understand": understand_value(item, "fused"),
    }
    missing = [f"{stage}_understand" for stage in STAGES if not projected[f"{stage}_understand"]]
    if missing:
        raise ValueError(f"Cannot project item id={projected['id']} to demo schema; missing={missing}")
    return projected


def to_clean_stage_cot_schema(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Emit the exact clean stage-CoT protocol used by the successful demo PKLs."""
    projected = {
        "id": str(item.get("id", "")),
        "visible_degraded_path": str(item["visible_degraded_path"]),
        "infrared_degraded_path": str(item["infrared_degraded_path"]),
        "visible_clean_path": str(item["visible_clean_path"]),
        "infrared_clean_path": str(item["infrared_clean_path"]),
        "fused_gt_path": str(item["fused_gt_path"]),
    }
    for stage in STAGES:
        raw_outputs = item.get("raw_qwen_outputs")
        raw_source = raw_outputs.get(stage, "") if isinstance(raw_outputs, Mapping) else ""
        source = raw_source or item.get(f"{stage}_cot", "")
        if not source:
            raise ValueError(f"Cannot build clean stage CoT for id={projected['id']}: missing {stage}_cot")
        cleaned = sanitize_stage_cot(source, stage)
        if "No clean field could be extracted from the source CoT." in cleaned:
            raise ValueError(f"Cannot build clean stage CoT for id={projected['id']}: incomplete {stage}_cot")
        projected[f"{stage}_cot"] = cleaned
    return projected


def degradation_components(label: str) -> Tuple[str, ...]:
    """Split level-2 labels without breaking the ``stripe_noise`` component."""
    components: List[str] = []
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
            components.extend(part for part in remaining.split("_") if part)
            break
        components.append(match)
        remaining = remaining[len(match) :].lstrip("_")
    return tuple(components) if components else (label,)


def canonical_degradation(label: object) -> str:
    """Treat differently ordered level-2 degradations as the same stratum."""
    return "+".join(sorted(degradation_components(str(label))))


def combo_key(item: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(item.get("infrared_level", "")),
        canonical_degradation(item.get("infrared_label", "")),
        str(item.get("visible_level", "")),
        canonical_degradation(item.get("visible_label", "")),
    )


def cot_format(item: Mapping[str, Any]) -> str:
    if all(item.get(f"{stage}_cot") for stage in STAGES):
        return "stage_cot"
    if item.get("final_cot"):
        return "final_cot"
    if all(item.get(f"{stage}_understand") for stage in STAGES):
        return "legacy_understand"
    return "missing"


def item_quality(item: Mapping[str, Any]) -> int:
    """Prefer strict golden stage CoT when old demo and new data share a pair."""
    fmt = cot_format(item)
    if fmt == "stage_cot":
        cot_valid = item.get("cot_valid")
        return 4 if isinstance(cot_valid, dict) and all(cot_valid.get(s) is True for s in STAGES) else 3
    if fmt == "final_cot":
        return 2
    if fmt == "legacy_understand":
        return 1
    return 0


def is_valid(
    item: Dict[str, Any],
    require_cot_valid: bool,
    allow_legacy_understand: bool,
    allow_missing_cot_valid: bool,
) -> bool:
    for key in PATH_KEYS:
        if not item.get(key):
            return False

    fmt = cot_format(item)
    if fmt == "missing":
        return False
    if fmt == "legacy_understand":
        return allow_legacy_understand

    if require_cot_valid and fmt == "stage_cot":
        cot_valid = item.get("cot_valid")
        if not isinstance(cot_valid, dict):
            return allow_missing_cot_valid
        if not all(cot_valid.get(stage) is True for stage in STAGES):
            return False

    return True


def pairing_errors(item: Mapping[str, Any], check_paths: bool) -> List[str]:
    """Check that the five images form one aligned AR training item."""
    errors: List[str] = []
    paths = {key: Path(str(item.get(key, ""))) for key in PATH_KEYS}

    expected_name = str(item.get("base_image", "")).strip()
    names = {key: path.name for key, path in paths.items() if str(path)}
    if not expected_name and names:
        expected_name = next(iter(names.values()))
    mismatched_names = {key: name for key, name in names.items() if name != expected_name}
    if mismatched_names:
        errors.append(f"image_name_mismatch:{mismatched_names}")

    if check_paths:
        for key, path in paths.items():
            if not path.is_file():
                errors.append(f"path_not_found:{key}")

    ir_level = str(item.get("infrared_level", ""))
    vis_level = str(item.get("visible_level", ""))
    if ir_level != vis_level:
        errors.append(f"cross_level_pair:{ir_level}!={vis_level}")

    path_expectations = {
        "infrared_degraded_path": (ir_level, str(item.get("infrared_label", ""))),
        "visible_degraded_path": (vis_level, str(item.get("visible_label", ""))),
    }
    for key, (level, label) in path_expectations.items():
        parts = paths[key].parts
        if level and level not in parts:
            errors.append(f"path_level_mismatch:{key}")
        if label and label not in parts:
            errors.append(f"path_label_mismatch:{key}")

    return errors


def select_items(
    items: Iterable[Dict[str, Any]],
    samples_per_combo: int,
    min_items_per_combo: int,
    seed: int,
    balance_base_images: bool,
    balance_base_images_by_level: bool,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, str], int]]:
    by_combo: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_combo[combo_key(item)].append(item)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    skipped: Dict[Tuple[str, str, str, str], int] = {}
    base_counts: Counter[str] = Counter()
    level_base_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    combo_order = sorted(by_combo, key=lambda combo: (len(by_combo[combo]), combo))
    for combo in combo_order:
        rows = sorted(by_combo[combo], key=item_key)
        if len(rows) < min_items_per_combo:
            skipped[combo] = len(rows)
            continue
        rng.shuffle(rows)
        if samples_per_combo > 0:
            if balance_base_images_by_level:
                level = combo[0]
                # First remove scene-level confounding inside each degradation level;
                # global exposure is the deterministic secondary tie-break objective.
                rows.sort(
                    key=lambda item: (
                        level_base_counts[level][base_image_key(item)],
                        base_counts[base_image_key(item)],
                    )
                )
            elif balance_base_images:
                # Stable sort preserves the seeded random order among equal-frequency bases.
                rows.sort(key=lambda item: base_counts[base_image_key(item)])
            rows = rows[:samples_per_combo]
        selected.extend(rows)
        base_counts.update(base_image_key(item) for item in rows)
        level_base_counts[combo[0]].update(base_image_key(item) for item in rows)

    rng.shuffle(selected)
    return selected, skipped


def combo_to_dict(combo: Tuple[str, str, str, str], count: int) -> Dict[str, Any]:
    ir_level, ir_label, vis_level, vis_label = combo
    return {
        "infrared_level": ir_level,
        "infrared_label": ir_label,
        "visible_level": vis_level,
        "visible_label": vis_label,
        "count": count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge Qwen3VL MSRS CoT JSON files into training raw_items.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files or glob patterns.")
    parser.add_argument("--output", type=Path, default=Path("data/msrs_raw_items_cot_level1_level2_balanced.json"))
    parser.add_argument(
        "--samples_per_combo",
        type=int,
        default=0,
        help="Random samples per degradation combo. 0 keeps all valid items.",
    )
    parser.add_argument(
        "--min_items_per_combo",
        type=int,
        default=1,
        help="Skip combos with fewer valid items than this threshold.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_invalid_cot", action="store_true")
    parser.add_argument(
        "--allow_legacy_understand",
        action="store_true",
        help="Accept and clean the early demo schema with three *_understand fields.",
    )
    parser.add_argument(
        "--allow_missing_cot_valid",
        action="store_true",
        help="Trust stage-CoT inputs from the successful demo source, which predates cot_valid flags.",
    )
    parser.add_argument(
        "--check_paths",
        action="store_true",
        help="Require all five image paths to exist while checking item alignment.",
    )
    parser.add_argument(
        "--strict_pairing",
        action="store_true",
        help="Fail instead of dropping records whose five image paths are not aligned.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON audit report for reproducibility and bias checks.",
    )
    parser.add_argument(
        "--output_schema",
        choices=("full", "legacy_understand", "clean_stage_cot"),
        default="full",
        help="Use clean_stage_cot to reproduce the supervision protocol of the successful demo PKLs.",
    )
    parser.add_argument(
        "--balance_base_images",
        action="store_true",
        help="Within each combo, prefer base images with fewer selections so scene exposure stays uniform.",
    )
    parser.add_argument(
        "--balance_base_images_by_level",
        action="store_true",
        help=(
            "Prefer scenes with fewer selections inside the current degradation level, "
            "then use global scene exposure as a tie-break. This removes scene-level confounding."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise ValueError("No input files found.")

    all_items: List[Dict[str, Any]] = []
    per_file_counts: Counter[str] = Counter()
    for path in paths:
        rows = load_items(path)
        per_file_counts[str(path)] = len(rows)
        all_items.extend(normalize_item_metadata(row) for row in rows)

    deduped: Dict[str, Dict[str, Any]] = {}
    invalid_cot_count = 0
    pairing_error_counts: Counter[str] = Counter()
    pairing_error_examples: List[Dict[str, Any]] = []
    for item in all_items:
        if args.allow_legacy_understand and cot_format(item) == "legacy_understand":
            item = normalize_legacy_understand(item)
        if not is_valid(
            item,
            require_cot_valid=not args.allow_invalid_cot,
            allow_legacy_understand=args.allow_legacy_understand,
            allow_missing_cot_valid=args.allow_missing_cot_valid,
        ):
            invalid_cot_count += 1
            continue
        errors = pairing_errors(item, check_paths=args.check_paths)
        if errors:
            pairing_error_counts.update(errors)
            if len(pairing_error_examples) < 10:
                pairing_error_examples.append({"id": item.get("id", ""), "errors": errors})
            continue
        key = pair_key(item)
        existing = deduped.get(key)
        if existing is None or item_quality(item) > item_quality(existing):
            deduped[key] = item

    if args.strict_pairing and pairing_error_counts:
        raise ValueError(
            "Found misaligned training items: "
            + json.dumps(dict(pairing_error_counts), ensure_ascii=False, sort_keys=True)
        )

    available_combo_counts = Counter(combo_key(item) for item in deduped.values())
    selected, skipped_combos = select_items(
        deduped.values(),
        samples_per_combo=args.samples_per_combo,
        min_items_per_combo=args.min_items_per_combo,
        seed=args.seed,
        balance_base_images=args.balance_base_images,
        balance_base_images_by_level=args.balance_base_images_by_level,
    )

    combo_counts = Counter(combo_key(item) for item in selected)
    level_counts = Counter(str(item.get("infrared_level", "")) for item in selected)
    base_image_counts = Counter(str(item.get("base_image", "")) for item in selected)
    base_image_counts_by_level: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in selected:
        base_image_counts_by_level[str(item.get("infrared_level", ""))].update(
            [str(item.get("base_image", ""))]
        )
    format_counts = Counter(cot_format(item) for item in selected)
    if args.output_schema == "legacy_understand":
        output_items = [to_legacy_understand_schema(item) for item in selected]
    elif args.output_schema == "clean_stage_cot":
        output_items = [to_clean_stage_cot_schema(item) for item in selected]
    else:
        output_items = selected
    output_ids = [str(item.get("id", "")) for item in output_items]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Selected output contains duplicate IDs, which would overwrite PKL files.")
    dump_items(output_items, args.output)
    print(
        f"[DONE] files={len(paths)} input_rows={len(all_items)} "
        f"invalid_cot={invalid_cot_count} pairing_errors={sum(pairing_error_counts.values())} "
        f"valid_unique={len(deduped)} selected={len(selected)}"
    )
    print(f"[DONE] output={args.output}")
    print(f"[DONE] output_schema={args.output_schema}")
    print("[FILES]")
    for path, count in per_file_counts.items():
        print(f"  {path}: {count}")
    print("[COMBOS]")
    for combo, count in sorted(combo_counts.items()):
        ir_level, ir_label, vis_level, vis_label = combo
        print(f"  ir={ir_level}/{ir_label} vis={vis_level}/{vis_label}: {count}")
    if skipped_combos:
        print("[SKIPPED_COMBOS]")
        for combo, count in sorted(skipped_combos.items()):
            ir_level, ir_label, vis_level, vis_label = combo
            print(
                f"  ir={ir_level}/{ir_label} vis={vis_level}/{vis_label}: "
                f"available={count} required={args.min_items_per_combo}"
            )
    print("[LEVELS]")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")
    print("[SOURCE_COT_FORMATS]")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count}")

    if args.report is not None:
        report = {
            "seed": args.seed,
            "samples_per_combo": args.samples_per_combo,
            "min_items_per_combo": args.min_items_per_combo,
            "inputs": [{"path": path, "rows": count} for path, count in per_file_counts.items()],
            "input_rows": len(all_items),
            "invalid_cot_rows": invalid_cot_count,
            "pairing_error_counts": dict(pairing_error_counts),
            "pairing_error_examples": pairing_error_examples,
            "valid_unique_rows": len(deduped),
            "selected_rows": len(selected),
            "output_schema": args.output_schema,
            "balance_base_images": args.balance_base_images,
            "balance_base_images_by_level": args.balance_base_images_by_level,
            "available_combos": [
                combo_to_dict(combo, count) for combo, count in sorted(available_combo_counts.items())
            ],
            "selected_combos": [
                combo_to_dict(combo, count) for combo, count in sorted(combo_counts.items())
            ],
            "skipped_combos": [
                combo_to_dict(combo, count) for combo, count in sorted(skipped_combos.items())
            ],
            "selected_levels": dict(sorted(level_counts.items())),
            "selected_cot_formats": dict(sorted(format_counts.items())),
            "selected_unique_base_images": len(base_image_counts),
            "selected_base_image_frequency": {
                "min": min(base_image_counts.values(), default=0),
                "max": max(base_image_counts.values(), default=0),
            },
            "selected_base_image_frequency_histogram": dict(
                sorted(Counter(base_image_counts.values()).items())
            ),
            "selected_base_image_frequency_by_level": {
                level: {
                    "unique": len(counts),
                    "min": min(counts.values(), default=0),
                    "max": max(counts.values(), default=0),
                    "histogram": dict(sorted(Counter(counts.values()).items())),
                }
                for level, counts in sorted(base_image_counts_by_level.items())
            },
        }
        dump_report(report, args.report)
        print(f"[DONE] report={args.report}")


if __name__ == "__main__":
    main()
