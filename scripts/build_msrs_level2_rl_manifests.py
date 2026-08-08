from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEVEL2_IR_LABEL = "noise+stripe_noise"
KNOWN_DEGRADATIONS = ("stripe_noise", "blur2", "blur4", "haze", "noise", "rain")
CANONICAL_ORDER = {
    "blur2": 0,
    "blur4": 1,
    "haze": 2,
    "noise": 3,
    "rain": 4,
    "stripe_noise": 5,
}

A_COMBOS = ("haze+noise", "haze+rain", "noise+rain")
B_COMBOS = ("blur4+haze", "blur4+noise", "blur4+rain")
C_COMBOS = ("blur2+noise", "blur2+rain", "blur2+haze")
D_COMBOS = ("blur2+blur4",)

SMOKE_COUNTS = {combo: 20 for combo in (*A_COMBOS, *B_COMBOS, *C_COMBOS)}
PREFLIGHT_COUNTS = {combo: 1 for combo in (*A_COMBOS, *B_COMBOS, *C_COMBOS)}
PILOT100_COUNTS = {
    **{combo: 16 for combo in A_COMBOS},
    **{combo: 12 for combo in B_COMBOS},
    "blur2+haze": 6,
    "blur2+noise": 5,
    "blur2+rain": 5,
}
PILOT200_COUNTS = {
    **{combo: 35 for combo in A_COMBOS},
    **{combo: 25 for combo in B_COMBOS},
    "blur2+haze": 7,
    "blur2+noise": 7,
    "blur2+rain": 6,
}
PILOT_COUNTS = {
    **{combo: 100 for combo in A_COMBOS},
    **{combo: 80 for combo in B_COMBOS},
    **{combo: 20 for combo in C_COMBOS},
}
MAIN_COUNTS = {
    **{combo: 250 for combo in A_COMBOS},
    **{combo: 180 for combo in B_COMBOS},
    **{combo: 70 for combo in C_COMBOS},
}

RL_KEYS = (
    "id",
    "infrared_degraded_path",
    "visible_degraded_path",
    "infrared_clean_path",
    "visible_clean_path",
    "fused_gt_path",
    "infrared_label",
    "visible_label",
    "infrared_level",
    "visible_level",
    "base_image",
)
PATH_KEYS = (
    "infrared_degraded_path",
    "visible_degraded_path",
    "infrared_clean_path",
    "visible_clean_path",
    "fused_gt_path",
)


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item index={index} is not an object in {path}")
        rows.append(item)
    return rows


def dump_json(items: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(items), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value)


def degradation_components(label: str) -> tuple[str, ...]:
    components: list[str] = []
    remaining = normalize_text(label).strip().replace("+", "_")
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
    return tuple(components)


def canonical_label(label: str) -> str:
    components = degradation_components(label)
    components = tuple(sorted(components, key=lambda name: (CANONICAL_ORDER.get(name, 999), name)))
    return "+".join(components)


def label_and_level_from_path(path: str) -> tuple[str, str]:
    parts = list(Path(path).parts)
    for level_name in ("level_0", "level_1", "level_2"):
        if level_name not in parts:
            continue
        index = parts.index(level_name)
        if index + 1 < len(parts) and level_name != "level_0":
            return canonical_label(parts[index + 1]), level_name
        return "", level_name
    return "", ""


def base_image_from_item(item: Mapping[str, Any]) -> str:
    value = normalize_text(item.get("base_image"))
    if value:
        return value
    item_id = normalize_text(item.get("id"))
    if item_id:
        return item_id.split("_ir-", 1)[0]
    for key in ("infrared_degraded_path", "visible_degraded_path", "fused_gt_path"):
        path = normalize_text(item.get(key))
        if path:
            return Path(path).stem
    return ""


def minimal_rl_item(item: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: normalize_text(item.get(key)) for key in RL_KEYS if key in item}
    ir_label, ir_level = label_and_level_from_path(normalize_text(item.get("infrared_degraded_path")))
    vis_label, vis_level = label_and_level_from_path(normalize_text(item.get("visible_degraded_path")))
    row["infrared_label"] = canonical_label(row.get("infrared_label", "")) or ir_label
    row["visible_label"] = canonical_label(row.get("visible_label", "")) or vis_label
    row["infrared_level"] = row.get("infrared_level") or ir_level
    row["visible_level"] = row.get("visible_level") or vis_level
    row["base_image"] = row.get("base_image") or base_image_from_item(item)
    return row


def is_level2_item(item: Mapping[str, Any]) -> bool:
    row = minimal_rl_item(item)
    return (
        normalize_text(row.get("infrared_level")) == "level_2"
        and normalize_text(row.get("visible_level")) == "level_2"
        and normalize_text(row.get("infrared_label")) == LEVEL2_IR_LABEL
    )


def group_level2_by_visible_label(items: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        row = minimal_rl_item(item)
        if not is_level2_item(row):
            continue
        visible_label = normalize_text(row.get("visible_label"))
        grouped[visible_label].append(row)
    return dict(grouped)


def sorted_candidates(items: Sequence[Mapping[str, Any]], *, seed: int, combo: str) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in items]
    # The per-combo seed makes smoke/pilot/main nested: first 20 are smoke, first N are pilot/main.
    rng = random.Random(f"{seed}:{combo}")
    rng.shuffle(candidates)
    return candidates


def sample_by_counts(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    counts: Mapping[str, int],
    *,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for combo in sorted(counts):
        requested = int(counts[combo])
        rows = grouped.get(combo, [])
        if len(rows) < requested:
            raise ValueError(
                f"{split_name}: visible_label={combo!r} has {len(rows)} rows, "
                f"but {requested} are required."
            )
        selected.extend(sorted_candidates(rows, seed=seed, combo=combo)[:requested])
    rng = random.Random(f"{seed}:{split_name}:global")
    rng.shuffle(selected)
    return selected


def sample_scene_disjoint_by_counts(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    counts: Mapping[str, int],
    *,
    seed: int,
    split_name: str,
    excluded_base_images: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_bases = set(excluded_base_images)
    for combo in sorted(counts):
        requested = int(counts[combo])
        combo_rows = []
        for row in sorted_candidates(grouped.get(combo, []), seed=seed, combo=combo):
            base_image = normalize_text(row.get("base_image"))
            if not base_image or base_image in used_bases:
                continue
            combo_rows.append(row)
            used_bases.add(base_image)
            if len(combo_rows) == requested:
                break
        if len(combo_rows) != requested:
            raise ValueError(
                f"{split_name}: could only find {len(combo_rows)}/{requested} scene-disjoint rows "
                f"for visible_label={combo!r}"
            )
        selected.extend(combo_rows)
    random.Random(f"{seed}:{split_name}:global").shuffle(selected)
    return selected


def path_errors(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for item in items:
        missing = [
            key
            for key in PATH_KEYS
            if not normalize_text(item.get(key)) or not Path(normalize_text(item.get(key))).is_file()
        ]
        if missing:
            errors.append({"id": normalize_text(item.get("id")), "missing": missing})
    return errors


def combo_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(normalize_text(item.get("visible_label")) for item in items)
    return dict(sorted(counts.items()))


def base_histogram(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(normalize_text(item.get("base_image")) for item in items)
    hist = Counter(counts.values())
    return {str(key): hist[key] for key in sorted(hist)}


def split_summary(items: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    errors = path_errors(items)
    return {
        "path": str(path),
        "rows": len(items),
        "visible_combo_counts": combo_counts(items),
        "unique_ids": len({normalize_text(item.get("id")) for item in items}),
        "unique_base_images": len({normalize_text(item.get("base_image")) for item in items}),
        "base_image_frequency_histogram": base_histogram(items),
        "path_error_count": len(errors),
        "path_error_examples": errors[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build MSRS level2 RL smoke/pilot/main manifests.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "dataset_final/MSRS/training_seed20260620_levelbalanced_v2/"
            "msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.json"
        ),
        help="Final clean 6000-item MSRS JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset_final/MSRS/rl_level2_seed20260709"),
    )
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--check_paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_all_level2", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = read_json_list(args.input)
    grouped = group_level2_by_visible_label(items)
    missing_combos = sorted(set((*A_COMBOS, *B_COMBOS, *C_COMBOS, *D_COMBOS)) - set(grouped))
    if missing_combos:
        raise ValueError(f"Missing expected level2 visible combos: {missing_combos}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)

    splits: dict[str, tuple[list[dict[str, Any]], Path]] = {}
    split_specs = {
        "preflight": PREFLIGHT_COUNTS,
        "pilot100": PILOT100_COUNTS,
        "pilot200": PILOT200_COUNTS,
        "smoke": SMOKE_COUNTS,
        "pilot": PILOT_COUNTS,
        "main": MAIN_COUNTS,
    }
    for split_name, counts in split_specs.items():
        rows = sample_by_counts(grouped, counts, seed=seed, split_name=split_name)
        path = output_dir / f"msrs_level2_rl_{split_name}_{len(rows)}_seed{seed}.json"
        dump_json(rows, path)
        splits[split_name] = (rows, path)

    pilot100_ids = {normalize_text(row.get("id")) for row in splits["pilot100"][0]}
    pilot200_extra100 = [
        row
        for row in splits["pilot200"][0]
        if normalize_text(row.get("id")) not in pilot100_ids
    ]
    extra100_path = (
        output_dir
        / f"msrs_level2_rl_pilot200_extra100_{len(pilot200_extra100)}_seed{seed}.json"
    )
    dump_json(pilot200_extra100, extra100_path)
    splits["pilot200_extra100"] = (pilot200_extra100, extra100_path)

    pilot200_bases = {
        normalize_text(row.get("base_image"))
        for row in splits["pilot200"][0]
    }
    dev = sample_scene_disjoint_by_counts(
        grouped,
        {combo: 5 for combo in (*A_COMBOS, *B_COMBOS, *C_COMBOS)},
        seed=seed,
        split_name="dev",
        excluded_base_images=pilot200_bases,
    )
    dev_path = output_dir / f"msrs_level2_rl_dev_{len(dev)}_seed{seed}.json"
    dump_json(dev, dev_path)
    splits["dev"] = (dev, dev_path)

    diagnostic = sample_by_counts(
        grouped,
        {"blur2+blur4": min(300, len(grouped["blur2+blur4"]))},
        seed=seed,
        split_name="diagnostic_blur2_blur4",
    )
    diagnostic_path = output_dir / f"msrs_level2_rl_diagnostic_blur2_blur4_{len(diagnostic)}_seed{seed}.json"
    dump_json(diagnostic, diagnostic_path)
    splits["diagnostic_blur2_blur4"] = (diagnostic, diagnostic_path)

    if args.write_all_level2:
        all_level2 = []
        for combo in sorted(grouped):
            all_level2.extend(sorted_candidates(grouped[combo], seed=seed, combo=combo))
        random.Random(f"{seed}:all_level2:global").shuffle(all_level2)
        all_path = output_dir / f"msrs_level2_rl_all_{len(all_level2)}_seed{seed}.json"
        dump_json(all_level2, all_path)
        splits["all_level2"] = (all_level2, all_path)

    report = {
        "input": str(args.input),
        "seed": seed,
        "source_rows": len(items),
        "level2_rows": sum(len(rows) for rows in grouped.values()),
        "available_level2_visible_combo_counts": {key: len(grouped[key]) for key in sorted(grouped)},
        "plans": {
            "preflight": dict(sorted(PREFLIGHT_COUNTS.items())),
            "pilot100": dict(sorted(PILOT100_COUNTS.items())),
            "pilot200": dict(sorted(PILOT200_COUNTS.items())),
            "pilot200_extra100": combo_counts(pilot200_extra100),
            "dev": {combo: 5 for combo in sorted((*A_COMBOS, *B_COMBOS, *C_COMBOS))},
            "smoke": dict(sorted(SMOKE_COUNTS.items())),
            "pilot": dict(sorted(PILOT_COUNTS.items())),
            "main": dict(sorted(MAIN_COUNTS.items())),
            "diagnostic_blur2_blur4": {"blur2+blur4": len(diagnostic)},
        },
        "splits": {
            name: split_summary(rows, path)
            for name, (rows, path) in sorted(splits.items())
        },
    }
    split_ids = {
        name: {normalize_text(row.get("id")) for row in rows}
        for name, (rows, _path) in splits.items()
    }
    dev_bases = {normalize_text(row.get("base_image")) for row in dev}
    expected_sizes = {
        "preflight": sum(PREFLIGHT_COUNTS.values()),
        "pilot100": sum(PILOT100_COUNTS.values()),
        "pilot200": sum(PILOT200_COUNTS.values()),
        "pilot200_extra100": 100,
        "dev": 45,
    }
    validations = {
        "expected_sizes": all(len(splits[name][0]) == size for name, size in expected_sizes.items()),
        "unique_ids": all(
            len(split_ids[name]) == len(rows)
            for name, (rows, _path) in splits.items()
        ),
        "preflight_nested_in_pilot100": split_ids["preflight"] <= split_ids["pilot100"],
        "pilot100_nested_in_pilot200": split_ids["pilot100"] <= split_ids["pilot200"],
        "pilot200_extra_disjoint_from_pilot100": not (
            split_ids["pilot200_extra100"] & split_ids["pilot100"]
        ),
        "pilot100_plus_extra_equals_pilot200": (
            split_ids["pilot100"] | split_ids["pilot200_extra100"]
        ) == split_ids["pilot200"],
        "dev_scene_disjoint_from_pilot200": not (dev_bases & pilot200_bases),
    }
    report["validations"] = validations
    if args.check_paths:
        bad_splits = [
            name
            for name, summary in report["splits"].items()
            if int(summary["path_error_count"]) > 0
        ]
        if bad_splits or not all(validations.values()):
            report["passed"] = False
            report["failed_splits"] = bad_splits
            report["failed_validations"] = [
                name for name, ok in validations.items() if not ok
            ]
        else:
            report["passed"] = True
    else:
        report["passed"] = None

    report_path = output_dir / f"msrs_level2_rl_manifest_report_seed{seed}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] output_dir={output_dir}")
    for name, (_rows, path) in sorted(splits.items()):
        print(f"[SPLIT] {name}: {path}")
    print(f"[REPORT] {report_path}")
    if args.check_paths and report["passed"] is not True:
        raise SystemExit(f"Path check failed for splits: {report.get('failed_splits')}")


if __name__ == "__main__":
    main()
