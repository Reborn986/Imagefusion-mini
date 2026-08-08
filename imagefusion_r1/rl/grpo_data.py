from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#读取数据，挑选完样本之后要改这里
KNOWN_DEGRADATIONS = ("stripe_noise", "noise", "blur2", "blur4", "haze", "rain")
REQUIRED_PATH_KEYS = (
    "infrared_degraded_path",
    "visible_degraded_path",
    "infrared_clean_path",
    "visible_clean_path",
    "fused_gt_path",
)


@dataclass(frozen=True)
class MSRSRLSample:
    id: str
    infrared_degraded_path: str
    visible_degraded_path: str
    infrared_clean_path: str
    visible_clean_path: str
    fused_gt_path: str
    infrared_label: str
    visible_label: str
    infrared_level: str = ""
    visible_level: str = ""
    base_image: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_manifest(path: str | Path) -> list[Mapping[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        items: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"JSONL row is not an object in {path}: {line[:120]}")
                    items.append(row)
        return items

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"MSRS RL manifest must be a list: {path}")
        if not all(isinstance(item, Mapping) for item in data):
            raise ValueError(f"MSRS RL manifest contains non-object items: {path}")
        return list(data)

    raise ValueError(f"Unsupported manifest suffix for {path}; expected .json or .jsonl")


def degradation_components(label: str) -> tuple[str, ...]:
    components: list[str] = []
    remaining = str(label or "").strip().replace("+", "_")
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


def _label_and_level_from_path(path: str) -> tuple[str, str]:
    parts = list(Path(str(path or "")).parts)
    for level_name in ("level_1", "level_2"):
        if level_name not in parts:
            continue
        index = parts.index(level_name)
        if index + 1 < len(parts):
            return parts[index + 1], level_name
    return "", ""


def _nonempty_str(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    return "" if value is None else str(value)


def normalize_sample(raw: Mapping[str, Any], index: int) -> MSRSRLSample:
    missing = [key for key in REQUIRED_PATH_KEYS if not _nonempty_str(raw, key)]
    if missing:
        raise KeyError(f"sample index={index} missing required path keys: {missing}")

    ir_label, ir_level = _label_and_level_from_path(_nonempty_str(raw, "infrared_degraded_path"))
    vis_label, vis_level = _label_and_level_from_path(_nonempty_str(raw, "visible_degraded_path"))
    ir_label = _nonempty_str(raw, "infrared_label") or ir_label
    vis_label = _nonempty_str(raw, "visible_label") or vis_label
    ir_level = _nonempty_str(raw, "infrared_level") or ir_level
    vis_level = _nonempty_str(raw, "visible_level") or vis_level

    if not degradation_components(ir_label):
        raise ValueError(f"sample index={index} has no parseable infrared_label: {ir_label!r}")
    if not degradation_components(vis_label):
        raise ValueError(f"sample index={index} has no parseable visible_label: {vis_label!r}")

    sample_id = _nonempty_str(raw, "id") or _nonempty_str(raw, "sample_id") or f"sample_{index:06d}"
    return MSRSRLSample(
        id=sample_id,
        infrared_degraded_path=_nonempty_str(raw, "infrared_degraded_path"),
        visible_degraded_path=_nonempty_str(raw, "visible_degraded_path"),
        infrared_clean_path=_nonempty_str(raw, "infrared_clean_path"),
        visible_clean_path=_nonempty_str(raw, "visible_clean_path"),
        fused_gt_path=_nonempty_str(raw, "fused_gt_path"),
        infrared_label=ir_label,
        visible_label=vis_label,
        infrared_level=ir_level,
        visible_level=vis_level,
        base_image=_nonempty_str(raw, "base_image"),
    )


def load_msrs_rl_samples(path: str | Path, *, max_samples: int = 0) -> list[MSRSRLSample]:
    raw_items = read_manifest(path)
    if max_samples > 0:
        raw_items = raw_items[:max_samples]
    samples = [normalize_sample(raw, index) for index, raw in enumerate(raw_items)]
    ids = [sample.id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("MSRS RL manifest contains duplicate ids after max_samples filtering.")
    return samples


def shuffled_epoch_indices(num_samples: int, *, seed: int, epoch: int) -> list[int]:
    indices = list(range(num_samples))
    rng = random.Random(int(seed) + int(epoch) * 100_003)
    rng.shuffle(indices)
    return indices


def iter_distributed_batches(
    samples: Sequence[MSRSRLSample],
    *,
    rank: int,
    world_size: int,
    per_device_batch_size: int,
    seed: int,
    epoch: int,
) -> Iterable[list[MSRSRLSample]]:
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    global_batch = world_size * per_device_batch_size
    order = shuffled_epoch_indices(len(samples), seed=seed, epoch=epoch)
    usable = (len(order) // global_batch) * global_batch
    order = order[:usable]
    for start in range(0, usable, global_batch):
        shard_start = start + rank * per_device_batch_size
        shard = order[shard_start : shard_start + per_device_batch_size]
        if len(shard) == per_device_batch_size:
            yield [samples[index] for index in shard]
