from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefusion_r1.rl.msrs_two_level_reward import (  # noqa: E402
    ImageLevelRewardResult,
    image_reward_from_qwen_scores,
    score_two_level_reward,
)
from imagefusion_r1.rl.qwen3vl_reward_judge import (  # noqa: E402
    DEFAULT_QWEN3VL8B_PATH,
    JudgeInput,
    Qwen3VLImageRewardJudge,
)
from imagefusion_r1.rl.reference_image_reward import score_three_image_reference  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_sample_dirs(eval_root: Path) -> Iterable[Path]:
    for status_path in sorted(eval_root.glob("**/status.json")):
        yield status_path.parent


def mean(values: List[float]) -> float | None:
    values = [value for value in values if value == value]
    if not values:
        return None
    return float(sum(values) / len(values))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(sample: Mapping[str, Any], fused_image_path: Path) -> str:
    payload = {
        "fused_sha256": file_sha256(fused_image_path),
        "infrared_degraded_path": str(sample.get("infrared_degraded_path", "")),
        "visible_degraded_path": str(sample.get("visible_degraded_path", "")),
        "infrared_label": str(sample.get("infrared_label", "")),
        "visible_label": str(sample.get("visible_label", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("cache_key", ""))
            raw = row.get("raw")
            if key and isinstance(raw, Mapping):
                cache[key] = raw
    return cache


def append_cache(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_completion(sample_dir: Path, status: Mapping[str, Any]) -> Mapping[str, Any]:
    text_path = sample_dir / "generated_text.txt"
    generated_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    del status
    images = []
    for filename in ("infrared_restored.png", "visible_restored.png", "fused_image.png"):
        path = sample_dir / filename
        if not path.is_file():
            continue
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return {"generated_text": generated_text, "generated_images": images}


def qwen_judge_input(sample: Mapping[str, Any], sample_dir: Path) -> JudgeInput | None:
    fused_image_path = sample_dir / "fused_image.png"
    if not fused_image_path.is_file():
        return None
    return JudgeInput(
        infrared_degraded_path=str(sample.get("infrared_degraded_path", "")),
        visible_degraded_path=str(sample.get("visible_degraded_path", "")),
        fused_image_path=str(fused_image_path),
        infrared_label=str(sample.get("infrared_label", "")),
        visible_label=str(sample.get("visible_label", "")),
        sample_id=str(sample.get("id", sample_dir.name)),
    )


def raw_from_image_reward(result: ImageLevelRewardResult) -> Mapping[str, Any]:
    if isinstance(result.raw, Mapping):
        return result.raw
    return {
        "artifact_suppression": result.artifact_suppression * 10.0,
        "visible_preservation": result.visible_preservation * 10.0,
        "infrared_preservation": result.infrared_preservation * 10.0,
        "fusion_naturalness": result.fusion_naturalness * 10.0,
        "semantic_consistency": result.semantic_consistency * 10.0,
        "overall": None if result.overall is None else result.overall * 10.0,
        "error": result.error,
    }


def summarize(rows: List[Mapping[str, Any]], eval_root: Path) -> Mapping[str, Any]:
    def collect(path: str) -> List[float]:
        values = []
        for row in rows:
            cursor: Any = row
            for key in path.split("."):
                if not isinstance(cursor, Mapping) or key not in cursor:
                    cursor = None
                    break
                cursor = cursor[key]
            if isinstance(cursor, (int, float)):
                values.append(float(cursor))
        return values

    return {
        "eval_root": str(eval_root),
        "num_samples": len(rows),
        "num_gate_ok": sum(1 for row in rows if row.get("format_gate", {}).get("ok") is True),
        "mean_total_reward": mean(collect("reward.score")),
        "mean_gated_reward": mean(collect("reward.gated_score")),
        "mean_text_reward": mean(collect("text_reward.score")),
        "mean_label_score": mean(collect("text_reward.label_score.score")),
        "mean_planning_score": mean(collect("text_reward.planning_score.score")),
        "mean_image_reward": mean(collect("image_reward.score")),
        "mean_reference_reward": mean(collect("reference_reward.score")),
        "mean_combined_image_reward": mean(collect("reward.combined_image_score")),
        "mean_infrared_psnr": mean(collect("reference_reward.infrared.psnr")),
        "mean_infrared_ssim": mean(collect("reference_reward.infrared.ssim")),
        "mean_visible_psnr": mean(collect("reference_reward.visible.psnr")),
        "mean_visible_ssim": mean(collect("reference_reward.visible.ssim")),
        "mean_fused_psnr": mean(collect("reference_reward.fused.psnr")),
        "mean_fused_ssim": mean(collect("reference_reward.fused.ssim")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Offline MSRS two-level reward evaluation for batch inference outputs.")
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--qwen_model_path", default=DEFAULT_QWEN3VL8B_PATH)
    parser.add_argument("--cache_jsonl", type=Path, default=Path("outputs/reward_cache/qwen3vl8b_msrs_reward_cache.jsonl"))
    parser.add_argument("--skip_image_reward", action="store_true")
    parser.add_argument("--qwen_batch_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--protocol", default="auto", choices=("auto", "full", "fused_only"))
    parser.add_argument("--lambda_text", type=float, default=0.1)
    parser.add_argument("--lambda_image", type=float, default=0.9)
    parser.add_argument("--reference_reward_weight", type=float, default=0.9)
    parser.add_argument("--qwen_reward_weight", type=float, default=0.1)
    parser.add_argument("--reference_psnr_floor", type=float, default=10.0)
    parser.add_argument("--reference_psnr_ceiling", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dirs = list(iter_sample_dirs(args.eval_root))
    output_jsonl = args.output_jsonl or (args.eval_root / "two_level_reward_details.jsonl")
    summary_json = args.summary_json or (args.eval_root / "two_level_reward_summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache_jsonl)
    judge = None
    if not args.skip_image_reward:
        judge = Qwen3VLImageRewardJudge(
            model_path=args.qwen_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    rows: List[Mapping[str, Any]] = []
    pending_inputs: List[tuple[int, str, JudgeInput]] = []

    for sample_dir in sample_dirs:
        status = read_json(sample_dir / "status.json")
        sample = read_json(sample_dir / "sample.json") if (sample_dir / "sample.json").exists() else {}
        completion = build_completion(sample_dir, status)
        reference_reward = score_three_image_reference(
            completion["generated_images"],
            sample,
            psnr_floor=args.reference_psnr_floor,
            psnr_ceiling=args.reference_psnr_ceiling,
        )

        image_reward = ImageLevelRewardResult(score=0.0, error="image_reward_skipped")
        qwen_input = qwen_judge_input(sample, sample_dir)
        key = ""
        if not args.skip_image_reward and qwen_input is not None:
            key = cache_key(sample, Path(qwen_input.fused_image_path))
            if key in cache:
                image_reward = image_reward_from_qwen_scores(cache[key])
            else:
                pending_inputs.append((len(rows), key, qwen_input))
                image_reward = ImageLevelRewardResult(score=0.0, error="pending_qwen_reward")
        elif not args.skip_image_reward:
            image_reward = ImageLevelRewardResult(score=0.0, error="missing_fused_image")

        reward = score_two_level_reward(
            completion,
            sample=sample,
            image_reward=image_reward,
            reference_reward=reference_reward,
            protocol=args.protocol,
            lambda_text=args.lambda_text,
            lambda_image=args.lambda_image,
            reference_weight=args.reference_reward_weight,
            qwen_weight=args.qwen_reward_weight,
        )
        rows.append(
            {
                "sample_dir": str(sample_dir),
                "checkpoint": sample_dir.parent.name,
                "id": str(sample.get("id", sample_dir.name)),
                "status_ok": bool(status.get("ok")),
                "format_gate": asdict(reward.format_gate),
                "text_reward": asdict(reward.text_reward),
                "image_reward": asdict(reward.image_reward),
                "reference_reward": reward.reference_reward.to_dict(),
                "reward": {
                    "score": reward.score,
                    "gated_score": reward.gated_score,
                    "text_modulation": reward.text_modulation,
                    "combined_image_score": reward.combined_image_score,
                },
                "qwen_cache_key": key,
            }
        )

    if judge is not None and pending_inputs:
        for start in range(0, len(pending_inputs), max(1, args.qwen_batch_size)):
            chunk = pending_inputs[start : start + max(1, args.qwen_batch_size)]
            results = judge.score_batch([item for _row_idx, _key, item in chunk])
            cache_rows = []
            for (row_idx, key, _item), image_reward in zip(chunk, results):
                cache_rows.append({"cache_key": key, "raw": raw_from_image_reward(image_reward)})
                row = dict(rows[row_idx])
                completion = build_completion(Path(row["sample_dir"]), read_json(Path(row["sample_dir"]) / "status.json"))
                sample = read_json(Path(row["sample_dir"]) / "sample.json")
                reference_reward = score_three_image_reference(
                    completion["generated_images"],
                    sample,
                    psnr_floor=args.reference_psnr_floor,
                    psnr_ceiling=args.reference_psnr_ceiling,
                )
                reward = score_two_level_reward(
                    completion,
                    sample=sample,
                    image_reward=image_reward,
                    reference_reward=reference_reward,
                    protocol=args.protocol,
                    lambda_text=args.lambda_text,
                    lambda_image=args.lambda_image,
                    reference_weight=args.reference_reward_weight,
                    qwen_weight=args.qwen_reward_weight,
                )
                row["image_reward"] = asdict(reward.image_reward)
                row["reference_reward"] = reward.reference_reward.to_dict()
                row["reward"] = {
                    "score": reward.score,
                    "gated_score": reward.gated_score,
                    "text_modulation": reward.text_modulation,
                    "combined_image_score": reward.combined_image_score,
                }
                rows[row_idx] = row
            append_cache(args.cache_jsonl, cache_rows)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, args.eval_root)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
