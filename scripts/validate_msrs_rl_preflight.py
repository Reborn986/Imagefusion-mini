from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def reward_rows(output_dir: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for path in sorted((output_dir / "logs").glob("reward_details_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping):
                    rows.append(row)
    return rows


def trainer_metrics(output_dir: Path) -> list[Mapping[str, Any]]:
    path = output_dir / "logs" / "msrs_two_level_rl_train.log"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if not value.startswith("{"):
                continue
            try:
                row = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and "micro_step" in row:
                rows.append(row)
    return rows


def latest_trainer_state(output_dir: Path) -> Mapping[str, Any] | None:
    candidates = sorted(output_dir.glob("step*/trainer_state.json"))
    candidates.extend(sorted(output_dir.glob("epoch*/trainer_state.json")))
    if not candidates:
        return None
    return max((read_json(path) for path in candidates), key=lambda row: int(row.get("update_step", 0)))


def nested(row: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = row
    for key in path.split("."):
        if not isinstance(cursor, Mapping):
            return default
        cursor = cursor.get(key, default)
    return cursor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate an MSRS RL preflight before launching 100-200 prompts.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_three_image_rate", type=float, default=0.95)
    parser.add_argument("--min_strict_gate_rate", type=float, default=0.95)
    parser.add_argument("--min_nonzero_group_rate", type=float, default=0.50)
    parser.add_argument("--min_reference_mean", type=float, default=0.40)
    parser.add_argument("--min_qwen_success_rate", type=float, default=0.90)
    parser.add_argument("--max_replay_logprob_error", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = reward_rows(args.output_dir)
    metrics = trainer_metrics(args.output_dir)
    state = latest_trainer_state(args.output_dir)
    if not rows:
        raise SystemExit(f"[FAIL] no reward rows found under {args.output_dir / 'logs'}")

    three_image_rate = sum(int(row.get("num_generated_images", -1)) == 3 for row in rows) / len(rows)
    strict_gate_rate = sum(nested(row, "reward.format_gate.ok") is True for row in rows) / len(rows)
    reference_scores = [
        float(nested(row, "reward.reference_reward.score"))
        for row in rows
        if isinstance(nested(row, "reward.reference_reward.score"), (int, float))
    ]
    reference_mean = sum(reference_scores) / len(reference_scores) if reference_scores else 0.0
    qwen_candidate_rows = [
        row for row in rows if nested(row, "reward.format_gate.ok") is True
    ]
    qwen_successes = sum(
        not str(nested(row, "reward.image_reward.error", "") or "").strip()
        for row in qwen_candidate_rows
    )
    qwen_success_rate = qwen_successes / max(1, len(qwen_candidate_rows))

    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("rank"), row.get("micro_step"), row.get("id"))].append(
            float(nested(row, "reward.score", 0.0))
        )
    group_stds = []
    for rewards in grouped.values():
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        group_stds.append(math.sqrt(variance))
    nonzero_group_rate = sum(std > 1e-4 for std in group_stds) / max(1, len(group_stds))

    replay_errors = [
        float(row["replay_logp_max_abs"])
        for row in metrics
        if isinstance(row.get("replay_logp_max_abs"), (int, float))
    ]
    grad_norms = [
        float(row["grad_norm"])
        for row in metrics
        if isinstance(row.get("grad_norm"), (int, float)) and float(row["grad_norm"]) > 0.0
    ]
    update_step = int(state.get("update_step", 0)) if state else 0
    elapsed_sec = max(
        (float(row.get("elapsed_sec", 0.0)) for row in metrics),
        default=0.0,
    )
    completed_micro_steps = max(
        (int(row.get("micro_step", 0)) for row in metrics),
        default=0,
    )
    sec_per_micro_step = (
        elapsed_sec / completed_micro_steps if completed_micro_steps > 0 else None
    )
    projected_pilot100_hours = (
        sec_per_micro_step * 33 / 3600.0 if sec_per_micro_step is not None else None
    )
    projected_pilot200_hours = (
        sec_per_micro_step * 66 / 3600.0 if sec_per_micro_step is not None else None
    )
    max_cuda_memory_gib = max(
        (float(row.get("max_cuda_memory_gib", 0.0)) for row in metrics),
        default=0.0,
    )

    checks = {
        "three_image_rate": three_image_rate >= args.min_three_image_rate,
        "strict_gate_rate": strict_gate_rate >= args.min_strict_gate_rate,
        "nonzero_group_rate": nonzero_group_rate >= args.min_nonzero_group_rate,
        "reference_mean": reference_mean >= args.min_reference_mean,
        "qwen_image_reward": bool(qwen_candidate_rows)
        and qwen_success_rate >= args.min_qwen_success_rate,
        "optimizer_update": update_step >= 1 and bool(grad_norms),
        "rollout_replay_parity": bool(replay_errors)
        and max(replay_errors) <= args.max_replay_logprob_error,
    }
    summary = {
        "output_dir": str(args.output_dir),
        "num_completions": len(rows),
        "num_groups": len(grouped),
        "three_image_rate": three_image_rate,
        "strict_gate_rate": strict_gate_rate,
        "nonzero_group_rate": nonzero_group_rate,
        "reference_mean": reference_mean,
        "qwen_candidate_completions": len(qwen_candidate_rows),
        "qwen_successes": qwen_successes,
        "qwen_success_rate": qwen_success_rate,
        "max_replay_logprob_error": max(replay_errors) if replay_errors else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "update_step": update_step,
        "elapsed_sec_before_final_checkpoint": elapsed_sec,
        "sec_per_micro_step": sec_per_micro_step,
        "projected_pilot100_hours": projected_pilot100_hours,
        "projected_pilot200_hours": projected_pilot200_hours,
        "max_cuda_memory_gib": max_cuda_memory_gib,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = args.output_dir / "preflight_validation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit("[FAIL] RL preflight did not pass; do not launch pilot100/pilot200.")


if __name__ == "__main__":
    main()
