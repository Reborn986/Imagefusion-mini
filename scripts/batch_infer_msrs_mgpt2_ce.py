from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefusion_r1.inference.inference_solver_mgpt2_ce import (  # noqa: E402
    ImageFusionMGPT2CESolver,
    infer_output_names_from_text,
    save_outputs,
)


TARGET_OUTPUTS = {
    "infrared": ("infrared_restored.png", "infrared_clean_path"),
    "visible": ("visible_restored.png", "visible_clean_path"),
    "fused": ("fused_image.png", "fused_gt_path"),
}


def safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")
    return value[:180] or fallback


def load_manifest(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list: {path}")
    if max_samples > 0:
        data = data[:max_samples]
    return data


def iter_shard(items: List[Dict[str, Any]], shard_id: int, num_shards: int) -> Iterable[tuple[int, Dict[str, Any]]]:
    for index, item in enumerate(items):
        if index % num_shards == shard_id:
            yield index, item


def set_sample_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rgb(path: Path, size: tuple[int, int]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32)


def global_ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    pred = pred.astype(np.float64)
    scores = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for channel in range(ref.shape[2]):
        x = ref[:, :, channel]
        y = pred[:, :, channel]
        mux = x.mean()
        muy = y.mean()
        varx = ((x - mux) ** 2).mean()
        vary = ((y - muy) ** 2).mean()
        cov = ((x - mux) * (y - muy)).mean()
        numerator = (2 * mux * muy + c1) * (2 * cov + c2)
        denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
        scores.append(float(numerator / denominator))
    return float(sum(scores) / len(scores))


def image_metrics(pred_path: Path, ref_path: Optional[Path]) -> Dict[str, Any]:
    if not pred_path.exists():
        return {"exists": False}
    if ref_path is None or not ref_path.is_file():
        return {"exists": True, "ref_exists": False}
    with Image.open(pred_path) as pred_image:
        size = pred_image.size
    pred = read_rgb(pred_path, size)
    if pred is None:
        return {"exists": False}
    ref = read_rgb(ref_path, size)
    if ref is None:
        return {"exists": False, "ref_exists": False}

    mse = float(np.mean((pred - ref) ** 2))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    if skimage_ssim is not None:
        min_side = min(pred.shape[:2])
        win_size = 7 if min_side >= 7 else max(3, min_side | 1)
        ssim = skimage_ssim(
            ref,
            pred,
            channel_axis=2,
            data_range=255.0,
            win_size=win_size,
        )
        ssim_kind = "skimage"
    else:
        ssim = global_ssim(ref, pred)
        ssim_kind = "global_fallback"
    return {
        "exists": True,
        "ref_exists": True,
        "psnr": psnr,
        "ssim": float(ssim),
        "ssim_kind": ssim_kind,
    }


def compute_metrics(sample_dir: Path, sample: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key, (filename, ref_key) in TARGET_OUTPUTS.items():
        ref_value = sample.get(ref_key)
        ref_path = Path(str(ref_value)) if ref_value else None
        metrics[key] = image_metrics(sample_dir / filename, ref_path)
    return metrics


def normalize_fused_only_output(sample_dir: Path) -> List[str]:
    """Give the single fused-only image a protocol-stable filename."""
    fused_path = sample_dir / "fused_image.png"
    if not fused_path.is_file():
        candidates = [
            sample_dir / "infrared_restored.png",
            sample_dir / "visible_restored.png",
            sample_dir / "extra_image_1.png",
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise RuntimeError(
                f"Expected one fused-only image in {sample_dir}, found {len(existing)}"
            )
        existing[0].replace(fused_path)

    (sample_dir / "outputs_manifest.txt").write_text(
        "num_generated_images=1\n"
        "generated_text_file=generated_text.txt\n"
        "0: fused_image.png\n",
        encoding="utf-8",
    )
    return ["fused_image.png"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Batch inference for MSRS mGPT2 CE checkpoints.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save_root", type=Path, required=True)
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
    parser.add_argument("--prefill_stage_tag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output_protocol",
        choices=("auto", "no_cot", "fused_only", "fusion_cot_three_images"),
        default="auto",
        help=(
            "auto expects the three-target protocol; no_cot disables CoT; fused_only expects "
            "fused CoT followed by exactly one clean fused image."
        ),
    )
    parser.add_argument(
        "--gpt_prefix",
        default=None,
        help=(
            "Optional raw Assistant prefix. Use '<cot>\\n' for fused-only CoT ablation "
            "checkpoints, or leave unset for protocol-specific defaults."
        ),
    )
    parser.add_argument("--stop_after_images", type=int, default=3)
    parser.add_argument("--force_image_stages", action="store_true")
    parser.add_argument(
        "--speculative_jacobi",
        action="store_true",
        help="Enable inference-only Speculative Jacobi decoding inside image-token blocks.",
    )
    parser.add_argument(
        "--speculative_jacobi_window",
        type=int,
        default=16,
        help="Maximum number of image tokens proposed by one Jacobi forward pass.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    items = load_manifest(args.manifest, args.max_samples)
    args.save_root.mkdir(parents=True, exist_ok=True)
    (args.save_root / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

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

    failures = 0
    for index, sample in iter_shard(items, args.shard_id, args.num_shards):
        sample_id = str(sample.get("id") or f"sample_{index:03d}")
        sample_dir = args.save_root / f"{index:03d}_{safe_name(sample_id, f'sample_{index:03d}')}"
        status_path = sample_dir / "status.json"
        if args.resume and status_path.exists():
            try:
                old_status = json.loads(status_path.read_text(encoding="utf-8"))
                if old_status.get("ok") is True:
                    if args.output_protocol == "fused_only":
                        old_status["output_names"] = normalize_fused_only_output(sample_dir)
                        old_status["metrics"] = compute_metrics(sample_dir, sample)
                        status_path.write_text(
                            json.dumps(old_status, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    print(f"[SKIP] {index:03d} {sample_id}", flush=True)
                    continue
            except json.JSONDecodeError:
                pass

        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        started_at = time.time()
        status: Dict[str, Any] = {
            "index": index,
            "id": sample_id,
            "ok": False,
            "model_path": args.model_path,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        }
        try:
            set_sample_seed(args.seed + index)
            if args.force_image_stages:
                generated_text, generated_images = solver.generate_forced_image_stages(
                    infrared_path=str(sample["infrared_degraded_path"]),
                    visible_path=str(sample["visible_degraded_path"]),
                    max_gen_len=args.max_gen_len,
                    image_top_k=args.image_top_k,
                    text_top_k=args.text_top_k,
                    cfg=args.cfg,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                )
            else:
                generated_text, generated_images = solver.generate(
                    infrared_path=str(sample["infrared_degraded_path"]),
                    visible_path=str(sample["visible_degraded_path"]),
                    max_gen_len=args.max_gen_len,
                    image_top_k=args.image_top_k,
                    text_top_k=args.text_top_k,
                    cfg=args.cfg,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    stream=False,
                    prefill_stage_tag=args.prefill_stage_tag,
                    stop_after_images=args.stop_after_images,
                    gpt_prefix=args.gpt_prefix,
                )

            save_outputs(sample_dir, generated_text, generated_images)
            if args.output_protocol == "fused_only" and len(generated_images) == 1:
                output_names = normalize_fused_only_output(sample_dir)
            else:
                output_names = infer_output_names_from_text(generated_text, len(generated_images))
            metrics = compute_metrics(sample_dir, sample)
            expected_images = 1 if args.output_protocol == "fused_only" else 3
            status.update(
                {
                    "ok": len(generated_images) == expected_images,
                    "num_generated_images": len(generated_images),
                    "expected_images": expected_images,
                    "output_names": output_names,
                    "metrics": metrics,
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )
            if not status["ok"]:
                failures += 1
        except Exception as exc:
            failures += 1
            status.update(
                {
                    "ok": False,
                    "error": repr(exc),
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )
            (sample_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            print(f"[ERROR] {index:03d} {sample_id}: {exc!r}", flush=True)
        finally:
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[DONE] {index:03d} {sample_id} ok={status.get('ok')} "
                f"elapsed={status.get('elapsed_sec')}s",
                flush=True,
            )

    if failures:
        raise SystemExit(f"{failures} sample(s) failed in shard {args.shard_id}.")


if __name__ == "__main__":
    main()
