from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REFERENCE_TARGETS: tuple[tuple[str, str], ...] = (
    ("infrared", "infrared_clean_path"),
    ("visible", "visible_clean_path"),
    ("fused", "fused_gt_path"),
)


@dataclass(frozen=True)
class ReferenceImageMetrics:
    score: float
    psnr: float = 0.0
    psnr_normalized: float = 0.0
    ssim: float = 0.0
    width: int = 0
    height: int = 0
    error: str = ""


@dataclass(frozen=True)
class ThreeImageReferenceReward:
    score: float
    infrared: ReferenceImageMetrics = field(default_factory=lambda: ReferenceImageMetrics(score=0.0))
    visible: ReferenceImageMetrics = field(default_factory=lambda: ReferenceImageMetrics(score=0.0))
    fused: ReferenceImageMetrics = field(default_factory=lambda: ReferenceImageMetrics(score=0.0))
    weighted_mean: float = 0.0
    weakest_modality: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rgb_array(image: Image.Image | str | Path) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _local_ssim(reference: np.ndarray, prediction: np.ndarray, window_size: int = 7) -> float:
    """Compute channel-averaged local SSIM without an optional skimage dependency."""
    if reference.shape != prediction.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, reference={reference.shape}")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(f"expected RGB HWC arrays, got {reference.shape}")

    ref = torch.from_numpy(np.ascontiguousarray(reference)).permute(2, 0, 1).unsqueeze(0)
    pred = torch.from_numpy(np.ascontiguousarray(prediction)).permute(2, 0, 1).unsqueeze(0)
    kernel = max(3, int(window_size) | 1)
    padding = kernel // 2
    mu_ref = F.avg_pool2d(ref, kernel, stride=1, padding=padding)
    mu_pred = F.avg_pool2d(pred, kernel, stride=1, padding=padding)
    ref_sq = mu_ref.square()
    pred_sq = mu_pred.square()
    ref_pred = mu_ref * mu_pred
    var_ref = F.avg_pool2d(ref.square(), kernel, stride=1, padding=padding) - ref_sq
    var_pred = F.avg_pool2d(pred.square(), kernel, stride=1, padding=padding) - pred_sq
    covariance = F.avg_pool2d(ref * pred, kernel, stride=1, padding=padding) - ref_pred

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * ref_pred + c1) * (2.0 * covariance + c2)
    denominator = (ref_sq + pred_sq + c1) * (var_ref + var_pred + c2)
    score = (numerator / denominator.clamp_min(1e-12)).mean().item()
    return float(max(-1.0, min(1.0, score)))


def score_reference_pair(
    prediction: Image.Image | str | Path,
    reference: Image.Image | str | Path,
    *,
    psnr_floor: float = 10.0,
    psnr_ceiling: float = 40.0,
    ssim_weight: float = 0.6,
    psnr_weight: float = 0.4,
) -> ReferenceImageMetrics:
    try:
        pred = _rgb_array(prediction)
        ref = _rgb_array(reference)
        if pred.shape != ref.shape:
            raise ValueError(
                "reference reward forbids implicit resizing: "
                f"prediction={pred.shape}, reference={ref.shape}"
            )
        mse = float(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
        psnr = 100.0 if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)
        span = max(1e-8, float(psnr_ceiling) - float(psnr_floor))
        psnr_normalized = max(0.0, min(1.0, (psnr - float(psnr_floor)) / span))
        ssim = _local_ssim(ref, pred)
        ssim_normalized = max(0.0, min(1.0, ssim))
        total_weight = max(1e-8, float(ssim_weight) + float(psnr_weight))
        score = (
            float(ssim_weight) * ssim_normalized + float(psnr_weight) * psnr_normalized
        ) / total_weight
        return ReferenceImageMetrics(
            score=float(max(0.0, min(1.0, score))),
            psnr=float(psnr),
            psnr_normalized=float(psnr_normalized),
            ssim=float(ssim),
            width=int(pred.shape[1]),
            height=int(pred.shape[0]),
        )
    except Exception as exc:
        return ReferenceImageMetrics(score=0.0, error=f"{type(exc).__name__}: {exc}")


def score_three_image_reference(
    generated_images: Sequence[Image.Image],
    sample: Mapping[str, Any],
    *,
    psnr_floor: float = 10.0,
    psnr_ceiling: float = 40.0,
) -> ThreeImageReferenceReward:
    if len(generated_images) != len(REFERENCE_TARGETS):
        return ThreeImageReferenceReward(
            score=0.0,
            error=f"expected exactly 3 generated images, got {len(generated_images)}",
        )

    metrics: dict[str, ReferenceImageMetrics] = {}
    for index, (name, path_key) in enumerate(REFERENCE_TARGETS):
        reference_path = str(sample.get(path_key, "") or "")
        if not reference_path:
            metrics[name] = ReferenceImageMetrics(score=0.0, error=f"missing sample field: {path_key}")
            continue
        metrics[name] = score_reference_pair(
            generated_images[index],
            reference_path,
            psnr_floor=psnr_floor,
            psnr_ceiling=psnr_ceiling,
        )

    infrared = metrics["infrared"]
    visible = metrics["visible"]
    fused = metrics["fused"]
    errors = [metric.error for metric in (infrared, visible, fused) if metric.error]
    if errors:
        return ThreeImageReferenceReward(
            score=0.0,
            infrared=infrared,
            visible=visible,
            fused=fused,
            error="; ".join(errors),
        )

    weighted_mean = 0.25 * infrared.score + 0.25 * visible.score + 0.50 * fused.score
    weakest = min(infrared.score, visible.score, fused.score)
    score = 0.70 * weighted_mean + 0.30 * weakest
    return ThreeImageReferenceReward(
        score=float(max(0.0, min(1.0, score))),
        infrared=infrared,
        visible=visible,
        fused=fused,
        weighted_mean=float(weighted_mean),
        weakest_modality=float(weakest),
    )
