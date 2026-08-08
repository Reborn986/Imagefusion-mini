from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from imagefusion_r1.rl.reference_image_reward import ThreeImageReferenceReward
from imagefusion_r1.rl.rewards_msrs import (
    completion_text_and_images,
    degradation_components,
    score_msrs_format,
)


IMAGE_PLACEHOLDER = "<|image|>"
KNOWN_DEGRADATIONS = ("stripe_noise", "noise", "blur2", "blur4", "haze", "rain")

DEGRADATION_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "noise": (
        r"\bnoise\b",
        r"grain",
        r"granul",
        r"speckl",
        r"random (?:pixel )?fluctuation",
    ),
    "stripe_noise": (
        r"stripe",
        r"striping",
        r"banding",
        r"horizontal (?:dark )?bands?",
        r"fixed[- ]pattern",
    ),
    "blur2": (r"blur", r"soft", r"loss of sharp", r"smear"),
    "blur4": (r"blur", r"soft", r"loss of sharp", r"smear"),
    "haze": (r"haze", r"fog", r"veil", r"low contrast", r"atmospheric"),
    "rain": (r"rain", r"streak", r"droplet", r"wet"),
}

PLANNING_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "infrared_preservation": (
        r"infrared",
        r"thermal",
        r"heat",
        r"target",
        r"salien",
        r"silhouette",
        r"contour",
        r"outline",
    ),
    "visible_preservation": (
        r"visible",
        r"color",
        r"texture",
        r"detail",
        r"edge",
        r"structure",
        r"spatial",
        r"layout",
    ),
    "artifact_suppression": (
        r"suppress",
        r"remove",
        r"reduce",
        r"eliminate",
        r"artifact",
        r"noise",
        r"stripe",
        r"haze",
        r"rain",
        r"blur",
    ),
    "fusion_intent": (
        r"fusion",
        r"fuse",
        r"combine",
        r"integrat",
        r"balance",
        r"complement",
    ),
}


@dataclass(frozen=True)
class FormatGateResult:
    score: float
    ok: bool
    protocol: str
    image_count_ok: bool
    placeholder_count: int
    num_generated_images: int | None
    missing: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LabelScoreResult:
    score: float
    infrared_score: float
    visible_score: float
    infrared_expected: tuple[str, ...]
    visible_expected: tuple[str, ...]
    infrared_covered: tuple[str, ...] = field(default_factory=tuple)
    visible_covered: tuple[str, ...] = field(default_factory=tuple)
    infrared_missing: tuple[str, ...] = field(default_factory=tuple)
    visible_missing: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlanningScoreResult:
    score: float
    covered_items: tuple[str, ...] = field(default_factory=tuple)
    missing_items: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TextLevelRewardResult:
    score: float
    label_score: LabelScoreResult
    planning_score: PlanningScoreResult


@dataclass(frozen=True)
class ImageLevelRewardResult:
    score: float
    artifact_suppression: float = 0.0
    visible_preservation: float = 0.0
    infrared_preservation: float = 0.0
    fusion_naturalness: float = 0.0
    semantic_consistency: float = 0.0
    overall: float | None = None
    raw: Mapping[str, Any] | None = None
    error: str = ""


@dataclass(frozen=True)
class TwoLevelRewardResult:
    score: float
    gated_score: float
    format_gate: FormatGateResult
    text_reward: TextLevelRewardResult
    image_reward: ImageLevelRewardResult
    reference_reward: ThreeImageReferenceReward
    combined_image_score: float
    text_modulation: float
    lambda_text: float
    lambda_image: float
    reference_weight: float
    qwen_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_tag(text: str, tag: str) -> str:
    pattern = re.compile(rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>", re.I | re.S)
    matches = [match.group(1).strip() for match in pattern.finditer(text)]
    return matches[-1] if matches else ""


def has_open_tag(text: str, tag: str) -> bool:
    return bool(re.search(rf"<\s*{re.escape(tag)}\s*>", text, flags=re.I))


def sample_label(sample: Mapping[str, Any] | None, modality: str, explicit_label: str = "") -> str:
    if explicit_label:
        return explicit_label
    if not sample:
        return ""

    label_key = "infrared_label" if modality == "infrared" else "visible_label"
    value = str(sample.get(label_key, "") or "")
    if value:
        return value

    path_key = "infrared_degraded_path" if modality == "infrared" else "visible_degraded_path"
    parts = list(Path(str(sample.get(path_key, "") or "")).parts)
    for level_name in ("level_1", "level_2"):
        if level_name in parts:
            index = parts.index(level_name)
            if index + 1 < len(parts):
                return parts[index + 1]
    return ""


def component_covered(component: str, text: str) -> bool:
    patterns = DEGRADATION_PATTERNS.get(component, (rf"\b{re.escape(component)}\b",))
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def score_component_coverage(expected_label: str, text: str) -> tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    expected = degradation_components(expected_label)
    if not expected:
        return 0.0, expected, (), expected
    covered = tuple(component for component in expected if component_covered(component, text))
    missing = tuple(component for component in expected if component not in covered)
    return float(len(covered) / len(expected)), expected, covered, missing


def protocol_from_text(text: str) -> str:
    if has_open_tag(text, "infrared_cot") or has_open_tag(text, "visible_cot") or has_open_tag(text, "fused_cot"):
        return "full"
    if has_open_tag(text, "cot"):
        return "fused_only"
    return "unknown"


def score_format_gate(
    completion: Any,
    *,
    protocol: str = "auto",
    expected_images: int = 3,
    require_decoded_images: bool = True,
) -> FormatGateResult:
    text, images = completion_text_and_images(completion)
    inferred_protocol = protocol_from_text(text)
    active_protocol = inferred_protocol if protocol == "auto" else protocol

    if active_protocol == "full":
        strict = score_msrs_format(
            completion,
            expected_images=expected_images,
            require_decoded_images=require_decoded_images,
            allow_open_final_image_tag=True,
        )
        problems = [*strict.missing_tags]
        problems.extend(f"empty:{tag}" for tag in strict.empty_tags)
        problems.extend(f"duplicate:{tag}" for tag in strict.duplicate_tags)
        if not strict.order_ok:
            problems.append("tag_order")
        return FormatGateResult(
            score=1.0 if strict.ok else 0.0,
            ok=strict.ok,
            protocol=active_protocol,
            image_count_ok=strict.image_count_ok,
            placeholder_count=strict.placeholder_count,
            num_generated_images=strict.num_generated_images,
            missing=tuple(problems),
        )

    placeholder_count = text.count(IMAGE_PLACEHOLDER)
    num_images = len(images) if images is not None else None
    if require_decoded_images:
        image_count_ok = num_images == expected_images
    elif num_images is None:
        image_count_ok = placeholder_count == expected_images
    else:
        image_count_ok = num_images == expected_images

    if active_protocol == "full":
        required_tags = (
            "infrared_cot",
            "clean_infrared_image",
            "visible_cot",
            "clean_visible_image",
            "fused_cot",
            "clean_fused_image",
        )
    elif active_protocol == "fused_only":
        required_tags = (
            "cot",
            "clean_infrared_image",
            "clean_visible_image",
            "clean_fused_image",
        )
    else:
        required_tags = ("clean_fused_image",)

    missing = tuple(tag for tag in required_tags if not has_open_tag(text, tag))
    placeholder_ok = placeholder_count == expected_images
    ok = active_protocol != "unknown" and image_count_ok and placeholder_ok and not missing
    return FormatGateResult(
        score=1.0 if ok else 0.0,
        ok=ok,
        protocol=active_protocol,
        image_count_ok=image_count_ok,
        placeholder_count=placeholder_count,
        num_generated_images=num_images,
        missing=missing,
    )


def modality_text_for_label(text: str, modality: str) -> str:
    direct = extract_tag(text, f"{modality}_degradation")
    if direct:
        return direct

    fused_parts = [
        extract_tag(text, "fused_understand"),
        extract_tag(text, "think"),
        extract_tag(text, "cot"),
        extract_tag(text, "fused_cot"),
    ]
    fused_text = " ".join(part for part in fused_parts if part).strip()
    return fused_text or text


def fused_planning_text(text: str) -> str:
    parts = [
        extract_tag(text, "fused_understand"),
        extract_tag(text, "fused_cot"),
        extract_tag(text, "cot"),
        extract_tag(text, "think"),
    ]
    value = " ".join(part for part in parts if part).strip()
    return value or text


def score_text_label(
    completion: Any,
    *,
    sample: Mapping[str, Any] | None = None,
    infrared_label: str = "",
    visible_label: str = "",
) -> LabelScoreResult:
    text, _images = completion_text_and_images(completion)
    ir_label = sample_label(sample, "infrared", infrared_label)
    vis_label = sample_label(sample, "visible", visible_label)

    ir_score, ir_expected, ir_covered, ir_missing = score_component_coverage(
        ir_label,
        modality_text_for_label(text, "infrared"),
    )
    vis_score, vis_expected, vis_covered, vis_missing = score_component_coverage(
        vis_label,
        modality_text_for_label(text, "visible"),
    )
    return LabelScoreResult(
        score=float(0.5 * ir_score + 0.5 * vis_score),
        infrared_score=ir_score,
        visible_score=vis_score,
        infrared_expected=ir_expected,
        visible_expected=vis_expected,
        infrared_covered=ir_covered,
        visible_covered=vis_covered,
        infrared_missing=ir_missing,
        visible_missing=vis_missing,
    )


def score_text_planning(completion: Any) -> PlanningScoreResult:
    text, _images = completion_text_and_images(completion)
    plan_text = fused_planning_text(text)
    covered = []
    missing = []
    for name, patterns in PLANNING_PATTERNS.items():
        if any(re.search(pattern, plan_text, flags=re.I) for pattern in patterns):
            covered.append(name)
        else:
            missing.append(name)
    score = len(covered) / max(1, len(PLANNING_PATTERNS))
    return PlanningScoreResult(score=float(score), covered_items=tuple(covered), missing_items=tuple(missing))


def score_text_level_reward(
    completion: Any,
    *,
    sample: Mapping[str, Any] | None = None,
    infrared_label: str = "",
    visible_label: str = "",
    label_weight: float = 0.7,
    planning_weight: float = 0.3,
) -> TextLevelRewardResult:
    label = score_text_label(
        completion,
        sample=sample,
        infrared_label=infrared_label,
        visible_label=visible_label,
    )
    planning = score_text_planning(completion)
    total_weight = max(1e-8, label_weight + planning_weight)
    score = (label_weight * label.score + planning_weight * planning.score) / total_weight
    return TextLevelRewardResult(score=float(max(0.0, min(1.0, score))), label_score=label, planning_score=planning)


def normalize_image_reward(value: ImageLevelRewardResult | Mapping[str, Any] | float | int | None) -> ImageLevelRewardResult:
    if isinstance(value, ImageLevelRewardResult):
        return value
    if isinstance(value, (int, float)):
        return ImageLevelRewardResult(score=float(max(0.0, min(1.0, value))))
    if isinstance(value, Mapping):
        return image_reward_from_qwen_scores(value)
    return ImageLevelRewardResult(score=0.0, error="missing_image_reward")


def normalize_reference_reward(
    value: ThreeImageReferenceReward | Mapping[str, Any] | float | int | None,
) -> ThreeImageReferenceReward:
    if isinstance(value, ThreeImageReferenceReward):
        return value
    if isinstance(value, (int, float)):
        return ThreeImageReferenceReward(score=float(max(0.0, min(1.0, value))))
    return ThreeImageReferenceReward(score=0.0, error="missing_reference_reward")


def _score_0_10(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(10.0, value))


def image_reward_from_qwen_scores(raw: Mapping[str, Any]) -> ImageLevelRewardResult:
    artifact = _score_0_10(raw, "artifact_suppression")
    visible = _score_0_10(raw, "visible_preservation")
    infrared = _score_0_10(raw, "infrared_preservation")
    natural = _score_0_10(raw, "fusion_naturalness")
    semantic = _score_0_10(raw, "semantic_consistency")
    weighted = (
        0.25 * artifact
        + 0.20 * visible
        + 0.20 * infrared
        + 0.20 * natural
        + 0.15 * semantic
    ) / 10.0
    overall = raw.get("overall")
    try:
        overall_value = None if overall is None else max(0.0, min(10.0, float(overall))) / 10.0
    except (TypeError, ValueError):
        overall_value = None
    return ImageLevelRewardResult(
        score=float(max(0.0, min(1.0, weighted))),
        artifact_suppression=artifact / 10.0,
        visible_preservation=visible / 10.0,
        infrared_preservation=infrared / 10.0,
        fusion_naturalness=natural / 10.0,
        semantic_consistency=semantic / 10.0,
        overall=overall_value,
        raw=dict(raw),
    )


def score_two_level_reward(
    completion: Any,
    *,
    sample: Mapping[str, Any] | None = None,
    image_reward: ImageLevelRewardResult | Mapping[str, Any] | float | int | None = None,
    reference_reward: ThreeImageReferenceReward | Mapping[str, Any] | float | int | None = None,
    protocol: str = "auto",
    lambda_text: float = 0.1,
    lambda_image: float = 0.9,
    reference_weight: float = 0.9,
    qwen_weight: float = 0.1,
    text_modulation_min: float = 1.0,
) -> TwoLevelRewardResult:
    gate = score_format_gate(completion, protocol=protocol)
    text = score_text_level_reward(completion, sample=sample)
    image = normalize_image_reward(image_reward)
    reference = normalize_reference_reward(reference_reward)

    # Keep the deterministic GT branch active even when it reports a policy
    # failure such as a wrong generated image size.  Its score is then zero and
    # the completion is penalized instead of silently promoting Qwen from a 10%
    # auxiliary judge to the entire image reward.  External data failures
    # (missing/corrupt GT) are removed group-wise by the trainer after logging.
    active_reference_weight = max(0.0, float(reference_weight))
    active_qwen_weight = max(0.0, float(qwen_weight))
    # The deterministic three-GT reward is the primary signal.  If the
    # auxiliary Qwen judge is unavailable, malformed, skipped, or inapplicable
    # for this completion, renormalize to the reference score instead of
    # injecting a false zero or discarding an expensive rollout.
    if str(image.error or "").strip():
        active_qwen_weight = 0.0
    image_weight_sum = active_reference_weight + active_qwen_weight
    if image_weight_sum > 0.0:
        combined_image = (
            active_reference_weight * reference.score + active_qwen_weight * image.score
        ) / image_weight_sum
    else:
        combined_image = 0.0

    task_weight_sum = max(1e-8, max(0.0, lambda_text) + max(0.0, lambda_image))
    base = (
        max(0.0, lambda_text) * text.score + max(0.0, lambda_image) * combined_image
    ) / task_weight_sum
    modulation = text_modulation_min + (1.0 - text_modulation_min) * text.score
    gated = gate.score * base
    score = gated * modulation
    return TwoLevelRewardResult(
        score=float(max(0.0, min(1.0, score))),
        gated_score=float(max(0.0, min(1.0, gated))),
        format_gate=gate,
        text_reward=text,
        image_reward=image,
        reference_reward=reference,
        combined_image_score=float(max(0.0, min(1.0, combined_image))),
        text_modulation=float(max(0.0, min(1.0, modulation))),
        lambda_text=lambda_text,
        lambda_image=lambda_image,
        reference_weight=reference_weight,
        qwen_weight=qwen_weight,
    )
