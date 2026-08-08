from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IMAGE_PLACEHOLDER = "<|image|>"

REQUIRED_TAG_ORDER: tuple[tuple[str, str], ...] = (
    ("infrared_cot", "cot"),
    ("clean_infrared_image", "image"),
    ("visible_cot", "cot"),
    ("clean_visible_image", "image"),
    ("fused_cot", "cot"),
    ("clean_fused_image", "image"),
)

STAGE_INNER_TAGS: Mapping[str, tuple[str, ...]] = {
    "infrared_cot": (
        "think",
        "answer",
        "infrared_degradation",
        "infrared_understand",
        "infrared_image",
    ),
    "visible_cot": (
        "think",
        "answer",
        "visible_degradation",
        "visible_understand",
        "visible_image",
    ),
    "fused_cot": (
        "think",
        "answer",
        "fused_understand",
        "fused_image",
    ),
}

KNOWN_DEGRADATIONS = ("stripe_noise", "noise", "blur2", "blur4", "haze", "rain")

DEGRADATION_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "noise": (
        r"\bnoise\b",
        r"grain",
        r"granul",
        r"speckl",
        r"random (?:pixel )?fluctuation",
    ),
    "stripe_noise": (r"strip", r"banding", r"banded artifact", r"fixed-pattern"),
    "blur2": (r"blur", r"soften", r"loss of sharp"),
    "blur4": (r"blur", r"soften", r"loss of sharp"),
    "haze": (r"haze", r"fog", r"veil", r"atmospheric scattering"),
    "rain": (r"rain", r"streak", r"droplet"),
}


@dataclass(frozen=True)
class _SyntheticTagMatch:
    start_index: int
    content: str

    def start(self) -> int:
        return self.start_index

    def group(self, index: int = 0) -> str:
        if index == 1:
            return self.content
        return self.content


@dataclass(frozen=True)
class FormatRewardResult:
    score: float
    ok: bool
    text_score: float
    image_score: float
    order_ok: bool
    image_count_ok: bool
    placeholder_count: int
    num_generated_images: int | None
    missing_tags: tuple[str, ...] = field(default_factory=tuple)
    empty_tags: tuple[str, ...] = field(default_factory=tuple)
    duplicate_tags: tuple[str, ...] = field(default_factory=tuple)
    extra_image_placeholders: int = 0
    final_image_tag_open: bool = False


@dataclass(frozen=True)
class ModalityDegradationScore:
    score: float
    expected_label: str
    expected_components: tuple[str, ...]
    covered_components: tuple[str, ...] = field(default_factory=tuple)
    missing_components: tuple[str, ...] = field(default_factory=tuple)
    unexpected_components: tuple[str, ...] = field(default_factory=tuple)
    field_missing: bool = False


@dataclass(frozen=True)
class DegradationTextRewardResult:
    score: float
    ok: bool
    infrared: ModalityDegradationScore
    visible: ModalityDegradationScore


def completion_text_and_images(completion: Any) -> tuple[str, Sequence[Any] | None]:
    """Normalize trainer completion shapes into generated text and images."""
    if isinstance(completion, str):
        return completion, None

    if isinstance(completion, Mapping):
        text = completion.get("generated_text", completion.get("text", completion.get("content", "")))
        images = completion.get("generated_images", completion.get("images"))
        return str(text or ""), images if isinstance(images, Sequence) else None

    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        if len(completion) >= 2 and isinstance(completion[0], str):
            images = completion[1]
            return completion[0], images if isinstance(images, Sequence) else None
        if len(completion) == 1 and isinstance(completion[0], Mapping):
            return completion_text_and_images(completion[0])

    return str(completion or ""), None


def _tag_pattern(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)


def _extract_last_tag(text: str, tag: str) -> str:
    matches = list(_tag_pattern(tag).finditer(text))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _find_required_tag_spans(
    text: str,
    *,
    allow_open_final_image_tag: bool,
) -> tuple[dict[str, re.Match[str] | _SyntheticTagMatch], list[str], list[str], list[str], bool]:
    matches: dict[str, re.Match[str] | _SyntheticTagMatch] = {}
    missing: list[str] = []
    empty: list[str] = []
    duplicate: list[str] = []
    final_image_tag_open = False

    for tag, _kind in REQUIRED_TAG_ORDER:
        found = list(_tag_pattern(tag).finditer(text))
        if not found and allow_open_final_image_tag and tag == "clean_fused_image":
            final_open = re.search(
                rf"<{re.escape(tag)}>\s*{re.escape(IMAGE_PLACEHOLDER)}\s*$",
                text,
                re.DOTALL,
            )
            if final_open:
                matches[tag] = _SyntheticTagMatch(final_open.start(), IMAGE_PLACEHOLDER)
                final_image_tag_open = True
                continue
        if not found:
            missing.append(tag)
            continue
        if len(found) > 1:
            duplicate.append(tag)
        matches[tag] = found[0]
        if not found[0].group(1).strip():
            empty.append(tag)

    for stage_tag, inner_tags in STAGE_INNER_TAGS.items():
        stage_match = matches.get(stage_tag)
        if stage_match is None:
            for inner_tag in inner_tags:
                missing.append(f"{stage_tag}.{inner_tag}")
            continue
        stage_text = stage_match.group(1)
        for inner_tag in inner_tags:
            found = list(_tag_pattern(inner_tag).finditer(stage_text))
            qualified = f"{stage_tag}.{inner_tag}"
            if not found:
                missing.append(qualified)
                continue
            if len(found) > 1:
                duplicate.append(qualified)
            matches[qualified] = found[0]
            if not found[0].group(1).strip():
                empty.append(qualified)

    return matches, missing, empty, duplicate, final_image_tag_open


def _required_order_ok(matches: Mapping[str, re.Match[str] | _SyntheticTagMatch]) -> bool:
    last_start = -1
    for tag, _kind in REQUIRED_TAG_ORDER:
        match = matches.get(tag)
        if match is None or match.start() <= last_start:
            return False
        last_start = match.start()
    return True


def _image_placeholder_count(text: str) -> int:
    return text.count(IMAGE_PLACEHOLDER)


def degradation_components(label: str) -> tuple[str, ...]:
    """Split compound MSRS degradation labels without breaking stripe_noise."""
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


def _label_from_degraded_path(path: str) -> str:
    parts = list(Path(str(path or "")).parts)
    for level_name in ("level_1", "level_2"):
        if level_name in parts:
            index = parts.index(level_name)
            if index + 1 < len(parts):
                return parts[index + 1]
    return ""


def _label_from_sample(sample: Mapping[str, Any] | None, modality: str) -> str:
    if not sample:
        return ""
    label_key = "infrared_label" if modality == "infrared" else "visible_label"
    label = str(sample.get(label_key, "") or "")
    if label:
        return label
    path_key = "infrared_degraded_path" if modality == "infrared" else "visible_degraded_path"
    return _label_from_degraded_path(str(sample.get(path_key, "") or ""))


def _component_covered(component: str, text: str) -> bool:
    patterns = DEGRADATION_PATTERNS.get(component, (rf"\b{re.escape(component)}\b",))
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _is_tolerated_extra_component(component: str, expected_components: Sequence[str]) -> bool:
    if component in ("blur2", "blur4") and any(item in ("blur2", "blur4") for item in expected_components):
        return True
    if component == "noise" and "stripe_noise" in expected_components:
        return True
    return False


def _score_modality_degradation(
    text: str,
    *,
    modality: str,
    expected_label: str,
    penalize_unexpected: bool,
) -> ModalityDegradationScore:
    tag = f"{modality}_degradation"
    field = _extract_last_tag(text, tag)
    expected_components = degradation_components(expected_label)
    if not field or not expected_components:
        return ModalityDegradationScore(
            score=0.0,
            expected_label=expected_label,
            expected_components=expected_components,
            missing_components=expected_components,
            field_missing=not bool(field),
        )

    covered = tuple(component for component in expected_components if _component_covered(component, field))
    missing = tuple(component for component in expected_components if component not in covered)
    unexpected = tuple(
        component
        for component in KNOWN_DEGRADATIONS
        if component not in expected_components
        and not _is_tolerated_extra_component(component, expected_components)
        and _component_covered(component, field)
    )

    coverage = len(covered) / len(expected_components)
    unexpected_penalty = min(0.25, 0.08 * len(unexpected)) if penalize_unexpected else 0.0
    score = max(0.0, coverage - unexpected_penalty)
    return ModalityDegradationScore(
        score=float(min(1.0, score)),
        expected_label=expected_label,
        expected_components=expected_components,
        covered_components=covered,
        missing_components=missing,
        unexpected_components=unexpected,
        field_missing=False,
    )


def score_msrs_format(
    completion: Any,
    *,
    expected_images: int = 3,
    require_decoded_images: bool = True,
    allow_open_final_image_tag: bool = True,
) -> FormatRewardResult:
    """Score MSRS three-stage text/image output format.

    The strict success condition requires the fixed IR -> VIS -> Fused tag order,
    non-empty required tags, exactly three image placeholders, and exactly three
    decoded images when image objects are available. In the current mGPT2
    inference path, decoding often stops immediately after the third image, so
    the final ``</clean_fused_image>`` can be tolerated while scoring.
    """
    text, images = completion_text_and_images(completion)
    matches, missing, empty, duplicate, final_image_tag_open = _find_required_tag_spans(
        text,
        allow_open_final_image_tag=allow_open_final_image_tag,
    )

    order_ok = _required_order_ok(matches)
    placeholder_count = _image_placeholder_count(text)
    extra_placeholders = max(0, placeholder_count - expected_images)

    num_images = len(images) if images is not None else None
    if require_decoded_images:
        image_count_ok = num_images == expected_images
    elif num_images is None:
        image_count_ok = placeholder_count == expected_images
    else:
        image_count_ok = num_images == expected_images

    total_required_tags = len(REQUIRED_TAG_ORDER) + sum(len(tags) for tags in STAGE_INNER_TAGS.values())
    present_nonempty = total_required_tags - len(set(missing)) - len(set(empty))
    tag_score = max(0.0, present_nonempty / total_required_tags)
    duplicate_penalty = min(0.20, 0.04 * len(set(duplicate)))
    order_score = 1.0 if order_ok else 0.0
    placeholder_score = 1.0 if placeholder_count == expected_images else max(
        0.0,
        1.0 - abs(placeholder_count - expected_images) / expected_images,
    )
    text_score = max(0.0, 0.75 * tag_score + 0.15 * order_score + 0.10 * placeholder_score - duplicate_penalty)

    if image_count_ok:
        image_score = 1.0
    elif num_images is None and not require_decoded_images:
        image_score = placeholder_score
    elif num_images is None:
        image_score = 0.0
    else:
        image_score = max(0.0, 1.0 - abs(num_images - expected_images) / expected_images)

    score = 0.70 * text_score + 0.30 * image_score
    strict_ok = (
        order_ok
        and image_count_ok
        and placeholder_count == expected_images
        and not missing
        and not empty
        and not duplicate
    )
    if strict_ok:
        score = 1.0

    return FormatRewardResult(
        score=float(max(0.0, min(1.0, score))),
        ok=strict_ok,
        text_score=float(max(0.0, min(1.0, text_score))),
        image_score=float(max(0.0, min(1.0, image_score))),
        order_ok=order_ok,
        image_count_ok=image_count_ok,
        placeholder_count=placeholder_count,
        num_generated_images=num_images,
        missing_tags=tuple(missing),
        empty_tags=tuple(empty),
        duplicate_tags=tuple(duplicate),
        extra_image_placeholders=extra_placeholders,
        final_image_tag_open=final_image_tag_open,
    )


def msrs_format_reward(
    completions: Iterable[Any],
    *,
    expected_images: int = 3,
    require_decoded_images: bool = True,
    allow_open_final_image_tag: bool = True,
    **_kwargs: Any,
) -> list[float]:
    """TRL/GRPO-compatible reward function for MSRS output format."""
    return [
        score_msrs_format(
            completion,
            expected_images=expected_images,
            require_decoded_images=require_decoded_images,
            allow_open_final_image_tag=allow_open_final_image_tag,
        ).score
        for completion in completions
    ]


def score_msrs_degradation_text(
    completion: Any,
    *,
    sample: Mapping[str, Any] | None = None,
    infrared_label: str = "",
    visible_label: str = "",
    penalize_unexpected: bool = False,
) -> DegradationTextRewardResult:
    """Score whether generated CoT names the expected IR/VIS degradations.

    This reward intentionally focuses on the semantic content inside
    ``<infrared_degradation>`` and ``<visible_degradation>``. Structural XML
    mistakes are handled by ``score_msrs_format`` so the two rewards can provide
    complementary learning signals during RL.
    """
    text, _images = completion_text_and_images(completion)
    ir_label = infrared_label or _label_from_sample(sample, "infrared")
    vis_label = visible_label or _label_from_sample(sample, "visible")
    infrared = _score_modality_degradation(
        text,
        modality="infrared",
        expected_label=ir_label,
        penalize_unexpected=penalize_unexpected,
    )
    visible = _score_modality_degradation(
        text,
        modality="visible",
        expected_label=vis_label,
        penalize_unexpected=penalize_unexpected,
    )
    score = 0.5 * infrared.score + 0.5 * visible.score
    return DegradationTextRewardResult(
        score=float(max(0.0, min(1.0, score))),
        ok=infrared.score >= 1.0 and visible.score >= 1.0,
        infrared=infrared,
        visible=visible,
    )


def _select_parallel_arg(values: Any, index: int) -> Any:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        if index < len(values):
            return values[index]
    return None


def msrs_degradation_text_reward(
    completions: Iterable[Any],
    *,
    samples: Sequence[Mapping[str, Any]] | None = None,
    infrared_labels: Sequence[str] | None = None,
    visible_labels: Sequence[str] | None = None,
    penalize_unexpected: bool = False,
    **_kwargs: Any,
) -> list[float]:
    """TRL/GRPO-compatible reward for IR/VIS degradation recognition."""
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        sample = _select_parallel_arg(samples, index)
        infrared_label = _select_parallel_arg(infrared_labels, index) or ""
        visible_label = _select_parallel_arg(visible_labels, index) or ""
        rewards.append(
            score_msrs_degradation_text(
                completion,
                sample=sample if isinstance(sample, Mapping) else None,
                infrared_label=str(infrared_label),
                visible_label=str(visible_label),
                penalize_unexpected=penalize_unexpected,
            ).score
        )
    return rewards
