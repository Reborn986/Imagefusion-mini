from __future__ import annotations

import re
from typing import Dict, Mapping, Tuple


STAGE_TAGS: Mapping[str, Tuple[str, ...]] = {
    "infrared": ("infrared_degradation", "infrared_understand", "infrared_image"),
    "visible": ("visible_degradation", "visible_understand", "visible_image"),
    "fused": ("fused_understand", "fused_image"),
}


POLLUTION_PHRASES = (
    "the user said",
    "do not output",
    "output format",
    "let me draft",
    "i need to make sure",
    "check the requirements",
    "the requirement says",
    "no markdown",
    "plain text",
    "write a full expert",
    "the ground-truth label",
    "the label is",
    "but i can't use that directly",
    "but i should not use",
    "now, writing the output",
    "now, for the final answer",
    "now, for the final answer structure",
    "now, for the answer part",
    "now, writing the output",
    "in the reasoning",
    "i think that's good",
    "with the three tags",
    "with the three parts",
    "which i did above",
    "which i have",
)


TAG_RE = re.compile(r"</?[^<>]+>")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def _normalize_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("<answer></think>", "</think>\n<answer>")
    value = value.replace("</think>\n</think>", "</think>")
    value = value.replace("<answer>\n</think>", "</think>\n<answer>")
    value = re.sub(r"</answer>\s*</answer>", "</answer>", value, flags=re.I)
    value = re.sub(r"<answer>\s*<answer>", "<answer>", value, flags=re.I)
    return value.strip()


def _collapse_spaces(text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def _strip_xml_tags(text: str) -> str:
    return TAG_RE.sub(" ", text)


def _remove_pollution_lines(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        lowered = stripped.lower()
        if any(phrase in lowered for phrase in POLLUTION_PHRASES):
            continue
        kept.append(stripped)
    return _collapse_spaces("\n".join(kept))


def _truncate_on_sentence(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    cut = max(
        text.rfind(". ", 0, max_chars),
        text.rfind("\n", 0, max_chars),
        text.rfind("; ", 0, max_chars),
    )
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].strip()


def _extract_tag_candidates(text: str, tag: str) -> list[str]:
    """Extract from every opening tag so an earlier unclosed tag cannot swallow a later valid one."""
    start_re = re.compile(rf"<\s*{re.escape(tag)}\s*>", flags=re.I)
    end_re = re.compile(rf"<\s*/\s*{re.escape(tag)}\s*>", flags=re.I)
    candidates = []
    for start in start_re.finditer(text):
        end = end_re.search(text, start.end())
        if end is None:
            continue
        value = text[start.end() : end.start()].strip()
        if value:
            candidates.append(value)
    return candidates


def _extract_first_tag(text: str, tag: str) -> str:
    candidates = _extract_tag_candidates(text, tag)
    return candidates[0] if candidates else ""


def _extract_last_tag(text: str, tag: str) -> str:
    candidates = _extract_tag_candidates(text, tag)
    if candidates:
        return candidates[-1]

    start = re.search(rf"<\s*{re.escape(tag)}\s*>", text, flags=re.I)
    if not start:
        return ""

    tail = text[start.end() :]
    next_tag = re.search(r"<\s*/?\s*[a-zA-Z_][^<>]*>", tail)
    if next_tag:
        tail = tail[: next_tag.start()]
    return tail.strip()


def _has_answer_markup(text: str, stage: str) -> bool:
    return any(
        re.search(rf"<\s*{re.escape(tag)}\s*>", text, flags=re.I)
        for tag in STAGE_TAGS[stage]
    )


def _synthesize_think(stage: str, fields: Mapping[str, str]) -> str:
    """Reconstruct only when the source reasoning is empty/broken, using its verified answer fields."""
    if stage == "infrared":
        return (
            f"The infrared degradation is diagnosed from the paired degraded and clean references: "
            f"{fields['infrared_degradation']} The restoration must account for what is damaged and "
            f"preserved: {fields['infrared_understand']} Therefore the clean infrared target should be: "
            f"{fields['infrared_image']}"
        )
    if stage == "visible":
        return (
            f"The visible degradation is diagnosed from the paired degraded and clean references: "
            f"{fields['visible_degradation']} The restoration must account for what is damaged and "
            f"preserved: {fields['visible_understand']} Therefore the clean visible target should be: "
            f"{fields['visible_image']}"
        )
    return (
        f"The fusion target should combine the complementary restored modalities while rejecting their "
        f"degradation artifacts: {fields['fused_understand']} The expected clean fused result is: "
        f"{fields['fused_image']}"
    )


def _clean_think(text: str, max_chars: int) -> str:
    think = _extract_first_tag(text, "think")
    if not think:
        # Qwen sometimes omits the opening <think> but still emits </think>.
        # Stop at the earliest control boundary so answer fields are not copied
        # into the reconstructed reasoning.
        boundaries = [
            match.start()
            for pattern in (r"<\s*/\s*think\s*>", r"<\s*answer\s*>")
            if (match := re.search(pattern, text, flags=re.I)) is not None
        ]
        think = text[: min(boundaries)] if boundaries else text

    think = re.sub(r"<\s*answer\s*>.*", "", think, flags=re.I | re.S)
    think = _strip_xml_tags(think)
    think = _remove_pollution_lines(think)
    think = _truncate_on_sentence(think, max_chars)
    if not think:
        think = "Assess the paired degraded and clean references, then summarize the restoration target."
    return think


def _clean_field(text: str, max_chars: int) -> str:
    text = _strip_xml_tags(text)
    text = _remove_pollution_lines(text)
    text = text.strip(" :\"'\n\t")
    text = _truncate_on_sentence(text, max_chars)
    return text


def sanitize_stage_cot(
    text: object,
    stage: str,
    max_think_chars: int = 1800,
    max_field_chars: int = 900,
) -> str:
    if stage not in STAGE_TAGS:
        raise ValueError(f"Unknown CoT stage: {stage}")

    normalized = _normalize_text(text)
    raw_think = _extract_first_tag(normalized, "think")
    think = _clean_think(normalized, max_chars=max_think_chars)

    fields: Dict[str, str] = {}
    for tag in STAGE_TAGS[stage]:
        value = _extract_last_tag(normalized, tag)
        fields[tag] = _clean_field(value, max_chars=max_field_chars)

    if len(think) < 100 or _has_answer_markup(raw_think, stage):
        think = _truncate_on_sentence(
            _collapse_spaces(_synthesize_think(stage, fields)),
            max_think_chars,
        )

    answer_lines = []
    for tag, value in fields.items():
        if not value:
            value = "No clean field could be extracted from the source CoT."
        answer_lines.append(f"<{tag}>{value}</{tag}>")

    return (
        "<think>\n"
        f"{think}\n"
        "</think>\n"
        "<answer>\n"
        + "\n".join(answer_lines)
        + "\n</answer>"
    )


def sanitize_item_cots(item: Mapping[str, object]) -> Dict[str, object]:
    cleaned = dict(item)
    for stage in STAGE_TAGS:
        key = f"{stage}_cot"
        if cleaned.get(key):
            cleaned[key] = sanitize_stage_cot(cleaned[key], stage)
    return cleaned


def cot_has_pollution(text: object) -> bool:
    normalized = _normalize_text(text).lower()
    return any(phrase in normalized for phrase in POLLUTION_PHRASES) or "<answer></think>" in normalized
