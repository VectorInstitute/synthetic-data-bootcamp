"""Heuristic filters for synthetic Q&A quality."""

from __future__ import annotations

import re
from collections.abc import Iterable

from aieng.syn_data.text.schemas import QASample


def normalize_for_dedupe(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def is_duplicate_question(
    sample: QASample,
    seen: set[str],
) -> bool:
    """Return True if the question was already seen."""
    key = normalize_for_dedupe(sample.question)
    if key in seen:
        return True
    seen.add(key)
    return False


def has_required_fields(sample: QASample) -> bool:
    """Check that required Q&A fields are present and non-empty."""
    return bool(
        sample.question.strip()
        and sample.gold_answer.strip()
        and sample.doc_id.strip()
        and sample.para_id.strip(),
    )


def detect_prompt_leakage(sample: QASample) -> bool:
    """Return True if the answer appears to copy the generation instruction."""
    instruction = sample.instruction.lower().strip()
    answer = sample.gold_answer.lower().strip()
    if not instruction:
        return False
    return instruction in answer or "return json" in answer


def within_length_bounds(
    sample: QASample,
    *,
    min_question_chars: int = 12,
    max_question_chars: int = 500,
    min_answer_chars: int = 8,
    max_answer_chars: int = 2000,
) -> bool:
    """Check basic length constraints for questions and answers."""
    question_len = len(sample.question.strip())
    answer_len = len(sample.gold_answer.strip())
    return (
        min_question_chars <= question_len <= max_question_chars
        and min_answer_chars <= answer_len <= max_answer_chars
    )


def apply_heuristic_filters(
    samples: Iterable[QASample],
) -> tuple[list[QASample], list[dict[str, str]]]:
    """Filter samples using cheap quality heuristics.

    Returns
    -------
    tuple[list[QASample], list[dict[str, str]]]
        Kept samples and rejection records with reasons.
    """
    kept: list[QASample] = []
    rejected: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for sample in samples:
        if not has_required_fields(sample):
            rejected.append({"id": sample.id, "reason": "missing_required_fields"})
            continue
        if not within_length_bounds(sample):
            rejected.append({"id": sample.id, "reason": "length_bounds"})
            continue
        if detect_prompt_leakage(sample):
            rejected.append({"id": sample.id, "reason": "prompt_leakage"})
            continue
        if is_duplicate_question(sample, seen_questions):
            rejected.append({"id": sample.id, "reason": "duplicate_question"})
            continue
        kept.append(sample)

    return kept, rejected
