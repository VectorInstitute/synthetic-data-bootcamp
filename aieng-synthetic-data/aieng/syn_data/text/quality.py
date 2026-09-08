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


def question_dedupe_key(sample: QASample) -> str:
    """Return the normalized key used for question deduplication."""
    return normalize_for_dedupe(sample.question)


def is_duplicate_question(sample: QASample, seen: set[str]) -> bool:
    """Return True if the question key is already in ``seen`` (does not mutate)."""
    return question_dedupe_key(sample) in seen


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
    question = sample.question.lower().strip()
    if instruction and (instruction in answer or instruction in question):
        return True
    return "return json" in answer or "return json" in question


def detect_answer_leakage(sample: QASample) -> bool:
    """Return True if the gold answer is copied into the question."""
    question = normalize_for_dedupe(sample.question)
    answer = normalize_for_dedupe(sample.gold_answer)
    if not question or not answer:
        return False
    # Exact containment of a reasonably long answer, or near-identical Q/A.
    if len(answer) >= 20 and answer in question:
        return True
    return question == answer


def within_length_bounds(
    sample: QASample,
    *,
    min_question_chars: int = 12,
    max_question_chars: int = 500,
    min_answer_chars: int = 8,
    max_answer_chars: int = 8000,
) -> bool:
    """Check basic length constraints for questions and answers."""
    question_len = len(sample.question.strip())
    answer_len = len(sample.gold_answer.strip())
    return (
        min_question_chars <= question_len <= max_question_chars
        and min_answer_chars <= answer_len <= max_answer_chars
    )


def heuristic_rejection_reasons(sample: QASample, seen: set[str]) -> list[str]:
    """Return all heuristic failure reasons for a sample (empty if it passes)."""
    reasons: list[str] = []
    if not has_required_fields(sample):
        reasons.append("missing_required_fields")
    if not within_length_bounds(sample):
        reasons.append("length_bounds")
    if detect_prompt_leakage(sample):
        reasons.append("prompt_leakage")
    if detect_answer_leakage(sample):
        reasons.append("answer_in_question")
    if is_duplicate_question(sample, seen):
        reasons.append("duplicate_question")
    return reasons


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
        reasons = heuristic_rejection_reasons(sample, seen_questions)
        if reasons:
            rejected.append(
                {
                    "id": sample.id,
                    "reason": reasons[0],
                    "reasons": ", ".join(reasons),
                    "question": sample.question[:120],
                },
            )
            continue

        # Only mark a question as seen once it is kept.
        seen_questions.add(question_dedupe_key(sample))
        kept.append(sample)

    return kept, rejected


def summarize_heuristic_rejections(
    rejected: list[dict[str, str]],
) -> dict[str, int]:
    """Count rejection records by primary reason."""
    counts: dict[str, int] = {}
    for row in rejected:
        reason = row.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
