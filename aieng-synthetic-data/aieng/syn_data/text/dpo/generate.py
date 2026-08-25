"""Teacher-LLM generation of SEC boundary questions and candidate answers."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast

import requests

from aieng.syn_data.text.clients import LLMClient, extract_json_text
from aieng.syn_data.text.dpo.config import DEFAULT_DPO_QUESTIONS, SEC_DOC_ID
from aieng.syn_data.text.dpo.prompts import (
    BOUNDARY_QUESTION_SYSTEM,
    CALIBRATION_CANDIDATE_SYSTEM,
    DEFAULT_BOUNDARY_INSTRUCTION,
    boundary_question_prompt,
    calibration_candidates_prompt,
)
from aieng.syn_data.text.dpo.schemas import (
    BoundaryQuestionKind,
    CalibrationPrompt,
    CandidateKind,
    PreferenceCandidate,
)
from aieng.syn_data.text.schemas import DocumentRole, Paragraph, ParagraphSplit


logger = logging.getLogger(__name__)

_QUESTION_KIND_CYCLE: tuple[BoundaryQuestionKind, ...] = (
    BoundaryQuestionKind.IN_SCOPE,
    BoundaryQuestionKind.OUT_OF_SCOPE,
    BoundaryQuestionKind.GRAY_BOUNDARY,
)

_CANDIDATE_KEYS: tuple[tuple[str, CandidateKind], ...] = (
    ("correctly_scoped", CandidateKind.CORRECTLY_SCOPED),
    ("overreaching", CandidateKind.OVERREACHING),
    ("underreaching", CandidateKind.UNDERREACHING),
    ("authority_misattribution", CandidateKind.AUTHORITY_MISATTRIBUTION),
)


def filter_sec_train_paragraphs(
    paragraphs: list[Paragraph],
    *,
    doc_id: str = SEC_DOC_ID,
) -> list[Paragraph]:
    """Keep train-split SEC / scope-boundary paragraphs only."""
    return [
        paragraph
        for paragraph in paragraphs
        if paragraph.doc_id == doc_id
        and paragraph.split == ParagraphSplit.TRAIN
        and paragraph.role == DocumentRole.SCOPE_BOUNDARY
    ]


def _complete_json(
    client: LLMClient,
    prompt: str,
    *,
    system: str,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    if hasattr(client, "complete_json"):
        return cast(
            dict[str, Any],
            client.complete_json(
                prompt,
                system=system,
                temperature=0.4,
                max_tokens=max_tokens,
            ),
        )
    raw = client.complete(prompt, system=system, temperature=0.4, max_tokens=max_tokens)
    return cast(dict[str, Any], json.loads(extract_json_text(raw)))


def generate_boundary_prompts(
    teacher: LLMClient,
    paragraphs: list[Paragraph],
    *,
    n_questions: int = DEFAULT_DPO_QUESTIONS,
) -> list[CalibrationPrompt]:
    """Generate grounded boundary questions from SEC train paragraphs.

    Cycles in-scope, out-of-scope, and gray-boundary types. Does not yet
    attach candidate answers; call :func:`generate_calibration_candidates`.
    """
    if not paragraphs:
        return []

    prompts: list[CalibrationPrompt] = []
    for index in range(n_questions):
        paragraph = paragraphs[index % len(paragraphs)]
        question_kind = _QUESTION_KIND_CYCLE[index % len(_QUESTION_KIND_CYCLE)]
        try:
            payload = _complete_json(
                teacher,
                boundary_question_prompt(paragraph.text, question_kind),
                system=BOUNDARY_QUESTION_SYSTEM,
                max_tokens=512,
            )
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            requests.HTTPError,
        ) as exc:
            logger.warning(
                "Skipping boundary question for %s: %s: %s",
                paragraph.para_id,
                type(exc).__name__,
                exc,
            )
            continue

        question = str(payload.get("question", "")).strip()
        if not question:
            logger.warning("Empty question for paragraph %s", paragraph.para_id)
            continue

        kind_raw = str(payload.get("question_kind", question_kind.value)).strip()
        try:
            parsed_kind = BoundaryQuestionKind(kind_raw)
        except ValueError:
            parsed_kind = question_kind

        prompts.append(
            CalibrationPrompt(
                id=f"dpo-{uuid.uuid4().hex[:8]}",
                question=question,
                doc_id=paragraph.doc_id,
                para_id=paragraph.para_id,
                context=paragraph.text,
                question_kind=parsed_kind,
                instruction=DEFAULT_BOUNDARY_INSTRUCTION,
                metadata={"requested_question_kind": question_kind.value},
            )
        )
    return prompts


def _parse_candidates(payload: dict[str, Any]) -> list[PreferenceCandidate]:
    candidates: list[PreferenceCandidate] = []
    for key, kind in _CANDIDATE_KEYS:
        item = payload.get(key, {})
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer", "")).strip()
        if not answer:
            continue
        candidates.append(
            PreferenceCandidate(
                kind=kind,
                answer=answer,
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return candidates


def generate_calibration_candidates(
    teacher: LLMClient,
    prompt: CalibrationPrompt,
) -> CalibrationPrompt:
    """Fill in four labeled candidate answers with one teacher JSON call."""
    payload = _complete_json(
        teacher,
        calibration_candidates_prompt(prompt.context, prompt.question),
        system=CALIBRATION_CANDIDATE_SYSTEM,
        max_tokens=2048,
    )
    prompt.candidates = _parse_candidates(payload)
    return prompt


def generate_calibration_set(
    teacher: LLMClient,
    paragraphs: list[Paragraph],
    *,
    n_questions: int = DEFAULT_DPO_QUESTIONS,
) -> list[CalibrationPrompt]:
    """Generate boundary questions and four candidates for each."""
    prompts = generate_boundary_prompts(teacher, paragraphs, n_questions=n_questions)
    completed: list[CalibrationPrompt] = []
    for prompt in prompts:
        try:
            completed.append(generate_calibration_candidates(teacher, prompt))
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            requests.HTTPError,
        ) as exc:
            logger.warning(
                "Skipping candidates for %s: %s: %s",
                prompt.id,
                type(exc).__name__,
                exc,
            )
    return completed
