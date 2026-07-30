"""LLM-as-judge helpers for synthetic data and model responses."""

from __future__ import annotations

import json
import logging
from typing import Any

from json_repair import repair_json

from aieng.syn_data.text.clients import LLMClient
from aieng.syn_data.text.prompts import (
    RESPONSE_JUDGE_SYSTEM_PROMPT,
    SYNTHETIC_QA_JUDGE_SYSTEM_PROMPT,
    absolute_response_judge_prompt,
    pairwise_response_judge_prompt,
    synthetic_qa_quality_prompt,
)
from aieng.syn_data.text.schemas import JudgeScore, QASample

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def build_absolute_judge_prompt(
    sample: QASample,
    model_answer: str,
) -> str:
    """Build a prompt that scores one model answer vs a reference gold answer."""
    return absolute_response_judge_prompt(sample, model_answer)


def build_pairwise_judge_prompt(
    sample: QASample,
    candidate_answer: str,
    reference_answer: str,
) -> str:
    """Build a prompt comparing two candidate answers."""
    return pairwise_response_judge_prompt(sample, candidate_answer, reference_answer)


def build_synthetic_qa_judge_prompt(sample: QASample) -> str:
    """Build a prompt that scores synthetic Q&A quality given source context."""
    return synthetic_qa_quality_prompt(sample)


def parse_judge_score(sample_id: str, payload: dict[str, Any]) -> JudgeScore:
    """Convert a judge JSON payload into a typed score object."""
    return JudgeScore(
        sample_id=sample_id,
        correctness=float(payload["correctness"]),
        coherence=float(payload["coherence"]),
        instruction_following=float(payload["instruction_following"]),
        factual_plausibility=float(payload["factual_plausibility"]),
        reasoning=str(payload.get("reasoning", "")),
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "correctness",
                "coherence",
                "instruction_following",
                "factual_plausibility",
                "reasoning",
            }
        },
    )


def _complete_judge_json(
    client: LLMClient,
    prompt: str,
    *,
    system: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    if hasattr(client, "complete_json"):
        return client.complete_json(
            prompt, system=system, temperature=0.0, max_tokens=max_tokens
        )
    raw = client.complete(
        prompt, system=system, temperature=0.0, max_tokens=max_tokens
    )
    payload = json.loads(repair_json(raw))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Model response did not contain a JSON object: {raw[:300]!r}"
        )
    return payload


def judge_response(
    client: LLMClient,
    sample: QASample,
    model_answer: str,
) -> JudgeScore:
    """Score a model answer using absolute LLM-as-judge evaluation.

    Use this after inference, when you have a model response to compare
    against ``sample.gold_answer``.
    """
    logger.info(
        "Scoring model answer for sample: %s (answer length: %d)",
        sample.id,
        len(model_answer),
    )
    prompt = build_absolute_judge_prompt(sample, model_answer)
    payload = _complete_judge_json(
        client, prompt, system=RESPONSE_JUDGE_SYSTEM_PROMPT
    )
    return parse_judge_score(sample.id, payload)


def judge_synthetic_sample(client: LLMClient, sample: QASample) -> JudgeScore:
    """Score synthetic Q&A quality given the source passage (pre-inference).

    Unlike :func:`judge_response`, this does **not** expect a model answer.
    It judges whether the question and gold answer form a good training pair
    relative to ``sample.context``.
    """
    logger.info("Scoring synthetic Q&A quality for sample: %s", sample.id)
    prompt = build_synthetic_qa_judge_prompt(sample)
    payload = _complete_judge_json(
        client, prompt, system=SYNTHETIC_QA_JUDGE_SYSTEM_PROMPT
    )
    return parse_judge_score(sample.id, payload)
