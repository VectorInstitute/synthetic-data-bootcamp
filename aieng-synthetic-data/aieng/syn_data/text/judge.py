"""LLM-as-judge helpers for synthetic data and model responses."""

from __future__ import annotations

import json
import logging
from typing import Any

from json_repair import repair_json

from aieng.syn_data.text.clients import LLMClient
from aieng.syn_data.text.schemas import JudgeScore, QASample


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for policy-document question answering. "
    "Score model outputs fairly and conservatively."
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def build_absolute_judge_prompt(
    sample: QASample,
    model_answer: str,
) -> str:
    """Build a prompt that scores one answer on four dimensions."""
    return (
        "Evaluate the model answer against the reference.\n"
        "Return JSON with numeric scores from 1 to 5 for:\n"
        "- correctness\n"
        "- coherence\n"
        "- instruction_following\n"
        "- factual_plausibility\n"
        "Also include a short 1-2 sentence max 'reasoning' string, no text outside JSON.\n\n"
        f"Question:\n{sample.question}\n\n"
        f"Reference answer:\n{sample.gold_answer}\n\n"
        f"Model answer:\n{model_answer}\n"
        "JSON format:"
        """
        {
            "correctness": <number>,
            "coherence": <number>,
            "instruction_following": <number>,
            "factual_plausibility": <number>,
            "reasoning": "<reasoning>"
        }
        """
    )


def build_pairwise_judge_prompt(
    sample: QASample,
    candidate_answer: str,
    reference_answer: str,
) -> str:
    """Build a prompt comparing two candidate answers."""
    return (
        "Compare answer A and answer B for the question below.\n"
        'Return JSON: {"winner": "A"|"B"|"tie", "reasoning": "..."}\n\n'
        f"Question:\n{sample.question}\n\n"
        f"Answer A:\n{candidate_answer}\n\n"
        f"Answer B:\n{reference_answer}\n"
    )


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


def judge_response(
    client: LLMClient,
    sample: QASample,
    model_answer: str,
) -> JudgeScore:
    """Score a model answer using absolute LLM-as-judge evaluation."""
    logger.info(
        f"Scoring model answer for sample: {sample.id} with model answer: {model_answer}"
    )
    prompt = build_absolute_judge_prompt(sample, model_answer)
    max_tokens = 256
    if hasattr(client, "complete_json"):
        payload = client.complete_json(
            prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0, max_tokens=max_tokens
        )
    else:
        raw = client.complete(
            prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0, max_tokens=max_tokens
        )

        payload = json.loads(repair_json(raw))
        if payload is None:
            raise ValueError(f"Failed to parse model JSON: {raw}")
    return parse_judge_score(sample.id, payload)


def judge_synthetic_sample(client: LLMClient, sample: QASample) -> JudgeScore:
    """Score a synthetic training sample against its own gold answer."""
    return judge_response(client, sample, sample.gold_answer)
