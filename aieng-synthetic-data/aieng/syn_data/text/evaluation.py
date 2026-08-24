"""Baseline inference and results comparison helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

import requests

from aieng.syn_data.text.schemas import JudgeScore, QASample


class InferenceClient(Protocol):
    """Minimal interface for the small model under evaluation."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a model answer."""
        ...


DEFAULT_EVAL_SYSTEM = (
    "You are a policy assistant. Follow instructions precisely. "
    "If a question is out of scope, refuse briefly and explain why."
)


def build_eval_prompt(sample: QASample) -> str:
    """Build a closed-book user prompt for training and evaluation.

    The source paragraph stays on ``sample.context`` for generation and
    judging. It is not shown to the student model.
    """
    parts = []
    if sample.instruction:
        parts.append(sample.instruction)
    parts.append(f"Question:\n{sample.question}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def run_inference(
    client: InferenceClient,
    samples: list[QASample],
    *,
    system: str = DEFAULT_EVAL_SYSTEM,
) -> list[dict[str, Any]]:
    """Run the small model on a list of evaluation samples."""
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        prompt = build_eval_prompt(sample)
        record: dict[str, Any] = {
            "id": sample.id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "model_answer": "",
            "failure_mode": (
                sample.failure_mode.value if sample.failure_mode else None
            ),
            "doc_id": sample.doc_id,
            "para_id": sample.para_id,
        }
        # Isolate failures so one bad LLM call does not discard prior predictions.
        try:
            record["model_answer"] = client.complete(
                prompt, system=system, temperature=0.0
            )
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            requests.RequestException,
        ) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(record)
    return predictions


def summarize_judge_scores(scores: list[JudgeScore]) -> dict[str, float]:
    """Compute mean judge scores across all samples."""
    if not scores:
        return {
            "correctness": 0.0,
            "coherence": 0.0,
            "instruction_following": 0.0,
            "factual_plausibility": 0.0,
            "average": 0.0,
        }
    total = len(scores)
    return {
        "correctness": sum(score.correctness for score in scores) / total,
        "coherence": sum(score.coherence for score in scores) / total,
        "instruction_following": sum(score.instruction_following for score in scores)
        / total,
        "factual_plausibility": sum(score.factual_plausibility for score in scores)
        / total,
        "average": sum(score.average for score in scores) / total,
    }


def summarize_by_failure_mode(
    scores: list[JudgeScore],
    samples_by_id: dict[str, QASample],
) -> dict[str, dict[str, float]]:
    """Aggregate judge scores grouped by failure mode."""
    grouped: dict[str, list[JudgeScore]] = defaultdict(list)
    for score in scores:
        sample = samples_by_id.get(score.sample_id)
        key = sample.failure_mode.value if sample and sample.failure_mode else "unknown"
        grouped[key].append(score)
    return {
        key: summarize_judge_scores(grouped_scores)
        for key, grouped_scores in grouped.items()
    }


def compare_summaries(
    baseline: dict[str, float],
    finetuned: dict[str, float],
) -> dict[str, float]:
    """Compute deltas between baseline and fine-tuned summaries."""
    keys = set(baseline) | set(finetuned)
    return {key: finetuned.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}
