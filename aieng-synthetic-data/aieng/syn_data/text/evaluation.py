"""Baseline inference and results comparison helpers."""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

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
    """Build the user prompt for baseline or fine-tuned evaluation."""
    parts = []
    if sample.instruction:
        parts.append(sample.instruction)
    if sample.context:
        parts.append(f"Context:\n{sample.context}")
    parts.append(f"Question:\n{sample.question}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def run_inference(
    client: InferenceClient,
    samples: list[QASample],
    *,
    system: str = DEFAULT_EVAL_SYSTEM,
    max_concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run the small model on a list of evaluation samples.

    Requests are dispatched concurrently on a thread pool (inference is
    I/O-bound on the model API); a semaphore caps how many calls are in
    flight at once so the batch respects the API's rate limits.

    Parameters
    ----------
    client : InferenceClient
        The small model client under evaluation.
    samples : list of QASample
        Evaluation samples to run inference on.
    system : str, optional
        System prompt to use for every completion.
    max_concurrency : int, optional
        Maximum number of inference calls allowed to run at the same time
        (default is 4).

    Returns
    -------
    list of dict
        One prediction record per sample, in the same order as `samples`.
    """
    if not samples:
        return []

    semaphore = threading.Semaphore(max_concurrency)

    def _run_one(sample: QASample) -> dict[str, Any]:
        prompt = build_eval_prompt(sample)
        with semaphore:
            model_answer = client.complete(prompt, system=system, temperature=0.0)
        return {
            "id": sample.id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "model_answer": model_answer,
            "failure_mode": (
                sample.failure_mode.value if sample.failure_mode else None
            ),
            "doc_id": sample.doc_id,
            "para_id": sample.para_id,
        }

    worker_count = max(1, min(len(samples), max_concurrency * 2))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_one, sample) for sample in samples]
        return [future.result() for future in futures]


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
