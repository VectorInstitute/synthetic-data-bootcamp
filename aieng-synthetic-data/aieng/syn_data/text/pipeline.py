"""High-level orchestration helpers used by the tutorial notebooks."""

from __future__ import annotations

import itertools
import random
import uuid
from pathlib import Path
from typing import Any

from aieng.syn_data.text.clients import LLMClient
from aieng.syn_data.text.config import (
    DEFAULT_JUDGE_THRESHOLD,
    DEFAULT_SYNTHETIC_TARGET_SIZE,
    DEFAULT_TEST_PARAS_PER_DOC,
    DOCUMENT_ROLE_BY_FAILURE,
)
from aieng.syn_data.text.datasets import list_domain_documents
from aieng.syn_data.text.documents import (
    load_document_text,
    paragraphs_from_document,
    sample_test_paragraphs,
)
from aieng.syn_data.text.evaluation import (
    summarize_by_failure_mode,
    summarize_judge_scores,
)
from aieng.syn_data.text.generation import (
    few_shot_generate,
    one_shot_generate,
    topic_controlled_generate,
    zero_shot_generate,
)
from aieng.syn_data.text.io import write_json, write_jsonl
from aieng.syn_data.text.judge import judge_response, judge_synthetic_sample
from aieng.syn_data.text.quality import apply_heuristic_filters
from aieng.syn_data.text.rag import generate_grounded_qa, grounding_overlap_score
from aieng.syn_data.text.schemas import (
    FailureMode,
    JudgeScore,
    Paragraph,
    ParagraphSplit,
    QASample,
)


def effective_test_holdout(
    n_paragraphs: int, requested: int = DEFAULT_TEST_PARAS_PER_DOC
) -> int:
    """Choose a safe test holdout count that leaves at least one train paragraph."""
    if n_paragraphs <= 2:
        return 1
    return min(requested, n_paragraphs - 1)


def effective_synthetic_target(
    train_paragraphs: list[Paragraph],
    requested: int = DEFAULT_SYNTHETIC_TARGET_SIZE,
) -> int:
    """Scale synthetic target to available train paragraphs for bootcamp demos."""
    if not train_paragraphs:
        return 0
    per_para_cap = 4
    demo_cap = max(12, len(train_paragraphs) * per_para_cap)
    return min(requested, demo_cap)


def build_paragraph_splits(
    domain: str = "finance",
    *,
    n_test_per_doc: int = DEFAULT_TEST_PARAS_PER_DOC,
    seed: int = 42,
    min_chars: int = 80,
) -> list[Paragraph]:
    """Load documents, chunk paragraphs, and assign train/test splits per document."""
    specs = list_domain_documents(domain)
    all_paragraphs: list[Paragraph] = []

    for spec in specs:
        if not spec.local_path:
            msg = f"Document '{spec.doc_id}' is missing a local_path."
            raise ValueError(msg)
        text = load_document_text(Path(spec.local_path))
        doc_paragraphs = paragraphs_from_document(spec, text, min_chars=min_chars)
        holdout = effective_test_holdout(len(doc_paragraphs), n_test_per_doc)
        test_paras, train_paras = sample_test_paragraphs(
            doc_paragraphs,
            n_test=holdout,
            seed=seed,
        )
        all_paragraphs.extend(test_paras)
        all_paragraphs.extend(train_paras)

    return all_paragraphs


def failure_modes_for_paragraph(paragraph: Paragraph) -> list[FailureMode]:
    """Return failure modes appropriate for a paragraph's document role."""
    return list(DOCUMENT_ROLE_BY_FAILURE.get(paragraph.role, ()))


def generate_test_qa_batch(
    teacher: LLMClient,
    test_paragraphs: list[Paragraph],
    *,
    questions_per_para: int = 2,
) -> list[QASample]:
    """Generate held-out, hard-to-answer test Q&A from test paragraphs.

    Uses the teacher model.
    """
    samples: list[QASample] = []
    for paragraph in test_paragraphs:
        modes = failure_modes_for_paragraph(paragraph)
        if not modes:
            modes = [FailureMode.DOMAIN_VOCABULARY_DRIFT]
        for offset in range(questions_per_para):
            failure_mode = modes[offset % len(modes)]
            sample = topic_controlled_generate(
                teacher,
                paragraph,
                failure_mode=failure_mode,
            )
            sample.id = f"test-{paragraph.para_id}-{offset}"
            sample.split = ParagraphSplit.TEST
            sample.failure_mode = failure_mode
            samples.append(sample)
    return samples


def compare_generation_strategies(
    teacher: LLMClient,
    paragraph: Paragraph,
    *,
    seed_example: QASample | None = None,
    few_shot_examples: list[QASample] | None = None,
) -> dict[str, QASample]:
    """Generate one sample per strategy for side-by-side comparison."""
    seed = seed_example or QASample(
        id="seed",
        question="What is the grace period for new purchases?",
        gold_answer="The grace period ends 21 days after the close of the billing cycle.",
        doc_id=paragraph.doc_id,
        para_id=paragraph.para_id,
        context=paragraph.text,
        instruction="Answer using the policy text. Respond in one sentence.",
    )
    few_shot = few_shot_examples or [seed]
    return {
        "zero_shot": zero_shot_generate(teacher, paragraph),
        "one_shot": one_shot_generate(teacher, paragraph, seed),
        "few_shot": few_shot_generate(teacher, paragraph, few_shot),
        "topic_controlled": topic_controlled_generate(teacher, paragraph),
    }


def generate_raw_synthetic_corpus(
    teacher: LLMClient,
    train_paragraphs: list[Paragraph],
    *,
    max_paragraphs: int = 5,
) -> list[QASample]:
    """Generate a small raw corpus using every prompting strategy.

    For each train paragraph run all strategies and collect everything.
    """
    selected = train_paragraphs[:max_paragraphs]
    if not selected:
        return []

    # The teacher bootstraps its own examples — a common synthetic-data pattern.
    # Here we use zero-shot as a simple baseline.
    seed = zero_shot_generate(teacher, selected[0])
    samples: list[QASample] = []

    # For each paragraph, generate a sample using each strategy.
    for paragraph in selected:
        generated = compare_generation_strategies(
            teacher,
            paragraph,
            seed_example=seed,
            few_shot_examples=[seed],
        )
        samples.extend(generated.values())
    return samples


def filter_with_judge(
    judge: LLMClient,
    samples: list[QASample],
    *,
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
) -> tuple[list[QASample], list[JudgeScore], list[dict[str, str]]]:
    """Apply heuristics then keep samples that pass the judge threshold."""
    kept, rejected = apply_heuristic_filters(samples)
    accepted: list[QASample] = []
    scores: list[JudgeScore] = []
    judge_rejected: list[dict[str, str]] = []

    for sample in kept:
        score = judge_synthetic_sample(judge, sample)
        scores.append(score)
        if score.average >= threshold:
            accepted.append(sample)
        else:
            judge_rejected.append(
                {
                    "id": sample.id,
                    "reason": "below_judge_threshold",
                    "average": str(round(score.average, 3)),
                },
            )

    return accepted, scores, rejected + judge_rejected


def generate_grounded_training_corpus(
    teacher: LLMClient,
    train_paragraphs: list[Paragraph],
    *,
    target_size: int | None = None,
    min_overlap: float = 0.15,
    seed: int = 42,
) -> list[QASample]:
    """Generate grounded Q&A pairs until the target size is reached."""
    if not train_paragraphs:
        return []

    goal = target_size or effective_synthetic_target(train_paragraphs)
    rng = random.Random(seed)
    pool = list(train_paragraphs)
    rng.shuffle(pool)
    cycle = itertools.cycle(pool)

    samples: list[QASample] = []
    attempts = 0
    max_attempts = goal * 3

    while len(samples) < goal and attempts < max_attempts:
        attempts += 1
        paragraph = next(cycle)
        try:
            sample = generate_grounded_qa(teacher, paragraph)
        except (KeyError, ValueError, TypeError):
            continue

        overlap = grounding_overlap_score(sample.gold_answer, sample.context)
        if overlap < min_overlap:
            continue

        sample.id = f"train-{uuid.uuid4().hex[:8]}"
        sample.metadata["grounding_overlap"] = round(overlap, 3)
        samples.append(sample)

    return samples


def score_predictions(
    judge: LLMClient,
    test_samples: list[QASample],
    predictions: list[dict[str, Any]],
) -> list[JudgeScore]:
    """Score model predictions with the judge model."""
    samples_by_id = {sample.id: sample for sample in test_samples}
    scores: list[JudgeScore] = []
    for prediction in predictions:
        sample = samples_by_id[prediction["id"]]
        scores.append(
            judge_response(judge, sample, prediction["model_answer"]),
        )
    return scores


def save_baseline_results(
    predictions: list[dict[str, Any]],
    scores: list[JudgeScore],
    test_samples: list[QASample],
    *,
    predictions_path: Path,
    scores_path: Path,
) -> dict[str, Any]:
    """Persist baseline predictions and aggregated judge scores."""
    write_jsonl(predictions_path, predictions)
    samples_by_id = {sample.id: sample for sample in test_samples}
    payload = {
        "overall": summarize_judge_scores(scores),
        "by_failure_mode": summarize_by_failure_mode(scores, samples_by_id),
        "num_samples": len(scores),
    }
    write_json(scores_path, payload)
    return payload
