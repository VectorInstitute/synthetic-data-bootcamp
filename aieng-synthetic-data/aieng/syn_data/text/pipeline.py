"""High-level orchestration helpers used by the tutorial notebooks."""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

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
    extract_topics,
    few_shot_generate,
    one_shot_generate,
    topic_controlled_generate,
    zero_shot_generate,
)
from aieng.syn_data.text.io import write_json, write_jsonl
from aieng.syn_data.text.judge import judge_response, judge_synthetic_sample
from aieng.syn_data.text.quality import apply_heuristic_filters
from aieng.syn_data.text.rag import (
    generate_instruction_back_translation_sample,
    grounding_overlap_score,
)
from aieng.syn_data.text.schemas import (
    FailureMode,
    JudgeScore,
    Paragraph,
    ParagraphSplit,
    QASample,
)
from aieng.syn_data.text.seed_examples import (
    default_few_shot_examples,
    default_seed_example,
)


logger = logging.getLogger(__name__)


def effective_test_holdout(
    n_paragraphs: int, requested: int = DEFAULT_TEST_PARAS_PER_DOC
) -> int:
    """Choose a safe test holdout count that leaves at least one train paragraph."""
    if n_paragraphs <= 1:
        return 0
    if n_paragraphs == 2:
        return 1
    return min(requested, n_paragraphs - 1)


def effective_synthetic_target(
    train_paragraphs: list[Paragraph],
    requested: int = DEFAULT_SYNTHETIC_TARGET_SIZE,
    *,
    per_paragraph: int = 4,
    one_per_paragraph: bool = False,
) -> int:
    """Scale a synthetic corpus target to the available train paragraphs.

    Use this for bootcamp demos where the full production target (e.g. 500–1k)
    would overspend teacher tokens on a small document set.

    Parameters
    ----------
    train_paragraphs:
        Train-split paragraphs available for generation.
    requested:
        Desired corpus size (from config or the caller).
    per_paragraph:
        Soft cap on samples per paragraph when cycling the pool
        (ignored when ``one_per_paragraph`` is True).
    one_per_paragraph:
        If True, never request more samples than there are paragraphs.
        Prefer this for instruction back-translation, where repeat visits to
        the same paragraph with the same prompt mostly yield duplicates that
        heuristics then discard.
    """
    if not train_paragraphs:
        return 0
    if one_per_paragraph:
        return min(requested, len(train_paragraphs))
    demo_cap = max(12, len(train_paragraphs) * per_paragraph)
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

    Uses topic-controlled generation: one Q&A per paragraph topic (capped by
    ``questions_per_para``), with failure modes rotated across topics.
    """
    samples: list[QASample] = []
    for paragraph in test_paragraphs:
        modes = failure_modes_for_paragraph(paragraph)
        if not modes:
            modes = [FailureMode.DOMAIN_VOCABULARY_DRIFT]
        # Paragraph-scoped topics (not document-wide) for precise, grounded Q&As.
        topics = extract_topics(teacher, paragraph)[:questions_per_para]
        if not topics:
            failure_mode = modes[0]
            sample = topic_controlled_generate(
                teacher,
                paragraph,
                failure_mode=failure_mode,
                max_topics=1,
            )[0]
            sample.id = f"test-{paragraph.para_id}-0"
            sample.split = ParagraphSplit.TEST
            sample.failure_mode = failure_mode
            samples.append(sample)
            continue

        for offset, topic in enumerate(topics):
            failure_mode = modes[offset % len(modes)]
            sample = topic_controlled_generate(
                teacher,
                paragraph,
                topics=[topic],
                failure_mode=failure_mode,
            )[0]
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
    topic_controlled_topic: str | None = None,
) -> dict[str, QASample]:
    """Generate one sample per strategy for side-by-side comparison."""
    seed = seed_example or default_seed_example(paragraph)
    few_shot = few_shot_examples or default_few_shot_examples(paragraph)
    # Side-by-side view keeps one topic-controlled sample; full per-topic
    # expansion is used in generate_test_qa_batch / topic_controlled_generate.
    topic_samples = topic_controlled_generate(
        teacher,
        paragraph,
        topics=[topic_controlled_topic] if topic_controlled_topic is not None else None,
        max_topics=1,
    )
    return {
        "zero_shot": zero_shot_generate(teacher, paragraph),
        "one_shot": one_shot_generate(teacher, paragraph, seed),
        "few_shot": few_shot_generate(teacher, paragraph, few_shot),
        "topic_controlled": topic_samples[0],
    }


def generate_raw_synthetic_corpus(
    teacher: LLMClient,
    train_paragraphs: list[Paragraph],
    *,
    max_paragraphs: int = 5,
    questions_per_para: int = 2,
) -> list[QASample]:
    """Generate an unfiltered training corpus from train paragraphs.

    Zero/one/few-shot each contribute one sample per paragraph (format and
    style). Topic-controlled generation is expanded separately: topics are
    extracted from the paragraph and one Q&A is produced per topic, capped by
    ``questions_per_para``. Failure modes are left unset — those are a test-set
    concern. The side-by-side demo in ``compare_generation_strategies`` still
    emits one topic-controlled sample.

    Parameters
    ----------
    teacher : LLMClient
        The teacher language model client used for generation.
    train_paragraphs : list of Paragraph
        List of paragraphs to generate synthetic Q&A samples from.
    max_paragraphs : int, optional
        Maximum number of paragraphs to use from `train_paragraphs` (default is 5).
    questions_per_para : int, optional
        Maximum topic-controlled Q&As per paragraph (default is 2).

    Returns
    -------
    list of QASample
        List containing all generated QASample objects across all paragraphs and
        strategies.
    """
    selected = train_paragraphs[:max_paragraphs]
    if not selected:
        return []

    generation_errors = (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        requests.HTTPError,
    )
    # The teacher bootstraps its own examples — a common synthetic-data pattern.
    # Here we use zero-shot as a simple baseline.
    seed = zero_shot_generate(teacher, selected[0])
    samples: list[QASample] = []

    # Isolate failures so one bad LLM call does not discard prior samples.
    for paragraph in selected:
        try:
            samples.append(zero_shot_generate(teacher, paragraph))
            samples.append(one_shot_generate(teacher, paragraph, seed))
            samples.append(few_shot_generate(teacher, paragraph, [seed]))
        except generation_errors as exc:
            logger.warning(
                "Skipping prompt-strategy samples for paragraph %s: %s: %s",
                paragraph.para_id,
                type(exc).__name__,
                exc,
            )

        try:
            topics = extract_topics(teacher, paragraph)[:questions_per_para]
            samples.extend(
                topic_controlled_generate(
                    teacher,
                    paragraph,
                    topics=topics,
                    max_topics=questions_per_para,
                )
            )
        except generation_errors as exc:
            logger.warning(
                "Skipping topic-controlled samples for paragraph %s: %s: %s",
                paragraph.para_id,
                type(exc).__name__,
                exc,
            )
    return samples


def filter_with_judge(
    judge: LLMClient,
    samples: list[QASample],
    *,
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
    on_progress: Callable[[], None] | None = None,
) -> tuple[list[QASample], list[JudgeScore], list[dict[str, str]]]:
    """Apply heuristics then keep samples that pass the judge threshold.

    Parameters
    ----------
    on_progress:
        Called once per finished sample. A Rich ``Progress.advance`` hook
        stays valid if judging is later parallelized.
    """
    kept, rejected = apply_heuristic_filters(samples)
    accepted: list[QASample] = []
    scores: list[JudgeScore] = []
    judge_rejected: list[dict[str, str]] = []

    for sample in kept:
        score = judge_synthetic_sample(judge, sample)
        scores.append(score)
        if on_progress is not None:
            on_progress()
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
    """Generate instruction-backtranslation Q&A until the target size is reached.

    Each sample asks the teacher for a question answered by a train paragraph.
    Defaults to at most one sample per paragraph so the pool is not cycled with
    an identical prompt (which mostly produces duplicates).

    Parameters
    ----------
    teacher:
        Teacher LLM client.
    train_paragraphs:
        Train-split paragraphs used as answer text.
    target_size:
        Desired number of samples. Defaults to one-per-paragraph demo scaling.
    min_overlap:
        Minimum lexical overlap between gold answer and passage.
    seed:
        RNG seed for paragraph shuffle.
    """
    if not train_paragraphs:
        return []
    # Preserve explicit target_size=0 without calling effective_synthetic_target
    # or generating LLM requests.
    goal = (
        target_size
        if target_size is not None
        else effective_synthetic_target(
            train_paragraphs,
            one_per_paragraph=True,
        )
    )
    # Avoid repeat visits that regenerate near-identical questions.
    goal = min(goal, len(train_paragraphs))

    rng = random.Random(seed)
    pool = list(train_paragraphs)
    rng.shuffle(pool)

    samples: list[QASample] = []
    seen_paras: set[str] = set()

    for paragraph in pool:
        if len(samples) >= goal:
            break
        if paragraph.para_id in seen_paras:
            continue
        seen_paras.add(paragraph.para_id)
        try:
            sample = generate_instruction_back_translation_sample(teacher, paragraph)
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            requests.HTTPError,
        ) as exc:
            logger.warning(
                "Skipping paragraph %s after generation failure: %s: %s",
                paragraph.para_id,
                type(exc).__name__,
                exc,
            )
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
        # Skip predictions with inference errors
        if prediction.get("error"):
            logger.warning(
                "Skipping prediction %s with inference error: %s",
                prediction.get("id"),
                prediction["error"],
            )
            continue
        sample = samples_by_id.get(prediction["id"])
        if sample is None:
            logger.warning("No test sample for prediction id %s", prediction["id"])
            continue

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
