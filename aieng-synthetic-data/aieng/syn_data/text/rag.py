"""Passage-grounded synthetic Q&A via instruction back-translation.

Also includes a lightweight lexical retriever for optional RAG-style workflows.
The notebook-04 training path uses instruction back-translation: given passage
text *y*, generate a question/instruction *x* for which *y* is a good answer
(https://openreview.net/forum?id=1oijHJBRsT).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from typing import Any

from aieng.syn_data.text.clients import LLMClient
from aieng.syn_data.text.prompts import (
    INSTRUCTION_BACKTRANSLATION_SYSTEM,
    instruction_backtranslation_prompt,
)
from aieng.syn_data.text.schemas import GenerationStrategy, Paragraph, QASample


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_index(
    paragraphs: list[Paragraph],
) -> tuple[list[Counter[str]], dict[str, float]]:
    """Build a lightweight TF-IDF representation for paragraphs."""
    doc_freq: Counter[str] = Counter()
    term_counters: list[Counter[str]] = []

    for paragraph in paragraphs:
        tokens = tokenize(paragraph.text)
        counts = Counter(tokens)
        term_counters.append(counts)
        doc_freq.update(set(counts))

    num_docs = max(len(paragraphs), 1)
    idf = {
        term: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }
    return term_counters, idf


def tfidf_vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    """Convert a term count vector into a TF-IDF vector."""
    total = sum(counts.values()) or 1
    return {
        term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()
    }


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve_paragraphs(
    query: str,
    paragraphs: list[Paragraph],
    *,
    top_k: int = 3,
) -> list[Paragraph]:
    """Retrieve the most relevant train paragraphs for a query.

    Not used by the default notebook-04 back-translation path (each sample is
    generated from a known paragraph). Kept for optional retrieval-augmented
    variants.
    """
    if not paragraphs:
        return []
    term_counters, idf = build_tfidf_index(paragraphs)
    query_vector = tfidf_vector(Counter(tokenize(query)), idf)

    scored: list[tuple[float, Paragraph]] = []
    for counts, paragraph in zip(term_counters, paragraphs, strict=True):
        paragraph_vector = tfidf_vector(counts, idf)
        score = cosine_similarity(query_vector, paragraph_vector)
        scored.append((score, paragraph))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [paragraph for _, paragraph in scored[:top_k]]


def grounded_qa_prompt(passage: str) -> str:
    """Backward-compatible alias for :func:`instruction_backtranslation_prompt`."""
    return instruction_backtranslation_prompt(passage)


def generate_grounded_qa(
    client: LLMClient,
    paragraph: Paragraph,
) -> QASample:
    """Generate a Q&A pair via instruction back-translation.

    The teacher proposes a question for which the paragraph text is a good
    answer. The gold answer is the passage itself (classic back-translation).
    """
    prompt = instruction_backtranslation_prompt(paragraph.text)
    if hasattr(client, "complete_json"):
        payload: dict[str, Any] = client.complete_json(
            prompt,
            system=INSTRUCTION_BACKTRANSLATION_SYSTEM,
        )
    else:
        raw = client.complete(
            prompt,
            system=INSTRUCTION_BACKTRANSLATION_SYSTEM,
        )
        payload = json.loads(raw)

    question = str(payload["question"]).strip()
    # Classic instruction back-translation: the passage *is* the answer.
    # Ignore any model-provided gold_answer so labels stay faithful to the source.
    gold_answer = paragraph.text.strip()

    return QASample(
        id=str(uuid.uuid4()),
        question=question,
        gold_answer=gold_answer,
        doc_id=paragraph.doc_id,
        para_id=paragraph.para_id,
        context=paragraph.text,
        role=paragraph.role,
        # No separate "instruction" field: the question *is* the instruction.
        instruction="",
        split=paragraph.split,
        metadata={
            "generation_strategy": (
                GenerationStrategy.INSTRUCTION_BACKTRANSLATION.value
            ),
        },
    )


def grounding_overlap_score(answer: str, passage: str) -> float:
    """Estimate lexical grounding as token overlap between answer and passage."""
    answer_tokens = set(tokenize(answer))
    passage_tokens = set(tokenize(passage))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & passage_tokens) / len(answer_tokens)
