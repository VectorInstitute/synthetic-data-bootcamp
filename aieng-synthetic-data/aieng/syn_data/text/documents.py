"""Document ingestion, paragraph chunking, and train/test splitting."""

from __future__ import annotations

import random
import re
from pathlib import Path

from aieng.syn_data.text.schemas import (
    DocumentSpec,
    Paragraph,
    ParagraphSplit,
)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing space."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_into_paragraphs(
    text: str,
    *,
    min_chars: int = 120,
) -> list[str]:
    """Split document text into paragraph-sized chunks."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    raw_parts = re.split(r"\n\s*\n+", normalized)
    paragraphs: list[str] = []
    buffer = ""

    for part in raw_parts:
        candidate = part.strip()
        if not candidate:
            continue
        if len(candidate) < min_chars and buffer:
            buffer = f"{buffer}\n\n{candidate}"
            continue
        if buffer:
            paragraphs.append(buffer.strip())
            buffer = ""
        if len(candidate) < min_chars:
            buffer = candidate
        else:
            paragraphs.append(candidate)

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs


def make_paragraph_id(doc_id: str, index: int) -> str:
    """Build a stable paragraph identifier."""
    return f"{doc_id}::p{index:04d}"


def paragraphs_from_document(
    spec: DocumentSpec,
    text: str,
    *,
    min_chars: int = 120,
) -> list[Paragraph]:
    """Chunk a document into paragraph records without split assignment."""
    chunks = chunk_text_into_paragraphs(text, min_chars=min_chars)
    return [
        Paragraph(
            doc_id=spec.doc_id,
            para_id=make_paragraph_id(spec.doc_id, index),
            text=chunk,
            role=spec.role,
            split=ParagraphSplit.TRAIN,
            index=index,
        )
        for index, chunk in enumerate(chunks)
    ]


def assign_test_train_split(
    paragraphs: list[Paragraph],
    *,
    test_para_ids: set[str],
) -> list[Paragraph]:
    """Mark paragraphs as test or train based on held-out IDs."""
    updated: list[Paragraph] = []
    for paragraph in paragraphs:
        split = (
            ParagraphSplit.TEST
            if paragraph.para_id in test_para_ids
            else ParagraphSplit.TRAIN
        )
        updated.append(
            Paragraph(
                doc_id=paragraph.doc_id,
                para_id=paragraph.para_id,
                text=paragraph.text,
                role=paragraph.role,
                split=split,
                index=paragraph.index,
            ),
        )
    return updated


def sample_test_paragraphs(
    paragraphs: list[Paragraph],
    *,
    n_test: int,
    seed: int = 42,
) -> tuple[list[Paragraph], list[Paragraph]]:
    """Randomly hold out paragraphs for the evaluation set."""
    if n_test <= 0:
        return [], list(paragraphs)
    if n_test >= len(paragraphs):
        msg = (
            f"Requested {n_test} test paragraphs but only "
            f"{len(paragraphs)} are available."
        )
        raise ValueError(msg)

    rng = random.Random(seed)
    indices = list(range(len(paragraphs)))
    rng.shuffle(indices)
    test_indices = set(indices[:n_test])

    test_paras: list[Paragraph] = []
    train_paras: list[Paragraph] = []
    for index, paragraph in enumerate(paragraphs):
        split = ParagraphSplit.TEST if index in test_indices else ParagraphSplit.TRAIN
        record = Paragraph(
            doc_id=paragraph.doc_id,
            para_id=paragraph.para_id,
            text=paragraph.text,
            role=paragraph.role,
            split=split,
            index=paragraph.index,
        )
        if split is ParagraphSplit.TEST:
            test_paras.append(record)
        else:
            train_paras.append(record)
    return test_paras, train_paras


def load_document_text(path: Path) -> str:
    """Load a plain-text document from disk."""
    if not path.exists():
        msg = f"Document not found: {path}"
        raise FileNotFoundError(msg)
    return normalize_whitespace(path.read_text(encoding="utf-8"))
