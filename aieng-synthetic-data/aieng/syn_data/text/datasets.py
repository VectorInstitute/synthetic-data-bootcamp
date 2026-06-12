"""Download and load domain policy documents."""

from __future__ import annotations

from pathlib import Path

import requests

from aieng.syn_data.text.config import DOCUMENTS_DIR, FINANCE_DOCUMENTS
from aieng.syn_data.text.documents import load_document_text, paragraphs_from_document
from aieng.syn_data.text.io import ensure_parent
from aieng.syn_data.text.schemas import DocumentSpec, Paragraph


def download_url(url: str, destination: Path, *, timeout: int = 60) -> Path:
    """Download a remote file to a local path."""
    ensure_parent(destination)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_document(spec: DocumentSpec, data_dir: Path | None = None) -> Path:
    """Download a document spec to the implementation data directory.

    Notes
    -----
    PDFs are saved as-is. Notebooks should convert PDFs to plain text before
    paragraph chunking, or provide pre-extracted ``.txt`` files locally.
    """
    base_dir = data_dir or DOCUMENTS_DIR
    if spec.local_path:
        destination = Path(spec.local_path)
    else:
        destination = base_dir / f"{spec.doc_id}.bin"

    if destination.exists():
        return destination

    if not spec.source_url:
        msg = f"No source URL configured for document '{spec.doc_id}'."
        raise ValueError(msg)

    return download_url(spec.source_url, destination)


def list_domain_documents(domain: str = "finance") -> list[DocumentSpec]:
    """Return configured document specs for a domain."""
    if domain == "finance":
        return list(FINANCE_DOCUMENTS)
    msg = f"Unsupported domain: {domain}"
    raise ValueError(msg)


def load_paragraphs_from_text_files(
    specs: list[DocumentSpec],
    *,
    text_suffix: str = ".txt",
) -> list[Paragraph]:
    """Load paragraphs from local plain-text files referenced by document specs."""
    paragraphs: list[Paragraph] = []
    for spec in specs:
        if not spec.local_path:
            msg = f"Document '{spec.doc_id}' is missing a local_path."
            raise ValueError(msg)
        text_path = Path(spec.local_path)
        if text_path.suffix != text_suffix:
            text_path = text_path.with_suffix(text_suffix)
        text = load_document_text(text_path)
        paragraphs.extend(paragraphs_from_document(spec, text))
    return paragraphs
