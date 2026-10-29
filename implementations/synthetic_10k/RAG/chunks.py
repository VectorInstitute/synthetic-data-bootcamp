"""Shared paragraph-level chunking of data/index.csv's item files.

Used by both RAG.step11_make_qa (to locate a gold passage for each question) and
RAG.step12_evaluate_retrievers (to build the retrieval corpus), so the two always
agree on what a "passage" is.

A chunk is one blank-line-separated block of an item file. That's also the unit
step10_build_pdfs.tableize() operates on: a financial-statement table's heading and
its rows have no blank lines between them, so a whole table section (e.g. "Assets"
plus its balance-sheet rows) falls out as one chunk -- a natural retrieval unit,
not an arbitrary fixed-size window.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from method_2.config import RAW_DIR


INDEX_PATH = RAW_DIR.parent / "index.csv"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    item: str
    source: str  # "real" or "synthetic"
    path: str
    text: str


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, dropping any empty fragments."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def load_chunks(
    index_path: Path = INDEX_PATH,
    *,
    sources: set[str] | None = None,
    tickers: set[str] | None = None,
) -> list[Chunk]:
    """Return every chunk from matching rows in data/index.csv, in file order.

    `sources`/`tickers` filter at the index-row level, before any file is read --
    the real corpus alone is ~730 rows including some 100,000+-word items, so
    filtering after reading every file (as an earlier version of this function did)
    cost minutes for callers that only need a handful of real companies (see
    RAG.step12_evaluate_retrievers's 6-company sample).
    """
    with open(index_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sources is not None:
        rows = [r for r in rows if r["source"] in sources]
    if tickers is not None:
        rows = [r for r in rows if r["ticker"] in tickers]

    chunks = []
    for row in rows:
        path = RAW_DIR.parent / row["path"]
        text = path.read_text(encoding="utf-8")
        for i, para in enumerate(chunk_text(text)):
            chunk_id = f"{row['ticker']}_{row['fiscal_year']}_{row['item']}_{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    ticker=row["ticker"],
                    company_name=row["company_name"],
                    fiscal_year=int(row["fiscal_year"]),
                    item=row["item"],
                    source=row["source"],
                    path=row["path"],
                    text=para,
                ),
            )
    return chunks
