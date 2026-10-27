"""Build data/index.csv: one row per parsed section, real and synthetic (Phase 8).

Real rows: data/raw/{ticker}/{fy}/metadata.json and data/parsed/{ticker}/{fy}/item_*.txt
(run src.step01_download_10k, then src.step02_parse_sections). Only sections with a
parsed file get a row; data/parse_status.csv already covers what's missing and why.

Synthetic rows: data/synthetic/{fictional_id}/{fy}/{spec.json,item_*.txt} (run
src.step06_make_specs through src.step08_generate_filings first). There's no real CIK,
filing date, or accession number for a fictional company, so those three columns are
blank on synthetic rows.

Every row carries a `source` column ("real" or "synthetic") so retrieval code (Phase 9)
can select, mix, or compare the two corpora from this one index rather than needing two.
Re-run this after the synthetic corpus changes, not just after the real one.

Run from the project directory:
    python -m src.step03_build_index
"""

import csv
import json
from pathlib import Path

from src.config import PARSED_DIR, RAW_DIR, get_companies, load_config
from src.step02_parse_sections import TARGET_ITEMS
from src.utils import setup_logging


logger = setup_logging("build_index")

INDEX_PATH = RAW_DIR.parent / "index.csv"
SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
INDEX_FIELDS = [
    "ticker",
    "company_name",
    "cik",
    "fiscal_year",
    "filing_date",
    "accession_number",
    "item",
    "path",
    "chars",
    "words",
    "source",
]


def index_rows(ticker: str, fiscal_year: int) -> list[dict]:
    """Return one row per parsed item for this company-year, or none if not parsed."""
    meta_path = RAW_DIR / ticker / str(fiscal_year) / "metadata.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = []
    for item in TARGET_ITEMS:
        item_path = PARSED_DIR / ticker / str(fiscal_year) / f"item_{item.lower()}.txt"
        if not item_path.exists():
            continue
        text = item_path.read_text(encoding="utf-8")
        rows.append(
            {
                "ticker": ticker,
                "company_name": meta["company_name"],
                "cik": meta["cik"],
                "fiscal_year": fiscal_year,
                "filing_date": meta["filing_date"],
                "accession_number": meta["accession_number"],
                "item": item,
                "path": str(item_path.relative_to(RAW_DIR.parent)),
                "chars": len(text),
                "words": len(text.split()),
                "source": "real",
            },
        )
    return rows


def synthetic_index_rows(spec_path: Path) -> list[dict]:
    """Return one row per item file for one fictional company-year."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    rows = []
    for item in TARGET_ITEMS:
        item_path = spec_path.parent / f"item_{item.lower()}.txt"
        if not item_path.exists():
            continue
        text = item_path.read_text(encoding="utf-8")
        rows.append(
            {
                "ticker": spec["fictional_id"],
                "company_name": spec["fictional_name"],
                "cik": "",
                "fiscal_year": spec["fiscal_year"],
                "filing_date": "",
                "accession_number": "",
                "item": item,
                "path": str(item_path.relative_to(RAW_DIR.parent)),
                "chars": len(text),
                "words": len(text.split()),
                "source": "synthetic",
            },
        )
    return rows


def main() -> None:
    """Build data/index.csv for every company-year in config.yaml, plus every fictional company."""
    cfg = load_config()
    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]

    rows = []
    for ticker in tickers:
        for fy in fiscal_years:
            rows.extend(index_rows(ticker, fy))
    n_real = len(rows)

    for spec_path in sorted(SYNTHETIC_DIR.glob("*/*/spec.json")):
        rows.extend(synthetic_index_rows(spec_path))
    n_synthetic = len(rows) - n_real

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s (%d real, %d synthetic)", len(rows), INDEX_PATH, n_real, n_synthetic)


if __name__ == "__main__":
    main()
