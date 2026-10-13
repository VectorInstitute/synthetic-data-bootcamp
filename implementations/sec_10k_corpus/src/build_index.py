"""Build data/index.csv: one row per parsed section.

Input: data/raw/{ticker}/{fy}/metadata.json and data/parsed/{ticker}/{fy}/item_*.txt
(run src.download_10k and src.parse_sections first). Only sections that have a parsed
file get a row; data/parse_status.csv already covers what's missing and why.

Run from the project directory:
    python -m src.build_index
"""

import csv
import json

from src.config import PARSED_DIR, RAW_DIR, get_companies, load_config
from src.parse_sections import TARGET_ITEMS
from src.utils import setup_logging


logger = setup_logging("build_index")

INDEX_PATH = RAW_DIR.parent / "index.csv"
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
            },
        )
    return rows


def main() -> None:
    """Build data/index.csv for every company-year in config.yaml."""
    cfg = load_config()
    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]

    rows = []
    for ticker in tickers:
        for fy in fiscal_years:
            rows.extend(index_rows(ticker, fy))

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), INDEX_PATH)


if __name__ == "__main__":
    main()
