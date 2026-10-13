"""Sample 50 random S&P 500 tickers for config.yaml's `companies.full` list.

Prints the result; it is not applied automatically, since config.yaml also holds
hand-maintained settings (identity placeholders, the test profile, fiscal years)
that a script shouldn't touch.

The constituent list is a frozen snapshot (data/sp500_constituents.csv, from
https://github.com/datasets/s-and-p-500-companies) rather than a live fetch, so the
sample is reproducible later even as S&P 500 membership changes. Citigroup is
excluded by name (its 10-K body has no plain-text Item headings, so it can't be
parsed by this pipeline); dotted share-class tickers (e.g. BRK.B) and duplicate
listings of the same company (by CIK, e.g. GOOGL/GOOG) are excluded so every
sampled ticker maps to one distinct, EDGAR-friendly company.

Run from the project directory:
    python -m src.step00_sample_companies
"""

import csv
import random

from src.config import PROJECT_DIR
from src.utils import setup_logging


logger = setup_logging("sample_companies")

CONSTITUENTS_PATH = PROJECT_DIR / "data" / "sp500_constituents.csv"
EXCLUDED_TICKERS = {"C"}  # Citigroup: no plain-text Item headings in its 10-K body
SAMPLE_SIZE = 50
SEED = 42


def eligible_tickers(path=CONSTITUENTS_PATH) -> list[str]:
    """Return S&P 500 tickers, one per company, excluding dotted and banned tickers."""
    seen_ciks = set()
    tickers = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["Symbol"].strip().upper()
            if "." in ticker or ticker in EXCLUDED_TICKERS:
                continue
            if row["CIK"] in seen_ciks:
                continue
            seen_ciks.add(row["CIK"])
            tickers.append(ticker)
    return sorted(tickers)


def main() -> None:
    """Sample and print 50 tickers for config.yaml."""
    pool = eligible_tickers()
    sample = sorted(random.Random(SEED).sample(pool, SAMPLE_SIZE))
    logger.info("Eligible pool: %d tickers. Sampled %d with seed %d.", len(pool), SAMPLE_SIZE, SEED)
    for ticker in sample:
        print(f"    - {ticker}")


if __name__ == "__main__":
    main()
