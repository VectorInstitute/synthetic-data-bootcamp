"""Download SEC XBRL companyfacts for every company in config.yaml.

Output: data/companyfacts/{ticker}.json, the raw response from
https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json for that company's CIK.
This is the source data for src.step05_make_calibration (Phase 3).

The CIK comes from an already-downloaded data/raw/{ticker}/{fy}/metadata.json, so
run src.step01_download_10k first. A status row per company goes to
data/companyfacts_status.csv.

Run from the project directory:
    python -m src.step04_fetch_companyfacts
"""

import csv
import json

from edgar import set_identity
from edgar.httprequests import get_with_retry
from edgar.urls import build_company_facts_url

from src.config import COMPANYFACTS_DIR, RAW_DIR, get_companies, get_sec_identity, load_config
from src.utils import setup_logging, with_retries, write_atomic


logger = setup_logging("fetch_companyfacts")

STATUS_PATH = COMPANYFACTS_DIR.parent / "companyfacts_status.csv"
STATUS_FIELDS = ["ticker", "cik", "status", "bytes", "message"]


def company_cik(ticker: str, fiscal_years: list[int]) -> int | None:
    """Return the CIK from any already-downloaded metadata.json for `ticker`."""
    for fy in fiscal_years:
        meta_path = RAW_DIR / ticker / str(fy) / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))["cik"]
    return None


def fetch_companyfacts(cik: int, label: str) -> bytes:
    """Download one company's companyfacts JSON, retrying transient failures."""

    def _get() -> bytes:
        response = get_with_retry(build_company_facts_url(cik))
        response.raise_for_status()
        return response.content

    return with_retries(_get, f"{label} companyfacts download", logger)


def process_company(ticker: str, fiscal_years: list[int]) -> dict:
    """Download one company's companyfacts, or return a row explaining why not."""
    row = {"ticker": ticker, "cik": "", "status": "", "bytes": 0, "message": ""}
    out_path = COMPANYFACTS_DIR / f"{ticker}.json"
    if out_path.exists():
        row.update(status="exists", bytes=out_path.stat().st_size)
        return row

    cik = company_cik(ticker, fiscal_years)
    if cik is None:
        logger.warning("%s: no downloaded metadata.json to read a CIK from; skipping", ticker)
        row.update(status="no_cik", message="run src.step01_download_10k first")
        return row
    row["cik"] = cik

    try:
        content = fetch_companyfacts(cik, ticker)
    except Exception as exc:
        logger.error("%s: companyfacts download failed: %s", ticker, exc)
        row.update(status="error", message=str(exc)[:200])
        return row

    write_atomic(out_path, content)
    logger.info("%s: saved companyfacts (%.1f KB)", ticker, len(content) / 1e3)
    row.update(status="downloaded", bytes=len(content))
    return row


def write_status(rows: list[dict]) -> None:
    """Write this run's per-company status rows to STATUS_PATH."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Download companyfacts for every company in config.yaml's run_profile."""
    cfg = load_config()
    set_identity(get_sec_identity(cfg))

    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]
    logger.info("Profile '%s': %d companies", cfg["run_profile"], len(tickers))

    COMPANYFACTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [process_company(ticker, fiscal_years) for ticker in tickers]
    write_status(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    logger.info("Done. Status counts: %s", counts)
    logger.info("Status file: %s", STATUS_PATH)


if __name__ == "__main__":
    main()
