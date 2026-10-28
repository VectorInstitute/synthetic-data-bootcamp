"""Download 10-K filings from SEC EDGAR into data/raw/{ticker}/{fiscal_year}/.

Each company-year folder contains:
    10k.htm        the primary 10-K exactly as filed (opens in a browser)
    10k.txt        readable plain text converted from 10k.htm
    ex13_*.htm     annual report exhibit(s), only for filings that include one
    metadata.json  ticker, CIK, fiscal year, filing date, accession number, file list

metadata.json is written last, so a folder counts as complete only when it exists.
Missing company-years are logged and skipped. A status row is written for every
company-year to data/download_status.csv.

Run from the project directory:
    python -m method_2.step01_download_10k
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from edgar import Company, set_identity
from edgar.exceptions import CompanyNotFoundError
from edgar.httprequests import get_with_retry

from method_2.config import RAW_DIR, get_companies, get_sec_identity, load_config
from method_2.html_to_text import html_to_text
from method_2.utils import setup_logging, with_retries, write_atomic


logger = setup_logging("download_10k")

FORM = "10-K"
STATUS_PATH = RAW_DIR.parent / "download_status.csv"
STATUS_FIELDS = ["ticker", "fiscal_year", "status", "accession_number", "filing_date", "message"]


def find_10k(company: Company, fiscal_year: int):
    """Return the original 10-K whose period of report falls in fiscal_year, or None."""
    # A 10-K is filed after year-end, so search the fiscal year and the following year
    filings = company.get_filings(form=FORM, amendments=False, year=[fiscal_year, fiscal_year + 1])
    if not filings:
        return None
    matches = [f for f in filings if str(f.period_of_report).startswith(str(fiscal_year))]
    if not matches:
        return None
    # If several match (rare), keep the most recently filed one
    return max(matches, key=lambda f: str(f.filing_date))


def fetch_bytes(url: str, description: str) -> bytes:
    """Download a SEC file with retries. Raises if SEC returns an error status."""

    def _get() -> bytes:
        response = get_with_retry(url)
        response.raise_for_status()
        return response.content

    return with_retries(_get, description, logger)


def is_complete(out_dir: Path) -> bool:
    """Return True if the folder has its metadata file, which is written last."""
    return (out_dir / "metadata.json").exists()


def save_filing(company: Company, filing, ticker: str, fiscal_year: int, out_dir: Path) -> dict:
    """Save one 10-K as HTML and text, plus any EX-13 exhibits, then write metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{ticker} FY{fiscal_year}"

    # filing.document is an Attachment object, so use its filename and URL
    primary = filing.document
    if primary is None or not primary.url:
        raise ValueError("filing has no primary 10-K document")
    html = fetch_bytes(primary.url, f"{label} 10-K download")
    write_atomic(out_dir / "10k.htm", html)
    text = html_to_text(html)
    write_atomic(out_dir / "10k.txt", text)

    # Many regional banks put MD&A and financials in an EX-13 annual report exhibit
    ex13_files = []
    for att in filing.attachments:
        if (att.document_type or "").startswith("EX-13"):
            suffix = Path(att.document or "").suffix or ".htm"
            name = f"ex13_{att.sequence_number}{suffix}"
            write_atomic(out_dir / name, fetch_bytes(att.url, f"{label} {name} download"))
            ex13_files.append(name)

    meta = {
        "ticker": ticker,
        "company_name": company.name,
        "cik": int(company.cik),
        "fiscal_year": fiscal_year,
        "form": filing.form,
        "period_of_report": str(filing.period_of_report),
        "filing_date": str(filing.filing_date),
        "accession_number": filing.accession_no,
        "primary_document": primary.document,
        "primary_document_url": primary.url,
        "filing_url": filing.homepage_url,
        "files": ["10k.htm", "10k.txt", *ex13_files],
        "ex13_files": ex13_files,
        "bytes_10k_htm": len(html),
        "chars_10k_txt": len(text),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_atomic(out_dir / "metadata.json", json.dumps(meta, indent=2))
    return meta


def load_company(ticker: str) -> tuple[Company | None, str, str]:
    """Look up a ticker. Returns (company, status, message); status is "" on success."""
    try:
        return with_retries(lambda: Company(ticker), f"{ticker} company lookup", logger), "", ""
    except CompanyNotFoundError:
        logger.error("%s: not found on EDGAR; skipping all its years", ticker)
        return None, "company_not_found", "ticker not found on EDGAR"
    except Exception as exc:
        logger.error("%s: company lookup failed; skipping all its years: %s", ticker, exc)
        return None, "error", str(exc)[:200]


def process_company_year(
    ticker: str,
    company: Company | None,
    company_error: tuple[str, str],
    fiscal_year: int,
) -> dict:
    """Download one 10-K, or return a status row explaining why it was skipped."""
    row = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "status": "",
        "accession_number": "",
        "filing_date": "",
        "message": "",
    }
    out_dir = RAW_DIR / ticker / str(fiscal_year)

    if is_complete(out_dir):
        meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        logger.info("%s FY%d: already downloaded, skipping", ticker, fiscal_year)
        row.update(status="exists", accession_number=meta["accession_number"], filing_date=meta["filing_date"])
        return row

    if company is None:
        status, message = company_error
        row.update(status=status, message=message)
        return row

    try:
        filing = find_10k(company, fiscal_year)
    except Exception as exc:
        logger.error("%s FY%d: 10-K search failed: %s", ticker, fiscal_year, exc)
        row.update(status="error", message=f"search failed: {str(exc)[:150]}")
        return row

    if filing is None:
        logger.warning("%s FY%d: no 10-K found on EDGAR; skipping", ticker, fiscal_year)
        row.update(status="not_found", message="no 10-K with this fiscal year")
        return row

    try:
        meta = save_filing(company, filing, ticker, fiscal_year, out_dir)
    except Exception as exc:
        logger.error("%s FY%d: download failed after retries: %s", ticker, fiscal_year, exc)
        row.update(status="error", message=f"download failed: {str(exc)[:150]}")
        return row

    logger.info(
        "%s FY%d: saved 10-K filed %s (%s, %.1f MB HTML, %d EX-13 file(s))",
        ticker,
        fiscal_year,
        meta["filing_date"],
        meta["accession_number"],
        meta["bytes_10k_htm"] / 1e6,
        len(meta["ex13_files"]),
    )
    row.update(status="downloaded", accession_number=meta["accession_number"], filing_date=meta["filing_date"])
    return row


def write_status(rows: list[dict]) -> None:
    """Write this run's per-company-year status rows to STATUS_PATH."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Download 10-Ks for the profile and fiscal years set in config.yaml."""
    cfg = load_config()
    set_identity(get_sec_identity(cfg))

    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]
    logger.info(
        "Profile '%s': %d companies x %d fiscal years",
        cfg["run_profile"],
        len(tickers),
        len(fiscal_years),
    )

    rows = []
    for ticker in tickers:
        company, status, message = load_company(ticker)
        for fy in fiscal_years:
            rows.append(process_company_year(ticker, company, (status, message), fy))

    write_status(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    logger.info("Done. Status counts: %s", counts)
    logger.info("Status file: %s", STATUS_PATH)


if __name__ == "__main__":
    main()
