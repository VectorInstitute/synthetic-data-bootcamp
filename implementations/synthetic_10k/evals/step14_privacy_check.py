"""Phase 11: how far is each fictional company's financials from the real company
it's templated on?

Every fictional company is calibrated from one real template (`spec.json`'s
`template.ticker`/`template.fiscal_year`), so the obvious privacy question a
real-company-name/ticker leak check (Phase 7) doesn't answer is: even without a
literal name leak, are the fictional numbers just the real company's own numbers
relabeled? If a fictional company's revenue, net income, and total assets are each
within a percent of its real template's, the "fictional" company is a thin disguise,
not a synthetic one.

This computes, per fictional company and per financial field, the percent difference
between the fictional value (`spec.json`'s `financials`) and the real template's
actual same-year value (`data/calibration.json`'s `company_years`, written once by
`step05_make_calibration` directly from XBRL, before any fictional company existed).

Output: data/privacy_report.csv, one row per (fictional company, financial field).

Run from implementations/synthetic_10k (after src.step06_make_specs):
    PYTHONPATH=dylan_pipeline:. python evals/step14_privacy_check.py
"""

import csv
import io
import json

from src.config import COMPANYFACTS_DIR, RAW_DIR
from src.utils import setup_logging, write_atomic


logger = setup_logging("privacy_check")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
CALIBRATION_PATH = COMPANYFACTS_DIR.parent / "calibration.json"
REPORT_PATH = RAW_DIR.parent / "privacy_report.csv"
REPORT_FIELDS = ["fictional_id", "fiscal_year", "template_ticker", "field", "real_value", "synthetic_value", "pct_diff"]

FIELDS = ["total_assets", "total_liabilities", "stockholders_equity", "revenues", "net_income", "operating_income"]


def load_real_company_years() -> dict[tuple[str, int], dict]:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {(r["ticker"], r["fiscal_year"]): r for r in calibration["company_years"]}


def privacy_rows(spec_path, real_company_years: dict[tuple[str, int], dict]) -> list[dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    fictional_id = spec["fictional_id"]
    template_ticker = spec["template"]["ticker"]
    template_fy = spec["template"]["fiscal_year"]
    real = real_company_years.get((template_ticker, template_fy))
    if real is None:
        raise SystemExit(f"{fictional_id}: no calibration.json record for template {template_ticker} FY{template_fy}")

    rows = []
    for field in FIELDS:
        if field not in spec["financials"] or field not in real:
            continue
        synthetic_value, real_value = spec["financials"][field], real[field]
        pct_diff = (synthetic_value - real_value) / real_value * 100 if real_value else None
        rows.append(
            {
                "fictional_id": fictional_id,
                "fiscal_year": spec["fiscal_year"],
                "template_ticker": template_ticker,
                "field": field,
                "real_value": real_value,
                "synthetic_value": synthetic_value,
                "pct_diff": round(pct_diff, 1) if pct_diff is not None else None,
            },
        )
    return rows


def main() -> None:
    """Build data/privacy_report.csv for every fictional company."""
    real_company_years = load_real_company_years()
    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))

    rows = []
    for spec_path in spec_paths:
        rows.extend(privacy_rows(spec_path, real_company_years))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(REPORT_PATH, buffer.getvalue())

    diffs = [abs(r["pct_diff"]) for r in rows if r["pct_diff"] is not None]
    closest = min(rows, key=lambda r: abs(r["pct_diff"]) if r["pct_diff"] is not None else float("inf"))
    logger.info(
        "Wrote %d rows to %s (%d companies). Median |pct diff| %.1f%%, closest single field %s/%s at %.1f%%",
        len(rows), REPORT_PATH, len(spec_paths),
        sorted(diffs)[len(diffs) // 2], closest["fictional_id"], closest["field"], closest["pct_diff"],
    )


if __name__ == "__main__":
    main()
