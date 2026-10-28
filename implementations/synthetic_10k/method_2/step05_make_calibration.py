"""Build data/calibration.json: percentile distributions of real financial metrics.

Input: data/companyfacts/{ticker}.json (run method_2.step04_fetch_companyfacts first).
Output: data/calibration.json, with one record per company-fiscal_year and, per
metric, the count and percentile distribution across every company-year reporting it.

Calibration is generic across sectors (see PLAN.md Phase 3 and open question 4): the
company list is a random S&P 500 sample, not a bank list, so the metrics below are
ones most filers report, rather than bank-specific ratios like CET1 or net interest
margin. Phase 4's spec generator should sample a real company_years record as a
template and perturb it, rather than sampling each metric's percentile independently,
since independent sampling can't be trusted to keep accounting identities (assets =
liabilities + equity) intact.

Run from the project directory:
    python -m method_2.step05_make_calibration
"""

import json
import statistics
from datetime import date, datetime, timezone

from method_2.config import COMPANYFACTS_DIR, RAW_DIR, get_companies, load_config
from method_2.utils import setup_logging, write_atomic


logger = setup_logging("make_calibration")

CALIBRATION_PATH = COMPANYFACTS_DIR.parent / "calibration.json"

# Each metric tries its tags in order and uses the first with a value for that year,
# since the same line item is tagged differently across filers and accounting-standard
# transitions (e.g. ASC 606 revenue recognition).
METRIC_TAGS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
    "revenues": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
}
# name -> (numerator, denominator), skipped when either side is missing or zero
DERIVED_RATIOS = {
    "net_margin": ("net_income", "revenues"),
    "operating_margin": ("operating_income", "revenues"),
    "return_on_equity": ("net_income", "stockholders_equity"),
    "return_on_assets": ("net_income", "total_assets"),
    "equity_to_assets": ("stockholders_equity", "total_assets"),
    "liabilities_to_assets": ("total_liabilities", "total_assets"),
}
PERCENTILES = (5, 25, 50, 75, 95)


def annual_value(tag_facts: dict, year: int) -> float | None:
    """Return `tag`'s value for the fiscal year ending in `year`, from a 10-K.

    Matched on the fact's own "end" date rather than its "fy" field: "fy" names the
    filing a fact was *reported in*, not the period it covers, since every 10-K
    re-reports the prior year's figures as comparatives. A duration fact (one with a
    "start") is required to span close to a full year, so a quarter reported inside a
    10-K isn't mistaken for the annual figure. Ties (the same period re-reported in a
    later filing) are broken by the most recently filed one.
    """
    candidates = []
    for unit_values in tag_facts.get("units", {}).values():
        for entry in unit_values:
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue
            end = entry.get("end", "")
            if not end.startswith(str(year)):
                continue
            start = entry.get("start")
            if start:
                span_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                if not 330 <= span_days <= 400:
                    continue
            candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: e.get("filed", ""))
    return candidates[-1]["val"]


def company_year_metrics(companyfacts: dict, year: int) -> dict[str, float]:
    """Return every METRIC_TAGS value found for `year`, trying each tag in order."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    metrics = {}
    for metric, tags in METRIC_TAGS.items():
        for tag in tags:
            if tag not in gaap:
                continue
            value = annual_value(gaap[tag], year)
            if value is not None:
                metrics[metric] = value
                break
    return metrics


def add_derived_ratios(metrics: dict[str, float]) -> None:
    """Add each DERIVED_RATIOS entry to `metrics` in place, where it's computable."""
    for name, (numerator, denominator) in DERIVED_RATIOS.items():
        if numerator in metrics and metrics.get(denominator):
            metrics[name] = metrics[numerator] / metrics[denominator]


def distribution(values: list[float]) -> dict:
    """Return count, min, max, and the PERCENTILES cut points for `values`."""
    ordered = sorted(values)
    cuts = statistics.quantiles(ordered, n=100, method="inclusive") if len(ordered) >= 2 else None
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        **{f"p{p}": (ordered[0] if cuts is None else cuts[p - 1]) for p in PERCENTILES},
    }


def main() -> None:
    """Build calibration.json from every company's downloaded companyfacts."""
    cfg = load_config()
    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]

    company_years = []
    missing: dict[str, list[str]] = {}
    for ticker in tickers:
        facts_path = COMPANYFACTS_DIR / f"{ticker}.json"
        if not facts_path.exists():
            logger.warning("%s: no companyfacts.json (run method_2.step04_fetch_companyfacts first); skipping", ticker)
            continue
        companyfacts = json.loads(facts_path.read_text(encoding="utf-8"))
        for fy in fiscal_years:
            if not (RAW_DIR / ticker / str(fy) / "metadata.json").exists():
                continue
            metrics = company_year_metrics(companyfacts, fy)
            add_derived_ratios(metrics)
            missing_metrics = [m for m in METRIC_TAGS if m not in metrics]
            if missing_metrics:
                missing[f"{ticker} FY{fy}"] = missing_metrics
            company_years.append({"ticker": ticker, "fiscal_year": fy, **metrics})

    all_metric_names = list(METRIC_TAGS) + list(DERIVED_RATIOS)
    distributions = {
        name: distribution([row[name] for row in company_years if name in row])
        for name in all_metric_names
        if any(name in row for row in company_years)
    }

    calibration = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/companyfacts/*.json (SEC XBRL), form 10-K, generic cross-sector metrics",
        "fiscal_years": fiscal_years,
        "n_company_years": len(company_years),
        "company_years": company_years,
        "distributions": distributions,
        "missing": missing,
    }
    write_atomic(CALIBRATION_PATH, json.dumps(calibration, indent=2))
    logger.info(
        "Wrote %d company-years, %d metrics, to %s",
        len(company_years),
        len(distributions),
        CALIBRATION_PATH,
    )


if __name__ == "__main__":
    main()
