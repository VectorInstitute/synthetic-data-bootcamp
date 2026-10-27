"""Turn parsed real filings into per-sector distributions for fictional companies.

Usage::

    python -m sec10k.calibrate --data data/sec10k

Reads ``financials.csv`` and ``index.csv`` written by ``build`` and writes
``calibration.json``: for each sector, quantiles of company size and of the
ratios that shape a set of financial statements (margins, cost mix, asset mix,
leverage, cash conversion), plus section-length quantiles for the narrative.
Financial companies are excluded by default: their statements use a different
layout (interest income, loans, deposits).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from importlib import resources
from pathlib import Path
from statistics import quantiles
from typing import Any


logger = logging.getLogger(__name__)

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
EXCLUDED_SECTORS = ("Financials",)
# Below this many companies a sector's ratio falls back to the pooled distribution.
MIN_SECTOR_N = 8

# Ratio name -> (numerator, denominator) columns of financials.csv.
RATIOS: dict[str, tuple[str, str]] = {
    "gross_margin": ("gross_profit", "revenue"),
    "cost_of_revenue_ratio": ("cost_of_revenue", "revenue"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "rd_ratio": ("rd_expense", "revenue"),
    "sga_ratio": ("sga_expense", "revenue"),
    "da_ratio": ("depreciation_amortization", "revenue"),
    "capex_ratio": ("capex", "revenue"),
    "ocf_margin": ("operating_cash_flow", "revenue"),
    "tax_rate": ("income_tax", "pretax_income"),
    "asset_turnover": ("revenue", "total_assets"),
    "equity_ratio": ("stockholders_equity", "total_assets"),
    "current_assets_ratio": ("current_assets", "total_assets"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "cash_ratio": ("cash", "total_assets"),
    "ppe_ratio": ("ppe_net", "total_assets"),
    "goodwill_ratio": ("goodwill", "total_assets"),
    "debt_ratio": ("long_term_debt", "total_assets"),
    "inventory_ratio": ("inventory", "revenue"),
    "receivables_ratio": ("receivables", "revenue"),
    "dividend_payout": ("dividends_paid", "net_income"),
}


def load_sectors(path: Path | None = None) -> dict[str, str]:
    """Return ticker -> GICS sector (the bundled table unless ``path`` is given)."""
    source = (
        path.read_text()
        if path
        else resources.files(__package__).joinpath("sectors.csv").read_text()
    )
    return {row["ticker"]: row["sector"] for row in csv.DictReader(source.splitlines())}


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if len(values) < 3:
        return None
    cuts = quantiles(values, n=20, method="inclusive")  # 5% steps
    out = {f"p{int(q * 100)}": round(cuts[int(q * 20) - 1], 4) for q in QUANTILES}
    out["n"] = len(values)
    return out


def company_ratios(row: dict[str, str]) -> dict[str, float]:
    """Compute the calibration ratios for one filing, skipping untagged inputs."""
    out: dict[str, float] = {}
    for name, (num, den) in RATIOS.items():
        a, b = _float(row.get(num)), _float(row.get(den))
        if a is None or b is None or b <= 0:
            continue
        if name == "gross_margin" and a == 0:
            continue
        if name in ("capex_ratio", "dividend_payout"):
            a = abs(a)  # some filers tag payments as negative
        out[name] = a / b
    if "gross_margin" not in out and "cost_of_revenue_ratio" in out:
        out["gross_margin"] = 1 - out["cost_of_revenue_ratio"]
    revenue = _float(row.get("revenue"))
    if revenue and revenue > 0:
        out["log10_revenue"] = math.log10(revenue)
    return out


def build_calibration(
    financials: list[dict[str, str]],
    index: list[dict[str, str]],
    sectors: dict[str, str],
    excluded: tuple[str, ...] = EXCLUDED_SECTORS,
) -> dict[str, Any]:
    """Compute pooled and per-sector quantiles of ratios and section lengths."""
    by_sector: dict[str, list[dict[str, float]]] = {}
    tickers_used = set()
    for row in financials:
        sector = sectors.get(row["ticker"], "Unknown")
        if sector in excluded:
            continue
        ratios = company_ratios(row)
        if "log10_revenue" not in ratios:
            continue
        tickers_used.add(row["ticker"])
        by_sector.setdefault(sector, []).append(ratios)

    metrics = ["log10_revenue", *RATIOS]
    pooled_rows = [r for rows in by_sector.values() for r in rows]
    pooled = {m: _quantiles([r[m] for r in pooled_rows if m in r]) for m in metrics}

    sector_out: dict[str, Any] = {}
    for sector, rows in sorted(by_sector.items()):
        stats: dict[str, Any] = {}
        for m in metrics:
            values = [r[m] for r in rows if m in r]
            stats[m] = (
                _quantiles(values)
                if len(values) >= MIN_SECTOR_N
                else {**(pooled[m] or {}), "pooled": True}
            )
        sector_out[sector] = {"n_companies": len(rows), "metrics": stats}

    section_words: dict[str, list[float]] = {}
    for row in index:
        if row["ticker"] in tickers_used and row["status"] in ("ok", "fallback"):
            section_words.setdefault(row["item"], []).append(float(row["words"]))

    return {
        "description": "Quantiles from real 10-K filings; ratios are fractions (0.25 = 25%).",
        "excluded_sectors": list(excluded),
        "n_companies": len(pooled_rows),
        "ratio_definitions": {k: f"{a} / {b}" for k, (a, b) in RATIOS.items()},
        "pooled": pooled,
        "sectors": sector_out,
        "section_words": {
            item: _quantiles(v) for item, v in sorted(section_words.items())
        },
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/sec10k"), help="Output folder of build"
    )
    parser.add_argument(
        "--sectors",
        type=Path,
        default=None,
        help="ticker,sector CSV (default: bundled)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    with (args.data / "financials.csv").open() as f:
        financials = list(csv.DictReader(f))
    with (args.data / "index.csv").open() as f:
        index = list(csv.DictReader(f))
    calibration = build_calibration(financials, index, load_sectors(args.sectors))
    out = args.data / "calibration.json"
    out.write_text(json.dumps(calibration, indent=2))
    logger.info(
        "Wrote %s from %d companies in %d sectors",
        out,
        calibration["n_companies"],
        len(calibration["sectors"]),
    )


if __name__ == "__main__":
    main()
