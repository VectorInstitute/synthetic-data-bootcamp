"""Sample fictional company specs from ``calibration.json``, with their statements.

Usage::

    python -m aieng.syn_data.sec10k.specs --data data/sec10k --n 20 --seed 0

Writes ``data/sec10k/synthetic/specs/{company_id}.json``, one file per company,
holding the spec (name, ticker, sector, size, ratios) and its reconciled
statements. Specs whose statements fail an accounting check are resampled.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

from aieng.syn_data.sec10k.calibrate import load_sectors
from aieng.syn_data.sec10k.statements import (
    build_statements,
    check_plausibility,
    check_statements,
)


logger = logging.getLogger(__name__)

QUANTILE_POINTS = (0.1, 0.25, 0.5, 0.75, 0.9)
# Ratios a generic statement layout needs; others in calibration.json are optional.
REQUIRED = [
    "gross_margin",
    "operating_margin",
    "sga_ratio",
    "da_ratio",
    "capex_ratio",
    "tax_rate",
    "asset_turnover",
    "equity_ratio",
    "current_assets_ratio",
    "current_ratio",
    "cash_ratio",
    "ppe_ratio",
    "debt_ratio",
    "receivables_ratio",
]
OPTIONAL = ["rd_ratio", "inventory_ratio", "goodwill_ratio", "dividend_payout"]
# Hard bounds that keep independently sampled ratios economically sensible.
BOUNDS = {
    "gross_margin": (0.05, 0.95),
    "operating_margin": (-0.2, 0.6),
    "tax_rate": (0.05, 0.35),
    "asset_turnover": (0.1, 4.0),
    "equity_ratio": (0.05, 0.8),
    "current_ratio": (0.4, 5.0),
    "debt_ratio": (0.0, 0.6),
    "dividend_payout": (0.0, 1.0),
}

_SYLLABLES = [
    "al",
    "ven",
    "tor",
    "ix",
    "qua",
    "mer",
    "lon",
    "dra",
    "sel",
    "ka",
    "ris",
    "nov",
    "tal",
    "zen",
    "or",
    "bri",
]
_SUFFIXES = {
    "Information Technology": ["Systems", "Technologies", "Software", "Semiconductor"],
    "Health Care": ["Therapeutics", "Health", "Medical", "Biosciences"],
    "Industrials": ["Industries", "Aerospace", "Logistics", "Machinery"],
    "Consumer Discretionary": ["Brands", "Retail", "Motors", "Leisure"],
    "Consumer Staples": ["Foods", "Consumer Products", "Beverages", "Household"],
    "Energy": ["Energy", "Petroleum", "Resources", "Midstream"],
    "Materials": ["Materials", "Chemicals", "Metals", "Minerals"],
    "Utilities": ["Power", "Electric", "Utilities", "Energy Services"],
    "Communication Services": ["Media", "Communications", "Networks", "Entertainment"],
    "Real Estate": ["Realty", "Properties", "Real Estate Trust", "Land"],
}
_STATES = [
    "DE",
    "CA",
    "TX",
    "NY",
    "IL",
    "OH",
    "GA",
    "NC",
    "MA",
    "WA",
    "CO",
    "MN",
    "PA",
    "NJ",
    "VA",
    "AZ",
]


def sample_quantile(q: dict[str, float], rng: random.Random) -> float:
    """Draw from a distribution given by its 10/25/50/75/90th percentiles.

    Uses linear interpolation between the percentiles and extends the outer
    segments linearly into the tails (to the 2nd and 98th percentiles).
    """
    xs = [q[f"p{int(p * 100)}"] for p in QUANTILE_POINTS]
    ps = [0.02, *QUANTILE_POINTS, 0.98]
    lo = xs[0] - (xs[1] - xs[0]) * (0.08 / 0.15)
    hi = xs[-1] + (xs[-1] - xs[-2]) * (0.08 / 0.15)
    xs = [lo, *xs, hi]
    u = rng.uniform(ps[0], ps[-1])
    for i in range(len(ps) - 1):
        if u <= ps[i + 1]:
            t = (u - ps[i]) / (ps[i + 1] - ps[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return xs[-1]


def _name(rng: random.Random, sector: str, taken: set[str]) -> tuple[str, str]:
    """Invent a company name and ticker that no real filer uses."""
    while True:
        word = "".join(
            rng.choice(_SYLLABLES) for _ in range(rng.choice((2, 3)))
        ).capitalize()
        ticker = (
            word[0] + "".join(rng.sample(word[1:].upper(), k=min(3, len(word) - 1)))
        ).upper()[:4]
        if word.upper() in taken or ticker in taken:
            continue
        taken.update({word.upper(), ticker})
        return f"{word} {rng.choice(_SUFFIXES.get(sector, ['Holdings']))}, Inc.", ticker


def sample_ratios(metrics: dict[str, Any], rng: random.Random) -> dict[str, float]:
    """Sample one value per ratio, then enforce cross-ratio consistency."""
    r: dict[str, float] = {}
    for name in REQUIRED + OPTIONAL:
        q = metrics.get(name)
        if not q or "p50" not in q:
            continue
        if name in OPTIONAL and q.get("pooled"):
            continue  # too few filers in the sector report it (e.g. R&D in Energy)
        value = sample_quantile(q, rng)
        lo, hi = BOUNDS.get(name, (0.0, float("inf")))
        r[name] = min(max(value, lo), hi)
    if r.get("rd_ratio", 0) < 0.005:
        r.pop("rd_ratio", None)
    # Operating costs must fit inside gross profit; keep 3% of revenue for SG&A.
    room = r["gross_margin"] - r.get("rd_ratio", 0.0)
    if room < 0.05:
        r["gross_margin"] = r.get("rd_ratio", 0.0) + 0.05
        room = 0.05
    r["operating_margin"] = min(r["operating_margin"], room - 0.03)
    r["sga_ratio"] = min(r["sga_ratio"], room - r["operating_margin"])
    # Working-capital and non-cash items between net income and operating cash flow.
    r["other_operating_cash_ratio"] = rng.gauss(0.01, 0.02)
    return {k: round(v, 4) for k, v in r.items()}


def sample_spec(
    calibration: dict[str, Any],
    rng: random.Random,
    *,
    taken: set[str],
    index: int,
    fiscal_year: int = 2024,
    sector: str | None = None,
) -> dict[str, Any]:
    """Sample one fictional company and compute its statements."""
    sectors = calibration["sectors"]
    if sector is None:
        names = sorted(sectors)
        sector = rng.choices(names, weights=[sectors[s]["n_companies"] for s in names])[
            0
        ]
    metrics = sectors[sector]["metrics"]
    name, ticker = _name(rng, sector, taken)
    revenue = round(
        10 ** sample_quantile(metrics["log10_revenue"], rng) / 1e6
    )  # $ millions
    spec = {
        "company_id": f"syn{index:04d}",
        "name": name,
        "ticker": ticker,
        "sector": sector,
        "state_of_incorporation": "DE",
        "headquarters_state": rng.choice(_STATES),
        "fiscal_year": fiscal_year,
        "fiscal_year_end": f"{fiscal_year}-12-31",
        "revenue": revenue,
        "revenue_growth": round(rng.gauss(0.06, 0.08), 4),
        "employees": max(
            200, int(round(revenue * 1e6 / rng.uniform(250_000, 900_000), -2))
        ),
        "ratios": sample_ratios(metrics, rng),
    }
    spec["statements"] = build_statements(spec)
    return spec


def generate(
    calibration: dict[str, Any], n: int, seed: int, real_names: set[str]
) -> list[dict[str, Any]]:
    """Generate ``n`` specs whose statements pass every accounting check."""
    rng = random.Random(seed)
    taken = set(real_names)
    specs: list[dict[str, Any]] = []
    attempts = 0
    while len(specs) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError(
                "Too many specs failed reconciliation; check calibration.json"
            )
        spec = sample_spec(calibration, rng, taken=taken, index=len(specs) + 1)
        errors = check_statements(spec["statements"]) + check_plausibility(
            spec["statements"]
        )
        if errors:
            logger.debug("Resampling %s: %s", spec["company_id"], errors)
            continue
        specs.append(spec)
    logger.info("Generated %d specs in %d attempts", n, attempts)
    return specs


def real_company_tokens(financials_csv: Path | None) -> set[str]:
    """Collect real tickers and name words so fictional names never reuse them."""
    tokens = {t.upper() for t in load_sectors()}
    if financials_csv and financials_csv.exists():
        with financials_csv.open() as f:
            for row in csv.DictReader(f):
                tokens.add(row["ticker"].upper())
                tokens.update(
                    w.strip(".,&").upper() for w in row["company"].split() if len(w) > 3
                )
    return tokens


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sec10k"),
        help="Folder with calibration.json",
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of fictional companies"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fiscal-year", type=int, default=2024)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    calibration = json.loads((args.data / "calibration.json").read_text())
    specs = generate(
        calibration,
        args.n,
        args.seed,
        real_company_tokens(args.data / "financials.csv"),
    )
    out_dir = args.data / "synthetic" / "specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (out_dir / f"{spec['company_id']}.json").write_text(json.dumps(spec, indent=2))
    logger.info("Wrote %d specs to %s", len(specs), out_dir)


if __name__ == "__main__":
    main()
