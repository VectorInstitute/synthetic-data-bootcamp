"""Run Amin's narrative checks on Dylan's fictional-company filings.

Amin's checks live in ../amin_pipeline/sec10k/validate.py and run
inside his narrative step on each generated section. This script applies the ones
that don't depend on his prompts to every filing under
../dylan_pipeline/data/synthetic/{fictional_id}/{fy}/:

- company_name: Items 1 and 7 name the fictional company.
- real_names: no S&P 500 company name appears (Dylan's sp500_constituents.csv, matched
  with Amin's case-sensitive patterns, which skip names that are ordinary words).
- number_density: Items 1, 1A and 1C cite few dollar amounts, as real filings do.
- identifiers: no variable names such as `gross_margin_pct` leaked from a prompt.
- numbers: every dollar amount traces to the statements; in Item 7, every percentage
  too.
- exemplar_overlap: under 5% of 8-word runs repeat the template company's own filing.
  Runs only where Dylan's data/parsed/{ticker}/{fy}/ exists (it is git-ignored), and
  is reported as SKIP otherwise.

Not run: the heading and target-length checks, which test Amin's prompt format (each
section starts with "Item N." and hits a length set from his calibration). Dylan's
own step09 report covers length against the real corpus.

Each `numbers` failure gets a note saying how many of its flagged figures Dylan's own
step09 check accepts (it allows more ratios and line combinations, such as total debt
or a line as a share of total assets). That needs Dylan's requirements installed.

Dylan's filings are single-year, so his financials.json is used for both the current
and the prior year: line changes are then zero and add no extra allowed figures.

Run from this folder (in the repo's `uv` environment):
    python amin_checks.py
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
DYLAN_DIR = EVALS_DIR.parent / "dylan_pipeline"
sys.path.insert(0, str(EVALS_DIR.parent / "amin_pipeline"))

from sec10k.validate import (  # noqa: E402
    _DOLLARS,
    _PERCENT,
    check_company_name,
    check_identifiers,
    check_number_density,
    check_numbers,
    check_overlap,
    check_real_names,
    real_name_patterns,
)


DEFAULT_OUT = EVALS_DIR / "results" / "amin_checks_on_dylan20" / "validation_report.csv"
ITEMS = {"1": "1", "1a": "1A", "1c": "1C", "7": "7", "8": "8"}
MILLION = 1e6


def leaves(d: dict) -> dict[str, float]:
    """Return every numeric leaf of a nested dict, in USD millions."""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(leaves(value))
        elif isinstance(value, (int, float)):
            out[key] = value / MILLION
    return out


def to_amin_spec(spec: dict[str, Any], financials: dict[str, Any]) -> dict[str, Any]:
    """Build the parts of Amin's spec format that his checks read."""
    inc = leaves(financials["income_statement"])
    revenue = inc.get("revenues", 0.0)
    income = {
        **inc,
        "revenue": revenue,
        "income_tax": inc.get("income_tax_expense", 0.0),
        "operating_income": inc.get("operating_income") or 0.0,
        "gross_profit": revenue - inc["cost_of_revenue"] if "cost_of_revenue" in inc else 0.0,
        "research_and_development": inc.get("research_and_development", 0.0),
        "selling_general_and_administrative": inc.get("selling_general_administrative", 0.0),
    }
    balance = leaves(financials["balance_sheet"])
    cur, prev = str(spec["fiscal_year"]), str(spec["fiscal_year"] - 1)
    return {
        "name": spec["fictional_name"],
        "statements": {
            "fiscal_years": [int(cur), int(prev)],
            "income_statement": {cur: income, prev: income},
            "balance_sheet": {cur: balance, prev: balance},
            "cash_flow_statement": {cur: leaves(financials["cash_flow_statement"])},
        },
    }


def dylan_expected(financials: dict[str, Any]) -> tuple[set[float], set[float]] | None:
    """Return Dylan's step09 allowed (dollars, percents), or None if it can't import.

    Used only to annotate failures: a figure Dylan's check accepts is a ratio or line
    combination that Amin's check doesn't compute, not a number with no source.
    """
    sys.path.insert(0, str(DYLAN_DIR))
    try:
        from src.step09_validate_filings import build_expected_figures  # noqa: PLC0415
    except ImportError:
        return None
    return build_expected_figures(financials)


def explain(text: str, financials: dict[str, Any], issues: list[str]) -> str:
    """Say how many figures behind a `numbers` failure Dylan's own check accepts."""
    expected = dylan_expected(financials) if issues else None
    if expected is None:
        return ""
    dollars, percents = expected
    figures = [m.group(0).strip() for m in _DOLLARS.finditer(text)] + [
        m.group(0).strip() for m in _PERCENT.finditer(text)
    ]
    flagged = [f for f in dict.fromkeys(figures) if any(f in issue for issue in issues)]
    scale = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
    accepted = 0
    for figure in flagged:
        number = float("".join(c for c in figure.split()[0] if c.isdigit() or c == "."))
        if figure.endswith("%") or figure.endswith("percent"):
            accepted += any(abs(number - p) <= 0.25 for p in percents)
        else:
            value = number * scale.get(figure.split()[-1].lower(), 1e6)
            accepted += any(abs(value - abs(d)) <= max(1e5, 0.01 * abs(d)) for d in dollars)
    return f"{accepted} of {len(flagged)} flagged figures are accepted by Dylan's step09 check"


def load_real_names() -> list:
    """Return Amin's name patterns for every S&P 500 company in Dylan's snapshot."""
    with open(DYLAN_DIR / "data" / "sp500_constituents.csv", encoding="utf-8") as f:
        return real_name_patterns([row["Security"] for row in csv.DictReader(f)])


def check_company(spec_path: Path, real_names: list) -> list[dict]:
    """Return one report row per check and Item for one fictional company."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    amin_spec = to_amin_spec(spec, financials)
    template = spec["template"]
    rows = []

    def add(check: str, item: str, issues: list[str] | None, note: str = "") -> None:
        status = "SKIP" if issues is None else ("FAIL" if issues else "PASS")
        rows.append(
            {
                "fictional_id": spec["fictional_id"],
                "fiscal_year": spec["fiscal_year"],
                "check": check,
                "item": item,
                "status": status,
                "detail": " ".join(issues or []),
                "note": note,
            },
        )

    for item, label in ITEMS.items():
        path = spec_path.parent / f"item_{item}.txt"
        if not path.exists():
            add("file_exists", label, ["missing"])
            continue
        text = path.read_text(encoding="utf-8")
        if label in ("1", "7"):
            add("company_name", label, check_company_name(text, amin_spec))
        add("real_names", label, check_real_names(text, real_names))
        if label in ("1", "1A", "1C"):
            add("number_density", label, check_number_density(text, label))
        add("identifiers", label, check_identifiers(text))
        issues = check_numbers(text, amin_spec, strict=label == "7")
        add("numbers", label, issues, explain(text, financials, issues))
        exemplar = DYLAN_DIR / "data" / "parsed" / template["ticker"] / str(template["fiscal_year"]) / path.name
        if exemplar.exists():
            add("exemplar_overlap", label, check_overlap(text, exemplar.read_text(encoding="utf-8")))
        else:
            add("exemplar_overlap", label, None)
    return rows


def main() -> None:
    """Check every fictional company in Dylan's pipeline and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    real_names = load_real_names()
    rows = []
    for spec_path in sorted((DYLAN_DIR / "data" / "synthetic").glob("*/*/spec.json")):
        rows.extend(check_company(spec_path, real_names))

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["fictional_id", "fiscal_year", "check", "item", "status", "detail", "note"],
    )
    writer.writeheader()
    writer.writerows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(buffer.getvalue(), encoding="utf-8")

    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "SKIP")}
    failed = sorted({r["fictional_id"] for r in rows if r["status"] == "FAIL"})
    print(f"{len(rows)} checks: {counts}. Companies with a failure: {len(failed)}. Report: {args.out}")
    for r in rows:
        if r["status"] == "FAIL":
            print(f"FAIL {r['fictional_id']} item {r['item']} {r['check']}: {r['detail'][:200]} [{r['note']}]")


if __name__ == "__main__":
    main()
