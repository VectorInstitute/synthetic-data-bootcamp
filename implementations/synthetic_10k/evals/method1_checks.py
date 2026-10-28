"""Run Method 1's narrative checks on Method 2's fictional-company filings.

Method 1's checks live in ../method_1/validate.py and run
inside its narrative step on each generated section. This script applies the ones
that don't depend on its prompts to every filing under
../method_2/data/synthetic/{fictional_id}/{fy}/ (or --synthetic-dir):

- company_name: Items 1 and 7 name the fictional company.
- real_names: no S&P 500 company name appears (Method 2's sp500_constituents.csv, matched
  with Method 1's case-sensitive patterns, which skip names that are ordinary words).
- number_density: Items 1, 1A and 1C cite few dollar amounts, as real filings do.
- identifiers: no variable names such as `gross_margin_pct` leaked from a prompt.
- numbers: every dollar amount traces to the statements; in Item 7, every percentage
  too.
- exemplar_overlap: under 5% of 8-word runs repeat the template company's own filing.
  Runs only where Method 2's data/parsed/{ticker}/{fy}/ exists (it is git-ignored), and
  is reported as SKIP otherwise.

Not run: the heading and target-length checks, which test Method 1's prompt format (each
section starts with "Item N." and hits a length set from its calibration). Method 2's
own step09 report covers length against the real corpus.

Each `numbers` failure gets a note saying how many of its flagged figures Method 2's own
step09 check accepts (it allows more ratios and line combinations, such as total debt
or a line as a share of total assets).

Method 2's filings are single-year, so its financials.json is used for both the current
and the prior year: line changes are then zero and add no extra allowed figures.

Run from implementations/synthetic_10k (in the repo's `uv` environment):
    uv run python -m evals.method1_checks
    uv run python -m evals.method1_checks --synthetic-dir DIR --out evals/results/NAME/report.csv
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
METHOD2_DIR = EVALS_DIR.parent / "method_2"
sys.path.insert(0, str(EVALS_DIR.parent))

from method_1.validate import (  # noqa: E402
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


DEFAULT_OUT = EVALS_DIR / "results" / "method1_checks_on_method2_claude20" / "validation_report.csv"
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


def to_method1_spec(spec: dict[str, Any], financials: dict[str, Any]) -> dict[str, Any]:
    """Build the parts of Method 1's spec format that its checks read."""
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


def method2_expected(financials: dict[str, Any]) -> tuple[set[float], set[float]] | None:
    """Return Method 2's step09 allowed (dollars, percents), or None if it can't import.

    Used only to annotate failures: a figure Method 2's check accepts is a ratio or line
    combination that Method 1's check doesn't compute, not a number with no source.
    """
    try:
        from method_2.step09_validate_filings import build_expected_figures  # noqa: PLC0415
    except ImportError:
        return None
    return build_expected_figures(financials)


def explain(text: str, financials: dict[str, Any], issues: list[str]) -> str:
    """Say how many figures behind a `numbers` failure Method 2's own check accepts."""
    expected = method2_expected(financials) if issues else None
    if expected is None:
        return ""
    dollars, percents = expected
    # Each figure maps to (number, unit), read from the regex groups so "$ 12.3 million"
    # and "$1,200" parse the same way.
    figures: dict[str, tuple[float, str]] = {}
    for m in _DOLLARS.finditer(text):
        figures.setdefault(m.group(0).strip(), (float(m.group(1).replace(",", "")), (m.group(2) or "million").lower()))
    for m in _PERCENT.finditer(text):
        figures.setdefault(m.group(0).strip(), (float(m.group(1)), "%"))
    flagged = [f for f in figures if any(f in issue for issue in issues)]
    scale = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
    accepted = 0
    for figure in flagged:
        number, unit = figures[figure]
        if unit == "%":
            accepted += any(abs(number - p) <= 0.25 for p in percents)
        else:
            value = number * scale[unit]
            accepted += any(abs(value - abs(d)) <= max(1e5, 0.01 * abs(d)) for d in dollars)
    return f"{accepted} of {len(flagged)} flagged figures are accepted by Method 2's step09 check"


def load_real_names() -> list:
    """Return Method 1's name patterns for every S&P 500 company in Method 2's snapshot."""
    with open(METHOD2_DIR / "data" / "sp500_constituents.csv", encoding="utf-8") as f:
        return real_name_patterns([row["Security"] for row in csv.DictReader(f)])


def check_company(spec_path: Path, real_names: list) -> list[dict]:
    """Return one report row per check and Item for one fictional company."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    method1_spec = to_method1_spec(spec, financials)
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
            add("company_name", label, check_company_name(text, method1_spec))
        add("real_names", label, check_real_names(text, real_names))
        if label in ("1", "1A", "1C"):
            add("number_density", label, check_number_density(text, label))
        add("identifiers", label, check_identifiers(text))
        issues = check_numbers(text, method1_spec, strict=label == "7")
        add("numbers", label, issues, explain(text, financials, issues))
        exemplar = METHOD2_DIR / "data" / "parsed" / template["ticker"] / str(template["fiscal_year"]) / path.name
        if exemplar.exists():
            add("exemplar_overlap", label, check_overlap(text, exemplar.read_text(encoding="utf-8")))
        else:
            add("exemplar_overlap", label, None)
    return rows


def main() -> None:
    """Check every fictional company in Method 2 and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=METHOD2_DIR / "data" / "synthetic",
        help="folder of {fictional_id}/{fy}/ companies in Method 2's layout",
    )
    args = parser.parse_args()

    real_names = load_real_names()
    rows = []
    for spec_path in sorted(args.synthetic_dir.glob("*/*/spec.json")):
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
