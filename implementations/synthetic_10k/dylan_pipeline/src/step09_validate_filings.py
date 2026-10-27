"""Phase 7: automatic validation gates for the fictional companies' narrative text.

Input: data/synthetic/{fictional_id}/{fy}/{spec.json,financials.json,item_*.txt}.
Output: data/validation_report.csv, one row per fictional_id/fy/check, plus a log
summary. Covers PLAN.md Phase 7's "automatic" gates:

- No real company name or ticker leaks into the narrative.
- Every dollar figure and percentage quoted in Item 7 and Item 8 traces back to
  financials.json, within the rounding the narrative text itself uses.
- The three financial-statement headings are present in Item 8.
- Every item file is substantive (not just a stub).

Section length against the real corpus's distribution is reported, not gated:
Phase 6 condensed every section by a documented, deliberate choice (see PLAN.md
Phase 6's "known gaps" paragraph), so gating on it would flag all 20 companies for
something that isn't a defect.

PLAN.md Phase 7's other two gates (a model-judged realism rubric on a sample, and
human review of about 10 filings) are judgment calls this script doesn't make.

Run from the project directory:
    python -m src.step09_validate_filings
"""

import csv
import io
import json
import re
from pathlib import Path

from src.config import PROJECT_DIR, RAW_DIR
from src.utils import setup_logging, write_atomic


logger = setup_logging("validate_filings")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
CONSTITUENTS_PATH = PROJECT_DIR / "data" / "sp500_constituents.csv"
PARSE_STATUS_PATH = RAW_DIR.parent / "parse_status.csv"
REPORT_PATH = RAW_DIR.parent / "validation_report.csv"

ITEMS = ["1", "1a", "1c", "7", "8"]
MIN_WORDS = 150  # Phase 6's own substantiveness floor, already verified by hand once

# Figures in Item 7/8 are rounded to roughly 4 significant figures (e.g. "$5,825.6
# million"); percentages are rounded to 1 decimal. Tolerances are a little looser
# than that rounding to absorb the narrative computing a ratio from its own rounded
# dollar figures rather than from financials.json's exact ones.
DOLLAR_RELATIVE_TOLERANCE = 0.01
DOLLAR_ABSOLUTE_FLOOR = 100_000
PERCENT_ABSOLUTE_TOLERANCE = 0.25

# Reference numbers every company's narrative is allowed to cite without it tracing
# to that company's own financials.json: 0% and 100% (trivial base-100 table rows)
# and 21% (the U.S. federal statutory corporate tax rate, cited as a comparison).
UNIVERSAL_PERCENTS = {0.0, 100.0, 21.0}

ITEM_8_REQUIRED_PATTERNS = [
    re.compile(r"Consolidated Balance Sheet", re.IGNORECASE),
    re.compile(r"Statement of (Income|Operations)", re.IGNORECASE),
    re.compile(r"Statement of Cash Flows", re.IGNORECASE),
]

UNIT_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

# A dollar figure, optionally parenthesized (negative) and optionally followed by a
# scale word (prose always states one; table rows rely on the "(in millions)"
# header instead, handled separately in parse_table_row).
PROSE_DOLLAR_RE = re.compile(
    r"\$\s?\(?-?[\d,]+(?:\.\d+)?\)?\s*(thousand|million|billion)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"(negative\s+)?\(?-?[\d,]+(?:\.\d+)?\)?%", re.IGNORECASE)


def to_float(raw: str) -> float | None:
    """Parse a figure like "$(339.7)" or "24.6%" into a signed float, or None.

    None covers "—" and similar placeholders for zero/not-applicable, which carry
    no numeric claim to check.
    """
    token = raw.strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    token = token.replace("$", "").replace("%", "").replace(",", "").strip()
    if not token or not re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return None
    value = float(token)
    return -abs(value) if negative else value


def load_real_companies() -> dict[str, str]:
    """Return {ticker: full security name} for every real S&P 500 constituent."""
    companies = {}
    with open(CONSTITUENTS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            companies[row["Symbol"].strip().upper()] = row["Security"].strip()
    return companies


def check_real_company_leaks(text: str, template_ticker: str, template_name: str) -> list[str]:
    """Return any case-sensitive ticker or case-insensitive name leak found in `text`.

    Scoped to the fictional company's own real template, not every real S&P 500
    company: that's the one real company whose exemplar text ever entered this
    company's generation, so it's the one a leak could plausibly come from.
    Ticker matching is case-sensitive and whole-word, since several real tickers
    used as templates here (BEN, ROP, DRI, FAST, BALL, USB) are also ordinary
    English words or acronyms in lowercase or mixed case ("benefit," "USB port"),
    and filings don't write a ticker symbol in running prose.
    """
    findings = []
    if re.search(rf"\b{re.escape(template_ticker)}\b", text):
        findings.append(f"ticker '{template_ticker}' appears as a standalone word")
    if re.search(re.escape(template_name), text, re.IGNORECASE):
        findings.append(f"template company name '{template_name}' appears")
    return findings


def flatten_leaves(d: dict) -> dict[str, float]:
    """Return every numeric leaf in a nested dict, keyed by its leaf name."""
    leaves = {}
    for key, value in d.items():
        if isinstance(value, dict):
            leaves.update(flatten_leaves(value))
        elif isinstance(value, (int, float)):
            leaves[key] = float(value)
    return leaves


# Specific sibling-leaf combinations actually seen stated as one figure in the
# narrative (e.g. "total debt," "goodwill and acquired intangible assets together").
# A blind all-pairs combination of every section's leaves was tried first and
# dropped: with 5-7 leaves per section, the combinatorics produce dozens of
# candidate values per company, enough that an unrelated, wrong number can land
# within tolerance of one by coincidence (confirmed with a deliberately wrong
# figure during development) -- which defeats the point of the check.
NAMED_COMBINATIONS = {
    "liabilities": [("short_term_debt", "long_term_debt")],
    "assets": [("goodwill", "intangible_assets_net")],
    "operating_expenses": [
        ("selling_general_administrative", "research_and_development"),
        ("cost_of_revenue", "research_and_development"),
    ],
}


def named_sums(section_name: str, d: dict) -> list[float]:
    """Return the sum of each of `section_name`'s NAMED_COMBINATIONS present in `d`."""
    return [d[a] + d[b] for a, b in NAMED_COMBINATIONS.get(section_name, []) if a in d and b in d]


def build_expected_figures(financials: dict) -> tuple[set[float], set[float]]:
    """Return (expected dollar values, expected percentages) derived from financials.json."""
    bs, inc, cf = financials["balance_sheet"], financials["income_statement"], financials["cash_flow_statement"]

    dollars = set(flatten_leaves(bs).values()) | set(flatten_leaves(inc).values()) | set(flatten_leaves(cf).values())
    dollars.update(named_sums("assets", bs["assets"]))
    dollars.update(named_sums("liabilities", bs["liabilities"]))
    dollars.update(named_sums("operating_expenses", inc["operating_expenses"]))

    percents = set(UNIVERSAL_PERCENTS)
    revenue = inc["revenues"]
    total_assets = bs["assets"]["total_assets"]
    pretax = inc["pretax_income"]
    equity = bs["equity"]["total_stockholders_equity"]

    def add_ratio(numerator: float | None, denominator: float) -> None:
        if numerator is not None and denominator:
            percents.add(round(numerator / denominator * 100, 6))

    add_ratio(inc["net_income"], revenue)
    add_ratio(inc.get("operating_income"), revenue)
    add_ratio(inc["net_income"], equity)
    add_ratio(inc["net_income"], total_assets)
    add_ratio(equity, total_assets)
    add_ratio(bs["liabilities"]["total_liabilities"], total_assets)
    add_ratio(inc["income_tax_expense"], pretax)
    if "cost_of_revenue" in inc["operating_expenses"]:
        gross_profit = revenue - inc["operating_expenses"]["cost_of_revenue"]
        dollars.add(gross_profit)
        add_ratio(gross_profit, revenue)

    if revenue:
        for value in list(flatten_leaves(inc).values()) + named_sums("operating_expenses", inc["operating_expenses"]):
            add_ratio(value, revenue)
    if total_assets:
        for value in list(flatten_leaves(bs["assets"]).values()) + named_sums("assets", bs["assets"]):
            add_ratio(value, total_assets)

    return dollars, percents


def parse_table_row(line: str) -> tuple[list[float], list[float]]:
    """Return (dollar values, percent values) from a "label | value[ | percent]" row.

    Bare values in these rows are in millions, per every statement table's own
    "(in millions)" header; this is the one place that scale isn't stated inline.
    """
    parts = [p.strip() for p in line.split(" | ")]
    if len(parts) < 2:
        return [], []
    dollars, percents = [], []
    value = to_float(parts[1])
    if value is not None:
        dollars.append(value * 1e6)
    if len(parts) >= 3:
        pct = to_float(parts[2])
        if pct is not None:
            percents.append(pct)
    return dollars, percents


def extract_figures(text: str) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Return every (value, source-line) dollar and percent figure found in `text`."""
    dollars, percents = [], []
    for line in text.splitlines():
        if " | " in line:
            line_dollars, line_percents = parse_table_row(line)
            dollars.extend((v, line) for v in line_dollars)
            percents.extend((v, line) for v in line_percents)

    for match in PROSE_DOLLAR_RE.finditer(text):
        # match.group(0) is e.g. "$5,825.6 million"; split off the trailing unit word.
        amount_str, unit = match.group(0).rsplit(None, 1)
        value = to_float(amount_str)
        if value is not None:
            dollars.append((value * UNIT_SCALE[unit.lower()], match.group(0)))

    for match in PERCENT_RE.finditer(text):
        said_negative = bool(match.group(1))
        numeric_part = match.group(0)[len(match.group(1)) :] if said_negative else match.group(0)
        value = to_float(numeric_part)
        if value is not None:
            percents.append((-abs(value) if said_negative else value, match.group(0)))

    return dollars, percents


def closest_match(value: float, candidates: set[float], *, relative: bool) -> bool:
    """Return True if `value` is within tolerance of any figure in `candidates`.

    Dollar figures are compared by magnitude, not sign: cash-flow-statement values
    are stored with an accounting sign (an outflow is negative), but the narrative
    routinely restates them as a positive amount in prose ("we used $X million of
    cash in investing activities"), which is a phrasing convention, not a numeric
    discrepancy to flag.
    """
    for c in candidates:
        if relative:
            if abs(abs(value) - abs(c)) <= max(DOLLAR_ABSOLUTE_FLOOR, DOLLAR_RELATIVE_TOLERANCE * abs(c)):
                return True
        elif abs(value - c) <= PERCENT_ABSOLUTE_TOLERANCE:
            return True
    return False


def real_section_stats() -> dict[str, list[int]]:
    """Return real corpus char counts per item, from parse_status.csv's ok rows."""
    stats: dict[str, list[int]] = {item: [] for item in ITEMS}
    with open(PARSE_STATUS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = row["item"].lower().removeprefix("item_")
            if item in stats and row["status"] == "ok":
                stats[item].append(int(row["chars"]))
    return stats


def validate_company(spec_path: Path, real_companies: dict[str, str], real_stats: dict[str, list[int]]) -> list[dict]:
    """Return one validation-report row per check for one fictional_id/fy."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    fictional_id, fy = spec["fictional_id"], spec["fiscal_year"]
    template_ticker = spec["template"]["ticker"]
    template_name = real_companies.get(template_ticker, template_ticker)

    rows = []

    def add(check: str, item: str, status: str, detail: str) -> None:
        rows.append({"fictional_id": fictional_id, "fiscal_year": fy, "check": check, "item": item, "status": status, "detail": detail})

    texts = {}
    for item in ITEMS:
        path = spec_path.parent / f"item_{item}.txt"
        if not path.exists():
            add("file_exists", item, "FAIL", "missing")
            continue
        text = path.read_text(encoding="utf-8")
        texts[item] = text
        word_count = len(text.split())
        add("substantive", item, "PASS" if word_count >= MIN_WORDS else "FAIL", f"{word_count} words")

        real_counts = real_stats.get(item, [])
        if real_counts:
            median = sorted(real_counts)[len(real_counts) // 2]
            add("length_vs_real", item, "INFO", f"{len(text)} chars vs real median {median} chars ({len(text) / median:.0%})")

        findings = check_real_company_leaks(text, template_ticker, template_name)
        add("real_company_leak", item, "FAIL" if findings else "PASS", "; ".join(findings) or "none found")

    if "8" in texts:
        missing_headings = [p.pattern for p in ITEM_8_REQUIRED_PATTERNS if not p.search(texts["8"])]
        add("required_headings", "8", "FAIL" if missing_headings else "PASS", "; ".join(missing_headings) or "all present")

    expected_dollars, expected_percents = build_expected_figures(financials)
    for item in ("7", "8"):
        if item not in texts:
            continue
        dollars, percents = extract_figures(texts[item])
        unmatched_dollars = [(v, ctx) for v, ctx in dollars if not closest_match(v, expected_dollars, relative=True)]
        unmatched_percents = [(v, ctx) for v, ctx in percents if not closest_match(v, expected_percents, relative=False)]
        total = len(dollars) + len(percents)
        unmatched = unmatched_dollars + unmatched_percents
        status = "PASS" if not unmatched else "FAIL"
        detail = f"{total - len(unmatched)}/{total} figures traced" if total else "no figures found"
        if unmatched:
            detail += "; unmatched: " + "; ".join(f"{v:g} in '{ctx.strip()[:80]}'" for v, ctx in unmatched[:10])
        add("numeric_consistency", item, status, detail)

    return rows


def main() -> None:
    """Run every automatic Phase 7 gate on every fictional company and write the report."""
    real_companies = load_real_companies()
    real_stats = real_section_stats()
    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if not spec_paths:
        logger.warning("No specs found under %s; run src.step06_make_specs first", SYNTHETIC_DIR)
        return

    all_rows = []
    for spec_path in spec_paths:
        all_rows.extend(validate_company(spec_path, real_companies, real_stats))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["fictional_id", "fiscal_year", "check", "item", "status", "detail"])
    writer.writeheader()
    writer.writerows(all_rows)
    write_atomic(REPORT_PATH, buffer.getvalue())

    failures = [r for r in all_rows if r["status"] == "FAIL"]
    companies_with_failures = {r["fictional_id"] for r in failures}
    logger.info(
        "Validated %d company-years, %d checks, %d failures across %d companies. Report: %s",
        len(spec_paths),
        len(all_rows),
        len(failures),
        len(companies_with_failures),
        REPORT_PATH,
    )
    for r in failures:
        logger.warning("FAIL %s FY%s %s item %s: %s", r["fictional_id"], r["fiscal_year"], r["check"], r["item"], r["detail"])


if __name__ == "__main__":
    main()
