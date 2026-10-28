"""Run Method 2's Phase 7 validation gates on filings from Method 1.

Method 2's gates live in ../method_2/step09_validate_filings.py and read its
folder layout. This script converts each EDGAR-style filing.html written by Method 1's
pipeline (../method_1/export.py) into that layout and
runs its checks on it without changing them.

Output, under --out-dir (default results/method2_gates_on_method1_showcase/):

- {ticker}/{fy}/spec.json, financials.json, item_1.txt ... item_8.txt: the filing
  converted into Method 2's layout.
- validation_report.csv: one row per filing x check, same columns as step09's report
  plus `note`, which explains a failure when it is a known difference between the two
  pipelines rather than a defect in the filing.

What differs from step09 and why:

- Method 1's filings carry two fiscal years in every statement. The prior year's
  values go under `cash_flow_statement.prior_year`, so step09's dollar check accepts
  them without changing its ratio checks. Table rows are written one value per row
  ("Revenue | 15,645" and "Revenue (prior year) | 13,661") because step09 reads a
  third column as a percentage.
- Method 1's filings don't record which real company's text was the style exemplar,
  so the per-template leak check is replaced by a check for every S&P 500 company name
  in Method 2's data/sp500_constituents.csv. Tickers are not checked across all 503
  companies: many (A, ALL, IT, ON, KEY) are ordinary words.
- Unmatched figures that equal a year-over-year change (in dollars or percent) or a
  prior-year ratio are noted, since step09 was written for single-year filings and
  doesn't expect them. Anything left over is listed as "not explained".

Run from implementations/synthetic_10k, in the repo's uv environment:
    uv run python -m evals.method2_gates evals/samples/method1_showcase/*.html
"""

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag


EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR.parent))

from method_2.step09_validate_filings import (  # noqa: E402
    ITEM_8_REQUIRED_PATTERNS,
    ITEMS,
    MIN_WORDS,
    build_expected_figures,
    closest_match,
    extract_figures,
    load_real_companies,
    real_section_stats,
)
from method_2.utils import setup_logging, write_atomic  # noqa: E402


logger = setup_logging("method2_gates")

DEFAULT_OUT_DIR = EVALS_DIR / "results" / "method2_gates_on_method1_showcase"
HTML_ITEM_IDS = {"1": "item-1", "1a": "item-1A", "1c": "item-1C", "7": "item-7", "8": "item-8"}

# Method 1's statement labels whose step09 key differs from a plain slug of the label.
KEY_OVERRIDES = {
    "revenue": "revenues",
    "selling_general_and_administrative": "selling_general_administrative",
    "other_operating_expense": "other_operating_expenses",
    "income_before_income_taxes": "pretax_income",
    "provision_for_income_taxes": "income_tax_expense",
    "cash_and_cash_equivalents": "cash_and_equivalents",
    "accounts_receivable_net": "accounts_receivable",
    "property_plant_and_equipment_net": "property_plant_equipment_net",
    "purchases_of_property_and_equipment": "capital_expenditures",
    "net_cash_provided_by_operating_activities": "cash_from_operations",
    "net_cash_used_in_investing_activities": "cash_from_investing",
    "net_cash_provided_by_investing_activities": "cash_from_investing",
    "net_cash_used_in_financing_activities": "cash_from_financing",
    "net_cash_provided_by_financing_activities": "cash_from_financing",
    "net_change_in_cash_and_cash_equivalents": "net_change_in_cash",
}
OPERATING_EXPENSE_KEYS = {
    "cost_of_revenue",
    "research_and_development",
    "selling_general_administrative",
    "other_operating_expenses",
}


def slug(label: str) -> str:
    """Return the step09-style key for a statement row label."""
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return KEY_OVERRIDES.get(key, key)


def cell_value(text: str) -> float | None:
    """Parse "$ 15,645" or "(613)" (USD millions) into dollars, or None."""
    token = text.replace("$", "").replace(",", "").strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return None
    value = float(token) * 1e6
    return -abs(value) if negative else value


def read_statement(table) -> tuple[str, list[tuple[str, str, list[float | None]]]]:
    """Return (caption, [(section, label, per-year values)]) for one fin table."""
    caption = table.find("caption").get_text("\n", strip=True)
    rows, section = [], ""
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) == 1:
            section = cells[0]
            continue
        rows.append((section, cells[0], [cell_value(c) for c in cells[1:]]))
    return caption, rows


def to_financials(statements: list[tuple[str, list]]) -> dict:
    """Map the three statements of Method 1's filings onto step09's schema."""
    out = {
        "balance_sheet": {"assets": {}, "liabilities": {}, "equity": {}},
        "income_statement": {"operating_expenses": {}},
        "cash_flow_statement": {
            "operating_activities": {},
            "investing_activities": {},
            "financing_activities": {},
            "prior_year": {},
        },
    }
    bs, inc, cf = out["balance_sheet"], out["income_statement"], out["cash_flow_statement"]
    for caption, rows in statements:
        for section, label, values in rows:
            key, current = slug(label), values[0]
            prior = values[1] if len(values) > 1 else None
            if current is None:
                continue
            if prior is not None:
                cf["prior_year"][f"{key}_prior"] = prior
            if "Operations" in caption or "Income" in caption:
                target = inc["operating_expenses"] if key in OPERATING_EXPENSE_KEYS else inc
            elif "Balance Sheet" in caption:
                if section.startswith("ASSETS"):
                    target = bs["assets"]
                elif key == "total_stockholders_equity" or "equity" in key and "liabilities" not in key:
                    target = bs["equity"]
                else:
                    target = bs["liabilities"]
            else:
                group = section.split()[0].lower() if section else ""
                target = cf.get(f"{group}_activities", cf)
            target[key] = current
    return out


def to_text(element) -> str:
    """Render one Item's HTML as text, tables as `label | value` rows.

    Blocks are separated by a blank line and a table's rows stay together in one
    block, as in Method 2's filings, so RAG.chunks splits both layouts the same way.
    """
    blocks = []
    for node in element:
        name = getattr(node, "name", None)
        if name == "table":
            lines = []
            caption = node.find("caption")
            if caption:
                lines.extend(caption.get_text("\n", strip=True).splitlines())
            for _section, label, values in read_statement(node)[1] if caption else []:
                if values and values[0] is not None:
                    lines.append(f"{label} | {values[0] / 1e6:,.0f}")
                if len(values) > 1 and values[1] is not None:
                    lines.append(f"{label} (prior year) | {values[1] / 1e6:,.0f}")
            if not caption:
                lines.extend(
                    " | ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"]))
                    for tr in node.find_all("tr")
                )
            blocks.append("\n".join(line for line in lines if line))
        elif name:
            blocks.append(node.get_text(" ", strip=True))
    return "\n\n".join(block for block in blocks if block)


def item_nodes(soup: BeautifulSoup, html_id: str) -> list:
    """Return the nodes between an Item's h2 and the next Item or Part heading."""
    heading = soup.find(id=html_id)
    if heading is None:
        return []
    nodes = []
    for sib in heading.next_siblings:
        if getattr(sib, "name", None) == "h2" or (
            isinstance(sib, Tag) and sib.name == "div" and "page" in (sib.get("class") or [])
        ):
            break
        nodes.append(sib)
    return nodes


def convert(html_path: Path, out_dir: Path) -> Path:
    """Write one filing.html into out_dir/{ticker}/{fy}/; return its spec path."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    fin_tables = soup.select("table.fin")
    statements = [read_statement(t) for t in fin_tables]
    company = statements[0][0].splitlines()[0]
    fy_match = re.search(r"FY(\d{4})", html_path.name) or re.search(
        r"fiscal year ended \w+ \d+, (\d{4})",
        soup.get_text(" "),
    )
    if fy_match is None:
        raise ValueError(f"{html_path.name}: no fiscal year in the file name or text")
    fiscal_year = int(fy_match.group(1))
    facts = soup.select("table.facts")
    cells = [td.get_text(strip=True) for t in facts for td in t.find_all("td")]
    ticker = next((c for c in cells if re.fullmatch(r"[A-Z]{1,5}", c)), re.sub(r"\W", "", company)[:4].upper())

    folder = out_dir / ticker / str(fiscal_year)
    folder.mkdir(parents=True, exist_ok=True)
    financials = to_financials(statements)
    spec = {
        "fictional_id": ticker,
        "fictional_name": company,
        "fiscal_year": fiscal_year,
        "source_html": html_path.name,
        "template": None,
    }
    write_atomic(folder / "spec.json", json.dumps(spec, indent=2))
    write_atomic(folder / "financials.json", json.dumps(financials, indent=2))
    for item, html_id in HTML_ITEM_IDS.items():
        heading = soup.find(id=html_id)
        if heading is None:
            continue
        body = to_text(item_nodes(soup, html_id))
        write_atomic(folder / f"item_{item}.txt", heading.get_text(" ", strip=True) + "\n\n" + body + "\n")
    return folder / "spec.json"


def two_year_figures(financials: dict) -> tuple[set[float], set[float]]:
    """Return (dollar changes, percentages) a two-year filing cites but step09 misses.

    Dollar changes are current minus prior for each statement line. Percentages are
    each line's year-over-year change, and the prior year's line-to-revenue and
    line-to-total-assets ratios (prior-year margins and asset mix).
    """
    prior = {k[: -len("_prior")]: v for k, v in financials["cash_flow_statement"].get("prior_year", {}).items()}
    current = {}

    def walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict) and k != "prior_year":
                walk(v)
            elif isinstance(v, (int, float)):
                current[k] = v

    walk(financials)
    deltas = {current[k] - p for k, p in prior.items() if k in current}
    percents = {(current[k] / p - 1) * 100 for k, p in prior.items() if k in current and p}
    for base in ("revenues", "total_assets"):
        if prior.get(base):
            percents |= {v / prior[base] * 100 for v in prior.values()}
    if prior.get("pretax_income"):
        percents.add(prior.get("income_tax_expense", 0) / prior["pretax_income"] * 100)
    if prior.get("total_stockholders_equity"):
        percents.add(prior.get("net_income", 0) / prior["total_stockholders_equity"] * 100)
    return deltas, percents


def validate(spec_path: Path, real_companies: dict[str, str], real_stats: dict[str, list[int]]) -> list[dict]:
    """Return one report row per check, mirroring step09.validate_company."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    rows = []

    def add(check: str, item: str, status: str, detail: str, note: str = "") -> None:
        rows.append(
            {
                "fictional_id": spec["fictional_id"],
                "fiscal_year": spec["fiscal_year"],
                "check": check,
                "item": item,
                "status": status,
                "detail": detail,
                "note": note,
            },
        )

    names = {name for name in real_companies.values() if len(name) > 3}
    texts = {}
    for item in ITEMS:
        path = spec_path.parent / f"item_{item}.txt"
        if not path.exists():
            add("file_exists", item, "FAIL", "missing")
            continue
        texts[item] = text = path.read_text(encoding="utf-8")
        words = len(text.split())
        add("substantive", item, "PASS" if words >= MIN_WORDS else "FAIL", f"{words} words")
        counts = sorted(real_stats.get(item, []))
        if counts:
            median = counts[len(counts) // 2]
            add(
                "length_vs_real",
                item,
                "INFO",
                f"{len(text)} chars vs real median {median} chars ({len(text) / median:.0%})",
            )
        hits = sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", text))
        add(
            "real_company_leak",
            item,
            "FAIL" if hits else "PASS",
            "; ".join(hits) or "none of 503 S&P 500 names found",
            "checks every S&P 500 name, case-sensitive so phrases like 'waste management' don't match; the exemplar company is not recorded in Method 1's filings",
        )

    if "8" in texts:
        missing = [p.pattern for p in ITEM_8_REQUIRED_PATTERNS if not p.search(texts["8"])]
        plural_only = [
            m
            for m in missing
            if m.startswith("Statement of")
            and re.search(r"Statements of (Income|Operations)", texts["8"], re.IGNORECASE)
        ]
        note = (
            "filing uses the plural 'Statements of Operations', which step09's pattern doesn't accept"
            if plural_only and len(plural_only) == len(missing)
            else ""
        )
        add("required_headings", "8", "FAIL" if missing else "PASS", "; ".join(missing) or "all present", note)

    dollars_expected, percents_expected = build_expected_figures(financials)
    deltas, two_year_percents = two_year_figures(financials)
    for item in ("7", "8"):
        if item not in texts:
            continue
        dollars, percents = extract_figures(texts[item])
        bad_dollars = [(v, c) for v, c in dollars if not closest_match(v, dollars_expected, relative=True)]
        bad_percents = [(v, c) for v, c in percents if not closest_match(v, percents_expected, relative=False)]
        unmatched = bad_dollars + bad_percents
        total = len(dollars) + len(percents)
        detail = f"{total - len(unmatched)}/{total} figures traced" if total else "no figures found"
        if unmatched:
            detail += "; unmatched: " + "; ".join(f"{v:g} in '{c.strip()[:80]}'" for v, c in unmatched[:15])
        unexplained = [(v, c) for v, c in bad_dollars if not closest_match(v, deltas, relative=True)]
        unexplained += [(v, c) for v, c in bad_percents if not closest_match(v, two_year_percents, relative=False)]
        note = ""
        if unmatched:
            note = (
                f"{len(unmatched) - len(unexplained)} of {len(unmatched)} unmatched figures are year-over-year "
                "changes or prior-year ratios, which step09 doesn't expect"
            )
            if unexplained:
                note += "; not explained: " + "; ".join(f"{v:g} in '{c.strip()[:60]}'" for v, c in unexplained)
        add("numeric_consistency", item, "PASS" if not unmatched else "FAIL", detail, note)
    return rows


def main() -> None:
    """Convert each filing.html, run the Phase 7 gates, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", nargs="+", type=Path, help="filing.html files written by Method 1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    real_companies = load_real_companies()
    real_stats = real_section_stats()
    all_rows = []
    for html_path in args.html:
        spec_path = convert(html_path, args.out_dir)
        all_rows.extend(validate(spec_path, real_companies, real_stats))

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["fictional_id", "fiscal_year", "check", "item", "status", "detail", "note"],
    )
    writer.writeheader()
    writer.writerows(all_rows)
    report = args.out_dir / "validation_report.csv"
    write_atomic(report, buffer.getvalue())

    failures = [r for r in all_rows if r["status"] == "FAIL"]
    logger.info(
        "Validated %d filings, %d checks, %d failures. Report: %s",
        len(args.html),
        len(all_rows),
        len(failures),
        report,
    )
    for r in failures:
        logger.warning(
            "FAIL %s FY%s %s item %s: %s %s",
            r["fictional_id"],
            r["fiscal_year"],
            r["check"],
            r["item"],
            r["detail"][:300],
            f"[{r['note']}]" if r["note"] else "",
        )


if __name__ == "__main__":
    main()
