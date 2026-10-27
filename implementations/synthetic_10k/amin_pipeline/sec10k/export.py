"""Assemble a synthetic 10-K into ``filing.txt`` and an EDGAR-style ``filing.html``.

Usage::

    python -m sec10k.export --data data/sec10k --companies syn0001

Reads the generated Items in ``synthetic/filings/{company_id}/item_*.txt`` and
the spec, adds the cover page, table of contents and formulaic Items
(``boilerplate.py``), renders Item 8 as formatted tables from the statements,
and writes both files next to the Items. ``narrative`` calls this after each
company; run it directly to re-render existing output. Open the HTML in a
browser and print to PDF to share it.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from sec10k.boilerplate import (
    HEADINGS,
    OUTLINE,
    cover_facts,
    template_items,
)
from sec10k.render import BALANCE_ORDER, LABELS, item_8, money
from sec10k.statements import INCOME_LINES


logger = logging.getLogger(__name__)

NOTICE = (
    "SYNTHETIC DOCUMENT. This annual report describes a fictional company. It was "
    "generated for research on synthetic data; it is not a filing with the SEC "
    "and none of its contents describe a real company or security."
)
_TOTALS = (
    "gross_profit",
    "operating_income",
    "pretax_income",
    "net_income",
    "total_current_assets",
    "total_assets",
    "total_current_liabilities",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
    "net_cash_from_operating_activities",
    "net_cash_from_investing_activities",
    "net_cash_from_financing_activities",
    "cash_end_of_year",
)
_BALANCE_SECTIONS = {
    "cash": "ASSETS",
    "total_current_liabilities": "LIABILITIES AND STOCKHOLDERS' EQUITY",
}
_CASH_SECTIONS = {
    "net_income": "OPERATING ACTIVITIES",
    "capital_expenditures": "INVESTING ACTIVITIES",
    "net_change_in_long_term_debt": "FINANCING ACTIVITIES",
}
_DOLLAR_ROWS = (
    "net_income",
    "total_assets",
    "total_liabilities_and_equity",
    "cash_end_of_year",
)
CSS = """
body { background: #fff; color: #000; font: 10pt/1.45 "Times New Roman", Times,
  serif; margin: 0; }
main { max-width: 7.5in; margin: 0 auto; padding: 0.6in 0.5in; }
.notice { border: 1px solid #999; background: #f4f4f4; padding: 8px 12px;
  font: 9pt/1.4 Arial, sans-serif; margin-bottom: 24px; }
.cover { text-align: center; }
.cover h1 { font-size: 16pt; margin: 4px 0; }
.cover .big { font-size: 22pt; font-weight: bold; margin: 18px 0 2px; }
.cover p { margin: 4px 0; }
.box { display: inline-block; width: 1em; }
table.facts { margin: 12px auto; border-collapse: collapse; }
table.facts td { padding: 2px 10px; text-align: left; vertical-align: top; }
h2.part { text-align: center; font-size: 12pt; margin: 36px 0 12px; }
h2.item { font-size: 11pt; margin: 28px 0 10px; }
h3 { font-size: 10pt; font-weight: bold; margin: 16px 0 6px; }
p { text-align: justify; margin: 0 0 9px; }
p.risk { font-weight: bold; font-style: italic; text-align: left; }
ul { margin: 0 0 9px 18px; padding: 0; }
.toc table { width: 100%; border-collapse: collapse; }
.toc td { padding: 1px 4px; }
.toc a { color: #000; text-decoration: none; }
table.fin { width: 100%; border-collapse: collapse; margin: 6px 0 22px; }
table.fin caption { font-weight: bold; text-align: center; margin-bottom: 4px; }
table.fin th { text-align: right; border-bottom: 1px solid #000; padding: 2px 6px; }
table.fin td { padding: 1px 6px; }
table.fin td.n { text-align: right; white-space: nowrap; width: 18%; }
table.fin tr.total td { font-weight: bold; }
table.fin tr.total td.n { border-top: 1px solid #000; }
table.fin tr.sec td { font-weight: bold; padding-top: 8px; }
.page { page-break-before: always; }
@media print { main { padding: 0; } }
"""


def _p(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<p{attr}>{html.escape(text)}</p>"


def _is_heading(line: str) -> bool:
    return len(line) <= 110 and not re.search(r"[.:;,]$", line) and "  " not in line


def text_to_html(text: str, item: str) -> str:
    """Turn a generated section into paragraphs, sub-headings and lists."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks and re.match(rf"item\s+{re.escape(item)}\.", blocks[0], re.I):
        blocks = blocks[1:]  # the Item heading is rendered separately
    out = []
    for i, block in enumerate(blocks):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if all(ln.startswith("- ") for ln in lines):
            items = "".join(f"<li>{html.escape(ln[2:])}</li>" for ln in lines)
            out.append(f"<ul>{items}</ul>")
        elif len(lines) == 1 and _is_heading(lines[0]):
            out.append(f"<h3>{html.escape(lines[0])}</h3>")
        elif (
            item == "1A"
            and len(lines) == 1
            and len(lines[0]) <= 320
            and lines[0].count(". ") == 0
            and i + 1 < len(blocks)
        ):
            out.append(_p(lines[0], "risk"))
        elif len(lines) > 1:  # one fact per line, such as comparison lists
            out.append("<p>" + "<br>".join(html.escape(ln) for ln in lines) + "</p>")
        else:
            out.append(_p(" ".join(lines)))
    return "\n".join(out)


def _amount(value: int, dollar: bool) -> str:
    sign = "$ " if dollar else ""
    return sign + money(value)


def _fin_table(
    caption: str,
    years: list[str],
    rows: list[tuple[str, list[int]]],
    sections: dict[str, str] | None = None,
) -> str:
    sections = sections or {}
    head = "".join(f"<th>Fiscal {y}</th>" for y in years)
    body = []
    first = True
    for key, values in rows:
        if key in sections:
            body.append(
                f'<tr class="sec"><td colspan="{len(years) + 1}">'
                f"{html.escape(sections[key])}</td></tr>"
            )
        # Net income opens the cash flow statement, so it is not a total there.
        total = key in _TOTALS and not (first and key == "net_income")
        dollar = first or key in _DOLLAR_ROWS
        cells = "".join(f'<td class="n">{_amount(v, dollar)}</td>' for v in values)
        cls = ' class="total"' if total else ""
        body.append(f"<tr{cls}><td>{html.escape(LABELS[key])}</td>{cells}</tr>")
        first = False
    return (
        f'<table class="fin"><caption>{html.escape(caption).replace(chr(10), "<br>")}<br>'
        f"<small>(in millions)</small></caption>"
        f"<tr><th></th>{head}</tr>{''.join(body)}</table>"
    )


def item_8_html(spec: dict[str, Any]) -> str:
    """Item 8 as formatted statement tables."""
    st = spec["statements"]
    years = [str(y) for y in st["fiscal_years"]]
    cur = years[0]
    income = [(k, [st["income_statement"][y][k] for y in years]) for k in INCOME_LINES]
    balance = [(k, [st["balance_sheet"][y][k] for y in years]) for k in BALANCE_ORDER]
    cash = [(k, [v]) for k, v in st["cash_flow_statement"][cur].items()]
    name = spec["name"]
    return "\n".join(
        [
            _fin_table(f"{name}\nConsolidated Statements of Operations", years, income),
            _fin_table(
                f"{name}\nConsolidated Balance Sheets",
                years,
                balance,
                _BALANCE_SECTIONS,
            ),
            _fin_table(
                f"{name}\nConsolidated Statement of Cash Flows",
                [cur],
                cash,
                _CASH_SECTIONS,
            ),
        ]
    )


def _cover_html(spec: dict[str, Any], facts: dict[str, str]) -> str:
    x, o = "&#9746;", "&#9744;"  # checked and unchecked boxes
    return f"""
<div class="cover">
<p><b>UNITED STATES<br>SECURITIES AND EXCHANGE COMMISSION</b><br>Washington, D.C. 20549</p>
<h1>FORM 10-K</h1>
<p>{x} ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</p>
<p>For the fiscal year ended {facts["fiscal_year_end"]}</p>
<p>or</p>
<p>{o} TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</p>
<p>Commission File Number: {facts["file_number"]}</p>
<p class="big">{html.escape(spec["name"])}</p>
<p>(Exact name of registrant as specified in its charter)</p>
<table class="facts"><tr><td>{facts["state"]}<br><small>(State of incorporation)</small></td>
<td>{facts["ein"]}<br><small>(I.R.S. Employer Identification No.)</small></td></tr></table>
<p>{html.escape(facts["address"])}<br><small>(Address of principal executive offices)</small></p>
<p>{facts["phone"]}<br><small>(Registrant's telephone number)</small></p>
<p>Securities registered pursuant to Section 12(b) of the Act:</p>
<table class="facts"><tr><td><u>Title of each class</u></td><td><u>Trading symbol</u></td>
<td><u>Name of each exchange on which registered</u></td></tr>
<tr><td>Common Stock, $0.01 par value</td><td>{spec["ticker"]}</td><td>{facts["exchange"]}</td></tr></table>
<p>Large accelerated filer {x} &nbsp; Accelerated filer {o} &nbsp; Non-accelerated filer {o}
&nbsp; Smaller reporting company {o} &nbsp; Emerging growth company {o}</p>
<p>Shell company (as defined in Rule 12b-2 of the Act): Yes {o} No {x}</p>
<p>The aggregate market value of the voting stock held by non-affiliates of the registrant
as of {facts["market_value_date"]} was approximately {facts["market_value"]}.
As of {facts["shares_date"]}, {facts["shares"]} shares of common stock were outstanding.</p>
<p><b>DOCUMENTS INCORPORATED BY REFERENCE</b><br>Portions of the registrant's Proxy Statement
for the {facts["proxy"]} are incorporated by reference into Part III.</p>
</div>"""


def _toc_html(present: list[str]) -> str:
    rows, part = [], ""
    for p, item, title in OUTLINE:
        if item not in present:
            continue
        if p != part:
            rows.append(f'<tr><td colspan="2"><b>{p}</b></td></tr>')
            part = p
        rows.append(
            f'<tr><td><a href="#item-{item}">Item {item}.</a></td>'
            f'<td><a href="#item-{item}">{html.escape(title)}</a></td></tr>'
        )
    return f'<div class="toc page"><h2 class="part">TABLE OF CONTENTS</h2><table>{"".join(rows)}</table></div>'


def collect_items(
    spec: dict[str, Any], folder: Path, only: Collection[str] | None = None
) -> dict[str, str]:
    """Return the generated Items in ``folder`` plus Item 8 and the templates.

    ``only`` keeps just the Items it names, e.g. ``{"1", "1A", "1C", "7", "8"}``.
    """
    items = template_items(spec)
    items["8"] = item_8(spec)
    for _, item, _title in OUTLINE:
        path = folder / f"item_{item.lower()}.txt"
        if item not in items and path.exists():
            items[item] = path.read_text()
    if only is not None:
        items = {k: v for k, v in items.items() if k in only}
    return items


def filing_text(spec: dict[str, Any], items: dict[str, str]) -> str:
    """All Items in 10-K order under a plain-text cover."""
    facts = cover_facts(spec)
    parts = [
        NOTICE,
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\n"
        "FORM 10-K\nANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE "
        f"SECURITIES EXCHANGE ACT OF 1934\nFor the fiscal year ended "
        f"{facts['fiscal_year_end']}\nCommission File Number: {facts['file_number']}"
        f"\n\n{spec['name']}\n{facts['state']} | {facts['ein']}\n{facts['address']}"
        f"\nCommon Stock, $0.01 par value | {spec['ticker']} | {facts['exchange']}",
    ]
    part = ""
    for p, item, _ in OUTLINE:
        if item not in items:
            continue
        if p != part:
            parts.append(p)
            part = p
        parts.append(items[item].strip())
    return "\n\n\n".join(parts) + "\n"


def filing_html(spec: dict[str, Any], items: dict[str, str]) -> str:
    """Render the whole filing as a self-contained HTML page."""
    facts = cover_facts(spec)
    body, part = [], ""
    for p, item, _ in OUTLINE:
        if item not in items:
            continue
        if p != part:
            body.append(f'<h2 class="part">{p}</h2>')
            part = p
        content = item_8_html(spec) if item == "8" else text_to_html(items[item], item)
        body.append(
            f'<h2 class="item" id="item-{item}">{html.escape(HEADINGS[item])}</h2>'
            f"\n{content}"
        )
    title = html.escape(f"{spec['name']} Form 10-K {spec['fiscal_year']}")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{CSS}</style></head><body><main>"
        f'<div class="notice">{html.escape(NOTICE)}</div>'
        f"{_cover_html(spec, facts)}{_toc_html(list(items))}"
        f'<div class="page">{"".join(body)}</div></main></body></html>\n'
    )


def write_filing(
    spec: dict[str, Any], folder: Path, only: Collection[str] | None = None
) -> dict[str, str]:
    """Write ``filing.txt`` and ``filing.html`` into ``folder``."""
    items = collect_items(spec, folder, only)
    (folder / "filing.txt").write_text(filing_text(spec, items))
    (folder / "filing.html").write_text(filing_html(spec, items))
    return items


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/sec10k"))
    parser.add_argument("--companies", nargs="*", help="company ids (default: all)")
    parser.add_argument(
        "--items", nargs="*", help="only these Items, e.g. 1 1A 1C 7 8 (default: all)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    synthetic = args.data / "synthetic"
    for folder in sorted((synthetic / "filings").iterdir()):
        spec_path = synthetic / "specs" / f"{folder.name}.json"
        if not spec_path.exists() or (
            args.companies and folder.name not in args.companies
        ):
            continue
        write_filing(json.loads(spec_path.read_text()), folder, args.items)
        logger.info("Wrote %s", folder / "filing.html")


if __name__ == "__main__":
    main()
