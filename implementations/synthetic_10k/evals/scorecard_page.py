r"""Build the dashboard's scorecard page from the per-section scorecard and RAG answers.

Reads section_scorecard's scorecard.csv (one row per set and Item, plus "all" rows for
the whole filing) and, if given, rag_answers' summary.csv files, and writes a static
page that links dashboard/scorecard.css. The best synthetic set per metric is shaded
with the scorecard's own rule (evals.section_scorecard.pick).

Run from implementations/synthetic_10k:
    uv run python -m evals.scorecard_page --scorecard evals/results/day3_scorecard/scorecard.csv \
        --rag hybrid=evals/results/day3_rag/summary.csv --rag bm25=evals/results/day3_rag_bm25/summary.csv
"""

import argparse
import contextlib
import csv
import html
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from evals.section_scorecard import ALL, ITEM_LABELS, REAL, WEIGHTS, pick  # noqa: E402


DEFAULT_OUT = ROOT / "dashboard" / "scorecard.html"
FINDINGS = HERE / "scorecard_findings.html"  # optional hand-written <li> items

SET_LABELS = {
    "m1_gemini": ("Method 1", "Gemini 3 Flash"),
    "m2_gemini": ("Method 2", "Gemini 3 Flash"),
    "m1_gpt4o": ("Method 1", "GPT-4o"),
    "m2_gpt4o": ("Method 2", "GPT-4o"),
    "m1_opus": ("Method 1", "Claude Opus 5.5"),
    "m2_opus": ("Method 2", "Claude Opus 5.5"),
    REAL: ("Real 10-Ks", "20-filing sample"),
}
DIRECTIONS = {"high": "higher is better", "real": "closer to real", "one": "closer to 1", "le_real": "at or below real"}
# Rows of each Item's table: group, metric, label, description.
METRICS = [
    ("Standard fact questions", "n_questions", "Questions", "10 per company, the same for every set"),
    ("Standard fact questions", "fact_coverage", "Facts stated", "share of questions whose answer this Item states"),
    ("Standard fact questions", "fact_recall_at_10", "Fact recall@10", "BM25 finds a stating passage in its top 10"),
    ("Standard fact questions", "fact_correct_pct", "Numbers correct %", "labelled figures equal to the statements"),
    ("Standard fact questions", "n_attached", "Labelled figures", "how many figures that check covers"),
    ("Realism", "length_ratio", "Length vs real", "median words over the real median"),
    ("Realism", "vocab_jaccard", "Vocabulary overlap", "top-1,000 terms shared with real"),
    ("Realism", "figures_per_1k", "Figures per 1k words", "number density"),
    ("Diversity and privacy", "cross_doc_overlap", "Repetition across filings", "8-word overlap within the set"),
    ("Diversity and privacy", "copied_from_real", "Copied from real", "8-word overlap with real filings"),
    ("Diversity and privacy", "self_retrieval_mrr", "Self-retrieval MRR", "can a sentence find its own passage"),
    ("Shown, not ranked", "traced_pct", "Figures traced %", "any figure matching any statement value"),
    ("Shown, not ranked", "median_words", "Median words", ""),
]
RAG_COLUMNS = [
    ("rag_answer_correct_pct", "RAG answer correct %"),
    ("retrieval_hit_pct", "Retrieval hit %"),
    ("rag_abstain_pct", "Abstained %"),
]


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV, numbers as floats where they parse."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k in ("set", "item"):
                    continue
                if v in ("", None):
                    row[k] = None
                else:
                    with contextlib.suppress(ValueError):
                        row[k] = float(v)
            rows.append(row)
    return rows


def cell(v) -> str:
    """Format a number for a table cell."""
    if v is None:
        return '<span class="na">not scored</span>'
    return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 10 else f"{v:g}"


def set_head(name: str) -> str:
    """Column header for a set."""
    method, model = SET_LABELS.get(name, (name, ""))
    cls = ' class="num real-col"' if name == REAL else ' class="num"'
    return f'<th{cls}><span class="sh-model">{html.escape(model)}</span><span class="sh-m">{html.escape(method)}</span></th>'


def chip(name: str | None) -> str:
    """A set name as a coloured chip."""
    if not name:
        return "<span class='muted'>none</span>"
    method, model = SET_LABELS.get(name, (name, ""))
    kind = "m1" if name.startswith("m1") else "m2"
    return f'<span class="chip {kind}">{html.escape(method)} · {html.escape(model)}</span>'


def item_section(item: str, rows: list[dict], sets: list[str]) -> str:
    """One Item's table, with the suggested set and per-metric best."""
    item_rows = [r for r in rows if r["item"] == item and r["set"] in sets]
    real_row = next((r for r in rows if r["item"] == item and r["set"] == REAL), None)
    suggested, best = pick(item_rows, real_row)
    shown = [*item_rows, *([real_row] if real_row else [])]
    body, group = [], None
    for g, metric, label, desc in METRICS:
        if g != group:
            body.append(f'<tr class="grp"><td colspan="{len(shown) + 1}">{html.escape(g)}</td></tr>')
            group = g
        direction = DIRECTIONS.get(WEIGHTS.get(metric, (0, ""))[1], "")
        weight = WEIGHTS.get(metric, (0, ""))[0]
        sub = " · ".join(x for x in (direction, f"weight {weight}" if weight else "", desc) if x)
        cells = []
        for r in shown:
            classes = ["num"]
            if r["set"] == REAL:
                classes.append("real-col")
            elif best.get(metric) == r["set"]:
                classes.append("best")
            cells.append(f'<td class="{" ".join(classes)}">{cell(r.get(metric))}</td>')
        body.append(
            f'<tr><th scope="row"><span class="mname">{html.escape(label)}</span><span class="mdir">{html.escape(sub)}</span></th>{"".join(cells)}</tr>',
        )
    label = ITEM_LABELS[item]
    number, _, name = label.partition(" ")[2].partition(" ")
    return f"""
<section class="item" id="item-{item}">
  <div class="item-head">
    <h2><span class="ino">Item {html.escape(number)}</span>{html.escape(name)}</h2>
    <dl class="picks"><div><dt>Suggested set</dt><dd>{chip(suggested)}</dd></div>
    <div><dt>Most numbers correct</dt><dd>{chip(best.get("fact_correct_pct"))}</dd></div>
    <div><dt>Best fact recall</dt><dd>{chip(best.get("fact_recall_at_10"))}</dd></div></dl>
  </div>
  <div class="scroll"><table class="grid">
    <thead><tr><th class="mcol">Metric</th>{"".join(set_head(r["set"]) for r in shown)}</tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table></div>
</section>"""


def whole_filing(rows: list[dict], rag: dict[str, dict[str, dict]], sets: list[str]) -> str:
    """The whole-filing table: fact questions plus each RAG run's answers."""
    order = [*sets, REAL]
    by_set = {r["set"]: r for r in rows if r["item"] == ALL}
    heads = [
        ("n_questions", "Questions"),
        ("fact_coverage", "Facts stated"),
        ("fact_recall_at_10", "Fact recall@10"),
        ("fact_correct_pct", "Numbers correct %"),
    ]
    rag_heads = [(run, col, f"{label} ({run})") for run in rag for col, label in RAG_COLUMNS]
    th = "".join(f'<th class="num">{html.escape(h)}</th>' for _, h in heads)
    th += "".join(f'<th class="num">{html.escape(h)}</th>' for _, _, h in rag_heads)
    body = []
    for name in order:
        r = by_set.get(name, {})
        method, model = SET_LABELS.get(name, (name, ""))
        cls = ' class="real-col"' if name == REAL else ""
        tds = "".join(f'<td class="num">{cell(r.get(m))}</td>' for m, _ in heads)
        tds += "".join(f'<td class="num">{cell(rag[run].get(name, {}).get(col))}</td>' for run, col, _ in rag_heads)
        body.append(f'<tr{cls}><th scope="row">{html.escape(method)} · {html.escape(model)}</th>{tds}</tr>')
    return f"""<div class="scroll"><table class="grid wide">
    <thead><tr><th class="mcol">Set</th>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>"""


def page(rows: list[dict], rag: dict[str, dict[str, dict]]) -> str:
    """The whole page."""
    sets = [s for s in SET_LABELS if s != REAL and any(r["set"] == s for r in rows)]
    findings = FINDINGS.read_text(encoding="utf-8") if FINDINGS.exists() else ""
    toc = "".join(
        f'<a href="#item-{i}">Item {ITEM_LABELS[i].split(" ")[1]} <span>{html.escape(ITEM_LABELS[i].split(" ", 2)[2])}</span></a>'
        for i in ITEM_LABELS
    )
    defs = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(desc)}.</dd></div>"
        for _, _, label, desc in METRICS
        if desc
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic 10-K Scorecard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="scorecard.css">
</head>
<body>
<div class="wrap">

<nav class="sitenav" aria-label="Dashboard pages">
  <a href="index.html">Results dashboard</a>
  <a href="scorecard.html" aria-current="page">Scorecard</a>
  <a href="scorecard_day2.html">Scorecard (Day 2)</a>
  <a href="financial_method_comparison.html">Financial method comparison</a>
  <a href="chatbot.html">Chatbot</a>
</nav>

<header>
  <span class="eyebrow">Vector Institute · Synthetic Data Bootcamp · Day 3 · Oct 29, 2026</span>
  <h1>Synthetic 10-K Scorecard</h1>
  <p class="lede">Six synthetic sets (two methods, three models) and a sample of real 10-Ks, asked the same ten financial questions per company and scored section by section.</p>
</header>
{f'<section><h2>Key findings</h2><ul class="findings">{findings}</ul></section>' if findings else ""}
<section>
  <h2>Standard fact questions and RAG answers</h2>
  <p class="muted">Every company gets the same ten questions (revenue, net income, operating income, total assets, total liabilities, equity, cash, operating cash flow, net and operating margin), with answers taken from its statements, never from the text. Facts stated and fact recall count every question, so a filing that never states a fact scores a miss. RAG answers run each question through the chatbot's retrieval and an LLM (gemini-3.1-flash-lite-preview, the same for every set) and check the number against the statements.</p>
  {whole_filing(rows, rag, sets)}
  <p class="note">Banks have no operating income, so their operating-income and operating-margin questions are dropped; real filings get nine questions, as operating cash flow isn't in the XBRL extract.</p>
</section>
<section>
  <h2>Per-section scorecard</h2>
  <p class="muted">Shaded = best synthetic set on that metric. The suggested set is the weighted rank across the ranked metrics; numbers correct and fact recall count twice. "Figures traced" is shown for reference only: any figure within tolerance of any statement value passes, so about a quarter of random numbers would.</p>
  <nav class="toc">{toc}</nav>
</section>
{"".join(item_section(i, rows, sets) for i in ITEM_LABELS)}
<section>
  <h2>Metric definitions</h2>
  <dl class="defs">{defs}</dl>
</section>
<footer><span>Built by <code>evals/scorecard_page.py</code> from <code>evals/section_scorecard.py</code> and <code>evals/rag_answers.py</code>; questions from <code>qa/fact_questions.py</code>.</span></footer>
</div>
</body>
</html>
"""


def main() -> None:
    """Write the page."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorecard", type=Path, required=True, help="section_scorecard's scorecard.csv")
    parser.add_argument("--rag", action="append", default=[], metavar="NAME=PATH", help="a rag_answers summary.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = read_csv(args.scorecard)
    rag = {}
    for spec in args.rag:
        name, path = spec.split("=", 1)
        rag[name] = {r["set"]: r for r in read_csv(Path(path))}
    args.out.write_text(page(rows, rag), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
