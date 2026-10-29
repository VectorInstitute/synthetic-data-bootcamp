r"""Score each 10-K section separately, so a method can be chosen per section.

For every synthetic set (a folder of filings from either method) and every Item
(1, 1A, 1C, 7, 8) this writes one row of metrics, next to the same metrics on real
10-Ks. It needs no API calls: retrieval is BM25 only, so it runs anywhere.

Metrics, with the direction that counts as better:

Figure accuracy (text-to-table consistency)
- fact_correct_pct (higher): of the figures the Item attaches to one of the ten
  standard facts' labels ("revenues of $X", "net margin was Y%", a "Total assets"
  table row), the share equal to that fact's current or prior-year value or their
  change (qa.fact_questions.correctness). Label-aware, so a figure can't pass by
  matching some unrelated number. Blank under MIN_ATTACHED attached figures.
- traced_pct (shown, not ranked): share of dollar amounts and percentages in the Item that trace
  to the company's statements or a value derived from them (Method 2's step09 rules,
  plus year-over-year changes and prior-year ratios for two-year filings, as in
  method2_gates). Real filings are traced the same way against their own XBRL
  headline figures (Method 1's financials.csv), so the real column says how much of
  a real filing's numbers come from its headline statements; segment, quarterly and
  other untagged figures can't trace there, so real stays well under 100%. Left
  blank when the Item has fewer than
  MIN_FIGURES_PER_DOC figures per filing on average, since a share of almost nothing
  says nothing (a set that writes no figures would otherwise score 100%).

Retrieval utility (what a RAG benchmark built on the set measures)
- fact_recall_at_10 (higher): the shared bank of ten standard fact questions per
  company (qa.fact_questions: revenue, net income, operating income, total assets,
  total liabilities, equity, cash, operating cash flow, net and operating margin),
  asked of every set the same way. BM25 over this Item's chunks; a question counts
  as a hit when a chunk stating its answer is in the top 10. Every question counts,
  so the denominator is fixed (10 x companies) and a set that never states a fact
  scores a miss on it. Real filings get the same questions from their XBRL figures
  (operating cash flow isn't in the XBRL extract, so nine per filing).
- fact_coverage (shown, not ranked): share of those questions whose answer is
  stated anywhere in this Item; recall@10 can't exceed it. The "all" rows score the
  whole filing.
- self_retrieval_mrr (closer to real): a sentence from each chunk, with the company's
  name removed, is the query; MRR of finding its own chunk among all chunks of this
  Item. Low means passages repeat across companies; real filings set the target.

Realism (fidelity)
- length_ratio (closer to 1): median words over the real median for the Item.
- vocab_jaccard (higher): overlap of the Item's 1,000 most frequent terms with real.
- figures_per_1k (closer to real): dollar and percent figures per 1,000 words.

Diversity and privacy
- cross_doc_overlap (at or below real): median share of a filing's 8-word sequences
  that also appear in the same Item of another filing in the set.
- copied_from_real (at or below real): median share of 8-word sequences found in the
  same Item of the real corpus. Real column: a held-out sample against the others.

The summary picks a suggested method per Item by a weighted rank (see WEIGHTS):
fact_correct_pct and fact_recall_at_10 count twice. traced_pct is no longer ranked:
any percentage within 0.25 points of ~100 allowed values passes, so a random
percentage traces about a quarter of the time, and it ignores what the figure is
labelled as. Comparisons for self-retrieval
and cross-document overlap use a real sample of the same size as each set, since
both depend on corpus size.

Output, under --out (default evals/results/section_scorecard/): scorecard.csv (one
row per set and Item, plus the real rows), scorecard.md (one table per Item and the
suggested pick).

Run from implementations/synthetic_10k, in the repo's uv environment:
    uv run python -m evals.section_scorecard
    uv run python -m evals.section_scorecard \
        --set m1_gemini=data/gemini20/method_1/synthetic/filings \
        --set m2_gemini=data/gemini20/method_2/synthetic --out evals/results/NAME
"""

import argparse
import csv
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from method_1.financials import DURATION_METRICS, INSTANT_METRICS  # noqa: E402
from method_2.step09_validate_filings import (  # noqa: E402
    UNIVERSAL_PERCENTS,
    build_expected_figures,
    closest_match,
    extract_figures,
)
from method_2.utils import setup_logging, write_atomic  # noqa: E402
from qa.fact_questions import (  # noqa: E402
    correctness,
    make_questions,
    real_facts,
    synthetic_facts,
    synthetic_prior_facts,
)
from qa.fact_questions import score as fact_score  # noqa: E402
from RAG.chunks import Chunk  # noqa: E402
from RAG.retrieval import Bm25Retriever, reciprocal_rank, tokenize  # noqa: E402

from evals.compare_sets import (  # noqa: E402
    DEFAULT_REAL_DIR,
    DEFAULT_REAL_FINANCIALS,
    DEFAULT_SETS,
    ITEMS,
    Company,
    RealFiling,
    ShingleHasher,
    company_chunks,
    item_texts,
    load_real_filings,
    load_real_records,
    load_set,
)
from evals.method2_gates import two_year_figures  # noqa: E402


logger = setup_logging("section_scorecard")

DEFAULT_OUT = HERE / "results" / "section_scorecard"
TOP_TERMS = 1000
QUERIES_PER_DOC = 5
MIN_QUERY_WORDS = 12
MIN_FIGURES_PER_DOC = 1  # below this average, traced_pct is not scored
MIN_ATTACHED = 5  # below this many label-attached figures in an Item, fact_correct_pct is not scored
REAL = "real"
# metric -> (weight, direction): "high" = higher is better, "real" = closer to the real
# value is better, "le_real" = at or below the real value is best, then closer to it.
WEIGHTS = {
    "fact_correct_pct": (2, "high"),
    "fact_recall_at_10": (2, "high"),
    "self_retrieval_mrr": (1, "real"),
    "length_ratio": (1, "one"),
    "vocab_jaccard": (1, "high"),
    "figures_per_1k": (1, "real"),
    "cross_doc_overlap": (1, "le_real"),
    "copied_from_real": (1, "le_real"),
}
INFO = ["n_docs", "median_words", "n_figures", "n_questions", "fact_coverage", "n_attached", "traced_pct"]
COLUMNS = ["set", "item", *INFO, *WEIGHTS]
ALL = "all"  # item value of the whole-filing fact-question rows
ITEM_LABELS = {
    "1": "Item 1 Business",
    "1a": "Item 1A Risk Factors",
    "1c": "Item 1C Cybersecurity",
    "7": "Item 7 MD&A",
    "8": "Item 8 Financial Statements",
}


def median(values: list[float]) -> float | None:
    """Return the median, or None for no values."""
    return statistics.median(values) if values else None


def figure_count(text: str) -> int:
    """Return how many dollar and percent figures `text` cites."""
    dollars, percents = extract_figures(text)
    return len(dollars) + len(percents)


def traced_share(text: str, financials: dict) -> tuple[int, int]:
    """Return (traced, total) figures in `text`, counting two-year figures as traced."""
    dollars_expected, percents_expected = build_expected_figures(financials)
    deltas, yoy_percents = two_year_figures(financials)
    dollars, percents = extract_figures(text)
    traced = sum(
        closest_match(v, dollars_expected, relative=True) or closest_match(v, deltas, relative=True) for v, _ in dollars
    )
    traced += sum(
        closest_match(v, percents_expected, relative=False) or closest_match(v, yoy_percents, relative=False)
        for v, _ in percents
    )
    return traced, len(dollars) + len(percents)


# XBRL metrics that aren't dollar amounts, or are so small that the dollar tolerance's
# absolute floor would let almost any figure match them.
REAL_NON_DOLLAR_METRICS = {"eps_diluted", "employees"}
REAL_REVENUE_RATIO_METRICS = [
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "pretax_income",
    "income_tax",
    "net_income",
    "rd_expense",
    "sga_expense",
    "depreciation_amortization",
    "operating_cash_flow",
    "capex",
    "dividends_paid",
]
REAL_ASSET_RATIO_METRICS = [
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "stockholders_equity",
    "cash",
    "ppe_net",
    "goodwill",
    "long_term_debt",
    "inventory",
    "receivables",
]


def load_real_metrics(financials_csv: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Return every XBRL metric in Method 1's financials.csv, per (ticker, fiscal year)."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    if not financials_csv.exists():
        logger.warning("No %s; real filings get no figures traced", financials_csv)
        return out
    with open(financials_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics = {}
            for m in [*DURATION_METRICS, *INSTANT_METRICS]:
                try:
                    metrics[m] = float(row[m])
                except (KeyError, TypeError, ValueError):
                    continue
            out[(row["ticker"], int(row["fiscal_year"]))] = metrics
    return out


def real_expected_figures(current: dict[str, float], prior: dict[str, float]) -> tuple[set[float], set[float]]:
    """Return (dollars, percents) a real filing can cite from its XBRL headline figures.

    The same kinds of values as a synthetic filing's: every line, gross profit, lines
    as a share of revenue and of total assets, the tax rate, ROE and ROA, and, when
    the prior fiscal year's row exists, the year-over-year changes and prior ratios.
    """
    dollars: set[float] = set()
    percents = set(UNIVERSAL_PERCENTS)
    for year in (current, prior):
        values = {k: v for k, v in year.items() if k not in REAL_NON_DOLLAR_METRICS}
        dollars |= set(values.values())
        revenue, assets = values.get("revenue"), values.get("total_assets")
        if revenue and "cost_of_revenue" in values and "gross_profit" not in values:
            values["gross_profit"] = revenue - values["cost_of_revenue"]
            dollars.add(values["gross_profit"])
        if revenue:
            percents |= {values[m] / revenue * 100 for m in REAL_REVENUE_RATIO_METRICS if m in values}
        if assets:
            percents |= {values[m] / assets * 100 for m in REAL_ASSET_RATIO_METRICS if m in values}
            if "net_income" in values:
                percents.add(values["net_income"] / assets * 100)
        if values.get("pretax_income") and "income_tax" in values:
            percents.add(values["income_tax"] / values["pretax_income"] * 100)
        if values.get("stockholders_equity") and "net_income" in values:
            percents.add(values["net_income"] / values["stockholders_equity"] * 100)
    for k, p in prior.items():
        if k in current and k not in REAL_NON_DOLLAR_METRICS:
            dollars.add(current[k] - p)
            if p:
                percents.add((current[k] / p - 1) * 100)
    return dollars, percents


def real_traced_share(text: str, expected: tuple[set[float], set[float]]) -> tuple[int, int]:
    """Return (traced, total) figures in a real filing's `text`."""
    dollars_expected, percents_expected = expected
    dollars, percents = extract_figures(text)
    traced = sum(closest_match(v, dollars_expected, relative=True) for v, _ in dollars)
    traced += sum(closest_match(v, percents_expected, relative=False) for v, _ in percents)
    return traced, len(dollars) + len(percents)


def top_terms(texts: list[str]) -> set[str]:
    """Return the most frequent terms across `texts`."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(tokenize(t))
    return {t for t, _ in counts.most_common(TOP_TERMS)}


def query_sentence(text: str, name_tokens: set[str], rng: random.Random) -> str | None:
    """One sentence of a chunk with the company's name removed, or None if too short."""
    sentences = [s for s in re.split(r"(?<=[.;:])\s+", text) if len(s.split()) >= MIN_QUERY_WORDS]
    if not sentences:
        return None
    words = [w for w in rng.choice(sentences).split() if re.sub(r"\W", "", w).lower() not in name_tokens]
    return " ".join(words)


def self_retrieval_mrr(docs: list[tuple[str, str, str]], seed: int) -> float | None:
    """Return the MRR of finding a chunk from one of its own sentences."""
    chunks = [
        (f"{doc_id}_{i}", name, para)
        for doc_id, name, text in docs
        for i, para in enumerate(text.split("\n\n"))
        if para.strip()
    ]
    if len(chunks) < 2:
        return None
    bm25 = Bm25Retriever([c[0] for c in chunks], [c[2] for c in chunks])
    rng = random.Random(seed)
    scores = []
    by_doc: dict[str, list[tuple[str, str, str]]] = {}
    for c in chunks:
        by_doc.setdefault(c[0].rsplit("_", 1)[0], []).append(c)
    for doc_chunks in by_doc.values():
        for chunk_id, name, text in rng.sample(doc_chunks, min(QUERIES_PER_DOC, len(doc_chunks))):
            query = query_sentence(text, {w.lower() for w in tokenize(name)}, rng)
            if query:
                scores.append(reciprocal_rank(bm25.rank(query), {chunk_id}))
    return round(statistics.mean(scores), 3) if scores else None


def cross_doc_overlap(texts: list[str]) -> float | None:
    """Median share of each text's 8-grams found in another text of the same group."""
    if len(texts) < 2:
        return None
    hasher = ShingleHasher()
    hashes = [np.unique(hasher.hashes(tokenize(t))) for t in texts]
    counts = Counter(h for hs in hashes for h in hs.tolist())
    shares = [float(np.mean([counts[h] > 1 for h in hs.tolist()])) for hs in hashes if len(hs)]
    return round(statistics.median(shares), 3) if shares else None


class RealItems:
    """The real corpus per Item: lengths, terms, figure density and an 8-gram index."""

    def __init__(self, filings: list[RealFiling], holdout: set[tuple[str, int]]) -> None:
        self.hasher = ShingleHasher()
        self.texts: dict[str, dict[tuple[str, int], str]] = {item: {} for item in ITEMS}
        for f in filings:
            for item, path in f.paths.items():
                self.texts[item][(f.ticker, f.fiscal_year)] = path.read_text(encoding="utf-8", errors="replace")
        self.index, self.other_index, self.holdout_hashes = {}, {}, {}
        for item, by_key in self.texts.items():
            all_h, other_h = [], []
            for key, text in by_key.items():
                h = self.hasher.hashes(tokenize(text))
                all_h.append(h)
                if key in holdout:
                    self.holdout_hashes.setdefault(item, []).append(h)
                else:
                    other_h.append(h)
            self.index[item] = np.unique(np.concatenate(all_h)) if all_h else np.zeros(0, dtype=np.uint64)
            self.other_index[item] = np.unique(np.concatenate(other_h)) if other_h else np.zeros(0, dtype=np.uint64)
        self.terms = {item: top_terms(list(by_key.values())) for item, by_key in self.texts.items()}
        self.median_words = {
            item: median([len(t.split()) for t in by_key.values()]) for item, by_key in self.texts.items()
        }
        self.figures_per_1k = {
            item: median([figure_count(t) / max(len(t.split()), 1) * 1000 for t in by_key.values()])
            for item, by_key in self.texts.items()
        }
        logger.info("Real corpus: %s filings per Item", {i: len(t) for i, t in self.texts.items()})

    def copied_share(self, text: str, item: str) -> float:
        """Return the share of `text`'s 8-grams found in the real corpus's same Item."""
        h = self.hasher.hashes(tokenize(text))
        return float(np.isin(h, self.index[item]).mean()) if len(h) else 0.0


def fact_cells(questions: list[dict], chunks: list[Chunk], correct: int, attached: int) -> dict:
    """The fact-question and fact-correctness cells of one row."""
    scores = fact_score(questions, chunks)
    return {
        "n_questions": scores["n_questions"],
        "fact_coverage": scores["fact_coverage"],
        "fact_recall_at_10": scores["fact_recall_at_10"],
        "n_attached": attached,
        "fact_correct_pct": round(correct / attached * 100, 1) if attached >= MIN_ATTACHED else None,
    }


def all_row(name: str, n_docs: int, questions: list[dict], chunks: list[Chunk], correct: int, attached: int) -> dict:
    """The whole-filing row: the fact questions over every chunk of the set."""
    return {"set": name, "item": ALL, "n_docs": n_docs, **fact_cells(questions, chunks, correct, attached)}


def set_rows(name: str, companies: list[Company], real: RealItems, seed: int) -> list[dict]:
    """Return one metrics row per Item for one synthetic set."""
    texts = {c.company_id: item_texts(c.folder) for c in companies}
    chunks, questions = [], []
    facts = {c.company_id: (synthetic_facts(c.financials), synthetic_prior_facts(c.financials)) for c in companies}
    for c in companies:
        mine = company_chunks(
            c.company_id,
            c.name,
            c.fiscal_year,
            {i: c.folder / f"item_{i}.txt" for i in texts[c.company_id]},
            "synthetic",
        )
        chunks.extend(mine)
        questions.extend(make_questions(c.company_id, c.name, c.fiscal_year, facts[c.company_id][0], mine))
    rows = []
    all_correct = all_attached = 0
    for item in ITEMS:
        docs = [(c, texts[c.company_id][item]) for c in companies if item in texts[c.company_id]]
        if not docs:
            continue
        words = [len(t.split()) for _, t in docs]
        traced = total = correct = attached = 0
        for c, t in docs:
            a, b = traced_share(t, c.financials)
            traced, total = traced + a, total + b
            a, b = correctness([t], *facts[c.company_id])
            correct, attached = correct + a, attached + b
        all_correct, all_attached = all_correct + correct, all_attached + attached
        real_words = real.median_words.get(item)
        rows.append(
            {
                "set": name,
                "item": item,
                "n_docs": len(docs),
                "median_words": round(statistics.median(words)),
                "n_figures": total,
                "traced_pct": round(traced / total * 100, 1) if total >= MIN_FIGURES_PER_DOC * len(docs) else None,
                **fact_cells(questions, [ch for ch in chunks if ch.item == item], correct, attached),
                "self_retrieval_mrr": self_retrieval_mrr([(c.company_id, c.name, t) for c, t in docs], seed),
                "length_ratio": round(statistics.median(words) / real_words, 2) if real_words else None,
                "vocab_jaccard": round(
                    len(top_terms([t for _, t in docs]) & real.terms[item])
                    / len(top_terms([t for _, t in docs]) | real.terms[item]),
                    3,
                )
                if real.terms.get(item)
                else None,
                "figures_per_1k": round(
                    statistics.median([figure_count(t) / max(len(t.split()), 1) * 1000 for _, t in docs]),
                    2,
                ),
                "cross_doc_overlap": cross_doc_overlap([t for _, t in docs]),
                "copied_from_real": round(statistics.median([real.copied_share(t, item) for _, t in docs]), 3)
                if len(real.index.get(item, []))
                else None,
            },
        )
    rows.append(all_row(name, len(companies), questions, chunks, all_correct, all_attached))
    return rows


def real_rows(
    sample: list[tuple[RealFiling, dict]],
    real: RealItems,
    seed: int,
    label: str,
    metrics: dict[tuple[str, int], dict[str, float]],
) -> list[dict]:
    """Return the same metrics on a real sample of the sets' size."""
    chunks, questions = [], []
    facts = {}
    for filing, _ in sample:
        key = (filing.ticker, filing.fiscal_year)
        facts[key] = (
            real_facts(metrics.get(key, {})),
            real_facts(metrics.get((filing.ticker, filing.fiscal_year - 1), {})),
        )
        mine = company_chunks(filing.ticker, filing.name, filing.fiscal_year, filing.paths, "real")
        chunks.extend(mine)
        questions.extend(make_questions(filing.ticker, filing.name, filing.fiscal_year, facts[key][0], mine))
    rows = []
    all_correct = all_attached = 0
    for item in ITEMS:
        docs = [
            (f, real.texts[item][(f.ticker, f.fiscal_year)])
            for f, _ in sample
            if (f.ticker, f.fiscal_year) in real.texts[item]
        ]
        if not docs:
            continue
        holdout = [
            float(np.isin(h, real.other_index[item]).mean()) for h in real.holdout_hashes.get(item, []) if len(h)
        ]
        traced = total = correct = attached = 0
        traced_docs = 0
        for f, t in docs:
            a, b = correctness([t], *facts[(f.ticker, f.fiscal_year)])
            correct, attached = correct + a, attached + b
            current = metrics.get((f.ticker, f.fiscal_year))
            if not current:
                continue
            a, b = real_traced_share(t, real_expected_figures(current, metrics.get((f.ticker, f.fiscal_year - 1), {})))
            traced, total, traced_docs = traced + a, total + b, traced_docs + 1
        rows.append(
            {
                "set": label,
                "item": item,
                "n_docs": len(docs),
                "median_words": round(mw) if (mw := real.median_words.get(item)) else None,
                "n_figures": total if traced_docs else None,
                "traced_pct": round(traced / total * 100, 1)
                if traced_docs and total >= MIN_FIGURES_PER_DOC * traced_docs
                else None,
                **fact_cells(questions, [ch for ch in chunks if ch.item == item], correct, attached),
                "self_retrieval_mrr": self_retrieval_mrr([(f.ticker, f.name, t) for f, t in docs], seed),
                "length_ratio": 1.0,
                "vocab_jaccard": 1.0,
                "figures_per_1k": round(fig, 2) if (fig := real.figures_per_1k.get(item)) is not None else None,
                "cross_doc_overlap": cross_doc_overlap([t for _, t in docs]),
                "copied_from_real": round(statistics.median(holdout), 3) if holdout else None,
            },
        )
        all_correct, all_attached = all_correct + correct, all_attached + attached
    rows.append(all_row(label, len(sample), questions, chunks, all_correct, all_attached))
    return rows


def pick(rows: list[dict], real_row: dict | None) -> tuple[str | None, dict[str, str]]:
    """Return the suggested set for one Item and the best set per metric."""
    best, points = {}, Counter()
    for metric, (weight, direction) in WEIGHTS.items():
        target = real_row.get(metric) if real_row else None
        scored = []
        for r in rows:
            v = r.get(metric)
            if v is None:
                continue
            if direction == "high":
                key = -v
            elif direction == "one":
                key = abs(v - 1)
            elif target is None:
                continue
            elif direction == "real":
                key = abs(v - target)
            else:  # le_real
                key = max(v - target, 0) * 10 + abs(v - target)
            scored.append((key, r["set"]))
        if len(scored) < 2:
            continue
        scored.sort()
        best[metric] = scored[0][1]
        for rank, (_, name) in enumerate(scored):
            points[name] += weight * (len(scored) - 1 - rank)
    return (points.most_common(1)[0][0] if points else None), best


def fmt(v) -> str:
    """Format a cell value."""
    return "" if v is None else f"{v:g}" if isinstance(v, float) else str(v)


def markdown(rows: list[dict], set_names: list[str]) -> str:
    """Render the per-Item tables and the suggested picks."""
    lines = [
        "# Per-section scorecard",
        "",
        "One table per Item. `real` is the same measure on real 10-Ks (for self-retrieval and cross-document overlap, a real sample of the same size as the sets). See `evals/section_scorecard.py` for each metric.",
        "",
        "## Suggested method per section",
        "",
        "| Item | Suggested set | Best figure accuracy | Best retrieval utility |",
        "|---|---|---|---|",
    ]
    tables = []
    for item in ITEMS:
        item_rows = [r for r in rows if r["item"] == item and r["set"] in set_names]
        if not item_rows:
            continue
        real_row = next((r for r in rows if r["item"] == item and r["set"] == REAL), None)
        suggested, best = pick(item_rows, real_row)
        lines.append(
            f"| {ITEM_LABELS[item]} | {suggested or ''} | {best.get('fact_correct_pct', '')} | {best.get('fact_recall_at_10', '')} |",
        )
        shown = [*item_rows, *([real_row] if real_row else [])]
        tables += [
            "",
            f"## {ITEM_LABELS[item]}",
            "",
            "| Metric | " + " | ".join(r["set"] for r in shown) + " |",
            "|---" * (len(shown) + 1) + "|",
        ]
        for metric in [*INFO, *WEIGHTS]:
            cells = []
            for r in shown:
                cell = fmt(r.get(metric))
                cells.append(f"**{cell}**" if best.get(metric) == r["set"] and cell else cell)
            tables.append(f"| {metric} | " + " | ".join(cells) + " |")
    whole = [r for r in rows if r["item"] == ALL and r["set"] in [*set_names, REAL]]
    fact_metrics = ["n_docs", "n_questions", "fact_coverage", "fact_recall_at_10", "n_attached", "fact_correct_pct"]
    lines += [
        "",
        "Suggested set = highest weighted rank across the metrics (fact correctness and fact retrieval count twice). Bold = best set on that metric.",
        "",
        "## Whole filing: standard fact questions",
        "",
        "Ten questions per company, the same for every set (qa/fact_questions.py), over all of a filing's chunks.",
        "",
        "| Set | " + " | ".join(fact_metrics) + " |",
        "|---" * (len(fact_metrics) + 1) + "|",
        *(f"| {r['set']} | " + " | ".join(fmt(r.get(m)) for m in fact_metrics) + " |" for r in whole),
        *tables,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score every set and write scorecard.csv and scorecard.md."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a synthetic set; repeat for each",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=DEFAULT_REAL_DIR,
        help="real filings as {ticker}/{fy}/item_*.txt",
    )
    parser.add_argument(
        "--real-financials",
        type=Path,
        default=DEFAULT_REAL_FINANCIALS,
        help="Method 1's financials.csv",
    )
    parser.add_argument("--real-sample", type=int, default=20, help="real filings in the same-size comparisons")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sets = dict(s.split("=", 1) for s in args.set) if args.set else {k: str(v) for k, v in DEFAULT_SETS.items()}
    companies = {name: load_set(name, Path(path), args.out) for name, path in sets.items() if Path(path).exists()}
    if missing := [n for n in sets if n not in companies]:
        logger.warning("Skipping sets with no folder: %s", ", ".join(missing))

    records = load_real_records(args.real_financials)
    filings = load_real_filings(args.real_dir, records)
    by_key = {(r["ticker"], r["fiscal_year"]): r for r in records if r.get("revenues")}
    rng = random.Random(args.seed)
    sample_filings = rng.sample(filings, min(args.real_sample, len(filings)))
    sample = [(f, by_key.get((f.ticker, f.fiscal_year), {})) for f in sample_filings]
    real = RealItems(filings, holdout={(f.ticker, f.fiscal_year) for f in sample_filings})

    rows = real_rows(sample, real, args.seed, REAL, load_real_metrics(args.real_financials))
    for name, cs in companies.items():
        logger.info("Scoring %s (%d companies)", name, len(cs))
        rows += set_rows(name, cs, real, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "scorecard.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_atomic(args.out / "scorecard.md", markdown(rows, list(companies)))
    logger.info("Wrote %s", args.out / "scorecard.md")


if __name__ == "__main__":
    main()
