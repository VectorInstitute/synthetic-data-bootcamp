r"""Fidelity, utility and privacy evals on several synthetic 10-K sets, side by side.

Each set is a folder of fictional filings from either pipeline: Method 1's
`filings/{company}/filing.html` (converted with method2_gates.convert) or Method 2's
`{fictional_id}/{fy}/` layout. Every set is scored the same way against one real
corpus, so the columns are comparable:

Fidelity, does it read and compute like a real 10-K?
- Method 2's Phase 7 gates in the form method2_gates.validate runs them (every S&P 500
  name instead of one template; prior-year figures allowed): substantive sections,
  Item 8 headings, and the share of dollar and percent figures in Items 7 and 8
  that trace to the statements.
- Gap to the real corpus (as in step13_gap_report): median words per Item,
  overlap of the 3,000 most frequent terms, dollar and percent figures per 1,000 words.

Utility, is it useful for a retrieval benchmark (as in RAG.step11/step12)?
- QA pairs built the same way for every set and for a sample of real filings:
  revenue, net income, total assets, net margin and operating margin, each answered
  by a verbatim span of one chunk. Method 2's scenario questions are counted separately,
  since only its scenario-bearing set can have them.
- Recall@5, recall@10, MRR and nDCG@10 for BM25, each embedding model, and a BM25 +
  embedding hybrid, on the set alone (synthetic_only), on the real sample alone
  (real_only, the same for every set) and on both together (mixed).
- Whether the retrievers rank in the same order on the set as on real filings
  (Spearman correlation of MRR across retrievers).

Privacy, could it expose a real company?
- Real company names in the text (Method 1's case-sensitive S&P 500 patterns), and for
  Method 2's sets the template company's own name or ticker.
- Distance from each fictional company's financials to the closest real
  company-year (median relative difference over shared fields), and for Method 2's sets
  to the template it was scaled from.
- Share of each filing's 8-word sequences that also appear in the real corpus. The
  "Real 10-Ks" column gives the same figure for the real sample against the other
  real filings, since boilerplate repeats across real filings too.

Embeddings: `--embedding-models` (default text-embedding-3-small, plus Method 2's local
potion-retrieval-32M when fastembed can load it). text-embedding-* models go through
Vector's proxy (OPENAI_API_KEY or SYNBENCH_API_KEY) and are cached on disk, see
RAG.retrieval.

Output, under --out (default results/compare_sets/): comparison.md (the three
tables), summary.json, and per-set CSVs (gates, qa_pairs, retrieval_eval,
privacy_financials, privacy_text), plus the real sample's QA pairs.

Run from implementations/synthetic_10k, in the repo's uv environment:
    uv run python -m evals.compare_sets
    uv run python -m evals.compare_sets \
        --set method1_gemini=data/gemini20/method_1/synthetic/filings \
        --set method2_claude=method_2/data/synthetic --real-sample 10
"""

import argparse
import csv
import io
import json
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from method_2.step09_validate_filings import (  # noqa: E402
    check_real_company_leaks,
    extract_figures,
    load_real_companies,
    real_section_stats,
)
from method_2.utils import setup_logging, write_atomic  # noqa: E402
from qa.fact_questions import make_questions, synthetic_facts  # noqa: E402
from RAG import step11_make_qa as qa  # noqa: E402
from RAG.chunks import Chunk, chunk_text  # noqa: E402
from RAG.retrieval import (  # noqa: E402
    POTION_MODEL_NAME,
    Bm25Retriever,
    SemanticRetriever,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
    tokenize,
)

from evals import method1_checks, method2_gates  # noqa: E402


logger = setup_logging("compare_sets")

DEFAULT_SETS = {
    "method1_gemini": ROOT / "data" / "gemini20" / "method_1" / "synthetic" / "filings",
    "method2_gemini": ROOT / "data" / "gemini20" / "method_2" / "synthetic",
    "method2_claude": ROOT / "method_2" / "data" / "synthetic",
}
SET_LABELS = {
    "method1_gemini": "Method 1 · Gemini",
    "method2_gemini": "Method 2 · Gemini",
    "method2_claude": "Method 2 · Claude",
}
DEFAULT_REAL_DIR = ROOT / "method_2" / "data" / "parsed"
DEFAULT_REAL_FINANCIALS = ROOT.parents[1] / "data" / "sec10k" / "financials.csv"
CALIBRATION_PATH = ROOT / "method_2" / "data" / "calibration.json"
CONSTITUENTS_PATH = ROOT / "method_2" / "data" / "sp500_constituents.csv"
DEFAULT_OUT = HERE / "results" / "compare_sets"

ITEMS = ["1", "1a", "1c", "7", "8"]
TOP_VOCAB_SIZE = 3000
SHINGLE = 8
COPY_THRESHOLD = 0.05
PRIVACY_FIELDS = [
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "revenues",
    "net_income",
    "operating_income",
]
# Method 1's financials.csv column names for the privacy fields.
REAL_FINANCIALS_COLUMNS = {"revenue": "revenues"}
RETRIEVAL_FIELDS = ["experiment", "retriever", "n_questions", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]


@dataclass
class Company:
    company_id: str
    name: str
    fiscal_year: int
    folder: Path
    spec: dict
    financials: dict


@dataclass
class RealFiling:
    ticker: str
    fiscal_year: int
    name: str
    paths: dict[str, Path]


# ---------- loading ----------


def load_set(name: str, path: Path, work_dir: Path) -> list[Company]:
    """Return the set's companies in Method 2's layout, converting Method 1's filing.html if needed."""
    htmls = sorted(path.glob("*/filing.html"))
    if htmls:
        converted = work_dir / name / "converted"
        spec_paths = [method2_gates.convert(h, converted) for h in htmls]
    else:
        spec_paths = sorted(path.glob("*/*/spec.json"))
    if not spec_paths:
        raise SystemExit(f"{name}: no filing.html or */*/spec.json under {path}")
    companies = []
    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
        companies.append(
            Company(
                spec["fictional_id"],
                spec["fictional_name"],
                int(spec["fiscal_year"]),
                spec_path.parent,
                spec,
                financials,
            ),
        )
    logger.info("%s: %d companies from %s", name, len(companies), path)
    return companies


def item_texts(folder: Path) -> dict[str, str]:
    return {item: p.read_text(encoding="utf-8") for item in ITEMS if (p := folder / f"item_{item}.txt").exists()}


def load_real_records(financials_csv: Path) -> list[dict]:
    """Real company-years with the privacy fields: Method 2's calibration.json plus Method 1's financials.csv."""
    records = []
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    for r in calibration["company_years"]:
        records.append(
            {
                "ticker": r["ticker"],
                "fiscal_year": int(r["fiscal_year"]),
                "source": "calibration.json",
                **{f: r.get(f) for f in PRIVACY_FIELDS},
            },
        )
    if financials_csv.exists():
        with open(financials_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = {
                    "ticker": row["ticker"],
                    "fiscal_year": int(row["fiscal_year"]),
                    "source": "financials.csv",
                    "name": row.get("company", ""),
                }
                for column, value in row.items():
                    field = REAL_FINANCIALS_COLUMNS.get(column, column)
                    if field in PRIVACY_FIELDS:
                        rec[field] = float(value) if value not in ("", None) else None
                records.append(rec)
    else:
        logger.warning("No %s; real financials come from calibration.json only", financials_csv)
    return records


def load_real_filings(real_dir: Path, records: list[dict]) -> list[RealFiling]:
    names = {}
    with open(CONSTITUENTS_PATH, encoding="utf-8") as f:
        names.update({row["Symbol"]: row["Security"] for row in csv.DictReader(f)})
    names.update({r["ticker"]: r["name"].title() for r in records if r.get("name")})
    filings = []
    for fy_dir in sorted(real_dir.glob("*/*")):
        if not fy_dir.name.isdigit():
            continue
        paths = {item: p for item in ITEMS if (p := fy_dir / f"item_{item}.txt").exists()}
        if paths:
            ticker = fy_dir.parent.name
            filings.append(RealFiling(ticker, int(fy_dir.name), names.get(ticker, ticker), paths))
    if not filings:
        raise SystemExit(f"No real filings under {real_dir} ({{ticker}}/{{fy}}/item_*.txt)")
    logger.info("Real corpus: %d filings from %s", len(filings), real_dir)
    return filings


# ---------- shared text statistics ----------


class ShingleHasher:
    """Hash 8-word sequences to 64-bit ints, so the real corpus fits in one sorted array."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def hashes(self, tokens: list[str]) -> np.ndarray:
        if len(tokens) < SHINGLE:
            return np.zeros(0, dtype=np.uint64)
        ids = np.fromiter(
            (self.vocab.setdefault(t, len(self.vocab) + 1) for t in tokens),
            dtype=np.uint64,
            count=len(tokens),
        )
        n = len(ids) - SHINGLE + 1
        h: np.ndarray = np.zeros(n, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(SHINGLE):
                h = h * np.uint64(1_000_003) + ids[j : j + n]
        return h


def text_stats(text: str) -> dict:
    tokens = tokenize(text)
    dollars, percents = extract_figures(text)
    n_words = max(len(text.split()), 1)
    return {
        "tokens": tokens,
        "words": len(text.split()),
        "dollars_per_1k": len(dollars) / n_words * 1000,
        "percents_per_1k": len(percents) / n_words * 1000,
    }


class RealCorpus:
    """One pass over every real filing: lengths, vocabulary, figure density, 8-gram index."""

    def __init__(self, filings: list[RealFiling], holdout: set[str]) -> None:
        self.hasher = ShingleHasher()
        self.words: dict[str, list[int]] = {item: [] for item in ITEMS}
        self.dollars_per_1k, self.percents_per_1k = [], []
        counts: Counter = Counter()
        all_hashes, other_hashes = [], []
        self.holdout_hashes: dict[str, list[np.ndarray]] = {}
        for i, filing in enumerate(filings, start=1):
            for item, path in filing.paths.items():
                stats = text_stats(path.read_text(encoding="utf-8", errors="replace"))
                self.words[item].append(stats["words"])
                self.dollars_per_1k.append(stats["dollars_per_1k"])
                self.percents_per_1k.append(stats["percents_per_1k"])
                counts.update(stats["tokens"])
                h = self.hasher.hashes(stats["tokens"])
                all_hashes.append(h)
                if filing.ticker in holdout:
                    self.holdout_hashes.setdefault(f"{filing.ticker}_{filing.fiscal_year}", []).append(h)
                else:
                    other_hashes.append(h)
            if i % 50 == 0:
                logger.info("Real corpus: %d/%d filings read", i, len(filings))
        self.top_terms = {t for t, _ in counts.most_common(TOP_VOCAB_SIZE)}
        self.index = np.unique(np.concatenate(all_hashes)) if all_hashes else np.zeros(0, dtype=np.uint64)
        self.other_index = np.unique(np.concatenate(other_hashes)) if other_hashes else np.zeros(0, dtype=np.uint64)
        logger.info("Real corpus: %d distinct %d-word sequences", len(self.index), SHINGLE)

    def copied_share(self, hashes: np.ndarray, index: np.ndarray | None = None) -> float:
        index = self.index if index is None else index
        return float(np.isin(hashes, index).mean()) if len(hashes) else 0.0

    def holdout_copied_shares(self) -> list[float]:
        """Each held-out real filing's 8-gram share found in the *other* real filings."""
        return [self.copied_share(np.concatenate(hs), self.other_index) for hs in self.holdout_hashes.values()]


# ---------- fidelity ----------


def run_gates(companies: list[Company]) -> list[dict]:
    real_companies, real_stats = load_real_companies(), real_section_stats()
    rows = []
    for c in companies:
        rows.extend(method2_gates.validate(c.folder / "spec.json", real_companies, real_stats))
    return rows


def gate_summary(rows: list[dict]) -> dict:
    def pass_rate(check: str) -> str:
        sel = [r for r in rows if r["check"] == check]
        # method2_gates notes a heading "failure" that is only the plural "Statements of".
        ok = [r for r in sel if r["status"] == "PASS" or r.get("note", "").startswith("filing uses the plural")]
        return f"{len(ok)}/{len(sel)}" if sel else "n/a"

    traced = total = explained = 0
    for r in rows:
        if r["check"] != "numeric_consistency":
            continue
        m = re.match(r"(\d+)/(\d+) figures traced", r["detail"])
        if m:
            traced += int(m.group(1))
            total += int(m.group(2))
        m = re.match(r"(\d+) of \d+ unmatched figures are year-over-year", r.get("note", ""))
        if m:
            explained += int(m.group(1))
    return {
        "substantive": pass_rate("substantive"),
        "required_headings": pass_rate("required_headings"),
        "numeric_consistency": pass_rate("numeric_consistency"),
        "figures_traced_pct": round(traced / total * 100, 1) if total else None,
        "figures_traced_or_yoy_pct": round((traced + explained) / total * 100, 1) if total else None,
    }


def gap_summary(companies: list[Company], real: RealCorpus) -> dict:
    words = {item: [] for item in ITEMS}
    dollars, percents = [], []
    counts: Counter = Counter()
    for c in companies:
        for item, text in item_texts(c.folder).items():
            stats = text_stats(text)
            words[item].append(stats["words"])
            dollars.append(stats["dollars_per_1k"])
            percents.append(stats["percents_per_1k"])
            counts.update(stats["tokens"])
    top = {t for t, _ in counts.most_common(TOP_VOCAB_SIZE)}
    out = {f"item_{item}_median_words": round(statistics.median(w)) if w else None for item, w in words.items()}
    out["vocab_jaccard"] = round(len(top & real.top_terms) / len(top | real.top_terms), 3) if top else None
    out["dollars_per_1k_words"] = round(statistics.median(dollars), 2) if dollars else None
    out["percents_per_1k_words"] = round(statistics.median(percents), 2) if percents else None
    return out


def real_gap_summary(real: RealCorpus) -> dict:
    out: dict[str, float | None] = {
        f"item_{item}_median_words": round(statistics.median(w)) if w else None for item, w in real.words.items()
    }
    out["vocab_jaccard"] = 1.0
    out["dollars_per_1k_words"] = round(statistics.median(real.dollars_per_1k), 2)
    out["percents_per_1k_words"] = round(statistics.median(real.percents_per_1k), 2)
    return out


# ---------- utility ----------


def company_chunks(company_id: str, name: str, fy: int, paths: dict[str, Path], source: str) -> list[Chunk]:
    chunks = []
    for item, path in paths.items():
        for i, para in enumerate(chunk_text(path.read_text(encoding="utf-8", errors="replace"))):
            chunks.append(
                Chunk(f"{source[0]}:{company_id}_{fy}_{item}_{i}", company_id, name, fy, item, source, str(path), para),
            )
    return chunks


def set_chunks(companies: list[Company]) -> list[Chunk]:
    out = []
    for c in companies:
        out.extend(
            company_chunks(
                c.company_id,
                c.name,
                c.fiscal_year,
                {i: c.folder / f"item_{i}.txt" for i in item_texts(c.folder)},
                "synthetic",
            ),
        )
    return out


def financial_fact_rows(
    source: str,
    ticker: str,
    name: str,
    fy: int,
    financials: dict,
    chunks: list[Chunk],
) -> list[dict]:
    """The shared fact questions (qa.fact_questions) that have a stating chunk, as QA rows.

    Retrieval evaluation needs a gold chunk, so unanswerable questions are left out
    here; section_scorecard counts them as misses instead.
    """
    by_id = {c.chunk_id: c for c in chunks}
    rows = []
    for q in make_questions(ticker, name, fy, synthetic_facts(financials), chunks):
        if not q["gold_chunk_ids"]:
            continue
        chunk = by_id[q["gold_chunk_ids"].split(";")[0]]
        answer = f"{q['answer']:.1f}%" if q["is_percent"] else f"${q['answer']:,.0f}"
        row = qa.qa_row(source, ticker, fy, chunk.item, "factual", q["question"], answer, chunk.chunk_id)
        row["gold_chunk_ids"] = q["gold_chunk_ids"]
        rows.append(row)
    return rows


def set_qa(companies: list[Company], chunks: list[Chunk]) -> list[dict]:
    rows = []
    for c in companies:
        mine = [ch for ch in chunks if ch.ticker == c.company_id and ch.fiscal_year == c.fiscal_year]
        for row in financial_fact_rows("synthetic", c.company_id, c.name, c.fiscal_year, c.financials, mine):
            rows.append({**row, "kind": "core"})
        if isinstance(c.spec.get("scenario"), dict):
            rows.extend({**row, "kind": "scenario"} for row in qa.scenario_qa_rows(c.spec, mine))
    return rows


def real_sample(filings: list[RealFiling], records: list[dict], n: int, seed: int) -> list[tuple[RealFiling, dict]]:
    """Pick n real filings that have Item 7 and XBRL revenue, deterministically."""
    by_key = {}
    for r in records:
        if r.get("revenues"):
            by_key.setdefault((r["ticker"], r["fiscal_year"]), r)
    candidates = [
        (f, by_key[(f.ticker, f.fiscal_year)])
        for f in filings
        if "7" in f.paths and (f.ticker, f.fiscal_year) in by_key
    ]
    rng = random.Random(seed)
    picked = rng.sample(candidates, min(n, len(candidates)))
    return sorted(picked, key=lambda p: (p[0].ticker, p[0].fiscal_year))


def real_financials(record: dict) -> dict:
    return {
        "income_statement": {k: record.get(k) for k in ("revenues", "net_income", "operating_income")},
        "balance_sheet": {"assets": {"total_assets": record.get("total_assets")}},
    }


def strip_none(d: dict) -> dict:
    return {k: (strip_none(v) if isinstance(v, dict) else v) for k, v in d.items() if v is not None}


def evaluate(qa_rows: list[dict], rankings: dict[str, list[str]], name: str, experiment: str) -> dict:
    n = len(qa_rows)
    sums = {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0}
    for q in qa_rows:
        ranking = rankings[q["qa_id"]]
        gold = set(filter(None, q.get("gold_chunk_ids", "").split(";"))) or {q["gold_chunk_id"]}
        sums["recall_at_5"] += recall_at_k(ranking, gold, 5)
        sums["recall_at_10"] += recall_at_k(ranking, gold, 10)
        sums["mrr"] += reciprocal_rank(ranking, gold)
        sums["ndcg_at_10"] += ndcg_at_k(ranking, gold, 10)
    return {
        "experiment": experiment,
        "retriever": name,
        "n_questions": n,
        **{k: round(v / n, 4) for k, v in sums.items()},
    }


def retrieval_rows(experiment: str, chunks: list[Chunk], qa_rows: list[dict], models: list[str]) -> list[dict]:
    ids = {c.chunk_id for c in chunks}
    qa_rows = [q for q in qa_rows if q["gold_chunk_id"] in ids]
    if not qa_rows:
        return []
    chunk_ids, texts, questions = (
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        [q["question"] for q in qa_rows],
    )
    logger.info("%s: %d chunks, %d questions", experiment, len(chunks), len(qa_rows))
    bm25 = Bm25Retriever(chunk_ids, texts)
    rankings = {"bm25": {q["qa_id"]: bm25.rank(q["question"]) for q in qa_rows}}
    for model in models:
        short = model.split("/")[-1]
        semantic = SemanticRetriever(chunk_ids, texts, model_name=model)
        sem = dict(zip([q["qa_id"] for q in qa_rows], semantic.rank_many(questions), strict=True))
        rankings[f"semantic:{short}"] = sem
        rankings[f"hybrid:bm25+{short}"] = {k: reciprocal_rank_fusion([rankings["bm25"][k], v]) for k, v in sem.items()}
    return [evaluate(qa_rows, r, name, experiment) for name, r in rankings.items()]


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v: list[float]) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in set(v):  # average ties
            idx = [i for i, x in enumerate(v) if x == value]
            r[idx] = np.mean(r[idx])
        return r

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 2)


def usable_models(models: list[str]) -> list[str]:
    """Drop local models fastembed can't load (e.g. no Hugging Face access), with a warning."""
    from RAG.retrieval import get_embedder  # noqa: PLC0415

    kept = []
    for model in models:
        try:
            get_embedder(model).embed(["warm-up"])
            kept.append(model)
        except Exception as e:  # noqa: BLE001 -- report and carry on with the rest
            logger.warning("Skipping embedding model %s: %s", model, e)
    if not kept:
        raise SystemExit("No embedding model could be loaded.")
    return kept


# ---------- privacy ----------


def fictional_values(c: Company) -> dict[str, float]:
    leaves = {}

    def walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict) and k != "prior_year":
                walk(v)
            elif isinstance(v, (int, float)):
                leaves.setdefault(k, float(v))

    walk(c.financials)
    leaves.setdefault("stockholders_equity", leaves.get("total_stockholders_equity"))
    return {f: leaves[f] for f in PRIVACY_FIELDS if leaves.get(f) is not None}


def distance(fictional: dict[str, float], real: dict) -> tuple[float, float, int] | None:
    """(median, max) relative difference over shared non-zero fields, and the field count."""
    diffs = [abs(v - real[f]) / abs(real[f]) for f, v in fictional.items() if real.get(f)]
    if len(diffs) < 4:
        return None
    return statistics.median(diffs), max(diffs), len(diffs)


def privacy_financial_rows(companies: list[Company], records: list[dict]) -> list[dict]:
    rows = []
    for c in companies:
        values = fictional_values(c)
        best = None
        for r in records:
            d = distance(values, r)
            if d and (best is None or d[0] < best[1][0]):
                best = (r, d)
        template = c.spec.get("template") or {}
        tmpl = next(
            (
                r
                for r in records
                if r["ticker"] == template.get("ticker") and r["fiscal_year"] == template.get("fiscal_year")
            ),
            None,
        )
        tdist = distance(values, tmpl) if tmpl else None
        rows.append(
            {
                "company_id": c.company_id,
                "nearest_real": f"{best[0]['ticker']} FY{best[0]['fiscal_year']}" if best else "",
                "nearest_median_pct": round(best[1][0] * 100, 1) if best else None,
                "nearest_max_pct": round(best[1][1] * 100, 1) if best else None,
                "fields_compared": best[1][2] if best else 0,
                "template": f"{template.get('ticker')} FY{template.get('fiscal_year')}" if tmpl else "",
                "template_median_pct": round(tdist[0] * 100, 1) if tdist else None,
            },
        )
    return rows


def privacy_text_rows(companies: list[Company], real: RealCorpus) -> list[dict]:
    name_patterns = method1_checks.load_real_names()
    sp500 = load_real_companies()
    rows = []
    for c in companies:
        texts = item_texts(c.folder)
        combined = "\n\n".join(texts.values())
        names = sorted(
            {
                n
                for t in texts.values()
                for issue in method1_checks.check_real_names(t, name_patterns)
                for n in issue.split(": ", 1)[1].rstrip(".").split(", ")
            },
        )
        template = c.spec.get("template") or {}
        template_hits = (
            check_real_company_leaks(combined, template["ticker"], sp500.get(template["ticker"], template["ticker"]))
            if template.get("ticker")
            else []
        )
        hashes = real.hasher.hashes(tokenize(combined))
        rows.append(
            {
                "company_id": c.company_id,
                "real_names": "; ".join(names),
                "template_leak": "; ".join(template_hits),
                "copied_8gram_pct": round(real.copied_share(hashes) * 100, 2),
            },
        )
    return rows


# ---------- report ----------


def fmt(v, suffix: str = "") -> str:
    if v is None or v == "":
        return "n/a"
    return f"{v:.2f}{suffix}" if isinstance(v, float) and not suffix else f"{v}{suffix}"


def median_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def markdown(summary: dict, set_names: list[str]) -> str:
    cols = ["Real 10-Ks"] + [SET_LABELS.get(s, s) for s in set_names]
    header = "| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"

    def row(label: str, reference, values: list) -> str:
        return f"| {label} | {reference} | " + " | ".join(values) + " |\n"

    sets = [summary["sets"][s] for s in set_names]
    real = summary["real"]
    out = ["# Fidelity, utility and privacy: synthetic sets vs real 10-Ks\n"]
    out.append(
        f"Real corpus: {real['n_filings']} filings. Real sample for retrieval: {real['n_sample']} filings. Embedding models: {', '.join(summary['embedding_models'])}.\n",
    )

    out.append("\n## Fidelity\n\n" + header)
    out.append(row("Companies", real["n_filings"], [str(s["n_companies"]) for s in sets]))
    for item in ITEMS:
        key = f"item_{item}_median_words"
        out.append(row(f"Median words, Item {item.upper()}", fmt(real["gap"][key]), [fmt(s["gap"][key]) for s in sets]))
    out.append(row("Top-3,000 term overlap with real (Jaccard)", "1.0", [fmt(s["gap"]["vocab_jaccard"]) for s in sets]))
    out.append(
        row(
            "Dollar figures per 1,000 words (median)",
            fmt(real["gap"]["dollars_per_1k_words"]),
            [fmt(s["gap"]["dollars_per_1k_words"]) for s in sets],
        ),
    )
    out.append(
        row(
            "Percent figures per 1,000 words (median)",
            fmt(real["gap"]["percents_per_1k_words"]),
            [fmt(s["gap"]["percents_per_1k_words"]) for s in sets],
        ),
    )
    out.append(row("Sections with 150+ words", "", [s["gates"]["substantive"] for s in sets]))
    out.append(
        row(
            "Item 8 has all three statement headings (singular or plural)",
            "",
            [s["gates"]["required_headings"] for s in sets],
        ),
    )
    out.append(row("Items 7 and 8 with every figure traced", "", [s["gates"]["numeric_consistency"] for s in sets]))
    out.append(row("Figures traced to the statements", "", [fmt(s["gates"]["figures_traced_pct"], "%") for s in sets]))
    out.append(
        row("… or explained as year-over-year", "", [fmt(s["gates"]["figures_traced_or_yoy_pct"], "%") for s in sets]),
    )

    out.append("\n## Utility (retrieval benchmark)\n\n" + header)
    out.append(
        row(
            "Core QA pairs (5 facts per company)",
            real["n_qa"],
            [f"{s['n_core_qa']} ({s['core_qa_per_company']}/company)" for s in sets],
        ),
    )
    out.append(row("Scenario QA pairs", "", [str(s["n_scenario_qa"]) for s in sets]))
    retrievers = [r["retriever"] for r in real["retrieval"]]
    for metric, label in (("recall_at_10", "Recall@10"), ("mrr", "MRR")):
        for name in retrievers:
            real_v = next((r[metric] for r in real["retrieval"] if r["retriever"] == name), None)
            vals = [
                fmt(
                    next(
                        (
                            r[metric]
                            for r in s["retrieval"]
                            if r["retriever"] == name and r["experiment"] == "synthetic_only"
                        ),
                        None,
                    ),
                )
                for s in sets
            ]
            out.append(row(f"{label}, {name}, on its own corpus", fmt(real_v), vals))
        for name in retrievers:
            vals = [
                fmt(
                    next(
                        (r[metric] for r in s["retrieval"] if r["retriever"] == name and r["experiment"] == "mixed"),
                        None,
                    ),
                )
                for s in sets
            ]
            out.append(row(f"{label}, {name}, synthetic + real mixed", "", vals))
    out.append(row("Retriever order matches real (Spearman, MRR)", "1.0", [fmt(s["ranking_spearman"]) for s in sets]))
    out.append(row("Best retriever (by MRR)", real["best_retriever"], [s["best_retriever"] for s in sets]))

    out.append("\n## Privacy\n\n" + header)
    out.append(
        row(
            "Filings naming a real S&P 500 company",
            "",
            [f"{s['n_real_name_filings']}/{s['n_companies']}" for s in sets],
        ),
    )
    out.append(row("Names found", "", [s["real_names_found"] or "none" for s in sets]))
    out.append(row("Template company's name or ticker in text", "", [s["template_leaks"] for s in sets]))
    out.append(
        row("Median distance to nearest real company-year", "", [fmt(s["nearest_median_pct"], "%") for s in sets]),
    )
    out.append(
        row("Companies within 5% of a real company on every field", "", [str(s["n_within_5pct_all"]) for s in sets]),
    )
    out.append(row("Median distance to own template", "", [fmt(s["template_median_pct"], "%") for s in sets]))
    out.append(
        row(
            "8-word sequences also in real filings (median)",
            fmt(real["copied_median_pct"], "%"),
            [fmt(s["copied_median_pct"], "%") for s in sets],
        ),
    )
    out.append(
        row("… highest single filing", fmt(real["copied_max_pct"], "%"), [fmt(s["copied_max_pct"], "%") for s in sets]),
    )
    out.append(row(f"Filings over {COPY_THRESHOLD:.0%} copied", "", [str(s["n_over_copy_threshold"]) for s in sets]))

    out.append(
        "\nNotes: the real column's lengths and densities cover every real filing; its retrieval figures are the "
        "real_only experiment and its copied share compares the real sample with the other real filings. Distances "
        "are the median relative difference over total assets, liabilities, equity, revenue, net income and operating "
        "income. Name hits are case-sensitive S&P 500 names, so some may be ordinary words; see privacy_text.csv.\n",
    )
    return "".join(out)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    buffer = io.StringIO()
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a synthetic set; repeat for each (default: the three gemini20/claude sets)",
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
        help="Method 1's financials.csv (XBRL metrics per filing)",
    )
    parser.add_argument("--real-sample", type=int, default=15, help="real filings in the retrieval sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedding-models", default=f"text-embedding-3-small,{POTION_MODEL_NAME}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sets = dict(s.split("=", 1) for s in args.set) if args.set else {k: str(v) for k, v in DEFAULT_SETS.items()}
    models = usable_models([m.strip() for m in args.embedding_models.split(",") if m.strip()])
    companies = {name: load_set(name, Path(path), args.out) for name, path in sets.items()}

    records = load_real_records(args.real_financials)
    filings = load_real_filings(args.real_dir, records)
    sample = real_sample(filings, records, args.real_sample, args.seed)
    real = RealCorpus(filings, holdout={f.ticker for f, _ in sample})

    # Real sample: QA and the real_only experiment, shared by every set.
    real_chunks, real_qa = [], []
    for filing, record in sample:
        chunks = company_chunks(filing.ticker, filing.name, filing.fiscal_year, filing.paths, "real")
        real_chunks.extend(chunks)
        real_qa.extend(
            {**r, "kind": "core"}
            for r in financial_fact_rows(
                "real",
                filing.ticker,
                filing.name,
                filing.fiscal_year,
                strip_none(real_financials(record)),
                chunks,
            )
        )
    for i, row in enumerate(real_qa):
        row["qa_id"] = f"real_{i:04d}"
    write_csv(args.out / "real" / "qa_pairs.csv", real_qa)
    real_retrieval = retrieval_rows("real_only", real_chunks, real_qa, models)
    write_csv(args.out / "real" / "retrieval_eval.csv", real_retrieval)
    real_mrr = {r["retriever"]: r["mrr"] for r in real_retrieval}
    copied_real = [s * 100 for s in real.holdout_copied_shares()]

    summary: dict[str, Any] = {
        "embedding_models": models,
        "real": {
            "n_filings": len(filings),
            "n_sample": len(sample),
            "n_qa": len(real_qa),
            "gap": real_gap_summary(real),
            "retrieval": real_retrieval,
            "best_retriever": max(real_mrr, key=lambda k: real_mrr[k]) if real_mrr else "n/a",
            "copied_median_pct": median_or_none(copied_real),
            "copied_max_pct": round(max(copied_real), 1) if copied_real else None,
        },
        "sets": {},
    }

    for name, comps in companies.items():
        out = args.out / name
        logger.info("=== %s ===", name)
        gates = run_gates(comps)
        write_csv(out / "gates.csv", gates)

        chunks = set_chunks(comps)
        qa_rows = set_qa(comps, chunks)
        for i, row in enumerate(qa_rows):
            row["qa_id"] = f"{name}_{i:04d}"
        write_csv(out / "qa_pairs.csv", qa_rows)
        retrieval = retrieval_rows("synthetic_only", chunks, qa_rows, models)
        retrieval += retrieval_rows("mixed", chunks + real_chunks, qa_rows + real_qa, models)
        write_csv(out / "retrieval_eval.csv", retrieval)
        own = {r["retriever"]: r["mrr"] for r in retrieval if r["experiment"] == "synthetic_only"}
        shared = [k for k in real_mrr if k in own]

        fin = privacy_financial_rows(comps, records)
        write_csv(out / "privacy_financials.csv", fin)
        text = privacy_text_rows(comps, real)
        write_csv(out / "privacy_text.csv", text)
        copied = [r["copied_8gram_pct"] for r in text]
        names = Counter(n for r in text for n in r["real_names"].split("; ") if n)
        n_core = sum(r["kind"] == "core" for r in qa_rows)
        has_template = any(r["template"] for r in fin)

        summary["sets"][name] = {
            "n_companies": len(comps),
            "gates": gate_summary(gates),
            "gap": gap_summary(comps, real),
            "n_core_qa": n_core,
            "core_qa_per_company": round(n_core / len(comps), 1),
            "n_scenario_qa": sum(r["kind"] == "scenario" for r in qa_rows),
            "retrieval": retrieval,
            "ranking_spearman": spearman([real_mrr[k] for k in shared], [own[k] for k in shared]),
            "best_retriever": max(own, key=lambda k: own[k]) if own else "n/a",
            "n_real_name_filings": sum(bool(r["real_names"]) for r in text),
            "real_names_found": ", ".join(f"{n} ({k})" for n, k in names.most_common(6)),
            "template_leaks": f"{sum(bool(r['template_leak']) for r in text)}/{len(comps)}"
            if has_template
            else "n/a (no template)",
            "nearest_median_pct": median_or_none([r["nearest_median_pct"] for r in fin]),
            "n_within_5pct_all": sum(
                1 for r in fin if r["nearest_max_pct"] is not None and r["nearest_max_pct"] <= COPY_THRESHOLD * 100
            ),
            "template_median_pct": median_or_none([r["template_median_pct"] for r in fin]) if has_template else None,
            "copied_median_pct": median_or_none(copied),
            "copied_max_pct": round(max(copied), 1) if copied else None,
            "n_over_copy_threshold": sum(1 for v in copied if v > COPY_THRESHOLD * 100),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "summary.json", json.dumps(summary, indent=2))
    report = markdown(summary, list(companies))
    write_atomic(args.out / "comparison.md", report)
    print(report)
    logger.info("Wrote %s", args.out / "comparison.md")


if __name__ == "__main__":
    main()
