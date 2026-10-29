"""Phase 10: quantify the gap between the real and synthetic corpora.

Four comparisons (PLAN.md Phase 10):
  1. Section length (chars/words per item, from data/index.csv directly).
  2. Vocabulary overlap (Jaccard over each corpus's most frequent terms).
  3. Number distributions (dollar-figure magnitude and percent-figure value, reusing
     method_2.step09_validate_filings's own figure extraction).
  4. BM25 score distributions, and whether retriever *ranking* (which of BM25/TF-IDF/
     hybrid wins) agrees between the synthetic_only and real_only experiments already
     in data/retrieval_eval.csv (RAG.step12_evaluate_retrievers) -- no need to re-run
     retrieval for that part, just re-read what Phase 9 already computed.

Comparisons 2 and 3 scan every row of the real corpus (729 rows, ~10.8M words), not a
sample: unlike Phase 9's dense-vector corpus, a streaming word/figure count needs no
matrix, so the full real corpus is cheap enough to use directly and gives an honest
answer instead of one bounded by the same 6-company convenience sample everywhere.
Comparison 4 reuses that 6-company sample's chunks, same as Phase 9, since it's
retrieval over those specific QA pairs.

Output: data/gap_report.csv, one row per metric, plus the narrative writeup in
PLAN.md Phase 10.

Run from implementations/synthetic_10k (after method_2.step03_build_index and
RAG.step12_evaluate_retrievers):
    uv run python -m evals.step13_gap_report
"""

import csv
import io
import statistics
from collections import Counter

from method_2.config import RAW_DIR
from method_2.step09_validate_filings import extract_figures
from method_2.utils import setup_logging, write_atomic
from RAG.chunks import load_chunks
from RAG.retrieval import Bm25Retriever, tokenize


logger = setup_logging("gap_report")

INDEX_PATH = RAW_DIR.parent / "index.csv"
EVAL_PATH = RAW_DIR.parent / "retrieval_eval.csv"
QA_PATH = RAW_DIR.parent / "qa_pairs.csv"
REPORT_PATH = RAW_DIR.parent / "gap_report.csv"
REPORT_FIELDS = ["comparison", "metric", "real", "synthetic", "note"]

REAL_SAMPLE_TICKERS = {"AEP", "BEN", "USB", "CVS", "NOC", "CRWD"}
TOP_VOCAB_SIZE = 3000


def read_index() -> list[dict]:
    with open(INDEX_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section_length_rows(index_rows: list[dict]) -> list[dict]:
    """Median chars/words per item, real vs synthetic (comparison 1)."""
    by_item: dict[str, dict[str, list[int]]] = {}
    for row in index_rows:
        item = row["item"]
        by_item.setdefault(item, {"real": [], "synthetic": []})
        by_item[item][row["source"]].append(int(row["words"]))

    rows = []
    for item in sorted(by_item):
        real_words, synth_words = by_item[item]["real"], by_item[item]["synthetic"]
        if not real_words or not synth_words:
            continue
        real_median, synth_median = statistics.median(real_words), statistics.median(synth_words)
        rows.append(
            {
                "comparison": "section_length",
                "metric": f"item_{item}_median_words",
                "real": round(real_median),
                "synthetic": round(synth_median),
                "note": f"synthetic is {synth_median / real_median:.1%} of real's median length",
            },
        )
    return rows


def iter_texts(index_rows: list[dict], source: str):
    """Yield each matching row's full file text, read once (no chunking needed here)."""
    for row in index_rows:
        if row["source"] != source:
            continue
        yield (RAW_DIR.parent / row["path"]).read_text(encoding="utf-8")


def vocab_overlap_row(index_rows: list[dict]) -> dict:
    """Jaccard overlap of each corpus's top-N most frequent terms (comparison 2)."""
    real_counts: Counter = Counter()
    for text in iter_texts(index_rows, "real"):
        real_counts.update(tokenize(text))
    synth_counts: Counter = Counter()
    for text in iter_texts(index_rows, "synthetic"):
        synth_counts.update(tokenize(text))

    real_top = {term for term, _ in real_counts.most_common(TOP_VOCAB_SIZE)}
    synth_top = {term for term, _ in synth_counts.most_common(TOP_VOCAB_SIZE)}
    jaccard = len(real_top & synth_top) / len(real_top | synth_top)
    coverage = len(synth_top & real_top) / len(synth_top)  # share of synthetic's vocab real also uses
    return {
        "comparison": "vocabulary_overlap",
        "metric": f"jaccard_top_{TOP_VOCAB_SIZE}_terms",
        "real": round(jaccard, 3),
        "synthetic": round(coverage, 3),
        "note": "real=Jaccard(real_top, synth_top); synthetic=share of synthetic's top terms also in real's top terms",
    }


def number_distribution_rows(index_rows: list[dict]) -> list[dict]:
    """Density and magnitude of dollar/percent figures per 1,000 words (comparison 3)."""
    rows = []
    stats = {}
    for source in ("real", "synthetic"):
        dollars_per_1k, percents_per_1k = [], []
        dollar_magnitudes, percent_values = [], []
        for text in iter_texts(index_rows, source):
            n_words = max(len(text.split()), 1)
            dollars, percents = extract_figures(text)
            dollars_per_1k.append(len(dollars) / n_words * 1000)
            percents_per_1k.append(len(percents) / n_words * 1000)
            dollar_magnitudes.extend(abs(v) for v, _ in dollars if v)
            percent_values.extend(v for v, _ in percents)
        stats[source] = {
            "dollars_per_1k_words": statistics.median(dollars_per_1k) if dollars_per_1k else 0.0,
            "percents_per_1k_words": statistics.median(percents_per_1k) if percents_per_1k else 0.0,
            "median_dollar_magnitude_millions": statistics.median(dollar_magnitudes) / 1e6
            if dollar_magnitudes
            else 0.0,
            "median_percent_value": statistics.median(percent_values) if percent_values else 0.0,
        }

    for metric in stats["real"]:
        rows.append(
            {
                "comparison": "number_distribution",
                "metric": metric,
                "real": round(stats["real"][metric], 2),
                "synthetic": round(stats["synthetic"][metric], 2),
                "note": "",
            },
        )
    return rows


def bm25_score_rows() -> list[dict]:
    """Median top-1 BM25 score per question, synthetic vs the real 6-company sample (comparison 4)."""
    with open(QA_PATH, encoding="utf-8") as f:
        qa_pairs = list(csv.DictReader(f))

    synth_chunks = load_chunks(sources={"synthetic"})
    real_chunks = load_chunks(sources={"real"}, tickers=REAL_SAMPLE_TICKERS)
    synth_bm25 = Bm25Retriever([c.chunk_id for c in synth_chunks], [c.text for c in synth_chunks])
    real_bm25 = Bm25Retriever([c.chunk_id for c in real_chunks], [c.text for c in real_chunks])

    synth_scores, real_scores = [], []
    for q in qa_pairs:
        bm25, bucket = (synth_bm25, synth_scores) if q["source"] == "synthetic" else (real_bm25, real_scores)
        bucket.append(bm25.top_score(q["question"]))

    return [
        {
            "comparison": "bm25_score_distribution",
            "metric": "median_top1_score",
            "real": round(statistics.median(real_scores), 2) if real_scores else None,
            "synthetic": round(statistics.median(synth_scores), 2) if synth_scores else None,
            "note": f"n_real={len(real_scores)}, n_synthetic={len(synth_scores)}; not directly comparable across corpora of different sizes, shape matters more than the absolute value",
        },
    ]


def retriever_ranking_rows() -> list[dict]:
    """Does the best/worst retriever (by recall@10) agree between synthetic_only and real_only?"""
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    by_experiment: dict[str, list[dict]] = {}
    for row in eval_rows:
        by_experiment.setdefault(row["experiment"], []).append(row)

    synth_ranked = sorted(by_experiment.get("synthetic_only", []), key=lambda r: -float(r["recall_at_10"]))
    real_ranked = sorted(by_experiment.get("real_only", []), key=lambda r: -float(r["recall_at_10"]))
    if not synth_ranked or not real_ranked:
        return []

    synth_order = [r["retriever"] for r in synth_ranked]
    real_order = [r["retriever"] for r in real_ranked]
    agrees = synth_order[0] == real_order[0]
    return [
        {
            "comparison": "retriever_ranking_agreement",
            "metric": "best_retriever_by_recall_at_10",
            "real": real_order[0],
            "synthetic": synth_order[0],
            "note": (
                f"{'agrees' if agrees else 'disagrees'}: synthetic_only order {synth_order}, "
                f"real_only order {real_order}"
            ),
        },
    ]


def main() -> None:
    """Run all four Phase 10 comparisons and write data/gap_report.csv."""
    index_rows = read_index()

    rows = []
    rows.extend(section_length_rows(index_rows))
    rows.append(vocab_overlap_row(index_rows))
    rows.extend(number_distribution_rows(index_rows))
    rows.extend(bm25_score_rows())
    rows.extend(retriever_ranking_rows())

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(REPORT_PATH, buffer.getvalue())

    logger.info("Wrote %d rows to %s", len(rows), REPORT_PATH)
    for r in rows:
        logger.info(
            "%-28s %-32s real=%-10s synthetic=%-10s %s",
            r["comparison"],
            r["metric"],
            r["real"],
            r["synthetic"],
            r["note"],
        )


if __name__ == "__main__":
    main()
