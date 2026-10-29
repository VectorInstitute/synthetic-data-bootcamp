"""Phase 9: evaluate BM25, a real semantic embedding retriever, and their
reciprocal-rank-fusion hybrid against data/qa_pairs.csv, across three corpus scopes
(PLAN.md Phase 9): synthetic only, real only, and mixed.

"Real" here means a sample of REAL_CORPUS_SAMPLE_SIZE real companies (seeded,
reproducible) drawn from every real company RAG.step11_make_qa generated
deterministic financial-fact questions for (~49 companies) -- not the full set, and
not the original 6-company hand-authored sample (AEP, BEN, USB, CVS, NOC, CRWD)
either. The full real corpus is ~380,000 paragraphs; embedding and holding that in
memory at once pushed this development sandbox (11GB total, shared with an active
editor session) into swap and didn't finish in reasonable time even after fixing
Bm25Retriever's own scalability bug (see RAG.retrieval's module docstring) -- a
better-resourced machine could raise REAL_CORPUS_SAMPLE_SIZE or drop the sampling
entirely. 15 companies (~115,000 paragraphs as of this writing) is still roughly 2x
the original 6-company sample's size and, more importantly, draws its QA pairs from
2.5x as many real companies using the same deterministic generation RAG.step11_make_qa
uses for every synthetic company, rather than a small hand-picked set.

Output: data/retrieval_eval.csv, one row per (experiment, retriever).

Run from implementations/synthetic_10k (after RAG.step11_make_qa):
    uv run python -m RAG.step12_evaluate_retrievers
"""

import csv
import io
import random

from method_2.config import RAW_DIR
from method_2.utils import setup_logging, write_atomic

from RAG.chunks import Chunk, INDEX_PATH, load_chunks
from RAG.retrieval import (
    Bm25Retriever,
    SemanticRetriever,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


logger = setup_logging("evaluate_retrievers")

QA_PATH = RAW_DIR.parent / "qa_pairs.csv"
EMBEDDING_CACHE_DIR = RAW_DIR.parent / "embedding_cache"
REAL_CORPUS_SAMPLE_SIZE = 15
REAL_CORPUS_SEED = 7


def real_sample_tickers() -> set[str]:
    """Return a seeded, reproducible sample of real tickers (see module docstring)."""
    with open(INDEX_PATH, encoding="utf-8") as f:
        tickers = sorted({r["ticker"] for r in csv.DictReader(f) if r["source"] == "real"})
    return set(random.Random(REAL_CORPUS_SEED).sample(tickers, min(REAL_CORPUS_SAMPLE_SIZE, len(tickers))))
EVAL_PATH = RAW_DIR.parent / "retrieval_eval.csv"
EVAL_FIELDS = ["experiment", "retriever", "n_questions", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]

def load_qa_pairs() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def qa_for(qa_pairs: list[dict], experiment: str) -> list[dict]:
    if experiment == "synthetic_only":
        return [q for q in qa_pairs if q["source"] == "synthetic"]
    if experiment == "real_only":
        return [q for q in qa_pairs if q["source"] == "real"]
    return qa_pairs


def evaluate(qa_pairs: list[dict], rankings_by_qa: dict[str, list[str]], name: str) -> dict:
    n = len(qa_pairs)
    recall5, recall10, rr_sum, ndcg_sum = 0.0, 0.0, 0.0, 0.0
    for q in qa_pairs:
        ranking = rankings_by_qa[q["qa_id"]]
        gold = {q["gold_chunk_id"]}
        recall5 += recall_at_k(ranking, gold, 5)
        recall10 += recall_at_k(ranking, gold, 10)
        rr_sum += reciprocal_rank(ranking, gold)
        ndcg_sum += ndcg_at_k(ranking, gold, 10)
    return {
        "retriever": name,
        "n_questions": n,
        "recall_at_5": recall5 / n,
        "recall_at_10": recall10 / n,
        "mrr": rr_sum / n,
        "ndcg_at_10": ndcg_sum / n,
    }


def load_corpus(experiment: str) -> list[Chunk]:
    """Load only the index rows this experiment needs (see module docstring)."""
    if experiment == "synthetic_only":
        return load_chunks(sources={"synthetic"})
    tickers = real_sample_tickers()
    if experiment == "real_only":
        return load_chunks(sources={"real"}, tickers=tickers)
    return load_chunks(sources={"synthetic"}) + load_chunks(sources={"real"}, tickers=tickers)


def run_experiment(experiment: str, all_qa: list[dict]) -> list[dict]:
    chunks = load_corpus(experiment)
    chunk_id_set = {c.chunk_id for c in chunks}
    qa_pairs = [q for q in qa_for(all_qa, experiment) if q["gold_chunk_id"] in chunk_id_set]
    if not qa_pairs:
        logger.warning("%s: no QA pairs with a gold chunk in this corpus scope; skipping", experiment)
        return []

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    logger.info("%s: embedding %d chunks for the semantic retriever", experiment, len(texts))
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bm25 = Bm25Retriever(chunk_ids, texts)
    semantic = SemanticRetriever(chunk_ids, texts, cache_path=EMBEDDING_CACHE_DIR / f"retrieval_{experiment}.npz")

    bm25_rankings = {q["qa_id"]: bm25.rank(q["question"]) for q in qa_pairs}
    semantic_rankings = dict(
        zip([q["qa_id"] for q in qa_pairs], semantic.rank_many([q["question"] for q in qa_pairs]), strict=True),
    )
    hybrid_rankings = {
        qa_id: reciprocal_rank_fusion([bm25_rankings[qa_id], semantic_rankings[qa_id]]) for qa_id in bm25_rankings
    }

    rows = []
    for name, rankings in [("bm25", bm25_rankings), ("semantic", semantic_rankings), ("hybrid_rrf", hybrid_rankings)]:
        result = evaluate(qa_pairs, rankings, name)
        result["experiment"] = experiment
        rows.append(result)
    return rows


def main() -> None:
    """Run all three Phase 9 experiments and write data/retrieval_eval.csv."""
    all_qa = load_qa_pairs()
    logger.info("Loaded %d QA pairs", len(all_qa))

    rows = []
    for experiment in ["synthetic_only", "real_only", "mixed"]:
        rows.extend(run_experiment(experiment, all_qa))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EVAL_FIELDS)
    writer.writeheader()
    writer.writerows([{k: r[k] for k in EVAL_FIELDS} for r in rows])
    write_atomic(EVAL_PATH, buffer.getvalue())

    logger.info("Wrote %d rows to %s", len(rows), EVAL_PATH)
    for r in rows:
        logger.info(
            "%-15s %-12s n=%-4d recall@5=%.2f recall@10=%.2f mrr=%.2f ndcg@10=%.2f",
            r["experiment"],
            r["retriever"],
            r["n_questions"],
            r["recall_at_5"],
            r["recall_at_10"],
            r["mrr"],
            r["ndcg_at_10"],
        )


if __name__ == "__main__":
    main()
