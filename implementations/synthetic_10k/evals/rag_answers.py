r"""End-to-end RAG answers to the shared fact questions, per set.

For each set, every company's ten standard questions (qa.fact_questions, the same
bank the scorecard uses) go through the RAG chatbot's pipeline: hybrid retrieval
(BM25 + semantic embeddings, fused by reciprocal rank) over all of the set's chunks,
then an LLM answers from the top passages only. The answer is scored against the
company's statements with the same tolerance as the scorecard (dollars within 1%,
percentages within 0.25 points). Every question counts, so a set whose filings never
state a fact scores a miss on it.

Metrics per set (summary.csv):
- rag_answer_correct_pct: share of questions answered with the true value.
- rag_abstain_pct: share the model said it couldn't answer from the passages.
- retrieval_hit_pct: share with a passage stating the answer in the top K.

Real filings get the same questions from their XBRL figures, over the same random
sample the scorecard uses (--real-sample, --seed).

Needs the Vector proxy (OPENAI_API_KEY, SYNBENCH_BASE_URL), so run it on Coder from
implementations/synthetic_10k:
    ../../.venv/bin/python -m evals.rag_answers --out evals/results/day3_rag \
        --set m1_gemini=data/gemini20/method_1/synthetic/filings \
        --set m2_gemini=data/gemini20/method_2/synthetic
"""

import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from method_1.narrative import UsageClient  # noqa: E402
from method_2.step09_validate_filings import closest_match  # noqa: E402
from method_2.utils import setup_logging  # noqa: E402
from qa.fact_questions import make_questions, real_facts, synthetic_facts  # noqa: E402
from RAG.chunks import Chunk  # noqa: E402
from RAG.retrieval import Bm25Retriever, SemanticRetriever, reciprocal_rank_fusion  # noqa: E402

from evals.compare_sets import (  # noqa: E402
    DEFAULT_REAL_DIR,
    DEFAULT_REAL_FINANCIALS,
    company_chunks,
    item_texts,
    load_real_filings,
    load_real_records,
    load_set,
)
from evals.section_scorecard import load_real_metrics  # noqa: E402


logger = setup_logging("rag_answers")

TOP_K = 5
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You answer questions about companies using only the 10-K passages provided. "
    'Reply with JSON only: {"value": <number or null>, "unit": "usd" | "percent"}. '
    "Give dollar amounts in full US dollars (e.g. 1250000000 for $1.25 billion) and "
    "percentages as a number of percent (e.g. 12.5). If the passages don't state the "
    'answer, reply {"value": null}.'
)


def parse_answer(text: str) -> float | None:
    """The numeric value from the model's JSON reply, or None for no answer."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        value = json.loads(match.group(0)).get("value") if match else None
    except (json.JSONDecodeError, AttributeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def format_context(passages: list[Chunk]) -> str:
    """Passages as the chatbot shows them to the model."""
    return "\n\n".join(
        f"[{i}] {c.company_name}, FY{c.fiscal_year}, Item {c.item}:\n{c.text}" for i, c in enumerate(passages, start=1)
    )


def answer_all(
    name: str,
    questions: list[dict],
    chunks: list[Chunk],
    client: UsageClient,
    workers: int,
) -> list[dict]:
    """Retrieve, answer and score every question over one corpus."""
    by_id = {c.chunk_id: c for c in chunks}
    ids, texts = list(by_id), [c.text for c in by_id.values()]
    bm25 = Bm25Retriever(ids, texts)
    semantic = SemanticRetriever(ids, texts)

    def one(q: dict) -> dict:
        fused = reciprocal_rank_fusion([bm25.rank(q["question"]), semantic.rank(q["question"])])[:TOP_K]
        gold = set(filter(None, q["gold_chunk_ids"].split(";")))
        prompt = f"Question: {q['question']}\n\nPassages:\n{format_context([by_id[c] for c in fused])}"
        try:
            reply = client.chat(SYSTEM_PROMPT, prompt, max_tokens=200).text
        except Exception as exc:  # noqa: BLE001 - one failed call shouldn't stop the set
            logger.warning("%s %s: %s", name, q["question"], exc)
            reply = ""
        value = parse_answer(reply)
        return {
            "set": name,
            **{k: q[k] for k in ("company_id", "fact", "question", "answer", "is_percent")},
            "retrieval_hit": bool(gold & set(fused)),
            "rag_value": value,
            "correct": value is not None and closest_match(value, {q["answer"]}, relative=not q["is_percent"]),
            "reply": reply.strip()[:200],
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, questions))


def summary(name: str, rows: list[dict]) -> dict:
    """One summary row for a set."""
    n = len(rows)
    return {
        "set": name,
        "n_questions": n,
        "rag_answer_correct_pct": round(sum(r["correct"] for r in rows) / n * 100, 1) if n else None,
        "rag_abstain_pct": round(sum(r["rag_value"] is None for r in rows) / n * 100, 1) if n else None,
        "retrieval_hit_pct": round(sum(r["retrieval_hit"] for r in rows) / n * 100, 1) if n else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Answer every set's questions and write answers.csv and summary.csv."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=PATH", help="a synthetic set; repeat")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--real-financials", type=Path, default=DEFAULT_REAL_FINANCIALS)
    parser.add_argument("--real-sample", type=int, default=20, help="0 skips real filings")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "rag_answers")
    args = parser.parse_args()

    client = UsageClient(args.model, temperature=0.0)
    answers, summaries = [], []

    if args.real_sample:
        records = load_real_records(args.real_financials)
        filings = load_real_filings(args.real_dir, records)
        sample = random.Random(args.seed).sample(filings, min(args.real_sample, len(filings)))
        metrics = load_real_metrics(args.real_financials)
        chunks, questions = [], []
        for f in sample:
            mine = company_chunks(f.ticker, f.name, f.fiscal_year, f.paths, "real")
            chunks.extend(mine)
            facts = real_facts(metrics.get((f.ticker, f.fiscal_year), {}))
            questions.extend(make_questions(f.ticker, f.name, f.fiscal_year, facts, mine))
        logger.info("real: %d questions over %d chunks", len(questions), len(chunks))
        rows = answer_all("real", questions, chunks, client, args.workers)
        answers += rows
        summaries.append(summary("real", rows))

    for spec in args.set:
        name, path = spec.split("=", 1)
        chunks, questions = [], []
        for c in load_set(name, Path(path), args.out):
            texts = item_texts(c.folder)
            mine = company_chunks(
                c.company_id,
                c.name,
                c.fiscal_year,
                {i: c.folder / f"item_{i}.txt" for i in texts},
                "synthetic",
            )
            chunks.extend(mine)
            questions.extend(make_questions(c.company_id, c.name, c.fiscal_year, synthetic_facts(c.financials), mine))
        logger.info("%s: %d questions over %d chunks", name, len(questions), len(chunks))
        rows = answer_all(name, questions, chunks, client, args.workers)
        answers += rows
        summaries.append(summary(name, rows))
        logger.info("%s", summaries[-1])

    write_csv(args.out / "answers.csv", answers)
    write_csv(args.out / "summary.csv", summaries)
    logger.info("Wrote %s", args.out / "summary.csv")


if __name__ == "__main__":
    main()
