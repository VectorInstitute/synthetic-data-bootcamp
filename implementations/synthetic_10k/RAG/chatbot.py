"""RAG chatbot demo over the synthetic 10-K corpus (ad hoc, built on top of
Phase 9's retrievers -- see RAG.retrieval and RAG.step12_evaluate_retrievers).
Also used by RAG.chatbot_server, which puts this on a small web UI alongside
dashboard/index.html.

Retrieval: BM25 + the real semantic embedding retriever (fastembed /
minishlab/potion-retrieval-32M), fused via reciprocal rank fusion -- the exact
hybrid_rrf pipeline step12 evaluates, over the 20 synthetic companies
(data/synthetic), chunked the same way RAG.chunks chunks them for step11/step12, so
a retrieved chunk_id lines up with data/qa_pairs.csv's gold_chunk_id.

Generation is optional and provider-pluggable, picked at startup from whichever API
key is set in the environment (or in a gitignored .env file next to this project,
see src.utils.load_dotenv): DeepSeek first (DEEPSEEK_API_KEY), then Claude
(ANTHROPIC_API_KEY), then retrieval-only -- same "retrieval doesn't need a key;
generation does" fallback as step08_generate_filings.

Two CLI modes:
    PYTHONPATH=dylan_pipeline python -m RAG.chatbot            interactive chat loop
    PYTHONPATH=dylan_pipeline python -m RAG.chatbot --demo     scripted scenario-analysis walkthrough: one
                                      analytical question per non-baseline scenario
                                      company (13 of data/qa_pairs.csv's 15 analytical
                                      rows span 13 companies), showing whether hybrid
                                      retrieval actually lands on the scenario-
                                      grounded passage, alongside each company's own
                                      scenario summary from its spec.json.
See RAG.chatbot_server for the browser UI covering the same two modes.
"""

import argparse
import csv
import json
import os
from collections.abc import Callable

import anthropic
import requests

from RAG.chunks import Chunk, load_chunks
from src.config import PROJECT_DIR, RAW_DIR
from RAG.retrieval import Bm25Retriever, SemanticRetriever, reciprocal_rank_fusion
from src.utils import load_dotenv, setup_logging


logger = setup_logging("chatbot")

DATA_DIR = RAW_DIR.parent
SYNTHETIC_DIR = DATA_DIR / "synthetic"
QA_PATH = DATA_DIR / "qa_pairs.csv"
ENV_PATH = PROJECT_DIR / ".env"

CLAUDE_MODEL = "claude-sonnet-5-5"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_TOKENS = 1024
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a RAG assistant answering questions about a synthetic SEC 10-K corpus "
    "of fictional companies. Answer ONLY using the numbered passages given in the "
    "user message -- if they don't contain the answer, say so plainly rather than "
    "guessing or using outside knowledge. Cite which passage(s) you used by their "
    "[n] label. Be concise."
)

Generator = Callable[[str, list[Chunk]], str]


def build_index() -> tuple[dict[str, Chunk], Bm25Retriever, SemanticRetriever]:
    """Load the 20 synthetic companies' chunks and build BM25 + semantic retrievers."""
    chunks = load_chunks(sources={"synthetic"})
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    logger.info("Indexing %d chunks from %d synthetic companies", len(chunks), len({c.ticker for c in chunks}))
    bm25 = Bm25Retriever(chunk_ids, texts)
    semantic = SemanticRetriever(chunk_ids, texts)
    return {c.chunk_id: c for c in chunks}, bm25, semantic


def retrieve(
    question: str,
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    k: int = TOP_K,
) -> list[Chunk]:
    """Rank with BM25 and semantic separately, fuse via RRF, return the top-k chunks."""
    fused = reciprocal_rank_fusion([bm25.rank(question), semantic.rank(question)])
    return [chunks_by_id[cid] for cid in fused[:k]]


def format_context(passages: list[Chunk]) -> str:
    blocks = [
        f"[{i}] {c.company_name} ({c.ticker}), FY{c.fiscal_year}, Item {c.item}:\n{c.text}"
        for i, c in enumerate(passages, start=1)
    ]
    return "\n\n".join(blocks)


def synthesize_claude(client: "anthropic.Anthropic", question: str, passages: list[Chunk]) -> str:
    prompt = f"Question: {question}\n\nPassages:\n{format_context(passages)}"
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def synthesize_deepseek(api_key: str, question: str, passages: list[Chunk]) -> str:
    """DeepSeek's chat completions API is OpenAI-compatible; plain `requests` is
    enough, no SDK needed for one call shape."""
    prompt = f"Question: {question}\n\nPassages:\n{format_context(passages)}"
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_generator() -> tuple[str, Generator | None]:
    """Pick a generation backend from the environment: DeepSeek first, then Claude,
    then retrieval-only (no key set for either). Reads .env (gitignored) if present,
    without overriding an already-exported shell variable."""
    load_dotenv(ENV_PATH)

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        return "deepseek", lambda question, passages: synthesize_deepseek(deepseek_key, question, passages)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        client = anthropic.Anthropic(api_key=anthropic_key)
        return "claude", lambda question, passages: synthesize_claude(client, question, passages)

    return "retrieval_only", None


def answer_question(
    question: str,
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    provider: str,
    generate: Generator | None,
) -> tuple[str, list[Chunk], str]:
    passages = retrieve(question, chunks_by_id, bm25, semantic)
    if generate is not None:
        return generate(question, passages), passages, provider
    return passages[0].text, passages, "retrieval_only"


def print_sources(passages: list[Chunk]) -> None:
    print("Sources:")
    for i, c in enumerate(passages, start=1):
        print(f"  [{i}] {c.ticker} FY{c.fiscal_year} item {c.item} ({c.chunk_id})")


def source_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "ticker": c.ticker,
        "company_name": c.company_name,
        "fiscal_year": c.fiscal_year,
        "item": c.item,
        "text": c.text,
    }


def load_scenario_info() -> dict[str, dict]:
    """ticker -> {"company_name", "type", "summary"}, read from each spec.json."""
    info = {}
    for spec_path in sorted(SYNTHETIC_DIR.glob("*/*/spec.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        scenario = spec.get("scenario") or {}
        info[spec["fictional_id"]] = {
            "company_name": spec["fictional_name"],
            "type": scenario.get("type", "baseline"),
            "summary": scenario.get("summary", ""),
        }
    return info


def load_analytical_qa() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["style"] == "analytical" and r["source"] == "synthetic"]


def demo_rows(
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    provider: str,
    generate: Generator | None,
) -> list[dict]:
    """One row per scenario company: does hybrid retrieval land on the scenario-
    grounded passage, not just a keyword-overlapping one? Shared by the CLI --demo
    and RAG.chatbot_server's GET /api/demo, so both show exactly the same data."""
    scenario_info = load_scenario_info()
    seen_tickers: set[str] = set()
    rows = []

    for row in load_analytical_qa():
        ticker = row["ticker"]
        if ticker in seen_tickers:
            continue  # one question per company caps this at 13 companies, not 15 rows
        seen_tickers.add(ticker)
        info = scenario_info.get(ticker, {})

        passages = retrieve(row["question"], chunks_by_id, bm25, semantic)
        hit = row["gold_chunk_id"] in {c.chunk_id for c in passages}
        if generate is not None:
            answer, answer_provider = generate(row["question"], passages), provider
        else:
            answer, answer_provider = row["answer"], "reference"

        rows.append(
            {
                "ticker": ticker,
                "company_name": info.get("company_name", ticker),
                "scenario_type": info.get("type", "unknown"),
                "scenario_summary": info.get("summary", ""),
                "question": row["question"],
                "gold_chunk_id": row["gold_chunk_id"],
                "hit": hit,
                "answer": answer,
                "answer_provider": answer_provider,
                "sources": [source_dict(c) for c in passages],
            },
        )
    return rows


def run_demo(
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    provider: str,
    generate: Generator | None,
) -> None:
    print("=" * 78)
    print("SCENARIO ANALYSIS DEMO")
    print("20 synthetic companies: 13 carry a deliberate scenario (M&A, margin crisis,")
    print("cash burn, regulatory pressure, ...); 7 are a baseline control group.")
    print("Each block: one company's scenario, an analytical question about it, and")
    print("whether hybrid (BM25 + semantic, RRF-fused) retrieval finds the grounded passage.")
    print("=" * 78)

    rows = demo_rows(chunks_by_id, bm25, semantic, provider, generate)
    for row in rows:
        print(f"\n--- {row['company_name']} ({row['ticker']}) -- scenario: {row['scenario_type']} ---")
        if row["scenario_summary"]:
            print(f"Scenario: {row['scenario_summary']}")
        print(f"\nQ: {row['question']}")
        print(
            f"Retrieval: {'HIT' if row['hit'] else 'MISS'} -- gold chunk {row['gold_chunk_id']} is "
            f"{'' if row['hit'] else 'NOT '}in the top-{len(row['sources'])} fused ranking",
        )
        label = "Reference answer (data/qa_pairs.csv)" if row["answer_provider"] == "reference" else "A"
        print(f"\n{label}: {row['answer']}")
        print()
        print("Sources:")
        for i, s in enumerate(row["sources"], start=1):
            print(f"  [{i}] {s['ticker']} FY{s['fiscal_year']} item {s['item']} ({s['chunk_id']})")

    print("\n" + "=" * 78)
    print(f"{len(rows)}/13 scenario companies covered.")
    if provider == "retrieval_only":
        print("Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY to see a synthesized answer instead of the reference one.")


def run_chat(
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    provider: str,
    generate: Generator | None,
) -> None:
    print("SEC 10-K synthetic corpus chatbot -- ask about any of the 20 fictional companies.")
    print("Type 'quit' or 'exit' to leave.\n")
    print(f"Generation backend: {provider}" + (" (no API key set -- showing top-retrieved passage)" if provider == "retrieval_only" else ""))
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer, passages, _ = answer_question(question, chunks_by_id, bm25, semantic, provider, generate)
        print(f"\nBot: {answer}\n")
        print_sources(passages)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted scenario-analysis walkthrough instead of an interactive chat.",
    )
    args = parser.parse_args()

    chunks_by_id, bm25, semantic = build_index()
    provider, generate = get_generator()
    if provider == "retrieval_only":
        logger.warning("No DEEPSEEK_API_KEY or ANTHROPIC_API_KEY set -- running retrieval-only (no synthesized answers)")
    else:
        logger.info("Generation backend: %s", provider)

    if args.demo:
        run_demo(chunks_by_id, bm25, semantic, provider, generate)
    else:
        run_chat(chunks_by_id, bm25, semantic, provider, generate)


if __name__ == "__main__":
    main()
