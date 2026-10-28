"""Phase 9: build data/qa_pairs.csv, one row per question with an exact answer and a
gold passage (chunk_id, see RAG.chunks).

Synthetic questions (20 companies): template-based and fully deterministic. Every
answer is a value already in that company's financials.json or spec.json scenario
facts; the gold chunk and the exact answer span are found by scanning the company's
own chunked narrative with the same figure-extraction and tolerance-matching logic
method_2.step09_validate_filings already uses for the numeric-consistency gate (so a
financial-fact question's answer is guaranteed to be a verbatim substring of its
gold chunk by construction, not by a separate check).

Real questions (6 companies): hand-authored from a short, grep-verified excerpt of
each filing (not the whole filing -- some run to 100,000+ words), matching the
template-ticker/fiscal-year sample Phase 7's model-judged review already used
(method_2.step09_validate_filings's docstring). Each answer is checked against the
actual chunked text before being written out; the script hard-fails if one isn't
found verbatim, which is what PLAN.md Phase 9 means by "check that each answer span
appears in the text."

A known simplification: each question records exactly one gold_chunk_id, even on
the rare case where the same figure is restated in more than one chunk (e.g. both a
prose sentence and a financial-statement table row). Retrieval of the other,
unrecorded chunk would be scored as a miss. See PLAN.md Phase 9.

Every question carries a `style`: "factual" (a single figure or named entity -- the
answer is the whole point) or "analytical" (why something happened, or what it
implies -- the answer is still a short exact span, but finding it requires reading
a passage that connects a cause to an effect, not just locating a labeled figure).
Analytical questions are grounded the same way: a chunk search, verified at
generation time, never a paraphrase. See PLAN.md Phase 9 for which chunks turned out
to state cause and effect together (no new grounding invented to force this).

Run from implementations/synthetic_10k (after method_2.step03_build_index):
    uv run python -m RAG.step11_make_qa
"""

import csv
import io
import json
import re
from typing import Any

from method_2.config import RAW_DIR
from method_2.step09_validate_filings import (
    SCENARIO_ANCHOR_FACT,
    SCENARIO_KEYWORDS,
    closest_match,
    extract_figures,
    to_float,
)
from method_2.utils import setup_logging, write_atomic

from RAG.chunks import Chunk, load_chunks


logger = setup_logging("make_qa")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
QA_PATH = RAW_DIR.parent / "qa_pairs.csv"
QA_FIELDS = ["qa_id", "source", "ticker", "fiscal_year", "item", "style", "question", "answer", "gold_chunk_id"]

# (path through financials.json, question template, is_percent)
FINANCIAL_FACTS = [
    (("income_statement", "revenues"), "What were {name}'s total revenues for fiscal {fy}?", False),
    (("income_statement", "net_income"), "What was {name}'s net income for fiscal {fy}?", False),
    (
        ("balance_sheet", "assets", "total_assets"),
        "What were {name}'s total assets as of the end of fiscal {fy}?",
        False,
    ),
]

# One "why" question per keyword-grounded scenario type (method_2.step06b_add_scenarios),
# phrased so answering it requires reading the passage that connects the scenario's
# cause to its effect, not just matching a category label. The two anchor-fact types
# (mergers_and_acquisitions, product_launch) are handled separately below since their
# answer is a proper noun, not a matched keyword.
SCENARIO_KEYWORD_QUESTIONS = {
    "competitive_displacement": "Why did {name}'s operating margin come under pressure during fiscal {fy}?",
    "margin_crisis": "Why did {name}'s margins compress to historically thin levels in fiscal {fy}?",
    "capital_structure": "What is driving {name}'s elevated financial leverage?",
    "cash_burn": "Why did {name} record an operating loss in fiscal {fy} rather than sustained profitability?",
    "regulatory_rate_case": "Why did {name} fall short of recovering its full requested costs in fiscal {fy}?",
    "weather_event": "Why did {name}'s fiscal {fy} capital spending include incremental restoration costs?",
    "regulatory_pricing_pressure": "Why did regulatory changes affect {name}'s margins during fiscal {fy}?",
    "regulatory_consent_order": "Why did {name}'s compliance-related costs increase during fiscal {fy}?",
}

# For these two scenario types, the narrative states the scenario's driver and its
# financial-statement outcome in the *same* paragraph (verified for BIDZ and WJE, the
# only two companies with these types, at generation time -- not assumed). That makes
# a richer question answerable from one passage: the exact phrase here is the
# qualitative severity language the narrative uses, not a bare number, so answering it
# requires reading past the figures to the sentence that characterizes them.
SEVERITY_PHRASE_QUESTIONS = {
    "margin_crisis": ("How severe does {name} describe its fiscal {fy} margin compression?", "thinnest margins"),
    "cash_burn": (
        "What does {name}'s fiscal {fy} operating loss reflect about its stage of growth, per its MD&A?",
        "ahead of sustained profitability",
    ),
}

# Verified present in the risk-factor item of every company with the matching
# scenario type -- not guaranteed by construction, checked directly in the narrative,
# same as every other grounding in this file.
ACQUISITION_RISK_PHRASE = "may not realize the anticipated benefits"
PRODUCT_LAUNCH_RISK_PHRASE = "may not realize the expected benefits"

# Hand-authored from the real company behind each sampled fictional company's template
# (ticker, fiscal_year) -- the same 6-company sample used for Phase 7's model-judged
# review. Each answer is a short, grep-verified verbatim excerpt, not a paraphrase.
REAL_QA = [
    ("AEP", 2025, "1", "How many employees did AEP's subsidiaries have as of December 31, 2025?", "17,581 employees"),
    (
        "AEP",
        2025,
        "1",
        "In what year was American Electric Power incorporated, and in which state?",
        "incorporated under the laws of the State of New York in 1906",
    ),
    (
        "BEN",
        2024,
        "7",
        "What was Franklin Resources' total assets under management (AUM) at September 30, 2024?",
        "$1,678.6 billion",
    ),
    (
        "BEN",
        2024,
        "7",
        "How much did Franklin Resources' AUM growth reflect from the acquisition of Putnam Investments?",
        "$148.3 billion from the acquisition of Putnam Investments",
    ),
    (
        "USB",
        2025,
        "7",
        "How much net income did U.S. Bancorp earn in 2025 compared with 2024?",
        "$7.6 billion in 2025 compared with $6.3 billion in 2024",
    ),
    (
        "USB",
        2025,
        "7",
        "What was U.S. Bancorp's diluted earnings per common share in 2025, and how much did it increase from 2024?",
        "$4.62 in 2025, representing a 21.9 percent increase",
    ),
    (
        "CVS",
        2024,
        "7",
        "How many retail locations did CVS Health have as of December 31, 2024?",
        "more than 9,000 retail locations",
    ),
    (
        "CVS",
        2024,
        "7",
        "Approximately how many pharmacy benefits manager plan members did CVS Health serve?",
        "approximately 90 million plan members",
    ),
    (
        "NOC",
        2025,
        "7",
        "How much cash did Northrop Grumman receive for the sale of its Immersive Mission Solutions (IMS) operating unit?",
        "$333 million in cash",
    ),
    (
        "NOC",
        2025,
        "7",
        "When did Northrop Grumman complete the divestiture of its training services business?",
        "May 24, 2025",
    ),
    ("CRWD", 2025, "1", "In what year was CrowdStrike founded?", "Founded in 2011"),
    ("CRWD", 2025, "7", "On what date does CrowdStrike's fiscal year end?", "fiscal years ended January 31, 2025"),
]


def get_field(d: Any, path: tuple[str, ...]) -> float | None:
    for key in path:
        if key not in d:
            return None
        d = d[key]
    return float(d)


def extract_answer_token(ctx: str, target_value: float, *, is_percent: bool) -> str:
    """Pull the short span matching `target_value` out of `ctx`.

    For prose, `ctx` from extract_figures is already the exact figure token (e.g.
    "$16,233.4 million"). For a table row, `ctx` is the whole "label | value[ | pct]"
    line; this picks out just the one cell that matches, not the whole row.
    """
    if " | " not in ctx:
        return ctx
    for part in [p.strip() for p in ctx.split(" | ")[1:]]:
        value = to_float(part)
        if value is None:
            continue
        scaled = value if is_percent else value * 1e6
        if closest_match(scaled, {target_value}, relative=not is_percent):
            return part
    return ctx


def find_figure_chunk(value: float, chunks: list[Chunk], *, is_percent: bool) -> tuple[Chunk, str] | None:
    """Return (chunk, exact answer span) for the first chunk whose text states `value`."""
    for chunk in chunks:
        dollars, percents = extract_figures(chunk.text)
        for v, ctx in percents if is_percent else dollars:
            if closest_match(v, {value}, relative=not is_percent):
                return chunk, extract_answer_token(ctx, value, is_percent=is_percent)
    return None


def find_phrase_chunk(phrase: str, chunks: list[Chunk]) -> Chunk | None:
    """Return the first chunk whose text contains `phrase` (case-insensitive)."""
    lowered = phrase.lower()
    return next((c for c in chunks if lowered in c.text.lower()), None)


def qa_row(
    source: str,
    ticker: str,
    fy: int,
    item: str,
    style: str,
    question: str,
    answer: str,
    gold_chunk_id: str,
) -> dict:
    return {
        "source": source,
        "ticker": ticker,
        "fiscal_year": fy,
        "item": item,
        "style": style,
        "question": question,
        "answer": answer,
        "gold_chunk_id": gold_chunk_id,
    }


def scenario_qa_rows(spec: dict, company_chunks: list[Chunk]) -> list[dict]:
    """Questions grounded in spec.json's scenario (method_2.step06b_add_scenarios); [] for baseline."""
    scenario = spec.get("scenario", {})
    scenario_type = scenario.get("type", "baseline")
    if scenario_type == "baseline":
        return []
    name, fy, ticker, facts = spec["fictional_name"], spec["fiscal_year"], spec["fictional_id"], scenario["facts"]
    rows = []

    if scenario_type in SCENARIO_ANCHOR_FACT:
        anchor_value = facts[SCENARIO_ANCHOR_FACT[scenario_type]]
        chunk = next((c for c in company_chunks if anchor_value in c.text), None)
        if chunk:
            question = (
                f"What company did {name} acquire during fiscal {fy}?"
                if scenario_type == "mergers_and_acquisitions"
                else f"What new product did {name} launch during fiscal {fy}?"
            )
            rows.append(qa_row("synthetic", ticker, fy, chunk.item, "factual", question, anchor_value, chunk.chunk_id))
        if scenario_type == "mergers_and_acquisitions":
            found = find_figure_chunk(float(facts["purchase_price"]), company_chunks, is_percent=False)
            if found:
                price_chunk, price_answer = found
                question = f"How much did {name} pay to acquire {facts['acquired_name']}?"
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        price_chunk.item,
                        "factual",
                        question,
                        price_answer,
                        price_chunk.chunk_id,
                    ),
                )
            risk_chunk = find_phrase_chunk(ACQUISITION_RISK_PHRASE, company_chunks)
            if risk_chunk:
                question = (
                    f"What integration risk does {name} face following its acquisition of {facts['acquired_name']}?"
                )
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        risk_chunk.item,
                        "analytical",
                        question,
                        ACQUISITION_RISK_PHRASE,
                        risk_chunk.chunk_id,
                    ),
                )
        if scenario_type == "product_launch":
            risk_chunk = find_phrase_chunk(PRODUCT_LAUNCH_RISK_PHRASE, company_chunks)
            if risk_chunk:
                question = f"What risk does {name} face related to its launch of {facts['product_name']}?"
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        risk_chunk.item,
                        "analytical",
                        question,
                        PRODUCT_LAUNCH_RISK_PHRASE,
                        risk_chunk.chunk_id,
                    ),
                )
        return rows

    template = SCENARIO_KEYWORD_QUESTIONS.get(scenario_type)
    keywords = SCENARIO_KEYWORDS.get(scenario_type, [])
    if not template:
        return rows
    for chunk in company_chunks:
        lowered = chunk.text.lower()
        matched_keyword = next((k for k in keywords if k in lowered), None)
        if matched_keyword:
            m = re.search(re.escape(matched_keyword) + r"\w*", chunk.text, re.IGNORECASE)
            answer = m.group(0) if m else matched_keyword
            rows.append(
                qa_row(
                    "synthetic",
                    ticker,
                    fy,
                    chunk.item,
                    "analytical",
                    template.format(name=name, fy=fy),
                    answer,
                    chunk.chunk_id,
                ),
            )

            severity = SEVERITY_PHRASE_QUESTIONS.get(scenario_type)
            if severity:
                severity_template, phrase = severity
                m2 = re.search(re.escape(phrase), chunk.text, re.IGNORECASE)
                if m2:
                    rows.append(
                        qa_row(
                            "synthetic",
                            ticker,
                            fy,
                            chunk.item,
                            "analytical",
                            severity_template.format(name=name, fy=fy),
                            m2.group(0),
                            chunk.chunk_id,
                        ),
                    )
            break
    return rows


def synthetic_qa_rows(spec_path, all_chunks: list[Chunk]) -> list[dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    ticker, fy, name = spec["fictional_id"], spec["fiscal_year"], spec["fictional_name"]
    company_chunks = [c for c in all_chunks if c.source == "synthetic" and c.ticker == ticker and c.fiscal_year == fy]

    rows = []
    for path, template, is_percent in FINANCIAL_FACTS:
        value = get_field(financials, path)
        if value is None:
            continue
        found = find_figure_chunk(value, company_chunks, is_percent=is_percent)
        if found:
            chunk, answer = found
            rows.append(
                qa_row(
                    "synthetic",
                    ticker,
                    fy,
                    chunk.item,
                    "factual",
                    template.format(name=name, fy=fy),
                    answer,
                    chunk.chunk_id,
                ),
            )

    revenue = get_field(financials, ("income_statement", "revenues"))
    net_income = get_field(financials, ("income_statement", "net_income"))
    operating_income = financials["income_statement"].get("operating_income")
    if revenue:
        found = (
            find_figure_chunk(net_income / revenue * 100, company_chunks, is_percent=True)
            if net_income is not None
            else None
        )
        if found:
            chunk, answer = found
            rows.append(
                qa_row(
                    "synthetic",
                    ticker,
                    fy,
                    chunk.item,
                    "factual",
                    f"What was {name}'s net margin for fiscal {fy}?",
                    answer,
                    chunk.chunk_id,
                ),
            )
        if operating_income is not None:
            found = find_figure_chunk(operating_income / revenue * 100, company_chunks, is_percent=True)
            if found:
                chunk, answer = found
                question = f"What was {name}'s operating margin for fiscal {fy}?"
                rows.append(qa_row("synthetic", ticker, fy, chunk.item, "factual", question, answer, chunk.chunk_id))

    rows.extend(scenario_qa_rows(spec, company_chunks))
    return rows


def real_qa_rows(chunks: list[Chunk]) -> list[dict]:
    by_key: dict[tuple[str, int, str], list[Chunk]] = {}
    for c in chunks:
        by_key.setdefault((c.ticker, c.fiscal_year, c.item), []).append(c)

    rows = []
    for ticker, fy, item, question, answer in REAL_QA:
        candidates = by_key.get((ticker, fy, item), [])
        chunk = next((c for c in candidates if answer in c.text), None)
        if chunk is None:
            raise SystemExit(f"Real QA answer not found verbatim: {ticker} FY{fy} item {item}: {answer!r}")
        rows.append(qa_row("real", ticker, fy, item, "factual", question, answer, chunk.chunk_id))
    return rows


def main() -> None:
    """Build data/qa_pairs.csv from the synthetic corpus and a 6-company real sample."""
    synthetic_chunks = load_chunks(sources={"synthetic"})

    rows = []
    for spec_path in sorted(SYNTHETIC_DIR.glob("*/*/spec.json")):
        rows.extend(synthetic_qa_rows(spec_path, synthetic_chunks))

    real_sample_tickers = {ticker for ticker, *_ in REAL_QA}
    real_chunks = load_chunks(sources={"real"}, tickers=real_sample_tickers)
    rows.extend(real_qa_rows(real_chunks))

    for i, row in enumerate(rows):
        row["qa_id"] = f"qa_{i:04d}"

    buffer_rows = [{field: row[field] for field in QA_FIELDS} for row in rows]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=QA_FIELDS)
    writer.writeheader()
    writer.writerows(buffer_rows)
    write_atomic(QA_PATH, buffer.getvalue())

    n_synthetic = sum(1 for r in rows if r["source"] == "synthetic")
    n_real = sum(1 for r in rows if r["source"] == "real")
    logger.info("Wrote %d QA pairs to %s (%d synthetic, %d real)", len(rows), QA_PATH, n_synthetic, n_real)


if __name__ == "__main__":
    main()
