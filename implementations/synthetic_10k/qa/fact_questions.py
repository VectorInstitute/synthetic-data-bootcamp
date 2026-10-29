"""One fixed bank of financial-fact questions, asked the same way of every set.

Every company gets the same ten questions, whichever method wrote its filing. The
answer to each comes from the company's statements (financials.json for a synthetic
company, XBRL headline figures for a real one), never from the text, so a set that
never states its revenue still gets the revenue question and scores a miss on it.
That keeps the denominator fixed at ten per company and makes sets comparable.

Per question this records the true value and every chunk whose text states it (the
same figure matching as method_2.step09's numeric gate: dollars within 1%, percentages
within 0.25 points). A question with no such chunk is unanswerable from the corpus.

Scores, for a list of questions and the chunks they are searched over:
- fact_coverage: share of questions whose answer is stated in at least one chunk.
- fact_recall_at_10: share of questions with a stating chunk in BM25's top 10.
  Unanswerable questions count as misses, so recall can never exceed coverage.

And one score on the text itself, label-aware, with no chance matches:
- fact_correct_pct: of the figures the text attaches to one of the ten facts' labels
  ("revenues of $X", "net margin was Y%", "Total assets | Z" table rows), the share
  equal to that fact's current or prior-year value or their difference. A figure
  next to "revenue" must be the revenue line, not any number in the statements.

Run from implementations/synthetic_10k; used by evals.section_scorecard and
evals.compare_sets.
"""

import re
from collections.abc import Iterable

from method_2.step09_validate_filings import (
    PERCENT_RE,
    PROSE_DOLLAR_RE,
    UNIT_SCALE,
    closest_match,
    extract_figures,
    to_float,
)
from RAG.chunks import Chunk
from RAG.retrieval import Bm25Retriever


K = 10

# (fact key, question template, is_percent). Every fact exists in both methods'
# financials.json and in Method 1's XBRL extract of real filings.
QUESTIONS: list[tuple[str, str, bool]] = [
    ("revenue", "What were {name}'s total revenues for fiscal {fy}?", False),
    ("net_income", "What was {name}'s net income for fiscal {fy}?", False),
    ("operating_income", "What was {name}'s operating income for fiscal {fy}?", False),
    ("total_assets", "What were {name}'s total assets at the end of fiscal {fy}?", False),
    ("total_liabilities", "What were {name}'s total liabilities at the end of fiscal {fy}?", False),
    ("stockholders_equity", "What was {name}'s total stockholders' equity at the end of fiscal {fy}?", False),
    ("cash", "How much cash and cash equivalents did {name} hold at the end of fiscal {fy}?", False),
    ("operating_cash_flow", "How much cash did {name} generate from operating activities in fiscal {fy}?", False),
    ("net_margin", "What was {name}'s net margin for fiscal {fy}?", True),
    ("operating_margin", "What was {name}'s operating margin for fiscal {fy}?", True),
]

# Leaf names in a synthetic financials.json (either method) for each dollar fact.
SYNTHETIC_KEYS = {
    "revenue": ("income_statement", "revenues"),
    "net_income": ("income_statement", "net_income"),
    "operating_income": ("income_statement", "operating_income"),
    "total_assets": ("balance_sheet", "total_assets"),
    "total_liabilities": ("balance_sheet", "total_liabilities"),
    "stockholders_equity": ("balance_sheet", "total_stockholders_equity"),
    "cash": ("balance_sheet", "cash_and_equivalents"),
    "operating_cash_flow": ("cash_flow_statement", "cash_from_operations"),
}


# How each fact is labelled in prose or a table row. Matched case-insensitively.
LABELS = {
    "revenue": r"(?:total |net )?revenues?|net sales|total sales",
    "net_income": r"net income|net earnings|net loss",
    "operating_income": r"operating income|income from operations|operating loss",
    "total_assets": r"total assets",
    "total_liabilities": r"total liabilities(?! and)",
    "stockholders_equity": r"(?:total )?(?:stockholders|shareholders)['’]? equity",
    "cash": r"cash and cash equivalents",
    "operating_cash_flow": r"(?:net )?cash (?:provided by|generated (?:from|by)|from) operating activities|operating cash flows?",
    "net_margin": r"net (?:profit )?margin",
    "operating_margin": r"operating margin",
}
WINDOW = 120  # characters after a label in which its figure must appear


def find_leaf(d: dict, key: str) -> float | None:
    """Return the first numeric value stored under `key` anywhere in `d`, skipping prior-year blocks."""
    for k, v in d.items():
        if k == key and isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict) and k != "prior_year":
            found = find_leaf(v, key)
            if found is not None:
                return found
    return None


def with_margins(facts: dict[str, float | None]) -> dict[str, float | None]:
    """Add net and operating margin (percent of revenue) to a dict of dollar facts."""
    revenue = facts.get("revenue")
    for margin, numerator in (("net_margin", "net_income"), ("operating_margin", "operating_income")):
        value = facts.get(numerator)
        facts[margin] = value / revenue * 100 if revenue and value is not None else None
    return facts


def synthetic_facts(financials: dict) -> dict[str, float | None]:
    """The ten facts from a synthetic company's financials.json."""
    facts = {fact: find_leaf(financials.get(section, {}), leaf) for fact, (section, leaf) in SYNTHETIC_KEYS.items()}
    return with_margins(facts)


def real_facts(xbrl: dict[str, float]) -> dict[str, float | None]:
    """The ten facts from a real filing's XBRL headline figures (Method 1's financials.csv names)."""
    names = {"revenue": "revenue", "stockholders_equity": "stockholders_equity"}
    facts = {fact: xbrl.get(names.get(fact, fact)) for fact in SYNTHETIC_KEYS}
    return with_margins(facts)


def stating_chunks(value: float, chunks: Iterable[Chunk], *, is_percent: bool) -> list[str]:
    """Ids of every chunk whose text states `value`."""
    ids = []
    for chunk in chunks:
        dollars, percents = extract_figures(chunk.text)
        if any(closest_match(v, {value}, relative=not is_percent) for v, _ in (percents if is_percent else dollars)):
            ids.append(chunk.chunk_id)
    return ids


def make_questions(
    company_id: str,
    name: str,
    fy: int,
    facts: dict[str, float | None],
    chunks: list[Chunk],
) -> list[dict]:
    """The fixed question bank for one company, with each answer's stating chunks.

    A fact the statements don't have (a real filing missing an XBRL tag) is skipped;
    for synthetic companies all ten always exist.
    """
    rows = []
    by_id = {c.chunk_id: c for c in chunks}
    for fact, template, is_percent in QUESTIONS:
        value = facts.get(fact)
        if value is None:
            continue
        gold = stating_chunks(value, chunks, is_percent=is_percent)
        rows.append(
            {
                "company_id": company_id,
                "fiscal_year": fy,
                "fact": fact,
                "question": template.format(name=name, fy=fy),
                "answer": round(value, 2),
                "is_percent": is_percent,
                "gold_chunk_ids": ";".join(gold),
                "gold_items": ";".join(sorted({by_id[g].item for g in gold})),
            },
        )
    return rows


def score(questions: list[dict], chunks: list[Chunk], k: int = K) -> dict[str, float | int | None]:
    """fact_coverage and fact_recall_at_k for `questions` searched over `chunks`.

    Only gold chunks among `chunks` count, so scoring one Item's chunks asks: could a
    retriever over this Item alone answer the standard questions?
    """
    if not questions:
        return {"n_questions": 0, "fact_coverage": None, f"fact_recall_at_{k}": None}
    ids = {c.chunk_id for c in chunks}
    bm25 = Bm25Retriever([c.chunk_id for c in chunks], [c.text for c in chunks]) if chunks else None
    covered = hits = 0
    for q in questions:
        gold = {g for g in q["gold_chunk_ids"].split(";") if g in ids}
        if not gold:
            continue
        covered += 1
        if bm25 and set(bm25.rank(q["question"])[:k]) & gold:
            hits += 1
    n = len(questions)
    return {"n_questions": n, "fact_coverage": round(covered / n, 3), f"fact_recall_at_{k}": round(hits / n, 3)}


def first_percent(text: str) -> float | None:
    m = PERCENT_RE.search(text)
    return to_float(m.group(0)) if m else None


def first_dollar(text: str) -> float | None:
    m = PROSE_DOLLAR_RE.search(text)
    if not m:
        return None
    amount = to_float(m.group(0)[: -len(m.group(1))])
    return amount * UNIT_SCALE[m.group(1).lower()] if amount is not None else None


def attached_figures(text: str, fact: str, *, is_percent: bool) -> list[float]:
    """Figures the text attaches to `fact`'s label: the first figure of the right kind
    within WINDOW characters after the label in the same sentence, or a table row whose
    label is the fact's label.
    """
    label = re.compile(rf"\b(?:{LABELS[fact]})\b", re.IGNORECASE)
    values = []
    for line in text.splitlines():
        if " | " in line:
            parts = [p.strip() for p in line.split(" | ")]
            if label.fullmatch(parts[0].strip("*: ")) and not is_percent:
                v = to_float(parts[1]) if len(parts) > 1 else None
                if v is not None:
                    values.append(v * 1e6)
            continue
        for m in label.finditer(line):
            tail = re.split(r"(?<=[.;])\s", line[m.end() : m.end() + WINDOW], maxsplit=1)[0]
            v = first_percent(tail) if is_percent else first_dollar(tail)
            if v is not None:
                values.append(v)
    return values


def correctness(
    texts: Iterable[str],
    facts: dict[str, float | None],
    prior: dict[str, float | None],
) -> tuple[int, int]:
    """(correct, attached) label-attached figures in `texts` for one company."""
    correct = total = 0
    for fact, _, is_percent in QUESTIONS:
        value = facts.get(fact)
        if value is None:
            continue
        allowed = {value}
        before = prior.get(fact)
        if before is not None:
            allowed |= {before} if is_percent else {before, value - before}
        for text in texts:
            for v in attached_figures(text, fact, is_percent=is_percent):
                total += 1
                correct += closest_match(v, allowed, relative=not is_percent)
    return correct, total


def synthetic_prior_facts(financials: dict) -> dict[str, float | None]:
    """Prior-year facts from a two-year financials.json (Method 1), else all None."""
    prior_block = financials.get("cash_flow_statement", {}).get("prior_year", {})
    facts = {fact: prior_block.get(f"{leaf}_prior") for fact, (_, leaf) in SYNTHETIC_KEYS.items()}
    return with_margins({k: (float(v) if v is not None else None) for k, v in facts.items()})
