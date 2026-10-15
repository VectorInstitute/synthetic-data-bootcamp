"""Checks on generated 10-K sections, run before a section is accepted.

Each check returns a list of human-readable issues; an empty list means the
section passed. The issues are also fed back to the model when a section is
regenerated, so they are phrased as instructions the model can act on.
"""

from __future__ import annotations

import re
from typing import Any

from aieng.syn_data.sec10k.boilerplate import HEADINGS
from aieng.syn_data.sec10k.render import derived_metrics


__all__ = ["HEADINGS", "validate_section"]
# Accepted word count, as a fraction of the target.
WORD_RANGE = (0.6, 1.5)
MAX_EXEMPLAR_OVERLAP = 0.05  # share of 8-word shingles also in the exemplar
SHINGLE = 8
# Most dollar amounts a narrative Item may cite; Items 7 and 7A are unlimited.
MAX_AMOUNTS = {"1": 3, "1A": 4, "1C": 0, "2": 0, "3": 2}
# Percentages any MD&A may cite without them being in the statements.
STANDARD_PERCENTAGES = {21.0, 100.0}

_DOLLARS = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion|thousand)\b)?",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b)", re.IGNORECASE)
_NAME_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "COMPANIES",
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "PLC",
    "LTD",
    "LIMITED",
    "LLC",
    "LP",
    "NV",
    "SA",
    "AG",
    "THE",
    "TRUST",
    "INTERNATIONAL",
    "&",
    "DE",
    "NEW",
}
# Single-word filer names that are also ordinary words in a 10-K.
_COMMON_WORDS = {
    "AMERICAN",
    "BLOCK",
    "BOOKING",
    "COHERENT",
    "FLEX",
    "SOUTHERN",
    "TARGET",
    "WATERS",
    "WILLIAMS",
}


def word_count(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def check_heading(text: str, item: str) -> list[str]:
    """Check that the section opens with its Item heading."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if not re.match(rf"item\s+{re.escape(item)}\.", first, re.IGNORECASE):
        return [f'Start the section with the heading line "{HEADINGS[item]}".']
    return []


def check_length(text: str, target: int) -> list[str]:
    """Check that the section is within ``WORD_RANGE`` of its target length."""
    n = word_count(text)
    lo, hi = (round(target * f) for f in WORD_RANGE)
    if n < lo:
        return [f"The section has {n} words; write at least {lo} (target {target})."]
    if n > hi:
        return [f"The section has {n} words; keep it under {hi} (target {target})."]
    return []


def check_company_name(text: str, spec: dict[str, Any]) -> list[str]:
    """Check that the fictional company is named in the text."""
    short = spec["name"].split(",")[0]
    if short.split()[0] not in text:
        return [f'Refer to the company by its name, "{spec["name"]}".']
    return []


def real_name_patterns(companies: list[str]) -> list[re.Pattern[str]]:
    """Turn real filer names ("ADOBE INC.") into case-sensitive word patterns.

    Each name matches in title case ("Adobe", "Archer-Daniels-Midland") or upper
    case ("CBRE"). Names that are ordinary words ("Target") are skipped.
    """
    patterns = {}
    for company in companies:
        words = [w.strip(".,()") for w in company.upper().split()]
        core = [w for w in words if w and w not in _NAME_SUFFIXES]
        if not core or (len(core) == 1 and core[0] in _COMMON_WORDS):
            continue
        upper = " ".join(core[:2])
        if len(upper) < 4:
            continue
        title = re.sub(
            r"(?<![a-z'’.])[a-z]", lambda m: m.group(0).upper(), upper.lower()
        )
        forms = "|".join(
            re.escape(form).replace("’", "['’]")
            for form in dict.fromkeys((title, upper))
        )
        patterns[upper] = re.compile(rf"\b(?:{forms})(?!\w)")
    return list(patterns.values())


def check_real_names(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    """No real filer may be named; competitors are described generically."""
    found = sorted({m.group(0) for p in patterns for m in [p.search(text)] if m})
    if found:
        return [
            "Remove these real company names and describe them generically: "
            + ", ".join(found)
            + "."
        ]
    return []


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def exemplar_overlap(text: str, exemplar: str) -> float:
    """Share of the text's 8-word shingles that also appear in the exemplar."""
    ours = _shingles(text)
    if not ours or not exemplar:
        return 0.0
    return len(ours & _shingles(exemplar)) / len(ours)


def copied_sentences(text: str, exemplar: str, limit: int = 4) -> list[str]:
    """Sentences of ``text`` that mostly repeat the exemplar word for word."""
    theirs = _shingles(exemplar)
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        ours = _shingles(sentence)
        if ours and len(ours & theirs) / len(ours) >= 0.5:
            found.append(sentence.strip()[:120])
    return found[:limit]


def check_overlap(text: str, exemplar: str) -> list[str]:
    """Check that the text does not copy the style exemplar."""
    overlap = exemplar_overlap(text, exemplar)
    if overlap > MAX_EXEMPLAR_OVERLAP:
        quoted = "; ".join(f'"{s}..."' for s in copied_sentences(text, exemplar))
        return [
            f"{overlap:.0%} of the text repeats the style example word for word; "
            "rewrite it in your own words. Copied sentences include: " + quoted
        ]
    return []


def check_number_density(text: str, item: str) -> list[str]:
    """Narrative Items other than MD&A should rarely quote dollar amounts."""
    limit = MAX_AMOUNTS.get(item)
    found = len(_DOLLARS.findall(text))
    if limit is not None and found > limit:
        return [
            f"Item {item} cites {found} dollar amounts; real filings cite few in "
            f"this Item. Keep at most {limit} and describe exposure in words."
        ]
    return []


def allowed_amounts(spec: dict[str, Any]) -> set[int]:
    """Dollar amounts ($ millions) the narrative may cite: lines and changes."""
    st = spec["statements"]
    cur, prev = (str(y) for y in st["fiscal_years"])
    amounts: set[int] = set()
    for table in ("income_statement", "balance_sheet"):
        for key, value in st[table][cur].items():
            prior = st[table][prev][key]
            amounts.update({abs(value), abs(prior), abs(value - prior)})
    amounts.update(abs(v) for v in st["cash_flow_statement"][cur].values())
    return amounts


def allowed_percentages(spec: dict[str, Any]) -> set[float]:
    """Percentages the narrative may cite: derived metrics and line changes."""
    st = spec["statements"]
    cur, prev = (str(y) for y in st["fiscal_years"])
    pcts = set(STANDARD_PERCENTAGES)
    pcts.update(abs(v) for v in derived_metrics(spec).values())
    for table in ("income_statement", "balance_sheet"):
        for key, value in st[table][cur].items():
            prior = st[table][prev][key]
            if prior:
                pcts.add(abs(round(100 * (value - prior) / abs(prior), 1)))
    for year in (cur, prev):
        income = st["income_statement"][year]
        for value in income.values():
            if income["revenue"]:
                pcts.add(abs(round(100 * value / income["revenue"], 1)))
    return pcts


def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def unmatched_amounts(text: str, spec: dict[str, Any]) -> list[str]:
    """Dollar amounts in ``text`` that match no statement line or change."""
    allowed = sorted(allowed_amounts(spec))
    scale = {"billion": 1000.0, "million": 1.0, "thousand": 0.001, "": 1.0}
    bad = []
    for match in _DOLLARS.finditer(text):
        raw, unit = match.group(1).replace(",", ""), (match.group(2) or "").lower()
        if not unit and "." in raw:
            continue  # per-share style amounts without a unit are not statement lines
        value = float(raw) * scale[unit]
        # Rounding to the stated precision, e.g. $1.2 billion covers +-50 million.
        tol = 0.5 * 10 ** -_decimals(raw) * scale[unit] + 0.5
        if not any(abs(value - a) <= tol for a in allowed):
            bad.append(match.group(0).strip())
    return bad


def unmatched_percentages(text: str, spec: dict[str, Any]) -> list[str]:
    """Percentages in ``text`` that match no derived metric or line change."""
    allowed = allowed_percentages(spec)
    bad = []
    for match in _PERCENT.finditer(text):
        raw = match.group(1)
        tol = 0.5 * 10 ** -_decimals(raw) + 0.051
        if not any(abs(float(raw) - a) <= tol for a in allowed):
            bad.append(match.group(0).strip())
    return bad


def check_numbers(text: str, spec: dict[str, Any], *, strict: bool) -> list[str]:
    """Dollar amounts must come from the statements; in MD&A, percentages too."""
    issues = []
    amounts = unmatched_amounts(text, spec)
    if amounts:
        issues.append(
            "These dollar amounts are not in the financial statements; use only "
            "figures from the facts sheet: " + ", ".join(dict.fromkeys(amounts)) + "."
        )
    if strict:
        pcts = unmatched_percentages(text, spec)
        if pcts:
            issues.append(
                "These percentages are not in the facts sheet; use only the "
                "derived percentages given: " + ", ".join(dict.fromkeys(pcts)) + "."
            )
    return issues


def validate_section(
    text: str,
    *,
    item: str,
    spec: dict[str, Any],
    target_words: int,
    exemplar: str = "",
    real_names: list[re.Pattern[str]] | None = None,
) -> list[str]:
    """Run every check for one section and return all issues found."""
    issues = check_heading(text, item) + check_length(text, target_words)
    if item in ("1", "7"):
        issues += check_company_name(text, spec)
    issues += check_real_names(text, real_names or [])
    issues += check_overlap(text, exemplar)
    issues += check_number_density(text, item)
    issues += check_numbers(text, spec, strict=item == "7")
    return issues
