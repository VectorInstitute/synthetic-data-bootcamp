"""Split downloaded 10-K text into Items 1, 1A, 1C, 7, and 8.

Input: data/raw/{ticker}/{fy}/10k.txt (and ex13_*.htm, for PGR, USB, and WFC).
Output: data/parsed/{ticker}/{fy}/item_1.txt, item_1a.txt, item_1c.txt, item_7.txt,
item_8.txt, plus one status row per company-year-item in data/parse_status.csv.
See SECTIONS.md for why these five items were chosen.

The table of contents repeats every Item heading, so each heading is matched wherever
it appears and the occurrence with the most text before the next heading is taken as
the body. A few large banks and insurers reprint the heading as a running header on
every page instead; those are detected by an unusually high occurrence count and
every stretch the heading owns is concatenated, rather than just the single biggest
one. Any item that comes back short and "incorporated by reference" instead of
written out goes through a second pass: the heading named in the cross-reference
sentence (quoted, or named in a few common phrasings) is searched for first in the
filing's EX-13 annual report exhibit (PGR, USB, WFC in this corpus), then in the 10-K
body itself, since large banks often repeat a running header like "Management's
discussion and analysis" dozens of times instead of writing Item 7's content under a
single numbered heading. Anything still missing or suspiciously short is written
anyway and flagged in parse_status.csv for manual review, rather than guessed at
further.

Run from the project directory:
    python -m src.parse_sections
"""

import csv
import re
import warnings
from pathlib import Path

from bs4 import XMLParsedAsHTMLWarning

from src.config import PARSED_DIR, RAW_DIR, get_companies, load_config
from src.html_to_text import html_to_text
from src.utils import setup_logging, write_atomic


logger = setup_logging("parse_sections")
# The EX-13 exhibits are HTML with an XML-ish inline-XBRL root tag; expected, not a bug.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TARGET_ITEMS = ["1", "1A", "1C", "7", "8"]

HEADING_RE = re.compile(
    r"^\s*ITEMS?\s+(\d{1,2}[A-C]?(?:\s*(?:,|and)\s*\d{1,2}[A-C]?)*)\b[\s.\-–—:|]*(.*)$",
    re.IGNORECASE,
)
ITEM_CODE_RE = re.compile(r"\d{1,2}[A-C]?", re.IGNORECASE)
# PulteGroup headers Item 1 as "ITEM I." (a capital letter, not the digit) every year.
ITEM_ONE_TYPO_RE = re.compile(r"^\s*ITEM\s+I\b[\s.\-–—:|]*(.*)$", re.IGNORECASE)

# Phrase extraction for an "incorporated by reference" sentence, in priority order.
QUOTE_RE = re.compile(r"[“”\"]([^“”\"]{3,200})[“”\"]")
DASH_SPLIT_RE = re.compile(r"\s+[–—-]\s+")
# A heading-style phrase: capitalized words, optionally joined by a lowercase connector
# ("Compliance and Operational Risk Management"). Case-sensitive on purpose, so a
# connector verb like "Refer to" before it is never mistaken for part of the heading.
_WORD = r"(?:[A-Z][\w&/’'()-]*|and|of|the|for|or)"
_PHRASE = rf"{_WORD}(?:\s+{_WORD}){{0,7}}"
SECTION_OF_RE = re.compile(rf"\b(?:the\s+)?({_PHRASE})\s+section of")
SEE_IN_RE = re.compile(rf"\bSee\s+({_PHRASE})\s+in(?: the| our)?\s+MD&A")
MDA_PHRASE_RE = re.compile(
    r"Management.s Discussion and Analysis(?: of Financial Condition and Results of Operations)?",
    re.IGNORECASE,
)
# Last resort for Item 8: every audited 10-K has this exact phrase before the financial
# statements. Only used when nothing more specific is named in the cross-reference text.
ITEM8_FALLBACK_PHRASE = "Report of Independent Registered Public Accounting Firm"

MIN_SECTION_CHARS = 400
# A real section is never this short, so a marker phrase found within this many
# characters means the whole thing is a cross-reference, not a passing mention buried
# in otherwise-real content (e.g. a Business section citing a proxy statement).
CROSS_REFERENCE_STUB_CHARS = 2000
CROSS_REFERENCE_MARKERS = (
    "incorporated by reference",
    "incorporated herein by reference",
    "incorporated into this item by reference",
    "incorporated into this report by reference",
    "can be found in the",
    "appears on page",
    "appear on pages",
    "for further information",
    "reference is made to",
)

STATUS_PATH = RAW_DIR.parent / "parse_status.csv"
STATUS_FIELDS = ["ticker", "fiscal_year", "item", "status", "chars", "method", "message"]


def find_headings(lines: list[str]) -> list[tuple[int, str]]:
    """Return (line_index, item_code) for every "Item N" heading, including TOC rows.

    A combined heading ("Items 1 and 2. Business and Properties") yields one entry per
    item code, so Item 1 is still found even when it shares a heading with Item 2.
    """
    headings = []
    for idx, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match:
            for code in ITEM_CODE_RE.findall(match.group(1)):
                headings.append((idx, code.upper()))
        elif ITEM_ONE_TYPO_RE.match(line):
            headings.append((idx, "1"))
    return headings


def best_span(markers: list[tuple[int, str]], item: str, total_lines: int) -> tuple[int, int] | None:
    """Return the (start, end) line range for `item`'s occurrence with the most text.

    `markers` pools every candidate boundary in the document: numbered Item headings,
    and (when resolving a cross-reference) every occurrence of another item's marker
    phrase. A repeated phrase lists almost no text before its next repeat, so the
    occurrence with the largest gap to the next marker (of any item) is the real one.
    """
    candidates = [line for line, code in markers if code == item]
    if not candidates:
        return None
    all_lines = sorted({line for line, _ in markers})
    best, best_gap = None, -1
    for start in candidates:
        later = [i for i in all_lines if i > start]
        end = later[0] if later else total_lines
        # >= so a tie (same gap) prefers the later occurrence: the TOC always comes
        # before the body, so when two candidates tie, the later one is the body.
        if end - start >= best_gap:
            best, best_gap = (start, end), end - start
    return best


def section_text(lines: list[str], span: tuple[int, int]) -> str:
    """Join the lines strictly between a heading and the next one."""
    start, end = span
    return "\n".join(lines[start + 1 : end]).strip()


# Above this many occurrences, a heading is a running header reprinted on every page
# (seen in large banks/insurers whose 10-K wraps a full annual report) rather than a
# normal table-of-contents-plus-body pair, and best_span's single "biggest gap" pick
# would return just one page of it.
REPEATED_HEADING_THRESHOLD = 5


def collect_text(markers: list[tuple[int, str]], item: str, lines: list[str], total_lines: int) -> str:
    """Return every stretch `item` owns, concatenated, instead of just the biggest one.

    Between two consecutive markers (of any item, in document order) the earlier one
    owns that stretch. Unlike best_span, this keeps every stretch a repeated heading
    owns rather than picking a single occurrence, so content split across many
    reprints of the same running header is no longer truncated to one page of it.
    """
    ordered = sorted(set(markers))
    chunks = []
    for i, (start, code) in enumerate(ordered):
        if code != item:
            continue
        end = ordered[i + 1][0] if i + 1 < len(ordered) else total_lines
        chunk = "\n".join(lines[start + 1 : end]).strip()
        if chunk:
            chunks.append(chunk)
    return "\n\n".join(chunks)


def is_cross_reference(text: str) -> bool:
    """Return True if `text` points elsewhere instead of holding the content itself.

    Checked against CROSS_REFERENCE_STUB_CHARS rather than MIN_SECTION_CHARS: USB's
    Item 8 cross-reference lists every financial statement heading by name and clears
    400 characters without containing any actual financial statements, but a real
    section is never as short as 2,000 characters either.
    """
    if len(text) > CROSS_REFERENCE_STUB_CHARS:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in CROSS_REFERENCE_MARKERS)


def extract_marker_phrases(cross_ref_text: str, item: str) -> list[str]:
    """Pull every candidate heading name a cross-reference sentence could point to.

    Returns all of them, not just the first match, so a wrong one (a quoted heading
    can just be naming Item 15's title instead of the financial statements) doesn't
    block a better one found a different way. Candidates: a quoted heading (split on
    a hierarchy like "Financial Review – Risk Factors" down to its leaf, "Risk
    Factors"); "the X section of ..."; "See X in the MD&A ..."; the official Item 7
    title; and, for Item 8 only, the one phrase every audited 10-K has.
    """
    phrases = []
    quoted = QUOTE_RE.search(cross_ref_text)
    if quoted:
        phrase = quoted.group(1).strip().rstrip(".,")
        phrases.append(DASH_SPLIT_RE.split(phrase)[-1].strip())
    section_of = SECTION_OF_RE.search(cross_ref_text)
    if section_of:
        phrases.append(section_of.group(1).strip())
    see_in = SEE_IN_RE.search(cross_ref_text)
    if see_in:
        phrases.append(see_in.group(1).strip())
    mda = MDA_PHRASE_RE.search(cross_ref_text)
    if mda:
        phrases.append(mda.group(0))
    if item == "8":
        phrases.append(ITEM8_FALLBACK_PHRASE)
    return phrases


def find_all_lines(lines: list[str], phrase: str) -> list[int]:
    """Return every line equal to `phrase`, ignoring case and a trailing period."""
    target = phrase.strip().rstrip(".").lower()
    return [i for i, line in enumerate(lines) if line.strip().rstrip(".").lower() == target]


def locate_spans(
    lines: list[str],
    items: set[str],
    cross_refs: dict[str, str],
    base_markers: list[tuple[int, str]],
    *,
    allow_synthetic_start: bool = False,
) -> dict[str, tuple[int, int]]:
    """Find each item's marker occurrences in `lines` and compute its best span.

    `base_markers` seeds the shared pool best_span() picks from: a document's own
    numbered headings for the same-document pass, or nothing for an EX-13 exhibit
    (which has no Item numbering of its own). With `allow_synthetic_start`, Item 8
    also gets a synthetic marker at the very start of the document, so a
    cross-reference that never names a heading (PGR's Item 8 just says "our Annual
    Report") still gets a span. This only makes sense for an EX-13 exhibit, where the
    financial statements can legitimately open the document; in the 10-K body itself
    they never do, and the synthetic marker would instead win by default and capture
    the cover page.
    """
    pool = list(base_markers)
    for item in items:
        phrases = extract_marker_phrases(cross_refs.get(item, ""), item)
        for phrase in phrases:
            pool.extend((line, item) for line in find_all_lines(lines, phrase))
        if item == "8" and allow_synthetic_start and phrases:
            pool.append((-1, item))

    spans = {}
    for item in items:
        span = best_span(pool, item, len(lines))
        if span is not None:
            spans[item] = span
    return spans


def ex13_candidate(out_dir: Path) -> Path | None:
    """Return the EX-13 exhibit for this company-year, if the filing included one."""
    matches = sorted(out_dir.glob("ex13_*.htm"))
    return matches[0] if matches else None


def load_ex13_lines(path: Path) -> list[str]:
    """Convert an EX-13 exhibit to text, caching it next to the HTML like 10k.txt."""
    cache = path.with_suffix(".txt")
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        text = html_to_text(path.read_bytes())
        write_atomic(cache, text)
    return text.splitlines()


def resolve_fallbacks(
    *,
    needing: set[str],
    sections: dict[str, tuple[str, str]],
    lines: list[str],
    headings: list[tuple[int, str]],
    out_dir: Path,
    claimed_spans: dict[str, tuple[int, int]],
) -> None:
    """Resolve as many `needing` items as possible, updating `sections` in place.

    All items needing fallback are resolved together, in one shared marker pool per
    stage: WFC's Item 7, 1A, and 1C cross-references all point into the same EX-13
    exhibit, and each one's marker is what correctly bounds the others (Item 7 stops
    where Item 1C's marker starts, and so on). Resolving them one at a time would lose
    that mutual boundary and let each span run too far.

    `claimed_spans` holds the line range every already-resolved item occupies in
    `lines`, so a same-document fallback for another item can be checked against them:
    a multi-registrant filer (e.g. AEP, which files jointly for several utility
    subsidiaries) can leave Item 7's own heading a stub while its content and Item 8's
    sit in the same unlabeled block, and without this check Item 7's fallback would
    claim that whole block too, duplicating what Item 8 already claimed.
    """
    cross_refs = {item: sections[item][0] for item in needing}

    ex13_path = ex13_candidate(out_dir)
    if ex13_path is not None:
        ex13_lines = load_ex13_lines(ex13_path)
        spans = locate_spans(ex13_lines, needing, cross_refs, base_markers=[], allow_synthetic_start=True)
        for item, span in spans.items():
            candidate = section_text(ex13_lines, span)
            if _is_acceptable(item, candidate, bounded=span[1] != len(ex13_lines)):
                sections[item] = (candidate, "ex13_fallback")
                needing.discard(item)

    if not needing:
        return
    spans = locate_spans(lines, needing, cross_refs, base_markers=headings)
    for item, span in spans.items():
        if _overlaps_claimed(span, claimed_spans):
            continue
        candidate = section_text(lines, span)
        if _is_acceptable(item, candidate, bounded=span[1] != len(lines)):
            sections[item] = (candidate, "body_alias_fallback")


def _overlaps_claimed(span: tuple[int, int], claimed_spans: dict[str, tuple[int, int]]) -> bool:
    """Return True if `span` overlaps more than a third of its length with a claim."""
    start, end = span
    length = end - start
    if length <= 0:
        return False
    for claimed_start, claimed_end in claimed_spans.values():
        overlap = min(end, claimed_end) - max(start, claimed_start)
        if overlap / length > 1 / 3:
            return True
    return False


def _is_acceptable(item: str, text: str, *, bounded: bool) -> bool:
    """Accept a fallback candidate unless it's suspiciously unbounded.

    Item 1C (narrow, normally well under 20,000 chars) is only accepted when its span
    ends at another item's marker rather than falling through to the end of the
    document: an actual Item 1C cross-reference always sits ahead of something else
    (Item 1A, 7, or 8), so an unbounded span is more likely a sweep of unrelated text
    than a real cybersecurity section, however long it comes out. A candidate that
    itself still reads as a cross-reference is rejected outright: a multi-registrant
    filing (e.g. AEP's) can point Item 7 at another cross-reference sentence instead
    of the real content, and that's better left `short` than accepted as resolved.
    """
    if len(text) < MIN_SECTION_CHARS or is_cross_reference(text):
        return False
    return bounded or item != "1C"


def process(ticker: str, fiscal_year: int) -> list[dict]:
    """Parse one company-year, writing item files and returning its status rows."""
    out_dir = RAW_DIR / ticker / str(fiscal_year)
    raw_path = out_dir / "10k.txt"
    if not raw_path.exists():
        logger.warning("%s FY%d: no 10k.txt (run download_10k first); skipping", ticker, fiscal_year)
        return [
            _status_row(
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
                status="not_downloaded",
                chars=0,
                method="",
                message="no 10k.txt",
            )
            for item in TARGET_ITEMS
        ]

    lines = raw_path.read_text(encoding="utf-8").splitlines()
    headings = find_headings(lines)

    sections: dict[str, tuple[str, str]] = {}
    claimed_spans: dict[str, tuple[int, int]] = {}
    for item in TARGET_ITEMS:
        if sum(1 for _, code in headings if code == item) > REPEATED_HEADING_THRESHOLD:
            text, method = collect_text(headings, item, lines, len(lines)), "body_repeated_heading"
        else:
            span = best_span(headings, item, len(lines))
            text, method = (section_text(lines, span), "body") if span else ("", "body")
            if span is not None:
                claimed_spans[item] = span
        sections[item] = (text, method)

    needing = {
        item
        for item in TARGET_ITEMS
        if len(sections[item][0]) < MIN_SECTION_CHARS or is_cross_reference(sections[item][0])
    }
    claimed_spans = {item: span for item, span in claimed_spans.items() if item not in needing}
    if needing:
        resolve_fallbacks(
            needing=needing,
            sections=sections,
            lines=lines,
            headings=headings,
            out_dir=out_dir,
            claimed_spans=claimed_spans,
        )

    parsed_dir = PARSED_DIR / ticker / str(fiscal_year)
    rows = []
    for item in TARGET_ITEMS:
        text, method = sections[item]
        if not text:
            rows.append(
                _status_row(
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    item=item,
                    status="missing",
                    chars=0,
                    method=method,
                    message="no heading found",
                ),
            )
            continue
        status = "ok" if len(text) >= MIN_SECTION_CHARS and not is_cross_reference(text) else "short"
        message = "" if status == "ok" else "short section, possibly an unresolved cross-reference"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(parsed_dir / f"item_{item.lower()}.txt", text + "\n")
        rows.append(
            _status_row(
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
                status=status,
                chars=len(text),
                method=method,
                message=message,
            ),
        )

    logger.info(
        "%s FY%d: %s",
        ticker,
        fiscal_year,
        ", ".join(
            f"{r['item']}={r['status']}({r['method']})" if r["method"] else f"{r['item']}={r['status']}" for r in rows
        ),
    )
    return rows


def _status_row(
    *,
    ticker: str,
    fiscal_year: int,
    item: str,
    status: str,
    chars: int,
    method: str,
    message: str,
) -> dict:
    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "item": item,
        "status": status,
        "chars": chars,
        "method": method,
        "message": message,
    }


def write_status(rows: list[dict]) -> None:
    """Write this run's per-company-year-item status rows to STATUS_PATH."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse Items 1, 1A, 1C, 7, and 8 for every company-year in config.yaml."""
    cfg = load_config()
    tickers = get_companies(cfg)
    fiscal_years = [int(y) for y in cfg["fiscal_years"]]
    logger.info(
        "Profile '%s': %d companies x %d fiscal years",
        cfg["run_profile"],
        len(tickers),
        len(fiscal_years),
    )

    rows = []
    for ticker in tickers:
        for fy in fiscal_years:
            rows.extend(process(ticker, fy))

    write_status(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    logger.info("Done. Status counts: %s", counts)
    logger.info("Status file: %s", STATUS_PATH)


if __name__ == "__main__":
    main()
