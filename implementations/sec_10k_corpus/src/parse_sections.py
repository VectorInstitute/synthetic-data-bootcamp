"""Split downloaded 10-K text into Items 1, 1A, 1C, 7, and 8.

Input: data/raw/{ticker}/{fy}/10k.txt (and ex13_*.htm, for PGR, USB, and WFC).
Output: data/parsed/{ticker}/{fy}/item_1.txt, item_1a.txt, item_1c.txt, item_7.txt,
item_8.txt, plus one status row per company-year-item in data/parse_status.csv.
See SECTIONS.md for why these five items were chosen.

The table of contents repeats every Item heading, so each heading is matched wherever
it appears and the occurrence with the most text before the next heading is taken as
the body. Item 7 and Item 8 are sometimes only "incorporated by reference" to an EX-13
annual report exhibit (PGR, USB, WFC in this corpus); for those, the heading named in
the cross-reference sentence is located inside the EX-13 text instead. Anything still
missing or suspiciously short is written anyway and flagged in parse_status.csv for
manual review, rather than guessed at further.

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
# PLAN.md routes only Item 7 and 8 to an EX-13 exhibit when they're a cross-reference.
EX13_FALLBACK_ITEMS = ("7", "8")

HEADING_RE = re.compile(
    r"^\s*ITEMS?\s+(\d{1,2}[A-C]?(?:\s*(?:,|and)\s*\d{1,2}[A-C]?)*)\b[\s.\-–—:|]*(.*)$",
    re.IGNORECASE,
)
ITEM_CODE_RE = re.compile(r"\d{1,2}[A-C]?", re.IGNORECASE)
QUOTE_RE = re.compile(r"[“”\"]([^“”\"]{3,90})[“”\"]")
MDA_PHRASE_RE = re.compile(
    r"Management.s Discussion and Analysis(?: of Financial Condition and Results of Operations)?",
    re.IGNORECASE,
)

MIN_SECTION_CHARS = 400
CROSS_REFERENCE_MARKERS = (
    "incorporated by reference",
    "incorporated herein by reference",
    "incorporated into this item by reference",
    "incorporated into this report by reference",
    "can be found in the",
    "appears on page",
    "appear on pages",
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
    return headings


def best_span(headings: list[tuple[int, str]], item: str, total_lines: int) -> tuple[int, int] | None:
    """Return the (start, end) line range for `item`'s occurrence with the most text.

    The TOC lists every heading with almost no text before the next one, so the
    occurrence with the largest gap to the next heading (of any item) is the body.
    """
    candidates = [line for line, code in headings if code == item]
    if not candidates:
        return None
    all_lines = sorted({line for line, _ in headings})
    best, best_gap = None, -1
    for start in candidates:
        later = [i for i in all_lines if i > start]
        end = later[0] if later else total_lines
        if end - start > best_gap:
            best, best_gap = (start, end), end - start
    return best


def section_text(lines: list[str], span: tuple[int, int]) -> str:
    """Join the lines strictly between a heading and the next one."""
    start, end = span
    return "\n".join(lines[start + 1 : end]).strip()


def is_cross_reference(text: str) -> bool:
    """Return True if `text` points elsewhere instead of holding the content itself.

    Checked regardless of length: USB's Item 8 cross-reference lists every financial
    statement heading by name, so it clears MIN_SECTION_CHARS without containing any
    actual financial statements.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in CROSS_REFERENCE_MARKERS)


def extract_marker_phrase(cross_ref_text: str) -> str | None:
    """Pull the heading name an "incorporated by reference" sentence points to."""
    quoted = QUOTE_RE.search(cross_ref_text)
    if quoted:
        return quoted.group(1).strip().rstrip(".,")
    mda = MDA_PHRASE_RE.search(cross_ref_text)
    return mda.group(0) if mda else None


def find_last_line(lines: list[str], phrase: str) -> int | None:
    """Return the last line equal to `phrase`, ignoring case and a trailing period."""
    target = phrase.strip().rstrip(".").lower()
    matches = [i for i, line in enumerate(lines) if line.strip().rstrip(".").lower() == target]
    return matches[-1] if matches else None


def ex13_spans(ex13_lines: list[str], cross_refs: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Locate Item 7 and Item 8 inside an EX-13 exhibit from their cross-reference text.

    PGR, USB, and WFC name the real heading in the sentence that replaces Item 7 or 8
    (e.g. "...under the heading 'Financial Review'..."). That heading is searched for
    in the EX-13 text, and whichever of Item 7 or 8 is found first runs until the other
    one starts (or, for PGR, until the start of the exhibit, since its Item 8
    cross-reference names no specific heading).
    """
    markers: dict[str, int] = {}
    for item in EX13_FALLBACK_ITEMS:
        phrase = extract_marker_phrase(cross_refs.get(item, ""))
        line = find_last_line(ex13_lines, phrase) if phrase else None
        if line is not None:
            markers[item] = line

    spans: dict[str, tuple[int, int]] = {}
    ordered = sorted(markers.items(), key=lambda kv: kv[1])
    for i, (item, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(ex13_lines)
        spans[item] = (start, end)

    missing = [item for item in EX13_FALLBACK_ITEMS if item not in spans]
    if len(missing) == 1 and ordered:
        _, other_start = ordered[0]
        spans[missing[0]] = (0, other_start)
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
    cross_refs: dict[str, str] = {}
    for item in TARGET_ITEMS:
        span = best_span(headings, item, len(lines))
        text = section_text(lines, span) if span else ""
        sections[item] = (text, "body")
        if item in EX13_FALLBACK_ITEMS:
            cross_refs[item] = text

    ex13_path = ex13_candidate(out_dir)
    if ex13_path is not None:
        needs_fallback = {
            item for item in EX13_FALLBACK_ITEMS if sections[item][0] and is_cross_reference(sections[item][0])
        }
        if needs_fallback:
            ex13_lines = load_ex13_lines(ex13_path)
            spans = ex13_spans(ex13_lines, cross_refs)
            for item in needs_fallback:
                if item not in spans:
                    continue
                text = section_text(ex13_lines, spans[item])
                if len(text) >= MIN_SECTION_CHARS:
                    sections[item] = (text, "ex13_fallback")

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
        status = "ok" if len(text) >= MIN_SECTION_CHARS else "short"
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
