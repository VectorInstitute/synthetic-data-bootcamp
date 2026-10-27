"""Split 10-K text into Items (1, 1A, 1C, 7, 7A, 8, ...).

Every Item heading appears twice: once in the table of contents and once in the
body. For each Item we take the occurrence followed by the most text before the
next heading, which skips the table of contents and short cross-references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Items extracted by default, with the minimum length (characters) below which a
# section is flagged as "short". Item 1C (cybersecurity) and 7A are often brief.
DEFAULT_ITEMS: dict[str, int] = {
    "1": 3000,
    "1A": 3000,
    "1C": 800,
    "7": 5000,
    "7A": 300,
    "8": 5000,
}

_HEADING = re.compile(
    r"^[ \t]*(?:part[ \t]+i{1,3}v?[ \t,.:\-–—]*)?items?[ \t]*(\d{1,2}[abc]?)(?![0-9a-z])[ \t]*[.:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)
# Lines that open the audited financial statements, for filings whose Item 8 only
# points elsewhere (e.g. "See Item 15" with the statements on F-pages at the end).
_STATEMENTS_START = re.compile(
    r"^(?:report of independent registered public accounting firm"
    r"|consolidated statements? of (?:income|operations|earnings)"
    r"|consolidated (?:income|earnings) statements?)\b[^\n]{0,80}$",
    re.IGNORECASE | re.MULTILINE,
)
_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+")
_STATEMENTS_END = re.compile(
    r"^(?:signatures|exhibit index|index to exhibits)\s*$", re.IGNORECASE | re.MULTILINE
)
_REFERENCE = re.compile(
    r"incorporated (?:herein |into this report )?by reference|exhibit 13|annual report to (?:share|stock)holders|can be found in",
    re.IGNORECASE,
)


@dataclass
class Section:
    """One extracted Item and how confident we are in it."""

    item: str
    text: str
    status: str  # ok | fallback | short | reference | missing
    start: int = -1


def find_headings(text: str) -> list[tuple[int, str]]:
    """Return ``(offset, item)`` for every line that starts with an Item heading."""
    return [(m.start(), m.group(1).upper()) for m in _HEADING.finditer(text)]


def split_sections(
    text: str, items: dict[str, int] | None = None
) -> dict[str, Section]:
    """Extract the requested Items from 10-K text.

    Status is ``missing`` when no heading was found, ``reference`` when a short
    section points elsewhere (for example to an EX-13 annual report that was not
    downloaded), ``short`` when it is below the item's minimum length, else ``ok``.
    """
    items = items or DEFAULT_ITEMS
    headings = find_headings(text)
    best: dict[str, tuple[int, int]] = {}
    for i, (start, item) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if item in items and (
            item not in best or end - start > best[item][1] - best[item][0]
        ):
            best[item] = (start, end)

    sections: dict[str, Section] = {}
    for item, min_chars in items.items():
        if item not in best:
            sections[item] = Section(item=item, text="", status="missing")
            continue
        start, end = best[item]
        body = text[start:end].strip()
        if len(body) >= min_chars:
            status = "ok"
        elif _REFERENCE.search(body):
            status = "reference"
        else:
            status = "short"
        sections[item] = Section(item=item, text=body, status=status, start=start)

    if "8" in sections and sections["8"].status != "ok":
        fallback = _find_statements(text, headings)
        if fallback is not None and len(fallback.text) >= items["8"]:
            sections["8"] = fallback
    return sections


def _find_statements(text: str, headings: list[tuple[int, str]]) -> Section | None:
    """Locate the financial statements by their own headings instead of "Item 8".

    Takes the first statements heading that is followed by real content (a table row
    with amounts, or the auditor's "To the ..." address) rather than a table-of-contents
    line, and
    runs to the next Item heading, the signatures, or the end of the document.
    """
    for match in _STATEMENTS_START.finditer(text):
        start = match.start()
        following = text[match.end() : match.end() + 1500]
        table_rows = [
            line
            for line in following.splitlines()
            if "\t" in line and _AMOUNT.search(line)
        ]
        if not (
            re.search(r"^to the ", following, re.IGNORECASE | re.MULTILINE)
            or len(table_rows) >= 5
        ):
            continue
        ends = [pos for pos, _ in headings if pos > start]
        end_match = _STATEMENTS_END.search(text, match.end())
        if end_match:
            ends.append(end_match.start())
        end = min(ends, default=len(text))
        return Section(
            item="8", text=text[start:end].strip(), status="fallback", start=start
        )
    return None
