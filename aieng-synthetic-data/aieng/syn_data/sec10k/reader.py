"""Read an EDGAR 10-K (inline XBRL .htm) into plain text and XBRL facts.

Uses only the standard library so it runs on a CPU-only workspace without extra
installs. One pass over the HTML produces:

- readable text, with a line break per block element and a tab between table cells,
- every numeric inline XBRL fact (``ix:nonFraction``) with its context,
- the ``dei:`` cover-page facts (fiscal year, period end, registrant name, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


_BLOCK_TAGS = {
    "address",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
}
_CELL_TAGS = {"td", "th"}
_SKIP_TAGS = {"script", "style", "ix:header", "head", "title"}

# Values that inline XBRL transforms render as words or dashes but mean zero.
_ZERO_WORDS = {"", "-", "—", "–", "no", "none", "nil", "zero"}


@dataclass
class Context:
    """An XBRL context: the period a fact covers and any dimension members."""

    context_id: str
    start: str | None = None
    end: str | None = None
    instant: str | None = None
    dimensions: dict[str, str] = field(default_factory=dict)

    @property
    def period_end(self) -> str | None:
        """Return the end date of a duration, or the date of an instant."""
        return self.end or self.instant


@dataclass
class Fact:
    """One numeric inline XBRL fact as reported in the filing."""

    concept: str
    value: float | None
    raw: str
    context_id: str
    unit: str | None
    decimals: str | None


@dataclass
class ParsedFiling:
    """Text, facts and contexts read from one filing."""

    text: str
    facts: list[Fact]
    contexts: dict[str, Context]
    dei: dict[str, str]


def parse_number(
    raw: str, scale: str | None, sign: str | None, fmt: str | None
) -> float | None:
    """Convert a displayed inline XBRL number to its full value.

    ``raw`` is the text shown in the filing (``"1,234"``), ``scale`` the power of ten
    the display omits (``"6"`` for millions), and ``sign="-"`` flips the sign.
    """
    text = raw.strip().strip("()").strip()
    fmt = (fmt or "").lower()
    if text.lower() in _ZERO_WORDS or fmt.endswith("fixed-zero"):
        value = 0.0
    else:
        if "comma-decimal" in fmt or "numcommadecimal" in fmt:
            text = text.replace(".", "").replace(" ", "").replace(",", ".")
        else:
            text = text.replace(",", "").replace(" ", "")
        try:
            value = float(text)
        except ValueError:
            return None
    if scale and scale.lstrip("-").isdigit():
        value *= 10 ** int(scale)
    if sign == "-":
        value = -value
    return value


class _FilingHTMLParser(HTMLParser):
    """Single-pass parser; see :func:`read_filing`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.facts: list[Fact] = []
        self.contexts: dict[str, Context] = {}
        self.dei: dict[str, str] = {}
        self._skip_depth = 0
        # Open ix:nonFraction / ix:nonNumeric elements: (tag, attrs, collected text).
        # Text is only collected for numbers and dei: cover facts, not long text blocks.
        self._open_facts: list[tuple[str, dict[str, str], list[str] | None]] = []
        self._context: Context | None = None
        self._context_field: str | None = None
        self._member_dim: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

        if tag == "ix:nonfraction":
            self._open_facts.append((tag, attr, []))
        elif tag == "ix:nonnumeric":
            self._open_facts.append(
                (tag, attr, [] if attr.get("name", "").startswith("dei:") else None)
            )
        elif tag == "xbrli:context":
            self._context = Context(context_id=attr.get("id", ""))
        elif self._context is not None:
            if tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant"):
                self._context_field = tag.split(":")[1]
            elif tag in ("xbrldi:explicitmember", "xbrldi:typedmember"):
                self._member_dim = attr.get("dimension", "")
                self._context.dimensions[self._member_dim] = ""

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS and not self._skip_depth:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")
        elif tag in _CELL_TAGS:
            self.chunks.append("\t")

        if (
            tag in ("ix:nonfraction", "ix:nonnumeric")
            and self._open_facts
            and self._open_facts[-1][0] == tag
        ):
            _, attr, parts = self._open_facts.pop()
            raw = "".join(parts or []).strip()
            if self._open_facts and self._open_facts[-1][2] is not None:
                self._open_facts[-1][2].append(raw)
            if parts is None:
                pass
            elif tag == "ix:nonfraction":
                self.facts.append(
                    Fact(
                        concept=attr.get("name", ""),
                        value=parse_number(
                            raw, attr.get("scale"), attr.get("sign"), attr.get("format")
                        ),
                        raw=raw,
                        context_id=attr.get("contextref", ""),
                        unit=attr.get("unitref") or None,
                        decimals=attr.get("decimals") or None,
                    ),
                )
            else:
                name = attr.get("name", "")
                self.dei.setdefault(name, re.sub(r"\s+", " ", raw))
                self.dei.setdefault(f"{name}#context", attr.get("contextref", ""))
        elif tag == "xbrli:context" and self._context is not None:
            self.contexts[self._context.context_id] = self._context
            self._context = None
        elif tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant"):
            self._context_field = None
        elif tag in ("xbrldi:explicitmember", "xbrldi:typedmember"):
            self._member_dim = None

    def handle_data(self, data: str) -> None:
        if self._context is not None:
            value = data.strip()
            if self._context_field and value:
                setattr(
                    self._context,
                    {"startdate": "start", "enddate": "end"}.get(
                        self._context_field, "instant"
                    ),
                    value,
                )
            elif self._member_dim is not None and value:
                self._context.dimensions[self._member_dim] += value
        if self._open_facts and self._open_facts[-1][2] is not None:
            self._open_facts[-1][2].append(data)
        if not self._skip_depth:
            self.chunks.append(data)


def clean_text(raw: str) -> str:
    """Collapse whitespace while keeping line and table-cell breaks."""
    text = raw.replace("\u200b", "").replace("\ufeff", "")
    # Any other whitespace (non-breaking, thin, em spaces, ...) becomes a plain space.
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = re.sub(r" *\t[ \t]*", "\t", text)
    text = re.sub(r"\t*\n\t*", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def read_filing(source: str | Path) -> ParsedFiling:
    """Parse a 10-K .htm file (or its HTML string) into text, facts and contexts."""
    html = (
        source
        if isinstance(source, str) and "<" in source[:2000]
        else Path(source).read_text(errors="replace")
    )
    parser = _FilingHTMLParser()
    parser.feed(html)
    parser.close()
    return ParsedFiling(
        text=clean_text("".join(parser.chunks)),
        facts=parser.facts,
        contexts=parser.contexts,
        dei=parser.dei,
    )
