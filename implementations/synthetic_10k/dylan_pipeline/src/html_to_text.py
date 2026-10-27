"""Convert SEC inline-XBRL 10-K HTML into readable plain text."""

import re

from bs4 import BeautifulSoup


# Hidden XBRL metadata and non-text elements, removed before extracting text
SKIP_TAGS = ["script", "style", "ix:header"]
# Elements that start a new line in the output
BLOCK_TAGS = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table"]


def html_to_text(html: bytes | str) -> str:
    """Return the readable text of a 10-K HTML document.

    Hidden XBRL metadata is dropped. Each table row becomes one line with its cells
    separated by " | ", and block elements start new lines.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        cells = [cell for cell in cells if cell not in ("", "$")]
        row.replace_with(f"\n{' | '.join(cells)}\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text().replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
