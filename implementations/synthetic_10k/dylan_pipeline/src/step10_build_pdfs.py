"""Render each fictional company's narrative as a standalone 10-K-style PDF.

Input: data/synthetic/{fictional_id}/{fy}/{spec.json,item_*.txt}.
Output: data/pdf_reports/{fictional_id}_{fy}.pdf (git-ignored; regenerate with this
script any time the narrative changes -- it is a derived artifact, not source data).

The narrative files store table rows as plain "label | value" lines (and
"label | value | percent" for the few three-column tables), not Markdown table
syntax, so `tableize()` promotes each contiguous run of pipe rows into a real
Pandoc pipe table before rendering. Two bugs bit this during development and are
worth not repeating:

- A pipe table glued directly onto a preceding non-blank line (e.g. a "Assets"
  label right above the first row) gets that line absorbed into the table header,
  corrupting it. Pandoc's pipe-table syntax requires a blank line both before and
  after the table block.
- `$` figures in the prose (e.g. "$5,825.6 million") must not be parsed as LaTeX
  math spans. `-f markdown-tex_math_dollars` disables that one Markdown extension
  while leaving everything else (including smart quotes/dashes) on.

Requires `pandoc` and a `--pdf-engine=xelatex` LaTeX install (both on PATH).

Run from the project directory:
    python -m src.step10_build_pdfs
"""

import json
import subprocess
from pathlib import Path

from src.config import PROJECT_DIR, RAW_DIR
from src.utils import setup_logging


logger = setup_logging("build_pdfs")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
OUTPUT_DIR = PROJECT_DIR / "data" / "pdf_reports"

# Order they appear in a real 10-K; see step09_validate_filings.ITEMS for the same list.
ITEM_TITLES = {
    "1": "Item 1. Business",
    "1a": "Item 1A. Risk Factors",
    "1c": "Item 1C. Cybersecurity",
    "7": "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "8": "Item 8. Financial Statements and Supplementary Data",
}


def tableize(text: str) -> str:
    """Turn every contiguous run of "label | value[ | percent]" lines into a Pandoc pipe table."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if " | " in line and line.strip():
            block = []
            while i < len(lines) and " | " in lines[i] and lines[i].strip():
                block.append([p.strip() for p in lines[i].split(" | ")])
                i += 1
            n_cols = max(len(row) for row in block)
            header = ["Line item", "Amount", "% of total"][:n_cols]
            if out and out[-1].strip():
                out.append("")
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join(["---"] * n_cols) + "|")
            for row in block:
                row = row + [""] * (n_cols - len(row))
                out.append("| " + " | ".join(row) + " |")
            out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def build_markdown(spec: dict, texts: dict[str, str]) -> str:
    """Assemble one company's cover page plus every item into a single Markdown document."""
    name, fid, fy = spec["fictional_name"], spec["fictional_id"], spec["fiscal_year"]
    parts = [
        f"# {name}\n",
        f"**Form 10-K for the Fiscal Year Ended {fy}**\n",
        f"Ticker: {fid} | Sector: {spec['sector']} | HQ: {spec['hq_state']}\n",
        (
            f"> **NOTE:** This is a synthetic filing generated for a RAG-evaluation benchmark "
            f"dataset. {name} is a fictional company; it does not exist. Every figure below is "
            f"internally consistent and traceable to this project's computed financial statements, "
            f"but nothing here describes a real company, and it should not be treated as an actual "
            f"SEC filing.\n"
        ),
        "\\newpage\n",
    ]
    item_keys = [k for k in ITEM_TITLES if k in texts]
    for idx, item in enumerate(item_keys):
        parts.append(f"# {ITEM_TITLES[item]}\n")
        parts.append(tableize(texts[item]))
        if idx < len(item_keys) - 1:
            parts.append("\n\\newpage\n")
    return "\n".join(parts)


def build_pdf(spec_path: Path) -> Path:
    """Render one company-year's PDF and return its output path."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    fid, fy = spec["fictional_id"], spec["fiscal_year"]

    texts = {}
    for item in ITEM_TITLES:
        item_path = spec_path.parent / f"item_{item}.txt"
        if item_path.exists():
            texts[item] = item_path.read_text(encoding="utf-8")

    markdown = build_markdown(spec, texts)
    md_path = OUTPUT_DIR / f"{fid}_{fy}.md"
    md_path.write_text(markdown, encoding="utf-8")

    pdf_path = OUTPUT_DIR / f"{fid}_{fy}.pdf"
    # Must end in ".pdf": pandoc infers the OUTPUT format from -o's extension, not
    # from --pdf-engine, so a ".pdf.tmp" path silently renders HTML instead.
    tmp_pdf_path = OUTPUT_DIR / f"{fid}_{fy}.tmp.pdf"
    subprocess.run(
        [
            "pandoc",
            str(md_path),
            "-f", "markdown-tex_math_dollars",
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-M", f"title={spec['fictional_name']} ({fid}) - Form 10-K",
            "-M", f"date=FY{fy}",
            "-o", str(tmp_pdf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tmp_pdf_path.replace(pdf_path)
    md_path.unlink()
    return pdf_path


def main() -> None:
    """Build a PDF for every fictional company-year under SYNTHETIC_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if not spec_paths:
        logger.warning("No specs found under %s; run src.step06_make_specs first", SYNTHETIC_DIR)
        return

    built, failed = [], []
    for spec_path in spec_paths:
        fid_fy = f"{spec_path.parent.parent.name}/{spec_path.parent.name}"
        try:
            pdf_path = build_pdf(spec_path)
            built.append(pdf_path)
            logger.info("%s: wrote %s", fid_fy, pdf_path)
        except subprocess.CalledProcessError as exc:
            failed.append(fid_fy)
            logger.error("%s: pandoc failed: %s", fid_fy, exc.stderr.strip()[-2000:])

    logger.info("Built %d PDFs in %s (%d failed)", len(built), OUTPUT_DIR, len(failed))
    if failed:
        logger.warning("Failed: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
