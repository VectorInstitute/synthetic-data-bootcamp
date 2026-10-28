"""Parse a folder of downloaded 10-K .htm files into sections, facts and an index.

Usage::

    python -m method_1.build --input path/to/10K_Reports --output data

Input files are named ``{filing-date}_{ticker}_{company}_{accession}.htm``. Output::

    parsed/{ticker}/{fy}/item_{1,1a,1c,7,7a,8}.txt  section text
    parsed/{ticker}/{fy}/full_text.txt              whole filing as text
    parsed/{ticker}/{fy}/facts.csv                  every numeric XBRL fact
    parsed/{ticker}/{fy}/metadata.json              cover facts + headline metrics
    index.csv                                       one row per section, with status
    financials.csv                                  one row per filing, metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from method_1.financials import (
    DURATION_METRICS,
    INSTANT_METRICS,
    fiscal_period_end,
    key_financials,
)
from method_1.reader import read_filing
from method_1.sections import DEFAULT_ITEMS, split_sections


logger = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "ticker",
    "company",
    "fiscal_year",
    "period_end",
    "filing_date",
    "accession",
    "item",
    "status",
    "chars",
    "words",
    "path",
    "source",
]
METRIC_COLUMNS = [*DURATION_METRICS, *INSTANT_METRICS]


@dataclass
class FilingName:
    """Fields encoded in a downloaded filing's file name."""

    filing_date: str
    ticker: str
    company: str
    accession: str

    @classmethod
    def from_path(cls, path: Path) -> FilingName:
        """Split ``2025-10-31_AAPL_Apple_Inc._0000320193-25-000079.htm``."""
        parts = path.stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected filing name: {path.name}")
        return cls(parts[0], parts[1], " ".join(parts[2:-1]), parts[-1])


def process_filing(path: Path, output: Path, source: str = "real") -> dict[str, Any]:
    """Parse one filing and write its outputs; return its index and metrics rows."""
    name = FilingName.from_path(path)
    filing = read_filing(path)
    period_end = fiscal_period_end(filing)
    fiscal_year = (
        filing.dei.get("dei:DocumentFiscalYearFocus")
        or (period_end or name.filing_date)[:4]
    )
    company = filing.dei.get("dei:EntityRegistrantName") or name.company
    out_dir = output / "parsed" / name.ticker / fiscal_year
    out_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "ticker": name.ticker,
        "company": company,
        "fiscal_year": fiscal_year,
        "period_end": period_end or "",
        "filing_date": name.filing_date,
        "accession": name.accession,
        "source": source,
    }
    index_rows = []
    for item, section in split_sections(filing.text, DEFAULT_ITEMS).items():
        section_path = out_dir / f"item_{item.lower()}.txt"
        section_path.write_text(section.text)
        index_rows.append(
            {
                **base,
                "item": item,
                "status": section.status,
                "chars": len(section.text),
                "words": len(section.text.split()),
                "path": str(section_path.relative_to(output)),
            },
        )

    with (out_dir / "facts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "concept",
                "value",
                "raw",
                "unit",
                "decimals",
                "start",
                "end",
                "instant",
                "dimensions",
            ]
        )
        for fact in filing.facts:
            ctx = filing.contexts.get(fact.context_id)
            writer.writerow(
                [
                    fact.concept,
                    fact.value,
                    fact.raw,
                    fact.unit,
                    fact.decimals,
                    ctx.start if ctx else "",
                    ctx.end if ctx else "",
                    ctx.instant if ctx else "",
                    json.dumps(ctx.dimensions) if ctx and ctx.dimensions else "",
                ],
            )

    metrics = key_financials(filing)
    dei = {k: v for k, v in filing.dei.items() if not k.endswith("#context")}
    metadata = {
        **base,
        "file": path.name,
        "n_facts": len(filing.facts),
        "text_chars": len(filing.text),
        "dei": dei,
        "metrics": metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "full_text.txt").write_text(filing.text)
    return {
        "index": index_rows,
        "financials": {**base, "n_facts": len(filing.facts), **metrics},
    }


def _safe_process(args: tuple[Path, Path, str]) -> dict[str, Any]:
    path, output, source = args
    try:
        return process_filing(path, output, source)
    except Exception as exc:  # one bad filing should not stop the batch
        logger.exception("Failed to parse %s", path.name)
        return {"error": f"{path.name}: {exc}"}


def build(
    input_dir: Path,
    output: Path,
    workers: int = 2,
    source: str = "real",
    limit: int | None = None,
) -> None:
    """Parse every .htm in ``input_dir`` and write the index and financials tables."""
    paths = sorted(input_dir.glob("*.htm"))[:limit]
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Parsing %d filings from %s with %d workers", len(paths), input_dir, workers
    )

    index_rows: list[dict[str, Any]] = []
    financial_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(
            pool.map(_safe_process, [(p, output, source) for p in paths]), start=1
        ):
            if "error" in result:
                errors.append(result["error"])
            else:
                index_rows.extend(result["index"])
                financial_rows.append(result["financials"])
            if i % 25 == 0:
                logger.info("%d/%d filings parsed", i, len(paths))

    with (output / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)
    with (output / "financials.csv").open("w", newline="") as f:
        fieldnames = [
            "ticker",
            "company",
            "fiscal_year",
            "period_end",
            "filing_date",
            "accession",
            "source",
            "n_facts",
            *METRIC_COLUMNS,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(financial_rows)
    (output / "errors.txt").write_text("\n".join(errors))

    counts: dict[str, int] = {}
    for row in index_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    logger.info("Sections by status: %s; %d filings failed", counts, len(errors))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Folder of downloaded 10-K .htm files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output folder (default: data)",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Parallel processes (default: 2)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Parse only the first N files"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build(args.input, args.output, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
