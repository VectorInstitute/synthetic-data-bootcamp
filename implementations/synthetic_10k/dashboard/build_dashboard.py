"""Phase 9/10/11 dashboard: build a self-contained dashboard/index.html.

Input: dashboard/template.html, dashboard/vendor/d3.v7.9.0.min.js, plus the five
result CSVs (data/validation_report.csv, qa_pairs.csv, retrieval_eval.csv,
gap_report.csv, privacy_report.csv) and one optional sixth, data/llm_judge_report.csv
(evals/llm_judge.py's model-judged scores) -- optional because, unlike the other
five, it needs a real API call rather than coming free with the rest of the
pipeline; its card on the built page just renders empty until it's been run once.
Output: dashboard/index.html.

The template has no embedded data and loads d3 from a CDN -- opened directly, its
script falls back to a live, cache-busted `fetch()` of the five CSVs (see the
template's own comment), which only works served over http:// (browsers block
`fetch()` over file://, and a local server also lets you edit the template and
reload without re-running this script); the CDN d3 also needs network either way.
This script instead (a) reads the five CSVs and injects them as a
`window.__DASHBOARD_DATA__` object, and (b) inlines the vendored d3 source in place
of the CDN `<script src>` tag, so the built file has zero network dependencies: open
it directly from a file manager, or double-click it, with no internet connection,
same as any other local file.

Like every other output in this pipeline, the built file goes stale the moment the
CSVs it was built from change -- re-run this after re-running step09/11/12/13/14, the
same as you'd re-run step12 after step11, or step03 after the synthetic corpus
changes. Nothing here refreshes itself.

Run from implementations/synthetic_10k (after step09, step11, step12, step13, step14):
    uv run python -m dashboard.build_dashboard
"""

import csv
import json
from pathlib import Path

from method_2.config import RAW_DIR
from method_2.utils import setup_logging, write_atomic


logger = setup_logging("build_dashboard")

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
TEMPLATE_PATH = DASHBOARD_DIR / "template.html"
VENDOR_D3_PATH = DASHBOARD_DIR / "vendor" / "d3.v7.9.0.min.js"
OUTPUT_PATH = DASHBOARD_DIR / "index.html"
DATA_DIR = RAW_DIR.parent

DATASETS = {
    "validation": "validation_report.csv",
    "qa": "qa_pairs.csv",
    "retrieval": "retrieval_eval.csv",
    "gap": "gap_report.csv",
    "privacy": "privacy_report.csv",
}
# Opt-in: needs a real API call (evals/llm_judge.py) rather than coming free with the
# rest of the pipeline, so unlike DATASETS above its absence isn't a build error --
# the dashboard just renders that one card empty until someone runs it.
OPTIONAL_DATASETS = {
    "judge": "llm_judge_report.csv",
}

D3_SCRIPT_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'


def load_csv(filename: str) -> list[dict]:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_csv_optional(filename: str) -> list[dict]:
    """Like load_csv, but a missing file is a warning, not a crash."""
    if not (DATA_DIR / filename).exists():
        logger.warning("%s not found; run `uv run python -m evals.llm_judge` to populate that card", filename)
        return []
    return load_csv(filename)


def main() -> None:
    """Embed the five result CSVs and the vendored d3 source into dashboard/template.html,
    writing dashboard/index.html.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if D3_SCRIPT_TAG not in template:
        raise SystemExit(f"Expected to find {D3_SCRIPT_TAG!r} in {TEMPLATE_PATH} to anchor the injected data")

    d3_source = VENDOR_D3_PATH.read_text(encoding="utf-8")
    data = {key: load_csv(filename) for key, filename in DATASETS.items()}
    data.update({key: load_csv_optional(filename) for key, filename in OPTIONAL_DATASETS.items()})

    # `</script` inside a JSON string (or, in principle, the vendored d3 source)
    # would otherwise close this script tag early -- the HTML parser doesn't know
    # it's inside a string -- none of the current data contains it, but a future
    # narrative edit could.
    data_json = json.dumps(data).replace("</script", "<\\/script")
    d3_inline_script = f"<script>\n{d3_source.replace('</script', '<\\/script')}\n</script>"
    data_script = f"<script>\nwindow.__DASHBOARD_DATA__ = {data_json};\n</script>"

    built = template.replace(D3_SCRIPT_TAG, f"{d3_inline_script}\n{data_script}")
    write_atomic(OUTPUT_PATH, built)

    counts = ", ".join(f"{key}={len(rows)}" for key, rows in data.items())
    logger.info("Wrote %s (d3 vendored, %s)", OUTPUT_PATH, counts)


if __name__ == "__main__":
    main()
