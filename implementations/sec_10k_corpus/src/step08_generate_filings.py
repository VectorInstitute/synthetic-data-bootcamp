"""Generate narrative text for fictional company filings via the Claude API (Phase 6).

PLACEHOLDER: this script is complete and would run, but hasn't been run against the
real API. The data/synthetic/*/{fy}/item_*.txt files in this repo were written
directly in a Claude Code session instead (see PLAN.md Phase 6 and open question 3),
so having this script in the repo costs nothing, and running it costs nothing until
someone actually does so with a real API key.

Input: data/synthetic/{fictional_id}/{fy}/spec.json and financials.json (run
src.step06_make_specs and src.step07_make_financials first), plus the matching real
exemplar at data/parsed/{template_ticker}/{template_fy}/item_*.txt. Output:
data/synthetic/{fictional_id}/{fy}/item_*.txt, one per target item, overwriting
whatever (if anything) is there.

Requires the `anthropic` package (see requirements.txt) and ANTHROPIC_API_KEY set in
the environment.

Run from the project directory:
    python -m src.step08_generate_filings
"""

import csv
import json
import os
from pathlib import Path

import anthropic

from src.config import PARSED_DIR, RAW_DIR
from src.utils import setup_logging, write_atomic


logger = setup_logging("generate_filings")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
MAX_TOKENS = 4096

# Generation order from PLAN.md Phase 6: Item 7 quotes the Item 8 numbers, so it's
# written before Item 8; earlier items are passed in as context for later ones.
GENERATION_ORDER = ["1", "1A", "7", "1C", "8"]
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1C": "Cybersecurity",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "8": "Financial Statements and Supplementary Data",
}
# One model for every item by default; PLAN.md Phase 6 suggests a stronger model for
# MD&A and a cheaper one for volume, so this is the knob for that, left untuned.
ITEM_MODELS = dict.fromkeys(GENERATION_ORDER, "claude-sonnet-5-5")


def load_json(path: Path) -> dict:
    """Return a JSON file's parsed contents."""
    return json.loads(path.read_text(encoding="utf-8"))


def real_exemplar(ticker: str, fiscal_year: int, item: str, max_chars: int = 6000) -> str:
    """Return an excerpt of the template's real item, as a style guide, not to copy."""
    path = PARSED_DIR / ticker / str(fiscal_year) / f"item_{item.lower()}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")[:max_chars]


def length_target(item: str) -> int:
    """Return the median real character count for `item`, from parse_status.csv."""
    counts = []
    with open(RAW_DIR.parent / "parse_status.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["item"] == item and row["status"] == "ok":
                counts.append(int(row["chars"]))
    counts.sort()
    return counts[len(counts) // 2] if counts else 5000


def relevant_numbers(item: str, financials: dict) -> dict:
    """Return only the figures `item` actually needs, not the full statements.

    Item 1 and 1A ground a couple of qualitative claims in numbers; Item 7 and 8 need
    everything, since 7 is the MD&A over those same figures and 8 presents them.
    """
    if item in ("7", "8"):
        return financials
    if item == "1":
        return {"revenues": financials["income_statement"]["revenues"]}
    if item == "1A":
        bs = financials["balance_sheet"]
        return {
            "total_assets": bs["assets"]["total_assets"],
            "total_liabilities": bs["liabilities"]["total_liabilities"],
            "total_stockholders_equity": bs["equity"]["total_stockholders_equity"],
        }
    return {}


def build_prompt(item: str, spec: dict, financials: dict, exemplar: str, prior_items: dict[str, str]) -> str:
    """Build the Phase 6 prompt.

    Spec facts, computed numbers, a real exemplar (style only), a length target, and
    whatever's already written for this company.
    """
    facts = {
        "fictional_name": spec["fictional_name"],
        "sector": spec["sector"],
        "sub_industry": spec["sub_industry"],
        "hq_state": spec["hq_state"],
        "fiscal_year": spec["fiscal_year"],
    }
    if item == "1C":
        facts["cybersecurity"] = spec["cybersecurity"]

    context = "\n\n".join(f"--- Item {k} (already written for this company) ---\n{v}" for k, v in prior_items.items())

    return f"""You are drafting Item {item}, "{ITEM_TITLES[item]}," of a FICTIONAL company's
Form 10-K, for a synthetic benchmark dataset. The company does not exist.

Company facts (ground every qualitative claim in these; never contradict them):
{json.dumps(facts, indent=2)}

Computed financial figures (ground every dollar figure and ratio in these exactly;
never invent a number that isn't derivable from this data):
{json.dumps(relevant_numbers(item, financials), indent=2)}

Already-written sections for this same company, for consistency:
{context or "(none yet)"}

A real company's {ITEM_TITLES[item]} section, as a STYLE and STRUCTURE guide only. Do
not copy its wording, its numbers, or any of its real facts (segment names, products,
locations, personnel, or incidents) — write an original section for the fictional
company above, in a similar register and with a similar heading structure:
---
{exemplar}
---

Target length: approximately {length_target(item)} characters. Do not mention the
real company shown above, or any other real company, by name.
"""


def generate_item(
    client: "anthropic.Anthropic",
    item: str,
    spec: dict,
    financials: dict,
    prior_items: dict[str, str],
) -> str:
    """Call the API for one item and return its text."""
    ticker = spec["template"]["ticker"]
    fiscal_year = spec["template"]["fiscal_year"]
    exemplar = real_exemplar(ticker, fiscal_year, item)
    prompt = build_prompt(item, spec, financials, exemplar, prior_items)
    response = client.messages.create(
        model=ITEM_MODELS[item],
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main() -> None:
    """Generate every item for every fictional company found under data/synthetic/."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. This script is a placeholder: the "
            "data/synthetic/*/item_*.txt files in this repo were written directly in "
            "a Claude Code session instead (see PLAN.md Phase 6), not through this "
            "script. Set ANTHROPIC_API_KEY to actually run it.",
        )
    client = anthropic.Anthropic(api_key=api_key)

    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if not spec_paths:
        logger.warning("No specs found under %s; run src.step06_make_specs first", SYNTHETIC_DIR)
        return

    for spec_path in spec_paths:
        spec = load_json(spec_path)
        financials = load_json(spec_path.parent / "financials.json")
        prior_items: dict[str, str] = {}
        for item in GENERATION_ORDER:
            text = generate_item(client, item, spec, financials, prior_items)
            write_atomic(spec_path.parent / f"item_{item.lower()}.txt", text)
            prior_items[item] = text
            logger.info("%s: wrote Item %s (%d chars)", spec["fictional_id"], item, len(text))


if __name__ == "__main__":
    main()
