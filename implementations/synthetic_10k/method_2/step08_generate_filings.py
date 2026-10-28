"""Generate narrative text for fictional company filings via the Claude API (Phase 6).

PLACEHOLDER: this script is complete and would run, but hasn't been run against the
real API. The data/synthetic/*/{fy}/item_*.txt files in this repo were written
directly in a Claude Code session instead (see PLAN.md Phase 6 and open question 3),
so having this script in the repo costs nothing, and running it costs nothing until
someone actually does so with a real API key.

Input: data/synthetic/{fictional_id}/{fy}/spec.json and financials.json (run
method_2.step06_make_specs and method_2.step07_make_financials first), plus the matching real
exemplar at data/parsed/{template_ticker}/{template_fy}/item_*.txt. Output:
data/synthetic/{fictional_id}/{fy}/item_*.txt, one per target item, overwriting
whatever (if anything) is there.

Two backends:
- `--backend anthropic` (default): the Claude API. Requires ANTHROPIC_API_KEY, from the
  environment or the repo-root .env. `--effort` (default high) sets how much the model
  thinks; thinking gets 32,000 tokens on top of --max-tokens.
- `--backend openai`: any OpenAI-compatible endpoint, such as Vector's proxy on the
  Coder workspace. Reads the endpoint from SYNBENCH_BASE_URL (default
  https://proxy.vectorinstitute.ai/v1) and the key from OPENAI_API_KEY or
  SYNBENCH_API_KEY, the same variables the bootcamp's notebooks use. Pass --model,
  e.g. gemini-3-flash-preview.

Item files that already exist (and aren't empty) are kept and passed on as context
unless --overwrite is given, so an interrupted run resumes with only the missing items.

Run from the project directory:
    python -m method_2.step08_generate_filings
    python -m method_2.step08_generate_filings --backend openai \
        --model gemini-3-flash-preview --workers 4
"""

import argparse
import csv
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from method_2.config import PARSED_DIR, RAW_DIR, SYNTHETIC_DIR
from method_2.utils import load_dotenv, setup_logging, write_atomic


logger = setup_logging("generate_filings")

MAX_TOKENS = 4096
THINKING_TOKENS = 32000  # Anthropic backend: room for thinking on top of max_tokens
DEFAULT_BASE_URL = "https://proxy.vectorinstitute.ai/v1"

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


def make_complete(backend: str, model: str | None, max_tokens: int, effort: str = "high") -> Callable[[str, str], str]:
    """Return a function prompt, item -> generated text, for the chosen backend."""
    if backend == "anthropic":
        import anthropic  # noqa: PLC0415

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Set it, or use --backend openai with "
                "Vector's proxy (see this module's docstring).",
            )
        client = anthropic.Anthropic(api_key=api_key)

        def complete_anthropic(prompt: str, item: str) -> str:
            # Current Claude models always think, so the response can open with a
            # thinking block: keep only the text blocks. Streamed, since thinking
            # plus a long section can outlast a plain request's timeout.
            with client.messages.stream(
                model=model or ITEM_MODELS[item],
                max_tokens=max_tokens + THINKING_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                output_config=cast("anthropic.types.OutputConfigParam", {"effort": effort}),
            ) as stream:
                message = stream.get_final_message()
            if message.stop_reason == "refusal":
                raise RuntimeError(f"Item {item}: the model refused ({message.stop_details})")
            return "".join(block.text for block in message.content if block.type == "text")

        return complete_anthropic

    from openai import OpenAI  # noqa: PLC0415

    if not model:
        raise SystemExit("--backend openai needs --model, e.g. gemini-3-flash-preview")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or SYNBENCH_API_KEY for --backend openai.")
    openai_client = OpenAI(base_url=os.environ.get("SYNBENCH_BASE_URL", DEFAULT_BASE_URL), api_key=api_key)

    def complete_openai(prompt: str, item: str) -> str:
        response = openai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return complete_openai


def generate_item(complete, item: str, spec: dict, financials: dict, prior_items: dict[str, str]) -> str:
    """Call the API for one item and return its text."""
    ticker = spec["template"]["ticker"]
    fiscal_year = spec["template"]["fiscal_year"]
    exemplar = real_exemplar(ticker, fiscal_year, item)
    prompt = build_prompt(item, spec, financials, exemplar, prior_items)
    return complete(prompt, item)


def generate_company(complete, spec_path: Path, overwrite: bool) -> None:
    """Write every item for one fictional company, in GENERATION_ORDER."""
    spec = load_json(spec_path)
    paths = {item: spec_path.parent / f"item_{item.lower()}.txt" for item in GENERATION_ORDER}
    if not overwrite and all(p.exists() for p in paths.values()):
        logger.info("%s: all items exist, skipping", spec["fictional_id"])
        return
    financials = load_json(spec_path.parent / "financials.json")
    prior_items: dict[str, str] = {}
    for item in GENERATION_ORDER:
        # Resume: reuse an item already written by an earlier, interrupted run, so a
        # rerun only pays for the items that are missing (or empty).
        if not overwrite and paths[item].exists() and paths[item].stat().st_size:
            prior_items[item] = paths[item].read_text(encoding="utf-8")
            continue
        text = generate_item(complete, item, spec, financials, prior_items)
        write_atomic(paths[item], text)
        prior_items[item] = text
        logger.info("%s: wrote Item %s (%d chars)", spec["fictional_id"], item, len(text))


def main() -> None:
    """Generate every item for every fictional company found under SYNTHETIC_DIR."""
    parser = argparse.ArgumentParser(description="Generate narrative for fictional companies.")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", help="one model for every item (default: ITEM_MODELS)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
        help="Anthropic backend only: thinking effort",
    )
    parser.add_argument("--workers", type=int, default=1, help="companies generated in parallel")
    parser.add_argument("--overwrite", action="store_true", help="regenerate finished companies")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # repo-root .env, if any
    complete = make_complete(args.backend, args.model, args.max_tokens, args.effort)

    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if not spec_paths:
        logger.warning("No specs found under %s; run method_2.step06_make_specs first", SYNTHETIC_DIR)
        return

    failures = []

    def run(spec_path: Path) -> None:
        try:
            generate_company(complete, spec_path, args.overwrite)
        except Exception:
            logger.exception("%s: generation failed", spec_path.parent.parent.name)
            failures.append(spec_path.parent.parent.name)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, spec_paths))
    logger.info("Done: %d companies, %d failed %s", len(spec_paths), len(failures), sorted(failures))


if __name__ == "__main__":
    main()
