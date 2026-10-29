"""LLM judge: does the narrative connect to the tables, and is it truthful? (Phase 7, automated)

PLAN.md Phase 7 lists a "model-judged realism rubric on a sample" as one of the two
non-automatic validation gates, and so far it has only been done once, by hand, as a
close read of 6/20 companies against four free-form dimensions (register,
sector-appropriate risk framing, cross-item consistency, absence of invented detail
-- see PLAN.md's "Model-judged" paragraph). This script automates that close read
over every company, with a narrower, more checkable rubric:

- table_grounding (1-5): does Item 7's narrative actually engage with what Item 8's
  tables show -- naming and explaining the figures that matter -- rather than just
  avoiding contradicting them?
- truthfulness (1-5): are Item 7's numbers, directions, and magnitude words ("modest",
  "significant") accurate against financials.json and Item 8? This is the one check
  here that method_2/step09_validate_filings.py's regex-based `numeric_consistency`
  gate can't do: that gate only confirms a quoted number exists somewhere in
  financials.json, within tolerance -- it says nothing about whether the sentence
  around that number is true. A narrative that calls a 40% revenue decline a "modest
  increase" still passes `numeric_consistency`, since "40%" and every dollar figure
  it cites are real; this check is what catches that.
- narrative_consistency (1-5): do Item 7 and Item 8 agree with each other and with
  spec.json's company facts and scenario, with no contradiction between sections?
- invented_detail: a count (not a 1-5 score) of specific names, events, or figures
  stated in Item 7 or Item 8 that aren't grounded in spec.json or financials.json.
- overall (1-5): holistic realism, folding in register and sector fit -- the closest
  analog to the original by-hand rubric's four dimensions combined.

Each dimension also gets a short rationale from the model, quoting or paraphrasing
the specific claim it's based on.

Input: data/synthetic/{fictional_id}/{fy}/{item_7.txt,item_8.txt,financials.json,
spec.json} (SYNTHETIC_DIR, same env var override step08/step09 use). Output:
data/llm_judge_report.csv, one row per company x fiscal_year x dimension, in the
same shape as validation_report.csv (fictional_id, fiscal_year, check, status,
detail) plus a numeric `score` column, so dashboard/build_dashboard.py and
dashboard/template.html render it with the same table/search/sort machinery as the
other four reports.

Two backends, same convention as method_2/step08_generate_filings.py:
- `--backend openai` (default): any OpenAI-compatible endpoint, such as Vector's
  proxy on the Coder workspace. Reads the endpoint from SYNBENCH_BASE_URL or
  BASE_URL (default https://proxy.vectorinstitute.ai/v1) and the key from
  OPENAI_API_KEY or SYNBENCH_API_KEY -- the same variables
  aieng.syn_data.synbench.llm.config and the bootcamp notebooks use. Default model
  gpt-4o: a different provider from the claude-sonnet-5-5 that wrote the narrative
  (method_2/step08_generate_filings.py's ITEM_MODELS), so the judge isn't grading its
  own homework. Uses the SDK's JSON mode (response_format={"type": "json_object"}),
  the same call shape aieng.syn_data.synbench.llm.chat_client.ChatClient.complete_json
  uses against this same proxy.
- `--backend anthropic`: the Claude API (ANTHROPIC_API_KEY). No JSON mode in this
  SDK's plain messages.create, so the prompt asks for JSON directly; the response is
  parsed the same way either backend returns it, with code-fence stripping.

Run from implementations/synthetic_10k, in the repo's uv environment:
    uv run python -m evals.llm_judge
    uv run python -m evals.llm_judge --limit 3 --workers 2   # smoke test before a full run
    uv run python -m evals.llm_judge --backend anthropic --model claude-sonnet-5-5
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR.parent))

from method_2.config import RAW_DIR, SYNTHETIC_DIR  # noqa: E402
from method_2.utils import load_dotenv, setup_logging, with_retries, write_atomic  # noqa: E402


logger = setup_logging("llm_judge")

REPORT_PATH = RAW_DIR.parent / "llm_judge_report.csv"
DEFAULT_BASE_URL = "https://proxy.vectorinstitute.ai/v1"
DEFAULT_MODEL = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
MAX_TOKENS = 1024

# 1-5 Likert checks scored directly by the model, plus `invented_detail`, a count.
SCORED_CHECKS = ["table_grounding", "truthfulness", "narrative_consistency", "overall"]

RUBRIC = {
    "table_grounding": (
        "Does Item 7's narrative actually engage with what Item 8's tables show -- "
        "naming and explaining the figures and line items that matter (the biggest "
        "dollar amounts, the biggest swings), not just failing to contradict them? "
        "A narrative that never mentions a line item Item 8 highlights, or discusses "
        "a figure Item 8 doesn't support, scores low."
    ),
    "truthfulness": (
        'Are Item 7\'s numbers, directions, and magnitude words ("modest", '
        '"significant", "sharply") accurate against the computed financials below '
        "and Item 8? A number can be correct while the characterization around it "
        'is false (e.g. calling a decline an increase, or a small change '
        '"significant") -- that still scores low here.'
    ),
    "narrative_consistency": (
        "Do Item 7 and Item 8 agree with each other, and with the company facts and "
        "scenario below, with no contradiction between sections?"
    ),
    "overall": (
        "Holistic realism: would an equity analyst reading Item 7 and Item 8 "
        "together find them to read like a genuine, internally coherent 10-K for "
        "this sector, folding in register and sector-appropriate framing as well as "
        "the three checks above?"
    ),
}

JSON_SCHEMA_HINT = """Return ONLY a single JSON object, no prose before or after it and no markdown code fence, with exactly this shape:
{
  "table_grounding": {"score": <int 1-5>, "rationale": "<1-2 sentences, citing the specific claim(s) you scored>"},
  "truthfulness": {"score": <int 1-5>, "rationale": "<...>"},
  "narrative_consistency": {"score": <int 1-5>, "rationale": "<...>"},
  "overall": {"score": <int 1-5>, "rationale": "<...>"},
  "invented_detail": {"count": <int >= 0>, "examples": ["<short quote or paraphrase>", ...], "rationale": "<...>"}
}
5 means no issues found; 1 means the section fails this check outright. Use the full
range -- most sections will not score 5 on every dimension."""


def load_json(path: Path) -> dict:
    """Return a JSON file's parsed contents."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(spec: dict, financials: dict, item_7: str, item_8: str) -> str:
    """Build the judge prompt: company facts, ground-truth figures, both sections, the rubric."""
    facts = {
        "fictional_name": spec["fictional_name"],
        "sector": spec["sector"],
        "sub_industry": spec["sub_industry"],
        "hq_state": spec["hq_state"],
        "fiscal_year": spec["fiscal_year"],
        "scenario": spec.get("scenario"),
    }
    rubric_text = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    return f"""You are an equity analyst fact-checking a FICTIONAL company's Form 10-K for a
synthetic benchmark dataset. The company does not exist; your job is to judge
whether its Item 7 (MD&A, the narrative) and Item 8 (Financial Statements, the
tables) hold together the way a real filing would.

Company facts and scenario:
{json.dumps(facts, indent=2)}

Computed ground-truth financial figures (what Item 7 and Item 8 are supposed to be
grounded in; treat anything that contradicts these as false):
{json.dumps(financials, indent=2)}

--- Item 7 (MD&A, the narrative) ---
{item_7}

--- Item 8 (Financial Statements, the tables) ---
{item_8}

Score the narrative against these four dimensions, 1 (fails) to 5 (no issues), plus
a count of invented details:
{rubric_text}
- invented_detail: any specific name, event, or figure stated in Item 7 or Item 8
  that isn't grounded in the company facts/scenario or the computed figures above
  (a count, not a 1-5 score).

{JSON_SCHEMA_HINT}
"""


def make_judge(backend: str, model: str) -> Callable[[str], str]:
    """Return a function prompt -> raw model output, for the chosen backend."""
    if backend == "anthropic":
        import anthropic  # noqa: PLC0415

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set. Set it, or use --backend openai (see this module's docstring).")
        client = anthropic.Anthropic(api_key=api_key)

        def judge_anthropic(prompt: str) -> str:
            message = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in message.content if block.type == "text")

        return judge_anthropic

    from openai import OpenAI  # noqa: PLC0415

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or SYNBENCH_API_KEY for --backend openai (see this module's docstring).")
    base_url = os.environ.get("SYNBENCH_BASE_URL") or os.environ.get("BASE_URL") or DEFAULT_BASE_URL
    client = OpenAI(base_url=base_url, api_key=api_key)

    def judge_openai(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return judge_openai


CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_judgment(raw: str) -> dict:
    """Parse and validate the model's JSON judgment. Raises on anything off-spec."""
    data = json.loads(CODE_FENCE_RE.sub("", raw.strip()))
    for check in SCORED_CHECKS:
        entry = data[check]
        score = int(entry["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"{check}: score {score} out of range 1-5")
        entry["score"] = score
        str(entry["rationale"])  # required key, must be present
    invented = data["invented_detail"]
    count = int(invented["count"])
    if count < 0:
        raise ValueError("invented_detail: negative count")
    invented["count"] = count
    return data


def status_for(check: str, score: int) -> str:
    """PASS/WARN/FAIL for one check's score. invented_detail is a count (lower is better)."""
    if check == "invented_detail":
        return "PASS" if score == 0 else ("WARN" if score == 1 else "FAIL")
    return "PASS" if score >= 4 else ("WARN" if score == 3 else "FAIL")


def judgment_rows(fictional_id: str, fiscal_year: int, judgment: dict) -> list[dict]:
    """Flatten one parsed judgment into report rows, one per dimension."""
    rows = []
    for check in SCORED_CHECKS:
        entry = judgment[check]
        rows.append(
            {
                "fictional_id": fictional_id,
                "fiscal_year": fiscal_year,
                "check": check,
                "score": entry["score"],
                "status": status_for(check, entry["score"]),
                "detail": entry["rationale"],
            },
        )
    invented = judgment["invented_detail"]
    detail = invented.get("rationale", "")
    if invented.get("examples"):
        detail = (detail + " " if detail else "") + "Examples: " + "; ".join(invented["examples"])
    rows.append(
        {
            "fictional_id": fictional_id,
            "fiscal_year": fiscal_year,
            "check": "invented_detail",
            "score": invented["count"],
            "status": status_for("invented_detail", invented["count"]),
            "detail": detail or "none found",
        },
    )
    return rows


def failure_row(fictional_id: str, fiscal_year: int, detail: str) -> list[dict]:
    """One FAIL row standing in for a company that couldn't be judged at all."""
    return [
        {
            "fictional_id": fictional_id,
            "fiscal_year": fiscal_year,
            "check": "overall",
            "score": "",
            "status": "FAIL",
            "detail": detail,
        },
    ]


def judge_company(judge: Callable[[str], str], spec_path: Path) -> list[dict]:
    """Return every report row for one fictional_id/fiscal_year, or one FAIL row on failure."""
    spec = load_json(spec_path)
    fictional_id, fiscal_year = spec["fictional_id"], spec["fiscal_year"]
    item_7_path, item_8_path = spec_path.parent / "item_7.txt", spec_path.parent / "item_8.txt"
    if not item_7_path.exists() or not item_8_path.exists():
        return failure_row(fictional_id, fiscal_year, "item_7.txt or item_8.txt missing, nothing to judge")

    financials = load_json(spec_path.parent / "financials.json")
    prompt = build_prompt(
        spec,
        financials,
        item_7_path.read_text(encoding="utf-8"),
        item_8_path.read_text(encoding="utf-8"),
    )
    try:
        judgment = with_retries(
            lambda: parse_judgment(judge(prompt)),
            f"{fictional_id} FY{fiscal_year}",
            logger,
        )
    except Exception as exc:  # noqa: BLE001 - any judge/parse failure becomes one FAIL row, not a crash
        logger.exception("%s FY%s: judging failed", fictional_id, fiscal_year)
        return failure_row(fictional_id, fiscal_year, f"judge call or parse failed: {exc}")
    return judgment_rows(fictional_id, fiscal_year, judgment)


def main() -> None:
    """Judge every company under SYNTHETIC_DIR and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", help="default: gpt-4o (openai) / claude-sonnet-5-5 (anthropic)")
    parser.add_argument("--workers", type=int, default=4, help="companies judged in parallel")
    parser.add_argument("--limit", type=int, help="judge only the first N companies (smoke test)")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # repo-root .env, if any

    model = args.model or DEFAULT_MODEL[args.backend]
    judge = make_judge(args.backend, model)

    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if args.limit:
        spec_paths = spec_paths[: args.limit]
    if not spec_paths:
        logger.warning("No specs found under %s; run method_2.step06_make_specs first", SYNTHETIC_DIR)
        return

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(lambda p: judge_company(judge, p), spec_paths):
            all_rows.extend(rows)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["fictional_id", "fiscal_year", "check", "score", "status", "detail"])
    writer.writeheader()
    writer.writerows(all_rows)
    write_atomic(args.out, buffer.getvalue())

    fails = [r for r in all_rows if r["status"] == "FAIL"]
    warns = [r for r in all_rows if r["status"] == "WARN"]
    logger.info(
        "Judged %d companies (backend=%s, model=%s): %d rows, %d FAIL, %d WARN. Report: %s",
        len(spec_paths),
        args.backend,
        model,
        len(all_rows),
        len(fails),
        len(warns),
        args.out,
    )
    for r in fails:
        logger.warning("FAIL %s FY%s %s: %s", r["fictional_id"], r["fiscal_year"], r["check"], str(r["detail"])[:300])


if __name__ == "__main__":
    main()
