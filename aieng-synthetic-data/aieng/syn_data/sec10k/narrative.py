"""Write the narrative Items of each fictional company's 10-K with an LLM.

Usage::

    # Write the prompts only (no API calls), to inspect them and their size
    python -m aieng.syn_data.sec10k.narrative --data data/sec10k --limit 2 --dry-run

    # Generate two filings with Gemini Flash through the Vector proxy
    python -m aieng.syn_data.sec10k.narrative --data data/sec10k --limit 2

For each spec in ``data/sec10k/synthetic/specs/`` the model writes Items 1, 1A,
7 and 1C, in that order, each prompt holding the company's facts sheet, the
items already written, a length target from ``calibration.json`` and an excerpt
of the same Item from a real company in the same sector as a style example.
Item 8 is rendered from the statements by code. Each section is validated
(``validate.py``) and regenerated once with the issues as feedback if it fails.

Writes ``data/sec10k/synthetic/filings/{company_id}/``: ``item_*.txt``,
``filing.txt`` (all Items with a cover), ``prompts/`` and ``metrics.json``
(time, tokens, attempts and issues per call). Every run also writes
``synthetic/runs/{run_id}.json`` and appends its totals to ``runs.jsonl``.

The Vector proxy is reached with the ``openai`` SDK at ``SYNBENCH_BASE_URL``
(default ``https://proxy.vectorinstitute.ai/v1``) with ``OPENAI_API_KEY``, as in
``implementations/agent_benchmark_generation/1-check_access_to_model.ipynb``.
``--model`` picks the model (default ``gemini-3-flash-preview``) and
``--list-models`` prints the ids the proxy serves.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from aieng.syn_data.sec10k.calibrate import load_sectors
from aieng.syn_data.sec10k.render import facts_sheet, item_8
from aieng.syn_data.sec10k.specs import sample_quantile
from aieng.syn_data.sec10k.validate import (
    HEADINGS,
    exemplar_overlap,
    real_name_patterns,
    validate_section,
    word_count,
)
from aieng.syn_data.synbench.llm.config import (
    get_api_key,
    get_base_url,
    get_reasoning_effort,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"
GENERATION_ORDER = ["1", "1A", "7", "1C"]
FILING_ORDER = ["1", "1A", "1C", "7", "8"]
# Real sections run long (median Item 1A is ~11,000 words); targets are sampled
# from calibration.json and capped here to keep a pilot cheap.
WORD_CAPS = {"1": 3000, "1A": 3500, "7": 3000, "1C": 900}
EXEMPLAR_WORDS = 1200
CONTEXT_WORDS = 1500  # of each earlier Item, shown for consistency

SYSTEM = """\
You write sections of annual reports on Form 10-K for a fictional US public \
company, in the register of real SEC filings: formal, specific, and organised \
under sub-headings.
Rules:
- Plain text only. No markdown: no #, no ** and no bullet symbols other than \
"-". Sub-headings go on their own line.
- The company and its numbers are given in the facts sheet. Never invent other \
dollar amounts or percentages; when you cite a figure, copy it from the facts \
sheet (amounts in millions as "$1,234 million", or billions with one decimal \
as "$1.2 billion").
- Never name a real company, including the one in the style example. Describe \
competitors, customers and suppliers generically.
- Do not copy sentences from the style example; follow its structure and tone \
only."""

GUIDANCE = {
    "1": (
        "Describe the business: overview, principal products and services, two "
        "or three reportable segments described qualitatively, customers, sales "
        "and marketing, competition, seasonality, intellectual property, "
        "government regulation, human capital (state the number of employees from "
        "the facts sheet) and available information. The only dollar figure you "
        "may cite is total revenue."
    ),
    "1A": (
        "List the material risks, grouped under sub-headings (for example risks "
        "related to the business and industry, to operations, to financing and "
        "capital structure, to legal and regulatory matters, and general risks). "
        "Each risk starts with a one-sentence heading line, followed by one or two "
        "paragraphs. Make the risks specific to this company's sector, cost "
        "structure and balance sheet. Avoid numbers."
    ),
    "7": (
        "Write MD&A with these parts: overview; results of operations comparing "
        "the current with the prior fiscal year line by line (revenue, cost of "
        "revenue and gross margin, each operating expense, operating income, "
        "interest expense, effective tax rate, net income), giving plausible "
        "qualitative drivers; liquidity and capital resources (cash, operating, "
        "investing and financing cash flows, capital expenditures, debt, "
        "dividends and share repurchases); and critical accounting estimates. "
        "Quote amounts and percentages exactly as they appear in the facts sheet."
    ),
    "1C": (
        "Describe cybersecurity risk management and strategy (processes, third "
        "parties, assessment of risks from threats) and governance (board "
        "oversight committee, the role and experience of the chief information "
        "security officer, reporting lines). Avoid numbers."
    ),
}


@dataclass
class Completion:
    """One model response with its usage and timing."""

    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None


class ChatClient(Protocol):
    """Anything that turns a system and user prompt into a ``Completion``."""

    def chat(self, system: str, prompt: str, *, max_tokens: int) -> Completion:
        """Generate one completion."""
        ...


class UsageClient:
    """Chat client for the Vector proxy that also returns token usage and latency.

    Connects as in ``implementations/agent_benchmark_generation/
    1-check_access_to_model.ipynb``: the ``openai`` SDK with ``base_url`` from
    ``SYNBENCH_BASE_URL`` (default ``https://proxy.vectorinstitute.ai/v1``) and
    the key from ``OPENAI_API_KEY``, both also read from a ``.env`` file. The SDK
    retries rate limits and server errors itself.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort or get_reasoning_effort(model)
        self.client = OpenAI(
            base_url=get_base_url(),
            api_key=get_api_key(),
            max_retries=max_retries,
            timeout=timeout,
        )

    def chat(self, system: str, prompt: str, *, max_tokens: int) -> Completion:
        """Call the chat completions API once (with the SDK's retries)."""
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            **extra,
        )
        return _completion(response.model_dump(), self.model, start)

    def list_models(self) -> list[str]:
        """Return the model ids served by the endpoint."""
        return sorted(m.id for m in self.client.models.list())


def _completion(payload: dict[str, Any], model: str, start: float) -> Completion:
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return Completion(
        text=choice["message"].get("content") or "",
        model=payload.get("model") or model,
        seconds=round(time.perf_counter() - start, 2),
        prompt_tokens=usage.get("prompt_tokens") or 0,
        completion_tokens=usage.get("completion_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
        reasoning_tokens=details.get("reasoning_tokens") or 0,
        finish_reason=choice.get("finish_reason"),
    )


@dataclass
class Corpus:
    """Real filings used for style examples, length targets and name checks."""

    data: Path
    calibration: dict[str, Any]
    sections: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    real_names: list[re.Pattern[str]] = field(default_factory=list)

    @classmethod
    def load(cls, data: Path) -> Corpus:
        """Read calibration.json, index.csv and the real company names."""
        corpus = cls(data, json.loads((data / "calibration.json").read_text()))
        sectors = load_sectors()
        companies = set()
        index = data / "index.csv"
        if index.exists():
            with index.open() as f:
                for row in csv.DictReader(f):
                    companies.add(row["company"])
                    if row["status"] != "ok" or row.get("source", "real") != "real":
                        continue
                    key = (sectors.get(row["ticker"], "Unknown"), row["item"])
                    corpus.sections.setdefault(key, []).append(row["path"])
        corpus.real_names = real_name_patterns(sorted(companies))
        return corpus

    def exemplar(self, sector: str, item: str, rng: random.Random) -> str:
        """Return an excerpt of a real same-sector Item ("" when none is on disk)."""
        paths = [
            p for p in self.sections.get((sector, item), []) if (self.data / p).exists()
        ]
        if not paths:
            return ""
        words = (self.data / rng.choice(sorted(paths))).read_text().split(" ")
        excerpt = " ".join(words[:EXEMPLAR_WORDS])
        return excerpt + (" [...]" if len(words) > EXEMPLAR_WORDS else "")

    def target_words(self, item: str, rng: random.Random, scale: float) -> int:
        """Sample a section length from the real distribution, then cap it."""
        q = self.calibration.get("section_words", {}).get(item)
        sampled = sample_quantile(q, rng) if q else WORD_CAPS[item]
        return max(200, int(round(min(sampled, WORD_CAPS[item]) * scale, -1)))


def build_prompt(
    spec: dict[str, Any],
    item: str,
    *,
    target: int,
    exemplar: str,
    earlier: dict[str, str],
) -> str:
    """Assemble the user prompt for one Item."""
    parts = [f"FACTS SHEET (all figures in USD millions)\n{facts_sheet(spec)}"]
    if exemplar:
        parts.append(
            f"STYLE EXAMPLE: the start of Item {item} from a real "
            f"{spec['sector']} company's 10-K. Follow its structure and tone; do "
            f"not copy its wording, names or numbers.\n<<<\n{exemplar}\n>>>"
        )
    for done, text in earlier.items():
        words = text.split(" ")
        shown = " ".join(words[:CONTEXT_WORDS])
        cut = " [...]" if len(words) > CONTEXT_WORDS else ""
        parts.append(
            f"ALREADY WRITTEN, Item {done} of this filing (stay consistent with "
            f"it):\n<<<\n{shown}{cut}\n>>>"
        )
    parts.append(
        f"TASK\nWrite {HEADINGS[item]} for {spec['name']}'s annual report for the "
        f"fiscal year ended {spec['fiscal_year_end']}. {GUIDANCE[item]}\n"
        f"Length: about {target} words. Start with the line "
        f'"{HEADINGS[item]}" and output only the section text.'
    )
    return "\n\n".join(parts)


def clean_output(text: str, item: str) -> str:
    """Strip markdown the model adds anyway and make sure the heading is there."""
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*[*•]\s+", "- ", text, flags=re.MULTILINE).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    if not re.match(rf"item\s+{re.escape(item)}\.", text, re.IGNORECASE):
        text = f"{HEADINGS[item]}\n\n{text}"
    return text + "\n"


def _max_tokens(target: int) -> int:
    # About 1.4 tokens per word, doubled for headroom, plus room for thinking.
    return int(target * 2.8) + 8192


def generate_item(
    client: ChatClient,
    spec: dict[str, Any],
    item: str,
    *,
    prompt: str,
    target: int,
    check: Callable[[str], list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    """Generate one Item, regenerating once with feedback if validation fails."""
    calls: list[dict[str, Any]] = []
    text = ""
    issues: list[str] = []
    for attempt in (1, 2):
        user = prompt
        if issues:
            user += (
                "\n\nA previous draft had these problems; fix all of them:\n- "
                + "\n- ".join(issues)
            )
        result = client.chat(SYSTEM, user, max_tokens=_max_tokens(target))
        text = clean_output(result.text, item)
        issues = check(text)
        record = asdict(result)
        del record["text"]
        calls.append(
            {
                "item": item,
                "attempt": attempt,
                **record,
                "words": word_count(text),
                "issues": issues,
            }
        )
        logger.info(
            "%s Item %s attempt %d: %d words, %d tokens, %.1fs, %d issues",
            spec["company_id"],
            item,
            attempt,
            word_count(text),
            result.total_tokens,
            result.seconds,
            len(issues),
        )
        if not issues:
            break
    return text, calls


def assemble_filing(spec: dict[str, Any], items: dict[str, str]) -> str:
    """Join the Items under a cover page in 10-K order."""
    cover = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nFORM 10-K\n"
        "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES "
        "EXCHANGE ACT OF 1934\n"
        f"For the fiscal year ended {spec['fiscal_year_end']}\n\n"
        f"{spec['name']}\n(Ticker: {spec['ticker']})\n"
        f"State of incorporation: {spec['state_of_incorporation']}\n"
        "This is a synthetic filing for a fictional company.\n"
    )
    part_1 = [items[i] for i in ("1", "1A", "1C") if i in items]
    part_2 = [items[i] for i in ("7", "8") if i in items]
    return "\n\n".join([cover, "PART I", *part_1, "PART II", *part_2])


def _totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("seconds", "prompt_tokens", "completion_tokens", "total_tokens")
    out: dict[str, Any] = {k: sum(c.get(k, 0) for c in calls) for k in keys}
    out["reasoning_tokens"] = sum(c.get("reasoning_tokens", 0) for c in calls)
    out["seconds"] = round(out["seconds"], 2)
    out["calls"] = len(calls)
    return out


def write_company(
    spec: dict[str, Any],
    corpus: Corpus,
    client: ChatClient | None,
    *,
    out_root: Path,
    length_scale: float = 1.0,
) -> dict[str, Any]:
    """Generate (or, without a client, only prompt) one company's filing."""
    cid = spec["company_id"]
    out = out_root / cid
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    rng = random.Random(cid)
    started = time.perf_counter()
    items: dict[str, str] = {"8": item_8(spec)}
    calls: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}
    for item in GENERATION_ORDER:
        target = corpus.target_words(item, rng, length_scale)
        exemplar = corpus.exemplar(spec["sector"], item, rng)
        earlier = {i: items[i] for i in GENERATION_ORDER if i in items}
        if client is None:  # dry run: show where earlier Items will go
            earlier = {
                i: f"[Item {i} as generated]"
                for i in GENERATION_ORDER[: GENERATION_ORDER.index(item)]
            }
        prompt = build_prompt(
            spec, item, target=target, exemplar=exemplar, earlier=earlier
        )
        (out / "prompts" / f"item_{item.lower()}.txt").write_text(
            SYSTEM + "\n\n" + prompt
        )
        sections[item] = {
            "target_words": target,
            "exemplar_words": word_count(exemplar),
            "prompt_chars": len(SYSTEM) + len(prompt),
        }
        if client is None:
            continue

        def check(
            text: str, item: str = item, exemplar: str = exemplar, target: int = target
        ) -> list[str]:
            return validate_section(
                text,
                item=item,
                spec=spec,
                target_words=target,
                exemplar=exemplar,
                real_names=corpus.real_names,
            )

        text, item_calls = generate_item(
            client, spec, item, prompt=prompt, target=target, check=check
        )
        items[item] = text
        calls += item_calls
        sections[item].update(
            words=word_count(text),
            attempts=len(item_calls),
            exemplar_overlap=round(exemplar_overlap(text, exemplar), 4),
            issues=item_calls[-1]["issues"],
        )
        (out / f"item_{item.lower()}.txt").write_text(text)

    (out / "item_8.txt").write_text(items["8"])
    dry_run = client is None
    if not dry_run:
        (out / "filing.txt").write_text(assemble_filing(spec, items))
    metrics = {
        "company_id": cid,
        "name": spec["name"],
        "sector": spec["sector"],
        "dry_run": dry_run,
        "passed": not dry_run and all(not s["issues"] for s in sections.values()),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "totals": _totals(calls),
        "sections": sections,
        "calls": calls,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run(
    specs: list[dict[str, Any]],
    corpus: Corpus,
    client: ChatClient | None,
    *,
    out_root: Path,
    workers: int = 1,
    length_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate filings for ``specs``, ``workers`` companies at a time."""

    def one(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            return write_company(
                spec, corpus, client, out_root=out_root, length_scale=length_scale
            )
        except Exception as exc:  # keep going; the run summary records it
            logger.exception("Failed on %s", spec["company_id"])
            return {"company_id": spec["company_id"], "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, specs))


def summarize(
    results: list[dict[str, Any]], *, run_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Run-level record: config, per-company totals and grand totals."""
    done = [r for r in results if "totals" in r]
    calls = [c for r in done for c in r["calls"]]
    totals = _totals(calls)
    n = max(1, len(done))
    return {
        "run_id": run_id,
        **config,
        "companies": len(results),
        "succeeded": len(done),
        "passed_validation": sum(r["passed"] for r in done),
        "errors": [r for r in results if "error" in r],
        "totals": totals,
        "per_company_avg": {
            "seconds": round(sum(r["wall_seconds"] for r in done) / n, 2),
            "total_tokens": round(totals["total_tokens"] / n),
        },
        "by_company": [
            {
                "company_id": r["company_id"],
                "passed": r["passed"],
                "wall_seconds": r["wall_seconds"],
                **r["totals"],
            }
            for r in done
        ],
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/sec10k"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--companies", nargs="*", help="company ids, e.g. syn0001")
    parser.add_argument("--limit", type=int, help="first N specs only")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reasoning-effort", help="passed through if set")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="redo finished")
    parser.add_argument("--dry-run", action="store_true", help="prompts only")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request

    client = None
    if not args.dry_run or args.list_models:
        client = UsageClient(
            args.model,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
    if args.list_models and client is not None:
        print("\n".join(client.list_models()))
        return

    synthetic = args.data / "synthetic"
    specs = [json.loads(p.read_text()) for p in sorted(synthetic.glob("specs/*.json"))]
    if args.companies:
        specs = [s for s in specs if s["company_id"] in args.companies]
    out_root = synthetic / "filings"
    if not args.overwrite and not args.dry_run:
        specs = [
            s for s in specs if not (out_root / s["company_id"] / "filing.txt").exists()
        ]
    specs = specs[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.perf_counter()
    results = run(
        specs,
        Corpus.load(args.data),
        client,
        out_root=out_root,
        workers=args.workers,
        length_scale=args.length_scale,
    )
    config = {
        "model": args.model,
        "dry_run": args.dry_run,
        "temperature": args.temperature,
        "length_scale": args.length_scale,
        "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    summary = summarize(results, run_id=run_id, config=config)
    runs = synthetic / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.json").write_text(json.dumps(summary, indent=2))
    line = {k: v for k, v in summary.items() if k not in ("by_company", "errors")}
    with (synthetic / "runs.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")
    t = summary["totals"]
    logger.info(
        "Run %s: %d/%d companies passed validation, %d calls, %d tokens "
        "(%d prompt, %d completion), %.0fs",
        run_id,
        summary["passed_validation"],
        len(specs),
        t["calls"],
        t["total_tokens"],
        t["prompt_tokens"],
        t["completion_tokens"],
        config["wall_seconds"],
    )


if __name__ == "__main__":
    main()
