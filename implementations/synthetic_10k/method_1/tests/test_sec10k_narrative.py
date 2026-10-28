"""Tests for narrative generation, its validators and its run metrics."""

import json
from pathlib import Path
from typing import Any

from method_1.narrative import (
    Completion,
    Corpus,
    clean_output,
    run,
    summarize,
)
from method_1.render import derived_metrics, item_8, metric_label, money
from method_1.specs import generate
from method_1.validate import (
    HEADINGS,
    check_identifiers,
    check_numbers,
    check_overlap,
    check_real_names,
    real_name_patterns,
    validate_section,
)

from .test_sec10k_specs import CALIBRATION


SPEC = generate(CALIBRATION, n=1, seed=5, real_names=set())[0]
INCOME = SPEC["statements"]["income_statement"]["2024"]


def _section(item: str, words: int, extra: str = "") -> str:
    facts = ""
    if item in ("1", "7"):
        revenue = f"Revenue was ${money(INCOME['revenue'])} million. "
        gross = derived_metrics(SPEC)["gross_margin_pct_current"]
        facts = f"{revenue}Gross margin was {gross}%. "
    filler = " ".join(f"word{i}" for i in range(words))
    return f"{HEADINGS[item]}\n\n{SPEC['name']} {facts}{extra}{filler}"


def test_item_8_renders_the_statement_numbers() -> None:
    """Item 8 quotes the statements exactly, with negatives in parentheses."""
    text = item_8(SPEC)
    assert text.startswith("Item 8.")
    assert money(INCOME["revenue"]) in text
    assert money(-1234) == "(1,234)"


def test_numbers_must_come_from_the_statements() -> None:
    """Statement amounts and derived percentages pass; invented ones do not."""
    billions = f"${INCOME['revenue'] / 1000:.1f} billion"
    good = _section("7", 10, f"Sales reached {billions}. ")
    assert check_numbers(good, SPEC, strict=True) == []
    bad = good + " Backlog was $987,654 million, up 77.7%."
    issues = check_numbers(bad, SPEC, strict=True)
    assert "$987,654 million" in issues[0]
    assert "77.7%" in issues[1]
    assert len(check_numbers(bad, SPEC, strict=False)) == 1


def test_real_names_and_copying_are_caught() -> None:
    """Real filer names and text copied from the exemplar are reported."""
    patterns = real_name_patterns(["ADOBE INC.", "THE HOME DEPOT, INC.", "3M CO"])
    assert check_real_names("We compete with Home Depot.", patterns)
    assert check_real_names("We compete with large retailers.", patterns) == []
    exemplar = " ".join(f"w{i}" for i in range(200))
    assert check_overlap(exemplar, exemplar)
    assert check_overlap(_section("1", 200), exemplar) == []


def test_validate_section_checks_length_and_heading() -> None:
    """A short section without a heading fails both checks."""
    issues = validate_section("Too short.", item="1A", spec=SPEC, target_words=500)
    assert any("heading" in i for i in issues)
    assert any("at least" in i for i in issues)
    assert (
        validate_section(_section("1A", 500), item="1A", spec=SPEC, target_words=500)
        == []
    )


def test_clean_output_strips_markdown_and_adds_heading() -> None:
    """Markdown headings and bold are removed and a missing heading is added."""
    text = clean_output("## Overview\n**Bold** text\n* point", "1C")
    assert text.startswith("Item 1C. Cybersecurity\n")
    assert "#" not in text and "**" not in text and "- point" in text


class FakeClient:
    """Return a bad draft for the first call of each Item, then good ones."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, system: str, prompt: str, *, max_tokens: int) -> Completion:
        """Answer from the heading and word target found in the prompt."""
        item = next(i for i, h in HEADINGS.items() if f'line "{h}"' in prompt)
        target = int(
            prompt.rsplit("about ", maxsplit=1)[-1].split(" words", maxsplit=1)[0]
        )
        first = not any(f'line "{HEADINGS[item]}"' in p for p in self.calls)
        self.calls.append(prompt)
        extra = "Backlog was $987,654 million. " if first and item == "7" else ""
        return Completion(
            text=_section(item, target, extra),
            model="fake",
            seconds=0.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


def _data(tmp_path: Path) -> Path:
    calibration: dict[str, Any] = {
        **CALIBRATION,
        "section_words": {
            i: {"p10": 300, "p25": 300, "p50": 300, "p75": 300, "p90": 300}
            for i in HEADINGS
        },
    }
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    return tmp_path


def test_run_regenerates_failures_and_records_usage(tmp_path: Path) -> None:
    """A failed Item is retried once and every call's time and tokens add up."""
    data = _data(tmp_path)
    out = data / "synthetic" / "filings"
    client = FakeClient()
    results = run([SPEC], Corpus.load(data), client, out_root=out)
    metrics = results[0]
    assert metrics["passed"]
    assert metrics["sections"]["7"]["attempts"] == 2
    assert metrics["sections"]["1"]["attempts"] == 1
    assert len(client.calls) == 8  # seven Items, one retried
    assert metrics["totals"]["total_tokens"] == 8 * 150
    assert metrics["totals"]["seconds"] == 4.0
    retry = [p for p in client.calls if "previous draft" in p]
    assert len(retry) == 1 and "$987,654 million" in retry[0]  # carries feedback
    folder = out / SPEC["company_id"]
    filing = (folder / "filing.txt").read_text()
    order = ["Item 1C.", "Item 2.", "Item 7.", "Item 7A.", "Item 8.", "Item 9A."]
    positions = [filing.index(h) for h in order]
    assert positions == sorted(positions)
    page = (folder / "filing.html").read_text()
    assert "Consolidated Balance Sheets" in page and "SYNTHETIC DOCUMENT" in page
    saved = json.loads((out / SPEC["company_id"] / "metrics.json").read_text())
    assert saved["calls"][0]["prompt_tokens"] == 100

    summary = summarize(results, run_id="r1", config={"model": "fake"})
    assert summary["passed_validation"] == 1
    assert summary["totals"]["calls"] == 8
    assert summary["by_company"][0]["total_tokens"] == 1200


def test_items_limits_the_filing_to_chosen_items(tmp_path: Path) -> None:
    """``only`` generates just those Items and the filing holds nothing else."""
    data = _data(tmp_path)
    out = data / "synthetic" / "filings"
    client = FakeClient()
    only = ["1", "1A", "1C", "7", "8"]
    metrics = run([SPEC], Corpus.load(data), client, out_root=out, only=only)[0]
    assert sorted(metrics["sections"]) == ["1", "1A", "1C", "7"]
    filing = (out / SPEC["company_id"] / "filing.txt").read_text()
    for heading in ("Item 1.", "Item 1A.", "Item 1C.", "Item 7.", "Item 8."):
        assert heading in filing
    for heading in ("Item 1B.", "Item 2.", "Item 7A.", "Item 9A.", "Item 15."):
        assert heading not in filing


def test_dry_run_writes_prompts_only(tmp_path: Path) -> None:
    """Without a client, prompts and Item 8 are written but no filing."""
    data = _data(tmp_path)
    out = data / "synthetic" / "filings"
    metrics = run([SPEC], Corpus.load(data), None, out_root=out)[0]
    folder = out / SPEC["company_id"]
    assert not (folder / "filing.txt").exists()
    assert (folder / "item_8.txt").exists()
    prompt = (folder / "prompts" / "item_7.txt").read_text()
    assert "[Item 1 as generated]" in prompt and SPEC["name"] in prompt
    assert metrics["sections"]["7"]["target_words"] == 300
    assert metrics["totals"]["calls"] == 0


def test_identifiers_are_flagged_and_labels_are_readable():
    """Metric keys from the prompt must not leak into the prose."""
    assert check_identifiers("Our gross_margin_pct_current was 37.9%.")
    assert not check_identifiers("Our gross margin was 37.9%.")
    assert metric_label("gross_margin_pct_current", 2024) == "Gross margin, fiscal 2024"
    assert (
        metric_label("selling_general_and_administrative_pct_of_revenue_prior", 2024)
        == "Selling, general and administrative as a share of revenue, fiscal 2023"
    )
    assert metric_label("revenue_growth_pct", 2024) == (
        "Revenue growth, fiscal 2024 versus 2023"
    )


class _FakeStream:
    """Stands in for ``client.messages.stream(...)``: a thinking block, then text."""

    def __init__(self, stop_reason: str = "end_turn", **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stop_reason = stop_reason

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        from types import SimpleNamespace  # noqa: PLC0415

        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Item 1. Business\n\nHello."),
            ],
            model="claude-opus-5-5",
            stop_reason=self.stop_reason,
            stop_details=None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=40),
        )


def test_anthropic_client_keeps_text_and_sets_effort(monkeypatch: Any) -> None:
    """Claude calls stream with an effort, skip thinking blocks and blank refusals."""
    from method_1 import narrative  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(
        narrative.make_client("claude-opus-5-5"), narrative.AnthropicUsageClient
    )
    assert isinstance(narrative.make_client("gpt-4o"), narrative.UsageClient)

    client = narrative.AnthropicUsageClient("claude-opus-5-5", effort="xhigh")
    calls: list[dict[str, Any]] = []

    def stream(**kwargs: Any) -> _FakeStream:
        calls.append(kwargs)
        return _FakeStream(**kwargs)

    monkeypatch.setattr(client.client.messages, "stream", stream)
    result = client.chat("system", "prompt", max_tokens=1000)
    assert result.text == "Item 1. Business\n\nHello."
    assert result.total_tokens == 140
    assert calls[0]["output_config"] == {"effort": "xhigh"}
    assert "temperature" not in calls[0]

    monkeypatch.setattr(
        client.client.messages, "stream", lambda **kw: _FakeStream("refusal", **kw)
    )
    assert client.chat("system", "prompt", max_tokens=1000).text == ""
