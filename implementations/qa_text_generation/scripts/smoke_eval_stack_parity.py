#!/usr/bin/env python3
"""Dump Ollama vs HF eval prompt/sampling details for one sample.

Prints:
  1. The exact OpenAI-compatible JSON body ``OpenAICompatibleClient`` posts
     (whether ``top_p`` / ``top_k`` / ``repeat_penalty`` are present).
  2. The HF ``apply_chat_template`` string vs the Ollama TEMPLATE rendering,
     with a character-by-character diff.
  3. ``ollama show <tag> --modelfile`` PARAMETER and TEMPLATE blocks.

Usage (from repo root, with the package installed and Ollama up)::

    python implementations/qa_text_generation/scripts/smoke_eval_stack_parity.py
    python implementations/qa_text_generation/scripts/smoke_eval_stack_parity.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import requests

# Allow running without installing if PYTHONPATH is unset.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PKG_ROOT = _REPO_ROOT / "aieng-synthetic-data"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from aieng.syn_data.text.clients import OpenAICompatibleClient  # noqa: E402
from aieng.syn_data.text.config import TEST_SET_PATH  # noqa: E402
from aieng.syn_data.text.evaluation import (  # noqa: E402
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_SYSTEM,
    build_eval_prompt,
)
from aieng.syn_data.text.io import load_typed_jsonl  # noqa: E402
from aieng.syn_data.text.paths import load_implementation_dotenv  # noqa: E402
from aieng.syn_data.text.schemas import QASample  # noqa: E402
from aieng.syn_data.text.small_model import create_small_model_client  # noqa: E402

SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "repetition_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "max_tokens",
    "num_predict",
    "options",
)

FALLBACK_SHOW_TAGS = ("qwen2.5:0.5b", "qwen2.5:0.5b-instruct")


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _demo_sample() -> QASample:
    return QASample(
        id="smoke-001",
        question="What is the APR for purchases?",
        gold_answer="The purchase APR is 18.99%.",
        doc_id="cfpb_credit_card_agreement",
        para_id="cfpb_credit_card_agreement::p0001",
        context="The Annual Percentage Rate (APR) for purchases is 18.99%.",
        instruction="Answer only from the context. Cite the rate exactly.",
    )


def load_one_sample() -> QASample:
    if TEST_SET_PATH.exists():
        samples = load_typed_jsonl(TEST_SET_PATH, QASample.from_dict)
        if samples:
            return samples[0]
        print(f"(test set empty at {TEST_SET_PATH}; using built-in sample)")
    else:
        print(f"(no test set at {TEST_SET_PATH}; using built-in sample)")
    return _demo_sample()


def build_openai_eval_body(
    client: OpenAICompatibleClient,
    prompt: str,
    *,
    system: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Return (url, json body) that ``complete()`` would POST. No network."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["json"] = kwargs.get("json")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": "(dry-run)"}}]}

        return _Response()

    with patch.object(requests, "post", side_effect=fake_post):
        client.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
    if "json" not in captured:
        raise RuntimeError("Failed to capture OpenAI-compatible request body.")
    return str(captured["url"]), dict(captured["json"])


def summarize_sampling_fields(body: dict[str, Any]) -> None:
    print("\nSampling-related keys on the wire:")
    present = []
    for key in SAMPLING_KEYS:
        if key in body:
            present.append(key)
            print(f"  EXPLICIT  {key}={body[key]!r}")
    nested = body.get("options")
    if isinstance(nested, dict):
        for key, value in nested.items():
            print(f"  EXPLICIT  options.{key}={value!r}")
    missing = [
        key
        for key in ("top_p", "top_k", "repeat_penalty", "repetition_penalty")
        if key not in body
        and not (isinstance(nested, dict) and key in nested)
    ]
    for key in missing:
        print(f"  ABSENT    {key}  → Ollama will use Modelfile PARAMETER or server default")
    if not present and "temperature" not in body:
        print("  (no sampling fields at all)")


def fetch_api_show(host: str, model: str) -> dict[str, Any] | None:
    url = f"http://{host}/api/show"
    try:
        response = requests.post(url, json={"name": model}, timeout=30)
        response.raise_for_status()
        return dict(response.json())
    except requests.RequestException as exc:
        print(f"GET {url} failed: {exc}")
        return None


def ollama_show_modelfile(tag: str) -> str | None:
    try:
        result = subprocess.run(
            ["ollama", "show", tag, "--modelfile"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"ollama show {tag!r} failed: {exc}")
        return None
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        print(f"ollama show {tag!r} exit {result.returncode}: {err}")
        return None
    return result.stdout


def split_modelfile_sections(modelfile: str) -> tuple[str, str, str]:
    """Return (PARAMETER lines, TEMPLATE block, SYSTEM line) from a Modelfile."""
    parameter_lines: list[str] = []
    system_line = "(no SYSTEM line)"
    for line in modelfile.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PARAMETER "):
            parameter_lines.append(line)
        if stripped.upper().startswith("SYSTEM "):
            system_line = line
    template = ""
    match = re.search(r'TEMPLATE\s+"""(.*?)"""', modelfile, re.DOTALL)
    if match:
        template = match.group(1)
    return (
        "\n".join(parameter_lines) if parameter_lines else "(no PARAMETER lines)",
        template,
        system_line,
    )


def render_ollama_system_prompt_template(
    template: str,
    *,
    system: str,
    prompt: str,
    response: str = "",
) -> str:
    """Expand the simple ``.System`` / ``.Prompt`` / ``.Response`` Ollama form.

    Message-range templates (``range .Messages``) are not executed; callers
    should fall back to ChatML when those constructs are present.
    """
    values = {"System": system, "Prompt": prompt, "Response": response}

    def if_block(source: str, var: str) -> str:
        pattern = re.compile(
            rf"{{{{-?\s*if\s+\.{var}\s*-?}}}}(.*?){{{{-?\s*end\s*-?}}}}",
            re.DOTALL,
        )

        def repl(match: re.Match[str]) -> str:
            return match.group(1) if values.get(var) else ""

        return pattern.sub(repl, source)

    rendered = template
    for var in ("System", "Prompt", "Response"):
        rendered = if_block(rendered, var)
        rendered = re.sub(
            rf"{{{{-?\s*\.{var}\s*-?}}}}",
            lambda _m, v=var: values[v],
            rendered,
        )
    rendered = rendered.replace("{{-", "").replace("-}}", "").replace("{{", "").replace("}}", "")
    return rendered


def fallback_qwen_chatml(system: str, prompt: str) -> str:
    """Qwen Instruct ChatML with a generation prompt (assistant header open)."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def render_hf_prompt(system: str, prompt: str, base_model: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return f"{system}\n\n{prompt}\nAnswer:"


def char_by_char_diff(left: str, right: str, *, left_name: str, right_name: str) -> None:
    print(f"\n{left_name} length: {len(left)}")
    print(f"{right_name} length: {len(right)}")
    if left == right:
        print("Strings are identical.")
        return
    limit = max(len(left), len(right))
    first = None
    mismatches = 0
    for index in range(limit):
        ca = left[index] if index < len(left) else None
        cb = right[index] if index < len(right) else None
        if ca != cb:
            mismatches += 1
            if first is None:
                first = index
    print(f"Mismatch count: {mismatches} character position(s)")
    assert first is not None
    window = 40
    start = max(0, first - window)
    end_l = min(len(left), first + window)
    end_r = min(len(right), first + window)
    print(f"First mismatch at index {first}:")
    print(f"  {left_name}[{first}]={left[first]!r}" if first < len(left) else f"  {left_name}: <missing>")
    print(f"  {right_name}[{first}]={right[first]!r}" if first < len(right) else f"  {right_name}: <missing>")
    print(f"  {left_name} context: {left[start:end_l]!r}")
    print(f"  {right_name} context: {right[start:end_r]!r}")
    print("\nAll mismatched positions (index, left, right):")
    shown = 0
    for index in range(limit):
        ca = left[index] if index < len(left) else "<EOF>"
        cb = right[index] if index < len(right) else "<EOF>"
        if ca != cb:
            print(f"  {index:4d}  {ca!r:8}  {cb!r}")
            shown += 1
            if shown >= 80:
                print(f"  ... truncated ({mismatches - shown} more)")
                break


def hf_generate_kwargs_at_eval() -> dict[str, Any]:
    """Document what ``Hf4BitInferenceClient.complete`` passes at temperature=0."""
    temperature = 0.0
    return {
        "max_new_tokens": DEFAULT_EVAL_MAX_TOKENS,
        "do_sample": temperature > 0.0,
        "temperature": max(temperature, 1e-5),
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "note": (
            "Eval calls complete(..., temperature=0.0). With do_sample=False, "
            "transformers ignores top_p/top_k but still applies repetition_penalty."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="POST the captured body to Ollama (still prints the payload first).",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip loading the Hugging Face tokenizer.",
    )
    args = parser.parse_args()

    # Prefer the RI .env so a leftover shell OLLAMA_MODEL (e.g. 3b) does not hide
    # the notebook 01 tag.
    load_implementation_dotenv(override=True)
    sample = load_one_sample()
    user_prompt = build_eval_prompt(sample)
    system = DEFAULT_EVAL_SYSTEM
    ollama_model = os.getenv("OLLAMA_MODEL") or os.getenv("SMALL_MODEL_NAME") or "qwen2.5:0.5b-instruct"
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    hf_model = os.getenv("SFT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    _banner("1. Ollama request payload (OpenAI-compatible /v1/chat/completions)")
    print(f"sample id: {sample.id}")
    print(f"SMALL_MODEL_NAME={os.getenv('SMALL_MODEL_NAME')!r}")
    print(f"OLLAMA_MODEL={os.getenv('OLLAMA_MODEL')!r}")
    print(f"SMALL_MODEL_BASE_URL={os.getenv('SMALL_MODEL_BASE_URL')!r}")
    print(f"Hf4BitInferenceClient.generate kwargs at eval temperature=0:\n{json.dumps(hf_generate_kwargs_at_eval(), indent=2)}")

    client = create_small_model_client()
    url, body = build_openai_eval_body(
        client,
        user_prompt,
        system=system,
        temperature=0.0,
        max_tokens=DEFAULT_EVAL_MAX_TOKENS,
    )
    print(f"\nPOST {url}")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    summarize_sampling_fields(body)
    print(
        "\nConclusion: OpenAICompatibleClient only sets model/messages/"
        "temperature/max_tokens. It does not send top_p, top_k, or repeat_penalty, "
        "so those are NOT explicitly zeroed; Ollama fills them from the Modelfile "
        "or its engine defaults (commonly top_k=40, top_p=0.9, repeat_penalty=1.1)."
    )

    if args.live:
        print("\n--live: sending the captured body...")
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {client.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        print(f"HTTP {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False)[:4000])
        except json.JSONDecodeError:
            print(response.text[:2000])

    native_chat = {
        "model": ollama_model,
        "messages": body["messages"],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": DEFAULT_EVAL_MAX_TOKENS,
        },
    }
    _banner("1b. Equivalent native /api/chat body (not used by the client)")
    print(
        "The notebooks talk to Ollama via /v1, not /api/chat. "
        "If you did use native chat with only temperature/num_predict set, "
        "the options blob would look like this — still no top_p/top_k/repeat_penalty:"
    )
    print(json.dumps(native_chat, indent=2, ensure_ascii=False))

    _banner("2. Rendered prompts (one sample) + character diff")
    print("--- user prompt (build_eval_prompt) ---")
    print(user_prompt)
    print("--- system ---")
    print(system)

    hf_rendered: str | None = None
    if not args.skip_hf:
        print(f"\nLoading HF tokenizer {hf_model!r} ...")
        try:
            hf_rendered = render_hf_prompt(system, user_prompt, hf_model)
        except Exception as exc:  # noqa: BLE001 — diagnostic script
            print(f"HF tokenizer failed: {exc}")
    else:
        print("Skipping HF tokenizer (--skip-hf).")

    show_payload = fetch_api_show(ollama_host, ollama_model)
    ollama_template = ""
    if show_payload:
        ollama_template = str(show_payload.get("template") or "")
        print("\n--- /api/show parameters ---")
        print(json.dumps(show_payload.get("parameters"), indent=2, ensure_ascii=False))
        print("\n--- /api/show system (Modelfile SYSTEM; chat .System overrides this) ---")
        print(show_payload.get("system"))
        print("\n--- /api/show template ---")
        print(ollama_template)

    uses_messages = ".Messages" in ollama_template or "range" in ollama_template
    if ollama_template and not uses_messages:
        ollama_rendered = render_ollama_system_prompt_template(
            ollama_template, system=system, prompt=user_prompt
        )
        ollama_label = "Ollama TEMPLATE (.System/.Prompt)"
    else:
        ollama_rendered = fallback_qwen_chatml(system, user_prompt)
        ollama_label = "Ollama Messages-path ChatML (no tools; last turn user + generation prompt)"
        if uses_messages:
            print(
                "\nTEMPLATE uses range/.Messages. Not executing Go templates; "
                "expanding the no-tools path: .System, then user content, then "
                "<|im_start|>assistant (same shape as the TEMPLATE else-branch)."
            )

    print(f"\n--- {ollama_label} ---")
    print(repr(ollama_rendered))
    print(ollama_rendered)
    if hf_rendered is not None:
        print("\n--- HF apply_chat_template (tokenize=False, add_generation_prompt=True) ---")
        print(repr(hf_rendered))
        print(hf_rendered)
        _banner("2b. Character-by-character diff (Ollama vs HF)")
        char_by_char_diff(
            ollama_rendered,
            hf_rendered,
            left_name="ollama",
            right_name="hf",
        )
        print(
            "\nNote: Hf4BitInferenceClient.complete uses this same apply_chat_template "
            "path; OpenAICompatibleClient sends messages and lets Ollama apply TEMPLATE."
        )
    else:
        print("\n(No HF string to diff.)")

    _banner("3. ollama show --modelfile (PARAMETER + TEMPLATE)")
    tags = []
    for tag in (ollama_model, *FALLBACK_SHOW_TAGS):
        if tag not in tags:
            tags.append(tag)
    for tag in tags:
        print(f"\n----- ollama show {tag} --modelfile -----")
        raw = ollama_show_modelfile(tag)
        if raw is None:
            continue
        params, template, system_line = split_modelfile_sections(raw)
        print("\n[PARAMETER]")
        print(params)
        print("\n[SYSTEM]")
        print(system_line)
        print("\n[TEMPLATE]")
        print(template if template else "(could not parse TEMPLATE)")
        license_at = raw.find("\nLICENSE ")
        trimmed = raw[:license_at] if license_at != -1 else raw
        print("\n[modelfile without LICENSE body]")
        print(trimmed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
