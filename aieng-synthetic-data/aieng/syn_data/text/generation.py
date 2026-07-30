"""Synthetic Q&A generation strategies using a teacher LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast

from aieng.syn_data.text.clients import LLMClient, extract_json_text
from aieng.syn_data.text.config import DEFAULT_DOMAIN, FAILURE_MODE_GUIDANCE
from aieng.syn_data.text.schemas import (
    FailureMode,
    GenerationStrategy,
    Paragraph,
    QASample,
)


logger = logging.getLogger(__name__)


def _base_generation_prompt(
    paragraph: Paragraph,
    *,
    domain: str = DEFAULT_DOMAIN,
    failure_mode: FailureMode | None = None,
    extra_instruction: str = "",
) -> str:
    failure_hint = ""
    if failure_mode is not None:
        failure_hint = (
            f"\nTarget failure mode: {failure_mode.value}\n"
            f"Guidance: {FAILURE_MODE_GUIDANCE[failure_mode]}\n"
        )
    return (
        f"Generate one challenging question-answer pair about this {domain} passage.\n"
        "The answer must be grounded in the passage only.\n"
        "Return JSON with keys: question, gold_answer\n"
        f"{failure_hint}"
        f"{extra_instruction}\n"
        f"Passage:\n{paragraph.text}"
    )


def zero_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    *,
    domain: str = DEFAULT_DOMAIN,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate a single Q&A pair without exemplars."""
    prompt = _base_generation_prompt(
        paragraph, domain=domain, failure_mode=failure_mode
    )
    payload = _parse_generation_response(client, prompt, domain=domain)
    logger.info("Payload: %s", payload)
    return _to_qa_sample(
        paragraph,
        payload,
        strategy=GenerationStrategy.ZERO_SHOT,
        failure_mode=failure_mode,
    )


def one_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    example: QASample,
    *,
    domain: str = DEFAULT_DOMAIN,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate a Q&A pair using one in-context example."""
    example_block = json.dumps(example.to_dict(), ensure_ascii=False, indent=2)
    extra = f"Example:\n{example_block}\n"
    prompt = _base_generation_prompt(
        paragraph,
        domain=domain,
        failure_mode=failure_mode,
        extra_instruction=extra,
    )
    payload = _parse_generation_response(client, prompt, domain=domain)
    return _to_qa_sample(
        paragraph,
        payload,
        strategy=GenerationStrategy.ONE_SHOT,
        failure_mode=failure_mode,
    )


def few_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    examples: list[QASample],
    *,
    domain: str = DEFAULT_DOMAIN,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate a Q&A pair using multiple in-context examples."""
    example_block = json.dumps(
        [example.to_dict() for example in examples],
        ensure_ascii=False,
        indent=2,
    )
    extra = f"Examples:\n{example_block}\n"
    prompt = _base_generation_prompt(
        paragraph,
        domain=domain,
        failure_mode=failure_mode,
        extra_instruction=extra,
    )
    payload = _parse_generation_response(client, prompt, domain=domain)
    return _to_qa_sample(
        paragraph,
        payload,
        strategy=GenerationStrategy.FEW_SHOT,
        failure_mode=failure_mode,
    )


def topic_controlled_generate(
    client: LLMClient,
    paragraph: Paragraph,
    *,
    topics: list[str] | None = None,
    domain: str = DEFAULT_DOMAIN,
    failure_mode: FailureMode | None = None,
    max_topics: int | None = None,
) -> list[QASample]:
    """Generate one Q&A pair per topic extracted from (or supplied for) a paragraph.

    Topics are **paragraph-scoped** on purpose: shorter context keeps questions
    precise and reduces the risk of hallucinated themes that can appear when
    topics are mined from a whole document.

    Parameters
    ----------
    client:
        Teacher LLM client.
    paragraph:
        Source paragraph for topic extraction and grounding.
    topics:
        Optional precomputed topics. If omitted, topics are extracted from the
        paragraph only.
    domain:
        Domain label injected into prompts (from ``DEFAULT_DOMAIN`` / config).
    failure_mode:
        Optional small-model failure mode to condition each generation.
    max_topics:
        Optional cap on how many topics (and thus Q&As) to generate.

    Returns
    -------
    list[QASample]
        One sample per topic, each tagged with ``metadata["topic"]``.
    """
    topic_list = list(topics) if topics is not None else extract_topics(
        client, paragraph, domain=domain
    )
    if not topic_list:
        topic_list = [f"general {domain} interpretation"]
    if max_topics is not None:
        topic_list = topic_list[: max(0, max_topics)]

    samples: list[QASample] = []
    for topic in topic_list:
        extra = (
            f"Focus topic: {topic}\n"
            f"The question must be specifically about this topic and answerable "
            f"from the passage alone.\n"
        )
        prompt = _base_generation_prompt(
            paragraph,
            domain=domain,
            failure_mode=failure_mode,
            extra_instruction=extra,
        )
        payload = _parse_generation_response(client, prompt, domain=domain)
        sample = _to_qa_sample(
            paragraph,
            payload,
            strategy=GenerationStrategy.TOPIC_CONTROLLED,
            failure_mode=failure_mode,
        )
        sample.metadata["topic"] = topic
        sample.metadata["candidate_topics"] = list(topic_list)
        samples.append(sample)
    return samples


def extract_topics(
    client: LLMClient,
    paragraph: Paragraph,
    *,
    domain: str = DEFAULT_DOMAIN,
    n_topics: int = 5,
) -> list[str]:
    """Extract concise topics present in a single paragraph (not the full document)."""
    prompt = (
        f"List {n_topics} concise {domain} topics present in the passage.\n"
        "Only include topics that are explicitly supported by the passage.\n"
        'Return JSON: {"topics": ["..."]}\n'
        f"Passage:\n{paragraph.text}"
    )
    system = f"You extract {domain} topics from a short passage."
    if hasattr(client, "complete_json"):
        payload = client.complete_json(prompt, system=system)
        return list(payload.get("topics", []))
    raw = client.complete(prompt, system=system)
    return list(json.loads(extract_json_text(raw)).get("topics", []))


def _parse_generation_response(
    client: LLMClient,
    prompt: str,
    *,
    domain: str = DEFAULT_DOMAIN,
) -> dict[str, Any]:
    system = f"You generate grounded {domain} Q&A from source text."
    if hasattr(client, "complete_json"):
        return cast(
            dict[str, Any],
            client.complete_json(
                prompt,
                system=system,
                max_tokens=2048,
            ),
        )
    raw = client.complete(prompt, system=system, max_tokens=2048)
    return cast(dict[str, Any], json.loads(extract_json_text(raw)))


def _to_qa_sample(
    paragraph: Paragraph,
    payload: dict[str, Any],
    *,
    strategy: GenerationStrategy,
    failure_mode: FailureMode | None = None,
) -> QASample:

    return QASample(
        id=str(uuid.uuid4()),
        question=payload["question"],
        gold_answer=payload["gold_answer"],
        doc_id=paragraph.doc_id,
        para_id=paragraph.para_id,
        context=paragraph.text,
        failure_mode=failure_mode,
        role=paragraph.role,
        instruction=payload.get("instruction", ""),
        split=paragraph.split,
        metadata={"generation_strategy": strategy.value},
    )
