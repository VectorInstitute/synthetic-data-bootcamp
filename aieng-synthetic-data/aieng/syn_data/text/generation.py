"""Synthetic Q&A generation strategies using a teacher LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from aieng.syn_data.text.clients import LLMClient, extract_json_text
from aieng.syn_data.text.config import FAILURE_MODE_GUIDANCE
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
        "Generate one challenging question-answer pair grounded in the passage.\n"
        "Return JSON with keys: question, gold_answer\n"
        f"{failure_hint}"
        f"{extra_instruction}\n"
        f"Passage:\n{paragraph.text}"
    )


def zero_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    *,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate a single Q&A pair without exemplars."""
    prompt = _base_generation_prompt(paragraph, failure_mode=failure_mode)
    payload = _parse_generation_response(client, prompt)
    logger.info(f"Payload: {payload}")
    return _to_qa_sample(paragraph, payload, strategy=GenerationStrategy.ZERO_SHOT, failure_mode=failure_mode)


def one_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    example: QASample,
    *,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate a Q&A pair using one in-context example."""
    example_block = json.dumps(example.to_dict(), ensure_ascii=False, indent=2)
    extra = f"Example:\n{example_block}\n"
    prompt = _base_generation_prompt(
        paragraph,
        failure_mode=failure_mode,
        extra_instruction=extra,
    )
    payload = _parse_generation_response(client, prompt)
    return _to_qa_sample(paragraph, payload, strategy=GenerationStrategy.ONE_SHOT, failure_mode=failure_mode)


def few_shot_generate(
    client: LLMClient,
    paragraph: Paragraph,
    examples: list[QASample],
    *,
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
        failure_mode=failure_mode,
        extra_instruction=extra,
    )
    payload = _parse_generation_response(client, prompt)
    return _to_qa_sample(paragraph, payload, strategy=GenerationStrategy.FEW_SHOT, failure_mode=failure_mode)


def topic_controlled_generate(
    client: LLMClient,
    paragraph: Paragraph,
    *,
    topics: list[str] | None = None,
    failure_mode: FailureMode | None = None,
) -> QASample:
    """Generate topics for the passage, pick one, then generate a Q&A pair.
    Run a two-step generation — first extract policy topics from the passage, then generate Q&A focused on one chosen topic. 
    Used in the test-set pipeline (generate_test_qa_batch) to get targeted, diverse questions per paragraph instead of one generic Q&A.
    """
    topic_list = topics or _generate_topics(client, paragraph)
    topic = topic_list[0] if topic_list else "general policy interpretation"
    extra = f"Focus topic: {topic}\n"
    prompt = _base_generation_prompt(
        paragraph,
        failure_mode=failure_mode,
        extra_instruction=extra,
    )
    payload = _parse_generation_response(client, prompt)
    sample = _to_qa_sample(paragraph, payload, strategy=GenerationStrategy.TOPIC_CONTROLLED, failure_mode=failure_mode)
    sample.metadata["topic"] = topic
    sample.metadata["candidate_topics"] = topic_list
    return sample


def _generate_topics(client: LLMClient, paragraph: Paragraph, *, n_topics: int = 5) -> list[str]:
    prompt = (
        f"List {n_topics} concise policy topics present in the passage.\n"
        'Return JSON: {"topics": ["..."]}\n'
        f"Passage:\n{paragraph.text}"
    )
    if hasattr(client, "complete_json"):
        payload = client.complete_json(prompt, system="You extract policy topics.")
        return list(payload.get("topics", []))
    raw = client.complete(prompt, system="You extract policy topics.")
    return list(json.loads(raw).get("topics", []))


def _parse_generation_response(client: LLMClient, prompt: str) -> dict[str, Any]:
    if hasattr(client, "complete_json"):
        return client.complete_json(
            prompt,
            system="You generate grounded policy Q&A.",
            max_tokens=2048,
        )
    raw = client.complete(prompt, system="You generate grounded policy Q&A.", max_tokens=2048)
    return json.loads(extract_json_text(raw))


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
