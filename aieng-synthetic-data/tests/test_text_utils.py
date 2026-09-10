"""Unit tests for text synthetic data utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aieng.syn_data.text.clients import extract_json_text
from aieng.syn_data.text.documents import (
    chunk_text_into_paragraphs,
    sample_test_paragraphs,
)
from aieng.syn_data.text.generation import (
    _base_generation_prompt,
    topic_controlled_generate,
)
from aieng.syn_data.text.io import read_jsonl, write_jsonl
from aieng.syn_data.text.pipeline import (
    build_paragraph_splits,
    effective_synthetic_target,
    effective_test_holdout,
)
from aieng.syn_data.text.prompts import (
    instruction_backtranslation_prompt,
    synthetic_qa_quality_prompt,
)
from aieng.syn_data.text.quality import (
    apply_heuristic_filters,
    detect_answer_leakage,
    summarize_heuristic_rejections,
)
from aieng.syn_data.text.rag import (
    generate_instruction_back_translation_sample,
    grounding_overlap_score,
    retrieve_paragraphs,
)
from aieng.syn_data.text.schemas import (
    DocumentRole,
    FailureMode,
    Paragraph,
    ParagraphSplit,
    QASample,
)
from aieng.syn_data.text.sft import qa_samples_to_messages


def _paragraph(doc_id: str, index: int, text: str) -> Paragraph:
    return Paragraph(
        doc_id=doc_id,
        para_id=f"{doc_id}::p{index:04d}",
        text=text,
        role=DocumentRole.POLICY_DENSE,
        split=ParagraphSplit.TRAIN,
        index=index,
    )


def test_extract_json_text_from_markdown_fence() -> None:
    raw = (
        "4. **Drafting the JSON:**\n"
        "```json\n"
        '{"question": "What is APR?", "gold_answer": "Annual Percentage Rate."}\n'
        "```"
    )
    assert json.loads(extract_json_text(raw)) == {
        "question": "What is APR?",
        "gold_answer": "Annual Percentage Rate.",
    }


def test_extract_json_text_from_embedded_object() -> None:
    raw = 'Here is the result: {"topics": ["billing", "fees"]} thanks.'
    assert json.loads(extract_json_text(raw)) == {"topics": ["billing", "fees"]}


def test_chunk_text_into_paragraphs_merges_short_sections() -> None:
    text = "Short.\n\nAlso brief."
    chunks = chunk_text_into_paragraphs(text, min_chars=30)
    assert len(chunks) == 1
    assert "Short." in chunks[0]
    assert "Also brief." in chunks[0]


def test_sample_test_paragraphs_holdout(tmp_path: Path) -> None:
    paragraphs = [
        _paragraph("doc", index, f"Paragraph {index} with enough content to count.")
        for index in range(6)
    ]
    test_paras, train_paras = sample_test_paragraphs(paragraphs, n_test=2, seed=7)
    assert len(test_paras) == 2
    assert len(train_paras) == 4
    assert all(item.split is ParagraphSplit.TEST for item in test_paras)


def test_apply_heuristic_filters_rejects_duplicates() -> None:
    sample = QASample(
        id="1",
        question="What is the grace period?",
        gold_answer="21 days after the close of the billing cycle.",
        doc_id="doc",
        para_id="doc::p0001",
    )
    duplicate = QASample(
        id="2",
        question="what is the grace period?",
        gold_answer="21 days.",
        doc_id="doc",
        para_id="doc::p0002",
    )
    kept, rejected = apply_heuristic_filters([sample, duplicate])
    assert len(kept) == 1
    assert rejected[0]["reason"] == "duplicate_question"
    assert summarize_heuristic_rejections(rejected)["duplicate_question"] == 1


def test_detect_answer_leakage() -> None:
    leaked = QASample(
        id="1",
        question="What is APR? Annual Percentage Rate is the yearly interest.",
        gold_answer="Annual Percentage Rate is the yearly interest.",
        doc_id="doc",
        para_id="doc::p0001",
    )
    clean = QASample(
        id="2",
        question="What is APR?",
        gold_answer="Annual Percentage Rate is the yearly interest.",
        doc_id="doc",
        para_id="doc::p0002",
    )
    assert detect_answer_leakage(leaked) is True
    assert detect_answer_leakage(clean) is False
    _, rejected = apply_heuristic_filters([leaked])
    assert rejected[0]["reason"] == "answer_in_question"


def test_instruction_backtranslation_prompt_asks_for_question_only() -> None:
    prompt = instruction_backtranslation_prompt("Grace period is 21 days.")
    assert "Instruction back-translation" in prompt
    assert "single key: question" in prompt


def test_generation_prompts_use_domain_not_hardcoded_policy() -> None:
    paragraph = _paragraph("doc", 0, "Grace period is 21 days after billing closes.")
    prompt = _base_generation_prompt(paragraph, domain="healthcare")
    assert "healthcare" in prompt
    assert "policy passage" not in prompt


def test_topic_controlled_generate_one_sample_per_topic() -> None:
    class _FakeClient:
        def complete_json(self, prompt: str, **kwargs: object) -> dict[str, Any]:
            if "Focus topic:" in prompt:
                topic = prompt.split("Focus topic:", 1)[1].split("\n", 1)[0].strip()
                return {
                    "question": f"What about {topic}?",
                    "gold_answer": "See passage.",
                }
            return {"topics": ["billing", "fees"]}

    paragraph = _paragraph("doc", 0, "Billing fees and grace periods apply.")
    samples = topic_controlled_generate(_FakeClient(), paragraph)  # type: ignore[arg-type]
    assert len(samples) == 2
    assert samples[0].metadata["topic"] == "billing"
    assert samples[1].metadata["topic"] == "fees"


def test_generate_instruction_back_translation_sample_forces_passage_as_gold_answer() -> (
    None
):
    class _FakeClient:
        def complete_json(self, prompt: str, **kwargs: object) -> dict[str, Any]:
            return {
                "question": "What is the grace period?",
                "gold_answer": "Model should not invent this.",
            }

    paragraph = _paragraph("doc", 0, "Your grace period is 21 days.")
    sample = generate_instruction_back_translation_sample(_FakeClient(), paragraph)  # type: ignore[arg-type]
    assert sample.gold_answer == paragraph.text
    assert sample.question == "What is the grace period?"


def test_synthetic_qa_quality_prompt_uses_passage_not_model_answer() -> None:
    sample = QASample(
        id="1",
        question="What is the grace period?",
        gold_answer="21 days.",
        doc_id="doc",
        para_id="doc::p0001",
        context="Your grace period for new purchases is 21 days.",
    )
    prompt = synthetic_qa_quality_prompt(sample)
    assert "Model answer" not in prompt
    assert "Passage:" in prompt
    assert sample.context in prompt


def test_retrieve_paragraphs_returns_best_match() -> None:
    paragraphs = [
        _paragraph(
            "doc", 0, "Annual percentage rate and finance charges for purchases."
        ),
        _paragraph("doc", 1, "Refund policy for damaged goods and shipping delays."),
    ]
    results = retrieve_paragraphs("What is the APR for purchases?", paragraphs, top_k=1)
    assert len(results) == 1
    assert "Annual percentage rate" in results[0].text


def test_grounding_overlap_score() -> None:
    score = grounding_overlap_score(
        "The grace period is 21 days after the billing cycle closes.",
        "Your grace period for new purchases is 21 days after the close of the billing cycle.",
    )
    assert score > 0.4


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    records = read_jsonl(path)
    assert records == [{"id": "a", "value": 1}, {"id": "b", "value": 2}]


def test_failure_mode_enum_values() -> None:
    assert FailureMode.REFUSAL_CALIBRATION.value == "refusal_calibration"


def test_effective_test_holdout() -> None:
    assert effective_test_holdout(7, requested=8) == 6
    assert effective_test_holdout(2, requested=8) == 1


def test_build_paragraph_splits_assigns_train_and_test() -> None:
    paragraphs = build_paragraph_splits("finance", n_test_per_doc=2, seed=7)
    splits = {paragraph.split for paragraph in paragraphs}
    assert ParagraphSplit.TEST in splits
    assert ParagraphSplit.TRAIN in splits


def test_effective_synthetic_target_scales_with_train_size() -> None:
    train = [
        _paragraph("doc", index, f"Train paragraph {index} with enough content.")
        for index in range(5)
    ]
    assert effective_synthetic_target(train, requested=500) == 20
    assert effective_synthetic_target(train, requested=500, one_per_paragraph=True) == 5


def test_qa_samples_to_messages_format() -> None:
    sample = QASample(
        id="1",
        question="What is APR?",
        gold_answer="Annual Percentage Rate.",
        doc_id="doc",
        para_id="doc::p0001",
    )
    rows = qa_samples_to_messages([sample])
    assert rows[0]["messages"][-1]["role"] == "assistant"
