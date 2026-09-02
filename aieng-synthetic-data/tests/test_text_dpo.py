"""Unit tests for DPO preference-pair helpers (no live LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aieng.syn_data.text.dpo.evaluation import (
    build_preference_judge_prompt,
    judge_model_preference,
    split_calibration_prompts,
    summarize_preferences,
)
from aieng.syn_data.text.dpo.generate import filter_sec_paragraphs
from aieng.syn_data.text.dpo.pairs import (
    candidates_to_dpo_pairs,
    summarize_rejected_kinds,
)
from aieng.syn_data.text.dpo.prompts import (
    DEFAULT_BOUNDARY_INSTRUCTION,
    boundary_question_prompt,
    calibration_candidates_prompt,
)
from aieng.syn_data.text.dpo.schemas import (
    BoundaryQuestionKind,
    CalibrationPrompt,
    CandidateKind,
    PreferenceCandidate,
    PreferencePair,
)
from aieng.syn_data.text.dpo.train import pairs_to_trl_rows
from aieng.syn_data.text.io import load_typed_jsonl, save_typed_jsonl
from aieng.syn_data.text.schemas import DocumentRole, Paragraph, ParagraphSplit


def _paragraph(
    doc_id: str, index: int, role: DocumentRole, split: ParagraphSplit
) -> Paragraph:
    return Paragraph(
        doc_id=doc_id,
        para_id=f"{doc_id}::p{index:04d}",
        text="Use a strong passphrase and turn on account alerts.",
        role=role,
        split=split,
        index=index,
    )


def _prompt_with_candidates(
    *,
    chosen: str = "The bulletin recommends a strong passphrase and does not tell you which stock to buy.",
    overreaching: str = "You should buy XYZ stock today for guaranteed gains.",
    underreaching: str = "I cannot help with anything about your investment account.",
    misattribution: str = "The SEC requires every firm to offer passkeys by law.",
    include_overreaching: bool = True,
) -> CalibrationPrompt:
    candidates = [
        PreferenceCandidate(CandidateKind.CORRECTLY_SCOPED, chosen, "hedged"),
        PreferenceCandidate(
            CandidateKind.UNDERREACHING, underreaching, "blank refusal"
        ),
        PreferenceCandidate(
            CandidateKind.AUTHORITY_MISATTRIBUTION,
            misattribution,
            "invents a mandate",
        ),
    ]
    if include_overreaching:
        candidates.insert(
            1,
            PreferenceCandidate(CandidateKind.OVERREACHING, overreaching, "advice"),
        )
    return CalibrationPrompt(
        id="dpo-test01",
        question="What passphrase practices does the bulletin recommend?",
        doc_id="sec_investor_bulletin",
        para_id="sec_investor_bulletin::p0001",
        context="Consider using a strong passphrase instead of a password.",
        question_kind=BoundaryQuestionKind.IN_SCOPE,
        instruction="Answer using the source passage.",
        candidates=candidates,
    )


def test_filter_sec_paragraphs() -> None:
    paragraphs = [
        _paragraph(
            "sec_investor_bulletin",
            0,
            DocumentRole.SCOPE_BOUNDARY,
            ParagraphSplit.TRAIN,
        ),
        _paragraph(
            "sec_investor_bulletin", 1, DocumentRole.SCOPE_BOUNDARY, ParagraphSplit.TEST
        ),
        _paragraph(
            "cfpb_credit_card_agreement",
            0,
            DocumentRole.POLICY_DENSE,
            ParagraphSplit.TRAIN,
        ),
    ]
    kept = filter_sec_paragraphs(paragraphs)
    assert {paragraph.para_id for paragraph in kept} == {
        "sec_investor_bulletin::p0000",
        "sec_investor_bulletin::p0001",
    }


def test_candidates_to_dpo_pairs_expands_three_rejected() -> None:
    pairs = candidates_to_dpo_pairs([_prompt_with_candidates()])
    assert len(pairs) == 3
    kinds = {pair.rejected_kind for pair in pairs}
    assert kinds == {
        CandidateKind.OVERREACHING,
        CandidateKind.UNDERREACHING,
        CandidateKind.AUTHORITY_MISATTRIBUTION,
    }
    assert all(pair.chosen.startswith("The bulletin recommends") for pair in pairs)
    assert all("Question:" in pair.prompt for pair in pairs)
    assert all("Context:" in pair.prompt for pair in pairs)
    assert all(
        "strong passphrase instead of a password" in pair.prompt for pair in pairs
    )


def test_candidates_to_dpo_pairs_skips_identical_and_missing() -> None:
    chosen = "Stay inside the bulletin and hedge."
    prompt = _prompt_with_candidates(
        chosen=chosen,
        overreaching=chosen,
        include_overreaching=True,
    )
    prompt.candidates = [
        PreferenceCandidate(CandidateKind.CORRECTLY_SCOPED, chosen),
        PreferenceCandidate(CandidateKind.OVERREACHING, chosen),
        PreferenceCandidate(CandidateKind.UNDERREACHING, ""),
        PreferenceCandidate(
            CandidateKind.AUTHORITY_MISATTRIBUTION,
            "The SEC guarantees your account against all fraud.",
        ),
    ]
    pairs = candidates_to_dpo_pairs([prompt])
    assert len(pairs) == 1
    assert pairs[0].rejected_kind == CandidateKind.AUTHORITY_MISATTRIBUTION


def test_candidates_to_dpo_pairs_skips_missing_chosen() -> None:
    prompt = _prompt_with_candidates()
    prompt.candidates = [
        PreferenceCandidate(CandidateKind.OVERREACHING, "Buy this ETF."),
    ]
    assert candidates_to_dpo_pairs([prompt]) == []


def test_summarize_rejected_kinds() -> None:
    pairs = candidates_to_dpo_pairs([_prompt_with_candidates()])
    counts = summarize_rejected_kinds(pairs)
    assert counts[CandidateKind.OVERREACHING.value] == 1
    assert counts[CandidateKind.UNDERREACHING.value] == 1
    assert counts[CandidateKind.AUTHORITY_MISATTRIBUTION.value] == 1
    assert CandidateKind.CORRECTLY_SCOPED.value not in counts


def test_preference_pair_jsonl_roundtrip(tmp_path: Path) -> None:
    pairs = candidates_to_dpo_pairs([_prompt_with_candidates()])
    path = tmp_path / "pairs.jsonl"
    save_typed_jsonl(path, pairs, to_dict=PreferencePair.to_dict)
    loaded = load_typed_jsonl(path, PreferencePair.from_dict)
    assert len(loaded) == len(pairs)
    assert loaded[0].rejected_kind == pairs[0].rejected_kind
    assert loaded[0].to_trl_row() == {
        "prompt": loaded[0].prompt,
        "chosen": loaded[0].chosen,
        "rejected": loaded[0].rejected,
    }


def test_pairs_to_trl_rows_uses_conversational_format() -> None:
    pair = candidates_to_dpo_pairs([_prompt_with_candidates()])[0]
    row = pairs_to_trl_rows([pair])[0]

    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1] == {"role": "user", "content": pair.prompt}
    assert row["chosen"] == [{"role": "assistant", "content": pair.chosen}]
    assert row["rejected"] == [{"role": "assistant", "content": pair.rejected}]


def test_calibration_prompt_jsonl_roundtrip(tmp_path: Path) -> None:
    prompt = _prompt_with_candidates()
    path = tmp_path / "candidates.jsonl"
    save_typed_jsonl(path, [prompt], to_dict=CalibrationPrompt.to_dict)
    loaded = load_typed_jsonl(path, CalibrationPrompt.from_dict)
    assert loaded[0].id == prompt.id
    assert loaded[0].chosen_answer() == prompt.chosen_answer()
    sample = loaded[0].to_qa_sample()
    assert sample.question == prompt.question
    assert sample.failure_mode is not None
    assert sample.failure_mode.value == "refusal_calibration"


def test_boundary_prompts_mention_question_kind() -> None:
    text = boundary_question_prompt("passage", BoundaryQuestionKind.OUT_OF_SCOPE)
    assert "out_of_scope" in text
    cand = calibration_candidates_prompt("passage", "Should I buy this stock?")
    assert "correctly_scoped" in cand
    assert "authority_misattribution" in cand
    assert "source passage" in DEFAULT_BOUNDARY_INSTRUCTION.lower()


def test_split_calibration_prompts_holds_out_each_kind() -> None:
    prompts: list[CalibrationPrompt] = []
    for kind in BoundaryQuestionKind:
        for index in range(2):
            prompt = _prompt_with_candidates()
            prompt.id = f"{kind.value}-{index}"
            prompt.question_kind = kind
            prompts.append(prompt)

    train, evaluation = split_calibration_prompts(prompts)

    assert len(train) == 3
    assert len(evaluation) == 3
    assert {prompt.question_kind for prompt in evaluation} == set(BoundaryQuestionKind)
    assert {prompt.id for prompt in train}.isdisjoint(
        {prompt.id for prompt in evaluation}
    )


class _OrderedJudge:
    """Prefer DPO by recognizing which response occupies each position."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return json.dumps(self.complete_json(prompt))

    def complete_json(self, prompt: str, **_: Any) -> dict[str, str]:
        answer_a = prompt.split("Response A:\n", 1)[1].split("\n\nResponse B:", 1)[0]
        return {
            "winner": "A" if "DPO answer" in answer_a else "B",
            "reasoning": "The DPO answer is better scoped.",
        }


def test_preference_judge_is_position_balanced_and_has_no_gold() -> None:
    prompt = _prompt_with_candidates()
    judge_prompt = build_preference_judge_prompt(prompt, "base", "dpo")
    assert prompt.context in judge_prompt
    assert prompt.chosen_answer() not in judge_prompt
    assert "no gold answer" in judge_prompt.lower()

    result = judge_model_preference(
        _OrderedJudge(), prompt, "Baseline answer", "DPO answer"
    )
    assert result.winner == "dpo"
    assert result.forward_winner == "dpo"
    assert result.reverse_winner == "dpo"

    summary = summarize_preferences([result])
    assert summary["overall"]["dpo_wins"] == 1
    assert summary["overall"]["dpo_preference_rate"] == 1.0
