"""Expand labeled candidates into TRL DPO preference pairs."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from json_repair import repair_json

from aieng.syn_data.text.clients import LLMClient
from aieng.syn_data.text.dpo.schemas import (
    REJECTED_CANDIDATE_KINDS,
    CalibrationPrompt,
    CandidateKind,
    PreferencePair,
)
from aieng.syn_data.text.evaluation import DEFAULT_EVAL_SYSTEM, build_eval_prompt
from aieng.syn_data.text.judge import build_pairwise_judge_prompt
from aieng.syn_data.text.prompts import RESPONSE_JUDGE_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def candidates_to_dpo_pairs(
    prompts: list[CalibrationPrompt],
    *,
    system: str = DEFAULT_EVAL_SYSTEM,
) -> list[PreferencePair]:
    """Turn each calibration prompt into chosen-vs-rejected DPO rows.

    The correctly scoped answer is always ``chosen``. Each other candidate
    becomes one ``rejected`` row. Empty answers and exact duplicates of the
    chosen text are skipped.
    """
    pairs: list[PreferencePair] = []
    for prompt in prompts:
        chosen = prompt.chosen_answer().strip()
        if not chosen:
            logger.warning("Skipping %s: missing correctly_scoped answer", prompt.id)
            continue
        sample = prompt.to_qa_sample()
        user_prompt = build_eval_prompt(sample)
        for kind in REJECTED_CANDIDATE_KINDS:
            candidate = prompt.candidate_by_kind(kind)
            if candidate is None:
                continue
            rejected = candidate.answer.strip()
            if not rejected or rejected == chosen:
                logger.warning(
                    "Skipping %s vs %s: empty or identical to chosen",
                    prompt.id,
                    kind.value,
                )
                continue
            pairs.append(
                PreferencePair(
                    id=f"{prompt.id}-{kind.value}",
                    prompt=user_prompt,
                    chosen=chosen,
                    rejected=rejected,
                    rejected_kind=kind,
                    prompt_id=prompt.id,
                    doc_id=prompt.doc_id,
                    para_id=prompt.para_id,
                    metadata={
                        "question_kind": prompt.question_kind.value,
                        "rejected_rationale": candidate.rationale,
                        "candidate_kind": kind.value,
                        "system": system,
                    },
                )
            )
    return pairs


def _pairwise_winner(
    judge: LLMClient,
    prompt: CalibrationPrompt,
    chosen: str,
    rejected: str,
) -> str:
    sample = prompt.to_qa_sample()
    judge_prompt = build_pairwise_judge_prompt(sample, chosen, rejected)
    if hasattr(judge, "complete_json"):
        payload = cast(
            dict[str, Any],
            judge.complete_json(
                judge_prompt,
                system=RESPONSE_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=256,
            ),
        )
    else:
        raw = judge.complete(
            judge_prompt,
            system=RESPONSE_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
        )
        parsed = json.loads(repair_json(raw))
        if not isinstance(parsed, dict):
            msg = f"Pairwise judge did not return a JSON object: {raw[:300]!r}"
            raise ValueError(msg)
        payload = parsed
    return str(payload.get("winner", "")).strip().upper()


def filter_pairs_with_judge(
    judge: LLMClient,
    prompts: list[CalibrationPrompt],
    pairs: list[PreferencePair],
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    """Keep pairs where the judge prefers chosen (A) over rejected (B)."""
    prompts_by_id = {prompt.id: prompt for prompt in prompts}
    kept: list[PreferencePair] = []
    dropped: list[PreferencePair] = []
    for pair in pairs:
        source = prompts_by_id.get(pair.prompt_id)
        if source is None:
            dropped.append(pair)
            continue
        try:
            winner = _pairwise_winner(judge, source, pair.chosen, pair.rejected)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning(
                "Judge failed for %s: %s: %s",
                pair.id,
                type(exc).__name__,
                exc,
            )
            dropped.append(pair)
            continue
        if winner == "A":
            kept.append(pair)
        else:
            pair.metadata["judge_winner"] = winner
            dropped.append(pair)
    return kept, dropped


def summarize_rejected_kinds(pairs: list[PreferencePair]) -> dict[str, int]:
    """Count preference pairs by rejected candidate kind."""
    counts: dict[str, int] = {kind.value: 0 for kind in CandidateKind}
    counts.pop(CandidateKind.CORRECTLY_SCOPED.value, None)
    for pair in pairs:
        counts[pair.rejected_kind.value] = counts.get(pair.rejected_kind.value, 0) + 1
    return counts
