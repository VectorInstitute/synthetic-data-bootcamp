"""DPO preference-pair generation and optional LoRA training."""

from aieng.syn_data.text.dpo.config import (
    DEFAULT_DPO_QUESTIONS,
    DPO_ADAPTER_DIR,
    DPO_CANDIDATES_PATH,
    DPO_EVAL_PROMPTS_PATH,
    DPO_PAIRS_PATH,
    DPO_PREFERENCE_RESULTS_PATH,
    SEC_DOC_ID,
)
from aieng.syn_data.text.dpo.evaluation import (
    PREFERENCE_JUDGE_SYSTEM,
    PreferenceJudgment,
    build_preference_judge_prompt,
    evaluate_model_preferences,
    generate_preference_responses,
    judge_model_preference,
    split_calibration_prompts,
    summarize_preferences,
)
from aieng.syn_data.text.dpo.generate import (
    filter_sec_paragraphs,
    generate_boundary_prompts,
    generate_calibration_candidates,
    generate_calibration_set,
)
from aieng.syn_data.text.dpo.pairs import (
    candidates_to_dpo_pairs,
    filter_pairs_with_judge,
    summarize_rejected_kinds,
)
from aieng.syn_data.text.dpo.prompts import DEFAULT_BOUNDARY_INSTRUCTION
from aieng.syn_data.text.dpo.schemas import (
    REJECTED_CANDIDATE_KINDS,
    BoundaryQuestionKind,
    CalibrationPrompt,
    CandidateKind,
    PreferenceCandidate,
    PreferencePair,
)
from aieng.syn_data.text.dpo.train import (
    build_dpo_dataset,
    pairs_to_trl_rows,
    train_lora_dpo,
)


__all__ = [
    "DEFAULT_BOUNDARY_INSTRUCTION",
    "DEFAULT_DPO_QUESTIONS",
    "DPO_ADAPTER_DIR",
    "DPO_CANDIDATES_PATH",
    "DPO_EVAL_PROMPTS_PATH",
    "DPO_PAIRS_PATH",
    "DPO_PREFERENCE_RESULTS_PATH",
    "PREFERENCE_JUDGE_SYSTEM",
    "REJECTED_CANDIDATE_KINDS",
    "SEC_DOC_ID",
    "BoundaryQuestionKind",
    "CalibrationPrompt",
    "CandidateKind",
    "PreferenceCandidate",
    "PreferenceJudgment",
    "PreferencePair",
    "build_preference_judge_prompt",
    "build_dpo_dataset",
    "candidates_to_dpo_pairs",
    "filter_pairs_with_judge",
    "filter_sec_paragraphs",
    "generate_boundary_prompts",
    "generate_calibration_candidates",
    "generate_calibration_set",
    "generate_preference_responses",
    "judge_model_preference",
    "evaluate_model_preferences",
    "pairs_to_trl_rows",
    "split_calibration_prompts",
    "summarize_rejected_kinds",
    "summarize_preferences",
    "train_lora_dpo",
]
