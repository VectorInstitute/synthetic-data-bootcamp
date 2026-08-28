"""Paths and demo defaults for DPO preference-pair generation."""

from __future__ import annotations

from aieng.syn_data.text.config import IMPLEMENTATION_DIR


SEC_DOC_ID = "sec_investor_bulletin"

DPO_DIR = IMPLEMENTATION_DIR / "DPO"
DPO_CANDIDATES_PATH = DPO_DIR / "synthetic" / "dpo_candidates.jsonl"
DPO_EVAL_PROMPTS_PATH = DPO_DIR / "synthetic" / "dpo_eval_prompts.jsonl"
DPO_PAIRS_PATH = DPO_DIR / "synthetic" / "dpo_preference_pairs.jsonl"
DPO_ADAPTER_DIR = IMPLEMENTATION_DIR / "models" / "dpo_lora_adapter"
DPO_PREFERENCE_RESULTS_PATH = DPO_DIR / "results" / "dpo_preference_evaluation.json"

# Small bootcamp demo: 8 questions x 3 rejected kinds ≈ 24 preference pairs.
DEFAULT_DPO_QUESTIONS = 8
