"""Paths and demo defaults for DPO preference-pair generation."""

from __future__ import annotations

from aieng.syn_data.text.config import DATA_DIR, IMPLEMENTATION_DIR


SEC_DOC_ID = "sec_investor_bulletin"

DPO_CANDIDATES_PATH = DATA_DIR / "synthetic" / "dpo_candidates.jsonl"
DPO_PAIRS_PATH = DATA_DIR / "synthetic" / "dpo_preference_pairs.jsonl"
DPO_ADAPTER_DIR = IMPLEMENTATION_DIR / "models" / "dpo_lora_adapter"

# Small bootcamp demo: 8 questions x 3 rejected kinds ≈ 24 preference pairs.
DEFAULT_DPO_QUESTIONS = 8
