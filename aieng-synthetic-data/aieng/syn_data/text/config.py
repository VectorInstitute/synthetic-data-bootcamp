"""Configuration constants for the policy-document QA workflow."""

from __future__ import annotations

from pathlib import Path

from aieng.syn_data.text.schemas import DocumentRole, DocumentSpec, FailureMode


DEFAULT_DOMAIN = "finance"

# Default relative paths under implementations/qa_text_generation/.
IMPLEMENTATION_DIR = Path("implementations/qa_text_generation")
DATA_DIR = IMPLEMENTATION_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PARAGRAPHS_PATH = DATA_DIR / "paragraphs.jsonl"
TEST_SET_PATH = DATA_DIR / "test" / "test_set.jsonl"
SYNTHETIC_RAW_PATH = DATA_DIR / "synthetic" / "synthetic_raw.jsonl"
SYNTHETIC_FILTERED_PATH = DATA_DIR / "synthetic" / "synthetic_filtered.jsonl"
SYNTHETIC_TRAIN_PATH = DATA_DIR / "synthetic" / "synthetic_train.jsonl"
RESULTS_DIR = DATA_DIR / "results"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
BASELINE_SCORES_PATH = RESULTS_DIR / "baseline_scores.json"
FINETUNED_PREDICTIONS_PATH = RESULTS_DIR / "finetuned_predictions.jsonl"
COMPARISON_REPORT_PATH = RESULTS_DIR / "comparison_report.json"

DEFAULT_TEST_PARAS_PER_DOC = 3
DEFAULT_SYNTHETIC_TARGET_SIZE = 50
DEFAULT_JUDGE_THRESHOLD = 3.5

FAILURE_MODE_GUIDANCE: dict[FailureMode, str] = {
    FailureMode.FORMAT_NON_COMPLIANCE: (
        "Ask for a specific output structure (JSON, numbered list, clause citation)."
    ),
    FailureMode.DOMAIN_VOCABULARY_DRIFT: (
        "Use precise domain terms from the passage (APR, grace period, fiduciary)."
    ),
    FailureMode.REFUSAL_CALIBRATION: (
        "Include in-scope and out-of-scope questions; gold answer should refuse when needed."
    ),
    FailureMode.MULTI_CONSTRAINT_COLLAPSE: (
        "Combine multiple policy rules or exceptions in a single question."
    ),
}

FINANCE_DOCUMENTS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        doc_id="cfpb_credit_card_agreement",
        title="CFPB Sample Credit Card Agreement",
        role=DocumentRole.POLICY_DENSE,
        domain="finance",
        source_url=(
            "https://files.consumerfinance.gov/f/documents/"
            "201401_cfpb_credit-card-agreement_english.pdf"
        ),
        local_path=str(DOCUMENTS_DIR / "cfpb_credit_card_agreement.txt"),
    ),
    DocumentSpec(
        doc_id="sec_investor_bulletin",
        title="SEC Investor Bulletin",
        role=DocumentRole.SCOPE_BOUNDARY,
        domain="finance",
        source_url="https://www.sec.gov/files/ib_fraud.pdf",
        local_path=str(DOCUMENTS_DIR / "sec_investor_bulletin.txt"),
    ),
)

DOCUMENT_ROLE_BY_FAILURE: dict[DocumentRole, tuple[FailureMode, ...]] = {
    DocumentRole.POLICY_DENSE: (
        FailureMode.FORMAT_NON_COMPLIANCE,
        FailureMode.DOMAIN_VOCABULARY_DRIFT,
        FailureMode.MULTI_CONSTRAINT_COLLAPSE,
    ),
    DocumentRole.SCOPE_BOUNDARY: (
        FailureMode.REFUSAL_CALIBRATION,
        FailureMode.FORMAT_NON_COMPLIANCE,
    ),
}
