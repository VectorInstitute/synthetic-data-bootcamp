"""Verification of draft tasks against domain rules, the FSM, and replay."""

from aieng.syn_data.synbench.verification.domain_checks import run_domain_checks
from aieng.syn_data.synbench.verification.errors import VerificationError
from aieng.syn_data.synbench.verification.pipeline import (
    verify_draft,
    verify_draft_or_raise,
)


__all__ = [
    "VerificationError",
    "run_domain_checks",
    "verify_draft",
    "verify_draft_or_raise",
]
