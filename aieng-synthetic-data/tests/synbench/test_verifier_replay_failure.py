"""Tests that verification rejects drafts which fail replay or FSM checks."""

import json
from pathlib import Path

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.verification.pipeline import verify_draft


FIXTURES = Path(__file__).parent / "fixtures"


def test_invalid_replay_fails(mock_retail_path):
    """A draft whose oracle actions cannot be replayed is rejected."""
    domain = load_domain(mock_retail_path)
    with open(FIXTURES / "tasks_invalid_replay.json") as f:
        draft = Task.model_validate(json.load(f))
    result = verify_draft(domain, draft)
    assert not result.verification_report.passed
    assert any(
        "Replay" in e or "not found" in e for e in result.verification_report.errors
    )


def test_invalid_fsm_fails(mock_retail_path):
    """A draft whose actions violate the FSM path is rejected."""
    domain = load_domain(mock_retail_path)
    with open(FIXTURES / "tasks_invalid_fsm.json") as f:
        draft = Task.model_validate(json.load(f))
    result = verify_draft(domain, draft)
    assert not result.verification_report.passed
