"""Tests that a well-formed draft passes verification and yields a target hash."""

import json
from pathlib import Path

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.verification.pipeline import verify_draft


FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_fixture_passes(mock_retail_path):
    """The valid fixture verifies and records a target database hash."""
    domain = load_domain(mock_retail_path)
    with open(FIXTURES / "tasks_valid.json") as f:
        draft = Task.model_validate(json.load(f))
    result = verify_draft(domain, draft)
    assert result.verification_report.passed
    assert result.verification_report.target_db_hash
