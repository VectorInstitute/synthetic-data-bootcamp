"""Tests that generation works end to end against the mock LLM."""

import os

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.generation.generator import TrajectoryGenerator
from aieng.syn_data.synbench.verification.pipeline import verify_draft


def test_mock_llm_draft_verifies(mock_retail_path):
    """A draft generated from the mock fixture passes verification."""
    os.environ["MOCK_LLM"] = "1"
    domain = load_domain(mock_retail_path)
    gen = TrajectoryGenerator(domain)
    draft = gen.generate_one("test_mock_1")
    result = verify_draft(domain, draft)
    assert result.verification_report.passed
