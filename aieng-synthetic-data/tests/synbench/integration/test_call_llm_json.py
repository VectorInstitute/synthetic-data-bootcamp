"""Integration test for the live JSON generation call."""

import pytest

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.generation.llm import call_llm_json
from aieng.syn_data.synbench.generation.prompt import PromptBuilder
from aieng.syn_data.synbench.generation.sampler import ConstraintSampler
from aieng.syn_data.synbench.schemas.tasks import Task


@pytest.mark.integration_test
def test_call_llm_json_returns_valid_task_draft(mock_retail_path):
    """A real model returns JSON that validates as a Task."""
    domain = load_domain(mock_retail_path)
    constraints = ConstraintSampler(domain, seed=42).sample()
    prompt = PromptBuilder().build(domain, constraints)

    data = call_llm_json(prompt)

    assert isinstance(data, dict)
    data.setdefault("id", "integration_test_draft")
    data.setdefault("task_type", constraints.task_type)

    draft = Task.model_validate(data)
    assert draft.description
    assert draft.task_type == constraints.task_type
    assert draft.user_scenario.initial_message
    assert draft.evaluation_criteria.actions
