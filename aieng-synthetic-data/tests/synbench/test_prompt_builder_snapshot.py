"""Tests that the generation prompt carries the domain context it must."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.generation.prompt import PromptBuilder
from aieng.syn_data.synbench.generation.sampler import (
    ConstraintSampler,
    SampleConstraints,
)


def test_prompt_contains_policy_tools_fsm(mock_retail_path):
    """The prompt includes policy, tools, FSM path, and sampled entities."""
    domain = load_domain(mock_retail_path)
    constraints = ConstraintSampler(domain).sample()
    prompt = PromptBuilder().build(domain, constraints)
    assert "Policy" in prompt or "policy" in prompt.lower()
    assert "get_order" in prompt
    assert constraints.task_type in prompt
    assert "FSM" in prompt or "path" in prompt.lower()
    assert constraints.primary_id in prompt
    assert constraints.entities["order_id"] in prompt
    assert domain.generation.agent_role in prompt
    assert "Do not include" in prompt
    assert "communicate_info" in prompt
    assert "Entity context" in prompt


def test_prompt_prefers_matching_seed_example(mock_retail_path):
    """The few-shot example matches the sampled task type when one exists."""
    domain = load_domain(mock_retail_path)
    constraints = ConstraintSampler(domain, seed=0).sample()
    prompt = PromptBuilder().build(domain, constraints)
    if constraints.task_type == "cancel":
        assert "seed_cancel" in prompt


def test_prompt_neutral_style_when_missing(mock_retail_path):
    """The prompt falls back to neutral wording when no style was sampled."""
    domain = load_domain(mock_retail_path)
    base = ConstraintSampler(domain, seed=1).sample()
    constraints = SampleConstraints(
        task_type=base.task_type,
        entities=base.entities,
        fsm_path=base.fsm_path,
        primary_id=base.primary_id,
        entity_context=base.entity_context,
        personality_style=None,
    )
    prompt = PromptBuilder().build(domain, constraints)
    assert "No personality style was sampled" in prompt
