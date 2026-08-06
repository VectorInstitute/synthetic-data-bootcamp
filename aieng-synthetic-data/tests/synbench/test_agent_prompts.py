"""Tests for the per-role system prompts."""

from aieng.syn_data.synbench.agents.prompts import (
    agent_system_prompt,
    user_sim_system_prompt,
)
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.tasks import Task, UserScenario


def test_agent_system_prompt_hides_instructions_and_description(mock_retail_path):
    """The agent under test must infer the goal from the conversation alone."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    prompt = agent_system_prompt(domain, task)
    assert "Instructions:" not in prompt
    assert task.user_scenario.instructions not in prompt
    assert task.description not in prompt
    assert "User name:" not in prompt
    assert task.id in prompt
    assert domain.generation.agent_role in prompt


def test_user_sim_uses_task_user_name_and_style(mock_retail_path):
    """The simulator prompt carries the task's customer name and style."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    assert task.user_scenario.user_name == "Alice Chen"
    assert task.user_scenario.personality_style == "rushed"
    prompt = user_sim_system_prompt(domain, task)
    assert "User name: Alice Chen" in prompt
    assert "Personality style: rushed" in prompt
    assert "Impatient and terse" in prompt or "impatient" in prompt.lower()
    assert domain.user_simulator["persona"] not in prompt
    assert task.user_scenario.instructions in prompt
    assert "Instructions:" in prompt


def test_legacy_persona_migrates_to_user_name():
    """An old ``persona`` field is mapped onto ``user_name``."""
    task = Task.model_validate(
        {
            "id": "legacy",
            "user_scenario": {
                "persona": "Alice Chen",
                "instructions": "Ask about the order.",
                "initial_message": "Hi",
            },
        }
    )
    assert task.user_scenario.user_name == "Alice Chen"
    assert "persona" not in task.user_scenario.model_dump()


def test_user_scenario_keeps_name_and_style_separate():
    """Customer identity and interaction style stay in distinct fields."""
    scenario = UserScenario(
        user_name="Alice Chen",
        personality_style="rushed",
        instructions="Ask about order ord_1001 status.",
        initial_message="Where's my order? Need an update now.",
    )
    assert scenario.user_name == "Alice Chen"
    assert scenario.personality_style == "rushed"
    assert "rushed" not in scenario.user_name
