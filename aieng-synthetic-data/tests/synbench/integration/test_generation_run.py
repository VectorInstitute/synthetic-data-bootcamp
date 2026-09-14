"""Integration test for a full generate-verify-write run."""

import json

import pytest

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.generation.generator import GenerationRun
from aieng.syn_data.synbench.schemas.tasks import Task


@pytest.mark.integration_test
def test_generation_run_and_verify_writes_tasks(mock_retail_path, tmp_path):
    """Generated tasks verify and are written to ``tasks.json``."""
    domain = load_domain(mock_retail_path)
    run = GenerationRun(domain, seed=42)
    verified_tasks, rejected_tasks = run.run_and_verify(n=3)
    tasks_path = run.write_tasks(verified_tasks, tmp_path)

    assert isinstance(verified_tasks, list)
    assert verified_tasks, f"no task verified; rejections={rejected_tasks}"
    assert len(rejected_tasks) + len(verified_tasks) == 3, (
        f"total tasks={len(rejected_tasks) + len(verified_tasks)} != 3"
    )
    assert all(isinstance(t, Task) for t in verified_tasks)
    for task in verified_tasks:
        assert task.id
        assert task.description
        assert task.user_scenario.initial_message
        assert task.evaluation_criteria.actions

    assert tasks_path == tmp_path / "tasks.json"
    assert tasks_path.exists()
    payload = json.loads(tasks_path.read_text())
    assert "tasks" in payload
    assert len(payload["tasks"]) == len(verified_tasks)
