"""Tests for aggregating per-task scores into benchmark metrics."""

from aieng.syn_data.synbench.agents.single import SingleToolAgent
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.evaluation.metrics import MetricsCollector


def test_pass_at_1_two_tasks(mock_retail_path):
    """The scripted agent solves both seed tasks, giving pass@1 of 1.0."""
    domain = load_domain(mock_retail_path)
    agent = SingleToolAgent(domain)
    metrics = MetricsCollector()
    for task in domain.seed_tasks[:2]:
        metrics.add(task.id, agent.run_and_score_task(task))
    assert metrics.pass_at_1() == 1.0
    summary = metrics.summary()
    assert summary["n_tasks"] == 2
