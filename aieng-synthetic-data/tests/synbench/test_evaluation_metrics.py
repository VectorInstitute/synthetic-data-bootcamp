"""Tests for aggregating per-task scores into benchmark metrics."""

from aieng.syn_data.synbench.evaluation.metrics import MetricsCollector
from aieng.syn_data.synbench.evaluation.scoring import ScoreResult


def _score(reward: float) -> ScoreResult:
    """Build a minimal ScoreResult for metrics aggregation tests."""
    return ScoreResult(
        reward=reward,
        db_reward=reward,
        communicate_reward=reward,
        target_db_hash="t",
        predicted_db_hash="t" if reward == 1.0 else "p",
        partial_action_match=reward,
        missing_communicate=[],
    )


def test_pass_at_1_two_tasks():
    """Two perfect scores yield pass@1 of 1.0."""
    metrics = MetricsCollector()
    metrics.add("task_a", _score(1.0))
    metrics.add("task_b", _score(1.0))
    assert metrics.pass_at_1() == 1.0
    summary = metrics.summary()
    assert summary["n_tasks"] == 2


def test_pass_at_1_partial():
    """One success and one failure yield pass@1 of 0.5."""
    metrics = MetricsCollector()
    metrics.add("task_a", _score(1.0))
    metrics.add("task_b", _score(0.0))
    assert metrics.pass_at_1() == 0.5
