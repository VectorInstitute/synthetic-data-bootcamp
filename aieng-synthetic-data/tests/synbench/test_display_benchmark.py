"""Tests for notebook benchmark summary display."""

from aieng.syn_data.synbench.display import show_benchmark_by_task_type
from aieng.syn_data.synbench.evaluation.metrics import MetricsCollector
from aieng.syn_data.synbench.evaluation.scoring import ScoreResult
from aieng.syn_data.synbench.schemas.tasks import Task, UserScenario


def _score(*, reward: float, turns: int) -> ScoreResult:
    return ScoreResult(
        reward=reward,
        db_reward=reward,
        communicate_reward=reward,
        target_db_hash="t",
        predicted_db_hash="t" if reward == 1.0 else "p",
        missing_communicate=[],
        dialogue_turns=turns,
    )


def _task(task_id: str, task_type: str, personality: str) -> Task:
    return Task(
        id=task_id,
        task_type=task_type,
        user_scenario=UserScenario(personality_style=personality),
    )


def test_show_benchmark_by_task_type_includes_model_and_avg_turns(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(
        "aieng.syn_data.synbench.display._emit", captured.append
    )
    metrics = MetricsCollector()
    metrics.add("a", _score(reward=1.0, turns=2))
    metrics.add("b", _score(reward=0.0, turns=4))
    tasks = [
        _task("a", "inquiry", "rushed"),
        _task("b", "inquiry", "anxious"),
    ]
    show_benchmark_by_task_type(metrics, tasks, agent_model="gemini-test")
    markdown = captured[0]
    assert "**Agent LLM:** `gemini-test`" in markdown
    assert "| avg_turns |" in markdown
    assert "| `inquiry` | 1 | 0 | 1 | 2 | 0.50 | 3.00 |" in markdown
