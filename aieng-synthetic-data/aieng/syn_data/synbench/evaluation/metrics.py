"""Aggregate per-task scores into benchmark-level metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aieng.syn_data.synbench.evaluation.scoring import ScoreResult


@dataclass
class RunMetrics:
    """Scores recorded for a single task run."""

    task_id: str
    reward: float
    db_reward: float
    communicate_reward: float
    partial_action_match: float


@dataclass
class MetricsCollector:
    """Accumulates per-task scores and reports benchmark-level aggregates."""

    runs: list[RunMetrics] = field(default_factory=list)

    def add(self, task_id: str, score: ScoreResult) -> None:
        """Record the score for one task run."""
        self.runs.append(
            RunMetrics(
                task_id=task_id,
                reward=score.reward,
                db_reward=score.db_reward,
                communicate_reward=score.communicate_reward,
                partial_action_match=score.partial_action_match,
            )
        )

    def pass_at_1(self) -> float:
        """Fraction of recorded runs that earned full reward."""
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.reward >= 1.0) / len(self.runs)

    def summary(self) -> dict[str, Any]:
        """Return aggregate metrics plus the individual run records."""
        return {
            "n_tasks": len(self.runs),
            "pass_at_1": self.pass_at_1(),
            "mean_db_reward": sum(r.db_reward for r in self.runs)
            / max(len(self.runs), 1),
            "mean_communicate_reward": sum(r.communicate_reward for r in self.runs)
            / max(len(self.runs), 1),
            "mean_partial_action_match": sum(r.partial_action_match for r in self.runs)
            / max(len(self.runs), 1),
            "runs": [r.__dict__ for r in self.runs],
        }
