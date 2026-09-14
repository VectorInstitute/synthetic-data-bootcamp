"""Score an agent trajectory against a task's oracle actions and required info."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aieng.syn_data.synbench.environment.communicate import CommunicateChecker
from aieng.syn_data.synbench.environment.core import replay_actions
from aieng.syn_data.synbench.environment.hashing import db_hash
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import RewardType, Task


@dataclass
class ScoreResult:
    """Reward breakdown for one scored trajectory."""

    reward: float
    db_reward: float
    communicate_reward: float
    target_db_hash: str
    predicted_db_hash: str
    missing_communicate: list[str]
    execution_ok: bool = True
    tool_errors: list[str] = field(default_factory=list)


def failed_execution_score(errors: list[str]) -> ScoreResult:
    """Zero reward when the live run aborted before a trajectory could be scored.

    Used for uncaught failures in ``run_task`` (LLM crash, unexpected exception).
    Domain tool errors such as "User not found" are *not* abort-worthy: the
    tool loop already returns them to the model so it can recover, and
    outcome-based scoring still zeros reward if the job was never completed.
    """
    return ScoreResult(
        reward=0.0,
        db_reward=0.0,
        communicate_reward=0.0,
        target_db_hash="",
        predicted_db_hash="",
        missing_communicate=[],
        execution_ok=False,
        tool_errors=list(errors),
    )


def score_trajectory(
    domain: DomainBundle,
    task: Task,
    agent_actions: list[Action],
    agent_messages: list[str],
    *,
    tool_errors: list[str] | None = None,
) -> ScoreResult:
    """Score a recorded trajectory by replaying it on a fresh environment.

    This is intentionally separate from the live env used during tool-calling:
    scoring is a pure function of (domain, task, actions, messages) so logged
    trajectories can be rescored offline without re-running the agent.

    ``tool_errors`` are diagnostic only. A failed ``find_user_id`` does not by
    itself force reward 0 — the agent may recover on a later turn. Replay still
    decides whether the final DB / communicate outcomes match the oracle.
    """
    errors = list(tool_errors or [])
    criteria = task.evaluation_criteria
    gold_env = replay_actions(domain, criteria.actions)
    target_hash = db_hash(gold_env.db)

    pred_env = replay_actions(domain, agent_actions, strict=False)
    predicted_hash = db_hash(pred_env.db)

    db_reward = 1.0 if predicted_hash == target_hash else 0.0

    comm_reward, missing = CommunicateChecker.check(
        criteria.communicate_info,
        agent_messages,
    )

    basis = criteria.reward_basis
    reward = 1.0
    if RewardType.DB in basis:
        reward *= db_reward
    if RewardType.COMMUNICATE in basis:
        reward *= comm_reward

    return ScoreResult(
        reward=reward,
        db_reward=db_reward,
        communicate_reward=comm_reward,
        target_db_hash=target_hash,
        predicted_db_hash=predicted_hash,
        missing_communicate=missing,
        execution_ok=True,
        tool_errors=errors,
    )


def score_agent_run(
    domain: DomainBundle,
    task: Task,
    run_task: Callable[[Task], Any],
) -> ScoreResult:
    """Run ``run_task`` and score it, mapping an aborted run to reward 0.

    Live tool errors are recorded on the session and attached to the result;
    they do not skip replay. Only exceptions that escape ``run_task`` skip
    agent replay, because there is no finished trajectory to evaluate.
    """
    try:
        session = run_task(task)
    except Exception as exc:
        return failed_execution_score([str(exc)])
    return score_trajectory(
        domain,
        task,
        session.agent_actions,
        session.agent_messages,
        tool_errors=session.tool_errors,
    )
