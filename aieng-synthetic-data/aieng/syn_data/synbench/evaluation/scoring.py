"""Score an agent trajectory against a task's oracle actions and required info."""

from __future__ import annotations

from dataclasses import dataclass

from aieng.syn_data.synbench.environment.communicate import CommunicateChecker
from aieng.syn_data.synbench.environment.core import replay_actions
from aieng.syn_data.synbench.environment.hashing import db_hash
from aieng.syn_data.synbench.schemas.actions import Action, action_fingerprint
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
    partial_action_match: float
    missing_communicate: list[str]


def score_trajectory(
    domain: DomainBundle,
    task: Task,
    agent_actions: list[Action],
    agent_messages: list[str],
) -> ScoreResult:
    """Score a recorded trajectory by replaying it on a fresh environment.

    This is intentionally separate from the live env used during tool-calling:
    scoring is a pure function of (domain, task, actions, messages) so logged
    trajectories can be rescored offline without re-running the agent.
    """
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

    oracle_fps = {action_fingerprint(a) for a in criteria.actions}
    agent_fps = {action_fingerprint(a) for a in agent_actions}
    partial = len(oracle_fps & agent_fps) / len(oracle_fps) if oracle_fps else 1.0

    return ScoreResult(
        reward=reward,
        db_reward=db_reward,
        communicate_reward=comm_reward,
        target_db_hash=target_hash,
        predicted_db_hash=predicted_hash,
        partial_action_match=partial,
        missing_communicate=missing,
    )
