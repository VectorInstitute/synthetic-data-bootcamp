"""Single-agent baseline: one LLM handling the whole conversation."""

from __future__ import annotations

from aieng.syn_data.synbench.agents.dialogue import run_user_dialogue
from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.user_sim import UserSimulator
from aieng.syn_data.synbench.evaluation.scoring import ScoreResult, score_trajectory
from aieng.syn_data.synbench.llm.client import LLMClient, get_client
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


class SingleToolAgent:
    """
    Single LLM agent with multi-turn tool-calling and optional user dialogue.

    Flow: Task → AgentSession → user dialogue (initial_message + user_sim) →
    ToolCallingLoop per utterance → score_trajectory().

    The loop mutates a live env so the LLM gets tool results. Scoring then
    replays recorded actions on a fresh env (τ-bench style) — it does not
    reuse the live session env, so evaluation stays offline-rescorable.
    """

    def __init__(
        self,
        domain: DomainBundle,
        max_turns: int = 12,
        max_dialogue_turns: int = 5,
        client: LLMClient | None = None,
    ):
        self.domain = domain
        self.max_turns = max_turns  # tool-call rounds per user utterance
        self.max_dialogue_turns = max_dialogue_turns  # user ↔ agent rounds
        self.client = client or get_client()
        self.user_sim = UserSimulator(self.client)
        self.executor = ToolCallingLoop(
            self.domain, client=self.client, max_turns=max_turns
        )

    def run_task(self, task: Task) -> AgentSession:
        """Run the user dialogue for ``task`` and return the finished session."""
        session = AgentSession(domain=self.domain, task=task)
        session.role_trace.append("initial_user_message")
        # Same dialogue helper as AgentPipeline, but only user_sim + executor.
        return run_user_dialogue(
            session,
            task,
            executor=self.executor,
            user_sim=self.user_sim,
            planner=None,
            critic=None,
            roles=["user_sim", "executor"],
            max_dialogue_turns=self.max_dialogue_turns,
        )

    def run_and_score_task(self, task: Task) -> ScoreResult:
        """Run ``task`` and score the resulting trajectory on a fresh environment."""
        session = self.run_task(task)
        return score_trajectory(
            self.domain, task, session.agent_actions, session.agent_messages
        )
