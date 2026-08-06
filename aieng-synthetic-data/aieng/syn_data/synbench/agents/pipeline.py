"""Multi-agent pipeline wiring the planner, executor, critic, and user simulator."""

from __future__ import annotations

from aieng.syn_data.synbench.agents.critic import Critic
from aieng.syn_data.synbench.agents.dialogue import run_user_dialogue
from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.planner import Planner
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.user_sim import UserSimulator
from aieng.syn_data.synbench.evaluation.scoring import ScoreResult, score_trajectory
from aieng.syn_data.synbench.llm.client import LLMClient, get_client
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


DEFAULT_ROLES = ["user_sim", "planner", "executor", "critic"]


class AgentPipeline:
    """Multi-role pipeline: user_sim → planner → executor → critic per dialogue turn."""

    def __init__(
        self,
        domain: DomainBundle,
        roles: list[str] | None = None,
        max_turns: int = 12,
        max_dialogue_turns: int = 5,
        client: LLMClient | None = None,
    ):
        self.domain = domain
        self.roles = roles or DEFAULT_ROLES
        self.max_turns = max_turns  # tool-call rounds per user utterance
        self.max_dialogue_turns = max_dialogue_turns  # user ↔ pipeline rounds
        self.client = client or get_client()
        self.user_sim = UserSimulator(self.client)
        self.planner = Planner(self.client)
        self.critic = Critic(self.client)
        self.executor = ToolCallingLoop(
            self.domain, client=self.client, max_turns=max_turns
        )

    def run_task(self, task: Task) -> AgentSession:
        """Run the full role dialogue for ``task`` and return the finished session."""
        session = AgentSession(domain=self.domain, task=task)
        return run_user_dialogue(
            session,
            task,
            executor=self.executor,
            user_sim=self.user_sim,
            planner=self.planner,
            critic=self.critic,
            roles=self.roles,
            max_dialogue_turns=self.max_dialogue_turns,
        )

    def run_and_score_task(self, task: Task) -> ScoreResult:
        """Run ``task`` and score the resulting trajectory on a fresh environment."""
        session = self.run_task(task)
        return score_trajectory(
            self.domain, task, session.agent_actions, session.agent_messages
        )
