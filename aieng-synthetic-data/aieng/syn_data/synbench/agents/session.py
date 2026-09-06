"""Mutable state carried through a single agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aieng.syn_data.synbench.environment.core import Environment
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


@dataclass
class AgentSession:
    """Chat transcript, recorded actions, and environment for one task run."""

    domain: DomainBundle
    task: Task
    messages: list[dict[str, Any]] = field(default_factory=list)
    agent_actions: list[Action] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    role_trace: list[str] = field(default_factory=list)
    plan: str = ""
    env: Environment | None = None

    def ensure_env(self) -> Environment:
        """Return the session environment, creating it on first use."""
        if self.env is None:
            self.env = Environment(self.domain)
        return self.env
