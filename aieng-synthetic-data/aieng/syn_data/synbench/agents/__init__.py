"""Agents that solve benchmark tasks: single-agent, multi-agent, and user simulation."""

from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.pipeline import DEFAULT_ROLES, AgentPipeline
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.single import SingleToolAgent


__all__ = [
    "AgentPipeline",
    "AgentSession",
    "DEFAULT_ROLES",
    "SingleToolAgent",
    "ToolCallingLoop",
]
