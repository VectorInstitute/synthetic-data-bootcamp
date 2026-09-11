"""Pydantic schemas for domains, tasks, tools, actions, and verification."""

from aieng.syn_data.synbench.schemas.actions import Action, normalize_action
from aieng.syn_data.synbench.schemas.domain import DomainBundle, DomainManifest
from aieng.syn_data.synbench.schemas.generation import (
    GenerationConfig,
    RelatedEntitySpec,
)
from aieng.syn_data.synbench.schemas.tasks import (
    EvaluationCriteria,
    RewardType,
    Task,
    UserScenario,
)
from aieng.syn_data.synbench.schemas.tools import ToolSpec, ToolType
from aieng.syn_data.synbench.schemas.verification import (
    VerificationReport,
    VerifiedTask,
)


__all__ = [
    "Action",
    "DomainBundle",
    "DomainManifest",
    "EvaluationCriteria",
    "GenerationConfig",
    "RelatedEntitySpec",
    "RewardType",
    "Task",
    "ToolSpec",
    "ToolType",
    "UserScenario",
    "VerificationReport",
    "VerifiedTask",
    "normalize_action",
]
