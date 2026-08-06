"""Schema for a loaded domain bundle and its manifest."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aieng.syn_data.synbench.schemas.generation import GenerationConfig
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.schemas.tools import ToolSpec


class DomainManifest(BaseModel):
    """Identifying metadata from a domain's ``manifest.json``."""

    name: str
    version: str = "0.1.0"
    description: str = ""


class DomainBundle(BaseModel):
    """Loaded domain artifacts from a domain directory.

    ``generation`` comes from ``generation.yaml`` and configures how task
    sampling / generation prompts bind to ``db`` (see ``GenerationConfig``).
    """

    root: Path
    manifest: DomainManifest
    policy: str
    db: dict[str, Any]
    tools: list[ToolSpec]
    state_machine: dict[str, Any]
    user_simulator: dict[str, Any]
    seed_tasks: list[Task]
    generation: GenerationConfig
    """From ``generation.yaml``: primary collection, id field, related joins,
    communicate hints, agent_role, etc. Used by sampler, PromptBuilder, and
    agent prompts — customize per domain without editing library code."""

    model_config = {"arbitrary_types_allowed": True}
