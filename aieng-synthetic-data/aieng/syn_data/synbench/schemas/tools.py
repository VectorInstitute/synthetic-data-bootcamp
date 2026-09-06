"""Schema describing the tools a domain exposes to the agent."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolType(StrEnum):
    """Whether a tool only reads state or also mutates it."""

    READ = "read"
    WRITE = "write"


class ToolSpec(BaseModel):
    """Name, description, and JSON-schema parameters for one domain tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    tool_type: ToolType = ToolType.READ
    fsm_tag: str | None = None
