"""Check that a task's oracle uses write tools iff its task type allows them."""

from __future__ import annotations

from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tools import ToolType


class TaskTypeValidationError(Exception):
    """Raised when an action sequence violates the task type's write policy."""


def validate_task_type_actions(
    domain: DomainBundle,
    task_type: str,
    actions: list[Action],
) -> None:
    """Raise ``TaskTypeValidationError`` if write-tool usage is wrong for ``task_type``.

    Scoring compares the final DB, not the route taken to reach it, so this
    check only enforces ``allow_write`` from ``task_types.yaml``.
    """
    types = domain.task_types
    if task_type not in types:
        raise TaskTypeValidationError(f"Unknown task_type: {task_type}")

    cfg = types[task_type]
    allow_write: bool = cfg.get("allow_write", True)

    write_tools = {t.name for t in domain.tools if t.tool_type == ToolType.WRITE}
    has_write = any(a.name in write_tools for a in actions)

    if not allow_write and has_write:
        raise TaskTypeValidationError(f"task_type '{task_type}' forbids write tools")

    if allow_write and not has_write:
        raise TaskTypeValidationError(
            f"task_type '{task_type}' requires at least one write tool"
        )
