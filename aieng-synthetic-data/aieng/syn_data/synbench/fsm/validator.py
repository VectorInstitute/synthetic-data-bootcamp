"""Check that a task's oracle actions follow its state-machine path."""

from __future__ import annotations

from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tools import ToolType


class FSMValidationError(Exception):
    """Raised when an action sequence violates the task type's FSM path."""


def _tool_tags(domain: DomainBundle) -> dict[str, str]:
    tags = {}
    for spec in domain.tools:
        tags[spec.name] = spec.fsm_tag or (
            "lookup" if spec.tool_type == ToolType.READ else "mutate"
        )
    return tags


def validate_actions_against_fsm(
    domain: DomainBundle,
    task_type: str,
    actions: list[Action],
) -> None:
    """Raise ``FSMValidationError`` if ``actions`` do not match ``task_type``."""
    sm = domain.state_machine.get("task_types", {})
    if task_type not in sm:
        raise FSMValidationError(f"Unknown task_type: {task_type}")

    cfg = sm[task_type]
    expected_path: list[str] = cfg.get("path", [])
    allow_write: bool = cfg.get("allow_write", True)
    tags = _tool_tags(domain)

    action_tags = [tags.get(a.name, "lookup") for a in actions]

    if len(action_tags) != len(expected_path):
        raise FSMValidationError(
            f"Action count {len(action_tags)} != expected path length {len(expected_path)}"
        )

    for i, (got, exp) in enumerate(zip(action_tags, expected_path, strict=True)):
        if got != exp:
            raise FSMValidationError(
                f"Step {i}: tool tag '{got}' does not match expected '{exp}'"
            )

    write_tools = {t.name for t in domain.tools if t.tool_type == ToolType.WRITE}
    has_write = any(a.name in write_tools for a in actions)

    if not allow_write and has_write:
        raise FSMValidationError(f"task_type '{task_type}' forbids write tools")

    if allow_write and not has_write and expected_path and "mutate" in expected_path:
        raise FSMValidationError(
            f"task_type '{task_type}' requires at least one write tool"
        )
