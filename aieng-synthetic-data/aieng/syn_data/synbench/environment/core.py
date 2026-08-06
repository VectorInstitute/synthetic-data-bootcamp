"""Environment that executes agent tool calls against a copy of the domain DB."""

from __future__ import annotations

import copy
import importlib.util
from typing import Any

from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle


class ToolDispatchError(Exception):
    """Raised when a tool call cannot be resolved or executed."""


def _load_toolkit(domain: DomainBundle) -> Any:
    tools_path = domain.root / "tools.py"
    spec = importlib.util.spec_from_file_location("domain_tools", tools_path)
    if spec is None or spec.loader is None:
        raise ToolDispatchError(f"Cannot load tools: {tools_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ToolKit


class Environment:
    """Domain ToolKit bound to a private, mutable copy of the domain database."""

    def __init__(self, domain: DomainBundle, db: dict[str, Any] | None = None):
        self.domain = domain
        self.db = copy.deepcopy(db if db is not None else domain.db)
        self._toolkit = _load_toolkit(domain)(self.db)
        self._tool_names = {t.name for t in domain.tools}

    def dispatch(self, action: Action) -> Any:
        """Execute ``action`` against the ToolKit and return the tool result."""
        if action.name not in self._tool_names:
            raise ToolDispatchError(f"Unknown tool: {action.name}")
        method = getattr(self._toolkit, action.name, None)
        if method is None:
            raise ToolDispatchError(f"Tool not implemented: {action.name}")
        try:
            return method(**action.arguments)
        except TypeError as e:
            raise ToolDispatchError(f"Invalid arguments for {action.name}: {e}") from e
        except ValueError as e:
            raise ToolDispatchError(str(e)) from e

    def copy(self) -> Environment:
        """Return an independent environment starting from the current DB state."""
        return Environment(self.domain, self.db)


def replay_actions(
    domain: DomainBundle,
    actions: list[Action],
    *,
    strict: bool = True,
) -> Environment:
    """Replay actions on a fresh environment (τ-bench style).

    Used for gold targets (strict) and for offline scoring of agent
    trajectories (strict=False so failed tool calls are skipped, matching
    the agent loop which records the action but continues after errors).
    """
    env = Environment(domain)
    for action in actions:
        try:
            env.dispatch(action)
        except Exception:
            if strict:
                raise
    return env
