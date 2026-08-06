"""Scripted LLM client that replays fixtures so CI can run without a provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aieng.syn_data.synbench.llm.client import LLMResponse
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import Task


FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "agent_traces"
AGENT_TRACES = FIXTURES
LLM_RESPONSES = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm_responses"
)


class MockLLMClient:
    """Scripted multi-turn responses for CI; keyed by session context."""

    def __init__(self) -> None:
        self._sessions: dict[str, int] = {}
        self._task_oracles: dict[str, list[dict[str, Any]]] = {}

    def register_task(self, task_id: str, turns: list[dict[str, Any]]) -> None:
        """Set the scripted turns for ``task_id`` and reset its turn counters."""
        self._task_oracles[task_id] = turns
        # Reset per-task turn counters so multi-task eval runs stay deterministic.
        prefix = f"{task_id}:"
        for key in list(self._sessions):
            if key.startswith(prefix):
                del self._sessions[key]

    def ensure_task(self, task_id: str, turns: list[dict[str, Any]]) -> None:
        """Register oracle turns only once so dialogue re-entry does not reset."""
        if task_id not in self._task_oracles:
            self.register_task(task_id, turns)

    @staticmethod
    def turns_from_task(task: Any) -> list[dict[str, Any]]:
        """Derive scripted turns from a task's oracle actions and required info."""
        t = task if isinstance(task, Task) else Task.model_validate(task)
        turns: list[dict[str, Any]] = []
        for a in t.evaluation_criteria.actions:
            turns.append({"tool_calls": [{"name": a.name, "arguments": a.arguments}]})
        comm = t.evaluation_criteria.communicate_info
        turns.append({"content": " ".join(comm) if comm else "Done."})
        return turns

    def _session_key(self, messages: list[dict[str, Any]]) -> str:
        for m in messages:
            if m.get("role") == "system" and "task_id:" in (m.get("content") or ""):
                for line in m["content"].splitlines():
                    if line.startswith("task_id:"):
                        return str(line.split(":", 1)[1].strip())
        return "default"

    def _role_from_messages(self, messages: list[dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "system":
                content = m.get("content") or ""
                if "You are the user simulator" in content:
                    return "user_sim"
                if "You are the planner" in content:
                    return "planner"
                if "You are the critic" in content:
                    return "critic"
                # Match agent_system_prompt role line for any domain agent_role.
                if "for this domain." in content and "user simulator" not in content:
                    return "executor"
        return "executor"

    def _turn_index(self, session_key: str, role: str) -> int:
        key = f"{session_key}:{role}"
        idx = self._sessions.get(key, 0)
        self._sessions[key] = idx + 1
        return idx

    def _load_trace(self, task_id: str) -> dict[str, Any] | None:
        path = AGENT_TRACES / f"{task_id}.json"
        if path.exists():
            with open(path) as f:
                data: dict[str, Any] = json.load(f)
            return data
        return None

    def _oracle_turns(self, task_id: str) -> list[dict[str, Any]]:
        """Build turns from task oracle when no fixture exists."""
        root = Path(__file__).resolve().parents[3]
        tasks_path = root / "domains" / "mock_retail" / "tasks.seed.json"
        if tasks_path.exists():
            with open(tasks_path) as f:
                data = json.load(f)
            for item in data.get("tasks", data):
                if item.get("id") == task_id:
                    task = Task.model_validate(item)
                    turns: list[dict[str, Any]] = []
                    for a in task.evaluation_criteria.actions:
                        turns.append(
                            {"tool_calls": [{"name": a.name, "arguments": a.arguments}]}
                        )
                    comm = task.evaluation_criteria.communicate_info
                    text = " ".join(comm) if comm else "Done."
                    turns.append({"content": text})
                    return turns
        return []

    def _get_turns(self, session_key: str) -> list[dict[str, Any]]:
        if session_key in self._task_oracles:
            return self._task_oracles[session_key]
        trace = self._load_trace(session_key)
        if trace and "turns" in trace:
            recorded: list[dict[str, Any]] = trace["turns"]
            return recorded
        if session_key != "default":
            oracle = self._oracle_turns(session_key)
            if oracle:
                return oracle
        return []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return the scripted response for the role and turn in ``messages``."""
        if json_mode:
            path = LLM_RESPONSES / "draft_cancel.json"
            with open(path) as f:
                return LLMResponse(content=json.dumps(json.load(f)))

        session_key = self._session_key(messages)
        role = self._role_from_messages(messages)
        turn_idx = self._turn_index(session_key, role)

        if role == "user_sim":
            return self._user_sim_response(messages, turn_idx)
        if role == "planner":
            return LLMResponse(
                content="1. Look up the order\n2. Apply the correct tool\n3. Reply to the customer"
            )
        if role == "critic":
            return LLMResponse(content="APPROVE")
        return self._executor_response(session_key, turn_idx)

    def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the recorded draft-task fixture as a decoded object."""
        response = self.complete(messages, json_mode=True)
        payload: dict[str, Any] = json.loads(response.content)
        return payload

    def _executor_response(self, session_key: str, turn_idx: int) -> LLMResponse:
        """Replay the executor's scripted turn, or close out the conversation."""
        turns = self._get_turns(session_key)
        if turn_idx >= len(turns):
            return LLMResponse(content="Done.")

        step = turns[turn_idx]
        if "tool_calls" in step:
            actions = [
                Action(name=tc["name"], arguments=tc.get("arguments", {}))
                for tc in step["tool_calls"]
            ]
            return LLMResponse(content="", tool_calls=actions)
        return LLMResponse(content=step.get("content", "Done."))

    def _user_sim_response(
        self, messages: list[dict[str, Any]], turn_idx: int
    ) -> LLMResponse:
        # Dialogue helper seeds initial_message; after an agent reply, end by default.
        if (
            any(m.get("role") == "assistant" and m.get("content") for m in messages)
            or turn_idx > 0
        ):
            return LLMResponse(content="[[DONE]]")
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                return LLMResponse(content=m["content"])
        for m in messages:
            if m.get("role") == "system" and "initial_message:" in (
                m.get("content") or ""
            ):
                for line in m["content"].splitlines():
                    if line.startswith("initial_message:"):
                        return LLMResponse(content=line.split(":", 1)[1].strip())
        return LLMResponse(content="[[DONE]]")
