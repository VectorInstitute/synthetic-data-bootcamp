"""Notebook-friendly display helpers for synbench objects.

Uses IPython rich display when available; otherwise prints Markdown text.
"""

from __future__ import annotations

import contextlib
import html
import json
from typing import Any

from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import Task


def _emit(markdown: str) -> None:
    try:
        from IPython.display import Markdown, display  # noqa: PLC0415

        display(Markdown(markdown))
    except ImportError:
        print(markdown)


def _emit_html(fragment: str) -> None:
    try:
        from IPython.display import HTML, display  # noqa: PLC0415

        display(HTML(fragment))
    except ImportError:
        print(fragment)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def show_task(task: Task, *, title: str | None = "Task") -> None:
    """Render a Task with scenario and evaluation criteria."""
    lines: list[str] = []
    if title:
        lines.append(f"### {title}")
    lines.extend(
        [
            f"**id:** `{task.id}`  ",
            f"**type:** `{task.task_type or '—'}`  ",
            f"**description:** {task.description or '—'}",
            "",
            "#### User scenario",
            f"- **user name:** {task.user_scenario.user_name or '—'}",
            f"- **personality style:** {task.user_scenario.personality_style or '—'}",
            f"- **instructions:** {task.user_scenario.instructions or '—'}",
            f"- **initial message:** {task.user_scenario.initial_message or '—'}",
            "",
            "#### Evaluation criteria",
        ]
    )
    basis = ", ".join(r.value for r in task.evaluation_criteria.reward_basis) or "—"
    lines.append(f"- **reward basis:** {basis}")
    lines.append(
        f"- **communicate info:** {task.evaluation_criteria.communicate_info or []}"
    )
    lines.append("- **oracle actions:**")
    actions = task.evaluation_criteria.actions
    if actions:
        for a in actions:
            lines.append(f"  - `{a.name}` `{json.dumps(a.arguments, default=str)}`")
    else:
        lines.append("  - *(none)*")
    _emit("\n".join(lines))


# ---------------------------------------------------------------------------
# Prompts / chat
# ---------------------------------------------------------------------------


def show_prompt(text: str, *, title: str = "Prompt") -> None:
    """Render a multi-line prompt as a fenced Markdown block."""
    _emit(f"### {title}\n\n{_fence(text)}")


def show_message(content: str, *, role: str = "user", title: str | None = None) -> None:
    """Render a single chat message."""
    heading = title or f"{role.capitalize()} message"
    _emit(f"### {heading}\n\n> {content.replace(chr(10), chr(10) + '> ')}")


def show_messages(messages: list[dict[str, Any]], *, title: str = "Messages") -> None:
    """Render a chat transcript with role-colored cards."""
    role_colors = {
        "system": ("#5b6b7c", "#eef1f4"),
        "user": ("#1f4b7a", "#e8f1fb"),
        "assistant": ("#1f5c3a", "#e8f7ee"),
        "tool": ("#6b4f1f", "#fbf3e2"),
    }
    cards: list[str] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, default=str)
        # Pretty-print JSON tool payloads when possible
        if role == "tool":
            with contextlib.suppress(TypeError, json.JSONDecodeError):
                content = json.dumps(json.loads(content), indent=2, default=str)
        fg, bg = role_colors.get(role, ("#333", "#f5f5f5"))
        cards.append(
            f"""
<div style="border-left:4px solid {fg}; background:{bg}; padding:10px 12px;
            margin:8px 0; border-radius:4px; font-family:ui-sans-serif,system-ui,sans-serif;">
  <div style="font-size:12px; font-weight:600; color:{fg}; margin-bottom:6px;">
    [{i}] {_esc(role)}
  </div>
  <pre style="margin:0; white-space:pre-wrap; word-break:break-word;
              font-size:13px; line-height:1.4;">{_esc(content)}</pre>
</div>"""
        )
    body = "".join(cards) if cards else "<em>(no messages)</em>"
    _emit_html(
        f"<div><h3 style='margin:0 0 8px; font-family:ui-sans-serif,system-ui,sans-serif;'>"
        f"{_esc(title)}</h3>{body}</div>"
    )


_ROLE_LABELS = {
    "system": "system",
    "user": "user",
    "assistant": "assistant (executor)",
    "tool": "tool",
}


def _format_message_body(msg: dict[str, Any]) -> str:
    """Format one chat message body for plain Markdown display."""
    role = str(msg.get("role", "?"))
    parts: list[str] = []

    content = msg.get("content")
    if content not in (None, ""):
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, default=str)
        elif role == "tool":
            with contextlib.suppress(TypeError, json.JSONDecodeError):
                content = json.dumps(json.loads(content), indent=2, default=str)
        parts.append(content)

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "?")
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                with contextlib.suppress(TypeError, json.JSONDecodeError):
                    args = json.dumps(json.loads(args), indent=2, default=str)
            else:
                args = json.dumps(args, indent=2, default=str)
            parts.append(f"tool_call `{name}`\n{args}")

    if not parts:
        return "*(empty)*"
    return "\n\n".join(parts)


def show_pipeline_messages(
    session: AgentSession | list[dict[str, Any]],
    *,
    title: str = "Pipeline messages",
) -> None:
    """Render all session messages by role in plain Markdown.

    Accepts an ``AgentSession`` or a raw message list. Shows every stored chat
    turn (system / user / assistant / tool) without colored cards. When given a
    session, also shows ``role_trace`` and the latest planner ``plan``.
    """
    messages: list[dict[str, Any]]
    role_trace: list[str] | None = None
    plan: str | None = None

    if isinstance(session, AgentSession):
        messages = session.messages
        role_trace = session.role_trace or None
        plan = session.plan or None
    else:
        messages = session

    lines: list[str] = [f"### {title}", ""]

    if role_trace:
        lines.append(f"**role_trace:** `{' → '.join(role_trace)}`")
        lines.append("")
    if plan:
        lines.append("**plan (planner):**")
        lines.append(_fence(plan))
        lines.append("")

    if not messages:
        lines.append("*(no messages)*")
        _emit("\n".join(lines))
        return

    for i, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        label = _ROLE_LABELS.get(role, role)
        lines.append(f"**[{i}] {label}**")
        body = _format_message_body(msg)
        # Indent body under the role heading for readability
        for line in body.splitlines() or ["*(empty)*"]:
            lines.append(f"    {line}" if line else "")
        lines.append("")

    _emit("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def show_actions(actions: list[Action], *, title: str = "Actions") -> None:
    """Render a list of tool-call actions."""
    lines = [f"### {title}"]
    if not actions:
        lines.append("*(none)*")
    else:
        for i, a in enumerate(actions):
            args = json.dumps(a.arguments, default=str)
            lines.append(f"{i + 1}. `{a.name}` — `{args}`")
    _emit("\n".join(lines))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def show_session(
    session: AgentSession,
    *,
    title: str = "AgentSession",
    show_chat: bool = True,
) -> None:
    """Summarize an AgentSession without dumping the full domain bundle."""
    domain = session.domain
    task = session.task
    env = session.env
    env_label = "ready" if env is not None else "not created"

    lines = [
        f"### {title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| **domain** | `{domain.manifest.name}` (`{domain.root}`) |",
        f"| **task** | `{task.id}` · `{task.task_type}` |",
        f"| **env** | {env_label} |",
        f"| **messages** | {len(session.messages)} |",
        f"| **agent_actions** | {len(session.agent_actions)} |",
        f"| **agent_messages** | {len(session.agent_messages)} |",
        f"| **role_trace** | {session.role_trace or '[]'} |",
        f"| **plan** | {session.plan or '—'} |",
    ]
    _emit("\n".join(lines))

    if session.agent_actions:
        show_actions(session.agent_actions, title="Agent actions")
    if session.agent_messages:
        quoted = []
        for m in session.agent_messages:
            quoted.append(f"> {m.replace(chr(10), chr(10) + '> ')}")
        _emit("### Agent replies\n\n" + "\n\n".join(quoted))
    if show_chat and session.messages:
        show_messages(session.messages, title="Chat transcript")
