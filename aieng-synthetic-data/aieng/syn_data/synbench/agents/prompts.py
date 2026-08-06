"""System prompts for each agent role."""

from __future__ import annotations

from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


def agent_system_prompt(domain: DomainBundle, task: Task, extra: str = "") -> str:
    """System prompt for the tool-calling agent under test.

    Intentionally omits ``user_scenario.instructions`` and ``task.description``
    so the agent must infer the goal from the live customer messages only.
    ``domain.generation.agent_role`` sets the role line.
    """
    # No instructions / description here — agent must use the live conversation only.
    role = domain.generation.agent_role
    parts = [
        f"You are the {role} for this domain.",
        f"task_id: {task.id}",
        "",
        "## Policy",
        domain.policy[:4000],
        "",
        "Resolve the customer's messages using the provided tools and policy.",
        "Do not invent goals beyond what the customer says.",
        "When finished, reply to the customer clearly.",
    ]
    if extra:
        parts.extend(["", "## Plan from planner", extra])
    return "\n".join(parts)


def planner_system_prompt(domain: DomainBundle, task: Task) -> str:
    """System prompt for the planner role (plan only, no tool calls)."""
    return "\n".join(
        [
            "You are the planner. Produce a short numbered plan (3-5 steps) for the agent.",
            "Do not call tools. Output plain text only.",
            f"task_id: {task.id}",
            "",
            "## Policy (excerpt)",
            domain.policy[:2000],
            "",
            f"Task: {task.description}",
        ]
    )


def critic_system_prompt(domain: DomainBundle, task: Task) -> str:
    """System prompt for the critic role (approve or give revision notes)."""
    return "\n".join(
        [
            "You are the critic. Review the agent plan, tool trace, and draft reply.",
            "Reply with exactly APPROVE if acceptable, otherwise brief revision notes.",
            f"task_id: {task.id}",
            "",
            "## Policy (excerpt)",
            domain.policy[:1500],
        ]
    )


def _style_description(domain: DomainBundle, style_name: str) -> str:
    """Look up behavior text for ``style_name`` from ``user_simulator.yaml``."""
    if not style_name:
        return ""
    for item in (domain.user_simulator or {}).get("personality_styles") or []:
        if isinstance(item, dict) and item.get("name") == style_name:
            return str(item.get("description") or "").strip()
    return ""


def user_sim_system_prompt(domain: DomainBundle, task: Task) -> str:
    """Prompt for the user simulator (evaluator-side; sees clean instructions).

    Uses ``user_name`` + ``personality_style`` from the task. Style behavior
    text is resolved from the domain catalog when available.
    """
    us = domain.user_simulator or {}
    scenario = task.user_scenario
    # Name and style stay separate on the task; look up style text from YAML.
    user_name = scenario.user_name or us.get("persona") or "a customer"
    style_name = scenario.personality_style or ""
    style_desc = _style_description(domain, style_name)
    style_line = (
        f"Personality style: {style_name}" + (f" — {style_desc}" if style_desc else "")
        if style_name
        else "Personality style: neutral"
    )
    return "\n".join(
        [
            "You are the user simulator. Write the next customer message only.",
            "In the conversation below, the messages addressed to you are the support agent",
            "speaking, and your own past customer messages are shown as your replies.",
            "Never act as the agent: do not call tools, or offer to help.",
            "Output the customer's next message as plain text, with no name or role prefix.",
            "Stay in character: use the customer name and personality style below.",
            "Pursue the instructions as your true goal; style may add noise but do not abandon the goal.",
            "Your first message was already sent as initial_message (often incomplete).",
            "Reveal identifiers and other details from Instructions when the agent asks,",
            "or when progress clearly requires them — do not dump everything unprompted.",
            "If the conversation is complete, reply with exactly [[DONE]].",
            f"task_id: {task.id}",
            f"initial_message: {scenario.initial_message}",
            "",
            f"User name: {user_name}",
            style_line,
            f"Instructions: {scenario.instructions}",
        ]
    )
