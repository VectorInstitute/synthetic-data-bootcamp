"""Shared multi-turn user dialogue for single- and multi-agent runners.

Contract
--------
1. Dialogue turn 0 uses ``task.user_scenario.initial_message`` directly (no
   user-sim rewrite of the first utterance).
2. Later turns call ``UserSimulator.respond`` until ``[[DONE]]`` or
   ``max_dialogue_turns``.
3. The helper owns appending user messages; the executor is called with
   ``initial_user_message=""`` so it does not re-seed the transcript.
"""

from __future__ import annotations

from aieng.syn_data.synbench.agents.critic import Critic
from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.planner import Planner
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.user_sim import UserSimulator
from aieng.syn_data.synbench.schemas.tasks import Task


def run_user_dialogue(
    session: AgentSession,
    task: Task,
    *,
    executor: ToolCallingLoop,
    user_sim: UserSimulator | None = None,
    planner: Planner | None = None,
    critic: Critic | None = None,
    roles: list[str] | None = None,
    max_dialogue_turns: int = 5,
) -> AgentSession:
    """Run up to ``max_dialogue_turns`` of user ↔ agent interaction.

    Parameters
    ----------
    roles:
        Subset of ``user_sim``, ``planner``, ``executor``, ``critic``.
        Defaults to all four when ``None``.
    """
    active = roles or ["user_sim", "planner", "executor", "critic"]
    last_agent_reply = ""
    # Turn 0 starts from the task's authored (styled) first message.
    user_message = task.user_scenario.initial_message

    for dialogue_turn in range(max_dialogue_turns):
        if dialogue_turn == 0:
            if not user_message:
                break
        else:
            # Later turns: simulated customer follow-ups until [[DONE]].
            if "user_sim" not in active or user_sim is None:
                break
            session.role_trace.append("user_sim")
            user_message = user_sim.respond(session, last_agent_reply)
            if user_sim.is_done(user_message) or not user_message.strip():
                break

        session.messages.append({"role": "user", "content": user_message})

        # Optional planner → executor (+ optional critic retry).
        plan = ""
        if "planner" in active and planner is not None:
            session.role_trace.append("planner")
            plan = planner.plan(session, user_message)
            session.plan = plan

        if "executor" in active:
            session.role_trace.append("executor")
            session = _run_executor_turn(
                session,
                task,
                executor=executor,
                plan=plan,
                critic=critic if "critic" in active else None,
            )
            last_agent_reply = (
                session.agent_messages[-1] if session.agent_messages else ""
            )

        # Without user_sim, one utterance is enough.
        if "user_sim" not in active:
            break

    return session


def _run_executor_turn(
    session: AgentSession,
    task: Task,
    *,
    executor: ToolCallingLoop,
    plan: str,
    critic: Critic | None,
) -> AgentSession:
    """Execute one user utterance; optionally critic-retry with session rollback."""
    # Snapshot so a critic reject can rewind a failed attempt.
    snap_messages = len(session.messages)
    snap_actions = len(session.agent_actions)
    snap_agent_msgs = len(session.agent_messages)
    snap_session_env = session.env.copy() if session.env is not None else None

    critic_notes = ""
    max_attempts = 2 if critic is not None else 1

    for attempt in range(max_attempts):
        if attempt > 0:
            # Rollback to the previous state
            session.messages = session.messages[:snap_messages]
            session.agent_actions = session.agent_actions[:snap_actions]
            session.agent_messages = session.agent_messages[:snap_agent_msgs]
            session.env = snap_session_env

        extra = plan
        if critic_notes:
            extra = f"{plan}\n\nCritic revision:\n{critic_notes}"

        # "" = dialogue already appended the user message; do not re-seed.
        session = executor.run(
            task,
            session=session,
            plan=extra,
            initial_user_message="",
        )
        if critic is None:
            break

        session.role_trace.append("critic")
        approved, critic_notes = critic.review(session)
        if approved:
            break

    return session
