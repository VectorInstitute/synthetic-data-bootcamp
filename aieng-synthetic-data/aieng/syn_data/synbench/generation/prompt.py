"""Build the LLM prompt that asks for one synthetic ``Task`` JSON object.

All domain-specific wording (agent role, entity ids, communicate hints, DB
snapshot) comes from ``SampleConstraints`` + ``domain.generation``. Prefer
editing ``generation.yaml`` over changing this module when adding a domain.
"""

from __future__ import annotations

import json

from aieng.syn_data.synbench.generation.sampler import SampleConstraints
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


def _seed_examples(domain: DomainBundle, task_type: str, limit: int = 2) -> list[Task]:
    """Pick few-shot seed tasks for the generation prompt.

    Prefers seeds whose ``task_type`` matches the sampled constraints, then
    fills with other seeds up to ``limit``. Seeds live in ``tasks.seed.json``.
    """
    matching = [t for t in domain.seed_tasks if t.task_type == task_type]
    others = [t for t in domain.seed_tasks if t.task_type != task_type]
    ordered = matching + others
    return ordered[:limit]


def _fsm_rules(domain: DomainBundle, task_type: str) -> str:
    """Human-readable FSM / oracle trajectory rules for the prompt.

    Derived from ``state_machine.yaml`` for ``task_type`` (path length, tag
    order, ``allow_write``). The verifier enforces the same rules later.
    """
    cfg = domain.state_machine.get("task_types", {}).get(task_type, {})
    path = cfg.get("path", [])
    allow_write = cfg.get("allow_write", True)
    description = cfg.get("description", "")
    n = len(path)
    path_str = " → ".join(path) if path else "(none)"
    write_rule = (
        "Include exactly one write tool as the final action."
        if allow_write and n > 1
        else "Use read-only tools only; do not call write tools."
    )
    return (
        f"- task_type `{task_type}`: {description}\n"
        f"- Oracle must contain exactly {n} action(s) in order: {path_str}\n"
        f"- {write_rule}"
    )


def _communicate_guidance(domain: DomainBundle, task_type: str) -> str:
    """Lookup ``generation.communicate_hints[task_type]``, or a generic default.

    Customize hints in ``generation.yaml`` so each domain can steer
    ``communicate_info`` without hardcoding task types in this file.
    """
    hints = domain.generation.communicate_hints
    return hints.get(
        task_type,
        "Short substrings the agent must mention; use [] if none.",
    )


def _customer_name_hint(domain: DomainBundle, constraints: SampleConstraints) -> str:
    """Concrete customer name for ``user_scenario.user_name``.

    If ``generation.persona_related`` / ``persona_field`` are set and the
    related row is present in ``entity_context``, return that name; otherwise
    a generic fallback.
    """
    cfg = domain.generation
    if not cfg.persona_related or cfg.persona_related not in cfg.related:
        return "a natural customer name for this domain"
    collection = cfg.related[cfg.persona_related].collection
    related = constraints.entity_context.get(collection, {})
    name = related.get(cfg.persona_field)
    if name:
        return f'"{name}"'
    return f"{cfg.persona_field} from related `{cfg.persona_related}` when available"


def _personality_style_block(constraints: SampleConstraints) -> str:
    """Format sampled interaction style for the generation prompt, if any."""
    style = constraints.personality_style
    if not style:
        return (
            "No personality style was sampled. Leave personality_style empty "
            "and use a neutral tone in initial_message."
        )
    name = style.get("name", "")
    description = style.get("description", "")
    return (
        f"- personality_style (must copy exactly): {name}\n"
        f"- behavior guidance for initial_message: {description}\n"
        "Set user_scenario.personality_style to the style name only. "
        "Do not put style into user_name or instructions."
    )


def _entity_constraints_block(constraints: SampleConstraints) -> str:
    """Format ``task_type`` + ``entities`` as bullet lines for the prompt.

    These values are hard constraints the draft JSON must reuse (ids, etc.).
    """
    lines = [f"- task_type: {constraints.task_type}"]
    for key, value in constraints.entities.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _minimal_output_template(domain: DomainBundle) -> str:
    """JSON skeleton shown in the prompt; ``id_field`` is domain-specific.

    Keeps the required Task shape stable while substituting the domain's
    primary id field name into example arguments / user-message hints.
    """
    id_field = domain.generation.id_field
    return json.dumps(
        {
            "description": (
                "High-level goal of the evaluation task; matches "
                "Constraints.task_type; no user-specific ids or args"
            ),
            "task_type": "<must match Constraints.task_type>",
            "user_scenario": {
                "user_name": "<customer name from entity context>",
                "personality_style": "<sampled style name, e.g. rushed>",
                "instructions": (
                    f"Simulator brief: specific task including {id_field} "
                    "and any other required args (no style)"
                ),
                "initial_message": (
                    "Incomplete first utterance in personality style; "
                    f"withhold {id_field} and other slots the agent should ask for"
                ),
            },
            "evaluation_criteria": {
                "actions": [
                    {
                        "name": "tool_name",
                        "arguments": {id_field: f"example_{id_field}"},
                    },
                ],
                "communicate_info": [],
                "reward_basis": ["DB", "COMMUNICATE"],
            },
        },
        indent=2,
    )


class PromptBuilder:
    """Assemble a single user prompt that asks the LLM for one ``Task`` as JSON.

    Domain customization knobs (all from config / sampler, not hardcoded here):

    - ``generation.agent_role``
    - sampled ``entities`` / ``primary_id``
    - ``entity_context`` (DB snapshot)
    - ``communicate_hints`` / FSM rules / seed examples
    """

    def build(
        self,
        domain: DomainBundle,
        constraints: SampleConstraints,
    ) -> str:
        """Return the full generation prompt string for ``constraints``.

        Parameters
        ----------
        domain:
            Loaded domain (policy, tools, seeds, ``generation`` config).
        constraints:
            Output of ``ConstraintSampler.sample()`` for this draft.
        """
        cfg = domain.generation
        tools_json = json.dumps(
            [t.model_dump() for t in domain.tools],
            indent=2,
        )
        seeds = json.dumps(
            [
                t.model_dump(exclude_none=True)
                for t in _seed_examples(domain, constraints.task_type)
            ],
            indent=2,
        )
        # Prompt-only snapshot; see SampleConstraints.entity_context
        entity_ctx = json.dumps(constraints.entity_context, indent=2)
        fsm_rules = _fsm_rules(domain, constraints.task_type)
        comm_guidance = _communicate_guidance(domain, constraints.task_type)
        # Name from DB; style from user_simulator.yaml — kept separate in the Task.
        name_hint = _customer_name_hint(domain, constraints)
        style_block = _personality_style_block(constraints)
        entity_block = _entity_constraints_block(constraints)
        primary_id = constraints.primary_id or constraints.entities.get(
            cfg.id_field, ""
        )

        return f"""Generate one synthetic benchmark task for domain '{domain.manifest.name}'.
The agent under test is a {cfg.agent_role}.

Return a single JSON object. Match the shape of the seed tasks exactly — no extra fields.

## Policy
{domain.policy[:1500]}

## Available tools
{tools_json}

## Sampled constraints (must use these values)
{entity_block}

## Entity context (from domain DB)
{entity_ctx}

## Sampled customer interaction style
{style_block}

## FSM / oracle trajectory rules
{fsm_rules}

## Example seed task(s)
{seeds}

## Required JSON shape
Use this skeleton. Do not add additional fields. For example, do not add action_id.

{_minimal_output_template(domain)}

## Field rules
- description: high-level description of the **evaluation task goal**. Must match
  Constraints.task_type (and policy). Do **not** include user-specific details such
  as {cfg.id_field}, names, addresses, or other instance args — those belong only
  in user_scenario.instructions (and oracle arguments).
- user_scenario.user_name: customer name only — use {name_hint}. Do not append style text.
- user_scenario.personality_style: copy the sampled style **name** exactly (e.g. rushed).
  Empty string only if no style was sampled.
- user_scenario.instructions: brief that guides the **user simulator LLM** toward this
  specific task instance. Include {cfg.id_field} {primary_id} and any other args the
  oracle needs (addresses, etc.). Only instructions (not description) carries these
  user-specific details. No personality theatrics. Must be consistent with
  description/task_type and the oracle actions. Not shown to the agent under test.
- user_scenario.initial_message: first customer utterance written **in the sampled
  personality style**. Prefer **partial disclosure** so multi-turn dialogue is needed:
  state the goal class clearly enough to start (e.g. cancel / status / address change),
  but **omit** {cfg.id_field} {primary_id} and other critical slots that live only in
  instructions — the simulator reveals them when the agent asks. May be messy or
  mention side topics. Exception: for policy-stress tasks (e.g. refuse_*), the ask
  itself may be blunt; still prefer withholding ids/args when that does not muddy
  the refusal scenario.
- evaluation_criteria.actions: oracle tool calls for the **specific task** grounded in
  instructions (ids/args) — ignore omissions and style distractions in initial_message.
  Each action is {{"name": "<tool>", "arguments": {{...}}}} only — no action_id.
  Use {cfg.id_field} {primary_id} in arguments where applicable. Respect entity state and policy.
- evaluation_criteria.communicate_info: {comm_guidance}
  Substring checks only; lowercase matching is applied.
- evaluation_criteria.reward_basis: always ["DB", "COMMUNICATE"].

## Do not include
- Structured {{"contains": ...}} objects anywhere — put substrings in communicate_info.
- Write tools for read-only task types (allow_write: false).
- Legacy ``persona`` field — use user_name + personality_style instead.

Output only the JSON object, no markdown fences or commentary.
"""
