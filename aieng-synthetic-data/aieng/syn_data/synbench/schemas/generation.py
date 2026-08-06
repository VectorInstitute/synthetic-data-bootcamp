"""Domain-owned config for task sampling and LLM task-generation prompts.

Loaded from each domain's ``generation.yaml`` into ``DomainBundle.generation``.
This is the main extension point when adding a non-retail domain: point the
sampler at your DB collections/fields and customize prompt wording without
editing library code.

Consumed by
-----------
- ``ConstraintSampler`` — picks a primary entity and related IDs from ``db.json``
- ``PromptBuilder`` — fills the generation prompt (role, entity snapshot, hints)
- ``agent_system_prompt`` — uses ``agent_role`` for the agent-under-test persona
- ``_validate_generation_readiness`` — checks config against ``db.json`` / FSM

See ``domains/mock_retail/generation.yaml`` for a fully commented example.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RelatedEntitySpec(BaseModel):
    """How to resolve a related row from the sampled primary record.

    Example (retail): primary = an order; related ``user_id`` looks up
    ``db["users"][order["user_id"]]`` so the prompt can include the customer name.

    Customize per domain by changing ``collection``, ``via``, and which
    ``fields`` to surface in the generation prompt's entity context.
    """

    collection: str = Field(
        description=(
            'Top-level key in db.json for the related table (e.g. "users", "accounts").'
        ),
    )
    via: str = Field(
        description=(
            "Field on the *primary* record whose value keys into "
            'db[collection] (e.g. order\'s "user_id").'
        ),
    )
    fields: list[str] = Field(
        default_factory=list,
        description=(
            "Related-record fields to include under entity_context[collection]. "
            "Empty = include all keys. Prompt-only; unused by verify/score."
        ),
    )


class GenerationConfig(BaseModel):
    """Per-domain settings that drive constraint sampling and task-generation prompts.

    Author these in ``domains/<name>/generation.yaml``. Required fields are
    ``primary_collection`` and ``id_field``; everything else is optional
    customization for prompts and richer entity snapshots.
    """

    agent_role: str = Field(
        default="support agent",
        description=(
            "Role label for the agent under test. Injected into the task-generation "
            'prompt and the runtime agent prompt ("You are the … for this domain."). '
            'Customize per domain, e.g. "airline support agent".'
        ),
    )
    primary_collection: str = Field(
        description=(
            "Top-level key in db.json from which ConstraintSampler picks a random "
            'record (e.g. "orders", "tickets").'
        ),
    )
    id_field: str = Field(
        description=(
            "Unique-id field on each primary record. Becomes the main key in "
            "SampleConstraints.entities and is what the generation prompt tells "
            'the LLM to use in tool args / user messages (e.g. "order_id").'
        ),
    )
    related: dict[str, RelatedEntitySpec] = Field(
        default_factory=dict,
        description=(
            "Named FK joins from the primary record. Dict keys become extra "
            "SampleConstraints.entities entries; values say where to look them up "
            "for prompt context. Omit if the primary record is self-contained."
        ),
    )
    context_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Primary-record fields copied into entity_context for the generation "
            "LLM (e.g. status so it only cancels pending orders). Empty = all "
            "fields. Prompt-only; unused by verify/score."
        ),
    )
    communicate_hints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-task_type guidance for evaluation_criteria.communicate_info. "
            "Keys should match state_machine.yaml task_types; missing keys fall back "
            "to a generic PromptBuilder default."
        ),
    )
    persona_related: str | None = Field(
        default=None,
        description=(
            "Optional key into `related` whose row supplies user_scenario.user_name "
            '(e.g. "user_id" + persona_field "name"). Leave null if names are '
            "not tied to a related table."
        ),
    )
    persona_field: str = Field(
        default="name",
        description=(
            "Field on the related record named by persona_related to use as user_name. "
            "Ignored when persona_related is null."
        ),
    )
