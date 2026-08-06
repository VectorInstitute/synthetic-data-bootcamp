"""Sample task-type + entity IDs for LLM task generation.

Uses ``DomainBundle.generation`` (from ``generation.yaml``) so domains with
different DB shapes do not require changes here — only config.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from aieng.syn_data.synbench.schemas.domain import DomainBundle


@dataclass
class SampleConstraints:
    """One draw of generation constraints for ``PromptBuilder``.

    Produced by ``ConstraintSampler.sample()``. Not persisted; only feeds the
    generation prompt. Verification/scoring use the resulting ``Task``, not
    this object.

    Attributes
    ----------
    task_type:
        Key from ``state_machine.yaml`` → ``task_types`` (e.g. ``cancel``).
    entities:
        String IDs the draft task must use. Always includes
        ``generation.id_field`` → primary id; plus one entry per key in
        ``generation.related`` (e.g. ``{"order_id": "ord_1001",
        "user_id": "user_alice"}``).
    fsm_path:
        Expected oracle FSM tag sequence for ``task_type`` (copied from the
        state machine for prompt wording; validator uses the state machine
        directly when verifying).
    primary_id:
        Convenience copy of ``entities[id_field]`` for prompt templates.
    entity_context:
        Prompt-only JSON-ish snapshot of the sampled primary row (and related
        rows). Built from ``context_fields`` / ``related.*.fields``. Lets the
        generation LLM see status, addresses, names, etc. Unused by verify
        or score.
    personality_style:
        Optional ``{"name", "description"}`` drawn from
        ``user_simulator.yaml`` → ``personality_styles``. The generation
        prompt copies ``name`` into ``user_scenario.personality_style`` and
        uses ``description`` to style ``initial_message`` only.
    """

    task_type: str
    entities: dict[str, str]
    fsm_path: list[str]
    primary_id: str = ""
    entity_context: dict[str, Any] = field(default_factory=dict)
    personality_style: dict[str, str] | None = None


class ConstraintSampler:
    """Randomly sample a ``task_type`` and a primary DB entity for generation.

    Sampling policy is entirely driven by ``domain.generation``:

    1. Choose a task type uniformly from ``state_machine["task_types"]``.
    2. Choose a record uniformly from ``db[primary_collection]``.
        Primary collection is the collection that is used to generate the task.
        It is specified in the ``generation.yaml`` file.
    3. Build ``entities`` from ``id_field`` + ``related`` joins.
        id_field is the field that is used to identify the primary record.
        related is the list of related collections that are used to generate the task.
        It is specified in the ``generation.yaml`` file.
    4. Build ``entity_context`` for the generation prompt.

    Parameters
    ----------
    domain:
        Loaded domain bundle (must include ``generation``).
    seed:
        RNG seed for reproducible samples (CLI ``--seed``, notebooks).
    """

    def __init__(self, domain: DomainBundle, seed: int = 42):
        self.domain = domain
        self.rng = random.Random(seed)
        # Shorthand for domains/<name>/generation.yaml
        self.cfg = domain.generation

    def sample(self) -> SampleConstraints:
        """Draw one set of constraints for a single draft task.

        Raises
        ------
        ValueError
            If there are no task types, or the primary collection is empty /
            missing (``validate_domain`` should catch this earlier).
        """
        task_types = list(self.domain.state_machine.get("task_types", {}).keys())
        if not task_types:
            raise ValueError("state_machine.yaml has no task_types to sample")
        task_type = self.rng.choice(task_types)
        # FSM path / allow_write for this task type (prompt + later verify)
        cfg = self.domain.state_machine["task_types"][task_type]

        collection = self.domain.db.get(self.cfg.primary_collection) or {}
        records = list(collection.values())
        if not records:
            raise ValueError(
                f"Cannot sample: db['{self.cfg.primary_collection}'] is empty"
            )
        record = self.rng.choice(records)
        primary_id = str(record[self.cfg.id_field])

        # IDs the LLM must reuse in the draft Task (tools + user message)
        entities: dict[str, str] = {self.cfg.id_field: primary_id}
        for key, rel in self.cfg.related.items():
            entities[key] = str(record.get(rel.via, ""))

        # Prompt-only DB snapshot (status, customer name, …).
        entity_context = _build_entity_context(self.domain, record)
        # Style is sampled separately from the order/user context.
        return SampleConstraints(
            task_type=task_type,
            entities=entities,
            fsm_path=cfg.get("path", []),
            primary_id=primary_id,
            entity_context=entity_context,
            personality_style=_sample_personality_style(
                self.domain.user_simulator or {}, self.rng
            ),
        )


def _sample_personality_style(
    user_simulator: dict[str, Any], rng: random.Random
) -> dict[str, str] | None:
    """Uniformly pick one ``personality_styles`` entry, or ``None`` if absent.

    Each entry should provide ``name`` and ``description`` strings. Invalid
    or empty entries are skipped so domains without styles stay compatible.
    """
    styles = user_simulator.get("personality_styles") or []
    valid: list[dict[str, str]] = []
    for item in styles:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name and description:
            valid.append({"name": name, "description": description})
    if not valid:
        return None
    return dict(rng.choice(valid))


def _build_entity_context(
    domain: DomainBundle, record: dict[str, Any]
) -> dict[str, Any]:
    """Build the prompt-only DB snapshot for a sampled primary ``record``.

    Structure
    ---------
    - Top-level keys from ``generation.context_fields`` (or all primary keys
      if that list is empty).
    - For each ``generation.related`` entry: a nested dict under the related
      ``collection`` name with the configured ``fields`` (or all fields).
    - Related entity key (e.g. ``user_id``) is also set at the top level when
      not already present from ``context_fields``.

    Customize via ``generation.yaml`` (``context_fields``, ``related.*.fields``);
    this helper should stay domain-agnostic.
    """
    cfg = domain.generation
    ctx: dict[str, Any] = {}
    fields = cfg.context_fields or list(record.keys())
    for name in fields:
        if name in record:
            ctx[name] = record[name]

    for key, rel in cfg.related.items():
        via_val = record.get(rel.via)
        related_rec = (domain.db.get(rel.collection) or {}).get(via_val, {})
        if not related_rec:
            continue
        rel_fields = rel.fields or list(related_rec.keys())
        ctx[rel.collection] = {
            f: related_rec[f] for f in rel_fields if f in related_rec
        }
        if key not in ctx and via_val is not None:
            ctx[key] = via_val
    return ctx
