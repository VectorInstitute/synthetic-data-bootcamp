"""Load a domain folder into a validated ``DomainBundle``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from aieng.syn_data.synbench.generation.sampler import ConstraintSampler
from aieng.syn_data.synbench.schemas.domain import DomainBundle, DomainManifest
from aieng.syn_data.synbench.schemas.generation import GenerationConfig
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.schemas.tools import ToolSpec


# Files every domain folder must provide. `generation.yaml` configures how
# ConstraintSampler / PromptBuilder bind to this domain's db.json shape —
# see synbench.schemas.generation and domains/mock_retail/generation.yaml.
REQUIRED_FILES = [
    "policy.md",
    "db.json",
    "tools.py",
    "state_machine.yaml",
    "user_simulator.yaml",
    "tasks.seed.json",
    "generation.yaml",
]


class DomainLoadError(Exception):
    """Raised when a domain bundle is incomplete or invalid."""


def _load_tools_module(tools_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("domain_tools", tools_path)
    if spec is None or spec.loader is None:
        raise DomainLoadError(f"Cannot load tools module: {tools_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_tool_specs"):
        raise DomainLoadError(f"{tools_path} must define get_tool_specs()")
    if not hasattr(module, "ToolKit"):
        raise DomainLoadError(f"{tools_path} must define ToolKit class")
    return module


def _load_generation_config(root: Path) -> GenerationConfig:
    """Parse and validate ``generation.yaml`` into a ``GenerationConfig``.

    This file is the domain's sampling/prompt customization surface
    (primary collection, id field, related joins, communicate hints, …).
    Schema errors surface as ``DomainLoadError`` so ``load_domain`` fails fast.
    """
    path = root / "generation.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        return GenerationConfig.model_validate(raw)
    except Exception as e:
        raise DomainLoadError(f"Invalid generation.yaml: {e}") from e


def load_domain(path: str | Path) -> DomainBundle:
    """Load a domain directory into a ``DomainBundle``.

    Requires every name in ``REQUIRED_FILES``, including ``generation.yaml``,
    which is attached as ``bundle.generation`` for sampler/prompt/agent use.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise DomainLoadError(f"Domain path is not a directory: {root}")

    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    if missing:
        raise DomainLoadError(
            f"Domain '{root.name}' is missing required files: {', '.join(missing)}"
        )

    tools_mod = _load_tools_module(root / "tools.py")
    tool_specs: list[ToolSpec] = tools_mod.get_tool_specs()

    with open(root / "db.json") as f:
        db = json.load(f)

    with open(root / "state_machine.yaml") as f:
        state_machine = yaml.safe_load(f)

    with open(root / "user_simulator.yaml") as f:
        user_simulator = yaml.safe_load(f)

    with open(root / "tasks.seed.json") as f:
        seed_raw = json.load(f)
    seed_tasks = [Task.model_validate(t) for t in seed_raw.get("tasks", seed_raw)]

    policy = (root / "policy.md").read_text()
    # Domain-owned sampling + generation-prompt settings (see generation.yaml)
    generation = _load_generation_config(root)
    manifest_data = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
    manifest = DomainManifest(
        name=manifest_data.get("name", root.name),
        version=manifest_data.get("version", "0.1.0"),
        description=manifest_data.get("description", ""),
    )

    return DomainBundle(
        root=root,
        manifest=manifest,
        policy=policy,
        db=db,
        tools=tool_specs,
        state_machine=state_machine,
        user_simulator=user_simulator,
        seed_tasks=seed_tasks,
        generation=generation,
    )


def _validate_generation_readiness(bundle: DomainBundle) -> list[str]:
    """Check that ``generation.yaml`` lines up with ``db.json`` / the FSM.

    Purpose
    -------
    ``load_domain`` only checks that files parse. A domain can load successfully
    and still crash at ``generate`` time (wrong collection name, missing
    ``id_field``, broken related refs). This function is called from
    ``validate_domain`` / ``synbench domain validate`` so authors catch those
    mismatches before running the LLM.

    What it checks
    --------------
    1. ``primary_collection`` exists in ``db.json`` and is a non-empty object.
    2. Each ``related.*.collection`` exists; ``persona_related`` names a
       ``related`` key when set.
    3. Every primary record has ``id_field`` and each related ``via`` field;
       related ids resolve; optional ``related.*.fields`` exist on those rows.
    4. ``communicate_hints`` keys are real ``task_types`` (typo catch).
    5. One ``ConstraintSampler(...).sample()`` smoke call if no errors yet.

    Returns a list of human-readable error strings (empty if ready to generate).
    """
    errors: list[str] = []
    cfg = bundle.generation
    task_types = bundle.state_machine.get("task_types") or {}

    # --- structural: can we even look up a primary table? ---
    if cfg.primary_collection not in bundle.db:
        errors.append(
            f"generation.primary_collection '{cfg.primary_collection}' not found in db.json"
        )
        return errors

    collection = bundle.db[cfg.primary_collection]
    if not isinstance(collection, dict) or not collection:
        errors.append(
            f"db['{cfg.primary_collection}'] must be a non-empty object keyed by entity id"
        )
        return errors

    # --- related table names + persona wiring ---
    for rel_key, rel in cfg.related.items():
        if rel.collection not in bundle.db:
            errors.append(
                f"generation.related.{rel_key}.collection '{rel.collection}' not found in db.json"
            )

    if cfg.persona_related and cfg.persona_related not in cfg.related:
        errors.append(
            f"generation.persona_related '{cfg.persona_related}' is not a key in generation.related"
        )

    # --- per-record: id_field, via fields, FK targets, optional related fields ---
    for eid, record in collection.items():
        errors.extend(_validate_record(bundle, cfg, eid, record))

    # --- communicate_hints keys should match FSM task_types (catch typos) ---
    for hint_type in cfg.communicate_hints:
        if hint_type not in task_types:
            errors.append(
                f"generation.communicate_hints key '{hint_type}' is not a state_machine task_type"
            )

    # --- end-to-end smoke: sampler must be able to produce one SampleConstraints ---
    if not errors:
        try:
            ConstraintSampler(bundle, seed=0).sample()
        except Exception as e:
            errors.append(f"ConstraintSampler smoke sample failed: {e}")

    return errors


def _validate_record(
    bundle: DomainBundle, cfg: GenerationConfig, eid: str, record: Any
) -> list[str]:
    """Check one primary record's ``id_field`` and its ``related`` joins."""
    if not isinstance(record, dict):
        return [f"db['{cfg.primary_collection}']['{eid}'] must be an object"]

    errors: list[str] = []
    if cfg.id_field not in record:
        errors.append(
            f"db['{cfg.primary_collection}']['{eid}'] missing id_field '{cfg.id_field}'"
        )

    for rel_key, rel in cfg.related.items():
        if rel.via not in record:
            errors.append(
                f"db['{cfg.primary_collection}']['{eid}'] missing related via field "
                f"'{rel.via}' (for related.{rel_key})"
            )
            continue
        via_val = record[rel.via]
        related_coll = bundle.db.get(rel.collection) or {}
        if via_val not in related_coll:
            errors.append(
                f"db['{cfg.primary_collection}']['{eid}'].{rel.via}={via_val!r} "
                f"not found in db['{rel.collection}']"
            )
            continue
        related_rec = related_coll[via_val]
        for field in rel.fields or []:
            if field not in related_rec:
                errors.append(
                    f"db['{rel.collection}']['{via_val}'] missing field '{field}' "
                    f"(generation.related.{rel_key}.fields)"
                )
    return errors


def validate_domain(path: str | Path) -> list[str]:
    """Return validation errors for a domain folder (empty list if OK).

    Covers loadability, tools/seeds/FSM basics, ToolKit method presence, and
    generation readiness (``generation.yaml`` ↔ ``db.json``). Used by
    ``synbench domain validate``.
    """
    errors: list[str] = []
    try:
        bundle = load_domain(path)
    except DomainLoadError as e:
        return [str(e)]

    if not bundle.tools:
        errors.append("No tools defined")
    if not bundle.seed_tasks:
        errors.append("No seed tasks in tasks.seed.json")
    if "task_types" not in bundle.state_machine:
        errors.append("state_machine.yaml must define task_types")

    tools_mod = _load_tools_module(Path(path).resolve() / "tools.py")
    try:
        tk = tools_mod.ToolKit(bundle.db.copy())
        for spec in bundle.tools:
            if not hasattr(tk, spec.name):
                errors.append(f"ToolKit missing method: {spec.name}")
    except Exception as e:
        errors.append(f"ToolKit init failed: {e}")

    # Catch generate-time failures early (wrong collection / id_field / FKs)
    errors.extend(_validate_generation_readiness(bundle))
    return errors
