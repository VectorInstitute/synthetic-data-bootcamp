"""Tests for ``generation.yaml`` parsing, sampling, and readiness validation."""

import json
from pathlib import Path

import pytest
import yaml

from aieng.syn_data.synbench.domain.loader import (
    DomainLoadError,
    load_domain,
    validate_domain,
)
from aieng.syn_data.synbench.generation.sampler import ConstraintSampler


def _copy_mock_retail(src: Path, dest: Path) -> Path:
    """Copy the domain's top-level files so a test can corrupt one safely."""
    dest.mkdir(parents=True)
    for path in src.iterdir():
        target = dest / path.name
        if path.is_file():
            target.write_bytes(path.read_bytes())
    return dest


def test_load_generation_config(mock_retail_path):
    """``generation.yaml`` is parsed onto the bundle's generation config."""
    bundle = load_domain(mock_retail_path)
    assert bundle.generation.primary_collection == "orders"
    assert bundle.generation.id_field == "order_id"
    assert "user_id" in bundle.generation.related
    assert bundle.generation.related["user_id"].collection == "users"
    assert "cancel" in bundle.generation.communicate_hints


def test_sampler_uses_generation_config(mock_retail_path):
    """Sampled constraints follow the configured id field and related joins."""
    domain = load_domain(mock_retail_path)
    constraints = ConstraintSampler(domain, seed=42).sample()
    assert constraints.primary_id
    assert constraints.entities["order_id"] == constraints.primary_id
    assert "user_id" in constraints.entities
    assert constraints.entities["user_id"].startswith("user_")
    assert "status" in constraints.entity_context
    assert "users" in constraints.entity_context
    assert "name" in constraints.entity_context["users"]
    assert constraints.fsm_path


def test_sampler_draws_personality_style(mock_retail_path):
    """A style is drawn from the domain's ``personality_styles`` catalog."""
    domain = load_domain(mock_retail_path)
    styles = domain.user_simulator.get("personality_styles") or []
    assert styles, "mock_retail should define personality_styles"
    names = {s["name"] for s in styles}
    constraints = ConstraintSampler(domain, seed=0).sample()
    assert constraints.personality_style is not None
    assert constraints.personality_style["name"] in names
    assert constraints.personality_style["description"]


def test_sampler_personality_style_none_without_config(mock_retail_path, tmp_path):
    """Domains without a style catalog sample no personality style."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "no_styles")
    us_path = domain_dir / "user_simulator.yaml"
    cfg = yaml.safe_load(us_path.read_text())
    cfg.pop("personality_styles", None)
    us_path.write_text(yaml.safe_dump(cfg))
    domain = load_domain(domain_dir)
    constraints = ConstraintSampler(domain, seed=0).sample()
    assert constraints.personality_style is None


def test_validate_domain_ok(mock_retail_path):
    """The bundled domain is generation-ready."""
    assert validate_domain(mock_retail_path) == []


def test_validate_missing_primary_collection(mock_retail_path, tmp_path):
    """A primary collection absent from ``db.json`` is reported."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "bad_domain")
    gen_path = domain_dir / "generation.yaml"
    cfg = yaml.safe_load(gen_path.read_text())
    cfg["primary_collection"] = "tickets"
    gen_path.write_text(yaml.safe_dump(cfg))

    errors = validate_domain(domain_dir)
    assert any("primary_collection" in e and "tickets" in e for e in errors)


def test_validate_missing_id_field_on_record(mock_retail_path, tmp_path):
    """A record missing the configured id field is reported."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "bad_id")
    db_path = domain_dir / "db.json"
    db = json.loads(db_path.read_text())
    del db["orders"]["ord_1001"]["order_id"]
    db_path.write_text(json.dumps(db))

    errors = validate_domain(domain_dir)
    assert any("id_field" in e and "ord_1001" in e for e in errors)


def test_validate_broken_related_ref(mock_retail_path, tmp_path):
    """A related reference pointing at a missing row is reported."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "bad_rel")
    db_path = domain_dir / "db.json"
    db = json.loads(db_path.read_text())
    db["orders"]["ord_1001"]["user_id"] = "user_missing"
    db_path.write_text(json.dumps(db))

    errors = validate_domain(domain_dir)
    assert any("user_missing" in e for e in errors)


def test_validate_unknown_communicate_hint(mock_retail_path, tmp_path):
    """A communicate hint keyed by an unknown task type is reported."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "bad_hint")
    gen_path = domain_dir / "generation.yaml"
    cfg = yaml.safe_load(gen_path.read_text())
    cfg["communicate_hints"]["not_a_task"] = "hint"
    gen_path.write_text(yaml.safe_dump(cfg))

    errors = validate_domain(domain_dir)
    assert any("not_a_task" in e for e in errors)


def test_missing_generation_yaml_raises(mock_retail_path, tmp_path):
    """A domain without ``generation.yaml`` fails to load."""
    domain_dir = _copy_mock_retail(mock_retail_path, tmp_path / "no_gen")
    (domain_dir / "generation.yaml").unlink()

    with pytest.raises(DomainLoadError, match="generation.yaml"):
        load_domain(domain_dir)
