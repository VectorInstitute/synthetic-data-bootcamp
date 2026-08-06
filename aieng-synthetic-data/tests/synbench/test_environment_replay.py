"""Tests that replaying oracle actions produces the expected database state."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.environment.core import replay_actions
from aieng.syn_data.synbench.environment.hashing import db_hash


def test_replay_cancel_changes_hash(mock_retail_path):
    """A write task's oracle actions mutate the database."""
    domain = load_domain(mock_retail_path)
    seed = domain.seed_tasks[1]
    env = replay_actions(domain, seed.evaluation_criteria.actions)
    initial = db_hash(domain.db)
    final = db_hash(env.db)
    assert initial != final


def test_replay_readonly_same_hash(mock_retail_path):
    """A read-only task's oracle actions leave the database unchanged."""
    domain = load_domain(mock_retail_path)
    seed = domain.seed_tasks[0]
    env = replay_actions(domain, seed.evaluation_criteria.actions)
    initial = db_hash(domain.db)
    final = db_hash(env.db)
    assert initial == final
