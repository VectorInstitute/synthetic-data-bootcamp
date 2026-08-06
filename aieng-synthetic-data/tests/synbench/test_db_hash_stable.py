"""Tests that the database hash is stable across irrelevant representation changes."""

from aieng.syn_data.synbench.environment.hashing import db_hash


def test_hash_invariant_key_order():
    """Key insertion order does not change the hash."""
    a = {"orders": {"x": 1}, "users": {"y": 2}}
    b = {"users": {"y": 2}, "orders": {"x": 1}}
    assert db_hash(a) == db_hash(b)


def test_hash_float_int():
    """Whole-number floats hash the same as their integer form."""
    a = {"v": 1.0}
    b = {"v": 1}
    assert db_hash(a) == db_hash(b)
