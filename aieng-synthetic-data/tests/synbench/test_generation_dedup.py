"""Tests that verification deduplicates drafts by oracle action sequence."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import EvaluationCriteria, Task, UserScenario
from aieng.syn_data.synbench.verification.pipeline import filter_verified


def _draft(did: str, oid: str) -> Task:
    """Build a minimal inquiry draft that looks up ``oid``."""
    return Task(
        id=did,
        task_type="inquiry",
        user_scenario=UserScenario(),
        evaluation_criteria=EvaluationCriteria(
            actions=[Action(name="get_order", arguments={"order_id": oid})]
        ),
    )


def test_dedup_same_actions(mock_retail_path):
    """Two drafts with identical actions collapse to one verified task."""
    domain = load_domain(mock_retail_path)
    drafts = [_draft("a", "ord_1001"), _draft("b", "ord_1001")]
    verified, rejected, seen = filter_verified(domain, drafts)
    assert len(verified) == 1
    assert len(rejected) == 1
    assert rejected[0][1] == [
        "Duplicate actions: exact sequence of tool calls and arguments"
    ]
    assert len(seen) == 1


def test_dedup_keeps_distinct_actions(mock_retail_path):
    """Read-only drafts targeting different orders are both kept."""
    domain = load_domain(mock_retail_path)
    drafts = [_draft("a", "ord_1001"), _draft("b", "ord_1002")]
    verified, rejected, seen = filter_verified(domain, drafts)
    assert len(verified) == 2
    assert rejected == []
    assert len(seen) == 2
