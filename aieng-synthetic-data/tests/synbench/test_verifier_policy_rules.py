"""Tests that verification enforces the domain's policy rules."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import EvaluationCriteria, Task
from aieng.syn_data.synbench.verification.pipeline import verify_draft


def test_refuse_cancel_with_write_fails(mock_retail_path):
    """A refuse_cancel task may not include a write tool call."""
    domain = load_domain(mock_retail_path)
    draft = Task(
        id="bad_refuse",
        task_type="refuse_cancel",
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(name="get_order", arguments={"order_id": "ord_1002"}),
                Action(name="cancel_order", arguments={"order_id": "ord_1002"}),
            ]
        ),
    )
    result = verify_draft(domain, draft)
    assert not result.verification_report.passed
    assert any(
        "path" in e.lower() or "write" in e.lower()
        for e in result.verification_report.errors
    )


def test_cancel_non_pending_fails_domain_check(mock_retail_path):
    """Cancelling a non-pending order is rejected by the domain checks."""
    domain = load_domain(mock_retail_path)
    draft = Task(
        id="bad_cancel",
        task_type="cancel",
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(name="get_order", arguments={"order_id": "ord_1002"}),
                Action(name="cancel_order", arguments={"order_id": "ord_1002"}),
            ]
        ),
    )
    result = verify_draft(domain, draft)
    assert not result.verification_report.passed
    assert any("non-pending" in e for e in result.verification_report.errors)
