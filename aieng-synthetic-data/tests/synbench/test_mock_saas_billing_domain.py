"""Tests for the scaled mock SaaS billing domain."""

from aieng.syn_data.synbench.domain.loader import load_domain, validate_domain
from aieng.syn_data.synbench.environment.core import Environment
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import EvaluationCriteria, Task
from aieng.syn_data.synbench.verification.pipeline import verify_draft


def test_domain_is_valid_and_scaled(mock_saas_billing_path):
    assert validate_domain(mock_saas_billing_path) == []
    domain = load_domain(mock_saas_billing_path)
    accounts = domain.db["accounts"]
    subscriptions = domain.db["subscriptions"]
    invoices = domain.db["invoices"]

    assert len(accounts) >= 40
    assert len(subscriptions) >= 80
    assert len(invoices) >= 150
    assert len({account["name"].lower() for account in accounts.values()}) == len(
        accounts
    )
    assert {invoice["status"] for invoice in invoices.values()} == {
        "open",
        "paid",
        "past_due",
        "void",
    }
    for invoice in invoices.values():
        assert invoice["account_id"] in accounts
        assert invoice["subscription_id"] in subscriptions
    for account_id in accounts:
        assert any(i["account_id"] == account_id for i in invoices.values())


def test_all_ten_seed_tasks_verify(mock_saas_billing_path):
    domain = load_domain(mock_saas_billing_path)
    assert len(domain.seed_tasks) == 10
    for task in domain.seed_tasks:
        result = verify_draft(domain, task)
        assert result.verification_report.passed, (
            task.id,
            result.verification_report.errors,
        )


def test_every_tool_dispatches(mock_saas_billing_path):
    domain = load_domain(mock_saas_billing_path)
    env = Environment(domain)
    assert env.dispatch(Action(name="find_account_id", arguments={"name": "Avery Kim"}))
    assert env.dispatch(
        Action(name="get_invoice", arguments={"invoice_id": "inv_avery_open"})
    )
    assert env.dispatch(
        Action(name="list_invoices", arguments={"account_id": "acct_avery"})
    )
    assert env.dispatch(
        Action(name="list_subscriptions", arguments={"account_id": "acct_avery"})
    )
    assert env.dispatch(
        Action(name="void_invoice", arguments={"invoice_id": "inv_avery_open"})
    )
    assert env.dispatch(
        Action(
            name="update_billing_email",
            arguments={"account_id": "acct_avery", "billing_email": "ap@example.com"},
        )
    )
    assert env.dispatch(
        Action(
            name="update_seats",
            arguments={"subscription_id": "sub_avery_1", "seats": 6},
        )
    )


def test_paid_invoice_cannot_be_voided(mock_saas_billing_path):
    domain = load_domain(mock_saas_billing_path)
    task = Task(
        id="bad_paid_void",
        task_type="void_invoice",
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(name="get_invoice", arguments={"invoice_id": "inv_avery_paid"}),
                Action(name="void_invoice", arguments={"invoice_id": "inv_avery_paid"}),
            ]
        ),
    )
    result = verify_draft(domain, task)
    assert not result.verification_report.passed
    assert any("ineligible" in error for error in result.verification_report.errors)


def test_canceled_subscription_seats_cannot_change(mock_saas_billing_path):
    domain = load_domain(mock_saas_billing_path)
    task = Task(
        id="bad_canceled_seats",
        task_type="update_seats",
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(name="get_invoice", arguments={"invoice_id": "inv_jordan_open"}),
                Action(
                    name="update_seats",
                    arguments={"subscription_id": "sub_jordan_1", "seats": 4},
                ),
            ]
        ),
    )
    result = verify_draft(domain, task)
    assert not result.verification_report.passed
    assert any("inactive" in error for error in result.verification_report.errors)
