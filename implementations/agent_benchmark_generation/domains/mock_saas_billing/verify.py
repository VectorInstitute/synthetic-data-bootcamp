"""Domain-specific verification for mock SaaS billing tasks."""

from __future__ import annotations

from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


def check_domain_rules(domain: DomainBundle, draft: Task) -> list[str]:
    """Return policy-invariant violations in a generated oracle."""
    checker = {
        "void_invoice": _check_void_invoice,
        "update_billing_email": _check_billing_email,
        "update_seats": _check_update_seats,
    }.get(draft.task_type)
    return checker(domain, draft.evaluation_criteria.actions) if checker else []


def _check_void_invoice(domain: DomainBundle, actions: list[Action]) -> list[str]:
    errors: list[str] = []
    for action in (a for a in actions if a.name == "void_invoice"):
        invoice_id = action.arguments.get("invoice_id")
        invoice = domain.db.get("invoices", {}).get(invoice_id, {})
        if invoice.get("status") not in {"open", "past_due"}:
            errors.append(f"void_invoice on ineligible invoice: {invoice_id}")
    return errors


def _check_billing_email(domain: DomainBundle, actions: list[Action]) -> list[str]:
    errors: list[str] = []
    for action in (a for a in actions if a.name == "update_billing_email"):
        account_id = action.arguments.get("account_id")
        if account_id not in domain.db.get("accounts", {}):
            errors.append(f"update_billing_email on missing account: {account_id}")
    return errors


def _check_update_seats(domain: DomainBundle, actions: list[Action]) -> list[str]:
    errors: list[str] = []
    for action in (a for a in actions if a.name == "update_seats"):
        subscription_id = action.arguments.get("subscription_id")
        subscription = domain.db.get("subscriptions", {}).get(subscription_id, {})
        if subscription.get("status") != "active":
            errors.append(f"update_seats on inactive subscription: {subscription_id}")
        seats = action.arguments.get("seats")
        if isinstance(seats, bool) or not isinstance(seats, int) or seats < 1:
            errors.append(f"update_seats requires positive integer seats: {seats}")
    return errors
