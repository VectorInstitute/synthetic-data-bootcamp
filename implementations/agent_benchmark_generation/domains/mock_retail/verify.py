"""Mock retail domain-specific checks for draft tasks."""

from __future__ import annotations

from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


def check_domain_rules(domain: DomainBundle, draft: Task) -> list[str]:
    """Domain-specific oracle invariants for draft tasks."""
    errors: list[str] = []
    task_type = draft.task_type

    if task_type == "cancel":
        cancel_actions = [a for a in draft.evaluation_criteria.actions if a.name == "cancel_order"]
        for a in cancel_actions:
            oid = a.arguments.get("order_id")
            order = domain.db.get("orders", {}).get(oid, {})
            if order.get("status") != "pending":
                errors.append(f"cancel_order on non-pending order: {oid}")

    return errors
