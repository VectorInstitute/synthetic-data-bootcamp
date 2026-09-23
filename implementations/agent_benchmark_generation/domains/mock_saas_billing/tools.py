"""Mock SaaS billing domain toolkit."""

from __future__ import annotations

import copy
from typing import Any

from aieng.syn_data.synbench.schemas.tools import ToolSpec, ToolType


def _spec(
    name: str,
    description: str,
    properties: dict[str, dict[str, str]],
    required: list[str],
    tool_type: ToolType = ToolType.READ,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        tool_type=tool_type,
    )


def get_tool_specs() -> list[ToolSpec]:
    """Declare tools exposed to the billing support agent."""
    string = {"type": "string"}
    integer = {"type": "integer"}
    return [
        _spec("find_account_id", "Find an account by customer full name", {"name": string}, ["name"]),
        _spec("get_invoice", "Get invoice details", {"invoice_id": string}, ["invoice_id"]),
        _spec("list_invoices", "List invoices owned by an account", {"account_id": string}, ["account_id"]),
        _spec(
            "list_subscriptions",
            "List subscriptions owned by an account",
            {"account_id": string},
            ["account_id"],
        ),
        _spec(
            "void_invoice",
            "Void an open or past-due invoice",
            {"invoice_id": string},
            ["invoice_id"],
            ToolType.WRITE,
        ),
        _spec(
            "update_billing_email",
            "Change an account billing email",
            {"account_id": string, "billing_email": string},
            ["account_id", "billing_email"],
            ToolType.WRITE,
        ),
        _spec(
            "update_seats",
            "Change seat count on an active subscription",
            {"subscription_id": string, "seats": integer},
            ["subscription_id", "seats"],
            ToolType.WRITE,
        ),
    ]


class ToolKit:
    """Implement billing tools over a mutable domain database."""

    def __init__(self, db: dict[str, Any]):
        self.db = db

    def find_account_id(self, name: str) -> dict[str, Any]:
        """Return the account matching a customer's full name."""
        needle = name.strip().lower()
        for account in self.db["accounts"].values():
            if account["name"].strip().lower() == needle:
                return copy.deepcopy(account)
        raise ValueError(f"Account not found for name: {name}")

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Return one invoice by ID."""
        invoice = self.db["invoices"].get(invoice_id)
        if invoice is None:
            raise ValueError(f"Invoice not found: {invoice_id}")
        return copy.deepcopy(invoice)

    def list_invoices(self, account_id: str) -> list[dict[str, Any]]:
        """Return all invoices owned by an account."""
        self._require_account(account_id)
        return [
            copy.deepcopy(invoice) for invoice in self.db["invoices"].values() if invoice["account_id"] == account_id
        ]

    def list_subscriptions(self, account_id: str) -> list[dict[str, Any]]:
        """Return all subscriptions owned by an account."""
        self._require_account(account_id)
        return [
            copy.deepcopy(subscription)
            for subscription in self.db["subscriptions"].values()
            if subscription["account_id"] == account_id
        ]

    def void_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Void an invoice when its current status permits it."""
        invoice = self.db["invoices"].get(invoice_id)
        if invoice is None:
            raise ValueError(f"Invoice not found: {invoice_id}")
        if invoice["status"] not in {"open", "past_due"}:
            raise ValueError(f"Cannot void invoice with status: {invoice['status']}")
        invoice["status"] = "void"
        return {"invoice_id": invoice_id, "status": "void"}

    def update_billing_email(self, account_id: str, billing_email: str) -> dict[str, Any]:
        """Set an account's billing email."""
        account = self._require_account(account_id)
        if "@" not in billing_email:
            raise ValueError("Invalid billing email")
        account["billing_email"] = billing_email
        return {"account_id": account_id, "billing_email": billing_email}

    def update_seats(self, subscription_id: str, seats: int) -> dict[str, Any]:
        """Set the seat count on an active subscription."""
        subscription = self.db["subscriptions"].get(subscription_id)
        if subscription is None:
            raise ValueError(f"Subscription not found: {subscription_id}")
        if subscription["status"] != "active":
            raise ValueError(f"Cannot update subscription with status: {subscription['status']}")
        if isinstance(seats, bool) or not isinstance(seats, int) or seats < 1:
            raise ValueError("Seats must be a positive integer")
        subscription["seats"] = seats
        return {"subscription_id": subscription_id, "seats": seats}

    def _require_account(self, account_id: str) -> dict[str, Any]:
        account: dict[str, Any] | None = self.db["accounts"].get(account_id)
        if account is None:
            raise ValueError(f"Account not found: {account_id}")
        return account
