"""Mock retail domain toolkit."""

from __future__ import annotations

import copy
from typing import Any

from aieng.syn_data.synbench.schemas.tools import ToolSpec, ToolType


def get_tool_specs() -> list[ToolSpec]:
    """Declare the tools this domain exposes to the agent."""
    return [
        ToolSpec(
            name="find_user_id",
            description=(
                "Look up a customer's user_id by their full name. "
                "Ask the customer for their name first, then call this tool "
                "before list_orders or any order mutation."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            tool_type=ToolType.READ,
            fsm_tag="lookup",
        ),
        ToolSpec(
            name="get_order",
            description="Get order details by order_id",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            tool_type=ToolType.READ,
            fsm_tag="lookup",
        ),
        ToolSpec(
            name="list_orders",
            description="List orders for a user_id",
            parameters={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            tool_type=ToolType.READ,
            fsm_tag="lookup",
        ),
        ToolSpec(
            name="cancel_order",
            description="Cancel a pending order",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            tool_type=ToolType.WRITE,
            fsm_tag="mutate",
        ),
        ToolSpec(
            name="update_shipping",
            description="Update shipping address for an order",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "shipping_address": {"type": "string"},
                },
                "required": ["order_id", "shipping_address"],
            },
            tool_type=ToolType.WRITE,
            fsm_tag="mutate",
        ),
    ]


class ToolKit:
    """Implementations of the mock retail tools over a mutable database."""

    def __init__(self, db: dict[str, Any]):
        self.db = db

    def find_user_id(self, name: str) -> dict[str, Any]:
        """Look up a customer by full name."""
        needle = name.strip().lower()
        for user in self.db["users"].values():
            if str(user.get("name", "")).strip().lower() == needle:
                return {
                    "user_id": user["user_id"],
                    "name": user["name"],
                    "email": user.get("email"),
                }
        raise ValueError(f"User not found for name: {name}")

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Return the order with ``order_id``."""
        order = self.db["orders"].get(order_id)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        return copy.deepcopy(order)

    def list_orders(self, user_id: str) -> list[dict[str, Any]]:
        """Return every order belonging to ``user_id``."""
        return [copy.deepcopy(o) for o in self.db["orders"].values() if o["user_id"] == user_id]

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel ``order_id``, which must still be pending."""
        order = self.db["orders"].get(order_id)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        if order["status"] != "pending":
            raise ValueError(f"Cannot cancel order with status: {order['status']}")
        order["status"] = "cancelled"
        return {"order_id": order_id, "status": "cancelled"}

    def update_shipping(self, order_id: str, shipping_address: str) -> dict[str, Any]:
        """Set a new shipping address on ``order_id``."""
        order = self.db["orders"].get(order_id)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        order["shipping_address"] = shipping_address
        return {"order_id": order_id, "shipping_address": shipping_address}
