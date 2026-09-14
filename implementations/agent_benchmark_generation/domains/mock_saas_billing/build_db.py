"""Build the deterministic mock SaaS billing database."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path


SEED_ACCOUNTS = [
    ("acct_avery", "Avery Kim", "Northstar Labs"),
    ("acct_noah", "Noah Patel", "Cedar Analytics"),
    ("acct_maya", "Maya Chen", "Orbit Works"),
    ("acct_luis", "Luis Ortega", "Summit Systems"),
    ("acct_priya", "Priya Shah", "Bluebird Studio"),
    ("acct_jordan", "Jordan Blake", "Harbor Digital"),
]
FIRST_NAMES = ["Amara", "Ben", "Chloe", "Diego", "Elena", "Finn", "Grace"]
LAST_NAMES = ["Adams", "Brooks", "Cole", "Diaz", "Evans", "Foster"]
PLANS = {
    "starter": (3, 2900),
    "pro": (10, 9900),
    "business": (25, 24900),
}
STATUSES = ["open", "paid", "past_due", "void"]
PINNED_INVOICES = {
    ("acct_avery", 0): ("inv_avery_open", "open"),
    ("acct_avery", 1): ("inv_avery_paid", "paid"),
    ("acct_noah", 0): ("inv_noah_open", "open"),
    ("acct_maya", 0): ("inv_maya_pastdue", "past_due"),
    ("acct_luis", 0): ("inv_luis_paid", "paid"),
    ("acct_priya", 0): ("inv_priya_open", "open"),
    ("acct_jordan", 0): ("inv_jordan_open", "open"),
}


def _slug(value: str) -> str:
    return value.lower().replace(" ", ".")


def build_database(seed: int = 42) -> dict:
    """Return 48 accounts, 96 subscriptions, and 192 invoices."""
    rng = random.Random(seed)
    generated = [
        (f"acct_{i:03d}", f"{first} {last}", f"{last} {i:02d} Software")
        for i, (first, last) in enumerate(
            ((first_name, last_name) for first_name in FIRST_NAMES for last_name in LAST_NAMES),
            start=7,
        )
    ]
    account_rows = SEED_ACCOUNTS + generated
    accounts: dict[str, dict] = {}
    subscriptions: dict[str, dict] = {}
    invoices: dict[str, dict] = {}
    base_date = date(2026, 1, 15)

    for account_index, (account_id, name, company) in enumerate(account_rows):
        plan = list(PLANS)[account_index % len(PLANS)]
        email_slug = _slug(name)
        accounts[account_id] = {
            "account_id": account_id,
            "name": name,
            "email": f"{email_slug}@example.com",
            "billing_email": f"billing+{email_slug}@example.com",
            "company": company,
            "plan": plan,
        }

        account_subscriptions = []
        for sub_index in range(2):
            subscription_id = f"sub_{account_id.removeprefix('acct_')}_{sub_index + 1}"
            sub_plan = list(PLANS)[(account_index + sub_index) % len(PLANS)]
            default_seats, monthly_cents = PLANS[sub_plan]
            status = (
                "canceled"
                if account_id == "acct_jordan" and sub_index == 0
                else "past_due"
                if account_index >= len(SEED_ACCOUNTS) and (account_index + sub_index) % 17 == 0
                else "active"
            )
            subscriptions[subscription_id] = {
                "subscription_id": subscription_id,
                "account_id": account_id,
                "plan": sub_plan,
                "status": status,
                "seats": default_seats + rng.randint(0, 4),
                "monthly_cents": monthly_cents,
            }
            account_subscriptions.append(subscription_id)

        for invoice_index in range(4):
            pinned = PINNED_INVOICES.get((account_id, invoice_index))
            invoice_id = pinned[0] if pinned else f"inv_{account_id.removeprefix('acct_')}_{invoice_index + 1}"
            status = pinned[1] if pinned else STATUSES[(account_index + invoice_index) % len(STATUSES)]
            subscription_id = account_subscriptions[invoice_index % 2]
            subscription = subscriptions[subscription_id]
            quantity = subscription["seats"]
            unit_cents = max(900, subscription["monthly_cents"] // quantity)
            amount_cents = quantity * unit_cents
            due_date = base_date + timedelta(days=14 * (account_index + invoice_index))
            invoices[invoice_id] = {
                "invoice_id": invoice_id,
                "account_id": account_id,
                "subscription_id": subscription_id,
                "status": status,
                "amount_cents": amount_cents,
                "currency": "USD",
                "due_date": due_date.isoformat(),
                "billing_address": f"{100 + account_index} Market St, Suite {invoice_index + 1}",
                "line_items": [
                    {
                        "description": f"{subscription['plan'].title()} plan seats",
                        "quantity": quantity,
                        "unit_cents": unit_cents,
                    },
                ],
            }

    return {
        "accounts": accounts,
        "subscriptions": subscriptions,
        "invoices": invoices,
    }


if __name__ == "__main__":
    output = Path(__file__).with_name("db.json")
    output.write_text(json.dumps(build_database(), indent=2) + "\n")
    print(f"Wrote {output}")
