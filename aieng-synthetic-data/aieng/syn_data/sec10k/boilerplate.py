"""Cover-page facts and the short, formulaic Items every 10-K carries.

Items such as 1B, 4, 6, 9 and 10 to 14 are near-identical across real filings
(Items 10 to 14 are usually incorporated by reference to the proxy statement),
so they are written from templates rather than by the model. Every fact here is
derived deterministically from the spec, so a filing renders the same each time.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

from aieng.syn_data.sec10k.places import CITIES, STATE_NAMES
from aieng.syn_data.sec10k.render import money


# Ordered as in a real 10-K: (part, item, heading).
OUTLINE = [
    ("PART I", "1", "Business"),
    ("PART I", "1A", "Risk Factors"),
    ("PART I", "1B", "Unresolved Staff Comments"),
    ("PART I", "1C", "Cybersecurity"),
    ("PART I", "2", "Properties"),
    ("PART I", "3", "Legal Proceedings"),
    ("PART I", "4", "Mine Safety Disclosures"),
    (
        "PART II",
        "5",
        "Market for Registrant's Common Equity, Related Stockholder Matters and "
        "Issuer Purchases of Equity Securities",
    ),
    ("PART II", "6", "[Reserved]"),
    (
        "PART II",
        "7",
        "Management's Discussion and Analysis of Financial Condition and Results "
        "of Operations",
    ),
    ("PART II", "7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("PART II", "8", "Financial Statements and Supplementary Data"),
    (
        "PART II",
        "9",
        "Changes in and Disagreements with Accountants on Accounting and "
        "Financial Disclosure",
    ),
    ("PART II", "9A", "Controls and Procedures"),
    ("PART II", "9B", "Other Information"),
    (
        "PART II",
        "9C",
        "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    ),
    ("PART III", "10", "Directors, Executive Officers and Corporate Governance"),
    ("PART III", "11", "Executive Compensation"),
    (
        "PART III",
        "12",
        "Security Ownership of Certain Beneficial Owners and Management and "
        "Related Stockholder Matters",
    ),
    (
        "PART III",
        "13",
        "Certain Relationships and Related Transactions, and Director Independence",
    ),
    ("PART III", "14", "Principal Accountant Fees and Services"),
    ("PART IV", "15", "Exhibits and Financial Statement Schedules"),
    ("PART IV", "16", "Form 10-K Summary"),
]
HEADINGS = {item: f"Item {item}. {title}" for _, item, title in OUTLINE}

_STREETS = ["Corporate Drive", "Innovation Way", "Commerce Parkway", "Market Plaza"]


def long_date(iso: str) -> str:
    """Format ``2024-12-31`` as ``December 31, 2024``."""
    d = date.fromisoformat(iso)
    return f"{d:%B} {d.day}, {d.year}"


def cover_facts(spec: dict[str, Any]) -> dict[str, str]:
    """Invent the cover-page identifiers for a fictional registrant."""
    rng = random.Random(f"{spec['company_id']}-cover")
    st = spec["statements"]
    net_income = st["income_statement"][str(spec["fiscal_year"])]["net_income"]
    pe = rng.uniform(14, 32) if net_income > 0 else rng.uniform(1.5, 4)
    market_cap = max(net_income * pe, spec["revenue"] * rng.uniform(0.8, 3))
    price = rng.uniform(35, 260)
    shares = market_cap * 1e6 / price
    hq = spec["headquarters_state"]
    city, zip3 = CITIES.get(hq, CITIES["DE"])
    fy = int(spec["fiscal_year"])
    return {
        "file_number": f"001-{rng.randint(10000, 42999)}",
        "ein": f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        "address": (
            f"{rng.randint(100, 9900)} {rng.choice(_STREETS)}, {city}, "
            f"{STATE_NAMES.get(hq, hq)} {zip3}{rng.randint(10, 99)}"
        ),
        "phone": f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}",
        "exchange": rng.choice(
            ["The Nasdaq Stock Market LLC", "New York Stock Exchange"]
        ),
        "market_value": f"${market_cap * 0.97 / 1000:,.1f} billion",
        "market_value_date": f"June 30, {fy}",
        "shares": f"{shares:,.0f}",
        "shares_date": f"February 14, {fy + 1}",
        "state": STATE_NAMES.get(spec["state_of_incorporation"], "Delaware"),
        "fiscal_year_end": long_date(spec["fiscal_year_end"]),
        "proxy": f"{fy + 1} Annual Meeting of Stockholders",
    }


def _the(exchange: str) -> str:
    return exchange if exchange.startswith("The ") else f"the {exchange}"


def _item_5(spec: dict[str, Any], facts: dict[str, str]) -> str:
    cash = spec["statements"]["cash_flow_statement"][str(spec["fiscal_year"])]
    dividends, buybacks = -cash["dividends_paid"], -cash["share_repurchases_net"]
    lines = [
        "Market Information",
        f"Our common stock is traded on {_the(facts['exchange'])} under the symbol "
        f'"{spec["ticker"]}".',
        "Dividends",
    ]
    if dividends > 0:
        lines.append(
            f"We paid cash dividends of ${money(dividends)} million in fiscal "
            f"{spec['fiscal_year']}. Future dividends are subject to declaration by "
            "our Board of Directors and will depend on our earnings, financial "
            "condition and capital requirements."
        )
    else:
        lines.append(
            "We have not paid cash dividends on our common stock and currently "
            "intend to retain earnings to fund the development of our business."
        )
    lines.append("Issuer Purchases of Equity Securities")
    if buybacks > 0:
        lines.append(
            f"During fiscal {spec['fiscal_year']} we repurchased shares of our "
            f"common stock, net of issuances, for ${money(buybacks)} million under "
            "programs authorized by our Board of Directors."
        )
    else:
        lines.append(
            "We did not make net repurchases of our common stock during the "
            "fiscal year."
        )
    return "\n\n".join(lines)


def template_items(spec: dict[str, Any]) -> dict[str, str]:
    """Return the formulaic Items as text, each starting with its heading."""
    facts = cover_facts(spec)
    proxy = (
        "The information required by this Item is incorporated by reference to "
        f"our definitive Proxy Statement for the {facts['proxy']}, to be filed "
        "with the SEC within 120 days after the end of the fiscal year covered "
        "by this report."
    )
    body = {
        "1B": "None.",
        "4": "Not applicable.",
        "5": _item_5(spec, facts),
        "6": "Not applicable.",
        "9": "None.",
        "9A": (
            "Evaluation of Disclosure Controls and Procedures\n\n"
            "Our management, with the participation of our Chief Executive Officer "
            "and Chief Financial Officer, evaluated the effectiveness of our "
            "disclosure controls and procedures as of the end of the period "
            "covered by this report. Based on that evaluation, they concluded that "
            "our disclosure controls and procedures were effective.\n\n"
            "Management's Report on Internal Control over Financial Reporting\n\n"
            "Management is responsible for establishing and maintaining adequate "
            "internal control over financial reporting. Management assessed the "
            f"effectiveness of our internal control as of {facts['fiscal_year_end']} "
            "using the criteria in Internal Control - Integrated Framework (2013) "
            "and concluded that it was effective. Our independent registered "
            "public accounting firm has issued an attestation report on our "
            "internal control over financial reporting.\n\n"
            "Changes in Internal Control over Financial Reporting\n\n"
            "There were no changes in our internal control over financial "
            "reporting during the fourth quarter that materially affected, or are "
            "reasonably likely to materially affect, our internal control over "
            "financial reporting."
        ),
        "9B": (
            "During the three months ended "
            f"{facts['fiscal_year_end']}, none of our directors or officers adopted "
            'or terminated a "Rule 10b5-1 trading arrangement" or "non-Rule 10b5-1 '
            'trading arrangement."'
        ),
        "9C": "Not applicable.",
        "10": proxy,
        "11": proxy,
        "12": proxy,
        "13": proxy,
        "14": proxy,
        "15": (
            "(a) The following documents are filed as part of this report:\n\n"
            "1. Financial Statements: see Item 8.\n\n"
            "2. Financial Statement Schedules: schedules are omitted because they "
            "are not applicable or the information is included in the financial "
            "statements.\n\n"
            "3. Exhibits: none are included with this synthetic filing."
        ),
        "16": "None.",
    }
    return {item: f"{HEADINGS[item]}\n\n{text}\n" for item, text in body.items()}
