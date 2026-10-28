"""Render a spec's statements as text: Item 8 tables and a facts sheet for prompts.

Item 8 is produced by code, never by the model, so its numbers are exactly the
spec's. The facts sheet gives the model every figure it may quote, plus derived
percentages, so it does not have to do arithmetic.
"""

from __future__ import annotations

from typing import Any

from method_1.places import CITIES, STATE_NAMES


LABELS = {
    "revenue": "Revenue",
    "cost_of_revenue": "Cost of revenue",
    "gross_profit": "Gross profit",
    "research_and_development": "Research and development",
    "selling_general_and_administrative": "Selling, general and administrative",
    "other_operating_expense": "Other operating expense",
    "operating_income": "Operating income",
    "interest_expense": "Interest expense",
    "pretax_income": "Income before income taxes",
    "income_tax": "Provision for income taxes",
    "net_income": "Net income",
    "cash": "Cash and cash equivalents",
    "receivables": "Accounts receivable, net",
    "inventory": "Inventories",
    "other_current_assets": "Other current assets",
    "total_current_assets": "Total current assets",
    "property_plant_and_equipment": "Property, plant and equipment, net",
    "goodwill": "Goodwill",
    "other_noncurrent_assets": "Other non-current assets",
    "total_assets": "Total assets",
    "total_current_liabilities": "Total current liabilities",
    "long_term_debt": "Long-term debt",
    "other_noncurrent_liabilities": "Other non-current liabilities",
    "total_liabilities": "Total liabilities",
    "total_equity": "Total stockholders' equity",
    "total_liabilities_and_equity": "Total liabilities and stockholders' equity",
    "depreciation_and_amortization": "Depreciation and amortization",
    "other_operating_items": "Changes in working capital and other",
    "net_cash_from_operating_activities": "Net cash provided by operating activities",
    "capital_expenditures": "Purchases of property and equipment",
    "other_investing_activities": "Other investing activities, net",
    "net_cash_from_investing_activities": "Net cash used in investing activities",
    "net_change_in_long_term_debt": "Net change in long-term debt",
    "dividends_paid": "Dividends paid",
    "share_repurchases_net": "Share repurchases, net of issuances",
    "net_cash_from_financing_activities": "Net cash used in financing activities",
    "net_change_in_cash": "Net change in cash and cash equivalents",
    "cash_beginning_of_year": "Cash and cash equivalents, beginning of year",
    "cash_end_of_year": "Cash and cash equivalents, end of year",
}
BALANCE_ORDER = [
    "cash",
    "receivables",
    "inventory",
    "other_current_assets",
    "total_current_assets",
    "property_plant_and_equipment",
    "goodwill",
    "other_noncurrent_assets",
    "total_assets",
    "total_current_liabilities",
    "long_term_debt",
    "other_noncurrent_liabilities",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
]


def money(value: int) -> str:
    """Format millions the way 10-K tables do: ``1,234`` or ``(56)``."""
    return f"({abs(value):,})" if value < 0 else f"{value:,}"


def _table(title: str, columns: list[str], rows: list[tuple[str, list[int]]]) -> str:
    width = max(len(label) for label, _ in rows) + 2
    lines = [
        title,
        "(in millions)".ljust(width) + "".join(c.rjust(14) for c in columns),
    ]
    lines += [
        label.ljust(width) + "".join(money(v).rjust(14) for v in values)
        for label, values in rows
    ]
    return "\n".join(lines)


def statement_tables(spec: dict[str, Any]) -> str:
    """Return the three statements as plain-text tables."""
    st = spec["statements"]
    years = [str(y) for y in st["fiscal_years"]]
    income = [
        (LABELS[k], [st["income_statement"][y][k] for y in years])
        for k in st["income_statement"][years[0]]
    ]
    balance = [
        (LABELS[k], [st["balance_sheet"][y][k] for y in years]) for k in BALANCE_ORDER
    ]
    cash = [(LABELS[k], [v]) for k, v in st["cash_flow_statement"][years[0]].items()]
    name = spec["name"]
    return "\n\n".join(
        [
            _table(
                f"{name}\nConsolidated Statements of Operations",
                [f"FY{y}" for y in years],
                income,
            ),
            _table(
                f"{name}\nConsolidated Balance Sheets",
                [f"FY{y}" for y in years],
                balance,
            ),
            _table(
                f"{name}\nConsolidated Statement of Cash Flows", [f"FY{years[0]}"], cash
            ),
        ],
    )


def derived_metrics(spec: dict[str, Any]) -> dict[str, float]:
    """Percentages a narrative typically quotes, computed once so the model need not."""
    st = spec["statements"]
    cur, prev = (st["income_statement"][str(y)] for y in st["fiscal_years"])

    def pct(a: float, b: float) -> float:
        return round(100 * a / b, 1) + 0.0 if b else 0.0  # + 0.0 drops -0.0

    out = {
        "revenue_growth_pct": pct(cur["revenue"] - prev["revenue"], prev["revenue"]),
        "net_income_growth_pct": pct(
            cur["net_income"] - prev["net_income"], abs(prev["net_income"])
        ),
        "operating_income_growth_pct": pct(
            cur["operating_income"] - prev["operating_income"],
            abs(prev["operating_income"]),
        ),
    }
    for label, line in (("current", cur), ("prior", prev)):
        out[f"gross_margin_pct_{label}"] = pct(line["gross_profit"], line["revenue"])
        out[f"operating_margin_pct_{label}"] = pct(
            line["operating_income"], line["revenue"]
        )
        out[f"net_margin_pct_{label}"] = pct(line["net_income"], line["revenue"])
        out[f"effective_tax_rate_pct_{label}"] = pct(
            line["income_tax"], line["pretax_income"]
        )
        for cost in ("research_and_development", "selling_general_and_administrative"):
            out[f"{cost}_pct_of_revenue_{label}"] = pct(line[cost], line["revenue"])
    return out


def metric_label(key: str, fiscal_year: int) -> str:
    """Turn ``gross_margin_pct_current`` into ``Gross margin, fiscal 2024``."""
    period = ""
    if key.endswith("_current"):
        key, period = key.removesuffix("_current"), f", fiscal {fiscal_year}"
    elif key.endswith("_prior"):
        key, period = key.removesuffix("_prior"), f", fiscal {fiscal_year - 1}"
    elif key.endswith("_growth_pct"):
        period = f", fiscal {fiscal_year} versus {fiscal_year - 1}"
    text = (
        key.replace("_pct_of_revenue", " as a share of revenue")
        .replace("_pct", "")
        .replace(
            "selling_general_and_administrative", "selling, general and administrative"
        )
        .replace("_", " ")
    )
    return text[0].upper() + text[1:] + period


def facts_sheet(spec: dict[str, Any]) -> str:
    """Return the profile, statements and derived percentages, for prompts."""
    hq = spec["headquarters_state"]
    city = CITIES.get(hq, CITIES["DE"])[0]
    profile = [
        f"Company name: {spec['name']} (ticker {spec['ticker']})",
        f"Sector: {spec['sector']}",
        f"Incorporated in: {STATE_NAMES.get(spec['state_of_incorporation'], 'Delaware')}",
        f"Headquarters: {city}, {STATE_NAMES.get(hq, hq)}",
        f"Fiscal year: {spec['fiscal_year']} (ended {spec['fiscal_year_end']})",
        f"Employees: about {spec['employees']:,}",
    ]
    fy = int(spec["fiscal_year"])
    derived = "\n".join(
        f"{metric_label(k, fy)}: {v}%" for k, v in derived_metrics(spec).items()
    )
    return (
        "\n".join(profile)
        + "\n\n"
        + statement_tables(spec)
        + "\n\nDerived percentages:\n"
        + derived
    )


def item_8(spec: dict[str, Any]) -> str:
    """Item 8 text: heading plus the statement tables."""
    return (
        "Item 8. Financial Statements and Supplementary Data\n\n"
        + statement_tables(spec)
        + "\n"
    )
