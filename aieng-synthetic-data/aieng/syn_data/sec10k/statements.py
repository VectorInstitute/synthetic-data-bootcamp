"""Compute financial statements that reconcile, from a fictional company's ratios.

All numbers are fixed here, in code, before any language model sees them. The
model only writes narrative around them, so every figure in a synthetic filing
can be checked against these tables.

Two fiscal years are built (the current year and the prior year) so the cash
flow statement and the equity roll-forward have an opening balance. Amounts are
in millions of US dollars, rounded to whole millions, and every subtotal is
recomputed from the rounded lines so the identities hold exactly.
"""

from __future__ import annotations

from typing import Any


INTEREST_RATE = 0.045  # on average long-term debt

INCOME_LINES = [
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "research_and_development",
    "selling_general_and_administrative",
    "other_operating_expense",
    "operating_income",
    "interest_expense",
    "pretax_income",
    "income_tax",
    "net_income",
]


def _round(values: dict[str, float]) -> dict[str, int]:
    return {k: round(v) for k, v in values.items()}


def income_statement(revenue: float, r: dict[str, float]) -> dict[str, int]:
    """Build one year's income statement; operating costs not modelled go to "other"."""
    cogs = revenue * (1 - r["gross_margin"])
    rd = revenue * r.get("rd_ratio", 0.0)
    sga = revenue * r["sga_ratio"]
    operating_income = revenue * r["operating_margin"]
    lines = _round(
        {
            "revenue": revenue,
            "cost_of_revenue": cogs,
            "research_and_development": rd,
            "selling_general_and_administrative": sga,
            "operating_income": operating_income,
            "interest_expense": r["_debt"] * INTEREST_RATE,
        },
    )
    lines["gross_profit"] = lines["revenue"] - lines["cost_of_revenue"]
    other = (
        lines["gross_profit"]
        - lines["research_and_development"]
        - lines["selling_general_and_administrative"]
    )
    other -= lines["operating_income"]
    if (
        other < 0
    ):  # margins sampled independently can over-explain costs; cut SG&A to fit
        lines["selling_general_and_administrative"] += other
        other = 0
    lines["other_operating_expense"] = other
    lines["pretax_income"] = lines["operating_income"] - lines["interest_expense"]
    lines["income_tax"] = round(max(lines["pretax_income"], 0) * r["tax_rate"])
    lines["net_income"] = lines["pretax_income"] - lines["income_tax"]
    return {k: lines[k] for k in INCOME_LINES}


def balance_sheet(revenue: float, r: dict[str, float]) -> dict[str, int]:
    """Build a balanced year-end balance sheet; "other" lines absorb residuals."""
    total_assets = revenue / r["asset_turnover"]
    current_assets = total_assets * r["current_assets_ratio"]
    cash = min(total_assets * r["cash_ratio"], current_assets * 0.8)
    receivables = revenue * r["receivables_ratio"]
    inventory = revenue * r.get("inventory_ratio", 0.0)
    if (
        cash + receivables + inventory > current_assets
    ):  # keep the parts inside the whole
        scale = current_assets * 0.95 / (cash + receivables + inventory)
        cash, receivables, inventory = (
            cash * scale,
            receivables * scale,
            inventory * scale,
        )
    noncurrent = total_assets - current_assets
    ppe = min(total_assets * r["ppe_ratio"], noncurrent * 0.9)
    goodwill = min(
        total_assets * r.get("goodwill_ratio", 0.0), (noncurrent - ppe) * 0.9
    )

    current_liabilities = current_assets / r["current_ratio"]
    debt = total_assets * r["debt_ratio"]
    equity = total_assets * r["equity_ratio"]
    if current_liabilities + debt + equity > total_assets:  # scale the claims to fit
        scale = total_assets * 0.97 / (current_liabilities + debt + equity)
        current_liabilities, debt, equity = (
            current_liabilities * scale,
            debt * scale,
            equity * scale,
        )

    b = _round(
        {
            "cash": cash,
            "receivables": receivables,
            "inventory": inventory,
            "total_current_assets": current_assets,
            "property_plant_and_equipment": ppe,
            "goodwill": goodwill,
            "total_assets": total_assets,
            "total_current_liabilities": current_liabilities,
            "long_term_debt": debt,
            "total_equity": equity,
        },
    )
    b["other_current_assets"] = (
        b["total_current_assets"] - b["cash"] - b["receivables"] - b["inventory"]
    )
    b["other_noncurrent_assets"] = (
        b["total_assets"]
        - b["total_current_assets"]
        - b["property_plant_and_equipment"]
        - b["goodwill"]
    )
    b["other_noncurrent_liabilities"] = (
        b["total_assets"]
        - b["total_current_liabilities"]
        - b["long_term_debt"]
        - b["total_equity"]
    )
    b["total_liabilities"] = (
        b["total_current_liabilities"]
        + b["long_term_debt"]
        + b["other_noncurrent_liabilities"]
    )
    b["total_liabilities_and_equity"] = b["total_liabilities"] + b["total_equity"]
    return b


def build_statements(spec: dict[str, Any]) -> dict[str, Any]:
    """Return income statements, balance sheets and a cash flow statement for ``spec``.

    ``spec`` needs ``revenue`` (current year, $M), ``revenue_growth`` and ``ratios``.
    """
    r = dict(spec["ratios"])
    fy = int(spec["fiscal_year"])
    years = {
        fy: spec["revenue"],
        fy - 1: spec["revenue"] / (1 + spec["revenue_growth"]),
    }

    balances, incomes = {}, {}
    for year, revenue in years.items():
        balances[year] = balance_sheet(revenue, r)
        r["_debt"] = balances[year]["long_term_debt"]
        incomes[year] = income_statement(revenue, r)

    cur, prev = balances[fy], balances[fy - 1]
    inc = incomes[fy]
    depreciation = round(years[fy] * r["da_ratio"])
    capex = round(years[fy] * r["capex_ratio"])
    # Operating cash = net income + depreciation + working capital and other.
    operating_cash_flow = (
        inc["net_income"]
        + depreciation
        + round(years[fy] * r.get("other_operating_cash_ratio", 0.0))
    )
    dividends = round(max(inc["net_income"], 0) * r.get("dividend_payout", 0.0))
    # Equity moves by net income - dividends - buybacks (negative = issuance).
    buybacks = (
        prev["total_equity"] + inc["net_income"] - dividends - cur["total_equity"]
    )
    debt_change = cur["long_term_debt"] - prev["long_term_debt"]
    financing = debt_change - dividends - buybacks
    # Other investing activity (securities, acquisitions) closes the cash bridge.
    other_investing = (
        (cur["cash"] - prev["cash"]) - operating_cash_flow - financing + capex
    )
    investing = -capex + other_investing
    cash_flow = {
        "net_income": inc["net_income"],
        "depreciation_and_amortization": depreciation,
        "other_operating_items": operating_cash_flow - inc["net_income"] - depreciation,
        "net_cash_from_operating_activities": operating_cash_flow,
        "capital_expenditures": -capex,
        "other_investing_activities": other_investing,
        "net_cash_from_investing_activities": investing,
        "net_change_in_long_term_debt": debt_change,
        "dividends_paid": -dividends,
        "share_repurchases_net": -buybacks,
        "net_cash_from_financing_activities": financing,
        "net_change_in_cash": operating_cash_flow + investing + financing,
        "cash_beginning_of_year": prev["cash"],
        "cash_end_of_year": cur["cash"],
    }
    return {
        "units": "USD millions",
        "fiscal_years": [fy, fy - 1],
        "income_statement": {str(y): incomes[y] for y in (fy, fy - 1)},
        "balance_sheet": {str(y): balances[y] for y in (fy, fy - 1)},
        "cash_flow_statement": {str(fy): cash_flow},
    }


def _check_balance_sheets(st: dict[str, Any]) -> list[str]:
    errors = []
    for year, b in st["balance_sheet"].items():
        if b["total_assets"] != b["total_liabilities_and_equity"]:
            errors.append(f"{year}: assets != liabilities + equity")
        parts = (
            b["cash"] + b["receivables"] + b["inventory"] + b["other_current_assets"]
        )
        if parts != b["total_current_assets"]:
            errors.append(f"{year}: current assets do not add up")
        if min(b.values()) < 0:
            errors.append(f"{year}: negative balance sheet line")
    return errors


def _check_income_statements(st: dict[str, Any]) -> list[str]:
    errors = []
    for year, i in st["income_statement"].items():
        if i["gross_profit"] != i["revenue"] - i["cost_of_revenue"]:
            errors.append(f"{year}: gross profit")
        opex = (
            i["research_and_development"]
            + i["selling_general_and_administrative"]
            + i["other_operating_expense"]
        )
        if i["operating_income"] != i["gross_profit"] - opex:
            errors.append(f"{year}: operating income")
        if i["net_income"] != i["pretax_income"] - i["income_tax"]:
            errors.append(f"{year}: net income")
        costs = (
            "cost_of_revenue",
            "research_and_development",
            "selling_general_and_administrative",
        )
        if min(i[k] for k in (*costs, "other_operating_expense")) < 0:
            errors.append(f"{year}: negative cost line")
    return errors


def _check_cash_flows(st: dict[str, Any]) -> list[str]:
    errors = []
    for year, c in st["cash_flow_statement"].items():
        if (
            c["cash_beginning_of_year"] + c["net_change_in_cash"]
            != c["cash_end_of_year"]
        ):
            errors.append(f"{year}: cash does not roll forward")
        if c["cash_end_of_year"] != st["balance_sheet"][year]["cash"]:
            errors.append(f"{year}: cash differs from the balance sheet")
        if c["net_income"] != st["income_statement"][year]["net_income"]:
            errors.append(f"{year}: cash flow net income differs from income statement")
        prior = st["balance_sheet"].get(str(int(year) - 1))
        if prior is not None:
            change = c["net_income"] + c["dividends_paid"] + c["share_repurchases_net"]
            if (
                prior["total_equity"] + change
                != st["balance_sheet"][year]["total_equity"]
            ):
                errors.append(f"{year}: equity does not roll forward")
    return errors


def check_statements(st: dict[str, Any]) -> list[str]:
    """Return the accounting identities that fail (empty when it all reconciles)."""
    return (
        _check_balance_sheets(st) + _check_income_statements(st) + _check_cash_flows(st)
    )
