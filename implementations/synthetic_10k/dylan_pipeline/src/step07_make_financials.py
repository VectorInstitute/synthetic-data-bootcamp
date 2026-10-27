"""Compute reconciling financial statements for each fictional company (Phase 5).

Input: data/synthetic/{fictional_id}/{fy}/spec.json (run src.step06_make_specs first),
the matching template record in data/calibration.json, and that template company's
data/companyfacts/{ticker}.json. Output:
data/synthetic/{fictional_id}/{fy}/financials.json: a balance sheet, income
statement, and cash-flow statement, every one reconciling exactly to spec.json's
totals.

Code builds every number here; nothing is left for a language model to invent later
(Phase 6 writes narrative text only, over numbers that are already fixed). Each line
item is the template's own ratio (that line divided by its own total, from the same
XBRL data Phase 3 and 4 already use) applied to the fictional company's total, so the
balance-sheet mix, expense structure, and cash-flow shape stay the kind of company the
spec is grounded in. Whatever a ratio basis can't explain (a tag the template doesn't
report, or simply the rest) becomes one named "other" line per statement, which also
forces each statement to reconcile exactly rather than approximately.

Run from the project directory:
    python -m src.step07_make_financials
"""

import json
import random
from datetime import datetime, timezone

from src.config import COMPANYFACTS_DIR, RAW_DIR, SYNTHETIC_DIR
from src.step05_make_calibration import annual_value
from src.utils import setup_logging, write_atomic


logger = setup_logging("make_financials")

CALIBRATION_PATH = RAW_DIR.parent / "calibration.json"
SEED = 42
RATIO_JITTER_RANGE = (0.85, 1.15)
# Used only when a template has no operating-income anchor at all (see
# build_income_statement): the U.S. federal statutory rate, and the band an
# expense-based pretax estimate's implied tax rate must fall in to be trusted
# instead of that statutory assumption.
STATUTORY_TAX_RATE = 0.21
SANE_TAX_RATE_RANGE = (-0.3, 0.5)

# name -> candidate XBRL tags, tried in order (the same line item is tagged
# differently across filers; see src.step05_make_calibration).
ASSET_TAGS = {
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "accounts_receivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves", "InventoryFinishedGoods"],
    "property_plant_equipment_net": ["PropertyPlantAndEquipmentNet"],
    "goodwill": ["Goodwill"],
    "intangible_assets_net": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
}
LIABILITY_TAGS = {
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent"],
    "accrued_liabilities": ["AccruedLiabilitiesCurrent"],
    "short_term_debt": ["DebtCurrent", "LongTermDebtCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
EQUITY_TAGS = {
    "additional_paid_in_capital": ["AdditionalPaidInCapitalCommonStock", "AdditionalPaidInCapital"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "accumulated_other_comprehensive_income": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
}
OPERATING_EXPENSE_TAGS = {
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "selling_general_administrative": ["SellingGeneralAndAdministrativeExpense"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
}
CASH_FLOW_LINE_TAGS = {
    "depreciation_and_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "capital_expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends_paid": ["PaymentsOfDividends"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
}
CASH_FLOW_SUBTOTAL_TAGS = {
    "cash_from_operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash_from_investing": ["NetCashProvidedByUsedInInvestingActivities"],
    "cash_from_financing": ["NetCashProvidedByUsedInFinancingActivities"],
}


def load_companyfacts(ticker: str) -> dict | None:
    """Return the template's cached companyfacts JSON, or None if not downloaded."""
    path = COMPANYFACTS_DIR / f"{ticker}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_calibration_templates() -> dict[tuple[str, int], dict]:
    """Return {(ticker, fiscal_year): metrics}, calibration.json's company_years.

    spec.json's own "template" field is only {ticker, fiscal_year} (a reference, not
    the metrics Phase 4 built the spec's financials from); this looks those metrics
    back up so the same total_assets, revenues, and ratios are used here.
    """
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {(row["ticker"], row["fiscal_year"]): row for row in data["company_years"]}


def template_tag_value(companyfacts: dict, year: int, tags: list[str]) -> float | None:
    """Return the first of `tags` with an annual value for `year` in the template."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag in gaap:
            value = annual_value(gaap[tag], year)
            if value is not None:
                return value
    return None


def scaled_line_items(
    rng: random.Random,
    companyfacts: dict,
    year: int,
    tags_by_name: dict[str, list[str]],
    *,
    template_total: float,
    fictional_total: float,
) -> dict[str, int]:
    """Scale each template ratio (tag / template_total) onto fictional_total.

    A name is only included if the template reports one of its tags; everything
    else is left for the caller's "other" plug to absorb.
    """
    items = {}
    if not template_total:
        return items
    for name, tags in tags_by_name.items():
        value = template_tag_value(companyfacts, year, tags)
        if value is None:
            continue
        ratio = (value / template_total) * rng.uniform(*RATIO_JITTER_RANGE)
        items[name] = round(fictional_total * ratio)
    return items


def with_plug(items: dict[str, int], total: int, plug_name: str) -> dict[str, int]:
    """Add one more line, equal to what's left, so `items` sums to exactly `total`."""
    return {**items, plug_name: total - sum(items.values())}


def build_balance_sheet(rng: random.Random, companyfacts: dict, template: dict, financials: dict) -> dict:
    """Build a balance sheet whose three sections sum exactly to `financials`'s totals.

    Every line's ratio basis is the template's own total assets (always available,
    unlike total liabilities; see src.step05_make_calibration's coverage notes), so
    the fictional balance sheet keeps the same relative shape before each section's
    plug reconciles it to the fictional company's own, already-fixed totals.
    """
    fy = template["fiscal_year"]
    assets_total = financials["total_assets"]
    liabilities_total = financials["total_liabilities"]
    equity_total = financials["stockholders_equity"]
    template_assets = template["total_assets"]

    assets = scaled_line_items(
        rng,
        companyfacts,
        fy,
        ASSET_TAGS,
        template_total=template_assets,
        fictional_total=assets_total,
    )
    assets = with_plug(assets, assets_total, "other_assets")

    liabilities = scaled_line_items(
        rng,
        companyfacts,
        fy,
        LIABILITY_TAGS,
        template_total=template_assets,
        fictional_total=assets_total,
    )
    liabilities = with_plug(liabilities, liabilities_total, "other_liabilities")

    equity = scaled_line_items(
        rng,
        companyfacts,
        fy,
        EQUITY_TAGS,
        template_total=template_assets,
        fictional_total=assets_total,
    )
    equity = with_plug(equity, equity_total, "other_equity")

    return {
        "assets": {**assets, "total_assets": assets_total},
        "liabilities": {**liabilities, "total_liabilities": liabilities_total},
        "equity": {**equity, "total_stockholders_equity": equity_total},
    }


def build_income_statement(rng: random.Random, companyfacts: dict, template: dict, financials: dict) -> dict:
    """Build an income statement tying exactly to `financials`'s income figures.

    Expense line items are the template's own ratio to revenue; an
    "other_operating_expenses" line absorbs the rest, sized so expenses plus it
    equal operating income, when the template reports one. Income tax is always
    the residual between pretax income and net income, rather than ratio-based, so
    net income reconciles exactly either way.
    """
    fy = template["fiscal_year"]
    revenue = financials["revenues"]
    template_revenue = template.get("revenues")
    net_income = financials["net_income"]
    operating_income = financials.get("operating_income")

    expenses = scaled_line_items(
        rng,
        companyfacts,
        fy,
        OPERATING_EXPENSE_TAGS,
        template_total=template_revenue,
        fictional_total=revenue,
    )
    if operating_income is not None:
        expenses = with_plug(expenses, revenue - operating_income, "other_operating_expenses")
        pretax_base = operating_income
    else:
        pretax_base = revenue - sum(expenses.values())

    interest_expense = 0
    if template_revenue:
        template_interest = template_tag_value(companyfacts, fy, ["InterestExpense"])
        if template_interest is not None:
            interest_expense = round(
                revenue * (template_interest / template_revenue) * rng.uniform(*RATIO_JITTER_RANGE),
            )
            # A bank's "interest expense" tag isn't on the same basis as the generic
            # revenue metric calibration uses, and can come out larger than revenue
            # itself; cap it so it can't single-handedly make pretax income nonsensical.
            interest_expense = min(interest_expense, round(revenue * 0.6))

    pretax_income = pretax_base - interest_expense
    if operating_income is None:
        implied_rate = (pretax_income - net_income) / pretax_income if pretax_income else None
        if implied_rate is None or not SANE_TAX_RATE_RANGE[0] <= implied_rate <= SANE_TAX_RATE_RANGE[1]:
            # No operating-income anchor from the template, and few or no expense
            # tags were captured either (common for banks, whose "cost of revenue"
            # isn't tagged the way an industrial company's is): the expense-based
            # estimate above left pretax wildly over- or under-stated, which would
            # otherwise force an unreasonable "tax expense" to make net income
            # reconcile. Assume the U.S. federal statutory rate instead.
            pretax_income = round(net_income / (1 - STATUTORY_TAX_RATE))
            expenses = with_plug(expenses, revenue - interest_expense - pretax_income, "other_operating_expenses")

    income_tax_expense = pretax_income - net_income

    statement = {"revenues": revenue, "operating_expenses": expenses}
    if operating_income is not None:
        statement["operating_income"] = operating_income
    statement["interest_expense"] = interest_expense
    statement["pretax_income"] = pretax_income
    statement["income_tax_expense"] = income_tax_expense
    statement["net_income"] = net_income
    return statement


def build_cash_flow_statement(rng: random.Random, companyfacts: dict, template: dict, financials: dict) -> dict:
    """Build a cash-flow statement shaped by the template's own ratios to revenue.

    Depreciation, capex, dividends, and buybacks are each the template's ratio to
    revenue; each of the three subtotals (operating, investing, financing) is
    separately targeted the same way, with one named "other" line per section
    absorbing whatever the explicit lines don't reach, so the subtotal always hits
    its target exactly.
    """
    fy = template["fiscal_year"]
    revenue = financials["revenues"]
    template_revenue = template.get("revenues")
    net_income = financials["net_income"]

    lines = scaled_line_items(
        rng,
        companyfacts,
        fy,
        CASH_FLOW_LINE_TAGS,
        template_total=template_revenue,
        fictional_total=revenue,
    )
    da = lines.get("depreciation_and_amortization", 0)
    capex = -abs(lines.get("capital_expenditures", 0))
    dividends = -abs(lines.get("dividends_paid", 0))
    buybacks = -abs(lines.get("share_repurchases", 0))

    def target_subtotal(tags: list[str]) -> int | None:
        if not template_revenue:
            return None
        value = template_tag_value(companyfacts, fy, tags)
        if value is None:
            return None
        return round(revenue * (value / template_revenue) * rng.uniform(*RATIO_JITTER_RANGE))

    operating = {"net_income": net_income, "depreciation_and_amortization": da}
    cfo_target = target_subtotal(CASH_FLOW_SUBTOTAL_TAGS["cash_from_operations"])
    operating = (
        with_plug(operating, cfo_target, "changes_in_working_capital_and_other")
        if cfo_target is not None
        else operating
    )
    operating.setdefault("changes_in_working_capital_and_other", 0)

    investing = {"capital_expenditures": capex}
    cfi_target = target_subtotal(CASH_FLOW_SUBTOTAL_TAGS["cash_from_investing"])
    investing = with_plug(investing, cfi_target, "other_investing_activities") if cfi_target is not None else investing
    investing.setdefault("other_investing_activities", 0)

    financing = {"dividends_paid": dividends, "share_repurchases": buybacks}
    cff_target = target_subtotal(CASH_FLOW_SUBTOTAL_TAGS["cash_from_financing"])
    financing = with_plug(financing, cff_target, "other_financing_activities") if cff_target is not None else financing
    financing.setdefault("other_financing_activities", 0)

    cash_from_operations = sum(operating.values())
    cash_from_investing = sum(investing.values())
    cash_from_financing = sum(financing.values())

    return {
        "operating_activities": {**operating, "cash_from_operations": cash_from_operations},
        "investing_activities": {**investing, "cash_from_investing": cash_from_investing},
        "financing_activities": {**financing, "cash_from_financing": cash_from_financing},
        "net_change_in_cash": cash_from_operations + cash_from_investing + cash_from_financing,
    }


def _lines_sum_to_total(section: dict, total_key: str) -> bool:
    """Return True if every other entry sums to the one named `total_key`."""
    lines = {k: v for k, v in section.items() if k != total_key}
    return sum(lines.values()) == section[total_key]


def check_reconciliation(statements: dict, financials: dict) -> list[str]:
    """Return a description of every subtotal that doesn't add up, if any."""
    problems = []
    bs = statements["balance_sheet"]
    if not _lines_sum_to_total(bs["assets"], "total_assets"):
        problems.append("balance_sheet.assets does not sum to total_assets")
    if not _lines_sum_to_total(bs["liabilities"], "total_liabilities"):
        problems.append("balance_sheet.liabilities does not sum to total_liabilities")
    if not _lines_sum_to_total(bs["equity"], "total_stockholders_equity"):
        problems.append("balance_sheet.equity does not sum to total_stockholders_equity")
    if (
        bs["assets"]["total_assets"]
        != bs["liabilities"]["total_liabilities"] + bs["equity"]["total_stockholders_equity"]
    ):
        problems.append("balance_sheet: assets != liabilities + equity")
    income = statements["income_statement"]
    if income["pretax_income"] - income["income_tax_expense"] != income["net_income"]:
        problems.append("income_statement: pretax_income - tax != net_income")
    return problems


def build_financials(rng: random.Random, spec: dict, template: dict, companyfacts: dict) -> dict:
    """Build and verify all three statements for one fictional company's spec.

    `template` is the full calibration.json record for spec["template"]'s
    (ticker, fiscal_year), not spec["template"] itself (see load_calibration_templates).
    """
    financials = spec["financials"]
    statements = {
        "balance_sheet": build_balance_sheet(rng, companyfacts, template, financials),
        "income_statement": build_income_statement(rng, companyfacts, template, financials),
        "cash_flow_statement": build_cash_flow_statement(rng, companyfacts, template, financials),
    }
    problems = check_reconciliation(statements, financials)
    if problems:
        raise RuntimeError(f"{spec['fictional_id']}: {'; '.join(problems)}")
    return statements


def main() -> None:
    """Build financials.json next to every data/synthetic/*/*/spec.json."""
    rng = random.Random(SEED)
    spec_paths = sorted(SYNTHETIC_DIR.glob("*/*/spec.json"))
    if not spec_paths:
        logger.warning("No specs found under %s; run src.step06_make_specs first", SYNTHETIC_DIR)
        return
    templates = load_calibration_templates()

    written = 0
    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        ticker = spec["template"]["ticker"]
        fiscal_year = spec["template"]["fiscal_year"]
        template = templates.get((ticker, fiscal_year))
        if template is None:
            logger.warning(
                "%s: no calibration record for template %s FY%d; skipping",
                spec["fictional_id"],
                ticker,
                fiscal_year,
            )
            continue
        companyfacts = load_companyfacts(ticker)
        if companyfacts is None:
            logger.warning("%s: no companyfacts.json for template %s; skipping", spec["fictional_id"], ticker)
            continue

        statements = build_financials(rng, spec, template, companyfacts)
        statements["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_path = spec_path.parent / "financials.json"
        write_atomic(out_path, json.dumps(statements, indent=2))
        written += 1
        logger.info(
            "%s: balance sheet, income statement, and cash flow written and reconciled",
            spec["fictional_id"],
        )

    logger.info("Wrote financials for %d/%d specs", written, len(spec_paths))


if __name__ == "__main__":
    main()
