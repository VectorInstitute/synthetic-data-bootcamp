"""Pick headline financial figures out of a filing's inline XBRL facts.

Companies tag the same line item with different us-gaap concepts, so each metric
lists its candidates in order of preference. Only consolidated facts (no
dimensions) for the filing's own fiscal year are used.
"""

from __future__ import annotations

from datetime import date

from method_1.reader import ParsedFiling


# Flow metrics: reported over the fiscal year (a duration context).
DURATION_METRICS: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenuesNetOfInterestExpense",
    ],
    "cost_of_revenue": [
        "us-gaap:CostOfRevenue",
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold",
    ],
    "gross_profit": ["us-gaap:GrossProfit"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "pretax_income": [
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax": ["us-gaap:IncomeTaxExpenseBenefit"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "rd_expense": ["us-gaap:ResearchAndDevelopmentExpense"],
    "sga_expense": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "depreciation_amortization": [
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationAmortizationAndAccretionNet",
        "us-gaap:DepreciationAndAmortization",
    ],
    "operating_cash_flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends_paid": [
        "us-gaap:PaymentsOfDividends",
        "us-gaap:PaymentsOfDividendsCommonStock",
    ],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
    "net_interest_income": ["us-gaap:InterestIncomeExpenseNet"],
}

# Stock metrics: reported at the fiscal year end (an instant context).
INSTANT_METRICS: dict[str, list[str]] = {
    "total_assets": ["us-gaap:Assets"],
    "current_assets": ["us-gaap:AssetsCurrent"],
    "total_liabilities": ["us-gaap:Liabilities"],
    "current_liabilities": ["us-gaap:LiabilitiesCurrent"],
    "stockholders_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "liabilities_and_equity": ["us-gaap:LiabilitiesAndStockholdersEquity"],
    "cash": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "ppe_net": ["us-gaap:PropertyPlantAndEquipmentNet"],
    "goodwill": ["us-gaap:Goodwill"],
    "long_term_debt": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"],
    "inventory": ["us-gaap:InventoryNet"],
    "receivables": ["us-gaap:AccountsReceivableNetCurrent"],
    "loans": [
        "us-gaap:LoansAndLeasesReceivableNetReportedAmount",
        "us-gaap:FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
    ],
    "deposits": ["us-gaap:Deposits"],
    "employees": ["dei:EntityNumberOfEmployees"],
}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days


def fiscal_period_end(filing: ParsedFiling) -> str | None:
    """Return the fiscal year-end date, read from the cover page's context."""
    for key in (
        "dei:DocumentPeriodEndDate#context",
        "dei:DocumentFiscalYearFocus#context",
        "dei:DocumentType#context",
    ):
        ctx = filing.contexts.get(filing.dei.get(key, ""))
        if ctx is not None and ctx.period_end:
            return ctx.period_end[:10]
    return None


def key_financials(filing: ParsedFiling) -> dict[str, float | None]:
    """Return headline metrics for the filing's fiscal year, ``None`` when untagged."""
    period_end = fiscal_period_end(filing)
    by_concept: dict[str, list[tuple[str, float]]] = {}
    for fact in filing.facts:
        ctx = filing.contexts.get(fact.context_id)
        if (
            fact.value is None
            or ctx is None
            or ctx.dimensions
            or ctx.period_end is None
        ):
            continue
        if period_end and ctx.period_end[:10] != period_end:
            continue
        if ctx.instant:
            kind = "instant"
        elif ctx.start and ctx.end and 330 <= _days(ctx.start, ctx.end) <= 380:
            kind = "year"
        else:
            continue
        by_concept.setdefault(fact.concept, []).append((kind, fact.value))

    def pick(concepts: list[str], kind: str) -> float | None:
        for concept in concepts:
            values = [v for k, v in by_concept.get(concept, []) if k == kind]
            if values:
                return values[0]
        return None

    result: dict[str, float | None] = {
        name: pick(c, "year") for name, c in DURATION_METRICS.items()
    }
    result.update({name: pick(c, "instant") for name, c in INSTANT_METRICS.items()})
    return result
