"""Tests for fictional company specs and statements that reconcile."""

import random

import pytest

from method_1.specs import generate, sample_quantile, sample_ratios
from method_1.statements import (
    build_statements,
    check_plausibility,
    check_statements,
)


def _q(p50: float, spread: float) -> dict[str, float]:
    return {
        "p10": p50 - 2 * spread,
        "p25": p50 - spread,
        "p50": p50,
        "p75": p50 + spread,
        "p90": p50 + 2 * spread,
        "n": 50,
    }


METRICS = {
    "log10_revenue": _q(10.0, 0.3),
    "gross_margin": _q(0.45, 0.1),
    "operating_margin": _q(0.15, 0.06),
    "net_margin": _q(0.1, 0.05),
    "rd_ratio": _q(0.08, 0.03),
    "sga_ratio": _q(0.15, 0.04),
    "da_ratio": _q(0.04, 0.01),
    "capex_ratio": _q(0.05, 0.02),
    "ocf_margin": _q(0.18, 0.05),
    "tax_rate": _q(0.2, 0.03),
    "asset_turnover": _q(0.7, 0.2),
    "equity_ratio": _q(0.4, 0.1),
    "current_assets_ratio": _q(0.35, 0.08),
    "current_ratio": _q(1.4, 0.3),
    "cash_ratio": _q(0.08, 0.03),
    "ppe_ratio": _q(0.25, 0.08),
    "goodwill_ratio": _q(0.15, 0.05),
    "debt_ratio": _q(0.25, 0.08),
    "inventory_ratio": _q(0.1, 0.03),
    "receivables_ratio": _q(0.15, 0.04),
    "dividend_payout": _q(0.35, 0.1),
}
CALIBRATION = {"sectors": {"Industrials": {"n_companies": 50, "metrics": METRICS}}}


def test_sample_quantile_stays_near_the_distribution() -> None:
    """Draws have the right median and stay within the extended tails."""
    rng = random.Random(1)
    draws = sorted(sample_quantile(_q(0.0, 1.0), rng) for _ in range(4000))
    assert draws[len(draws) // 2] == pytest.approx(0.0, abs=0.1)
    assert draws[0] > -2.0 - 1.1 and draws[-1] < 2.0 + 1.1


def test_sample_ratios_leave_room_for_operating_costs() -> None:
    """Operating margin plus R&D plus SG&A never exceeds gross margin."""
    rng = random.Random(2)
    for _ in range(500):
        r = sample_ratios(METRICS, rng)
        assert (
            r["operating_margin"] + r.get("rd_ratio", 0) + r["sga_ratio"]
            <= r["gross_margin"] + 1e-3
        )


def test_generated_statements_reconcile() -> None:
    """Every generated company reconciles and has a unique fictional name."""
    specs = generate(CALIBRATION, n=50, seed=3, real_names={"ACME"})
    assert len({s["name"] for s in specs}) == 50
    for spec in specs:
        assert check_statements(spec["statements"]) == []
        income = spec["statements"]["income_statement"]["2024"]
        assert income["revenue"] == spec["revenue"]


def test_check_statements_catches_a_broken_balance_sheet() -> None:
    """Changing one line breaks the identity and is reported."""
    spec = generate(CALIBRATION, n=1, seed=4, real_names=set())[0]
    statements = build_statements(spec)
    statements["balance_sheet"]["2024"]["cash"] += 1
    errors = check_statements(statements)
    assert any("current assets" in e for e in errors)
    assert any("cash differs from the balance sheet" in e for e in errors)


def test_statements_are_plausible_and_years_differ() -> None:
    """Generated companies pass the plausibility checks; FY-1 is not a copy."""
    specs = generate(CALIBRATION, n=20, seed=6, real_names=set())
    margins = []
    for spec in specs:
        st = spec["statements"]
        assert check_plausibility(st) == []
        balance = st["balance_sheet"]["2024"]
        assert balance["other_current_assets"] <= 0.31 * balance["total_current_assets"]
        cash = st["cash_flow_statement"]["2024"]
        ppe = balance["property_plant_and_equipment"]
        assert -cash["capital_expenditures"] <= 0.41 * ppe
        cur, prev = (st["income_statement"][y] for y in ("2024", "2023"))
        margins.append(
            cur["gross_profit"] / cur["revenue"]
            - prev["gross_profit"] / prev["revenue"]
        )
    assert any(abs(m) > 0.002 for m in margins)
