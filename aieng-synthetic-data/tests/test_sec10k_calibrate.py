"""Tests for per-sector calibration of fictional company specs."""

import pytest

from aieng.syn_data.sec10k.calibrate import (
    build_calibration,
    company_ratios,
    load_sectors,
)


def _row(ticker: str, revenue: float, gross: float, net: float) -> dict[str, str]:
    return {
        "ticker": ticker,
        "revenue": str(revenue),
        "gross_profit": str(gross),
        "net_income": str(net),
        "total_assets": str(revenue * 2),
        "capex": str(-revenue * 0.05),
        "cost_of_revenue": "",
    }


def test_company_ratios() -> None:
    """Ratios divide the right columns and skip untagged ones."""
    ratios = company_ratios(_row("X", 1000, 400, 100))
    assert ratios["gross_margin"] == pytest.approx(0.4)
    assert ratios["net_margin"] == pytest.approx(0.1)
    assert ratios["asset_turnover"] == pytest.approx(0.5)
    assert ratios["capex_ratio"] == pytest.approx(0.05)  # sign normalised
    assert ratios["log10_revenue"] == pytest.approx(3)
    assert "rd_ratio" not in ratios


def test_gross_margin_from_cost_of_revenue() -> None:
    """Filers without a gross profit tag get it from cost of revenue."""
    ratios = company_ratios({"ticker": "X", "revenue": "100", "cost_of_revenue": "70"})
    assert ratios["gross_margin"] == pytest.approx(0.3)


def test_build_calibration_excludes_financials_and_pools_small_sectors() -> None:
    """Financials are dropped; a sector below the minimum size uses pooled quantiles."""
    sectors = {f"T{i}": "Information Technology" for i in range(10)}
    sectors.update({"U0": "Utilities", "U1": "Utilities", "B0": "Financials"})
    financials = [_row(t, 1000 * (i + 1), 500, 100) for i, t in enumerate(sectors)]
    index = [{"ticker": "T0", "item": "1", "status": "ok", "words": "5000"}] * 3
    cal = build_calibration(financials, index, sectors)
    assert set(cal["sectors"]) == {"Information Technology", "Utilities"}
    assert cal["n_companies"] == 12
    assert cal["sectors"]["Information Technology"]["metrics"]["net_margin"]["n"] == 10
    assert cal["sectors"]["Utilities"]["metrics"]["net_margin"]["pooled"] is True
    assert cal["section_words"]["1"]["p50"] == 5000


def test_bundled_sectors() -> None:
    """The bundled sector table covers the downloaded tickers."""
    sectors = load_sectors()
    assert sectors["AAPL"] == "Information Technology"
    assert sectors["JPM"] == "Financials"
    assert len(sectors) == 305
