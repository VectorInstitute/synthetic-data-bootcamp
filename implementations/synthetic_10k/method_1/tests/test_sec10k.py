"""Tests for the 10-K parser: text, sections, inline XBRL facts and headline metrics."""

from pathlib import Path

import pytest

from method_1 import key_financials, read_filing, split_sections
from method_1.build import FilingName, process_filing
from method_1.reader import parse_number


FILLER = "Lorem ipsum dolor sit amet. " * 200

FILING_HTML = f"""
<html><body>
<div style="display:none"><ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="FY">2024</ix:nonNumeric>
</ix:hidden><ix:resources>
<xbrli:context id="FY"><xbrli:entity><xbrli:identifier>1</xbrli:identifier></xbrli:entity>
<xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="FY_seg"><xbrli:entity><xbrli:identifier>1</xbrli:identifier>
<xbrli:segment><xbrldi:explicitMember dimension="srt:SegmentAxis">x:WidgetsMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
<xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="I"><xbrli:entity><xbrli:identifier>1</xbrli:identifier></xbrli:entity>
<xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>
</ix:resources></ix:header></div>
<p><ix:nonNumeric name="dei:EntityRegistrantName" contextRef="FY">Example&#160;Corp</ix:nonNumeric></p>
<table><tr><td>Item&#160;1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
<tr><td>Item 7.</td><td>MD&amp;A</td><td>9</td></tr>
<tr><td>Item 8.</td><td>Financial Statements</td><td>20</td></tr></table>
<p>Item 1. Business</p><p>We make widgets. {FILLER}</p>
<p>Item 1A. Risk Factors</p><p>Widgets may fail. {FILLER}</p>
<p>Item&#8201;7. Management's Discussion and Analysis</p><p>Revenue grew. {FILLER} {FILLER}</p>
<p>Item 8. Financial Statements</p>
<table><tr><td>Revenue</td><td>$<ix:nonFraction name="us-gaap:Revenues" contextRef="FY" unitRef="usd" decimals="-6" scale="6">1,250</ix:nonFraction></td></tr>
<tr><td>Widget revenue</td><td><ix:nonFraction name="us-gaap:Revenues" contextRef="FY_seg" unitRef="usd" scale="6">900</ix:nonFraction></td></tr>
<tr><td>Net loss</td><td>(<ix:nonFraction name="us-gaap:NetIncomeLoss" contextRef="FY" unitRef="usd" scale="6" sign="-">75</ix:nonFraction>)</td></tr>
<tr><td>Total assets</td><td><ix:nonFraction name="us-gaap:Assets" contextRef="I" unitRef="usd" scale="6">3,400</ix:nonFraction></td></tr>
<tr><td>Goodwill</td><td><ix:nonFraction name="us-gaap:Goodwill" contextRef="I" unitRef="usd" format="ixt:fixed-zero">—</ix:nonFraction></td></tr>
</table><p>{FILLER} {FILLER}</p>
<p>Item 9. Changes in Accountants</p><p>None.</p>
</body></html>
"""


@pytest.fixture
def filing_path(tmp_path: Path) -> Path:
    """Write the example filing with a 10K_Reports-style file name."""
    path = tmp_path / "2025-02-14_EXMP_Example_Corp_0000000001-25-000001.htm"
    path.write_text(FILING_HTML)
    return path


def test_parse_number() -> None:
    """Displayed values are scaled, signed and dash-means-zero."""
    assert parse_number("1,250", "6", None, None) == 1_250_000_000
    assert parse_number("75", "6", "-", None) == -75_000_000
    assert parse_number("—", None, None, "ixt:fixed-zero") == 0
    assert parse_number("1.234,5", None, None, "ixt:num-comma-decimal") == 1234.5
    assert parse_number("n/a", None, None, None) is None


def test_read_filing_text_and_facts(filing_path: Path) -> None:
    """Hidden header text is dropped, facts and contexts are read."""
    filing = read_filing(filing_path)
    assert not filing.text.startswith("2024")  # hidden header is not in the text
    assert filing.dei["dei:EntityRegistrantName"] == "Example Corp"
    assert filing.dei["dei:DocumentFiscalYearFocus"] == "2024"
    assert len(filing.facts) == 5
    assert filing.contexts["FY_seg"].dimensions == {
        "srt:SegmentAxis": "x:WidgetsMember"
    }
    assert "Item 7. Management's" in filing.text  # thin space normalised


def test_split_sections_skips_table_of_contents(filing_path: Path) -> None:
    """Each Item comes from the body, not the table of contents."""
    sections = split_sections(read_filing(filing_path).text)
    assert sections["1"].status == "ok"
    assert sections["1"].text.startswith("Item 1. Business\nWe make widgets.")
    assert sections["7"].status == "ok"
    assert "Item 8" not in sections["7"].text
    assert sections["1C"].status == "missing"


def test_reference_status() -> None:
    """A short Item pointing to an annual report exhibit is a reference."""
    text = "Item 7. MD&A\nInformation in response to this Item 7 can be found in the 2024 Annual Report.\nItem 8. FS\nSee above."
    sections = split_sections(text, {"7": 1000, "8": 1000})
    assert sections["7"].status == "reference"
    assert sections["8"].status == "short"


def test_item8_fallback_to_statements() -> None:
    """When Item 8 points to Item 15, the statements are found by their own heading."""
    rows = "\n".join(f"Line {i}\t1,{i:03d}\t2,{i:03d}" for i in range(20))
    text = (
        "Item 7. MD&A\nDiscussion.\nItem 8. Financial Statements\nSee Item 15.\n"
        "Item 15. Exhibits\nIndex.\nSIGNATURES\nSigned.\n"
        f"Consolidated Statements of Operations\n{rows}\nNotes. {FILLER}\nExhibit Index\nEx 1"
    )
    section = split_sections(text, {"8": 1000})["8"]
    assert section.status == "fallback"
    assert section.text.startswith("Consolidated Statements of Operations")
    assert "Exhibit Index" not in section.text


def test_key_financials(filing_path: Path) -> None:
    """Headline metrics use consolidated facts only."""
    metrics = key_financials(read_filing(filing_path))
    assert metrics["revenue"] == 1_250_000_000
    assert metrics["net_income"] == -75_000_000
    assert metrics["total_assets"] == 3_400_000_000
    assert metrics["goodwill"] == 0
    assert metrics["capex"] is None


def test_process_filing_writes_outputs(filing_path: Path, tmp_path: Path) -> None:
    """One filing produces section files, facts, metadata and index rows."""
    assert FilingName.from_path(filing_path).ticker == "EXMP"
    out = tmp_path / "out"
    result = process_filing(filing_path, out)
    filing_dir = out / "parsed" / "EXMP" / "2024"
    assert (filing_dir / "item_1a.txt").read_text().startswith("Item 1A. Risk Factors")
    assert (filing_dir / "facts.csv").read_text().count("\n") == 6
    assert {row["item"] for row in result["index"]} == {"1", "1A", "1C", "7", "7A", "8"}
    assert result["financials"]["revenue"] == 1_250_000_000
