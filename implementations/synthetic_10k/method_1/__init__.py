"""Parse real SEC 10-K filings into sections and XBRL facts."""

from method_1.financials import fiscal_period_end, key_financials
from method_1.reader import Context, Fact, ParsedFiling, read_filing
from method_1.sections import DEFAULT_ITEMS, Section, split_sections
