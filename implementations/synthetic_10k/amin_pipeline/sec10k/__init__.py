"""Parse real SEC 10-K filings into sections and XBRL facts."""

from sec10k.financials import fiscal_period_end, key_financials
from sec10k.reader import Context, Fact, ParsedFiling, read_filing
from sec10k.sections import DEFAULT_ITEMS, Section, split_sections
