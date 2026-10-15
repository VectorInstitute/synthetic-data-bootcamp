"""Parse real SEC 10-K filings into sections and XBRL facts."""

from aieng.syn_data.sec10k.financials import fiscal_period_end, key_financials
from aieng.syn_data.sec10k.reader import Context, Fact, ParsedFiling, read_filing
from aieng.syn_data.sec10k.sections import DEFAULT_ITEMS, Section, split_sections
