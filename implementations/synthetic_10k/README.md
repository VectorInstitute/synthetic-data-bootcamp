# Synthetic 10-K reports

Goal: generate fictional companies' 10-K filings (business, risk factors, cybersecurity, MD&A and financial statements) whose numbers reconcile, and use them to benchmark retrieval and Q&A on equity-analysis questions against real filings.

## Step 1: parse the real filings

Input is a folder of 10-K main documents downloaded from EDGAR as `.htm` (inline XBRL), named `{filing-date}_{ticker}_{company}_{accession}.htm`.

```bash
uv run python -m aieng.syn_data.sec10k.build --input path/to/10K_Reports --output data/sec10k --workers 4
```

The parser uses only the Python standard library and takes about a minute for 305 filings (1.3 GB) on 4 CPUs. For each filing it writes:

| Output | Contents |
|---|---|
| `parsed/{ticker}/{fy}/item_{1,1a,1c,7,7a,8}.txt` | Business, Risk Factors, Cybersecurity, MD&A, Market Risk, Financial Statements |
| `parsed/{ticker}/{fy}/full_text.txt` | The whole filing as text, with tabs between table cells |
| `parsed/{ticker}/{fy}/facts.csv` | Every numeric inline XBRL fact with its period and dimensions |
| `parsed/{ticker}/{fy}/metadata.json` | Cover-page (`dei:`) facts and headline metrics |
| `index.csv` | One row per section: status, length, path |
| `financials.csv` | One row per filing: revenue, net income, assets, cash flow, ... |

Section status: `ok`; `fallback` (Item 8 only points elsewhere, so the statements were found by their own headings); `short`; `reference` (the Item points to an annual-report exhibit that is not in the download); `missing` (no Item heading found).

### Results on the 305-filing corpus (filed in 2025)

- 268 of 305 filings have usable Items 1, 1A, 7 and 8.
- Item 8: 240 found under their heading and 46 more by the fallback.
- Headline metrics: revenue for 295 filings, net income for 299, total assets for 300.
- Known gaps: GE, Honeywell, Intel, Illumina, McDonald's, Citigroup, Morgan Stanley and Cardinal Health organise the body by their own headings with a cross-reference index, so Items can't be cut by heading. U.S. Bancorp, Wells Fargo, BNY and IBM put MD&A and the financial statements in an EX-13 exhibit that isn't in the download.

## Next steps

See the project plan: calibrate per-sector distributions from `financials.csv`, sample fictional company specs, compute reconciling statements in code, then have an LLM write the narrative sections around those numbers, validate, and evaluate retrieval on synthetic vs real filings.
