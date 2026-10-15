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

## Step 2: calibrate fictional companies

```bash
uv run python -m aieng.syn_data.sec10k.calibrate --data data/sec10k
```

Writes `data/sec10k/calibration.json`. It holds, per GICS sector (from the bundled `sectors.csv`), the 10th/25th/50th/75th/90th percentiles of:
- size (`log10_revenue`)
- the ratios that shape a set of statements: margins, R&D/SG&A/D&A/capex intensity, tax rate, asset turnover, equity, cash, PP&E, goodwill and debt shares of assets, working-capital ratios and dividend payout
- section lengths in words for each Item

Financial companies are excluded, since their statements use a different layout. A ratio that fewer than 8 companies in a sector report falls back to the pooled distribution and is marked `"pooled": true`. Utilities' gross margin is one example, because utilities don't report gross profit.

On the 305-filing corpus this covers 254 non-financial companies in 10 sectors. Pooled medians: revenue about $21B (these are S&P 500 companies), gross margin 49%, operating margin 17%, net margin 12%, tax rate 19.5%.

## Step 3: fictional companies and their statements

```bash
uv run python -m aieng.syn_data.sec10k.specs --data data/sec10k --n 20 --seed 0
```

Writes `data/sec10k/synthetic/specs/syn0001.json` and so on. Each file holds one fictional non-financial company:
- an invented name and ticker, checked against real tickers and real company-name words
- its sector, size, employees and growth
- the ratios sampled from `calibration.json`
- two years of statements in USD millions: income statement, balance sheet, and cash flow statement

All numbers are computed in code, so the language model only ever writes narrative around fixed figures. Sampled ratios are made consistent with each other: operating costs fit inside gross profit, and the parts of assets fit inside the totals. Lines that aren't modelled ("other") absorb the rounding residuals. Every company must pass `check_statements` or it is resampled. The checks are:
- assets = liabilities + equity
- the current assets add up
- gross profit and operating income tie
- cash rolls forward and matches the balance sheet
- net income flows into the cash flow statement
- equity rolls forward by net income less dividends and buybacks
- no negative cost or balance-sheet lines

## Step 4: write the narrative with an LLM

```bash
# Check which model ids the endpoint serves, then preview the prompts (no API calls)
uv run python -m aieng.syn_data.sec10k.narrative --data data/sec10k --list-models
uv run python -m aieng.syn_data.sec10k.narrative --data data/sec10k --limit 2 --dry-run

# Pilot: two companies with Gemini Flash
uv run python -m aieng.syn_data.sec10k.narrative --data data/sec10k --limit 2
```

The API comes from `OPENAI_BASE_URL` and `OPENAI_API_KEY`, which are already set in the Vector Coder workspaces. The default model is `gemini-2.5-flash`; pass `--model` if the endpoint names it differently.

For each spec the model writes Items 1, 1A, 7 and 1C in that order. Each prompt holds:
- the facts sheet: the company profile, the three statements and derived percentages (margins, growth, tax rate), so the model never does arithmetic
- the Items already written, so later sections stay consistent
- an excerpt of the same Item from a real company in the same sector, as a style example (read from `parsed/`)
- a length target sampled from the real section lengths in `calibration.json`, capped at 3,000 to 3,500 words (900 for Item 1C) to keep a pilot cheap; `--length-scale` scales it

Item 8 is rendered from the statements by code. Each section is then validated, and regenerated once with the problems as feedback if it fails. The checks are:
- the Item heading is there and the length is 60% to 150% of the target
- every dollar amount matches a statement line or a year-over-year change, and in Item 7 every percentage matches a derived metric
- no real filer is named (names come from `index.csv`)
- less than 5% of the text's 8-word sequences are copied from the style example
- the fictional company is named in Items 1 and 7

Output goes to `data/sec10k/synthetic/filings/{company_id}/`: `item_*.txt`, `filing.txt` (all Items under a cover page), the exact `prompts/` sent, and `metrics.json`. Finished companies are skipped on the next run unless you pass `--overwrite`.

### Metrics

Every call records its wall-clock time, prompt, completion, reasoning and total tokens, model, finish reason, word count, attempt number and validation issues. `metrics.json` holds these per call, per section and as company totals. Each run also writes `synthetic/runs/{run_id}.json` (config, per-company totals, grand totals and errors) and appends one line of totals to `synthetic/runs.jsonl`, so runs can be compared over time.

## Next steps

- Run the two-company pilot, read the filings and tune the prompts and checks.
- Scale to the 20 specs, then to a few hundred.
- Downstream evaluation: retrieval over synthetic versus real filings.
