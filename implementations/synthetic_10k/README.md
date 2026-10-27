# Synthetic 10-K reports (team BMO-1)

Two pipelines that build fictional companies' 10-K filings, and shared evals that check each pipeline's output with both pipelines' validators. The pipelines were built in parallel, use different data and methods, and are kept separate so either can be run, compared, or reused.

```
synthetic_10k/
  README.md            this overview
  amin_pipeline/       Amin's pipeline: the sec10k package, its tests and README
  dylan_pipeline/      Dylan's pipeline, self-contained (src/step00..step09, data/, PLAN.md)
  evals/               cross-pipeline validation scripts, sample inputs and results
```

## How the two pipelines differ

| | Amin's pipeline ([README](amin_pipeline/README.md)) | Dylan's pipeline ([README](dylan_pipeline/README.md)) |
|---|---|---|
| Real filings | 305 S&P 500 10-Ks filed in 2025 (mostly FY2024), main document only, from a local download | 50 S&P 500 companies sampled at random, FY2023 to FY2025, downloaded with edgartools, including EX-13 exhibits |
| Sections parsed | Items 1, 1A, 1C, 7, 7A, 8, plus every inline XBRL fact in the filing | Items 1, 1A, 1C, 7, 8; XBRL facts from SEC's companyfacts API |
| Calibration | Per GICS sector: percentiles of about 20 ratios and of section lengths | Generic across sectors: percentiles of 6 totals and 6 ratios |
| Fictional company | Ratios sampled from its sector's percentiles | One real company-year as a template, scaled by a random size factor |
| Statements | Two years (current and prior), income statement, balance sheet, cash flow | One year, line items from the template company's own ratios |
| Narrative | Written by an LLM through Vector's proxy (Gemini by default), with checks and up to 3 retries per section | Written in a Claude Code session; the API script (`step08`) is built but has not been run |
| Output | Items 1 to 16 as text, plus an EDGAR-style HTML and PDF filing | Items 1, 1A, 1C, 7, 8 as text (condensed by design) |
| Built-in checks | `validate.py`, run on each section during generation | `step09_validate_filings.py`, run on the finished filings |

## Evals

[`evals/`](evals/README.md) runs each pipeline's validator on the other pipeline's filings, without changing either validator.

| Validator | Run on | Checks | Pass | Fail | Result |
|---|---|---|---|---|---|
| Dylan's step09 | his own 20 companies | 360 | 260 (+100 info) | 0 | [report](dylan_pipeline/data/validation_report.csv) |
| Dylan's step09 | Amin's 2 showcase filings | 36 | 22 (+10 info) | 4 | [report](evals/results/phase7_on_amin_showcase/validation_report.csv) |
| Amin's checks | Dylan's 20 companies | 400 (+100 skipped) | 375 | 25 | [report](evals/results/amin_checks_on_dylan20/validation_report.csv) |

None of the failures is a defect in a filing. Each one comes from a difference in what the two validators expect:

- **Dylan's checks on Amin's filings:** Dylan's Item 8 heading pattern accepts only "Statement of Operations", not the plural that Amin's filings use (2 failures). His number check expects single-year filings, so it flags year-over-year changes and prior-year margins in MD&A (2 failures, 17 figures). Every figure in Item 8 traces to the statements.
- **Amin's checks on Dylan's filings:** Amin's number check allows fewer derived figures. Every figure behind the 25 failures is one that Dylan's check accepts, such as a sum of lines (total debt) or a ratio Amin's check doesn't compute. No real company names, leaked variable names or missing company names were found.
- The exemplar-overlap check (copying from the real template filing) was skipped for Dylan's filings, because his parsed real filings are git-ignored. It runs automatically where `dylan_pipeline/data/parsed/` exists.
