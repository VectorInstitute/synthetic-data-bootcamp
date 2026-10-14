# Synthetic 10-K Dataset: Plan and Open Questions

Goal: build a synthetic 10-K dataset for evaluating RAG systems on equity-analysis questions.
Real SEC 10-K filings serve three roles: style templates, calibration for realistic numbers, and
a real-data benchmark.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Project setup, config, identity handling | Done |
| 2 | Download 10-K filings from EDGAR | Done: 50 companies x FY2023-2025 (150 company-years, 147 filings, 3 not found) |
| 3 | Parse filings into sections (Items 1, 1A, 1C, 7, 8) | Done: 724/735 sections ok, 5 short, 6 missing; see `data/parse_status.csv` |
| 4 | Metadata index (`data/index.csv`) | Done: 729 rows |
| 5 | XBRL companyfacts JSON per company | Done: 50/50 companies |
| 6 | Summary script | Not started |
| - | Calibration (`data/calibration.json`) | Done: 147 company-years, 12 metrics; see "Calibration profile" |
| - | Company specs (Phase 4) | Done: 20 fictional companies; see `data/synthetic/` |
| - | Financial statements (Phase 5) | Done: balance sheet, income statement, cash flow per company, all reconciling |
| - | Narrative generation (Phase 6) | M4 pilot done: 2/20 companies; see "Phase 6" |
| - | QA, retrieval evaluation | Not started |

### Downloaded data

- Location: `data/raw/{ticker}/{fy}/`, with `10k.htm` (the 10-K as filed), `10k.txt` (readable text),
  `ex13_*.htm` (annual report exhibit, when present), and `metadata.json`.
- Companies: 50 S&P 500 companies sampled at random (`python -m src.step00_sample_companies`,
  `random.Random(42).sample` over `data/sp500_constituents.csv`, a frozen snapshot). Citigroup is
  excluded from the sampling pool by name: its 10-K body has no plain-text Item headings at all
  (see "Parsed sections"), so this pipeline can't parse it. This replaced an earlier list of 15
  hand-picked US banks plus 35 random companies; see "Leftover folders" below for what that left
  behind in `data/raw/`.
- 3 company-years have no 10-K on EDGAR and are skipped (logged `not_found` in
  `download_status.csv`): RDDT FY2023 (IPO'd 2024), SW FY2023 (formed by a 2024 merger), and
  TXT FY2025 (its FY2025 10-K's period of report is dated just after New Year's Day, in the next
  calendar year, so it doesn't match fiscal_year's "starts with this year" check — a labeling
  edge case in `find_10k`, not a missing filing).
- EX-13 annual reports are present for PGR, USB, and WFC (all three years each). Most other filings
  keep MD&A and financials in the 10-K body.
- Reproduce: `python -m src.step01_download_10k` (skips anything already downloaded).

### Parsed sections

- Location: `data/parsed/{ticker}/{fy}/item_1.txt`, `item_1a.txt`, `item_1c.txt`, `item_7.txt`,
  `item_8.txt`. See `SECTIONS.md` for which items these are and why.
- Result over the 147 downloaded filings x 5 items (735 sections; the 3 not-found company-years
  have no file to parse): 724 ok, 5 short, 6 missing. Every row's status, character count, and
  extraction method are in `data/parse_status.csv`. See `SECTIONS.md` for what each remaining
  short/missing row is and why it's not worth chasing further.
- Reproduce: `python -m src.step02_parse_sections` (always reparses; it's local and fast, no need to skip).

### Calibration profile

- `data/companyfacts/{ticker}.json`: the raw SEC XBRL companyfacts response per company (about
  180 MB total; git-ignored, reproduce with `python -m src.step04_fetch_companyfacts`).
- `data/calibration.json`: 147 company-years (every downloaded filing), 6 generic cross-sector
  metrics (total assets, total liabilities, stockholders' equity, revenues, net income, operating
  income) plus 6 derived ratios (net/operating margin, return on equity/assets, equity/liabilities
  to assets), each with its percentile distribution and the per-company-year values it's drawn
  from. Not every company reports every tag (e.g. 111/147 report a total `Liabilities` figure,
  107/147 report `OperatingIncomeLoss`); `data/calibration.json`'s `missing` field says which
  company-years lack which metric. Reproduce with `python -m src.step05_make_calibration`.
- Resolves open question 4 (below): calibration is generic across sectors, not bank-specific, to
  match the random S&P 500 company list.

## Plan

### Phase 1: Parse filings into sections (step 3) — done

- Input: `10k.txt`. Output as above.
- The table of contents repeats every Item heading. The body occurrence is the heading followed
  by the most text before the next heading, which also makes combined headings like
  `Items 1 and 2. Business and Properties` resolve to Item 1.
- When any item is only a cross-reference instead of written out, the heading named in the
  cross-reference sentence (e.g. "under 'Financial Review'") is located in the filing's EX-13
  exhibit if it has one (PGR, USB, WFC), or else elsewhere in the 10-K body itself — large banks
  and insurers often write Item 7's real content under an unnumbered heading instead of the
  numbered one, sometimes repeated as a running header on every page.
- A heading repeated as a running header (same detection as above, just without a cross-reference
  to trigger it) is unioned across every page it owns instead of returning just one page of it.
- Item 1C exists only from FY2023 onward (and only for December fiscal year-ends at that; see
  `SECTIONS.md` for the known gaps this leaves).
- Missing or suspiciously short sections are still written and flagged with their status in
  `data/parse_status.csv` for manual review, rather than guessed at further.

### Phase 2: Index and XBRL facts (steps 4 and 5) — done

- `data/index.csv`: one row per section, with ticker, company name, CIK, fiscal year, filing date,
  accession number, item, path, character count, and word count. `python -m src.step03_build_index`.
- XBRL companyfacts: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`, one file per
  company, fetched through edgartools' `get_with_retry` so it uses the same identity and rate
  limit. `python -m src.step04_fetch_companyfacts`.

### Phase 3: Calibration profile — done, generic (resolves open question 4)

- Decided: calibration is generic across sectors, not bank-specific — the company list is now a
  random S&P 500 sample (only KEY, PGR, and USB are incidentally banks), so fictional companies
  are generic companies from here on, not fictional banks. See "Calibration profile" above for
  what was actually built: `python -m src.step05_make_calibration`, from the metrics and ratios in
  `src/step05_make_calibration.py`'s `METRIC_TAGS` and `DERIVED_RATIOS` rather than bank-specific ones
  like CET1 or net interest margin.
- `data/calibration.json` keeps the full per-company-year records, not just percentile summaries,
  specifically so Phase 4 can sample a real company-year as a template and perturb it. Independent
  per-metric percentile sampling can't be trusted to keep accounting identities (assets =
  liabilities + equity) intact; starting from one real, internally-consistent company-year can.

### Phase 4: Company spec generator — done

- 20 fictional companies (resolves open question 1), one fiscal year each, sampled without
  replacement from `calibration.json`'s 147 company-years as templates. `python -m src.step06_make_specs`;
  output at `data/synthetic/{fictional_id}/{fy}/spec.json` (committed, 80 KB total).
- Each spec's sector, sub-industry, and HQ state carry over directly from its template's real
  company (via `data/sp500_constituents.csv`), so the financial profile and the business
  description Phase 6 will write stay consistent with each other.
- Financials are scaled from the template by a random size factor, then rebuilt from the
  template's own ratios (equity/assets, revenue/assets, margins) jittered within a tight band,
  rather than sampled independently per calibration percentile — keeps the fictional company
  recognizably the same *kind* of company as its template without being identical to it.
- Assets = liabilities + equity by construction (equity and liabilities are rounded first, then
  liabilities is set to assets minus that rounded equity, so independent rounding can't break the
  identity) and then checked (`check_accounting_identity`), with up to 5 retries if it ever fails.
- Every fictional name and ticker is checked against all 503 real S&P 500 names and tickers (not
  just the 50 downloaded) and against every other fictional company already generated, so none
  collides. Verified on the current run: all 20 pass the identity check, and there are zero
  name/ticker collisions, real or duplicate.
- Cybersecurity facts for Item 1C: a security framework, a board-oversight line, and (15% of the
  time, matching how rarely real filings report one) a generic disclosed-incident summary.
- Not yet done: Item 1A risk-factor themes and Item 1/7 narrative-supporting facts beyond sector
  and size aren't in the spec; PLAN.md's original Phase 4 wording didn't call for them beyond what
  Item 1C needs, and Phase 6's prompts can likely derive them from sector + financials + a real
  exemplar. Revisit if Phase 6's narrative pilot needs more structure than that.

### Phase 5: Compute financial statements, don't write them — done

- `python -m src.step07_make_financials`; output at
  `data/synthetic/{fictional_id}/{fy}/financials.json` (committed, 160 KB total): a balance sheet,
  income statement, and cash-flow statement per fictional company, built from `spec.json`'s totals
  plus the line-item mix pulled from the template company's own `data/companyfacts/{ticker}.json`
  (the same source Phase 3 uses). Every line item is the template's own ratio (that line divided
  by its own total) applied to the fictional company's total, so the mix, expense structure, and
  cash-flow shape stay the kind of company the spec is grounded in, not an arbitrary combination.
- Every table reconciles exactly, not approximately: whatever a ratio basis can't explain (a tag
  the template doesn't report, or simply what's left) becomes one named "other" line per statement,
  sized so the explicit lines plus it hit the control total on the nose. Verified independently
  (outside the script's own check) on all 20: assets = liabilities + equity, every subtotal sums
  to its total, pretax income minus tax equals net income, and the three cash-flow subtotals sum
  to the net change in cash.
- The language model writes only narrative text (Phase 6); every number here is already fixed.
- Not every template reports every line item (financial and utility companies especially: no
  inventory, no traditional "cost of revenue," etc.), so some fictional companies show more of
  their statement in an "other" line than others. Accepted as consistent with Phase 3's "generic
  calibration" decision (open question 4) rather than modeled per sector.

### Phase 6: Narrative generation per Item — M4 pilot done, scale-up not started

- **Pilot:** 2 fictional companies written end to end — PRG (Amberwood Technologies Corp.,
  templated on ROP) and NDE (Harborview Logistics Corp., templated on AZO) — all 5 items each, at
  `data/synthetic/{PRG,NDE}/{fy}/item_*.txt` (committed). Written directly in this session rather
  than through an API script, using each real template's own Item as a structural/style guide
  (not copied), grounded strictly in that company's `spec.json` and `financials.json` figures.
- **Verified (Phase 7's automatic gates, run by hand on the pilot):** every ratio and dollar figure
  quoted in Item 7 and Item 8 matches `financials.json` exactly (operating/net/gross margin,
  effective tax rate, equity/assets, goodwill+intangibles/assets, inventory/assets, total debt —
  independently recomputed from the JSON and compared); no real company name or the template's own
  ticker appears in either company's text (one false-positive substring, "interoperate" containing
  "roper", checked and dismissed); Item 1, 1A, 7, 1C, and 8 all have the heading structure real
  filings use for that item.
- **Known gaps against the original design, by choice, for the pilot:** sections are condensed
  (300-700 words each) rather than matched to the real length distribution, since writing full
  length (thousands of words per item) for a 2-company pilot wasn't a good use of a pilot's
  purpose; operational color not in the spec (store counts, employee counts, segment sub-brand
  names) is either omitted or described qualitatively rather than invented as a specific fact, to
  avoid presenting an ungrounded number as if it were computed; Item 8 omits a signed audit
  opinion, since that would require inventing a fictional accounting firm and partner.
- **Open question 3 (model and budget) — resolved for now:** `src.step08_generate_filings` is
  built (Claude API, `anthropic` package, configurable per-item model) but deliberately left as a
  placeholder — never run against the real API, no cost incurred. All 20 companies continue to be
  written directly in this session instead, the same way as the 2-company pilot, for as long as
  that's the chosen approach. Revisit if that changes: the script is ready to use once there's a
  real `ANTHROPIC_API_KEY` and a budget decision.
- Original design, still the target for a full build: generation order Item 1, then 1A, then 7
  (which quotes the Item 8 numbers), then 1C, then the Item 8 notes, with earlier output passed in
  as context; each prompt containing spec facts, computed numbers, one real exemplar, a length
  target from the real distribution, and the real heading outline.

### Phase 7: Validation gates

- Automatic: MD&A figures match the Item 8 tables, ratios match computed values, section lengths
  fall within the real distribution, required headings are present, and no real company names
  appear.
- Model-judged: realism rubric on a sample.
- Human: review about 10 random filings.
- A failing filing gets one regeneration. If it still fails, it is dropped and logged.

### Phase 8: Same layout as the real corpus

- `data/synthetic/{fictional_id}/{fy}/item_*.txt`, with index rows marked `source = synthetic`.
  The same index and retrieval code then works on both corpora.

### Phase 9: QA and retrieval evaluation

- Synthetic filings: the spec holds the true values, so QA pairs can have exact answers and gold
  passage locations.
- Real filings: generate questions with a model, then check that each answer span appears in the
  text.
- Retrievers: BM25, dense embeddings, and hybrid (reciprocal rank fusion).
- Metrics: recall@k, MRR, nDCG@10.
- Experiments: synthetic only, real only, and mixed.

### Phase 10: Synthetic-vs-real gap report

- Compare section lengths, vocabulary overlap, number distributions, and BM25 score distributions.
- Check whether the retriever ranking on synthetic matches the ranking on real. If it does,
  synthetic data is a valid proxy for the real benchmark. If not, the report says so.

### Command sequence (once built)

```
python -m src.step01_download_10k          # done
python -m src.step02_parse_sections        # done (Phase 1 / step 3)
python -m src.step03_build_index           # done (Phase 2 / step 4)
python -m src.step04_fetch_companyfacts    # done (Phase 2 / step 5)
python -m src.step05_make_calibration      # done (Phase 3)
python -m src.step06_make_specs            # done (Phase 4)
python -m src.step07_make_financials       # done (Phase 5)
python -m src.step08_generate_filings  # built, placeholder (Phase 6; see above)
python -m src.step09_validate_filings  # not built yet (Phase 7)
python -m src.step10_make_qa           # not built yet (Phase 9)
python -m src.step11_evaluate_retrievers  # not built yet (Phase 9)
```

### Risks

- **Verbatim leakage:** narrative built from real exemplars can echo real phrasing. Keep exemplars
  short and flag long n-gram overlaps with the source.
- **Uniform prose:** generated text is usually more regular than real filings. The gap report
  should say so.
- **XBRL gaps:** not every company reports every tag (111/147 report total `Liabilities`,
  107/147 report `OperatingIncomeLoss`; see "Calibration profile"). `step05_make_calibration.py` tries
  fallback tags per metric and records what's still missing rather than guessing.
- **Name impersonation:** fictional companies must not use real names or tickers.

## Open questions

1. **Synthetic corpus size — resolved:** 20 fictional companies, one fiscal year each (not a
   multi-year trajectory). See Phase 4.
2. **Sections:** only Items 1, 1A, 1C, 7, and 8 as specified, or full 10-Ks?
3. **Model and budget:** which model, and what budget? Narrative cost scales with
   companies x sections.
4. **Calibration scope — resolved:** generic calibration, generic fictional companies (not
   fictional banks). See Phase 3 and "Calibration profile" above.
5. **Embedding model:** which model for the dense retriever?
6. **Sharing the raw data — resolved:** published as a public GitHub Release asset
   (`data-v1`, `sec10k_raw_fy2023-2025.tar.gz` + a SHA-256 manifest).
7. **Repository visibility — resolved:** the GitHub repo is public; confirmed intended.
8. **Leftover folders:** `data/raw/` now has two generations of superseded company-years that are
   deliberately left alone (not deleted, not reparsed, not in `config.yaml`'s company list):
   15 companies from the original stopped regional-bank run (FY2022-2024), and 44 companies from
   the first 50-company list this project used before it was replaced by a fully random sample
   (FY2023-2025; see "Downloaded data"). Citigroup's data was deleted outright rather than left
   over, since it's now excluded from sampling entirely. Delete the other leftovers, or keep them?
9. **EX-13 handling — resolved:** built, and now covers every item (not just 7 and 8): whichever
   of Items 1, 1A, 1C, 7, or 8 is a cross-reference gets the heading named in its cross-reference
   sentence located inside `ex13_*.htm` (PGR, USB, WFC) or the 10-K body itself. See "Parsed
   sections" above and `SECTIONS.md` for what's still short or missing and why.

## Milestones

- **M1 — done:** parser runs on all 50 companies x 3 years; exceptions are in
  `data/parse_status.csv` (5 short, 6 missing out of 735 sections, all explained above).
- **M2 — done:** index (729 rows), companyfacts (50/50 companies), and `calibration.json`
  (147 company-years, 12 metrics).
- **M3 — done:** 20 fictional company specs (Phase 4) and their full balance sheet, income
  statement, and cash-flow tables (Phase 5) are built, with every subtotal and identity checked.
- **M4 — pilot done, review pending:** narrative pilot on 2 fictional companies (PRG, NDE) written
  and hand-validated against Phase 7's automatic gates (see Phase 6). Human review of the pilot,
  and the model/budget decision for scaling to all 20 companies, are still open.
- **M5:** full synthetic corpus.
- **M6:** QA, retrieval evaluation, and gap report.
