# Synthetic 10-K Dataset: Plan and Open Questions

Goal: build a synthetic 10-K dataset for evaluating RAG systems on equity-analysis questions.
Real SEC 10-K filings serve three roles: style templates, calibration for realistic numbers, and
a real-data benchmark.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Project setup, config, identity handling | Done |
| 2 | Download 10-K filings from EDGAR | Done: 50 companies x FY2023-2025 (150 company-years, 147 filings, 3 not found) |
| 3 | Parse filings into sections (Items 1, 1A, 1C, 7, 8) | Done: 721/735 sections ok, 5 short, 9 missing; see `data/parse_status.csv` |
| 4 | Metadata index (`data/index.csv`) | Not started |
| 5 | XBRL companyfacts JSON per company | Not started |
| 6 | Summary script | Not started |
| - | Calibration, company specs, filing generation, QA, retrieval evaluation | Not started |

### Downloaded data

- Location: `data/raw/{ticker}/{fy}/`, with `10k.htm` (the 10-K as filed), `10k.txt` (readable text),
  `ex13_*.htm` (annual report exhibit, when present), and `metadata.json`.
- Companies: 50 S&P 500 companies sampled at random (`python -m src.sample_companies`,
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
- Reproduce: `python -m src.download_10k` (skips anything already downloaded).

### Parsed sections

- Location: `data/parsed/{ticker}/{fy}/item_1.txt`, `item_1a.txt`, `item_1c.txt`, `item_7.txt`,
  `item_8.txt`. See `SECTIONS.md` for which items these are and why.
- Result over the 147 downloaded filings x 5 items (735 sections; the 3 not-found company-years
  have no file to parse): 721 ok, 5 short, 9 missing. Every row's status, character count, and
  extraction method are in `data/parse_status.csv`. See `SECTIONS.md` for what each remaining
  short/missing row is and why it's not worth chasing further.
- Reproduce: `python -m src.parse_sections` (always reparses; it's local and fast, no need to skip).

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

### Phase 2: Index and XBRL facts (steps 4 and 5)

- `data/index.csv`: one row per section, with ticker, company name, CIK, fiscal year, filing date,
  accession number, item, path, character count, and word count.
- XBRL companyfacts: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`, one file per
  company, fetched through edgartools' `get_with_retry` so it uses the same identity and rate limit.

### Phase 3: Calibration profile

- **Stale assumption, needs a decision (see open question 4):** this phase was designed around a
  stable set of 15 hand-picked banks for bank-specific calibration. The company list is now a
  random S&P 500 sample instead (see "Downloaded data"), which only incidentally includes KEY,
  PGR, and USB as banks. Calibrating fictional *banks* from 3 real ones isn't enough; calibrating
  fictional companies generally, across sectors, is a different and larger design than what's
  written below.
- From the companyfacts data, extract per bank-year: total assets, loans, deposits, equity, net
  income, CET1 ratio, net interest margin, and return on equity.
- Compute percentile distributions and save them as `calibration.json`.
- Calibration uses only the 15 banks. The 35 S&P 500 companies are not banks, so they don't
  calibrate bank specs. They are still useful as real filings for the benchmark.

### Phase 4: Company spec generator

- Each fictional bank is a JSON spec: fictional name, HQ state, fiscal year, size sampled from the
  calibration distributions, balance-sheet mix, capital ratios, loan concentrations (for example,
  commercial real estate share), footprint, employees and branches, and the cybersecurity facts
  that Item 1C needs.
- Reject specs that break accounting identities: assets equal liabilities plus equity, and loans do
  not exceed assets.
- Check every fictional name against the real company list, so no fictional bank carries a real
  name or ticker.

### Phase 5: Compute financial statements, don't write them

- Code builds the balance sheet, income statement, and cash-flow tables from the spec. Every table
  reconciles.
- The language model writes only narrative text. All numbers are fixed before any prompt.

### Phase 6: Narrative generation per Item

- Generation order: Item 1, then 1A, then 7 (which quotes the Item 8 numbers), then 1C, then the
  Item 8 notes. Earlier output is passed in as context so sections stay consistent.
- Each prompt contains: spec facts, computed numbers, one real section of the same Item as a style
  exemplar (not to be copied), a length target from the real distribution, and the real heading
  outline.
- Choose the model after a small pilot: a stronger model for MD&A, a cheaper one for volume.

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
python -m src.download_10k          # done
python -m src.parse_sections        # done (Phase 1 / step 3)
python -m src.build_index           # Phase 2 (step 4)
python -m src.fetch_companyfacts    # Phase 2 (step 5)
python -m src.make_calibration      # Phase 3
python -m src.make_specs            # Phase 4
python -m src.generate_filings      # Phases 5 and 6
python -m src.validate_filings      # Phase 7
python -m src.make_qa               # Phase 9
python -m src.evaluate_retrievers   # Phase 9
```

### Risks

- **Verbatim leakage:** narrative built from real exemplars can echo real phrasing. Keep exemplars
  short and flag long n-gram overlaps with the source.
- **Uniform prose:** generated text is usually more regular than real filings. The gap report
  should say so.
- **XBRL gaps:** some banks do not report every tag, so calibration needs fallbacks.
- **Name impersonation:** fictional banks must not use real names or tickers.

## Open questions

1. **Synthetic corpus size:** how many synthetic companies, and which fiscal years? For example,
   20 companies for FY2023 and FY2024.
2. **Sections:** only Items 1, 1A, 1C, 7, and 8 as specified, or full 10-Ks?
3. **Model and budget:** which model, and what budget? Narrative cost scales with
   companies x sections.
4. **Calibration scope — now open again:** the company list is a random S&P 500 sample, not a
   bank list (see Phase 3). Options: calibrate per sector instead of banks-only; pick a fixed
   sub-list of real banks for calibration regardless of what's in the random sample; or make
   every fictional company a generic company instead of a fictional bank. Which do you want?
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
  `data/parse_status.csv` (5 short, 9 missing out of 735 sections, all explained above).
- **M2:** index, companyfacts, and `calibration.json`.
- **M3:** spec generator and financial-statement computation, with reconciliation tests.
- **M4:** narrative pilot on 2 fictional banks, validators, and review.
- **M5:** full synthetic corpus.
- **M6:** QA, retrieval evaluation, and gap report.
