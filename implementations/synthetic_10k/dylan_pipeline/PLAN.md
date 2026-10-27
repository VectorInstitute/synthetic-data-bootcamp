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
| 4 | Metadata index (`data/index.csv`) | Done: 829 rows (729 real + 100 synthetic, Phase 8) |
| 5 | XBRL companyfacts JSON per company | Done: 50/50 companies |
| 6 | Summary script | Not started |
| - | Calibration (`data/calibration.json`) | Done: 147 company-years, 12 metrics; see "Calibration profile" |
| - | Company specs (Phase 4) | Done: 20 fictional companies; see `data/synthetic/` |
| - | Scenario retrofit (Phase 4.5) | Done: 13/20 companies have a scenario, 7/20 baseline |
| - | Financial statements (Phase 5) | Done: balance sheet, income statement, cash flow per company, all reconciling |
| - | Narrative generation (Phase 6) | Done: 20/20 companies, 100 files; see "Phase 6" |
| - | Validation gates (Phase 7) | Automatic and model-judged done, 0 failures; human review open |
| - | Same-layout index (Phase 8) | Done: one `data/index.csv`, `source` column distinguishes real/synthetic |
| - | QA, retrieval evaluation (Phase 9) | Done: 130 QA pairs (115 factual, 15 analytical), BM25/semantic/hybrid x 3 experiments |
| - | Synthetic-vs-real gap report (Phase 10) | Done: full retriever ranking agrees; length/vocab/numbers differ |
| - | Privacy: distance from real template (Phase 11) | Done: no company's financials sit within 5% of its template on every field |

## Evaluation framework: fidelity, utility, privacy

A synthetic corpus built to stand in for real 10-Ks has to answer three separate questions, and
the dashboard (`dashboard/index.html`) is organized around them rather than around the phases that
produced the numbers (phases map onto these three unevenly — Phase 7 alone spans two of them):

- **Fidelity** — does it look and compute like a real 10-K? Structural fidelity (required
  statement headings, substantive section length, internal numeric consistency between narrative
  and financials — Phase 7's `required_headings`/`substantive`/`numeric_consistency`/
  `scenario_grounding` checks) and distributional fidelity (section length, vocabulary overlap,
  numeric density vs. the real corpus — Phase 10's gap report).
- **Utility** — is it actually useful for the task it exists to support (building and evaluating a
  retrieval benchmark)? Phase 9's QA pairs and recall@5/@10/MRR/nDCG@10 scores are the benchmark
  itself; Phase 10's retriever-ranking-agreement check is the validation that conclusions drawn from
  the synthetic corpus (e.g. "BM25 beats this embedding model here") transfer to the real one.
- **Privacy** — since every fictional company is templated on one real company, does the corpus
  avoid exposing that real company? Two checks, both from Phase 7 and the new Phase 11: no real
  company name or ticker ever appears in the narrative (`real_company_leak`, 0 found across 373
  checks), and no fictional company's financials are close enough to its real template's to make the
  disguise pointless (`step14_privacy_check.py` — median 20.0% absolute difference per field across
  all 20 companies; closest single company, BIDZ, still differs by a 4.1% median and an 8.2% max
  across its 6 fields; no company is within 5% on every field).

### Phase 11: Privacy — distance from each fictional company's real template

`src/step14_privacy_check.py` writes `data/privacy_report.csv`, 112 rows (one per fictional company
per financial field: total assets, total liabilities, stockholders' equity, revenues, net income,
operating income where present). For each row it looks up the real template's actual same-year
value directly from `data/calibration.json`'s `company_years` (written by `step05_make_calibration`
from XBRL, before any fictional company existed) and computes the percent difference from the
fictional value.

- **Why this check and not just the leak check:** Phase 7's `real_company_leak` check already
  confirms no real company's *name* or *ticker* appears in the narrative, but that alone doesn't
  rule out the fictional numbers simply being the real company's own numbers relabeled — which would
  make the fictional company a disguise, not a synthetic one. This check answers that directly.
- **Result:** median absolute difference per field, across all 112 rows, is 20.0%. Every company has
  at least one field that differs substantially (max per-company difference ranges from 8.2%, BIDZ,
  to 113.1%, NDE — the latter because NDE's scenario deliberately gives it a small *positive* equity
  cushion where its real template, AutoZone, actually carries deeply negative equity). The closest
  individual field found anywhere is NDE's revenue, 0.5% from AutoZone's actual FY2025 revenue — a
  coincidence of where the calibration sampling landed on that one field, not a pattern: NDE's other
  four fields differ by 7.4%-113.1%, and no company is close on every field. Reported honestly rather
  than smoothed over, since a single close field is a legitimate thing a privacy audit should surface.

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

### Phase 4.5: Scenario retrofit — done, all 20 companies

- Raised after Phase 6/7 were both already complete: spec.json only held size/sector/margin facts
  plus one cybersecurity flag, so the narrative had nothing distinguishing to ground beyond generic
  sector-template boilerplate — which is also the root cause of Phase 7's "uniform prose" finding
  (the same Liquidity paragraph and commitments sentence verbatim across all 20 companies). It also
  meant there was no structured ground truth for *qualitative* claims (an acquisition, a product
  launch, a regulatory event) the way there already was for numbers, which matters for Phase 9: a
  synthetic company's QA pairs are only as good as what's actually grounded in its spec.
- `python -m src.step06b_add_scenarios`; adds a `scenario` field (`type`, `summary`, `facts`) to
  each `spec.json`. Scenario types, hand-assigned per company based on fit with its existing
  profile (see the script's `SCENARIOS` dict for the full rationale and exact source of every
  figure):
  - **Mergers & acquisitions** — EASS, FTC, PRG. Each "acquired" a fictional target company this
    fiscal year; the purchase price is the exact pre-existing `other_investing_activities` dollar
    figure from that company's own `financials.json` (EASS $249.7M, FTC $2,211.2M, PRG $2,943.0M —
    PRG's generic "acquisitions of businesses" phrasing already matched this almost exactly). Item 7's
    Liquidity paragraph was reworded in place to attribute the existing cash outflow to the named
    acquisition instead of a generic cause; Item 8 gained one new numbered note per company.
  - **Product launch** — GOG ("SentinelFlow," an anomaly-detection module), grounding the existing
    R&D-heavy operating loss in a specific, named cause instead of a generic one.
  - **Competitive displacement** — HZP (a customer migrated workloads to a competitor).
  - **Margin crisis** — BIDZ, whose 0.4% net margin was already the thinnest of the 20; reframed
    explicitly as historically compressed and unsustainable without correction, not just reported
    neutrally.
  - **Capital structure** — NDE and TVZ (same AutoZone-style template, ~2% equity/assets each):
    framed, matching how AutoZone itself frames it in real filings, as a deliberate debt-funded
    share-repurchase strategy reflecting management confidence, explicitly **not** distress — worded
    independently for each company despite sharing a template and a scenario type.
  - **Cash burn** — WJE (existing operating/net loss reframed as deliberate pre-profitability
    investment, consistent with its growth stage).
  - **Regulatory rate case** — BAI (an unfavorable rate case outcome limited cost recovery).
  - **Weather event** — VQWP (storm restoration layered onto, not replacing, the existing capital
    program — explicitly not claimed as the entire investing outflow, since that would be
    financially implausible for a single utility in one year).
  - **Regulatory pricing pressure** — KMW (named pharmacy-benefit-management and drug-pricing
    measures as the driver of its already-thin margin and high effective tax rate).
  - **Regulatory consent order** — GKRH (a consent order with a federal banking regulator, tied to
    its existing "other operating expenses" figure).
  - **Baseline** — CRJ, FAI, JFG, PKPM, SKR, TOUN, XTE (7/20): explicitly "ordinary course of
    business, no material scenario event." Real 10-Ks are mostly like this, so a benchmark where
    every company has a dramatic event would itself be unrealistic.
- **Deliberately out of scope:** true bankruptcy / going-concern distress. None of the 20 companies
  has a distressed balance sheet, and forcing one would mean actually lowering a company's equity or
  raising its leverage — real financial-statement changes that risk breaking Phase 5's already-
  validated reconciliation, not just a narrative reframe. "Elevated financial risk" was folded into
  BIDZ's margin crisis and NDE/TVZ's capital structure instead, both grounded in figures that were
  already there.
- **financials.json is untouched for all 20 companies** (verified: `git diff --stat` on every
  `financials.json` shows no changes) — every scenario fact is copied from a number already
  computed in Phase 5, so Phase 7's numeric-consistency gate needed no changes to keep passing.
- Narrative was retrofitted into the 13 non-baseline companies' `item_1.txt`/`item_1a.txt`/
  `item_7.txt`/`item_8.txt` (not all 4 files for every company — several were left untouched where
  the existing content already covered the scenario well enough that forcing an edit would have
  been redundant), via 4 parallel sub-agents grouped by scenario similarity (M&A trio; tech-sector
  trio; regulated-industry quartet; balance-sheet-story trio), the same fork-based process Phase 6
  used. Every edit was a pure insertion or an in-place reattribution of an *existing* number's
  stated cause — no dollar figure or percentage was changed anywhere.
- **Verified independently** (not just trusting each sub-agent's self-report, per the same
  discipline as Phase 6): re-ran `src.step09_validate_filings` after all 4 sub-agents finished —
  373 checks, 0 failures, across all 20 companies. Spot-checked several of the trickiest edits by
  hand (PRG and FTC's Liquidity-paragraph reattribution, GKRH's new risk paragraph, NDE/TVZ's
  capital-structure framing) and confirmed they read naturally and match their intended tone.
  Checked the 4 new fictional entities (3 acquired companies, 1 product name) against all 503 real
  S&P 500 names for collisions: none found.
- **New automatic gate added to Phase 7** (`src/step09_validate_filings.py`'s
  `scenario_grounding` check): for every non-baseline company, confirms its scenario is actually
  grounded somewhere in Items 1/1A/7/8, not just recorded in `spec.json`. M&A and product-launch
  scenarios are checked by exact string match on the acquired company's or product's name; the rest
  are checked by keyword (e.g. "consent order," "rate case," "repurchase"), since there's no single
  proper noun to anchor on for something like a regulatory pricing change. All 13 pass.

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
- **Fixed:** when a template has neither an operating-income anchor (missing `operating_margin` in
  calibration) nor its expenses tagged in a way `OPERATING_EXPENSE_TAGS` catches (both true for a
  few banks and industrials), the original expense-based pretax estimate could be wildly off and
  forced the income-tax-expense plug to absorb the gap — 93%+ "effective tax rates" for EASS
  (BALL), GKRH (USB), SKR (NTRS), and TOUN (TXT). Now checked against a sane range and, when it
  fails, replaced with a U.S. statutory-rate (21%) estimate instead; `interest_expense` is also
  capped at 60% of revenue, since a bank's real tag isn't on the same basis as the generic revenue
  metric and could otherwise exceed revenue outright (SKR). All 4 now show a plausible rate (21%,
  21%, 36%, 21%); the other 16 companies are byte-identical to before the fix.
- Not every template reports every line item (financial and utility companies especially: no
  inventory, no traditional "cost of revenue," etc.), so some fictional companies show more of
  their statement in an "other" line than others. Accepted as consistent with Phase 3's "generic
  calibration" decision (open question 4) rather than modeled per sector.

### Phase 6: Narrative generation per Item — done, all 20 companies

- All 20 fictional companies written end to end: all 5 items each, at
  `data/synthetic/{fictional_id}/{fy}/item_*.txt` (committed, 100 files). Written directly in
  Claude Code sessions rather than through an API script — the first 2 (PRG, NDE) as a pilot, the
  remaining 18 via 5 parallel sub-agents grouped by sector (Utilities; Financials; Health Care and
  Materials; Information Technology; Industrials and Consumer Discretionary), each given the exact
  pre-computed figures for its companies so no sub-agent had to do its own arithmetic. Every
  company uses its real template's own Item as a structural/style guide (not copied), grounded
  strictly in that company's `spec.json` and `financials.json` figures.
- **Verified (Phase 7's automatic gates, run by hand on all 20 after every sub-agent reported its
  own pass):** every ratio and dollar figure quoted in Item 7 and Item 8 matches `financials.json`
  exactly (net margin, equity/assets, and others, independently recomputed and compared to what
  each item_7.txt states); every balance sheet, income statement, and cash-flow statement
  reconciles; all 100 files are substantive (none under 150 words); no real template company's
  full name appears anywhere across all 100 files, and no bare ticker appears as a standalone
  word (checked separately from the expected substring false positives — "ben" inside "benefit",
  "rop" inside "appropriate"/"property", "dri" inside "driven", "fast" inside "breakfast" — which
  are not real leaks). TOUN's disclosed cybersecurity incident is threaded consistently through
  Item 1A, 1C, and 8.
- A handful of companies have figures that don't fit the standard narrative cleanly, each handled
  explicitly rather than left to look broken: FTC, GOG, HZP, and WJE have a net tax *benefit*
  (negative tax expense) at a small or negative pretax base, which would produce a meaningless
  "percentage" (e.g. -80.8%, 208.6%) if stated literally — these are described in dollars and
  qualitatively instead, never as a literal rate. GKRH, SKR, and EASS have no separate operating-
  income line (see Phase 5's fix), so their Item 7/8 go straight from revenue and expenses to
  pretax income. GKRH, SKR, BAI, CRJ, and VQWP (banks and utilities) have 90%+ of assets in one
  "other_assets" line; described qualitatively as consistent with that kind of balance sheet
  (loans/securities for banks, utility plant for electric utilities) rather than invented detail.
- **Known gaps against the original design, by choice:** sections are condensed (roughly
  400-900 words each) rather than matched to the real length distribution, since matching it
  (thousands of words per item, times 100 files) wasn't a good use of hand-written effort over an
  API-driven approach; operational color not in the spec (store counts, employee counts, segment
  sub-brand names) is either omitted or described qualitatively rather than invented as a specific
  fact, to avoid presenting an ungrounded number as if it were computed; Item 8 omits a signed
  audit opinion, since that would require inventing a fictional accounting firm and partner.
- **Open question 3 (model and budget) — resolved:** `src.step08_generate_filings` is built
  (Claude API, `anthropic` package, configurable per-item model) but deliberately left as a
  placeholder — never run against the real API, no cost incurred. All 20 companies were written
  directly in Claude Code sessions instead. Revisit if a future expansion (more companies, a
  multi-year trajectory, full-length sections) makes hand-writing impractical: the script is ready
  to use once there's a real `ANTHROPIC_API_KEY` and a budget decision.
- Original design, still the target for a full build: generation order Item 1, then 1A, then 7
  (which quotes the Item 8 numbers), then 1C, then the Item 8 notes, with earlier output passed in
  as context; each prompt containing spec facts, computed numbers, one real exemplar, a length
  target from the real distribution, and the real heading outline.

### Phase 7: Validation gates — automatic and model-judged done; human review open

- **Automatic — done.** `python -m src.step09_validate_filings`; output at
  `data/validation_report.csv` (committed, 360 rows: 20 companies x 18 checks). Result on the
  current corpus: 0 failures. Checks:
  - **Numeric consistency (Item 7 and Item 8):** every dollar figure and percentage extracted from
    the narrative must be within tolerance (1% relative for dollars, 0.25 points for percentages)
    of a value derived from that company's own `financials.json` — either a leaf figure, a specific
    named combination seen in the narrative (total debt, goodwill + intangibles, SG&A + R&D,
    cost of revenue + R&D, gross profit), or one of six standard ratios (net/operating margin, ROE,
    ROA, equity/assets, effective tax rate). Dollar figures are compared by magnitude, not sign,
    since cash-flow figures are stored with an accounting sign but routinely restated as a positive
    amount in prose ("we used $X million of cash in investing activities").
  - **Real-company leak:** scoped to each fictional company's own real template (not all 503 S&P 500
    companies), since that's the one real company whose exemplar text ever entered its generation.
    Ticker matching is case-sensitive and whole-word, since several templates used here (BEN, ROP,
    DRI, FAST, BALL, USB) are also ordinary English words or acronyms in lowercase or mixed case.
  - **Required headings:** Item 8 contains a consolidated balance sheet, income/operations
    statement, and cash-flow statement heading.
  - **Substantive:** every item file is at least 150 words (Phase 6's own floor).
  - **Length vs. real distribution:** reported, not gated — Phase 6 condensed every section by a
    documented, deliberate choice, so gating on length would flag all 20 companies for something
    that isn't a defect.
  - **Calibration note:** a first version of the numeric check used blind all-pairs sums of each
    statement section's own leaves (to catch combination figures like "total debt") instead of the
    named combinations above. Dropped after a deliberately-wrong injected figure ($9,999.9 million
    in place of PRG's real revenue) was accepted as a false match — with 5-7 leaves per section, the
    combinatorics produced ~99 candidate values per company, enough that a wrong number could land
    within tolerance of one by coincidence. Re-verified after the fix with the same injected-error
    test (caught) and an injected real-company-name/ticker leak (caught), confirming the gate
    actually discriminates rather than rubber-stamping.
- **Model-judged — done, 6/20 companies (one from each Phase 6 sector batch, including both
  special cases).** Read in full and scored against a realism rubric (register, sector-appropriate
  risk framing, cross-item consistency, absence of invented detail): BAI (Utilities/AEP), FTC
  (Financials/BEN, tax-benefit case), GKRH (Financials/USB, bank with no operating-income line), KMW
  (Health Care/CVS), XTE (Industrials/NOC), WJE (Information Technology/CRWD, net-loss tax-benefit
  case).
  - **Strengths:** sector-appropriate risk framing throughout (GKRH's bank-specific credit/interest-
    rate risk and explicit note that its costs aren't broken into industrial cost-of-revenue/SG&A
    categories; XTE's government-contract termination-for-convenience and security-clearance
    language; WJE's subscription-billed-in-advance explanation for why operating cash flow is
    strongly positive despite a net loss); correct hedged 10-K register throughout; every
    cross-reference ("See Item 1A," "As described in Item 1C") lands on content that's actually
    there; the tax-benefit and no-operating-income special cases read naturally, not as a workaround.
  - **Weakness, consistent with the "Uniform prose" risk below:** the Item 7 "Liquidity and Capital
    Resources" paragraph follows an almost identical sentence template in every company regardless
    of sector ("We generated $X of cash from operating activities... We used $Y in investing
    activities... Financing activities provided/used $Z... cash increased/decreased by $W"), and the
    Item 8 commitments-and-contingencies sentence is word-for-word identical across all 20 companies.
    Both are authentic 10-K boilerplate individually, but their exact, invariant repetition across a
    bank, a utility, a defense contractor, and a cybersecurity company in the same corpus is a
    detectable synthetic fingerprint — the single most "generated" feature found across the sample.
    Not a correctness defect; noted for Phase 10's gap report.
  - No real template name or ticker found in the 6-company close read (consistent with the
    automatic gate's result across all 20).
- **Human — open.** Needs about 10 filings read by a person, not a model. Not started; this is the
  one remaining Phase 7 item.
- A failing filing gets one regeneration. If it still fails, it is dropped and logged. Hasn't come
  up yet: nothing has failed.

### Phase 8: Same layout as the real corpus — done

- `src.step03_build_index` now appends `data/synthetic/{fictional_id}/{fy}/item_*.txt` to the same
  `data/index.csv` it already built from the real corpus, instead of a separate file. A new
  `source` column (`real` or `synthetic`) is the only schema change; `cik`, `filing_date`, and
  `accession_number` are blank on synthetic rows since there's no real-world equivalent. Result:
  829 rows (729 real + 100 synthetic — 20 companies x 5 items). Retrieval code (Phase 9) can
  select, mix, or compare the two corpora by filtering on `source` rather than needing two
  indexes or two code paths.
- Re-run `step03_build_index` after the synthetic corpus changes, not just after the real one —
  it's not implicitly triggered by `step06`-`step09`.

### Phase 9: QA and retrieval evaluation — done

- **Chunking (`src/chunks.py`), shared by QA generation and retrieval.** A chunk is one
  blank-line-separated block of an item file — the same unit `step10_build_pdfs.tableize()`
  operates on, so a financial-statement table (heading + rows, no blank lines between them) falls
  out as one chunk rather than being split mid-table. `load_chunks()` filters at the index-row
  level (`sources`/`tickers`) before reading any file, not after — the real corpus alone is ~730
  rows including items over 100,000 words, so filtering after reading cost minutes the first time
  this was built the naive way.
- **QA generation (`src/step11_make_qa.py`) — `data/qa_pairs.csv`, 130 rows (118 synthetic, 12
  real).** Every row has a `style`: `factual` (a single figure or named entity — 115 rows) or
  `analytical` (why something happened, or what it implies — 15 rows). Analytical questions are
  grounded exactly the same way as factual ones (a chunk search, verified at generation time,
  never a paraphrase); the difference is the question requires reading a passage that connects a
  cause to an effect, not locating a labeled figure.
  - *Synthetic, fully deterministic, no model call.* Every answer is a value already in that
    company's `financials.json` or `spec.json` scenario (Phase 4.5). The gold chunk and exact
    answer span are found by scanning the company's own chunks with `step09_validate_filings`'s
    own figure-extraction and tolerance-matching logic (`extract_figures`, `closest_match`) — so a
    financial-fact answer is guaranteed to be a verbatim substring of its gold chunk by
    construction, not by a separate check. Four **factual** question types: revenue, net income,
    total assets, net margin (+ operating margin where `operating_income` exists — 16/20
    companies); plus, for the 13 non-baseline companies, a factual scenario question (the
    acquired company's name, the purchase price, or the launched product's name).
  - **Analytical questions, added after the first pass read as pure factoid lookups** (revenue,
    margin, "what company did X acquire") with no question requiring synthesis. Three kinds, all
    grounded by verifying the exact phrase or keyword is actually present at generation time, same
    rigor as every other question in this file:
    - *Rephrased scenario questions* (9 keyword-grounded types, e.g. "Why did {company}'s margins
      compress to historically thin levels in fiscal {fy}?" instead of "What type of rate pressure
      did {company} cite…"). Same grounding and answer as before, reframed as a cause, not a
      category label.
    - *Risk-implication questions*, newly grounded in each company's Item 1A risk factors (not
      reused from an existing chunk): the 3 M&A companies and the 1 product-launch company each
      have a risk paragraph reading "We may not realize the anticipated/expected benefits of our
      acquisition/launch of X…" — verified present in all 4 before use, not assumed from the
      M&A/product-launch narrative pattern elsewhere.
    - *Severity-synthesis questions*, for the 2 scenario types (`margin_crisis` — BIDZ,
      `cash_burn` — WJE) where the narrative happens to state the scenario's driver and its
      financial-statement outcome in the *same* paragraph — checked per company, not assumed from
      the scenario type; no other scenario type had this co-location. The answer is the
      qualitative severity phrase the narrative itself uses ("thinnest margins",
      "ahead of sustained profitability"), not a bare percentage, so answering requires reading
      past the figures to the sentence that characterizes them.
  - Hit rate: every one of these generates deterministically from facts already verified present;
    0 silent failures.
  - *Real, hand-authored, not model-generated.* "Generate questions with a model" above originally
    meant calling the Claude API; there's no `ANTHROPIC_API_KEY` configured (same gap as Phase 6 —
    `step08_generate_filings.py` is a placeholder for that reason), and reading whole real filings
    to write questions would mean ingesting 100,000+ words for the larger ones. Instead: grep-based
    short excerpts (a paragraph or two, not the whole item) from the same 6-company template sample
    Phase 7's model-judged review already used (AEP/FY2025, BEN/FY2024, USB/FY2025, CVS/FY2024,
    NOC/FY2025, CRWD/FY2025), 2 hand-written questions each. Every answer is checked against the
    actual chunked text at generation time — the script hard-fails (`SystemExit`) if one isn't
    found verbatim — which is what "check that each answer span appears in the text" means here.
  - *Known simplification:* each question records exactly one `gold_chunk_id`, even if the same
    figure happens to be restated in a second chunk (a prose sentence and a table row, say).
    Retrieval of that second, unrecorded chunk would be scored as a miss. Not corrected for; the
    effect is to slightly understate recall, not overstate it.
- **Retrieval (`src/retrieval.py`) and evaluation (`src/step12_evaluate_retrievers.py`) —
  `data/retrieval_eval.csv`.**
  - **BM25** via `rank-bm25`. **Semantic**: real sentence embeddings via `fastembed` (ONNX Runtime,
    no torch) with `minishlab/potion-retrieval-32M`, a *static* embedding model distilled from a
    larger transformer (Model2Vec) — embedding is a token lookup plus pooling, not a transformer
    forward pass, so it's genuinely semantic without per-query transformer inference cost. Cosine
    similarity over L2-normalized vectors, plain numpy. **Hybrid**: reciprocal rank fusion of the
    two (`k=60`).
  - **Calibration note, round 1 (vocabulary determinism):** an early version used a TF-IDF "dense"
    retriever instead (log-scaled term frequency x log IDF, capped at an 8,000-term vocabulary) —
    not a neural embedding, since `sentence-transformers` (pulls in `torch`) was tried first and
    timed out after 3 minutes in this sandboxed environment with no prior download cache. That
    TF-IDF version picked its vocabulary via `Counter.most_common()`, which breaks count ties by
    first-insertion order into the counter — an order that depended on `set(tokens)`'s
    hash-randomized iteration, so two runs over the *identical* corpus produced slightly different
    vocabularies and therefore slightly different recall/MRR each run. Fixed at the time by sorting
    candidates by `(-count, term)` explicitly; verified by diffing two full runs byte-for-byte.
  - **Calibration note, round 2 (replacing TF-IDF with a real embedding model):** TF-IDF worked but
    was lexical, not semantic, and it collapsed badly on the real corpus (recall@5 0.08 — see the
    superseded numbers this replaced, in git history). Revisited after torch was ruled out:
    `fastembed` needs only ONNX Runtime (~20MB) and was tried with its default transformer model,
    `BAAI/bge-small-en-v1.5` — installs fine, but measured at ~9 texts/second on this machine's 16
    cores. The real 6-company sample alone chunks into ~60,000 paragraphs (`real_only` and `mixed`
    both embed it), which would take roughly two hours at that rate. Switched to
    `minishlab/potion-retrieval-32M` (a static, non-transformer embedding model): ~1,900
    texts/second on the same corpus, under a minute end to end, with comparable semantic
    separation in a spot check (query-answer cosine 0.62 vs. query-unrelated-text cosine 0.03). Hit
    one more snag along the way: `fastembed==0.9.0` declares `huggingface-hub<2.0` but imports an
    internal module (`_snapshot_download`) that huggingface-hub removed in its 1.x release, so a
    plain `pip install fastembed` resolved huggingface-hub 1.33.0 and broke on import; fixed by
    pinning `huggingface-hub==0.36.2` in `requirements.txt`.
  - **Metrics:** recall@5, recall@10, MRR, nDCG@10 (binary relevance, one gold chunk per question).
  - **Experiments.** `synthetic_only` (118 Qs against the 20 synthetic companies' chunks),
    `real_only` (12 Qs against the same 6 real sample companies' chunks — not all 50 real
    companies/150 filings, which would mean embedding on the order of 500,000 paragraphs), `mixed`
    (all 130 Qs against the union of both).
  - **Result** (`data/retrieval_eval.csv`):

    | experiment | retriever | n | recall@5 | recall@10 | MRR | nDCG@10 |
    |---|---|---|---|---|---|---|
    | synthetic_only | bm25 | 118 | 0.49 | 0.57 | 0.44 | 0.46 |
    | synthetic_only | semantic | 118 | 0.50 | 0.54 | 0.40 | 0.43 |
    | synthetic_only | hybrid_rrf | 118 | 0.48 | 0.56 | 0.44 | 0.46 |
    | real_only | bm25 | 12 | 0.50 | 0.58 | 0.29 | 0.36 |
    | real_only | semantic | 12 | 0.33 | 0.33 | 0.21 | 0.23 |
    | real_only | hybrid_rrf | 12 | 0.33 | 0.42 | 0.23 | 0.27 |
    | mixed | bm25 | 130 | 0.48 | 0.49 | 0.40 | 0.42 |
    | mixed | semantic | 130 | 0.45 | 0.45 | 0.35 | 0.38 |
    | mixed | hybrid_rrf | 130 | 0.46 | 0.48 | 0.40 | 0.42 |

    (Adding the 6 analytical questions in the "diversity" revision below shifted these by a point
    or two — e.g. synthetic_only's recall@5 ordering between bm25 and semantic flips — but not the
    qualitative story: BM25 still edges out the semantic retriever on `real_only`, and the full
    ranking agreement in Phase 10 below is unaffected.)

    BM25 beats the semantic retriever on all three experiments, but nowhere near as dramatically as
    TF-IDF's collapse on `real_only` (0.50 vs. 0.33 recall@5, not 0.50 vs. 0.08) — a real embedding
    model generalizes far better to the real corpus's longer, more varied documents than a
    fixed-vocabulary lexical vector did, even though it's still a lighter, distilled model rather
    than a full transformer encoder. BM25's edge makes sense for this task specifically: the QA
    pairs are precise fact lookups (a dollar figure, a company name, a date), where exact lexical
    matching is a strong baseline, and a static embedding model trades some of a full encoder's
    nuance for its ~200x speed advantage. Hybrid RRF again sits at or slightly below BM25 alone —
    equal-weight fusion with a second ranking that's consistently weaker pulls the combined result
    down, not up. Take the `real_only` numbers directionally, not precisely: n=12 means one
    question's outcome moves recall@5 by ~8 points. None of this is Phase 10's synthetic-vs-real
    comparison (which asks whether a *ranking* agrees across corpora, a different question) — that's
    still open.

### Phase 10: Synthetic-vs-real gap report — done

`src/step13_gap_report.py` writes `data/gap_report.csv`, 12 rows across the four comparisons below.
Comparisons 2 and 3 scan the *full* real corpus (729 rows, ~10.8M words), not the 6-company
sample everywhere else in Phase 9/10 — a streaming word/figure count needs no matrix, unlike the
dense-vector corpus, so there was no reason to bound it. Comparisons 1 and 4 reuse data Phase 8/9
already produced (`data/index.csv`, `data/retrieval_eval.csv`) rather than recomputing anything.

1. **Section length** (`data/index.csv`'s own `words` column, median per item): synthetic is
   3.6%-31.2% of real's median length per item (Item 1C, the shortest and most boilerplate section
   in both corpora, is closest at 31.2%; Item 8, where real filings carry pages of footnotes
   synthetic never generates, is furthest at 1.6%). Not a new finding — this is the condensation
   Phase 6 already documented as a deliberate choice, just quantified precisely for the first time
   here, per-item rather than as one qualitative note.
2. **Vocabulary overlap** (Jaccard over each corpus's 3,000 most frequent tokens): 0.404 symmetric
   Jaccard; 67.1% of synthetic's top-3,000 terms also appear in real's top-3,000. The asymmetry
   makes sense — synthetic's narrative was written from real exemplars as templates, so its
   (much smaller) vocabulary is largely a subset of real's; real's top terms include large amounts
   of real-filing-specific and numeric vocabulary (subsidiary names, exhibit cross-references,
   boilerplate legal phrasing) that condensed synthetic narrative never needed.
3. **Number distributions** (every dollar/percent figure found by `step09_validate_filings`'s own
   `extract_figures`, across every file): synthetic states **more** numbers per 1,000 words than
   real (4.02 vs. 1.48 dollar figures; 3.09 vs. 1.75 percent figures) — expected, since Phase 6's
   narrative was deliberately written to state and restate the key statement figures the validation
   gate checks, where real MD&A spends much more prose on qualitative discussion between numbers.
   Median dollar magnitude differs sharply ($429M real vs. $2,126M synthetic) and median percent
   value even more so (100.0% real vs. 23.8% synthetic) — read both with a caveat, not as a deep
   semantic gap: real's figure *count* is dominated by small, incidental numbers across 10.8M words
   (per-share amounts, headcounts, ownership-table rows), and real's 100.0% median percent in
   particular looks like an artifact of how common exact "100%" entries are in real
   percentage-of-total tables (the same reason `step09_validate_filings.UNIVERSAL_PERCENTS`
   excludes 0%/100%/21% from its own matching), not a sign real filings "run near 100% margins."
   Synthetic's narrower, curated set of figures (financial-statement totals, margins, tax rates)
   skews toward company-scale totals and meaningful ratios instead.
4. **BM25 score distribution and retriever ranking agreement** (reusing Phase 9's `Bm25Retriever`
   and `data/retrieval_eval.csv`): median top-1 BM25 score is 23.97 (real sample, n=12) vs. 18.64
   (synthetic, n=118) — not directly comparable in absolute terms, since BM25's IDF term depends on
   each corpus's own size and term statistics, and the real sample's 6 companies chunk into far more
   paragraphs than all 20 synthetic companies combined. The more decision-relevant question is
   retriever **ranking**: by recall@10, both `synthetic_only` and `real_only` rank retrievers the
   same way — `[bm25, hybrid_rrf, semantic]` — **the full three-way order agrees**, not just the top
   choice. (With the superseded TF-IDF retriever, only the top choice agreed; the full order didn't
   — see the Phase 9 "Calibration note, round 2" above. Swapping in a real embedding model changed
   this result, not just the retriever's own numbers.)

**Verdict:** synthetic is a reasonable proxy for *which retriever to pick* — here, the full ranking
agrees, not just the top choice — and for exercising the retrieval/evaluation pipeline end to end.
It is not a proxy for realistic document length, numeric density, or absolute retrieval-score
magnitude — all three differ by a large, now-quantified margin. Anyone using this corpus for
retriever benchmarking should treat synthetic results as directional for retriever choice, not as
an estimate of real-world recall or score values.

### Command sequence (once built)

```
python -m src.step01_download_10k          # done
python -m src.step02_parse_sections        # done (Phase 1 / step 3)
python -m src.step03_build_index           # done (Phase 2 / step 4; real rows only at this point)
python -m src.step04_fetch_companyfacts    # done (Phase 2 / step 5)
python -m src.step05_make_calibration      # done (Phase 3)
python -m src.step06_make_specs            # done (Phase 4)
python -m src.step06b_add_scenarios        # done (Phase 4.5)
python -m src.step07_make_financials       # done (Phase 5)
python -m src.step08_generate_filings  # built, placeholder (Phase 6; see above)
python -m src.step09_validate_filings  # done: 0 failures, 20 companies (Phase 7)
python -m src.step03_build_index           # re-run: done (Phase 8; now adds the 100 synthetic rows)
python -m src.step10_build_pdfs        # done (ad hoc; renders each filing to a standalone PDF)
python -m src.step11_make_qa           # done: 130 QA pairs (Phase 9)
python -m src.step12_evaluate_retrievers  # done: data/retrieval_eval.csv (Phase 9)
python -m src.step13_gap_report        # done: data/gap_report.csv (Phase 10)
python -m src.step14_privacy_check     # done: data/privacy_report.csv (Phase 11)
python -m src.step15_build_dashboard   # done: dashboard/index.html (ad hoc; embeds the 5 result CSVs)
```

### Risks

- **Verbatim leakage:** narrative built from real exemplars can echo real phrasing. Keep exemplars
  short and flag long n-gram overlaps with the source.
- **Uniform prose:** generated text is usually more regular than real filings. The gap report
  should say so. **Observed, not just anticipated:** Phase 7's model-judged pass found the Item 7
  "Liquidity and Capital Resources" paragraph and the Item 8 commitments-and-contingencies sentence
  repeat in an almost invariant template across all 20 companies regardless of sector — the clearest
  synthetic fingerprint found in the corpus. See Phase 7.
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
5. **Embedding model — resolved:** `fastembed` (ONNX Runtime, no torch) with
   `minishlab/potion-retrieval-32M`, a static embedding model distilled from a larger transformer —
   `sentence-transformers` (pulls in `torch`) timed out installing, and fastembed's own default
   transformer model was too slow at the real corpus's scale (~9 texts/second would mean ~2 hours
   for the real sample alone). See Phase 9.
6. **Sharing the raw data — resolved:** published as a public GitHub Release asset
   (`data-v1`, `sec10k_raw_fy2023-2025.tar.gz` + a SHA-256 manifest). `data/companyfacts/*.json`
   (184MB, git-ignored for the same reason) is shared the same way: `companyfacts-v1`,
   `sec10k_companyfacts.tar.gz` (13MB gzipped) + `manifest.sha256` inside the tarball +
   `SHA256SUMS.txt` of the tarball itself — `gh release download companyfacts-v1 -R
   dylanwang2111/synthetic-data-bootcamp -p "sec10k_companyfacts.tar.gz"`, then
   `tar -xzf sec10k_companyfacts.tar.gz -C data`.
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
- **M2 — done:** index (729 real rows at the time; 829 after Phase 8 added the synthetic ones),
  companyfacts (50/50 companies), and `calibration.json` (147 company-years, 12 metrics).
- **M3 — done:** 20 fictional company specs (Phase 4) and their full balance sheet, income
  statement, and cash-flow tables (Phase 5) are built, with every subtotal and identity checked.
- **M4/M5 — done:** all 20 fictional companies' narrative written and validated against Phase 7's
  automatic gates (`src.step09_validate_filings`, 0 failures) and a model-judged realism pass on a
  6-company sample (see Phase 7). Human review of about 10 filings is the one Phase 7 item still
  open.
- **M6 — done:** QA and retrieval evaluation (Phase 9, 130 QA pairs) and the synthetic-vs-real gap
  report (Phase 10) are both complete. Every planned phase is now done except Phase 7's human
  review of about 10 filings, which needs an actual person.
