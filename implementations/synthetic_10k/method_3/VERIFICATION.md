# Verification — 2026-10-28

## Standalone HTML — 2026-10-29

- Added `dashboard/financial_method_comparison.html` (53,726,478 bytes). Existing
  `dashboard/index.html` and `template.html` are unchanged.
- All CSS, D3, application code, source metadata and compressed financial pools are
  embedded. Four executable inline scripts compile successfully; no external asset
  references or live API requests remain.
- Contains 10 sector pools, each with 2,000 companies per native financial method;
  the real reference contains 228 eligible filings. Pools were generated with seed 42.
- Actual-artifact Node checks decoded embedded data and verified default 200 samples
  per method, 2,000 Utilities samples per method, unique IDs, stored statements and
  accounting results, fixed real reference and reproducible sampling seeds.
- 36 Python tests passed (new functionality plus original parser/spec/calibration
  suites); 4 JavaScript tests passed. The full narrative collection was attempted but
  is blocked in the current checkout by missing `anthropic` dependency. No narrative
  generation is needed for this standalone financial dashboard.
- Browser preview is unverified for this artifact: Browser Use disallows `file://`
  URLs. No alternate browser or indirect preview was used to circumvent that policy.
  Open the saved HTML directly in a recent browser for the local-file visual check.
- The standalone UI explicitly labels fixed-pool resampling, sampling seed and pool
  generation diagnostics. It does not claim to execute Python on slider changes.
- A large Utilities pool exposed finite native name reservations consumed by rejected
  profiles. The comparison adapter now releases those reservations (original source
  unchanged); a reproducing regression failed before and passed after the fix.

## Relocation requested by user

All new files are consolidated under `implementations/synthetic_10k/method_3/`;
UI assets are in its `dashboard/` subfolder. The previous
`implementations/method_3` and separate `comparison_dashboard` directories no longer
exist. Source adapters read the sibling original methods and real data without edits.
The launcher and README run from `implementations/synthetic_10k/`. The revised layout
regressions pass: 43 Python tests and 2 JavaScript tests passed after this relocation.

- New Python functionality: 16 tests passed.
- Original Method 1 suite: 25 tests passed, including narrative/validation tests.
- Combined pytest run: **41 passed**, 4.94 seconds. Test dependencies were installed
  only under ignored `method_3/data/test-deps`; no model calls or global installs.
- JavaScript statistics: **2 passed** using `node --test`.
- CLI generated 200 companies per method and saved model and statement JSON.
- Actual source: 305 parsed filings; 228 eligible common records. Remaining records
  are explicitly excluded for sector mapping, financial layout, or missing core data.
- Every existing sector plus all-sector mode generated 20 companies per method.
  Maximum all-sector batch generated 2,000 per method; maximum IT batch also tested.
- Source hashes in the saved model match current original Python source hashes.
  `git diff --exit-code -- method_1 method_2` passed; no existing method code changed.
- Browser (in-app): 12 charts rendered; no page error; industry change correctly
  selected 52 real IT reports; slider 20 and 2,000 changed all synthetic counts and
  left the real count fixed. Revenue log scale and balance-sheet/cash-flow selection
  were exercised. Default all-sector/200 view restored for delivery.
- CSV downloaded from browser: 112 records = 60 synthetic + 52 real for IT/20.
  JSON downloaded from browser: three groups of 20 plus 52 real observations.
- Browser reads use same-origin local API; assets are local, including vendored D3.

## Independent review and resolved findings

1. Fiscal-period mismatch: exact period-end filtering and native lookup calendar year
   now align template headline totals and detail facts. CRWD January-year-end cash
   matching and a small exact-period fixture are covered by regression tests.
2. Unsupported optional R&D: sector support below native Method 1's threshold uses
   the same synthetic omission policy, separately disclosed from missing real values.
   Utilities regression checks zero R&D. Omitted tiny driver changes are also counted.
3. Singleton sector: explicit pooled-model fallback; startup of other sectors is
   covered by regression. No second review was needed after covered fixes.

## Implementation rulings

- Work directly in the user-supplied checkout so newly supplied untracked data remains
  available. New modules only; no branches, commits, pushes, or hosted publication.
- Source-load pure original functions through AST to avoid CLI logging/dependencies.
  This relies on their local function/constant interfaces; future refactoring of the
  original scripts may require updating the adapter. Tests exercise current interfaces.
- Use comparison-only Method 2 identities and with-replacement templates to support
  2,000 samples. Financial functions are unchanged; this is disclosed in UI and README.
- Use the approved local buildless page rather than a hosted Data runtime.
- Running the server directly inside an execution session caused loopback timeouts.
  A hidden background process launched through native PowerShell Start-Process works;
  the delivered local URL was verified in browser. No firewall/network settings changed.

## Limits

### Dashboard metric extension (2026-10-29)

- Follow-up visual: added per-method real/synthetic scatter overlays for all 15
  selectable ratio pairs, full-data least-squares lines, shared axes, Pearson
  coefficients, selected-pair similarity and observed counts. Circles/dashed lines
  identify real data and diamonds/solid lines identify synthetic data. Plotted
  points are capped at 500 per population; fits and scores use all complete pairs.
  Rebuilt the standalone HTML with the original embedded pools. Nine Node tests
  pass, including finite fitted coefficients and score agreement with the pair
  evaluation on actual pools at slider bounds in every sector. Local-file browser
  rendering remains unverified under the existing preview restriction.

- Added JavaScript KSComplement, numeric Column Pair Trends (Pearson similarity),
  and feature-subset DCR with a uniform numeric random-baseline protection score.
  UI exposes formulas, features, valid counts, per-column/per-pair detail and privacy
  limitations. JSON exports include results and settings. Existing generators are
  unchanged; original embedded sample banks and generation timestamp are preserved.
- Rebuilt `dashboard/financial_method_comparison.html` with inline evaluation code;
  the existing local API server also serves the new evaluation module.
- Eight Node tests passed, including tests on actual delivered HTML: all inline
  JavaScript compiles, no API calls remain, and embedded pools evaluate at 20 and
  2,000 samples for all ten sectors and the pooled selection. Twenty Method 3 Python
  tests passed with the bundled Python runtime and workspace-local pytest temp path.
- Independent read-only metric/integration review found no critical or important
  issues. `git diff --check` passed; original Method 1/2 sources have no changes.
- Local-file browser rendering is not verified: the app's browser blocks file URLs.
  No alternate-browser or HTTP workaround was used to bypass that restriction.

In-sample comparison, not held-out validation. No learned multi-year dynamics. Optional
zeros and missing-value policies are modeling choices. Native gates differ, and their
corresponding rejection/repair effects remain part of the comparison. Real statement
inspection covers extracted reported metrics, not all disclosures. Gaussian copulas
cannot learn arbitrary dependence. No hardcoded claim of Method 3 superiority.
