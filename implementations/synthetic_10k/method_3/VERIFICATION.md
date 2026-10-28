# Verification — 2026-10-28

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

In-sample comparison, not held-out validation. No learned multi-year dynamics. Optional
zeros and missing-value policies are modeling choices. Native gates differ, and their
corresponding rejection/repair effects remain part of the comparison. Real statement
inspection covers extracted reported metrics, not all disclosures. Gaussian copulas
cannot learn arbitrary dependence. No hardcoded claim of Method 3 superiority.
