# Joint financial generation and three-method comparison

Status: approved by the user's “实现” instruction; implemented and verified.

## Intent and agreed requirements

Add a Gaussian copula financial generator without modifying the algorithms or saved
outputs of Method 1 and Method 2. Build a Chinese-language interactive dashboard:
all three synthetic methods on the left, actual financial distributions on the
right, with a slider controlling synthetic sample count. The purpose is to inspect
whether joint generation preserves within-sector relationships better than
independent ratio sampling and real-company template perturbation.

This is a local repository feature. No LLM calls are necessary to generate numeric
statements. Existing narrative and retrieval dashboards remain intact.

## Verified source and constraints

- Source: `implementations/synthetic_10k/method_2/data/parsed/`.
- 305 company filings, each with `metadata.json`, `facts.csv`, and section text.
- Metadata contains already-extracted metrics in USD. The fact CSV preserves
  concept, value, unit, period, and dimensions for detailed statement adapters.
- Available headline coverage: revenue 295, assets 300, equity 298, net income 299,
  operating income 237, operating cash flow 296. Missing is not zero.
- The existing S&P constituents table maps 277 tickers directly; 28 need normalized
  ticker/name matching or an explicit unmapped classification. No sector guesses.
- Method 1 excludes Financials and expects its sector table, which is absent in this
  checkout. Supply an explicit sector mapping to its calibration API and construct
  specs through existing functions without changing its source or monkeypatching.
- Both methods' cash-flow and balance-sheet definitions differ. Normalize only
  comparable fields and label single-year versus multi-year coverage accurately.
- Current default Python lacks numpy, PyYAML, and pytest. Locate the bundled runtime
  or prepare isolated dependencies during implementation; do not alter global Python.

## Approaches

The user accepted the proposed Gaussian copula approach in the conversation.

1. Chosen: Gaussian copula for joint driver sampling, reusing Method 1's accounting
   engine. Small dependency footprint, inspectable dependence model, suitable baseline
   for the available small per-sector samples. Does not capture arbitrary dependence.
2. TabDDPM: possible future replacement behind the same sampler interface. Requires
   training infrastructure and stronger validation with limited sector data.
3. Independent marginals: retain as Method 1 comparison; it does not learn joint
   dependence and is not the new implementation.

## New modules and source isolation

Add `method_3/` for data preparation, a reusable copula class, generation CLI,
normalized statement adapters, and tests. Add a separate comparison dashboard
folder rather than replacing `dashboard/index.html`. New generated assets live in
a new output directory. No regeneration into original saved-output folders.

The adapter imports the existing pure financial functions. For Method 2 it converts
consolidated USD annual/instant CSV facts into an in-memory companyfacts-shaped
input, respecting fiscal periods and excluding segment facts. It calls the existing
template/statement routines without changing their formulas or source data.
Unknown or unavailable fields stay missing. No invented detailed financial inputs.

## Model and accounting

- Fit on a joint vector including log revenue and the ratios consumed by the
  accounting engine. Do not independently draw size outside this vector.
- Use empirical marginals, average ranks for ties, normal scores, and a regularized
  positive-semidefinite dependence matrix. Constant columns must remain supported.
- Estimate dependence within sectors; shrink unstable estimates toward a documented
  pooled structure. Report unique companies, usable sample counts, missingness, and
  fallback decisions. A fallback is visible, not silently presented as sector evidence.
- Missing required drivers use a documented sector/pooled fit-time imputation policy;
  optional structurally absent costs are handled separately. Preserve a missingness
  report and label generated versus directly observed fields.
- Save fitted model metadata and seed for reproducibility. Sample entire vectors,
  then apply compatible bounds and explicit consistency constraints.
- Reuse `method_1.statements.build_statements`, `check_statements`, and
  `check_plausibility`. Current/prior-year statements follow that engine; no claim
  that a one-year copula learns temporal dynamics.
- Record attempted, accepted, rejected, and adjusted samples, with reasons and
  driver adjustment sizes. Do not silently force arbitrary acceptance counts.

## Fair comparison and data scope

Calibrate Method 1, choose Method 2 templates, and fit Method 3 using the same eligible
source corpus. For selected sectors, request the same synthetic count for each method.
Use the same sector mixture in the all-sector view, so sector mixing does not explain
correlation differences. Report exclusions and differences caused by each method's
native gates. Default comparison excludes Financials, matching Method 1's scope.

Method 2's original CLI samples templates without replacement and caps its count at
available company-years. To support a larger slider without modifying it, the adapter
calls its original per-template functions with replacement and explicitly labels this
extension in the UI and export. Repeated templates are not independent real evidence.

This first dashboard is an in-sample fidelity explorer, not proof of generalization,
privacy, or downstream utility. Do not label a fitted real comparison as held-out.

## Dashboard behavior (implementation choices proposed by assistant)

- Local Python generation service and buildless HTML/JS, using vendored D3 if useful;
  opens locally, with no hosting requirement. Bind to localhost.
- Desktop layout: approximately two-thirds width for three color-coded synthetic
  methods, one-third for the fixed real reference. Responsive stacking on small screens.
- Slider: 20–2,000 synthetic companies **per method**, default 200. Debounce changes;
  latest request wins and pending/failed generation is shown. Real sample count does
  not change with the slider. Fixed-seed results are reproducible.
- Global industry filter and seed input; show real usable counts and training scope.
- Distributions: shared bins and axes, normalized percentages rather than raw counts.
  Financial amounts support log scale with explicit handling of nonpositive values.
  Never silently clip tails; disclose overflow or show complete ranges.
- Dependence: selectable driver scatter plot, within-sector correlation matrices,
  pairwise sample counts, undefined correlations displayed as unavailable, and a
  correlation-difference summary over commonly supported pairs.
- Statements: select a synthetic company in each method and inspect income statement,
  balance sheet, cash flow, and reconciliation details. Real statement fields retain
  their reported values and missing entries; do not force real accounting discrepancies
  into synthetic balancing rules.
- Quality: median/IQR, distribution distance, accounting gates, rejection/adjustment
  statistics, and fallback indicators with metric definitions.
- Download normalized sample CSV/JSON and detailed synthetic statement JSON. Source
  details show corpus location, fiscal years, units, sample exclusions, and limitations.

## Verification and acceptance

- Confirm Method 1 and Method 2 source hashes remain unchanged.
- Test deterministic copula sampling, PSD/constant/tied columns, missingness and sparse
  sectors, and dependence preservation on a controlled dataset with a known relation.
- Verify generated accounting identities and that adjustments/rejections are counted.
- Test fact adapter unit, dimension, annual-period, and duplicate selection rules.
- Verify each method returns the requested count or an explicit diagnostic failure;
  Method 2's template reuse is disclosed. Match sector composition across methods.
- Exercise industry selection, slider boundaries, seed changes, loading/error states,
  distributions, dependence matrices, statements, and downloads in browser QA.
- Compare normalized fields against known source metadata/facts and native outputs.
- Report actual test results and any unresolved limitations. No assertion that Method 3
  outperforms either baseline unless the measured results support it.

## Next step

Implemented in `implementations/synthetic_10k/method_3/`, with the dashboard in its `dashboard/` subfolder. Verification and implementation
decisions are recorded in the implementation plan and `method_3/VERIFICATION.md`.
