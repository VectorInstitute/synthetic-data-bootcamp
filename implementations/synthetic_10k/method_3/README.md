# Method 3: joint financial generation

Jointly sample company size and financial drivers with a regularized Gaussian
copula, then calculate statements through Method 1's existing accounting engine.
The original Method 1 and Method 2 files and saved results are not modified.

## Run the dashboard

### One-file offline dashboard

Open `../dashboard/financial_method_comparison.html` directly in a recent browser.
It includes the charts, styles, 228 eligible real observations, and native-generated
per-sector pools with complete synthetic statements. No server or internet is needed.

The slider selects 20–2,000 samples per method from these **fixed embedded pools**.
"Sampling seed" changes the selection; it does not execute the Python generators.
Generation diagnostics describe the full embedded pools, not the resampled selection.
Sector filters, plots, statement selection and exports all work inside the HTML file.

Rebuild the snapshot from `implementations/synthetic_10k` when source data or generators change:

```powershell
python -m method_3.build_standalone
```

The default output is `dashboard/financial_method_comparison.html`, leaving the
existing `dashboard/index.html` intact. The default pool contains 2,000 companies
per method per sector, with generation seed 42. `--pool-size`, `--seed`, `--parsed`,
and `--output` can be supplied to rebuild a different snapshot. Compression is embedded
in the file; opening it requires browser support for `DecompressionStream('gzip')`.

### Live Python generation dashboard

All new code, dashboard assets, tests, documentation, and generated outputs live
under `implementations/synthetic_10k/method_3/`. The existing methods and real input files remain
read-only dependencies under `implementations/synthetic_10k/`.

From `implementations/synthetic_10k`, using a Python environment with numpy:

```powershell
python -m pip install -r method_3/requirements.txt
python -m method_3.server --port 8766
```

Open http://127.0.0.1:8766. On Windows you can also run:

```powershell
powershell -File method_3/dashboard/start_dashboard.ps1
```

The launcher checks the repository's `.venv`, the bundled Codex Python, and then
the Python on PATH. It does not install anything into the global environment.
The live generation dashboard requires the local server; opening its source `index.html` directly cannot
call the generator. The server only listens on 127.0.0.1. Stop it with Ctrl+C.

Controls include industry, fixed random seed, and a slider for **20–2,000 companies
per synthetic method**. Real observations stay fixed within the selected industry.
The left side shows all three methods; the right side shows the actual reports.
Distributions use common bins/axes and fractions of valid observations. Scatter
plots show at most 500 evenly selected points while statistics use every valid pair.
The page includes correlation matrices, statement inspection, rejection and
adjustment diagnostics, and CSV/full-JSON download.

## Generate files or reuse the engine

```powershell
python -m method_3.generate --n 200 --seed 42
python -m method_3.generate --n 500 --sector "Information Technology" --output method_3/data/technology
```

Outputs: `comparison.json` (all methods and real reference), `joint_financials.json`
(joint-method statements), and `joint_model.json` (empirical marginals, learned
dependence, missingness, fallback decisions, and original-source hashes).

```python
from method_3.comparison import ComparisonEngine
engine = ComparisonEngine()
result = engine.generate(n=200, seed=42, sector="all")
joint_companies = result["methods"][2]["rows"]
```

The fitted `GaussianCopula` is also independently reusable through `fit`, `sample`,
and `to_dict`. Its input columns must have some observed values, and it requires
at least two rows. The engine handles singleton sectors through an explicit pooled
fallback instead of claiming a sector-specific estimate.

## Source, comparison scope, and interpretation

Default input is `implementations/synthetic_10k/method_2/data/parsed/*/*/metadata.json` and each filing's
`facts.csv`. No SEC network or model API calls are needed. Explicit sector mapping
comes from `method_2/data/sp500_constituents.csv`; missing sectors are excluded,
not guessed. Financial companies are excluded to match Method 1's accounting layout.
Use `--parsed PATH` on either CLI for another corpus with the same schema.

All three methods use the same eligible source population and exactly the same
synthetic industry schedule for a given seed. Method 1 retains its native percentile
sampling, constraints, and rejection gates. Method 2 retains its native financial
scaling/jitter and detailed statement functions. To exceed its native CLI's finite
template/name limits, the comparison adapter selects templates **with replacement**
within industry and assigns comparison-only IDs. This extension is disclosed in UI.

Original pure functions are loaded from their local source AST without running CLI
imports, log creation, or requiring unrelated edgartools/PyYAML dependencies. Their
function bodies are unchanged. The adapter only accepts consolidated annual or
instant USD facts at the **exact report period end**; it excludes segments, quarters,
non-USD facts, and duplicates. Native detailed lookup receives the calendar year of
that end date, while the displayed original fiscal label is retained. This matters
for January year ends and mislabeled fiscal-year metadata.

Method 3 uses empirical per-column marginals and tied-rank Gaussian scores. Its
sector correlation estimates shrink toward the pooled correlation matrix and are
projected to a positive definite matrix. Missing fit cells use medians, with explicit
pooled fallback when a sector has fewer than three observations in a required column.
Optional ratios reported by fewer than eight companies in a sector are omitted
(modeled as zero) following Method 1's policy, rather than borrowing another sector's
R&D/asset profile. This is a modeling assumption; missing real observations remain
missing. Diagnostics preserve observed and missing counts and the optional-zero list.

Revenue is jointly sampled with ratios. The residual operating-cash driver is
estimated from `(OCF - net income - depreciation)/revenue` where reported. Subsequent
constraints and accounting operations can change the sampled ratios; diagnostics
include realized changes and rounding. For Methods 1/2, internal adjustment rates
are unavailable and are not falsely reported as zero. Their native residual rules
and economic constraints differ, so the comparison concerns the full pipelines.

Amounts in Method 1/3 detailed statements are rounded USD millions. Method 2's are
USD; normalized comparison metrics are always USD. The asset turnover definition is
revenue divided by year-end assets (matching the existing engine), not average assets.
Methods 1/3 have rule-generated prior-year statements; the copula does not learn a
time-series model. Real statement inspection shows the extracted reported metrics,
not every disclosure/note in the original filing. Fiscal labels are retained even
where source metadata is unusual; source diagnostics expose inconsistent labels.

This is **in-sample fidelity exploration**. Gaussian copulas do not model arbitrary
nonlinear/tail dependence, and rejection/repair changes the output distribution.
A smaller correlation/KS difference is not proof of generalization, privacy, or
downstream benchmark utility. No superiority claim is hardcoded into the page.

## Tests

The standalone HTML also includes KSComplement for the 16 displayed financial
metrics and numeric Column Pair Trends for the 15 unique pairs of six ratios in
the correlation matrix. The latter uses Pearson correlation similarity; missing
or constant pairs are unavailable and coverage is shown. These are JavaScript
implementations of the documented SDMetrics formulas on explicit feature subsets,
not an SDMetrics QualityReport run. Each method has expandable per-column and
per-pair results; JSON exports retain evaluation details.

Column Pair Trends also has a selectable overlay plot for each method: real circles
and synthetic diamonds, separate least-squares lines, shared axes, selected-pair
similarity, Pearson coefficients and complete-observation counts. The selector
covers the same 15 ratio pairs as the aggregate. Each population displays at most
500 points; all complete observations contribute to the fitted lines and scores.

Privacy diagnostics show range-normalized mean-absolute DCR on revenue, asset
turnover, operating margin, equity/assets and cash/assets. Constant/unobserved real
features are excluded; missing-to-observed distances are 1 and two missing values
match at 0. A shared seeded uniform baseline draws observed numeric ranges without
missing values; its median DCR provides a capped median-ratio protection score.
The real panel uses leave-one-out distances. Zero distance only means a match on
the active feature subset. These are closeness diagnostics, not privacy guarantees;
there is no independent held-out set to assess overfitting protection. Higher DCR
can reflect unrealistic records. Sample count and sector changes recalculate all
metrics; real reference data stays fixed within the chosen sector.

```powershell
python -m unittest discover -s method_3/tests -v
node --test method_3/dashboard/statistics.test.cjs method_3/dashboard/standalone.test.cjs method_3/dashboard/evaluation.test.cjs method_3/dashboard/artifact-evaluation.test.cjs
```

Tests cover reproducibility, dependence, tied/constant/missing columns, exact fiscal
period matching, sector fallbacks, structural optional ratios, accounting identities,
2,000-company generation, common industry counts, input bounds, safe static paths,
and null-aware statistics. The original source hashes are compared before/after
generation. Existing Method 1's pytest suite remains separate.
