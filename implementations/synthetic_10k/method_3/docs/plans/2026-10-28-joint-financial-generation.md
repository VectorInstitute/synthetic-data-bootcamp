# Joint financial generation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task.

**Goal:** Add a joint financial sampler and local interactive three-method comparison.
**Architecture:** Read parsed metadata and consolidated XBRL facts, fit sector copulas,
reuse both original algorithms through side-effect-free source adapters, and serve
normalized results to a buildless Chinese dashboard.
**Tech stack:** Python 3.12+, numpy, unittest, HTTP server, HTML/CSS/JavaScript.
**Spec:** ../specs/2026-10-28-joint-financial-generation-design.md

## Global constraints
- Preserve all Method 1 and Method 2 source files and saved outputs.
- Use local parsed source, never fabricate observed missing fields.
- Slider 20–2,000 companies per method; fixed real reference and matched sector counts.
- Disclose fit-time imputation, template reuse, constraints, and in-sample comparison.

## Review focus
- Sparse sectors and missing columns: finite fit with visible fallback diagnostics.
- Dimensional/quarterly/non-USD facts: excluded from the detailed adapter.
- Failed generation: bounded retries with explicit error, no partial success claims.
- Rapid slider changes: latest result wins, failed generation cannot show stale exports.
- Unknown sectors and accounting differences: visible exclusions, no silent relabeling.

## Task 1: Joint model and source ingestion
Files: method_3/copula.py, sources.py, native.py, tests/test_core.py.
Interfaces: GaussianCopula.fit(matrix), sample(n, seed); load_corpus(path),
companyfacts(row), native_functions(name).
- [ ] Write and run failing unittest cases for seeded dependence, tied/constant data,
  sparse/missing columns, and annual/unit/dimension selection.
- [ ] Implement empirical marginals, shrinkage and PSD repair; read-only native adapter.
- [ ] Run unittest core cases; expected all passing.

## Task 2: Generator and comparison API
Files: method_3/generate.py, comparison.py, server.py, tests/test_generation.py.
Interfaces: ComparisonEngine.generate(n, seed, sector) returns real, methods, metadata;
python -m method_3.generate; python -m method_3.server.
- [ ] Write and run failing count/reconciliation/reproducibility/sector tests.
- [ ] Reuse existing accounting and template functions, collect diagnostics, normalize USD.
- [ ] Implement bounded server routes, safe static serving, JSON and CSV exports.
- [ ] Run core/generation tests, generate actual 20/200/2,000 batches and boundary errors.

## Task 3: Interactive dashboard and delivery
Files: method_3/dashboard/index.html, app.js, style.css; method_3/README.md.
- [ ] Define UI smoke acceptance in test checks before authoring controls.
- [ ] Implement matched distribution/scatter/correlation views, statements, controls,
  diagnostics, source inspection, exports, keyboard focus, responsive layout.
- [ ] Browser QA: initial load, sector and slider changes, sample tables and downloads.
- [ ] Run native Method 1 suite where dependencies permit; record failures explicitly.
- [ ] Verify original hashes, fresh reviewer, fix material findings with regression tests.

## Completion ledger
Task 1: complete — copula/source/period tests RED→GREEN.
Task 2: complete — all methods, accounting, seed, bounds, matched sectors and 2,000
samples tested. Singleton fallback and optional ratio policies covered by regressions.
Task 3: complete — browser verified controls, 12 charts, statements and actual downloads.
Final review: three material findings fixed with regressions; combined relevant suite
41/41 passed; JavaScript 2/2 passed. Original source hashes unchanged. See
`implementations/synthetic_10k/method_3/VERIFICATION.md` for evidence and rulings.

Relocation: at the user's request, all new code/assets/tests/docs/output now reside
under `implementations/synthetic_10k/method_3/`. Original methods and input corpus stay under
`implementations/synthetic_10k/` as read-only dependencies. Added layout regressions.

## Execution ledger
User authorized implementation on 2026-10-28; execute continuously in this session.
Ruling: work in the existing checkout because supplied parsed data is untracked and
the user requested this repository. Only new modules are written; no branch/push/commit.
Ruling: source-load original pure functions via AST to avoid original CLI import-time
logging and unrelated edgartools/PyYAML dependencies. Algorithms and function bodies
remain unchanged and their original hashes are checked.
Ruling: explicit local dashboard approved in design takes precedence over default
hosted Data/Sites runtime scaffolding. Use the repository's buildless local pattern.
