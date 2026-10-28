"""Three original/accounting-compatible generators on a common real population."""
from collections import Counter
import math
import random
from pathlib import Path
import numpy as np
from method_3.copula import GaussianCopula
from method_3.native import native_functions, source_hashes
from method_3.sources import DEFAULT_PARSED, load_corpus, read_companyfacts, template_record


def driver_adjustments(raw, ratios, actual):
    """Count removed optional drivers as zero, including small R&D omissions."""
    changes = {}
    for key, before in raw.items():
        if key == 'log10_revenue':
            continue
        after = actual.get(key, ratios.get(key, 0))
        if abs(after - before) > 1e-8:
            changes[key] = dict(before=before, after=after, delta=after-before)
    return changes


def derived_metrics(metrics):
    m = dict(metrics)
    pairs = dict(gross_margin=('gross_profit', 'revenue'),
                 operating_margin=('operating_income', 'revenue'),
                 net_margin=('net_income', 'revenue'),
                 asset_turnover=('revenue', 'total_assets'),
                 equity_ratio=('stockholders_equity', 'total_assets'),
                 liabilities_ratio=('total_liabilities', 'total_assets'),
                 cash_ratio=('cash', 'total_assets'),
                 rd_ratio=('rd_expense', 'revenue'),
                 ocf_margin=('operating_cash_flow', 'revenue'))
    for key, (a, b) in pairs.items():
        m[key] = m[a] / m[b] if m.get(a) is not None and m.get(b) not in (None, 0) else None
    return m


def normalized_m1(spec):
    st, fy = spec['statements'], str(spec['fiscal_year'])
    inc, bs, cf = st['income_statement'][fy], st['balance_sheet'][fy], st['cash_flow_statement'][fy]
    mapping = dict(revenue=inc['revenue'], gross_profit=inc['gross_profit'],
                   operating_income=inc['operating_income'], net_income=inc['net_income'],
                   total_assets=bs['total_assets'], total_liabilities=bs['total_liabilities'],
                   stockholders_equity=bs['total_equity'], cash=bs['cash'],
                   rd_expense=inc['research_and_development'],
                   operating_cash_flow=cf['net_cash_from_operating_activities'])
    return derived_metrics({k: v * 1e6 for k, v in mapping.items()})


def normalized_m2(spec, st):
    inc, bs, cf = st['income_statement'], st['balance_sheet'], st['cash_flow_statement']
    expenses = inc['operating_expenses']
    revenue = inc['revenues']
    cost = expenses.get('cost_of_revenue')
    return derived_metrics(dict(revenue=revenue,
        gross_profit=revenue - cost if cost is not None else None,
        operating_income=inc.get('operating_income'), net_income=inc['net_income'],
        total_assets=bs['assets']['total_assets'],
        total_liabilities=bs['liabilities']['total_liabilities'],
        stockholders_equity=bs['equity']['total_stockholders_equity'],
        cash=bs['assets'].get('cash_and_equivalents'),
        rd_expense=expenses.get('research_and_development'),
        operating_cash_flow=cf['operating_activities']['cash_from_operations']))


class ComparisonEngine:
    def __init__(self, parsed=DEFAULT_PARSED):
        self.rows, self.provenance = load_corpus(parsed)
        self.hashes = source_hashes()
        self.m1 = native_functions('method_1/specs.py')
        self.accounting = native_functions('method_1/statements.py')
        self.m1['build_statements'] = self.accounting['build_statements']
        self.cal = native_functions('method_1/calibrate.py')
        self.m2 = native_functions('method_2/step06_make_specs.py')
        self.m2st = native_functions('method_2/step07_make_financials.py')
        self.annual = native_functions('method_2/step05_make_calibration.py')['annual_value']
        self.m2st['annual_value'] = self.annual
        self.by_sector = {s: [r for r in self.rows if r['sector'] == s]
                          for s in sorted({r['sector'] for r in self.rows})}
        # Existing calibration API receives the explicit local sector mapping.
        flat = [{**r['metrics'], 'ticker': r['ticker']} for r in self.rows]
        self.calibration = self.cal['build_calibration'](flat, [], {r['ticker']: r['sector'] for r in self.rows})
        self.features = ['log10_revenue', *self.m1['REQUIRED'], *self.m1['OPTIONAL'], 'other_operating_cash_ratio']
        vectors = []
        for row in self.rows:
            ratios = self.cal['company_ratios'](row['metrics'])
            m = row['metrics']
            if all(m.get(k) is not None for k in ('operating_cash_flow', 'net_income', 'depreciation_amortization')):
                ratios['other_operating_cash_ratio'] = (m['operating_cash_flow'] - m['net_income'] - m['depreciation_amortization']) / m['revenue']
            vectors.append([ratios.get(k, np.nan) for k in self.features])
        self.matrix = np.asarray(vectors)
        self.pooled = GaussianCopula.fit(self.matrix)
        self.models, self.model_info = {}, {}
        for sector, rows in self.by_sector.items():
            indexes = [i for i, r in enumerate(self.rows) if r['sector'] == sector]
            matrix = self.matrix[indexes].copy()
            missing = ~np.isfinite(matrix)
            borrowed, optional_zero = [], []
            for j, feature in enumerate(self.features):
                if feature in self.m1['OPTIONAL'] and np.isfinite(matrix[:, j]).sum() < self.cal['MIN_SECTOR_N']:
                    matrix[:, j] = 0
                    optional_zero.append(feature)
                elif np.isfinite(matrix[:, j]).sum() < 3:
                    # Explicit fit-time pooled median fallback; observed cells survive.
                    matrix[missing[:, j], j] = np.median(self.pooled.marginals[j])
                    borrowed.append(feature)
            weight = min(0.85, len(self.features) / (len(rows) + len(self.features)))
            self.models[sector] = (self.pooled if len(rows) < 2 else
                GaussianCopula.fit(matrix, shrinkage=weight, prior=self.pooled.correlation))
            self.model_info[sector] = dict(n=len(rows), shrinkage=weight,
                fit_policy='pooled_singleton' if len(rows) < 2 else 'sector_shrunk',
                pooled_fallback=borrowed, optional_zero=optional_zero,
                missing={k: int(missing[:, j].sum()) for j, k in enumerate(self.features)},
                observed={k: int((~missing[:, j]).sum()) for j, k in enumerate(self.features)})

    def model_snapshot(self):
        return dict(features=self.features, source=self.provenance, diagnostics=self.model_info,
                    pooled=self.pooled.to_dict(),
                    sectors={s: m.to_dict() for s, m in self.models.items()}, source_hashes=self.hashes)

    def statement_inputs(self, row):
        template = template_record(row)
        end = row['period_end']
        if not end:
            raise ValueError(f"{row['ticker']}: missing actual period end")
        # Original annual_value selects the calendar year of the fact end, which
        # need not equal the company's fiscal label (e.g. January fiscal endings).
        template['fiscal_year'] = int(end[:4])
        facts = read_companyfacts(str(Path(row['path']) / 'facts.csv'), row['filed'], end)
        return template, facts

    def generate(self, n=200, seed=42, sector='all'):
        if isinstance(n, bool) or not isinstance(n, int) or not 20 <= n <= 2000:
            raise ValueError('Sample count must be an integer from 20 to 2,000')
        if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Seed must be an integer from 0 to 4294967295')
        if sector != 'all' and sector not in self.by_sector:
            raise ValueError('Unknown sector')
        rng = random.Random(seed)
        sectors = list(self.by_sector)
        # One schedule reused across all methods: sector composition cannot confound results.
        schedule = ([sector] * n if sector != 'all' else
                    rng.choices(sectors, weights=[len(self.by_sector[s]) for s in sectors], k=n))
        counts = Counter(schedule)
        real = [dict(id=r['ticker'], name=r['name'], sector=r['sector'],
                     fiscal_year=r['fiscal_year'], period_end=r['period_end'], metrics=derived_metrics(r['metrics']),
                     statements={'units': 'USD', 'reported_metrics': r['metrics']},
                     checks=None) for r in self.rows if sector == 'all' or r['sector'] == sector]
        methods = [self._generate_m1(schedule, seed), self._generate_m2(schedule, seed),
                   self._generate_joint(schedule, seed)]
        return dict(real=real, methods=methods, n=n, seed=seed, sector=sector,
                    sector_counts=dict(counts), sectors=sectors, provenance=self.provenance,
                    model_info={s: self.model_info[s] for s in counts},
                    source_hashes=self.hashes,
                    limitations=['The same real observations are used for fitting and comparison: this is an in-sample fidelity check.',
                      'Unmapped sectors, financial companies and records missing core metrics are excluded. See source details.',
                      'Method 2 samples templates with replacement within sectors. Its original CLI samples without replacement and caps the count.',
                      'Methods 1 and 3 include a rule-generated prior year. Method 2 is single-year. The joint model does not learn temporal dynamics.',
                      'Zero and missing are different. Fitting uses median imputation; real distributions are not imputed.',
                      'Cash ratio is cash/assets. Liabilities ratio is total liabilities/assets, not interest-bearing debt/assets.'])

    def _generate_m1(self, schedule, seed):
        rng = random.Random(seed)
        taken = {r['ticker'] for r in self.rows}
        output, attempts, reasons = [], 0, Counter()
        for i, sector in enumerate(schedule):
            for trial in range(300):
                attempts += 1
                spec = self.m1['sample_spec'](self.calibration, rng, taken=taken,
                    index=i + 1, fiscal_year=2025, sector=sector)
                issues = self.accounting['check_statements'](spec['statements']) + self.accounting['check_plausibility'](spec['statements'])
                if not issues:
                    output.append(dict(id=spec['company_id'], name=spec['name'], sector=sector,
                        fiscal_year=spec['fiscal_year'], metrics=normalized_m1(spec),
                        statements=spec['statements'], checks=[], input_drivers=spec['ratios']))
                    break
                reasons.update(issues)
            else:
                raise ValueError(f'Method 1 failed in {sector} after 300 consecutive attempts')
        return dict(id='method_1', label='Method 1 · Independent sampling', rows=output,
                    diagnostics=dict(attempted=attempts, accepted=len(output),
                     rejected=attempts-len(output), rejection_reasons=dict(reasons),
                     adjusted=None, adjustment_note='Internal adjustments in the original method are not separately measured; they are not reported as zero.'))

    def _generate_m2(self, schedule, seed):
        rng = random.Random(seed)
        output, used = [], Counter()
        for i, sector in enumerate(schedule):
            row = rng.choice(self.by_sector[sector])
            template = template_record(row)
            # Reuse the original financial routine. Comparison-only IDs avoid the
            # finite native name bank (20 * 20 * 5 names) at the 2,000 sample limit.
            spec = dict(fictional_id=f'template{i+1:04d}',
                        fictional_name=f'Template Company {i+1:04d}',
                        fiscal_year=template['fiscal_year'],
                        financials=self.m2['build_financials'](rng, template))
            statement_template, facts = self.statement_inputs(row)
            st = self.m2st['build_financials'](rng, spec, statement_template, facts)
            issues = self.m2st['check_reconciliation'](st, spec['financials'])
            used[row['ticker']] += 1
            output.append(dict(id=spec['fictional_id'], name=spec['fictional_name'], sector=sector,
                fiscal_year=spec['fiscal_year'], metrics=normalized_m2(spec, st),
                statements={'units': 'USD', **st}, checks=issues, template=row['ticker'], period_end=row['period_end']))
        return dict(id='method_2', label='Method 2 · Template perturbation', rows=output,
                    diagnostics=dict(attempted=len(output), accepted=len(output), rejected=0,
                        adjusted=None, adjustment_note='Original scaling, perturbation and residual rules are retained. Adjustment rates are not separately measured.',
                        template_reuses=sum(v-1 for v in used.values()), unique_templates=len(used)))

    def _generate_joint(self, schedule, seed):
        rng = random.Random(seed)
        output, attempts, adjusted, reasons, changes = [], 0, 0, Counter(), Counter()
        magnitude, vectors, offsets = Counter(), {}, Counter()
        for sector, count in Counter(schedule).items():
            vectors[sector] = self.models[sector].sample(max(100, count * 8), rng.randrange(2**32))
        for i, sector in enumerate(schedule):
            for trial in range(300):
                attempts += 1
                if offsets[sector] >= len(vectors[sector]):
                    vectors[sector] = self.models[sector].sample(100, rng.randrange(2**32))
                    offsets[sector] = 0
                raw = dict(zip(self.features, map(float, vectors[sector][offsets[sector]])))
                offsets[sector] += 1
                ratios = {k: v for k, v in raw.items() if k != 'log10_revenue'}
                for key in self.m1['REQUIRED'] + self.m1['OPTIONAL']:
                    lo, hi = self.m1['BOUNDS'].get(key, (0, float('inf')))
                    ratios[key] = min(max(ratios[key], lo), hi)
                for key in self.model_info[sector]['optional_zero']:
                    ratios.pop(key, None)
                # Same explicit cross-ratio constraints as Method 1, without redrawing columns.
                if ratios.get('rd_ratio', 0) < 0.005:
                    ratios.pop('rd_ratio', None)
                ratios['gross_margin'] = max(ratios['gross_margin'], ratios.get('rd_ratio', 0) + 0.05)
                room = ratios['gross_margin'] - ratios.get('rd_ratio', 0)
                ratios['operating_margin'] = min(ratios['operating_margin'], room - 0.03)
                ratios['sga_ratio'] = min(ratios['sga_ratio'], room - ratios['operating_margin'])
                spec = dict(company_id=f'joint{i+1:04d}', name=f'Joint Company {i+1:04d}',
                    sector=sector, fiscal_year=2025,
                    revenue=max(1, round(10 ** raw['log10_revenue'] / 1e6)),
                    revenue_growth=round(rng.gauss(0.06, 0.08), 4), ratios=ratios)
                spec['statements'] = self.accounting['build_statements'](spec)
                issues = self.accounting['check_statements'](spec['statements']) + self.accounting['check_plausibility'](spec['statements'])
                if issues:
                    reasons.update(issues)
                    continue
                metrics = normalized_m1(spec)
                # Capture actual realized ratio changes caused inside the accounting engine too.
                actual = self.cal['company_ratios'](self._statement_metrics(spec))
                deltas = driver_adjustments(raw, ratios, actual)
                adjusted += bool(deltas)
                for key, value in deltas.items():
                    changes[key] += 1
                    magnitude[key] += abs(value['delta'])
                output.append(dict(id=spec['company_id'], name=spec['name'], sector=sector,
                    fiscal_year=2025, metrics=metrics, statements=spec['statements'], checks=[],
                    raw_drivers=raw, input_drivers=ratios, adjustments=deltas))
                break
            else:
                raise ValueError(f'Joint generation failed in {sector} after 300 consecutive attempts')
        return dict(id='method_3', label='Method 3 · Joint copula', rows=output,
            diagnostics=dict(attempted=attempts, accepted=len(output), rejected=attempts-len(output),
                adjusted=adjusted, rejection_reasons=dict(reasons), adjustments_by_field=dict(changes),
                mean_absolute_adjustment={k: magnitude[k]/v for k, v in changes.items()},
                adjustment_note='Includes constraint repairs, accounting-engine adjustments and ratio changes caused by rounding to USD millions.'))

    @staticmethod
    def _statement_metrics(spec):
        st, fy = spec['statements'], str(spec['fiscal_year'])
        inc, b, c = st['income_statement'][fy], st['balance_sheet'][fy], st['cash_flow_statement'][fy]
        return dict(revenue=inc['revenue'], gross_profit=inc['gross_profit'],
            operating_income=inc['operating_income'], net_income=inc['net_income'],
            rd_expense=inc['research_and_development'], sga_expense=inc['selling_general_and_administrative'],
            income_tax=inc['income_tax'], pretax_income=inc['pretax_income'],
            total_assets=b['total_assets'], stockholders_equity=b['total_equity'],
            current_assets=b['total_current_assets'], current_liabilities=b['total_current_liabilities'],
            cash=b['cash'], ppe_net=b['property_plant_and_equipment'], goodwill=b['goodwill'],
            long_term_debt=b['long_term_debt'], inventory=b['inventory'], receivables=b['receivables'],
            depreciation_amortization=c['depreciation_and_amortization'],
            capex=abs(c['capital_expenditures']), dividends_paid=abs(c['dividends_paid']))
