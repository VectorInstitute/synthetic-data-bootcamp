"""Read existing parsed filings and normalize consolidated annual USD facts."""
import csv
import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from method_3.native import ROOT

DEFAULT_PARSED = ROOT / 'method_2/data/parsed'


def facts_to_companyfacts(facts, filed, period_end=None):
    result = {'facts': {'us-gaap': {}}}
    seen = set()
    for f in facts:
        concept = f.get('concept', '')
        if not concept.startswith('us-gaap:') or f.get('unit', '').lower() not in ('usd', 'iso4217:usd'):
            continue
        if f.get('dimensions', '').strip() not in ('', '{}', 'null'):
            continue
        try:
            value = float(f['value'])
            if not math.isfinite(value):
                continue
            end = f.get('instant') or f.get('end')
            date.fromisoformat(end)
            if period_end and end != period_end:
                continue
            start = f.get('start')
            if start and not 330 <= (date.fromisoformat(end) - date.fromisoformat(start)).days <= 400:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        key = (concept, start or '', end, value)
        if key in seen:
            continue
        seen.add(key)
        entry = {'val': value, 'end': end, 'form': '10-K', 'filed': filed}
        if start:
            entry['start'] = start
        tag = concept.split(':', 1)[1]
        result['facts']['us-gaap'].setdefault(tag, {'units': {'USD': []}})['units']['USD'].append(entry)
    return result


@lru_cache(maxsize=400)
def read_companyfacts(path, filed, period_end=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return facts_to_companyfacts(csv.DictReader(f), filed, period_end)


def load_corpus(parsed=DEFAULT_PARSED):
    parsed = Path(parsed).resolve()
    with (ROOT / 'method_2/data/sp500_constituents.csv').open(encoding='utf-8-sig') as f:
        companies = list(csv.DictReader(f))
    normalize = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
    tickers = {r['Symbol'].replace('.', '-'): r for r in companies}
    names = {normalize(r['Security']): r for r in companies}
    rows, excluded = [], []
    for path in sorted(parsed.glob('*/*/metadata.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        company = tickers.get(d['ticker'].replace('.', '-')) or names.get(normalize(d.get('company', '')))
        sector = company['GICS Sector'] if company else None
        metrics = {k: (float(v) if isinstance(v, (float, int)) and math.isfinite(v) else None)
                   for k, v in d.get('metrics', {}).items()}
        reason = None
        if not sector:
            reason = 'unmapped_sector'
        elif sector == 'Financials':
            reason = 'financial_sector_layout'
        elif any(metrics.get(k) is None for k in ('revenue', 'total_assets', 'stockholders_equity', 'net_income')):
            reason = 'missing_core_metrics'
        elif metrics['revenue'] <= 0 or metrics['total_assets'] <= 0:
            reason = 'nonpositive_size'
        if reason:
            excluded.append(dict(ticker=d['ticker'], reason=reason))
            continue
        rows.append(dict(ticker=d['ticker'], name=d.get('company', d['ticker']),
                         sector=sector, sub_industry=company.get('GICS Sub-Industry', ''),
                         fiscal_year=int(d['fiscal_year']), period_end=d.get('period_end'),
                         filed=d.get('filing_date', ''),
                         path=str(path.parent), metrics=metrics))
    if not rows:
        raise ValueError(f'No usable filings in {parsed}')
    return rows, dict(path=str(parsed), total=len(rows) + len(excluded), eligible=len(rows),
                      excluded=excluded, units='USD', scope='in_sample',
                      years=sorted({r['fiscal_year'] for r in rows}),
                      period_years=sorted({int(r['period_end'][:4]) for r in rows if r['period_end']}),
                      fiscal_label_warnings=[dict(ticker=r['ticker'], fiscal_year=r['fiscal_year'], period_end=r['period_end'])
                          for r in rows if r['period_end'] and abs(r['fiscal_year']-int(r['period_end'][:4]))>1])


def template_record(row):
    m = row['metrics']
    result = dict(ticker=row['ticker'], fiscal_year=row['fiscal_year'],
                  total_assets=m['total_assets'], stockholders_equity=m['stockholders_equity'],
                  revenues=m['revenue'], net_income=m['net_income'],
                  equity_to_assets=m['stockholders_equity'] / m['total_assets'],
                  net_margin=m['net_income'] / m['revenue'])
    if m.get('operating_income') is not None:
        result.update(operating_income=m['operating_income'],
                      operating_margin=m['operating_income'] / m['revenue'])
    return result
