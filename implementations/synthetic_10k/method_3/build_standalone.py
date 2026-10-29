"""Build one offline HTML dashboard with native-generated per-sector sample pools.

Run from implementations/synthetic_10k:
    python -m method_3.build_standalone
"""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
from method_3.comparison import ComparisonEngine
from method_3.native import PACKAGE_DIR, ROOT
from method_3.sources import DEFAULT_PARSED

OUTPUT = ROOT / 'dashboard/financial_method_comparison.html'
ASSETS = PACKAGE_DIR / 'dashboard'


def inline_script(source):
    return '<script>\n' + re.sub(r'</script', r'<\\/script', source, flags=re.I) + '\n</script>'


def render_html(snapshot):
    page = (ASSETS / 'index.html').read_text(encoding='utf-8')
    page = re.sub(r'<script\b[^>]*\bsrc="[^"]+"[^>]*></script>', '', page)
    page = page.replace('<link rel="stylesheet" href="/style.css">',
                        '<style>' + (ASSETS / 'style.css').read_text(encoding='utf-8') + '</style>')
    page = page.replace('Local data lab', 'Standalone snapshot')
    page = page.replace('Random seed', 'Sampling seed').replace('Generate again', 'Resample')
    page = page.replace('Synthetic samples per method', 'Samples per method')
    page = page.replace('<div class="scope">',
        '<p class="source" style="margin:15px 0">Offline snapshot: controls resample fixed embedded pools. '
        'They do not run the Python generators. Generation diagnostics describe the full source pools.</p><div class="scope">')
    app = (ASSETS / 'app.js').read_text(encoding='utf-8')
    online_generate = "const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n,seed,sector}),signal:controller.signal});const result=await response.json();if(!response.ok)throw new Error(result.error||'Generation failed');"
    online_info = "const response=await fetch('/api/info');if(!response.ok)throw new Error('Unable to load source data');const info=await response.json();"
    if online_generate not in app or online_info not in app:
        raise ValueError('Dashboard API interface changed; update the standalone adapter')
    app = app.replace(online_generate, 'const result=await window.financialDataProvider.generate(n,seed,sector);')
    app = app.replace(online_info, 'const info=await window.financialDataProvider.info();')
    app = app.replace('Generating and checking three sets of statements…', 'Sampling embedded financial statements…')
    app = app.replace('Settings changed; generation pending…', 'Settings changed; resampling pending…')
    app = app.replace('Generation and reconciliation', 'Pool generation and reconciliation')
    app = app.replace('Attempts / rejections', 'Pool attempts / rejections')
    app = app.replace('Rejection rate', 'Pool rejection rate')
    app = app.replace('Accepted samples with adjustments', 'Adjusted pool samples')
    app = app.replace('Loading failed. Check the local server.', 'Loading failed. Use a recent browser with gzip decompression support.')
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    scripts = '<script type="application/json" id="embedded-financial-data">' + encoded + '</script>\n'
    for name in ['vendor/d3.v7.9.0.min.js', 'statistics.js', 'evaluation.js', 'standalone.js']:
        scripts += inline_script((ASSETS / name).read_text(encoding='utf-8'))
    scripts += inline_script(app)
    return page.replace('</body>', scripts + '\n</body>')


def build_snapshot(engine, pool_size=2000, seed=42):
    reference = engine.generate(20, seed, 'all')
    shared = {k: reference[k] for k in ['real','sectors','provenance','source_hashes','limitations']}
    shared['model_info'] = engine.model_info
    pools = {}
    for sector in shared['sectors']:
        result = engine.generate(pool_size, seed, sector)
        data = json.dumps({'methods': result['methods']}, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
        pools[sector] = base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode('ascii')
        print(f'{sector}: {pool_size:,} per method, {len(data)/1e6:.1f} MB raw, {len(pools[sector])/1e6:.1f} MB embedded', flush=True)
    return dict(schema_version=1, built_at=datetime.now(timezone.utc).isoformat(),
                source_generation_seed=seed, pool_size=pool_size, shared=shared, pools=pools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parsed', type=Path, default=DEFAULT_PARSED)
    parser.add_argument('--pool-size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 200 <= args.pool_size <= 2000:
        parser.error('--pool-size must be 200–2,000 (dashboard defaults to 200)')
    engine = ComparisonEngine(args.parsed)
    snapshot = build_snapshot(engine, args.pool_size, args.seed)
    html = render_html(snapshot)
    # Keep supported slider bounds consistent with the actual embedded pool size.
    html = html.replace('max="2000"', f'max="{args.pool_size}"').replace('<span>2,000</span>', f'<span>{args.pool_size:,}</span>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    print(f'Standalone dashboard: {args.output.resolve()} ({args.output.stat().st_size/1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
