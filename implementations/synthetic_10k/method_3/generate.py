"""Reproducible CLI for the joint generator and three-method comparison."""
import argparse
import json
from pathlib import Path
from method_3.comparison import ComparisonEngine
from method_3.native import PACKAGE_DIR
from method_3.sources import DEFAULT_PARSED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parsed', type=Path, default=DEFAULT_PARSED)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sector', default='all')
    parser.add_argument('--output', type=Path, default=PACKAGE_DIR / 'data')
    args = parser.parse_args()
    engine = ComparisonEngine(args.parsed)
    result = engine.generate(args.n, args.seed, args.sector)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (args.output / 'joint_model.json').write_text(json.dumps(engine.model_snapshot(), ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (args.output / 'joint_financials.json').write_text(json.dumps(result['methods'][2], ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f"Generated {args.n} companies per method; {len(result['real'])} real references. Output: {args.output.resolve()}")


if __name__ == '__main__':
    main()
