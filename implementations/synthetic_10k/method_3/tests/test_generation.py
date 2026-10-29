"""Integration tests using the supplied real filing corpus."""
import unittest
from unittest.mock import patch
from collections import Counter
from method_3.comparison import ComparisonEngine
from method_3.native import source_hashes


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = source_hashes()
        cls.engine = ComparisonEngine()

    def test_three_methods_reconcile_match_sectors_and_preserve_sources(self):
        out = self.engine.generate(20, 42, 'all')
        self.assertEqual(len(out['methods']), 3)
        counts = []
        for method in out['methods']:
            self.assertEqual(len(method['rows']), 20)
            counts.append(Counter(r['sector'] for r in method['rows']))
            for row in method['rows']:
                m = row['metrics']
                self.assertLessEqual(abs(m['total_assets'] - m['total_liabilities'] - m['stockholders_equity']), 1)
                self.assertEqual(row['checks'], [])
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])
        self.assertEqual(self.before, source_hashes())

    def test_seed_real_baseline_and_selected_sector(self):
        sector = 'Information Technology'
        a = self.engine.generate(20, 7, sector)
        b = self.engine.generate(20, 7, sector)
        self.assertEqual(a['methods'], b['methods'])
        self.assertEqual(a['real'], self.engine.generate(30, 7, sector)['real'])
        self.assertTrue(all(r['sector'] == sector for m in a['methods'] for r in m['rows']))

    def test_bad_controls_rejected(self):
        for n in (0, 19, 2001):
            with self.assertRaises(ValueError):
                self.engine.generate(n, 42, 'all')
        with self.assertRaises(ValueError):
            self.engine.generate(20, 42, 'nonexistent')

    def test_maximum_slider_does_not_exhaust_native_name_bank(self):
        out = self.engine.generate(2000, 42, 'Information Technology')
        self.assertTrue(all(len(m['rows']) == 2000 for m in out['methods']))
        self.assertEqual(len({r['id'] for r in out['methods'][1]['rows']}), 2000)

    def test_optional_unsupported_sector_costs_not_borrowed_from_technology(self):
        out = self.engine.generate(20, 42, 'Utilities')
        self.assertTrue(all(r['input_drivers'].get('rd_ratio', 0) == 0 for r in out['methods'][2]['rows']))

    def test_singleton_sector_falls_back_without_blocking_other_sectors(self):
        rows, provenance = self.engine.rows, self.engine.provenance
        sparse = [r for r in rows if r['sector'] != 'Real Estate']
        sparse += [next(r for r in rows if r['sector'] == 'Real Estate')]
        with patch('method_3.comparison.load_corpus', return_value=(sparse, provenance)):
            engine = ComparisonEngine()
        self.assertEqual(engine.model_info['Real Estate']['fit_policy'], 'pooled_singleton')
        self.assertEqual(len(engine.generate(20, 42, 'Information Technology')['methods'][2]['rows']), 20)

    def test_native_detail_adapter_uses_current_filing_period(self):
        row = next(r for r in self.engine.rows if r['ticker'] == 'CRWD')
        template, facts = self.engine.statement_inputs(row)
        self.assertEqual(template['fiscal_year'], int(row['period_end'][:4]))
        cash = self.engine.m2st['template_tag_value'](facts, template['fiscal_year'],
                    ['CashAndCashEquivalentsAtCarryingValue'])
        self.assertEqual(cash, row['metrics']['cash'])

    def test_removed_small_optional_driver_is_counted_as_adjustment(self):
        from method_3.comparison import driver_adjustments
        changes = driver_adjustments({'log10_revenue': 10, 'rd_ratio': 0.004}, {}, {'rd_ratio': 0})
        self.assertEqual(changes['rd_ratio']['delta'], -0.004)

    def test_rejected_profiles_do_not_exhaust_fictional_name_reservations(self):
        original = self.engine.m1['sample_spec']
        checks = self.engine.accounting['check_statements']
        sizes = []
        def record(*args, **kwargs):
            sizes.append(len(kwargs['taken']))
            return original(*args, **kwargs)
        calls = [0]
        def reject_first(statements):
            calls[0] += 1
            return ['forced rejection'] if calls[0] == 1 else checks(statements)
        with patch.dict(self.engine.m1, {'sample_spec': record}), patch.dict(self.engine.accounting, {'check_statements': reject_first}):
            self.engine._generate_m1(['Utilities'], 42)
        self.assertEqual(sizes[0], sizes[1])


if __name__ == '__main__':
    unittest.main()
