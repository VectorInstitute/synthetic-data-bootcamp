"""Behavioral tests for joint sampling and real-fact adapters."""
import unittest
import numpy as np
from method_3.copula import GaussianCopula
from method_3.sources import facts_to_companyfacts


class CopulaTests(unittest.TestCase):
    def test_joint_dependence_and_seed(self):
        x = np.linspace(1, 100, 200)
        model = GaussianCopula.fit(np.column_stack([x, x ** 2]), shrinkage=0.03)
        a = model.sample(1500, 42)
        np.testing.assert_array_equal(a, model.sample(1500, 42))
        self.assertGreater(np.corrcoef(a.T)[0, 1], 0.9)
        independent = a[:, 1].copy()
        np.random.default_rng(1).shuffle(independent)
        self.assertLess(abs(np.corrcoef(a[:, 0], independent)[0, 1]), 0.1)

    def test_missing_tied_constant_and_psd(self):
        matrix = np.array([[1, 3, 0], [1, np.nan, 0], [2, 5, 0], [3, 7, 0]])
        model = GaussianCopula.fit(matrix)
        self.assertGreaterEqual(np.linalg.eigvalsh(model.correlation).min(), -1e-9)
        sample = model.sample(40, 3)
        self.assertTrue(np.isfinite(sample).all())
        np.testing.assert_array_equal(sample[:, 2], np.zeros(40))
        self.assertEqual(model.imputed_count, 1)

    def test_unobserved_column_cannot_silently_become_zero(self):
        with self.assertRaises(ValueError):
            GaussianCopula.fit(np.array([[1, np.nan], [2, np.nan]]))


class FactTests(unittest.TestCase):
    def test_exact_period_matching_for_january_fiscal_year_end(self):
        facts = [dict(concept='us-gaap:Assets', value=str(v), unit='usd',
                      start='', end='', instant=end, dimensions='')
                 for v,end in [(100,'2025-01-31'),(90,'2024-01-31')]]
        out = facts_to_companyfacts(facts, '2025-03-01', period_end='2025-01-31')
        self.assertEqual([e['val'] for e in out['facts']['us-gaap']['Assets']['units']['USD']], [100])

    def test_fact_adapter_excludes_segments_quarters_and_non_usd(self):
        def fact(value, **kw):
            return dict(concept='us-gaap:Revenues', value=str(value), unit='usd',
                        start='2024-01-01', end='2024-12-31', instant='', dimensions='', **kw)
        good = fact(100)
        segment = {**good, 'value': '999', 'dimensions': '{"segment":"x"}'}
        quarter = {**good, 'value': '888', 'start': '2024-10-01'}
        eur = {**good, 'value': '777', 'unit': 'eur'}
        out = facts_to_companyfacts([good, segment, quarter, eur], '2025-02-01')
        entries = out['facts']['us-gaap']['Revenues']['units']['USD']
        self.assertEqual([e['val'] for e in entries], [100])


if __name__ == '__main__':
    unittest.main()
