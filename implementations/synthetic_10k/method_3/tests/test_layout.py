"""Ensure all new assets live under implementations/synthetic_10k/method_3."""
import unittest
from pathlib import Path
from method_3.native import PACKAGE_DIR, ROOT
from method_3.server import DASHBOARD


class LayoutTests(unittest.TestCase):
    def test_package_is_a_direct_child_of_synthetic_10k(self):
        self.assertEqual(PACKAGE_DIR.name, 'method_3')
        self.assertEqual(PACKAGE_DIR.parent.name, 'synthetic_10k')
        self.assertEqual(ROOT, PACKAGE_DIR.parent)

    def test_dashboard_is_inside_the_new_package(self):
        self.assertEqual(DASHBOARD, PACKAGE_DIR / 'dashboard')
        self.assertTrue((DASHBOARD / 'index.html').is_file())
        self.assertTrue((DASHBOARD / 'vendor/d3.v7.9.0.min.js').is_file())
        self.assertEqual(ROOT / 'method_3', PACKAGE_DIR)
        self.assertFalse((ROOT.parent / 'method_3').exists())
        self.assertFalse((ROOT / 'comparison_dashboard').exists())


if __name__ == '__main__':
    unittest.main()
