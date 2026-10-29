"""HTTP input and static surface contract."""
import unittest
from method_3.server import parse_controls, static_file


class ServerTests(unittest.TestCase):
    def test_control_defaults_and_invalid_inputs(self):
        self.assertEqual(parse_controls({}), (200, 42, 'all'))
        for value in ('20.5', True, None, 2001):
            with self.assertRaises(ValueError):
                parse_controls({'n': value})

    def test_static_paths_cannot_escape_dashboard(self):
        self.assertEqual(static_file('/').name, 'index.html')
        self.assertEqual(static_file('/evaluation.js').name, 'evaluation.js')
        for path in ('/../method_2/config.yaml', '/method_3/server.py', '/.env'):
            self.assertIsNone(static_file(path))

    def test_dashboard_has_expected_controls_and_chart_slots(self):
        page = static_file('/').read_text(encoding='utf-8')
        for item in ('id="sampleCount"', 'id="sector"', 'id="synthetic"', 'id="real"',
                     'id="metric"', 'id="downloadCsv"'):
            self.assertIn(item, page)
