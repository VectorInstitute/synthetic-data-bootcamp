"""Standalone packaging must embed its assets and avoid server dependencies."""
import unittest
from method_3.build_standalone import render_html


class StandaloneTests(unittest.TestCase):
    def test_single_html_has_no_external_assets_or_api_calls(self):
        page = render_html({'schema_version': 1, 'pools': {}, 'shared': {}, 'marker': '</script><script>unsafe</script>'})
        self.assertNotIn('<script defer src=', page)
        self.assertNotIn('<link rel="stylesheet"', page)
        self.assertNotIn("fetch('/api/", page)
        self.assertIn('id="embedded-financial-data"', page)
        self.assertIn('Sampling seed', page)
        self.assertIn('Resample', page)
        self.assertIn('fixed embedded pools', page)
        self.assertNotIn('</script><script>unsafe</script>', page)
        self.assertIn('\\u003c/script>', page)


if __name__ == '__main__':
    unittest.main()
