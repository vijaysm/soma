# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Tests of docs/figures/check_figures.py on a small site built in a scratch directory.

    python3 tests/docs/test_check_figures.py
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in docs/figures
SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "check_figures.py"
spec = importlib.util.spec_from_file_location("check_figures", SCRIPT)
cf = importlib.util.module_from_spec(spec)
sys.modules["check_figures"] = cf
spec.loader.exec_module(cf)

PAGE = """\
# Page

```{figure} ../figures/img/a.svg
:alt: The a
:width: 60%

The caption of a.
```

```{figure} ../figures/img/b.png
:width: 50%
:alt: The b

The caption of b.
```
"""


class CheckFiguresTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name) / "docs"
        self.write("guide/page.md", PAGE)
        self.write("figures/img/a.svg", "<svg/>")
        self.write("figures/img/b.png", "png")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_a_good_site_passes(self):
        self.assertEqual(cf.check(self.docs), [])

    def test_missing_image(self):
        (self.docs / "figures/img/b.png").unlink()
        errors = cf.check(self.docs)
        self.assertEqual(len(errors), 1, errors)
        self.assertEqual("guide/page.md:10: figures/img/b.png does not exist", errors[0])

    def test_figure_without_alt(self):
        self.write("guide/page.md", PAGE.replace(":alt: The a\n", ""))
        errors = cf.check(self.docs)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/a.svg has no :alt:", errors[0])

    def test_figure_without_caption(self):
        self.write("guide/page.md", PAGE.replace("The caption of b.\n", ""))
        errors = cf.check(self.docs)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/b.png has no caption", errors[0])

    def test_orphan_image(self):
        self.write("figures/img/c.svg", "<svg/>")
        errors = cf.check(self.docs)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/c.svg is not shown by any page", errors[0])

    def test_image_too_large(self):
        self.write("figures/img/a.svg", "x" * 101)
        errors = cf.check(self.docs, max_file=100, max_total=10_000)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/a.svg is 101 bytes, over 100", errors[0])

    def test_images_too_large_together(self):
        errors = cf.check(self.docs, max_file=100, max_total=8)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/ holds 9 bytes, over 8", errors[0])

    def test_pages_under_plans_and_build_are_not_pages(self):
        self.write("plans/x.md", PAGE.replace("a.svg", "none.svg"))
        self.write("_build/x.md", PAGE.replace("a.svg", "none.svg"))
        self.assertEqual(cf.check(self.docs), [])

    def test_figures_shown_as_code_are_not_figures(self):
        example = "````text\n```{figure} ../figures/img/<name>.png\n:alt: x\n\nx\n```\n````\n"
        self.write("guide/other.md", "# Other\n\n" + example)
        self.assertEqual(cf.check(self.docs), [])

    def test_main(self):
        self.assertEqual(cf.main(["--docs", str(self.docs)]), 0)
        self.write("figures/img/c.svg", "<svg/>")
        self.assertEqual(cf.main(["--docs", str(self.docs)]), 1)


if __name__ == "__main__":
    unittest.main()
