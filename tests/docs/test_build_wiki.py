# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Tests of docs/wiki/build_wiki.py on a small site built in a scratch directory.

    python3 tests/docs/test_build_wiki.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in docs/wiki
SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "wiki" / "build_wiki.py"
spec = importlib.util.spec_from_file_location("build_wiki", SCRIPT)
bw = importlib.util.module_from_spec(spec)
sys.modules["build_wiki"] = bw  # dataclasses look their module up there
spec.loader.exec_module(bw)

INDEX = """\
# Site

Start with [the guide](guide/a.md).

```{toctree}
:maxdepth: 2
:caption: Guide

guide/a
guide/b
```
"""

A = """\
# Page A

See [B's details](b.md#details), [this page's end](#the-end), [the code](../../src/lib.cpp),
[the sources](../../src/) and [a site](https://example.org/x.md).

```{note}
A note with [a link](b.md).
```

```{warning}
Mind $x_1 + x_2$.
```

```cpp
// a code block is left alone: [x](nowhere.md) {ref}`y`
```

Inline code keeps `[x](nowhere.md)` too.

## The end
"""

B = """\
# Page `B`: the second

```{include} ../../NOTES.md
:start-line: 1
```

## Details

## Details
"""

NOTES = """\
# Notes title, skipped by start-line
Notes link to [the library](src/lib.cpp).
"""


class Site:
    def __init__(self, tmp: Path, overrides: dict | None = None):
        files = {"docs/index.md": INDEX, "docs/guide/a.md": A, "docs/guide/b.md": B, "NOTES.md": NOTES,
                 "src/lib.cpp": "int x;\n"}
        files.update(overrides or {})
        for name, text in files.items():
            path = tmp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        bw.ROOT, bw.DOCS = tmp.resolve(), (tmp / "docs").resolve()

    @staticmethod
    def build():
        return bw.build("https://github.com/o/r", "main", "https://x.readthedocs.io/en/latest/")


class BuildWikiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.saved = (bw.ROOT, bw.DOCS)

    def tearDown(self):
        bw.ROOT, bw.DOCS = self.saved
        self._tmp.cleanup()

    def converted(self, overrides=None):
        Site(self.tmp, overrides)
        return Site.build()

    def test_slugs_match_myst(self):
        self.assertEqual(bw.slugify("An application outside SOMA's tree"), "an-application-outside-somas-tree")
        self.assertEqual(bw.slugify("`refine_uniform` and [links](x.md)"), "refine_uniform-and-links")
        self.assertEqual(bw.slugify("MOAB .h5m files"), "moab-h5m-files")
        self.assertEqual(bw.wiki_name("MOAB .h5m files"), "MOAB-h5m-files")
        self.assertEqual(bw.wiki_name("Page `B`: the second"), "Page-B-the-second")

    def test_pages_links_and_directives(self):
        files, errors = self.converted()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(files), ["Home.md", "Page-A.md", "Page-B-the-second.md", "_Sidebar.md"])
        home, a, b = files["Home.md"], files["Page-A.md"], files["Page-B-the-second.md"]
        self.assertTrue(a.startswith(bw.MARKER + "docs/guide/a.md;"))
        self.assertNotIn("# Page A", a)  # the wiki shows the page name as the title
        self.assertIn("[the guide](Page-A)", home)
        self.assertIn("## Guide\n\n- [Page A](Page-A)\n- [Page `B`: the second](Page-B-the-second)", home)
        self.assertIn("[B's details](Page-B-the-second#details)", a)
        self.assertIn("[this page's end](#the-end)", a)
        self.assertIn("[the code](https://github.com/o/r/blob/main/src/lib.cpp)", a)
        self.assertIn("[the sources](https://github.com/o/r/tree/main/src)", a)
        self.assertIn("[a site](https://example.org/x.md)", a)
        self.assertIn("> **Note:** A note with [a link](Page-B-the-second).", a)
        self.assertIn("> **Warning:** Mind $x_1 + x_2$.", a)
        self.assertIn("// a code block is left alone: [x](nowhere.md) {ref}`y`", a)
        self.assertIn("Inline code keeps `[x](nowhere.md)` too.", a)
        self.assertNotIn("Notes title", b)
        self.assertIn("Notes link to [the library](https://github.com/o/r/blob/main/src/lib.cpp).", b)
        self.assertIn("**Guide**\n\n- [Page A](Page-A)", files["_Sidebar.md"])

    def test_duplicate_headings_get_numbered_anchors(self):
        _, errors = self.converted({"docs/guide/a.md": A + "\n[second](b.md#details-1) [third](b.md#details-2)\n"})
        self.assertEqual(len(errors), 1)
        self.assertIn("no heading #details-2", errors[0])

    def test_errors(self):
        cases = {
            "[x](b.md#nowhere)": "no heading #nowhere",
            "[x](missing.md)": "missing.md does not exist",
            "[x](../orphan.md)": "is not in any toctree",
            "```{eval-rst}\n.. note:: x\n```": "{eval-rst} is not supported",
            "See {ref}`target`.": "{role}",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                _, errors = self.converted({"docs/guide/a.md": A + "\n" + text + "\n", "docs/orphan.md": "# Orphan\n"})
                self.assertEqual(len(errors), 1, errors)
                self.assertIn(message, errors[0])

    def test_two_pages_with_one_title_are_an_error(self):
        _, errors = self.converted({"docs/guide/b.md": "# Page A\n"})
        self.assertTrue(any("would both be the wiki page Page-A" in e for e in errors), errors)

    def test_write_replaces_only_generated_pages(self):
        files, _ = self.converted()
        out = self.tmp / "wiki"
        out.mkdir()
        (out / "Old-page.md").write_text(bw.MARKER + "docs/old.md; edit that file, not this page. -->\n")
        (out / "Hand-written.md").write_text("Written in the wiki.\n")
        written, removed = bw.write(files, out)
        self.assertEqual((written, removed), (4, 1))
        self.assertFalse((out / "Old-page.md").exists())
        self.assertEqual((out / "Hand-written.md").read_text(), "Written in the wiki.\n")
        self.assertEqual((out / "Page-A.md").read_text(), files["Page-A.md"])

    def test_figure_becomes_raw_image(self):
        figure = "```{figure} ../figures/img/x.svg\n:alt: An x\n:width: 60%\n\nThe x, `drawn`.\n```\n"
        files, errors = self.converted({"docs/guide/a.md": A + "\n" + figure, "docs/figures/img/x.svg": "<svg/>"})
        self.assertEqual(errors, [])
        self.assertIn("![An x](https://raw.githubusercontent.com/o/r/main/docs/figures/img/x.svg)\n\n*The x, `drawn`.*",
                      files["Page-A.md"])

    def test_figure_options_in_any_order_and_no_caption(self):
        figure = "```{figure} ../figures/img/x.png\n:width: 50%\n:alt: An x\n```\n"
        files, errors = self.converted({"docs/guide/a.md": A + "\n" + figure, "docs/figures/img/x.png": "png"})
        self.assertEqual(errors, [])
        page = files["Page-A.md"]
        self.assertIn("![An x](https://raw.githubusercontent.com/o/r/main/docs/figures/img/x.png)", page)
        self.assertNotIn("*", page.split("x.png)")[1].split("\n\n")[0])

    def test_missing_figure_is_an_error(self):
        figure = "```{figure} ../figures/img/none.svg\n:alt: None\n\nNothing.\n```\n"
        _, errors = self.converted({"docs/guide/a.md": A + "\n" + figure})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("figures/img/none.svg does not exist", errors[0])

    def test_mermaid_passes_through(self):
        diagram = "```mermaid\nflowchart LR\n  a --> b\n```\n"
        files, errors = self.converted({"docs/guide/a.md": A + "\n" + diagram})
        self.assertEqual(errors, [])
        self.assertIn(diagram, files["Page-A.md"])

    def test_link_with_a_code_label(self):
        a = A + "\nA [`B` page](b.md#details), a [`lib`](../../src/lib.cpp), and `[x](nowhere.md)` kept.\n"
        files, errors = self.converted({"docs/guide/a.md": a})
        self.assertEqual(errors, [])
        page = files["Page-A.md"]
        self.assertIn("A [`B` page](Page-B-the-second#details), a [`lib`](https://github.com/o/r/blob/main/src/lib.cpp)",
                      page)
        self.assertIn("`[x](nowhere.md)` kept.", page)

    def test_api_reference_links_to_read_the_docs(self):
        index = INDEX.replace("guide/b\n```", "guide/b\nreference/api/index\n```")
        api_index = "# API reference\n\n```{toctree}\n:maxdepth: 1\n\ncore/mesh\n```\n"
        api_page = "# `soma/core/mesh.hpp`\n\n```{doxygenfile} soma/core/mesh.hpp\n```\n"
        a = A + ("\nSee [the mesh](../reference/api/core/mesh.md#details) and "
                 "[the reference](../reference/api/index.md).\n")
        files, errors = self.converted({"docs/index.md": index, "docs/guide/a.md": a,
                                        "docs/reference/api/index.md": api_index,
                                        "docs/reference/api/core/mesh.md": api_page})
        self.assertEqual(errors, [])
        rtd = "https://x.readthedocs.io/en/latest"
        self.assertEqual(sorted(files), ["Home.md", "Page-A.md", "Page-B-the-second.md", "_Sidebar.md"])
        self.assertIn(f"- [API reference]({rtd}/reference/api/index.html)", files["Home.md"])
        self.assertIn(f"- [API reference]({rtd}/reference/api/index.html)", files["_Sidebar.md"])
        self.assertIn(f"[the mesh]({rtd}/reference/api/core/mesh.html#details)", files["Page-A.md"])
        self.assertIn(f"[the reference]({rtd}/reference/api/index.html)", files["Page-A.md"])

    def test_main_check_and_failure(self):
        Site(self.tmp)
        self.assertEqual(bw.main(["--check"]), 0)
        (self.tmp / "docs/guide/a.md").write_text(A + "\n[x](b.md#nowhere)\n")
        out = self.tmp / "wiki"
        self.assertEqual(bw.main(["--out", str(out)]), 1)
        self.assertFalse(out.exists())  # nothing written on an error


if __name__ == "__main__":
    unittest.main()
