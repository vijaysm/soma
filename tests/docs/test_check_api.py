# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Tests of docs/reference/check_api.py on small trees built in a scratch directory.

    python3 tests/docs/test_check_api.py
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in docs/reference
SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "reference" / "check_api.py"
spec = importlib.util.spec_from_file_location("check_api", SCRIPT)
ca = importlib.util.module_from_spec(spec)
sys.modules["check_api"] = ca
spec.loader.exec_module(ca)

HEADER = """\
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef A_HPP_
#define A_HPP_

/// \\file
/// What it is.

/// A function.
int f();

#endif  // A_HPP_
"""

HEADERS_MD = """\
# Public headers

| Header | Provides |
| --- | --- |
| `soma/core/a.hpp` | the a |
| `soma/core/hidden.hpp` | **internal**: hidden things |
"""


def page(name: str) -> str:
    return f"# soma/{name}\n\n```{{doxygenfile}} soma/{name}\n```\n"


class CheckApi(unittest.TestCase):
    def tree(self, d: str) -> Path:
        root = Path(d)
        for rel, text in {
            "include/soma/core/a.hpp": HEADER,
            "include/soma/core/hidden.hpp": HEADER,
            "include/soma/compat/dropin/moab/Core.hpp": "// a shim\nint g();\n",
            "docs/reference/headers.md": HEADERS_MD,
            "docs/reference/api/core/a.md": page("core/a.hpp"),
        }.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        return root

    def test_a_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ca.problems(self.tree(d)), [])

    def test_public_header_without_page(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.tree(d)
            (root / "include/soma/core/b.hpp").write_text(HEADER)
            self.assertEqual(ca.problems(root), ["soma/core/b.hpp: no page docs/reference/api/core/b.md"])

    def test_page_for_no_public_header(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.tree(d)
            (root / "docs/reference/api/core/hidden.md").write_text(page("core/hidden.hpp"))
            (root / "docs/reference/api/core/gone.md").write_text(page("core/gone.hpp"))
            self.assertEqual(
                ca.problems(root),
                [
                    "docs/reference/api/core/gone.md: no public header soma/core/gone.hpp",
                    "docs/reference/api/core/hidden.md: no public header soma/core/hidden.hpp",
                ],
            )

    def test_page_naming_another_header(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.tree(d)
            (root / "docs/reference/api/core/a.md").write_text(page("core/other.hpp"))
            self.assertEqual(
                ca.problems(root), ["docs/reference/api/core/a.md: no doxygenfile directive for soma/core/a.hpp"]
            )

    def test_unconverted_comment(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.tree(d)
            p = root / "include/soma/core/a.hpp"
            p.write_text(p.read_text().replace("/// A function.", "// A function."))
            self.assertEqual(ca.problems(root), ["include/soma/core/a.hpp:10: a comment not yet in Doxygen form"])


if __name__ == "__main__":
    unittest.main()
