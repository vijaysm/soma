# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Tests of tools/doxify_headers.py on small headers.

    python3 tests/docs/test_doxify.py
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in tools/
SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "doxify_headers.py"
spec = importlib.util.spec_from_file_location("doxify_headers", SCRIPT)
dx = importlib.util.module_from_spec(spec)
sys.modules["doxify_headers"] = dx
spec.loader.exec_module(dx)

HEAD = """\
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef X_HPP_
#define X_HPP_
"""


def header(body: str) -> str:
    return HEAD + body + "\n#endif  // X_HPP_\n"


class Doxify(unittest.TestCase):
    def check(self, before: str, after: str) -> None:
        self.assertEqual(dx.doxify(header(before)), header(after))

    def test_file_comment_becomes_file_doc(self):
        self.check(
            "\n// What the header does,\n// in two lines.\n//\n// More.\n\n#include <x>\n",
            "\n/// \\file\n/// What the header does,\n/// in two lines.\n///\n/// More.\n\n#include <x>\n",
        )

    def test_block_above_declaration(self):
        self.check(
            "\nnamespace a {\n\n// A class.\nclass C {\n public:\n  // A member.\n  int f() const;\n};\n\n"
            "// A function\n// of two lines.\nint g(int x);\n\n// An alias.\nusing T = int;\n\n"
            "// A template.\ntemplate <class M>\nstruct S;\n\n}  // namespace a\n",
            "\nnamespace a {\n\n/// A class.\nclass C {\n public:\n  /// A member.\n  int f() const;\n};\n\n"
            "/// A function\n/// of two lines.\nint g(int x);\n\n/// An alias.\nusing T = int;\n\n"
            "/// A template.\ntemplate <class M>\nstruct S;\n\n}  // namespace a\n",
        )

    def test_trailing_member_comment(self):
        self.check(
            "\nstruct P {\n  int a = 0;     // the a\n  double bb;     // the bb, which\n"
            "                 // runs on\n};\nenum class E { x,  // first\n  y };\n",
            "\nstruct P {\n  int a = 0;     ///< the a\n  double bb;     ///< the bb, which\n"
            "                 ///< runs on\n};\nenum class E { x,  ///< first\n  y };\n",
        )

    def test_blank_separated_block_stays(self):
        src = "\nnamespace a {\n\n// A note on what follows.\n\nint g();\n\n}  // namespace a\n"
        self.check(src, src)

    def test_block_before_preprocessor_or_close_stays(self):
        src = "\nnamespace a {\n\nint g();\n// Before a close.\n}  // namespace a\n// Before a line.\n#include <y>\n"
        self.check(src, src)

    def test_block_before_namespace_or_forward_declaration_stays(self):
        src = (
            "\nint a();\n\n// -----\n// Section\n// -----\nnamespace detail {\n}  // namespace detail\n\n"
            "// About what comes after.\nclass Later;\nstruct Other;\n"
        )
        self.check(src, src)

    def test_body_comments_stay(self):
        src = (
            "\ninline int f(int x) {\n  // Inside.\n  int y = x;  // trailing\n  return y;\n}\n\n"
            "struct Q {\n  int g() const {\n    // In a member body.\n    return 1;\n  }\n};\n"
        )
        self.check(src, src)

    def test_indented_code_fenced(self):
        self.check(
            "\n// Use it so:\n//\n//   f(1);\n//   g(2);\n//\n// and done.\nint f(int);\n",
            "\n/// Use it so:\n///\n/// ```\n///   f(1);\n///   g(2);\n/// ```\n///\n/// and done.\nint f(int);\n",
        )

    def test_code_run_with_blank_inside(self):
        self.check(
            "\n// A table:\n//   a   1\n//\n//   b   2\nint f(int);\n",
            "\n/// A table:\n/// ```\n///   a   1\n///\n///   b   2\n/// ```\nint f(int);\n",
        )

    def test_spdx_and_guard_stay(self):
        src = "\n#include <x>\n"
        self.assertTrue(dx.doxify(header(src)).startswith(HEAD))

    def test_strings_are_not_comments(self):
        src = '\nstruct R {\n  const char* p = "http://x";\n};\n'
        self.check(src, src)

    def test_idempotent(self):
        texts = [
            header("\n// File.\n\n// A.\nint a;\nstruct S {\n  int b;  // B\n};\n// Note.\n\n/// Already.\nint c;\n"),
            header("\n// Use:\n//   f(1);\nint f(int);\n"),
        ]
        for t in texts:
            once = dx.doxify(t)
            self.assertEqual(dx.doxify(once), once)

    def test_unconverted_lists_lines(self):
        t = header("\n// File.\n\nnamespace a {\n// A.\nint a;\n// Note.\n\n}  // namespace a\n")
        self.assertEqual(dx.unconverted(t), [7, 10])
        self.assertEqual(dx.unconverted(dx.doxify(t)), [])

    def test_directory_walk_skips_the_shims(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "soma"
            for rel in ("core/a.hpp", "compat/core.hpp", "compat/dropin/moab/Core.hpp", "compat/serial_mpi/mpi.hpp"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("")
            found = sorted(p.relative_to(root).as_posix() for p in dx.headers([str(root)]))
            self.assertEqual(found, ["compat/core.hpp", "core/a.hpp"])

    def test_check_mode_on_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.hpp"
            p.write_text(header("\n// File.\n\n// A.\nint a;\n"))
            self.assertEqual(dx.main(["--check", str(p)]), 1)
            self.assertEqual(dx.main([str(p)]), 0)
            self.assertEqual(dx.main(["--check", str(p)]), 0)
            once = p.read_text()
            dx.main([str(p)])
            self.assertEqual(p.read_text(), once)


if __name__ == "__main__":
    unittest.main()
