#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Turn the plain comments of C++ headers into Doxygen comments, in place.

    python3 tools/doxify_headers.py [--check] PATHS...

PATHS are headers or directories (searched for *.hpp, less the MOAB and MPI
stand-ins under compat/dropin and compat/serial_mpi). The rules:

- the first comment block after the include guard's #define, before any code
  and followed by a blank or preprocessor line, becomes the file's
  documentation: "/// \\file", then its text as "///" lines;
- a "//" block directly above a declaration in namespace or class scope
  becomes "///" lines; one followed by a blank line, a preprocessor line, a
  closing brace, an access specifier, a namespace or a forward declaration
  stays "//" (a note, not documentation);
- a trailing comment after a member or enumerator ("int x;  // ...") becomes
  "///<", and so does a comment-only line under it indented further;
- comments inside function bodies, the SPDX and copyright lines, and "///"
  comments stay as they are;
- in a converted block, a run of lines indented two spaces or more past the
  comment marker (example code, tables) is fenced with ``` so that Doxygen
  keeps it verbatim.

The result does not depend on what was converted before: running the script
again changes nothing. --check changes nothing; it prints path:line for each
block the script would convert and exits 1 if there is any.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCOPE = re.compile(r'\b(namespace|class|struct|union|enum)\b|extern\s+"C"')
ACCESS = re.compile(r"^(public|protected|private)\s*:")
# A namespace (documented nowhere per header: Doxygen would merge the texts)
# or a forward declaration (the class is documented where it is defined).
NOT_DOCUMENTED = re.compile(r"^(inline\s+)?namespace\b|^(class|struct|union)\s+\w+\s*;")


def split_comment(line: str) -> tuple[str, str]:
    """The line's code and its trailing // comment ("" if none), quotes respected."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif line.startswith("//", i):
            return line[:i], line[i:]
        i += 1
    return line, ""


def is_plain(stripped: str) -> bool:
    """A "//" comment line that is neither Doxygen's nor the licence header's."""
    return (
        stripped.startswith("//")
        and not stripped.startswith("///")
        and not stripped.startswith(("// SPDX-License-Identifier", "// Copyright"))
    )


def fence(texts: list[str]) -> list[str]:
    """Texts with each run of code lines (two spaces or more in) inside ``` lines."""
    code = [t.startswith("  ") and t.strip() != "" for t in texts]
    out: list[str] = []
    i = 0
    while i < len(texts):
        if not code[i]:
            out.append(texts[i])
            i += 1
            continue
        last = i
        j = i
        while j < len(texts) and (code[j] or texts[j].strip() == ""):
            if code[j]:
                last = j
            j += 1
        out.append("```")
        out.extend(texts[i : last + 1])
        out.append("```")
        i = last + 1
    return out


def convert(text: str) -> tuple[str, list[int]]:
    """The converted text and the 1-based first lines of the blocks converted."""
    lines = text.split("\n")
    out: list[str] = []
    converted: list[int] = []
    stack: list[str] = []  # "scope" or "body" per open brace
    statement = ""  # code since the last ; { or }
    guard = file_done = False
    trailing_indent: int | None = None  # indent of the last line with a ///< comment
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        in_body = "body" in stack
        if is_plain(stripped):
            j = i
            while j < len(lines) and is_plain(lines[j].strip()):
                j += 1
            block = lines[i:j]
            following = lines[j].strip() if j < len(lines) else ""
            indent = len(line) - len(line.lstrip())
            if not in_body and trailing_indent is not None and indent > trailing_indent:
                out.extend(b.replace("//", "///<", 1) for b in block)
                converted.append(i + 1)
                i = j
                continue
            trailing_indent = None
            file_comment = guard and not file_done and (following == "" or following.startswith("#"))
            if guard:
                file_done = True
            note = (
                following == ""
                or following.startswith(("#", "}"))
                or ACCESS.match(following) is not None
                or NOT_DOCUMENTED.match(following) is not None
            )
            if in_body or (note and not file_comment):
                out.extend(block)
                i = j
                continue
            texts = fence([re.sub(r"^\s*// ?", "", b) for b in block])
            pad = " " * indent
            if file_comment:
                out.append(pad + "/// \\file")
            out.extend(pad + "/// " + t if t else pad + "///" for t in texts)
            converted.append(i + 1)
            i = j
            continue
        if stripped.startswith("#define") and not guard:
            guard = True
        if stripped and not stripped.startswith("#") and guard and not file_done:
            file_done = True  # code before any comment: the header has no file comment
        code, comment = split_comment(line)
        if comment and is_plain(comment) and code.strip() and not in_body and re.search(r"[;,]\s*$", code):
            out.append(code + comment.replace("//", "///<", 1))
            converted.append(i + 1)
            trailing_indent = len(line) - len(line.lstrip())
        else:
            out.append(line)
            if comment.startswith("///<") and code.strip():
                trailing_indent = len(line) - len(line.lstrip())
            elif stripped and not (stripped.startswith("///<") and trailing_indent is not None):
                trailing_indent = None
        for c in code:
            if c == "{":
                stack.append("scope" if SCOPE.search(statement) and "body" not in stack else "body")
                statement = ""
            elif c == "}":
                if stack:
                    stack.pop()
                statement = ""
            elif c == ";":
                statement = ""
            else:
                statement += c
        statement += " "
        i += 1
    return "\n".join(out), converted


def doxify(text: str) -> str:
    """The header text with its comments turned into Doxygen comments."""
    return convert(text)[0]


def unconverted(text: str) -> list[int]:
    """1-based lines where doxify would convert a block or a trailing comment."""
    return convert(text)[1]


# Headers that stand in for MOAB's and MPI's at their include paths: they keep
# those libraries' comment style and are not part of SOMA's API reference.
SHIMS = ("compat/dropin", "compat/serial_mpi")


def headers(paths: list[str]) -> list[Path]:
    """The headers named, and those under the directories named, less the shims."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(h for h in sorted(p.rglob("*.hpp")) if not any(s in h.as_posix() for s in SHIMS))
        else:
            found.append(p)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="change nothing; list what would be converted")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    left = 0
    for path in headers(args.paths):
        text = path.read_text()
        if args.check:
            for line in unconverted(text):
                print(f"{path}:{line}: a comment not yet in Doxygen form")
                left += 1
        else:
            new = doxify(text)
            if new != text:
                path.write_text(new)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
