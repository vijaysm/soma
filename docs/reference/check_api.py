#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Check that the API reference covers the public headers.

    python3 docs/reference/check_api.py [--root DIR]

Fails (exit 1, one line per problem) unless:

- every public header, soma/<path>.hpp, has a page docs/reference/api/<path>.md
  whose doxygenfile directive names it;
- every page there names a public header;
- no header comment is left out of Doxygen form (tools/doxify_headers.py --check).

Public headers are those under include/soma less the internal ones (marked
**internal** in docs/reference/headers.md) and the MOAB and MPI stand-ins under
compat/dropin and compat/serial_mpi. See docs/developer/writing-docs.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("doxify_headers", ROOT / "tools" / "doxify_headers.py")
doxify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doxify)

INTERNAL = re.compile(r"^\|\s*`(soma/[^`]+\.hpp)`\s*\|\s*\*\*internal\*\*", re.M)
DIRECTIVE = re.compile(r"^```\{doxygenfile\}\s+(\S+)\s*$", re.M)


def public_headers(root: Path) -> list[str]:
    """soma/<path>.hpp for each public header, sorted."""
    internal = set(INTERNAL.findall((root / "docs" / "reference" / "headers.md").read_text()))
    include = root / "include"
    return sorted(
        name
        for name in (h.relative_to(include).as_posix() for h in doxify.headers([str(include / "soma")]))
        if name not in internal
    )


def problems(root: Path) -> list[str]:
    found: list[str] = []
    api = root / "docs" / "reference" / "api"
    public = public_headers(root)
    for name in public:
        page = api / (name[len("soma/") : -len(".hpp")] + ".md")
        rel = page.relative_to(root).as_posix()
        if not page.exists():
            found.append(f"{name}: no page {rel}")
        elif name not in DIRECTIVE.findall(page.read_text()):
            found.append(f"{rel}: no doxygenfile directive for {name}")
    pages = api.rglob("*.md") if api.exists() else []
    named = {"soma/" + p.relative_to(api).with_suffix(".hpp").as_posix() for p in pages}
    for name in sorted(named - set(public) - {"soma/index.hpp"}):
        found.append(f"docs/reference/api/{name[len('soma/'):-len('.hpp')]}.md: no public header {name}")
    for header in doxify.headers([str(root / "include" / "soma")]):
        rel = header.relative_to(root).as_posix()
        lines = doxify.unconverted(header.read_text())
        found.extend(f"{rel}:{line}: a comment not yet in Doxygen form" for line in lines)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository (default: this one)")
    args = parser.parse_args(argv)
    found = problems(args.root)
    for p in found:
        print(p)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
