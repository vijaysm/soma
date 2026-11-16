#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Check the figures of the pages under docs/.

    python3 docs/figures/check_figures.py [--docs DIR]

Fails (exit 1, one line per problem) unless:

- every {figure} of a page names an image that exists, and has :alt: and a
  caption;
- every file in docs/figures/img/ is shown by some page;
- no image is over 300 KB, and the images together are under 6 MB.

Pages are the .md files under docs/, except those under plans/ and _build/.
The images are made by docs/figures/make_figures.sh; see
docs/developer/making-figures.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^\s*(`{3,}|:{3,})(.*)$")
FIGURE = re.compile(r"^\{figure\}\s+(\S+)\s*$")
OPTION = re.compile(r"^:([a-z-]+):\s*(.*)$")
NOT_PAGES = {"plans", "_build"}


def figures(page: Path):
    """(line, image, alt, caption) of each figure of a page; other fenced blocks are skipped whole."""
    lines = page.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        mark, start = m.group(1), i + 1
        end = start
        while end < len(lines) and lines[end].strip() != mark:
            end += 1
        i = end + 1
        figure = FIGURE.match(m.group(2).strip())
        if not figure:
            continue
        target = figure.group(1)
        options, body = {}, [b.strip() for b in lines[start:end]]
        while body and OPTION.match(body[0]):
            key, value = OPTION.match(body[0]).groups()
            options[key] = value
            body = body[1:]
        caption = " ".join(b.strip() for b in body if b.strip())
        yield start, (page.parent / target).resolve(), options.get("alt", "").strip(), caption


def check(docs: Path, max_file: int = 300_000, max_total: int = 6_000_000) -> list[str]:
    docs = docs.resolve()
    img = docs / "figures" / "img"
    errors: list[str] = []
    shown: set[Path] = set()
    for page in sorted(docs.rglob("*.md")):
        if NOT_PAGES & set(page.relative_to(docs).parts):
            continue
        where = page.relative_to(docs).as_posix()
        for line, image, alt, caption in figures(page):
            name = image.relative_to(docs).as_posix() if image.is_relative_to(docs) else str(image)
            if not image.is_file():
                errors.append(f"{where}:{line}: {name} does not exist")
                continue
            shown.add(image)
            if not alt:
                errors.append(f"{where}:{line}: {name} has no :alt:")
            if not caption:
                errors.append(f"{where}:{line}: {name} has no caption")
    total = 0
    for image in sorted(img.glob("*")) if img.is_dir() else []:
        if not image.is_file():
            continue
        name = image.relative_to(docs).as_posix()
        size = image.stat().st_size
        total += size
        if image.resolve() not in shown:
            errors.append(f"{name} is not shown by any page")
        if size > max_file:
            errors.append(f"{name} is {size} bytes, over {max_file}")
    if total > max_total:
        errors.append(f"figures/img/ holds {total} bytes, over {max_total}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs", type=Path, default=DOCS, help="the docs directory (default: this one)")
    args = parser.parse_args(argv)
    errors = check(args.docs)
    for e in errors:
        print(f"check_figures: {e}", file=sys.stderr)
    if errors:
        return 1
    print("check_figures: every figure is present, described and shown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
