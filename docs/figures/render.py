# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Draws the documentation figures from the data soma_figure_data wrote.

    pvpython docs/figures/render.py <data dir> <img dir> all | <figure>...
    pvpython docs/figures/render.py --list

Each figure is a function registered with @figure(name, sets, kind) in a
figure module (figures_<area>.py, sharing figlib.py): it reads its data sets'
files from <data dir>/<set>/ and writes <img dir>/<name>.<kind>, a PNG
rendered by ParaView or an SVG drawn by matplotlib (see style.py). Only
the figures named are drawn, so an unrelated change never rewrites an image.
Run it through docs/figures/make_figures.sh, which makes the data first; see
docs/developer/making-figures.md.

Exits 0 on success, 1 when a figure's data is missing, 2 for an unknown figure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import figures_algorithms  # noqa: E402,F401  (each module registers its figures)
import figures_concepts  # noqa: E402,F401
import figures_parallel  # noqa: E402,F401
from figlib import FIGURES  # noqa: E402


def main(argv):
    if argv == ["--list"]:
        for name, f in sorted(FIGURES.items()):
            print(f"{name} {f['kind']} {','.join(f['sets'])}")
        return 0
    if len(argv) < 3:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    data, img, names = Path(argv[0]), Path(argv[1]), argv[2:]
    names = sorted(FIGURES) if names == ["all"] else names
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        print(f"render: unknown figure(s) {' '.join(unknown)}; known: {' '.join(sorted(FIGURES))}", file=sys.stderr)
        return 2
    img.mkdir(parents=True, exist_ok=True)
    for name in names:
        f = FIGURES[name]
        missing = [s for s in f["sets"] if not (data / s).is_dir()]
        if missing:
            print(f"render: {name} needs data set(s) {' '.join(missing)} in {data}", file=sys.stderr)
            return 1
        path = img / f"{name}.{f['kind']}"
        f["draw"](data, path)
        print(f"render: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
