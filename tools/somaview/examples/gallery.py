#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Render a gallery of the ground-truth SOMA files to PNGs, offscreen.

    python examples/gallery.py [--data tests/data] --out gallery_out

Each image is independent: a failure in one (e.g. no GPU / display server available for the
pyvista-based renders -- see the package README) is reported and does not stop the others. The
yt slice does not need a GPU (yt's SlicePlot is matplotlib/Agg); the rest are pyvista (VTK/
OpenGL) and need a real or software OpenGL context.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _hex_fields(data: Path, out: Path) -> None:
    import somaview as sv

    sv.plot(str(data / "hex_fields.soma"), scalars="pressure", clip="x",
           screenshot=str(out / "hex_fields_clipped_pressure.png"), window_size=(1000, 800))


def _ne20(data: Path, out: Path, ne20: str | None = None) -> None:
    import somaview as sv

    path = Path(ne20) if ne20 else data / "ne20_2km_centroids.soma"
    if not path.exists():
        raise FileNotFoundError(f"{path}: the real 6-part ne20 file was not found (pass --ne20)")
    sv.plot(str(path), scalars="spacing_km", screenshot=str(out / "ne20_spacing_km.png"),
           window_size=(1000, 800))


def _materials(data: Path, out: Path) -> None:
    import somaview as sv

    sv.plot(str(data / "64bricks_1khex.soma"), scalars="set:MATERIAL_SET",
           screenshot=str(out / "64bricks_material_set.png"), window_size=(1000, 800))


def _highorder3d(data: Path, out: Path) -> None:
    import somaview as sv

    sv.plot(str(data / "highorder3d.soma"), show_edges=True,
           screenshot=str(out / "highorder3d_curved.png"), window_size=(1000, 800))


def _mixed3d(data: Path, out: Path) -> None:
    import somaview as sv

    sv.plot(str(data / "mixed3d.soma"), show_edges=True, surface="never",
           screenshot=str(out / "mixed3d_edges.png"), window_size=(1000, 800))


def _amr2d(data: Path, out: Path) -> None:
    import somaview as sv

    sv.plot(str(data / "amr2d.soma"), scalars="phi", box_outlines=True,
           screenshot=str(out / "amr2d_phi_boxes.png"), window_size=(1000, 800))


def _amr3d_yt_slice(data: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import yt

    import somaview.yt as svyt

    ds = svyt.load_amr(str(data / "amr3d.soma"))
    p = yt.SlicePlot(ds, "z", ("stream", "u_0"))
    p.save(str(out / "amr3d_yt_slice_u.png"))


GALLERY = [
    ("ne20_spacing_km.png", "ne20 (6-part CONUS mesh), colored by spacing_km", _ne20),
    ("hex_fields_clipped_pressure.png", "hex_fields.soma, clipped and colored by pressure", _hex_fields),
    ("64bricks_material_set.png", "64bricks_1khex.soma, colored by set:MATERIAL_SET", _materials),
    ("highorder3d_curved.png", "highorder3d.soma, curved (quadratic) cells with edges", _highorder3d),
    ("mixed3d_edges.png", "mixed3d.soma (tet, pyramid, prism, hex, knife), with edges", _mixed3d),
    ("amr2d_phi_boxes.png", "amr2d.soma, colored by phi, with AMR box outlines", _amr2d),
    ("amr3d_yt_slice_u.png", "amr3d.soma, a yt slice colored by u (no GPU needed)", _amr3d_yt_slice),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "tests" / "data"),
                    help="directory of ground-truth .soma files (made by tests/make_data.sh)")
    ap.add_argument("--out", default="gallery_out", help="output directory for PNGs")
    ap.add_argument("--ne20", default=None, help="path to a real multi-part file for the ne20 render (default: "
                    "<data>/ne20_2km_centroids.soma)")
    ap.add_argument("--run-one", default=None, help=argparse.SUPPRESS)  # internal: one item, in-process
    args = ap.parse_args(argv)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_name = {filename: fn for filename, _, fn in GALLERY}
    if args.run_one is not None:
        if args.run_one == "ne20_spacing_km.png":
            _ne20(data, out, args.ne20)
        else:
            by_name[args.run_one](data, out)
        return 0

    # Each item renders in its own subprocess: a renderer crash (with no GPU or display, a
    # segfault, not a Python exception) then costs only that one image, not the rest of the
    # gallery.
    import subprocess

    ok, failed = [], []
    for filename, caption, _ in GALLERY:
        cmd = [sys.executable, __file__, "--data", str(data), "--out", str(out), "--run-one", filename]
        if args.ne20:
            cmd += ["--ne20", args.ne20]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and (out / filename).exists():
            ok.append((filename, caption))
            print(f"[ok]     {caption} -> {out / filename}")
        else:
            failed.append((filename, caption))
            print(f"[FAILED] {caption} (exit {r.returncode})", file=sys.stderr)
            if r.stderr:
                print(r.stderr.strip().splitlines()[-1], file=sys.stderr)
    print(f"\n{len(ok)} of {len(GALLERY)} rendered to {out}")
    if failed:
        print(f"{len(failed)} failed (see above) -- a pyvista failure here usually means no "
              "GPU/OpenGL context is available; see the package README.", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
