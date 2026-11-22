# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Cross-checks soma_reader.hpp (via the compiled cpp_driver) against somaview's Python core.

For each ground-truth file (and every part/split-file variation this exercises), runs
cpp_driver as a subprocess, parses its binary dump, reads the same selection with somaview, and
asserts the two agree: point coordinates by gid, cell types and connectivity (as sorted corner
gid sets, orientation-independent), every field's values (NaN-aware), and set labels. Tallies
and prints the number of individual comparisons that passed.

Usage: python compare_with_python.py [--ne20] [--verbose]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # somaview/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))  # soma_split

import somaview as sv  # noqa: E402
import soma_split  # noqa: E402

DRIVER = HERE / "cpp_driver"
DATA = Path(os.environ.get("SOMAVIEW_DATA", ROOT / "tests" / "data"))
NE20 = os.environ.get("SOMAVIEW_NE20", "")

PASS = {"n": 0}
FAIL = []


def check(cond, msg):
    if cond:
        PASS["n"] += 1
    else:
        FAIL.append(msg)


def allclose_nan(a, b, msg, atol=1e-8, rtol=1e-6):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    check(a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}")
    if a.shape == b.shape:
        check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), f"{msg}: values differ")


def read_sections(path):
    """Parses the cpp_driver binary dump into {name: ndarray}."""
    out = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (nlen,) = struct.unpack("<I", hdr)
            if nlen == 0:
                break
            name = f.read(nlen).decode()
            dtype_code, ncomp = struct.unpack("<BI", f.read(5))
            (nrows,) = struct.unpack("<Q", f.read(8))
            dtype = np.float64 if dtype_code == 0 else np.int64
            n = nrows * ncomp
            buf = f.read(n * 8)
            arr = np.frombuffer(buf, dtype=dtype).reshape(nrows, ncomp) if nrows else np.empty((0, ncomp), dtype)
            out[name] = arr
    return out


def run_driver(path, parts=None, dims=None, cached=False, out=None):
    out = out or (HERE / "_dump.bin")
    parts_s = ",".join(str(p) for p in parts) if parts is not None else ""
    dims_s = ",".join(str(d) for d in dims) if dims is not None else ""
    args = [str(DRIVER), str(path), str(out), parts_s, dims_s]
    if cached:
        args.append("cached")
    from somaview.cells import wedge_legacy
    env = dict(os.environ)
    env["SOMAVIEW_VTK_WEDGE"] = "legacy" if wedge_legacy() else "current"
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cpp_driver failed on {path} parts={parts} dims={dims}: {r.stderr}")
    return read_sections(out)


def corner_sets_from_sections(sec, dim, key="cell_corner_gids"):
    off = sec[f"cell_corner_offsets:{dim}"].ravel()
    flat = sec[f"{key}:{dim}"].ravel()
    gids = sec[f"cell_gid:{dim}"].ravel()
    out = {}
    for c in range(len(gids)):
        out[int(gids[c])] = tuple(int(x) for x in flat[off[c]:off[c + 1]])
    return out


def corner_sets_from_piece(piece, dim):
    b = piece.blocks[dim]
    out = {}
    for c in range(len(b)):
        ids = b.connectivity[b.offsets[c]:b.offsets[c + 1]]
        out[int(b.gids[c])] = tuple(int(x) for x in piece.point_gids[ids])  # VTK order
    return out


def compare_piece(sec, piece, label, dims=None):
    """Cross-checks one cpp_driver dump against the somaview MeshPiece it should match."""
    g1 = sec["point_gid"].ravel()
    x1 = sec["point_xyz"]
    order1 = np.argsort(g1)
    g1s, x1s = g1[order1], x1[order1]
    g2 = piece.point_gids
    order2 = np.argsort(g2)
    g2s, x2s = g2[order2], piece.points[order2]
    check(np.array_equal(g1s, g2s), f"{label}: point gid sets differ")
    if np.array_equal(g1s, g2s):
        check(np.allclose(x1s, x2s), f"{label}: point coords differ")
    check(int(sec["halo_points"].ravel()[0]) == piece.halo_points, f"{label}: halo_points differ")

    for name, arr in piece.point_data.items():
        key = f"point_data:{name}"
        if key not in sec:
            FAIL.append(f"{label}: missing point field {name} in cpp dump")
            continue
        a = np.asarray(arr).reshape(len(g2), -1)
        _, a_sorted = g2s, a[order2]
        _, c_sorted = g1s, sec[key][order1]
        allclose_nan(a_sorted, c_sorted, f"{label}: point field {name}")

    dims = dims if dims is not None else list(piece.blocks)
    for d in dims:
        block = piece.blocks[d]
        cs_cpp = corner_sets_from_sections(sec, d)
        cs_py = corner_sets_from_piece(piece, d)
        check(set(cs_cpp) == set(cs_py), f"{label}: dim{d} cell gid sets differ")
        same = 0
        for gid, corners in cs_py.items():
            if gid in cs_cpp and tuple(sorted(cs_cpp[gid])) == tuple(sorted(corners)):
                same += 1
        check(same == len(cs_py), f"{label}: dim{d} corner sets differ ({same}/{len(cs_py)} match)")
        vtk_cpp = corner_sets_from_sections(sec, d, key="cell_vtk_gids")
        ordered = sum(1 for gid, corners in cs_py.items() if vtk_cpp.get(gid) == corners)
        check(ordered == len(cs_py), f"{label}: dim{d} VTK node order differs ({ordered}/{len(cs_py)} match)")

        types_cpp = {int(g): int(t) for g, t in zip(sec[f"cell_gid:{d}"].ravel(), sec[f"cell_type:{d}"].ravel())}
        types_py = {int(g): int(t) for g, t in zip(block.gids, block.types)}
        check(types_cpp == types_py, f"{label}: dim{d} vtk cell types differ")

        order_py = np.argsort(block.gids)
        gpy = block.gids[order_py]
        for name, arr in block.cell_data.items():
            key = f"cell_data:{d}:{name}"
            if key not in sec:
                FAIL.append(f"{label}: missing cell field {name} (dim {d}) in cpp dump")
                continue
            gcpp = sec[f"cell_gid:{d}"].ravel()
            order_cpp = np.argsort(gcpp)
            check(np.array_equal(gcpp[order_cpp], gpy), f"{label}: dim{d} cell gid order mismatch for {name}")
            a = np.asarray(arr).reshape(len(block.gids), -1)[order_py]
            c = sec[key][order_cpp]
            allclose_nan(a, c, f"{label}: dim{d} cell field {name}")


def check_file(name, path=None, dims=None):
    path = path or (DATA / name)
    if not Path(path).exists():
        print(f"skip {name}: not present")
        return
    sec = run_driver(path, dims=dims)
    piece = sv.read(str(path), dims=dims)
    compare_piece(sec, piece, name, dims=dims)
    # also check the HaloCache-backed path gives identical results
    sec_cached = run_driver(path, dims=dims, cached=True)
    compare_piece(sec_cached, piece, name + " [cached]", dims=dims)
    print(f"ok  {name}: {PASS['n']} checks so far, {len(FAIL)} failed")


def check_parts(name, path, nparts, label):
    for p in range(nparts):
        sec = run_driver(path, parts=[p])
        piece = sv.read(str(path), parts=[p])
        compare_piece(sec, piece, f"{label} part {p}")
    # whole file too
    sec = run_driver(path)
    piece = sv.read(str(path))
    compare_piece(sec, piece, f"{label} whole")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne20", action="store_true", help="also check the real 6-part ne20 file")
    ap.add_argument("--skip-split", action="store_true")
    args = ap.parse_args()

    if not DRIVER.exists():
        print("cpp_driver not built; run tests/build.sh first", file=sys.stderr)
        sys.exit(2)

    for name in ["hex_fields.soma", "mixed3d.soma", "highorder3d.soma", "highorder2d.soma",
                "polygons.soma", "hexes_mixed.soma", "64bricks_1khex.soma", "recMeshOcn128.soma",
                "hybrid_3quad_1tri.soma"]:
        check_file(name)
    check_file("hexes_mixed.soma", dims=[1, 2, 3])

    if args.ne20:
        if Path(NE20).exists():
            check_file("ne20_2km_centroids.soma", path=NE20)
            check_parts("ne20_2km_centroids.soma", NE20, 6, "ne20")
        else:
            print(f"skip ne20: {NE20} not present")

    if not args.skip_split:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for name, nparts in [("hex_fields.soma", 4), ("highorder3d.soma", 3),
                                 ("recMeshOcn128.soma", 5), ("polygons.soma", 3)]:
                src = DATA / name
                if not src.exists():
                    continue
                dst = td / f"split_{name}_{nparts}.soma"
                soma_split.split(str(src), str(dst), nparts)
                check_parts(name, dst, nparts, f"split({name},{nparts})")

    print(f"\n{PASS['n']} checks passed, {len(FAIL)} failed")
    if FAIL:
        for m in FAIL[:50]:
            print("FAIL:", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
