# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""soma_reader.hpp's cells closed at hanging vertices and its node values of AMR fields, against
somaview's Python core; and damaged "refine:birth" records, which must not crash it.

usage: python compare_hanging.py <cpp_driver> <workdir> [FILE.soma ...]

Writes its own files (tests/hanging_files.py, tests/amr_files.py) into <workdir>: the meshes whole and
part by part; any mesh FILE given is compared too. Build the driver with -fsanitize=address.
"""
import os
import pathlib
import subprocess
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent / "src"), str(HERE.parent.parent / "tests"), str(HERE)]
os.environ.setdefault("SOMAVIEW_VTK_WEDGE", "current")  # of both readers

import amr_files  # noqa: E402
import hanging_files as hf  # noqa: E402
from compare_with_python import read_sections  # noqa: E402
from somaview.amr import AmrFile, smooth  # noqa: E402
from somaview.format import FileKind, SomaFile, detect  # noqa: E402
from somaview.mesh import read_piece  # noqa: E402

passed, failed = 0, []


def check(ok, what):
    global passed
    passed += bool(ok)
    if not ok:
        failed.append(what)


def drive(driver, path, out, parts=()):
    r = subprocess.run([driver, str(path), str(out), ",".join(map(str, parts)), "", "cached", "closed"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stderr


def mesh(driver, path, work):
    with SomaFile(path) as sf:
        dim, nparts = sf.dim, sf.nparts
    closed = 0
    for parts in [list(range(nparts))] + ([[p] for p in range(nparts)] if nparts > 1 else []):
        label = f"{path.name} {parts if len(parts) == 1 else 'all'}"
        rc, err = drive(driver, path, work / "c.bin", parts)
        check(rc == 0, f"{label}: {err[-300:]}")
        if rc:
            continue
        sec, p = read_sections(work / "c.bin"), read_piece(path, parts, close_hanging=True)
        b = p.cells
        conn, off = sec[f"cell_conn:{dim}"].ravel(), sec[f"cell_corner_offsets:{dim}"].ravel()
        check(np.array_equal(sec[f"cell_type:{dim}"].ravel(), b.types), f"{label}: types")
        check(np.array_equal(off, b.offsets), f"{label}: offsets")
        check(len(sec["point_gid"]) == len(p.points), f"{label}: points")
        if len(conn) != len(b.connectivity):
            check(False, f"{label}: connectivity")
            continue
        scale = np.abs(p.points).max()
        check(np.abs(sec["point_xyz"][conn] - p.points[b.connectivity]).max() <= 1e-14 * scale, f"{label}: corners")
        check(np.array_equal(sec["point_gid"].ravel()[conn], p.point_gids[b.connectivity]), f"{label}: gids")
        src = sec[f"cell_source:{dim}"].ravel()
        check(np.array_equal(src, b.source if b.source is not None else np.empty(0)), f"{label}: source")
        for k, a in p.point_data.items():
            v = sec[f"point_data:{k}"].reshape(len(p.points), -1)[conn]
            tol = 1e-6 if a.dtype == np.float32 else 1e-14  # the driver reads, and so averages, doubles
            want = np.asarray(a, float).reshape(len(a), -1)[b.connectivity]
            check(np.allclose(v, want, rtol=tol, atol=tol * np.nanmax(np.abs(want), initial=1), equal_nan=True), f"{label}: {k}")
        for k, a in b.cell_data.items():
            v = sec[f"cell_data:{dim}:{k}"]
            check(np.array_equal(v, np.asarray(a, float).reshape(v.shape), equal_nan=True), f"{label}: {k}")
        closed += b.source is not None
    return closed


def amr(driver, path, work):
    rc, err = drive(driver, path, work / "a.bin")
    check(rc == 0, f"{path.name}: {err[-300:]}")
    if rc:
        return
    sec = read_sections(work / "a.bin")
    with AmrFile(path) as af:
        for f in af.fields:
            for l in range(af.nlevels):
                want = np.concatenate([v.ravel() for v in smooth(af, f.name, l)])
                got = sec[f"amr_smooth:{f.name}:{l}"].ravel()
                check(got.shape == want.shape and np.abs(got - want).max() <= 1e-14 * np.abs(want).max(), f"{path.name}: {f.name} level {l}")


def damaged(driver, work):
    """Births that name themselves, each other, or wrong counts: no crash, no endless recursion."""
    f = work / "damaged.soma"
    for case in range(6):
        hf.write(f, 3 if case % 2 else 2, hf.leaves(3 if case % 2 else 2, 2, 1, lambda x, l: x[0] < 0.5), 2, 1)
        with h5py.File(f, "r+") as h:
            g = next(g for g in h["soma/fields"].values() if g.attrs["name"] == b"refine:birth")
            d = next(g[k] for k in g if k.startswith("dim"))["values"]
            b = d[...]
            gid = h["soma/dim0/gid"][...].ravel()
            if case < 2:  # every vertex of the first cell made between two others of it
                c = h[f"soma/dim{3 if case % 2 else 2}/conn"][...].ravel()[:4]
                at = {int(g): i for i, g in enumerate(gid)}
                for i in range(4):
                    b[at[int(c[i])]] = [2, c[i - 1], c[i - 2], 0, 0]
            elif case < 4:
                b[:, 0] = 99
            else:
                b[:] = [2, gid[0], gid[0], 0, 0]
            d[...] = b
        rc, err = drive(driver, f, work / "d.bin")
        read_piece(f, close_hanging=True)
        check(rc in (0, 1) and "Sanitizer" not in err and "runtime error" not in err, f"damaged {case}: {rc} {err[-300:]}")


def main():
    driver, work = sys.argv[1], pathlib.Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    closed = 0
    for dim, n, depth, parts, warp in [(2, 3, 2, 1, 0.0), (2, 4, 3, 5, 0.3), (3, 2, 1, 1, 0.0), (3, 3, 2, 4, 0.3)]:
        f = work / f"hanging{dim}_{parts}.soma"
        hf.write(f, dim, hf.leaves(dim, n, depth, lambda x, l: np.linalg.norm(x - 0.4) < 0.45 / (l + 1)), n, depth, parts, warp)
        closed += mesh(driver, f, work)
    check(closed >= 8, f"only {closed} pieces had hanging vertices")
    for dim, boxes in ((2, amr_files.BOXES_2D), (3, amr_files.BOXES_3D)):
        f = work / f"amr{dim}.soma"
        bumpy = lambda x, y, z, l: np.sin(9 * x) * np.cos(7 * y) + z * z + 0.3 * l  # noqa: E731
        amr_files.write(f, dim, boxes, {"c": ("cell", 2, lambda x, y, z, l: np.stack([bumpy(x, y, z, l), x - y])), "n": ("node", 1, bumpy)})
        amr(driver, f, work)
    damaged(driver, work)
    for path in map(pathlib.Path, sys.argv[3:]):
        (amr if detect(path) == FileKind.amr else mesh)(driver, path, work)
    print(f"{passed} checks passed, {len(failed)} failed")
    for w in failed[:20]:
        print("FAIL", w)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
