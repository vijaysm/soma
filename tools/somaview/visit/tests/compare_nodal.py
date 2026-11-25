# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""soma_reader.hpp's sub-quads of spectral-element fields, against files whose values are known and
against somaview's Python core; and damaged files, which it must refuse.

usage: python compare_nodal.py <cpp_driver> <workdir> [FILE.soma ...]

Writes its own files (tests/nodal_files.py) into <workdir>: every one whole and part by part; any
FILE given is compared with the Python core too. Build the driver with -fsanitize=address so that an
over-read in a damaged file is caught.
"""
import pathlib
import subprocess
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent / "src"), str(HERE.parent.parent / "tests"), str(HERE)]

import nodal_files as nf  # noqa: E402
import somaview as sv  # noqa: E402
from compare_with_python import read_sections  # noqa: E402

TOL = 1e-12  # positions, relative to the mesh's size; values must be equal
CASES = [dict(p=1), dict(p=3), dict(p=4), dict(p=8, n=2), dict(p=3, radius=6.37e6), dict(p=3, nodes=8),
         dict(p=4, nodes=9), dict(p=5, nodes=9, radius=2.0), dict(p=3, parts=3, n=5),
         dict(p=3, parts=3, partial=True), dict(p=3, partial=True), dict(p=4, nodes=9, radius=2.0, parts=4)]
DAMAGE = {
    "counts beyond the values": ("counts", lambda a: a.__setitem__(0, 999)),
    "negative count": ("counts", lambda a: a.__setitem__(0, -16)),
    "counts of two quads swapped unevenly": ("counts", lambda a: a.__setitem__(slice(0, 2), [[15], [17]])),
    "domain row beyond the elements": ("domain", lambda a: a.__setitem__(0, 999)),
    "domain row repeated": ("domain", lambda a: a.__setitem__(1, a[0])),
    "domain row negative": ("domain", lambda a: a.__setitem__(0, -1)),
}
passed, failed = 0, []


def check(ok, what):
    global passed
    passed += bool(ok)
    if not ok:
        failed.append(what)


def drive(driver, path, out, parts=None):
    r = subprocess.run([driver, str(path), str(out), ",".join(map(str, parts or [])), "2", "nodal"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stderr


def same(sec, piece, p, label):
    tag = f":{p}"
    size = np.abs(piece.points).max()
    check(np.array_equal(sec["nodal_gid" + tag].ravel(), piece.point_gids), f"{label}: points")
    check(np.array_equal(sec["nodal_conn" + tag].ravel(), piece.cells.connectivity), f"{label}: sub-quads")
    check(np.abs(sec["nodal_xyz" + tag] - piece.points).max() <= TOL * size, f"{label}: positions")
    names = {k.split(":", 2)[2] for k in sec if k.startswith("nodal_data" + tag + ":")}
    check(names == set(piece.point_data), f"{label}: fields {names} against {set(piece.point_data)}")
    for name, v in piece.point_data.items():
        got = sec.get(f"nodal_data{tag}:{name}")
        check(got is not None and np.array_equal(got, np.asarray(v, float).reshape(len(got), -1), equal_nan=True),
              f"{label}: {name}")


def known(sec, pos, fields, p, label):
    gid, npts = sec[f"nodal_gid:{p}"].ravel(), (p + 1) ** 2
    e, k = gid // npts - 101, gid % npts
    check(np.abs(sec[f"nodal_xyz:{p}"] - pos[e, k]).max() <= TOL * np.abs(pos).max(), f"{label}: known positions")
    for name, v in fields.items():
        check(np.array_equal(sec[f"nodal_data:{p}:{name}"], v[e, k], equal_nan=True), f"{label}: known {name}")


def main(driver, work, files):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / "dump.bin"
    for case in CASES:
        path = work / "case.soma"
        _, _, pos, fields = nf.write(path, **case)
        for parts in [None] + [[q] for q in range(case.get("parts", 1))]:
            label = f"{case} parts {parts}"
            code, err = drive(driver, path, out, parts)
            check(code == 0, f"{label}: {err}")
            if code == 0:
                sec = read_sections(out)
                same(sec, sv.read_nodal(path, parts), case["p"], label)
                known(sec, pos, fields, case["p"], label)
    for f in files:
        with sv.SomaFile(f) as sf:
            orders, nparts = sorted({x.order for x in sf.fields if x.nodal_on_quads}), sf.nparts
        for parts in [None] + ([[q] for q in range(nparts)] if nparts > 1 else []):
            code, err = drive(driver, f, out, parts)
            check(code == 0, f"{f}: {err}")
            for p in orders if code == 0 else []:
                same(read_sections(out), sv.read_nodal(f, parts, order=p), p, f"{f} parts {parts}")
    for name, (column, change) in DAMAGE.items():
        path = work / "bad.soma"
        nf.write(path, partial=True)
        with h5py.File(path, "r+") as h:
            d = h[f"soma/fields/f0/dim2/{column}"]
            a = d[...]
            change(a)
            d[...] = a
        code, err = drive(driver, path, out)
        check(code == 1 and "error: " in err and "AddressSanitizer" not in err, f"damaged, {name}: exit {code} {err[:300]}")
    print(f"{passed} checks passed, {len(failed)} failed")
    for m in failed[:30]:
        print("FAIL:", m)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
