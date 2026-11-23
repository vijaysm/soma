# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Malformed SOMA files must make soma_reader.hpp throw, never read out of bounds.

usage: python check_malformed.py <cpp_driver> <good.soma> <workdir>

Writes one damaged copy of <good.soma> per case into <workdir> (h5py) and runs <cpp_driver> on
each. A case passes when the driver exits 1 with an "error: " message and no AddressSanitizer
report; build the driver with -fsanitize=address so that an over-read is caught even when it does
not crash. The undamaged file must read with exit 0. <good.soma> is a 2D mesh file of one part,
with fields and mesh values, e.g. soma-convert data/hybrid_3quad_1tri.h5m good.soma.
"""
import pathlib
import shutil
import subprocess
import sys

import h5py


def set_attr(path, value):
    def edit(f):
        group, name = path.rsplit("@", 1)
        f[group].attrs[name] = value
    return edit


def edit_attr(path, change):
    def edit(f):
        group, name = path.rsplit("@", 1)
        words = list(f[group].attrs[name])
        f[group].attrs[name] = change(words)
    return edit


def set_rows(path, values):
    def edit(f):
        for row, value in values.items():
            f[path][row] = value
    return edit


def with_word(i, v):
    def change(words):
        words[i] = v
        return words
    return change


CASES = {
    "root meta of 5 words": edit_attr("soma@meta", lambda w: w[:5]),
    "more parts than table rows": edit_attr("soma@meta", with_word(2, 3)),
    "field meta of 10 words": edit_attr("soma/fields/f0@meta", lambda w: w[:10]),
    "field ndims beyond its meta": edit_attr("soma/fields/f0@meta", with_word(16, 40)),
    "value default shorter than ncomp": edit_attr("soma/values/v0@meta", with_word(1, 1000)),
    "counts sum beyond conn": set_rows("soma/dim2/counts", {0: 100}),
    "quad with 2 corners": set_rows("soma/dim2/counts", {0: 2}),
}


def run(driver, soma, out):
    r = subprocess.run([driver, str(soma), str(out)], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stderr


def main(driver, good, work):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    code, err = run(driver, good, work / "good.bin")
    if code != 0:
        print(f"FAIL undamaged file: exit {code}\n{err}")
        return 1
    failed = 0
    for i, (name, edit) in enumerate(CASES.items()):
        bad = work / f"bad{i}.soma"
        shutil.copyfile(good, bad)
        with h5py.File(bad, "r+") as f:
            edit(f)
        code, err = run(driver, bad, work / f"bad{i}.bin")
        ok = code == 1 and "error: " in err and "AddressSanitizer" not in err
        failed += not ok
        first = err.strip().splitlines()[0] if err.strip() else ""
        print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {code}: {first[:150]}")
    print(f"{len(CASES) - failed}/{len(CASES)} malformed files refused")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
