# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Cross-checks soma_reader.hpp's AMR reading against somaview.amr.AmrFile, for both ground
truth AMR checkpoints (amr2d.soma, amr3d.soma)."""
import struct
import subprocess
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import somaview as sv  # noqa: E402
from somaview.amr import AmrFile  # noqa: E402

DATA = Path(os.environ.get("SOMAVIEW_DATA", ROOT / "tests" / "data"))
DRIVER = HERE / "cpp_driver"

PASS = 0
FAIL = []


def check(cond, msg):
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(msg)


def read_sections(path):
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


def check_file(name):
    path = DATA / name
    if not path.exists():
        print("skip", name)
        return
    out = HERE / "_amr_dump.bin"
    r = subprocess.run([str(DRIVER), str(path), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sec = read_sections(out)
    with AmrFile(str(path)) as af:
        check(int(sec["amr_nlevels"].ravel()[0]) == af.nlevels, f"{name}: nlevels")
        check(int(sec["amr_ratio"].ravel()[0]) == af.ratio, f"{name}: ratio")
        check(int(sec["amr_dim"].ravel()[0]) == af.dim, f"{name}: dim")
        for l in range(af.nlevels):
            boxes_cpp = sec[f"amr_box:{l}"]
            boxes_py = af.boxes[l]
            check(np.array_equal(boxes_cpp, boxes_py), f"{name}: level {l} boxes differ")
            for f in af.fields:
                shape_key = f"amr_field_shape:{f.name}:{l}"
                data_key = f"amr_field:{f.name}:{l}"
                shapes = sec[shape_key]
                flat = sec[data_key]
                py_vals = af.read(f.name, l)
                off = 0
                ok = True
                for b, v in enumerate(py_vals):
                    ni, nj, nk = int(shapes[b, 0]), int(shapes[b, 1]), int(shapes[b, 2])
                    npts = ni * nj * nk * f.ncomp
                    # Both sides are box-after-box, component-major (i fastest, component
                    # slowest) raw file order: compare the flat sequences directly.
                    cpp_box = flat[off:off + npts, 0]
                    v_box = v.reshape(-1)
                    if not np.allclose(v_box, cpp_box, atol=1e-6):
                        ok = False
                    off += npts
                check(ok, f"{name}: field {f.name} level {l} values differ")


for name in ["amr2d.soma", "amr3d.soma"]:
    check_file(name)

print(f"{PASS} checks passed, {len(FAIL)} failed")
if FAIL:
    for m in FAIL:
        print("FAIL:", m)
    sys.exit(1)
