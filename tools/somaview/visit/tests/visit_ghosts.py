# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Run in VisIt's CLI: contours of a zonal field, averaged at the nodes, must be those of the same mesh in
one part however many parts it has: VisIt then finds each domain's ghost zones by the global ids.

  WORK=<dir> [NP=<ranks>] visit -nowin -cli -s visit_ghosts.py < /dev/null

WORK holds u<dim>_<parts>.soma (conforming) and g<dim>_<parts>.soma (hanging vertices), written by
`python visit_ghosts.py <dir>`. Prints VISITRESULT lines and exits 1 on a difference."""
import os
import sys

FILES = {"u2": (1, 4), "u3": (1, 4), "g2": (1, 5), "g3": (1, 4)}


def write(work):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
    import hanging_files as hf
    import numpy as np

    for name, parts in FILES.items():
        dim = int(name[1])
        n, depth = {"u2": (16, 0), "u3": (8, 0), "g2": (4, 3), "g3": (3, 2)}[name]
        boxes = hf.leaves(dim, n, depth, lambda x, l: np.linalg.norm(x - 0.4) < 0.45 / (l + 1))
        for p in parts:
            hf.write(os.path.join(work, "%s_%d.soma" % (name, p)), dim, boxes, n, depth, p)


def size(path, dim, variable, value):
    OpenDatabase(path)
    AddPlot("Pseudocolor", variable)
    AddOperator("Isosurface")
    a = IsosurfaceAttributes()
    a.contourMethod, a.contourValue, a.variable = a.Value, (value,), variable
    SetOperatorOptions(a)
    DrawPlots()
    Query("Total Length" if dim == 2 else "3D surface area")
    v = GetQueryOutputValue()
    DeleteAllPlots()
    CloseDatabase(path)
    return v


def main():
    work, bad = os.environ["WORK"], 0
    if os.environ.get("NP"):
        OpenComputeEngine("localhost", ("-np", os.environ["NP"]))
    SuppressMessages(1)
    for name, parts in sorted(FILES.items()):
        for variable, value in (("c", 0.7), ("f", 0.3)):
            got = [size("%s/%s_%d.soma" % (work, name, p), int(name[1]), variable, value) for p in parts]
            ok = got[0] > 0.1 and abs(got[1] - got[0]) <= 1e-7 * got[0]  # VisIt's contours are of float precision
            bad += not ok
            print("VISITRESULT ghosts %s %s %s %s" % (name, variable, got, "ok" if ok else "FAIL"))
    return bad


if "AddPlot" in globals():
    try:
        status = 1 if main() else 0
    except Exception as e:
        print("VISITRESULT ghosts error %s" % e)
        status = 1
    sys.stdout.flush()
    sys.exit(status)
else:
    write(sys.argv[1])
