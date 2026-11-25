# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Run in VisIt's CLI: contours of an AMR field on mesh_closed have no seams between levels; the read
option that leaves cells with hanging vertices as stored; and soma_gid, the cell a picked zone is of.

  WORK=<dir> [NP=<ranks>] visit -nowin -cli -s visit_closed.py < /dev/null

WORK holds what `python visit_closed.py <dir>` writes. VisIt writes the contours beside them:
`python visit_closed.py <dir> check` then counts their loose edges. Prints VISITRESULT lines; exits 1 on a
failure."""
import glob
import os
import sys

VALUES = (0.1234, 0.4)


def write(work):
    import json

    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(here, "..", "..", "tests"), os.path.join(here, "..", "..", "src")]
    import amr_files as af
    import hanging_files as hf
    import somaview as sv

    bump = lambda x, y, z, l: np.sin(9 * x) * np.cos(7 * y) + np.sin(8 * z + x) + 0.3 * l  # noqa: E731
    af.write(os.path.join(work, "a3.soma"), 3, [*af.BOXES_3D, [((12, 12, 12), (19, 19, 19))]], {"c": ("cell", 1, bump)})
    expect = {}
    for parts in (1, 4):
        f = os.path.join(work, "h3_%d.soma" % parts)
        boxes = hf.leaves(3, 3, 2, lambda x, l: np.linalg.norm(x - 0.4) < 0.45 / (l + 1))
        hf.write(f, 3, boxes, 3, 2, parts=parts)
        p = sv.read(f, parts=[0], close_hanging=True)
        expect["h3_%d" % parts] = {"stored": len(boxes), "closed": len(sv.read(f, close_hanging=True).cells),
                                   "gids": [int(g) for g in p.cells.gids]}
    with open(os.path.join(work, "expect.json"), "w") as out:
        json.dump(expect, out)


def check(work):
    """Loose edges of each exported contour inside the unit box, and whether there are triangles at all."""
    import numpy as np
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData, vtkFeatureEdges
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkIOLegacy import vtkGenericDataObjectReader

    loose = {}
    for name in ("smooth", "closed"):
        for i in range(len(VALUES)):
            app = vtkAppendPolyData()
            for f in glob.glob(os.path.join(work, "%s_%d" % (name, i), "*.vtk")):
                r = vtkGenericDataObjectReader()
                r.SetFileName(f)
                r.Update()
                g = vtkGeometryFilter()
                g.SetInputData(r.GetOutput())
                g.Update()
                app.AddInputData(g.GetOutput())
            clean = vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(2e-6)  # VisIt writes and contours in single precision
            e = vtkFeatureEdges()
            e.SetInputConnection(clean.GetOutputPort())
            e.BoundaryEdgesOn()
            e.FeatureEdgesOff()
            e.ManifoldEdgesOff()
            e.NonManifoldEdgesOff()
            e.Update()
            o, n = e.GetOutput(), clean.GetOutput().GetNumberOfCells()
            inside = 0
            if o.GetNumberOfCells():
                x = vtk_to_numpy(o.GetPoints().GetData()).astype(float)
                ends = x[vtk_to_numpy(o.GetLines().GetConnectivityArray()).reshape(-1, 2)]
                on = np.zeros(len(ends), bool)
                for d in range(3):
                    for side in (0.0, 1.0):
                        on |= (np.abs(ends[:, :, d] - side) < 1e-5).all(1)
                inside = int((~on).sum())
            loose.setdefault(name, []).append((inside, n))
    ok = all(i == 0 and n > 100 for i, n in loose["closed"]) and all(i > 0 for i, n in loose["smooth"])
    print("VISITRESULT seams %s %s" % (loose, "ok" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    import json

    work, failed = os.environ["WORK"], 0
    if int(os.environ.get("NP", "1")) > 1:
        OpenComputeEngine("localhost", ("-np", os.environ["NP"]))  # noqa: F821

    def result(name, ok, what):
        print("VISITRESULT %s %s %s" % (name, what, "ok" if ok else "FAIL"))
        return 0 if ok else 1

    OpenDatabase(os.path.join(work, "a3.soma"))  # noqa: F821
    for name in ("smooth", "closed"):
        for i, v in enumerate(VALUES):
            DeleteAllPlots()  # noqa: F821
            AddPlot("Pseudocolor", name + "/c")  # noqa: F821
            AddOperator("Isosurface")  # noqa: F821
            a = IsosurfaceAttributes()  # noqa: F821
            a.contourMethod, a.contourValue = a.Value, (v,)
            SetOperatorOptions(a)  # noqa: F821
            failed += result("draw", DrawPlots() == 1, "%s %g" % (name, v))  # noqa: F821
            e = ExportDBAttributes()  # noqa: F821
            e.db_type, e.filename, e.dirname, e.variables = "VTK", "%s_%d" % (name, i), work, ()
            ExportDatabase(e)  # noqa: F821
    DeleteAllPlots()  # noqa: F821
    AddPlot("Pseudocolor", "closed/c")  # noqa: F821
    DrawPlots()  # noqa: F821
    Query("Volume")  # noqa: F821
    v = GetQueryOutputValue()  # noqa: F821
    failed += result("volume", abs(v - 1) < 1e-6, v)

    expect = json.load(open(os.path.join(work, "expect.json")))
    for name in sorted(expect):
        f = os.path.join(work, name + ".soma")
        for closing in (1, 0):
            DeleteAllPlots()  # noqa: F821
            CloseDatabase(f)  # noqa: F821
            o = GetDefaultFileOpenOptions("SOMA")  # noqa: F821
            o["Close cells at hanging vertices"] = closing
            SetDefaultFileOpenOptions("SOMA", o)  # noqa: F821
            OpenDatabase(f, 0, "SOMA")  # noqa: F821
            AddPlot("Pseudocolor", "soma_gid")  # noqa: F821
            DrawPlots()  # noqa: F821
            Query("NumZones", use_actual_data=0)  # noqa: F821
            zones = int(GetQueryOutputValue())  # noqa: F821
            want = expect[name]["closed" if closing else "stored"]
            failed += result(name, zones == want, "closing %d: %d zones of %d" % (closing, zones, want))
            if closing:
                gids = expect[name]["gids"]
                for zone in (0, len(gids) // 2, len(gids) - 1):
                    got = PickByZone(zone, 0, vars=("soma_gid",))  # noqa: F821
                    got = got.get("soma_gid") if got else None
                    failed += result(name, got == gids[zone], "zone %d is of cell %s, %s" % (zone, got, gids[zone]))
    sys.exit(1 if failed else 0)


if "AddPlot" in globals():
    try:
        main()
    except Exception as e:
        print("VISITRESULT closed error %s" % e)
        sys.stdout.flush()
        sys.exit(1)
elif sys.argv[-1] == "check":
    sys.exit(check(sys.argv[1]))
else:
    write(sys.argv[1])
