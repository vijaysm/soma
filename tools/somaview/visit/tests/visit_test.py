# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
# VisIt CLI driver: VISIT_TEST_EXPECTED=e.json VISIT_TEST_OUT=dir VISIT_TEST_NP=n visit -nowin -cli -s visit_test.py
# Opens every file with the SOMA plugin, checks NumZones and MinMax against the Python core's
# expectations, and saves a render of each. NPROCS > 1 starts a parallel compute engine.
import json
import os
import sys

expected = json.load(open(os.environ["VISIT_TEST_EXPECTED"]))
outdir = os.environ["VISIT_TEST_OUT"]
nprocs = int(os.environ.get("VISIT_TEST_NP", "1"))
os.makedirs(outdir, exist_ok=True)
results = []
if nprocs > 1:
    ok = OpenComputeEngine("localhost", ("-np", str(nprocs)))
    results.append({"engine": "parallel", "np": nprocs, "started": bool(ok)})

s = SaveWindowAttributes()
s.format = s.PNG
s.width, s.height = 1400, 1000
s.family = 0
s.outputToCurrentDirectory = 0
s.outputDirectory = outdir
s.resConstraint = s.NoConstraint

for path, e in expected.items():
    r = {"path": os.path.basename(path), "np": nprocs}
    try:
        if not OpenDatabase(path):
            r["error"] = "OpenDatabase failed"
            results.append(r)
            continue
        md = GetMetaData(path)
        r["meshes"] = [md.GetMeshes(i).name for i in range(md.GetNumMeshes())]
        r["domains"] = md.GetMeshes(0).numBlocks
        r["scalars"] = sorted(md.GetScalars(i).name for i in range(md.GetNumScalars()))
        var = e.get("var")
        names = set(r["scalars"]) | {md.GetVectors(i).name for i in range(md.GetNumVectors())}
        if var and var not in names and "nodal/" + var in names:
            var = "nodal/" + var  # a field name shared by several dimensions is qualified
        r["var"] = var
        if var:
            AddPlot("Pseudocolor", var)
        else:
            AddPlot("Mesh", r["meshes"][0])
        if e["kind"] == "mesh" and e["dim"] == 3 and var:
            AddPlot("Mesh", "mesh")
        DrawPlots()
        # The zones the reader returned (original data), and those after VisIt's own pipeline
        # (actual data: ghost zones it generates between domains, tessellated high-order cells).
        Query("NumZones", use_actual_data=0)
        r["NumZones"] = GetQueryOutputValue()
        r["NumZones_text"] = GetQueryOutputString().strip()[:300]
        Query("NumZones", use_actual_data=1)
        r["NumZones_actual"] = GetQueryOutputValue()
        r["NumZones_actual_text"] = GetQueryOutputString().strip()[:300]
        if var:
            SetActivePlots(0)
            Query("MinMax")
            mm = GetQueryOutputValue()
            r["MinMax_text"] = GetQueryOutputString().strip()[:300]
            r["MinMax"] = [mm[0], mm[1]] if isinstance(mm, (tuple, list)) and len(mm) >= 2 else None
        expect_zones = e["uncovered"] if e["kind"] == "amr" else e["cells"]
        r["expect_zones"] = expect_zones
        r["zones_ok"] = int(r["NumZones"]) == int(expect_zones) or (e["kind"] == "amr" and int(r["NumZones"]) == e["cells"])
        if var:
            r["expect_minmax"] = [e["min"], e["max"]]
            r["minmax_ok"] = r["MinMax"] is not None and \
                abs(r["MinMax"][0] - e["min"]) <= 1e-6 * max(1, abs(e["min"])) and \
                abs(r["MinMax"][1] - e["max"]) <= 1e-6 * max(1, abs(e["max"]))
        v = GetView3D()
        v.viewNormal = (0.55, -0.6, 0.58)
        v.viewUp = (0, 0, 1)
        SetView3D(v)
        a = GetAnnotationAttributes()
        a.userInfoFlag = 0
        a.databaseInfoFlag = 0
        SetAnnotationAttributes(a)
        s.fileName = "visit_" + os.path.splitext(os.path.basename(path))[0] + ("_np%d" % nprocs)
        SetSaveWindowAttributes(s)
        SaveWindow()
        DeleteAllPlots()
        CloseDatabase(path)
    except Exception as ex:  # report and go on, on a fresh engine if this one died
        r["error"] = repr(ex)
        try:
            DeleteAllPlots()
            CloseDatabase(path)
            CloseComputeEngine()
            if nprocs > 1:
                OpenComputeEngine("localhost", ("-np", str(nprocs)))
        except Exception as ex2:
            r["recover_error"] = repr(ex2)
    results.append(r)
    print("VISITRESULT " + json.dumps(r), flush=True)

print("VISITSUMMARY " + json.dumps(results), flush=True)
sys.exit(0)
