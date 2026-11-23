# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Side-by-side quality renders of two CONUS meshes at one window (pvbatch under MPI).

    mpiexec -n P pvbatch conus_compare.py SomaReader.py PREPASS.json OUTDIR HALF_KM NAME=FILE ...

The window is centred at the prepass's fine_edge cell (the outermost finest-level cell), in the
stereographic projection centred at the finest level's mean direction; HALF_KM is its ground
half-height (the projected half-height is HALF_KM * k / R, k = 2 / (1 + cos theta)). For each
mesh it renders min_angle and area_ratio with cell edges, unlit, colour bars drawn later.
"""

import json
import math
import sys
from pathlib import Path

from paraview.simple import *  # noqa: F401,F403

plugin, prepass, outdir, half_km = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
meshes = dict(a.split("=", 1) for a in sys.argv[5:])
outdir.mkdir(parents=True, exist_ok=True)
P = json.load(open(prepass))
R_KM = P["radius"] / 1000.0
RADIUS = P["radius"]
lon0, lat0 = math.radians(P["center"]["lon"]), math.radians(P["center"]["lat"])
C = [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)]
E = [-math.sin(lon0), math.cos(lon0), 0.0]
N = [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def project(u):
    d = 1.0 + dot(u, C)
    return 2 * dot(u, E) / d, 2 * dot(u, N) / d


LoadPlugin(plugin, ns=globals())
target = P["fine_edge"]["unit"]
tx, ty = project(target)
k = 2.0 / (1.0 + dot(target, C))
half = half_km * k / R_KM
SIZE = [1500, 1200]
FIELDS = {"min_angle": (30.0, 90.0, ("Inferno (matplotlib)", "Inferno", "inferno")),
          "area_ratio": (1.0, 7.0, ("Viridis (matplotlib)", "Viridis", "viridis"))}
meta = {"half_km": half_km, "k": k, "size": SIZE, "fields": {f: [lo, hi] for f, (lo, hi, _) in FIELDS.items()},
        "renders": {}}
for name, path in meshes.items():
    reader = SomaReader(FileNames=[path])  # noqa: F821
    reader.PointArrayStatus = []
    reader.CellArrayStatus = list(FIELDS)
    reader.SetFieldLabels = 0
    reader.PartIds = 0
    reader.GlobalIds = 0
    clip = Clip(Input=reader)
    clip.ClipType = "Plane"
    angle = math.degrees(2.5 * half_km / R_KM) + 0.02
    clip.ClipType.Origin = [RADIUS * math.cos(math.radians(angle)) * a for a in target]
    clip.ClipType.Normal = list(target)
    clip.Crinkleclip = 1
    clip.Invert = 0  # keep the cap about the target (ParaView keeps the other side by default)
    calc = Calculator(Input=clip)
    calc.CoordinateResults = 1
    calc.ResultArrayName = "stereographic"
    den = f"(mag(coords)+({C[0]!r})*coordsX+({C[1]!r})*coordsY+({C[2]!r})*coordsZ)"
    x = f"2*(({E[0]!r})*coordsX+({E[1]!r})*coordsY+({E[2]!r})*coordsZ)/{den}"
    y = f"2*(({N[0]!r})*coordsX+({N[1]!r})*coordsY+({N[2]!r})*coordsZ)/{den}"
    calc.Function = f"({x})*iHat+({y})*jHat+0*kHat"
    for field, (lo, hi, presets) in FIELDS.items():
        v = CreateView("RenderView")
        v.ViewSize = SIZE
        v.OrientationAxesVisibility = 0
        v.UseColorPaletteForBackground = 0
        v.Background = [1, 1, 1]
        v.CameraParallelProjection = 1
        v.CameraPosition = [tx, ty, 10]
        v.CameraFocalPoint = [tx, ty, 0]
        v.CameraViewUp = [0, 1, 0]
        v.CameraParallelScale = half
        SetActiveView(v)
        d = Show(calc, v)
        ColorBy(d, ("CELLS", field))
        lut = GetColorTransferFunction(field)
        used = None
        for preset in presets:
            try:
                if lut.ApplyPreset(preset, True):
                    used = preset
                    break
            except RuntimeError:
                continue
        lut.RescaleTransferFunction(lo, hi)
        lut.AutomaticRescaleRangeMode = "Never"
        d.SetScalarBarVisibility(v, False)
        d.Ambient, d.Diffuse = 1.0, 0.0
        d.SetRepresentationType("Surface With Edges")
        d.EdgeColor = [0.1, 0.1, 0.1]
        d.LineWidth = 0.9
        fname = f"compare_{name}_{field}.png"
        SaveScreenshot(str(outdir / fname), v, ImageResolution=SIZE)
        meta["renders"][fname] = {"mesh": name, "file": path, "field": field, "preset": used}
        print("rendered", fname, flush=True)
        Delete(d)
        Delete(v)
json.dump(meta, open(outdir / "compare_meta.json", "w"), indent=1)
