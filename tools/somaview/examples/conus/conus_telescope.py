# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Telescoping stereographic zoom on a regionally refined SOMA mesh (pvbatch under MPI).

    mpiexec -n P pvbatch conus_telescope.py SomaReader.py FILE PREPASS.json OUTDIR

Panel a: the whole refined region, projection centred at the finest level's mean direction c.
Panels b-f: windows centred at the finest-level cell farthest from c (where the 200 m region
steps down to coarser levels), each 4 times smaller than the last, coloured by refinement
level, cell edges drawn once cells are a few pixels across.

Stereographic scale: a point at angle theta from c is magnified by k = 2 / (1 + cos theta), so a
window of ground half-height h km there spans h k / R in projected units (R the sphere radius).
"""

import json
import math
import sys
from pathlib import Path

from paraview.simple import *  # noqa: F401,F403

plugin, path, prepass, outdir = sys.argv[1:5]
out = Path(outdir)
out.mkdir(parents=True, exist_ok=True)
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


def unit_of(lon, lat):
    lo, la = math.radians(lon), math.radians(lat)
    return [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]


LoadPlugin(plugin, ns=globals())
reader = SomaReader(FileNames=[path])  # noqa: F821
reader.PointArrayStatus = []
reader.CellArrayStatus = ["refine:level"]
reader.SetFieldLabels = 0
reader.PartIds = 0
reader.GlobalIds = 0


def cap(src, axis, angle_deg):
    clip = Clip(Input=src)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = [RADIUS * math.cos(math.radians(angle_deg)) * a for a in axis]
    clip.ClipType.Normal = list(axis)
    clip.Crinkleclip = 1
    clip.Invert = 0
    return clip


def stereo(src):
    calc = Calculator(Input=src)
    calc.CoordinateResults = 1
    calc.ResultArrayName = "stereographic"
    den = f"(mag(coords)+({C[0]!r})*coordsX+({C[1]!r})*coordsY+({C[2]!r})*coordsZ)"
    x = f"2*(({E[0]!r})*coordsX+({E[1]!r})*coordsY+({E[2]!r})*coordsZ)/{den}"
    y = f"2*(({N[0]!r})*coordsX+({N[1]!r})*coordsY+({N[2]!r})*coordsZ)/{den}"
    calc.Function = f"({x})*iHat+({y})*jHat+0*kHat"
    return calc


def graticule(view, step, window):
    """Parallels and meridians every `step` degrees that cross `window` (cx, cy, half_x, half_y),
    merged into one polydata: one actor, however many lines (one actor per line exhausts the
    texture units of an offscreen context)."""
    cx, cy, hx, hy = window
    lat_c, lon_c = P["center"]["lat"], P["center"]["lon"]
    lines = []
    for lat in range(int(lat_c // step * step) - 40, int(lat_c) + 50, step):
        if abs(lat) < 89:
            lines.append([(lon, lat) for lon in [lon_c - 70 + 0.1 * step * k for k in range(int(1400 / step) + 1)]])
    for lon in range(int(lon_c // step * step) - 60, int(lon_c) + 70, step):
        lines.append([(lon, lat) for lat in [max(-89.0, lat_c - 45) + 0.1 * step * k for k in range(int(900 / step) + 1)]
                      if lat < 89.5])
    sources = []
    for ln in lines:
        pts = []
        for lon, lat in ln:
            u = unit_of(lon, lat)
            if dot(u, C) > math.cos(math.radians(55)):
                x, y = project(u)
                if abs(x - cx) < 1.5 * hx and abs(y - cy) < 1.5 * hy:
                    pts += [x, y, 0.002]
        if len(pts) >= 6:
            sources.append(PolyLineSource(Points=pts))
    if not sources:
        return
    append = globals().get("AppendGeometry") or globals().get("AppendPolyData")
    d = Show(append(Input=sources), view)
    d.ColorArrayName = ["POINTS", ""]
    d.AmbientColor = d.DiffuseColor = [0.35, 0.35, 0.35]
    d.Opacity = 0.55
    d.LineWidth = 1.2


# Discrete colours per refinement level (0 coarsest), sampled from viridis; the composer draws
# the key with the same table.
LEVEL_RGB = [[0.267, 0.005, 0.329], [0.255, 0.267, 0.529], [0.165, 0.471, 0.557],
             [0.133, 0.659, 0.518], [0.478, 0.820, 0.318], [0.992, 0.906, 0.145]]


def color_levels(display):
    ColorBy(display, ("CELLS", "refine:level"))
    lut = GetColorTransferFunction("refine:level")
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = sum(([str(l), str(l)] for l in range(6)), [])
    lut.IndexedColors = sum(LEVEL_RGB, [])
    display.SetScalarBarVisibility(GetActiveView(), False)


SIZE = [1500, 1200]
ASPECT = SIZE[0] / SIZE[1]
target = P["fine_edge"]["unit"]
tx, ty = project(target)
k_t = 2.0 / (1.0 + dot(target, C))
panels = [("a", -0.06, -0.04, 0.50, None)]  # overview, shifted so that window b lies well inside it
for letter, half_km in zip("bcdef", (600.0, 150.0, 37.5, 9.375, 2.34375)):
    panels.append((letter, tx, ty, half_km * k_t / R_KM, half_km))
edge_width = {"d": 0.35, "e": 0.9, "f": 1.6}

META = {"size": SIZE, "target": {"lonlat": P["fine_edge"]["lonlat"], "xy": [tx, ty], "k": k_t},
        "center": P["center"], "radius_km": R_KM, "panels": {}}
overview = None
for letter, cx, cy, half, half_km in panels:
    if letter == "a":
        overview = overview or stereo(cap(reader, C, 45))
        src = overview
    else:
        angle = math.degrees(2.5 * half_km / R_KM) + 0.02  # a cap larger than the window's diagonal
        src = stereo(cap(reader, target, angle))
    v = CreateView("RenderView")
    v.ViewSize = SIZE
    v.OrientationAxesVisibility = 0
    v.UseColorPaletteForBackground = 0
    v.Background = [1, 1, 1]
    v.CameraParallelProjection = 1
    v.CameraPosition = [cx, cy, 10]
    v.CameraFocalPoint = [cx, cy, 0]
    v.CameraViewUp = [0, 1, 0]
    v.CameraParallelScale = half
    SetActiveView(v)
    d = Show(src, v)
    color_levels(d)
    d.Ambient, d.Diffuse = 1.0, 0.0  # unlit: the drawn colours are exactly the key's
    if letter in edge_width:
        d.SetRepresentationType("Surface With Edges")
        d.EdgeColor = [0.1, 0.1, 0.1]
        d.LineWidth = edge_width[letter]
    if letter in "ab":
        graticule(v, 10 if letter == "a" else 2, (cx, cy, half * ASPECT, half))
    fname = f"telescope_{letter}.png"
    SaveScreenshot(str(out / fname), v, ImageResolution=SIZE)
    META["panels"][letter] = {"file": fname, "cx": cx, "cy": cy, "half": half,
                              "half_km": half_km if half_km else half * R_KM / 1.0,
                              "edges": letter in edge_width}
    print("rendered", fname, flush=True)
    Delete(d)
json.dump(META, open(out / "telescope_meta.json", "w"), indent=1)
