# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Stereographic renders of mesh quality on a regionally refined SOMA mesh (pvbatch under MPI).

    mpiexec -n P pvbatch conus_stereo.py SomaReader.py FILE PREPASS.json OUTDIR

The projection is centred at the finest refinement level's mean direction c (from the
pre-pass), with east e and north n spanning the tangent plane there; a point p maps to

    X = 2 (p . e) / (|p| + p . c),    Y = 2 (p . n) / (|p| + p . c)

(units of the sphere radius). Each rank reads its own parts, keeps the whole cells within a cap
around c (crinkle clip, so cell values are untouched), projects its points, and IceT composites.
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
R_EARTH_KM = 6371.22
lon0, lat0 = math.radians(P["center"]["lon"]), math.radians(P["center"]["lat"])
C = [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)]
E = [-math.sin(lon0), math.cos(lon0), 0.0]
N = [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)]
RADIUS = P["radius"]


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
reader.CellArrayStatus = ["min_angle", "scaled_jacobian", "refine:level"]
reader.SetFieldLabels = 0
reader.PartIds = 0
reader.GlobalIds = 0


def cap(src, axis, angle_deg):
    """The whole cells within angle_deg of the unit direction axis."""
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


def graticule(view, step=10):
    """Parallels and meridians every `step` degrees within 50 degrees of the centre."""
    lines = []
    lat_c, lon_c = P["center"]["lat"], P["center"]["lon"]
    for lat in range(int(lat_c // step * step) - 40, int(lat_c) + 50, step):
        if abs(lat) < 89:
            lines.append([(lon, lat) for lon in [lon_c - 70 + 0.5 * k for k in range(281)]])
    for lon in range(int(lon_c // step * step) - 60, int(lon_c) + 70, step):
        lines.append([(lon, lat) for lat in [max(-89.0, lat_c - 45) + 0.5 * k for k in range(181)] if lat < 89.5])
    for ln in lines:
        pts = []
        for lon, lat in ln:
            u = unit_of(lon, lat)
            if dot(u, C) > math.cos(math.radians(55)):
                x, y = project(u)
                pts += [x, y, 0.002]
        if len(pts) >= 6:
            src = PolyLineSource(Points=pts)
            d = Show(src, view)
            d.ColorArrayName = ['POINTS', '']
            d.AmbientColor = d.DiffuseColor = [0.35, 0.35, 0.35]
            d.Opacity = 0.55
            d.LineWidth = 1.2


def view_of(cx, cy, half_height, size):
    v = CreateView("RenderView")
    v.ViewSize = size
    v.OrientationAxesVisibility = 0
    v.UseColorPaletteForBackground = 0
    v.Background = [1, 1, 1]
    v.CameraParallelProjection = 1
    v.CameraPosition = [cx, cy, 10]
    v.CameraFocalPoint = [cx, cy, 0]
    v.CameraViewUp = [0, 1, 0]
    v.CameraParallelScale = half_height
    return v


PRESETS = {"inferno": ("Inferno (matplotlib)", "Inferno", "inferno", "Black-Body Radiation"),
           "viridis": ("Viridis (matplotlib)", "Viridis", "viridis", "Cool to Warm")}


def apply_preset(lut, family):
    """The first of the preset's names this ParaView has (names changed across versions)."""
    for name in PRESETS[family]:
        try:
            if lut.ApplyPreset(name, True):
                return name
        except RuntimeError:
            continue
    raise RuntimeError(f"no {family} preset in this ParaView")


def color(display, view, name, title, lo, hi, preset):
    ColorBy(display, ("CELLS", name))
    lut = GetColorTransferFunction(name)
    META.setdefault("_pending", {})["preset_used"] = apply_preset(lut, preset)
    lut.RescaleTransferFunction(lo, hi)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.UseBelowRangeColor = 1
    lut.BelowRangeColor = [0.0, 0.85, 1.0]  # cells below the range stand out
    display.SetScalarBarVisibility(view, False)


META = {}


def caption(view, text, **kw):
    """Recorded for compose_conus.py, which writes titles and colour bars beside the image."""
    META.setdefault("_pending", {}).update(text=text, **kw)


S = P["stats"]
ang_lo, jac_lo = 72.0, 0.95  # about the 0.1st percentiles; cells below are drawn in the flag colour
below = {"min_angle": S["min_angle"]["count_below_72"], "scaled_jacobian": S["scaled_jacobian"]["count_below_0.95"]}
proj = stereo(cap(reader, C, 45))
SIZE = [2400, 1500]
info = f"{P['cells']:,} quadrilaterals, {P['ranks']} reading ranks".replace(",", " ")

for name, title, lo, hi, preset, fname in (
        ("min_angle", "minimum angle (degrees)", ang_lo, 90.0, "inferno", "conus_stereo_min_angle.png"),
        ("scaled_jacobian", "scaled Jacobian", jac_lo, 1.0, "viridis", "conus_stereo_scaled_jacobian.png")):
    v = view_of(0.0, 0.0, 0.40, SIZE)
    d = Show(proj, v)
    color(d, v, name, title, lo, hi, preset)
    graticule(v)
    caption(v, f"{Path(path).name}: {title}, stereographic projection centred at "
               f"{P['center']['lat']:.1f} N, {-P['center']['lon']:.1f} W; graticule every 10 degrees\n{info}; "
               f"min {S[name]['min']:.3g}, median {S[name]['p50']:.3g}; {below[name]} cells below {lo:g} in cyan")
    SaveScreenshot(str(out / fname), v, ImageResolution=SIZE)
    META[fname] = dict(META.pop("_pending"), field=name, title=title, lo=lo, hi=hi, cmap=preset)
    print("rendered", fname, flush=True)


def zoom(key, cells_half_height, fname, text, lo=None, marker=False):
    u = P[key]["unit"]
    spacing = P[key]["spacing_km"]
    half = cells_half_height * spacing / R_EARTH_KM
    angle = math.degrees(4 * half)  # a cap comfortably larger than the window
    local = stereo(cap(reader, u, max(angle, 0.05)))
    x, y = project(u)
    v = view_of(x, y, half, [1800, 1350])
    d = Show(local, v)
    d.SetRepresentationType("Surface With Edges")
    d.EdgeColor = [0.15, 0.15, 0.15]
    d.LineWidth = 0.6
    lo = ang_lo if lo is None else lo
    color(d, v, "min_angle", "minimum angle (degrees)", lo, 90.0, "inferno")
    caption(v, text, marker=marker, half_height_km=half * R_EARTH_KM)
    SaveScreenshot(str(out / fname), v, ImageResolution=[1800, 1350])
    META[fname] = dict(META.pop("_pending"), field="min_angle", title="minimum angle (degrees)", lo=lo, hi=90.0,
                       cmap="inferno")
    print("rendered", fname, flush=True)


we = P["fine_edge"]
zoom("fine_edge", 90, "conus_stereo_transition_zoom.png",
     f"Edge of the finest level: {we['lonlat'][1]:.2f} N, {-we['lonlat'][0]:.2f} W "
     f"({we['angle_deg']:.1f} degrees from the centre)")
wa = P["worst_min_angle"]
zoom("worst_min_angle", 8, "conus_stereo_worst_cell_zoom.png",
     f"Worst cell by minimum angle: {wa['value']:.2f} degrees at {wa['lonlat'][1]:.3f} N, {-wa['lonlat'][0]:.3f} W "
     f"(cell spacing {wa['spacing_km']:.1f} km), marked", lo=30.0, marker=True)
import json as _json
_json.dump(META, open(out / "meta.json", "w"), indent=1)
