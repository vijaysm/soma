# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""ParaView renders of SOMA files through the reader plugin (pvbatch; run under mpiexec for a
distributed read with IceT compositing).

    mpiexec -n 4 pvbatch pv_gallery.py SomaReader.py OUTDIR name=path [name=path ...]

Names select the view: hexmpi, conus, bricks, highorder, amr.
"""

import sys
from pathlib import Path

from paraview.simple import *  # noqa: F401,F403

plugin, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
LoadPlugin(plugin, ns=globals())
SIZE = [1600, 1000]


#: Colormap presets by family; the first name this ParaView has is used (names changed across versions).
PRESETS = {"Viridis (matplotlib)": ("Viridis (matplotlib)", "Viridis", "viridis", "Cool to Warm"),
           "Inferno (matplotlib)": ("Inferno (matplotlib)", "Inferno", "inferno", "Black-Body Radiation")}


def preset(lut, name):
    for candidate in PRESETS.get(name, (name,)):
        try:
            if lut.ApplyPreset(candidate, True):
                return
        except RuntimeError:
            continue


def view():
    v = CreateView("RenderView")
    v.ViewSize = SIZE
    v.OrientationAxesVisibility = 0
    v.Background = [1, 1, 1]
    v.UseColorPaletteForBackground = 0
    return v


def bar(d, v, title):
    d.SetScalarBarVisibility(v, True)
    b = GetScalarBar(d.LookupTable, v)
    b.Title = title
    b.ComponentTitle = ""
    b.TitleColor = b.LabelColor = [0.1, 0.1, 0.1]
    b.TitleFontSize = 18
    b.LabelFontSize = 16


def hexmpi(path):
    """Hex box written on 4 ranks: clipped, colored by pressure, velocity glyphs; and by the process
    that read each cell."""
    r = SomaReader(FileNames=[path])
    v = view()
    clip = Clip(Input=r)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = [0.5, 0.4, 0.3]
    clip.ClipType.Normal = [1, 1, 0]
    surf = ExtractSurface(Input=clip)
    d = Show(surf, v)
    ColorBy(d, ("POINTS", "pressure"))
    d.LookupTable = GetColorTransferFunction("pressure")
    preset(d.LookupTable, "Cool to Warm")
    d.RescaleTransferFunctionToDataRange(True)
    bar(d, v, "pressure")
    g = Glyph(Input=surf, GlyphType="Arrow")
    g.OrientationArray = ["POINTS", "velocity"]
    g.ScaleArray = ["POINTS", "No scale array"]
    g.ScaleFactor = 0.04
    g.GlyphMode = "Every Nth Point"
    g.Stride = 23
    gd = Show(g, v)
    gd.ColorArrayName = ['POINTS', '']
    gd.AmbientColor = gd.DiffuseColor = [0.15, 0.15, 0.15]
    o = Show(Outline(Input=r), v)
    o.AmbientColor = o.DiffuseColor = [0.3, 0.3, 0.3]
    v.CameraPosition = [2.2, -1.6, 1.6]
    v.CameraFocalPoint = [0.5, 0.4, 0.3]
    v.CameraViewUp = [0, 0, 1]
    ResetCamera(v)
    SaveScreenshot(str(outdir / "pv_hex_4p_pressure.png"), v, ImageResolution=SIZE)
    # the distributed read: color by the process that holds each cell
    v2 = view()
    pid = ProcessIds(Input=ExtractSurface(Input=r))
    if "GenerateCellData" in pid.ListProperties():  # ParaView >= 5.11: point ids only by default
        pid.GenerateCellData = 1
    pid.UpdatePipeline()
    pid_name = next(n for n in ("CellProcessIds", "ProcessId") if n in pid.CellData.keys())
    d2 = Show(pid, v2)
    ColorBy(d2, ("CELLS", pid_name))
    d2.LookupTable = GetColorTransferFunction(pid_name)
    preset(d2.LookupTable, "Rainbow Uniform")
    d2.RescaleTransferFunctionToDataRange(True)
    d2.SetRepresentationType("Surface With Edges")
    d2.EdgeColor = [0.2, 0.2, 0.2]
    bar(d2, v2, "reading process")
    v2.CameraPosition = [2.2, -1.6, 1.6]
    v2.CameraFocalPoint = [0.5, 0.4, 0.3]
    v2.CameraViewUp = [0, 0, 1]
    ResetCamera(v2)
    SaveScreenshot(str(outdir / "pv_hex_4p_process.png"), v2, ImageResolution=SIZE)


def conus(path):
    """A regionally refined cubed-sphere mesh: cells colored by spacing_km, log scale."""
    r = SomaReader(FileNames=[path])
    v = view()
    d = Show(r, v)
    arr = "spacing_km"
    ColorBy(d, ("CELLS", arr))
    lut = GetColorTransferFunction(arr)
    preset(lut, "Viridis (matplotlib)")
    d.RescaleTransferFunctionToDataRange(True)
    lut.MapControlPointsToLogSpace()
    lut.UseLogScale = 1
    bar(d, v, "spacing (km)")
    v.CameraPosition = [-0.9, -3.0, 2.6]
    v.CameraFocalPoint = [0, 0, 0]
    v.CameraViewUp = [0, 0, 1]
    ResetCamera(v)
    cam = v.GetActiveCamera()
    cam.Zoom(1.25)
    SaveScreenshot(str(outdir / "pv_conus_spacing.png"), v, ImageResolution=SIZE)


def bricks(path):
    r = SomaReader(FileNames=[path])
    v = view()
    shrink = Shrink(Input=r)
    shrink.ShrinkFactor = 0.92
    d = Show(shrink, v)
    ColorBy(d, ("CELLS", "set:MATERIAL_SET"))
    lut = GetColorTransferFunction("set:MATERIAL_SET")
    preset(lut, "Rainbow Uniform")
    d.RescaleTransferFunctionToDataRange(True)
    bar(d, v, "MATERIAL_SET")
    v.CameraPosition = [3, -2.5, 2.5]
    v.CameraViewUp = [0, 0, 1]
    ResetCamera(v)
    SaveScreenshot(str(outdir / "pv_bricks_material.png"), v, ImageResolution=SIZE)


def highorder(path):
    r = SomaReader(FileNames=[path])
    v = view()
    s = ExtractSurface(Input=r)
    s.NonlinearSubdivisionLevel = 4
    d = Show(s, v)
    ColorBy(d, ("POINTS", "radius"))
    preset(GetColorTransferFunction("radius"), "Inferno (matplotlib)")
    d.RescaleTransferFunctionToDataRange(True)
    bar(d, v, "radius")
    e = Show(FeatureEdges(Input=s), v)
    e.ColorArrayName = ['POINTS', '']
    e.AmbientColor = e.DiffuseColor = [0.1, 0.1, 0.1]
    e.LineWidth = 2
    v.CameraPosition = [3.5, -6, 4]
    v.CameraFocalPoint = [3.3, 0.5, 0.5]
    v.CameraViewUp = [0, 0, 1]
    ResetCamera(v)
    SaveScreenshot(str(outdir / "pv_highorder_curved.png"), v, ImageResolution=SIZE)


def amr(path):
    r = SomaReader(FileNames=[path])
    v = view()
    sl = Slice(Input=r)
    sl.SliceType = "Plane"
    sl.SliceType.Origin = [0.5, 0.5, 0.5]
    sl.SliceType.Normal = [0, 0, 1]
    d = Show(sl, v)
    ColorBy(d, ("CELLS", "u", "0"))
    preset(GetColorTransferFunction("u"), "Cool to Warm")
    d.RescaleTransferFunctionToDataRange(True)
    d.SetRepresentationType("Surface With Edges")
    d.EdgeColor = [0.25, 0.25, 0.25]
    bar(d, v, "u[0]")
    o = Show(Outline(Input=r), v)
    o.AmbientColor = o.DiffuseColor = [0.3, 0.3, 0.3]
    v.CameraPosition = [0.5, 0.5, 3]
    v.CameraFocalPoint = [0.5, 0.5, 0.5]
    v.CameraViewUp = [0, 1, 0]
    ResetCamera(v)
    SaveScreenshot(str(outdir / "pv_amr3d_slice.png"), v, ImageResolution=SIZE)


for spec in sys.argv[3:]:
    name, path = spec.split("=", 1)
    globals()[name](path)
    print("rendered", name, flush=True)
