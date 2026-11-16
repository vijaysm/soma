# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Draws the documentation figures from the data soma_figure_data wrote.

    pvpython docs/figures/render.py <data dir> <img dir> all | <figure>...
    pvpython docs/figures/render.py --list

Each figure is a function registered with @figure(name, sets, kind): it reads
its data sets' files from <data dir>/<set>/ and writes <img dir>/<name>.<kind>,
a PNG rendered by ParaView or an SVG drawn by matplotlib (see style.py). Only
the figures named are drawn, so an unrelated change never rewrites an image.
Run it through docs/figures/make_figures.sh, which makes the data first; see
docs/developer/making-figures.md.

Exits 0 on success, 1 when a figure's data is missing, 2 for an unknown figure.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import style  # noqa: E402

FIGURES = {}


def figure(name, sets, kind):
    def register(draw):
        FIGURES[name] = {"sets": sets, "kind": kind, "draw": draw}
        return draw

    return register


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- ParaView


def pv_view():
    """A fresh render view: white, no orientation axes, the shared size."""
    import paraview.simple as pv

    pv.ResetSession()
    view = pv.CreateView("RenderView")
    view.ViewSize = list(style.PNG_SIZE)
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = [1.0, 1.0, 1.0]
    view.EnableRenderOnInteraction = 0
    return pv, view


def pv_color_by(pv, view, display, field, title, association="CELLS", cmap="Viridis", integer=False):
    """Colours a display by a field, with a scalar bar; integer fields get one colour per value."""
    pv.ColorBy(display, (association, field))
    lut = pv.GetColorTransferFunction(field)
    lut.ApplyPreset(cmap, True)
    display.RescaleTransferFunctionToDataRange(True, False)
    if integer:
        lo, hi = display.Input.CellData[field].GetRange() if association == "CELLS" else \
            display.Input.PointData[field].GetRange()
        lut.Discretize = 1
        lut.NumberOfTableValues = int(round(hi - lo)) + 1
        lut.RescaleTransferFunction(lo - 0.5, hi + 0.5)
    display.SetScalarBarVisibility(view, True)
    bar = pv.GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0.0, 0.0, 0.0]
    bar.TitleFontSize = bar.LabelFontSize = 28
    bar.ScalarBarLength = 0.5
    if integer:
        bar.AutomaticLabelFormat = 0
        bar.LabelFormat = "{:.0f}"
        bar.AddRangeLabels = 0
    bar.WindowLocation = "Any Location"
    bar.Position = [0.86, 0.25]
    return lut


def pv_surface(pv, view, source):
    """Shows a source as a surface with dark edges and soft light, so side faces keep their colour."""
    display = pv.Show(source, view)
    display.SetRepresentationType("Surface With Edges")
    display.EdgeColor = [0.1, 0.1, 0.1]
    display.LineWidth = 2.0
    display.Ambient = 0.6
    display.Diffuse = 0.45
    display.Specular = 0.0
    return display


def pv_save(pv, view, path, camera):
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal"]
    view.CameraViewUp = camera["up"]
    view.CameraParallelProjection = camera.get("parallel", 0)
    if "parallel_scale" in camera:
        view.CameraParallelScale = camera["parallel_scale"]
    pv.Render(view)
    pv.SaveScreenshot(str(path), view, ImageResolution=list(style.PNG_SIZE), TransparentBackground=0,
                      CompressionLevel="9")


# ---------------------------------------------------------------- figures


@figure("box-mesh-gids", ["box"], "png")
def box_mesh_gids(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "box" / "box.vtk")])
    display = pv_surface(pv, view, mesh)
    pv_color_by(pv, view, display, "gid", "cell gid", integer=True)
    pv_save(pv, view, path, {"position": [10.5, -8.0, 8.5], "focal": [1.9, 1.5, 0.9], "up": [0, 0, 1]})


# ---------------------------------------------------------------- driver


def main(argv):
    if argv == ["--list"]:
        for name, f in sorted(FIGURES.items()):
            print(f"{name} {f['kind']} {','.join(f['sets'])}")
        return 0
    if len(argv) < 3:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    data, img, names = Path(argv[0]), Path(argv[1]), argv[2:]
    names = sorted(FIGURES) if names == ["all"] else names
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        print(f"render: unknown figure(s) {' '.join(unknown)}; known: {' '.join(sorted(FIGURES))}", file=sys.stderr)
        return 2
    img.mkdir(parents=True, exist_ok=True)
    for name in names:
        f = FIGURES[name]
        missing = [s for s in f["sets"] if not (data / s).is_dir()]
        if missing:
            print(f"render: {name} needs data set(s) {' '.join(missing)} in {data}", file=sys.stderr)
            return 1
        path = img / f"{name}.{f['kind']}"
        f["draw"](data, path)
        print(f"render: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
