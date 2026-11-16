# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""What the figure modules share: the registry of figures, ParaView helpers,
the orthographic camera of the 3D schematics, and the ownership colours.

A figure module registers each figure with @figure(name, sets, kind); render.py
imports the modules and draws the figures asked for. See render.py.
"""

import json

import style

FIGURES = {}


def figure(name, sets, kind):
    def register(draw):
        FIGURES[name] = {"sets": sets, "kind": kind, "draw": draw}
        return draw

    return register


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    """Renders with the given camera and saves the shared size.

    A first render lets ParaView reset the camera to the data, as it does on a
    view's first render; the camera set after it is then the one drawn.
    """
    pv.Render(view)
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal"]
    view.CameraViewUp = camera["up"]
    view.CameraParallelProjection = camera.get("parallel", 0)
    if "parallel_scale" in camera:
        view.CameraParallelScale = camera["parallel_scale"]
    pv.Render(view)
    pv.SaveScreenshot(str(path), view, ImageResolution=list(style.PNG_SIZE), TransparentBackground=0,
                      CompressionLevel="9")


def pv_label(pv, view, text, position):
    """A caption in the view at a normalised position (lower left is 0, 0)."""
    label = pv.Text(Text=text)
    display = pv.Show(label, view)
    display.WindowLocation = "Any Location"
    display.Position = position
    display.FontSize = 30
    display.Color = [0.0, 0.0, 0.0]
    return display


class Camera:
    """An orthographic view with z up, seen from the direction `eye` (from the object to the viewer)."""

    def __init__(self, eye):
        n = sum(e * e for e in eye) ** 0.5
        self.eye = [e / n for e in eye]
        right = [-self.eye[1], self.eye[0], 0.0]  # (0, 0, 1) x eye
        n = sum(r * r for r in right) ** 0.5
        self.right = [r / n for r in right]
        e, r = self.eye, self.right
        self.up = [e[1] * r[2] - e[2] * r[1], e[2] * r[0] - e[0] * r[2], e[0] * r[1] - e[1] * r[0]]

    def __call__(self, p):
        return (sum(a * b for a, b in zip(self.right, p)), sum(a * b for a, b in zip(self.up, p)))

    def depth(self, p):
        """Larger is nearer the viewer."""
        return sum(a * b for a, b in zip(self.eye, p))


# Front, right and above: the view of the reference cells.
CELL_VIEW = Camera((0.45, -1.0, 0.55))


def face_normal(points, face):
    a, b, c = points[face[0]], points[face[1]], points[face[-1]]
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def in_front(points, face, camera=CELL_VIEW):
    return camera.depth(face_normal(points, face)) > 0


def centroid(points):
    return [sum(p[k] for p in points) / len(points) for k in range(len(points[0]))]


OWNERSHIP = {  # kind: (colour, label)
    "interior": ("#56B4E9", "interior"),
    "shared": ("#E69F00", "shared, owned here"),
    "shared_copy": ("#F0E442", "shared copy"),
    "ghost": ("#CC79A7", "ghost"),
}
