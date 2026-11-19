# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""The figures of the algorithms pages (docs/algorithms, docs/amr and docs/io)."""

from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

import style
from figlib import CELL_VIEW, Camera, centroid, figure, load_json, pv_color_by, pv_label, pv_save, pv_surface, pv_view

# The tet's children from further round, where the four in its middle separate.
TET_VIEW = Camera((1.0, -0.35, 0.45))
# A plane mesh seen from above, flat: no shading.
TOP = {"up": [0, 1, 0], "parallel": 1}


def save_png(pv, view, path, camera):
    """pv_save, then the image cropped to its drawing and cut to 256 colours (no dithering)."""
    pv_save(pv, view, path, camera)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    # Cut the blank bands round the drawing, keeping a margin, so the page shows it larger.
    box = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).getbbox()
    if box:
        m = 24
        rgb = rgb.crop((max(box[0] - m, 0), max(box[1] - m, 0), min(box[2] + m, rgb.width),
                        min(box[3] + m, rgb.height)))
    small = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    small.save(path, optimize=True)


def labels(pv, view, field, values, fmt):
    """The colour bar of a field with the given tick values only."""
    bar = pv.GetScalarBar(pv.GetColorTransferFunction(field), view)
    bar.UseCustomLabels = 1
    bar.CustomLabels = values
    bar.AutomaticLabelFormat = 0
    bar.LabelFormat = fmt
    bar.AddRangeLabels = 0


def flat(display):
    display.Ambient, display.Diffuse = 1.0, 0.0
    return display


def shared_range(pv, view, field, sources, association="CELLS", fmt=None):
    """One colour scale over several displays of a field: the union of their ranges."""
    lo, hi = float("inf"), float("-inf")
    for source in sources:
        source.UpdatePipeline()
        data = source.CellData if association == "CELLS" else source.PointData
        a, b = data[field].GetRange()
        lo, hi = min(lo, a), max(hi, b)
    lut = pv.GetColorTransferFunction(field)
    lut.RescaleTransferFunction(lo, hi)
    lut.AutomaticRescaleRangeMode = "Never"
    if fmt:
        bar = pv.GetScalarBar(lut, view)
        bar.RangeLabelFormat = fmt
    return lo, hi


def side_by_side(pv, view, left, right, shift, field, title, association="CELLS", integer=False):
    """Shows `left` and `right` moved by `shift` in x, coloured by one field on one scale."""
    moved = pv.Transform(Input=right)
    moved.Transform.Translate = [shift, 0.0, 0.0]
    for source in (left, moved):
        display = flat(pv_surface(pv, view, source))
        display.LineWidth = 1.5
        pv_color_by(pv, view, display, field, title, association=association, integer=integer)
    return moved


@figure("laplacian-smoothing", ["smooth"], "png")
def laplacian_smoothing(data, path):
    pv, view = pv_view()
    before = pv.LegacyVTKReader(FileNames=[str(data / "smooth" / "before.vtk")])
    after = pv.LegacyVTKReader(FileNames=[str(data / "smooth" / "after.vtk")])
    side_by_side(pv, view, before, after, 1.25, "area", "cell area / h²")
    shared_range(pv, view, "area", [before, after], fmt="{:.2f}")
    labels(pv, view, "area", [0.6, 0.8, 1.0, 1.2, 1.4], "{:.1f}")
    pv_label(pv, view, "inner vertices moved", [0.07, 0.8])
    pv_label(pv, view, "after 10 sweeps", [0.55, 0.8])
    # The squares span x in [0, 2.25] and y in [0, 1]; the view is 2.8 wide, its right sixth the colour bar.
    save_png(pv, view, path, {"position": [1.35, 0.5, 10.0], "focal": [1.35, 0.5, 0.0], **TOP, "parallel_scale": 1.0})


REGIONAL_RADIUS = 6.37122e6


def regional_turn(lat, lon):
    """A point (degrees) as set_regional turns the sphere: (37 N, 95.5 W) to +z, north +y, east +x."""
    import math

    d = math.pi / 180
    x, y, z = (math.cos(lat * d) * math.cos(lon * d), math.cos(lat * d) * math.sin(lon * d), math.sin(lat * d))
    t, u = 95.5 * d, (37 - 90) * d
    a, b = x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
    a2, c2 = a * math.cos(u) + z * math.sin(u), -a * math.sin(u) + z * math.cos(u)
    return [REGIONAL_RADIUS * 1.002 * q for q in (b, -a2, c2)]


def regional_box(pv, view, lat0, lat1, lon0, lon1):
    """The box's outline, a little above the sphere."""
    ring = [(lat0, lon0 + (lon1 - lon0) * k / 40) for k in range(41)] + \
           [(lat0 + (lat1 - lat0) * k / 20, lon1) for k in range(21)] + \
           [(lat1, lon1 - (lon1 - lon0) * k / 40) for k in range(41)] + \
           [(lat1 - (lat1 - lat0) * k / 20, lon0) for k in range(21)]
    line = pv.PolyLineSource(Points=[c for p in ring for c in regional_turn(*p)])
    display = pv.Show(line, view)
    display.DiffuseColor = display.AmbientColor = [0.85, 0.1, 0.1]
    display.LineWidth = 4.0
    return line


@figure("regional-levels", ["regional"], "png")
def regional_levels(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "regional" / "refined.vtk")])
    # Colour alone: the 33 km cells' edges would fill the picture (the close-up shows the cells).
    display = flat(pv_surface(pv, view, mesh))
    display.SetRepresentationType("Surface")
    pv_color_by(pv, view, display, "refine:level", "refinement level", integer=True)
    # Three levels: the caption names their colours (an integer bar of three values labels poorly).
    display.SetScalarBarVisibility(view, False)
    regional_box(pv, view, 24, 50, -125, -66)
    # The refined region spans about x in [-3.8, 4.3] Mm, y in [-2.2, 2.4] Mm.
    save_png(pv, view, path, {"position": [3.0e5, 0.0, 2.0e7], "focal": [3.0e5, 0.0, REGIONAL_RADIUS], **TOP,
                              "parallel_scale": 3.15e6})


@figure("regional-quality", ["regional"], "png")
def regional_quality(data, path):
    pv, view = pv_view()
    left = pv.LegacyVTKReader(FileNames=[str(data / "regional" / "refined.vtk")])
    right = pv.LegacyVTKReader(FileNames=[str(data / "regional" / "centroids.vtk")])
    # Each panel a map 900 km across about a transition, the second moved 1150 km to the right.
    half, shift = 4.5e5, 1.15e6
    centre = REGIONAL_ZOOM
    for k, source in enumerate((left, right)):
        # The cells about the centre (whole cells, near side only), then moved into their panel.
        window = pv.ExtractCellsByRegion(Input=source)
        window.IntersectWith = "Box"
        window.IntersectWith.Position = [centre[0] - half, centre[1] - half, 0.5 * REGIONAL_RADIUS]
        window.IntersectWith.Length = [2 * half, 2 * half, REGIONAL_RADIUS]
        window.Extractintersected = 1
        t = pv.Transform(Input=window)
        t.Transform.Translate = [-centre[0] + k * shift, -centre[1], 0.0]
        display = flat(pv_surface(pv, view, t))
        display.LineWidth = 1.0
        pv_color_by(pv, view, display, "min_angle", "least angle (°)")
    lut = pv.GetColorTransferFunction("min_angle")
    lut.RescaleTransferFunction(30.0, 90.0)
    lut.AutomaticRescaleRangeMode = "Never"
    labels(pv, view, "min_angle", [30, 45, 60, 75, 90], "{:.0f}")
    pv_label(pv, view, "refined", [0.07, 0.86])
    pv_label(pv, view, "20 sweeps of centroids", [0.45, 0.86])
    # Whole cells reach past each panel's 900 km: 100 km of margin, and the colour bar's sixth on the right.
    margin = 1.0e5
    width = (2 * half + shift + 2 * margin) / (5 / 6)
    focal = [-half - margin + width / 2, 0.0, REGIONAL_RADIUS]
    save_png(pv, view, path, {"position": [focal[0], 0.0, 2.0e7], "focal": focal, **TOP,
                              "parallel_scale": width / 1.4 / 2})


# The close-up's centre in set_regional's turned frame (m): south-west of the box, where the 33 km cells (level 2)
# meet the 100 km band (level 1, one cell wide there) and the 300 km cells.
REGIONAL_ZOOM = (-3.08e6, -1.0e6)


@figure("poisson-solution", ["poisson"], "png")
def poisson_solution(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "poisson" / "poisson.vtk")])
    # The cells behind the plane y = 1.6, and the layer it cuts: the half y >= 1.5 of the box, whole cells.
    half = pv.Clip(Input=mesh)
    half.ClipType = "Plane"
    half.ClipType.Origin = [1.5, 1.6, 1.5]
    half.ClipType.Normal = [0.0, -1.0, 0.0]
    half.Crinkleclip = 1
    display = pv_surface(pv, view, half)
    display.LineWidth = 1.0
    pv_color_by(pv, view, display, "u", "u", association="POINTS")
    pv.GetScalarBar(pv.GetColorTransferFunction("u"), view).RangeLabelFormat = "{:.3f}"
    save_png(pv, view, path, {"position": [-4.5, -9.0, 7.5], "focal": [1.7, 2.3, 1.4], "up": [0, 0, 1],
                              "parallel": 1, "parallel_scale": 3.0})


@figure("advected-bump", ["advect"], "png")
def advected_bump(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "advect" / "advect.vtk")])
    # The layer of cells under the bump's centre (z in [4/3, 1.5]), cut through its centres.
    layer = pv.Slice(Input=mesh)
    layer.SliceType = "Plane"
    layer.SliceType.Origin = [1.5, 1.5, 1.5 - 1.0 / 12.0]
    layer.SliceType.Normal = [0.0, 0.0, 1.0]
    layer.Triangulatetheslice = 0
    panels = [(layer, "start")]
    for shift, field in ((4.6, "upwind"), (9.2, "bounded")):
        moved = pv.Transform(Input=layer)
        moved.Transform.Translate = [shift, 0.0, 0.0]
        panels.append((moved, field))
    for source, field in panels:
        display = flat(pv_surface(pv, view, source))
        display.LineWidth = 0.5
        pv_color_by(pv, view, display, field, "u")
    # All on the scale of the start, where the bump peaks.
    layer.UpdatePipeline()
    lo, hi = layer.CellData["start"].GetRange()
    for _, field in panels:
        pv.GetColorTransferFunction(field).RescaleTransferFunction(0.0, hi)
    labels(pv, view, "start", [0.0, 0.2, 0.4, 0.6, 0.8], "{:.1f}")
    for source, field in panels[1:]:
        pv.HideScalarBarIfNotNeeded(pv.GetColorTransferFunction(field), view)
        pv.GetDisplayProperties(source, view).SetScalarBarVisibility(view, False)
    pv_label(pv, view, "start", [0.11, 0.8])
    pv_label(pv, view, "upwind, half a turn", [0.33, 0.8])
    pv_label(pv, view, "bounded, half a turn", [0.62, 0.8])
    # The squares span x in [-0.5, 12.7] and y in [-0.5, 3.5]; the view is 15.7 wide, room for the colour bar.
    save_png(pv, view, path, {"position": [7.3, 1.5, 20.0], "focal": [7.3, 1.5, 0.0], **TOP, "parallel_scale": 5.6})


@figure("remap-field", ["remap"], "png")
def remap_field(data, path):
    pv, view = pv_view()
    source = pv.LegacyVTKReader(FileNames=[str(data / "remap" / "source.vtk")])
    target = pv.LegacyVTKReader(FileNames=[str(data / "remap" / "target.vtk")])
    side_by_side(pv, view, source, target, 6.0, "u", "u")
    shared_range(pv, view, "u", [source, target], fmt="{:.2f}")
    labels(pv, view, "u", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], "{:.1f}")
    pv_label(pv, view, "source: 64 cells", [0.12, 0.8])
    pv_label(pv, view, "target: 1152 triangles", [0.47, 0.8])
    # The meshes span x in [-2.73, 8.74] and y in [-2.01, 2.74]; the view is 14 wide.
    save_png(pv, view, path, {"position": [3.9, 0.37, 30.0], "focal": [3.9, 0.37, 0.0], **TOP, "parallel_scale": 5.0})


@figure("remap-sphere", ["sphere"], "png")
def remap_sphere(data, path):
    pv, view = pv_view()
    source = pv.LegacyVTKReader(FileNames=[str(data / "sphere" / "sphere-source.vtk")])
    target = pv.LegacyVTKReader(FileNames=[str(data / "sphere" / "sphere-target.vtk")])
    moved = pv.Transform(Input=target)
    moved.Transform.Translate = [0.0, 2.3, 0.0]
    for s in (source, moved):
        display = pv_surface(pv, view, s)
        display.LineWidth = 1.0
        pv_color_by(pv, view, display, "u", "u")
    shared_range(pv, view, "u", [source, target], fmt="{:.2f}")
    labels(pv, view, "u", [1.0, 1.5, 2.0, 2.5, 3.0], "{:.1f}")
    pv_label(pv, view, "icosphere: 1280 triangles", [0.03, 0.84])
    pv_label(pv, view, "cubed sphere: 2400 quads", [0.45, 0.84])
    # Seen from +x, y to the right: the spheres span y in [-1, 3.3]; the view is 5.4 wide.
    save_png(pv, view, path, {"position": [12.0, 1.6, 4.0], "focal": [0.0, 1.6, 0.0], "up": [0, 0, 1],
                              "parallel": 1, "parallel_scale": 1.93})


@figure("marked-refinement", ["marked"], "png")
def marked_refinement(data, path):
    pv, view = pv_view()
    hanging = pv.LegacyVTKReader(FileNames=[str(data / "marked" / "hanging.vtk")])
    conforming = pv.LegacyVTKReader(FileNames=[str(data / "marked" / "conforming.vtk")])
    side_by_side(pv, view, hanging, conforming, 6.0, "level", "level", integer=True)
    points = load_json(data / "marked" / "hanging.json")
    dots = pv.ProgrammableSource()
    dots.OutputDataSetType = "vtkPolyData"
    dots.Script = (
        "import vtk\n"
        f"P = {points!r}\n"
        "pts = vtk.vtkPoints()\n"
        "cells = vtk.vtkCellArray()\n"
        "for i, (x, y) in enumerate(P):\n"
        "    pts.InsertNextPoint(x, y, 0.01)\n"
        "    cells.InsertNextCell(1)\n"
        "    cells.InsertCellPoint(i)\n"
        "out = self.GetPolyDataOutput()\n"
        "out.SetPoints(pts)\n"
        "out.SetVerts(cells)\n")
    display = pv.Show(dots, view)
    display.SetRepresentationType("Points")
    display.RenderPointsAsSpheres = 1
    display.PointSize = 13.0
    display.AmbientColor = display.DiffuseColor = [0.835, 0.369, 0.0]  # Okabe-Ito vermillion
    labels(pv, view, "level", [1, 2, 3], "{:.0f}")
    pv_label(pv, view, "with hanging vertices (dots)", [0.06, 0.8])
    pv_label(pv, view, "conforming", [0.56, 0.8])
    # The meshes span x in [-2.73, 8.73] and y in [-2, 2.73]; the view is 14 wide.
    save_png(pv, view, path, {"position": [3.9, 0.37, 30.0], "focal": [3.9, 0.37, 0.0], **TOP, "parallel_scale": 5.0})


def outline(ax, xy, **kwargs):
    ax.plot([p[0] for p in xy] + [xy[0][0]], [p[1] for p in xy] + [xy[0][1]], **kwargs)


@figure("remap-overlap", ["overlap"], "svg")
def remap_overlap(data, path):
    d = load_json(data / "overlap" / "overlap.json")
    cell = d["target"][d["cell"]]
    fig, (whole, zoom) = style.svg_figure(12, 5.4, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.0]})
    for ax in (whole, zoom):
        for c in d["source"]:
            outline(ax, c, color="#444444", lw=1.0, zorder=2)
        for c in d["target"]:
            outline(ax, c, color=style.OKABE_ITO[5], lw=0.8, ls=(0, (4, 3)), zorder=1)
    outline(whole, cell, color=style.OKABE_ITO[6], lw=2.5, zorder=4)
    # Not orange or vermillion, which the target cell's outline uses.
    colours = [style.OKABE_ITO[k] for k in (2, 3, 4, 5, 7)] + ["#bbbbbb"]
    for i, p in enumerate(d["pieces"]):
        colour = colours[i % len(colours)]
        zoom.fill([q[0] for q in p["piece"]], [q[1] for q in p["piece"]], fc=colour, ec="none", alpha=0.75,
                  zorder=0)
        m = centroid(p["piece"])
        small = p["area"] < 0.03 * d["area"]
        # A piece too small for its label gets it beside, with a leader.
        zoom.annotate(f"{p['area'] / d['area']:.3f}", m, xytext=(28, -22) if small else (0, 0),
                      textcoords="offset points", ha="center", va="center", fontsize=10, zorder=5,
                      arrowprops={"arrowstyle": "-", "lw": 0.8} if small else None,
                      bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    outline(zoom, cell, color=style.OKABE_ITO[6], lw=2.5, zorder=4)
    xs, ys = [q[0] for q in cell], [q[1] for q in cell]
    pad = 0.12 * max(max(xs) - min(xs), max(ys) - min(ys))
    zoom.set_xlim(min(xs) - pad, max(xs) + pad)
    zoom.set_ylim(min(ys) - pad, max(ys) + pad)
    whole.set_title("source cells (solid) under a 6 x 6 target grid (dashed)", fontsize=11, y=-0.1)
    zoom.set_title(f"target cell T and its {len(d['pieces'])} overlaps S ∩ T, labelled |S ∩ T| / |T|",
                   fontsize=11, y=-0.1)
    style.save_svg(fig, path)


def child_faces(child):
    """The faces of a child cell by its corner count: triangle, quad, tet or hex (canonical corner order)."""
    n = len(child)
    if n == 4 and len(set(round(p[2], 9) for p in child)) > 1:
        return [[0, 1, 3], [1, 2, 3], [0, 3, 2], [0, 2, 1]]
    if n == 8:
        return [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 3, 2, 1], [4, 5, 6, 7]]
    return [list(range(n))]


@figure("refinement-children", ["children"], "svg")
def refinement_children(data, path):
    cells = load_json(data / "children" / "children.json")
    order = ["tri", "quad", "tet", "hex"]
    fig, axes = style.svg_figure(13, 3.6, ncols=4)
    for ax, name in zip(axes, order):
        cell = cells[name]
        three = cell["dim"] == 3
        camera = TET_VIEW if name == "tet" else CELL_VIEW
        project = camera if three else (lambda q: (q[0], q[1]))
        mid = centroid(cell["parent"])
        # Each child pushed the same distance away from the parent's centre, so that all of them show.
        spread = {"tri": 0.12, "quad": 0.12, "tet": 0.75, "hex": 0.45}[name]
        drawn = []
        for i, child in enumerate(cell["children"]):
            c = centroid(child)
            r = max(sum((c[k] - mid[k]) ** 2 for k in range(3)) ** 0.5, 1e-9)
            shift = [spread * (c[k] - mid[k]) / r for k in range(3)]
            pts = [[q[k] + shift[k] for k in range(3)] for q in child]
            drawn.append((camera.depth(centroid(pts)) if three else 0.0, i, pts))
        for _, i, pts in sorted(drawn):
            colour = (style.CATEGORY + ["#999999"])[i]
            c = centroid(pts)
            faces = child_faces(pts) if three else [list(range(len(pts)))]
            for f in faces:
                fc = centroid([pts[v] for v in f])
                a, b, e = pts[f[0]], pts[f[1]], pts[f[-1]]
                u = [b[k] - a[k] for k in range(3)]
                w = [e[k] - a[k] for k in range(3)]
                normal = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
                if sum(normal[k] * (fc[k] - c[k]) for k in range(3)) < 0:
                    normal = [-x for x in normal]
                if three and camera.depth(normal) <= 0:
                    continue
                xy = [project(pts[v]) for v in f]
                ax.fill([p[0] for p in xy], [p[1] for p in xy], fc=colour, ec="#222222", lw=1.0)
        parent = [project(q) for q in cell["parent"]]
        if not three:
            outline(ax, parent, color="#222222", lw=1.0, ls=(0, (4, 3)))
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.06, f"{name}: {len(cell['children'])} children", fontsize=12,
                 ha="center")
        ax.autoscale()
        ax.margins(0.06)
    style.save_svg(fig, path)


@figure("grid-index-search", ["search"], "svg")
def grid_index_search(data, path):
    d = load_json(data / "search" / "search.json")
    g = d["grid"]
    fig, ax = style.svg_figure(8.5, 7.6)
    (x0, y0), (wx, wy), (nx, ny, _) = g["origin"], g["width"], g["n"]
    for i in range(nx + 1):
        ax.plot([x0 + i * wx] * 2, [y0, y0 + ny * wy], color="#bbbbbb", lw=0.8, ls=(0, (2, 2)), zorder=0)
    for j in range(ny + 1):
        ax.plot([x0, x0 + nx * wx], [y0 + j * wy] * 2, color="#bbbbbb", lw=0.8, ls=(0, (2, 2)), zorder=0)
    for c in d["cells"]:
        ax.fill([p[0] for p in c], [p[1] for p in c], fc="#e8e8e8", ec="white", lw=1.0, zorder=1)
    for b in d["boxes"]:
        outline(ax, [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])], color="#777777", lw=0.6, zorder=2)
    colours = [style.OKABE_ITO[6], style.OKABE_ITO[5]]
    for q, colour in zip(d["queries"], colours):
        lo_x, lo_y, hi_x, hi_y = q["box"]
        # The bins the query visits.
        i0, i1 = int((lo_x - x0) // wx), int((hi_x - x0) // wx)
        j0, j1 = int((lo_y - y0) // wy), int((hi_y - y0) // wy)
        ax.fill([x0 + i0 * wx, x0 + (i1 + 1) * wx, x0 + (i1 + 1) * wx, x0 + i0 * wx],
                [y0 + j0 * wy, y0 + j0 * wy, y0 + (j1 + 1) * wy, y0 + (j1 + 1) * wy], fc=colour, alpha=0.16,
                ec="none", zorder=2.5)
        for k in q["found"]:
            b = d["boxes"][k]
            outline(ax, [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])], color=colour, lw=1.4, zorder=3)
        outline(ax, [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)], color=colour, lw=3.0, zorder=4)
        ax.text(hi_x + 0.05, hi_y + 0.05, f"query: {len(q['found'])} boxes", color=colour, fontsize=11,
                ha="left", va="bottom", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_xlim(x0 - 0.1, x0 + nx * wx + 0.1)
    ax.set_ylim(y0 - 0.1, y0 + ny * wy + 0.1)
    ax.set_title(f"boxes around 64 cells, the index's {nx} x {ny} bins (dotted), two queries (bold)\n"
                 "and the boxes each meets; shaded: the bins a query visits", fontsize=11, y=-0.1)
    style.save_svg(fig, path)


def cells_of(box, dx, origin=(0.0, 0.0)):
    """The corners of a 2D box of cells [lo i, lo j, hi i, hi j] of width dx."""
    i0, j0, i1, j1 = box
    x0, y0 = origin[0] + i0 * dx, origin[1] + j0 * dx
    x1, y1 = origin[0] + (i1 + 1) * dx, origin[1] + (j1 + 1) * dx
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def grid_lines(ax, box, dx, **kwargs):
    """The cell lines inside a box of cells of width dx, as one path (NaN between the lines)."""
    i0, j0, i1, j1 = box
    xs, ys = [], []
    for i in range(i0, i1 + 2):
        xs += [i * dx, i * dx, float("nan")]
        ys += [j0 * dx, (j1 + 1) * dx, float("nan")]
    for j in range(j0, j1 + 2):
        xs += [i0 * dx, (i1 + 1) * dx, float("nan")]
        ys += [j * dx, j * dx, float("nan")]
    ax.plot(xs, ys, **kwargs)


LEVEL_COLOURS = ["#000000", style.OKABE_ITO[5], style.OKABE_ITO[3]]


@figure("amr-hierarchy", ["hierarchy"], "svg")
def amr_hierarchy(data, path):
    d = load_json(data / "hierarchy" / "hierarchy.json")
    n0 = d["domain"][2] + 1
    dx = [1.0 / (n0 << level) for level in range(3)]
    fig, axes = style.svg_figure(13, 4.9, ncols=3)
    titles = ["level 0: tagged cells, and the level-1 boxes", "level 1: tagged cells, and the level-2 boxes",
              "the hierarchy: three levels, ratio 2"]
    for k, ax in enumerate(axes):
        grid_lines(ax, d["domain"], dx[0], color="#dddddd", lw=0.5, zorder=0)
        if k < 2:
            if k == 1:
                for box in d["levels"][1]:
                    ax.fill(*zip(*cells_of(box, dx[1])), fc="white", ec=LEVEL_COLOURS[1], lw=0.6, zorder=1)
                    grid_lines(ax, box, dx[1], color="#cccccc", lw=0.4, zorder=1)
            for c in d["tags"][k]:
                ax.fill(*zip(*cells_of(c, dx[k])), fc=style.OKABE_ITO[1], ec="none", zorder=2)
            for box in d["levels"][k + 1]:
                ax.fill(*zip(*cells_of(box, dx[k + 1])), fc="none", ec=LEVEL_COLOURS[k + 1], lw=1.6, zorder=3)
        else:
            for level in (1, 2):
                for box in d["levels"][level]:
                    ax.fill(*zip(*cells_of(box, dx[level])), fc="white", ec="none", zorder=level)
                    grid_lines(ax, box, dx[level], color="#bbbbbb", lw=0.35, zorder=level)
                for box in d["levels"][level]:
                    ax.fill(*zip(*cells_of(box, dx[level])), fc="none", ec=LEVEL_COLOURS[level], lw=1.6,
                            zorder=3)
        ax.fill(*zip(*cells_of(d["domain"], dx[0])), fc="none", ec="#000000", lw=1.2, zorder=4)
        ax.set_title(titles[k], fontsize=11, y=-0.09)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
    style.save_svg(fig, path)


def inside(cell, box):
    return box[0] <= cell[0] <= box[2] and box[1] <= cell[1] <= box[3]


@figure("coarse-fine", ["coarsefine"], "svg")
def coarse_fine(data, path):
    d = load_json(data / "coarsefine" / "coarsefine.json")
    r, g = d["ratio"], d["ghosts"]
    h = 1.0 / r  # fine cells, in coarse cell widths
    fig, (fill, flux) = style.svg_figure(12, 5.6, ncols=2)
    filled = {}
    for op in d["filled"]:
        i0, j0, i1, j1 = op["region"]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                filled[(op["fine"], i, j)] = True
    for ax in (fill, flux):
        grid_lines(ax, d["domain"], 1.0, color="#bbbbbb", lw=0.7, zorder=0)
        for box in d["fine"]:
            ax.fill(*zip(*cells_of(box, h)), fc="white", ec="none", zorder=1)
            grid_lines(ax, box, h, color="#999999", lw=0.5, zorder=1)
    # The ghosts of the first fine box: from level 0 where the plan fills them, else from the other fine box.
    b = 0
    i0, j0, i1, j1 = d["fine"][b]
    for i in range(i0 - g, i1 + g + 1):
        for j in range(j0 - g, j1 + g + 1):
            if inside((i, j), d["fine"][b]):
                continue
            corners = cells_of([i, j, i, j], h)
            if (b, i, j) in filled:
                fill.fill(*zip(*corners), fc=style.OKABE_ITO[2], ec="#555555", lw=0.4, alpha=0.8, zorder=2)
            else:
                fill.fill(*zip(*corners), fc="white", ec=style.OKABE_ITO[3], lw=0.4, hatch="////", zorder=2)
    for k, box in enumerate(d["fine"]):
        fill.fill(*zip(*cells_of(box, h)), fc="none", ec=style.OKABE_ITO[5] if k == b else "#333333", lw=2.2,
                  zorder=3)
        flux.fill(*zip(*cells_of(box, h)), fc="none", ec="#333333", lw=1.2, zorder=3)
    fill.text(4.1, 12.7, "fine box", color=style.OKABE_ITO[5], fontsize=11, weight="bold")
    # The register's coarse faces, and the coarse cells outside that reflux corrects.
    for f in d["faces"]:
        i0, j0, i1, j1 = f["faces"]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if f["dir"] == 0:
                    seg = ([i, i], [j, j + 1])
                    cell = (i - 1, j) if f["side"] == 0 else (i, j)
                else:
                    seg = ([i, i + 1], [j, j])
                    cell = (i, j - 1) if f["side"] == 0 else (i, j)
                flux.fill(*zip(*cells_of([cell[0], cell[1], cell[0], cell[1]], 1.0)), fc=style.OKABE_ITO[4],
                          ec="none", alpha=0.6, zorder=1)
                flux.plot(*seg, color=style.OKABE_ITO[6], lw=3.2, solid_capstyle="butt", zorder=4)
    for ax in (fill, flux):
        ax.set_xlim(2.0, 14.0)
        ax.set_ylim(2.0, 14.0)
    fill.set_title(f"the ghosts ({g} fine cells) of one fine box: filled from level 0 (blue),\n"
                   "or by the other fine box (hatched)", fontsize=11, y=-0.13)
    flux.set_title("the flux register's coarse faces (red) on the fine level's outline,\n"
                   "and the coarse cells outside they correct (yellow)", fontsize=11, y=-0.13)
    style.save_svg(fig, path)


@figure("soma-read-parts", ["readparts"], "svg")
def soma_read_parts(data, path):
    d = load_json(data / "readparts" / "readparts.json")
    panels = [
        ("wrote4", "a file of N = 4 parts", "part"),
        ("read2", "read on P = 2 ranks: parts 0, 1 on rank 0,\nparts 2, 3 on rank 1", "rank"),
        None,
        ("wrote2", "a file of N = 2 parts", "part"),
        ("read4", "read on P = 4 ranks, rebalance off:\npart q on rank 2q; ranks 1, 3 empty", "rank"),
        ("rebalanced", "read on P = 4 ranks, rebalance on\n(the default): every rank gets cells", "rank"),
    ]
    fig, axes = style.svg_figure(13, 8.6, nrows=2, ncols=3)
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            continue
        key, title, what = panel
        owner = d[key]
        for c, o in zip(d["cells"], owner):
            ax.fill([p[0] for p in c], [p[1] for p in c], fc=style.CATEGORY[o], ec="#333333", lw=0.6)
        used = sorted(set(owner))
        # The legend in the empty corner right of the lower arm.
        for k, o in enumerate(used):
            ax.add_patch(matplotlib_rect((1.35, -0.5 - 0.42 * k), 0.3, 0.3, style.CATEGORY[o]))
            ax.text(1.8, -0.35 - 0.42 * k, f"{what} {o}", va="center", fontsize=10)
        ax.set_title(title, fontsize=11, y=-0.2)
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.1, 2.8)
    fig.text(0.07, 0.75, "N > P", fontsize=13, weight="bold", ha="center")
    fig.text(0.07, 0.3, "N < P", fontsize=13, weight="bold", ha="center")
    style.save_svg(fig, path)


def matplotlib_rect(corner, w, h, colour):
    return Rectangle(corner, w, h, fc=colour, ec="#333333", lw=0.6)
