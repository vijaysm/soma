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


# ---------------------------------------------------------------- figures


@figure("box-mesh-gids", ["box"], "png")
def box_mesh_gids(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "box" / "box.vtk")])
    display = pv_surface(pv, view, mesh)
    pv_color_by(pv, view, display, "gid", "cell gid", integer=True)
    pv_save(pv, view, path, {"position": [10.5, -8.0, 8.5], "focal": [1.9, 1.5, 0.9], "up": [0, 0, 1]})


# ---------------------------------------------------------------- schematics in 3D

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


def reference_cells_json(data):
    """reference.json, with the reference elements ([-1, 1] wide) halved to unit size for drawing."""
    cells = load_json(data / "reference" / "reference.json")
    for cell in cells.values():
        cell["points"] = [[c / 2 for c in p] for p in cell["points"]]
    return cells


def centroid(points):
    return [sum(p[k] for p in points) / len(points) for k in range(len(points[0]))]


@figure("reference-cells", ["reference"], "svg")
def reference_cells(data, path):
    cells = reference_cells_json(data)
    order = ["tri", "quad", "tet", "pyramid", "prism", "hex"]
    fig, axes = style.svg_figure(10.5, 6.6, nrows=2, ncols=3)
    for ax, name in zip(axes.flat, order):
        cell = cells[name]
        x = cell["points"]
        p = [CELL_VIEW(q) for q in x] if cell["dim"] == 3 else [(q[0], q[1]) for q in x]
        mid = centroid(p)
        faces = cell["faces"]
        for i, (a, b) in enumerate(cell["edges"]):
            visible = cell["dim"] == 2 or any(a in f and b in f and in_front(x, f) for f in faces)
            ax.plot([p[a][0], p[b][0]], [p[a][1], p[b][1]], color="#222222", lw=1.8,
                    ls="-" if visible else (0, (4, 3)), zorder=1)
            m = centroid([p[a], p[b]])
            ax.text(m[0], m[1], f"{i}", color="#777777", fontsize=8.5, ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"}, zorder=2)
        for i, q in enumerate(p):
            d = [q[0] - mid[0], q[1] - mid[1]]
            r = max((d[0] ** 2 + d[1] ** 2) ** 0.5, 1e-9)
            ax.plot(*q, "o", color=style.OKABE_ITO[5], ms=6, zorder=3)
            ax.text(q[0] + 0.15 * d[0] / r, q[1] + 0.15 * d[1] / r, str(i), fontsize=12, weight="bold",
                    ha="center", va="center", color=style.OKABE_ITO[5], zorder=4,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"})
        ax.set_title(name, fontsize=13, y=-0.12)
        ax.autoscale()
        ax.margins(0.12)
    style.save_svg(fig, path)


@figure("canonical-sides", ["reference"], "svg")
def canonical_sides(data, path):
    cells = reference_cells_json(data)
    fig, axes = style.svg_figure(11, 5.4, ncols=2)
    # Each cell from a direction that shows none of its faces edge-on.
    views = {"hex": Camera((0.55, -1.0, 0.5)), "tet": Camera((1.0, -0.75, 0.55))}
    for ax, name in zip(axes.flat, ["hex", "tet"]):
        x = cells[name]["points"]
        camera = views[name]
        faces = list(enumerate(cells[name]["faces"]))
        # Far faces first, so near ones are drawn over them.
        faces.sort(key=lambda f: camera.depth(centroid([x[v] for v in f[1]])))
        for i, face in faces:
            n = face_normal(x, face)
            length = sum(c * c for c in n) ** 0.5
            fc = centroid([x[v] for v in face])
            shift = [0.8 * n[k] / length for k in range(3)]
            # Each face pulled out along its normal and shrunk towards its centre.
            corners3 = [[fc[k] + shift[k] + 0.72 * (x[v][k] - fc[k]) for k in range(3)] for v in face]
            corners = [camera(c) for c in corners3]
            front = in_front(x, face, camera)
            colour = style.CATEGORY[i % len(style.CATEGORY)]
            ax.fill(*zip(*corners), color=colour, alpha=0.35 if front else 0.15, zorder=1 if not front else 3)
            ax.plot(*zip(*(corners + corners[:1])), color=colour, lw=1.6, zorder=2 if not front else 4,
                    ls="-" if front else (0, (4, 3)))
            centre = camera([fc[k] + shift[k] for k in range(3)])
            for v, c in zip(face, corners):
                inset = [c[k] + 0.22 * (centre[k] - c[k]) for k in range(2)]
                ax.text(*inset, str(v), fontsize=10, ha="center", va="center", zorder=6,
                        color="#000000" if front else "#666666")
            # The face's sense: an arrow from its first vertex towards its second.
            a = [corners[0][k] + 0.42 * (centre[k] - corners[0][k]) for k in range(2)]
            b = [corners[1][k] + 0.42 * (centre[k] - corners[1][k]) for k in range(2)]
            ax.annotate("", xy=[a[k] + 0.75 * (b[k] - a[k]) for k in range(2)], xytext=a, zorder=6,
                        arrowprops={"arrowstyle": "-|>", "color": "#000000" if front else "#888888", "lw": 1.3})
            ax.text(*centre, f"f{i}", fontsize=12, weight="bold", ha="center", va="center", zorder=7,
                    color="#000000" if front else "#666666")
        ax.set_title(f"{name}: faces pulled out along their outward normals", fontsize=12, y=-0.1)
        ax.autoscale()
        ax.margins(0.04)
    style.save_svg(fig, path)


# ---------------------------------------------------------------- tables and graphs

OWNERSHIP = {  # kind: (colour, label)
    "interior": ("#56B4E9", "interior"),
    "shared": ("#E69F00", "shared, owned here"),
    "shared_copy": ("#F0E442", "shared copy"),
    "ghost": ("#CC79A7", "ghost"),
}


@figure("segment-table", ["segments"], "svg")
def segment_table(data, path):
    import matplotlib.patches as mpatches

    tables = [load_json(p) for p in sorted((data / "segments").glob("segments-*.json"))]
    bars = [(t["rank"], d) for t in tables for d in t["dims"]]
    fig, ax = style.svg_figure(11, 1.25 * len(bars) + 0.9)
    ax.set_aspect("auto")
    width = 10.0
    for row, (rank, d) in enumerate(bars):
        y = -1.25 * row
        scale = width / d["count"]
        kind = "vertices" if d["dim"] == 0 else ("cells" if d["dim"] == max(e["dim"] for e in tables[0]["dims"])
                                                 else f"dimension {d['dim']}")
        ax.text(-0.2, y + 0.3, f"rank {rank}\n{kind}", ha="right", va="center", fontsize=11)
        for g in d["segments"]:
            x0, x1 = g["begin"] * scale, g["end"] * scale
            ax.add_patch(mpatches.Rectangle((x0, y), x1 - x0, 0.6, fc=OWNERSHIP[g["ownership"]][0], ec="#333333",
                                            lw=1.0))
            ax.text((x0 + x1) / 2, y + 0.3, f"{g['topology']}\n{g['end'] - g['begin']}", ha="center", va="center",
                    fontsize=9)
        # The lids where the ranges end, and the names of the table's ends there.
        ends = {}
        for name in ("interior_end", "owned_end", "shared_end"):
            ends.setdefault(d[name], []).append(name)
        cuts = sorted({0, d["count"], *ends, *(g["end"] for g in d["segments"])})
        for lid in cuts:
            ax.text(lid * scale, y - 0.06, str(lid), ha="center", va="top", fontsize=8.5, color="#444444")
        for lid, names in ends.items():
            if 0 < lid < d["count"]:
                ax.plot([lid * scale] * 2, [y + 0.6, y + 0.78], color="#000000", lw=1.2)
                ax.text(lid * scale, y + 0.8, " = ".join(names), ha="center", va="bottom", fontsize=8.5)
        ax.text(width + 0.15, y + 0.3, f"count {d['count']}", ha="left", va="center", fontsize=9.5,
                color="#444444")
    handles = [mpatches.Patch(fc=c, ec="#333333", label=label) for c, label in OWNERSHIP.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=10)
    ax.set_xlim(-1.6, width + 1.4)
    ax.set_ylim(-1.25 * (len(bars) - 1) - 0.5, 1.15)
    style.save_svg(fig, path)


def draw_quads(ax, g, cell_fill=None):
    """The mesh of graphs.json: cells outlined and numbered, vertices numbered."""
    pts = g["points"]
    for c, cell in enumerate(g["cells"]):
        xy = [pts[v] for v in cell]
        ax.fill(*zip(*xy), color=(cell_fill or {}).get(c, "#ffffff"), ec="#333333", lw=1.4, zorder=1)
        m = centroid(xy)
        ax.text(*m, f"c{c}", ha="center", va="center", fontsize=11, weight="bold", zorder=3)
    for v, p in enumerate(pts):
        ax.plot(*p, "o", color="#333333", ms=4, zorder=2)
        ax.text(p[0] + 0.07, p[1] + 0.09, f"v{v}", fontsize=9, color="#333333", zorder=3,
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.85})


@figure("graph-kinds", ["graphs"], "svg")
def graph_kinds(data, path):
    g = load_json(data / "graphs" / "graphs.json")
    fig, axes = style.svg_figure(11, 4.2, ncols=2)
    vertex, cell = 5, 1
    row = g["vertex_cells"][vertex]
    draw_quads(axes[0], g, {c: style.OKABE_ITO[1] for c in row})
    p = g["points"][vertex]
    axes[0].plot(*p, "o", color=style.OKABE_ITO[6], ms=11, zorder=4)
    axes[0].set_title(f"upward (0, 2): row of v{vertex} = cells {' '.join(map(str, row))}", fontsize=11.5, y=-0.16)
    nbrs = g["cell_neighbours"][cell]
    draw_quads(axes[1], g, {cell: style.OKABE_ITO[6], **{c: style.OKABE_ITO[2] for c in nbrs}})
    axes[1].set_title(f"bridge (2, 1): row of c{cell} = cells {' '.join(map(str, nbrs))}", fontsize=11.5, y=-0.16)
    for ax in axes:
        ax.autoscale()
        ax.margins(0.06)
    style.save_svg(fig, path)


# ---------------------------------------------------------------- fields and edits


def pv_label(pv, view, text, position):
    """A caption in the view at a normalised position (lower left is 0, 0)."""
    label = pv.Text(Text=text)
    display = pv.Show(label, view)
    display.WindowLocation = "Any Location"
    display.Position = position
    display.FontSize = 30
    display.Color = [0.0, 0.0, 0.0]
    return display


@figure("cell-and-vertex-fields", ["fields"], "png")
def cell_and_vertex_fields(data, path):
    pv, view = pv_view()
    cells = pv.LegacyVTKReader(FileNames=[str(data / "fields" / "fields-cell.vtk")])
    vertices = pv.LegacyVTKReader(FileNames=[str(data / "fields" / "fields-vertex.vtk")])
    shifted = pv.Transform(Input=vertices)
    shifted.Transform.Translate = [6.0, 0.0, 0.0]
    for source, association in ((cells, "CELLS"), (shifted, "POINTS")):
        display = pv_surface(pv, view, source)
        display.LineWidth = 2.0
        display.EdgeColor = [0.15, 0.15, 0.15]
        display.Ambient, display.Diffuse = 1.0, 0.0  # flat: a plane mesh needs no shading
        pv_color_by(pv, view, display, "u", "u = x + y", association=association)
    # One colour scale for both: the range of the vertex field, which holds the extremes of x + y.
    lut = pv.GetColorTransferFunction("u")
    vertices.UpdatePipeline()
    lut.RescaleTransferFunction(*vertices.PointData["u"].GetRange())
    bar = pv.GetScalarBar(lut, view)
    bar.RangeLabelFormat = "{:.1f}"
    pv_label(pv, view, "a cell field: one value per cell", [0.1, 0.84])
    pv_label(pv, view, "a vertex field, interpolated in each cell", [0.46, 0.84])
    # The copies span x in [-2.73, 8.73] and y in [-2, 2.73]; the view is 14.9 wide (1.4 times
    # twice the parallel scale), leaving the right sixth for the colour bar.
    pv_save(pv, view, path, {"position": [3.98, 0.6, 30.0], "focal": [3.98, 0.6, 0.0], "up": [0, 1, 0],
                             "parallel": 1, "parallel_scale": 5.3})


@figure("mesh-edit", ["edit"], "svg")
def mesh_edit(data, path):
    states = [("before: a 4 x 3 box of quads,\nlid in bold, gid below", "edit-before.json"),
              ("after apply(): removed cells keep their lids;\nthe added quad is lid 12, in the tail",
               "edit-applied.json"),
              ("after compress(): lids renumbered in gid order;\ngids kept", "edit-compressed.json")]
    fig, axes = style.svg_figure(13, 3.9, ncols=3)
    for ax, (title, name) in zip(axes, states):
        snap = load_json(data / "edit" / name)
        pts = snap["points"]
        first_tail = len(snap["cells"]) - snap["tail"]
        for cell in snap["cells"]:
            xy = [pts[v] for v in cell["corners"]]
            added = snap["edited"] and cell["lid"] >= first_tail
            if not cell["alive"]:
                ax.fill(*zip(*xy), fc="#eeeeee", ec="#999999", lw=1.0, hatch="///", ls=(0, (4, 3)), zorder=1)
            else:
                ax.fill(*zip(*xy), fc=style.OKABE_ITO[1] if added else "#ffffff", ec="#333333", lw=1.3, zorder=1)
            m = centroid(xy)
            colour = "#888888" if not cell["alive"] else "#000000"
            ax.text(m[0], m[1] + 0.12, str(cell["lid"]), ha="center", va="center", fontsize=12, weight="bold",
                    color=colour, zorder=3)
            ax.text(m[0], m[1] - 0.2, f"gid {cell['gid']}", ha="center", va="center", fontsize=8, color=colour,
                    zorder=3)
        ax.set_title(title, fontsize=11, y=-0.3)
        ax.set_xlim(-0.15, 5.15)
        ax.set_ylim(-0.15, 3.15)
    style.save_svg(fig, path)


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
