# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""The figures of the concept pages (docs/concepts and the box mesh)."""

import style
from figlib import (CELL_VIEW, OWNERSHIP, Camera, centroid, face_normal, figure, in_front, load_json, pv_color_by,
                    pv_label, pv_save, pv_surface, pv_view)


@figure("box-mesh-gids", ["box"], "png")
def box_mesh_gids(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "box" / "box.vtk")])
    display = pv_surface(pv, view, mesh)
    pv_color_by(pv, view, display, "gid", "cell gid", integer=True)
    pv_save(pv, view, path, {"position": [10.5, -8.0, 8.5], "focal": [1.9, 1.5, 0.9], "up": [0, 0, 1]})


def reference_cells_json(data):
    """reference.json, with the reference elements ([-1, 1] wide) halved to unit size for drawing."""
    cells = load_json(data / "reference" / "reference.json")
    for cell in cells.values():
        cell["points"] = [[c / 2 for c in p] for p in cell["points"]]
    return cells


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
