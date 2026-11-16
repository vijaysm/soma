# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""The figures of the parallel pages (docs/parallel)."""

import matplotlib.patches as mpatches

import style
from figlib import OWNERSHIP, centroid, figure, load_json, pv_save, pv_surface, pv_view

# The rank figures 13 and 15 show: of the four, the one holding every kind of vertex.
SHOWN_RANK = 2


def load_parts(data):
    return [load_json(data / "parts" / f"parts-{r}.json") for r in range(4)]


def draw_whole_mesh(ax, parts):
    """Every rank's owned cells, faint: the mesh a part belongs to."""
    for part in parts:
        for cell in part["cells"]:
            if cell["layer"] == 0:
                ax.fill(*zip(*cell["xy"]), fc="#ffffff", ec="#cccccc", lw=0.6, zorder=0)


def finish(ax):
    ax.autoscale()
    ax.margins(0.03)


@figure("partition-4-ranks", ["parts"], "png")
def partition_4_ranks(data, path):
    pv, view = pv_view()
    mesh = pv.LegacyVTKReader(FileNames=[str(data / "parts" / "parts.vtk")])
    display = pv_surface(pv, view, mesh)
    display.Ambient, display.Diffuse = 1.0, 0.0
    display.LineWidth = 1.5
    pv.ColorBy(display, ("CELLS", "rank"))
    lut = pv.GetColorTransferFunction("rank")
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = [v for r in range(4) for v in (str(r), f"rank {r}")]
    lut.IndexedColors = [c for r in range(4) for c in matplotlib_rgb(style.CATEGORY[r])]
    display.SetScalarBarVisibility(view, True)
    bar = pv.GetScalarBar(lut, view)
    bar.Title = "owner"
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0.0, 0.0, 0.0]
    bar.TitleFontSize = bar.LabelFontSize = 28
    bar.WindowLocation = "Any Location"
    bar.Position = [0.84, 0.3]
    # The mesh spans x in [-2.73, 2.73] and y in [-2, 2.73].
    pv_save(pv, view, path, {"position": [0.45, 0.37, 30.0], "focal": [0.45, 0.37, 0.0], "up": [0, 1, 0],
                             "parallel": 1, "parallel_scale": 2.75})


def matplotlib_rgb(colour):
    import matplotlib.colors

    return list(matplotlib.colors.to_rgb(colour))


@figure("rank-part", ["parts"], "svg")
def rank_part(data, path):
    parts = load_parts(data)
    part = parts[SHOWN_RANK]
    fig, ax = style.svg_figure(8, 7)
    draw_whole_mesh(ax, parts)
    for cell in part["cells"]:
        if cell["layer"] <= 1:
            kind = "interior" if cell["layer"] == 0 else "ghost"
            ax.fill(*zip(*cell["xy"]), fc=OWNERSHIP[kind][0], alpha=0.35, ec="#555555", lw=0.8, zorder=1)
    for v in part["vertices"]:
        ax.plot(*v["xy"], "o", ms=6.5, mfc=OWNERSHIP[v["kind"]][0], mec="#222222", mew=0.8, zorder=3)
    handles = [mpatches.Patch(fc=OWNERSHIP["interior"][0], alpha=0.35, ec="#555555", label="owned cell"),
               mpatches.Patch(fc=OWNERSHIP["ghost"][0], alpha=0.35, ec="#555555", label="ghost cell")]
    handles += [ax.plot([], [], "o", ms=7, mfc=OWNERSHIP[k][0], mec="#222222", label=f"{OWNERSHIP[k][1]} vertex")[0]
                for k in ("interior", "shared", "shared_copy", "ghost")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10)
    finish(ax)
    style.save_svg(fig, path)


@figure("sfc-curve", ["curve"], "svg")
def sfc_curve(data, path):
    import matplotlib.cm

    centroids = load_json(data / "curve" / "curve.json")["centroids"]
    fig, ax = style.svg_figure(6.5, 6.5)
    for i in range(9):
        ax.plot([i, i], [0, 8], color="#bbbbbb", lw=0.8, zorder=0)
        ax.plot([0, 8], [i, i], color="#bbbbbb", lw=0.8, zorder=0)
    cmap = matplotlib.colormaps[style.FIELD_CMAP]
    n = len(centroids)
    for k in range(n - 1):
        (x0, y0), (x1, y1) = centroids[k], centroids[k + 1]
        ax.plot([x0, x1], [y0, y1], color=cmap(k / (n - 1)), lw=2.6, solid_capstyle="round", zorder=1)
    for k, (x, y) in enumerate(centroids):
        # Each label sits in its cell's lower-left corner, clear of the curve through the centre.
        ax.text(x - 0.44, y - 0.44, str(k), ha="left", va="bottom", fontsize=7.5, color="#444444", zorder=2)
    ax.plot(*centroids[0], "o", color=cmap(0.0), ms=8, zorder=3)
    ax.plot(*centroids[-1], "s", color=cmap(1.0), ms=8, zorder=3)
    ax.set_xlim(-0.1, 8.1)
    ax.set_ylim(-0.1, 8.1)
    style.save_svg(fig, path)


@figure("ghost-layers", ["parts"], "svg")
def ghost_layers(data, path):
    parts = load_parts(data)
    part = parts[SHOWN_RANK]
    fig, ax = style.svg_figure(8, 7)
    draw_whole_mesh(ax, parts)
    fills = {0: ("#dddddd", "owned"), 1: (style.CATEGORY[0], "ghost layer 1"), 2: (style.CATEGORY[1], "ghost layer 2")}
    for cell in part["cells"]:
        ax.fill(*zip(*cell["xy"]), fc=fills[cell["layer"]][0], alpha=0.9 if cell["layer"] else 1.0, ec="#555555",
                lw=0.8, zorder=1)
    handles = [mpatches.Patch(fc=c, ec="#555555", label=label) for c, label in fills.values()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10)
    finish(ax)
    style.save_svg(fig, path)


@figure("merge-meshes", ["merge"], "svg")
def merge_meshes(data, path):
    fig, axes = style.svg_figure(12, 4.6, ncols=2)
    for ax, state in zip(axes, ("before", "after")):
        for r in range(2):
            snap = load_json(data / "merge" / f"merge-{r}-{state}.json")
            # Before the merge, the two meshes are drawn pulled apart, so both copies of x = 3 show.
            dx = 0.35 * r if state == "before" else 0.0
            for cell in snap["cells"]:
                xy = [(p[0] + dx, p[1]) for p in cell["xy"]]
                ax.fill(*zip(*xy), fc=style.CATEGORY[r], alpha=0.25, ec="#444444", lw=1.0, zorder=1)
            for v in snap["vertices"]:
                x, y = v["xy"][0] + dx, v["xy"][1]
                on_line = abs(v["xy"][0] - 3) < 1e-9
                colour = OWNERSHIP[v["kind"]][0]
                # After the merge both ranks hold each vertex on x = 3: rank 0's kind fills the left
                # half of its marker, rank 1's the right half.
                fill = ("left" if r == 0 else "right") if state == "after" and on_line else "full"
                ax.plot(x, y, "o", ms=11 if on_line else 4, mfc=colour, mec="#222222", mew=0.8, fillstyle=fill,
                        mfcalt="none", zorder=3)
                if on_line:
                    side = -1 if r == 0 else 1
                    if state == "after" and r == 1:
                        continue  # the same vertex as rank 0's, labelled once
                    ax.text(x + side * 0.12 if state == "before" else x + 0.12, y + 0.14, str(v["gid"]),
                            ha="right" if side < 0 and state == "before" else "left", fontsize=10, zorder=4,
                            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85})
        ax.set_title(f"{state} merge_mesh: " + ("each rank's own vertices on x = 3" if state == "before"
                                                 else "one vertex each, held on both ranks"), fontsize=11, y=-0.16)
        finish(ax)
        ax.margins(0.08)
    handles = [mpatches.Patch(fc=style.CATEGORY[r], alpha=0.25, ec="#444444", label=f"rank {r}'s cells")
               for r in range(2)]
    handles += [axes[1].plot([], [], "o", ms=7, mfc=OWNERSHIP[k][0], mec="#222222",
                             label=f"{OWNERSHIP[k][1]} vertex")[0] for k in ("interior", "shared", "shared_copy")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    style.save_svg(fig, path)
