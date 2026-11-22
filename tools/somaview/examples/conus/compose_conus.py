# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Compose the CONUS figures from the renders of conus_job.sh and conus_compare.py.

ParaView draws the mesh only; this script adds the titles, colour bars, panel letters, window
outlines, scale bars and keys with matplotlib, and cuts the results to the documentation's
limits (256 colours, under 300 KB):

    python compose_conus.py stereo OUTDIR FIGDIR            # conus_stereo.py's four renders
    python compose_conus.py telescope OUTDIR FIGDIR NAME    # conus_telescope.py's six panels
    python compose_conus.py compare CMPDIR FIGDIR           # conus_compare.py's renders of p3 and p4
    python compose_conus.py docs FIGDIR IMGDIR              # docs/figures/img/conus-*.png

OUTDIR is conus_job.sh's output (prepass.json, levels.json, img/); NAME labels the mesh in the
telescope's title. Scale bars are ground distances: a stereographic projection centred at c
enlarges lengths at a point u by k = 2 / (1 + u.c), so a projected half-height h is h / k on the
ground. Needs numpy, matplotlib and Pillow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.patches import Rectangle

mpl.use("Agg")
STYLE = {"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
         "patch.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
         "axes.titlelocation": "left", "legend.frameon": False, "xtick.major.size": 3,
         "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200}
CYAN = (0.0, 0.85, 1.0)  # cells below a colour bar's range
LEVEL_RGB = [[0.267, 0.005, 0.329], [0.255, 0.267, 0.529], [0.165, 0.471, 0.557],
             [0.133, 0.659, 0.518], [0.478, 0.820, 0.318], [0.992, 0.906, 0.145]]


def scale_bar(ax, x0, y0, px, lw=(6, 2.5)):
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=lw[0], solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=lw[1], solid_capstyle="butt")
    return ax


# -- stereo: the whole region by min_angle and scaled_jacobian, and two close-ups -------------

ZOOM_TARGET = {"conus_stereo_transition_zoom.png": "fine_edge", "conus_stereo_worst_cell_zoom.png": "worst_min_angle"}


def stereo(outdir: Path, figdir: Path) -> None:
    meta = json.load(open(outdir / "img/meta.json"))
    P = json.load(open(outdir / "prepass.json"))
    S, ncell, C = P["stats"], P["cells"], np.array(P["center"]["unit"])
    titles = {
        "conus_stereo_min_angle.png": f"Minimum angle: median {S['min_angle']['p50']:.1f} deg; 99.9% of the "
                                      f"{ncell / 1e6:.1f}M cells are at or above 73 deg",
        "conus_stereo_scaled_jacobian.png": f"Scaled Jacobian: median {S['scaled_jacobian']['p50']:.3f}; lowest "
                                            f"{S['scaled_jacobian']['min']:.2f}, none below 0.5",
        "conus_stereo_transition_zoom.png": "Refinement transitions at the edge of the 200 m region: the poorest "
                                            "cells are the transition templates",
        "conus_stereo_worst_cell_zoom.png": f"The worst cell ({S['min_angle']['min']:.1f} deg) lies in the coarse "
                                            "transition bands offshore, not in the 200 m region",
    }
    below = {"conus_stereo_min_angle.png": f"below 72 deg: {S['min_angle']['count_below_72']:,} cells",
             "conus_stereo_scaled_jacobian.png": "below 0.95: "
                                                 f"{S['scaled_jacobian'].get('count_below_0.95', 0):,} cells",
             "conus_stereo_transition_zoom.png": "below 72 deg", "conus_stereo_worst_cell_zoom.png": None}
    for fname, m in meta.items():
        img = imread(outdir / "img" / fname)
        h, w = img.shape[:2]
        figw = 12.0
        figh = figw * h / w * 0.93 + 0.8
        fig = plt.figure(figsize=(figw, figh))
        ax = fig.add_axes([0.0, 0.0, 0.9, (figh - 0.75) / figh])
        ax.imshow(img)
        ax.set_axis_off()
        cax = fig.add_axes([0.915, 0.12, 0.018, 0.62])
        cmap = mpl.colormaps[m["cmap"]].copy()
        cmap.set_under(CYAN)
        under = below[fname] is not None
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(m["lo"], m["hi"]), cmap), cax=cax,
                          extend="min" if under else "neither")
        cb.set_label(m["title"])
        if under:
            fig.text(0.905, 0.085, "cyan: " + below[fname], fontsize=8.5, ha="left", va="top",
                     wrap="half_height_km" not in m)
        fig.text(0.01, 1 - 0.18 / figh, titles[fname], fontsize=11, ha="left", va="top")
        if m.get("marker"):
            ax.plot([w / 2], [h / 2], marker="o", ms=26, mfc="none", mec="white", mew=2.5)
            ax.plot([w / 2], [h / 2], marker="o", ms=26, mfc="none", mec="black", mew=1.0)
        if "half_height_km" in m:
            k = 2.0 / (1.0 + np.array(P[ZOOM_TARGET[fname]]["unit"]) @ C)
            half_km = m["half_height_km"] / k
            L = 20 if half_km > 60 else 5
            px = L / (2 * half_km / h)
            x0, y0 = w * 0.04, h * 0.95
            scale_bar(ax, x0, y0, px)
            ax.text(x0 + px / 2, y0 - h * 0.02, f"{L} km", ha="center", va="bottom", fontsize=9.5,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        fig.savefig(figdir / fname, dpi=200)
        plt.close(fig)
        print("composed", fname)


# -- telescope: six windows, each 4 times smaller than the last -------------------------------

def nice_len(km):
    for L in [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2]:
        if L <= km:
            return L
    return 0.1


def telescope(outdir: Path, figdir: Path, name: str) -> None:
    m = json.load(open(outdir / "img/telescope_meta.json"))
    lev = json.load(open(outdir / "levels.json"))
    W, Hh = m["size"]
    A = W / Hh
    order = "abcdef"
    fig = plt.figure(figsize=(15, 10.4))
    gs = fig.add_gridspec(2, 3, left=0.01, right=0.99, top=0.885, bottom=0.085, wspace=0.03, hspace=0.13)
    titles = {"a": "whole refined region", "f": "200 m cells meet the transition templates"}
    for i, L in enumerate(order):
        p = m["panels"][L]
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(imread(outdir / "img" / p["file"]))
        ax.set_xlim(0, W)
        ax.set_ylim(Hh, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.3")
            s.set_linewidth(0.8)
        height = 2 * p["half_km"]
        hlabel = f"{height:,.0f} km" if height >= 100 else (f"{height:.0f} km" if height >= 10 else f"{height:.1f} km")
        ax.set_title(f"window height {hlabel}" + (f": {titles[L]}" if L in titles else ""), loc="left",
                     fontsize=9.5, pad=4)
        ax.text(-0.005, 1.012, L, transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontweight="bold")
        if i + 1 < len(order):  # outline of the next window
            q = m["panels"][order[i + 1]]
            cx, cy, h = p["cx"], p["cy"], p["half"]

            def px(X, Y):
                return ((X - (cx - h * A)) / (2 * h * A) * W, ((cy + h) - Y) / (2 * h) * Hh)

            x0, y0 = px(q["cx"] - q["half"] * A, q["cy"] + q["half"])
            x1, y1 = px(q["cx"] + q["half"] * A, q["cy"] - q["half"])
            for col, lw in (("white", 3.5), ("black", 1.4)):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=lw))
            ax.text(x1 + 12, y0, order[i + 1], ha="left", va="top", fontsize=11, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        # ground scale bar: panel a at the projection centre, the others at the window centre
        km_per_px = height / Hh
        Lkm = nice_len(0.25 * W * km_per_px)
        bar = Lkm / km_per_px
        bx, by = (0.95 * W - bar, 0.93 * Hh) if L == "a" else (0.05 * W, 0.93 * Hh)
        scale_bar(ax, bx, by, bar, lw=(6, 2.4))
        ax.text(bx + bar / 2, by - 0.025 * Hh, f"{Lkm:g} km" + (" (at centre)" if L == "a" else ""), ha="center",
                va="bottom", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    tgt, c = m["target"]["lonlat"], m["center"]
    ncells = sum(r["cells"] for r in lev.values())
    ns = "S" if c["lat"] < 0 else "N"
    ew = "W" if c["lon"] < 0 else "E"
    fig.text(0.01, 0.985, f"Telescoping zoom on {name} ({ncells / 1e6:.1f}M quadrilaterals): from the whole refined "
             f"region to the corner of the 200 m region at {tgt[1]:.2f} N, {-tgt[0]:.2f} W", ha="left", va="top",
             fontsize=11)
    fig.text(0.01, 0.955, f"Stereographic projection centred at {abs(c['lat']):.1f} {ns}, {abs(c['lon']):.1f} {ew}; "
             "each window is 4 times smaller than the last and is outlined in the panel before it; cells coloured "
             "by refinement level, edges drawn in d to f", ha="left", va="top", fontsize=9.5)
    kx = 0.115
    fig.text(0.01, 0.045, "refinement level\n(mean cell size)", ha="left", va="center", fontsize=9.5)
    for l in range(len(lev)):
        r = lev[str(l)]
        mean = r["spacing_km_mean"]
        size = f"{mean:.0f} km" if mean >= 5 else (f"{mean:.1f} km" if mean >= 1 else f"{mean * 1000:.0f} m")
        x = kx + 0.148 * l
        fig.patches.append(Rectangle((x, 0.03), 0.022, 0.03, transform=fig.transFigure, fc=LEVEL_RGB[l], ec="0.3",
                                     lw=0.6))
        last = l == len(lev) - 1
        fig.text(x + 0.027, 0.045, f"{l}{' (finest)' if last else ''}: {size}, {r['cells'] / 1e6:.2f}M cells"
                 if r["cells"] >= 1e6 else f"{l}: {size}, {r['cells']:,} cells", ha="left", va="center", fontsize=8.5)
    fig.savefig(figdir / "conus_telescope_zoom.png", dpi=150)
    plt.close(fig)
    print("composed conus_telescope_zoom.png")


# -- compare: one window in two meshes, least angle and area ratio ----------------------------

def compare(cmpdir: Path, figdir: Path) -> None:
    from PIL import Image
    mpl.rcParams.update({"font.size": 10, "xtick.labelsize": 8, "ytick.labelsize": 8})
    if (cmpdir / "img/compare_meta.json").exists():
        cmpdir = cmpdir / "img"
    meta = json.load(open(cmpdir / "compare_meta.json"))
    H = meta["size"][1]
    px_per_km = H / (2 * meta["half_km"])
    rows = [("p3", "p3: area ratio bound kept (--area-ratio keep)"), ("p4", "p4: no area ratio bound")]
    cols = [("min_angle", "inferno", "Least angle (degrees)"),
            ("area_ratio", "viridis", "Largest area ratio to an edge neighbour")]
    fig = plt.figure(figsize=(11, 9.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.045], left=0.01, right=0.99, top=0.955, bottom=0.06,
                          wspace=0.03, hspace=0.09)
    for i, (mesh, title) in enumerate(rows):
        for j, (field, _, _) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(Image.open(cmpdir / f"compare_{mesh}_{field}.png").convert("RGB"),
                      interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.6)
                s.set_color("0.3")
            if j == 0:
                ax.text(0, 1.012, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
            if i == 1 and j == 0:
                L = 5 * px_per_km
                x0, y0 = 60, H - 70
                ax.add_patch(mpl.patches.Rectangle((x0 - 25, y0 - 75), L + 50, 110, color="white", alpha=0.85, lw=0))
                ax.plot([x0, x0 + L], [y0, y0], color="black", lw=3, solid_capstyle="butt")
                ax.text(x0 + L / 2, y0 - 14, "5 km", ha="center", va="bottom", fontsize=9)
    for j, (field, cmap, label) in enumerate(cols):
        cax = fig.add_subplot(gs[2, j])
        pos = cax.get_position()
        cax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.12, pos.height])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(*meta["fields"][field]), cmap), cax=cax,
                          orientation="horizontal")
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
    fig.savefig(figdir / "conus_p3_p4.png", dpi=1600 / 11)
    plt.close(fig)
    print("composed conus_p3_p4.png")


# -- docs: 256 colours, under 300 KB --------------------------------------------------------------

def docs_png(src: Path, dst: Path, width: int, cmaps=()) -> int:
    """Resize to ``width`` and cut to 256 colours. A colour bar is a small part of a figure, so the
    palette is chosen from the figure plus wide strips of each colour map in ``cmaps``."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    if cmaps:
        train = Image.new("RGB", (im.width, im.height + 300 * len(cmaps)))
        train.paste(im, (0, 0))
        for i, name in enumerate(cmaps):
            rgb = (mpl.colormaps[name](np.linspace(0, 1, im.width))[:, :3] * 255).astype(np.uint8)
            train.paste(Image.fromarray(rgb[None].repeat(300, 0)), (0, im.height + 300 * i))
        palette = train.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q = im.quantize(palette=palette, dither=Image.Dither.NONE)
    else:
        q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q.save(dst, optimize=True)
    return dst.stat().st_size


DOCS = [("conus_telescope_zoom.png", "conus-telescope.png", 1600, ()),
        ("conus_stereo_min_angle.png", "conus-min-angle.png", 1400, ("inferno",)),
        ("conus_stereo_transition_zoom.png", "conus-transition-zoom.png", 1600, ("inferno",)),
        ("conus_p3_p4.png", "conus-p3-p4.png", 1500, ("inferno", "viridis"))]


def main(argv):
    mpl.rcParams.update(STYLE)
    cmd, args = argv[1], [Path(a) for a in argv[2:4]]
    if cmd != "docs":
        args[1].mkdir(parents=True, exist_ok=True)
    if cmd == "stereo":
        stereo(*args)
    elif cmd == "telescope":
        telescope(*args, argv[4])
    elif cmd == "compare":
        compare(*args)
    elif cmd == "docs":
        for src, dst, width, cmaps in DOCS:
            if (args[0] / src).exists():
                print(dst, docs_png(args[0] / src, args[1] / dst, width, cmaps), "bytes")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv)
