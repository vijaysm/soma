# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Compose the renderer galleries of the documentation's visualization pages.

    python compose_gallery.py PARAVIEW_DIR VISIT_DIR PYVISTA_DIR OUTDIR [IMGDIR]

PARAVIEW_DIR holds paraview/examples/pv_gallery.py's renders, VISIT_DIR visit/tests/visit_test.py's
(4-rank engine, and the CONUS file on 8), PYVISTA_DIR examples/gallery.py's. Writes
gallery_paraview.png, gallery_visit.png and gallery_pyvista_yt.png into OUTDIR and, given
IMGDIR, their documentation copies visualization-*.png (1600 pixels wide, 256 colours).
Needs numpy, matplotlib and Pillow.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
STYLE = {"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
         "patch.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
         "axes.titlelocation": "left", "legend.frameon": False, "xtick.major.size": 3,
         "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200}


def trim(im, pad=12):
    """Crop the white margin of a render."""
    mask = (im[..., :3] < 0.97).any(axis=2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return im
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.shape[1])
    return im[y0:y1, x0:x1]


def one(pattern):
    found = sorted(glob.glob(str(pattern)))
    if not found:
        raise SystemExit(f"compose_gallery: no render {pattern}")
    return found[0]


def sheets(pv: Path, vi: Path, py: Path):
    return {
        "gallery_paraview.png": ("ParaView 6.1.1 pvbatch, SOMA reader plugin (4 or 8 MPI ranks, IceT compositing)", [
            (pv / "pv_hex_4p_process.png", "hex mesh written on 4 ranks, coloured by reading rank"),
            (pv / "pv_hex_4p_pressure.png", "same mesh clipped, pressure and velocity glyphs"),
            (pv / "pv_bricks_material.png", "64 MOAB bricks (4 parts), set:MATERIAL_SET"),
            (pv / "pv_highorder_curved.png", "quadratic tet10 to hex27 cells, curved surfaces"),
            (pv / "pv_amr3d_slice.png", "3D AMR checkpoint (4 parts), slice of u"),
            (pv / "pv_conus_spacing.png", "CONUS 200 m, 50.3M cells (8 ranks), spacing_km")]),
        "gallery_visit.png": ("VisIt 3.5.0, SOMA database plugin (parallel engine)", [
            (vi / "visit_hex_7p_np4.png", "hex mesh written on 7 ranks, pressure (4 ranks)"),
            (vi / "visit_tet_4p_np4.png", "tet mesh written on 4 ranks, pressure (4 ranks)"),
            (vi / "visit_mixed3d_np4.png", "tet, pyramid, prism, hex and knife cells"),
            (vi / "visit_ocn_3p_np4.png", "ocean mesh (3 parts), nodal/GLOBAL_ID"),
            (vi / "visit_amr2d_np4.png", "2D AMR checkpoint, phi"),
            (one(vi / "visit_conus*_np8.png"), "CONUS 200 m, 50.3M cells (8 ranks), spacing_km")]),
        "gallery_pyvista_yt.png": ("pyvista (offscreen) and yt front-ends", [
            (py / "hex_fields_clipped_pressure.png", "pyvista: hex mesh clipped, pressure"),
            (py / "64bricks_material_set.png", "pyvista: 64 MOAB bricks, set:MATERIAL_SET"),
            (py / "highorder3d_curved.png", "pyvista: quadratic cells with edges"),
            (py / "mixed3d_edges.png", "pyvista: tet, pyramid, prism, hex, knife"),
            (py / "amr2d_phi_boxes.png", "pyvista: 2D AMR, phi with box outlines"),
            (py / "amr3d_yt_slice_u.png", "yt: slice of 3D AMR checkpoint, u")]),
    }


DOCS = {"gallery_paraview.png": "visualization-paraview.png", "gallery_visit.png": "visualization-visit.png",
        "gallery_pyvista_yt.png": "visualization-pyvista-yt.png"}


def main(argv):
    if len(argv) < 5:
        raise SystemExit(__doc__)
    pv, vi, py, out = (Path(a) for a in argv[1:5])
    img = Path(argv[5]) if len(argv) > 5 else None
    out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)
    for name, (title, items) in sheets(pv, vi, py).items():
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
        for ax, (f, cap) in zip(axs.flat, items):
            ax.imshow(trim(mpimg.imread(f)))
            ax.set_axis_off()
            ax.set_title(cap, loc="left", fontsize=9.5)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        print("composed", name)
        if img is not None:
            from PIL import Image
            im = Image.open(out / name).convert("RGB")
            if im.width > 1600:
                im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
            q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            q.save(img / DOCS[name], optimize=True)
            print(DOCS[name], (img / DOCS[name]).stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv)
