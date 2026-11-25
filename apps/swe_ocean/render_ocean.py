# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Frames of swe_ocean snapshots: surface relative vorticity, surface height and speed on a map.

    python render_ocean.py 'movie.step*.soma' --dt 16.7549 --out frames --log-speed
    ffmpeg -framerate 4 -i frames/frame%04d.png -pix_fmt yuv420p ocean.mp4

The snapshots carry no time, so a frame's time is its step (from the file name) times --dt, the dt of
the run's start-up report. The vorticity is the radial part of curl u, from each element's GLL
derivative (exact for the element's polynomial). Land points of the kept coastal elements (H at the
floor, 10 m) are grey; removed land elements are not in the mesh and show the beige background. The
colour limits are the same for every frame: by default from the last snapshot (99.5th percentile of
|vorticity| and of |eta|, 99.9th of speed). --log-speed draws speed on a log scale from a thousandth
of its limit (--speed-min), so the slow interior shows beside the coasts. The red dot marks the
fastest point, which moves between coastal points of nearly the same speed; --no-marker drops it.

Needs somaview (tools/somaview, with read_nodal) and pyvista: pip install 'tools/somaview[pyvista]'.
"""

import argparse
import glob
import os
import re

import numpy as np
import pyvista as pv
import somaview as sv
from somaview.nodal import gll_points
from somaview.vtkconv import to_pyvista

pv.OFF_SCREEN = True


def derivative_matrix(g):
    """D[i, j] = l_j'(g_i) for the Lagrange basis l on the points g."""
    n = len(g)
    w = np.array([1 / np.prod([g[j] - g[k] for k in range(n) if k != j]) for j in range(n)])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / w[i] / (g[i] - g[j])
        D[i, i] = -D[i].sum()
    return D


def vorticity(x, u, p):
    """n . curl u at every GLL point; x and u are (elements * (p+1)^2, 3), point k = i + (p+1) j."""
    q = p + 1
    D = derivative_matrix(gll_points(p))
    x, u = x.reshape(-1, q, q, 3), u.reshape(-1, q, q, 3)
    d_xi = lambda f: np.einsum("ik,ejkc->ejic", D, f)
    d_eta = lambda f: np.einsum("jk,ekic->ejic", D, f)
    x_xi, x_eta, u_xi, u_eta = d_xi(x), d_eta(x), d_xi(u), d_eta(u)
    n = x / np.linalg.norm(x, axis=-1, keepdims=True)
    J = np.einsum("...c,...c", n, np.cross(x_xi, x_eta))[..., None]
    a1, a2 = np.cross(x_eta, n) / J, np.cross(n, x_xi) / J  # the dual (contravariant) basis
    return np.einsum("...c,...c", n, np.cross(a1, u_xi) + np.cross(a2, u_eta)).reshape(-1)


def load(path, h_min, p):
    """The piece, its points, and the drawn fields (NaN on land points)."""
    piece = sv.read_nodal(path, fields=["u", "H", "eta"])
    x = np.asarray(piece.points)
    u = np.asarray(piece.point_data["u"])
    fields = {"zeta": vorticity(x, u, p), "eta": np.array(piece.point_data["eta"], dtype=float).reshape(-1),
              "speed": np.linalg.norm(u, axis=1)}
    land = np.asarray(piece.point_data["H"]) <= h_min * (1 + 1e-9)
    for v in fields.values():
        v[land] = np.nan
    return piece, x, fields


def flat_map(piece, x, fields):
    """The sub-quads in longitude-latitude degrees, cells across the date line drawn at both edges."""
    grid = to_pyvista(piece)
    n = x / np.linalg.norm(x, axis=1)[:, None]
    lon = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    flat = pv.UnstructuredGrid(grid.cells, grid.celltypes, np.c_[lon, lat, np.zeros_like(lon)])
    for name, v in fields.items():
        flat[name] = v
    conn = grid.cells.reshape(-1, 5)[:, 1:]
    span = lon[conn].max(axis=1) - lon[conn].min(axis=1)
    wrap = np.nonzero(span > 180)[0]
    if wrap.size:
        pieces = [flat.extract_cells(np.nonzero(span <= 180)[0])]
        for shift in (360.0, -360.0):
            c = flat.extract_cells(wrap)
            pts = np.asarray(c.points).copy()
            move = pts[:, 0] < 0 if shift > 0 else pts[:, 0] > 0
            pts[move, 0] += shift
            c.points = pts
            pieces.append(c)
        flat = pieces[0].merge(pieces[1:], merge_points=False)
    return flat.clip_box((-180, 180, -90, 90, -1, 1), invert=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="snapshot glob, quoted, e.g. 'movie.step*.soma'")
    ap.add_argument("--dt", type=float, required=True, help="the run's dt in seconds")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--order", type=int, default=3, help="the run's --p")
    ap.add_argument("--h-min", type=float, default=10.0, help="the land depth, m")
    ap.add_argument("--zeta-max", type=float, help="vorticity colour limit, 1/s")
    ap.add_argument("--eta-max", type=float, help="surface height colour limit, m")
    ap.add_argument("--speed-max", type=float, help="speed colour limit, m/s")
    ap.add_argument("--log-speed", action="store_true", help="speed on a log scale")
    ap.add_argument("--speed-min", type=float, help="lower speed limit with --log-speed (default max / 1000)")
    ap.add_argument("--no-marker", action="store_true", help="no dot at the fastest point")
    ap.add_argument("--width", type=int, default=1800)
    a = ap.parse_args()

    step_of = lambda f: int(re.search(r"step(\d+)", os.path.basename(f)).group(1))
    files = sorted(glob.glob(a.pattern), key=step_of)
    if not files:
        raise SystemExit(f"no files match {a.pattern}")
    os.makedirs(a.out, exist_ok=True)
    if None in (a.zeta_max, a.eta_max, a.speed_max):
        _, _, last = load(files[-1], a.h_min, a.order)
        a.zeta_max = a.zeta_max or float(np.nanpercentile(np.abs(last["zeta"]), 99.5))
        a.eta_max = a.eta_max or float(np.nanpercentile(np.abs(last["eta"]), 99.5))
        a.speed_max = a.speed_max or float(np.nanpercentile(last["speed"], 99.9))
    speed_min = (a.speed_min or a.speed_max / 1000) if a.log_speed else 0.0
    if a.log_speed and not 0 < speed_min < a.speed_max:
        raise SystemExit(f"--speed-min {speed_min} must be in (0, {a.speed_max})")
    print(f"colour limits: vorticity +-{a.zeta_max:.2e} 1/s, eta +-{a.eta_max:.2e} m, "
          f"speed {speed_min:.2e}..{a.speed_max:.3f} m/s{' (log)' if a.log_speed else ''}")
    panels = [("zeta", "RdBu_r", (-a.zeta_max, a.zeta_max), "relative vorticity (1/s)", False),
              ("eta", "PuOr_r", (-a.eta_max, a.eta_max), "surface height eta (m)", False),
              ("speed", "viridis", (speed_min, a.speed_max), "speed (m/s)", a.log_speed)]

    panel = a.width // 2 + 110  # a 2:1 map and room for the colour bar
    for i, f in enumerate(files):
        step = step_of(f)
        days = step * a.dt / 86400
        piece, x, fields = load(f, a.h_min, a.order)
        speed = fields["speed"]
        k = int(np.nanargmax(speed))
        n = x[k] / np.linalg.norm(x[k])
        lo, la = np.degrees(np.arctan2(n[1], n[0])), np.degrees(np.arcsin(n[2]))
        flat = flat_map(piece, x, fields)
        if a.log_speed:
            flat["speed"] = np.maximum(flat["speed"], speed_min)  # below the scale: its lowest colour; NaN stays
        pl = pv.Plotter(shape=(len(panels), 1), window_size=(a.width, len(panels) * panel), off_screen=True,
                        border=False)
        for row, (name, cmap, clim, title, log) in enumerate(panels):
            pl.subplot(row, 0)
            pl.set_background("#e3dccf")
            bar = dict(title=title, vertical=False, position_x=0.25, position_y=0.01, width=0.5, height=0.06,
                       color="black", fmt="%.1e", title_font_size=16, label_font_size=14)
            pl.add_mesh(flat.copy(), scalars=name, cmap=cmap, clim=clim, log_scale=log, nan_color="#9a9a9a",
                        lighting=False, scalar_bar_args=bar)
            if name == "speed":
                if not a.no_marker:
                    pl.add_points(np.array([[lo, la, 1.0]]), color="red", point_size=12,
                                  render_points_as_spheres=True)
                pl.add_text(f"fastest {speed[k]:.3f} m/s at {lo:.1f}E {la:.1f}N", position="upper_right",
                            font_size=10, color="black")
            pl.add_text(f"t = {days:.3f} d  (step {step})", position="upper_left", font_size=11, color="black")
            pl.camera.parallel_projection = True
            pl.camera.focal_point = (0, -6, 0)
            pl.camera.position = (0, -6, 1000)
            pl.camera.up = (0, 1, 0)
            pl.camera.parallel_scale = 0.5 * panel / a.width * 360 * 1.02
            pl.reset_camera_clipping_range()
        out = os.path.join(a.out, f"frame{i:04d}.png")
        pl.screenshot(out)
        pl.close()
        print(f"{out}: {os.path.basename(f)}, t {days:.4f} d, max speed {speed[k]:.4f} m/s")


if __name__ == "__main__":
    main()
