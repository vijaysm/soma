# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Load the SOMA ParaView reader plugin, read a file, extract the surface, color it by a chosen
array, and save a screenshot. A pvbatch example: runs single-process or under MPI (one SOMA
piece per rank, as the reader's ``RequestData`` arranges via ``UPDATE_PIECE_NUMBER``/
``UPDATE_NUMBER_OF_PIECES``; see the note at the end of this file for ``pvserver``).

Usage::

    pvbatch --mpi examples/render_soma.py path/to/file.soma array_name output.png
    mpiexec -n 4 pvbatch --mpi examples/render_soma.py path/to/file.soma array_name output.png

(or ``pvpython --no-mpi ... --force-offscreen-rendering`` for a single, non-MPI process, e.g.
when testing on a workstation without a display).

Tested with ParaView 6.1.1's EGL ``pvbatch`` on Linux, on 4 ranks, against hex_fields.soma.
Rendering needs an OpenGL context (a GPU through EGL, OSMesa, or a display); run offscreen with
``DISPLAY`` unset, since a stale ``DISPLAY`` makes some ranks try X and render nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import paraview.simple as smp

PLUGIN = Path(__file__).resolve().parent.parent / "SomaReader.py"


def render(path: str, array_name: str, output_png: str, size=(1200, 800)) -> None:
    smp.LoadPlugin(str(PLUGIN), remote=True, ns=globals())
    reader = SomaReader(FileNames=[path])  # noqa: F821 (injected by LoadPlugin's ns=globals())
    reader.UpdatePipelineInformation()

    surface = smp.ExtractSurface(Input=reader)
    display = smp.Show(surface)

    # Try cell data first, then point data (ExtractSurface keeps both).
    info = surface.GetDataInformation()
    on_cells = info.GetCellDataInformation().GetArrayInformation(array_name) is not None
    smp.ColorBy(display, ("CELLS" if on_cells else "POINTS", array_name))
    display.RescaleTransferFunctionToDataRange(True)
    display.SetScalarBarVisibility(smp.GetActiveView(), True)

    view = smp.GetActiveView()
    view.ViewSize = list(size)
    view.OrientationAxesVisibility = 0
    smp.ResetCamera()
    smp.Render()
    smp.SaveScreenshot(output_png, view, ImageResolution=list(size))


# -- running under pvserver with MPI ------------------------------------------------------------
#
# This script uses paraview.simple, which talks to a ParaView server (pvbatch *is* that server,
# running in the same process). For a real cluster:
#
#   mpiexec -n P pvserver --mpi
#
# starts a parallel server of P ranks; from a client (pvpython, the ParaView GUI, or a second
# pvbatch acting as a client) do::
#
#   from paraview.simple import Connect
#   Connect("myhost", 11111)
#
# before LoadPlugin/SomaReader. The plugin must be on every rank's SOMAVIEW_PATH or next to an
# installed somaview (ParaView loads and runs the plugin's Python on each server rank). Each rank
# then reads the SOMA parts RequestData assigns it (parts_for_rank(N, P, rank)); if mpi4py is
# already imported in that process (e.g. a server launched by an MPI-aware mpi4py bootstrap),
# somaview.read exchanges halo vertices in one alltoall instead of scanning the sharing tables.
# A client-only script (no Connect) runs the reader in-process instead, as above.

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <path.soma> <array_name> <output.png>")
        sys.exit(1)
    render(sys.argv[1], sys.argv[2], sys.argv[3])
