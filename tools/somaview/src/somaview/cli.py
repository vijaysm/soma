# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Command-line interface: ``somaview info|render|convert``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .format import FileKind, SomaFile, detect
from .mesh import read_piece


def _info_mesh(path: str) -> None:
    with SomaFile(path) as sf:
        print(repr(sf))
        for d in sf.written_dims:
            counts = sf.counts_by_part(d)
            print(f"  dim {d}: {int(counts.sum())} entities, by part {counts.tolist()}")
        if sf.fields:
            print("fields:")
        for f in sf.fields:
            note = "" if f.drawable else f" (GLL order {f.order}: read_nodal)" if f.nodal_on_quads else f" (not drawable: {f.layout.name}{', variable' if f.variable else ''})"
            print(f"  {f.name}: {f.dtype}, {f.ncomp} comp, on dims {f.dims}{note}")
        if sf.values:
            print("mesh values:")
        for name, v in sf.values.items():
            shown = f" = {v.value.tolist()}" if v.value is not None else ""
            print(f"  {name}: {v.dtype}, {v.ncomp} comp{shown}")
        t = sf.time()
        if t is not None:
            print(f"time: {t}")


def _info_amr(path: str) -> None:
    from .amr import AmrFile

    with AmrFile(path) as af:
        print(repr(af))
        print(f"domain: cells {af.domain_lo.tolist()} .. {af.domain_hi.tolist()}, "
              f"physical {af.lo.tolist()} .. {af.hi.tolist()}, periodic {af.periodic}")
        for l in range(af.nlevels):
            print(f"  level {l}: {len(af.boxes[l])} boxes, spacing {af.spacing(l).tolist()}")
        print("fields:")
        for f in af.fields:
            print(f"  {f.name}: {f.dtype}, {f.ncomp} comp, centering {f.centering(af.dim)}")


def _cmd_info(args) -> int:
    kind = detect(args.file)
    if kind is None:
        print(f"{args.file}: not a SOMA file", file=sys.stderr)
        return 1
    (_info_amr if kind == FileKind.amr else _info_mesh)(args.file)
    return 0


def _cmd_render(args) -> int:
    from .plot import plot

    clip = args.clip
    kw = {}
    if args.parts is not None:
        kw["parts"] = args.parts
    if args.dims is not None:
        kw["dims"] = args.dims
    if args.nodal or args.lagrange or args.resolution is not None:
        kw.update(nodal="lagrange" if args.lagrange else True, resolution=args.resolution, bounded=not args.unbounded)
    plot(args.file, scalars=args.scalars, clip=clip, screenshot=args.output, off_screen=True, **kw)
    print(f"wrote {args.output}")
    return 0


def _write_vtu(path: str, out: str, parts, dims) -> None:
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    from .vtkconv import to_vtk

    piece = read_piece(path, parts, dims=dims)
    ug = to_vtk(piece)
    w = vtkXMLUnstructuredGridWriter()
    w.SetFileName(out)
    w.SetInputData(ug)
    w.Write()


def _write_pvtu(path: str, out: str, parts, dims) -> None:
    from vtkmodules.vtkIOParallelXML import vtkXMLPUnstructuredGridWriter

    from .vtkconv import to_vtk

    with SomaFile(path) as sf:
        use_parts = sorted(range(sf.nparts) if parts is None else parts)
    w = vtkXMLPUnstructuredGridWriter()
    w.SetFileName(out)
    w.SetNumberOfPieces(len(use_parts))
    for k, p in enumerate(use_parts):
        piece = read_piece(path, [p], dims=dims)
        w.SetInputData(to_vtk(piece))
        w.SetStartPiece(k)
        w.SetEndPiece(k)
        w.SetWriteSummaryFile(k == len(use_parts) - 1)
        w.Write()


def _cmd_convert(args) -> int:
    ext = Path(args.output).suffix.lower()
    dims = args.dims
    dim = dims[0] if dims else None
    if ext == ".vtkhdf":
        from .vtkhdf import write_vtkhdf

        write_vtkhdf(args.file, args.output, parts=args.parts, dim=dim)
    elif ext == ".vtu":
        _write_vtu(args.file, args.output, args.parts, dims)
    elif ext == ".pvtu":
        _write_pvtu(args.file, args.output, args.parts, dims)
    else:
        print(f"convert: unknown output extension {ext!r} (use .vtkhdf, .vtu or .pvtu)", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somaview", description="Inspect, render and convert SOMA files.")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print a SOMA file's parts, dims, fields (or AMR levels)")
    info.add_argument("file")
    info.set_defaults(func=_cmd_info)

    render = sub.add_parser("render", help="render a SOMA file to a PNG with pyvista")
    render.add_argument("file")
    render.add_argument("-s", "--scalars", default=None, help="field to color by")
    render.add_argument("-o", "--output", required=True, help="output PNG path")
    render.add_argument("--parts", type=int, nargs="+", default=None)
    render.add_argument("--dims", type=int, nargs="+", default=None)
    render.add_argument("--nodal", action="store_true", help="draw a spectral-element field, on sub-quads")
    render.add_argument("--lagrange", action="store_true", help="draw it on VTK Lagrange cells")
    render.add_argument("--resolution", type=int, default=None, help="draw it on N x N sub-quads per element")
    render.add_argument("--unbounded", action="store_true", help="with --resolution, the polynomial beyond its element's values")
    render.add_argument("--clip", default=None, help="clip plane normal: x, y, z, or -x, -y, -z")
    render.set_defaults(func=_cmd_render)

    convert = sub.add_parser("convert", help="convert a SOMA mesh to .vtkhdf, .vtu or .pvtu")
    convert.add_argument("file")
    convert.add_argument("output")
    convert.add_argument("--parts", type=int, nargs="+", default=None)
    convert.add_argument("--dims", type=int, nargs="+", default=None)
    convert.set_defaults(func=_cmd_convert)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
