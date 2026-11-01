# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
#
# Writes the reference legacy VTK files in this directory with VTK's own
# vtkUnstructuredGridWriter (ASCII), in file versions 5.1 and 4.2.
# Run with ParaView's Python: pvpython make_reference.py [output directory]
# The SOMA reader tests restate the values below; keep the two in step.
# VTK writes reals with 11 significant digits (fewer for float), so every real
# below is a literal of few digits: the text then reads back as the same value.

import os
import sys

import vtk

INF = float("inf")
NAN = float("nan")


def grid(points, cells):
    g = vtk.vtkUnstructuredGrid()
    p = vtk.vtkPoints()
    p.SetDataTypeToDouble()
    for xyz in points:
        p.InsertNextPoint(xyz)
    g.SetPoints(p)
    for code, ids in cells:
        g.InsertNextCell(code, len(ids), ids)
    return g


def array(cls, name, ncomp, values, component_names=None, units=None):
    a = cls()
    a.SetName(name)
    a.SetNumberOfComponents(ncomp)
    a.SetNumberOfTuples(len(values) // ncomp)
    for i, v in enumerate(values):
        a.SetValue(i, v)
    if component_names:
        for k, n in enumerate(component_names):
            a.SetComponentName(k, n)
    if units:
        a.GetInformation().Set(vtk.vtkDataArray.UNITS_LABEL(), units)
    return a


def write(g, out_dir, stem):
    for version in (51, 42):
        w = vtk.vtkUnstructuredGridWriter()
        w.SetFileName(os.path.join(out_dir, "%s_%d.vtk" % (stem, version)))
        w.SetInputData(g)
        w.SetFileTypeToASCII()
        w.SetFileVersion(version)
        if not w.Write():
            raise RuntimeError("cannot write " + stem)


def tri_quad_2d():
    points = [(0.1, -0.25, 0), (1.1, -0.25, 0), (2.1, -0.25, 0), (0.1, 0.75, 0),
              (1.1, 0.75, 0), (2.1, 0.75, 0), (3.1, 0.25, 0)]
    cells = [(5, [1, 2, 4]), (9, [0, 1, 4, 3]), (5, [2, 5, 4]), (5, [2, 6, 5])]
    g = grid(points, cells)
    pd = g.GetPointData()
    pd.AddArray(array(vtk.vtkDoubleArray, "temperature", 1,
                      [1.5, -2.25, 0.1, 3.0, 0.001, 12345.678, -0.5]))
    pd.AddArray(array(vtk.vtkDoubleArray, "velocity", 3,
                      [i + 0.5 * k for i in range(7) for k in range(3)],
                      component_names=["vx", "vy", "vz"], units="m/s"))
    pd.AddArray(array(vtk.vtkFloatArray, "pf", 1, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    pd.SetActiveScalars("temperature")
    pd.SetActiveVectors("velocity")
    cd = g.GetCellData()
    cd.AddArray(array(vtk.vtkIntArray, "material", 1, [10, -20, 30, -2147483648]))
    cd.AddArray(array(vtk.vtkLongLongArray, "gid64", 1,
                      [2**40 + 1, -(2**40) - 2, 9223372036854775807, -9223372036854775808]))
    cd.AddArray(array(vtk.vtkFloatArray, "cf", 3,
                      [0.25 * c + k for c in range(4) for k in range(3)]))
    time = array(vtk.vtkDoubleArray, "TIME", 1, [1.5])
    g.GetFieldData().AddArray(time)
    return g


def mixed_3d():
    cube = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    apex = [(0.5, 0.5, 1.5)]
    tet = [(3, 0, 0), (4, 0, 0), (3, 1, 0), (3, 0, 1)]
    wedge = [(5, 0, 0), (6, 0, 0), (5, 1, 0), (5, 0, 1), (6, 0, 1), (5, 1, 1)]
    points = cube + apex + tet + wedge
    cells = [(13, [13, 14, 15, 16, 17, 18]), (10, [9, 10, 11, 12]),
             (12, [0, 1, 2, 3, 4, 5, 6, 7]), (14, [4, 5, 6, 7, 8])]
    g = grid(points, cells)
    pd = g.GetPointData()
    pd.AddArray(array(vtk.vtkDoubleArray, "pd", 1, [0.125 * i for i in range(19)]))
    pd.AddArray(array(vtk.vtkFloatArray, "pv", 3,
                      [i + k / 4.0 for i in range(19) for k in range(3)]))
    pd.SetActiveScalars("pd")
    pd.SetActiveVectors("pv")
    cd = g.GetCellData()
    cd.AddArray(array(vtk.vtkIntArray, "ci", 1, [13, 10, 12, 14]))
    cd.AddArray(array(vtk.vtkLongLongArray, "cl", 1, [-1, 2**33, -(2**35), 7]))
    cd.AddArray(array(vtk.vtkDoubleArray, "cd", 3,
                      [1.5, NAN, -2.25, INF, 0.0, -INF, 1e-300, 1e300, -0.0, 3.0, 4.0, 5.0]))
    cd.AddArray(array(vtk.vtkUnsignedCharArray, "flag", 1, [0, 1, 254, 255]))
    cd.SetActiveScalars("ci")
    return g


def polygons():
    points = [(0, 0, 0), (2, 0, 0), (3, 1, 0), (1, 2, 0), (-1, 1, 0),
              (3, 1, 0), (5, 0, 0), (6, 1, 0), (6, 2, 0), (4, 3, 0), (3, 2, 0),
              (0, 3, 0), (1, 3, 0)]
    cells = [(7, [5, 6, 7, 8, 9, 10]), (7, [0, 1, 2, 3, 4]), (5, [4, 3, 11]),
             (7, [3, 10, 9, 12, 11])]
    g = grid(points, cells)
    pd = g.GetPointData()
    pd.AddArray(array(vtk.vtkDoubleArray, "h", 1, [0.5 * i - 1 for i in range(13)]))
    pd.SetActiveScalars("h")
    cd = g.GetCellData()
    cd.AddArray(array(vtk.vtkIntArray, "pid", 1, [6, 5, 3, 5]))
    cd.AddArray(array(vtk.vtkFloatArray, "area", 1, [5.5, 6.0, 1.5, 3.25]))
    return g


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    write(tri_quad_2d(), out_dir, "tri_quad_2d")
    write(mixed_3d(), out_dir, "mixed_3d")
    write(polygons(), out_dir, "polygons")


main()
