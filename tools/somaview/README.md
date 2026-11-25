# somaview

Visualization of SOMA files (SOMA's parallel HDF5 mesh format and its AMR checkpoints) in
ParaView, VisIt, pyvista and yt, and export to VTKHDF. One reading core reads exactly the parts
a process is given, so every renderer scales the way SOMA itself does: each MPI rank reads its
own block of parts with one hyperslab per dataset, plus its halo.

The documentation is in SOMA's pages: `docs/visualization/overview.md`, `paraview.md`, `visit.md`
and `python.md`. This file summarizes the package and its validation.

| Renderer | Entry point | Parallelism |
| --- | --- | --- |
| ParaView | `paraview/SomaReader.py` (Python reader plugin) | pvserver / pvbatch pieces, one block of parts each; IceT compositing |
| VisIt | `visit/` (C++ STMD database plugin over `soma_reader.hpp`) | domains = parts, distributed by VisIt's parallel engine |
| pyvista | `somaview.plot`, `somaview.read_pyvista`, `somaview render` | one process (any subset of parts) |
| yt | `somaview.yt.load` | one process; AMR boxes become yt grid patches |
| VTKHDF | `somaview.vtkhdf.write_vtkhdf`, `somaview convert` | one partition per part; collective MPI-IO write with `comm=` |

## Install

```
pip install -e tools/somaview[all]     # numpy, h5py, plus pyvista, vtk, yt, mpi4py
pip install -e tools/somaview          # core only: numpy, h5py
```

ParaView: load `paraview/SomaReader.py` as a plugin (see `paraview/README.md`). VisIt: build
the plugin against a VisIt install with `xml2info`, `xml2cmake`, `cmake`, `make` (see
`visit/README.md`).

## Reading

```python
import somaview as sv

sf = sv.open("restart.soma")                 # SomaFile (or AmrFile): parts, dims, fields, sets
piece = sv.read("restart.soma")              # every part, merged by gid: a MeshPiece
piece = sv.read("restart.soma", parts=[3, 4], dims=[2, 3], point_fields=["T"])
grid = sv.read_vtk("restart.soma")           # vtkUnstructuredGrid, or vtkOverlappingAMR

from mpi4py import MPI                       # under mpiexec: each rank its own parts,
piece = sv.read("restart.soma", comm=MPI.COMM_WORLD)   # the halo by one all-to-all
```

A part stores only the vertices it owns; the elements of a part also reference vertices that
other parts own (the halo). The file records the sharing (`dim0/shared`), and the halo is found
in one of two ways:

* scan (default, no communication): read the other parts' sharing tables in bounded chunks and
  keep the rows whose copies are on the parts being read. Any process can read any part on its
  own (a VisIt domain, a notebook looking at part 17).
* exchange (`comm=`): each rank reads only its own parts' sharing tables and sends its owned
  shared vertices, with their point field values, to the ranks that hold copies, in one
  all-to-all (the algorithm of SOMA's `read_soma`). I/O and memory per rank are proportional to
  its own parts and their interface.

Parts are assigned as SOMA assigns them: at P ranks, part q goes to rank floor(qP/N).

## What is drawn

* Elements of the mesh's dimension by default; lower written dimensions as extra blocks on
  request.
* Linear cells in VTK's corner order. A SOMA (MOAB) prism has its base triangle counterclockwise
  seen from the top; VTK changed `vtkWedge`'s convention in 9.7: for VTK <= 9.6 (ParaView <= 6.1,
  VisIt 3.5) the corners are reordered [0 2 1 3 5 4], for VTK >= 9.7 they are used as stored.
  The choice follows the VTK in use (`SOMAVIEW_VTK_WEDGE=legacy|current` overrides).
* High-order elements become VTK quadratic cells (edge3, tri6, tri7, quad8, quad9, tet10,
  pyramid13, prism15, hex20, hex27) with edge and face nodes permuted to VTK's order.
* Polygons as VTK polygons; knives as VTK polyhedra (VisIt: degenerate hexahedra); SOMA
  polyhedra (stored by vertices only) as convex point sets.
* Fields with one fixed-size row per entity: point data on vertices, cell data on elements;
  entities outside a field's domain get NaN. Fields on entity sets become cell labels
  `set:<name>`. Variable-length, handle and element-node fields are listed as skipped.
* Spectral-element fields (GLL values on quads) by `sv.read_nodal(path)`: a piece of order x order
  sub-quads per element with the stored values as point data (`sv.plot(piece, scalars="eta")`).
* `soma_part` (owning part of each cell) and `soma_gid` (global ids, also VTK GlobalIds).
* AMR checkpoints: one grid per box, coarse cells covered by finer boxes blanked.

## pyvista, yt, command line, VTKHDF

```python
sv.plot("restart.soma", scalars="pressure", clip="x", screenshot="out.png")
sv.plot("checkpoint.soma", scalars="phi", box_outlines=True)

import somaview.yt as svyt
ds = svyt.load("checkpoint.soma")            # AMR: yt.load_amr_grids
ds = svyt.load("restart.soma")               # mesh: yt.load_unstructured_mesh

from somaview.vtkhdf import write_vtkhdf
write_vtkhdf("restart.soma", "restart.vtkhdf")                              # one partition per part
write_vtkhdf("restart.soma", "restart.vtkhdf", comm=MPI.COMM_WORLD)         # collective write
```

```
somaview info restart.soma
somaview render restart.soma -s pressure -o out.png --clip x
somaview convert restart.soma out.vtkhdf
```

`plot()` extracts the external surface of a volume mesh before rendering and clips the full
volume first, so the cut face carries the field. An unknown `scalars` name raises, listing the
arrays present. The VTKHDF file is a partitioned UnstructuredGrid (format 2.2): each partition
holds its own halo, with partition-local `Connectivity` and `Offsets` as the specification
requires, so VTK's parallel `vtkHDFReader` reads one partition per rank.

## Scaling (two Intel Xeon Gold 5317, CONUS 200 m mesh: 50,304,352 quadrilaterals, 48 parts)

| Ranks | Halo exchange read (s) | Halo scan read (s) | Peak memory per rank |
| --- | --- | --- | --- |
| 8 | 9.51 | 8.37 | not measured |
| 16 | 5.90 | 4.75 | not measured |
| 24 | 3.77 | 3.20 | 905 MiB |
| 48 | 2.91 | 2.49 | 502 MiB |

One part alone (1,048,007 cells) reads in 1.08 s in one process (360 MiB). The VisIt plugin
reads the same file on an 8-rank engine with zone count and field range equal to the Python
core's. `examples/conus/` computes mesh-quality statistics under MPI and renders stereographic
projections of minimum angle and scaled Jacobian with pvbatch (16 ranks, 65 s, for the
50,226,202-cell `conus200m_p3_optimized.soma`).

## Validation

* `tests/test_core.py`, `tests/test_frontends.py`: the core against files SOMA itself wrote
  (`tests/gen/`, `soma-convert` of MOAB meshes; `tests/make_data.sh` makes them with an installed
  SOMA) with SOMA's own reader as the reference (run
  in a separate process, `tests/soma_oracle.py`): coordinates, element corners and every field
  by gid; field values against the generator's formulas; outward normals of every linear 3D
  cell and positive volume of every quadratic cell in the VTK in use; set labels; every part
  read alone; the exchange path; VTKHDF round trips with every cell's corners checked.
  64 passed, 1 skipped (VTK 9.5, macOS); 42 passed on Linux (VTK 9.7), the rest skipped for
  absent optional inputs.
  Environment: `SOMAVIEW_DATA` (ground-truth files, default `tests/data`), `SOMAVIEW_SOMA_PYTHON`
  (a SOMA build's `python/` directory; the comparisons with SOMA's own reader skip without
  it), `SOMAVIEW_NE20` (an optional real multi-part file).
* `tests/mpi/` (`validate_mpi.sh`): SOMA's MPI writer makes true multi-part files (hex, tet,
  quadrilateral meshes and AMR at 4 and 7 ranks; MOAB brick and ocean meshes;
  `tests/gen/make_mpi_test_data.cpp`); `mpi_validate.py` reads each under
  `mpiexec` at 1 to 8 ranks with both halo methods: 66 of 66 runs correct. The collective VTKHDF
  write at 4 ranks reads back with every cell correct.
* `paraview/tests/test_paraview.py`: all checks pass (ParaView 6.1.0 macOS, 6.1.1 Linux);
  pvbatch at 1, 2, 4 and 7 ranks covers every cell exactly once.
* `visit/tests/`: the C++ reader against the Python core (618 + 24 checks, both prism
  conventions; 844 on the SOMA-MPI-written files), and VisIt 3.5 itself on 16 files, serial and
  4-rank engines, plus CONUS on 8 ranks: zone counts and field ranges equal the Python core's.

## Notes for SOMA

* SOMA's own `vis.py` and `vtk_writer.cpp` write prisms in stored order, which is correct for
  VTK >= 9.7 and inverted (negative volume) for VTK <= 9.6.
* VTK pitfall: `vtkCellArray.SetData` keeps a reference to the numpy buffers only if they are
  kept alive; somaview attaches them to the VTK arrays.
