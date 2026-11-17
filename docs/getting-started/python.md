# Python

The Python package `soma` runs SOMA from scripts: build or read a mesh, edit it, smooth or refine
it, remap fields between meshes, look at it and write it, on any number of ranks under `mpiexec`.
It wraps the C++ library with [nanobind](https://nanobind.readthedocs.io); the work is SOMA's, and
the arrays a mesh hands out share its memory.

This release runs on the host Kokkos backends (Serial, OpenMP, Threads); device memory is not
exposed to Python.

## Building and installing

The bindings need Python 3.9 or newer with nanobind and NumPy, and, in a build with MPI, mpi4py
built against the same MPI library as SOMA (the import checks this and refuses a mismatch):

```bash
MPICC=/path/to/mpi/bin/mpicc python -m pip install --no-binary mpi4py mpi4py
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON \
  -DSOMA_ENABLE_PYTHON=ON -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

The package is then importable from the build tree with `PYTHONPATH=build/python`.
`cmake --install build` puts it under the install prefix, in `lib/python3.X/site-packages/soma`
(add that directory to `PYTHONPATH`); to install into the interpreter's own `site-packages`, set
`SOMA_PYTHON_INSTALL_DIR` to it, for example
`-DSOMA_PYTHON_INSTALL_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('platlib'))")`. pyvista is optional; only
`soma.to_pyvista` needs it.

## Running under MPI

Every rank runs the same script, and calls named *collective* below must be made by every rank:

```python
from mpi4py import MPI
import soma

comm = soma.Comm(MPI.COMM_WORLD)
mesh = soma.box_mesh(comm, 8, 8, 8, element=soma.Topology.tet)
print(comm.rank, mesh.owned_count(3))
```

```bash
mpiexec -n 4 python script.py
```

`soma.Comm` wraps an mpi4py communicator (any one, split or duplicated) and keeps it alive;
`soma.Comm.world()` and `soma.Comm.self()` are shortcuts. A build without MPI accepts only
`soma.Comm.self()`.

## What a script can do

The operations below are the same set the C++ library offers and a later Fortran interface will
offer; the C++ pages describe each one in depth.

| Group | Python | C++ |
|---|---|---|
| Build | `MeshBuilder(dim)`, `.add_vertices(gids, xyz)`, `.add_elements(topology, gids, vertex_gids)`, `build_local_part(comm, builder)` | [Building meshes](../concepts/building-meshes.md) |
| Boxes | `box_mesh(comm, nx, ny, nz=1, element=Topology.hex, lx=1, ly=1, lz=1)`, split in layers across the ranks | `soma::testing::box_mesh` |
| Query | `mesh.dim`, `count(d)`, `owned_count(d)`, `interior_count(d)`, `segments(d)`, `coords()`, `conn(d)`, `conn_offsets(d)`, `gids(d)`, `topology(d)`, `owner_ranks(comm, d)`, `build_adjacencies(comm, pairs)` | [The mesh model](../concepts/mesh-model.md) |
| Edit | `MeshEdit(mesh)`: `add_vertices`, `add_elements`, `add_polygons`, `remove`, `apply()`; `mesh.compress()` | [Editing meshes](../concepts/editing.md) |
| Fields | `mesh.add_field(name, dim, ncomp=1, dtype)`, `field(name, dim=None)`, `fields()`, `remove_field(name)` | [Fields](../concepts/fields.md) |
| Files | `read_file(comm, path)`, `write_file(comm, mesh, path, vtk_fields=None)`, `detect_format(path)`, `convert_mesh_file(comm, src, dst)` | [Reading and writing meshes](../io/overview.md) |
| Parallel | `partition_by_curve`, `migrate`, `rebalance`, `mesh.add_ghosts(comm, layers, bridge_dim)`, `remove_ghosts`, `HaloExchange(comm, mesh, dim).update / .reduce` | [Distributed meshes](../parallel/distributed-meshes.md), [Ghosts and halo exchange](../parallel/ghosts-and-halos.md) |
| Smoothing | `laplacian_smooth(comm, mesh, iterations, bridge_dim, tolerance, fixed)` | [Laplacian smoothing](../algorithms/smoothing.md) |
| Refinement | `refine_uniform(comm, mesh, levels)`, `refine_marked(comm, mesh, marks)` | [Mesh refinement](../algorithms/refinement.md) |
| Remap | `RemapOperator(comm, source, target, ...)`, `.apply(comm, u_source, u_target)` | [Conservative remap](../algorithms/remap.md) |
| Looking | `to_pyvista(mesh, fields=None, owned_only=True)` | (Python only) |

Each call's docstring (`help(soma.laplacian_smooth)`) gives its arguments. C++ errors arrive as
Python exceptions with SOMA's message: `ValueError` for invalid arguments (most of them),
`IndexError` for out-of-range lookups, and `soma.Error` (a `RuntimeError`) otherwise. A collective
call that fails on one rank fails on every rank where the C++ call is collective-safe, as the file,
migration and ghost calls are.

## Arrays and their lifetime

`coords()`, `gids(d)`, `conn(d)`, `conn_offsets(d)`, `topology(d)` and every field array are NumPy
arrays over the mesh's own memory: no copy is made, and writing into `coords()` or a field moves
the mesh or changes the field. Coordinates and fields are writable, the rest read-only. Two-index
arrays (`coords()`, fields) are column-major, as SOMA stores them.

An array keeps its memory alive by itself: it stays valid after the mesh is deleted. Calls that
change a mesh in place (`MeshEdit.apply`, `compress`, `add_ghosts`, `remove_ghosts`) give it new
storage; arrays taken before still read the old values. Take arrays again after such a call.
`migrate`, `rebalance`, `refine_uniform` and `refine_marked` return new meshes and leave their
input as it was.

Two calls return copies: `owner_ranks` (SOMA keeps ownership as ranges, not per entity) and
`to_pyvista`.

At exit the module finalizes Kokkos. Meshes and arrays still alive then are left to the operating
system rather than freed, so a script may end with meshes in global variables.

## Looking at a mesh

`soma.to_pyvista(mesh)` gives the rank's part as a `pyvista.UnstructuredGrid`, with fields on
vertices as point data and fields on elements as cell data:

```python
grid = soma.to_pyvista(mesh, fields=["u"])
grid.plot(scalars="u")
```

For the whole mesh of a parallel run, write it with `write_file(comm, mesh, "out.vtk")` (gathered
to one file) or `"out.soma"` and open it in ParaView.

## A complete script

`python/examples/pipeline.py` builds a box of tets, perturbs and smooths it, remaps a linear field
to a second box and writes both meshes. `examples/pipeline.cpp` takes the same steps in C++, and
the test `python.pipeline` runs both at one and three ranks and requires the same digests of the
coordinates and remapped values, to the last bit:

```bash
mpiexec -n 4 python python/examples/pipeline.py out
```
