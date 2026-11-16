# Overview

This page says what SOMA is, what it can do, how it splits work between the host and a device,
and where in these pages each part is described. Read it first, then
[install SOMA](installation.md) and work through the [quick start](quickstart.md).

## What SOMA is

SOMA (Scalable Operators for Mesh Abstractions) is a C++20 library for unstructured meshes,
built on [Kokkos](https://kokkos.org) and, optionally, MPI. A mesh is read or built on the host,
distributed over MPI ranks, and mirrored to a device with only the topology, adjacencies and
fields a computation asks for. Kernels then traverse it through a device-copyable handle, the
same code running on GPUs or on host threads.

Kokkos 5.0 or newer is the only required dependency. MPI, HDF5 and the other libraries are
optional and off by default; see [installation](installation.md).

## Capabilities by area

### Mesh model

`soma::Mesh<MemSpace>` (`soma/core/mesh.hpp`) is the move-only owner of one rank's part of a
mesh in one memory space; `soma::HostMesh` and `soma::DeviceMesh` name the host and default
device instantiations. Every entity has a local id (lid) on its rank and a global id (gid)
across ranks. Each dimension is ordered by ownership first (interior, shared and owned here,
shared copies owned elsewhere, ghosts), so ownership is a comparison with the ends of a segment
table (`soma/core/segments.hpp`). Element topologies follow MOAB's numbering (`soma::Topology`
in `soma/core/topology.hpp`): edges, triangles, quads, polygons, tets, pyramids, prisms, knives,
hexes and polyhedra, with high-order elements stored MOAB's way. `soma::MeshBuilder`
(`soma/core/mesh_builder.hpp`) assembles a mesh from vertex and element blocks named by gid, in
file order or along a Hilbert curve (`Ordering::type_major_sfc`, the default). See
[the mesh model](../concepts/mesh-model.md) and [building meshes](../concepts/building-meshes.md).

### Topology

A mesh holds only the adjacency graphs it is asked for. `AdjacencyRequest` names them by
`(from, to)` pair or bridge, and `build_adjacencies` derives the edges and faces a request
touches, with downward orientation codes, from device-callable canonical side tables
(`soma/topology/`). See [topology](../concepts/topology.md).

### Fields, sets and edits

The field registry (`soma/fields/registry.hpp`, reached through `mesh.fields()`) holds
fixed-length and variable-length fields on vertices, edges, faces, cells and entity sets.
`soma::EntitySets` (`soma/core/sets.hpp`) holds a mesh's entity sets, and `soma::MeshEdit`
(`soma/core/mesh_edit.hpp`) records additions and removals and applies them. See
[fields](../concepts/fields.md), [entity sets](../concepts/sets.md) and
[editing meshes](../concepts/editing.md).

### Parallel distribution and ghosts

`soma::Comm` (`soma/parallel/comm.hpp`) is the single seam to MPI, with a serial
implementation when MPI is off. `build_local_part` (`soma/parallel/distributed.hpp`) turns each
rank's elements into the local part of a distributed mesh with its ownership and sharing;
`partition_by_curve`, `migrate` and `rebalance` move cells between ranks;
`mesh.add_ghosts(comm, GhostOptions{...})` adds ghost layers and `HaloExchange`
(`soma/parallel/halo_exchange.hpp`) updates copies and ghosts from their owners.
`match_points` and `merge_mesh` number and merge vertices by coordinates, and exact sums
(`soma/parallel/exact_sum.hpp`) give results that are the same bit for bit on any number of
ranks and threads. See the pages under *Parallel meshes*, starting with
[Comm](../parallel/comm.md) and [distributed meshes](../parallel/distributed-meshes.md).

### Input and output

- SOMA's own HDF5 format, read and written in parallel (`read_soma`, `write_soma` in
  `soma/io/soma_file.hpp`; needs `SOMA_ENABLE_HDF5`).
- MOAB `.h5m` files (`read_h5m`, `write_h5m` in `soma/io/h5m.hpp`; needs `SOMA_ENABLE_HDF5`).
- ASCII legacy VTK files (`read_vtk`, `write_vtk`), with no dependency.
- `read_file` and `write_file` (`soma/io/mesh_file.hpp`) choose the format for you: by content
  when reading, by extension when writing.
- Checkpoints of block-structured AMR hierarchies in the SOMA format, and VTK overlapping AMR
  output.
- The `soma-convert` tool converts between `.h5m` and SOMA files.

See the [I/O overview](../io/overview.md).

### Algorithms

- Laplacian smoothing of vertex coordinates (`soma/algorithms/laplacian.hpp`):
  [smoothing](../algorithms/smoothing.md).
- Linear finite elements on triangles, quads, tets and hexes (`soma/algorithms/fem.hpp`,
  `soma::FemSpace` in `soma/algorithms/fem_space.hpp`): [finite elements](../algorithms/fem.md).
- First-order upwind finite volumes (`soma::FvSpace` in `soma/algorithms/fv.hpp`):
  [finite volumes](../algorithms/finite-volumes.md).
- CSR matrices and conjugate gradients (`soma/algorithms/csr.hpp`, `soma/algorithms/cg.hpp`),
  with optional views as KokkosKernels matrices and PETSc matrices and vectors:
  [linear algebra](../algorithms/linear-algebra.md).
- Spatial search by a uniform grid (`soma::GridIndex`) or, optionally, ArborX:
  [spatial search](../algorithms/spatial-search.md).
- First-order conservative remap (`soma::RemapOperator` in `soma/algorithms/remap.hpp`)
  between plane meshes, volume meshes, and meshes on a sphere: [remap](../algorithms/remap.md).
- Uniform and marked refinement, with or without hanging vertices
  (`soma/algorithms/refine.hpp`): [refinement](../algorithms/refinement.md).

### Block-structured AMR

`soma::amr` (headers in `soma/amr/`) holds patch-based adaptive mesh refinement: boxes in index
space, levels and their halos, interpolation between levels, flux registers, clustering, and a
distributed `soma::amr::Hierarchy`. See [block AMR](../amr/block-amr.md).

### MOAB compatibility

With `SOMA_ENABLE_MOAB_COMPAT`, `soma::compat` offers MOAB's `Core` interface, ranges, tags and
sets over SOMA meshes, and a drop-in mode that puts headers at MOAB's include paths so that MOAB
applications build against SOMA unchanged. See [MOAB compatibility](../compat/moab-compat.md).

## The host and device model

SOMA keeps the full mesh on the host and copies to a device only what a computation needs:

- A `HostMesh` holds everything: topology, coordinates, adjacency graphs, fields, sets and the
  distribution. Reading, building, editing, distributing and writing happen here.
- `mesh.mirror<Space>(options)` makes a deep copy in another memory space. Its
  `soma::TransferOptions` (`soma/core/transfer.hpp`) choose the dimensions, coordinates, fields,
  adjacency graphs and sets copied; the default is lean: all dimensions, coordinates and sets, no
  fields and no adjacency graphs. Graphs are copied from the source or rebuilt on the
  destination.
- `mesh.view()` returns a `soma::MeshView` (`soma/core/mesh_view.hpp`), a device-copyable handle
  of unmanaged views. Kernels capture it by value and call its `KOKKOS_INLINE_FUNCTION` members
  (`count`, `vertex`, `x`, `gid`, `adj`, `owned`, ...).
- `Mesh` cannot be copied, because a copied Kokkos View aliases its data; deep copies are spelled
  `clone()` or `mirror<Space>()`.

A program starts with a `soma::ScopeGuard` (`soma/runtime/scope_guard.hpp`), which initialises
MPI (in MPI builds) and Kokkos and finalises them in reverse, because Kokkos forbids creating a
View before it is initialised. See [runtime](../concepts/runtime.md).

## Map of the documentation

| Section | Pages | Read it to |
| --- | --- | --- |
| Getting started | [installation](installation.md), [quick start](quickstart.md), [examples](examples.md) | build SOMA and run a first program |
| Concepts | [mesh model](../concepts/mesh-model.md), [building meshes](../concepts/building-meshes.md), [topology](../concepts/topology.md), [fields](../concepts/fields.md), [sets](../concepts/sets.md), [editing](../concepts/editing.md), [runtime](../concepts/runtime.md) | learn the data model on one rank |
| Parallel meshes | [Comm](../parallel/comm.md), [distributed meshes](../parallel/distributed-meshes.md), [ghosts and halos](../parallel/ghosts-and-halos.md), [matching and merging](../parallel/matching-and-merging.md), [reproducible sums](../parallel/reproducible-sums.md), [transport](../parallel/transport.md) | distribute meshes over MPI ranks |
| Input and output | [overview](../io/overview.md), [SOMA format](../io/soma-format.md), [.h5m](../io/h5m.md), [VTK](../io/vtk.md), [AMR checkpoints](../io/amr-checkpoints.md) | read and write files |
| Algorithms | [smoothing](../algorithms/smoothing.md), [FEM](../algorithms/fem.md), [finite volumes](../algorithms/finite-volumes.md), [linear algebra](../algorithms/linear-algebra.md), [spatial search](../algorithms/spatial-search.md), [remap](../algorithms/remap.md), [refinement](../algorithms/refinement.md) | use the numerical operators |
| AMR | [block AMR](../amr/block-amr.md) | work with patch-based hierarchies |
| MOAB compatibility | [moab-compat](../compat/moab-compat.md) | run MOAB code over SOMA |
| Developer guide | [architecture](../developer/architecture.md), [building and testing](../developer/building-and-testing.md), [benchmarks](../developer/benchmarks.md), [contributing](../developer/contributing.md), [provenance](../developer/provenance.md) | work on SOMA itself |
| Reference | [CMake options](../reference/cmake-options.md), [headers](../reference/headers.md), [changelog](../reference/changelog.md) | look up options and headers |
