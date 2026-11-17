# SOMA

SOMA (Scalable Operators for Mesh Abstractions) is a Kokkos-native library for unstructured
meshes: build or read a mesh, distribute it over MPI ranks, keep on a device only the topology,
adjacencies and fields you ask for, and run the same kernels on GPUs or host threads. It reads and
writes its own parallel HDF5 format, MOAB `.h5m` files and legacy VTK files, and ships smoothing,
finite element and finite volume operators, conservative remap (in the plane, in volumes and on
the sphere) and adaptive refinement.

Start with the [overview](getting-started/overview.md) and [installation](getting-started/installation.md),
then the [quick start](getting-started/quickstart.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/installation
getting-started/quickstart
getting-started/examples
getting-started/python
getting-started/fortran
getting-started/from-imoab
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/mesh-model
concepts/building-meshes
concepts/topology
concepts/fields
concepts/sets
concepts/editing
concepts/runtime
```

```{toctree}
:maxdepth: 2
:caption: Parallel meshes

parallel/comm
parallel/distributed-meshes
parallel/ghosts-and-halos
parallel/matching-and-merging
parallel/reproducible-sums
parallel/transport
```

```{toctree}
:maxdepth: 2
:caption: Input and output

io/overview
io/soma-format
io/h5m
io/vtk
io/amr-checkpoints
```

```{toctree}
:maxdepth: 2
:caption: Algorithms

algorithms/smoothing
algorithms/fem
algorithms/finite-volumes
algorithms/linear-algebra
algorithms/spatial-search
algorithms/remap
algorithms/refinement
```

```{toctree}
:maxdepth: 2
:caption: Adaptive mesh refinement

amr/block-amr
```

```{toctree}
:maxdepth: 2
:caption: MOAB compatibility

compat/moab-compat
```

```{toctree}
:maxdepth: 2
:caption: Developer guide

developer/architecture
developer/building-and-testing
developer/benchmarks
developer/contributing
developer/writing-docs
developer/making-figures
developer/provenance
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cmake-options
reference/headers
reference/api/index
reference/changelog
```
