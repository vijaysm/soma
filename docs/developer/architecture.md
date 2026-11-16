# Architecture

How SOMA is organised: its directories and how they depend on each other, which code runs on the
host and which on a device, how collectives and errors behave across ranks, what SOMA guarantees
about determinism, and how it is tested. Read this before changing the library; users may find
the sections on collectives and determinism useful too.

## Layout of the repository

| Path | Contents |
| --- | --- |
| `include/soma/<dir>/` | the public headers, one directory per component (listed in [public headers](../reference/headers.md)) |
| `src/<dir>/` | the compiled sources, mirroring `include/soma/` |
| `tests/` | the test suite: `unit/`, `reference/`, `golden/`, `compat/`, `policy/`, `packaging/` |
| `bench/` | benchmarks, built with `SOMA_ENABLE_MOAB_REFERENCE` (see [benchmarks](benchmarks.md)) |
| `examples/` | example programs, built with `SOMA_ENABLE_EXAMPLES`, and `examples/app/`, a standalone project built against an installed SOMA (see [examples](../getting-started/examples.md)) |
| `tools/` | command-line tools (`soma_convert`), built with `SOMA_ENABLE_HDF5` |
| `cmake/` | find and fetch modules, the install rules and `config.hpp.in` |
| `data/` | small meshes used by tests and examples |
| `docs/` | this documentation and `CHANGELOG.md` |

CMake builds one library, `soma` (alias `soma::soma`), from every directory under `src/` except
`compat/`. With `SOMA_ENABLE_MOAB_COMPAT` it also builds `soma_compat` (`soma::compat`) on top of
`soma`, and the interface target `soma_compat_dropin` (`soma::compat_dropin`). Kokkos is the only
required dependency; every optional one is off by default, and a requested dependency that cannot
be found is a configure error (see [CMake options](../reference/cmake-options.md)).

## Components and their dependencies

The components, from the bottom up, with the other components their public headers include:

| Component | Namespace | Includes headers of | Role |
| --- | --- | --- | --- |
| `runtime` | `soma` | none | process start-up (`ScopeGuard`) and process-wide options |
| `core` | `soma` | `topology`, `fields` | the mesh: `Mesh`, `MeshView`, segment tables, builders, edits, sets |
| `topology` | `soma`, `soma::topology`, `soma::canonical` | `core` | canonical numbering, sub-entity derivation, adjacency graphs |
| `fields` | `soma` | `core` | domains, spaces, layouts and the field registry |
| `parallel` | `soma`, `soma::transport` | `core`, `topology` | `Comm`, the transport layer, distributed meshes, ghosts, halos, matching and merging |
| `io` | `soma`, `soma::io` | `core`, `parallel`, `amr` | the SOMA format, `.h5m`, legacy VTK, AMR checkpoints |
| `algorithms` | `soma`, `soma::fem` | `core`, `topology`, `parallel`, `runtime` | smoothing, finite elements and volumes, linear algebra, search, remap, refinement |
| `amr` | `soma::amr` | `parallel` | patch-based AMR on integer boxes |
| `testing` | `soma::testing` | `core` | structured box meshes for tests |
| `compat` | `soma::compat` | `core` | MOAB's API over a SOMA host mesh |

`core`, `topology` and `fields` form the data model together: a `Mesh` owns its adjacency store
(`soma/topology/adjacency.hpp`) and its field registry (`soma/fields/registry.hpp`), and
`soma/core/mesh.hpp` includes `soma/topology/build.hpp` at its end, so including the mesh brings
in all three. `parallel` builds on the data model; `io`, `algorithms` and `amr` build on
`parallel`. The AMR layer does not use the unstructured mesh at all: it needs only `Comm` and the
transport layer's `CommPlan`. `compat` sits on top of everything, using the mesh, its edits and
the readers and writers.

### Builds with and without optional dependencies

The data model and every algorithm are the same in every build. Optional dependencies change which
sources are compiled, not which functions exist:

- **MPI.** `soma/parallel/comm.hpp` is the single communication seam. Algorithms call `Comm`
  unconditionally; without MPI it is a single-rank identity (rank 0 of 1), compiled from
  `src/parallel/comm_serial.cpp` instead of `comm_mpi.cpp`. That header, `src/parallel/comm_*.cpp`
  and `src/runtime/scope_guard.cpp` are the only files allowed to test `SOMA_ENABLE_MPI`; the test
  `policy.mpi_seam` enforces it.
- **HDF5.** Without `SOMA_ENABLE_HDF5` the SOMA-format and `.h5m` functions still exist but throw
  (`read_h5m` and `write_h5m` throw and `h5m_available()` is false; the SOMA-format functions
  throw `std::runtime_error`).
- **KokkosKernels, PETSc, ArborX.** Each adds a header's worth of functionality
  (`soma/algorithms/kokkos_sparse.hpp`, `petsc.hpp`, `arborx_index.hpp`) and nothing else.

The generated `soma/config.hpp` records the options that change the ABI or the public headers,
among them `SOMA_LO_64` and `SOMA_OFFSET_64` (64-bit local ids and CSR offsets) and `SOMA_DEBUG`
(device-side bounds and contract checks). Because the ordinal widths are ABI-breaking, the
installed package is compatible within a minor version only.

## Host and device code

SOMA is written against Kokkos and runs the same kernels on host threads or a GPU.

- **Memory spaces.** A `Mesh<MemSpace>` owns one rank's mesh in one memory space; `HostMesh` is
  `Mesh<Kokkos::HostSpace>` and `DeviceMesh` the default execution space's memory space. Copies are
  explicit (`clone()`, `mirror<Space>(TransferOptions)`), because copying a Kokkos view aliases.
  Every SOMA view is `LayoutLeft` (`soma/core/view_types.hpp`), so host and device types differ only
  in memory space and mirrors assign without conversion.
- **Host-side work.** Building, editing and distributing meshes happen on the host: `MeshBuilder`
  produces a `HostMesh`, `MeshEdit` edits a host mesh, and the parallel construction functions
  take `HostMesh&`. A device mirror then carries only what `TransferOptions` names (by default all
  topology dimensions, coordinates and sets; no fields and no adjacency graphs).
- **Kernels.** Kernels take device-copyable handles, never the owning objects: `MeshView` for a
  mesh (unmanaged views, valid while the mesh is alive and unchanged in topology), `Field` and
  `VarField` for fields, `LevelView` for AMR data. Type erasure exists only in the host-side field
  registry; kernels receive concretely typed views. Small geometric kernels (canonical tables,
  sides, polygon and polyhedron clipping, `amr::Box`) are written to be callable in device code,
  with fixed capacity and no allocation.
- **Plans.** Communication patterns are computed once on the host and replayed: `HaloExchange`,
  `transport::CommPlan`, and the AMR `CopyPlan`, `HaloPlan`, `CoarseFinePlan`,
  `AverageDownPlan` and `FluxRegister`. Plans keep device copies of what their kernels read.
- **Messages and GPUs.** Whether device buffers are passed to MPI is set at configure time by
  `SOMA_GPU_AWARE_MPI` (on by default with MPI and a GPU backend) and can be turned off at run time
  with the environment variable `SOMA_GPU_AWARE_MPI=0`. The AMR copies stage their messages
  through host memory.

## Collectives and errors

Functions that communicate are documented as collective in their header comments, and every rank
of the communicator must call them, in the same order. Where several non-blocking operations of
one plan are in flight together (a split AMR fill, for instance), they must begin in the same
order on every rank, as MPI collectives must.

The error policy follows from that: a collective that detects bad input on one rank must not leave
the other ranks waiting in a later collective. Collective functions therefore check their input,
agree on the outcome, and throw on every rank. The header comments say so explicitly, for example:

- `transport::route` and `transport::rendezvous` throw `std::invalid_argument` on every rank if
  any rank passes bad input;
- `resolve_sharing`, `build_local_part`, `migrate`, `ghost_targets`, `match_points` and
  `merge_mesh` throw on every rank, with the conditions listed in their headers;
- `write_soma` checks everything on every rank before touching a file and throws on every rank,
  and `amr::write_vtk_amr` with a communicator does the same.

Exceptions are standard: `std::invalid_argument` for bad input, `std::runtime_error` for failures
of the environment (files, MPI), `std::logic_error` for calls in the wrong state, and a few
specific types derived from them (`FieldClash`, `FieldOverlap`, `StaleField`,
`MissingAdjacency`, `MissingSetMembers`). Many operations give the strong guarantee and say so; for
example `build_adjacencies` builds on a working copy and installs it with non-throwing moves, a
throw from `MeshEdit::apply` leaves the mesh unchanged, and an AMR regrid that throws leaves the hierarchy
and its fields unchanged.

## Determinism

SOMA aims for results that do not depend on the number of threads, and where stated, not on the
number of ranks either. The guarantees are documented per function; the main ones are:

- **Topology.** Every adjacency graph is a deterministic function of the topology, whatever the
  backend or thread count: upward graphs are built with atomics, then each row is sorted.
  Sub-entities are numbered by the lowest-creator rule, without hashing.
- **Sums.** `ExactSums` adds doubles exactly into a fixed-point accumulator and rounds once, so the
  same terms give the same bits on any number of threads and ranks
  ([reproducible sums](../parallel/reproducible-sums.md)).
- **Exchanges.** `HaloExchange::reduce` combines in ascending rank order, so every holder computes
  the same result bit for bit, whatever the operation.
- **Sparse algebra.** A `CsrMatrix` keeps each row's columns ascending by gid, so a product is the
  same bit for bit on any number of ranks and threads; `laplacian_smooth` sums neighbours in gid
  order.
- **Reproducible mode.** The process-wide `reproducible` option (`SOMA_REPRODUCIBLE=1` or
  `--soma-reproducible`) makes the algorithms that can give bit-for-bit results on any number of
  ranks and threads do so at some cost, for example `FemSpace`'s dot products and `cg`, which then
  sum exactly ([runtime](../concepts/runtime.md)).
- **Partitions and numbering.** `partition_by_curve` depends only on the elements and the rank
  count; `transport::rank_keys` with distinct keys gives indices that do not depend on the rank
  count; the rule that makes ghost layers does not depend on the partition.
- **AMR.** Copies and fills into one box are disjoint, so they give the same result at any thread
  count; a distributed regrid makes the boxes one process would.

The test `test_cg.threads` checks that reproducible CG gives the same digests at different thread
counts.

## Testing strategy

Tests use GoogleTest and CTest (`SOMA_ENABLE_TESTS`); see
[building and testing](building-and-testing.md) for how to run them. Every test executable links a
shared `main` (`tests/soma_test_main.cpp`) that holds a `ScopeGuard` and reduces the result over
all ranks, so a failure on any rank fails the run on every rank.

| Directory | What it checks |
| --- | --- |
| `tests/unit/` | each component on its own; tests that communicate are registered once per rank count (typically 1, 2, 3, 4, 5, 7 and 8) |
| `tests/reference/` | SOMA against MOAB as the reference implementation (`SOMA_ENABLE_MOAB_REFERENCE`): canonical tables, node tables, adjacencies, ghosts, `.h5m` files, distributed meshes against `ParallelComm`, smoothing, finite elements and volumes, remap, refinement, and the compat layer |
| `tests/golden/` | the parallel sub-entity derivation against a serial reference derivation |
| `tests/compat/` | the MOAB example gate: MOAB examples built against MOAB and against `soma_compat_dropin` must print and write the same |
| `tests/policy/` | rules on the source tree: `SOMA_ENABLE_MPI` is tested only in the communication seam (`policy.mpi_seam`); test bodies hold no device lambdas, which CUDA refuses there (`policy.device_lambdas`); a parallel HDF5 without MPI is refused at configure time (`policy.hdf5_guard`) |
| `tests/packaging/` | an install into an empty prefix and an out-of-tree consumer that finds SOMA only through `find_package` (`packaging.find_package`); the same for `examples/app/` (`packaging.example_app`) |

Reference tests compare results that both libraries define, matched in a way that does not depend
on numbering where the numberings may differ (for example, entities matched by their coordinates).
MOAB is used only by tests and benchmarks, never by the library.
