# Changelog

All notable changes to SOMA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Build

- Kokkos >= 5.0 is the only required dependency. MPI, GoogleTest, MOAB and the other
  libraries are optional, OFF by default, and a configure error when requested but missing.
- `CMAKE_BUILD_TYPE` defaults to `RelWithDebInfo` for single-configuration generators.
- `SOMA_FETCH_KOKKOS=ON` downloads and builds Kokkos when no installed copy is found. An
  installed Kokkos always wins.
- The fetched Kokkos enables Serial plus OpenMP, or Threads when OpenMP is unusable, and
  `Kokkos_ARCH_NATIVE`. Any `Kokkos_*` option the user sets is respected.
- `SOMA_KOKKOS_GPU` selects the GPU backend of a fetched Kokkos (`AUTO`, `NONE`, `CUDA`,
  `HIP`, `SYCL`). A request the compiler cannot build is a configure error, never a silent CPU
  fallback.
- `SOMA_GPU_AWARE_MPI` (`AUTO`, `ON`, `OFF`) sets the compile-time default for passing device
  buffers to MPI. The generated `soma/config.hpp` also honors the environment variable
  `SOMA_GPU_AWARE_MPI=0|1` at run time.
- On AppleClang, SOMA seeds `FindOpenMP` from Homebrew libomp so that Kokkos with its OpenMP
  backend can be found.
- The configure summary line reports the Kokkos source, host and GPU backends, MPI, tests,
  GPU-aware MPI and build type.
- `-Wall -Wextra` are applied to SOMA targets for compilers that accept them.
- CPM is at 0.43.2.

### Runtime and Comm

- `soma::Comm` is the single seam to MPI, with a serial implementation and an MPI
  implementation. It provides `rank`, `size`, `barrier`, `allreduce`, `exscan` and `bcast`.
- Reductions are defined for `std::int32_t`, `std::int64_t`, `std::uint64_t` and `double`, for
  `Op::sum`, `min`, `max` and `prod`, on one value or a span. Other types fail to link.
- `Comm::self()` gives a single-rank communicator.
- MPI errors carry the MPI error string, and counts above `INT_MAX` are rejected.
- `soma::ScopeGuard` initializes MPI and then Kokkos, and tears them down in reverse order.
- The fixed-width types `LO`, `Offset`, `GO` and `Real` are defined in `soma/core/types.hpp`.
  `SOMA_LO_64` and `SOMA_OFFSET_64` select 64-bit widths.

### Topology

- A `Topology` enum whose values equal MOAB's `EntityType` and the `.h5m` element type
  numbers, so a reader can map element groups without a translation table.
- Constexpr, device-callable canonical side tables for edges and faces of every fixed-arity
  topology, with vertex order that encodes orientation.
- Variable-arity topologies (polygon, polyhedron) have no table. The tables return 0 sides.

### Node tables

- Reference-element node coordinates for edge, triangle, quad, tetrahedron, prism, hex and
  pyramid, as equispaced or GLL sets up to order 32, with `tensor_lexicographic`,
  `hierarchical` and `moab_native` orderings, serendipity sets at order 2, permutations between
  orderings, and a cache of device Views.

### Testing

- GoogleTest harness with a shared `main` that creates the runtime scope first and agrees on
  the exit code across ranks. `soma_add_test` registers MPI tests at 1, 2 and 4 ranks with a
  60 second timeout.
- `SOMA_FETCH_GTEST=ON` downloads GoogleTest. Without it, a missing GoogleTest is a clear
  configure error.
- Smoke tests for Kokkos initialization, a parallel reduction on the default execution space,
  and a check that the default execution space is the expected host backend.
- Comm tests cover every reduction operation and type, exclusive scan, broadcast from any root,
  and the single-rank communicator, at one, two and four ranks.
- Canonical table tests check face orientation consistency, index ranges, the absence of
  variable-arity tables, the enum values, and evaluation on the default execution space
  against the host.
- With `SOMA_ENABLE_MOAB_REFERENCE=ON`, the tables are compared position by position with
  `moab::CN`, and the polygon behavior of MOAB is pinned.
- A policy test fails if `SOMA_ENABLE_MPI` is tested outside the Comm seam.
- Every test has an injected-fault record in `docs/negative_controls.md`.
- `soma::testing::box_mesh` builds structured hex, quad and triangle boxes with closed-form
  counts, gids, coordinates and orientation, so tests do not depend on a file or a mesher.

### Packaging

- `cmake --install` installs the library, headers, generated `config.hpp`, and the package
  `soma` with the target `soma::soma`. The version is compatible within a minor version.
- With a fetched Kokkos, Kokkos is installed into the same prefix, and the installed config
  pins `Kokkos_DIR` to it, also after the prefix is relocated.
- The installed config replays the AppleClang libomp settings, so consumers configure without
  rediscovering OpenMP.
- `SOMA_INSTALL` defaults to ON only when SOMA is the top-level project.
- An out-of-tree packaging test installs to a clean prefix, builds a consumer, and runs it (at
  two ranks for MPI builds). It also works with multi-config generators through `ctest -C`.

### Documentation

- `docs/LESSONS_FROM_MOABX.md` lists the lessons carried over from moabX and those learned in
  SOMA.
- `docs/remote_builds.md` describes configuring and testing on NVIDIA, AMD and Intel GPUs.
- `docs/negative_controls.md` records each injected fault and what it showed.
- `docs/CONTINUITY.md` records the verified state and how to resume.
