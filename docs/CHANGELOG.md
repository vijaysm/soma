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

### VTK files

SOMA never links VTK; it writes and reads legacy VTK files itself.

- `write_vtk` writes a `HostMesh` as an ASCII legacy VTK unstructured grid, POSIX only and with
  no other dependency, in one of two layouts chosen by `VtkWriteOptions::version`:
  `VtkLegacyVersion::v5_1` (the default), the VTK 9 layout with `OFFSETS`/`CONNECTIVITY
  vtktypeint64` and the `vtktypeint64`/`vtktypeuint64` array names, or `VtkLegacyVersion::v3_0`
  with a vertex count per cell and `long`/`unsigned_long`. The default was 3.0 before; pass
  `{VtkLegacyVersion::v3_0}` for the old bytes, which are unchanged. Both contain cells of the
  mesh dimension, and `per_entity` fields over full domains as `FIELD` arrays in `POINT_DATA`
  and `CELL_DATA`. Subset, stale and uncommitted fields, and
  `knife` and `polyhedron` elements, are rejected. Output goes through a temporary file and a
  rename, and is byte-for-byte deterministic.
- The `write_vtk` overloads gained a trailing `VtkWriteOptions` parameter. Calls compile
  unchanged, but taking the address of an overload with its old signature no longer does.
- `i8` fields are written as `signed_char` in version 5.1, so a file `read_vtk` read with `char`
  or `signed_char` arrays writes back. Version 3.0 still rejects them: widen them to `i32`.
- The temporary file has a unique name next to the target and is created exclusively, so an
  existing `<path>.tmp` or a symlink there is never touched and concurrent writers do not
  collide. The target is replaced by a new file whose mode comes from the umask; nothing is
  fsync'ed. I/O errors name the path and the cause. The temporary name adds 7 bytes, so a file
  name of 249 to 255 bytes fails where names are limited to 255 bytes.
- NaN, infinite and subnormal values, in fields or coordinates, throw `std::invalid_argument`
  before any file is touched; VTK readers stop at them.
- In version 3.0 only, more than `INT32_MAX` points, or a `CELLS` section of more than
  `INT32_MAX` integers, throws `std::length_error`: those readers read the counts as `int`.
  Version 5.1 has no such limit.
- A mesh whose Views do not match its counts (a mirror without coordinates or vertices), or
  whose connectivity offsets do not start at 0, throws `std::invalid_argument` instead of
  reading past the end of a View or writing an inconsistent `CELLS` section.
- Wedges follow the VTK 9.7 vertex order. Older readers (ParaView 6.1 and earlier) show them
  with inverted orientation.
- Needs floating-point `std::to_chars` (libstdc++ 11+, libc++ 14+, macOS 13.3+).
- `read_vtk` reads ASCII legacy unstructured grids of versions 2.0, 3.0, 4.x and 5.1 (both cell
  layouts) into a `HostMesh` built in file order: point gids 1..n and cell gids 1..n per
  dimension in file order, the mesh dimension being the largest cell dimension. `FIELD`,
  `SCALARS`, `VECTORS`, `NORMALS`, `TENSORS`, `TENSORS6` and `TEXTURE_COORDINATES` arrays of
  `POINT_DATA` and `CELL_DATA` become committed `per_entity` tags, placed by gid; `char` maps to
  `i8`, `short` to `i32` and `unsigned_short` to `u32`. Dataset field data, lookup tables, color
  scalars and `METADATA` blocks are skipped. BINARY files, other datasets, unknown cell types,
  `CELL_DATA` over cells of several dimensions, files without cells, non-finite coordinates and
  malformed or inconsistent sections throw `std::runtime_error` naming the file and line.
- `GLOBAL_IDS`, and `PEDIGREE_IDS` of an integer type, become `i64` tags keeping the array name;
  ids of a real type throw. String and variant arrays (string `PEDIGREE_IDS`, and string or
  variant arrays in any `FIELD` block) are skipped, so files carrying labels read without them.
- `char` arrays take values from -128 to 255 and wrap 128..255 to negative `i8`, as VTK writes a
  `char` array where `char` is signed; `signed_char` keeps -128..127. Integer values of `OFFSETS`
  and `CONNECTIVITY` must fit their declared type.
- Polyhedra (cell type 42) throw "polyhedron cells are not supported"; point ids are checked once
  the cell types are known, and a bad id names the cell and the id. Polygons have 3 to 65535
  vertices. A section holds at most 1024 arrays. `%00` in names, a sign after a sign (`+-5`) and
  counts beyond `int64` throw; quoted tokens in messages are cut to 64 characters. Errors from the
  mesh builder or the field registry are rethrown as `std::runtime_error` naming the file.
- Not checked, as in VTK: degenerate and duplicate cells are read as they are, and an array named
  `METADATA` is taken for a `METADATA` block. Wedges in files written by VTK 9.6 or older come in
  with inverted orientation, since the reader follows the VTK 9.7 vertex order. Files without
  cells, which `write_vtk` can write, are rejected: a mesh needs a cell dimension.
- `data/vtk_reference/` holds files written by ParaView 6.1.0-RC1 (VTK 9.6) in versions 5.1 and
  4.2, and the script that wrote them; the reader tests check them value by value.

### Mesh

- `soma::Mesh<MemSpace>` is the move-only owner of one rank's mesh: per-dimension counts,
  CSR connectivity, topology, global ids and segment tables, plus coordinates, an epoch and a
  field registry. Copies are spelled `clone()`, `mirror<Space>()` and `mirror_view<Space>()`;
  accessors reject a dimension outside [0, 3], and `view()` is deleted on temporaries.
- `soma::MeshView` is the device-copyable kernel handle of unmanaged Views.
- `soma::TransferOptions` selects the dimensions, coordinates and fields a mirror copies.
  Keeping elements or coordinates without the vertices is rejected.
- `soma::MeshBuilder` assembles a mesh from vertex and element blocks given by global ids,
  validates connectivity, polygon offsets, arity and dimensions, and orders every dimension.
  A NaN or infinite vertex coordinate is rejected, naming the vertex gid, under every ordering.
- `Ordering::type_major_sfc` orders vertices and elements along a Hilbert curve of their
  coordinates and centroids within each topology and arity. It is the default ordering;
  `Ordering::file` keeps gid order. The curve uses 32 bits per axis over one or two axes and
  21 over three. Every axis is scaled by the widest extent, so the curve follows the shape
  of the mesh and a thin or noisy axis is not stretched over the whole grid. An axis no
  wider than one cell of the widest axis is left out, so noise off a plane does not change
  the order, and boxes whose extent overflows still spread over the grid.
- `soma::vertex_span_report(mesh, dim)` gives the mean and largest vertex span (max - min
  vertex local id) of the elements of one dimension, a host-side measure of how well an
  ordering keeps the vertices of an element close in memory.
- `Domain::all`, `Domain::segments` (with `SegmentFilter` on topology, arity and ownership)
  and `Domain::from_list` take a mesh, and `make_space` builds spaces from the mesh topology.

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
  A length whose product with its cell count is not finite is rejected, naming the axis.

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
