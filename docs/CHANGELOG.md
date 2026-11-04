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
- `AdjacencyRequest` names graphs by `(from, to)` pair or bridge `(dim, via)`, 32 slots in a
  `uint32_t`; a request from a dimension to itself is rejected. `(d, 0)` is the connectivity:
  always present, never stored.
- The mesh holds only the graphs it was asked for. `Mesh::adjacency`, `bridge` and
  `adjacencies()` read them on the host and throw `MissingAdjacency` for one it does not hold;
  `MeshView::adj`, `adj_codes` and `bridge` read them in kernels (under `SOMA_DEBUG` a missing
  graph aborts).
- Downward codes are `(offset << 1) | reversed` of the stored sub-entity against the parent's
  canonical side, the semantics of MOAB `CN::SideNumber`. Upward codes are the local side
  index in the parent, so a row longer than 256 throws `std::length_error`. Bridges have none.
  Downward rows follow the side order; upward and bridge rows are sorted.
- `build_adjacencies(mesh, request[, exec])` derives the edges and faces a request touches and
  builds every graph it depends on, keeping only the requested ones. Results are equal bit for
  bit on Serial, OpenMP and Threads at any thread count. A failure leaves the mesh unchanged;
  a repeated request does nothing.
- Edges and faces come from the lowest-creator rule over a sorted vertex-to-element transpose:
  no hash table, no global sort. They are numbered by (topology, arity), then by (creator
  element, side), so segments are contiguous and follow the element order. Polygon edges
  follow the vertex cycle.
- Explicit edges and faces in the input are adopted: the derived entity with the same vertex
  set keeps the explicit gid and vertex order, and new ones are numbered from the largest
  explicit gid + 1. An explicit entity that is no side of an element, that repeats another's
  vertices, or whose vertices are no cycle throws `std::invalid_argument` naming the gid.
  Deriving on a mesh with vertices but no elements throws.
- Deriving a dimension bumps the mesh epoch (every field goes stale) only when it renumbers
  explicit entities or a live field is defined on it; `FieldRegistry::uses_dimension` tells.
- CSR scans sum in 64 bits and throw `std::length_error` when a total does not fit `Offset`.

- `soma/topology/nodes.hpp`: MOAB's mid-node configurations. `mid_node_bits(t, count)` gives
  the mid-edge (2), mid-face (4) and mid-region (8) bits of a node count, as `CN::HasMidNodes`,
  or -1 for a count that is none; `num_nodes(t, bits)` the count; `node_parent(t, count, k)` the
  sub-entity node k sits on, as `CN::HONodeParent`. Checked against MOAB for every type and
  every count up to 27 (35 configurations beyond the linear ones).
- Derivation keeps the nodes of the high-order sides it adopts (a Quad8 face of a hex);
  derived sides have none.

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
- High-order elements, MOAB's way: the connectivity keeps the corners, and each element's other
  nodes (mid-edge, mid-face, mid-region, in MOAB's order) are vertices listed in
  `EntityBlock::node_offsets`/`nodes` (`Mesh::has_nodes`, `node_offsets`, `nodes`;
  `MeshView::num_nodes` and `node`, corners first). Vertices that are nodes and no element's
  corner are numbered after the others, so `[0, Mesh::corners())` are the corners; a hanging
  node that is another element's corner stays among them. Neighbours need not agree on a
  shared side's nodes. Mirrors, clones and moves carry the nodes.
- `MeshBuilder::add_elements(t, gids, node_gids, nodes_per_element)` takes the corners then
  the other nodes of each element; a count that is no configuration of `t`, or an unknown node
  gid, throws. Elements are ordered by (topology, arity, node count, key, gid), so each node
  count is contiguous within its segment; the curve key is the corners' centroid.
- `Domain::all`, `Domain::segments` (with `SegmentFilter` on topology, arity and ownership)
  and `Domain::from_list` take a mesh, and `make_space` builds spaces from the mesh topology.

- `TransferOptions::adjacencies` selects the graphs a mirror holds: copied from the source
  (`AdjSource::copy`) or built on the destination (`AdjSource::rebuild`, the default). A graph
  needs both of its dimensions in `dims`. `clone()` copies every held graph. A memory space
  without a default execution space compiles, and asking it to rebuild throws.

### Entity sets and value kinds

- `soma::EntitySets` holds a mesh's sets: per-slot sorted members (entities of dimensions 0 to
  3 and other sets), MOAB flags (`set_track_owner`, `set_unique`, `set_ordered`), the
  sequence of an ordered set with its repeats, children and parents. `members_recursive`
  follows contained sets, cycles included, once. `SetsView::contains` is a binary search in
  kernels. `SetBuilder` validates and installs sets with the strong guarantee.
- Sets are the fifth field slot (`set_dim = 4`): one tag stays one family across entities and
  sets, and `Domain::all`, `from_list` and the new `Domain::from_set` work on it. Deriving
  entities renumbers set members with them.
- Mirrors and clones copy sets by default (`TransferOptions::sets`); members of a dimension a
  mirror leaves out throw `MissingSetMembers` when asked for.
- Each set row takes the smallest of the forms of a domain block: every entity of the slot
  (nothing stored), runs (2 runs + 1 ids) or the sorted list. `SetRow` reads a row in place on
  the host and in kernels: `r[k]`, `r.size`, `row(lid)`, `contains(lid)` and an iterator over
  the runs; `members_of` returns it, and `kinds`/`offsets`/`data`/`entities` expose the storage.
  `SetBuilder::add(spec, runs)` takes `SetRun{slot, first, count}` and packs rows without listing
  them. `Domain::from_set` takes a row's runs as they are. The forms, their choice and their
  search (`soma/core/lid_runs.hpp`) are shared with domains.
- `VarField` is a variable-length field: a CSR over a `per_entity` space whose lengths are
  fixed when `FieldRegistry::add_var` adds it, fetched with `get_var`, mirrored like any field.
  It has a family of its own; `get` and `get_var` refuse each other's kind.
- `FieldOptions::handle` marks a `GO` field (fixed or variable-length) whose values are handles
  of entities or sets, 0 for none; the mark is part of the family key. A handle is
  `make_handle(slot, gid)`, the slot above bit 60 and the gid below, split by `handle_slot` and
  `handle_gid`: gids are unique within a slot only, as MOAB ids are within a type. `HandleRef`
  resolves handles to (slot, local id) on the host and in kernels, on every mesh, derived edges
  and faces included; it refuses a gid repeated within a slot or outside [0, 2^60).
- `FieldRegistry::get<T>(name, space)` returns only fixed-length fields and throws when several
  of one name and type, differing by component count or handle mark, share the space.
- `Mesh::values()` holds `MeshValue`s, values of the mesh as a whole with their type,
  component count, handle mark, MOAB storage class, default and global bytes. Every mirror
  copies them.
- `write_vtk` rejects variable-length fields.

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
- Topology tests check closed forms on boxes (vertex valence; 7208 edges, 6656 faces and 1024
  boundary faces on 8 x 16 x 16; stencil sizes 26 and 6 at an interior vertex), an independent
  set-based check on boxes, mixed elements and polygons, and bitwise equality across execution
  spaces. A serial golden derivation ported from moabX agrees incidence by incidence.
- With `SOMA_ENABLE_MOAB_REFERENCE=ON`, every bundled `.h5m` mesh is checked against MOAB
  `get_adjacencies(create_if_missing)`: sub-entities, downward sense and offset
  (`CN::SideNumber`), upward graphs, bridges and ring neighbourhoods match exactly.
- With `SOMA_ENABLE_MOAB_REFERENCE=ON`, `read_h5m` is checked against MOAB `load_file` on
  every bundled mesh and on two files MOAB writes at test time: every set's flags, members,
  sequence, children and parents, and every tag value on entities, sets and the mesh,
  variable-length and handle values included.

### MOAB files

- `SOMA_ENABLE_HDF5` (default OFF; point `HDF5_ROOT` at the installation) builds `read_h5m`,
  which reads a MOAB `.h5m` file on one process with the HDF5 C API, no MOAB involved. A
  parallel HDF5 with `SOMA_ENABLE_MPI=OFF` is a configure error; the package config records the
  HDF5 used. Without HDF5, `read_h5m` throws a `std::runtime_error` saying how to enable it, and
  `h5m_available()` is false.
- `read_h5m` reads vertices (1 to 3 coordinate columns), every Edge, Tri, Quad, Polygon (fixed
  arity groups and `poly_indices` lists), Tet, Pyramid, Prism and Hex group, with file ids as
  gids and lower-dimensional groups kept as explicit entities. Every fixed-length tag on vertices
  and elements becomes one per_entity field per dimension (full when every entity has a value),
  with the file's default and global values: 8-, 32- and 64-bit integers, f32, f64, arrays,
  opaque tags (`u8`, one component per byte) and bit tags (one `u8`). Value types it does not
  read are listed in an `H5mReport` instead. Files
  that are not MOAB meshes, element types it does not read (Polyhedron, Knife), node counts
  that are no configuration of their type, out-of-range or overlapping ids, malformed `poly_indices` and duplicate tag
  values throw `std::runtime_error` naming the file and the dataset; HDF5 prints nothing.
- `read_h5m` reads entity sets, with or without tags: contents id by id or as ranges that may
  cross groups, children, parents and flags; a set's gid is its file id. Tag values on sets
  become fields on the set slot, variable-length tags `VarField`s, handle tags handle fields
  whose values are the handles of the file ids' slots, and tags with values on no entity `MeshValue`s. A set member,
  child, parent or handle that names nothing, a malformed set list or `var_indices`, and an
  ordered set stored as ranges throw. The report now lists only 16-bit integers, bit tags wider
  than 8 bits, variable-length arrays and other HDF5 classes, and is empty on every bundled mesh.
- `read_h5m` turns a ranged set's ranges into runs of local ids without listing them, so a set
  over every vertex stores nothing, whatever the file's size.
- `soma::io::id_runs`, `moab_ranged` and `set_contents` encode a set's contents as MOAB does:
  (start, count) pairs exactly when the set is unordered, not empty and the pairs are not longer
  than the list. The rule agrees with every set of the bundled files; MOAB reads a sparse-class
  tag stored as dense values, which the legacy writer will use.
- `MeshBuilder::build` looks contiguous vertex gids up by index, skips sorts whose input is
  already sorted, computes centroids only for curve keys, copies in-order blocks whole and builds
  segments from their runs: the same mesh, about 10x faster on file input. Under
  `Ordering::type_major_sfc` it computes keys on host threads and orders by a stable radix sort
  on the key, which gives the same order.
- `sfc::hilbert_index` works without branches, with the axis count as a template parameter: the
  same keys, 3.7x faster each.
- `read_h5m` places tag values by lid in one pass instead of sorting them, and builds each full
  domain once for every tag that covers it.
- `write_h5m` writes a mesh in MOAB's legacy `.h5m` format, read back by MOAB `load_file` and
  `read_h5m` alike. Entities: the vertices, `mesh.dim()`, the dimensions the mesh was built with
  (`Mesh::input`) and those that fields, sets or handles use; `H5mWriteOptions::dims[d]` writes
  or leaves out any dimension, leaving one out only when a reader can rebuild it from the
  written ones; one group per element type and node count, file ids 1-based in gid order, so a file
  read and written back keeps its file ids. Sets with their flags, ordered sequences, children
  and parents, contents ranged where MOAB would. One tag per field family, dense on the blocks
  its values cover and sparse elsewhere, every MOAB kind: arrays, bit and opaque tags
  (`TagFormat`), variable-length tags, handles written as file ids, defaults, globals, mesh
  values, and always `GLOBAL_ID`: the mesh's own, or else the gids (a gid beyond i32 throws).
  `H5mWriteOptions::compression` (deflate level 0–9, default 0) chunks, shuffles and deflates
  the datasets. `H5mWriteOptions::adjacencies` (an `AdjacencyRequest`) caches the graphs it
  names under `/tstt/soma`, which MOAB ignores; `read_h5m` installs them
  (`H5mReadOptions::adjacencies`, default true) after checking their shape. The file is written to a temporary next to the target and renamed over it; a
  mesh it cannot write throws before any file is touched; equal meshes give equal bytes.
- Each field family keeps its legacy tag format, `TagFormat{storage_class, kind, nbits}`
  (`FieldRegistry::tag_format`, `set_tag_format`, checked against the value type); `read_h5m`
  records it for fields and mesh values (`MeshValue::format`), and rejects classes MOAB does
  not have.
- `read_h5m` reads high-order groups (`Hex27`, `Tri6`, ...) in every configuration MOAB
  writes, mixed and non-conforming ones too; `write_h5m` writes each node count as its own
  `<Type><count>` group with the corners then the nodes, and rejects leaving out a dimension
  whose entities have nodes. MOAB parity: every configuration of the bundled hexes, 2D and
  one-of-each-type meshes, converted by MOAB, reads with MOAB's counts, coordinates and node
  lists, and MOAB loads SOMA's copies alike.
- `EntityBlock::input` and `Mesh::input(d)` mark the dimensions `MeshBuilder` was given;
  derivation and mirrors keep the mark.

### Benchmarks

- `bench/bench_adjacency` times derivation and adjacency queries for SOMA on host threads
  against MOAB, the only baseline, with a discarded warmup, median and MAD, environment
  disclosure, JSON lines, and a checksum comparison that fails the run on a mismatch.
  `bench/run_adjacency.sh` repeats it at threads = 1, 2, 4, ..., N. Results are in
  `docs/benchmarks.md`.
- `bench/bench_h5m_read` times `read_h5m` against MOAB `load_file` on the bundled meshes, the
  large ones when present, and a MOAB-written box of N^3 hexes (`--box N`, default 64); the
  `soma_sfc` workflow times the read in `Ordering::type_major_sfc`.
- `bench/bench_h5m_write` times `write_h5m` against MOAB `write_file` on the same meshes, and
  `write_h5m` at a deflate level (`--compression L`, default 4), and prints the file sizes;
  every written file is loaded by MOAB and checked. `bench_h5m_read` also times reading cached adjacencies
  against building them.

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
