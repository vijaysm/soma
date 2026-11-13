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
- `SOMA_ENABLE_HDF5=ON` needs HDF5 1.14 or newer. SOMA files are written in the HDF5 1.14 file
  format (`H5F_LIBVER_V114`), whatever newer library writes them.
- CPM is at 0.43.2.
- `SOMA_ENABLE_EXAMPLES=ON` builds the example programs (it needs `SOMA_ENABLE_HDF5`); with
  tests on, each runs on a bundled mesh.
- `SOMA_ENABLE_KOKKOSKERNELS=ON` finds KokkosKernels (built with the installed Kokkos SOMA
  uses; a fetched Kokkos is a configure error). `SOMA_ENABLE_PETSC=ON` (with MPI) finds PETSc
  3.19 or later, real double scalars, through its pkg-config file under `PETSC_DIR` (and
  `PETSC_ARCH`). A PETSc that carries its own MOAB cannot be built with the MOAB reference.
  Both are replayed for consumers by `somaConfig.cmake`.

### Runtime and Comm

- `soma::Comm` is the single seam to MPI, with a serial implementation and an MPI
  implementation. It provides `rank`, `size`, `barrier`, `allreduce`, `exscan` and `bcast`.
- Reductions are defined for `std::int32_t`, `std::int64_t`, `std::uint64_t` and `double`, for
  `Op::sum`, `min`, `max` and `prod`, on one value or a span. Other types fail to link.
- `Comm::self()` gives a single-rank communicator.
- MPI errors carry the MPI error string, and counts above `INT_MAX` are rejected.
- `Comm::alltoall` and `Comm::alltoallv` (counts per rank, contiguous in rank order),
  checked alike in both builds.
- Non-blocking point-to-point: `isend`, `issend`, `irecv` (typed or bytes) return a move-only
  `Request` (`test`, `wait`, `Request::wait_all`; destroying an unfinished one waits);
  `iprobe` returns an `Envelope` (source, tag, bytes); `ibarrier`; `next_sequence`, a count
  per communicator (an MPI attribute), so ranks agree on tags for successive exchanges.
  Without MPI, rank 0 sends to itself through a queue matched by tag in send order, and
  waiting for a message nothing can send throws `std::logic_error`.
- `soma::ScopeGuard` initializes MPI and then Kokkos, and tears them down in reverse order.
- The fixed-width types `LO`, `Offset`, `GO` and `Real` are defined in `soma/core/types.hpp`.
  `SOMA_LO_64` and `SOMA_OFFSET_64` select 64-bit widths.
- `Comm::exscan` on a span scans each entry on its own (one call for a vector of counts).
- `soma::reproducible()` (`set_reproducible`, `SOMA_REPRODUCIBLE=0|1`,
  `--soma-reproducible[=0|1]`), off by default, asks algorithms for results the same bit for
  bit on any number of ranks and threads, for one build, at some cost.
- `soma::ExactSums<K>` (`soma/parallel/exact_sum.hpp`) sums doubles exactly into 68 radix-2^32
  digits and rounds once, to nearest: the same bits in any order and split. A Kokkos reducer
  (`ExactSumReducer`, `reduce_exact`) and `allreduce_exact` (one allreduce of integers) carry
  it over threads and ranks; a term not finite makes its sum NaN.
- `detail::comm_counts()` counts the collectives and routed exchanges this process has
  entered, for tests that bound them.
- `Comm::allgatherv` gathers each rank's span of trivially copyable values on every rank, in
  rank order, with the counts if asked; one collective.
- `Comm::split(color, key)` returns an owned `SplitComm` (freed with it) grouping the ranks of
  a color in key order; a negative color joins no group. One collective.

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

### Mesh edits

- `soma::MeshEdit` records additions (vertices, elements with or without nodes, polygons) and
  removals on a host mesh; `apply()` makes them visible together. Existing entities keep their
  local ids, added ones follow them in tail segments, removed ones stay in place, masked
  (`Mesh::alive`, `num_live`, `MeshView::alive`). Gids never change; entities added without
  gids take the next ones. Removing a vertex a live element uses throws, as MOAB refuses it.
- Elements above the mesh's dimension raise it; the old elements become explicit sides.
- `apply` grows fields over every entity of a dimension with their default, keeps the epoch,
  drops removed entities from sets, and builds held graphs again over the live entities, on a
  compact copy carried back to the edited lids; sides that lost their completeness are derived
  again, keeping their gids, with fields and sets following by gid.
- `compress()` rebuilds the mesh from its live entities through `MeshBuilder` in its ordering,
  so the result equals building the final mesh from scratch, carries fields
  (`FieldRegistry::remap_into`, stale fields as they were), sets and held graphs across, bumps
  the epoch and returns the renumbering. `write_h5m` and `write_vtk` write an edited mesh as its
  compressed copy.
- Set renumbering can drop members without a new local id.
- `apply()` compresses the mesh once removed entities fill a threshold share of some
  dimension's slots (default 0.2, measured), reporting the renumbering in
  `EditReport::compressed`; `MeshEdit::set_compress_threshold` sets it per edit, 0 never
  compresses. `removed_fraction(mesh)` gives the share.
- Process-wide options (`soma/runtime/options.hpp`): the compress threshold, read from
  `SOMA_COMPRESS_THRESHOLD` and `--soma-compress-threshold=<x>` (which wins) by `parse_options`,
  which `ScopeGuard` calls; SOMA's arguments are removed from argv.

### MOAB-compatible API

- `soma::compat` (`SOMA_ENABLE_MOAB_COMPAT`, target `soma_compat`) offers MOAB's `Core`
  interface over a SOMA host mesh: file I/O (`.h5m`, `.vtk`), entity, coordinate, connectivity
  and adjacency queries, creation and deletion, tags of every kind MOAB has, sets with parents
  and children, `ScdInterface::construct_box`, `Range`, `CN`, MOAB's error macros and message
  format, and the part of `ProgOptions` the examples use. Tags and sets live on the compat side;
  the mesh holds topology and coordinates.
- Handles keep MOAB's bit layout and MOAB's allocation of ids (blocks per type and node count,
  freed ids reused), so an application sees the handles MOAB would give it. Compat never
  compresses its own mesh; creations are staged and applied at the next query.
- Tag reads and writes on deleted entities fail with `MB_ENTITY_NOT_FOUND`, alone or in a Range,
  also where MOAB accepts them for dense tags. compat never compresses its mesh.
- `load_file` rejects a high-order mesh whose elements disagree on the node of a shared edge or
  face, naming both elements, and numbers what it reads in file order.
- Drop-in mode (`soma_compat_dropin`) puts headers at MOAB's paths first on the include path, so
  MOAB applications build unchanged; builds without MPI get a one-rank `mpi.h` stand-in.
- Twelve of MOAB's examples, built against MOAB and against compat, give the same output and
  files (`compat.gate.*`).

### Distributed transport

- `soma::transport::route` delivers fixed-width records (`RecordBatch`: `dest`, `source`,
  byte rows of one width on every rank) by the crystal router (clean-room after gslib, with its
  odd-count partner rule), `alltoallv`, or NBX (`Strategy`; `automatic` takes `alltoallv` up to
  64 ranks, a placeholder until measured at scale). Every strategy returns the same records
  grouped by source rank in each sender's order. A bad destination, a ragged payload, or a
  width or strategy that differs between ranks throws `std::invalid_argument` on every rank.
  Every received count is checked against what the ranks sent (P times the largest batch)
  before any buffer is sized from it; a corrupt message throws `std::runtime_error`.
- `CommPlan` / `learn_plan` keep the pattern of one routed exchange and replay it with one
  round of point-to-point messages: `forward` (values per sent record to where route() put
  them) and `reverse` (answers back to the sender's order). Learnt from both halves of the
  exchange, so no handshake is needed.
- `CommPlan::from_counts` builds a plan from the counts each rank sends to and receives from
  each other rank, with no exchange, and `begin_forward` / `begin_forward_bytes` start a forward
  under a caller's tag and return a move-only `Pending` to wait on, so other work can run while
  the messages travel.
- `rendezvous(keys, payload, answer_width, answer)` sends each record to
  `owner(key, P) = mix(key) % P` (the splitmix64 finalizer, so owners depend only on the key
  and P), lets the owner answer every arrival, and returns the answers in key order; failures
  in `answer` reach every rank.
- Records are staged through host memory (one copy in, one out); transport uses tags 32000 to
  32767.
- `rendezvous_to` is `rendezvous` with destinations the caller chooses (blocks of ids, say).
- `rank_keys(comm, keys, words, distinct)` ranks fixed-width keys across ranks by a sample sort
  (splitters chosen on rank 0 and broadcast); with `distinct`, equal keys share one index and
  the indices do not depend on P.

### Distributed meshes

- `Ownership` gains `shared_copy`, and `SegmentTable` `shared_end`: each dimension is ordered
  interior, shared and owned here, shared and owned elsewhere, ghosts, so ownership is a
  comparison with segment ends (`MeshView::owned`, `interior`, `shared`).
  `SegmentFilter::Own` selects each kind. `kind_bounds` gives each kind's range and
  `split_segments` splits segments at cuts.
- `MeshBuilder::set_ownership(dim, gids, kinds, groups)` orders a dimension by ownership, then
  topology, arity and nodes, then neighbour group; `set_key_box` gives every rank the same curve
  box. `vertex_gids()`, `vertex_coordinates()` and `element_gids(dim)` return what was added.
- High-order vertices: within each ownership kind, corners come before node-only vertices, in
  segments of their own. `Mesh::corner_end(kind)`, `MeshView::corner_ends` and
  `MeshView::corner(v)`; `corners()` is the interior kind's end, so an all-interior mesh keeps
  `[0, corners())`.
- `partition_by_curve(comm, centroids, gids)` cuts elements into runs of a Morton curve of
  their centroids whose sizes differ by at most one.
- `resolve_sharing(comm, gids)` finds every rank holding each gid (the lowest owns it).
- `build_local_part(comm, builder, ordering)` builds a rank's part with the ownership and
  sharing of every dimension it holds, kept in the internal fields `soma:sharing` (the other
  ranks holding each shared entity) and `soma:owner` (the owner of each shared copy). Its
  checks and the box of every rank's vertices take one reduction, and the build one more.
- `build_distributed_adjacencies(comm, mesh, request)` builds graphs on a part and gives
  derived edges and faces gids that do not depend on P (the largest explicit gid plus the rank
  of their sorted corner gids), sharing and ownership order; a distributed dimension that the
  derivation extends is distributed again.
- `HaloExchange(comm, mesh, dim)`: `update` (owners to copies) and `reduce` (every holder
  gets the same bits, combined in ascending rank order) over the shared entities, each also
  split into `begin_update` / `begin_reduce`, which return a move-only `Pending` with `test()`
  and `end()`; several may be in flight and end in any order.
- `migrate(comm, mesh, dest, ordering)` moves each element to a given rank with its closure
  and returns the new part: entities keep their gids, corners, nodes and coordinates;
  dimensions only derived are derived again (same gids); graphs held on any rank are built
  again; set memberships and field values (layouts, options, tag formats) follow from each
  entity's owner through a gid directory; mesh values are copied. Sets, fields and mesh values
  that differ between ranks, a bad `dest` or an edited mesh throw on every rank.
- Ghost layers: `mesh.add_ghosts(comm, GhostOptions{layers, bridge_dim, sets})` and
  `mesh.remove_ghosts(comm)`, in place and chained. Layer i is every element held elsewhere
  sharing a `bridge_dim` side with layer i - 1, whatever the partition (sliver parts, empty
  ranks, layers deeper than a part), found by one round per layer over the holders of each
  side's lowest vertex. Entities held before keep their ids; ghosts keep their gids and follow
  the shared copies, by topology and owner; `soma:owner` covers them. Fields come from the
  owners, set membership with `sets`. `resolve_sharing` and `build_local_part` take ghosts;
  derived sides of ghost elements only are ghosts. `HaloExchange::update` writes ghosts,
  `reduce` leaves them. Parity with MOAB's `exchange_ghost_cells` for the first layer.
  `Mesh::ghost_layers()` and `ghost_bridge_dim()` record the ghosts held (0 after
  `remove_ghosts`), and copies keep them.
  `add_ghosts` returns after one reduction when no rank has anything to ghost and none holds
  ghosts. It sends each ghosting rank its ghosts' closures (after owners give the copies of
  their shared entities what they hold) and appends them to what the rank holds, which stays in
  place: only the ghosts are ordered (as the part built whole orders them) and only the ghosts'
  own sides are derived. The rounds start from the elements with a shared vertex, index the
  sides of those within `layers - 1` of them, and skip telling vertex holders in the last
  round; owners keep the ranks they answer, so no round tells them. Copies' versions go in
  one exchange and closures in another, with notices telling owners who holds their closure
  entities; owners then push field values and set memberships to the ranks holding copies
  and ghosts, which look up only their shared and ghost entities. So the cost follows the
  ghost layer, not the part, with no gid directory: on a 64^3 box one layer takes 28 to 37
  ms at 2 to 8 ranks (281 to 115 ms before), as MOAB's `exchange_ghost_cells` at 8 ranks,
  two layers 47 to 83 ms, below MOAB's at 4 and 8. The part
  is the one rebuilding it through directories gives (`detail::add_ghosts_by_rebuild`), entity
  for entity. An entity held only for an explicit edge (such an edge's far corner) is no
  longer lost when another rank ghosts it.
- `build_distributed_adjacencies` keeps the sharing of the dimensions it does not extend when
  deriving beside explicit entities renews the epoch.
- `permute_entities(mesh, dim, new_of_old, table, gids)` reorders one dimension with
  everything that refers to it (connectivity, graphs, sets, fields), optionally with new gids.
- `read_h5m(comm, path, options)` reads a file in parallel: every rank reads the metadata and
  a slice of each dataset; elements go by the file's `PARALLEL_PARTITION` sets when there are
  at least as many as ranks (`H5mReadOptions::file_parts`), else by the curve; explicit
  entities of lower dimensions go to every rank holding all their corners; sets, tags and
  cached graphs follow (graphs are built on each part). Each rank's entities, set members and
  values equal the serial read's by gid. With a parallel HDF5 the slices are read collectively
  over MPI-IO (`H5mReadOptions::collective`, default on there; asking for it with a serial
  HDF5 throws `std::invalid_argument`); every rank joins each read, empty slices too, and a
  bad slice throws on every rank after the reads that must stay in step.
- `build_local_part(comm, builder, sharing, ordering)` takes the sharing of each dimension
  instead of resolving it through a gid directory.
- `match_points(comm, xyz, tol)` numbers points by their coordinates over all ranks: exactly
  (bit-identical coordinates, `-0.0` equal to `+0.0`) by one sort of their 192-bit keys, or
  within a tolerance (`PointNorm::linf`, `l1`, `l2`; single linkage) by grid cells with halos
  and a distributed union-find. Ids, and each cluster's representative point, equal the
  single-node `hash3d_exact` numbering of all points together, whatever the rank count or
  order. NaN coordinates, partial points, ranks giving different tolerances and, with a
  tolerance, unsafe floating point (`-ffast-math`) throw on every rank.
- `merge_vertices(comm, builder, tol)` makes the matching vertices of every rank's builder one
  (MOAB's MergeMesh on `Comm::self()`, ParallelMergeMesh over a communicator), and
  `number_elements(comm, builder)` gives elements gids from their corner sets, so parts with
  gids of their own, or new vertices added on both sides of a part boundary, build with
  `build_local_part` and share by gid. An element left naming a vertex twice throws on every
  rank, and no builder changes. `MeshBuilder::rename_vertices`, `rename_elements` and
  `element_corners` are the steps underneath.
- `transport::rank_keys_held` also returns, per key, the other ranks giving it: the rank of the
  key's range sees every copy. `match_points` returns this as each point's `Sharing`, and
  `merge_vertices` and `number_elements` as their entities', so `build_merged_part(comm,
  builder, tol)` builds the part without `resolve_sharing`'s gid directory. Tolerance clusters
  are joined across ranks by FastSV rounds, logarithmic in the longest chain.
- `merge_mesh(comm, mesh, tol)` merges a built mesh in place, serial or distributed: its live
  vertices that match on any rank become one, elements and explicit sides are numbered by their
  corner sets, and the mesh is built again with the graphs it held, so parts refined each on
  its own with `MeshEdit` (gids of their own) share their new boundary vertices after. Sets,
  fields and mesh values follow each entity; where entities on a rank become one, the least
  lid's are kept. A mesh with ghosts is refused on every rank.
- `point_key`, `point_of_key` and `point_keys(exec, xyz, keys)` (`soma/core/point_key.hpp`)
  compute the exact keys in device code, equal to `hash3d_exact`'s; `match_points` computes its
  keys and NaN check with them in one parallel pass.

### SOMA files

- `write_soma(comm, mesh, path, SomaWriteOptions{compression})` writes a native SOMA file
  (`/soma`, `format_version` 1): part p is what rank p owned, each per-part dataset the parts'
  rows concatenated, located by one part table. Per written dimension: gids, topology,
  corners and nodes as gids, coordinates, and the parts holding copies of each owned shared
  entity; fields per part with their options and domains; sets with each part's members;
  mesh values. Ghosts and copies are not written. A write takes 7 collectives and no routed
  exchange whatever the rank count or the number of fields and sets. With a parallel HDF5 the
  ranks write collectively over MPI-IO in one multi-dataset write; with a serial one each
  rank's rows go to rank 0 in turn. The file appears whole or not at all (temporary and
  rename); the same mesh at the same rank count gives the same bytes through either sink.
- `read_soma(comm, path, SomaReadOptions{ordering, rebalance, adjacencies})` reads a file of
  N parts at P ranks, part q on rank floor(qP/N). Owners send their shared entities, with
  field values and set memberships, to the ranks holding copies, and the sharing comes from
  the file: a restart read (N = P) gives each rank the part it wrote with 6 collectives and
  one routed exchange, 7 with MPI-IO at P > 1. There the file is opened once, collectively,
  and rank 0 alone reads its metadata and where each dataset's rows are, sent to every rank
  in one broadcast; every rank then reads its rows in one collective MPI-IO read on the file
  HDF5 opened. HDF5 itself makes 4 broadcasts (opening the file) instead of 63, and the other
  ranks no metadata reads. Compressed datasets, or those stored in another type or byte order,
  are read through HDF5. With N < P, `rebalance` migrates elements by the curve so every rank
  gets some. The result equals the mesh written by gid: entities, coordinates, connectivity,
  nodes, sets in order, fields with domains, defaults, global values and formats, mesh values;
  derived dimensions not written are derived again.
- `SomaWriteOptions::derived` also writes every dimension held on some rank (derived edges and
  faces with their gids and sharing). Dimensions written but derived in the mesh written read
  back complete and marked derived, so building the writer's graphs derives nothing and a
  restart makes no gid directory pass. A dimension is written for sets only when some set has
  members of it.
- `MpiIoHints` in `SomaWriteOptions` and `SomaReadOptions` pass `cb_nodes`, `cb_buffer_size`,
  `romio_cb_write`/`romio_cb_read`, `striping_factor` and `striping_unit` to MPI-IO, and align
  datasets with `H5Pset_alignment`. With `aggregators` = M in [1, P], a write sends each rank's
  rows to the first rank of its group floor(rM/P), and only those M ranks open the file and
  write their group's rows; a read has those ranks read their group's metadata and rows and
  send each rank its own. The file stays one file and is the same as without aggregators.
  M > 1 needs a parallel HDF5; invalid or differing options throw on every rank.
- Meshes that cannot be written and files that cannot be read throw on every rank.
- `detect_format(path)` tells a MOAB `.h5m` file from a SOMA file by content;
  `convert_mesh_file(comm, in, out, compression)` and the `soma-convert` tool convert either
  way (legacy files are read in parallel; SOMA files are gathered on rank 0 and written by
  `write_h5m`, which stays serial).
- `read_file(comm, path)` and `write_file(comm, mesh, path)` (`soma/io/mesh_file.hpp`) read
  and write a mesh without naming its format: a read goes by content (SOMA, MOAB `.h5m`, or
  legacy VTK, read on rank 0 and dealt to the ranks along the curve with its fields), a write
  by extension (`.soma` in parallel; `.h5m` and `.vtk` gathered and written by rank 0). A
  failure of rank 0 alone reaches every rank. `detect_format` also recognises legacy VTK
  files. The examples use them.

### Patch-based AMR

- `soma::amr` (headers in `include/soma/amr/`): boxes in ijk index space for cell, node and
  face data (`Box`, `IndexType`, `BoxList` algebra), levels of disjoint boxes ordered along a
  Morton or Hilbert curve (`BoxArray`, `chop`, `Distribution`), and `LevelField`: all boxes of
  a level in one allocation, with `for_each_point` running over a level in one launch.
- Same-level halos (`HaloPlan`, faces, edges, corners, periodic images, physical boundaries
  through a callback), copies between box sets (`CopyPlan`), coarse-fine halos by piecewise
  constant or conservative MC-limited linear prolongation (`CoarseFinePlan`), restriction
  (`average_down`) and refluxing (`FluxRegister`). Each point is written by one source, so
  results do not depend on the thread count.
- Quadratic prolongation (`Interp::quadratic`): conservative for cell averages, Lagrange at
  nodes, tensor products for faces; exact for data of degree 2 in each direction, unlimited.
- `Hierarchy`: levels checked for alignment and proper nesting, plans cached, halos filled and
  data restricted across all levels; `regrid` tags cells, clusters them with Berger–Rigoutsos
  (`cluster`), keeps the levels nested, and moves every field to the new boxes (copied where the
  old ones were, prolonged elsewhere, then restricted), all fields or none.
- `write_vtk_amr` writes a hierarchy as VTK overlapping AMR (`.vthb` and one `.vti` per box),
  which ParaView reads with the cells under finer levels hidden.
- Distributed AMR: every rank holds every level's boxes and their `Distribution` (cut along the
  curve, by point count or by weights), and a `LevelField` holds only the boxes its rank owns.
  `CopyPlan`, `HaloPlan`, `CoarseFinePlan`, `AverageDownPlan` and `FluxRegister` take a
  communicator: copies within a rank stay local, the rest go as one message per neighbour
  rank, packed and unpacked by device kernels and staged through host memory. Halo fills
  split into `begin_fill_halo` and `end()` around interior work. Results equal those of one
  process bit for bit at any rank count.
- `Hierarchy(comm, geometry, options, fine)` distributes every level; `HierarchyOptions::cost`
  weights boxes for the cut, or measured costs passed to `regrid` override it. A distributed
  `regrid` gathers the tagged blocks in one collective, so every rank clusters the same boxes
  as one process would; a tagger failing on one rank throws on every rank.
- `write_soma(comm, hierarchy, fields, path)` checkpoints a hierarchy and its fields (any of the
  eight value types) in the SOMA format: each level's boxes and each field's valid points, box
  after box in curve order, in 6 collectives whatever the rank count or number of fields.
  `read_soma_amr(comm, path)` restarts it at any rank count, cut by the reader's costs, each
  rank reading its own run of rows; ghosts are filled by the caller. With MPI-IO at P > 1 the
  file is opened once: rank 0 alone reads the header and where each column is, sent to every
  rank in one broadcast, and every rank reads its rows of all fields in one collective MPI-IO
  read (compressed or converted columns through HDF5). `detect_format` reports
  `MeshFormat::soma_amr`, and the mesh and AMR readers each refuse the other's files.
- `write_vtk_amr(comm, path, hierarchy, fields)` writes each rank's boxes as `.vti` pieces and
  the `.vthb` index on rank 0 once every piece is in place; the files equal those one process
  writes.

### Algorithms

- `soma::laplacian_smooth(comm, mesh, SmoothOptions)` (`soma/algorithms/laplacian.hpp`) moves
  each free owned vertex to the mean of its neighbours' coordinates, Jacobi, for `iterations`
  sweeps or until no vertex moves farther than `tolerance`. Neighbours are the other corners of
  the entities of dimension `bridge_dim` around a vertex (the cells by default, as MOAB's
  LaplacianSmoother; 1 for edges), read off the cells' canonical sides. The skin stays, or the
  vertices of a `fixed` mask. Each vertex sums in ascending gid order, so the result is the
  same, bit for bit, on any number of ranks and threads. On more than one rank the mesh needs a
  ghost layer through vertices; a halo update refreshes copies and ghosts after every sweep.
- It equals MOAB's smoother bit for bit on one rank and within 1e-14 on ghosted partitions of
  one to eight ranks. Ten sweeps of a 64^3 box of hexes, setup included, one thread a rank:
  128 ms against MOAB's 8222 at one rank, 29 against 468 at eight.
- `examples/laplacian_smooth.cpp` reads a `.h5m` or SOMA file on any number of ranks, adds a
  ghost layer, smooths and writes a SOMA file (a VTK file on one rank).
- `soma/algorithms/fem.hpp` holds linear finite element kernels for triangles, quads, tets
  and hexes (P1 and Q1): Gauss rules, shape functions and their gradients, the least Jacobian
  over the corners and the Gauss points, and element stiffness, mass, load and squared error.
  An inverted, degenerate or non-finite cell gives zero matrices and a NaN error.
- `soma::FemSpace<MemSpace>(comm, mesh, Assembly)` (`soma/algorithms/fem_space.hpp`) numbers
  a mesh's vertices for linear elements: rows are the owned vertices, columns all the rank's
  vertices in gid order. It builds the sparsity once and assembles the stiffness matrix
  (`laplacian`) and a load vector (`load`) by `Assembly::gather` (each row sums its cells in
  gid order: the same, bit for bit, on any number of ranks and threads), `atomic`
  (`Kokkos::atomic_add`), or `colored` (cells colored greedily by gid, a color at a time:
  the same on any number of threads). `boundary()` marks the skin's vertices, `update` the
  halo, `numbering` a global index of the owned vertices rank by rank. On more than one rank
  the mesh needs a ghost layer through vertices. High-order, edited or inverted meshes, a
  missing ghost layer and vertices in no cell are rejected on every rank.
- `soma::CsrMatrix` with `multiply` and `apply_dirichlet` (`soma/algorithms/csr.hpp`), and
  `soma::cg` (`soma/algorithms/cg.hpp`): Jacobi-preconditioned conjugate gradients, two
  allreduces and one halo update an iteration, confirming convergence with the true residual.
  A manufactured solution's L2 error falls as h^2 on smoothly warped boxes of all four element
  types (rates checked between 1.9 and 2.2).
- `soma::to_kokkos_sparse` (`soma/algorithms/kokkos_sparse.hpp`, `SOMA_ENABLE_KOKKOSKERNELS`)
  views a `CsrMatrix` as a `KokkosSparse::CrsMatrix` without a copy;
  `soma::to_petsc_matrix`, `to_petsc_vector` and `from_petsc_vector`
  (`soma/algorithms/petsc.hpp`, `SOMA_ENABLE_PETSC`) make PETSc `MATAIJ` matrices (by COO) and
  vectors in `FemSpace::numbering` order. PETSc's CG with Jacobi takes the iterations SOMA's
  takes, to one, and reaches its solution to 1e-9.
- On a 64^3 box of hexes, one thread a rank (bench_parallel), the space takes 202 ms,
  gathered assembly 246 and CG to 1e-10 518 at one rank; 40, 53 and 148 at eight. Atomic and
  colored assembly time the same as gathered to within 2 % at one rank.
- The element kernels skip work without changing a bit of the matrices: corner determinants
  are read off the cells' edges, the degree-2 Gauss points' determinants serve both the
  validity check and the integration, the stiffness matrix is filled by symmetry, and
  `laplacian()` and `load()` count bad cells as they assemble instead of in a pass of their
  own. On a 64^3 box of warped hexes, one rank: assembly 776 ms to 237 at one thread (123 to 33
  at eight), the space 397 to 199. `least_jacobian` is NaN for a cell with a coordinate not
  finite or a determinant that overflows; a NaN no longer gives way to a later determinant.
- Reproducible (`soma::reproducible()`, or `CgOptions::reproducible` for one solve), a
  `FemSpace` gathers whatever assembly it is asked for (`assembly()` tells), and its `dot()`
  and `cg` sum exactly, with as many allreduces: CG gives the same iterates and solution, bit
  for bit, on any number of ranks and threads. Ranks must agree on the switch (every rank
  throws otherwise), and a reproducible solve needs a gathering space. On the 64^3 box, one
  thread a rank, CG takes 1.28 times as long at one rank and 1.13 to 1.23 at two to eight;
  1.14 at eight threads on one rank.
- `examples/poisson.cpp` solves -lap u = 1, u = 0 on the boundary, on a `.h5m` or SOMA mesh on
  any number of ranks, by any assembly, and writes u as a vertex field.
- `soma::FvSpace<MemSpace>(comm, mesh)` (`soma/algorithms/fv.hpp`) advects a cell field by
  first-order upwind finite volumes and forward Euler on tris, quads, tets and hexes with one
  ghost layer: `normal_flux(a)` evaluates a velocity functor at the face centroids, `max_dt`
  gives the largest stable step, `step` takes one (with a value flowing in at the boundary)
  and updates the ghosts once, and `total` sums V u, exactly when reproducible. Face geometry
  follows the corners' gids and each cell sums its faces in a fixed order, so the steps are
  the serial ones bit for bit on any number of ranks and threads. Cells may be ordered either
  way round; a cell folded over its neighbour, of no positive volume, or (in 2D) off a plane
  z = const, a side of three cells, or no ghost layer on more than one rank, throws on every
  rank.
- `examples/advect.cpp` carries a bump around a `.h5m` or SOMA mesh by a rigid rotation on any
  number of ranks and writes u as a cell field.
- `soma::GridIndex<MemSpace>` (`soma/algorithms/grid_index.hpp`) bins axis-aligned boxes in a
  uniform grid (Kokkos only) and returns, for each query box, the boxes it meets within a
  tolerance, each once and in order, the same at any thread count.
- `soma::RemapOperator<MemSpace>(comm, source, target, RemapOptions)`
  (`soma/algorithms/remap.hpp`) is a first-order conservative remap between plane meshes of
  triangles, quads and polygons, or volume meshes of tets, hexes, prisms and pyramids:
  weights |T n S| / |T| from clipped overlaps, rows ordered by source gid, `apply` in one
  exchange. Non-convex plane cells are split into triangles; volume cells into tets about
  their centroids, quad faces cut from their least-gid corner so that neighbours' tets fill
  space. The meshes may be partitioned differently and need no ghosts; the operator is the
  same bit for bit on any partition, number of threads and listing of a cell's corners, and
  cells are clipped about a corner, so accuracy does not depend on the distance from the
  origin. `RemapOptions::merge_tolerance`
  (default 1e-12, a length) moves corners within it onto clipping lines and planes, merges
  clipped corners and drops degenerate overlaps; a cell thinner than it, crossing itself, off
  a plane z = const (2D) or not star-shaped about its centroid (3D) throws on every rank, as
  does a tolerance that is bad or differs between ranks. Plane overlaps agree with MOAB's
  `Intx2MeshInPlane` to rounding.
- `examples/remap.cpp` remaps a cell field from a `.h5m` or SOMA mesh onto a grid of triangles
  or tets over its bounding box and prints both totals.
- `soma::refine_uniform(comm, mesh, RefineOptions)` (`soma/algorithms/refine.hpp`) splits
  every edge of a plane mesh of triangles, quads and polygons, or a volume mesh of tets, hexes,
  prisms and pyramids (mixed meshes included), into two, `levels` times: tris and quads into
  4, polygons into n quads about their centroid, tets into 8 (the octahedron on its shortest
  diagonal), hexes and prisms into 8, pyramids into 6 pyramids and 4 tets. Every quad face gets
  a centre, so meshes stay conforming. New vertices are corner averages in gid order and gids
  follow from edge, face and cell gids, so the refined mesh is the same, gid for gid and bit
  for bit, on any number of ranks; children stay on their parent's rank. It returns child ->
  parent and parent -> children maps; dense cell fields pass to children and floating vertex
  fields are averaged onto new vertices (exact for linear fields); other fields are named as
  dropped. Matches MOAB's `NestedRefine` on tri, quad, tet and hex meshes (tets up to ties
  between equally short diagonals).
- `examples/refine.cpp` refines a `.h5m` or SOMA mesh `--levels` times and prints the counts
  and the error of a linear vertex field.

### Testing

- `policy.device_lambdas` fails when a test body holds a `KOKKOS_LAMBDA` (CUDA refuses extended
  lambdas in private member functions); test kernels live in named functions.

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
- With `SOMA_ENABLE_MOAB_REFERENCE=ON`, MPI and HDF5, `test_parallel_parity_moab` reads the
  bundled partitioned meshes with MOAB's `READ_PART` (shared entities resolved, one ghost
  layer) and builds SOMA's parts from the same elements on each rank: for vertices and
  elements, matched by coordinates, the local, owned, shared and ghost sets, owners, sharing
  ranks, global owned counts, and an update of doubles and integers equal MOAB's at 1 to 8
  ranks.

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
- At the full width of two or three axes the host follows the Hilbert curve by state tables, three
  levels a step in 3D and four in 2D, read off the transpose and checked against it: the same
  keys, 10x (3D) and 17x (2D) faster. `MeshBuilder::build` orders (key, position) pairs and
  moves records once: by one pass on the key's highest 16 varying bits and sorts of the small
  buckets, ties by gid, the same order; ownership and repeated element gids are looked up in
  tables over dense gid ranges. `Sharing::groups` sorts only shared rows. `read_soma` finds
  local ids and rows by tables over dense gids and indexes set memberships only for copies it
  sends. A one-rank read of a 64^3 box from a SOMA file takes 55 ms instead of 194 (MOAB 46),
  and 132 instead of 155 at 8 ranks (MOAB 266); `build_local_part` 42 instead of 118 ms at one
  rank and 18 instead of 32 at 8 (MOAB's `resolve_shared_ents` 235 and 68).
- `read_h5m` finds rows and local ids by tables over dense file ids, and forwards ranged set
  members by a search of the holders' sorted ids. `CommPlan` sends records already grouped by
  destination in place and counts the others by rank; `route` moves rows already in destination
  order rather than bucketing them. A one-rank read of a 64^3 box from an `.h5m` file takes
  99 ms instead of 266 (MOAB 45), and 51 instead of 100 at 8 ranks (MOAB 107); a one-rank
  `write_soma` 17 instead of 48 (MOAB 22).
- Records a rank routes to itself no longer travel: `route` copies them once, by runs, into
  their place among the arrivals. `CommPlan` is built in fewer passes and answers grouped
  records in place. `read_h5m` keeps the elements it places in one flat list rather than a
  vector each. `resolve_sharing` checks repeated gids and orders holders by tables over dense
  gids, and on a single rank shares nothing without an exchange; `read_h5m` on a single rank
  keeps every element where it was read. `SetBuilder::install` sorts the runs of every set of a
  dimension by one count over its entities when they are dense. A one-rank read of a 64^3 box
  from an `.h5m` file takes 43 ms instead of 99 (MOAB 48), and 75 instead of 91 at two ranks
  (MOAB 172); from a SOMA file 47 instead of 52 (MOAB 46); `build_local_part` 27 instead of 40
  at one rank (MOAB's `resolve_shared_ents` 232).
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
  `bench/run_adjacency.sh` repeats it at threads = 1, 2, 4, ..., N.
- `bench/bench_h5m_read` times `read_h5m` against MOAB `load_file` on the bundled meshes, the
  large ones when present, and a MOAB-written box of N^3 hexes (`--box N`, default 64); the
  `soma_sfc` workflow times the read in `Ordering::type_major_sfc`.
- `bench/bench_h5m_write` times `write_h5m` against MOAB `write_file` on the same meshes, and
  `write_h5m` at a deflate level (`--compression L`, default 4), and prints the file sizes;
  every written file is loaded by MOAB and checked. `bench_h5m_read` also times reading cached adjacencies
  against building them.
- `bench/bench_edit` times queries on a mesh after `MeshEdit::apply` against the compact mesh
  (1, 10 and 50 % of the hexes of a box removed and created again), `apply` and `compress`
  against building the mesh, and compat's tag, connectivity and adjacency calls against MOAB's
  on the same edits, checksummed.
- `bench/bench_transport` (MPI builds) times `route` with each strategy against MOAB's crystal
  router (`gs_transfer`) on dense and sparse destination patterns, checked by count and
  checksum on every rank.
- `bench/bench_parallel` (MPI and HDF5 builds) times, at P ranks on a box of n^3 hexes in
  partition sets that MOAB writes, the parallel read (independent and collective) against
  `READ_PART` with resolution, `build_local_part` against `resolve_shared_ents`, `add_ghosts`
  against `exchange_ghost_cells`, a vertex update of 1 and 3 doubles (fused and split)
  against `exchange_tags`, and `write_soma` / `read_soma` against `WRITE_PART` and a
  `READ_PART` load, checking owned counts and updated values. Its `smooth` and `poisson`
  sections time Laplacian smoothing against MOAB's, and `FemSpace`, the three assemblies and
  CG alone; its `advect` section, `FvSpace` and upwind steps alone; its `remap` section, a
  `RemapOperator` between plane grids built and applied, against MOAB's in-plane intersection
  at one rank, and between volume grids; its `refine` section, `refine_uniform` of hexes and
  tets, against MOAB's `NestedRefine` of the hexes at one rank. `bench/run_parallel.sh` runs it
  over rank counts and sizes into JSON files; its header lists what a run at scale needs.

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
