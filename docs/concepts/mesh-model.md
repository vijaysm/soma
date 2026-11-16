# The mesh model

This page describes how SOMA represents one rank's part of an unstructured mesh: the entity
dimensions and topologies, global and local ids, the segment tables that order entities by
ownership, the `Mesh` class and its copies in other memory spaces, the `MeshView` that kernels
use, coordinates and high-order nodes, and the integer types everything is built on. The terms
defined here (gid, lid, owned, shared, ghost, segment, epoch, ...) are used throughout the rest
of the documentation.

## Dimensions, entities and topologies

A mesh holds entities of four dimensions: vertices (0), edges (1), faces (2) and regions (3).
`Mesh::dim()` is the mesh dimension $D$, between 1 and 3 for a built mesh; the entities of
dimension $D$ are the *elements*, and entities of a lower dimension are *sides* (edges, and faces
in 3D). Sides are either given explicitly to the builder or derived from the elements on request
(see [topology and adjacencies](topology.md)).

The shape of an entity is a `soma::Topology` (`soma/core/topology.hpp`):

```cpp
enum class Topology : std::uint8_t {
  vertex = 0, edge = 1, tri = 2, quad = 3, polygon = 4, tet = 5,
  pyramid = 6, prism = 7, knife = 8, hex = 9, polyhedron = 10,
};
```

The values equal MOAB's `moab::EntityType` and the `/tstt/elemtypes` enumeration of the `.h5m`
format, so they are never renumbered. Device-callable helpers describe each topology:

| Function | Meaning |
| --- | --- |
| `dimension(t)` | 0 for `vertex`, 1 for `edge`, 2 for `tri`, `quad`, `polygon`, 3 for the others |
| `num_vertices(t)` | corner count: 1, 2, 3, 4 (`quad`, `tet`), 5 (`pyramid`), 6 (`prism`), 7 (`knife`), 8 (`hex`); 0 for `polygon` and `polyhedron` |
| `has_fixed_topology(t)` | false for `polygon` and `polyhedron`, whose vertex count varies |
| `name(t)` | `"tri"`, `"hex"`, ... |

Polygons are supported as faces (added with `MeshBuilder::add_polygons`); polyhedra are listed
for file compatibility but cannot be built yet.

```{figure} ../figures/img/reference-cells.svg
:alt: The tri, quad, tet, pyramid, prism and hex, each with its corners numbered in bold and its edges numbered in grey; edges hidden behind the cell are dashed
:width: 100%

The local numbering of the fixed topologies: corners in bold, edges in grey, numbered as
`soma::canonical` gives them (see [canonical side tables](topology.md#canonical-side-tables)).
An element's connectivity lists its corners in this order.
```

## Integer and real types

`soma/core/types.hpp` fixes the types that every array in SOMA uses:

| Type | Default | Meaning |
| --- | --- | --- |
| `LO` | `std::int32_t` (`std::int64_t` with `SOMA_LO_64`) | local ordinal: an entity index on this rank |
| `Offset` | `std::int32_t` (`std::int64_t` with `SOMA_OFFSET_64`) | CSR offsets (connectivity, adjacency rows) |
| `GO` | `std::int64_t` | global ordinal: identity across ranks |
| `Real` | `double` | coordinates and geometry |

`Offset` is separate from `LO` because the number of connectivity or adjacency entries
overflows 32 bits long before the number of entities does. `SOMA_LO_64` and `SOMA_OFFSET_64`
are CMake options recorded in the generated `soma/config.hpp`; both change the ABI, so an
application must be compiled against the same configuration as the library (see
[CMake options](../reference/cmake-options.md)). Counts that do not fit are reported rather
than truncated: for example, a box mesh whose counts exceed `LO` or `Offset` throws
`std::overflow_error`, and the CSR scans throw `std::length_error`.

The sentinels are `invalid_lo = -1` and `invalid_go = -1`.

## Global ids and local ids

Every entity has two names.

- Its **gid** (global id, a `GO`) is its identity across ranks and across runs. For a mesh read
  from a MOAB `.h5m` file it is the file id. Gids are unique *within a dimension*: two elements
  of one dimension may not share a gid, whatever their topologies, but vertex 1 and edge 1 are
  different entities. Gids need not be contiguous or start anywhere in particular.
- Its **lid** (local id, an `LO`) is its index on this rank, in $[0, \text{count}(d))$. Lids
  index every per-entity array: connectivity, coordinates, adjacency rows, field rows. They are
  assigned by the builder's ordering (below) and can change when the mesh changes (ghosting,
  compression, migration, derivation of sides); gids do not.

`mesh.gid(d)` is the View of gids by lid. Connectivity is stored by lid, so going from lid to gid
is an array read; going from gid to lid needs a search (the builder does this once when it
translates its gid-based input).

## Ownership and segment tables

On a distributed mesh an entity can be held by several ranks. Each rank classifies every entity
it holds by `soma::Ownership` (`soma/core/segments.hpp`):

| Kind | Meaning |
| --- | --- |
| `interior` | held by this rank only |
| `shared` | held by several ranks and **owned** by this one |
| `shared_copy` | held by several ranks and owned by another one; a copy of the owner's entity |
| `ghost` | a copy of another rank's entity held for neighbourhood access; never owned here |

An entity is **owned** when it is `interior` or `shared`. A one-rank mesh (anything
`MeshBuilder::build` returns) has interior entities only. Shared entities are created by
[`build_local_part`](../parallel/distributed-meshes.md) and ghosts by
[`Mesh::add_ghosts`](../parallel/ghosts-and-halos.md).

Each dimension is sorted by ownership first, then topology, then arity, so each ownership kind
is one contiguous range of lids and ownership is a comparison with a range end. The
**segment table** records the ranges:

```cpp
struct Segment {
  LO begin = 0;
  LO end = 0;
  Topology topology = Topology::vertex;
  std::uint16_t arity = 0;
  Ownership ownership = Ownership::interior;
  LO size() const noexcept { return end - begin; }
};

struct SegmentTable {
  std::vector<Segment> segments;
  LO count = 0;
  LO interior_end = 0;  // [0, interior_end): interior
  LO owned_end = 0;     // [interior_end, owned_end): shared and owned here
  LO shared_end = 0;    // [owned_end, shared_end): shared copies; the rest ghosts
};
```

So for dimension `d`:

| Lids | Entities |
| --- | --- |
| `[0, interior_end)` | interior |
| `[interior_end, owned_end)` | shared, owned here |
| `[0, owned_end)` | everything owned here |
| `[owned_end, shared_end)` | shared copies |
| `[shared_end, count)` | ghosts |

```{figure} ../figures/img/segment-table.svg
:alt: Four bars, the vertices and the cells of rank 0 and of rank 1, each divided into coloured lid ranges: interior, shared and owned here, shared copy, and ghost, with the lid where each range ends and the names of the table's ends
:width: 100%

The segment tables of `data/hybrid_3quad_1tri.h5m` read on two ranks with one ghost layer. Rank 0
owns the 14 vertices the ranks share and rank 1 holds copies of them; cells are never shared, so
their tables have `interior_end = owned_end = shared_end`. Within a kind, the segments split by
topology: the triangles come before the quads.
```

A **segment** is a maximal run of entities with one ownership kind, topology and arity (vertex
count; for vertices 1). Uniform arity lets a kernel be instantiated with a compile-time vertex
count per segment. `mesh.segments(d)` returns the segments and `mesh.segment_table(d)` the whole
table; `kind_bounds(table)` returns the five bounds so that kind `k` is
`[bounds[k], bounds[k + 1])`. Segments are host-side; kernels get the range ends through
`MeshView`.

## The `Mesh` class

`soma::Mesh<MemSpace>` (`soma/core/mesh.hpp`) owns one rank's mesh in one Kokkos memory space.
Two aliases cover the usual cases:

```cpp
using HostMesh = Mesh<Kokkos::HostSpace>;
using DeviceMesh = Mesh<Kokkos::DefaultExecutionSpace::memory_space>;
```

In a build without a GPU backend the two are the same type. Construction, editing, ghosting,
migration and file I/O work on a `HostMesh`; kernels can run on a mesh in any space.

### Ownership semantics

A `Mesh` is move-only. Copying is deleted because copying a `Kokkos::View` aliases its data: a
silent alias would make a "copy" and its original change together. Deep copies are spelled
`clone()` (same space) or `mirror<Space>()` (any space). A moved-from mesh is genuinely empty
(zero entities of every dimension), not an alias of the moved-to one.

### Per-dimension data

Each dimension holds a CSR connectivity block. The accessors take a dimension in $[0, 3]$ and
throw `std::out_of_range` otherwise:

| Accessor | Contents |
| --- | --- |
| `count(d)` | number of entities (removed ones included on an edited mesh) |
| `conn_offsets(d)` | `count(d) + 1` offsets into `conn(d)`; empty for vertices |
| `conn(d)` | corner vertex lids of each entity, in the order given (MOAB's vertex order for the topology) |
| `topology(d)` | one `std::uint8_t` per entity: its `Topology` |
| `gid(d)` | one `GO` per entity |
| `segments(d)`, `segment_table(d)` | the segment table above |
| `input(d)` | true when the builder was given entities of `d` (vertices always) |
| `has_nodes(d)`, `node_offsets(d)`, `nodes(d)` | high-order nodes (below) |
| `alive(d)`, `tail(d)`, `num_live(d)` | edit state (see [editing meshes](editing.md)) |

All Views returned are const and `LayoutLeft` (`soma/core/view_types.hpp` defines `View1`,
`View2` and the unmanaged, read-only `UView1` used in kernel handles).

### Whole-mesh data

| Accessor | Contents |
| --- | --- |
| `dim()` | mesh dimension |
| `coords()` | vertex coordinates (below); writable |
| `adjacency(from, to)`, `bridge(dim, via)`, `has_adjacency`, `has_bridge`, `adjacencies()` | adjacency graphs (see [topology](topology.md)) |
| `sets()` | entity sets (see [entity sets](sets.md)) |
| `fields()` | the field registry (see [fields](fields.md)) |
| `values()` | values of the mesh as a whole (see [fields](fields.md)) |
| `ordering()` | the ordering the builder used |
| `ghost_layers()`, `ghost_bridge_dim()` | ghost layers last placed by `add_ghosts` (0 and 0 without ghosts) |
| `epoch()` | topology version (below) |
| `edited()` | true while the mesh holds edits not yet compressed |

### The epoch

`epoch()` is a topology version. `MeshBuilder::build` starts it at 1, and an operation that
changes topology in a way that invalidates field domains calls `bump_epoch()`, which also tells
the field registry. Fields defined at an older epoch are then **stale**: fetching one throws
`StaleField`. The rules for each operation are on the pages that describe it
([topology](topology.md), [fields](fields.md), [editing](editing.md)).

## Coordinates

`mesh.coords()` returns a `CoordView<MemSpace>`, a `Kokkos::View<Real*[3], LayoutLeft, MemSpace>`
with one row per vertex lid. Every mesh stores three components; 2D meshes (and box meshes of
quads or triangles) have `z = 0`. The coordinates are the one writable part of a mesh's
geometry: algorithms such as [Laplacian smoothing](../algorithms/smoothing.md) move vertices in
place. Units are whatever the input used; SOMA does not convert them.

## High-order elements

An element can carry nodes beyond its corners, as MOAB's high-order elements do. The
connectivity `conn(d)` always holds the corners only, so every algorithm that needs only the
linear shape works unchanged; the extra nodes are kept apart:

- `node_offsets(d)` (`count(d) + 1` entries) and `nodes(d)` list, per element, the vertex lids of
  its nodes beyond its corners, in MOAB's order. Both are empty (`has_nodes(d)` is false) when no
  element of `d` has any.
- MOAB's order is: corners, then one node on each edge in canonical edge order, one on each face
  in canonical face order, and one in the region, as each configuration has them.
  `soma/topology/nodes.hpp` gives `mid_node_bits(t, count)` (the configuration with `count`
  nodes, or -1), `num_nodes(t, bits)`, and `node_parent(t, count, k)` (the dimension and
  canonical index of the sub-entity node `k` sits on). Edges, triangles, quads, tets, pyramids,
  prisms and hexes have mid-node configurations; knives have corners only.
- Vertices that are only high-order nodes (no element's corner) follow the corner vertices of
  their ownership kind, in segments of their own: kind `k`'s corners are
  `[kind begin, corner_end(k))` and its node-only vertices `[corner_end(k), kind end)`.
  `corners()` is `corner_end(Ownership::interior)`; on a one-rank mesh `[0, corners())` are the
  corner vertices.

`MeshView::num_nodes(d, e)` and `MeshView::node(d, e, k)` give corners and nodes together, in
that order.

## Entity ordering and locality

`soma::Ordering` (`soma/core/ordering.hpp`) selects how `MeshBuilder::build` orders each
dimension within the ownership, topology and arity keys above:

| Value | Order |
| --- | --- |
| `Ordering::file` | elements by gid; vertices by gid |
| `Ordering::type_major_sfc` | elements by the Hilbert key of their corners' centroid, then gid; vertices by Hilbert key, then gid |
| `Ordering::tiled` | declared but not implemented: `build` throws `std::invalid_argument` |

`default_ordering` is `Ordering::type_major_sfc`. The Hilbert keys use one bounding box, that of
the vertices (or the box given by `MeshBuilder::set_key_box`, which distributed builds use so
that keys agree across ranks). Ordering along a space-filling curve keeps the vertices of one
element close together in memory. `mesh.ordering()` reports the ordering used, and `compress()`
after an edit orders the mesh the same way.

`soma/core/locality.hpp` measures the result: `vertex_span_report(mesh, dim)` returns, over the
elements of dimension `dim` (1 to 3; `std::out_of_range` otherwise), the mean and maximum of
`max - min` of each element's vertex lids. Smaller spans mean an element gathers its coordinates
from a narrower window. It reads on the host, so a device mesh must be mirrored first.

To impose an ordering of your own, `permute_entities(mesh, dim, new_of_old, segments, gids = {})`
reorders one dimension of a host mesh: entity `e` becomes `new_of_old[e]`, and gids, topology,
connectivity, nodes, coordinates, held graphs, set members and fields follow. The new segment
table must be the one `make_segments` gives for the reordered entities; the call throws
`std::invalid_argument` otherwise, or for an argument that is not a permutation, and
`std::logic_error` on an edited mesh. It advances the epoch and leaves the mesh unchanged when
it throws.

## Copies in other memory spaces

A mesh is copied into another memory space with `mirror`:

```cpp
template <class Dst> Mesh<Dst> mirror(const TransferOptions& opts = {}) const;       // always allocates
template <class Dst> Mesh<Dst> mirror_view(const TransferOptions& opts = {}) const;  // aliases in the same space
Mesh clone() const;                                                                 // deep, everything
```

- `mirror<Dst>()` always allocates in `Dst`, even when `Dst` is the source's space.
- `mirror_view<Dst>()` shares storage with the source when `Dst` is the same memory space, and
  copies otherwise (even when `Dst` can access the source's memory). Writes through either then
  reach both; name the result accordingly.
- `clone()` is a deep copy in the same space with every field and every held graph.

What a mirror copies is chosen by `soma::TransferOptions` (`soma/core/transfer.hpp`). The
default is lean:

| Member | Default | Meaning |
| --- | --- | --- |
| `dims` | `{true, true, true, true}` | dimensions to copy |
| `coordinates` | `true` | copy vertex coordinates |
| `fields` | `{}` | field names to copy; `{"*"}` for all |
| `adjacencies` | empty | graphs the destination must hold |
| `adj_source` | `AdjSource::rebuild` | `copy` the graphs from the source, or `rebuild` them in the destination |
| `sets` | `true` | copy entity sets |

`TransferOptions::everything()` also sets `fields = {"*"}`. Element connectivity and
coordinates refer to vertices, so keeping either without `dims[0]` throws
`std::invalid_argument`, as does asking for a graph between dimensions that are not both kept.
With `AdjSource::copy` the source must hold each requested graph (`MissingAdjacency`
otherwise); with `rebuild` the destination builds them from its own topology with
[`build_adjacencies`](topology.md). Sets keep the members of the kept dimensions; asking a
mirror for the members of a dropped one throws `MissingSetMembers`. Mirrors carry the epoch,
the ordering, the ghost layer record and the [mesh values](fields.md).

From `tests/unit/test_mesh_fields.cpp`, a device copy carrying one field:

```cpp
soma::TransferOptions opt;
opt.fields = {"temperature"};
const auto dev = mesh.mirror<Kokkos::DefaultExecutionSpace::memory_space>(opt);
```

## `MeshView`: the kernel handle

Kernels never capture a `Mesh`. They capture a `soma::MeshView<MemSpace>`
(`soma/core/mesh_view.hpp`), a trivially copyable bundle of unmanaged Views and range ends,
obtained from `mesh.view()`. It holds no reference counts, so copying it into a lambda costs
nothing per launch. A view is valid only while its mesh is alive and its topology unchanged;
`view()` on a temporary mesh is deleted so that a view cannot dangle at once.

Device-callable members:

| Member | Returns |
| --- | --- |
| `dim`, `count(d)` | mesh dimension, entity count |
| `topology(d, e)`, `num_vertices(d, e)`, `vertex(d, e, k)` | topology, corner count and corner `k` of entity `e` |
| `num_nodes(d, e)`, `node(d, e, k)` | corners and high-order nodes together |
| `x(v, k)` | coordinate `k` of vertex `v` |
| `gid(d, e)` | global id |
| `interior(d, e)`, `owned(d, e)`, `shared(d, e)` | ownership tests by range end; a ghost is `e >= shared_ends[d]` |
| `corner(v)` | false for a vertex that is only a high-order node |
| `alive(d, e)` | false for an entity removed by an edit not yet compressed |
| `adj(from, to, e)`, `adj_codes(from, to, e)`, `bridge(d, via, e)` | adjacency rows (see [topology](topology.md)) |

The public arrays (`counts`, `interior_ends`, `owned_ends`, `shared_ends`, `offsets`, `conn`,
`gids`, `coords`, ...) can also be read directly.

A reduction over the regions of a box mesh on the default device, adapted from
`tests/unit/test_box_mesh.cpp`:

```cpp
#include "soma/core/mesh.hpp"
#include "soma/testing/box_mesh.hpp"

using Dev = Kokkos::DefaultExecutionSpace::memory_space;

const soma::HostMesh host = soma::testing::box_mesh({4, 4, 4, soma::Topology::hex});
const soma::Mesh<Dev> dev = host.mirror<Dev>();
const soma::MeshView<Dev> v = dev.view();
std::int64_t total = 0;
Kokkos::parallel_reduce(
    "gid_sum", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.count(3)),
    KOKKOS_LAMBDA(const soma::LO e, std::int64_t& acc) { acc += v.gid(3, e); }, total);
```

Under the `SOMA_DEBUG` CMake option, `MeshView` checks in kernels that a requested adjacency is
held and aborts otherwise; without it no check is made.

## Handles

MOAB names an entity by a handle that packs its type and id. SOMA stores entity references in
fields the same way, through *handle fields* (`FieldOptions::handle`, see [fields](fields.md)).
`soma/core/handle_ref.hpp` defines the encoding: a handle names a slot (a dimension 0 to 3, or
`set_dim` = 4 for the entity sets) and a gid, with the slot + 1 in the bits above bit 60 and the
gid below, so 0 is the null handle.

| Function | Meaning |
| --- | --- |
| `make_handle(slot, gid)` | the handle, or 0 if the slot is outside $[0, 4]$ or the gid outside $[0, 2^{60} - 1]$ |
| `handle_slot(h)` | the slot, or -1 for the null handle and values no `make_handle` gives |
| `handle_gid(h)` | the gid (meaningful when `handle_slot(h) >= 0`) |

`HandleRef<MemSpace>::build(mesh)` takes a snapshot of a mesh's gids and resolves handles to a
`HandleTarget{slot, lid}` on the host or in kernels (`resolve(h)`); handles the mesh does not
hold resolve to an invalid target. It throws `std::invalid_argument` if a gid repeats within a
slot or cannot be held in a handle. Build it again after the mesh changes.

## Summary of terms

| Term | Meaning |
| --- | --- |
| dimension | 0 vertices, 1 edges, 2 faces, 3 regions |
| element | an entity of the mesh dimension |
| side | an entity of a lower dimension than the elements (an edge, or a face in 3D) |
| gid | global id: identity of an entity within its dimension, across ranks |
| lid | local id: index of an entity on this rank |
| interior / shared / shared copy / ghost | the four ownership kinds |
| owned | interior or shared |
| segment | a run of entities with one ownership kind, topology and arity |
| segment table | the segments of a dimension and its range ends |
| slot | a dimension 0 to 3, or `set_dim` (4) for the entity sets |
| epoch | the topology version that field domains are checked against |
| corner / node | corners define the linear shape; nodes are the extra points of high-order elements |
