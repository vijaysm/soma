# Topology and adjacencies

A built mesh holds its vertices, its elements and any sides it was given, with the
element-to-vertex connectivity. Everything else (edges and faces, upward graphs such as
vertex-to-element, element-to-element neighbours) is built on request by
`build_adjacencies`. This page describes the graphs, how to request and read them, how edges and
faces are derived and numbered, and the canonical side tables they follow, which match MOAB's.

## Graph kinds

Adjacency graphs are CSR arrays between entity dimensions (`soma/topology/adjacency.hpp`):

| Graph | Rows | Codes |
| --- | --- | --- |
| `(d, 0)`, $d \ge 1$ | the connectivity itself; always present | none |
| `(from, to)`, from > to (downward) | the sub-entities of each entity, in the parent's canonical side order | orientation: `(offset << 1) \| reversed`, the stored sub-entity against the parent's canonical side (MOAB's `CN::SideNumber`) |
| `(from, to)`, from < to (upward) | the entities each entity is a side (or vertex) of, sorted by (target, code) | local side index in the parent (local vertex index for `from = 0`) |
| bridge `(dim, via)` | the other entities of dimension `dim` that share an entity of dimension `via`, sorted | none |

For example, `(3, 2)` gives each region its faces, `(0, 3)` each vertex its regions, and the
bridge `(3, 2)` each region its face neighbours. Every graph is a deterministic function of the
topology, whatever the backend or the thread count: upward rows are built with atomics and then
sorted, so the fill order of threads does not show.

## Requesting graphs

`soma::AdjacencyRequest` names a set of graphs:

```cpp
class AdjacencyRequest {
 public:
  AdjacencyRequest& add(int from, int to);
  AdjacencyRequest& add_bridge(int dim, int via);
  bool has(int from, int to) const noexcept;
  bool has_bridge(int dim, int via) const noexcept;
  bool empty() const noexcept;
  std::uint32_t bits() const noexcept;
  static AdjacencyRequest from_bits(std::uint32_t bits);
  friend AdjacencyRequest operator|(AdjacencyRequest a, AdjacencyRequest b) noexcept;
};
```

`add` and `add_bridge` return the request, so they chain; they throw `std::invalid_argument` for
a dimension outside $[0, 3]$ or `from == to`.

`build_adjacencies` (`soma/topology/build.hpp`) is the one entry point:

```cpp
template <class MemSpace>
void build_adjacencies(Mesh<MemSpace>& mesh, AdjacencyRequest req);

template <class ExecSpace>
void build_adjacencies(Mesh<typename ExecSpace::memory_space>& mesh, AdjacencyRequest req,
                       const ExecSpace& exec);
```

From `tests/unit/test_build_adjacencies.cpp`, on a $3 \times 3 \times 3$ box of hexes:

```cpp
soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(2, 3));
// mesh.count(2) == 108: the faces were derived
// mesh.count(1) == 0:   edges were not asked for
// mesh.adjacencies() == AdjacencyRequest{}.add(2, 3)
```

Behaviour:

- It derives the side dimensions that the requested graphs touch, builds every graph the
  requested ones depend on, and keeps only the requested ones, together with the graphs the mesh
  held before. Intermediate graphs are dropped.
- Graphs and derived entities are reused: a second call with the same request does nothing, and
  a later request does not derive held sides again.
- It works on a mesh in any memory space; the two-argument form runs in the default execution
  space of the mesh's memory space (the default device space for a device mesh, the default host
  space for a host mesh). For memory spaces without a default execution space, pass one.
- **Strong guarantee:** everything is built on an aliasing working copy and installed at the
  end with non-throwing moves; a throw leaves the mesh as it was.
- It throws `std::invalid_argument` for a graph involving a dimension above `mesh.dim()`, when
  sides must be derived but the mesh has vertices and no elements of its dimension, for a device
  mesh holding removed entities (see [editing](editing.md)), and when explicit sides cannot be
  adopted (below).

### Epoch and fields

Deriving a dimension that held explicit entities renumbers them, and deriving one that a live
field is defined on changes that field's domain: either advances the [epoch](mesh-model.md), so
every field becomes stale. Filling an empty dimension that no field uses keeps the epoch, so
fields stay current. To keep fields current, request the sides you need before defining fields
on them. Entity sets follow a renumbering: their members keep their gids.

On a host mesh holding removed entities, the graphs are built over the live entities; see
[editing meshes](editing.md).

### Distributed meshes

On a distributed mesh, call the collective `build_distributed_adjacencies(comm, mesh, request)`
(`soma/parallel/distributed.hpp`) instead. It builds the graphs on the local part, then gives
each newly derived dimension its ownership, ordering and sharing, as the vertices have, with
gids that do not depend on the number of ranks. See
[distributed meshes](../parallel/distributed-meshes.md).

## Reading graphs

On the host, `Mesh` gives:

| Member | Returns |
| --- | --- |
| `has_adjacency(from, to)` | whether the graph is held (`(d, 0)` for $1 \le d \le$ `dim()` is the connectivity) |
| `has_bridge(dim, via)` | whether the bridge is held |
| `adjacency(from, to)` | an `AdjacencyGraph<MemSpace>`: `offsets`, `targets`, `codes` (const Views) |
| `bridge(dim, via)` | the bridge, with empty `codes` |
| `adjacencies()` | the graphs held, as a request |

`adjacency` and `bridge` throw `MissingAdjacency` (a `std::logic_error`) for a graph the mesh
does not hold, and `std::invalid_argument` for a pair that is not two distinct dimensions.
Row `e` of a graph is `targets[offsets[e] .. offsets[e + 1])`.

In kernels, `MeshView` (see [the mesh model](mesh-model.md)) returns rows as
`Row<const LO>`, a pointer and a size valid while the mesh lives. From
`tests/unit/test_adjacency_store.cpp`:

```cpp
const soma::MeshView<M> v = mesh.view();
Kokkos::parallel_reduce(
    "weighted_row_sum", Kokkos::RangePolicy<typename M::execution_space>(0, v.count(2)),
    KOKKOS_LAMBDA(const soma::LO e, long long& acc) {
      const soma::Row<const soma::LO> r = v.adj(2, 1, e);              // the face's edges
      const soma::Row<const std::uint8_t> c = v.adj_codes(2, 1, e);    // their orientations
      for (soma::LO i = 0; i < r.size; ++i) acc += 100LL * r[i] + c[i];
      const soma::Row<const soma::LO> b = v.bridge(2, 0, e);           // faces sharing a vertex
      for (soma::LO i = 0; i < b.size; ++i) acc += 7LL * b[i];
      const soma::Row<const soma::LO> verts = v.adj(2, 0, e);          // the connectivity
      for (soma::LO i = 0; i < verts.size; ++i) acc += 1000000LL * verts[i];
    },
    total);
```

`adj_codes` is empty for `(d, 0)`. Kernels do not check that a graph is held unless SOMA is
built with `SOMA_DEBUG`, which aborts on a missing one.

## Derived edges and faces

Sides of dimension $k$ (edges; faces in 3D) are derived from the elements of the mesh
dimension $D$ (`soma/topology/derive.hpp`) by a *lowest-creator* rule: side $s$ of element $e$
has a vertex set $S$; among the elements around the lowest vertex of $S$ whose sides include
$S$, the one with the smallest (element, side) pair creates the sub-entity. This needs no hash
table and no global sort; each side costs a scan of one vertex's elements.

- **Numbering:** derived entities are ordered by (topology, arity), then by (creator element,
  side), so they follow the element order and each segment is contiguous.
- **Connectivity:** the creator's side, in the creator's canonical order. Other elements see the
  side with an orientation code (below).
- **Gids:** on a single part, new entities are numbered from the largest explicit gid of the
  dimension plus one, in local id order. On a distributed mesh,
  `build_distributed_adjacencies` gives gids that do not depend on the number of ranks.
- **Explicit sides** (sides given to the builder) are adopted: the derived entity with the same
  vertex set keeps the explicit gid and the explicit vertex order (and its high-order nodes).
  An explicit entity that is no side of any element, or two explicit entities with one vertex
  set, throw `std::invalid_argument` naming the gid.
- Downward graphs between existing entities (for example faces to edges) look up each side among
  the entities around its lowest vertex; a side with no such entity throws `std::logic_error`.

The derived topology is checked against MOAB's `get_adjacencies` (sub-entities, downward graphs
with orientation, upward graphs and bridges) on every bundled `.h5m` mesh by
`tests/reference/test_adjacency_moab.cpp`.

## Canonical side tables

`soma/topology/canonical.hpp` gives the sides of each fixed topology as local vertex indices.
The values mirror `moab::CN::SubEntityVertexIndices` and are compared position by position
against MOAB by `tests/reference/test_canonical_moab.cpp`. The vertex order of a side encodes its
sense relative to the parent, so it must never be sorted. The functions are `constexpr` and
callable in CUDA, HIP and SYCL kernels.

```cpp
namespace soma::canonical {
struct SideDef {
  Topology topology = Topology::vertex;
  std::uint8_t num_vertices = 0;
  std::uint8_t v[4] = {0, 0, 0, 0};
};
constexpr int num_sides(Topology topo, int dim) noexcept;   // sides of dimension dim (1 or 2)
constexpr SideDef side(Topology topo, int dim, int i) noexcept;  // default SideDef when out of range
}
```

Edges (`dim = 1`):

| Topology | Edges |
| --- | --- |
| tri | (0,1) (1,2) (2,0) |
| quad | (0,1) (1,2) (2,3) (3,0) |
| tet | (0,1) (1,2) (2,0) (0,3) (1,3) (2,3) |
| pyramid | (0,1) (1,2) (2,3) (3,0) (0,4) (1,4) (2,4) (3,4) |
| prism | (0,1) (1,2) (2,0) (0,3) (1,4) (2,5) (3,4) (4,5) (5,3) |
| knife | (0,1) (1,2) (2,3) (3,0) (0,4) (1,5) (2,6) (3,5) (4,5) (5,6) |
| hex | (0,1) (1,2) (2,3) (3,0) (0,4) (1,5) (2,6) (3,7) (4,5) (5,6) (6,7) (7,4) |

Faces (`dim = 2`):

| Topology | Faces |
| --- | --- |
| tri, quad | the element itself: (0,1,2), (0,1,2,3) |
| tet | (0,1,3) (1,2,3) (0,3,2) (0,2,1) |
| pyramid | (0,1,4) (1,2,4) (2,3,4) (3,0,4) (0,3,2,1) |
| prism | (0,1,4,3) (1,2,5,4) (0,3,5,2) (0,2,1) (3,4,5) |
| knife | (0,1,5,4) (1,2,6,5) (2,3,5,6) (3,0,4,5) (0,3,2,1) |
| hex | (0,1,5,4) (1,2,6,5) (2,3,7,6) (3,0,4,7) (0,3,2,1) (4,5,6,7) |

An edge has one edge, itself. Polygons have no table: their edges are the vertex cycle, edge $i$
running from vertex $i$ to vertex $(i + 1) \bmod n$.

`soma/topology/sides.hpp` applies the tables to a mesh entity, in kernels: `topology::side_count(m,
d, e, k)` is the number of sides of dimension $k$ of entity `e`, and `topology::side_of(m, d, e,
k, s)` returns side `s` as a `topology::Side` holding its topology and mesh vertex lids. Both
handle polygons.

### Orientation codes

`soma/topology/orientation.hpp` compares two vertex lists naming the same entity, with the
semantics of MOAB's `CN::ConnectivityMatch(a, b)`:

```cpp
namespace soma::topology {
struct Match { bool same = false; int offset = 0; bool reversed = false; };
Match connectivity_match(const LO* a, const LO* b, int n);
constexpr std::uint8_t orientation_code(int offset, bool reversed);
constexpr int code_offset(std::uint8_t code);
constexpr bool code_reversed(std::uint8_t code);
constexpr int code_sense(std::uint8_t code);  // +1 as stored, -1 reversed
}
```

`offset` is the position of `b[0]` in `a`, and `reversed` says that `b` runs through `a`
backwards; for two vertices, `b` is reversed whenever it is not identical to `a`. The codes of a
downward graph are `orientation_code(offset, reversed)` of the stored sub-entity against the
parent's canonical side, so `code_sense` gives the sign with which an element sees a face or an
edge (for fluxes, for example).

## Building blocks

The pieces `build_adjacencies` uses are public, for code that builds graphs of its own:

- `topology::transpose` (`soma/topology/transpose.hpp`) builds an upward graph from a downward
  one: count with atomics, scan, fill, then sort each row by (source, local index) so that the
  result does not depend on the threads. It throws `std::length_error` if a source row is longer
  than 256 entries (the local index must fit the code) or if the total does not fit `Offset`.
- `topology::bridge` (`soma/topology/bridge.hpp`) builds a bridge from a graph `dim -> via` and
  its transpose: each row gathers the candidates, then sorts and deduplicates them.
