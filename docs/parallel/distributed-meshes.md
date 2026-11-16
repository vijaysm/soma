# Distributed meshes

A distributed mesh in SOMA is one `HostMesh` per rank, the *local part*, holding the elements
the rank was given and every entity they use. This page explains how a local part is built,
how ownership and sharing are recorded, how edges and faces derived after the build agree on
their global ids, and how elements are partitioned, migrated and rebalanced. For ghost layers
and exchanging values between parts, see [ghosts and halo exchange](ghosts-and-halos.md).

## Owned, shared and ghost entities

Every entity has a global id (gid) and, on each rank holding it, a local id (lid). On a rank,
each entity of each dimension is one of four kinds (`soma::Ownership`, `soma/core/segments.hpp`):

| Kind | Meaning |
|---|---|
| `interior` | Held only by this rank. |
| `shared` | Held by several ranks and owned by this one. |
| `shared_copy` | Held by several ranks and owned by another; a copy of the owner's entity. |
| `ghost` | A copy of another rank's entity added for its neighbourhood (see [ghosts](ghosts-and-halos.md)); never owned. |

The owner of an entity held by several ranks is **the lowest rank holding it** (other than as a
ghost). Each dimension is ordered by kind, so each kind is a contiguous range of lids and
ownership is a comparison with the ends of the dimension's `SegmentTable`
(`mesh.segment_table(d)`):

| Range | Entities |
|---|---|
| `[0, interior_end)` | interior |
| `[interior_end, owned_end)` | shared, owned here |
| `[owned_end, shared_end)` | shared copies |
| `[shared_end, count)` | ghosts |

So `[0, owned_end)` is what this rank owns: a loop over it visits every entity of the global
mesh exactly once across the ranks. Within a kind, entities are further ordered by topology,
arity and node count, then by *neighbour set* (the set of other ranks holding them, so entities
shared with the same ranks are contiguous), then by the mesh's [ordering](#orderings).

Kernels test ownership with these ends and never read per-entity flags. The sharing is kept in
two internal fields, from which exchanges are built once:

| Field | Defined on | Contents |
|---|---|---|
| `soma:sharing` (`soma::sharing_field`) | the shared entities of both kinds; row `e - interior_end` | variable-length `int`: the other ranks holding the entity, ascending |
| `soma:owner` (`soma::owner_field`) | the shared copies and ghosts; row `e - owned_end` | `int`: the owning rank |

Both are declared in `soma/parallel/distributed.hpp`. This snippet, from
`tests/unit/test_distributed_mesh.cpp`, reads them:

```cpp
const soma::HostMesh m = soma::build_local_part(c, std::move(builder));
const auto& t = m.segment_table(0);
const auto sharing = m.fields().get_var<int>(soma::sharing_field, 0);
const auto owner = m.fields().get<int>(soma::owner_field, 0);
// sharing.rows() == t.shared_end - t.interior_end
// owner.rows()   == t.shared_end - t.owned_end
```

## Building a local part

`soma/parallel/distributed.hpp` declares three overloads of `build_local_part`. Each is
collective and returns this rank's part:

```cpp
[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        Ordering ordering = default_ordering);

[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        const std::array<std::vector<GO>, 4>& ghosts,
                                        Ordering ordering = default_ordering);

[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        const std::array<std::optional<Sharing>, 4>& sharing,
                                        Ordering ordering = default_ordering);
```

Each rank fills a `MeshBuilder` with the elements it holds and every vertex they use, naming
everything by gid; the same gid on two ranks is the same entity. From `examples/remap.cpp`:

```cpp
soma::MeshBuilder b(dim);
b.add_vertices(verts, xyz);   // gids, then x, y, z interleaved
b.add_elements(dim == 3 ? soma::Topology::tet : soma::Topology::tri, cells, conn);
return soma::build_local_part(comm, std::move(b));
```

What the first overload does:

- Every dimension the builder holds on some rank gets its ownership and sharing. Vertices held by
  several ranks are shared; so are explicit edges or faces given to the builder on several ranks.
- Curve keys for the [ordering](#orderings) are computed in the box of every rank's vertices, so
  keys are comparable across ranks.
- It throws as `MeshBuilder::build` and `resolve_sharing` do, **on every rank** if any rank's
  input is invalid.

The second overload also takes, per dimension, the gids this rank holds *only as ghosts* (through
ghost elements). They are never shared, come after the shared copies ordered by owner, and
`soma:owner` gives their owner, the lowest rank holding them otherwise. It throws
`std::invalid_argument` on every rank if a rank names a ghost it does not hold or one no rank
holds otherwise. Most users get ghosts from `Mesh::add_ghosts` instead.

The third overload is for sharing already known, for example from a file read back at its part
count: `sharing[d]`, when given, lists for each gid of dimension `d` the builder holds, in the
builder's order (`vertex_gids()`, `element_gids(d)`), the other ranks holding it, and the gid
directory is not consulted for `d`. The lowest rank holding an entity owns it. The ranks must
agree (what one rank says it shares with another, that one says too); **this is not checked**.
A sharing without one row per gid throws `std::invalid_argument` on every rank.

### Resolving sharing

`soma/parallel/sharing.hpp` declares the primitive behind the first two overloads:

```cpp
[[nodiscard]] Sharing resolve_sharing(const Comm& comm, std::span<const GO> gids,
                                      std::span<const std::uint8_t> ghost = {});
```

Collective: for the gids this rank holds, which other ranks hold each. Every gid goes to a
directory rank (`transport::owner` of the gid), which sends each holder of a gid held more than
once the other holders. Traffic and memory scale with the shared gids; a gid held on one rank
costs one record. `ghost[i] != 0` marks gid `i` as held here only as a ghost: ghost holders learn
its owner, and no other rank learns of them. It throws `std::invalid_argument` on every rank if
any rank names a gid twice, gives flags but not one per gid, or holds as a ghost a gid no rank
holds otherwise.

The result, `Sharing`, holds per gid given, in order:

| Member | Meaning |
|---|---|
| `offsets`, `ranks` | The other ranks holding gid `i`, ascending: `ranks[offsets[i]]` to `ranks[offsets[i + 1] - 1]` (empty for an interior gid). |
| `ghost_owner` | The owner of a gid held here as a ghost, else -1; empty when no ghosts were given. |
| `kind(i, me)` | `interior`, `shared`, `shared_copy` (owned by `ranks[offsets[i]]`) or `ghost`. |
| `owner(i, me)` | The lowest rank holding gid `i` other than as a ghost. |
| `groups()` | Per gid, a group number equal for gids held by the same other ranks, numbering the distinct rank sets in ascending lexicographic order, so groups agree in order on every rank. |

## Edges and faces derived on a distributed mesh

```cpp
void build_distributed_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request);
```

Collective: builds the requested graphs on a local part, then gives each newly derived
dimension (edges, faces) its distributed form: ownership, ordering by ownership and neighbour
set, and `soma:sharing` and `soma:owner` as the vertices have. It throws on every rank if any rank
fails. From `tests/unit/test_ghosts.cpp`:

```cpp
soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
```

On a distributed part, use this rather than plain `build_adjacencies`, which knows nothing of
the other ranks.

**How the ranks agree on gids of derived entities.** A derived entity is identified by its
sorted corner gids. Its gid is the largest explicit gid of its dimension on any rank, plus one,
plus the rank (position) of its sorted corner gids among all such keys over every rank, found by
a distributed sort (`transport::rank_keys`, see [transport](transport.md)). The gid therefore
depends only on the mesh, **not on the number of ranks or on the partition**. Explicit entities
(those given to the builder) keep their gids and must be present on every rank holding them.

## Orderings

```cpp
enum class Ordering : std::uint8_t { file, type_major_sfc, tiled };
inline constexpr Ordering default_ordering = Ordering::type_major_sfc;
```

(`soma/core/ordering.hpp`.) Ownership always comes first in the order of a dimension, then
topology, arity and node count, then the neighbour-set group; within that, `MeshBuilder::build`
orders by:

| Ordering | Elements | Vertices |
|---|---|---|
| `file` | gid | gid |
| `type_major_sfc` (default) | Hilbert key of the corners' centroid, then gid | Hilbert key, then gid |
| `tiled` | not supported: throws | |

Within each kind, vertices that are nodes of high-order elements and no element's corner come
after the others. The Hilbert ordering places entities that are close in space close in memory.

### Space-filling curve keys

`soma/core/sfc.hpp` (namespace `soma::sfc`) computes the curve keys. They work in device code,
except `KeyBox::make`, which runs on the host.

- `KeyBox::make(lo, hi)` maps a bounding box to a grid. Only the axes on which the box has
  extent take part, so a planar mesh gets a 2D curve and a single point gets key 0; an axis whose
  extent is at most one cell of the widest axis is dropped too. Every kept axis is divided by the
  widest extent, so the grid covers a cube and keeps the shape of the mesh.
- The grid has `bits_per_axis(n)` bits per axis: 32 for one or two axes, 21 for three, so a key
  fits 64 bits.
- `hilbert_key(box, x, y, z)` is the index along the Hilbert curve of the point's cell (Skilling's
  transpose algorithm); `morton_key(box, x, y, z)` is the Z-order key in the same grid.
  Coordinates outside the box, and NaN, clamp to the box.

## Partitioning

`soma/parallel/partition.hpp` partitions without a graph:

```cpp
[[nodiscard]] std::vector<int> partition_by_curve(const Comm& comm, std::span<const Real> centroids,
                                                  std::span<const GO> gids);
```

Collective: the destination rank of each element given here, from its centroid (x, y, z
interleaved) and gid. Elements are ordered along a **Morton** curve of their centroids in the box
of every rank's centroids (ties broken by gid), then cut into $P$ runs whose sizes differ by at
most one. The order does not depend on which rank holds which element, so **the result depends
only on the elements and $P$**. It throws `std::invalid_argument` on every rank if any rank passes
a centroid count other than its gid count, or a centroid that is not finite.

## Migrating and rebalancing

`soma/parallel/migrate.hpp`:

```cpp
[[nodiscard]] HostMesh migrate(const Comm& comm, const HostMesh& mesh, std::span<const int> dest,
                               Ordering ordering = default_ordering);

[[nodiscard]] HostMesh rebalance(const Comm& comm, const HostMesh& mesh, Ordering ordering = default_ordering);
```

`migrate` moves each element of the mesh's dimension to rank `dest[e]` (one rank per element, by
lid) with its closure, and returns the new part; the mesh given is not changed. A copy of a
shared element follows its owner's destination. Each entity is placed by its owner:

- an element on the rank given for it;
- a lower-dimensional entity of a dimension given to the builder on any rank, on every rank
  receiving all its corners;
- a vertex on every rank receiving an entity it is a corner or node of.

What no rank receives stays with its owner.

What is kept:

- Entities keep their gids, topology, corners and nodes; vertices keep their coordinates.
- Dimensions given to the builder on any rank are given again; the others are derived again with
  the graphs held on any rank (derived gids do not depend on the partition), and
  `build_local_part` resolves the sharing.
- Each entity's set memberships and field values are those its owner held. Sets keep their gids,
  flags and links; an ordered set lists the entries of each owner in its order, owners by rank.
- Fields keep their layout, options and tag format. A field's domain on a dimension is full where
  it was full on every rank, else the entities that had a value. Mesh values are copied.

`migrate` is collective and throws on every rank: `std::invalid_argument` for a `dest` without
one rank of `comm` per element, or for sets, fields (beyond their values on entities) or mesh
values that differ between ranks; `std::logic_error` for an edited mesh (call `compress()`
first).

`rebalance` moves the owned elements to the ranks `partition_by_curve` gives them by their
centroids (their corners' average), with `migrate`: parts along the curve whose element counts
differ by at most one, whatever the partition given. It is collective and throws as `migrate`
and `partition_by_curve` do. From `examples/refine.cpp`, after a local refinement:

```cpp
soma::RefineResult r = soma::refine_marked(comm, *from, marks, {.conforming = conforming});
fine = soma::rebalance(comm, r.mesh);
```

## Determinism across rank counts

| What | Depends on |
|---|---|
| Owner of a shared entity | Which ranks hold it (the lowest). |
| Gids of derived edges and faces | The mesh only; not $P$ or the partition. |
| `partition_by_curve` destinations | The elements and $P$; not which rank held what. |
| `rebalance` result | The elements and $P$; not the partition given. |
| Lids within a part | The part's entities, the ordering and the global vertex box. |

Local ids are not global: compare entities across runs and rank counts by gid.
