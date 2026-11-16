# Matching points and merging meshes

Pieces of a mesh generated or read independently often repeat the vertices along their seams,
each copy with its own gid. SOMA makes coincident points one by their coordinates alone,
exactly or within a tolerance, on one rank or over all ranks. This page covers `match_points`
(`soma/parallel/point_match.hpp`), which numbers points, and the merge functions built on it
(`soma/parallel/merge.hpp`), which make coincident vertices and elements one entity before or
after a mesh is built. They correspond to MOAB's `MergeMesh` and `ParallelMergeMesh`.

## Matching points

```cpp
enum class PointNorm : std::uint8_t { linf, l1, l2 };

// tol == 0: exact matching.
struct PointTolerance {
  Real tol = 0;
  PointNorm norm = PointNorm::l2;
};

[[nodiscard]] PointIds match_points(const Comm& comm, std::span<const Real> xyz, PointTolerance tol = {});
```

`match_points` is collective: each rank passes the points it holds (x, y, z interleaved) and
receives an id per point, over the points of every rank.

### Tolerance semantics

| Mode | Two points share an id when |
|---|---|
| exact (`tol == 0`, the default) | Their coordinates are equal bit for bit, with `-0.0` equal to `+0.0`. A point 1 ulp away gets another id. |
| tolerance (`tol > 0`) | They are connected by a chain of pairs, each pair within `tol` (single linkage). |

A pair $a, b$ is within `tol` when, with $r_j = |a_j - b_j| / \mathrm{tol}$:

| `norm` | Test |
|---|---|
| `PointNorm::linf` | $\max_j r_j \le 1$ |
| `PointNorm::l1` | $\sum_j r_j \le 1$ |
| `PointNorm::l2` (default) | $\sum_j r_j^2 \le 1$ |

Dividing by `tol` first keeps the test free of underflow and overflow in the squares. `tol` is in
the units of the coordinates.

```{warning}
Matching within a tolerance is single linkage: if A is within `tol` of B and B of C, all three
share an id even when A and C are further apart than `tol`. A tolerance comparable to the
mesh's edge lengths can therefore merge whole regions. Choose `tol` well below the smallest
distance between distinct points.
```

Points with an infinite coordinate share an id only with exact copies of themselves. A NaN
coordinate is an error.

### Results

```cpp
struct PointIds {
  std::vector<GO> ids;    // per point given, in [0, count)
  GO count = 0;           // clusters over all ranks
  std::vector<Real> xyz;  // per point, its cluster's least member (x, y, z)
  Sharing sharing;        // per point, the other ranks holding a point of its cluster
};
```

- Ids are contiguous in $[0, \mathrm{count})$ and number the clusters by their least member in
  x-major order (coordinates compared through their bit patterns as ordered keys).
- **Ids do not depend on the number of ranks, on how the points are spread over them, or on their
  order.** They equal the single-rank `spatial_exact::assign_ids_exact` and `assign_ids_tol`
  (below) on all points together.
- `xyz` gives, per point, its cluster's representative, the least member; merged vertices are
  placed there.
- `sharing` lists, per point, the other ranks holding a point of the same cluster, found by the
  same sort: the sharing of the vertex the cluster becomes (see
  [distributed meshes](distributed-meshes.md) for `Sharing`).

### Errors

Thrown on every rank: `std::invalid_argument` if any rank gives a NaN coordinate or a coordinate
count that is not a multiple of 3, or if the ranks give different tolerances, or a negative, NaN
or infinite one; `std::runtime_error` when a tolerance is given and some rank's floating point is
not strict IEEE.

```{warning}
Tolerance matching needs strict IEEE-754 double arithmetic. Do not build with `-ffast-math`,
`-Ofast` or similar flags (for Intel's compilers, use `-fp-model=precise`). A one-time
self-check detects an unsafe build or CPU state and makes tolerance matching throw. Exact matching
uses integer operations on the bit patterns only and is unaffected by floating-point flags.
```

### How it works

Exact ids come from one global sort of 3-word keys (`transport::rank_keys`, see
[transport](transport.md)); each coordinate's key is its IEEE bits made an unsigned integer that
sorts as the coordinate does (`soma::point_key` in `soma/core/point_key.hpp`, which works in
device code). With a tolerance, each distinct point goes to the owner of its grid cell (cells of
a power-of-two width in $(\mathrm{tol}, 2\,\mathrm{tol}]$) and, as a halo, to the owners of the
neighbouring cells it comes within `tol` of; owners test the pairs of their cells, and the
connected components are joined across owners by FastSV rounds, whose number grows with the
logarithm of the longest chain, not its length. If `tol` is so large that most points fall into a
few cells, the pair tests degrade towards quadratic cost.

### Single-rank numbering

`soma/core/hash3d_exact.hpp` is the single-rank, header-only reference that `match_points`
agrees with, in namespace `spatial_exact`, for points of any dimension `D`:

```cpp
template <std::size_t D> IdResult<D> assign_ids_exact(const std::vector<Point<D>>& pts);
template <std::size_t D> IdResult<D> assign_ids_tol(const std::vector<Point<D>>& pts, double tol, Norm norm = Norm::L2);

template <std::size_t D>
struct IdResult {
  std::vector<Point<D>> unique_points;  // M representatives, key-ordered
  std::vector<int64_t> ids;             // N entries in [0, M)
};
```

`Point<D>` is `std::array<double, D>` and `Norm` is `L1`, `L2` or `Linf`. `fp_environment_ok()`
runs the floating-point self-check. Within SOMA, prefer `match_points` with `Comm::self()`.

## Merging a mesh before it is built

On each rank, fill a `MeshBuilder` with that rank's part, then merge:

```cpp
MeshBuilder b = ...;   // this rank's part
HostMesh part = build_merged_part(comm, std::move(b), {1e-10, PointNorm::l2});
```

```cpp
[[nodiscard]] HostMesh build_merged_part(const Comm& comm, MeshBuilder&& builder, PointTolerance tol = {},
                                         MergeGids gids = MergeGids::keep, Ordering ordering = default_ordering);
```

`build_merged_part` is `merge_vertices`, then `number_elements`, then `build_local_part` given the
sharing their sorts found, so no gid directory is consulted. The steps are also available
separately:

```cpp
Sharing merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol = {},
                       MergeGids gids = MergeGids::keep);

std::array<std::optional<Sharing>, 4> number_elements(const Comm& comm, MeshBuilder& builder,
                                                      MergeGids gids = MergeGids::keep);
```

- `merge_vertices` (collective): the vertices of every rank's builder that match become one
  vertex, at the cluster's least member, on every rank holding one of them; elements name it by
  its gid. It returns the sharing of the merged vertices in `vertex_gids()` order.
- `number_elements` (collective): elements of one dimension with the same set of corner gids, on
  any rank, become one element. It returns, for each dimension holding elements on some rank,
  their sharing in `element_gids(d)` order.

On one rank, pass `Comm::self()` and build with `MeshBuilder::build`, as in
`tests/unit/test_merge.cpp`:

```cpp
MeshBuilder b(2);
// ... two patches meeting along a seam ...
soma::merge_vertices(Comm::self(), b);
soma::number_elements(Comm::self(), b);
const soma::HostMesh m = std::move(b).build();
```

### Choosing gids

Coordinates decide which entities are one, and so which ranks share them; gids only name them.
`MergeGids` says how:

| Value | Gids |
|---|---|
| `keep` (default) | An entity keeps its gid when no other entity, on any rank, has it. Entities made one keep the least of their gids. An entity whose gid is taken by another (as when each part was numbered alone), or not trusted (in `merge_mesh`, added since the mesh was built), takes a new gid above every gid in use, numbered by its coordinates (or corners), the same on any number of ranks. Gids you keep maps of (degrees of freedom, say) stay valid. |
| `renumber` | Every gid is new: 1 plus the entity's rank among all entities by coordinates (vertices) or corner gids (elements). The numbering depends on the geometry alone. |

### Errors

`merge_vertices` throws as `match_points` does; `std::overflow_error` on every rank when new gids
would not fit a `GO`; and `std::invalid_argument` on the ranks whose merge
`MeshBuilder::rename_vertices` refuses (`std::runtime_error` on the others). Every rank throws, and
no builder changes, if one does. `number_elements` and `build_merged_part` throw as these do.

## Merging a built mesh

```cpp
struct MergeOptions {
  PointTolerance tol{};
  MergeGids gids = MergeGids::keep;
  std::optional<std::array<std::vector<LO>, 4>> added;
};

Renumbering merge_mesh(const Comm& comm, HostMesh& mesh, const MergeOptions& options = {});
```

`merge_mesh` (collective) merges a built mesh in place: its live vertices that match, on any
rank, become one, and the mesh is built again in its ordering, with the graphs it held (on any
rank) and its explicit sides. A typical use is after a local refinement, when ranks have created
vertices along their shared boundary independently. From `tests/unit/test_merge.cpp`:

```cpp
soma::MergeOptions how{.tol = {tol, PointNorm::l2}};
(void)soma::merge_mesh(c, m, how);
```

- **Gids with `keep`.** Entities the mesh was built with keep their gids. Entities added since
  (`MergeOptions::added`: gids each rank chose alone with `MeshEdit`) take the gid of an entity
  they coincide with, or new gids above every gid of their dimension, the same on every rank
  holding them.
- **`added`.** The lids, per dimension, of the entities added since the mesh was built. Unset, the
  mesh's tail of added entities is used; `MeshEdit::apply` loses that tail when it compresses the
  mesh, so either pass `added` or keep the mesh uncompressed (`MeshEdit::set_compress_threshold(0)`).
- **Values.** Sets, fields and mesh values follow each entity: a vertex by its merged gid, any
  other entity by its merged corners. Where entities on a rank become one, the one of least lid
  keeps its values and memberships, so a merged vertex may hold different values on the ranks
  sharing it until an owner-to-copies [halo update](ghosts-and-halos.md). Derived sides that no
  longer exist lose their values.
- **Result.** A `Renumbering`: old lid to new lid per dimension, `invalid_lo` for an entity
  merged into another or removed.

Errors, on every rank: `std::invalid_argument` if a rank holds ghosts (remove them first), and on
a distributed mesh if a kept gid names different points on two ranks (an added entity not named
in `added`), naming the gid; otherwise as `merge_vertices`. A throw before the mesh is replaced
leaves it unchanged.
