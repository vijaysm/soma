# Building meshes

This page shows how to make a `soma::HostMesh` from your own arrays with `soma::MeshBuilder`,
how to build the local part of a distributed mesh, how to make structured box meshes for tests,
and what the builder checks. To read a mesh from a file instead, see the
[I/O overview](../io/overview.md). The terms used here (gid, lid, segment, ownership) are
defined in [the mesh model](mesh-model.md).

## `MeshBuilder`

`soma::MeshBuilder` (`soma/core/mesh_builder.hpp`) collects vertices and elements named by
gid, then orders every dimension and translates the connectivity to local ids in one step.

```cpp
class MeshBuilder {
 public:
  explicit MeshBuilder(int dim);

  void add_vertices(std::span<const GO> gids, std::span<const Real> xyz);
  void add_elements(Topology t, std::span<const GO> gids, std::span<const GO> vertex_gids);
  void add_elements(Topology t, std::span<const GO> gids, std::span<const GO> node_gids,
                    int nodes_per_element);
  void add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                    std::span<const GO> vertex_gids);

  HostMesh build(Ordering ordering = default_ordering) &&;
  // ... accessors, renaming and distributed-mesh setup, below
};
```

- `MeshBuilder(dim)` fixes the mesh dimension, 1, 2 or 3 (`std::invalid_argument` otherwise).
- `add_vertices(gids, xyz)` adds one vertex per gid; `xyz` holds three coordinates per vertex,
  interleaved (`x0, y0, z0, x1, ...`). Give `z = 0` for planar meshes.
- `add_elements(t, gids, vertex_gids)` adds elements of a fixed topology `t`, each named by its
  gid, with `num_vertices(t)` vertex gids per element in MOAB's vertex order for `t`.
- `add_polygons(gids, offsets, vertex_gids)` adds polygons as CSR: polygon `i` has the vertex
  gids `vertex_gids[offsets[i] .. offsets[i + 1])`.
- `build(ordering)` consumes the builder (call it on `std::move(builder)`) and returns the
  mesh. The [ordering](mesh-model.md) defaults to `Ordering::type_major_sfc`; `Ordering::file`
  orders by gid.

Each `add_*` call may be repeated; blocks accumulate. Elements of the mesh dimension are the
elements; elements of a lower dimension (edges in a 2D or 3D mesh, faces in a 3D mesh) are kept
as **explicit sides**. When sides are later derived, explicit ones are adopted and keep their
gids and vertex order (see [topology](topology.md)); `mesh.input(d)` tells which dimensions the
builder was given.

The mesh is allocated with Kokkos, so a `soma::ScopeGuard` (or your own Kokkos initialisation)
must exist first; see [runtime and options](runtime.md).

### Example

Two quads and a triangle, from `tests/unit/mesh_fixture.hpp`. The vertex gids are deliberately
unsorted and given in no particular order:

```cpp
#include <vector>

#include "soma/core/mesh_builder.hpp"

soma::HostMesh two_quads_and_a_triangle() {
  using soma::GO;
  using soma::Real;
  soma::MeshBuilder b(2);
  const std::vector<GO> vg{50, 10, 70, 30, 20, 60, 40};
  const std::vector<Real> xyz{1, 1, 0, 0, 0, 0, 3, 0.5, 0, 2, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0};
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, std::vector<GO>{7, 3},
                 std::vector<GO>{10, 20, 50, 40, 20, 30, 60, 50});
  b.add_elements(soma::Topology::tri, std::vector<GO>{5}, std::vector<GO>{30, 70, 60});
  return std::move(b).build(soma::Ordering::file);
}
```

The result has `dim() == 2`, `count(0) == 7`, `count(2) == 3` and `count(1) == 0` (no edges until
they are asked for). Under `Ordering::file` the vertices are ordered by gid (10, 20, ..., 70) and
the faces by topology, then gid: the triangle (gid 5) first, then the quads 3 and 7, in two
segments. The connectivity holds local ids.

### High-order elements

The four-argument `add_elements` takes `nodes_per_element` node gids per element: the corners,
then the other nodes in MOAB's order (`soma/topology/nodes.hpp`, described in
[the mesh model](mesh-model.md)). The count must be one of the configurations of `t`
(`mid_node_bits(t, count) >= 0`); `num_vertices(t)` adds linear elements. From
`tests/unit/test_high_order.cpp`, a 9-node quad beside a linear quad:

```cpp
b.add_elements(soma::Topology::quad, std::vector<GO>{1},
               std::vector<GO>{10, 20, 50, 40, 15, 35, 45, 25, 55}, 9);
b.add_elements(soma::Topology::quad, std::vector<GO>{2}, std::vector<GO>{20, 30, 60, 50}, 4);
```

Neighbouring elements need not agree on their nodes: here the shared edge has a mid-node in one
element and not in the other.

### What `build` does

`build` sorts the entities of every dimension by the key

1. ownership kind (interior on one rank; see below for distributed builds),
2. topology, arity and number of high-order nodes,
3. the neighbour-set group given by `set_ownership` (distributed builds only),
4. the ordering: gid for `Ordering::file`; Hilbert key, then gid, for `Ordering::type_major_sfc`,

so that each ownership kind is contiguous and each (topology, arity) run is one segment.
Vertices that are only high-order nodes come after the corner vertices of their kind. The
Hilbert keys of all dimensions use one bounding box, that of the vertices, unless
`set_key_box(lo, hi)` gives another.

The result starts at epoch 1, holds no adjacency graphs beyond the connectivity, no sets, no
fields and no mesh values.

## Inspecting and renaming builder input

The builder exposes what it has collected, which merging and readers use:

| Member | Returns |
| --- | --- |
| `vertex_gids()` | the vertex gids added so far, in the order added |
| `vertex_coordinates()` | their coordinates, interleaved |
| `element_gids(dim)` | gids of the elements of `dim` added so far, in the order added (empty outside $[1, 3]$) |
| `element_corners(dim, offsets, corners)` | their corner gids, as CSR, in the same order |

`rename_vertices(gids, xyz)` and `rename_elements(dim, gids)` give the collected entities new
gids; entities given one gid become one. They are the step that turns independently numbered
pieces into one mesh, and are normally called for you by the merge functions of
[matching and merging](../parallel/matching-and-merging.md). Both throw `std::invalid_argument`
and change nothing when the input is inconsistent: a vertex gid given twice, one gid for
vertices at different points, an element that would name a vertex twice, elements given one
gid that differ, or a call after `set_ownership` named entities of that dimension.

## Distributed meshes

On several ranks, each rank adds the elements it holds with every vertex they use, and the
collective `build_local_part` (`soma/parallel/distributed.hpp`) builds the local part:

```cpp
[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        Ordering ordering = default_ordering);
```

It finds which entities are shared between ranks (the lowest rank holding an entity owns it),
sets their ownership with `MeshBuilder::set_ownership`, sets a common key box with
`set_key_box` so that curve keys are comparable across ranks, and builds. It throws on every
rank if any rank's input is invalid. Overloads take ghost gids or a sharing already known. See
[distributed meshes](../parallel/distributed-meshes.md).

`set_ownership(dim, gids, kinds, groups)` and `set_key_box(lo, hi)` can also be called
directly. `set_ownership` names the ownership kind and a neighbour-set group of entities of
`dim` by gid (unnamed entities are interior); it throws `std::invalid_argument` for a dimension
outside $[0, \text{dim}]$, spans of different lengths, or (at build) a gid named twice or one
that names no entity. `set_key_box` throws unless the corners are finite with `lo <= hi`.

## Box meshes

`soma/testing/box_mesh.hpp` builds structured boxes with closed-form counts, coordinates and
orientation, for tests and examples that must not depend on a file or a mesher. It is part of
the library, in namespace `soma::testing`.

```cpp
struct BoxSpec {
  int nx = 1, ny = 1, nz = 1;
  Topology element = Topology::hex;
  Real lx = 1, ly = 1, lz = 1;
  Ordering ordering = Ordering::file;
};

BoxCounts box_counts(const BoxSpec& spec);  // vertices, elements, connectivity; allocates nothing
HostMesh box_mesh(const BoxSpec& spec);
```

```cpp
#include "soma/testing/box_mesh.hpp"

const soma::HostMesh hexes = soma::testing::box_mesh({4, 4, 4, soma::Topology::hex});
const soma::HostMesh quads = soma::testing::box_mesh({10, 10, 1, soma::Topology::quad});
```

The numbering, from the header:

| Entity | Gid and layout |
| --- | --- |
| vertex $(i, j, k)$ | gid $1 + i + (n_x + 1)(j + (n_y + 1)k)$, at $(l_x i / n_x,\ l_y j / n_y,\ l_z k / n_z)$; the far faces sit exactly at $l_x, l_y, l_z$ |
| hex | gid $1 + i + n_x(j + n_y k)$; corners $(i,j,k), (i{+}1,j,k), (i{+}1,j{+}1,k), (i,j{+}1,k)$, then the same at $k + 1$ |
| tet | hex $h$ (0-based) gives tets $6h + 1 \ldots 6h + 6$ (Kuhn's split); every tet has positive volume and neighbouring hexes split alike, so faces match |
| quad | gid $1 + i + n_x j$; corners $(i,j), (i{+}1,j), (i{+}1,j{+}1), (i,j{+}1)$ |
| tri | quad $q$ (0-based) gives triangles $2q + 1 = (v_0, v_1, v_2)$ and $2q + 2 = (v_0, v_2, v_3)$ |

Supported elements are `hex`, `tet`, `quad` and `tri`. Quads and triangles lie in the plane
$z = 0$, make a 2D mesh, and need `nz == 1`. The default ordering of a box is `Ordering::file`,
so local ids follow the gids; set `spec.ordering` for another.

`box_counts` and `box_mesh` throw `std::invalid_argument` for an unsupported element, a cell
count below 1, `nz != 1` for a planar element, or a length that is not positive and finite (or
whose product with its cell count is not finite; `lz` is only read for hexes and tets), and
`std::overflow_error` when a count does not fit `LO` (vertices, elements) or `Offset`
(connectivity entries). The counts are computed in 64-bit arithmetic before anything is
allocated.

## Validation and errors

The builder checks identities and the shape of its input, and throws `std::invalid_argument`,
naming the offending gid where there is one, unless noted otherwise. Checks made when a block is added (a rejected block
adds nothing, and the builder stays usable):

| Call | Rejected input |
| --- | --- |
| `MeshBuilder(dim)` | `dim` outside 1 to 3 |
| `add_vertices` | `xyz` without three values per gid; a NaN or infinite coordinate |
| `add_elements` | `Topology::polygon` (use `add_polygons`); `Topology::polyhedron` (not supported); a topology of dimension outside $[1, \text{dim}]$; a node count that is no configuration of the topology; a connectivity size that does not match; a connectivity too large for `Offset` (`std::length_error`) |
| `add_polygons` | a mesh dimension below 2; `offsets` without one entry more than `gids`, not starting at 0, decreasing, or not ending at `vertex_gids.size()`; a polygon with fewer than 3 or more than 65535 vertices |

Checks made by `build`:

- a vertex gid added twice;
- an element (or high-order node) naming a vertex gid that was not added;
- two elements of one dimension with one gid, whatever their topologies (the same gid in two
  dimensions is allowed);
- `Ordering::tiled`, which is not implemented;
- the `set_ownership` checks above.

`build` checks identities, not geometry: it does not test element orientation, volume or
validity. Algorithms that need valid cells check them themselves (for example, the finite
element and finite volume spaces reject inverted or degenerate cells).
