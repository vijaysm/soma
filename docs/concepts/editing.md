# Editing meshes

`soma::MeshEdit` adds and removes vertices and elements of an existing host mesh in batches,
without rebuilding it. This page describes how an edit is recorded and applied, the *edited*
state a mesh is left in, how `compress()` returns it to the compact form the builder produces,
and which operations accept an edited mesh. The [mesh model](mesh-model.md) defines the terms
used here.

## Recording and applying an edit

`MeshEdit` (`soma/core/mesh_edit.hpp`) works on a `HostMesh`:

```cpp
class MeshEdit {
 public:
  explicit MeshEdit(HostMesh& mesh);

  LO add_vertices(std::span<const Real> xyz);
  LO add_vertices(std::span<const GO> gids, std::span<const Real> xyz);
  LO add_elements(Topology t, std::span<const LO> node_lids, int nodes_per_element = 0);
  LO add_elements(Topology t, std::span<const GO> gids, std::span<const LO> node_lids,
                  int nodes_per_element = 0);
  LO add_polygons(std::span<const Offset> offsets, std::span<const LO> vertex_lids);
  void remove(int d, std::span<const LO> lids);

  EditReport apply();
  void set_compress_threshold(double t);
};
```

Additions and removals are recorded and checked as they are given, and become visible together
when `apply()` is called. Unlike `MeshBuilder`, an edit names vertices by **local id**: the
`add_*` calls return the local id of the first entity they add, and elements may name vertices
added earlier in the same edit. `nodes_per_element` counts corners and high-order nodes, as in
`MeshBuilder`; 0 means `num_vertices(t)`.

The usage pattern, from the header and `tests/unit/test_mesh_edit.cpp`:

```cpp
#include "soma/core/mesh_edit.hpp"

soma::MeshEdit edit(mesh);
const soma::LO v = edit.add_vertices(std::vector<soma::Real>{-1, 0, 0, -1, 1, 0});  // lids v, v + 1
edit.add_elements(soma::Topology::quad, std::vector<soma::LO>{v, a, b, v + 1});      // a, b: existing lids
edit.remove(2, std::vector<soma::LO>{3});
const soma::EditReport r = edit.apply();
// r.added[d], r.removed[d]: counts per dimension; r.compressed: set if apply() compressed
```

After `apply()` the edit is empty and can be used again. A throw from `apply()` leaves the mesh
unchanged and the edit as it was.

### Rules

- **Local ids.** Existing entities keep their local ids. Added entities follow them, in the order
  they were added: `mesh.tail(d)` is how many there are at the end of dimension `d`. Removed
  entities stay in place, masked, until the mesh is compressed.
- **Gids.** Entities added without gids take the next gids after the largest gid of their
  dimension (removed entities included), in the order added. Given gids must be new to the
  dimension.
- **Removal.** `remove(d, lids)` names entities that exist before this edit. Removing a vertex
  that a live element uses, as a corner or a node, throws unless the element is removed in the
  same edit, as MOAB refuses it. Edges and faces can always be removed. Removing an entity twice
  throws.
- **Raising the dimension.** Elements of a dimension above the mesh's raise it (as on an empty
  mesh); the elements of the old dimension become explicit sides, which must then be sides of
  some element when sides are derived.
- **Input checks.** The `add_*` calls throw `std::invalid_argument` for coordinates without three
  values per vertex or with a NaN, a vertex lid out of range, a node count that is no
  configuration of the topology, polygons given to `add_elements` or polyhedra, and malformed
  polygon offsets; `remove` throws for a dimension the mesh does not have or a lid out of range.

### What follows an edit

- **Sets:** removed entities leave every set.
- **Fields:** a field over every entity of a dimension grows, the added entities taking the
  field's default (zero without one); other fields keep their members.
- **Adjacency graphs** the mesh holds are built again over the live entities; rows of removed
  entities are empty, and no row names a removed entity.
- **Sides:** adding or removing elements or sides makes the derived sides incomplete, so a held
  graph that touches them derives them again. Sides keep their gids (new ones take the next
  gids, and those no live element has any more go); their local ids are renumbered, and sets and
  fields follow them through their gids.
- **Epoch:** an edit does not change the [epoch](mesh-model.md), so fields stay current.

## The edited state

Until it is compressed, a mesh that took an edit is **edited**: `mesh.edited()` is true when some
dimension has added entities in its tail or removed entities in place.

| Member | Meaning |
| --- | --- |
| `edited()` | true while edits are not compressed |
| `count(d)` | every entity, removed ones included |
| `num_live(d)` | the entities not removed |
| `alive(d)` | one byte per entity, 0 for a removed one; empty when none is removed |
| `tail(d)` | entities added after the others were ordered |
| `MeshView::alive(d, e)` | the same test in kernels; always true on a mesh that was not edited |

`removed_fraction(mesh)` is the largest fraction, over the dimensions holding entities, of slots
holding removed entities (0 when not edited).

An edited mesh keeps working for code that skips removed entities: a kernel over
`[0, count(d))` must test `v.alive(d, e)`. Added entities follow all the others, in segments of
their own in the order added, so the space-filling-curve order no longer covers the tail. When
entities are added to a dimension, its [segment table](mesh-model.md) range ends are all set to
its count: every entity of that dimension then counts as interior. That is why most algorithms
ask for a compressed mesh, and why distributed parts need the merge described below.

### Operations on edited meshes

| Operation | On an edited mesh |
| --- | --- |
| `build_adjacencies` on a host mesh | builds over the live entities (above) |
| `build_adjacencies` on a device mesh | throws `std::invalid_argument` if removed entities are held; build on the host or compress first |
| mirrors and `clone()` | copy the edit state (alive masks and tails) |
| `write_vtk`, `write_h5m` | write the mesh as its compressed copy would be |
| `write_soma` | writes the compressed copy on one rank; throws `std::logic_error` on every rank on more |
| `permute_entities`, `add_ghosts`, `migrate` | throw `std::logic_error`: compress first |
| `laplacian_smooth`, `FemSpace`, `FvSpace`, `RemapOperator` | throw `std::invalid_argument` (on every rank, collectively) |

## Compressing

```cpp
Renumbering compress();   // HostMesh only
```

`compress()` drops the removed entities and orders the added ones with the others, as
`MeshBuilder` orders a mesh built from the live entities in the mesh's `ordering()`: vertices,
elements and explicit sides with their gids, nodes and coordinates. Fields, sets and held graphs
follow; fields stay current and the epoch advances (fetch field handles again). It does nothing
on a mesh that is not edited.

It returns a `Renumbering`, whose `new_lid[d]` maps each old local id of dimension `d` to its new
one (`invalid_lo` for a removed entity); `new_lid[d]` is empty for a dimension left as it was.
Use it to carry your own per-entity arrays across the compression.

```{figure} ../figures/img/mesh-edit.svg
:alt: Three copies of a 4 by 3 box of quads labelled with local ids and gids: before an edit; after apply, with cells 5 and 6 hatched as removed and a new quad numbered 12 added at the right; after compress, with the eleven live cells renumbered 0 to 10 and their gids unchanged
:width: 100%

One `MeshEdit` on a $4 \times 3$ box of quads removes cells 5 and 6 and adds a quad on two new
vertices. After `apply()` the removed cells keep their slots, masked, and the new quad follows
the others as lid 12 with the next gid, 13. `compress()` drops the removed cells and orders the
rest as `MeshBuilder` would, by gid: the new quad becomes lid 10. Gids never change.
```

### Automatic compression

Edited meshes pay for the extra slots and the mask on every pass. `apply()` therefore compresses
the mesh itself when `removed_fraction` reaches the **compress threshold**, and reports the
renumbering in `EditReport::compressed`. The threshold is the process-wide
`soma::compress_threshold()` (default `default_compress_threshold` = 0.2) unless
`edit.set_compress_threshold(t)` gives another for that edit; 0 never compresses, and values
outside $[0, 1]$ throw `std::invalid_argument`. The process-wide value is set with
`set_compress_threshold`, the `SOMA_COMPRESS_THRESHOLD` environment variable or the
`--soma-compress-threshold=` option (see [runtime and options](runtime.md)).

From `tests/unit/test_mesh_edit.cpp`, removing one of five quads (20 % of the quad slots) at the
default threshold compresses:

```cpp
soma::HostMesh m = soma::testing::box_mesh({5, 1, 1, soma::Topology::quad});
soma::MeshEdit edit(m);
edit.remove(2, std::vector<soma::LO>{1});
const soma::EditReport r = edit.apply();
// r.compressed.has_value(), !m.edited(), m.count(2) == 4
// r.compressed->new_lid[2][1] == soma::invalid_lo
```

To keep the edited state (to batch several edits before one compression, for example), set the
edit's threshold to 0 and call `compress()` when done.

## Edits on distributed meshes

`MeshEdit` changes one rank's part and knows nothing of other ranks: the gids it gives are
chosen by each rank alone, and adding entities resets the dimension's ownership ranges (above),
so the sharing of a distributed part no longer holds after an edit. The collective
`merge_mesh(comm, mesh, options)` (`soma/parallel/merge.hpp`) rebuilds it in place: live
vertices that match on any rank become one, and entities added since the mesh was built take the
gid of an entity they coincide with, or new gids above every gid of their dimension, the same on
every rank holding them. It identifies the added entities by the mesh's tail, which compression
loses, so either keep the edited state (`set_compress_threshold(0)`) until the merge, or pass
their local ids in `MergeOptions::added`. It throws on every rank if a rank holds ghosts: remove
them before editing. See [matching and merging](../parallel/matching-and-merging.md).
