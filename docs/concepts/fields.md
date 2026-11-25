# Fields

A field is an array of values attached to some entities of a mesh: a temperature per vertex, a
material id per element, nodal values of a high-order discretisation, or a MOAB tag read from a
file. This page describes the field registry that every mesh owns, how a field's entities
(its *domain*) and value points (its *layout*) are described, how to define, read and write
fields on the host and in kernels, and what happens to fields when the mesh is copied, edited or
redistributed. It also covers mesh values, the values of a mesh as a whole.

## The registry

Each `Mesh` owns a `soma::FieldRegistry<MemSpace>` (`soma/fields/registry.hpp`), reached through
`mesh.fields()`. The registry is host-side and type-erased; kernels never see it, only the
concretely typed Kokkos Views of its fields.

A field is defined by four things: a **name**, a **value type**, a **component count**
(`ncomp`) and a **space**. A second definition with all four equal throws `FieldClash`. A field
is identified by a `FieldId`, an index that is never reused: a removed field leaves a gap.

The quickest way to make a field is `add_tag`, which defines a dense, MOAB-style tag with one
value per entity of a dimension. From `tests/unit/test_mesh_fields.cpp`:

```cpp
auto mesh = soma_test::two_quads_and_a_triangle();
auto t = mesh.fields().add_tag<double>("temperature", 0, 1);       // 7 rows, one per vertex
auto q = mesh.fields().add_tag<std::int32_t>("material", 2, 1);    // 3 rows, one per face
t.view()(3, 0) = 4.5;                                               // vertex lid 3, component 0
```

`add_tag<T>(name, dim, ncomp, options)` is shorthand for `add` over a full per-entity space,
described below.

### Value types

`soma::DataType` (`soma/fields/data_type.hpp`) lists the value types a field can hold:

| `DataType` | C++ type |
| --- | --- |
| `i8`, `u8` | `std::int8_t`, `std::uint8_t` |
| `i32`, `u32` | `std::int32_t`, `std::uint32_t` |
| `i64`, `u64` | `std::int64_t`, `std::uint64_t` |
| `f32`, `f64` | `float`, `double` |

`data_type_of<T>()` maps a C++ type to its `DataType` (any other type fails to compile),
`name(t)` gives `"f64"` and so on, and `dispatch_data_type(t, f)` calls
`f.template operator()<T>()` for the matching type, for code that handles fields of any type.

### Options

`FieldOptions<T>` (`soma/fields/field.hpp`) carries what a field may have beyond its values:

| Member | Default | Meaning |
| --- | --- | --- |
| `default_value` | none | `ncomp` values: the value of an entity that was never set, and the initial fill. Without one, fields start zeroed and `has_default()` is false |
| `global_value` | none | one value per mesh, as MOAB tags can carry |
| `handle` | `false` | the values are [handles](mesh-model.md) of entities or sets (`make_handle`; 0 for none); needs `T = GO` |

From `tests/unit/test_mesh_edit.cpp`:

```cpp
auto all = m.fields().add<double>(
    "a", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 0), soma::Layout::per_entity()),
    1, {.default_value = std::vector<double>{-5}});
```

## Domains: which entities

A `soma::Domain<MemSpace>` (`soma/fields/domain.hpp`) is an immutable snapshot of a set of local
ids, one block per dimension (or per [slot](mesh-model.md): `set_dim` for the entity sets). Each
block is held in the cheapest of three forms, chosen automatically:

| Form | Members | Lookup |
| --- | --- | --- |
| full | every entity of the dimension | identity, no storage |
| intervals | a few contiguous runs | binary search on the run starts |
| list | sorted local ids | binary search |

Factories that take a mesh are in `soma/fields/domain_mesh.hpp` (include it to use them):

| Factory | Members |
| --- | --- |
| `Domain::all(mesh, dim)` | every entity of `dim` (or every set, for `set_dim`) |
| `Domain::segments(mesh, dim, filter)` | the segments a `SegmentFilter` selects, by topology, arity and ownership |
| `Domain::from_list(mesh, dim, sorted_lids)` | strictly increasing local ids |
| `Domain::from_set(mesh, set, slot)` | the members of entity set `set` in one slot |

`SegmentFilter` has optional `topology` and `arity` and an `ownership` of `Own::any`, `owned`,
`interior`, `ghost`, `shared` (both kinds) or `shared_copy`, so for example a field on the owned
hexes of a mesh is one call. Mesh-free overloads (`Domain::all(dim, count, epoch, gids)`,
`from_intervals`, `from_list`) build domains from raw counts, and `Domain::combine(x, y)` joins
two domains of distinct dimensions built at the same epoch.

Each domain records the mesh [epoch](mesh-model.md) it was built at, and a fingerprint of its
members' gids. Two domains compare equal when they have the same epoch, blocks and members.

## Layouts and spaces: which value points

A `soma::Layout` (`soma/fields/layout.hpp`) says how many value points each entity carries:

| Layout | Value points |
| --- | --- |
| `Layout::per_entity()` | one per entity (the default) |
| `Layout::element_nodes(family, order, ordering, serendipity)` | the nodes of a nodal element of that order, owned by the element |
| `Layout::shared_nodes(...)`, `Layout::quadrature(rule)` | representable, but cannot back a space yet (`std::invalid_argument`) |

`NodeFamily` is `lagrange_equispaced`, `gll` or `gauss`; `NodeOrdering` is `moab_native`,
`tensor_lexicographic` or `hierarchical`; `max_node_order` is 32. `num_element_nodes(t, layout,
arity)` gives the node count of a topology, or 0 where no node table exists: for the families
other than `gauss`, $(p+1)(p+2)/2$ on triangles, $(p+1)^2$ on quads, $(p+1)(p+2)(p+3)/6$ on
tets, $(p+1)^2(p+2)/2$ on prisms and $(p+1)^3$ on hexes; pyramids at order 1 (and order 2,
equispaced); polygons at order 1; serendipity only at order 2 (Quad8, Hex20, Prism15,
Pyramid13); never knives or polyhedra.

A `soma::Space<MemSpace>` (`soma/fields/space.hpp`) is a domain times a layout. It fixes the row
numbering of every field defined on it: per-entity spaces give one row per member, in domain
order; element-node spaces give `num_element_nodes` consecutive rows per member. Build one with
`make_space(mesh, domain, layout)`, which supplies the element topologies that element-node
layouts need. `space.entity_rows(dim, lid)` returns the rows of an entity (invalid for a
non-member), and `space.view()` is the device-copyable form.

```{figure} ../figures/img/cell-and-vertex-fields.png
:alt: The same plane mesh twice, coloured by u = x + y; on the left each cell has one flat colour, on the right the colour varies smoothly inside each cell
:width: 90%

The function $u = x + y$ on `data/hybrid_3quad_1tri.h5m` as two per-entity fields: on the left a
field on the cells (dimension 2), one value per cell, here $u$ at the cell's corners' mean; on
the right a field on the vertices (dimension 0), drawn interpolated across each cell.
```

`soma/fields/node_tables.hpp` gives the reference coordinates of the nodes: `gll_points_1d(p)`,
`make_node_table(key)`, `make_node_permutation(from, to)` between orderings, and the
process-wide cache `NodeTables<MemSpace>::get(key)` and `permutation(from, to)`. Simplex GLL
nodes use Warburton's warp-and-blend construction. The cache's Views are unmanaged and valid
until `NodeTables::clear()` or `Kokkos::finalize`, whichever comes first.

## Defining, reading and writing fields

```cpp
template <class T>
Field<T, MemSpace> add(std::string name, const Space<MemSpace>& space, int ncomp,
                       FieldOptions<T> options = {});
template <class T>
Field<T, MemSpace> add_tag(std::string name, int dim, int ncomp, FieldOptions<T> options = {});
template <class T>
VarField<T, MemSpace> add_var(std::string name, const Space<MemSpace>& space,
                              std::span<const Offset> offsets, std::span<const T> values,
                              FieldOptions<T> options = {});
```

`add` throws `std::invalid_argument` for `ncomp < 1` or a default or global value without
`ncomp` values, `FieldClash` for a repeated definition, `FieldOverlap` (below), and `StaleField`
for a space built at another epoch.

A `Field<T, MemSpace>` is a typed handle that aliases the registry's storage:

| Member | Returns |
| --- | --- |
| `view()` | the `View2<T, MemSpace>` of `rows() x ncomp()` values (`LayoutLeft`); capture this in kernels, not the `Field` |
| `rows()`, `ncomp()` | its extents |
| `space()`, `domain()` | where the rows are |
| `id()`, `revision()` | identity and storage version |
| `default_value()`, `has_default()`, `global_value()`, `has_global()` | the options |
| `value_or_default(dim, lid, c)` | host read at the entity's first value point, or the default (zero without one) |

Fields are fetched again by name with `get<T>(name)`, `get<T>(name, dim)` (the one with members
on `dim`), `get<T>(name, space)` or `get<T>(id)`. `get` throws `std::out_of_range` for an unknown
name, `std::invalid_argument` when several fields share the name (use an overload that picks one)
or when `T` is not the stored type, and `StaleField` for a field of an older epoch. `contains`,
`info(id)` (a `FieldInfo` with name, type, `ncomp`, layout, rows, dimensions and revision) and
`list()` describe what is there; `remove(id)` frees a field.

### Variable-length fields

`add_var` defines a `VarField`: a CSR of values over a per-entity space, where member `r` holds
`values[offsets[r] .. offsets[r + 1])`. `offsets` has one entry more than the space has members,
starts at 0, never decreases and ends at `values.size()`. The values can be written; the lengths
are fixed when the field is added. Fetch it with `get_var`; `get` on a variable-length field (or
`get_var` on a fixed one) throws `std::invalid_argument`.

### Growing a field: pending writes

A field over a subset of entities can grow. `set_pending<T>(id, lid, values)` writes one
entity's values on the host: a member is written straight into the data, while a non-member is
buffered and joins the domain at the next `commit(id)` (or `commit_all()`). From
`tests/unit/test_field_commit.cpp`:

```cpp
auto f = reg.add<double>("q", subset_of_3_faces, 2);
reg.set_pending<double>(f.id(), 12, std::vector<double>{12.0, 12.5});  // not a member: buffered
reg.commit(f.id());
const auto g = reg.get<double>(f.id());   // 4 rows now, sorted by entity
```

A commit replaces the field's storage: it keeps its id and epoch, and its revision advances.
Handles taken before then refer to the old storage; they stay readable, but writes through
them are lost. `is_current(field)` tells, and `get` fetches the current handle.
`pending_count(id)` and `discard_pending(id)` manage the buffer. Mirrors commit the pending
writes of the fields they copy first.

## Families and lookups by entity

Fields that share a name, value type, component count, layout and handle mark form a
**family**; the domain is not part of the key. A family lets one name cover disjoint parts of a
mesh with separate fields, as a MOAB tag can be dense on some entities and absent on others.
Members of a family must be disjoint: adding an overlapping one throws `FieldOverlap`, and so
does a pending write to an entity another member holds.

`family<T>(name, ncomp, layout, handle)` returns the `FamilyId` (`std::out_of_range` if none).
For each dimension the family builds, on first use, an index from entity to (member, row)
(`soma/fields/resolver.hpp`), chosen by how the members cover the dimension: none, one full
member, one subset member, several members of intervals, or a byte-per-entity owner map for
general subsets (at most 255 members per dimension).

- `lookup(fam, dim, lid)` returns a `FamilyHit{slot, row}`, and `value<T>(fam, dim, lid, comp)`
  the value, or the family default when no member has the entity. These host conveniences fetch
  the field each call; use the Views for bulk access.
- `family_handle(fam)` returns a `FamilyHandle` that owns the current indexes; its `view()` is a
  device-copyable `FamilyView` with `lookup(dim, lid)`. Keep the handle alive while any kernel
  uses the view (`view()` on a temporary handle does not compile). A handle never sees later
  changes to the family.

```{warning}
The registry is not thread safe for mutation, and lazily built indexes make even some const
calls write. To read a registry from several host threads, take every `family_handle` first,
make no changes meanwhile, and let the threads use the handles' views and the field Views.
```

## Mesh values

Values of the mesh as a whole, rather than of its entities (MOAB tags of class 3, or tags
declared with values on no entity), are kept in `mesh.values()`, a `soma::MeshValues`
(`soma/core/mesh_values.hpp`) keyed by name:

| Member | Meaning |
| --- | --- |
| `set(name, MeshValue)` | adds or replaces |
| `contains(name)`, `at(name)`, `erase(name)` | lookup (`at` throws `std::out_of_range`) |
| `names()`, `size()`, `empty()` | in name order |

A `MeshValue` holds a `DataType`, `ncomp` (0 for variable length), a handle mark, a `TagFormat`,
and an optional default and global value as bytes; `default_as<T>()` and `global_as<T>()` read
them typed (`std::invalid_argument` for the wrong type). Mesh values are host-only and every
mirror copies them. The surface a mesh is on is one of them
([mesh geometry](../io/soma-format.md#mesh-geometry)).

`TagFormat` (`soma/fields/tag_format.hpp`) records how a legacy `.h5m` file stores a tag (MOAB
storage class, plain, bit or opaque values), so that `write_h5m` restores what `read_h5m` read;
for families it is read and set with `tag_format(fam)` and `set_tag_format(fam, f)`.

## How fields follow the mesh

A field's rows are tied to local ids, so every operation that renumbers entities either carries
the fields along or marks them stale:

| Operation | Fields |
| --- | --- |
| `mesh.mirror<Dst>(opts)` | copies the fields named in `opts.fields` (none by default, `{"*"}` for all), committing their pending writes first; a field on a dropped dimension is skipped; a selected stale field throws `StaleField`. `mirror_view` in the same space shares field data |
| `mesh.clone()` | deep copy of every field |
| [`build_adjacencies`](topology.md) | deriving a dimension a live field uses, or one that held explicit entities, advances the epoch: every field becomes stale |
| `bump_epoch()` | every field becomes stale |
| [`MeshEdit::apply`](editing.md) | fields over every entity of a dimension grow, added entities at the default; other fields keep their members; sides derived again keep their values through their gids; the epoch does not change |
| [`compress()`](editing.md) and `permute_entities` | values follow their entities; the epoch advances and the fields move to it, so they stay current (handles must be fetched again) |
| [`add_ghosts`](../parallel/ghosts-and-halos.md) | ghosts and shared copies hold their owners' values; fetch field handles again |
| [`migrate`, `rebalance`](../parallel/distributed-meshes.md) | each entity's values are those its owner held; a field's domain is full where it was full on every rank |

After the epoch advances, stale fields can still be removed, but not fetched; define them again
on the new topology. To avoid that, build the adjacencies you need before defining fields on
derived sides.
