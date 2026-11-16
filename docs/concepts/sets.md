# Entity sets

Entity sets group entities and other sets, as MOAB's meshsets do: material blocks, boundary
condition sets, partitions read from a file. This page describes how sets are stored, how to
make them with `SetBuilder`, how to query them on the host and in kernels, and how they follow
changes to the mesh.

## The model

A mesh's sets are `mesh.sets()`, an `soma::EntitySets<MemSpace>` (`soma/core/sets.hpp`). A mesh
has none until a `SetBuilder` installs them. Each set has:

- a **gid**: sets are ordered by gid, which is strictly increasing, and named on a rank by their
  index $0 \ldots$ `count() - 1`. Sets read from a file keep their file ids as gids;
- **members** per [slot](mesh-model.md): entities of dimensions 0 to 3 by local id, and other
  sets (slot `set_dim`) by index. Each row is sorted and unique, and held in the smallest of three
  forms (full, intervals or list), as field domains are;
- **flags**, MOAB's meshset flags;
- for ordered sets, the **contents in order**, repeats included;
- **parent** and **child** links to other sets, independent of membership.

| Flag | Value | Meaning |
| --- | --- | --- |
| `set_track_owner` | `0x1` | MOAB's tracking flag, kept |
| `set_unique` | `0x2` | an unordered set (the default of `SetSpec`) |
| `set_ordered` | `0x4` | contents kept in order, with repeats |

`set_flag_bits` (`0x7`) is the mask of the flags a set may have; MOAB's ranged-storage bit
(`0x8`) in a file is not kept.

Flags, ordered contents and links stay on the host; membership rows are also available on the
device.

## Making sets

`soma::SetBuilder` (`soma/core/set_builder.hpp`) collects sets on the host and installs them on a
mesh at once:

```cpp
struct SetItem { int slot = 0; GO id = 0; };   // an entity (slot 0-3, local id) or a set (set_dim, gid)

struct SetSpec {
  GO gid = 0;
  std::uint8_t flags = set_unique;
  std::vector<SetItem> contents;
  std::vector<GO> children;   // by gid
  std::vector<GO> parents;    // by gid
};

struct SetRun { int slot = 0; GO first = 0; GO count = 0; };

class SetBuilder {
 public:
  void add(SetSpec s);
  void add(SetSpec s, std::vector<SetRun> runs);  // unordered sets only
  template <class MemSpace> void install(Mesh<MemSpace>& mesh) &&;
};
```

Members that are entities are named by local id; members, children and parents that are sets are
named by gid. A `SetRun` adds `count` consecutive members from `first` without listing them,
which is how a set covering a large range is built cheaply.

From `tests/unit/test_entity_sets.cpp`, four sets on the two-quads-and-a-triangle fixture
(after its edges are derived):

```cpp
soma::HostMesh mesh = two_quads_and_a_triangle();
soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(2, 1));
soma::SetBuilder b;
b.add(SetSpec{300, soma::set_unique, {{set_dim, 200}}, {}, {}});
b.add(SetSpec{100, soma::set_unique | soma::set_track_owner,
              {{2, 2}, {2, 0}, {2, 2}, {0, 6}, {set_dim, 300}}, {400}, {400}});
b.add(SetSpec{200, soma::set_ordered, {{0, 3}, {2, 1}, {0, 3}, {1, 0}, {set_dim, 100}}, {}, {}});
b.add(SetSpec{400, soma::set_unique, {}, {}, {100}});
std::move(b).install(mesh);
```

Set 100 then has faces `{0, 2}` (the repeat is dropped), vertex 6 and set index 2 (gid 300);
set 200 keeps its five entries in order, repeats included. Containment may form cycles (here
100 contains 300, which contains 200, which contains 100).

`install` throws `std::invalid_argument` for a flag outside `set_flag_bits`, a repeated gid, an
entity that is not in the mesh, a member, child or parent that is not one of the sets, a run of
no members, or runs on an ordered set; and `std::logic_error` if the mesh has sets already.
Nothing changes when it throws.

## Querying sets

Host queries on `EntitySets`, with a set named by its index:

| Member | Returns |
| --- | --- |
| `count()` | number of sets |
| `gid()` | the View of set gids, strictly increasing |
| `index_of(gid)` | the index of a set, or `std::nullopt` |
| `flags(set)`, `ordered(set)` | flags |
| `members_of(set, slot)` | a `SetRow`: the sorted, unique members in one slot |
| `sequence(set)` | the contents of an ordered set as `SetEntry{slot, lid}`, in order with repeats; `std::logic_error` for an unordered set |
| `children(set)`, `parents(set)` | linked set indices |
| `members_recursive(set, dim)` | local ids of `dim` in the set and every set it contains, directly or not; sorted and unique, following cycles once |
| `has_members(slot)` | false for a dimension a mirror did not copy |

A `SetRow` reads one row in place: `size`, `r[k]` (the `k`-th member), `row(lid)` (its position,
or `invalid_lo`), `contains(lid)`, and an iterator, so
`std::vector<LO>(r.begin(), r.end())` lists the members. It is valid while the sets live.

In kernels, `mesh.sets().view()` returns a `SetsView<MemSpace>` with `row(slot, set)` and
`contains(slot, set, lid)`. From `tests/unit/test_entity_sets.cpp`:

```cpp
const auto dev = host.mirror<DevMem>();
const soma::SetsView<DevMem> view = dev.sets().view();
Kokkos::parallel_for(
    "set_contains", Kokkos::RangePolicy<DevMem::execution_space>(0, n),
    KOKKOS_LAMBDA(const soma::LO l) { out(l) = view.contains(slot, set, l) ? 1 : 0; });
```

`view()` on temporary sets is deleted, as for `Mesh::view()`.

## Fields on sets

Sets are field slot `set_dim`: `mesh.fields().add_tag<int>("GLOBAL_ID", set_dim, 1)` defines one
value per set, and `Domain::all(mesh, set_dim)` and `Domain::from_set(mesh, set, slot)` build
domains over sets or over a set's members (see [fields](fields.md)). `Domain::segments` does not
apply to sets.

## How sets follow the mesh

Set members are local ids, so they are renumbered with the entities:

- **Mirrors** copy the sets unless `TransferOptions::sets` is false, with the members of the
  kept dimensions; asking for the members of a dropped dimension throws `MissingSetMembers`.
- **Deriving sides** (`build_adjacencies`) renumbers explicit sides; set members keep their gids.
- **Edits** (`MeshEdit`): removed entities leave every set; sides derived again keep their
  members through their gids; `compress()` and `permute_entities` renumber the members. See
  [editing meshes](editing.md).
- **Ghosts:** `add_ghosts` puts set members on ghosts only when its options ask for it.
- **Migration:** sets keep their gids, flags and links; each entity's memberships are those its
  owner held, and an ordered set lists the entries of each owner in its order, owners by rank.

`EntitySets::renumbered(dim, new_lid, count, drop)` is the operation behind these: it maps the
members of one dimension, throwing `std::logic_error` for a member without a new id unless
`drop` is set, in which case the member leaves the set.
