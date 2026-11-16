# Ghosts and halo exchange

Ghosts are read-only copies of other ranks' elements, with their closure, added around a local
part so that stencils and assembly near the part boundary see their whole neighbourhood. A halo
exchange moves values between an entity's owner and its copies and ghosts. This page covers
adding and removing ghost layers (`Mesh::add_ghosts`, options in `soma/parallel/ghosts.hpp`) and
exchanging values (`soma::HaloExchange`, `soma/parallel/halo_exchange.hpp`). It assumes a mesh
built as in [distributed meshes](distributed-meshes.md).

## Ghost layers

### Options

```cpp
struct GhostOptions {
  int layers = 1;      // 0: no ghosts
  int bridge_dim = 0;  // below the mesh's dimension
  bool sets = false;   // ghosts join the sets their owners' entities are in
};
```

| Option | Default | Meaning |
|---|---|---|
| `layers` | 1 | Number of ghost layers; 0 means none. |
| `bridge_dim` | 0 | Dimension of the sides through which elements are neighbours: 0 joins elements sharing a vertex, `dim - 1` elements sharing a facet. Must be in `[0, dim)`. |
| `sets` | `false` | When true, ghosts join the sets their owners' entities are in. |

**The layer rule.** Two elements of the mesh's dimension are joined when they have a common
side of dimension `bridge_dim`, identified by its corners whether or not the mesh holds that
side. Layer 0 is the elements a rank holds other than as ghosts; layer $i$ is every element held
elsewhere joined to layer $i - 1$. The rule does not depend on the partition: however thin the
parts in between, a rank receives exactly the elements within `layers` steps of its own.

### Adding and removing ghosts

`add_ghosts` and `remove_ghosts` are members of `Mesh` (`soma/core/mesh.hpp`), for host meshes
only:

```cpp
Mesh& add_ghosts(const Comm& comm, const GhostOptions& options);
Mesh& remove_ghosts(const Comm& comm);   // add_ghosts with no layers
int ghost_layers() const noexcept;
int ghost_bridge_dim() const noexcept;
```

From `examples/laplacian_smooth.cpp`, one layer of ghosts through vertices around a smoothing
step, removed before writing:

```cpp
soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, options);
mesh.remove_ghosts(comm);
soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{}});
```

Behaviour:

- **Collective.** Every rank must call `add_ghosts` (and `remove_ghosts`) with the same options.
- **In place, ids kept.** Every entity held other than as a ghost keeps its lid and gid. Ghosts,
  with their gids, take the lids after the shared copies of each dimension (the
  `[shared_end, count)` range of the segment table), ordered by topology, owner, then the mesh's
  ordering. They replace any ghosts held before, so calling `add_ghosts` again with other options
  gives the new layers, not more.
- **Closure.** Ghost elements come with their vertices and with any edges and faces the mesh
  holds (explicit or derived); entities a rank holds only through ghost elements are ghosts.
- **Values.** Ghosts and shared copies hold their owners' field values when the call returns;
  set memberships reach ghosts only with `options.sets`.
- **Handles.** Field handles must be fetched again after the call.
- `ghost_layers()` and `ghost_bridge_dim()` return what `add_ghosts` last placed (0 and 0 for a
  mesh without ghosts); copies and mirrors of the mesh keep them.

Errors, thrown on every rank: `std::invalid_argument` for options that differ between ranks, a
negative layer count, a `bridge_dim` outside `[0, dim)`, or a mesh without sharing at $P > 1$;
`std::logic_error` for an edited mesh (call `compress()` first); and otherwise as `migrate`
does.

Several algorithms require a ghost layer on more than one rank, for example `FvSpace`
([finite volumes](../algorithms/finite-volumes.md)) and `laplacian_smooth`
([smoothing](../algorithms/smoothing.md)); their pages say which layers they need.

```{note}
`merge_mesh` refuses a mesh with ghosts (see [matching and merging](matching-and-merging.md)):
remove them first.
```

## Halo exchange

`HaloExchange<MemSpace>` exchanges and reduces values over the shared entities and ghosts of one
dimension of a distributed mesh. It is built once from the mesh's `soma:sharing` and
`soma:owner` fields and reused:

| Operation | Effect |
|---|---|
| `update(comm, values)` | Each owner's values overwrite its shared copies and ghosts. |
| `reduce(comm, values, op)` | Every holder's values are combined with `op` and the result written on every holder. Ghosts take no part: their values are neither read nor written. |

`values` is a `View2<T, MemSpace>` (`Kokkos::View<T**, Kokkos::LayoutLeft, MemSpace>`) with
`mesh.count(dim)` rows and any number of components; `T` is any trivially copyable type the
reduction can combine. `op` is a `soma::Op` (`sum`, `min`, `max`, `prod`).

```cpp
template <class MemSpace>
class HaloExchange {
 public:
  HaloExchange(const Comm& comm, const Mesh<MemSpace>& mesh, int dim);
  [[nodiscard]] const std::vector<int>& neighbours() const noexcept;

  template <class T> void update(const Comm& comm, const View2<T, MemSpace>& values) const;
  template <class T> void reduce(const Comm& comm, const View2<T, MemSpace>& values, Op op) const;

  template <class T> [[nodiscard]] Pending<T> begin_update(const Comm& comm, const View2<T, MemSpace>& values) const;
  template <class T> [[nodiscard]] Pending<T> begin_reduce(const Comm& comm, const View2<T, MemSpace>& values, Op op) const;
};
```

From `tests/unit/test_halo_exchange.cpp`, owners' values reaching their copies:

```cpp
const soma::HaloExchange<Kokkos::HostSpace> x(c, m, 0);   // vertices
soma::View2<double, Kokkos::HostSpace> v("v", static_cast<std::size_t>(m.count(0)), 2);
// ... fill the owned rows [0, m.segment_table(0).owned_end) ...
x.update(c, v);   // copies and ghosts now hold their owners' values
```

A common pattern is to accumulate into every holder's rows, `reduce` with `Op::sum` so every
holder has the total, then `update` to refresh the ghosts, which `reduce` leaves alone.

### Behaviour

- **Construction is collective**: ghost holders tell each owner which of its entities they hold.
  It throws `std::invalid_argument` if `dim` is out of range or the mesh has no sharing fields on
  `dim` (it was not built by `build_local_part`), and on every rank if a rank holds a ghost its
  owner does not own. `neighbours()` lists the ranks this one exchanges with, ascending.
- **Exchanges are neighbour-to-neighbour**, not global collectives: each rank sends and receives
  only with its neighbours. Call each exchange on every rank, in the same order.
- **No handshake.** Both sides list the entities they share with a neighbour by gid, so the lists
  match without extra messages.
- **Bit-for-bit reductions.** `reduce` combines each entity's values in ascending rank order, the
  holder's own value in its place, so every holder computes the same result bit for bit,
  whatever the op. The result can still change with the partition, since the set of holders
  does.
- **Shape check.** Values whose row count is not `mesh.count(dim)` throw `std::invalid_argument`.
- **Memory spaces.** Packing and unpacking are Kokkos kernels over flat index arrays in the
  mesh's memory space, so a device mesh exchanges device views. Messages are staged through host
  memory; device buffers are not passed to MPI.
- **Tags.** Updates use tag 32610 and reductions 32611 of the communicator.

### Split (non-blocking) exchanges

`update` and `reduce` are each a `begin_*` followed at once by `end()`. The split form lets work
overlap the messages:

```cpp
auto pending = x.begin_update(c, v);   // packs the rows sent and posts the messages
// ... work that does not write the rows exchanged ...
while (!pending.test()) { /* more work */ }   // optional: advance and poll
pending.end();                         // waits, then writes the rows received
```

- `begin_update` and `begin_reduce` pack the rows sent, post the messages and return a
  `Pending<T>`. `end()` waits and writes the rows received (a reduction reads each holder's own
  rows at that point); `test()` advances the messages and returns whether they are done.
- Between `begin_*` and `end()` the caller must not write the rows exchanged. Nothing is written
  to `values` before `end()`.
- Exchanges begun in the same order on every rank may be in flight together and end in any
  order (`tests/unit/test_halo_exchange.cpp`, `ExchangesInFlightEndInAnyOrder`).
- A `Pending` is move-only and does not depend on the `HaloExchange` that made it. One dropped
  unfinished completes its messages and writes nothing. `end()` on an exchange that has ended or
  was moved from throws `std::logic_error`.

## What is collective

| Call | Scope |
|---|---|
| `Mesh::add_ghosts`, `Mesh::remove_ghosts` | Collective over `comm`, same options on every rank. |
| `HaloExchange` constructor | Collective over `comm`. |
| `update`, `reduce`, `begin_update`, `begin_reduce` | Messages with neighbours only; call on every rank in the same order. |
| `Pending::end`, `Pending::test` | Local completion of this rank's messages. |
