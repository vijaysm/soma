# Block-structured AMR

SOMA includes a patch-based (block-structured) adaptive mesh refinement layer in the namespace
`soma::amr`, under `include/soma/amr/`. It works on integer index spaces, not on the unstructured
`Mesh`: a level is a set of disjoint boxes of cells, finer levels are refined from coarser ones by
an integer ratio, and data live on the boxes as Kokkos views. Use it for finite-difference or
finite-volume solvers on Cartesian grids that refine where a tagging criterion asks for it. Every
piece works on one process and over MPI ranks; checkpoints of hierarchies are described in
[AMR checkpoints](../io/amr-checkpoints.md).

The layer is built from small parts that can be used alone:

| Header | Provides |
| --- | --- |
| `soma/amr/box.hpp` | `IntVect`, `IndexType`, `Box` |
| `soma/amr/box_list.hpp` | `BoxList` and set operations on lists of boxes |
| `soma/amr/box_array.hpp` | `BoxArray` (a level's boxes in curve order), `chop`, `Distribution` |
| `soma/amr/cluster.hpp` | Berger–Rigoutsos clustering of tagged cells (`cluster`) |
| `soma/amr/level_field.hpp` | `LevelField`, `LevelView`, `Patch`, `for_each_point` |
| `soma/amr/copy.hpp` | `CopyPlan`, `copy`, `begin_copy`, `Periodicity` |
| `soma/amr/halo.hpp` | `HaloPlan`, `fill_halo`, `begin_fill_halo` |
| `soma/amr/interlevel.hpp` | `CoarseFinePlan`, `fill_coarse_fine`, `AverageDownPlan`, `average_down` |
| `soma/amr/flux_register.hpp` | `FluxRegister` |
| `soma/amr/hierarchy.hpp` | `Geometry`, `HierarchyOptions`, `Hierarchy`, `MultiLevelField` |
| `soma/amr/vtk_amr.hpp` | `write_vtk_amr` (VTK overlapping AMR output) |

## Index space: boxes and index types

`soma/amr/box.hpp` defines the index space. Everything in it is device-callable.

- `IntVect` is three integers, `v[3]`, with `+` and `-`.
- A `Box` is a closed range `[lo, hi]` of integer points in 2 or 3 dimensions. In 2D the k range
  is `[0, 0]` and no operation changes it. The default-constructed box is empty.
- An `IndexType` says, per direction, whether the box's points are cells or nodes (bit `d` of
  `nodal` set: nodal in direction `d`). `IndexType::cell()` is cell-centred everywhere,
  `IndexType::node(dim)` nodal in every direction, `IndexType::face(d)` nodal in direction `d`
  only (the faces normal to `d`). A box of $n$ cells in a direction has $n + 1$ nodes there.

```cpp
Box(IntVect lo, IntVect hi, int dim = 3, IndexType type = IndexType::cell());
```

Boxes of different index types or dimensions are never mixed: operations on them return an empty
box. The operations are `&` (intersection), `contains`, `intersects`, `grow(n)` (all directions,
negative `n` shrinks), `grow(d, n)` and `grow(d, n_lo, n_hi)`, `shift`, `refine(ratio)` (a cell
covers `ratio` cells per direction; a node maps to the node at the same place), `coarsen(ratio)`
(the smallest coarser box covering this one), `convert(IndexType)` (the same region as points of
another index type), and the queries `length(d)`, `num_pts()`, `empty()`, `same_kind`.

### Box lists

`soma/amr/box_list.hpp` declares `using BoxList = std::vector<Box>;` and host-side set operations
on lists of boxes of one kind. Results are lists of pairwise disjoint boxes; inputs to `subtract`
and `complement` may overlap.

| Function | Result |
| --- | --- |
| `subtract(const Box& a, const Box& b)` | the points of `a` not in `b`, as at most `2 * dim` boxes |
| `subtract(const BoxList& a, const BoxList& b)` | the points of the boxes of `a` in none of `b` |
| `complement(const Box& domain, const BoxList& boxes)` | the points of `domain` in none of the boxes |
| `intersect(const BoxList& boxes, const Box& b)` | the non-empty intersections with `b` |
| `refine`, `coarsen`, `grow` | each box refined, coarsened or grown |
| `num_pts(boxes)` | the point count (each point once only when the boxes are disjoint) |
| `disjoint(boxes)`, `covers(boxes, b)` | whether the boxes are disjoint; whether they cover every point of `b` |

### Box arrays and distributions

A `BoxArray` (`soma/amr/box_array.hpp`) holds the boxes of one level, ordered along a
space-filling curve of their lower corners: `Curve::morton` (the default) or `Curve::hilbert`.

```cpp
BoxArray(BoxList boxes, const Box& domain, Curve curve = Curve::morton);
static BoxArray scratch(BoxList boxes, const Box& domain);
```

The constructor throws `std::invalid_argument`, naming the boxes, unless every box is of the
domain's kind, inside it, and disjoint from the others. `BoxArray::scratch` keeps boxes for
temporary data in the order given and does not check them.

`chop(boxes, max_box, blocking_factor = 1)` cuts each box into pieces no longer than `max_box` in
any direction; with a blocking factor `b > 1`, a box whose length is a multiple of `b` is cut at
multiples of `b` from its lower end.

A `Distribution` gives each box an owner rank, `owner[i]`. Ranks take contiguous runs of the
curve:

- `Distribution::along_curve(boxes, nranks)` gives each rank about the same number of points (a
  box goes to the rank holding its middle point);
- `Distribution::along_curve(boxes, nranks, weights)` does the same by one weight per box, so each
  rank's load is within the heaviest box of an equal share. It throws `std::invalid_argument`
  unless there is a finite, non-negative weight per box and they do not all vanish.
- `owned(rank)` lists the boxes of `rank` in curve order.

### Clustering tagged cells

`cluster` (`soma/amr/cluster.hpp`) implements Berger–Rigoutsos clustering (M. Berger and
I. Rigoutsos, IEEE Trans. Systems, Man and Cybernetics 21(5), 1991):

```cpp
struct ClusterOptions {
  double efficiency = 0.7;
  int block = 1;
};
BoxList cluster(const std::vector<IntVect>& tags, const Box& domain, const ClusterOptions& options);
```

Tagged cells are first coarsened into blocks of `block` cells; boxes of blocks are split
recursively until each is at least `efficiency` full of tagged blocks (or is a single block). A
box is split at a hole in the tag signature closest to the middle, else at the strongest
inflection of its Laplacian, else at the middle of its longest side. The result covers every tag,
lies in the domain, is aligned to the block, and its boxes are disjoint. The same tags give the
same boxes whatever their order. It throws `std::invalid_argument` for an efficiency outside
$(0, 1]$, a block below 1, a domain whose lower corner and lengths are not multiples of the
block, or a tag outside the domain.

## Level fields

A `LevelField<T, MemSpace>` (`soma/amr/level_field.hpp`) holds data on the boxes of one level:
per box, the points of its index type grown by `ghosts` in every direction, times `ncomp`
components. All boxes share one allocation with per-box offsets; each box is column-major in
i, j, k, then the component. `MemSpace` defaults to the default execution space's memory space.

```cpp
// The boxes are cell boxes; the field holds the points of `type` on them, every box on this process.
LevelField(const BoxArray& boxes, IndexType type, int ncomp, int ghosts,
           const std::string& label = "amr::LevelField");
// Only the boxes `rank` owns under `distribution`, in curve order.
LevelField(const BoxArray& boxes, const Distribution& distribution, int rank, IndexType type, int ncomp,
           int ghosts, const std::string& label = "amr::LevelField");
```

A field over many ranks holds only the boxes its rank owns, numbered from 0 in curve order:
`global(b)` is the level's id of the field's box `b`, and `local(g)` the field's index of the
level's box `g` (-1 when another rank owns it). `valid(b)` and `grown(b)` are the box's points
without and with ghosts, in the field's index type. `fill(value)` sets every point, ghosts
included.

Kernels use the device-copyable `LevelView` (`field.view()`): `patch(b)` returns a `Patch<T>`
indexed by the global point, `patch(b)(i, j, k, c)`. `for_each_point(field, region, f)` calls
`f(b, i, j, k)` once for every point of every box, over `Region::valid` or `Region::grown`, in one
launch over the points of all boxes, and fences. From the header:

```cpp
amr::LevelField<double> u(boxes, amr::IndexType::cell(), 1, 2);
auto v = u.view();
amr::for_each_point(u, amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
  v.patch(b)(i, j, k) = 1.0;
});
```

## Copies and halo exchange

### Copy plans

A `CopyPlan` (`soma/amr/copy.hpp`) fills given regions of destination boxes from a source level.
It is computed once and replayed:

```cpp
// regions[b]: the points of destination box b to fill, in `type`.
CopyPlan(const BoxArray& src, const std::vector<BoxList>& regions, IndexType type, Periodicity periodic,
         int src_ghosts = 0);
CopyPlan(const Comm& comm, const BoxArray& src, const Distribution& src_owner, const std::vector<BoxList>& regions,
         const Distribution& dst_owner, IndexType type, Periodicity periodic, int src_ghosts = 0,
         int tag = default_tag);

template <class T> void copy(const LevelField<T>& src, const LevelField<T>& dst, const CopyPlan& plan);
template <class T> PendingCopy<T> begin_copy(const LevelField<T>& src, const LevelField<T>& dst, const CopyPlan& plan);
```

Points are filled from the source's valid points first (the first source along the curve wins
where sources share points), then, with `src_ghosts > 0`, from the ghost points of the source
boxes (which the caller has filled), through periodic images too (`Periodicity` holds one flag
per direction). What no source covers is left unchanged and listed by `uncovered()`. The copies
of one destination box are disjoint, so the result does not depend on the thread count. Copies
cover components up to the smaller of the two fields' counts.

Over many ranks every rank computes the same copies from the replicated boxes and keeps its part:
local copies, and per other rank the points it sends and receives, in the same order on both
sides, so no message is needed to set the plan up. A copy posts the receives, packs and sends
(device kernels, staged through host memory), copies locally, then waits and unpacks. The split
form lets work overlap the messages:

```cpp
auto pending = amr::begin_copy(src, dst, plan);   // work may go here
pending.end();
```

`end()` is idempotent; destroying a `PendingCopy` before `end()` waits and drops what arrived.
`begin_copy` throws `std::invalid_argument` for fields that do not match the plan's index type,
source ghosts, owners or rank.

```{warning}
Fills of one plan that are in flight together must begin in the same order on every rank, as MPI
collectives must.
```

### Same-level halo exchange

A `HaloPlan` (`soma/amr/halo.hpp`), computed once for a level's boxes, an index type and a ghost
width, lists the copies that fill the ghost points of every box from the valid points of the
others (faces, edges and corners), across periodic directions too, and the ghost regions outside
the domain.

```cpp
HaloPlan(const BoxArray& boxes, IndexType type, int ghosts, Periodicity periodic);
HaloPlan(const Comm& comm, const BoxArray& boxes, const Distribution& distribution, IndexType type, int ghosts,
         Periodicity periodic, int tag = default_tag);

template <class T>
void fill_halo(const LevelField<T>& field, const HaloPlan& plan,
               const std::function<void(int box, const Box& region, int dir, int side)>& physical = {});
```

`fill_halo` makes the copies, then calls `physical(box, region, dir, side)` for each ghost region
outside the domain (`boundary()`, side 0 low, 1 high), after the copies so that corners next to a
periodic direction are set. Ghost points that no box covers keep their values; `uncovered()`
lists them per box, and the coarse-fine fill takes them. As with copies, the copies of one box are
disjoint, so a fill gives the same result at any thread count. `begin_fill_halo(field, plan)`
returns a `PendingHalo` whose `end()` finishes the copies (it does not call a boundary callback).
The field's index type must be the plan's and its ghost width at least the plan's, or the fill
throws `std::invalid_argument`.

## Operators between levels

`soma/amr/interlevel.hpp` holds the operators between a level and the next coarser one, with
refinement ratio $r$.

### Prolongation: the coarse-fine fill

A `CoarseFinePlan` fills the ghost points of a fine level that no fine box covers
(`HaloPlan::uncovered`) from the coarse level. `Interp` chooses the rule:

| `Interp` | Cell directions | Node directions |
| --- | --- | --- |
| `piecewise_constant` | the coarse value | linear between the two coarse nodes around the fine one |
| `linear` | the coarse value plus a slope limited by the monotonized-central limiter times the offset of the fine cell centre (conservative, exact for linear data) | linear |
| `quadratic` | the average over the fine cell of the quadratic whose averages over the coarse cells $q-1, q, q+1$ are theirs (conservative) | the quadratic through the three coarse nodes |

Fine nodes on coarse nodes are copied. Cell data are thus piecewise constant or conservative
linear, node data multilinear, and face data linear across the faces and constant or
limited-linear along them. Quadratic prolongation is exact for data of degree at most 2 in each
direction; it is not limited, so it suits smooth data.

```cpp
CoarseFinePlan(const BoxArray& fine, const BoxArray& coarse, IndexType type, int fine_ghosts, int ratio,
               Periodicity periodic, Interp interp, int coarse_ghosts = 1);
CoarseFinePlan(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner, const BoxArray& coarse,
               const Distribution& coarse_owner, IndexType type, int fine_ghosts, int ratio, Periodicity periodic,
               Interp interp, int coarse_ghosts = 1, int tag = default_tag);

template <class T>
void fill_coarse_fine(const LevelField<T>& fine, const LevelField<T>& coarse, const CoarseFinePlan& plan);
```

The coarse level must cover the coarse points needed, by its valid points or by ghosts the caller
has filled first (one ghost for the linear slopes next to the domain boundary); a fine level
properly nested in the coarse one does. The constructor throws `std::invalid_argument` naming a
fine box whose ghosts the coarse level does not cover. Overloads taking `targets` fill given
points of each fine box instead, valid points included, which is how a regrid prolongs data onto
new boxes.

### Restriction

`average_down` sets each coarse point under the fine valid points to the average of the fine
points it covers: cell averages (conservative), node injection, face averages over the coincident
fine faces. Fine boxes must be aligned to the ratio.

```cpp
AverageDownPlan(const BoxArray& fine, const BoxArray& coarse, IndexType type, int ratio);
AverageDownPlan(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner, const BoxArray& coarse,
                const Distribution& coarse_owner, IndexType type, int ratio, int tag = default_tag);

template <class T> void average_down(const LevelField<T>& fine, const LevelField<T>& coarse, const AverageDownPlan& plan);
template <class T> void average_down(const LevelField<T>& fine, const LevelField<T>& coarse, int ratio);
```

Averages are computed where the fine box lives, into scratch boxes, then copied to the owners of
the coarse boxes. `average_down` throws `std::invalid_argument` for fields of different index
types or component counts, fields not distributed as the plan's boxes, or (without a plan) a fine
box not aligned to the ratio.

### Flux registers

A `FluxRegister` (`soma/amr/flux_register.hpp`) keeps conservative schemes conservative across a
coarse-fine interface. Its faces are the coarse faces on the boundary of the fine level whose
coarse cell outside is not under the fine level (and not outside the domain).

```cpp
FluxRegister(const BoxArray& fine, const BoxArray& coarse, int ratio, Periodicity periodic, int ncomp);
FluxRegister(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner, const BoxArray& coarse,
             const Distribution& coarse_owner, int ratio, Periodicity periodic, int ncomp, int tag = default_tag);

void reset() const;
template <class T> void set_coarse(int dir, const LevelField<T>& coarse_flux, double scale) const;
template <class T> void add_fine(int dir, const LevelField<T>& fine_flux, double scale) const;
template <class T> void reflux(const LevelField<T>& state) const;
```

Fluxes are face fields, one per direction (`IndexType::face(d)`); the state is cell data. In a
step, `set_coarse` records the coarse fluxes on the register's faces and `add_fine` adds the fine
fluxes through the same faces, averaged over the $r^{\mathrm{dim}-1}$ fine faces of each, once
per fine step when fine steps are smaller. `reflux` then corrects the coarse cells outside the
fine level:

$$
u \mathrel{+}= -(F_\text{fine} - F_\text{coarse}) \text{ for the cell below a face},\qquad
u \mathrel{+}= +(F_\text{fine} - F_\text{coarse}) \text{ for the cell above it}.
$$

The caller scales the fluxes through `scale`: $\Delta t / \Delta x$ of the coarse level, times
the fraction of the coarse step for each fine step. Each coarse cell is updated once per direction
and side, so the result does not depend on the thread count. Adapted from
`tests/unit/test_amr_flux.cpp`, with one coarse step of `dtdx` and two fine half steps:

```cpp
const soma::amr::FluxRegister reg(fine, coarse, r, periodic, 1);
reg.reset();
for (int d = 0; d < dim; ++d) {
  reg.set_coarse(d, fc[d], dtdx);
  reg.add_fine(d, ff[d], dtdx / 2);  // first fine step
  reg.add_fine(d, ff[d], dtdx / 2);  // second fine step
}
soma::amr::average_down(uf, uc, r);
reg.reflux(uc);
```

Over many ranks a face's coarse flux lives with the owner of the coarse box (`set_coarse` and
`reflux` are local there) and its fine sum with the owner of the fine box (`add_fine` is local
there). `reflux` first sends the fine sums to the coarse side, one message per neighbour, so it is
collective over the neighbours. `set_coarse` and `add_fine` throw `std::invalid_argument` unless
given a face field normal to the direction, and `reflux` unless given cell data.

## Hierarchies

A `Hierarchy` (`soma/amr/hierarchy.hpp`) ties levels together over one domain. Level 0 covers the
domain; each finer level, refined by the ratio from the one below, is a union of disjoint boxes
aligned to the blocking factor and properly nested in the level below (by `nesting` cells of that
level, or up to the domain boundary). The hierarchy keeps the halo, coarse-fine and restriction
plans of its levels, built on first use (not safe to call concurrently).

```cpp
struct Geometry {
  Box domain;                                // the level-0 cells
  std::array<double, 3> lo{0.0, 0.0, 0.0};   // the physical box they cover
  std::array<double, 3> hi{1.0, 1.0, 1.0};
  Periodicity periodic{};
};

Hierarchy(const Geometry& geometry, const HierarchyOptions& options, const std::vector<BoxList>& fine = {});
Hierarchy(const Comm& comm, const Geometry& geometry, const HierarchyOptions& options,
          const std::vector<BoxList>& fine = {});
```

Without `fine` the hierarchy has level 0 only, the domain chopped to `max_box`; with it, levels 1
to `fine.size()` take those boxes (each in its level's cells).

| `HierarchyOptions` | Default | Meaning |
| --- | --- | --- |
| `max_level` | `0` | the finest level a regrid may make |
| `ratio` | `2` | refinement ratio between levels: 2 or 4 |
| `blocking_factor` | `8` | boxes are aligned to it; a multiple of the ratio |
| `max_box` | `32` | longest box side; a multiple of the blocking factor |
| `efficiency` | `0.7` | the clustering efficiency of regrids |
| `nesting` | `1` | proper nesting width, in cells of the level below |
| `curve` | `Curve::morton` | the curve ordering each level's boxes |
| `cost` | empty | `std::function<double(int level, const Box& box)>`, a box's cost for load balancing; empty: its point count. It must give every rank the same value |

The constructors throw `std::invalid_argument` for invalid options (the constraints in the table,
and a domain of whole blocks) or for boxes outside their domain, overlapping, not aligned to the
blocking factor or not properly nested.

Fields over every level are `MultiLevelField<T>`, a `std::vector<LevelField<T>>`. From the header:

```cpp
amr::Hierarchy h(amr::Geometry{amr::Box({0, 0, 0}, {63, 63, 63})}, {.max_level = 2});
auto u = h.make_field<double>(amr::IndexType::cell(), 1, 2);  // 1 component, 2 ghosts
h.fill_halo(u);
h.average_down(u);
```

- `fill_halo(u, interp = Interp::linear, physical = {})` works level by level from 0: ghosts from
  the same level, then (above level 0) the uncovered ones from the level below, then those outside
  the domain through `physical`. `begin_fill_halo` starts every level's same-level fill so that
  work can go between, and its `end()` finishes them coarse first; the field and `physical` must
  outlive the pending fill.
- `average_down(u)` restricts each level's valid points to the level below, from the finest down.
- `make_flux_register(level, ncomp)` makes a flux register between `level` and `level - 1`.
- `boxes(level)`, `distribution(level)`, `domain(level)`, `dx(level, d)`, `num_levels()` and
  `finest_level()` describe the levels.

A physical boundary is a callable
`physical(const LevelField<T>& field, int level, int box, const Box& region, int dir, int side)`
that fills the ghost points `region` of box `box` outside the domain; a generic lambda serves
fields of every element type. `NoBoundary` leaves them as they are.

### Regridding

A `Tagger`, `std::function<void(int level, const TagField& tags)>`, marks the cells of a level to
refine. `TagField` is `LevelField<std::uint8_t>`, one byte per cell, nonzero where the level
should be refined; tags arrive zeroed, without ghosts.

- `make_grids(tagger)` returns the boxes of new levels 1, 2, ... from the tags of levels 0 to
  `min(finest, max_level - 1)`, built from the top down: the tagged cells of level $l$, with the
  cells under the new level $l+2$ grown by the nesting, are clustered on blocks of
  `blocking_factor / ratio` cells, refined, and chopped to `max_box`. Empty levels on top are
  dropped; level 0 stays.
- `transfer(from, u, interp, physical)` returns field `u` of hierarchy `from` on this hierarchy's
  boxes (same geometry and ratio): level by level from 0, prolonged from the new level below,
  overwritten where the old level had valid points, then restricted from the finest level down.
  Linear and quadratic prolongation need at least one ghost.
- `regrid(tagger, interp, physical, fields...)` tags, makes the new levels and moves every field
  to them. On an exception the hierarchy and the fields are unchanged. An overload taking
  `measured[l][b]`, the measured cost of box `b` of this rank's part of level `l`, balances by
  those costs: a new box costs the old boxes' costs in proportion to its overlap with them, and
  `options.cost` where it overlaps none.

Adapted from `tests/unit/test_amr_regrid.cpp`:

```cpp
Hierarchy::Tagger ring(const Hierarchy& h, double cx) {
  return [&h, cx](int level, const TagField& tags) {
    tag_ring(tags, h.dx(level, 0), h.geometry().domain.dim(), cx, h.geometry().periodic[0]);
  };
}

Hierarchy h(geom, opts);
auto u = h.make_field<double>(IndexType::cell(), 2, 1);
h.regrid(ring(h, 0.5), Interp::linear, soma::amr::NoBoundary{}, u);
```

where `tag_ring` sets `tags.view().patch(b)(i, j, k) = 1` in a `for_each_point` kernel on the
cells to refine.

### Over many ranks

With the constructor taking a `Comm`, every rank holds every level's boxes, each level cut along
the curve by box cost (point count, or `options.cost`), and fields hold the boxes their rank owns.
Fills, restriction and regrids are collective. A regrid gathers the tagged blocks, so every rank
makes the same boxes as one process would, and rebalances. The tests in
`tests/unit/test_amr_distributed.cpp` check that fills and restriction over several ranks give the
same values as one process.

## VTK output

`write_vtk_amr` (`soma/amr/vtk_amr.hpp`) writes a hierarchy as VTK overlapping AMR, which ParaView
and VTK's XML readers read:

```cpp
struct VtkAmrField {
  std::string name;
  const MultiLevelField<double>* field = nullptr;
};

void write_vtk_amr(const Comm& comm, const std::filesystem::path& path, const Hierarchy& hierarchy,
                   const std::vector<VtkAmrField>& fields);
void write_vtk_amr(const std::filesystem::path& path, const Hierarchy& hierarchy,
                   const std::vector<VtkAmrField>& fields);
```

`path` must end in `.vthb`; it lists the boxes of every level, each written as image data to
`<stem>/<stem>_<level>_<box>.vti` beside it. Each box holds the valid points of every field (cell
fields as cell data, node fields as point data, components interleaved, ASCII in the shortest form
that reads back bitwise equal) and `vtkGhostType`, 8 (hidden) on the cells a finer level covers.
Points are at the geometry's lower corner plus their index times the level's spacing. From
`tests/unit/test_amr_vtk.cpp`:

```cpp
soma::amr::write_vtk_amr(path, h, {{"u", &u}, {"w.node-1", &w}});
```

Behaviour that matters:

- Bad input throws `std::invalid_argument` before any file is written: a path without the `.vthb`
  extension, a field that does not match the hierarchy's levels or is neither cell- nor
  node-centred, a name that is empty, repeated, `vtkGhostType`, or not made of letters, digits,
  `_`, `.` and `-`, or a value that is not finite.
- Each file is written to a temporary and renamed into place, the `.vthb` last, so a failure never
  leaves a `.vthb` naming missing files; write failures throw `std::runtime_error`.
- The overload taking a communicator is collective over the hierarchy's ranks: each rank writes
  the pieces of the boxes it owns, and rank 0 writes the `.vthb` once every piece is in place. The
  files are those the serial overload writes for the whole hierarchy. Bad input on any rank, or
  ranks giving different fields, throws `std::invalid_argument` on every rank; a failed write
  throws `std::runtime_error` on every rank. The overload without a communicator works over
  `Comm::self()` and needs a hierarchy of one rank.

Only `double` fields are written. To save and restart a hierarchy with its fields, use the SOMA
format's [AMR checkpoints](../io/amr-checkpoints.md).

## Determinism

- Copies, halo fills and coarse-fine fills give the same result at any thread count: the copies
  into one box are disjoint, the first source along the curve winning where sources overlap.
- A flux register updates each coarse cell once per direction and side.
- Clustering gives the same boxes for the same tags whatever their order, and a distributed regrid
  makes the boxes one process would make.
