# Mesh refinement

SOMA refines unstructured meshes in parallel: uniformly, splitting every cell, or where cells
are marked, keeping a 2:1 balance and either leaving hanging vertices or closing the mesh with
transition cells. Quad meshes for spectral elements can also be refined where marked into
quads alone, conforming ([all-quad conforming refinement](#all-quad-conforming-refinement)).
The refined mesh is the same, gid for gid and bit for bit, on any number of ranks. All the functions here are in `soma/algorithms/refine.hpp`, except `rebalance`, which is
in `soma/parallel/migrate.hpp`.

## How cells are split

Every edge is split in two, and each cell into children that tile it. A cell's new nodes are
its edges' midpoints, its quad faces' centres, and its centre where it has one:

| Cell | Children | How |
| --- | --- | --- |
| tri | 4 tris | the corners' halves and the middle one |
| quad | 4 quads | with the centre |
| polygon | $n$ quads | one per corner, with the centroid (corners averaged) |
| tet | 8 tets | the corners' halves and four about the shortest diagonal of the octahedron left (the first in edge order of those as short) |
| hex | 8 hexes | with the faces' centres and the centre |
| prism | 8 prisms | with the quad faces' centres |
| pyramid | 6 pyramids + 4 tets | with the base's centre: the corners' halves, one upside down under the apex's, and four tets |

A corner's child is the cell halved about it, so children are oriented as their parent is.
Every quad face gets a centre and no triangle does, so cells of any type meeting at a face split
it alike: refining a conforming mesh gives a conforming mesh, mixed meshes included.

A new vertex is the average of its corners (an edge's two, a face's four, a cell's all),
summed in gid order, so every rank holding it computes the same bits.

```{figure} ../figures/img/refinement-children.svg
:alt: A triangle and a square each split into four children, and a tetrahedron and a cube each split into eight, the children pulled apart
:width: 90%

The children `refine_uniform` makes of a tri, a quad, a tet and a hex, each pushed away from the cell's
centre so that all show (the parent dashed in 2D).
```

### Gids

Gids do not depend on the number of ranks. With $V$, $E$ and $F$ the largest vertex, edge and
face gid on any rank (edges and faces derived as `build_distributed_adjacencies` derives
them):

- an edge's midpoint is $V +$ its edge gid;
- a quad face's centre is $V + E +$ its face gid (3D);
- with the cells numbered from 1 in gid order ($n = \text{gid} - G + 1$, $G$ the least cell
  gid), a cell's centre is $V + E + F + n$ ($F = 0$ in 2D), and its child $c$ is
  $K(n - 1) + c + 1$, $K$ the most children of any cell.

### Star-shaped cells

A child's corners are its parent's mapped from reference coordinates, so a cell that is not
star-shaped about its corners' average (a non-convex quad or polygon; in 3D a strongly
distorted cell) would have children turned against it. Plane cells are checked and refused;
volume cells are not checked. `refine_marked` checks only the plane cells it splits, so a mesh
holding such cells (for example the dart-shaped children of L-shaped polygons) can still be
refined where they are not marked.

## Uniform refinement

```cpp
struct RefineOptions {
  int levels = 1;  // passes; each splits every edge in two
};

[[nodiscard]] RefineResult refine_uniform(const Comm& comm, const HostMesh& mesh, RefineOptions options = {});
```

```cpp
const soma::RefineResult fine = soma::refine_uniform(comm, mesh, {.levels = 2});
fine.mesh;       // the refined local part, children on their parent's rank
fine.parent[c];  // the cell of `mesh` that fine cell c lies in
```

Each level is `refine_marked` (below) with every cell marked, so uniform refinement keeps the
same refinement state as marked refinement:

- the cell field `refine:level` goes one up per level;
- `refine:birth` is recorded for each new vertex, so `refine_marked` can follow;
- vertices made earlier on the cells' edges and faces are reused;
- a mesh that has hanging vertices keeps them, and they are listed in `RefineResult::hanging`;
- transition cells of a conforming result are put back first.

On a conforming mesh without that state the result is plain uniform refinement as described
above.

`refine_uniform` is collective. It throws `std::invalid_argument` on every rank if any rank's
mesh is not of dimension 2 (tris, quads, polygons) or 3 (tets, hexes, prisms, pyramids), is
edited (changed by `MeshEdit::apply` and not compressed since) or has high-order nodes, or
holds a quad or polygon not star-shaped about its corners' average; if `levels < 1` or the
ranks differ in `levels` or dimension; or if a gid of the refined mesh would not fit a `GO`.

## The result

```cpp
struct RefineResult {
  HostMesh mesh;
  std::vector<LO> parent;
  std::vector<Offset> child_offsets;
  std::vector<LO> children;
  std::vector<std::string> dropped;
  std::vector<LO> hanging;
  std::vector<Offset> hanging_offsets{0};
  std::vector<GO> hanging_masters;
  std::vector<Real> hanging_weights;
};
```

| Member | Meaning |
| --- | --- |
| `mesh` | The refined local part. Children stay on their parent's rank. It has no ghosts: add them again as needed. |
| `parent` | For each cell of `mesh`, the owned cell of the input it lies in, as a local id of the input. For a cell in a group of transition cells that the call put back, the group's least cell by gid. |
| `child_offsets`, `children` | The children of the input's owned cells: cell `p`'s are `children[child_offsets[p], child_offsets[p + 1])`, local ids of `mesh`. |
| `dropped` | Names of the fields not carried over. |
| `hanging` | The hanging vertices of `mesh` (local ids): each the midpoint of an edge or the centre of a quad face that a cell still has whole. |
| `hanging_offsets`, `hanging_masters`, `hanging_weights` | Hanging vertex `i` takes the value $\sum_k w_k\, \text{value}(m_k)$ over `k` in `[hanging_offsets[i], hanging_offsets[i + 1])`, masters $m_k$ given by gid. A master may itself be hanging, one level up. |

### Fields

Fields carry over where they are dense: one value per entity of a full domain, per-entity
layout, not variable-length and not handles (see [fields](../concepts/fields.md)). Their
default and global values come with them.

- A cell field's children take the parent's value, so integrals of cell averages are kept.
- A vertex field of floating-point values keeps the values of the old vertices and gives a
  new vertex the average of its corners' values: exact for linear, bilinear and trilinear
  fields.

Vertex values are read wherever a rank holds them, shared copies and ghosts included, so these
must hold their owners' values (update them first) for ranks to agree. Other fields, edge and
face fields, and integer vertex fields are dropped and named in `dropped`. The internal
`soma:` fields are rebuilt.

## Marked refinement

```cpp
enum class VertexMatching : std::uint8_t {
  recorded,     // by the corners recorded in "refine:birth"
  coordinates,  // also by coordinates, for vertices with no record (infer_births)
};

struct MarkedOptions {
  bool conforming = false;
  VertexMatching vertices = VertexMatching::recorded;
  PointTolerance tol{};  // VertexMatching::coordinates: exact by default
};

[[nodiscard]] RefineResult refine_marked(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                                         MarkedOptions options = {});
```

`marks` holds one entry per owned cell, in local id order. The owned cells with a nonzero mark,
and the cells the 2:1 balance needs, are split once as `refine_uniform` splits them; the others
stay whole.

| Option | Default | Meaning |
| --- | --- | --- |
| `conforming` | `false` | `false`: leave hanging vertices. `true`: leave none (bisection or transition cells, below). |
| `vertices` | `VertexMatching::recorded` | How vertices made by earlier refinement are found: by their recorded births, or also by coordinates for vertices without a record. |
| `tol` | exact | With `VertexMatching::coordinates`, the tolerance of the coordinate match (`PointTolerance`, from `soma/parallel/point_match.hpp`; see [matching and merging](../parallel/matching-and-merging.md)). |

### Refinement state

The refinement state lives in fields of the mesh, so later calls, on any partition, can
use it:

- `refine:level` (a cell field) holds each cell's level, taken as 0 where the field is missing;
- `refine:birth` (a vertex field) records, for each vertex refinement made at an edge's
  midpoint or a quad face's centre, the corners it lies between (a count, then up to four
  gids). Later calls, on any rank, find and reuse such a vertex through a directory keyed by
  those corners, whatever the partition;
- `refine:green` (a cell field) records, on transition cells, the cell each replaced (below);
  it is zero on other cells.

### Balance

A cell is split when a cell touching it at a corner, or at a vertex born on its edges or
faces (or on theirs, one level down), would otherwise be two levels finer than it.

### Non-conforming refinement (the default)

Unsplit cells next to split ones keep their edges and faces whole, so the new vertices on
those edges and faces are hanging. `RefineResult::hanging` lists them, each with its masters
and weights; a hanging vertex's value is its masters' average.

### Conforming refinement

With `options.conforming`, no hanging vertex is left:

- **Triangles alone, or tets alone: bisection.** Each marked cell is cut in two at the
  midpoint of its longest edge (the one of least corner gids among equals), and any cell with
  a vertex on one of its edges is cut so again, until none is left (longest-edge, or Rivara,
  bisection). Each cut counts as a level.
- **Any other mesh: transition cells.** Cells are split as in non-conforming refinement, then
  each cell left with hanging vertices is replaced by transition cells. In 2D these are
  triangles from its centre to its sides, divided at their hanging vertices. In 3D they are a
  cone from its centre over each face: tets over triangles, pyramids over quads, a face split
  as its neighbour split it or else, with hanging vertices on its edges, fanned into triangles
  from its centre (which both cells on it share).

Transition cells keep their parent's level and are cells like any other in the result. Each
records the cell it replaced in `refine:green` (its gid, topology, corner count and corners'
gids). The next call to `refine_marked` puts each group back as that cell before it refines
(red-green refinement). A group split between ranks is first gathered on the rank of its least
cell by gid; the restored cell takes that cell's gid, fields and level, and it is marked when
any cell of its group is.

### Finding earlier vertices by coordinates

By default (`VertexMatching::recorded`) earlier vertices are found by their recorded births
alone. This needs no coordinates, so vertices may move between calls (for example by
[smoothing](smoothing.md)).

With `options.vertices = VertexMatching::coordinates`, `refine_marked` first finds, as
`infer_births` does (below), the births of vertices that have no record, exactly or within
`options.tol`. This costs one `match_points` over the owned vertices and the points asked, per
depth of refinement found.

### Behaviour and errors

Gids, coordinates and fields follow `refine_uniform` (a vertex made at a centre takes its
corners' average), and the result is the same on any number of ranks. `refine_marked` is
collective. It throws as `refine_uniform` does, except that it checks for star shape only the
plane cells it splits, and throws `std::invalid_argument` on every rank if a rank's marks are
not one per owned cell.

## Rebuilding births: `infer_births`

```cpp
void infer_births(const Comm& comm, HostMesh& mesh, PointTolerance tol = {});
```

`infer_births` records in `refine:birth` the corners each vertex lies between, found by
coordinates (`match_points`, exactly or within `tol`) as refinement would have placed it: at
the midpoint of an owned cell's edge or the centre of its quad face, and then, as refinement
cuts them, at the midpoints of those edges' halves, of a triangle's halves about an edge's
midpoint and of its four children, and at the centres of a quad's four children, to any depth.

Use it for meshes with hanging vertices whose records are missing: read from a format that
drops the field, made by other means, or taken from an AMR hierarchy. `refine_marked` then
reuses those vertices and lists them as hanging.

- Recorded births stay as they are.
- Cell levels are not inferred: `refine:level` is 0 where missing.
- A vertex moved since it was made (smoothed, snapped) is not found.

It is collective and throws as `refine_marked` does and as `match_points` does for `tol`.

## All-quad conforming refinement

```cpp
struct QuadRefineOptions {
  std::optional<Real> sphere_radius;  // project new vertices onto this sphere about the origin
};
[[nodiscard]] RefineResult refine_quads(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                                        QuadRefineOptions options = {});
```

`refine_quads` refines a mesh of quads where cells are marked and keeps it a mesh of quads
with no hanging vertex. This is what [spectral elements](spectral-elements.md) need: their
elements are quads, and a hanging vertex would need constraints the continuous space does not
have. `refine_marked` with `conforming` closes a quad mesh with triangles instead, and without
it leaves hanging vertices. Use `refine_quads` for a regionally refined mesh of the sphere or
of a planar domain; use `refine_marked` for everything else.

A marked cell is split into nine (each edge in three), and its neighbours into transition
quads (Schneiders' vertex-based templates). The input must be conforming: hanging vertices
are not detected, and a mesh that has them (such as `refine_marked`'s default output) gets
more. Each call splits once; there is no 2:1 balance
across calls. The output is order 1: a later SemSpace maps its elements bilinearly, projected
onto the sphere.

### Templates

Marks move from cells to their corners: a marked cell marks its four. Each cell then splits by
which of its corners are marked. With reference corners $v_0 = (0,0)$, $v_1 = (1,0)$,
$v_2 = (1,1)$ and $v_3 = (0,1)$, in the cell's corner order, the four cases are:

| Marked corners | Children | New vertices (reference) | Children, in order |
| --- | --- | --- | --- |
| none | 1 | none | $[v_0 v_1 v_2 v_3]$ (the cell, whole) |
| one ($v_0$) | 3 | $a\,(1/3,0)$, $c\,(0,1/3)$, $p\,(1/3,1/3)$ | $[v_0 a p c]$, $[a v_1 v_2 p]$, $[c p v_2 v_3]$ |
| two adjacent ($v_0$, $v_1$) | 7 | $a\,(1/3,0)$, $b\,(2/3,0)$, $d\,(1,1/3)$, $c\,(0,1/3)$, $p\,(1/3,1/3)$, $q\,(2/3,1/3)$, $r\,(1/3,2/3)$, $s\,(2/3,2/3)$ | $[v_0 a p c]$, $[a b q p]$, $[b v_1 d q]$, $[c p r v_3]$, $[p q s r]$, $[q d v_2 s]$, $[r s v_2 v_3]$ |
| all four | 9 | the $3 \times 3$ grid $(i/3, j/3)$ | row by row from $v_0$ |

The other marked corners are these cases turned about the cell. A cell with two opposite
corners or three corners marked has no template: it is **promoted**, all four of its corners
marked. That can change its neighbours' cases, so promotion repeats until no cell on any rank
changes. Marks only grow, and a promoted cell's marks stay promoted as more are added, so the
result does not depend on the order cells are visited in, or on the ranks.

An edge's new vertices depend on its two ends alone: both marked, vertices at 1/3 and 2/3;
one marked, a vertex at 1/3 from it; none, none. The two cells on an edge therefore split it
alike, and the result is conforming.

### Positions, gids and levels

- **Positions.** A vertex on an edge is $(2/3)\,lo + (1/3)\,hi$ or $(1/3)\,lo + (2/3)\,hi$,
  with $lo$ and $hi$ the edge's ends by gid. A vertex inside a cell is the bilinear map of the
  cell's corners at its reference position, with the weights summed in the corners' gid order.
  Every rank holding a vertex therefore computes the same bits.
- **Gids.** Let $V$ and $E$ be the largest vertex and edge gids on any rank (edges derived as
  `build_distributed_adjacencies` derives them), and number the cells $n = \mathrm{gid} - G + 1$,
  with $G$ the least cell gid.
  - The vertex on edge $e$ nearer its lower-gid end is $V + 2(e - 1) + 1$; the other is
    $V + 2(e - 1) + 2$.
  - The new vertex nearest corner $j$ inside cell $n$ is $V + 2E + 4(n - 1) + j + 1$.
  - Child $c$ (0-based, in the template's order) is $9(n - 1) + c + 1$. A whole cell is its
    own child 0, so every cell is renumbered, even with no marks.
- **Levels.** The cell field `refine:level` (0 where missing) goes up by one for the nine
  children of a cell split fully; transition children and whole cells keep their parent's
  level. No other state is kept, so a later call takes the result as a mesh like any other.
- **Fields** carry over as `refine_uniform` carries them. A new vertex's value is weighted as
  its position is, before projection, so linear fields come through exactly in the plane.

### The sphere and validity

With `options.sphere_radius = R`, each new vertex $x$ is projected onto the sphere,
$x \leftarrow R\,x/|x|$; the input's vertices stay where they are.

Each split cell, and each of its children (projected), must turn as the cell does at every
corner, about the cell's area vector $\sum_i x_i \times x_{i+1}$. A bilinear quad that turns
positively at all four corners has a positive Jacobian everywhere, so at its GLL points too.
A cell that fails, such as a dart (one corner reflex), throws on every rank, naming the least
such cell gid. Cells left whole are not checked.

### Errors

`refine_quads` is collective. It throws `std::invalid_argument` on every rank, with a message
beginning `refine_quads:`, if:

- any rank's mesh holds a cell other than a quad, is not of dimension 2, has edits or has
  high-order nodes (Quad9);
- a rank's marks are not one per owned cell;
- a rank's sphere radius is not positive and finite, or the ranks differ in it (or in giving
  one);
- a split cell or one of its children turns against the cell;
- a gid of the result would not fit a `GO`.

### Example: refining towards a coastline

Mark by distance to a curve, refine, and refine again closer in:

```cpp
#include "soma/algorithms/refine.hpp"
#include "soma/sem/cubed_sphere.hpp"

const soma::Real R = 6.37122e6;
soma::sem::CubedSphereSpec spec;
spec.ne = 30;
spec.radius = R;
soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);
// distance(mesh, e): a cell's distance to the coastline, from the application's data.
for (const soma::Real within : {500e3, 150e3}) {
  std::vector<soma::Real> closeness;
  for (soma::LO e = 0; e < mesh.segment_table(2).owned_end; ++e) closeness.push_back(-distance(mesh, e));
  const auto marks = soma::mark_threshold(closeness, -within);
  mesh = soma::refine_quads(comm, mesh, marks, {.sphere_radius = R}).mesh;
}
```

Each pass leaves the cells it split on their parent's rank; call `soma::rebalance` between
passes to even the ranks out. Add ghosts and edges before building a SemSpace on the result,
as for the cubed sphere.

### Regionally refined meshes from elsewhere

E3SM's regionally refined meshes (RRM) and SQuadGen's are Exodus files (`.g`). Convert one
with MOAB's `mbconvert` and read the result:

```bash
mbconvert ne30x4_rrm.g ne30x4_rrm.h5m
```

```cpp
soma::HostMesh mesh = soma::read_h5m(comm, "ne30x4_rrm.h5m");
```

SOMA has no Exodus reader of its own.

## Marking

```cpp
[[nodiscard]] std::vector<std::int8_t> mark_threshold(std::span<const Real> indicator, Real threshold);
[[nodiscard]] std::vector<std::int8_t> mark_fraction(const Comm& comm, std::span<const Real> indicator, Real fraction);
```

Both take an indicator per owned cell, in local id order, and return marks for `refine_marked`
or `refine_quads`.

- `mark_threshold` marks the cells whose indicator exceeds `threshold`. It is local.
- `mark_fraction` is Doerfler marking: it marks the cells of largest indicator whose indicators
  add up to at least `fraction` of their sum over every rank. Precisely, it marks the cells
  whose indicator is at least the largest cutoff that reaches the fraction, so cells of equal
  indicator are marked alike. A fraction of 0, or a sum of 0, marks nothing. Indicators should
  be non-negative, such as squared error estimates. The sums are rounded as each partition adds
  them. It is collective and throws `std::invalid_argument` on every rank for a fraction outside
  $[0, 1]$, or an indicator that is negative or not finite on any rank.

## Rebalancing

Refinement keeps children on their parent's rank, so refining one region makes the ranks
holding it heavier. `soma::rebalance` (`soma/parallel/migrate.hpp`) evens them out:

```cpp
[[nodiscard]] HostMesh rebalance(const Comm& comm, const HostMesh& mesh, Ordering ordering = default_ordering);
```

It moves the owned cells to the ranks `partition_by_curve` gives their centroids (their
corners' average), with `migrate`: parts along the curve whose cell counts differ by at most
one, whatever the partition given. Each entity takes the field values its owner held, so
the `refine:` fields move with their cells and vertices. It is collective and throws as `migrate` and `partition_by_curve` do. See
[distributed meshes](../parallel/distributed-meshes.md).

## Example

`examples/refine.cpp` reads a mesh, gives each vertex the linear field $w = x + 2y + 3z$ and
each cell its gid as `part`, and refines it `--levels` times: uniformly, or with `--marked F`
the cells holding the fraction `F` of an indicator, rebalancing after each round. Its marked
loop:

```cpp
const soma::HostMesh* from = &mesh;
for (int level = 0; level < options.levels; ++level) {
  const auto marks = soma::mark_fraction(comm, indicator(*from, corner), fraction);
  soma::RefineResult r = soma::refine_marked(comm, *from, marks, {.conforming = conforming});
  hanging = 0;
  for (const soma::LO v : r.hanging) hanging += v < r.mesh.segment_table(0).owned_end ? 1 : 0;
  hanging = comm.allreduce(hanging, soma::Op::sum);
  fine = soma::rebalance(comm, r.mesh);
  from = &fine;
}
```

It prints the cell and vertex counts before and after, the hanging vertices left, and the
largest error of `w` on the refined vertices, which is zero up to rounding because new
vertices average linear fields exactly:

```bash
mpiexec -n 4 ./refine --levels 3 --marked 0.5 in.h5m out.soma
mpiexec -n 4 ./refine --levels 3 --marked 0.5 --conforming in.h5m out.soma
```

```{figure} ../figures/img/marked-refinement.png
:alt: Two copies of the Y-shaped mesh refined towards the lower left corner and coloured by refinement level 1 to 3: on the left with dots on the hanging vertices along the level boundaries, on the right with triangles closing the coarser cells next to finer ones
:width: 90%

The example's marked loop on `data/hybrid_3quad_1tri.h5m`: 3 levels, each marking the fraction 0.5 of the
indicator towards the lower left corner, coloured by `refine:level`. Non-conforming (left, the default): 730
cells, with the 35 hanging vertices of the last level as dots. Conforming (right): 929 cells, transition
triangles closing the cells next to finer ones. Each pass marks the mesh it has, so the two runs refine
different cells from the second level on, not the same cells closed differently.
```

## Memory spaces and limits

- Refinement runs on the host: the functions take and return a `HostMesh` (built with
  `build_local_part`).
- The result has no ghosts.
- Volume cells are not checked for star shape; a strongly distorted cell can give inverted
  children.
- Fields other than dense cell fields and floating-point vertex fields are dropped.
