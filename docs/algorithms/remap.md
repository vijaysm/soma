# Conservative remap

`soma::RemapOperator` transfers a cell field from one mesh (the source) to another (the
target) by first-order conservative remap. It works between plane meshes, volume meshes and
meshes on a sphere, on any number of ranks, with the two meshes partitioned independently. The
operator, and every field remapped with it, are the same bit for bit on any partition and
thread count.

## What it computes

For each target cell $T$,

$$
u_T = \sum_S w_{TS}\, u_S, \qquad w_{TS} = \frac{|T \cap S|}{|T|},
$$

where $|T \cap S|$ is the area (in 3D the volume) the two cells share. Two properties follow:

- where the source covers a target cell, its weights sum to one, so constants are kept;
- where each mesh covers the other, $\sum_T |T|\, u_T = \sum_S |S|\, u_S$: the total (mass)
  is conserved.

The overlaps are found by clipping each pair of candidate cells against each other.

## API

Header: `soma/algorithms/remap.hpp`.

```cpp
enum class SpatialSearch { grid, arborx };
enum class Surface { plane, sphere };
enum class SphereClip { gnomonic, great_circle };

struct RemapOptions {
  Real merge_tolerance = 1e-12;
  SpatialSearch search = SpatialSearch::grid;
  Surface surface = Surface::plane;
  Real radius = 0;
  SphereClip sphere_clip = SphereClip::gnomonic;
};

template <class MemSpace>
class RemapOperator {
 public:
  RemapOperator(const Comm& comm, const Mesh<MemSpace>& source, const Mesh<MemSpace>& target,
                const RemapOptions& options = {});
  LO rows() const;  // owned target cells
  LO cols() const;  // source cells that arrived here
  const View1<Offset, MemSpace>& offsets() const;
  const View1<LO, MemSpace>& columns() const;
  const View1<Real, MemSpace>& weights() const;
  const View1<Real, MemSpace>& overlap() const;      // |T n S| per entry
  const View1<Real, MemSpace>& target_area() const;  // |T| per row
  const View1<GO, MemSpace>& source_gid() const;     // per column
  void apply(const Comm& comm, const View1<const Real, MemSpace>& u_source,
             const View1<Real, MemSpace>& u_target) const;
};
```

| Option | Default | Meaning |
| --- | --- | --- |
| `merge_tolerance` | `1e-12` | The merge tolerance (below): a length in the meshes' units, finite and positive. |
| `search` | `SpatialSearch::grid` | How candidate cells are found: `GridIndex`, or `ArborXIndex` with `SpatialSearch::arborx` (see [spatial search](spatial-search.md)). |
| `surface` | `Surface::plane` | For meshes of dimension 2: in a plane $z = \text{const}$, or on a sphere about the origin. |
| `radius` | `0` | On a sphere: its radius; 0 takes the largest distance of a vertex from the origin. |
| `sphere_clip` | `SphereClip::gnomonic` | On a sphere: how cells are clipped (below). |

Every option must be the same on every rank.

The operator is held as compressed sparse rows: row $t$ (an owned target cell) has entries
`k` in `[offsets()(t), offsets()(t + 1))`, with column `columns()(k)` (a source cell that
arrived on this rank, whose gid is `source_gid()(columns()(k))`), weight `weights()(k)` and
overlap `overlap()(k)`. A row's entries are ordered by source gid.

### Usage

From `examples/remap.cpp`, which remaps a field from a mesh read from a file onto a grid of
triangles or tets covering it:

```cpp
const soma::HostMesh source = soma::read_file(comm, in);
soma::HostMesh target = grid(comm, dim, n, lo, hi);  // built with build_local_part
soma::RemapOptions options;                          // or {.merge_tolerance = 1e-10}
const soma::RemapOperator<Kokkos::HostSpace> remap(comm, source, target, options);
const View u = values(source);  // one value per owned source cell
View ut("ut", static_cast<std::size_t>(remap.rows()));
remap.apply(comm, soma::View1<const double, Kokkos::HostSpace>(u), ut);
```

The example also builds the operator from the target back to the source, whose
`target_area()` gives $|S|$ for each source cell, to print both totals:

```cpp
const soma::RemapOperator<Kokkos::HostSpace> back(comm, target, source, options);  // for |S|
double before = 0, after = 0;
for (soma::LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(e);
for (soma::LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * ut(e);
before = comm.allreduce(before, soma::Op::sum);
after = comm.allreduce(after, soma::Op::sum);
```

```bash
mpiexec -n 4 ./remap --n 24 --tolerance 1e-12 in.h5m out.soma
```

### `apply`

`apply(comm, u_source, u_target)` is collective. `u_source` holds a value per owned source
cell (at least), `u_target` receives one per owned target cell (at least). It replays the
routing the constructor found: one exchange with the ranks found, through host memory, then
each row's sum in `MemSpace`. A rank whose vectors are too short throws
`std::invalid_argument` after taking part in the exchange, so the other ranks are not left
waiting (they do not throw).

## Plane meshes

Meshes of dimension 2 with `Surface::plane` (the default). Each cell's corners must lie in one
plane $z = \text{const}$; the cells are taken in the $xy$ plane. Cells may be triangles,
quads or polygons of at most 64 corners.

Each cell is turned counter-clockwise and started at its least-gid corner, so every rank sees
the same polygon whatever order the cell lists its corners in. Convex cells are clipped
directly (Sutherland and Hodgman's algorithm, `soma/algorithms/polygon.hpp`). A non-convex
cell is split into triangles by ear clipping and its overlaps are summed over them. A polygon
counts as convex when no corner turns clockwise by more than the tolerance; corners in a line
are allowed.

A cell thinner than the tolerance (twice its area over its longest side), crossing itself, or
off a plane $z = \text{const}$ is refused (see [errors](#errors)).

The overlaps agree with MOAB's `Intx2MeshInPlane` to $10^{-13}$ of the target cell's area
(`tests/reference/test_remap_moab.cpp`, on perturbed triangle and quad grids, which MOAB
needs to be convex).

## Volume meshes

Meshes of dimension 3 of tets, hexes, prisms and pyramids, mixed as needed. Two methods are
used, chosen per cell:

- **Clipped whole.** A hex, prism or pyramid that is convex with flat faces (each corner
  within the tolerance of its faces' planes) is clipped as a polyhedron cut by the other
  cell's face planes (`soma/algorithms/convex_clip.hpp`). Its corners are listed from the
  rotation of it whose corner gids come least, so the result still does not depend on how the
  cell lists them.
- **Split into tets.** Any other cell is split into tets: a tet is itself; another cell has
  one tet per triangle of its faces, with apex at its centroid. A quad face is cut along the
  diagonal from its least-gid corner, so that two cells sharing a face cut it alike and their
  tets fill space without gaps, non-planar and non-convex faces included. Faces are taken in
  the order of their corner gids. Tets are clipped by one another's face planes
  (`soma/algorithms/tet_clip.hpp`).

The two give the same overlaps to rounding. Clipping whole is much faster: the changelog
records building the operator between hex grids of $24^3$ and $23^3$ cells in 0.25 s instead
of 5.0 s, and hexes onto tets in 1.1 s instead of 2.9 s, measured on one rank.

A cell with a tet thinner than the tolerance, or not star-shaped about its centroid (its tets
turned unlike one another), is refused. An inverted cell (all its tets turned the other way)
is accepted and taken at its volume's magnitude.

## Meshes on a sphere

Meshes of dimension 2 with `surface = Surface::sphere`: the cells lie on a sphere about the
origin, their edges are great-circle arcs, and areas are spherical
(`soma/algorithms/sphere_clip.hpp`). Cells may be triangles, quads or polygons of at most 64
corners, as in the plane.

```cpp
const soma::RemapOperator<Kokkos::HostSpace> remap(
    comm, source, target, {.surface = soma::Surface::sphere, .radius = radius, .sphere_clip = soma::SphereClip::great_circle});
```

The radius is `RemapOptions::radius`, or, if that is 0, the largest distance of any vertex of
either mesh from the origin. A corner off the sphere by more than the tolerance is refused.
Each cell is turned counter-clockwise as seen from outside; a non-convex cell is split by ear
clipping in the plane touching the sphere at its corners' mean. Overlaps and areas are in the
meshes' units (they sum to $4\pi r^2$ over a whole sphere), and the merge tolerance is a length
in those units, as in the plane.

Two clipping methods are available:

| | `SphereClip::gnomonic` (default) | `SphereClip::great_circle` |
| --- | --- | --- |
| Method | Both cells projected from the centre onto the plane of a cube face (great circles become straight lines), clipped there as plane polygons, and the overlap's corners taken back onto the sphere; as MOAB's `Intx2MeshOnSphere`. | The cell cut by the plane through the centre and each edge of the other cell in turn. |
| Cell size | Two cells reaching past the face's horizon, as cells wider than about 45 degrees may, cannot be projected: the constructor throws. | Cells of any size. |
| Cost | A few percent faster. | |

The two agree to rounding (`tests/unit/test_remap_sphere.cpp` checks $10^{-12}$ of a target
cell's area), and with MOAB's overlaps to about $10^{-15}$ of a cell's area
(`tests/reference/test_remap_sphere_moab.cpp` checks $10^{-13}$, on cubed spheres and
icosahedral meshes). The changelog records that between cubed spheres of about 14000 cells
the two take 0.60 s and 0.62 s, and MOAB 2.0 s, measured on one rank. Use `great_circle` for
coarse meshes with wide cells; `tests/unit/test_remap_sphere.cpp` remaps from a cubed sphere
of one cell per face, which `gnomonic` refuses.

## The merge tolerance

`RemapOptions::merge_tolerance` is a length in the meshes' units, the same on every rank. It
decides:

- when a corner is on a clipping line or plane: a corner outside by at most the tolerance is
  moved onto it (one inside is kept, so that of two cells sharing an edge or face one, not
  both, takes the band between);
- when clipped corners merge: those closer than it become one;
- when an overlap is dropped: one thinner than it is no overlap;
- which cells are refused as too thin or, on a sphere, as off the sphere.

Merging moves areas by up to about the tolerance times a cell's perimeter, so weights sum to
one only to that accuracy. Keep the tolerance far below the cells' size: the default
$10^{-12}$ suits cells of any size above about $10^{-6}$. Meshes whose corners differ by less than
the tolerance are treated as coincident: `tests/unit/test_remap.cpp` checks that a target
grid shifted by $10^{-14}$ gives each target cell its own source cell alone, while one shifted
by $10^{-9}$ also picks up slivers of the neighbours.

Cells are clipped about a corner of the clipping cell, so accuracy does not depend on the
distance from the origin.

## Partitioning and bit-for-bit results

The source and target may be partitioned differently and need no ghosts. The constructor:

1. shares the bounding box of each rank's owned target cells (an allgather of $P$ boxes);
2. sends each owned source cell, already split into the pieces it is clipped as, to every rank
   whose box it meets (`transport::route`; see [transport](../parallel/transport.md));
3. on each rank, indexes the source cells that arrived with a `GridIndex` (or an `ArborXIndex`)
   and clips each owned target cell against its candidates.

Because a cell's pieces are made by its owner and sent as they are, each cell is started at
its least-gid corner, and a row's entries are ordered by source gid, the operator and
`apply()` are the same bit for bit on any partition and number of threads, and on any listing
of a cell's corners. Both indexes find the same candidates in the same order, so the choice of
search does not change a bit either. The unit tests compare distributed operators with the
serial one bit for bit (`tests/unit/test_remap.cpp`, `test_remap3d.cpp`,
`test_remap_sphere.cpp`).

The changelog records that, between uniform grids, the two searches take about the same time,
since clipping dominates.

## Memory spaces

The overlaps are clipped by Kokkos on the host (a device mesh is mirrored to the host for
this). The operator's arrays then live in `MemSpace`, where `apply()` runs; values cross
ranks through host memory.

## Errors

The constructor is collective and throws `std::invalid_argument` on every rank, so no rank is
left waiting, if any rank's input is bad:

- the meshes are not both of dimension 2 or both of dimension 3, are edited (changed by
  `MeshEdit::apply` and not compressed since), or have high-order nodes;
- a cell is not a triangle, quad or polygon of at most 64 corners (2D), or a tet, hex, prism
  or pyramid (3D);
- a cell is thinner than the tolerance, crosses itself, or is off a plane $z = \text{const}$
  or off the sphere (2D), or is not star-shaped about its centroid (3D);
- the merge tolerance is not finite and positive, the radius is negative or not finite, or
  either differs between ranks;
- the search, surface or sphere clipping differs between ranks, or the search is
  `SpatialSearch::arborx` in a SOMA built without `SOMA_ENABLE_ARBORX`;
- with `SphereClip::gnomonic`, two cells reach past their cube face's horizon (use
  `SphereClip::great_circle`).

The message names whether the source or the target mesh holds a bad cell.

## Limits

- First order: the remapped field is piecewise constant on the target cells.
- Cell fields only, one value per owned cell.
- Mass is conserved only where each mesh covers the other; a target cell partly outside the
  source gets weights summing to less than one, and one that meets no source cell gets 0.
