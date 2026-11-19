# Laplacian smoothing

`soma::laplacian_smooth` improves the shape of a mesh's cells by moving each free vertex to the
mean of its neighbours' coordinates. Use it to relax a mesh after moving its boundary,
refining it or reading it from a generator that leaves badly shaped cells. The result is the
same, bit for bit, on any number of ranks and threads.

## What it does

Each sweep is a Jacobi iteration: every free owned vertex moves to the mean of its neighbours'
coordinates as they were before the sweep. The algorithm is the one in MOAB's
`LaplacianSmoother` example.

```{figure} ../figures/img/laplacian-smoothing.png
:alt: Two 12 by 12 grids of quads coloured by cell area: on the left with the inner vertices moved at random and areas from about half to one and a half cells, on the right after smoothing, a regular grid of equal cells
:width: 90%

A 12 by 12 grid of unit-square quads with its inner vertices moved by up to 0.3 of a cell (left), and after 10
sweeps of `laplacian_smooth` (right); the skin stays. Each cell is coloured by its area over the unmoved cell's:
from 0.46 to 1.61 before, from 0.99 to 1.01 after.
```

The neighbours of a vertex are the other corners of the entities of dimension `bridge_dim`
around it:

- by default the cells, so an interior vertex of a hex mesh has 26 neighbours;
- with `bridge_dim = 1`, the edges, so only the vertices joined to it by an edge.

Neighbours are read off the cells' canonical sides (see [topology](../concepts/topology.md)),
so the mesh need not hold edges or faces. Each vertex sums its neighbours in ascending
gid order, which is what makes the result independent of the partition and thread count.

## API

Header: `soma/algorithms/laplacian.hpp`.

```cpp
enum class SmoothMethod : std::uint8_t { neighbours, centroids };

struct SmoothOptions {
  int iterations = 1;
  int bridge_dim = -1;
  Real tolerance = 0;
  std::span<const std::uint8_t> fixed;
  SmoothMethod method = SmoothMethod::neighbours;
  std::optional<Real> sphere_radius;
};

struct SmoothReport {
  int iterations = 0;     // sweeps made
  Real largest_move = 0;  // in the last sweep, over every rank; 0 unless tolerance > 0
};

template <class MemSpace>
SmoothReport laplacian_smooth(const Comm& comm, Mesh<MemSpace>& mesh, const SmoothOptions& options = {});
```

| Option | Default | Meaning |
| --- | --- | --- |
| `iterations` | `1` | Number of sweeps to make, at most. |
| `bridge_dim` | `-1` | Dimension of the entities that make two vertices neighbours: `-1` for the mesh's dimension (the cells), otherwise 1 to `dim`. |
| `tolerance` | `0` | If greater than 0, stop once no vertex moved farther than this in a sweep. A length in the mesh's units. |
| `fixed` | empty | One entry per vertex (by lid); a nonzero entry keeps that vertex where it is. Empty: the skin stays fixed. |
| `method` | `neighbours` | Where a free vertex moves: the mean of its neighbours' coordinates, or (`centroids`, meshes of dimension 2) the mean of its cells' centroids. See [methods](#methods). |
| `sphere_radius` | none | Project every moved vertex onto the sphere of this radius about the origin after each sweep. See [on a sphere](#on-a-sphere). |

`SmoothReport::iterations` is the number of sweeps made; it is less than `options.iterations`
when the tolerance stopped the iteration early. `largest_move` is the largest distance a
vertex moved in the last sweep, over every rank, and is computed only when `tolerance > 0`.

## Methods

- **`neighbours`** (the default): each free vertex moves to the mean of its neighbours'
  coordinates, as above.
- **`centroids`** (meshes of dimension 2): each free vertex moves to the mean of the centroids
  (corners' averages) of the cells holding it, the cells summed in ascending gid order, so the
  result is again the same on any number of ranks and threads. A uniform grid stays as it is.

On a refined mesh both raise the corner angles of the cells, but they differ in what else they
change. On `refine_quads`' cubed sphere (ne 4, refined about a face's middle), 20 sweeps of each:

| | none | `neighbours` | `centroids` |
| --- | --- | --- | --- |
| mean of each cell's least angle | 75.0° | 78.7° | 79.1° |
| cells with a least angle under 45° | 28 | 12 | 12 |
| worst ratio of longest to shortest edge | 3.3 | 8.2 | 5.1 |
| change in the refined cells' mean area | | +3.5% | +4.5% |

`centroids` stretches cells less and keeps sizes about as well. Weighting the centroids by
their cells' areas (Lloyd's iteration at a uniform density) would instead even the sizes out:
in the same test the refined cells grew by 87%, undoing the refinement.

Neither method raises the single least angle of a refined mesh, and both can lower it: by about
1 degree on ne30 refined to 12 km over the United States, by 6 to 7 degrees on the
two-region run of the [regional refinement example](refinement.md#example-regionally-refined-meshes-of-the-sphere)
(31.7 degrees to 26.1 with `neighbours`, 25.1 with `centroids`). The worst cells are transition
cells, whose corners their templates set; smoothing improves the many cells around them, not
those. [Optimization](#optimization) does.

## On a sphere

A closed surface has no skin, so every vertex moves, and the mean of points on a sphere lies
inside it: without projection each sweep shrinks a sphere. With `sphere_radius = R`, every
moved vertex is set to $R\,x/|x|$ after each sweep. The vertices that do not move stay where
they are.

## Optimization

`quality_smooth` (`soma/algorithms/quality_smooth.hpp`, host meshes of triangles and quads)
moves each free vertex to improve the cells around it rather than to an average:

```cpp
soma::QualitySmoothOptions o;
o.sphere_radius = 6.37122e6;  // move on the sphere
const soma::QualitySmoothReport r = soma::quality_smooth(comm, mesh, o);
```

| Option | Default | Meaning |
| --- | --- | --- |
| `shape_sweeps` | 10 | Sweeps lowering the sum of the cells' corner condition numbers. |
| `angle_sweeps` | 5 | Sweeps raising the least corner angle of the cells below `angle_threshold`. |
| `polish_sweeps` | 10 | Shape sweeps again, kept above the least angle the angle sweeps reached. |
| `angle_threshold` | 60 | Degrees in (0, 90]. |
| `tolerance` | 0 | > 0: a phase ends once no vertex moves farther than this times its shortest edge. |
| `fixed` | empty | As for `laplacian_smooth`: the skin stays when empty. |
| `sphere_radius` | none | Move in the sphere's tangent plane and project back onto it. |

- **Shape:** each corner's condition number $\kappa = |A|_F^2 / (2 \det A)$, $A$ its two edges
  (in the vertex's tangent plane) against the ideal corner, a square's or an equilateral
  triangle's (Knupp). $\kappa$ is 1 for the ideal corner, does not depend on the cell's size,
  and grows without bound as the corner degenerates; steps that make any $\det A \le 0$ are
  refused.
- **Angle:** the steepest ascent of the least angle over the active angles (Freitag and
  Plassmann). Angles alone bound neither a quad's aspect nor its largest angle (a step can make
  a long thin rectangle, or a corner near 180 degrees), so a step must also leave no corner of
  the patch with a condition number above the patch's worst.
- **Never worse:** every step is a line search kept only if its objective improves, and the
  least angle of the mesh never falls during a phase.
- **Order:** vertices are coloured so that no two of one colour share a cell (Jones and
  Plassmann, priorities hashed from gids); a sweep moves one colour at a time with a halo update
  after each, and sums each patch in cell-gid order. The result is the same bit for bit on any
  number of ranks and threads.

On `refine_quads`' cubed sphere (ne 4, refined about a face's middle), against 20 sweeps of
`centroids`:

| | `centroids` | `quality_smooth` |
| --- | --- | --- |
| least angle | 33.1° | 40.1° |
| mean of each cell's least angle | 79.1° | 79.4° |
| worst ratio of longest to shortest edge | 5.1 | 3.3 |
| change in the refined cells' mean spacing | +2.0% | +1.6% |

On ne30 refined to 12 km over the United States (the
[example](refinement.md#example-regionally-refined-meshes-of-the-sphere)), from the centroid
result, the least angle goes from 30.1 to 43.3 degrees and the worst edge ratio from 4.8 to 3.2,
at about 15 s on one core for 24436 cells. Its condition number does not see sizes, so the
largest area ratio between neighbours grows (5.6 to 7.2 there).

It throws `std::invalid_argument` on every rank, as `laplacian_smooth` does, for a mesh not of
dimension 2, a cell neither a triangle nor a quad, an inverted corner, an edited mesh, a missing
ghost layer on more than one rank, negative sweeps, a threshold outside (0, 90], a bad tolerance,
mask or radius, or ranks differing in their options.

## Usage

From `examples/laplacian_smooth.cpp`, which reads a mesh on any number of ranks, smooths it
and writes it back:

```cpp
soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
soma::SmoothOptions options;
options.iterations = 10;
const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, options);
mesh.remove_ghosts(comm);
soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{}});
```

The example takes `--iterations K`, `--bridge-dim B` and `--tolerance T`:

```bash
mpiexec -n 4 ./laplacian_smooth --iterations 20 --tolerance 1e-8 in.h5m out.soma
```

## Boundary handling

Fixed vertices do not move. They are:

- the vertices whose entry in `options.fixed` is nonzero, if a mask is given;
- otherwise the skin: the corners of the sides of dimension `dim - 1` that only one cell has.

A vertex with no neighbours also stays where it is. To keep a different set of vertices fixed
(for example, to let boundary vertices slide), pass a mask with one entry per vertex of the
local part, ghosts included.

## Parallel behaviour

`laplacian_smooth` is collective: every rank must call it, with the same options.

On more than one rank the mesh must hold a ghost layer through vertices,
`GhostOptions{1, 0}` (see [ghosts and halos](../parallel/ghosts-and-halos.md)). Every cell
around a vertex the rank holds is then also held, so all of a vertex's neighbours are present
and the skin found on each rank is the global one. Owners move their vertices; after every
sweep a halo update gives shared copies and ghosts their owners' new coordinates. On return,
copies and ghosts therefore hold their owners' coordinates, which is what
[`FemSpace`](fem.md) expects of a mesh whose coordinates have moved.

Because each vertex sums its neighbours (or its cells) in gid order and only owners compute, the coordinates
after smoothing are the same, bit for bit, on any number of ranks and threads. This holds
whether or not `soma::reproducible()` is on (see [reproducible sums](../parallel/reproducible-sums.md)).

## Memory spaces

`laplacian_smooth` accepts a mesh in any memory space. The neighbour graph, the cells of each
vertex (for `centroids`) and the skin are built on the host (a device mesh is mirrored to the host for this), once per call; the sweeps
then run as Kokkos kernels in the mesh's memory space.

## Errors

`laplacian_smooth` throws `std::invalid_argument` on every rank if any rank passes:

- a negative iteration count;
- a `bridge_dim` other than `-1` or a value in `[1, dim]`;
- a negative or non-finite tolerance;
- a `fixed` mask whose length is not the vertex count;
- a mesh of dimension 0, or an edited mesh (changed by `MeshEdit::apply` and not compressed
  since with `compress()`; see [editing](../concepts/editing.md));
- on more than one rank, a mesh without a ghost layer through vertices;
- `centroids` on a mesh not of dimension 2;
- a sphere radius that is not positive and finite;

or if ranks differ in `iterations`, `bridge_dim`, `tolerance`, `method` or the sphere radius
(or in giving one). All ranks take part in the
check (one reduction), so no rank is left waiting.
