# Finite elements

SOMA provides linear finite elements on unstructured meshes: element kernels for single cells
(`soma/algorithms/fem.hpp`) and a space of one unknown per vertex that assembles a stiffness
matrix and load vectors in parallel (`soma/algorithms/fem_space.hpp`). Use them to solve
scalar elliptic problems such as Poisson's equation, with the solvers described in
[matrices and solvers](linear-algebra.md).

## Elements

The elements are P1 on triangles and tets and Q1 on quads and hexes, with corners in MOAB's
canonical order. The reference cells are the unit triangle $(0,0), (1,0), (0,1)$ and tet
$(0,0,0), (1,0,0), (0,1,0), (0,0,1)$, and $[-1, 1]^d$ for quads and hexes. Cells of
dimension 2 must lie in a plane $z = \text{const}$. No other topology, and no high-order
nodes, are supported.

Everything in `fem.hpp` is in namespace `soma::fem`, device-callable and allocation-free:

| Function | What it gives |
| --- | --- |
| `quadrature(Topology t, int degree)` | The smallest rule exact for polynomials of `degree` on the reference cell: Gauss-Legendre products on quads and hexes (up to degree 5), rules up to degree 4 on triangles and tets. |
| `shape(t, xi, n, dn)` | Shape function values and reference gradients at `xi`; the node count. |
| `cell_of(view, dim, e)` | A `fem::Cell`: topology, dimension, node count and corner coordinates of cell `e` of a `MeshView`. |
| `at_point(c, xi, n, g)` | Shape values, physical gradients and the Jacobian determinant at `xi`. |
| `least_jacobian(c)` | The smallest Jacobian determinant over the corners and Gauss points. |
| `stiffness(c, k)` | $k_{ab} = \int \nabla N_a \cdot \nabla N_b$. |
| `mass(c, m)` | $m_{ab} = \int N_a N_b$, exact on affine cells. |
| `load(c, f, r)` | $r_a = \int f N_a$ by the degree-2 rule; `f(x, y, z)` device-callable. |
| `squared_error(c, u, exact)` | $\int (\sum_a u_a N_a - u_\text{exact})^2$ by a degree-4 rule (5 on quads and hexes). |

The stiffness matrix is exact for simplices and for parallelogram-shaped quads and hexes.

```cpp
fem::Cell c = fem::cell_of(view, dim, e);  // view = mesh.view()
Real k[fem::max_nodes][fem::max_nodes];
const Real det = fem::stiffness(c, k);  // !(det > 0): inverted, flat or unsupported
```

### Cell validity

Cells must be positively oriented. `least_jacobian(c)` is positive for a usable cell. It is NaN
for a cell that cannot be used at all: an unsupported topology, a coordinate that is not
finite, a determinant that overflows, or a 2D cell with corners off one plane
$z = \text{const}$. For quads and hexes it checks the corners as well as the Gauss points,
since a folded cell inverts first at a corner.

`stiffness`, `mass` and `load` return `least_jacobian(c)`. Unless it is greater than 0, the
element matrix or vector is zero and must not be used: test the return value and fail on it.
`squared_error` returns NaN for such a cell, so a sum over cells shows the bad one.

## The space: `FemSpace`

Header: `soma/algorithms/fem_space.hpp`.

```cpp
enum class Assembly : std::uint8_t { gather, atomic, colored };

template <class MemSpace>
class FemSpace {
 public:
  FemSpace(const Comm& comm, const Mesh<MemSpace>& mesh, Assembly assembly = Assembly::gather);
  LO rows() const noexcept;               // owned vertices
  LO cols() const noexcept;               // all vertices here
  Assembly assembly() const noexcept;     // as laplacian() assembles
  bool reproducible() const noexcept;     // soma::reproducible() when made
  View1<Real, MemSpace> vector(const char* label = "soma::FemSpace::vector") const;
  CsrMatrix<MemSpace> pattern() const;
  CsrMatrix<MemSpace> laplacian(const Comm& comm) const;
  template <class F> View1<Real, MemSpace> load(const Comm& comm, const F& f) const;
  View1<const std::uint8_t, MemSpace> boundary() const;
  void update(const Comm& comm, const View1<Real, MemSpace>& x) const;
  void multiply(const Comm& comm, const CsrMatrix<MemSpace>& a, const View1<Real, MemSpace>& x,
                const View1<Real, MemSpace>& y) const;
  Numbering<MemSpace> numbering(const Comm& comm) const;
  Real dot(const Comm& comm, const View1<const Real, MemSpace>& x, const View1<const Real, MemSpace>& y) const;
};
```

A space has one unknown per vertex. A rank's matrix rows are its owned vertices
`[0, rows())`; its columns are all the vertices it holds, `[0, cols())`: owned, then shared
copies and ghosts (see [the mesh model](../concepts/mesh-model.md)). A vector holds one entry
per vertex here; after `update()` the copies and ghosts hold their owners' values. Matrices
are [`CsrMatrix`](linear-algebra.md) in the same layout.

| Member | Behaviour |
| --- | --- |
| `vector()` | A vector of zeros, `cols()` entries. |
| `pattern()` | The sparsity of the owned rows (each owned vertex with the vertices that share a cell with it), values zero. |
| `laplacian(comm)` | Collective. The stiffness matrix of $-\nabla \cdot \nabla$, without boundary conditions. |
| `load(comm, f)` | Collective. $b_v = \int f N_v$ for the owned vertices, 0 for the others. `f(x, y, z)` must be device-callable. |
| `boundary()` | Per vertex, 1 on the mesh's boundary (a corner of a side of dimension `dim - 1` that only one cell has; with hanging vertices, one that no other such side lies in or has in it), as its owner finds it. Never a hanging vertex. |
| `constrained()` | Whether the mesh has hanging vertices (see [hanging vertices](#hanging-vertices)). |
| `hanging()` | Per vertex, 1 where it hangs; empty when not constrained. |
| `distribute(comm, x)` | Collective. Each hanging vertex's value from its masters', then each owner's entries to its copies and ghosts (two updates); nothing when not constrained. Call it on a solution. |
| `update(comm, x)` | Collective. Each owner's entries of `x` overwrite its copies and ghosts. |
| `multiply(comm, a, x, y)` | Collective. Updates `x`, then $y = A x$ over the owned rows. |
| `numbering(comm)` | Collective. A global numbering: owned vertices numbered rank by rank, copies and ghosts their owners' numbers (one exscan, one allreduce, one update). |
| `dot(comm, x, y)` | Collective. $\sum x_i y_i$ over every rank's owned vertices; one allreduce. |

### Requirements on the mesh

- Dimension 2 or 3, cells all triangles, quads, tets or hexes, no high-order nodes, not edited
  (changed by `MeshEdit::apply` and not compressed since).
- Every cell positively oriented (`least_jacobian > 0`).
- No hex face against the two triangles of two tets: the space would not be continuous there (a
  linear solution is no longer reproduced exactly).
- Hanging vertices, if any, with a record of the side each was born on (`"refine:birth"`, which
  `refine_marked` writes and `infer_births` recovers); see [hanging vertices](#hanging-vertices).
- On more than one rank, a ghost layer through vertices, `GhostOptions{1, 0}` (see
  [ghosts and halos](../parallel/ghosts-and-halos.md)). Every cell around an owned vertex is
  then present, and owners assemble their rows whole, without messages. The mesh must also
  hold its sharing fields, as a mirror does with `TransferOptions::everything()`.
- Every owned vertex belongs to some cell on its rank.

The constructor is collective and throws `std::invalid_argument` on every rank if any rank's
mesh breaks one of these, if a rank's element matrices would not fit `Offset`, if ranks
differ in `soma::reproducible()`, or, with hanging vertices on more than one rank, if a rank
lacks a master of a hanging vertex it needs or a cell around one (add a ghost layer: two
suffice).

A space keeps views into the mesh: it is valid while the mesh is alive and its topology
unchanged. Coordinates may move between assemblies, provided copies and ghosts hold their
owners' coordinates ([`laplacian_smooth`](smoothing.md) leaves them so). `laplacian()` and
`load()` throw `std::invalid_argument` on every rank, after assembling, if a cell they
assemble has become inverted or degenerate since the space was made.

`update()` throws `std::invalid_argument` on a rank whose vector does not hold exactly
`cols()` entries; that rank leaves the exchange and the others complete it.

## Hanging vertices

`refine_marked` without `MarkedOptions::conforming` leaves vertices on edges and faces that a
neighbouring cell still has whole. A space of independent values there would not be continuous.
`FemSpace` constrains them instead, as `hanging_constraints` (`soma/algorithms/refine.hpp`)
gives them: a vertex born on a side that a cell still has whole takes the average of the side's
corners,

$$
u_h = \sum_k w_k\, u_{m_k}, \qquad w_k = 1/n \text{ over the side's } n \text{ corners},
$$

a corner that hangs itself (one level up) taking its own masters in turn. These are the traces
of the coarse cell's linear, bilinear or trilinear element at the edge midpoint or face centre,
so the constrained space is continuous: a linear function is reproduced exactly, and the error
falls at the uniform meshes' rate.

With $P$ the map from the free vertices to all of them, the space assembles $P^T A P$ and
$P^T b$ on the free vertices' rows. A hanging vertex's row is the identity's, its load 0, so the
matrix stays symmetric (to rounding: $P^T A P$ sums the same products in other orders) and
positive definite after Dirichlet elimination, and CG solves it as before. Hanging vertices are
never boundary vertices: Dirichlet values reach them through their masters. After solving, call
`distribute(comm, x)` to set them:

```cpp
const soma::FemSpace<Kokkos::HostSpace> space(comm, mesh);  // mesh from refine_marked
soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(comm);
auto b = space.load(comm, f);
soma::apply_dirichlet(a, b, space.boundary(), g);
soma::cg(comm, space, a, b, x);
space.distribute(comm, x);  // hanging vertices from their masters
```

A constrained space assembles by gather, whatever `Assembly` it was given: each owned row sums
its terms (its own cells', then each hanging vertex it masters, by gid, weighted, a hanging
corner spread to its masters) in an order fixed by gid, so the matrix and load are the same bit
for bit on any number of ranks and threads.

Its row for a master needs the cells around the hanging vertices it masters, which lie one layer
further out than its own: on more than one rank give the mesh two ghost layers through vertices,
`GhostOptions{2, 0}`. The constructor checks it: each needed hanging vertex's cell count against
its owner's, and an exact balance (the sum, over each hanging vertex's owner, of
$w_k\,\kappa(m_k, h)$ over its masters, less the same over each master's row of the hanging
vertices it took, for a number $\kappa$ fixed by the two gids) that is zero only when every
master's row took every hanging vertex. `hanging_constraints` gathers every hanging vertex's
record on every rank (five gids each), to resolve masters that hang.

Measured (`FemSpace.ConvergesWithHangingVertices`): $-\Delta u = f$ with
$u = \prod_i \sin(\pi x_i)$ on boxes refined once where $x < 0.4$; the largest error at a vertex
falls at rates 2.09, 1.96 (tri), 1.93, 1.96 (quad), 2.03, 2.18 (tet), 1.98, 2.02 (hex).

## Assembly modes

`Assembly` selects how `laplacian()` assembles. `load()` always gathers.

| Mode | How | Reproducibility |
| --- | --- | --- |
| `Assembly::gather` (default) | The element matrices are computed once per cell, then each owned row sums its cells' entries, the cells in ascending gid order. | The same, bit for bit, on any number of ranks and threads. |
| `Assembly::atomic` | Each cell with an owned corner computes its element matrix and adds its entries to the owned rows with atomic additions. | Summation order, and so rounding, varies between runs. |
| `Assembly::colored` | Cells that share no vertex form a colour (greedy, cells taken by gid). Colours take turns; the cells of one colour add their entries without atomics. | The same on any number of threads, not of ranks. |

Atomic and coloured assembly store no element matrices; gathered assembly stores one per cell
here. In practice the choice matters little: the changelog records atomic and coloured
assembly taking the same time as gathered to within 2 % on a $64^3$ box of hexes, measured on
one rank.

## Example: Poisson's equation

`examples/poisson.cpp` solves $-\nabla^2 u = 1$ with $u = 0$ on the boundary, on a mesh read
from a file on any number of ranks, by Jacobi-preconditioned conjugate gradients. The core,
lightly abridged:

```cpp
struct One {
  KOKKOS_INLINE_FUNCTION double operator()(double, double, double) const { return 1; }
};

soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
const soma::FemSpace<Kokkos::HostSpace> space(comm, mesh, assembly);
soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(comm);
auto b = space.load(comm, One{});
const auto zero = space.vector();
soma::apply_dirichlet(a, b, space.boundary(), soma::View1<const double, Kokkos::HostSpace>(zero));
auto u = space.vector();
soma::CgOptions options;
options.rtol = 1e-10;
const soma::CgReport report =
    soma::cg(comm, space, a, soma::View1<const double, Kokkos::HostSpace>(b), u, options);
```

The example then copies `u` into a vertex field, removes the ghosts and writes the mesh:

```bash
mpiexec -n 4 ./poisson --assembly gather --rtol 1e-10 in.h5m out.soma
```

```{figure} ../figures/img/poisson-solution.png
:alt: The back half of a box of hexes coloured by the solution u: zero on the outer faces, rising to about 0.9 at the centre of the cut face
:width: 75%

The solution of `poisson` on `data/64bricks_1khex.h5m` (12 by 12 by 12 hexes on $[-0.5, 3.5]^3$), $u = 0$ on the
boundary. The back half of the box, cut along the cell faces through the centre, shows the inside, where $u$
peaks at 0.909.
```

`apply_dirichlet` and `cg` are described in [matrices and solvers](linear-algebra.md).

## Reproducible mode

With `soma::reproducible()` on when a space is made (set by `SOMA_REPRODUCIBLE=1` or
`--soma-reproducible`; see [runtime](../concepts/runtime.md)):

- the space gathers whatever assembly it was asked for (`assembly()` reports
  `Assembly::gather`);
- `dot()` sums exactly and rounds once (see [reproducible sums](../parallel/reproducible-sums.md));
- `cg` sums exactly by default (`CgOptions::reproducible` follows the space).

The matrix, load vector, dot products and CG iterates are then the same, bit for bit, on any
number of ranks and threads, for one build. All ranks must agree on the switch; the
constructor throws on every rank otherwise. `reproducible()` on the space reports the setting
it was made with.

Without it, a gathered matrix and every load vector are already the same on any partition;
the dot products and CG's reductions may then differ in the last bits.

## Memory spaces

`FemSpace<MemSpace>` works on a mesh in any memory space. Its plan (sparsity, cell lists,
colours) is built once on the host, from a host mirror for a device mesh, and copied to
`MemSpace`; assembly, `multiply` and `dot` run as Kokkos kernels there.
