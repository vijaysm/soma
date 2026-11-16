# Finite volumes

`soma::FvSpace` advects a cell-centred scalar by finite volumes, explicit in time, on meshes of
triangles, quads, tets or hexes, on any number of ranks, by one of two schemes:

- `FvScheme::upwind` (the default): first-order upwind and forward Euler;
- `FvScheme::bounded`: second order where the solution is smooth, with no new extrema (to rounding):
  flux-corrected transport over a linear reconstruction, in two stages.

It solves

$$
\frac{\partial u}{\partial t} + \nabla \cdot (\mathbf{a} u) = 0
$$

for a given velocity field $\mathbf{a}(x, y, z)$. The steps are the same, bit for bit, on any
number of ranks and threads.

## The upwind scheme

Each face $f$ of an owned cell carries:

- its centroid $\mathbf{x}_f$ and area vector $\mathbf{A}_f$ (in 2D an edge's normal times
  its length; in 3D, for a quad face, half the cross product of its diagonals, exact for a
  bilinear face);
- the normal flux $\phi_f = \mathbf{a}(\mathbf{x}_f) \cdot \mathbf{A}_f$, from its left cell
  to its right;
- the flux $\phi_f u$ taken from its upwind side: the cell it leaves, or the value `inflow`
  where $\mathbf{a}$ enters the domain through the boundary.

A step updates each owned cell $c$ by

$$
u_c \leftarrow u_c - \frac{\Delta t}{V_c} \sum_{f \in c} s_{cf} \, \phi_f u_{\text{upwind}(f)},
$$

with $s_{cf} = +1$ where $c$ is the face's left cell (the one $\mathbf{A}_f$ points away from)
and $-1$ where it is the right. A face's flux leaves one cell and
enters the other with opposite signs, so the total $\sum_c V_c u_c$ changes only through the
boundary.

A cell's volume is $\frac{1}{d} \sum_f \mathbf{A}_f \cdot (\mathbf{x}_f - \mathbf{x}_0)$ over
its outward faces, $\mathbf{x}_0$ its first corner. A cell whose sides turn inwards (a
clockwise polygon) is taken the other way round, so cells may be listed in either
orientation.

## The bounded scheme

Upwinding is monotone, but its error is first order: it spreads a bump and lowers its peak at
every step (the figure below). The bounded scheme keeps upwind's two good properties, exact
conservation and no new extrema, and is second order where $u$ is smooth. It is flux-corrected
transport (Boris and Book) with Zalesak's limiter. One stage from $u$:

1. The upwind result $u^L$, as above. It has no new extrema.
2. Each owned cell's gradient $\mathbf{g}_c$, by least squares over its face neighbours:
   $\mathbf{g}_c$ minimises $\sum_n (u_n - u_c - \mathbf{g}_c \cdot (\mathbf{x}_n - \mathbf{x}_c))^2$
   with $\mathbf{x}$ the cells' centroids (their corners' means). The matrix is inverted once, when
   the space is made. A cell whose neighbours do not span the space (a corner cell with one
   neighbour) gets $\mathbf{g}_c = 0$.
3. Each face's second-order flux $\phi_f \bigl(u_c + \mathbf{g}_c \cdot (\mathbf{x}_f - \mathbf{x}_c)\bigr)$,
   with $c$ the upwind cell, and the correction
   $A_f = \Delta t \, (\text{second-order flux} - \text{upwind flux})$. At an inflow boundary
   there is none.
4. Zalesak's limiter. A cell's bounds are the least and greatest of $u$ and $u^L$ over itself and
   its face neighbours. $R^+_c$ is the fraction of its incoming corrections the cell can take
   and stay below its upper bound; $R^-_c$, of its outgoing ones and stay above its lower bound.
   Each face takes $\alpha_f A_f$, with $\alpha_f$ the lesser of what the cell it leaves allows
   ($R^-$) and what the cell it enters allows ($R^+$).
5. $u_c \leftarrow u^L_c - \frac{1}{V_c} \sum_{f \in c} s_{cf} \, \alpha_f A_f$.

A step is two stages averaged with the start (Heun's method, the strong-stability-preserving
Runge-Kutta method of second order): $u^{(1)} = S(u)$, $u^{(2)} = S(u^{(1)})$,
$u \leftarrow (u + u^{(2)}) / 2$.

What it guarantees, with $\Delta t \le$ `max_dt`:

- **Conservation.** Each limited correction leaves one cell and enters the other, as the upwind
  flux does, so the total changes only through the boundary.
- **No new extrema**, where each cell's fluxes $\sum_f s_{cf} \phi_f$ sum to zero (a
  divergence-free flow; exactly so for a linear $\mathbf{a}$ on cells with planar sides). Each
  stage keeps every cell within its bounds, and the average of two such stages with the start
  stays within the range of the start (and of `inflow`), to rounding: a value at an extremum may
  pass it by a few units in the last place. Where the discrete divergence is not
  zero, upwinding itself can leave the range, and so can this scheme, by as much.
- **Second order** where $u$ is smooth. At a smooth peak the limiter takes over and the scheme
  is first order there: it clips the peak a little, as any scheme that makes no new extrema
  must. Strict TVD is not the aim: in more than one dimension, a TVD scheme is at most first
  order (Goodman and LeVeque, 1985).

A stage exchanges ghosts three times: the gradients with $u^L$, the limiter's $R^\pm$, and the
new $u$. A step does six exchanges, and about three times the arithmetic of an upwind step. The
space holds its scratch (the gradients, $u^L$, $R^\pm$, the corrections and the two stages)
only when made bounded.

### Why the result does not depend on the partition

A face's geometry is computed from its corners in an order fixed by their gids (from the least
corner towards its lesser neighbour), not in the order a rank happens to hold them, and each
owned cell sums its faces' fluxes in its own side order. Every rank holding a face therefore
computes the same bits, and the update is the serial one on any partition. The bounded scheme's
centroids sum the corners in gid order, its gradients and limiter sum over each cell's sides in
order, and a face shared by two ranks takes the owners' $R^\pm$ on both, so it too is the same bit
for bit on any number of ranks and threads.

## API

Header: `soma/algorithms/fv.hpp`.

```cpp
template <class MemSpace>
class FvSpace {
 public:
  FvSpace(const Comm& comm, const Mesh<MemSpace>& mesh, FvScheme scheme = FvScheme::upwind);
  FvScheme scheme() const noexcept;
  LO rows() const noexcept;    // owned cells
  LO cols() const noexcept;    // all cells here
  LO faces() const noexcept;   // the faces of the owned cells
  bool reproducible() const noexcept;
  View1<const Real, MemSpace> volume() const;  // per owned cell (area in 2D)
  View1<Real, MemSpace> vector(const char* label = "soma::FvSpace::vector") const;
  template <class F> View1<Real, MemSpace> normal_flux(const F& a) const;
  Real max_dt(const Comm& comm, const View1<const Real, MemSpace>& phi) const;
  void step(const Comm& comm, const View1<const Real, MemSpace>& phi, const View1<Real, MemSpace>& u, Real dt,
            Real inflow = 0) const;
  void update(const Comm& comm, const View1<Real, MemSpace>& u) const;
  Real total(const Comm& comm, const View1<const Real, MemSpace>& u) const;
};
```

| Member | Behaviour |
| --- | --- |
| `vector()` | A vector of zeros, one entry per cell here (`cols()`), owned cells first. |
| `normal_flux(a)` | $\phi_f$ per face (`faces()` entries). `a(x, y, z)` returns a `Kokkos::Array<Real, 3>` and must be device-callable. |
| `max_dt(comm, phi)` | Collective. The largest stable step: the minimum over owned cells of the volume over the outflow $\sum_f \max(0, \pm\phi_f)$; infinite where nothing flows out anywhere. One allreduce. |
| `step(comm, phi, u, dt, inflow)` | Collective. Upwind: one forward Euler step of the owned cells, then one halo update of `u`. Bounded: two stages averaged with `u`, six halo updates. |
| `update(comm, u)` | Collective. Each owner's value overwrites its ghosts'. |
| `total(comm, u)` | Collective. $\sum_c V_c u_c$ over owned cells, the conserved quantity; exact (rounded once) when `reproducible()`. One allreduce. |

`step` reads `u` on ghost cells, so on entry the ghosts must hold their owners' values: fill all
`cols()` entries, or call `update()` first. `step` leaves them updated for the next step. The
space's scratch is shared by copies of the space, so make one step at a time.

`enum class FvScheme { upwind, bounded };` is in `soma/algorithms/fv.hpp` too.

## Usage

From `examples/advect.cpp`, which carries a Gaussian bump about the vertical axis through the
centre of the mesh's bounding box. Its flow turns once in unit time out to two thirds of the
box's half-width $h$, slows smoothly to rest at $0.9 h$, and is still beyond, so nothing crosses
the walls and the total stays to rounding:

```cpp
// A rotation about (cx, cy), one turn in unit time out to radius r1, slowing
// smoothly (a smoothstep in the radius) to rest at r2 and still beyond; with
// r1 and r2 infinite, a rigid rotation.
struct Rotation {
  double cx, cy, r1, r2;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double x, double y, double) const {
    constexpr double omega = 2 * 3.14159265358979323846;
    const double dx = x - cx, dy = y - cy, r = Kokkos::sqrt(dx * dx + dy * dy);
    double s = 1;
    if (r >= r2) {
      s = 0;
    } else if (r > r1) {
      const double t = (r2 - r) / (r2 - r1);
      s = t * t * (3 - 2 * t);
    }
    return {-omega * s * dy, omega * s * dx, 0};
  }
};

soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
const soma::FvSpace<Kokkos::HostSpace> space(comm, mesh, scheme);  // soma::FvScheme::bounded by default
const auto phi = space.normal_flux(a);  // a: a Rotation
const soma::View1<const double, Kokkos::HostSpace> fluxes(phi);
auto u = space.vector();
// ... set u on every cell here, ghosts included ...
before = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
dt = cfl * space.max_dt(comm, fluxes);
for (int n = 0; n < steps; ++n) space.step(comm, fluxes, u, dt);
after = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
```

The example prints the totals before and after and the range of $u$ after, and writes `u` as a
cell field:

```bash
mpiexec -n 4 ./advect --steps 100 --cfl 0.5 --scheme bounded in.h5m out.soma
```

```{figure} ../figures/img/advected-bump.png
:alt: Three 24 by 24 grids of cells coloured by u: a compact bright bump right of the centre; after half a turn by upwinding, a wide faint bump left of the centre; after half a turn by the bounded scheme, a compact bump of about two thirds the height there
:width: 100%

The layer of cells under the bump's centre, on `data/64bricks_1khex.h5m` refined once (24 by 24 by 24
hexes), in the example's flow at CFL 0.5, on one colour scale: the start (left), and after half a turn
upwind (middle) and bounded (right). The peak falls from 0.83 to 0.21 upwind and to 0.54 bounded; both
keep the total to rounding, and the bounded result stays within the start's range (to rounding). The
flow is divergence-free in its rigid core but, sampled at face centroids, not exactly so where it slows,
so the guarantee of no new extrema covers the core only; the figure's data set checks the range over all
cells.
```

## Requirements and errors

The constructor is collective. It throws `std::invalid_argument` on every rank if any rank's
mesh:

- is not of dimension 2 or 3, is edited (changed by `MeshEdit::apply` and not compressed
  since), has high-order nodes, or holds a cell other than a tri, quad, tet or hex;
- holds a side that more than two cells share;
- holds a cell of no positive volume, or one turned against a neighbour (folded over it);
- on more than one rank, has no ghost layer (through any bridge dimension; see
  [ghosts and halos](../parallel/ghosts-and-halos.md)), which is needed so that every face of
  an owned cell is present;

or if ranks differ in `soma::reproducible()`, which makes `total()` exact.

Vectors must hold exactly `cols()` entries and flux views exactly `faces()`. A rank whose
vectors are of the wrong size throws `std::invalid_argument`; in the collective calls that
rank leaves the exchange and the others complete it.

## Limits

- Upwind is first order in space and time, bounded second order away from extrema; both are
  stable, and bounded makes no new extrema (to rounding), for $\Delta t \le$ `max_dt`.
- The bounded scheme's guarantee of no new extrema holds for flows with no discrete divergence;
  its gradients use face neighbours only, so a cell with fewer than $d$ independent ones (a
  corner cell) is first order.
- The space keeps the mesh's geometry as it was when made. If the coordinates move, make a
  new space.
- `inflow` is one value for the whole boundary.

## Memory spaces

The geometry (face areas, centroids, cell volumes) is computed once on the host, from a host
mirror for a device mesh, and copied to `MemSpace`, with the bounded scheme's centroids and
least-squares matrices. `normal_flux`, `max_dt`, `step` and `total` run as Kokkos kernels in
`MemSpace`.
