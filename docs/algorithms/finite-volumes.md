# Finite volumes

`soma::FvSpace` advects a cell-centred scalar by first-order upwind finite volumes and forward
Euler in time, on meshes of triangles, quads, tets or hexes, on any number of ranks. It solves

$$
\frac{\partial u}{\partial t} + \nabla \cdot (\mathbf{a} u) = 0
$$

for a given velocity field $\mathbf{a}(x, y, z)$. The steps are the same, bit for bit, on any
number of ranks and threads.

## The scheme

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

### Why the result does not depend on the partition

A face's geometry is computed from its corners in an order fixed by their gids (from the least
corner towards its lesser neighbour), not in the order a rank happens to hold them, and each
owned cell sums its faces' fluxes in its own side order. Every rank holding a face therefore
computes the same bits, and the update is the serial one on any partition.

## API

Header: `soma/algorithms/fv.hpp`.

```cpp
template <class MemSpace>
class FvSpace {
 public:
  FvSpace(const Comm& comm, const Mesh<MemSpace>& mesh);
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
| `step(comm, phi, u, dt, inflow)` | Collective. One forward Euler step of the owned cells, then one halo update of `u`. |
| `update(comm, u)` | Collective. Each owner's value overwrites its ghosts'. |
| `total(comm, u)` | Collective. $\sum_c V_c u_c$ over owned cells, the conserved quantity; exact (rounded once) when `reproducible()`. One allreduce. |

`step` reads `u` on ghost cells, so on entry the ghosts must hold their owners' values: fill all
`cols()` entries, or call `update()` first. `step` leaves them updated for the next step. The
space's flux scratch is shared by copies of the space, so make one step at a time.

## Usage

From `examples/advect.cpp`, which carries a Gaussian bump around the centre of the mesh's
bounding box by a rigid rotation in the $xy$ plane:

```cpp
// A rigid rotation about (cx, cy), one turn in unit time.
struct Rotation {
  double cx, cy;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double x, double y, double) const {
    constexpr double omega = 2 * 3.14159265358979323846;
    return {-omega * (y - cy), omega * (x - cx), 0};
  }
};

soma::HostMesh mesh = soma::read_file(comm, in);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
const soma::FvSpace<Kokkos::HostSpace> space(comm, mesh);
const auto phi = space.normal_flux(a);  // a: a Rotation
const soma::View1<const double, Kokkos::HostSpace> fluxes(phi);
auto u = space.vector();
// ... set u on every cell here, ghosts included ...
before = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
dt = cfl * space.max_dt(comm, fluxes);
for (int n = 0; n < steps; ++n) space.step(comm, fluxes, u, dt);
after = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
```

The example prints the totals before and after and writes `u` as a cell field:

```bash
mpiexec -n 4 ./advect --steps 100 --cfl 0.5 in.h5m out.soma
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

- First order in space and time; the scheme is stable for $\Delta t \le$ `max_dt`.
- The space keeps the mesh's geometry as it was when made. If the coordinates move, make a
  new space.
- `inflow` is one value for the whole boundary.

## Memory spaces

The geometry (face areas, centroids, cell volumes) is computed once on the host, from a host
mirror for a device mesh, and copied to `MemSpace`. `normal_flux`, `max_dt`, `step` and
`total` run as Kokkos kernels in `MemSpace`.
