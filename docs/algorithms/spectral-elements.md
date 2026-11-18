# Spectral elements

SOMA provides continuous spectral elements of any order $p$ from 1 to 32 on quads, in the plane or
on a sphere: the discretization of HOMME's dynamical core. A space stores each element's own values
at its $(p + 1)^2$ Gauss-Lobatto-Legendre (GLL) points; direct stiffness summation (DSS) makes them
continuous. The geometry can be linear (Quad4) or quadratic (Quad8, Quad9) whatever $p$ is. Everything
is in namespace `soma::sem`.

| Header | What it gives |
| --- | --- |
| [`soma/sem/gll.hpp`](../reference/api/sem/gll.md) | the 1D GLL weights and derivative matrix |
| [`soma/sem/cubed_sphere.hpp`](../reference/api/sem/cubed_sphere.md) | the equiangular cubed sphere, split across the ranks |
| [`soma/sem/sem_space.hpp`](../reference/api/sem/sem_space.md) | `SemSpace`: points, geometry, `dss`, `mass`, `integral` |
| [`soma/sem/operators.hpp`](../reference/api/sem/operators.md) | gradient, divergence, vorticity, tangent projection, Laplacian, hyperviscosity |

```cpp
soma::sem::CubedSphereSpec spec;
spec.ne = 30;                     // ne30: about 100 km on the Earth
spec.radius = 6.371e6;
soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);
mesh.add_ghosts(comm, soma::GhostOptions{1, 0});  // on more than one rank
soma::build_distributed_adjacencies(comm, mesh, soma::AdjacencyRequest{}.add(2, 1));

const soma::sem::SemSpace<Mem> space(comm, mesh, 3, soma::sem::SemGeometry::sphere(spec.radius));
auto h = mesh.fields().add<double>("h", space.space(), 1).view();  // a value per GLL point
auto lap = mesh.fields().add<double>("lap", space.space(), 1).view();
soma::sem::laplacian(comm, space, h, lap);                         // continuous
const double mass = space.integral(comm, h);                       // the same bits on any P
```

## The cubed sphere

`cubed_sphere_mesh(comm, spec)` builds $6\,n_e^2$ quads: $n_e \times n_e$ on each face of the cube
$[-1, 1]^3$, at equal angles: the face point of angles $(a, b) \in [-\pi/4, \pi/4]^2$ is
$(\tan a, \tan b)$, projected radially onto the sphere of the radius asked for. Element gids are
$1 + f n_e^2 + i + n_e j$ for face $f$ in 0..5, and the corners run counter-clockwise seen from
outside. A vertex's gid comes from its place on the cube's surface lattice, so no gid depends on the
number of ranks. Rank $r$ builds the $r$-th block of consecutive element gids. With
`geometry_order = 2` the elements are Quad9, their mid-edge and centre nodes on the sphere.
`std::invalid_argument` reports $n_e < 1$, a radius that is not positive and finite, or another
order.

## The space

`SemSpace<MemSpace>(comm, mesh, p, geometry)` is collective. Point $k = i + (p + 1)\,j$ of element
$e$ is row $e\,(p + 1)^2 + k$ of `space()`, an `element_nodes` space of GLL nodes, so fields made on
it hold one value per point. $i$ runs along the element's edge from corner 0 to corner 1, $j$ along
the edge from corner 0 to corner 3, at the reference points $(\xi_i, \eta_j)$ of
`soma::gll_points_1d(p)`.

The element map $y(\xi, \eta)$ is bilinear in the corners or, with 8 or 9 nodes per quad,
biquadratic (Quad8 serendipity, Quad9 Lagrange, in MOAB's node order). On a sphere of radius $R$
each point is projected, $x = R\,y / |y|$, and the tangent vectors are derivatives of the projected
map: $a_i = (R / |y|)(I - \hat y \hat y^T)\,\partial y / \partial \xi_i$. Per point the space holds,
in three Cartesian components each:

| View | Columns | Value |
| --- | --- | --- |
| `x()` | 3 | the position |
| `covariant()` | 6 | $a_1 = \partial x / \partial \xi$, $a_2 = \partial x / \partial \eta$ |
| `contravariant()` | 6 | $a^1, a^2$: tangent, with $a^i \cdot a_j = \delta_{ij}$ |
| `jacobian()` | 1 | $J = \lvert a_1 \times a_2 \rvert$ |
| `local_mass()` | 1 | $w_i w_j J$, the element's quadrature weight |
| `mass()` | 1 | the assembled mass: the local masses of every element holding the point |

Every element on the rank has its geometry, ghosts included; the owned ones come first. In the
plane the area is exact; on the sphere it converges spectrally ($4\pi$ to $3 \times 10^{-10}$ at
$n_e = 2$, $p = 8$), and quadratic geometry converges at about $h^4$ (measured 4.0) where linear converges at about
$h^2$ (measured 2.0).

The constructor throws `std::invalid_argument` on every rank, naming the smallest offending element
gid where there is one, when a rank's mesh is not of dimension 2, $p$ is outside [1, 32], an element
is not a quad of 4, 8 or 9 nodes, an element's normal $a_1 \times a_2$ does not point along $+z$ (the
plane) or outward (the sphere) at every point, or, on more than one rank, the mesh has no ghost
layer through vertices or no distributed edges. On one rank the space derives the edges itself
when the mesh has none.

## Direct stiffness summation

`dss(comm, u)` gives each point shared by several elements, on a vertex or an edge,
$\sum_e m_e u_e / \sum_e m_e$, with $m_e$ the elements' local masses there; interior points are
left as they are. `u` has a row per point and any number of components. The result is the same, bit
for bit, on any number of ranks and threads:

1. Ghost elements take their owners' values (a halo update of whole elements), so only owned
   elements' values are read.
2. The owner of each vertex and edge sums $m_e u_e$ over the elements around it, in ascending
   element gid. A ghost layer through vertices puts every such element on the owner.
3. A halo update gives the copies and ghosts of each vertex and edge their owner's sum.
4. Every element's shared points take the sum divided by the assembled mass.

Points along an edge are counted from its vertex of smaller gid, which every rank holding the edge
agrees on. `integral(comm, u, c)` sums $m\,u$ over the owned points in exact arithmetic, so it too is
the same on any number of ranks and threads.

## Operators

The strong-form operators differentiate within each element with the GLL derivative matrix
$D_{ij} = \ell_j'(\xi_i)$, at every element on the rank; their results are continuous after `dss`.
With $f_\xi, f_\eta$ the reference derivatives:

| Operator | Value |
| --- | --- |
| `gradient(s, f, grad)` | $a^1 f_\xi + a^2 f_\eta$ |
| `divergence(s, v, div)` | $\frac{1}{J}\left[\partial_\xi (J\, v \cdot a^1) + \partial_\eta (J\, v \cdot a^2)\right]$ |
| `vorticity(s, v, vort)` | $\frac{1}{J}\left[\partial_\xi (v \cdot a_2) - \partial_\eta (v \cdot a_1)\right]$, the curl along $a_1 \times a_2$ |
| `project_tangent(s, v)` | $v$ less its component along $a_1 \times a_2$ |
| `laplacian(comm, s, f, lap)` | $-\sum w J\, \nabla\phi \cdot \nabla f$ per element, summed by DSS and divided by the assembled mass |
| `hyperviscosity(comm, s, f, out)` | the Laplacian applied twice |

GLL quadrature makes the divergence sum by parts: its integral is the net flux through the
boundary, zero on the sphere to rounding. The Laplacian is the weak form, so on a mesh with a
boundary it imposes a zero normal derivative there. Spherical harmonics are its eigenfunctions to
truncation error: for $Y_3^2$ at $n_e = 4$ the error falls from $1.7 \times 10^{-5}$ at $p = 8$ to
$3.6 \times 10^{-9}$ at $p = 12$. `laplacian` and `hyperviscosity` are collective and, like `dss`,
the same bit for bit on any number of ranks and threads.
