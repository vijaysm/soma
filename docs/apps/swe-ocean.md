# Barotropic ocean

`swe_ocean` solves the barotropic (single-layer) ocean equations on SOMA's
[spectral elements](../algorithms/spectral-elements.md): the shallow-water equations of the
[atmosphere](swe-atm.md) with a coast, a real bathymetry, wind stress and bottom drag, and
passive tracers on the levels of a vertical coordinate. The mesh is the cubed sphere refined
toward the coastline of a height file (ETOPO1), with land and closed basins removed. Two analytic
cases in the plane (Stommel's gyre and a Kelvin wave) verify the discretization. A run gives the
same bits on any number of MPI ranks and host threads. It lives under `apps/swe_ocean/` and is
built with `-DSOMA_ENABLE_APPS=ON`:

```sh
cmake -S . -B build -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON -DSOMA_ENABLE_APPS=ON
cmake --build build --target swe_ocean
mpiexec -n 8 build/apps/swe_ocean/swe_ocean --case wind --ne 30 --coast-km 50 \
    --bathymetry ETOPO1_Ice_g_gmt4.nc --days 5 --report-hours 24 --save-mesh ocean_ne30.soma
mpiexec -n 8 build/apps/swe_ocean/swe_ocean --case wind --mesh ocean_ne30.soma --days 60
```

Without HDF5 the analytic cases run, but the global cases cannot read a height file or a saved
mesh: `--bathymetry` is refused with a message naming `SOMA_ENABLE_HDF5`.

## Equations

The unknowns are the free surface $\eta$ and the horizontal velocity $\mathbf u$ (three Cartesian
components, kept tangent) over the depth $H > 0$; $D = H + \eta$ is the total depth. In the
vector-invariant form of the atmosphere,

$$
\frac{\partial \eta}{\partial t} = -\nabla \cdot (D \mathbf u + \mathbf F),
\qquad \mathbf F = \nu \nabla (L \eta),
$$

$$
\frac{\partial \mathbf u}{\partial t} = -(\zeta + f)\, \hat{\mathbf n} \times \mathbf u
  - \nabla\left(g \eta + \tfrac12 |\mathbf u|^2\right) + \frac{\boldsymbol\tau}{\rho_0 D}
  - C_d \frac{|\mathbf u| \mathbf u}{D} - \nu \nabla^4 \mathbf u,
$$

with $\zeta$ the relative vorticity, $f = 2\Omega \sin\phi$, $\boldsymbol\tau$ the wind stress and
$L$ the weak Laplacian, so that $-\nabla \cdot \mathbf F = -\nu \nabla^4 \eta$ is written as the
divergence of a flux the tracers can share. The constants: $a = 6.37122 \times 10^6$ m, $g =
9.80616$ m s$^{-2}$, $\Omega = 7.292 \times 10^{-5}$ s$^{-1}$, $\rho_0 = 1025$ kg m$^{-3}$, $C_d =
2.5 \times 10^{-3}$.

`--linear` drops $\zeta$ and $\tfrac12|\mathbf u|^2$ and takes $D = H$ in the flux, the stress and
the drag; the analytic cases run in this mode, with linear drag $r \mathbf u$ and $\nu = 0$.

## Discretization

- **Space and time.** As the atmosphere's: `soma::sem::SemSpace` of order $p$, strong gradient and
  vorticity, every tendency made continuous by DSS, SSP-RK3, HOMME's $\nabla^4$ hyperviscosity.
- **Coast.** The space's boundary (`soma::sem::make_boundary`) gives each coast point a unit normal,
  the average of its skin edges' outward normals. After each stage the velocity at a coast point
  loses its normal component (`soma::sem::remove_normal`); at a corner where the coast turns by 90
  degrees or more, and at a pinch point, it is set to zero. The continuity and tracer equations use
  the weak divergence (`soma::sem::divergence_weak`, HOMME's `divergence_sphere_wk`): zero flux
  through the coast is built in, so the integral of $\eta$ is conserved to rounding.
- **Viscosity.** By default $\nu = 10^{15} (\Delta x_{\min} / \Delta x_{30})^{3.2}$ m$^4$ s$^{-1}$,
  $\Delta x_{\min}$ the shortest element edge on any rank and $\Delta x_{30} = \pi a / 60$ the
  edge at ne30: the coast-refined elements set it.
- **Time step.** The smaller of the advective limit $c\, \min (\delta / (|\mathbf u| + \sqrt{g D}))$,
  $\delta$ a point's least GLL gap, and the hyperviscous limit $2.5 c / (\nu \rho(\nabla^4))$, at
  $t = 0$, with $c = 0.4$. `--dt` replaces both.
- **Blow-up.** A step that leaves a depth not positive or a value not finite stops every rank:
  `swe_ocean: blew up at step N, t = T s`. The driver exits with status 1.

## Mesh and bathymetry

The global cases build the mesh in one pipeline (`ocean::build_ocean_mesh`):

1. the cubed sphere at `--ne`;
2. refined toward the coastline (`soma::refine_graded`): ETOPO1's 0 m contour, decimated to half
   of `--coast-km`, marks every cell within the band of its level, so that cells near the coast
   reach a GLL spacing of `--coast-km` and the levels grade outward over `--transition` cells;
3. smoothed by `soma::quality_smooth`;
4. the bathymetry averaged at every GLL point: the area-weighted mean depth of the ETOPO1 cells in
   the disc of the point's assembled mass (land counting as depth 0), bilinear below a 2 km
   footprint. Each rank reads only the windows of the file its points need;
5. elements with no ocean point removed, then every element not joined through shared edges to the
   world ocean (a seed point in the central Pacific): inland seas, lakes and basins cut off by
   unresolved straits. Both areas are reported;
6. the bathymetry averaged again on the final mesh and smoothed by diffusion until the slope factor
   $r_{x0} = |H_a - H_b| / (H_a + H_b)$ between neighbouring GLL points is at most 0.2, with $H
   \ge H_{\min} = 10$ m.

`--save-mesh FILE` writes the mesh with $H$, the roughness and the land flags as a SOMA file;
`--mesh FILE` reads it back at any number of ranks, giving the same $H$ by element and the same
run (`swe_ocean.mesh_reuse` checks this).

## Vertical coordinate and tracers

Tracers ride on $K$ levels (`--levels`, 10 by default) in every column. Level $k$ has the
thickness

$$
h_k = h^0_k(H) \left(1 + \frac{\eta}{H}\right), \qquad \sum_k h^0_k = H,
$$

so the levels stretch with the column as z-star's do and none vanishes while $D > 0$. Only the
reference thicknesses $h^0_k$ differ (`--vertical`):

- **`s`** (the default): ROMS's terrain-following coordinate, Vtransform 2 and Vstretching 4
  (Shchepetkin and McWilliams 2009), with $\theta_s = 5$, $\theta_b = 2$, $h_c = 100$ m
  (`--theta-s`, `--theta-b`, `--hc`).
- **`zstar`**: geopotential interfaces $z_k = -5500\,(e^{3k/K} - 1)/(e^3 - 1)$ m cut off at the
  bottom: $z^0_k = \max(z_k, -H + (K - k) h_{\min})$, so the bottom wet level is a partial cell
  and the levels below it vanish to $h_{\min} = 10^{-3}$ m.

Each level carries the share $h^0_k / H$ of the barotropic flux $D\mathbf u + \mathbf F$; the
vertical flux $w$ follows from the levels' continuity, upward from the bottom, and is zero at the
surface to rounding. A tracer obeys the flux form with the same weak divergence, upwinded vertical
fluxes taken implicitly in each stage, and a thickness-weighted biharmonic along the levels. A
uniform tracer stays uniform, and every tracer's total is conserved to rounding, in either
coordinate.

Known limits, accepted while tracers are passive (no density acts on the flow):

- **`s`.** Over steep slopes, diffusion along s-levels mixes across what would be isopycnals, and
  a baroclinic pressure gradient would carry the sigma-coordinate error. Neither matters for a
  passive tracer and a barotropic flow; a baroclinic model would need rotated diffusion and a
  density Jacobian. The run reports ROMS's slope factor $r_{x0}$ and the Haney number $r_{x1}$ of
  both coordinates' reference interfaces, so that a baroclinic successor knows where it stands.
  (In `zstar`, $r_{x1}$ is very large by construction wherever a vanished level lies beside a wet
  one; geopotential levels do not suffer the pressure-gradient error it measures.)
- **`zstar`.** There is no limiter. In a vanished level a tracer is set by its thick neighbours'
  fluxes and can overshoot: in the tracer tests on a synthetic Earth (50 steps) the largest
  $|T|$ of a varying tracer grows from 0.90 to 3.76 in `zstar`, against 0.99 to 1.01 in `s`. The
  overshoot lives in levels 1 mm thick and carries no mass; a limiter is for a later stage.

## Cases

| `--case` | Case | Checks |
| --- | --- | --- |
| `stommel` | Stommel's (1948) wind-driven gyre: a square basin of 1000 km on the beta plane, flat bottom 100 m, $\tau_x = -0.1 \cos(\pi y / L)$, linear drag $r = 2 \times 10^{-6}$ s$^{-1}$; run to steady state and compared with the analytic velocity | Coriolis, beta, drag, forcing, closed walls |
| `kelvin` | A Gaussian Kelvin wave packet along the wall of a channel of 4000 x 1000 km on the f-plane, run 2000 km and compared with the translated analytic packet | the coast condition, gravity-wave speed |
| `rest` | the ocean at rest over real bathymetry, no forcing | $\eta = 0$, $\mathbf u = 0$ stays exactly so |
| `wind` | a zonal wind stress over real bathymetry, quadratic drag | stability, conservation, the Drake Passage transport |

The wind case's stress is $\tau_x = \tau_0\, (-\cos 3\phi)$ eastward, $\tau_0 = 0.1$ N m$^{-2}$
(`--tau0`): easterlies of $\tau_0$ at the equator, none at 30 degrees, westerlies of $\tau_0$ at
60 degrees, none at the poles. The tracers start as a temperature $2 + 25 \cos^2\phi\, e^{z / 500}$
and a uniform salinity 35.

The analytic cases use `--ne` as the elements along the basin's or channel's side ($n_x$; the
channel has $n_x / 4$ across) and ignore `--days`: Stommel runs until the relative change of
$\mathbf u$ over $1/r$ is below $10^{-7}$, Kelvin until the packet has travelled 2000 km.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--case NAME` | `wind` | `stommel`, `kelvin`, `rest` or `wind` |
| `--ne N`, `--order P` | 30, 3 (`stommel` 8, 4; `kelvin` 40, 4) | elements along a cube face's edge (the analytic cases: along the basin); GLL order |
| `--coast-km KM` | 10 | GLL spacing near the coast; 0 or less: no refinement |
| `--transition N` | 3 | cells per refinement band |
| `--bathymetry FILE` | | the height file (NetCDF-4, read through HDF5); the global cases need it or `--mesh` |
| `--bathy-var`, `--bathy-lat`, `--bathy-lon` | `z`, `y`, `x` | its variable and coordinate names (ETOPO1's) |
| `--mesh FILE`, `--save-mesh FILE` | | read a saved mesh instead of building one; save the mesh built (or read) |
| `--vertical s\|zstar`, `--levels K` | `s`, 10 | the tracers' vertical coordinate |
| `--theta-s`, `--theta-b`, `--hc` | 5, 2, 100 | the s-coordinate's stretching |
| `--days D` | 1 | run length; the last step is shortened to land on it |
| `--dt S`, `--nu V` | from the limits, as above | time step and viscosity |
| `--tau0 T` | 0.1 | the wind case's stress (N m$^{-2}$) |
| `--linear` | | the linear mode |
| `--report-hours H` | the run length | report interval; a report lands on the last whole step at or before its time, so reports leave the run's bits alone |
| `--output PREFIX` | | at the end, `PREFIX.soma` (below); for the analytic cases too |
| `--output-reports` | | also `PREFIX.stepN.soma` at every report but the last |
| `--latlon NLAT,NLON` | | at the end, `PREFIX_eta.npy`, `PREFIX_H.npy`, `PREFIX_u_east.npy`, `PREFIX_u_north.npy` (NaN over land), `PREFIX_lat.npy` and `PREFIX_lon.npy`, at latitudes from -90 to 90 inclusive and longitudes from 0 every 360/NLON degrees |

A bad option prints the usage and exits with status 2.

The SOMA snapshot holds the mesh with fields of its GLL space: `eta`, `u` (3 Cartesian
components), `H`, `h` (the $K$ level thicknesses) and `tracer0`, `tracer1` (each tracer's $K$
level values $T_k$).

## Reports

A global run first reports what the build did: the cells, the removed land and disconnected
areas, $r_{x0}$, the smoothing sweeps, the largest window of the height file one rank read, the
$r_{x1}$ of both coordinates, then the area, volume, step, viscosity and the coast and no-slip
points. Each report then gives the time, the step, the kinetic energy $\tfrac12 \rho_0 \int D
|\mathbf u|^2$, the largest speed, the change of $\int \eta$ over the ocean's volume, each
tracer's relative change of total, and the Drake Passage transport. At the end it prints the
simulated days per wall-clock hour and the state's digest, a hash of every element's values in
gid order, the same on any number of ranks.

The Drake Passage transport is the integral of $D\, \mathbf u \cdot \hat{\mathbf e}_{\rm east}$
along the meridian 67 W from the Antarctic coast to South America, in Sv ($10^6$ m$^3$ s$^{-1}$),
eastward positive (`ocean::meridian_transport`). The meridian is sampled every 0.01 degree from
80 S to 45 S. Each sample is located in its element by Gauss-Newton on the element's map (the
degree-$p$ interpolant of its GLL points, projected to the sphere; a point on a shared edge goes
to the element of least gid), and the model's interpolant of $D\,\mathbf u \cdot \hat{\mathbf
e}_{\rm east}$ is evaluated there. A sample no element holds is land, and so is one where the
interpolated land flags of the kept elements reach 1/2 (an island smaller than an element, such
as Tierra del Fuego at ne30, keeps its elements at the least depth); the run of wet samples
around 60 S, from coast to coast, is integrated by the trapezoidal rule. The coast is found to
within one sample (1.1 km), where the flow is slow, and inside the run the rule's error is
second order in 0.01 degree. On a uniform eastward flow over the cubed sphere at ne4 $p = 4$ the
transport is within $10^{-5}$ of the exact value (`MeridianTransport.uniform_flow`). Only the
owner of each sample's element contributes, so the transport is the same, bit for bit, on any
number of ranks.

## Verification

The tests run under `ctest` (`ctest --test-dir build -R "ocean|swe_ocean"`), each in under 60 s:

| Check | Result |
| --- | --- |
| Stommel, $p = 4$, $n_x$ = 4, 8 | relative $l_2$ velocity error 5.5e-3, 4.0e-4 (ratio 13.7); $n_x$ = 6, 12: 1.2e-3, 8.2e-5 (ratio 15), measured once (44 s at 4 ranks, over the tests' budget) |
| Kelvin, $n_x$ = 40, $p$ = 2, 4, 6 | relative $l_2$ $\eta$ error 2.2e-2, 4.2e-5, 3.7e-7 |
| conservation | $\int \eta$ to $10^{-13}$ of $\int |\eta|$; every tracer's total to rounding in both coordinates |
| coast | no normal velocity at coast points to rounding; zero at no-slip points |
| `rest` | $\eta$ and $\mathbf u$ stay exactly 0 |
| uniform tracer | stays uniform to $10^{-15}$ over the synthetic Earth's bathymetry, in both coordinates; $w_0$ at the surface 3e-15 of $\max|\nabla\cdot(D\mathbf u)|$ |
| bathymetry | the footprint mean of an analytic field; $r_{x0} \le 0.2$ after smoothing; each rank reads only its windows (counted) |
| reproducibility | the same digest at 1 to 8 ranks and 1, 2, 4 threads; the driver's Kelvin run at 1, 2 and 4 ranks (`swe_ocean.kelvin.digests`); a mesh saved at 2 ranks and read at 3 (`swe_ocean.mesh_reuse`) |

Errors converge exponentially in $p$ for Kelvin, and at about $h^4$ for Stommel at $p = 4$ (the
western boundary layer, $r / \beta = 100$ km, is 0.4 of an element at $n_x = 4$).

## A global run

ETOPO1 at ne30, $p = 3$, coasts refined to 50 km, 8 ranks of 1 thread (M2 Max):

```text
swe_ocean: case wind, p 3, 8 rank(s); mesh built at ne 30 in 41.6 s
cells 36875; removed 1.22307e+08 km^2 of land, 1.80289e+06 km^2 not joined to the world ocean
bathymetry: rx0 0.2772 (r_max 0.20, NOT smoothed to r_max), 10000 smoothing sweeps, peak window 9741600 values
rx1 of 10 levels: s 3.3704, zstar 1329353.1831
area 3.859895e+14 m^2, volume 1.344950e+18 m^3; dt 16.7549 s, nu 7.31274e+11 m^4/s, 10 levels (s), 2 tracers
coast points 6494, no-slip points 717
t    0.1000 d  step      516  KE 1.3984e+14 J  max|u| 1.084e-01 m/s  mass +8.25e-26  tracers +0.00e+00 +0.00e+00
simulated 0.1 days in 516 steps, 359.767 s: 1.001 days per hour
```

- The mesh build (coastline, refinement, two bathymetry averages, smoothing) takes 42 s and
  under 720 MB per rank; the saved mesh is 13 MB.
- The smoothing stops at its sweep limit with $r_{x0} = 0.28$ above the bound of 0.2: the run
  says so rather than failing. The step (17 s) is set by the hyperviscous and advective limits
  of the 50 km coastal elements; at 1 simulated day per hour, 60 days and 10 km coasts are runs
  for a cluster.
- Mass changes by $10^{-25}$ of the volume and the tracers by $10^{-16}$: rounding.
- With the land flags, the Drake Passage section runs from 69.3 S to 54.9 S (Tierra del Fuego);
  the transport is spinning up from rest (0.13 Sv after 0.01 day).

## Real bathymetry: ETOPO1

The global cases read ETOPO1 (NOAA's 1 arc-minute global relief, ice surface, grid registered:
10801 x 21601 heights in metres, negative below sea level). Download and convert it once to
NetCDF-4 (the reader uses HDF5), chunked so that a window of rows is a few chunks:

```sh
curl -O https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz
gunzip ETOPO1_Ice_g_gmt4.grd.gz
nccopy -k nc4 -d 4 -s -c y/540,x/540 ETOPO1_Ice_g_gmt4.grd ETOPO1_Ice_g_gmt4.nc
```

The file holds `z` over `y` (latitude) and `x` (longitude), the reader's defaults. Other grids
work too: grid- or cell-registered, either latitude order, any variable and axis names
(`--bathy-var`, `--bathy-lat`, `--bathy-lon`). The data are not part of SOMA.

## Reproducibility

Every sum that crosses elements or ranks is partition-independent: DSS, integrals (exact sums),
the bathymetry's footprints (each point sums its cells in a fixed order), the smoothing's
reductions, the coastline (each edge seen by one rank, decimated in a fixed order) and the
digest's gather. The mesh is the same gid for gid on any number of ranks. A run therefore has the
same bits on any number of ranks and threads, and with any report interval.

## References

- H. Stommel (1948). The westward intensification of wind-driven ocean currents. *Trans. Am.
  Geophys. Union* 29, 202-206.
- A. F. Shchepetkin and J. C. McWilliams (2009). Correction and commentary for "Ocean forecasting
  in terrain-following coordinates: formulation and skill assessment of the regional ocean
  modeling system" by Haidvogel et al. *J. Comput. Phys.* 228, 8985-9000.
- C. Amante and B. W. Eakins (2009). ETOPO1 1 arc-minute global relief model: procedures, data
  sources and analysis. NOAA Technical Memorandum NESDIS NGDC-24.
- C.-W. Shu and S. Osher (1988). Efficient implementation of essentially non-oscillatory
  shock-capturing schemes. *J. Comput. Phys.* 77, 439-471.
