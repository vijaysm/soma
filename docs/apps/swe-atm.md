# Shallow-water atmosphere

`swe_atm` solves the shallow-water equations on the rotating sphere with SOMA's
[spectral elements](../algorithms/spectral-elements.md), on the cubed sphere, as HOMME's
shallow-water model does. It runs the standard test cases of Williamson et al. (1992) and
Galewsky et al. (2004) and reports their errors, conservation and throughput. A run gives the
same bits on any number of MPI ranks and host threads. It lives under `apps/swe_atm/` and is built
with `-DSOMA_ENABLE_APPS=ON`:

```sh
cmake -S . -B build -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON -DSOMA_ENABLE_APPS=ON
cmake --build build --target swe_atm
mpiexec -n 8 build/apps/swe_atm/swe_atm --case w5 --ne 30 --order 3 --days 15 --report-every 24
```

Without HDF5 the model runs as well, but cannot read reference files or write SOMA files; those
options then fail with a message naming `SOMA_ENABLE_HDF5`.

## Equations

The unknowns are the fluid depth $h$ and the horizontal velocity $\mathbf u$. The velocity is
stored as its three Cartesian components and kept tangent to the sphere. The vector-invariant
form is

$$
\frac{\partial \mathbf u}{\partial t} = -(\zeta + f)\, \hat{\mathbf n} \times \mathbf u
  - \nabla\left(g (h + h_s) + \tfrac12 |\mathbf u|^2\right) - \nu \nabla^4 \mathbf u,
\qquad
\frac{\partial h}{\partial t} = -\nabla \cdot (h \mathbf u) - \nu \nabla^4 h,
$$

with $\zeta = \hat{\mathbf n} \cdot \nabla \times \mathbf u$ the relative vorticity, $f$ the
Coriolis parameter, $h_s$ the topography and $\hat{\mathbf n}$ the outward normal. The constants
are Williamson's: $R = 6.37122 \times 10^6$ m, $\Omega = 7.292 \times 10^{-5}$ s$^{-1}$ and
$g = 9.80616$ m s$^{-2}$.

## Discretization

- **Space.** `soma::sem::SemSpace` of order $p$ on `soma::sem::cubed_sphere_mesh` with $n_e$ quads
  along each face's edge. Each element holds $(p + 1)^2$ GLL points.
- **Operators.** Gradient, divergence and vorticity are the strong-form spectral operators. The
  vorticity is made continuous (DSS) before it multiplies $\hat{\mathbf n} \times \mathbf u$.
- **Continuity.** Each tendency is made continuous by DSS, and the velocity tendency is projected
  tangent.
- **Time stepping.** SSP-RK3 (Shu and Osher):

  $$
  y_1 = y + \Delta t\, F(y), \quad
  y_2 = \tfrac34 y + \tfrac14 (y_1 + \Delta t\, F(y_1)), \quad
  y \leftarrow \tfrac13 y + \tfrac23 (y_2 + \Delta t\, F(y_2)).
  $$
- **Dissipation.** $\nabla^4$ hyperviscosity in HOMME's form. For $h$ it is the weak Laplacian
  applied twice. For $\mathbf u$ it is the vector Laplacian
  $\nabla\delta + \hat{\mathbf n} \times \nabla\zeta + 2\mathbf u / R^2$ applied twice, which leaves
  rigid rotation undamped.
- **Viscosity.** By default $\nu = 10^{15} (30 / n_e)^{3.2}$ m$^4$ s$^{-1}$: HOMME's ne30 value,
  scaled with the grid spacing to the power 3.2. Case 2 runs without viscosity unless `--nu` is
  given.
- **Time step.** The smaller of an advective and a hyperviscous limit:

  $$
  \Delta t = \min\left(c \frac{\Delta x}{\max(|\mathbf u| + \sqrt{g h})},\;
                       \frac{2.5\, c}{\nu\, \rho(\nabla^4)}\right).
  $$

  - $c$ is `--cfl` (0.4 by default).
  - $\Delta x = \frac{\pi R}{2 n_e} \cdot \frac{\xi_1 - \xi_0}{2}$ is the smallest GLL spacing.
  - The speed is the largest at $t = 0$.
  - $\rho(\nabla^4)$ is the spectral radius of the discrete operator: the larger of the scalar
    and vector one, each from 40 power iterations at startup.

  The hyperviscous limit binds at coarse resolution. At $n_e = 3$, $p = 4$ the advective step
  alone blows up within a few steps. At ne30 $p = 3$ the advective limit binds (about 140 s).
  `--dt` replaces both limits.
- **Blow-up.** A step that leaves a depth not positive or a value not finite stops every rank:
  `swe: blew up at step N, t = T s`. The driver exits with status 1.

## Test cases

| `--case` | Case | Source |
| --- | --- | --- |
| `w2` | steady zonal geostrophic flow, axis tilted by `--alpha`; the exact solution is the initial state | Williamson et al. (1992), case 2 |
| `w5` | zonal flow over an isolated conical mountain (2000 m, radius $\pi/9$, at $(3\pi/2, \pi/6)$) | Williamson et al. (1992), case 5 |
| `w6` | Rossby-Haurwitz wave of wavenumber 4 | Williamson et al. (1992), case 6 |
| `galewsky` | barotropically unstable mid-latitude jet with a height bump of 120 m | Galewsky et al. (2004) |
| `galewsky-steady` | the same jet without the bump: a steady state | Galewsky et al. (2004) |

The formulas are in `apps/swe_atm/cases.cpp`, which writes out case 6's $A$, $B$ and $C$. For
Galewsky's jet, the balanced height comes from integrating the balance equation by Simpson's rule
on $10^4$ intervals, with $h_0$ set so that the mean depth is 10 km.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--case NAME` | `w2` | the test case |
| `--ne N`, `--order P` | 4, 3 | elements along a face's edge; GLL order |
| `--days D` | 1 | run length; the last step is shortened to land on it |
| `--alpha A` | 0 | case 2's rotation angle (radians) |
| `--cfl C`, `--dt S`, `--nu NU` | 0.4, from the limits, as above | time step and viscosity |
| `--report-every HOURS` | the run length | report interval; a report lands on the last whole step at or before its time (the last on the end), so reports leave the run's bits alone |
| `--vtk STEM` | | at the end, `STEM.<rank>.vtk`: each owned element as $p \times p$ quads with point fields `h`, `hs`, `u` |
| `--soma FILE` | | at the end, the mesh in SOMA's format with fields `h`, `hs`, `u` on the GLL space |
| `--latlon NLAT,NLON,STEM` | | at the end, `STEM_h.npy`, `STEM_vorticity.npy`, `STEM_lat.npy` and `STEM_lon.npy`: depth and vorticity at latitudes from -90 to 90 inclusive and longitudes from 0 every 360/NLON degrees |
| `--reference FILE` | | NetCDF-4 reference solution of the free-surface height $h + h_s$ to compare the final state against |
| `--ref-var`, `--ref-lat`, `--ref-lon`, `--ref-time` | `h`, `lat`, `lon`, -1 | the reference's variable and coordinate names, and its time index (-1: the last) |

Each report gives the time, the step, and the relative change of mass and total energy since
$t = 0$. The steady cases also report the normalised $l_1$, $l_2$ and $l_\infty$ depth errors
against the initial (exact) state and the largest velocity error. At the end the driver prints the
reference errors, the simulated days per wall-clock hour and the state's digest. The digest is a
hash of every element's values in gid order, the same on any number of ranks.

## Errors

Height errors are Williamson's normalised norms:

$$
l_1 = \frac{I|h - h_t|}{I|h_t|}, \quad
l_2 = \frac{\sqrt{I (h - h_t)^2}}{\sqrt{I h_t^2}}, \quad
l_\infty = \frac{\max|h - h_t|}{\max|h_t|}.
$$

Like Williamson's, they measure the free-surface height $h + h_s$ (the depth itself for every
case but 5, which alone has topography). For case 2, $I$ is the spectral-element integral (exact
sums). Against a reference file, which must hold the free-surface height, the model's
$h + h_s$ is evaluated at the reference's grid points. Each point is located on the cubed
sphere (a point on an element edge goes to the element of smallest gid) and evaluated by its
element's Lagrange interpolant. $I$ is then a sum weighted by $\cos\phi$ and each point's share
of the latitude and longitude spacing.

The interpolation goes from the model to the reference grid, so the reference adds no
interpolation error of its own.

Reference files are NetCDF-4 (convert a netCDF-3 file once with `nccopy -k nc4`), read through
HDF5. The variable may be (lat, lon) or (time, lat, lon). Latitudes may run either way, and
longitudes may be in [0, 360) or [-180, 180).

## Benchmarks

`apps/swe_atm/bench/` holds the benchmark runs. They are reported rather than run by `ctest`.

- **`run_cases.sh NP [REFDIR]`**
  - Case 2's $l_2$ error after 5 days at $n_e$ = 4, 8, 16 and $p$ = 2 to 8.
  - Cases 5 and 6 at ne30 $p = 3$ for 15 days, against `REFDIR/case5.nc` and `REFDIR/case6.nc`.
    `REFDIR` defaults to `apps/swe_atm/reference`, which holds T213 reference solutions of the
    free-surface height, days 0 to 15. They were computed with NCAR's spectral transform model
    (STSWM 2.0) at the parameters of NCAR's published reference runs (Jakob-Chien, Hack and
    Williamson 1995); `apps/swe_atm/reference/README_STSWM.md` records the procedure.
  - A case passes when its $l_2$ error is at most twice the mark in `REFDIR/marks.txt`. Each line
    there reads `case5 <l2> <source>`. No paper gives HOMME's $l_2$ error against the T213
    references at ne30 np4, so the marks are this model's own errors at ne30, kept so that a
    change cannot double them; the checks below are what makes them trustworthy.
- **`galewsky.sh NP [OUTDIR]`**
  - The unperturbed jet at ne30 for 6 days, reporting its drift.
  - The perturbed jet at ne30, ne60 and ne120 to day 6, writing vorticity on a 361 x 720 grid.
  - `compare_npy.py` then checks that the ne30 and ne60 fields' differences from ne120 fall.

The pass marks:

| Case | Mark |
| --- | --- |
| 2 | exponential convergence in $p$, down to rounding-limited errors |
| 5, 6 | $l_2$ height error at day 15, ne30 $p = 3$, at most twice the mark; the error falls with $n_e$ to the reference's own level |
| Galewsky | the unperturbed jet stays steady; the perturbed run converges toward the ne120 run |

Results (v0.1.0; 8 ranks of 1 thread, M2 Max), normalised $l_2$ height error at day 15:

| | ne15 | ne30 | ne60 |
| --- | --- | --- | --- |
| case 5 against T213 | 8.87e-4 | 8.45e-4 | 8.50e-4 |
| case 5 against SOMA ne60 | 4.87e-4 | 8.4e-5 | |
| case 6 against T213 | 1.27e-2 | 6.07e-3 | 2.52e-3 |

- Case 6 converges toward T213, its error falling by a factor of 2.1 to 2.4 per doubling of
  $n_e$.
- Case 5 meets T213 at about 8.5e-4 from ne15 on: the model converges well below that (ne30
  against ne60, 8.4e-5; with `--latlon 181,360` and areas weighted by $\cos\phi$), so the level is
  the reference's own difference from the converged solution. HOMME's error at ne30 np4 against
  a 0.125-degree HOMME run is about 6.7e-5 (Guba et al. 2014, *Geosci. Model Dev.* 7,
  2803-2816, Fig. 12, read from the plot), within a factor of 1.3 of ours.
- Case 2 converges exponentially in $p$, to 2.6e-13 at ne16 $p = 8$.

The scripts take the executable from `SWE_ATM` (default `build/apps/swe_atm/swe_atm`) and the
launcher from `MPIEXEC`. On the M2 Max, ne30 $p = 3$ case 5 runs at about 250 simulated days per
hour on 4 ranks of 2 threads.

## Reproducibility

Every sum that crosses elements or ranks is partition-independent: DSS, integrals (exact sums) and
the digest's gather. Every other operation is local to a point. A run therefore has the same bits
on any number of ranks and threads, and with any report interval. The tests check this from 1 to
8 ranks (`SweModel.*`, the `swe_atm.case2.digests` driver run at 1, 2 and 4 ranks), at 1, 3 and 4
threads (`test_swe_model.threads`) and at three report intervals (`swe_atm.case5.report_interval`).

## References

- D. L. Williamson, J. B. Drake, J. J. Hack, R. Jakob and P. N. Swarztrauber (1992). A standard
  test set for numerical approximations to the shallow water equations in spherical geometry.
  *J. Comput. Phys.* 102, 211-224.
- R. Jakob-Chien, J. J. Hack and D. L. Williamson (1995). Spectral transform solutions to the
  shallow water test set. *J. Comput. Phys.* 119, 164-187.
- J. Galewsky, R. K. Scott and L. M. Polvani (2004). An initial-value problem for testing numerical
  models of the global shallow-water equations. *Tellus A* 56, 429-440.
- C.-W. Shu and S. Osher (1988). Efficient implementation of essentially non-oscillatory
  shock-capturing schemes. *J. Comput. Phys.* 77, 439-471.
