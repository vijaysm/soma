# Example programs

SOMA ships six example programs in `examples/`, one per algorithm (two for refinement), the command-line tool
`soma-convert` in `tools/`, and `examples/app/`, an application built outside SOMA's tree (see
[below](#an-application-outside-somas-tree)). Each is a complete program of a page or two that reads a mesh on any
number of ranks, runs one algorithm and writes the result; read them alongside the algorithm
pages. This page gives each one's purpose, command line, output and the tests that run it.

## Building and running them

The examples read and write meshes, so they need HDF5:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos \
  -DSOMA_ENABLE_HDF5=ON -DSOMA_ENABLE_EXAMPLES=ON [-DSOMA_ENABLE_MPI=ON]
cmake --build build -j
```

The programs are built under `build/examples/` and `soma-convert` under `build/tools/`;
`soma-convert` is built whenever `SOMA_ENABLE_HDF5` is on, and installed by `cmake --install`.
The examples are not installed.

Every program shares the same structure:

- It starts with `soma::ScopeGuard` and runs on `soma::Comm::world()`, so it runs as it is on one
  rank or under `mpiexec -n P`. SOMA's runtime options (`--soma-reproducible`,
  `--soma-compress-threshold=<x>`) and Kokkos's (`--kokkos-num-threads=<n>`, ...) may be added
  to any command line.
- It reads `<in>` with `soma::read_file`, which accepts a SOMA file, a MOAB `.h5m` file or a
  legacy VTK file by content, and writes `<out>` with `soma::write_file`, as a SOMA, `.h5m` or
  `.vtk` file by its extension (see [I/O](../io/overview.md)). On more than one rank a SOMA
  output file is written in parallel; `.h5m` and `.vtk` are gathered on the first rank.
- The first rank prints one line on success, starting with the program's name as invoked
  (`argv[0]`). The exit status is 0 on success, 1 on failure (the
  message goes to standard error) and 2 for bad arguments (with a usage line).

The bundled meshes in `data/` used below are `64bricks_1khex.h5m`, a box of $12^3 = 1728$ hexes,
and `hybrid_3quad_1tri.h5m`, a plane mesh.

## laplacian_smooth

```text
laplacian_smooth [--iterations K] [--bridge-dim B] [--tolerance T] <in> <out>
```

Adds one ghost layer through vertices (`GhostOptions{1, 0, false}`), smooths the vertex
coordinates by `K` Jacobi Laplacian sweeps with `soma::laplacian_smooth` (the skin stays fixed),
removes the ghosts and writes the mesh without fields. `--iterations` defaults to 10;
`--bridge-dim` and `--tolerance` set `SmoothOptions::bridge_dim` (neighbours through entities of
that dimension; the cells by default) and `SmoothOptions::tolerance` (stop once no vertex moves
farther than `T` in a sweep; 0, the default, never stops early). See
[smoothing](../algorithms/smoothing.md).

It prints the number of sweeps made, with `--tolerance` the largest move in the last sweep, and
the file written:

```text
laplacian_smooth: 5 sweeps; wrote smoothed.vtk
```

## poisson

```text
poisson [--assembly gather|atomic|colored] [--rtol R] <in> <out>
```

Solves $-\Delta u = 1$ with $u = 0$ on the boundary by linear finite elements on a mesh of
triangles, quads, tets or hexes: adds a ghost layer through vertices, builds a
`soma::FemSpace<Kokkos::HostSpace>` with the chosen assembly (`gather`, the default; `atomic`;
`colored`), assembles the stiffness matrix and load vector, applies the Dirichlet condition with
`soma::apply_dirichlet`, and solves with Jacobi-preconditioned conjugate gradients (`soma::cg`)
to a relative residual of `R` (default `1e-10`). It writes $u$ as the vertex field `"u"` and
prints whether CG converged, its iterations and residual, and the largest value of $u$. The exit
status is 1 if CG did not converge. See [finite elements](../algorithms/fem.md) and
[linear algebra](../algorithms/linear-algebra.md).

## advect

```text
advect [--steps N] [--cfl C] [--scheme upwind|bounded] [--flow confined|rigid] <in> <out>
```

Carries a Gaussian bump about the vertical axis through the centre of the mesh's bounding box by
finite volumes (`soma::FvSpace`), after adding a ghost layer through vertices. It takes `N` steps
(default 100) of `C` (in $(0, 1]$, default 0.5) times the largest stable step
`FvSpace::max_dt`, nothing flowing in, by the scheme `--scheme` names (default `bounded`; see
[finite volumes](../algorithms/finite-volumes.md)). The flow `confined` (the default) turns once in
unit time out to two thirds of the box's half-width $h$ in $x$ and $y$, slows smoothly to rest at
$0.9 h$ and is still beyond, so nothing crosses the walls; the bump starts at $h / 3$ from the axis,
of width $h / 6$. The flow `rigid` turns the whole box, across its walls, with the bump at a quarter
of the box's width from the axis and of an eighth of it. It writes $u$ as the cell field `"u"`
and prints the scheme, the flow, the step count and size, the total $\sum_e V_e u_e$ before and
after (to 16 significant digits) and the range of $u$ after.

With `--soma-reproducible` the totals are summed exactly, and they are the same, bit for bit, on
any number of ranks and threads; the tests compare them to the last digit on one and two ranks.

## remap

```text
remap [--n N] [--tolerance T] <in> <out>
```

Reads a plane mesh of triangles, quads or polygons, or a volume mesh of tets, hexes, prisms or
pyramids, gives each cell the value $2 + \sin x \cos y$ at the mean of its corners, and remaps
it conservatively with `soma::RemapOperator<Kokkos::HostSpace>` onto a grid over the mesh's
bounding box (widened slightly on each side): $N \times N$ squares (default $N = 24$) each cut into
two triangles in 2D, or $N^3$ cubes each cut into six tets in 3D. The grid is built
distributed, its slabs dealt to the ranks, with `MeshBuilder` and `build_local_part`.
`--tolerance` sets `RemapOptions::merge_tolerance`. Since the grid covers every source cell, the
total $\sum |S| u_S$ is conserved; the program prints it on the source and on the grid and
writes the grid with the cell field `"u"`. See [remap](../algorithms/remap.md).

## refine

```text
refine [--levels L] [--marked F [--conforming]] <in> <out>
```

Gives each vertex the linear field $w = x + 2y + 3z$ and each cell its gid as the field
`"part"`, then refines the mesh `L` times (default 1):

- without `--marked`, uniformly (`soma::refine_uniform`);
- with `--marked F`, each round marks the cells that hold the fraction `F` of an indicator
  favouring large cells near the mesh's lowest corner (`soma::mark_fraction`), refines them
  (`soma::refine_marked`, leaving hanging vertices, or none with `--conforming`) and rebalances
  the parts (`soma::rebalance`).

It prints the global cell and vertex counts before and after, the hanging vertices left, and the
largest $|w - (x + 2y + 3z)|$ over the refined vertices, which is at rounding level because
averages of a linear field are exact; then it writes the refined mesh with `"w"` and `"part"`.
See [refinement](../algorithms/refinement.md).

## regional_refine

```text
regional_refine [--ne N | --mesh FILE] [--radius R] [--km K] [--np P]
                [--box LAT0,LAT1,LON0,LON1]... [--buffer B] [--smooth S]
                [--max-passes M] [--save FILE] <out-prefix>
```

Builds a regionally refined mesh of the sphere: the cubed sphere `ne N` (default 30), or a quad
mesh on a sphere read from `FILE` (a mesh refined before, say), refined with
`soma::refine_quads` until the grid spacing in the lat/lon boxes is about `K` km (default 12),
then smoothed by `S` sweeps (default 20) of each `soma::laplacian_smooth` method. The default
box is the contiguous United States; repeat `--box` for several regions. Spacing is measured as
on the Earth, so a mesh on the unit sphere refines as the Earth-sized one does. It prints the passes,
the cell counts, the spacing in and out of the boxes and a table of cell quality, and writes
three meshes with quality fields. See
[the example in refinement](../algorithms/refinement.md#example-regionally-refined-meshes-of-the-sphere).

```text
./regional_refine: passes 2; cells 5400 -> 28096; radius 6.37122e+06
spacing in the boxes: min 10.40 km, mean 11.04 km over 13671 cells; outside: 10.56 to 111.11 km
mesh        min angle      mean < 45 deg   aspect  jacobian     area
refined         30.56     79.69      580    3.589     0.506    4.590  conus_refined.vtk
neighbours      29.18     81.15      399    7.100     0.454    5.600  conus_neighbours.vtk
centroids       29.58     80.91      414    4.768     0.473    5.858  conus_centroids.vtk
```

## soma-convert

```text
soma-convert [--compression N] <in> <out>
```

Converts a MOAB `.h5m` file into a SOMA file, or a SOMA file into a MOAB `.h5m` file, by the
input's content (`soma::convert_mesh_file` in `soma/io/soma_file.hpp`). Run it under `mpiexec`
to convert in parallel: a `.h5m` input is read in parallel and written with `write_soma`; a SOMA
input is read in parallel, gathered on rank 0 and written there with `write_h5m`. `--compression`
is the deflate level of the file written (default 0). It prints, for example,

```text
soma-convert: wrote m.soma as SOMA
```

and fails (status 1) for an input of neither format. See [the SOMA format](../io/soma-format.md).

## An application outside SOMA's tree

```text
soma_app [--n N] [<mesh>] <out>
```

`examples/app/` is a CMake project of its own, not part of SOMA's build: it reaches SOMA only
through an installed package, as a downstream application does.

```cmake
cmake_minimum_required(VERSION 3.24)
project(soma_app LANGUAGES CXX)
find_package(soma REQUIRED)
add_executable(soma_app main.cpp)
target_link_libraries(soma_app PRIVATE soma::soma)
```

Build it against an install of SOMA (see [installation](installation.md)):

```bash
cmake --install build --prefix $HOME/soma
cmake -S examples/app -B app-build -DCMAKE_PREFIX_PATH="$HOME/soma;/path/to/kokkos"
cmake --build app-build
mpiexec -n 4 app-build/soma_app --n 16 out.soma
```

It reads `<mesh>` with `soma::read_file`, or without one builds an $N^3$ grid of hexes over the
unit cube with `soma::MeshBuilder` and `soma::build_local_part` (default $N = 8$, slabs of cells
dealt to the ranks). It gives each cell the density $1 + x + 2y + 3z$ at its corners' mean,
refines the mesh once with `soma::refine_uniform` (the density carries to the children), remaps
the density conservatively onto the refined cells with `soma::RemapOperator`, and writes the
refined mesh with both fields. When `<out>` is a SOMA file it reads it back. It fails unless a hex
mesh's cell count grows eightfold, the mass $\sum_T |T|\,u_T$ is the same on both meshes to
$10^{-12}$, and the file read back holds the remapped density as written:

```text
OK soma app: 2 rank(s), cells 216 -> 1728, mass 4 conserved to 2.2e-16, wrote two.soma and read it back
```

## How the tests run them

With `SOMA_ENABLE_TESTS=ON` as well, `examples/CMakeLists.txt` registers a ctest for each
example on a bundled mesh, and with MPI the same run at two ranks (names ending in `.np2`,
`TIMEOUT 60`, `PROCESSORS 2`). Most check the printed line with a regular expression, so the
expected results below hold on one and two ranks alike:

| Test | Command (inputs in `data/`) | Checked output |
| --- | --- | --- |
| `example_laplacian_smooth` | `laplacian_smooth --iterations 5 64bricks_1khex.h5m smoothed.vtk` | exit status 0 |
| `example_laplacian_smooth.np2` | the same at two ranks with `--tolerance 1e-9`, writing `.soma` | exit status 0 |
| `example_poisson`, `.np2` | `poisson 64bricks_1khex.h5m poisson.vtk`; at two ranks `--assembly colored`, writing `.soma` | `max u 9.090723e-01` |
| `example_advect`, `.np2` | `advect --soma-reproducible 64bricks_1khex.h5m advect.vtk` | `total 2.061703740944819e-01 before, 2.061703740944818e-01 after; u in [`(the least, at rounding level, not pinned)`, 1.199141e-01]` |
| `example_advect.upwind` | `advect --soma-reproducible --scheme upwind --flow rigid 64bricks_1khex.h5m advect_upwind.vtk` | `total 6.948772654115442e-01 before, 4.446464981696669e-01 after` |
| `example_remap`, `.np2` | `remap hybrid_3quad_1tri.h5m remap.vtk` | `total 2.7464101622e+01 on the source, 2.7464101622e+01 on the grid` |
| `example_remap3d` | `remap --n 12 64bricks_1khex.h5m remap3d.vtk` | `total 1.2894214008e+02 on the source, 1.2894214008e+02 on the grid` |
| `example_refine`, `.np2` | `refine 64bricks_1khex.h5m refined.vtk` | `cells 1728 -> 13824, vertices 2197 -> 15625, 0 hanging` |
| `example_refine_plane` | `refine --levels 2 hybrid_3quad_1tri.h5m refined2.vtk` | `cells 64 -> 1024, vertices 75 -> 969, 0 hanging` |
| `example_refine_marked` | `refine --levels 3 --marked 0.5 hybrid_3quad_1tri.h5m marked.vtk` | `cells 64 -> 730, vertices 75 -> 751, 35 hanging` |
| `example_refine_conforming`, `.np2` | `refine --levels 2 --marked 0.4 --conforming 64bricks_1khex.h5m conforming.vtk` | `cells 1728 -> 39998, vertices 2197 -> 29419, 0 hanging` |
| `example_regional_refine`, `.np2` | `regional_refine --ne 4 --km 300 --smooth 3 --save regional.soma regional` (no `--save` at two ranks) | `passes 1; cells 96 -> 556` |
| `example_regional_refine_again` | `regional_refine --mesh regional.soma --km 300 --smooth 0 regional_again` | `passes 0; cells 556 -> 556` |
| `example_regional_refine_antimeridian` | `regional_refine --ne 4 --km 300 --smooth 3 --box -20,20,170,-170 regional_dateline` | `passes 1; cells 96 -> 576` |
| `example_regional_refine_arctic` | `regional_refine --ne 4 --km 300 --smooth 0 --box 60,90,-180,180 regional_arctic` | `passes 1; cells 96 -> 448` |
| `example_regional_refine_everywhere` | `regional_refine --ne 4 --km 300 --smooth 0 --box -90,90,-180,180 regional_all` | `passes 1; cells 96 -> 864` and `outside: none` |
| `example_regional_refine_unit_sphere` | `regional_refine --ne 4 --radius 1 --km 300 --smooth 0 regional_unit` | `passes 1; cells 96 -> 556; radius 1 (spacing as on the Earth)` |

Each `refine` test also requires the linear field error to print as `d.de-14` to `d.de-19`, or
`0.0e+00`: below $10^{-13}$.

`tests/CMakeLists.txt` runs `soma-convert` too, at one rank and, with MPI, at three
(`soma_convert.*.np1`, `soma_convert.*.np3`): `to_soma` converts `data/hexes_mixed.h5m` to a SOMA
file and expects `as SOMA` in its output; `to_h5m` converts that file back with
`--compression 4` and expects `as MOAB`; `rejects` gives it the VTK file
`data/hexes_mixed.vtk` and expects it to fail. Run them all with

```bash
ctest --test-dir build -R "example_|soma_convert"
```

`packaging.example_app` installs SOMA into an empty prefix, builds `examples/app/` against it,
and runs `soma_app --n 6` at one rank and, with MPI, two; it expects
`cells 216 -> 1728` in the output.

See [building and testing](../developer/building-and-testing.md) for the rest of the suite.
