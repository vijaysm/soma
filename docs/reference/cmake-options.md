# CMake options

This page lists every CMake option and cache variable SOMA defines, with its default and meaning,
the variables that point CMake at dependencies, and the macros the configure step writes into
`soma/config.hpp`. For how to use them together, see [installation](../getting-started/installation.md)
and [building and testing](../developer/building-and-testing.md).

## Features and dependencies

Defined in the top-level `CMakeLists.txt`. Every dependency option is `OFF` by default; one that
is turned on but whose dependency cannot be found is a configure error.

| Option | Default | Meaning |
| --- | --- | --- |
| `SOMA_ENABLE_MPI` | `OFF` | Multi-rank distribution through MPI (`find_package(MPI COMPONENTS CXX)`). Off, `soma::Comm` is a one-rank serial implementation. |
| `SOMA_ENABLE_HDF5` | `OFF` | SOMA files and MOAB `.h5m` files through HDF5 1.14 or newer; also builds `soma-convert`. A parallel HDF5 needs `SOMA_ENABLE_MPI`. Off, the HDF5 readers and writers throw `std::runtime_error` naming this option. |
| `SOMA_ENABLE_TESTS` | `OFF` | Build the test suite (needs GoogleTest). |
| `SOMA_FETCH_GTEST` | `OFF` | Download GoogleTest (v1.17.0) when no installed copy is found. |
| `SOMA_FETCH_KOKKOS` | `OFF` | Download and build Kokkos when no installed Kokkos 5.0 or newer is found; ignored when one is. |
| `SOMA_KOKKOS_VERSION` | `5.1.0` | The Kokkos tag fetched with `SOMA_FETCH_KOKKOS`. |
| `SOMA_KOKKOS_GPU` | `AUTO` | GPU backend of a fetched Kokkos: `AUTO`, `NONE`, `CUDA`, `HIP` or `SYCL`. `AUTO` picks one from the toolchain found. |
| `SOMA_ENABLE_MOAB_REFERENCE` | `OFF` | Build the tests that compare SOMA with MOAB, and the benchmarks (needs MOAB). |
| `SOMA_ENABLE_EXAMPLES` | `OFF` | Build the example programs (needs `SOMA_ENABLE_HDF5`). |
| `SOMA_ENABLE_APPS` | `OFF` | Build the applications under `apps/`: the [shallow-water atmosphere](../apps/swe-atm.md) `swe_atm` (reference files and SOMA output need `SOMA_ENABLE_HDF5`) and the [barotropic ocean](../apps/swe-ocean.md) `swe_ocean` (real bathymetry and SOMA files need `SOMA_ENABLE_HDF5`). |
| `SOMA_ENABLE_DOC_FIGURES` | `OFF` | Build `soma_figure_data`, which makes the data of the documentation's figures (needs `SOMA_ENABLE_MPI` and `SOMA_ENABLE_HDF5`; see [making figures](../developer/making-figures.md)). |
| `SOMA_ENABLE_MOAB_COMPAT` | `OFF` | Build `soma::compat`, MOAB's API over SOMA meshes (targets `soma::compat` and `soma::compat_dropin`). |
| `SOMA_ENABLE_KOKKOSKERNELS` | `OFF` | KokkosSparse views of SOMA matrices. Needs an installed Kokkos, the one KokkosKernels was built with. |
| `SOMA_ENABLE_PETSC` | `OFF` | PETSc matrices and vectors from SOMA's. Needs `SOMA_ENABLE_MPI` and PETSc 3.19 or newer with real double scalars. |
| `SOMA_ENABLE_ARBORX` | `OFF` | ArborX's bounding volume hierarchy as the remap's search. Needs an installed Kokkos, the one ArborX was built with. |

## Types, checks and installation

| Option | Default | Meaning |
| --- | --- | --- |
| `SOMA_LO_64` | `OFF` | 64-bit local ordinals (`soma::LO`; 32-bit otherwise). ABI-breaking. |
| `SOMA_OFFSET_64` | `OFF` | 64-bit CSR offsets (`soma::Offset`; 32-bit otherwise). ABI-breaking. |
| `SOMA_DEBUG` | `OFF` | Device-side bounds and contract checks; for example a `MeshView` kernel reading an adjacency graph the mesh does not hold aborts. |
| `SOMA_GPU_AWARE_MPI` | `AUTO` | Whether device buffers may be passed to MPI: `AUTO`, `ON` or `OFF`. `AUTO` resolves to `ON` when MPI is enabled and Kokkos has a GPU backend, `OFF` otherwise. `ON` needs `SOMA_ENABLE_MPI`. Recorded only, as `SOMA_GPU_AWARE_MPI_DEFAULT` in `soma/config.hpp`: no exchange reads it yet, and messages are staged through host memory. |
| `SOMA_INSTALL` | `ON` when SOMA is the top-level project, else `OFF` | Generate install and export rules. Also enables the `packaging.find_package` test. |
| `SOMA_LARGE_DATA_DIR` | empty | Directory of large meshes. The benchmarks `bench_adjacency`, `bench_h5m_read` and `bench_h5m_write` add the large meshes they find there. |
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | Set by SOMA for single-configuration generators when you give none. |

## Test-only variables

Defined in `tests/CMakeLists.txt` and `tests/compat/examples.cmake`; they only matter with
`SOMA_ENABLE_TESTS=ON`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOMA_TEST_PARALLEL_HDF5_ROOT` | empty | A parallel HDF5 installation. When set, the test `policy.hdf5_guard` checks that configuring with it and `SOMA_ENABLE_MPI=OFF` is refused. |
| `SOMA_PACKAGING_SHADOW_PREFIX` | `/opt/kokkos` | With a fetched Kokkos, the prefix of another Kokkos that the packaging test puts first on the consumer's search path, to check that the Kokkos installed with SOMA still wins. Skipped if the directory does not exist. Defined when `SOMA_INSTALL` is on. |
| `SOMA_MOAB_EXAMPLES_DIR` | found from `MOAB_DIR` | MOAB's `share/examples` directory, for the `compat.gate.*` tests (with `SOMA_ENABLE_MOAB_COMPAT` and `SOMA_ENABLE_MOAB_REFERENCE`). When empty, SOMA looks for `share/examples` three directories above `MOAB_DIR`; without it the gate is off. |

`SOMA_PROBE_NVCC` and `SOMA_PROBE_HIP`, when defined, replace SOMA's search for `nvcc` and for a
HIP toolchain in `SOMA_KOKKOS_GPU=AUTO`; they exist to test that logic.

## Options of the fetched Kokkos

With `SOMA_FETCH_KOKKOS`, SOMA sets these Kokkos options unless you set them yourself; anything
you pass on the command line wins:

| Kokkos option | SOMA's choice |
| --- | --- |
| `Kokkos_ENABLE_SERIAL` | `ON` |
| `Kokkos_ENABLE_OPENMP` / `Kokkos_ENABLE_THREADS` | OpenMP when CMake finds a usable OpenMP, otherwise Threads |
| `Kokkos_ENABLE_CUDA` / `HIP` / `SYCL` | as `SOMA_KOKKOS_GPU` decides; left alone if you set any of the three |
| `Kokkos_ARCH_NATIVE` | `ON` unless you set any `Kokkos_ARCH_*` option |

## Finding dependencies

SOMA finds its dependencies with CMake's own mechanisms, so the usual hint variables apply. Each
`<Package>_ROOT` may also be an environment variable.

| Dependency | Found by | Hint |
| --- | --- | --- |
| Kokkos | `find_package(Kokkos 5.0)` | `Kokkos_ROOT`, `Kokkos_DIR` or `CMAKE_PREFIX_PATH` |
| MPI | `find_package(MPI)` (CMake's `FindMPI`) | `MPI_HOME`, or the MPI compiler wrappers on `PATH`; `MPIEXEC_EXECUTABLE`, `MPIEXEC_PREFLAGS` for the test launcher |
| HDF5 | `find_package(HDF5 COMPONENTS C)` (CMake's `FindHDF5`) | `HDF5_ROOT` |
| GoogleTest | `find_package(GTest)` | `GTest_ROOT` or `CMAKE_PREFIX_PATH`; or `SOMA_FETCH_GTEST=ON` |
| MOAB | `find_package(MOAB)` | `MOAB_DIR`, `MOAB_ROOT` or `CMAKE_PREFIX_PATH` |
| KokkosKernels | `find_package(KokkosKernels)` | `KokkosKernels_ROOT` |
| ArborX | `find_package(ArborX)` | `ArborX_ROOT` |
| PETSc | `pkg_check_modules(PETSc>=3.19)` on its pkg-config file | `PETSC_DIR` (an install prefix or source tree) and `PETSC_ARCH` (an in-place build), as cache or environment variables |
| OpenMP (AppleClang) | Homebrew `libomp`, when `FindOpenMP` fails | `brew install libomp`; `CMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` turns the fallback off |

`HDF5_ROOT`, the KokkosKernels and ArborX directories and the PETSc pkg-config directory found
at configure time are recorded in the installed `somaConfig.cmake`, so consumers find the same
installations.

## Macros in `soma/config.hpp`

The configure step generates `soma/config.hpp` (installed with the headers), which records the
options that change the ABI or the public headers. Each `SOMA_*` switch below is defined to 0 or
1.

| Macro | From |
| --- | --- |
| `SOMA_VERSION_MAJOR`, `SOMA_VERSION_MINOR`, `SOMA_VERSION_PATCH` | `project(soma VERSION ...)` |
| `SOMA_ENABLE_MPI` | `SOMA_ENABLE_MPI` |
| `SOMA_LO_64`, `SOMA_OFFSET_64` | the options of the same name |
| `SOMA_DEBUG` | `SOMA_DEBUG` |
| `SOMA_ENABLE_KOKKOSKERNELS`, `SOMA_ENABLE_PETSC`, `SOMA_ENABLE_ARBORX` | the options of the same name |
| `SOMA_GPU_AWARE_MPI_DEFAULT` | `SOMA_GPU_AWARE_MPI`, resolved (recorded only) |

`SOMA_ENABLE_HDF5` is not a macro: without HDF5 the same functions exist and throw.
