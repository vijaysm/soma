# Installation

This page lists SOMA's requirements, shows how to configure and build it with CMake against an
installed Kokkos or one SOMA downloads, how to turn on the optional dependencies, and how to
install SOMA and use it from another CMake project. Every option named here is listed in the
[CMake options reference](../reference/cmake-options.md).

## Requirements

| Requirement | Version | Needed for |
| --- | --- | --- |
| CMake | 3.24 or newer | configuring (`cmake_minimum_required(VERSION 3.24)`) |
| C++ compiler | C++20 | everything; the VTK writer also needs floating-point `std::to_chars` (libstdc++ 11+, libc++ 14+, macOS 13.3+) |
| Kokkos | 5.0 or newer | everything (the only required dependency) |
| MPI | any with a C++ CMake target | `SOMA_ENABLE_MPI`: more than one rank |
| HDF5 | 1.14 or newer | `SOMA_ENABLE_HDF5`: SOMA and MOAB `.h5m` files, `soma-convert`, the examples |
| GoogleTest | | `SOMA_ENABLE_TESTS` (or downloaded with `SOMA_FETCH_GTEST`) |
| MOAB | | `SOMA_ENABLE_MOAB_REFERENCE`: reference tests and benchmarks only |
| KokkosKernels | built with the same installed Kokkos | `SOMA_ENABLE_KOKKOSKERNELS` |
| ArborX | built with the same installed Kokkos | `SOMA_ENABLE_ARBORX` |
| PETSc | 3.19 or newer, real double scalars, shared libraries | `SOMA_ENABLE_PETSC` (also needs MPI) |

Every optional dependency is off by default, and one that is requested but cannot be found is a
configure error: nothing is silently disabled.

## Configuring with an installed Kokkos

Point CMake at a Kokkos 5 installation with `CMAKE_PREFIX_PATH` or `Kokkos_ROOT`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
```

SOMA uses the backends that Kokkos was built with. If that Kokkos has no parallel host backend
(OpenMP, Threads or HPX), the configure step warns that host code will run serially.

Single-configuration generators get `CMAKE_BUILD_TYPE=RelWithDebInfo` unless you choose another.
At the end of the configure step a status line reports the Kokkos version and source
(`installed` or `fetched`), its host and GPU backends, and the state of each SOMA option.

## Configuring with a downloaded Kokkos

Without an installed Kokkos, `-DSOMA_FETCH_KOKKOS=ON` downloads and builds one inside the SOMA
build (the tag in `SOMA_KOKKOS_VERSION`, `5.1.0` by default):

```bash
cmake -S . -B build -DSOMA_FETCH_KOKKOS=ON
```

An installed Kokkos always takes precedence: if `find_package(Kokkos 5.0)` succeeds,
`SOMA_FETCH_KOKKOS` is ignored with a status message. To force the download, hide the installed
one (remove it from `CMAKE_PREFIX_PATH` and `Kokkos_ROOT`, or pass
`-DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON`).

The fetched Kokkos is configured as follows; any `Kokkos_*` option you pass yourself wins over
these choices:

- the Serial backend is on;
- OpenMP is the host backend when CMake finds a usable OpenMP, otherwise Threads;
- `Kokkos_ARCH_NATIVE` is on unless you set a `Kokkos_ARCH_*` option;
- the GPU backend follows `SOMA_KOKKOS_GPU`.

### GPU backends

`SOMA_KOKKOS_GPU` takes `AUTO` (the default), `NONE`, `CUDA`, `HIP` or `SYCL`, and applies only
to a fetched Kokkos. With `AUTO` SOMA looks for a toolchain: `nvcc` selects CUDA; `hipcc`,
`amdclang++`, `ROCM_PATH` or `HIP_PATH` select HIP; an `icpx` C++ compiler selects SYCL;
otherwise no GPU backend is enabled. Setting any of `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`
or `Kokkos_ENABLE_SYCL` yourself leaves the GPU choice to you.

The C++ compiler must be able to build the backend chosen, or the configure step fails (it never
falls back to the CPU silently):

| Backend | C++ compiler |
| --- | --- |
| CUDA | a CUDA-capable `clang++`, or the `bin/nvcc_wrapper` of a Kokkos source tree or installation (the fetched Kokkos's wrapper does not exist yet when the compiler is chosen) |
| HIP | `hipcc` or `amdclang++` |
| SYCL | `icpx` |

For example:

```bash
cmake -S . -B build -DSOMA_FETCH_KOKKOS=ON -DSOMA_KOKKOS_GPU=HIP -DCMAKE_CXX_COMPILER=hipcc
cmake -S . -B build -DSOMA_FETCH_KOKKOS=ON -DSOMA_KOKKOS_GPU=NONE   # CPU only
```

Kokkos detects the GPU architecture only when a GPU is visible at configure time; on a login
node without one, set the matching `Kokkos_ARCH_*` option.

## Enabling optional features

Each feature is a `SOMA_ENABLE_*` option. A typical MPI build with file I/O and tests:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos \
  -DSOMA_ENABLE_MPI=ON \
  -DSOMA_ENABLE_HDF5=ON -DHDF5_ROOT=/path/to/hdf5 \
  -DSOMA_ENABLE_TESTS=ON -DSOMA_FETCH_GTEST=ON \
  -DSOMA_ENABLE_EXAMPLES=ON
```

Constraints the configure step enforces:

- `SOMA_ENABLE_HDF5` needs HDF5 1.14 or newer, found through CMake's `FindHDF5` (set
  `HDF5_ROOT`). A parallel HDF5 links MPI, so it is accepted only with `SOMA_ENABLE_MPI=ON`;
  without MPI, point `HDF5_ROOT` at a serial HDF5.
- `SOMA_ENABLE_EXAMPLES` needs `SOMA_ENABLE_HDF5`. The `soma-convert` tool is built whenever
  HDF5 is on.
- `SOMA_ENABLE_KOKKOSKERNELS` and `SOMA_ENABLE_ARBORX` need an installed Kokkos, the one those
  libraries were built with; with a fetched Kokkos they are configure errors. Set
  `KokkosKernels_ROOT` and `ArborX_ROOT`.
- `SOMA_ENABLE_PETSC` needs `SOMA_ENABLE_MPI` and `PETSC_DIR` (plus `PETSC_ARCH` for an
  in-place PETSc build), as CMake variables or in the environment. PETSc is found through its
  pkg-config file and must be 3.19 or newer with real double scalars; only shared PETSc
  libraries link correctly, since pkg-config's `Libs.private` are not linked.
- `SOMA_ENABLE_MOAB_REFERENCE` finds MOAB with `find_package(MOAB)` and links its imported
  target. A PETSc built with its own MOAB cannot be combined with the MOAB reference, because its
  headers would shadow the reference MOAB's.
- `SOMA_GPU_AWARE_MPI=ON` needs `SOMA_ENABLE_MPI`.

`SOMA_LO_64` and `SOMA_OFFSET_64` widen the local ordinal `soma::LO` and the CSR offset
`soma::Offset` to 64 bits. They change the ABI, are recorded in the generated
`soma/config.hpp`, and must match between SOMA and everything built against it.

## Apple and OpenMP

AppleClang ships without OpenMP, so a Kokkos built with its OpenMP backend cannot be found
unaided on macOS. When the compiler is AppleClang and CMake finds no OpenMP, SOMA looks for
Homebrew's `libomp` (`brew --prefix libomp`, `/opt/homebrew/opt/libomp`,
`/usr/local/opt/libomp`) and seeds CMake's `FindOpenMP` from it. Install it with:

```bash
brew install libomp
```

The installed package configuration replays these settings, so projects that use an installed
SOMA configure without finding OpenMP again. Pass `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` to
turn the fallback off.

## Installing

Install rules are generated when `SOMA_INSTALL` is on, which is the default when SOMA is the
top-level project (a parent project that adds SOMA with `add_subdirectory` must ask for them):

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DCMAKE_INSTALL_PREFIX=/opt/soma
cmake --build build -j
cmake --install build
```

This installs:

- the library `libsoma` and the headers under `include/soma/`, with the generated
  `soma/config.hpp`;
- the CMake package `soma` (`somaConfig.cmake`, `somaConfigVersion.cmake`,
  `somaTargets.cmake`) under `lib/cmake/soma`, exporting `soma::soma`;
- with `SOMA_ENABLE_MOAB_COMPAT`, the targets `soma::compat` and `soma::compat_dropin` and the
  compat headers (left out otherwise);
- with `SOMA_ENABLE_HDF5`, the `soma-convert` tool under `bin`;
- with a fetched Kokkos, Kokkos itself, into the same prefix.

The package version is compatible within a minor version (`SameMinorVersion`), since
`SOMA_LO_64` and `SOMA_OFFSET_64` can change the ABI.

## Using SOMA from another CMake project

`find_package(soma)` finds the installed package and its dependencies; link `soma::soma`. This
is the consumer from `tests/packaging/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(soma_packaging_test LANGUAGES CXX)

find_package(soma REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE soma::soma)
```

Configure it with the SOMA prefix on the search path:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/soma
```

`somaConfig.cmake` finds again what SOMA was built with:

- Kokkos 5.0 or newer. When SOMA fetched Kokkos, the config points `Kokkos_DIR` at the Kokkos
  installed in the same prefix (unless you set `Kokkos_DIR`), so another Kokkos on the search
  path cannot shadow it.
- MPI, KokkosKernels and ArborX when they were enabled; the latter two at the directories SOMA
  found them in, unless you set `KokkosKernels_DIR` or `ArborX_DIR`.
- PETSc through the pkg-config directory SOMA used, unless the consumer already defines
  `PkgConfig::PETSC`.
- HDF5 when it was enabled (SOMA links it privately, but the static library still needs it),
  at the `HDF5_ROOT` given when SOMA was configured, unless you set one.

The variables `SOMA_ENABLE_MPI`, `SOMA_ENABLE_HDF5`, `SOMA_LO_64`, `SOMA_OFFSET_64`,
`SOMA_ENABLE_KOKKOSKERNELS`, `SOMA_ENABLE_ARBORX`, `SOMA_ENABLE_PETSC` and `SOMA_KOKKOS_SOURCE`
are set after `find_package(soma)`, so a consumer can test them. In C++, the options that change
the ABI or the public headers are macros in `soma/config.hpp` (see
[CMake options](../reference/cmake-options.md)).

The ctest `packaging.find_package` checks all this: it installs the build tree into an empty
prefix, configures, builds and runs this consumer (at two ranks too in MPI builds); see
[building and testing](../developer/building-and-testing.md). For a fuller application built
this way, see `examples/app/` in [examples](examples.md).
