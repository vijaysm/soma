# Building and testing

This page describes SOMA's test suite: how to build it, how tests are registered and named, how
to run them with `ctest`, and the special kinds of tests (MOAB reference tests, golden, policy,
packaging and thread-comparison tests). For configuring SOMA itself, see
[installation](../getting-started/installation.md); every option is listed in
[CMake options](../reference/cmake-options.md).

## Building the tests

Turn on `SOMA_ENABLE_TESTS`. The tests use GoogleTest: an installed one is found with
`find_package(GTest)`, or `SOMA_FETCH_GTEST=ON` downloads v1.17.0. Without either, the configure
step stops with a message saying so.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos \
  -DSOMA_ENABLE_TESTS=ON -DSOMA_FETCH_GTEST=ON \
  -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON -DSOMA_ENABLE_EXAMPLES=ON
cmake --build build -j
ctest --test-dir build -j 8
```

Which tests exist depends on the options: MPI adds multi-rank runs; HDF5 adds the `.h5m` and
SOMA-file tests and the `soma-convert` tests (without HDF5, `test_h5m_unavailable` checks that
the I/O functions refuse cleanly); `SOMA_ENABLE_EXAMPLES` adds the example runs
([examples](../getting-started/examples.md)); `SOMA_ENABLE_KOKKOSKERNELS`, `SOMA_ENABLE_PETSC`
and `SOMA_ENABLE_ARBORX` add `test_kokkos_sparse`, `test_petsc` and `test_arborx_index`;
`SOMA_ENABLE_MOAB_REFERENCE` and `SOMA_ENABLE_MOAB_COMPAT` add the tests below.

Tests read the bundled meshes in `data/` (compiled in as `SOMA_TEST_DATA_DIR`).

## The test main

Every test executable links `tests/soma_test_main.cpp`, a shared `main` that constructs a
`soma::ScopeGuard` before GoogleTest parses its arguments (so no test creates a Kokkos View
before Kokkos is initialised, and Kokkos has removed its own options from `argv`), runs the
tests, and takes the largest exit code over all ranks, so a failure on any rank fails the run on
every rank.

## How tests are registered

`tests/CMakeLists.txt` defines

```cmake
# soma_add_test(<name> SOURCES ... [RANKS 1 2 4] [LIBS ...])
```

which builds the executable `<name>` and registers it in one of two ways:

- **Without `RANKS`, or without MPI**: `gtest_discover_tests` registers each GoogleTest case as
  its own ctest, named `Suite.Case` (for example `MeshBuilder.CountsAndDimension` from
  `test_mesh_builder`). Discovery runs when the tests run (`DISCOVERY_MODE PRE_TEST`).
- **With `RANKS` in an MPI build**: the executable is registered once per rank count, as
  `<name>.np<P>`, run as `${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} P ${MPIEXEC_PREFLAGS}
  <exe> ${MPIEXEC_POSTFLAGS}`, with `TIMEOUT 60` and `PROCESSORS P`. Most distributed tests use
  `RANKS 1 2 3 4 5 7 8`, so `test_remap` becomes `test_remap.np1` to `test_remap.np8` (no
  `np6`); a few use fewer (`test_comm` runs at 1, 2, 3, 4 and 8 ranks; `test_refine_moab` at
  one).

The `MPIEXEC_*` variables come from CMake's `FindMPI`. To run eight ranks on a machine with fewer
cores, pass your launcher's flag through `MPIEXEC_PREFLAGS` (for example
`-DMPIEXEC_PREFLAGS=--oversubscribe` with Open MPI).

## Running ctest

```bash
ctest --test-dir build                     # everything
ctest --test-dir build -j 8                # in parallel; PROCESSORS keeps MPI runs within 8 slots
ctest --test-dir build -R test_remap       # tests whose name matches a regular expression
ctest --test-dir build -R '\.np4$'         # every MPI test at four ranks
ctest --test-dir build -R '^MeshBuilder\.' # one discovered GoogleTest suite
ctest --test-dir build -E moab             # all but the MOAB reference tests
ctest --test-dir build --output-on-failure
ctest --test-dir build --timeout 600       # a default timeout for tests that set none
```

A single test executable can also be run directly, with GoogleTest's options after SOMA's and
Kokkos's:

```bash
mpiexec -n 4 build/tests/test_ghosts --gtest_filter='Ghosts.*'
build/tests/test_cg --kokkos-num-threads=4
```

Timeouts set by the suite: 60 s for each MPI run, 300 s for the thread-comparison, packaging,
HDF5-guard and compat-gate tests.

## Kinds of tests

### Unit tests

`tests/unit/test_*.cpp`, one executable per header or feature, from `test_smoke` (Kokkos starts
and runs a reduction on the default execution space) to the algorithm tests. Shared fixtures
(`mesh_fixture.hpp`, `distributed_fixture.hpp`, `refine_meshes.hpp`, ...) live beside them.

### Thread-comparison tests

`test_cg.threads` and `test_fv.threads` run one GoogleTest filter (`Cg.SolutionDigests`,
`Fv.SolutionDigests`) at 1, 3 and 4 host threads (through `KOKKOS_NUM_THREADS`) with
`tests/unit/compare_threads.cmake`. The test prints a `digest reproducible <hex>` line, which
must be the same at every thread count, and may print a `digest plain <hex>` line, which must
differ somewhere when a run had more than one thread, so that the comparison is shown able to
fail. These check that reproducible CG and the upwind steps are the same bit for bit on any
number of threads.

### Golden test

`test_golden_derivation` checks topology derivation incidence by incidence against a separate
serial reference implementation (`tests/golden/serial_derivation.cpp`).

### MOAB reference tests

With `SOMA_ENABLE_MOAB_REFERENCE=ON`, the tests in `tests/reference/` compare SOMA with MOAB
on the same input:

| Test | Needs | Compares |
| --- | --- | --- |
| `test_canonical_moab`, `test_node_tables_moab` | | canonical side tables and high-order node tables against `moab::CN` |
| `test_adjacency_moab` | | derived sub-entities, downward graphs with orientation, upward graphs and bridges against MOAB `get_adjacencies` on every bundled `.h5m` mesh |
| `test_h5m_moab` | HDF5 | `read_h5m` against MOAB `load_file`, and MOAB reading what `write_h5m` and `convert_mesh_file` write |
| `test_ghosts_moab` | MPI | ghost layers against MOAB's `exchange_ghost_cells` |
| `test_parallel_parity_moab` | MPI, HDF5 | owned, shared and ghost entities and halo updates against MOAB's `READ_PART` |
| `test_laplacian_moab` | MPI, HDF5 | Laplacian smoothing against MOAB's smoother (`bench/moab_smooth.hpp`) |
| `test_fem_moab` | MPI, HDF5 | `FemSpace`'s sparsity against MOAB's adjacencies |
| `test_fv_moab` | MPI, HDF5 | `FvSpace`'s upwind steps against an upwind scheme on MOAB's faces |
| `test_remap_moab`, `test_remap_sphere_moab` | MPI, HDF5 | remap overlap areas against MOAB's intersection in the plane (`Intx2MeshInPlane`) and on the sphere |
| `test_refine_moab` | MPI, HDF5 | uniform refinement against MOAB's `NestedRefine`, one rank |

With `SOMA_ENABLE_MOAB_COMPAT` as well, `test_compat_range_moab` and (with HDF5)
`test_compat_core_moab` compare `soma::compat` with MOAB, and the `compat.gate.*` tests build
ten of MOAB's own examples twice, against MOAB and against `soma::compat_dropin`, run both on
the same input, and require the same standard output and files that read the same in MOAB
(`tests/compat/compare_example.cmake`). The example sources are read from MOAB's
`share/examples` (`SOMA_MOAB_EXAMPLES_DIR`); without them the gate is off. `test_compat_error`
and `test_compat_prog_options` need only `SOMA_ENABLE_MOAB_COMPAT`.

### Policy tests

CMake scripts in `tests/policy/` that check the sources rather than run code:

- `policy.mpi_seam` fails if `SOMA_ENABLE_MPI` appears in any source outside the `Comm` seam
  (`include/soma/parallel/comm.hpp`, `src/parallel/comm_mpi.cpp`, `src/parallel/comm_serial.cpp`,
  `src/runtime/scope_guard.cpp`), so MPI-only code paths cannot hide where a build without MPI
  never compiles them.
- `policy.device_lambdas` fails if a GoogleTest body (`TEST`, `TEST_F`, `TEST_P`, `TYPED_TEST`)
  contains `KOKKOS_LAMBDA`: those bodies are private member functions, where CUDA refuses
  extended lambdas. Kernels belong in named free functions.
- `policy.hdf5_guard`, registered only when `SOMA_TEST_PARALLEL_HDF5_ROOT` names a parallel
  HDF5, configures SOMA with it and `SOMA_ENABLE_MPI=OFF` and requires the configure step to
  refuse it with its explanatory message.

### Packaging test

`packaging.find_package` (registered when `SOMA_INSTALL` is on) installs the build tree into an
empty prefix under the build directory, checks that the library, every header, `config.hpp` and
the package files are there (and Kokkos's, for a fetched Kokkos), then configures, builds and
runs the consumer in `tests/packaging/` against that prefix through `find_package(soma)`. The
consumer must print `OK soma consumer: hex has 6 faces, 1 rank(s)`, and in MPI builds also run at
two ranks. With a fetched Kokkos it also checks that the consumer used the Kokkos in the SOMA
prefix, even with another Kokkos (`SOMA_PACKAGING_SHADOW_PREFIX`) first on the search path. It
passes the build type through, so it works with multi-configuration generators via
`ctest -C <config>`.

`packaging.example_app` installs the build tree into another empty prefix the same way, then
configures, builds and runs the application in `examples/app/` against it (see
[examples](../getting-started/examples.md)) with `--n 6`, writing a SOMA file when
`SOMA_ENABLE_HDF5` is on and a legacy VTK file otherwise. It must print
`OK soma app: 1 rank(s), cells 216 -> 1728`, and in MPI builds the same at two ranks.

### Example and tool tests

`example_*` and `soma_convert.*` run the example programs and `soma-convert` on bundled meshes
and check their printed results; see [examples](../getting-started/examples.md).

## Other build configurations

- `SOMA_DEBUG=ON` turns on device-side bounds and contract checks (for example, a kernel reading
  an adjacency graph the mesh does not hold aborts).
- `SOMA_LO_64=ON` and `SOMA_OFFSET_64=ON` build with 64-bit local ordinals and CSR offsets.
- `SOMA_ENABLE_MOAB_REFERENCE=ON` with `CMAKE_BUILD_TYPE=Release` builds the
  [benchmarks](benchmarks.md).

SOMA's CMake has no sanitizer or coverage option; flags for those can be passed through the
usual `CMAKE_CXX_FLAGS`. `-Wall -Wextra` are applied to SOMA's own targets for the compilers that
accept them. `test_fem_elements` is compiled with `-ffp-contract=on` under Clang and GCC, since
its bit-for-bit comparisons hold only where expressions are contracted alike.
