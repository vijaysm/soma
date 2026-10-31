# SOMA — Scalable Operators for Mesh Abstractions

Kokkos-native unstructured mesh infrastructure: read a mesh on the host, mirror a lean copy
(topology, coordinates, and only the adjacencies and fields you ask for) to a device, and
traverse, query, and manipulate mesh and fields with the same kernels on GPUs or host threads.

**Required:** Kokkos >= 5.0. **Optional:** MPI, HDF5, GoogleTest, MOAB (tests and benchmarks
only), KokkosKernels, ArborX, PETSc.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
```

Without an installed Kokkos, configure with `-DSOMA_FETCH_KOKKOS=ON` to download and build one
(an installed Kokkos always takes precedence). The fetched build enables Serial plus OpenMP, or
Threads when OpenMP is unavailable, and a GPU backend chosen by `SOMA_KOKKOS_GPU`
(`AUTO`, `NONE`, `CUDA`, `HIP`, `SYCL`). Any `Kokkos_*` option you pass wins over these defaults.

Design: [docs/plans/2026-10-30-soma-design.md](docs/plans/2026-10-30-soma-design.md).
Implementation plan: [docs/plans/2026-10-30-soma-implementation-plan.md](docs/plans/2026-10-30-soma-implementation-plan.md).
Current state and how to resume: [docs/CONTINUITY.md](docs/CONTINUITY.md).

License: BSD-3-Clause (see [LICENSE](LICENSE)).
