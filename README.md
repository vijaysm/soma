# SOMA — Scalable Operators for Mesh Abstractions

Kokkos-native unstructured mesh infrastructure: read a mesh on the host, mirror a lean copy
(topology, coordinates, and only the adjacencies and fields you ask for) to a device, and
traverse, query, and manipulate mesh and fields with the same kernels on GPUs or host threads.

**Status (v0.1.0):** verified on host backends (Serial, OpenMP) with MPI; the GPU backends can be
configured but are untested in this release.

**Required:** Kokkos >= 5.0. **Optional:** MPI, HDF5 >= 1.14, GoogleTest, MOAB (tests and benchmarks
only), KokkosKernels, ArborX, PETSc.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
```

Without an installed Kokkos, configure with `-DSOMA_FETCH_KOKKOS=ON` to download and build one
(an installed Kokkos always takes precedence). The fetched build enables Serial plus OpenMP, or
Threads when OpenMP is unavailable, and a GPU backend chosen by `SOMA_KOKKOS_GPU`
(`AUTO`, `NONE`, `CUDA`, `HIP`, `SYCL`). Any `Kokkos_*` option you pass wins over these defaults.

Python: with `-DSOMA_ENABLE_PYTHON=ON` (nanobind, NumPy, and mpi4py built against SOMA's MPI),
the package `soma` runs SOMA from scripts under `mpiexec`, with NumPy arrays over the mesh's own
memory; see [Python](docs/getting-started/python.md).

Fortran and C: with `-DSOMA_ENABLE_FORTRAN=ON` (or `-DSOMA_ENABLE_C_API=ON` for C alone), the
Fortran 2008 module `soma` and the C header `soma/c/soma.h` offer the same operations, with
1-based ids and pointers into the mesh's memory from Fortran; see
[Fortran and C](docs/getting-started/fortran.md) and, for codes using iMOAB,
[From iMOAB](docs/getting-started/from-imoab.md).

Documentation: the site under [docs/](docs/index.md) (Sphinx with MyST Markdown, published on
Read the Docs through `.readthedocs.yaml`), with an API reference that Doxygen generates from the
headers' comments. To build it locally (Doxygen on the `PATH`; 1.17.0 is the version checked):

```bash
python3 -m pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

The same pages are converted to the GitHub wiki by `docs/wiki/build_wiki.py`; see
[writing documentation](docs/developer/writing-docs.md).

Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md).

License: BSD-3-Clause (see [LICENSE](LICENSE)).
