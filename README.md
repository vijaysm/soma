# SOMA — Scalable Operators for Mesh Abstractions

Kokkos-native unstructured mesh infrastructure: read a mesh on the host, mirror a lean copy
(topology, coordinates, and only the adjacencies and fields you ask for) to a device, and
traverse, query, and manipulate mesh and fields with the same kernels on GPUs or host threads.

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
