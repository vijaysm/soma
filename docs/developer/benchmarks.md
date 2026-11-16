# Benchmarks

The programs in `bench/` time SOMA against MOAB, its only baseline, on the same input, and check
that both computed the same thing before reporting a time. This page says what each one measures,
how to build and run it, and how to read its output. No timings are quoted here: run them on your
machine.

## Building

The benchmarks are built with `SOMA_ENABLE_MOAB_REFERENCE=ON` (which needs MOAB), some only with
further options. Build them in `Release` for numbers worth comparing:

```bash
cmake -S . -B build-bench -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/moab" \
  -DCMAKE_BUILD_TYPE=Release -DSOMA_ENABLE_MOAB_REFERENCE=ON \
  -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON [-DSOMA_ENABLE_MOAB_COMPAT=ON]
cmake --build build-bench -j
```

| Program | Built when | Runs |
| --- | --- | --- |
| `bench_adjacency` | always (with the MOAB reference) | serial, host threads |
| `bench_h5m_read`, `bench_h5m_write` | `SOMA_ENABLE_HDF5` | serial |
| `bench_edit` | `SOMA_ENABLE_HDF5` and `SOMA_ENABLE_MOAB_COMPAT` | serial |
| `bench_transport`, `bench_merge` | `SOMA_ENABLE_MPI` | `mpiexec -n P` |
| `bench_parallel` | `SOMA_ENABLE_MPI` and `SOMA_ENABLE_HDF5` | `mpiexec -n P` |

The executables are under `build-bench/bench/`. The build type and compiler are compiled into
each one and printed with its results.

## Common behaviour and output

Every benchmark uses the harness in `bench/harness.hpp`:

- Each measurement runs once untimed as a warmup, then `--repeats N` times (default 5); only the
  timed function is measured, after an untimed setup each time.
- It reports the median and the median absolute deviation (MAD) over the repeats, never a total.
  Across ranks, the reported time is the largest rank's median (the slowest rank sets the time of
  a bulk-synchronous step), with the smallest beside it; `ns_per_op` divides that time by the
  operations summed over ranks.
- It prints the environment first: mesh, build type, compiler, Kokkos host backend, ranks, host
  threads and repeats. With more than one thread it notes that SOMA runs threaded while MOAB is
  serial, so `threads=1` is the data-structure comparison.
- Every result is checked against MOAB's (or against closed forms); on a mismatch the program
  exits with status 2.

Without `--json`, each timing is one line, followed for a SOMA and MOAB pair by the ratio:

```text
[env] mesh=... build_type=Release compiler=... backend=OpenMP ranks=1 threads=8 repeats=5 (plus 1 discarded warmup)
[bench] <benchmark> workflow=<workflow> median_sec=... mad_sec=... min_rank_sec=... ns_per_op=... ops=... repeats=5
[bench] <benchmark> speedup(moab/soma)=... noise=...
```

`noise` is the larger of the two relative spreads (MAD over median); above 0.10 the line ends in
`WARNING:high-variance`, since a ratio within the spread is not a result.

With `--json`, each timing is one JSON object per line, with the keys `benchmark`, `workflow`,
`mesh`, `build_type`, `compiler`, `backend`, `ranks`, `threads`, `repeats`, `median_sec`,
`mad_sec`, `min_rank_sec`, `ns_per_op` and `ops`; the environment and ratio lines are not
printed. Lines can be appended to one file and filtered with `grep` or `jq`.

Host threads are set as for any Kokkos program, for example `--kokkos-num-threads=8` or
`OMP_NUM_THREADS`.

## The programs

### bench_adjacency

```text
bench_adjacency [--json] [--repeats N] [mesh.h5m ...]
```

Topology derivation and adjacency queries: SOMA on host threads against MOAB's
`get_adjacencies`. Its workflows include `derive` (creating the edges and faces), and the
vertex-to-element graph and bridges through vertices (`vertex_to_element`,
`vertex_via_element`, `element_via_vertex`). Each checks that both sides found graphs of the
same sizes. Without meshes it runs the bundled `64bricks_1khex.h5m`, `hex_2048.h5m`,
`moab_i.h5m` and `recMeshOcn128.h5m`, plus large meshes found in `SOMA_LARGE_DATA_DIR`. The
meshes are loaded by MOAB and handed to SOMA's `MeshBuilder`, so HDF5 is not needed in SOMA.

`bench/run_adjacency.sh` runs it at 1, 2, 4, ... threads up to the core count (or
`MAX_THREADS`), one process per count since Kokkos fixes its thread count at initialisation,
with `--json` on; JSON lines go to standard output:

```bash
bench/run_adjacency.sh build-bench [bench_adjacency arguments] > adjacency.json
```

### bench_h5m_read

```text
bench_h5m_read [--json] [--repeats N] [--box N] [mesh.h5m ...]
```

Reading a `.h5m` file: SOMA's `read_h5m` against MOAB's `load_file`, both serial, each read
checked to find the same vertex and element counts. SOMA reads every fixed-length tag on
vertices and elements. The `soma_sfc` workflow reads in `Ordering::type_major_sfc`; `soma_build`
reads without tags and builds vertex-to-element, side-to-element and element-to-element graphs;
`soma_cached` reads the same graphs from a copy that `write_h5m` cached them in. Without meshes
it reads the bundled ones, large ones from `SOMA_LARGE_DATA_DIR`, and a box of $N^3$ hexes
(`--box`, default 64) that MOAB writes to a temporary directory first.

### bench_h5m_write

```text
bench_h5m_write [--json] [--repeats N] [--box N] [--compression L] [mesh.h5m ...]
```

Writing a `.h5m` file: SOMA's `write_h5m` against MOAB's `write_file`, both serial, each writing
the mesh it read from the same file. Both files are loaded by MOAB and checked to hold the input's
entity counts per dimension, and the file sizes are printed beside the times. SOMA also writes at
deflate level `L` (default 4; 0 skips it). The meshes are chosen as for `bench_h5m_read`.

### bench_edit

```text
bench_edit [--json] [--repeats N] [--box n] [--fractions f,f,...]
```

Queries on edited meshes. A fraction $f$ of the hexes of an $n^3$ box (default $n = 40$;
fractions default `0,0.01,0.1,0.5`) is removed and created again with the same vertices, so the
edited mesh holds the same hexes as the compact one, $f$ of them masked and $f$ appended. It
times:

- native: connectivity, the vertex-to-hex graph and a per-hex field read and written after
  `MeshEdit::apply`, against the compact mesh; and `apply` and `compress` against building the
  box;
- compat: `tag_get_data` and `tag_set_data` (in bulk and one handle at a time),
  `get_connectivity` and vertex-to-hex `get_adjacencies` on a `soma::compat::Core` against a
  `moab::Core` that made the same calls.

Every query is checksummed against its compact counterpart and MOAB's. It ends with the removed
fraction at which the field write costs twice its compact cost, printed as a value for this
machine's `--soma-compress-threshold`:

```text
[bench] field write passes 2x compact at a removed fraction of ...: --soma-compress-threshold=...
```

### bench_transport

```text
mpiexec -n P bench_transport [--json] [--repeats N] [--records n]
```

`soma::transport::route` against MOAB's crystal router (`gs_transfer` on a `TupleList`). Every
rank sends `n` records (default 100000) of three integers, to destinations drawn uniformly from
all ranks (`dense`) or to the four neighbours at rank $\pm 1$ and $\pm 2$ (`sparse`, a halo's
pattern). SOMA routes them with each of its strategies; every result is checked against MOAB's
by count and checksum on every rank.

### bench_merge

```text
mpiexec -n P bench_merge [--json] [--repeats N] [--n edge]
```

Coordinate matching against MOAB's `ParallelMergeMesh` and `MergeMesh`. Each rank holds a slab
of whole layers of an $n^3$ box of hexes (default $n = 40$), its vertices numbered on their own,
as parts meshed or refined apart are. It times `match_points` (exactly, and within `1e-9` with
each rank's $z$ offset slightly), merging the slabs into one distributed mesh within `1e-9` two
ways (`build_merged_part`, and `merge_vertices`, `number_elements` and `build_local_part`)
against `ParallelMergeMesh::merge`, and on one rank a soup of hexes each with its own eight
vertices merged against `MergeMesh::merge_entities`. The owned vertices and hexes must add up to
the box's.

### bench_parallel

```text
mpiexec -n P bench_parallel [--json] [--repeats N] [--n edge] [--layers L] [--parts K] [--dir path]
```

Distributed meshes against MOAB's `ParallelComm` at P ranks, on an $n^3$ box of hexes (default
$n = 32$) in `K` `PARALLEL_PARTITION` sets of whole layers (default 64, at least P), written once
by MOAB on rank 0 into a fresh subdirectory of `--dir` (default the system temporary directory),
which is removed at the end. Its sections, each timing SOMA and, where MOAB has a counterpart,
MOAB on the same file:

| Section | SOMA | MOAB |
| --- | --- | --- |
| read | `read_h5m` over the ranks (independent, and collective with a parallel HDF5) | `load_file` with `READ_PART`, shared entities resolved |
| sharing | `build_local_part` from each rank's hexes | `resolve_shared_ents` |
| ghosts | `add_ghosts`, one layer or `--layers` | `exchange_ghost_cells` |
| halo | an update of a vertex field of 1 and 3 doubles (`HaloExchange`, fused and split into begin and end) | `exchange_tags` |
| smooth | 10 Jacobi sweeps of `laplacian_smooth`, setup included | the smoother in `bench/moab_smooth.hpp` |
| poisson | `FemSpace` for each assembly, the stiffness matrix gathered, by atomics and colored, Jacobi CG to `1e-10`, plain and reproducible | none |
| advect | `FvSpace` and 10 forward Euler upwind steps, plain and reproducible | none |
| remap | `RemapOperator` between plane grids of about $n^3$ cells, built and applied; hexes onto tets and onto hexes; by ArborX too when SOMA has it | `Intx2MeshInPlane`, at one rank |
| remap_sphere | `RemapOperator` between cubed spheres by gnomonic projection and by great circles | `Intx2MeshOnSphere`, at one rank |
| refine | `refine_uniform` of hexes and tets; `refine_marked` of the cells of odd gid (with hanging vertices, conforming, tets bisected) | `NestedRefine` of the hexes, at one rank |
| write, restart | `write_soma`, `read_soma` | `write_file` with `PARALLEL=WRITE_PART`, `load_file` with `READ_PART` |

Every result is checked: owned counts add up to the box's, ghosts hold their owners' values after
each update, the three assemblies agree, CG converges, advection keeps a constant state constant,
the remap keeps the total, and refinement makes the expected cells.

`bench/run_parallel.sh` runs it over rank counts and box sizes, one JSON file per case:

```bash
bench/run_parallel.sh BUILD OUT "1 2 4 8" "32 64" [repeats] [mpiexec]
```

It writes `OUT/<P>_<n>_t<T>.json` (standard error beside it in `.err`), running each case at one
thread per rank (the data-structure comparison) and at `SOMA_BENCH_THREADS` threads (default 1),
with `OMP_NUM_THREADS` set and `OMP_PROC_BIND=false`. `repeats` defaults to 5 and the launcher
to `$MPIEXEC` or `mpiexec`. It prints one status line per case and exits non-zero if any run
failed (status 2 if SOMA and MOAB disagreed).

At scale, the script's header asks for a build with MPI, a parallel HDF5, the MOAB reference and
`Release`; a run from a job script with `MPIEXEC` set to the system's launcher (with its binding
options); `OUT` on the parallel file system, since the box and the written files go there; and
the file system's striping recorded beside the results. Rank 0 holds the $O(n^3)$ box in memory
while MOAB writes it.
