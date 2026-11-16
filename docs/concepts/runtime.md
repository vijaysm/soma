# Runtime and options

A SOMA program must start MPI (in MPI builds) and Kokkos before it touches a mesh, and stop them
afterwards. This page describes `soma::ScopeGuard`, which does both, SOMA's process-wide options
and the environment variables and command-line arguments that set them, and how host threads
and devices are chosen.

## `ScopeGuard`

`soma::ScopeGuard` (`soma/runtime/scope_guard.hpp`) owns the runtime's lifetime:

```cpp
class [[nodiscard]] ScopeGuard {
 public:
  ScopeGuard(int& argc, char**& argv);
  ~ScopeGuard();
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};
```

The constructor, in order:

1. reads SOMA's options from the environment and the command line (`parse_options`, below),
   removing SOMA's own arguments from `argv`; a bad value throws `std::invalid_argument` before
   anything is started;
2. in an MPI build (`SOMA_ENABLE_MPI`), initialises MPI with `MPI_Init_thread` at
   `MPI_THREAD_FUNNELED`, unless MPI is already initialised (Kokkos host threads never call MPI);
3. initialises Kokkos with `Kokkos::initialize(argc, argv)`, unless Kokkos is already
   initialised.

The destructor finalises, in reverse order, only what the constructor started. An application
that initialises MPI or Kokkos itself keeps control of them; the guard leaves them alone.

Kokkos forbids constructing a View before `Kokkos::initialize`, and every mesh holds Views, so
every SOMA program, tests and benchmarks included, starts with a guard, and every mesh, field and
View must be destroyed before the guard goes out of scope. Declaring the guard first in `main`
and the meshes after it gives that order. From `examples/laplacian_smooth.cpp`:

```cpp
int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  // ... read, compute and write meshes here ...
  return 0;
}
```

Kokkos removes its own command-line options from `argv`, and the guard removes SOMA's, so any
other argument parser (GoogleTest's, for one) must run after the guard is constructed. Note that
`argc` and `argv` are taken by reference for this reason.

## Process-wide options

`soma/runtime/options.hpp` holds two options that apply to the whole process:

| Option | Default | Environment | Command line | Meaning |
| --- | --- | --- | --- | --- |
| compress threshold | `0.2` | `SOMA_COMPRESS_THRESHOLD=<x>` | `--soma-compress-threshold=<x>` | `MeshEdit::apply` compresses the mesh when, in some dimension, at least this fraction of the slots holds removed entities; 0 never compresses. Must be in $[0, 1]$ |
| reproducible | off | `SOMA_REPRODUCIBLE=0\|1` | `--soma-reproducible` (on), `--soma-reproducible=0\|1` | results bit for bit the same on any number of ranks and threads, where an algorithm can give that at some cost |

```bash
SOMA_COMPRESS_THRESHOLD=0.3 ./app
./app --soma-compress-threshold=0.3      # the command line wins over the environment
SOMA_REPRODUCIBLE=1 ./app
./app --soma-reproducible
```

The value of `--soma-reproducible` must be attached with `=`; a value after a space is not read
as its value.

The functions:

```cpp
inline constexpr double default_compress_threshold = 0.2;
double compress_threshold() noexcept;
void set_compress_threshold(double t);   // std::invalid_argument outside [0, 1] or NaN
bool reproducible() noexcept;
void set_reproducible(bool on) noexcept;
void parse_options(int& argc, char**& argv);
```

`parse_options` reads `SOMA_COMPRESS_THRESHOLD` and `SOMA_REPRODUCIBLE`, then the two
command-line options, and removes those options from `argv` (`argc` shrinks and `argv[argc]`
stays null). It throws `std::invalid_argument`, naming the option, for a threshold that is not
a number in $[0, 1]$ or a switch that is not 0 or 1; nothing is set then. `ScopeGuard` calls it;
call it yourself only if you initialise Kokkos without a guard.

### The compress threshold

Edited meshes pay for their extra slots and alive mask on every pass. The header records one
measurement behind the default: on an $80^3$ box of hexes (8 threads, Apple M-series), a field
write over every live hex cost 1.6 times its compact cost with 5 % of the slots removed, 1.8 times
at 17 % and 2.2 times at 23 %, passing 2 near 20 %. `bench_edit` prints the crossover it measures,
as a value for another machine. A single edit can override the threshold with
`MeshEdit::set_compress_threshold`; see [editing meshes](editing.md).

### Reproducibility

With `reproducible` off (the default), results may differ in the last bits between runs on
different numbers of ranks or threads, in exchange for speed. On, an algorithm that can give
bit-for-bit results does: a `FemSpace` gathers whatever assembly it is asked for, and its `dot()`
and the conjugate gradient solver sum exactly (see [reproducible sums](../parallel/reproducible-sums.md));
`laplacian_smooth` is reproducible either way. Algorithms read the option when they start (a
`FemSpace` when it is made) and agree on it across ranks; `FemSpace` and `FvSpace` throw
`std::invalid_argument` if the ranks differ. An option of an algorithm's own
(`CgOptions::reproducible`) overrides it.

## Environment variables

SOMA itself reads exactly two environment variables, both through `parse_options`:

| Variable | Values |
| --- | --- |
| `SOMA_COMPRESS_THRESHOLD` | a number in $[0, 1]$ |
| `SOMA_REPRODUCIBLE` | `0` or `1` |

Everything else in the environment is read by Kokkos, MPI and the libraries SOMA uses.

## Threads and devices

SOMA runs its kernels in Kokkos execution spaces and does not choose threads or devices itself:
`Kokkos::initialize(argc, argv)` does, from Kokkos's own command-line options and environment
variables. For example, the host thread count:

```bash
./app --kokkos-num-threads=8
KOKKOS_NUM_THREADS=8 ./app
```

The test suite uses `KOKKOS_NUM_THREADS` to check that results agree at 1, 3 and 4 threads
(`tests/unit/compare_threads.cmake`). With the OpenMP backend, the OpenMP variables
(`OMP_NUM_THREADS`, `OMP_PROC_BIND`) apply as usual. Under MPI, Kokkos maps one device per
node-local rank from the launcher's environment; SOMA adds no binding of its own. See the Kokkos
documentation for the full list of options.

Which backends exist (OpenMP, Threads, HPX or Serial on the host; CUDA, HIP or SYCL on a device) is
fixed when Kokkos is built; see [installation](../getting-started/installation.md) and
[CMake options](../reference/cmake-options.md).

## Configuration macros

The generated header `soma/config.hpp` records the CMake options that change the ABI or the
public headers, as 0 or 1: `SOMA_ENABLE_MPI`, `SOMA_LO_64`, `SOMA_OFFSET_64`, `SOMA_DEBUG`,
`SOMA_ENABLE_KOKKOSKERNELS`, `SOMA_ENABLE_PETSC` and `SOMA_ENABLE_ARBORX`, with the version as
`SOMA_VERSION_MAJOR`, `SOMA_VERSION_MINOR` and `SOMA_VERSION_PATCH`. Guard code that uses
optional features with them, for example `#if SOMA_ENABLE_MPI`. `SOMA_DEBUG` turns on
device-side bounds and contract checks, such as the check that a kernel reads an adjacency the
mesh holds.

## Process-wide state

Besides the two options, SOMA keeps a process-wide cache of the reference node tables of
high-order layouts (`NodeTables`, see [fields](fields.md)), built on first use and emptied by a
hook at `Kokkos::finalize`. Views taken from it must not be used after that.
