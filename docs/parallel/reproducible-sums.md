# Reproducible sums

A floating-point sum depends on the order of its terms, and a parallel reduction's order depends
on the number of threads and ranks, so plain sums can differ in their last bits between runs at
different scales. SOMA offers sums that do not depend on that order (`soma/parallel/exact_sum.hpp`)
and a process-wide *reproducible* switch under which algorithms use them. This page explains both
and what exactly is guaranteed bit for bit.

## Exact sums

Each term is added **exactly** to a fixed-point accumulator spanning every finite double (68
signed 32-bit digits held in `int64` words), and the total is rounded to the nearest double
(ties to even) once, at the end. Integer addition is associative, so the same terms give the
same bits on any number of threads and ranks, however they are split. Since only the final total
is rounded, the result is the correctly rounded sum of the terms: at least as accurate as a
plain sum, and exact under cancellation.

```cpp
template <int K>
struct ExactSums {
  KOKKOS_INLINE_FUNCTION void add(int k, Real x);   // adds x to sum k
  KOKKOS_INLINE_FUNCTION Real value(int k) const;   // sum k rounded; NaN if a term was not finite
  KOKKOS_INLINE_FUNCTION void join(const ExactSums& o);
  KOKKOS_INLINE_FUNCTION void normalize();
};

template <int K> struct ExactSumReducer;   // a Kokkos reducer of ExactSums<K>

template <int K, class Exec, class Body>
ExactSums<K> reduce_exact(const char* label, LO first, LO last, const Body& body);

template <int K>
std::array<Real, K> allreduce_exact(const Comm& comm, ExactSums<K> sums);
```

`ExactSums<K>` holds `K` independent sums. A typical use, from the header:

```cpp
ExactSums<2> s;
Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(LO i, ExactSums<2>& acc) {
  acc.add(0, x(i) * y(i));
  acc.add(1, x(i) * x(i));
}, ExactSumReducer<2>(s));
const auto [xy, xx] = allreduce_exact(comm, s);   // one allreduce of integers
```

- `ExactSumReducer<K>` is a Kokkos reducer into a host value; it normalises on every join.
- One accumulator takes at most $2^{28}$ terms between normalisations
  (`exact::terms_per_pass`). A single `parallel_reduce` over a longer range can exceed that in one
  thread's accumulator; `reduce_exact` splits the range `[first, last)` into passes of at most
  $2^{28}$ indices (the body adding at most one term per sum per index), and returns the
  normalised total. Use it for ranges that may be long. From `tests/unit/test_exact_sum.cpp`:

  ```cpp
  const auto s = soma::reduce_exact<1, Kokkos::DefaultHostExecutionSpace>(
      "test::long", 0, n,
      KOKKOS_LAMBDA(const LO i, ExactSums<1>& acc) { acc.add(0, static_cast<Real>(i % 7) - 3.25); });
  return s.value(0);
  ```

- `allreduce_exact` is collective: every rank's sums are added with **one** `allreduce` of
  integers, then rounded; every rank receives the same `std::array<Real, K>`.
- A NaN or infinite term makes its sum NaN (on every rank, after `allreduce_exact`); the count of
  such terms is carried with the digits. A finite total beyond the range of a double rounds to
  infinity.
- `value()` needs no `normalize()` first.

### Cost and limits

- A term costs a few integer operations; each accumulator is 68 `int64` digits, 552 bytes per sum
  with its non-finite count. That is a large reduction value on a GPU.
- Exact sums are tested on host execution spaces.
- **The terms themselves must be the same bits wherever they are computed.** That holds within
  one build, with each expression contracted alike (as compilers do by default). It does not hold
  under `-ffast-math`, which also removes the NaN checks. A dot product computed this way sums the
  *rounded* products exactly; it is reproducible, not the exact dot product.

## The reproducible switch

`soma/runtime/options.hpp`:

```cpp
bool reproducible() noexcept;
void set_reproducible(bool on) noexcept;
```

When on, algorithms that can give results the same bit for bit on any number of ranks and threads,
at some cost, do so; off (the default) they may differ in the last bits, and run faster. It is set
from the environment or the command line by `parse_options`, which `ScopeGuard` calls, or by
`set_reproducible`:

```bash
SOMA_REPRODUCIBLE=1 ./app            # 0 or 1
./app --soma-reproducible            # on; or --soma-reproducible=0|1
```

The command line wins over the environment. A value that is not 0 or 1 throws
`std::invalid_argument`, naming the option. See [runtime and options](../concepts/runtime.md).

Algorithms read the switch when they start (a space when it is made) and check that the ranks
agree on it; an option of their own overrides it. What it changes:

| Algorithm | With `reproducible()` on |
|---|---|
| `FemSpace` ([FEM](../algorithms/fem.md)) | Gathers whatever assembly it is asked for (`assembly()` tells which it uses) and `dot()` sums exactly. Throws `std::invalid_argument` if ranks differ in `reproducible()`. |
| `cg` ([linear algebra](../algorithms/linear-algebra.md)) | Every sum is exact, with as many allreduces as before; the space must gather. `CgOptions::reproducible` overrides the space's setting. |
| `FvSpace` ([finite volumes](../algorithms/finite-volumes.md)) | `total()` is exact (rounded once). Throws if ranks differ in `reproducible()`. |
| `laplacian_smooth` ([smoothing](../algorithms/smoothing.md)) | No change: reproducible either way. |

## What is bit for bit

Within one build, on any number of ranks and threads:

- `ExactSums` totals, through `ExactSumReducer`, `reduce_exact` and `allreduce_exact`, for the same
  terms.
- `FemSpace` matrices with gather assembly (the default), and `dot()` with `reproducible()` on.
- `cg` iterates and solution with reproducible sums, given a matrix and right-hand side that are
  the same on every partition, as `FemSpace::laplacian()` and `load()` are.
- `FvSpace` steps (each cell sums its faces' fluxes in a fixed order), and `total()` with
  `reproducible()` on.
- `laplacian_smooth` (each vertex sums its neighbours in ascending gid order).
- `HaloExchange::reduce`: every holder of an entity gets the same bits
  ([halo exchange](ghosts-and-halos.md)). This is agreement between ranks within one run; the
  result can change with the partition.

Not covered: `Comm::allreduce` of doubles (MPI's order); `FemSpace` atomic assembly, and
colored assembly across rank counts (it is the same on any number of threads), with
`reproducible()` off; and results across different builds, compilers or flags.
