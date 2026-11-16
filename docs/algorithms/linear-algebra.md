# Matrices and solvers

SOMA's sparse matrices hold a rank's rows of an operator over vertices, as
[`FemSpace`](fem.md) assembles them. This page covers the matrix type and its operations
(`soma/algorithms/csr.hpp`), the built-in conjugate gradient solver (`soma/algorithms/cg.hpp`)
and the optional bridges to KokkosKernels (`soma/algorithms/kokkos_sparse.hpp`) and PETSc
(`soma/algorithms/petsc.hpp`).

## `CsrMatrix`

Header: `soma/algorithms/csr.hpp`.

```cpp
template <class MemSpace>
struct CsrMatrix {
  LO rows = 0;
  LO cols = 0;
  View1<Offset, MemSpace> offsets;  // rows + 1
  View1<LO, MemSpace> columns;
  View1<Real, MemSpace> values;
};
```

A rank's matrix is in compressed sparse rows. Its rows are the rank's owned vertices
`[0, rows)`; its columns are all the vertices it holds, `[0, cols)`: owned, then shared copies
and ghosts. Each row's columns are in ascending gid order, and sums along a row run in that
order, so a product is the same, bit for bit, on any number of ranks and threads.

### Operations

```cpp
template <class MemSpace>
void multiply(const CsrMatrix<MemSpace>& a, const View1<const Real, MemSpace>& x, const View1<Real, MemSpace>& y);

template <class MemSpace>
void apply_dirichlet(const CsrMatrix<MemSpace>& a, const View1<Real, MemSpace>& b,
                     const View1<const std::uint8_t, MemSpace>& fixed, const View1<const Real, MemSpace>& values);
```

`multiply` computes $y_i = \sum_j A_{ij} x_j$ over the rows. `x` holds `cols` entries and `y`
at least `rows`. It is local: on more than one rank, `x`'s copies and ghosts must already
hold their owners' values. `FemSpace::multiply(comm, a, x, y)` updates them first and is the
usual way to call it. It throws `std::invalid_argument` if `x` or `y` is too short.

`apply_dirichlet` imposes Dirichlet conditions by symmetric elimination:

- the row of a fixed vertex becomes the identity's row, with $b_i$ set to its value;
- every other row moves the terms of its fixed columns to $b$ and zeroes them.

`fixed` and `values` hold one entry per column (vertex); they must be the same on every rank
that holds the vertex, and values of free vertices are not read. A symmetric matrix stays
symmetric. The call is local and throws `std::invalid_argument` if `b` is shorter than the
rows, or `fixed` or `values` than the columns. `FemSpace::boundary()` gives a suitable `fixed`.

## Conjugate gradients

Header: `soma/algorithms/cg.hpp`.

```cpp
struct CgOptions {
  Real rtol = 1e-10;  // relative to ||b||
  Real atol = 0;
  int max_iterations = 1000;
  bool jacobi = true;  // precondition by the diagonal
  std::optional<bool> reproducible;  // unset: the space's; on, the space must gather
};

struct CgReport {
  int iterations = 0;
  Real residual = 0;  // ||b - A x||: computed directly when converged
  bool converged = false;
};

template <class MemSpace>
CgReport cg(const Comm& comm, const FemSpace<MemSpace>& space, const CsrMatrix<MemSpace>& a,
            const View1<const Real, MemSpace>& b, const View1<Real, MemSpace>& x, const CgOptions& options = {});
```

`cg` solves $A x = b$ for a symmetric positive definite matrix over a `FemSpace`'s vertices
by conjugate gradients, preconditioned by the diagonal unless `jacobi` is false. `x` is the
initial guess on entry and the solution on return, with its copies and ghosts holding their
owners' values.

| Option | Default | Meaning |
| --- | --- | --- |
| `rtol` | `1e-10` | Stop when the residual norm is at most `rtol` times $\lVert b \rVert$ ... |
| `atol` | `0` | ... or at most `atol`, whichever is larger. |
| `max_iterations` | `1000` | Iterations before giving up. |
| `jacobi` | `true` | Precondition by the diagonal. |
| `reproducible` | unset | Sum exactly. Unset: the space's `reproducible()`. |

Norms are taken over every rank's owned entries. Each iteration makes one halo update (for the
product) and two allreduces. Once the recurrence's residual meets
$\max(\text{rtol}\,\lVert b \rVert, \text{atol})$, the true residual $b - A x$ is computed (one
more update and allreduce); it must meet the target too, or CG restarts from it. So a
converged report's `residual` is the true one.

With $b = 0$, `x` is set to 0 at once and the solve reports convergence. CG stops without
converging when the iterations run out, when $(p, A p)$ is not positive (the matrix is not
positive definite), or when a norm is not finite.

`cg` is collective. It throws `std::invalid_argument` on every rank if any rank passes a
negative or non-finite tolerance, a negative iteration count, an `x` not of the space's
`cols()` entries, a `b` shorter than its `rows()`, a matrix not of the space's shape, or, with
`jacobi`, a row whose diagonal is not positive (or whose inverse is not finite); if
`reproducible` is on with a space that does not gather; or if ranks differ in their options.

### Reproducible solves

With `reproducible` on (by default the space's, which follows `soma::reproducible()` when the
space was made), every sum in CG is exact (see
[reproducible sums](../parallel/reproducible-sums.md)), with as many allreduces as before, and
the space must assemble by `Assembly::gather`. With a matrix and right-hand side that are the
same on every partition, as the space's `laplacian()` and `load()` are, the iterates and `x`
are then the same, bit for bit, on any number of ranks and threads. The changelog records
the cost on a $64^3$ box of hexes, one thread a rank: CG took 1.28 times as long on one rank.

For an example of `cg` with `FemSpace`, see the Poisson example in
[finite elements](fem.md#example-poissons-equation).

## KokkosKernels

Header: `soma/algorithms/kokkos_sparse.hpp`. Needs SOMA configured with
`-DSOMA_ENABLE_KOKKOSKERNELS=ON` (see [CMake options](../reference/cmake-options.md));
including the header otherwise is a compile error.

```cpp
template <class MemSpace>
using KokkosSparseMatrix =
    KokkosSparse::CrsMatrix<Real, LO, Kokkos::Device<typename MemSpace::execution_space, MemSpace>, void, Offset>;

template <class MemSpace>
KokkosSparseMatrix<MemSpace> to_kokkos_sparse(const CsrMatrix<MemSpace>& a);
```

`to_kokkos_sparse` views a `CsrMatrix` as a `KokkosSparse::CrsMatrix` sharing its views: there
is no copy, and writes through either show in both. Rows are the owned vertices, columns all
the rank's vertices. As with `multiply`, on more than one rank `x`'s copies and ghosts must
hold their owners' values first (`FemSpace::update`). Adapted from
`tests/unit/test_kokkos_sparse.cpp`:

```cpp
const soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(comm);
const auto k = soma::to_kokkos_sparse(a);
space.update(comm, x);  // x: cols() entries
KokkosSparse::spmv("N", Real{1}, k, x, Real{0}, Kokkos::subview(y, std::make_pair(LO{0}, a.rows)));
```

A KokkosKernels product may sum a row in another order than `multiply`, so it can differ from
it by rounding.

## PETSc

Header: `soma/algorithms/petsc.hpp`. Needs SOMA configured with `-DSOMA_ENABLE_PETSC=ON` (which
needs MPI, and `PETSC_DIR`, plus `PETSC_ARCH` for an in-place PETSc build); including the
header otherwise is a compile error. PETSc must be built with real double scalars.

```cpp
template <class MemSpace>
Mat to_petsc_matrix(const Comm& comm, const FemSpace<MemSpace>& space, const CsrMatrix<MemSpace>& a);

template <class MemSpace>
Vec to_petsc_vector(const Comm& comm, const FemSpace<MemSpace>& space, const View1<const Real, MemSpace>& x);

template <class MemSpace>
void from_petsc_vector(const Comm& comm, const FemSpace<MemSpace>& space, Vec v, const View1<Real, MemSpace>& x);
```

Rows and entries are numbered by `FemSpace::numbering` (owned vertices rank by rank), each
rank's rows its owned vertices. The matrix is always `MATAIJ`, built by COO assembly, whatever
the options database says; use `MatConvert` for another type. Values pass through host
memory. PETSc must be initialised before these calls, and the caller destroys the objects they
return.

- `to_petsc_matrix` is collective. It throws `std::invalid_argument` on every rank if any
  rank's matrix is not of the space's shape or the global row count exceeds `PetscInt`, and
  `std::runtime_error` where a PETSc call fails (having destroyed the matrix).
- `to_petsc_vector` is collective: a vector of `x`'s owned entries. It throws
  `std::invalid_argument` on every rank if any rank's `x` is shorter than the space's rows,
  and `std::runtime_error` where PETSc fails.
- `from_petsc_vector` is collective: `x`'s owned entries from `v`, then its copies and ghosts
  from their owners. It throws `std::invalid_argument` on every rank if any rank's `v` does
  not hold the space's rows or `x` is not of its `cols()`.

Adapted from `tests/unit/test_petsc.cpp`, a PETSc CG solve of a space's system (error checks
omitted):

```cpp
Mat m = soma::to_petsc_matrix(comm, space, a);
KSP ksp = nullptr;
PC pc = nullptr;
KSPCreate(comm.native(), &ksp);
KSPSetOperators(ksp, m, m);
KSPSetType(ksp, KSPCG);
KSPGetPC(ksp, &pc);
PCSetType(pc, PCJACOBI);
Vec pb = soma::to_petsc_vector(comm, space, soma::View1<const Real, Kokkos::HostSpace>(b));
Vec pu = nullptr;
VecDuplicate(pb, &pu);
KSPSolve(ksp, pb, pu);
auto u = space.vector();
soma::from_petsc_vector(comm, space, pu, u);  // u's copies and ghosts updated
VecDestroy(&pb);
VecDestroy(&pu);
KSPDestroy(&ksp);
MatDestroy(&m);
```
