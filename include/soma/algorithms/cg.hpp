// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_CG_HPP_
#define SOMA_ALGORITHMS_CG_HPP_

// Conjugate gradients, Jacobi-preconditioned, for a symmetric positive
// definite matrix over a FemSpace's vertices:
//
//   soma::CgOptions o;
//   o.rtol = 1e-10;
//   const soma::CgReport r = soma::cg(comm, space, a, b, x, o);  // x: the initial guess
//
// Each iteration makes one halo update (the product) and two allreduces:
// (p, A p), then (r, z) and (r, r) together. Once the recurrence's residual
// meets max(rtol ||b||, atol), the norms over every rank's owned entries,
// the true residual b - A x is computed (one more update and allreduce): it
// must meet the target too, or CG restarts from it. On return x's copies and
// ghosts hold their owners' values.
//
// Reproducible (CgOptions::reproducible, by default the space's, which is
// soma::reproducible() when the space was made), every sum is exact
// (soma/parallel/exact_sum.hpp), as many allreduces as before, and the space
// must gather: with its matrix (and b) the same on every partition, as the
// space's laplacian() and load() are, the iterates and x are the same bit
// for bit on any number of ranks and threads.

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/exact_sum.hpp"

namespace soma {

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

namespace detail {

// z = r / d (or r) over the rows; returns the local (r, z) and (r, r).
template <class MemSpace>
std::array<Real, 2> precondition(LO rows, const View1<const Real, MemSpace>& inverse, bool jacobi,
                                 const View1<const Real, MemSpace>& r, const View1<Real, MemSpace>& z) {
  using Exec = typename MemSpace::execution_space;
  Real rz = 0, rr = 0;
  Kokkos::parallel_reduce(
      "soma::cg::precondition", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows),
      KOKKOS_LAMBDA(const LO i, Real& a, Real& b) {
        const Real zi = jacobi ? r(i) * inverse(i) : r(i);
        z(i) = zi;
        a += r(i) * zi;
        b += r(i) * r(i);
      },
      rz, rr);
  return {rz, rr};
}

// One step: x += alpha p, r -= alpha q, then z = M^-1 r; returns the local
// (r, z) and (r, r).
template <class MemSpace>
std::array<Real, 2> step(LO rows, Real alpha, const View1<const Real, MemSpace>& inverse, bool jacobi,
                         const View1<const Real, MemSpace>& p, const View1<const Real, MemSpace>& q,
                         const View1<Real, MemSpace>& x, const View1<Real, MemSpace>& r,
                         const View1<Real, MemSpace>& z) {
  using Exec = typename MemSpace::execution_space;
  Real rz = 0, rr = 0;
  Kokkos::parallel_reduce(
      "soma::cg::step", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows),
      KOKKOS_LAMBDA(const LO i, Real& a, Real& b) {
        x(i) += alpha * p(i);
        const Real ri = r(i) - alpha * q(i);
        r(i) = ri;
        const Real zi = jacobi ? ri * inverse(i) : ri;
        z(i) = zi;
        a += ri * zi;
        b += ri * ri;
      },
      rz, rr);
  return {rz, rr};
}

// precondition and step with the sums exact; precondition adds (b, b)
// as the third when with_b, in a pass of its own: a kernel reduces at most
// two exact sums on a GPU (exact::most_sums_on_device).
template <class MemSpace>
ExactSums<3> precondition_exact(LO rows, const View1<const Real, MemSpace>& inverse, bool jacobi,
                                const View1<const Real, MemSpace>& r, const View1<Real, MemSpace>& z,
                                const View1<const Real, MemSpace>& b, bool with_b) {
  const ExactSums<2> two = reduce_exact<2, typename MemSpace::execution_space>(
      "soma::cg::precondition_exact", 0, rows, KOKKOS_LAMBDA(const LO i, ExactSums<2>& acc) {
        const Real zi = jacobi ? r(i) * inverse(i) : r(i);
        z(i) = zi;
        acc.add(0, r(i) * zi);
        acc.add(1, r(i) * r(i));
      });
  ExactSums<3> out;
  out.set(0, two, 0);
  out.set(1, two, 1);
  if (with_b) out.set(2, local_dot_exact(rows, b, b), 0);
  return out;
}

template <class MemSpace>
ExactSums<2> step_exact(LO rows, Real alpha, const View1<const Real, MemSpace>& inverse, bool jacobi,
                        const View1<const Real, MemSpace>& p, const View1<const Real, MemSpace>& q,
                        const View1<Real, MemSpace>& x, const View1<Real, MemSpace>& r,
                        const View1<Real, MemSpace>& z) {
  return reduce_exact<2, typename MemSpace::execution_space>(
      "soma::cg::step_exact", 0, rows, KOKKOS_LAMBDA(const LO i, ExactSums<2>& acc) {
        x(i) += alpha * p(i);
        const Real ri = r(i) - alpha * q(i);
        r(i) = ri;
        const Real zi = jacobi ? ri * inverse(i) : ri;
        z(i) = zi;
        acc.add(0, ri * zi);
        acc.add(1, ri * ri);
      });
}

// 1 / A[i][i] over the rows; counts the rows where that is not finite and
// positive.
template <class MemSpace>
LO inverse_diagonal(const CsrMatrix<MemSpace>& a, const View1<Real, MemSpace>& inverse) {
  using Exec = typename MemSpace::execution_space;
  const auto offsets = a.offsets;
  const auto columns = a.columns;
  const auto values = a.values;
  LO bad = 0;
  Kokkos::parallel_reduce(
      "soma::cg::diagonal", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, a.rows),
      KOKKOS_LAMBDA(const LO i, LO& count) {
        Real d = 0;
        for (Offset k = offsets(i); k < offsets(i + 1); ++k) {
          if (columns(k) == i) d = values(k);
        }
        const Real inv = 1 / d;
        const bool good = d > 0 && Kokkos::isfinite(inv);
        inverse(i) = good ? inv : 0;
        count += good ? 0 : 1;
      },
      bad);
  return bad;
}

}  // namespace detail

// Collective. Throws std::invalid_argument on every rank if any rank passes
// a negative or non-finite tolerance, a negative iteration count, an x not
// of the space's cols() entries, a b shorter than its rows(), a matrix not
// the space's shape, or with jacobi a row whose diagonal is not positive (or
// whose inverse is not finite), or reproducible with a space that does not
// gather; or if ranks differ in their options. With
// b = 0, x = 0 at once. Stops without converging when the iterations run
// out, (p, A p) is not positive (A is not positive definite) or a norm is
// not finite.
template <class MemSpace>
CgReport cg(const Comm& comm, const FemSpace<MemSpace>& space, const CsrMatrix<MemSpace>& a,
            const View1<const Real, MemSpace>& b, const View1<Real, MemSpace>& x, const CgOptions& options = {}) {
  using Exec = typename MemSpace::execution_space;
  using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>;
  const LO rows = space.rows();
  const auto cols = static_cast<std::size_t>(space.cols());
  const bool finite = std::isfinite(options.rtol) && std::isfinite(options.atol);
  const bool shape = a.rows == rows && a.cols == space.cols() && b.extent(0) >= static_cast<std::size_t>(rows) &&
                     x.extent(0) == cols;
  const bool repro = options.reproducible.value_or(space.reproducible());
  const bool bad = !finite || options.rtol < 0 || options.atol < 0 || options.max_iterations < 0 || !shape ||
                   (repro && space.assembly() != Assembly::gather);
  const View1<Real, MemSpace> inverse("soma::cg::inverse", options.jacobi ? static_cast<std::size_t>(rows) : 0);
  const LO bad_diagonal = !bad && options.jacobi ? detail::inverse_diagonal(a, inverse) : 0;

  // One reduction of the flags and of each option and its negative, which
  // agree only if every rank passed the same; then every rank goes on.
  const Real rtol = finite ? options.rtol : 0, atol = finite ? options.atol : 0;
  const auto it = static_cast<Real>(options.max_iterations), jac = options.jacobi ? Real{1} : Real{0};
  const Real ex = repro ? Real{1} : Real{0};
  std::array<Real, 12> f{bad ? Real{1} : Real{0}, static_cast<Real>(bad_diagonal), rtol, -rtol, atol, -atol, it, -it,
                         jac, -jac, ex, -ex};
  comm.allreduce(std::span<Real>(f), Op::max);
  if (f[0] != 0 || f[2] != -f[3] || f[4] != -f[5] || f[6] != -f[7] || f[8] != -f[9] || f[10] != -f[11]) {
    throw std::invalid_argument(
        "cg: a rank passes a negative or non-finite tolerance, a negative iteration count, vectors or a matrix not "
        "of the space's shape, reproducible with a space that does not gather, or ranks differ in their options");
  }
  if (f[1] != 0) throw std::invalid_argument("cg: a row's diagonal is not positive");

  const View1<Real, MemSpace> r("soma::cg::r", cols), z("soma::cg::z", cols), p("soma::cg::p", cols),
      q("soma::cg::q", cols);
  const View1<const Real, MemSpace> cinverse(inverse);
  // r = b - A x, z = M^-1 r, p = z: (r, z) and (r, r), and (b, b) on the
  // first call, in one sum.
  const auto restart = [&](bool with_b) {
    space.multiply(comm, a, x, q);
    Kokkos::parallel_for(
        "soma::cg::residual", Policy(0, rows), KOKKOS_LAMBDA(const LO i) { r(i) = b(i) - q(i); });
    std::array<Real, 3> s{};
    if (repro) {
      s = allreduce_exact(comm, detail::precondition_exact(rows, cinverse, options.jacobi,
                                                           View1<const Real, MemSpace>(r), z, b, with_b));
    } else {
      const auto [rz, rr] = detail::precondition(rows, cinverse, options.jacobi, View1<const Real, MemSpace>(r), z);
      s = {rz, rr, with_b ? detail::local_dot(rows, b, b) : Real{0}};
      comm.allreduce(std::span<Real>(s.data(), with_b ? 3 : 2), Op::sum);
    }
    Kokkos::deep_copy(Exec{}, p, z);
    return s;
  };
  const std::array<Real, 3> s = restart(true);

  CgReport report;
  if (s[2] == 0) {
    // b = 0: the solution is 0.
    Kokkos::parallel_for("soma::cg::zero", Policy(0, rows), KOKKOS_LAMBDA(const LO i) { x(i) = 0; });
    Exec{}.fence();
    space.update(comm, x);
    report.converged = true;
    return report;
  }
  const Real target = Kokkos::max(rtol * std::sqrt(s[2]), atol);
  Real rz = s[0];
  report.residual = std::sqrt(s[1]);
  bool exact = true;  // r is b - A x computed directly, not by the recurrence
  for (;;) {
    if (!std::isfinite(report.residual) || !std::isfinite(target)) break;
    if (report.residual <= target) {
      if (exact) {
        report.converged = true;
        break;
      }
      const std::array<Real, 3> t = restart(false);
      rz = t[0];
      report.residual = std::sqrt(t[1]);
      exact = true;
      continue;
    }
    if (report.iterations >= options.max_iterations) break;
    space.multiply(comm, a, p, q);
    const View1<const Real, MemSpace> cp(p), cq(q);
    const Real pq = repro ? allreduce_exact(comm, detail::local_dot_exact(rows, cp, cq))[0]
                          : comm.allreduce(detail::local_dot(rows, cp, cq), Op::sum);
    if (!(pq > 0)) break;
    std::array<Real, 2> sums{};
    if (repro) {
      sums = allreduce_exact(comm, detail::step_exact(rows, rz / pq, cinverse, options.jacobi, cp, cq, x, r, z));
    } else {
      sums = detail::step(rows, rz / pq, cinverse, options.jacobi, cp, cq, x, r, z);
      comm.allreduce(std::span<Real>(sums), Op::sum);
    }
    ++report.iterations;
    exact = false;
    report.residual = std::sqrt(sums[1]);
    const Real beta = sums[0] / rz;
    rz = sums[0];
    Kokkos::parallel_for(
        "soma::cg::direction", Policy(0, rows), KOKKOS_LAMBDA(const LO i) { p(i) = z(i) + beta * p(i); });
  }
  Exec{}.fence();
  space.update(comm, x);
  return report;
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_CG_HPP_
