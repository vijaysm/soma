// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SEM_OPERATORS_HPP_
#define SOMA_SEM_OPERATORS_HPP_

/// \file
/// Spectral-element operators on the values of a SemSpace: rows as the
/// space's points, vectors as three Cartesian components.
///
/// ```
///   soma::sem::gradient(space, h, grad);          // element-local
///   space.dss(comm, grad);                        // continuous
///   soma::sem::divergence(space, u, div);
///   soma::sem::vorticity(space, u, zeta);
///   soma::sem::laplacian(comm, space, h, lap);     // weak form, assembled
///   soma::sem::hyperviscosity(comm, space, h, del4);
/// ```
///
/// The strong-form operators differentiate within each element with the GLL
/// derivative matrix D, at every element on the rank (ghosts included); their
/// results are discontinuous across elements until dss. With f_xi, f_eta the
/// derivatives along an element's reference axes and a_i, a^i, J its
/// geometry (SemSpace):
/// ```
///   gradient     a^1 f_xi + a^2 f_eta
///   divergence   (1/J) [d(J v.a^1)/dxi + d(J v.a^2)/deta]
///   vorticity    (1/J) [d(v.a_2)/dxi - d(v.a_1)/deta], the curl's component
///                along a_1 x a_2 (up, or outward on the sphere)
/// ```
/// The divergence sums by parts exactly: its integral over the mesh is the
/// net flux through the boundary, zero on a sphere, to rounding. laplacian is
/// the weak form, -sum w J grad(phi) . grad(f) per element, summed over the
/// elements holding each point and divided by the assembled mass: on a mesh
/// with a boundary it imposes a zero normal derivative there. It and
/// hyperviscosity (the Laplacian applied twice) are collective and read only
/// the owned elements; every element's result is written, as dss does.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/sem_space.hpp"

namespace soma::sem {

namespace detail {

/// Views taken by the operators, the memory space deduced from the space.
template <class MemSpace>
using In = std::type_identity_t<View2<const Real, MemSpace>>;
template <class MemSpace>
using Out = std::type_identity_t<View2<Real, MemSpace>>;

template <class MemSpace, class V>
void check_values(const SemSpace<MemSpace>& s, const V& v, std::size_t components, const char* who,
                  const char* what) {
  const auto rows = static_cast<std::size_t>(s.num_elements()) * static_cast<std::size_t>(s.points_per_element());
  if (v.extent(0) != rows || v.extent(1) != components) {
    throw std::invalid_argument(std::string("soma::sem::") + who + ": " + what + " is " +
                                std::to_string(v.extent(0)) + " x " + std::to_string(v.extent(1)) + ", not " +
                                std::to_string(rows) + " x " + std::to_string(components));
  }
}

template <class MemSpace>
using PointPolicy = Kokkos::RangePolicy<typename MemSpace::execution_space, Kokkos::IndexType<LO>>;

/// The reference derivatives at point r = e (p + 1)^2 + i + (p + 1) j of
/// columns c0 (along xi) and c1 (along eta) of g.
template <class MemSpace>
struct Derivatives {
  View2<const Real, MemSpace> d;
  int np;

  KOKKOS_INLINE_FUNCTION Real xi(const View2<Real, MemSpace>& g, LO r, int c) const {
    const LO k = r % (np * np), i = k % np, first = r - i;
    Real s = 0;
    for (int m = 0; m < np; ++m) s += d(i, m) * g(first + m, c);
    return s;
  }
  KOKKOS_INLINE_FUNCTION Real eta(const View2<Real, MemSpace>& g, LO r, int c) const {
    const LO k = r % (np * np), j = k / np, first = r - np * j;
    Real s = 0;
    for (int n = 0; n < np; ++n) s += d(j, n) * g(first + np * n, c);
    return s;
  }
  /// The weak (transposed) forms: sum_m D(m, i) g(m, j) and sum_n D(n, j) g(i, n).
  KOKKOS_INLINE_FUNCTION Real xi_t(const View2<Real, MemSpace>& g, LO r, int c) const {
    const LO k = r % (np * np), i = k % np, first = r - i;
    Real s = 0;
    for (int m = 0; m < np; ++m) s += d(m, i) * g(first + m, c);
    return s;
  }
  KOKKOS_INLINE_FUNCTION Real eta_t(const View2<Real, MemSpace>& g, LO r, int c) const {
    const LO k = r % (np * np), j = k / np, first = r - np * j;
    Real s = 0;
    for (int n = 0; n < np; ++n) s += d(n, j) * g(first + np * n, c);
    return s;
  }
};

template <class MemSpace>
View2<Real, MemSpace> scratch(const SemSpace<MemSpace>& s, int components, const char* label) {
  return View2<Real, MemSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                               static_cast<std::size_t>(s.num_elements()) *
                                   static_cast<std::size_t>(s.points_per_element()),
                               static_cast<std::size_t>(components));
}

}  // namespace detail

/// f: 1 component; grad: 3. Element-local.
template <class MemSpace>
void gradient(const SemSpace<MemSpace>& s, const detail::In<MemSpace>& f, const detail::Out<MemSpace>& grad) {
  detail::check_values(s, f, 1, "gradient", "f");
  detail::check_values(s, grad, 3, "gradient", "grad");
  const auto g = detail::scratch(s, 1, "soma::sem::gradient::f");
  Kokkos::deep_copy(g, f);
  const detail::Derivatives<MemSpace> d{s.derivative(), s.order() + 1};
  const auto b = s.contravariant();
  Kokkos::parallel_for(
      "soma::sem::gradient", detail::PointPolicy<MemSpace>(0, static_cast<LO>(f.extent(0))),
      KOKKOS_LAMBDA(const LO r) {
        const Real fx = d.xi(g, r, 0), fy = d.eta(g, r, 0);
        for (int k = 0; k < 3; ++k) grad(r, k) = b(r, k) * fx + b(r, 3 + k) * fy;
      });
  typename MemSpace::execution_space{}.fence();
}

/// v: 3 components; div: 1. Element-local.
template <class MemSpace>
void divergence(const SemSpace<MemSpace>& s, const detail::In<MemSpace>& v, const detail::Out<MemSpace>& div) {
  detail::check_values(s, v, 3, "divergence", "v");
  detail::check_values(s, div, 1, "divergence", "div");
  const auto flux = detail::scratch(s, 2, "soma::sem::divergence::flux");
  const auto b = s.contravariant();
  const auto jac = s.jacobian();
  const auto n = static_cast<LO>(v.extent(0));
  Kokkos::parallel_for(
      "soma::sem::divergence::flux", detail::PointPolicy<MemSpace>(0, n), KOKKOS_LAMBDA(const LO r) {
        for (int a = 0; a < 2; ++a) {
          flux(r, a) = jac(r, 0) * (v(r, 0) * b(r, 3 * a) + v(r, 1) * b(r, 3 * a + 1) + v(r, 2) * b(r, 3 * a + 2));
        }
      });
  const detail::Derivatives<MemSpace> d{s.derivative(), s.order() + 1};
  Kokkos::parallel_for(
      "soma::sem::divergence", detail::PointPolicy<MemSpace>(0, n),
      KOKKOS_LAMBDA(const LO r) { div(r, 0) = (d.xi(flux, r, 0) + d.eta(flux, r, 1)) / jac(r, 0); });
  typename MemSpace::execution_space{}.fence();
}

/// v: 3 components; vort: 1, the curl's normal component. Element-local.
template <class MemSpace>
void vorticity(const SemSpace<MemSpace>& s, const detail::In<MemSpace>& v, const detail::Out<MemSpace>& vort) {
  detail::check_values(s, v, 3, "vorticity", "v");
  detail::check_values(s, vort, 1, "vorticity", "vort");
  const auto cov = detail::scratch(s, 2, "soma::sem::vorticity::covariant");
  const auto a = s.covariant();
  const auto jac = s.jacobian();
  const auto n = static_cast<LO>(v.extent(0));
  Kokkos::parallel_for(
      "soma::sem::vorticity::covariant", detail::PointPolicy<MemSpace>(0, n), KOKKOS_LAMBDA(const LO r) {
        for (int i = 0; i < 2; ++i) {
          cov(r, i) = v(r, 0) * a(r, 3 * i) + v(r, 1) * a(r, 3 * i + 1) + v(r, 2) * a(r, 3 * i + 2);
        }
      });
  const detail::Derivatives<MemSpace> d{s.derivative(), s.order() + 1};
  Kokkos::parallel_for(
      "soma::sem::vorticity", detail::PointPolicy<MemSpace>(0, n),
      KOKKOS_LAMBDA(const LO r) { vort(r, 0) = (d.xi(cov, r, 1) - d.eta(cov, r, 0)) / jac(r, 0); });
  typename MemSpace::execution_space{}.fence();
}

/// v: 3 components; its component along a_1 x a_2 removed, in place.
template <class MemSpace>
void project_tangent(const SemSpace<MemSpace>& s, const detail::Out<MemSpace>& v) {
  detail::check_values(s, v, 3, "project_tangent", "v");
  const auto a = s.covariant();
  Kokkos::parallel_for(
      "soma::sem::project_tangent", detail::PointPolicy<MemSpace>(0, static_cast<LO>(v.extent(0))),
      KOKKOS_LAMBDA(const LO r) {
        const Real n[3] = {a(r, 1) * a(r, 5) - a(r, 2) * a(r, 4), a(r, 2) * a(r, 3) - a(r, 0) * a(r, 5),
                           a(r, 0) * a(r, 4) - a(r, 1) * a(r, 3)};
        const Real nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        const Real along = (v(r, 0) * n[0] + v(r, 1) * n[1] + v(r, 2) * n[2]) / nn;
        for (int k = 0; k < 3; ++k) v(r, k) -= along * n[k];
      });
  typename MemSpace::execution_space{}.fence();
}

/// Collective. f, lap: 1 component. The weak Laplacian,
/// assembled: continuous, every element's points written.
template <class MemSpace>
void laplacian(const Comm& comm, const SemSpace<MemSpace>& s, const detail::In<MemSpace>& f,
               const detail::Out<MemSpace>& lap) {
  detail::check_values(s, f, 1, "laplacian", "f");
  detail::check_values(s, lap, 1, "laplacian", "lap");
  const auto g = detail::scratch(s, 1, "soma::sem::laplacian::f");
  Kokkos::deep_copy(g, f);
  const auto flux = detail::scratch(s, 2, "soma::sem::laplacian::flux");
  const auto b = s.contravariant();
  const auto m = s.local_mass();
  const detail::Derivatives<MemSpace> d{s.derivative(), s.order() + 1};
  const LO owned = s.owned_elements() * s.points_per_element();
  // m (a^i . grad f): the integrand of each reference direction.
  Kokkos::parallel_for(
      "soma::sem::laplacian::flux", detail::PointPolicy<MemSpace>(0, owned), KOKKOS_LAMBDA(const LO r) {
        const Real fx = d.xi(g, r, 0), fy = d.eta(g, r, 0);
        const Real g11 = b(r, 0) * b(r, 0) + b(r, 1) * b(r, 1) + b(r, 2) * b(r, 2);
        const Real g12 = b(r, 0) * b(r, 3) + b(r, 1) * b(r, 4) + b(r, 2) * b(r, 5);
        const Real g22 = b(r, 3) * b(r, 3) + b(r, 4) * b(r, 4) + b(r, 5) * b(r, 5);
        flux(r, 0) = m(r, 0) * (g11 * fx + g12 * fy);
        flux(r, 1) = m(r, 0) * (g12 * fx + g22 * fy);
      });
  // Each element's -sum m grad(phi) . grad(f), over its local mass: dss then
  // sums the elements' integrals and divides by the assembled mass.
  Kokkos::parallel_for(
      "soma::sem::laplacian", detail::PointPolicy<MemSpace>(0, owned),
      KOKKOS_LAMBDA(const LO r) { lap(r, 0) = -(d.xi_t(flux, r, 0) + d.eta_t(flux, r, 1)) / m(r, 0); });
  typename MemSpace::execution_space{}.fence();
  s.dss(comm, lap);
}

/// Collective. laplacian applied twice: f, out 1 component.
template <class MemSpace>
void hyperviscosity(const Comm& comm, const SemSpace<MemSpace>& s, const detail::In<MemSpace>& f,
                    const detail::Out<MemSpace>& out) {
  const auto once = detail::scratch(s, 1, "soma::sem::hyperviscosity::laplacian");
  laplacian(comm, s, f, once);
  laplacian(comm, s, View2<const Real, MemSpace>(once), out);
}

}  // namespace soma::sem

#endif  // SOMA_SEM_OPERATORS_HPP_
