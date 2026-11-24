// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "tracers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/sem/operators.hpp"

namespace ocean {

namespace {

using soma::LO;
using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<LO>>;

Policy rows(const Values& v) { return Policy(0, static_cast<LO>(v.extent(0))); }

}  // namespace

Tracers::Tracers(const Space& space, ConstValues H, const VerticalSpec& spec, int count, double nu_T)
    : space_(&space), spec_(spec), K_(spec.levels), count_(count), nu_(nu_T), H_(H) {
  if (count < 0) throw std::invalid_argument("swe_ocean: " + std::to_string(count) + " tracers");
  if (!(nu_T >= 0 && std::isfinite(nu_T))) {
    std::ostringstream s;
    s << "swe_ocean: tracer viscosity " << nu_T << " negative";
    throw std::invalid_argument(s.str());
  }
  const auto n = static_cast<std::size_t>(space.num_elements() * space.points_per_element());
  if (H.extent(0) != n || H.extent(1) != 1) {
    throw std::invalid_argument("swe_ocean: H is " + std::to_string(H.extent(0)) + " x " +
                                std::to_string(H.extent(1)) + ", not " + std::to_string(n) + " x 1");
  }
  if (K_ < 1) throw std::invalid_argument("swe_ocean: the number of levels must be at least 1");
  const auto K = static_cast<std::size_t>(K_), nt = static_cast<std::size_t>(count);
  h0_ = Values("ocean::tracers::h0", n, K), ratio_ = Values("ocean::tracers::ratio", n, K);
  eta_ = Values("ocean::tracers::eta", n, 1), h_ = Values("ocean::tracers::h", n, K);
  hT_ = Values("ocean::tracers::hT", n, K * nt), anomaly_ = Values("ocean::tracers::anomaly", n, K * nt);
  w_ = Values("ocean::tracers::w", n, K + 1);
  hs_ = Values("ocean::tracers::h_stage", n, K), kappa_ = Values("ocean::tracers::kappa", n, K);
  lower_ = Values("ocean::tracers::lower", n, K), diag_ = Values("ocean::tracers::diag", n, K);
  upper_ = Values("ocean::tracers::upper", n, K);
  Fk_ = Values("ocean::tracers::F_k", n, 3 * K);
  divF_ = Values("ocean::tracers::div_F_k", n, K), T_ = Values("ocean::tracers::T", n, K);
  level_ = Values("ocean::tracers::level", n, 1), vec_ = Values("ocean::tracers::vec", n, 3);
  grad_ = Values("ocean::tracers::grad", n, 3), div_ = Values("ocean::tracers::div", n, 1);
  mean_.assign(nt, 0.0);
  std::vector<double> column(K);
  for (std::size_t r = 0; r < n; ++r) {
    reference_thicknesses(spec, H(r, 0), column);
    for (std::size_t k = 0; k < K; ++k) {
      h0_(r, k) = column[k];
      ratio_(r, k) = column[k] / H(r, 0);
    }
  }
  thicken(eta_);
}

void Tracers::thicken(ConstValues eta, bool keep) {
  const auto h0 = h0_, h = h_, a = anomaly_, e = eta_;
  const auto H = H_;
  const int K = K_, nt = count_;
  Kokkos::parallel_for(
      "ocean::tracers::thicken", rows(h_), KOKKOS_LAMBDA(const LO r) {
        e(r, 0) = eta(r, 0);
        for (int k = 0; k < K; ++k) {
          const double before = h(r, k);
          h(r, k) = h0(r, k) * (1 + eta(r, 0) / H(r, 0));
          if (keep) {
            for (int t = 0; t < nt; ++t) a(r, t * K + k) = a(r, t * K + k) / before * h(r, k);
          }
        }
      });
  Kokkos::fence();
  compose();
}

void Tracers::compose() {
  const auto h = h_, a = anomaly_, hT = hT_;
  const int K = K_;
  for (int t = 0; t < count_; ++t) {
    const double mean = mean_[static_cast<std::size_t>(t)];
    Kokkos::parallel_for(
        "ocean::tracers::compose", rows(h_), KOKKOS_LAMBDA(const LO r) {
          for (int k = 0; k < K; ++k) hT(r, t * K + k) = mean * h(r, k) + a(r, t * K + k);
        });
    Kokkos::fence();
  }
}

void Tracers::set(const soma::Comm& comm,
                  const std::function<double(const std::array<double, 3>&, double z, int)>& value) {
  const auto x = space_->x();
  for (std::size_t r = 0; r < h_.extent(0); ++r) {
    const std::array<double, 3> p{x(r, 0), x(r, 1), x(r, 2)};
    double top = eta_(r, 0);
    for (int k = 0; k < K_; ++k) {
      const auto kk = static_cast<std::size_t>(k);
      const double middle = top - h_(r, kk) / 2;
      for (int t = 0; t < count_; ++t) {
        const auto c = static_cast<std::size_t>(t * K_ + k);
        anomaly_(r, c) = value(p, middle, t);  // T, for now
        hT_(r, c) = anomaly_(r, c) * h_(r, kk);
      }
      top -= h_(r, kk);
    }
  }
  if (count_ == 0) return;
  double volume = 0;
  for (int k = 0; k < K_; ++k) volume += space_->integral(comm, h_, k);
  for (int t = 0; t < count_; ++t) {
    const double mean = volume > 0 ? total(comm, t) / volume : 0.0;
    mean_[static_cast<std::size_t>(t)] = mean;
    for (std::size_t r = 0; r < h_.extent(0); ++r) {
      for (int k = 0; k < K_; ++k) {
        const auto c = static_cast<std::size_t>(t * K_ + k);
        anomaly_(r, c) = (anomaly_(r, c) - mean) * h_(r, static_cast<std::size_t>(k));
      }
    }
  }
  compose();
}

double Tracers::total(const soma::Comm& comm, int tracer) const {
  if (tracer < 0 || tracer >= count_) {
    throw std::out_of_range("swe_ocean: tracer " + std::to_string(tracer) + " of " + std::to_string(count_));
  }
  double sum = 0;
  for (int k = 0; k < K_; ++k) sum += space_->integral(comm, hT_, tracer * K_ + k);
  return sum;
}

double Tracers::w_surface_max(const soma::Comm& comm) const {
  const auto owned = static_cast<std::size_t>(space_->owned_elements() * space_->points_per_element());
  double most = 0;
  for (std::size_t r = 0; r < owned; ++r) most = std::max(most, std::abs(w_(r, 0)));
  return comm.allreduce(most, soma::Op::max);
}

void Tracers::tendency(const soma::Comm& comm, ConstValues eta, ConstValues G, ConstValues deta, ConstValues a,
                       Values da) {
  const Space& s = *space_;
  const int K = K_;
  const auto H = H_;
  const auto h0 = h0_, ratio = ratio_, hs = hs_, Fk = Fk_, divF = divF_, w = w_, T = T_;
  const auto level = level_, vec = vec_, grad = grad_, div = div_, kappa = kappa_;

  // The stage's thicknesses and level fluxes F_k = r_k G.
  Kokkos::parallel_for(
      "ocean::tracers::levels", rows(hs_), KOKKOS_LAMBDA(const LO r) {
        for (int k = 0; k < K; ++k) {
          hs(r, k) = h0(r, k) * (1 + eta(r, 0) / H(r, 0));
          for (int c = 0; c < 3; ++c) Fk(r, 3 * k + c) = ratio(r, k) * G(r, c);
        }
      });
  Kokkos::fence();
  // The diffusive flux's weight: each level's least stage thickness over the element, at its every point.
  const LO ppe = s.points_per_element();
  Kokkos::parallel_for(
      "ocean::tracers::kappa", Policy(0, s.num_elements()), KOKKOS_LAMBDA(const LO e) {
        for (int k = 0; k < K; ++k) {
          double least = hs(e * ppe, k);
          for (LO q = 1; q < ppe; ++q) least = Kokkos::min(least, hs(e * ppe + q, k));
          for (LO q = 0; q < ppe; ++q) kappa(e * ppe + q, k) = least;
        }
      });
  Kokkos::fence();
  // Their divergences, level by level, with the operator of the continuity equation.
  for (int k = 0; k < K; ++k) {
    Kokkos::parallel_for(
        "ocean::tracers::level_flux", rows(vec_), KOKKOS_LAMBDA(const LO r) {
          for (int c = 0; c < 3; ++c) vec(r, c) = Fk(r, 3 * k + c);
        });
    Kokkos::fence();
    soma::sem::divergence_weak(comm, s, vec, div);
    Kokkos::parallel_for(
        "ocean::tracers::level_divergence", rows(div_), KOKKOS_LAMBDA(const LO r) { divF(r, k) = div(r, 0); });
    Kokkos::fence();
  }
  // w from the bottom up: w_K = 0, w_(k-1) = w_k - div_wk(F_k) - r_k d eta/dt (level k in column k - 1).
  Kokkos::parallel_for(
      "ocean::tracers::w", rows(w_), KOKKOS_LAMBDA(const LO r) {
        w(r, K) = 0;
        for (int k = K - 1; k >= 0; --k) w(r, k) = w(r, k + 1) - divF(r, k) - ratio(r, k) * deta(r, 0);
      });
  Kokkos::fence();

  const double nu = nu_;
  for (int t = 0; t < count_; ++t) {
    // The anomaly T_k - Tbar.
    Kokkos::parallel_for(
        "ocean::tracers::T", rows(T_), KOKKOS_LAMBDA(const LO r) {
          for (int k = 0; k < K; ++k) T(r, k) = a(r, t * K + k) / hs(r, k);
        });
    Kokkos::fence();
    for (int k = 0; k < K; ++k) {
      // Advection along the level, -div_wk(F_k T_k).
      Kokkos::parallel_for(
          "ocean::tracers::advective_flux", rows(vec_), KOKKOS_LAMBDA(const LO r) {
            for (int c = 0; c < 3; ++c) vec(r, c) = Fk(r, 3 * k + c) * T(r, k);
          });
      Kokkos::fence();
      soma::sem::divergence_weak(comm, s, vec, div);
      Kokkos::parallel_for(
          "ocean::tracers::tendency", rows(div_),
          KOKKOS_LAMBDA(const LO r) { da(r, t * K + k) = -div(r, 0); });
      Kokkos::fence();
      if (!(nu > 0)) continue;
      // The thickness-weighted biharmonic: -nu_T div_wk(kappa_k grad(L_k T_k)), L_k T = div_wk(kappa_k grad T) /
      // h_k.
      Kokkos::parallel_for(
          "ocean::tracers::level_T", rows(level_), KOKKOS_LAMBDA(const LO r) { level(r, 0) = T(r, k); });
      Kokkos::fence();
      for (int pass = 0; pass < 2; ++pass) {
        soma::sem::gradient(s, level, grad);
        Kokkos::parallel_for(
            "ocean::tracers::diffusive_flux", rows(vec_), KOKKOS_LAMBDA(const LO r) {
              for (int c = 0; c < 3; ++c) vec(r, c) = kappa(r, k) * grad(r, c);
            });
        Kokkos::fence();
        soma::sem::divergence_weak(comm, s, vec, div);
        if (pass == 0) {
          Kokkos::parallel_for(
              "ocean::tracers::laplacian", rows(level_),
              KOKKOS_LAMBDA(const LO r) { level(r, 0) = div(r, 0) / hs(r, k); });
        } else {
          Kokkos::parallel_for(
              "ocean::tracers::diffusion", rows(div_),
              KOKKOS_LAMBDA(const LO r) { da(r, t * K + k) -= nu * div(r, 0); });
        }
        Kokkos::fence();
      }
    }
  }
}

void Tracers::vertical(ConstValues eta, double c, Values a) {
  const int K = K_;
  const auto H = H_;
  const auto h0 = h0_, hs = hs_, w = w_, x = T_, lower = lower_, diag = diag_, upper = upper_;
  for (int t = 0; t < count_; ++t) {
    Kokkos::parallel_for(
        "ocean::tracers::vertical", rows(hs_), KOKKOS_LAMBDA(const LO r) {
          // The column's system for the anomaly x_k = T_k - Tbar at the stage's end (levels 1-based as in the
          // header): h_k x_k + c (wx)_(k-1) - c (wx)_k = a_k, a the explicit update, (wx) upwinded and zero at the
          // surface and the bottom. A level's outflow sits on its diagonal and is its neighbour's inflow: the
          // columns are diagonally dominant while h_k > 0, so the Thomas sweep is stable and makes no new extremum.
          // Below, column k holds level k + 1 and interface j lies between columns j - 1 and j.
          for (int k = 0; k < K; ++k) {
            hs(r, k) = h0(r, k) * (1 + eta(r, 0) / H(r, 0));
            lower(r, k) = 0, diag(r, k) = hs(r, k), upper(r, k) = 0;
            x(r, k) = a(r, t * K + k);
            if (k > 0) {
              const double up = c * w(r, k);
              if (up > 0) {
                diag(r, k) += up;
              } else {
                lower(r, k) = up;
              }
            }
            if (k + 1 < K) {
              const double up = c * w(r, k + 1);
              if (up > 0) {
                upper(r, k) = -up;
              } else {
                diag(r, k) -= up;
              }
            }
          }
          upper(r, 0) /= diag(r, 0);
          x(r, 0) /= diag(r, 0);
          for (int k = 1; k < K; ++k) {
            const double pivot = diag(r, k) - lower(r, k) * upper(r, k - 1);
            upper(r, k) /= pivot;
            x(r, k) = (x(r, k) - lower(r, k) * x(r, k - 1)) / pivot;
          }
          for (int k = K - 2; k >= 0; --k) x(r, k) -= upper(r, k) * x(r, k + 1);
          // The fluxes of the solution, applied in flux form: the column's sum of a is unchanged.
          const auto flux = [&](int j) {
            if (j == 0 || j == K) return 0.0;
            return c * w(r, j) * (w(r, j) > 0 ? x(r, j) : x(r, j - 1));
          };
          for (int k = 0; k < K; ++k) a(r, t * K + k) += flux(k + 1) - flux(k);
        });
    Kokkos::fence();
  }
}

}  // namespace ocean
