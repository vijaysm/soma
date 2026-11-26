// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "constants.hpp"
#include "soma/algorithms/grid_index.hpp"
#include "soma/fields/node_tables.hpp"

namespace ocean {

namespace {

using P3 = std::array<double, 3>;
using soma::GO;
using soma::LO;

double dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The Lagrange basis on the nodes z at s, and its derivative.
void lagrange(const std::vector<double>& z, double s, std::vector<double>& l, std::vector<double>& dl) {
  const std::size_t n = z.size();
  for (std::size_t i = 0; i < n; ++i) {
    double value = 1, slope = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == i) continue;
      const double d = z[i] - z[j];
      // (value * (s - z_j))' = slope (s - z_j) + value, all over d.
      slope = (slope * (s - z[j]) + value) / d;
      value = value * (s - z[j]) / d;
    }
    l[i] = value;
    dl[i] = slope;
  }
}

constexpr GO no_element = std::numeric_limits<GO>::max();
constexpr double inside_tolerance = 1e-9;

}  // namespace

struct Sampler::Index {
  Kokkos::View<double**, Kokkos::LayoutLeft, Host> boxes;
  std::unique_ptr<soma::GridIndex<Host>> grid;
  std::vector<double> nodes;
};

Sampler::Sampler(const Space& space, const soma::HostMesh& mesh)
    : space_(&space), mesh_(&mesh), index_(std::make_unique<Index>()) {
  const LO owned = space.owned_elements(), ppe = space.points_per_element();
  const auto x = space.x();
  index_->nodes = soma::gll_points_1d(space.order());
  index_->boxes = Kokkos::View<double**, Kokkos::LayoutLeft, Host>("ocean::Sampler::boxes",
                                                                   static_cast<std::size_t>(owned), 6);
  for (LO e = 0; e < owned; ++e) {
    P3 lo{x(static_cast<std::size_t>(e * ppe), 0), x(static_cast<std::size_t>(e * ppe), 1),
          x(static_cast<std::size_t>(e * ppe), 2)};
    P3 hi = lo;
    for (LO k = 0; k < ppe; ++k) {
      const auto r = static_cast<std::size_t>(e * ppe + k);
      for (std::size_t d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], x(r, d));
        hi[d] = std::max(hi[d], x(r, d));
      }
    }
    const double margin = 0.25 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    for (std::size_t d = 0; d < 3; ++d) {
      index_->boxes(static_cast<std::size_t>(e), d) = lo[d] - margin;
      index_->boxes(static_cast<std::size_t>(e), d + 3) = hi[d] + margin;
    }
  }
  index_->grid = std::make_unique<soma::GridIndex<Host>>(index_->boxes);
}

Sampler::~Sampler() = default;

std::vector<double> Sampler::sample(const soma::Comm& comm, ConstValues values,
                                    std::span<const std::array<double, 3>> points) const {
  const Space& s = *space_;
  const LO ppe = s.points_per_element();
  const auto rows = static_cast<std::size_t>(s.num_elements()) * static_cast<std::size_t>(ppe);
  const std::int32_t bad = comm.allreduce(static_cast<std::int32_t>(values.extent(0) != rows ? 1 : 0), soma::Op::max);
  if (bad != 0) throw std::invalid_argument("Sampler::sample: values need a row per point of the space");
  const std::size_t nc = values.extent(1), n = points.size();
  const bool sphere = s.geometry().kind == soma::sem::SemGeometry::Kind::sphere;
  const double radius = sphere ? s.geometry().radius : 1;

  // The points as the space sees them: on the sphere at its radius, in the plane at z = 0.
  std::vector<P3> where(n);
  Kokkos::View<double**, Kokkos::LayoutLeft, Host> queries("ocean::Sampler::queries", n, 6);
  for (std::size_t q = 0; q < n; ++q) {
    P3 p = points[q];
    if (sphere) {
      const double len = std::sqrt(dot(p, p));
      for (double& c : p) c = len > 0 ? c / len * radius : 0;
    } else {
      p[2] = 0;
    }
    where[q] = p;
    for (std::size_t d = 0; d < 3; ++d) queries(q, d) = queries(q, d + 3) = p[d];
  }
  const auto pairs = index_->grid->query(queries);

  const auto x = s.x();
  const auto gid = mesh_->gid(2);
  const std::vector<double>& z = index_->nodes;
  const std::size_t np = z.size();
  std::vector<double> lx(np), dlx(np), ly(np), dly(np);
  // Gauss-Newton for the point p in element e: true and (xi, eta) when the element holds it.
  const auto locate = [&](LO e, const P3& p, double& xi, double& eta) {
    const P3 dir = sphere ? P3{p[0] / radius, p[1] / radius, p[2] / radius} : P3{0, 0, 0};
    const auto base = static_cast<std::size_t>(e * ppe);
    xi = 0;
    eta = 0;
    for (int it = 0; it < 50; ++it) {
      lagrange(z, xi, lx, dlx);
      lagrange(z, eta, ly, dly);
      P3 X{0, 0, 0}, A{0, 0, 0}, B{0, 0, 0};
      for (std::size_t j = 0; j < np; ++j) {
        for (std::size_t i = 0; i < np; ++i) {
          const std::size_t r = base + i + np * j;
          for (std::size_t d = 0; d < 3; ++d) {
            X[d] += lx[i] * ly[j] * x(r, d);
            A[d] += dlx[i] * ly[j] * x(r, d);
            B[d] += lx[i] * dly[j] * x(r, d);
          }
        }
      }
      P3 res{X[0] - p[0], X[1] - p[1], X[2] - p[2]};
      if (sphere) {
        if (dot(X, dir) <= 0) return false;
        // The parts normal to the direction: X along it is the point.
        const double rx = dot(X, dir), ax = dot(A, dir), bx = dot(B, dir);
        for (std::size_t d = 0; d < 3; ++d) {
          res[d] = X[d] - rx * dir[d];
          A[d] -= ax * dir[d];
          B[d] -= bx * dir[d];
        }
      } else {
        res[2] = A[2] = B[2] = 0;
      }
      const double a11 = dot(A, A), a12 = dot(A, B), a22 = dot(B, B), det = a11 * a22 - a12 * a12;
      if (!(det > 0)) return false;
      const double g1 = -dot(A, res), g2 = -dot(B, res);
      const double dxi = (a22 * g1 - a12 * g2) / det, deta = (a11 * g2 - a12 * g1) / det;
      xi += dxi;
      eta += deta;
      if (!(std::abs(xi) < 3 && std::abs(eta) < 3)) return false;
      if (std::max(std::abs(dxi), std::abs(deta)) < 1e-13) {
        return std::abs(xi) <= 1 + inside_tolerance && std::abs(eta) <= 1 + inside_tolerance;
      }
    }
    return false;
  };

  // Each point's least holding gid here, its value, then the least over ranks.
  std::vector<GO> best(n, no_element);
  std::vector<double> mine(n * nc, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (soma::Offset k = pairs.offsets(q); k < pairs.offsets(q + 1); ++k) {
      const LO e = pairs.boxes(static_cast<std::size_t>(k));
      if (gid(e) >= best[q]) continue;
      double xi = 0, eta = 0;
      if (!locate(e, where[q], xi, eta)) continue;
      best[q] = gid(e);
      xi = std::clamp(xi, -1.0, 1.0);
      eta = std::clamp(eta, -1.0, 1.0);
      lagrange(z, xi, lx, dlx);
      lagrange(z, eta, ly, dly);
      const auto base = static_cast<std::size_t>(e * ppe);
      for (std::size_t c = 0; c < nc; ++c) {
        double v = 0;
        for (std::size_t j = 0; j < np; ++j) {
          for (std::size_t i = 0; i < np; ++i) v += lx[i] * ly[j] * values(base + i + np * j, c);
        }
        mine[q * nc + c] = v;
      }
    }
  }
  std::vector<GO> least = best;
  comm.allreduce(std::span<GO>(least), soma::Op::min);
  // Only the owner of the least gid adds its value: a sum of one value and zeros, exact.
  for (std::size_t q = 0; q < n; ++q) {
    if (best[q] != least[q]) std::fill_n(mine.begin() + static_cast<std::ptrdiff_t>(q * nc), nc, 0.0);
  }
  comm.allreduce(std::span<double>(mine), soma::Op::sum);
  for (std::size_t q = 0; q < n; ++q) {
    if (least[q] == no_element) {
      std::fill_n(mine.begin() + static_cast<std::ptrdiff_t>(q * nc), nc, std::numeric_limits<double>::quiet_NaN());
    }
  }
  return mine;
}

Transport meridian_transport(const soma::Comm& comm, const Sampler& sampler, const Space& space, ConstValues H,
                             ConstValues eta, ConstValues u, ConstValues land, const Section& section) {
  if (space.geometry().kind != soma::sem::SemGeometry::Kind::sphere) {
    throw std::invalid_argument("meridian_transport: needs a space on the sphere");
  }
  const Section& c = section;
  const auto in_range = [](double lat) { return lat >= -90 && lat <= 90; };
  if (!(c.spacing > 0) || !std::isfinite(c.spacing) || !in_range(c.lat_south) || !in_range(c.lat_north) ||
      !(c.lat_south <= c.lat_start && c.lat_start <= c.lat_north) || !std::isfinite(c.lon)) {
    throw std::invalid_argument("meridian_transport: needs lat_south <= lat_start <= lat_north in [-90, 90] and a "
                                "positive spacing");
  }
  const double deg = std::numbers::pi / 180;
  const auto count = static_cast<std::size_t>(std::floor((c.lat_north - c.lat_south) / c.spacing + 1e-9)) + 1;
  std::vector<P3> points(count);
  const double lon = c.lon * deg;
  for (std::size_t i = 0; i < count; ++i) {
    const double lat = (c.lat_south + c.spacing * static_cast<double>(i)) * deg;
    points[i] = {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
  }
  // D and the eastward velocity at every row; e_east is the same along the meridian.
  const P3 east{-std::sin(lon), std::cos(lon), 0};
  const bool flags = land.extent(0) > 0;
  Values flow("ocean::meridian_transport::flow", H.extent(0), 2);
  for (std::size_t r = 0; r < H.extent(0); ++r) {
    flow(r, 0) = (H(r, 0) + eta(r, 0)) * (u(r, 0) * east[0] + u(r, 1) * east[1] + u(r, 2) * east[2]);
    flow(r, 1) = flags && r < land.extent(0) ? land(r, 0) : 0.0;
  }
  const std::vector<double> both = sampler.sample(comm, flow, points);
  // The integrand, NaN where dry.
  std::vector<double> q(count);
  for (std::size_t i = 0; i < count; ++i) {
    q[i] = both[2 * i + 1] >= 0.5 ? std::numeric_limits<double>::quiet_NaN() : both[2 * i];
  }

  Transport t;
  const auto start = static_cast<std::size_t>(std::lround((c.lat_start - c.lat_south) / c.spacing));
  if (start >= count || std::isnan(q[start])) return t;
  std::size_t first = start, last = start;
  while (first > 0 && !std::isnan(q[first - 1])) --first;
  while (last + 1 < count && !std::isnan(q[last + 1])) ++last;
  double sum = 0;
  for (std::size_t i = first; i < last; ++i) sum += 0.5 * (q[i] + q[i + 1]);
  t.sv = sum * a * c.spacing * deg / 1e6;
  t.lat_from = c.lat_south + c.spacing * static_cast<double>(first);
  t.lat_to = c.lat_south + c.spacing * static_cast<double>(last);
  t.wet = true;
  return t;
}

}  // namespace ocean
