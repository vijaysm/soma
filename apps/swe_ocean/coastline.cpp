// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "coastline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "soma/algorithms/grid_index.hpp"
#include "soma/core/view_types.hpp"

namespace ocean {

namespace {

using P3 = std::array<double, 3>;
using soma::LO;
using soma::Real;

constexpr double deg = std::numbers::pi / 180;
constexpr double earth_km = a / 1000;

double dot(const P3& u, const P3& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }
P3 cross(const P3& u, const P3& v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
}
P3 unit(const P3& u) {
  const double n = std::sqrt(dot(u, u));
  return {u[0] / n, u[1] / n, u[2] / n};
}
double angle(const P3& u, const P3& v) { return std::atan2(std::sqrt(dot(cross(u, v), cross(u, v))), dot(u, v)); }

// A kept point and its bin, as gathered.
struct Coast {
  double x, y, z;
  std::uint64_t bin;
};

// The decimation bins: latitude bins of h degrees, each circle of latitude in the least number of equal
// longitude bins at most h / cos(c) wide (c the bin's central latitude), at least one.
class Bins {
 public:
  explicit Bins(double h_deg) : h_(h_deg), rows_(std::max<std::int64_t>(1, std::llround(std::ceil(180 / h_deg)))) {}

  [[nodiscard]] std::uint64_t operator()(double lat, double lon) const {
    const auto row = static_cast<std::int64_t>(std::floor((lat + 90) / h_));
    const std::int64_t i = std::clamp<std::int64_t>(row, 0, rows_ - 1);
    const double centre = std::clamp(-90 + (static_cast<double>(i) + 0.5) * h_, -90.0, 90.0);
    const double columns = std::max(1.0, std::ceil(360 * std::cos(centre * deg) / h_));
    double turn = (lon + 180) / 360;
    turn -= std::floor(turn);
    const auto j = std::clamp<std::int64_t>(static_cast<std::int64_t>(std::floor(turn * columns)), 0,
                                            static_cast<std::int64_t>(columns) - 1);
    return (static_cast<std::uint64_t>(i) << 32U) | static_cast<std::uint64_t>(j);
  }

 private:
  double h_;
  std::int64_t rows_;
};

// The great-circle distance (radians) from p to the cell with corners q (unit vectors): 0 inside it (on the
// inner side of every side, the sides oriented by the cell's area vector), else the least distance to a side.
double distance(const P3& p, const std::vector<P3>& q, double sense) {
  bool inside = true;
  double least = std::numbers::pi;
  for (std::size_t i = 0; i < q.size(); ++i) {
    const P3& u = q[i];
    const P3& v = q[(i + 1) % q.size()];
    const P3 n = unit(cross(u, v));
    const double side = dot(p, n);
    inside = inside && sense * side >= 0;
    const P3 foot = unit({p[0] - side * n[0], p[1] - side * n[1], p[2] - side * n[2]});
    const bool between = dot(cross(u, foot), n) >= 0 && dot(cross(foot, v), n) >= 0;
    least = std::min(least, between ? std::asin(std::min(std::abs(side), 1.0)) : std::min(angle(p, u), angle(p, v)));
  }
  return inside ? 0 : least;
}

}  // namespace

std::vector<std::array<double, 3>> coastline(const soma::Comm& comm, const GridSpec& spec, double spacing_km,
                                             std::size_t strip_rows) {
  if (!std::isfinite(spacing_km) || !(spacing_km > 0)) throw std::invalid_argument("coastline: spacing_km must be > 0");
  if (strip_rows == 0) throw std::invalid_argument("coastline: strip_rows must be > 0");
  const GridInfo info = open_grid(comm, spec);
  const std::size_t N = info.lat.size(), M = info.lon.size();
  const auto P = static_cast<std::size_t>(comm.size()), r = static_cast<std::size_t>(comm.rank());
  const std::size_t first = r * N / P, last = (r + 1) * N / P;  // this rank's rows: first .. last - 1
  const Bins bin(spacing_km / 2 / earth_km / deg);

  std::vector<Coast> kept;
  std::unordered_set<std::uint64_t> seen;
  const auto crossing = [&](double z0, double z1, double lat0, double lon0, double lat1, double lon1) {
    if ((z0 <= 0) == (z1 <= 0)) return;
    const double t = z0 / (z0 - z1);
    const double lat = lat0 + t * (lat1 - lat0), lon = lon0 + t * (lon1 - lon0);
    const std::uint64_t b = bin(lat, lon);
    if (!seen.insert(b).second) return;
    kept.push_back({std::cos(lat * deg) * std::cos(lon * deg), std::cos(lat * deg) * std::sin(lon * deg),
                    std::sin(lat * deg), b});
  };
  // Row i's edges, in (column, edge) order: the row edge east (wrapping on a periodic grid), then the column
  // edge north to `above` (none for the top row).
  const auto edges = [&](std::size_t i, std::span<const double> row, std::span<const double> above) {
    for (std::size_t j = 0; j < M; ++j) {
      if (j + 1 < M) {
        crossing(row[j], row[j + 1], info.lat[i], info.lon[j], info.lat[i], info.lon[j + 1]);
      } else if (info.periodic) {
        crossing(row[j], row[0], info.lat[i], info.lon[j], info.lat[i], info.lon[j] + info.dlon);
      }
      if (!above.empty()) crossing(row[j], above[j], info.lat[i], info.lon[j], info.lat[i + 1], info.lon[j]);
    }
  };
  // A read failure on one rank throws on every rank (with its message there), before the gather.
  std::string failure;
  try {
    if (first < last) {
      // Rows first .. end - 1, one past the band for its column edges, by strips in ascending order; the row
      // before a strip is carried over.
      const std::size_t end = std::min(last + 1, N);
      std::vector<double> strip, previous;
      for (std::size_t i0 = first; i0 < end; i0 += strip_rows) {
        const std::size_t k = std::min(strip_rows, end - i0);
        strip.assign(k * M, 0);
        const Window w{info.south_first ? i0 : N - (i0 + k), k, 0, M};
        read_window(spec, info, w, strip_rows, [&](std::size_t i, std::span<const double> row) {
          std::copy(row.begin(), row.end(), strip.begin() + static_cast<std::ptrdiff_t>((i - i0) * M));
        });
        for (std::size_t s = 0; s < k; ++s) {
          const std::span<const double> row(strip.data() + s * M, M);
          if (!previous.empty()) edges(i0 + s - 1, previous, row);
          previous.assign(row.begin(), row.end());
        }
      }
      // The grid's top row has no row above it; its row edges are still this rank's.
      if (last == N) edges(N - 1, previous, {});
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  const std::int32_t failed = failure.empty() ? 0 : 1;
  if (comm.allreduce(failed, soma::Op::max) != 0) {
    throw std::runtime_error(failure.empty() ? "coastline: failed to read the grid on another rank" : failure);
  }

  // Every rank's kept points in rank order (row order), then the first of each bin.
  const std::vector<Coast> all = comm.allgatherv(std::span<const Coast>(kept));
  std::vector<std::array<double, 3>> out;
  seen.clear();
  for (const Coast& c : all) {
    if (seen.insert(c.bin).second) out.push_back({c.x, c.y, c.z});
  }
  return out;
}

soma::GradedTarget coast_target(std::vector<std::array<double, 3>> points) {
  using Boxes = soma::GridIndex<Kokkos::HostSpace>::Boxes;
  // Each point a box of half-width pad (6 mm on the Earth): zero-extent boxes would give the index one bin
  // (it sizes bins by the boxes' mean extent); with a tiny extent it takes its budget of bins (at most 2n).
  constexpr double pad = 1e-9;
  struct State {
    std::vector<P3> points;
    Kokkos::View<Real**, Kokkos::LayoutLeft, Kokkos::HostSpace> boxes;
    std::unique_ptr<soma::GridIndex<Kokkos::HostSpace>> index;
  };
  auto state = std::make_shared<State>();
  state->points = std::move(points);
  const std::size_t n = state->points.size();
  state->boxes = decltype(state->boxes)("ocean::coast_target::boxes", n, 6);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t c = 0; c < 3; ++c) {
      state->boxes(k, c) = state->points[k][c] - pad;
      state->boxes(k, c + 3) = state->points[k][c] + pad;
    }
  }
  if (n > 0) state->index = std::make_unique<soma::GridIndex<Kokkos::HostSpace>>(Boxes(state->boxes));

  return [state](const soma::HostMesh& m, const std::vector<Real>& grow) {
    const LO owned = m.segment_table(2).owned_end;
    if (grow.size() != static_cast<std::size_t>(owned)) {
      throw std::invalid_argument("coast_target: grow must hold one value per owned cell");
    }
    std::vector<std::uint8_t> marked(grow.size(), 0);
    if (!state->index) return marked;
    const auto mv = m.view();
    const auto x = m.coords();
    // The cells asked: corners (unit vectors), orientation, and a query box about the centroid reaching the
    // circumradius plus grow (a chord: every point within that angle lies within it on every axis).
    std::vector<LO> asked;
    std::vector<std::vector<P3>> corners;
    std::vector<double> sense;
    for (LO e = 0; e < owned; ++e) {
      if (!(grow[static_cast<std::size_t>(e)] >= 0)) continue;
      std::vector<P3> q;
      for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
        const LO v = mv.vertex(2, e, i);
        q.push_back(unit({x(v, 0), x(v, 1), x(v, 2)}));
      }
      asked.push_back(e);
      corners.push_back(std::move(q));
    }
    Kokkos::View<Real**, Kokkos::LayoutLeft, Kokkos::HostSpace> queries("ocean::coast_target::queries",
                                                                       asked.size(), 6);
    for (std::size_t k = 0; k < asked.size(); ++k) {
      const auto& q = corners[k];
      P3 area{0, 0, 0}, mean{0, 0, 0};
      for (std::size_t i = 0; i < q.size(); ++i) {
        const P3 t = cross(q[i], q[(i + 1) % q.size()]);
        for (std::size_t c = 0; c < 3; ++c) {
          area[c] += t[c];
          mean[c] += q[i][c];
        }
      }
      sense.push_back(dot(area, mean) >= 0 ? 1 : -1);
      const P3 centre = unit(mean);
      double radius = 0;
      for (const P3& p : q) radius = std::max(radius, angle(centre, p));
      const double reach = std::min(radius + grow[static_cast<std::size_t>(asked[k])] / earth_km, std::numbers::pi);
      // The chord of the reach, widened by a relative 1e-9 and the boxes' pad against rounding.
      const double half = 2 * std::sin(reach / 2) * (1 + 1e-9) + 2 * pad;
      for (std::size_t c = 0; c < 3; ++c) {
        queries(k, c) = centre[c] - half;
        queries(k, c + 3) = centre[c] + half;
      }
    }
    const auto pairs = state->index->query(Boxes(queries));
    for (std::size_t k = 0; k < asked.size(); ++k) {
      const double g = grow[static_cast<std::size_t>(asked[k])];
      for (auto t = pairs.offsets(k); t < pairs.offsets(k + 1); ++t) {
        const P3& p = state->points[static_cast<std::size_t>(pairs.boxes(t))];
        if (distance(p, corners[k], sense[k]) * earth_km <= g) {
          marked[static_cast<std::size_t>(asked[k])] = 1;
          break;
        }
      }
    }
    return marked;
  };
}

}  // namespace ocean
