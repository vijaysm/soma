// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "bathymetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/sem/operators.hpp"
#include "vertical.hpp"

namespace ocean {

namespace {

constexpr double deg = std::numbers::pi / 180;
constexpr int power_iterations = 20;

std::size_t all_rows(const Space& s) {
  return static_cast<std::size_t>(s.num_elements()) * static_cast<std::size_t>(s.points_per_element());
}

std::size_t owned_rows(const Space& s) {
  return static_cast<std::size_t>(s.owned_elements()) * static_cast<std::size_t>(s.points_per_element());
}

std::int64_t wrap(std::int64_t j, std::int64_t n) {
  j %= n;
  return j < 0 ? j + n : j;
}

// A point's footprint and what it gathers. Rows are normalised (ascending latitude) indices; columns run from
// col_lo for ncol columns, wrapping on a periodic grid. The bilinear stencil is rows ib, ib + 1 and columns
// jb, jb + 1 (wrapped), at fractions t and s between them.
struct Footprint {
  double u[3] = {0, 0, 0};
  double chord2 = 0;  // the squared chord of the footprint's angular radius
  bool disc = false;
  std::int64_t row_lo = 0, row_hi = 0, col_lo = 0, ncol = 0;
  std::int64_t ib = 0, jb = 0;
  double t = 0, s = 0;
  double weight = 0, sum = 0, sum2 = 0, bilinear = 0;
};

Footprint footprint(const GridInfo& info, const double* x, double mass, double radius, double bilinear_below) {
  Footprint f;
  const auto nlat = static_cast<std::int64_t>(info.lat.size()), nlon = static_cast<std::int64_t>(info.lon.size());
  const double norm = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  for (int c = 0; c < 3; ++c) f.u[c] = x[c] / norm;
  const double lat = std::asin(std::clamp(f.u[2], -1.0, 1.0)) / deg, lon = std::atan2(f.u[1], f.u[0]) / deg;
  const double rho = std::sqrt(mass / std::numbers::pi), delta = rho / radius;
  f.disc = rho >= bilinear_below;
  f.chord2 = 4 * std::sin(delta / 2) * std::sin(delta / 2);

  // The bilinear stencil.
  const double lat0 = info.lat.front(), lon0 = info.lon.front();
  f.ib = std::clamp(static_cast<std::int64_t>(std::floor((lat - lat0) / info.dlat)), std::int64_t{0}, nlat - 2);
  f.t = std::clamp((lat - info.lat[static_cast<std::size_t>(f.ib)]) / info.dlat, 0.0, 1.0);
  const double u = (lon - lon0) / info.dlon;
  if (info.periodic) {
    f.jb = static_cast<std::int64_t>(std::floor(u));
    f.s = std::clamp(u - static_cast<double>(f.jb), 0.0, 1.0);
  } else {
    f.jb = std::clamp(static_cast<std::int64_t>(std::floor(u)), std::int64_t{0}, nlon - 2);
    f.s = std::clamp(u - static_cast<double>(f.jb), 0.0, 1.0);
  }
  std::int64_t row_lo = f.ib, row_hi = f.ib + 1, col_lo = f.jb, col_hi = f.jb + 1;
  if (f.disc) {
    // The disc's latitudes, a row beyond each side against rounding; every longitude when it reaches a pole,
    // else asin(sin(delta) / cos(lat)) either side, a column beyond.
    const double dd = delta / deg;
    row_lo = std::min(row_lo, static_cast<std::int64_t>(std::floor((lat - dd - lat0) / info.dlat)) - 1);
    row_hi = std::max(row_hi, static_cast<std::int64_t>(std::ceil((lat + dd - lat0) / info.dlat)) + 1);
    if (lat + dd >= 90 || lat - dd <= -90) {
      col_lo = 0;
      col_hi = nlon - 1;
    } else {
      const double hw = std::asin(std::min(1.0, std::sin(delta) / std::cos(lat * deg))) / deg;
      col_lo = std::min(col_lo, static_cast<std::int64_t>(std::floor((lon - hw - lon0) / info.dlon)) - 1);
      col_hi = std::max(col_hi, static_cast<std::int64_t>(std::ceil((lon + hw - lon0) / info.dlon)) + 1);
    }
  }
  f.row_lo = std::max<std::int64_t>(row_lo, 0);
  f.row_hi = std::min(row_hi, nlat - 1);
  if (info.periodic && col_hi - col_lo + 1 >= nlon) {
    col_lo = 0;
    col_hi = nlon - 1;
  } else if (!info.periodic) {
    col_lo = std::max<std::int64_t>(col_lo, 0);
    col_hi = std::min(col_hi, nlon - 1);
  }
  f.col_lo = col_lo;
  f.ncol = col_hi - col_lo + 1;
  return f;
}

// The windows of one strip of rows r0 .. r1: the runs of columns the candidates' footprints reach, runs closer
// than `merge` columns joined (fewer, slightly larger reads).
std::vector<Window> strip_windows(const GridInfo& info, const std::vector<Footprint*>& candidates,
                                  std::int64_t r0, std::int64_t r1) {
  const auto nlat = static_cast<std::int64_t>(info.lat.size()), nlon = static_cast<std::int64_t>(info.lon.size());
  std::vector<std::int64_t> change(static_cast<std::size_t>(nlon) + 1, 0);
  const auto mark = [&](std::int64_t a, std::int64_t b) {  // columns a .. b - 1, 0 <= a < b <= nlon
    ++change[static_cast<std::size_t>(a)];
    --change[static_cast<std::size_t>(b)];
  };
  for (const Footprint* f : candidates) {
    if (f->ncol >= nlon) {
      mark(0, nlon);
      continue;
    }
    const std::int64_t a = wrap(f->col_lo, nlon), b = a + f->ncol;
    if (b <= nlon) {
      mark(a, b);
    } else {
      mark(a, nlon);
      mark(0, b - nlon);
    }
  }
  std::vector<std::pair<std::int64_t, std::int64_t>> runs;  // [start, end)
  std::int64_t depth = 0;
  for (std::int64_t j = 0; j < nlon; ++j) {
    depth += change[static_cast<std::size_t>(j)];
    if (depth <= 0) continue;
    if (!runs.empty() && runs.back().second == j) {
      runs.back().second = j + 1;
    } else {
      runs.emplace_back(j, j + 1);
    }
  }
  const std::int64_t merge = std::max<std::int64_t>(1, nlon / 64);
  std::vector<std::pair<std::int64_t, std::int64_t>> joined;
  for (const auto& run : runs) {
    if (!joined.empty() && run.first - joined.back().second < merge) {
      joined.back().second = run.second;
    } else {
      joined.push_back(run);
    }
  }
  const auto rows = static_cast<std::size_t>(r1 - r0 + 1);
  const auto lat0 = static_cast<std::size_t>(info.south_first ? r0 : nlat - 1 - r1);
  std::vector<Window> out;
  for (const auto& [a, b] : joined) {
    out.push_back({lat0, rows, static_cast<std::size_t>(a), static_cast<std::size_t>(b - a)});
  }
  return out;
}

// Gathers every footprint's cells from the grid, strip by strip of rows (ascending), each strip read once.
// Returns the largest window read.
std::size_t gather(const BathymetryOptions& o, const GridInfo& info, std::vector<Footprint>& feet) {
  if (feet.empty()) return 0;
  const auto nlon = static_cast<std::int64_t>(info.lon.size());
  const double cell = info.dlat * deg * info.dlon * deg;
  std::vector<double> cos_lat(info.lat.size()), sin_lat(info.lat.size());
  for (std::size_t i = 0; i < info.lat.size(); ++i) {
    cos_lat[i] = std::cos(info.lat[i] * deg);
    sin_lat[i] = std::sin(info.lat[i] * deg);
  }
  std::vector<double> cos_lon(info.lon.size()), sin_lon(info.lon.size());
  for (std::size_t j = 0; j < info.lon.size(); ++j) {
    cos_lon[j] = std::cos(info.lon[j] * deg);
    sin_lon[j] = std::sin(info.lon[j] * deg);
  }
  std::int64_t first = feet.front().row_lo, last = feet.front().row_hi;
  for (const Footprint& f : feet) {
    first = std::min(first, f.row_lo);
    last = std::max(last, f.row_hi);
  }
  const auto strip = static_cast<std::int64_t>(o.strip_rows);
  std::size_t peak = 0;
  std::vector<std::int64_t> column(static_cast<std::size_t>(nlon), -1);  // grid column -> place in a row of buf
  std::vector<double> buf;
  std::vector<Footprint*> candidates;
  for (std::int64_t r0 = first; r0 <= last; r0 += strip) {
    const std::int64_t r1 = std::min(r0 + strip - 1, last);
    candidates.clear();
    for (Footprint& f : feet) {
      if (f.row_lo <= r1 && f.row_hi >= r0) candidates.push_back(&f);
    }
    if (candidates.empty()) continue;
    const std::vector<Window> windows = strip_windows(info, candidates, r0, r1);
    std::fill(column.begin(), column.end(), -1);
    std::size_t width = 0;
    for (const Window& w : windows) {
      for (std::size_t j = 0; j < w.nlon; ++j) column[w.lon0 + j] = static_cast<std::int64_t>(width + j);
      width += w.nlon;
      peak = std::max(peak, w.nlat * w.nlon);
    }
    buf.assign(static_cast<std::size_t>(r1 - r0 + 1) * width, 0.0);
    std::size_t offset = 0;
    for (const Window& w : windows) {
      read_window(o.grid, info, w, o.strip_rows, [&](std::size_t i, std::span<const double> row) {
        double* out = buf.data() + (static_cast<std::int64_t>(i) - r0) * static_cast<std::int64_t>(width) + offset;
        std::copy(row.begin(), row.end(), out);
      });
      offset += w.nlon;
    }
    // Depth, positive down; land is 0.
    const auto depth = [&](std::int64_t i, std::int64_t j) {
      const std::int64_t at = column[static_cast<std::size_t>(wrap(j, nlon))];
      if (at < 0) throw std::logic_error("ocean: a footprint reaches a column its strip did not read");
      const double z = buf[static_cast<std::size_t>((i - r0) * static_cast<std::int64_t>(width) + at)];
      return z > 0 ? 0.0 : -z;
    };
    // Each footprint in the strip: its rows ascending, its columns from its western edge.
    for (Footprint* fp : candidates) {
      Footprint& f = *fp;
      const std::int64_t lo = std::max(f.row_lo, r0), hi = std::min(f.row_hi, r1);
      for (std::int64_t i = lo; i <= hi; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        if (f.disc) {
          const double w = cos_lat[ii] * cell;
          for (std::int64_t k = 0; k < f.ncol; ++k) {
            const std::int64_t j = f.col_lo + k;
            const auto jj = static_cast<std::size_t>(wrap(j, nlon));
            const double dx = f.u[0] - cos_lat[ii] * cos_lon[jj], dy = f.u[1] - cos_lat[ii] * sin_lon[jj];
            const double dz = f.u[2] - sin_lat[ii];
            if (dx * dx + dy * dy + dz * dz > f.chord2) continue;
            const double d = depth(i, j);
            f.weight += w;
            f.sum += w * d;
            f.sum2 += w * d * d;
          }
        }
        if (i == f.ib || i == f.ib + 1) {
          const double across = (1 - f.s) * depth(i, f.jb) + f.s * depth(i, f.jb + 1);
          f.bilinear += (i == f.ib ? 1 - f.t : f.t) * across;
        }
      }
    }
  }
  return peak;
}

// Each owned element's largest rx0 between neighbouring points along i and j; returns their maximum.
double element_rx0(const Space& space, const ConstValues& H, std::vector<double>& per_element) {
  const int np = space.order() + 1, ppe = space.points_per_element();
  per_element.assign(static_cast<std::size_t>(space.owned_elements()), 0.0);
  double worst = 0;
  for (soma::LO e = 0; e < space.owned_elements(); ++e) {
    const std::size_t base = static_cast<std::size_t>(e) * static_cast<std::size_t>(ppe);
    double m = 0;
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        const double h = H(base + static_cast<std::size_t>(i + np * j), 0);
        if (i + 1 < np) m = std::max(m, rx0(h, H(base + static_cast<std::size_t>(i + 1 + np * j), 0)));
        if (j + 1 < np) m = std::max(m, rx0(h, H(base + static_cast<std::size_t>(i + np * (j + 1)), 0)));
      }
    }
    per_element[static_cast<std::size_t>(e)] = m;
    worst = std::max(worst, m);
  }
  return worst;
}

// The largest eigenvalue of -L by power iterations from a checkerboard over each element's points (scaled
// by the point index, so no symmetry hides a mode): the Rayleigh quotient of the last iterate.
double largest_eigenvalue(const soma::Comm& comm, const Space& space) {
  const std::size_t n = all_rows(space);
  const int np = space.order() + 1, ppe = space.points_per_element();
  Values v("ocean::smooth::v", n, 1), w("ocean::smooth::Lv", n, 1), product("ocean::smooth::product", n, 1);
  for (std::size_t r = 0; r < n; ++r) {
    const int k = static_cast<int>(r % static_cast<std::size_t>(ppe)), i = k % np, j = k / np;
    v(r, 0) = ((i + j) % 2 == 0 ? 1.0 : -1.0) * (1 + 0.1 * k / ppe);
  }
  space.dss(comm, v);
  const auto dot = [&](const Values& a, const Values& b) {
    for (std::size_t r = 0; r < n; ++r) product(r, 0) = a(r, 0) * b(r, 0);
    return space.integral(comm, product);
  };
  double lambda = 0;
  for (int it = 0; it < power_iterations; ++it) {
    soma::sem::laplacian(comm, space, ConstValues(v), w);
    lambda = -dot(v, w) / dot(v, v);
    const double norm = std::sqrt(dot(w, w));
    if (!(norm > 0)) break;
    for (std::size_t r = 0; r < n; ++r) v(r, 0) = w(r, 0) / norm;
  }
  if (!(lambda > 0) || !std::isfinite(lambda)) {
    throw std::runtime_error("ocean: smooth found no positive eigenvalue of the Laplacian");
  }
  return lambda;
}

}  // namespace

Bathymetry average(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh,
                   const BathymetryOptions& options) {
  if (space.geometry().kind != soma::sem::SemGeometry::Kind::sphere) {
    throw std::invalid_argument("ocean: bathymetry needs a space on a sphere");
  }
  const std::int32_t mismatch = static_cast<soma::LO>(mesh.gid(2).extent(0)) != space.num_elements() ? 1 : 0;
  if (comm.allreduce(mismatch, soma::Op::max) != 0) {
    throw std::invalid_argument("ocean: bathymetry's space is not over the mesh given");
  }
  if (options.strip_rows == 0) throw std::invalid_argument("ocean: bathymetry needs strip_rows > 0");
  const GridInfo info = open_grid(comm, options.grid);
  if (info.lat.size() < 2 || info.lon.size() < 2) {
    throw std::invalid_argument("ocean: bathymetry needs a grid of at least 2 rows and 2 columns");
  }
  const std::size_t n = all_rows(space), owned = owned_rows(space);
  const double radius = space.geometry().radius;
  const auto x = space.x();
  const auto mass = space.mass();
  std::vector<Footprint> feet;
  feet.reserve(owned);
  for (std::size_t r = 0; r < owned; ++r) {
    const double p[3] = {x(r, 0), x(r, 1), x(r, 2)};
    feet.push_back(footprint(info, p, mass(r, 0), radius, options.bilinear_below_m));
  }
  Bathymetry b;
  std::string failure;
  try {
    b.peak_window = gather(options, info, feet);
  } catch (const std::exception& e) {
    failure = e.what();
  }
  const std::int32_t failed = failure.empty() ? 0 : 1;
  if (comm.allreduce(failed, soma::Op::max) != 0) {
    throw std::runtime_error(failure.empty() ? "ocean: bathymetry failed to read its grid on another rank"
                                             : failure);
  }
  b.H = Values("ocean::H", n, 1);
  b.roughness = Values("ocean::roughness", n, 1);
  for (std::size_t r = 0; r < owned; ++r) {
    const Footprint& f = feet[r];
    if (f.disc && f.weight > 0) {
      const double mean = f.sum / f.weight;
      b.H(r, 0) = mean;
      b.roughness(r, 0) = std::max(0.0, f.sum2 / f.weight - mean * mean);
    } else {
      b.H(r, 0) = f.bilinear;
    }
  }
  space.dss(comm, b.H);
  space.dss(comm, b.roughness);
  b.land.resize(n);
  for (std::size_t r = 0; r < n; ++r) b.land[r] = b.H(r, 0) <= 0 ? 1 : 0;
  return b;
}

double max_rx0(const soma::Comm& comm, const Space& space, const ConstValues& H) {
  if (H.extent(0) != all_rows(space) || H.extent(1) != 1) {
    throw std::invalid_argument("ocean: max_rx0 needs one value per row of the space");
  }
  std::vector<double> per_element;
  return comm.allreduce(element_rx0(space, H, per_element), soma::Op::max);
}

void smooth(const soma::Comm& comm, const Space& space, Bathymetry& b, const BathymetryOptions& options) {
  const std::size_t n = all_rows(space);
  if (b.H.extent(0) != n || b.H.extent(1) != 1) {
    throw std::invalid_argument("ocean: smooth needs H with one value per row of the space");
  }
  if (!(options.r_max > 0 && options.r_max < 1)) throw std::invalid_argument("ocean: smooth needs r_max in (0, 1)");
  if (!(options.H_min > 0)) throw std::invalid_argument("ocean: smooth needs H_min > 0");
  if (options.sweep_limit < 0) throw std::invalid_argument("ocean: smooth needs sweep_limit >= 0");
  const auto raise = [&] {
    for (std::size_t r = 0; r < n; ++r) b.H(r, 0) = std::max(b.H(r, 0), options.H_min);
  };
  raise();
  const int ppe = space.points_per_element();
  std::vector<double> per_element;
  double worst = comm.allreduce(element_rx0(space, b.H, per_element), soma::Op::max);
  b.sweeps = 0;
  if (worst > options.r_max && options.sweep_limit > 0) {
    const double kappa = 0.5 / largest_eigenvalue(comm, space);
    Values mask("ocean::smooth::mask", n, 1), lap("ocean::smooth::laplacian", n, 1);
    while (worst > options.r_max && b.sweeps < options.sweep_limit) {
      // 1 at every point of an element too steep; dss spreads it to the points its neighbours share.
      for (std::size_t r = 0; r < n; ++r) {
        const std::size_t e = r / static_cast<std::size_t>(ppe);
        mask(r, 0) = e < per_element.size() && per_element[e] > options.r_max ? 1.0 : 0.0;
      }
      space.dss(comm, mask);
      soma::sem::laplacian(comm, space, ConstValues(b.H), lap);
      for (std::size_t r = 0; r < n; ++r) {
        if (mask(r, 0) > 0) b.H(r, 0) += kappa * lap(r, 0);
      }
      ++b.sweeps;
      worst = comm.allreduce(element_rx0(space, b.H, per_element), soma::Op::max);
    }
  }
  raise();
  b.rx0 = max_rx0(comm, space, b.H);
  b.smoothed = b.rx0 <= options.r_max;
}

}  // namespace ocean
