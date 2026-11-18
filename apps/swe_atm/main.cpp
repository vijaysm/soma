// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// swe_atm: the shallow-water atmosphere on the cubed sphere. Runs a test case,
// reports conservation (and case 2's errors) as it goes, and at the end the
// errors against a reference file, the throughput and the state's digest.
// See docs/apps/swe-atm.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cases.hpp"
#include "diagnostics.hpp"
#include "latlon.hpp"
#include "model.hpp"
#include "output.hpp"
#include "reference.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/sem/operators.hpp"

namespace {

constexpr const char* usage =
    "usage: swe_atm [--case w2|w5|w6|galewsky|galewsky-steady] [--ne N] [--order P] [--days D]\n"
    "               [--alpha A] [--cfl C] [--dt S] [--nu NU] [--report-every HOURS]\n"
    "               [--vtk STEM] [--soma FILE] [--latlon NLAT,NLON,STEM]\n"
    "               [--reference FILE] [--ref-var NAME] [--ref-lat NAME] [--ref-lon NAME] [--ref-time I]\n";

struct LatLon {
  int nlat = 0, nlon = 0;
  std::string stem;
};

struct Options {
  swe::ModelOptions model;
  double days = 1;
  std::optional<double> report_hours;
  std::optional<std::string> vtk, soma;
  std::optional<LatLon> latlon;
  std::optional<swe::ReferenceSpec> reference;
  swe::ReferenceSpec ref_names;
};

double number(std::string_view option, const std::string& text) {
  std::size_t used = 0;
  double v = 0;
  try {
    v = std::stod(text, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != text.size() || !std::isfinite(v)) {
    throw std::invalid_argument("swe_atm: " + std::string(option) + " takes a number, not '" + text + "'");
  }
  return v;
}

int integer(std::string_view option, const std::string& text) {
  const double v = number(option, text);
  if (v != std::floor(v) || std::abs(v) > 1e9) {
    throw std::invalid_argument("swe_atm: " + std::string(option) + " takes an integer, not '" + text + "'");
  }
  return static_cast<int>(v);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help" || key == "-h") throw std::invalid_argument(usage);
    if (i + 1 >= argc) throw std::invalid_argument("swe_atm: " + std::string(key) + " needs a value\n" + usage);
    const std::string value = argv[++i];
    if (key == "--case") {
      o.model.test = swe::case_from_name(value);
    } else if (key == "--ne") {
      o.model.ne = integer(key, value);
    } else if (key == "--order") {
      o.model.order = integer(key, value);
    } else if (key == "--days") {
      o.days = number(key, value);
    } else if (key == "--alpha") {
      o.model.alpha = number(key, value);
    } else if (key == "--cfl") {
      o.model.cfl = number(key, value);
    } else if (key == "--dt") {
      o.model.dt = number(key, value);
    } else if (key == "--nu") {
      o.model.nu = number(key, value);
    } else if (key == "--report-every") {
      o.report_hours = number(key, value);
      if (!(*o.report_hours > 0)) throw std::invalid_argument("swe_atm: --report-every must be positive");
    } else if (key == "--vtk") {
      o.vtk = value;
    } else if (key == "--soma") {
      o.soma = value;
    } else if (key == "--latlon") {
      const auto a = value.find(','), b = value.find(',', a == std::string::npos ? a : a + 1);
      if (a == std::string::npos || b == std::string::npos || b + 1 >= value.size()) {
        throw std::invalid_argument("swe_atm: --latlon takes NLAT,NLON,STEM, not '" + value + "'");
      }
      LatLon l{integer(key, value.substr(0, a)), integer(key, value.substr(a + 1, b - a - 1)), value.substr(b + 1)};
      if (l.nlat < 2 || l.nlon < 1) throw std::invalid_argument("swe_atm: --latlon needs NLAT >= 2 and NLON >= 1");
      o.latlon = l;
    } else if (key == "--reference") {
      o.reference = swe::ReferenceSpec{};
      o.reference->file = value;
    } else if (key == "--ref-var") {
      o.ref_names.variable = value;
    } else if (key == "--ref-lat") {
      o.ref_names.lat = value;
    } else if (key == "--ref-lon") {
      o.ref_names.lon = value;
    } else if (key == "--ref-time") {
      o.ref_names.time_index = integer(key, value);
    } else {
      throw std::invalid_argument("swe_atm: unknown option " + std::string(key) + "\n" + usage);
    }
  }
  if (!(o.days >= 0)) throw std::invalid_argument("swe_atm: --days must not be negative");
  if (o.reference) {
    const std::filesystem::path file = o.reference->file;
    *o.reference = o.ref_names;
    o.reference->file = file;
  }
  return o;
}

// A copy of a state view.
swe::Values copy(const swe::ConstValues& v, const char* label) {
  swe::Values out(label, v.extent(0), v.extent(1));
  Kokkos::deep_copy(out, v);
  return out;
}

// The largest |u - u0| over owned points and ranks.
double max_difference(const soma::Comm& comm, const swe::SweModel& m, const swe::ConstValues& u0) {
  const auto u = m.u();
  const auto owned = static_cast<std::size_t>(m.space().owned_elements() * m.space().points_per_element());
  double worst = 0;
  for (std::size_t r = 0; r < owned; ++r) {
    worst = std::max(worst, std::hypot(u(r, 0) - u0(r, 0), u(r, 1) - u0(r, 1), u(r, 2) - u0(r, 2)));
  }
  return comm.allreduce(worst, soma::Op::max);
}

void write_latlon(const soma::Comm& comm, const swe::SweModel& m, const LatLon& l) {
  std::vector<double> lat, lon;
  for (int i = 0; i < l.nlat; ++i) lat.push_back(-90.0 + 180.0 * i / (l.nlat - 1));
  for (int j = 0; j < l.nlon; ++j) lon.push_back(360.0 * j / l.nlon);
  const auto& s = m.space();
  swe::Values zeta("swe_atm::vorticity", m.h().extent(0), 1);
  soma::sem::vorticity(s, m.u(), zeta);
  s.dss(comm, zeta);
  const std::vector<double> h = swe::sample(comm, s, m.mesh(), m.options().ne, m.h(), 0, lat, lon);
  const std::vector<double> z = swe::sample(comm, s, m.mesh(), m.options().ne, zeta, 0, lat, lon);
  swe::collectively(comm, [&] {
    if (comm.rank() != 0) return;
    const auto rows = static_cast<std::size_t>(l.nlat), cols = static_cast<std::size_t>(l.nlon);
    swe::write_npy(l.stem + "_h.npy", h, rows, cols);
    swe::write_npy(l.stem + "_vorticity.npy", z, rows, cols);
    swe::write_npy(l.stem + "_lat.npy", lat, rows, 1);
    swe::write_npy(l.stem + "_lon.npy", lon, cols, 1);
  });
}

int run(const soma::Comm& comm, const Options& o) {
  const bool root = comm.rank() == 0;
  swe::SweModel m(comm, o.model);
  // The steady cases (case 2, Galewsky's unperturbed jet) report their drift
  // from the initial state, case 2's exact solution.
  const bool steady = o.model.test == swe::Case::williamson2 || o.model.test == swe::Case::galewsky_steady;
  const swe::Values h0 = copy(m.h(), "swe_atm::h0"), u0 = copy(m.u(), "swe_atm::u0");
  const double mass0 = swe::mass(comm, m.space(), m.h());
  const double energy0 = swe::energy(comm, m.space(), m.h(), m.hs(), m.u());
  if (root) {
    std::printf("swe_atm: case %s, ne %d, p %d, %d rank(s); dt %.6g s, nu %.6g m^4/s\n",
                std::string(swe::case_name(o.model.test)).c_str(), o.model.ne, o.model.order, comm.size(), m.dt(),
                m.nu());
  }
  const auto report = [&] {
    const double mass = swe::mass(comm, m.space(), m.h());
    const double energy = swe::energy(comm, m.space(), m.h(), m.hs(), m.u());
    std::string errors;
    if (steady) {
      const swe::Norms n = swe::height_errors(comm, m.space(), m.h(), h0);
      const double du = max_difference(comm, m, u0);
      char text[160];
      std::snprintf(text, sizeof text, "  h l1 %.6e  l2 %.6e  linf %.6e  u max %.6e m/s", n.l1, n.l2, n.linf, du);
      errors = text;
    }
    if (root) {
      std::printf("t %10.4f d  step %8ld  mass %+.3e  energy %+.3e%s\n", m.time() / 86400, m.steps(),
                  (mass - mass0) / mass0, (energy - energy0) / energy0, errors.c_str());
    }
  };
  report();
  const double end = o.days * 86400;
  const auto start = std::chrono::steady_clock::now();
  const double every = o.report_hours ? *o.report_hours * 3600 : end;
  for (long k = 1; m.time() < end; ++k) {
    m.advance(comm, std::min(end, static_cast<double>(k) * every));
    report();
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (o.reference) {
    const swe::Reference ref = swe::read_reference(comm, *o.reference);
    const swe::Norms n = swe::compare(comm, m.space(), m.mesh(), o.model.ne, m.h(), ref);
    if (root) {
      std::printf("reference %s: l1 %.6e  l2 %.6e  linf %.6e\n", o.reference->file.string().c_str(), n.l1, n.l2,
                  n.linf);
    }
  }
  if (o.vtk) swe::write_vtk_subquads(comm, m, *o.vtk);
  if (o.soma) swe::write_soma_state(comm, m, *o.soma);
  if (o.latlon) write_latlon(comm, m, *o.latlon);
  const std::uint64_t digest = m.digest(comm);
  if (root) {
    std::printf("simulated %.6g days in %.3f s: %.4g days per hour\n", end / 86400, seconds,
                seconds > 0 ? end / 86400 / (seconds / 3600) : 0.0);
    std::printf("digest %016llx\n", static_cast<unsigned long long>(digest));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  try {
    return run(comm, parse(argc, argv));
  } catch (const std::exception& e) {
    // Every rank fails alike (options are the same everywhere; the model's
    // failures are collective): rank 0 says why.
    if (comm.rank() == 0) std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
