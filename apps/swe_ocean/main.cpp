// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// swe_ocean: the barotropic ocean. Runs an analytic case in the plane (stommel, kelvin) and reports its errors, or
// a global case on real bathymetry (rest, wind): builds the ocean mesh from a height file (or reads one saved
// before), reports what the build did, then steps, reporting energy, conservation and the Drake Passage transport
// as it goes, and at the end the throughput and the state's digest. See docs/apps/swe-ocean.md.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bathymetry.hpp"
#include "cases.hpp"
#include "constants.hpp"
#include "model.hpp"
#include "ocean_mesh.hpp"
#include "output.hpp"
#include "sample.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "vertical.hpp"

namespace {

constexpr const char* usage =
    "usage: swe_ocean --case stommel|kelvin|rest|wind [--ne N] [--order P] [--coast-km KM] [--transition N]\n"
    "                 [--bathymetry FILE] [--bathy-var NAME] [--bathy-lat NAME] [--bathy-lon NAME]\n"
    "                 [--mesh FILE] [--save-mesh FILE] [--vertical s|zstar] [--levels K]\n"
    "                 [--theta-s S] [--theta-b B] [--hc M] [--days D] [--dt S] [--nu V] [--tau0 T] [--linear]\n"
    "                 [--report-hours H] [--output PREFIX] [--output-reports] [--latlon NLAT,NLON]\n";

// A bad command line: usage, exit 2.
struct BadOption : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

enum class Case { stommel, kelvin, rest, wind };

struct Options {
  Case test = Case::wind;
  std::optional<int> ne, order;
  ocean::MeshOptions mesh;
  std::optional<std::filesystem::path> bathymetry, mesh_file, save_mesh;
  ocean::ModelOptions model;
  double days = 1, tau0 = 0.1;
  std::optional<double> report_hours;
  std::optional<std::string> output;
  bool output_reports = false;
  std::optional<std::array<int, 2>> latlon;
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
    throw BadOption("swe_ocean: " + std::string(option) + " takes a number, not '" + text + "'\n" + usage);
  }
  return v;
}

int integer(std::string_view option, const std::string& text) {
  const double v = number(option, text);
  if (v != std::floor(v) || std::abs(v) > 1e9) {
    throw BadOption("swe_ocean: " + std::string(option) + " takes an integer, not '" + text + "'\n" + usage);
  }
  return static_cast<int>(v);
}

Case case_from_name(const std::string& name) {
  if (name == "stommel") return Case::stommel;
  if (name == "kelvin") return Case::kelvin;
  if (name == "rest") return Case::rest;
  if (name == "wind") return Case::wind;
  throw BadOption("swe_ocean: unknown case '" + name + "' (stommel, kelvin, rest, wind)\n" + usage);
}

const char* case_name(Case c) {
  switch (c) {
    case Case::stommel:
      return "stommel";
    case Case::kelvin:
      return "kelvin";
    case Case::rest:
      return "rest";
    case Case::wind:
      return "wind";
  }
  return "?";
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help" || key == "-h") throw BadOption(usage);
    if (key == "--linear") {
      o.model.linear = true;
      continue;
    }
    if (key == "--output-reports") {
      o.output_reports = true;
      continue;
    }
    if (i + 1 >= argc) throw BadOption("swe_ocean: " + std::string(key) + " needs a value\n" + usage);
    const std::string value = argv[++i];
    if (key == "--case") {
      o.test = case_from_name(value);
    } else if (key == "--ne") {
      o.ne = integer(key, value);
    } else if (key == "--order") {
      o.order = integer(key, value);
    } else if (key == "--coast-km") {
      o.mesh.coast_km = number(key, value);
    } else if (key == "--transition") {
      o.mesh.transition = integer(key, value);
    } else if (key == "--bathymetry") {
      o.bathymetry = value;
    } else if (key == "--bathy-var") {
      o.mesh.bathymetry.grid.variable = value;
    } else if (key == "--bathy-lat") {
      o.mesh.bathymetry.grid.lat = value;
    } else if (key == "--bathy-lon") {
      o.mesh.bathymetry.grid.lon = value;
    } else if (key == "--mesh") {
      o.mesh_file = value;
    } else if (key == "--save-mesh") {
      o.save_mesh = value;
    } else if (key == "--vertical") {
      if (value == "s") {
        o.model.vertical.kind = ocean::Vertical::s;
      } else if (value == "zstar") {
        o.model.vertical.kind = ocean::Vertical::zstar;
      } else {
        throw BadOption("swe_ocean: --vertical takes s or zstar, not '" + value + "'\n" + usage);
      }
    } else if (key == "--levels") {
      o.model.vertical.levels = integer(key, value);
      if (o.model.vertical.levels < 1) {
        throw BadOption("swe_ocean: --levels must be at least 1\n" + std::string(usage));
      }
    } else if (key == "--theta-s") {
      o.model.vertical.theta_s = number(key, value);
    } else if (key == "--theta-b") {
      o.model.vertical.theta_b = number(key, value);
    } else if (key == "--hc") {
      o.model.vertical.hc = number(key, value);
    } else if (key == "--days") {
      o.days = number(key, value);
    } else if (key == "--dt") {
      o.model.dt = number(key, value);
    } else if (key == "--nu") {
      o.model.nu = number(key, value);
    } else if (key == "--tau0") {
      o.tau0 = number(key, value);
    } else if (key == "--report-hours") {
      o.report_hours = number(key, value);
      if (!(*o.report_hours > 0)) throw BadOption("swe_ocean: --report-hours must be positive\n" + std::string(usage));
    } else if (key == "--output") {
      o.output = value;
    } else if (key == "--latlon") {
      const auto comma = value.find(',');
      if (comma == std::string::npos) {
        throw BadOption("swe_ocean: --latlon takes NLAT,NLON, not '" + value + "'\n" + usage);
      }
      const std::array<int, 2> n{integer(key, value.substr(0, comma)), integer(key, value.substr(comma + 1))};
      if (n[0] < 2 || n[1] < 1) {
        throw BadOption("swe_ocean: --latlon needs NLAT >= 2 and NLON >= 1\n" + std::string(usage));
      }
      o.latlon = n;
    } else {
      throw BadOption("swe_ocean: unknown option " + std::string(key) + "\n" + usage);
    }
  }
  if (!(o.days >= 0)) throw BadOption("swe_ocean: --days must not be negative\n" + std::string(usage));
  const bool global = o.test == Case::rest || o.test == Case::wind;
  if (global && !o.bathymetry && !o.mesh_file) {
    throw BadOption(std::string("swe_ocean: --case ") + case_name(o.test) +
                    " needs --bathymetry FILE or --mesh FILE\n" + usage);
  }
  if (o.bathymetry && o.mesh_file) {
    throw BadOption("swe_ocean: give --bathymetry or --mesh, not both\n" + std::string(usage));
  }
#ifndef SWE_OCEAN_HAVE_HDF5
  for (const auto& [given, name] : {std::pair{o.bathymetry.has_value(), "--bathymetry"},
                                    std::pair{o.output.has_value(), "--output"},
                                    std::pair{o.save_mesh.has_value(), "--save-mesh"},
                                    std::pair{o.mesh_file.has_value(), "--mesh"}}) {
    if (given) {
      throw BadOption(std::string("swe_ocean: ") + name +
                      " needs HDF5; this build has none (configure with SOMA_ENABLE_HDF5)\n");
    }
  }
#endif
  if ((o.latlon || o.output_reports) && !o.output) {
    throw BadOption("swe_ocean: --latlon and --output-reports need --output PREFIX\n" + std::string(usage));
  }
  return o;
}

// The analytic cases: errors, steps and digest; output at the end through the case's hook.
int run_analytic(const soma::Comm& comm, const Options& o) {
  const bool root = comm.rank() == 0;
  const ocean::CaseFinish finish = [&](ocean::OceanModel& m) {
    if (o.output) ocean::write_snapshot(comm, m, *o.output + ".soma");
  };
  const auto start = std::chrono::steady_clock::now();
  ocean::CaseResult r;
  int nx = 0, order = 0;
  if (o.test == Case::stommel) {
    nx = o.ne.value_or(8);
    order = o.order.value_or(4);
    r = ocean::run_stommel(comm, nx, order, ocean::StommelSpec{}, finish);
  } else {
    nx = o.ne.value_or(40);
    order = o.order.value_or(4);
    r = ocean::run_kelvin(comm, nx, order, ocean::KelvinSpec{}, finish);
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (root) {
    std::printf("swe_ocean: case %s, nx %d, p %d, %d rank(s)\n", case_name(o.test), nx, order, comm.size());
    std::printf("relative error l2 %.6e  max %.6e  (%s), %ld steps in %.3f s\n", r.error_l2, r.error_max,
                o.test == Case::stommel ? "velocity at steady state" : "eta after 2000 km", r.steps, seconds);
    std::printf("digest %016llx\n", static_cast<unsigned long long>(r.digest));
  }
  return 0;
}

// The largest Haney number rx1 between neighbouring GLL points of the owned elements (along i and along j) of the
// columns' reference interfaces in the coordinate spec. Collective.
double max_rx1(const soma::Comm& comm, const ocean::Space& space, const ocean::ConstValues& H,
               const ocean::VerticalSpec& spec) {
  const int np = space.order() + 1, ppe = space.points_per_element();
  const auto K = static_cast<std::size_t>(spec.levels);
  const auto rows = static_cast<std::size_t>(space.owned_elements() * ppe);
  std::vector<double> z(rows * (K + 1)), h0(K);
  for (std::size_t r = 0; r < rows; ++r) {
    ocean::reference_thicknesses(spec, H(r, 0), h0);
    z[r * (K + 1)] = 0;
    for (std::size_t k = 0; k < K; ++k) z[r * (K + 1) + k + 1] = z[r * (K + 1) + k] - h0[k];
  }
  const auto column = [&](std::size_t r) { return std::span<const double>(z.data() + r * (K + 1), K + 1); };
  double worst = 0;
  for (soma::LO e = 0; e < space.owned_elements(); ++e) {
    const auto base = static_cast<std::size_t>(e * ppe);
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        const std::size_t r = base + static_cast<std::size_t>(i + np * j);
        if (i + 1 < np) worst = std::max(worst, ocean::rx1(column(r), column(r + 1)));
        if (j + 1 < np) worst = std::max(worst, ocean::rx1(column(r), column(r + static_cast<std::size_t>(np))));
      }
    }
  }
  return comm.allreduce(worst, soma::Op::max);
}

// The kinetic energy rho0 / 2 times the integral of D |u|^2 (J). Collective.
double kinetic_energy(const soma::Comm& comm, const ocean::OceanModel& m) {
  const auto eta = m.eta(), H = m.H(), u = m.u();
  ocean::Values e("swe_ocean::energy", eta.extent(0), 1);
  for (std::size_t r = 0; r < eta.extent(0); ++r) {
    e(r, 0) = 0.5 * ocean::rho0 * (H(r, 0) + eta(r, 0)) * (u(r, 0) * u(r, 0) + u(r, 1) * u(r, 1) + u(r, 2) * u(r, 2));
  }
  return m.space().integral(comm, e);
}

// The largest speed over owned points and ranks (m/s).
double max_speed(const soma::Comm& comm, const ocean::OceanModel& m) {
  const auto u = m.u();
  const auto owned = static_cast<std::size_t>(m.space().owned_elements() * m.space().points_per_element());
  double worst = 0;
  for (std::size_t r = 0; r < owned; ++r) worst = std::max(worst, std::hypot(u(r, 0), u(r, 1), u(r, 2)));
  return comm.allreduce(worst, soma::Op::max);
}

// The wind case's stress: tau_x = tau0 (-cos(3 phi)) eastward, phi the latitude: easterlies of tau0 at the equator,
// none at 30 degrees, westerlies of tau0 at 60 degrees, none at the poles.
std::function<void(const ocean::Space&, ocean::Values)> zonal_wind(double tau0) {
  return [tau0](const ocean::Space& s, ocean::Values tau) {
    const auto x = s.x();
    for (std::size_t r = 0; r < tau.extent(0); ++r) {
      const double rho = std::hypot(x(r, 0), x(r, 1)), phi = std::atan2(x(r, 2), rho);
      const double east = rho > 0 ? -tau0 * std::cos(3 * phi) / rho : 0.0;
      tau(r, 0) = -x(r, 1) * east;
      tau(r, 1) = x(r, 0) * east;
      tau(r, 2) = 0;
    }
  };
}

int run_global(const soma::Comm& comm, Options o) {
  const bool root = comm.rank() == 0;
  const auto build_start = std::chrono::steady_clock::now();
  o.mesh.ne = o.ne.value_or(30);
  o.mesh.order = o.order.value_or(3);
  ocean::OceanMesh mesh;
  if (o.mesh_file) {
    mesh = ocean::read_ocean_mesh(comm, *o.mesh_file, o.mesh.order);
  } else {
    o.mesh.bathymetry.grid.file = *o.bathymetry;
    mesh = ocean::build_ocean_mesh(comm, o.mesh);
  }
  if (o.save_mesh) ocean::write_ocean_mesh(comm, mesh, *o.save_mesh);
  const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();

  // What the build did, before the model takes the mesh.
  const auto cells = comm.allreduce(static_cast<std::int64_t>(mesh.space->owned_elements()), soma::Op::sum);
  const auto peak = comm.allreduce(static_cast<std::uint64_t>(mesh.bathymetry.peak_window), soma::Op::max);
  const double rx0 = ocean::max_rx0(comm, *mesh.space, mesh.bathymetry.H);
  ocean::VerticalSpec s_spec = o.model.vertical, z_spec = o.model.vertical;
  s_spec.kind = ocean::Vertical::s;
  z_spec.kind = ocean::Vertical::zstar;
  const double rx1_s = max_rx1(comm, *mesh.space, mesh.bathymetry.H, s_spec);
  const double rx1_z = max_rx1(comm, *mesh.space, mesh.bathymetry.H, z_spec);
  const char* smoothing = o.mesh_file                ? "as saved"
                          : mesh.bathymetry.smoothed ? "smoothed"
                                                     : "NOT smoothed to r_max";
  if (root) {
    const std::string source =
        o.mesh_file ? "read from " + o.mesh_file->string() : "built at ne " + std::to_string(o.mesh.ne);
    std::printf("swe_ocean: case %s, p %d, %d rank(s); mesh %s in %.1f s\n", case_name(o.test), o.mesh.order,
                comm.size(), source.c_str(), build_seconds);
    std::printf("cells %lld; removed %.6g km^2 of land, %.6g km^2 not joined to the world ocean\n",
                static_cast<long long>(cells), mesh.removed_land_km2, mesh.removed_disconnected_km2);
    // A read mesh was not built here: no sweeps, windows or coastline to report.
    const bool built = !o.mesh_file;
    const std::string sweeps = built ? std::to_string(mesh.bathymetry.sweeps) + " smoothing sweeps (" +
                                           std::to_string(mesh.bathymetry.clip_sweeps) + " clipping)"
                                     : "n/a smoothing sweeps";
    const std::string window = built ? std::to_string(peak) + " values" : "n/a";
    const std::string coastline = built ? std::to_string(mesh.coastline_points) : "n/a";
    std::printf("bathymetry: rx0 %.4f (r_max %.2f, %s), %s, peak window %s, %s coastline points\n", rx0,
                o.mesh.bathymetry.r_max, smoothing, sweeps.c_str(), window.c_str(), coastline.c_str());
    std::printf("rx1 of %d levels: s %.4f, zstar %.4f\n", o.model.vertical.levels, rx1_s, rx1_z);
  }

  // The land flags of kept elements, for the transport (the model keeps the space's rows).
  ocean::Values land("swe_ocean::land", mesh.bathymetry.land.size(), 1);
  for (std::size_t r = 0; r < mesh.bathymetry.land.size(); ++r) land(r, 0) = mesh.bathymetry.land[r];
  if (o.test == Case::wind) o.model.forcing.wind = zonal_wind(o.tau0);
  ocean::OceanModel m(comm, std::move(mesh), o.model);
  // Tracers: a temperature warm at the surface of the tropics, a uniform salinity.
  m.tracers().set(comm, [](const std::array<double, 3>& x, double z, int t) {
    if (t != 0) return 35.0;
    const double r = std::hypot(x[0], x[1], x[2]), c = std::hypot(x[0], x[1]) / r;
    return 2 + 25 * c * c * std::exp(z / 500);
  });
  const ocean::Sampler sampler(m.space(), m.mesh());
  const ocean::Values ones("swe_ocean::ones", m.H().extent(0), 1);
  Kokkos::deep_copy(ones, 1.0);
  const double area = m.space().integral(comm, ones);
  const double volume = m.space().integral(comm, m.H());
  const double mass0 = m.mass(comm);
  std::vector<double> tracer0;
  for (int t = 0; t < m.tracers().count(); ++t) tracer0.push_back(m.tracers().total(comm, t));
  const auto coast = comm.allreduce(static_cast<std::int64_t>(m.boundary().coast_points), soma::Op::sum);
  const auto no_slip = comm.allreduce(static_cast<std::int64_t>(m.boundary().no_slip_points), soma::Op::sum);
  if (root) {
    std::printf("area %.6e m^2, volume %.6e m^3; dt %.6g s, nu %.6g m^4/s, %d levels (%s), %d tracers\n", area,
                volume, m.dt(), m.nu(), m.tracers().levels(),
                o.model.vertical.kind == ocean::Vertical::s ? "s" : "zstar", m.tracers().count());
    std::printf("coast points %lld, no-slip points %lld\n", static_cast<long long>(coast),
                static_cast<long long>(no_slip));
  }
  const auto report = [&] {
    const double energy = kinetic_energy(comm, m), speed = max_speed(comm, m);
    const double mass = m.mass(comm);
    std::string drift;
    for (int t = 0; t < m.tracers().count(); ++t) {
      const double total = m.tracers().total(comm, t), base = tracer0[static_cast<std::size_t>(t)];
      char text[48];
      std::snprintf(text, sizeof text, " %+.2e", base != 0 ? (total - base) / std::abs(base) : total);
      drift += text;
    }
    const ocean::Transport drake = ocean::meridian_transport(comm, sampler, m.space(), m.H(), m.eta(), m.u(), land);
    if (root) {
      std::printf("t %9.4f d  step %8ld  KE %.4e J  max|u| %.3e m/s  mass %+.2e  tracers%s  Drake %+.4f Sv",
                  m.time() / 86400, m.steps(), energy, speed, (mass - mass0) / volume, drift.c_str(), drake.sv);
      if (drake.wet) {
        std::printf(" (%.2f..%.2f)\n", drake.lat_from, drake.lat_to);
      } else {
        std::printf(" (closed)\n");
      }
    }
  };
  report();
  long reported = m.steps();
  const double end = o.days * 86400;
  const double every = o.report_hours ? *o.report_hours * 3600 : end;
  const auto start = std::chrono::steady_clock::now();
  // Reports on the way land on whole steps (at or just before each report time), so they leave the run's bits
  // alone; the last lands on the end.
  for (long k = 1; m.time() < end; ++k) {
    const double t = static_cast<double>(k) * every;
    if (t < end) {
      m.advance_whole_steps(comm, t, end);
    } else {
      m.advance(comm, end);
    }
    // Reports asked for more often than the step would repeat the last one: one per step.
    if (m.steps() == reported) continue;
    reported = m.steps();
    report();
    if (o.output_reports && m.time() < end) {
      ocean::write_snapshot(comm, m, *o.output + ".step" + std::to_string(m.steps()) + ".soma");
    }
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (o.output) {
    ocean::write_snapshot(comm, m, *o.output + ".soma");
    if (o.latlon) ocean::write_latlon(comm, m, sampler, (*o.latlon)[0], (*o.latlon)[1], *o.output);
  }
  const std::uint64_t digest = m.digest(comm);
  if (root) {
    std::printf("simulated %.6g days in %d steps, %.3f s: %.4g days per hour\n", end / 86400,
                static_cast<int>(m.steps()), seconds, seconds > 0 ? end / 86400 / (seconds / 3600) : 0.0);
    std::printf("digest %016llx\n", static_cast<unsigned long long>(digest));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  try {
    const Options o = parse(argc, argv);
    return o.test == Case::stommel || o.test == Case::kelvin ? run_analytic(comm, o) : run_global(comm, o);
  } catch (const BadOption& e) {
    // Options are the same on every rank: rank 0 says why.
    if (comm.rank() == 0) std::fprintf(stderr, "%s", e.what());
    return 2;
  } catch (const std::exception& e) {
    // The model's failures are collective: every rank fails alike, rank 0 says why.
    if (comm.rank() == 0) std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
