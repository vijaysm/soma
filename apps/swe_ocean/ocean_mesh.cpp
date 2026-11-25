// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "ocean_mesh.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "coastline.hpp"
#include "constants.hpp"
#include "soma/algorithms/components.hpp"
#include "soma/algorithms/quality_smooth.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/exact_sum.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace ocean {

namespace {

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Real;
using P3 = std::array<double, 3>;

constexpr double deg = std::numbers::pi / 180;

// The names the file gives the bathymetry and the removed areas.
constexpr const char* H_field = "ocean:H";
constexpr const char* roughness_field = "ocean:roughness";
constexpr const char* land_field = "ocean:land";
constexpr const char* land_area_value = "ocean:removed_land_km2";
constexpr const char* disconnected_area_value = "ocean:removed_disconnected_km2";

double dot(const P3& u, const P3& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }
P3 cross(const P3& u, const P3& v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
}
P3 unit(const P3& u) {
  const double n = std::sqrt(dot(u, u));
  return {u[0] / n, u[1] / n, u[2] / n};
}

// A ghost layer through vertices and distributed edges, as the space and edge_reachable need on more than one
// rank (on one they derive the edges themselves).
void prepare(const soma::Comm& comm, HostMesh& mesh) {
  if (comm.size() == 1) return;
  mesh.add_ghosts(comm, soma::GhostOptions{1, 0});
  soma::build_distributed_adjacencies(comm, mesh, soma::AdjacencyRequest{}.add(2, 1));
}

std::unique_ptr<Space> make_space(const soma::Comm& comm, HostMesh& mesh, int order) {
  return std::make_unique<Space>(comm, mesh, order, soma::sem::SemGeometry::sphere(a));
}

// quality_smooth on the unit sphere, every vertex free (the skin of a closed surface is empty), with the ghost
// layers it needs added and removed.
void optimize(const soma::Comm& comm, HostMesh& mesh, const MeshOptions& o) {
  if (o.optimize[0] + o.optimize[1] + o.optimize[2] == 0) return;
  const bool bounded = o.area_ratio >= 1;
  if (comm.size() > 1) mesh.add_ghosts(comm, soma::GhostOptions{bounded ? 2 : 1, 0});
  soma::QualitySmoothOptions q;
  q.shape_sweeps = o.optimize[0];
  q.angle_sweeps = o.optimize[1];
  q.polish_sweeps = o.optimize[2];
  q.area_ratio_bound = o.area_ratio;
  q.sphere_radius = 1;
  (void)soma::quality_smooth(comm, mesh, q);
  if (comm.size() > 1) mesh.remove_ghosts(comm);
}

// Whether owned element e (corners counter-clockwise seen from outside) holds the unit vector p: p on the inner
// side of, or on, each of its great-circle edges, and in its hemisphere.
bool contains(const HostMesh& mesh, LO e, const P3& p) {
  const auto view = mesh.view();
  const auto x = mesh.coords();
  const LO n = view.num_vertices(2, e);
  std::vector<P3> c(static_cast<std::size_t>(n));
  P3 mid{0, 0, 0};
  for (LO q = 0; q < n; ++q) {
    const LO v = view.vertex(2, e, q);
    c[static_cast<std::size_t>(q)] = unit({x(v, 0), x(v, 1), x(v, 2)});
    for (std::size_t k = 0; k < 3; ++k) mid[k] += c[static_cast<std::size_t>(q)][k];
  }
  if (dot(mid, p) <= 0) return false;
  for (LO q = 0; q < n; ++q) {
    const P3& u = c[static_cast<std::size_t>(q)];
    const P3& w = c[static_cast<std::size_t>((q + 1) % n)];
    if (dot(cross(u, w), p) < 0) return false;
  }
  return true;
}

// The gid of the element holding the seed (the least on a tie), the same on every rank; throws on every rank
// when there is none or it has no ocean point.
GO seed_element(const soma::Comm& comm, const HostMesh& mesh, const MeshOptions& o,
                const std::vector<std::uint8_t>& ocean_element) {
  const P3 p{std::cos(o.seed_lat * deg) * std::cos(o.seed_lon * deg),
             std::cos(o.seed_lat * deg) * std::sin(o.seed_lon * deg), std::sin(o.seed_lat * deg)};
  const auto gid = mesh.gid(2);
  constexpr auto none = std::numeric_limits<std::int64_t>::max();
  std::int64_t found = none;
  for (LO e = 0; e < static_cast<LO>(ocean_element.size()); ++e) {
    if (gid(e) < found && contains(mesh, e, p)) found = gid(e);
  }
  found = comm.allreduce(found, soma::Op::min);
  const std::string where = "(" + std::to_string(o.seed_lat) + ", " + std::to_string(o.seed_lon) + ")";
  if (found == none) throw std::runtime_error("build_ocean_mesh: no element contains the seed " + where);
  std::int32_t land = 0;
  for (LO e = 0; e < static_cast<LO>(ocean_element.size()); ++e) {
    if (gid(e) == found && ocean_element[static_cast<std::size_t>(e)] == 0) land = 1;
  }
  if (comm.allreduce(land, soma::Op::max) != 0) {
    throw std::runtime_error("build_ocean_mesh: the seed " + where + " lies in a land element (gid " +
                             std::to_string(found) + ", no point with H > 0)");
  }
  return static_cast<GO>(found);
}

// The owned elements of `mesh` that `keep` flags (quads, one flag per owned element), as a new mesh built
// from their gids, corners and coordinates (build_local_part resolves the sharing by gid), with their
// refine:level and the first pass's land flags (`land`, points_per_element per owned element) as tags of the
// elements: the tags travel with the elements, by gid, through the rebalance and the ghost layers. A MeshEdit
// of a distributed mesh leaves the sharing of the vertices and edges along the cut as it was (only the rank
// that removes an entity knows it is gone), so the part is built again instead.
HostMesh kept_part(const soma::Comm& comm, const HostMesh& mesh, const std::vector<std::uint8_t>& keep,
                   const std::vector<std::uint8_t>& land, LO ppe) {
  const auto view = mesh.view();
  const auto x = mesh.coords();
  const auto egid = mesh.gid(2);
  const auto vgid = mesh.gid(0);
  const bool levelled = mesh.fields().contains("refine:level");
  const std::vector<std::int32_t> level = soma::refine_levels(mesh);
  std::vector<GO> elements, corners;
  std::map<GO, std::array<Real, 3>> xyz;
  std::map<GO, std::pair<std::int32_t, LO>> kept;  // gid: level, lid in `mesh`
  for (LO e = 0; e < static_cast<LO>(keep.size()); ++e) {
    if (keep[static_cast<std::size_t>(e)] == 0) continue;
    if (view.num_vertices(2, e) != 4) throw std::logic_error("build_ocean_mesh: an element is not a quad");
    elements.push_back(egid(e));
    kept[egid(e)] = {level[static_cast<std::size_t>(e)], e};
    for (LO q = 0; q < 4; ++q) {
      const LO v = view.vertex(2, e, q);
      corners.push_back(vgid(v));
      xyz[vgid(v)] = {x(v, 0), x(v, 1), x(v, 2)};
    }
  }
  std::vector<GO> vertices;
  std::vector<Real> coords;
  for (const auto& [g, p] : xyz) {
    vertices.push_back(g);
    coords.insert(coords.end(), p.begin(), p.end());
  }
  soma::MeshBuilder b(2);
  b.add_vertices(vertices, coords);
  b.add_elements(soma::Topology::quad, elements, corners);
  HostMesh part = soma::build_local_part(comm, std::move(b), mesh.ordering());
  const auto gid = part.gid(2);
  const auto flags = part.fields().add_tag<std::uint8_t>(land_field, 2, static_cast<int>(ppe)).view();
  for (LO e = 0; e < part.count(2); ++e) {
    const LO from = kept.at(gid(e)).second;
    for (LO k = 0; k < ppe; ++k) flags(e, k) = land[static_cast<std::size_t>(from * ppe + k)];
  }
  if (levelled) {
    const auto levels = part.fields().add_tag<std::int32_t>("refine:level", 2, 1).view();
    for (LO e = 0; e < part.count(2); ++e) levels(e, 0) = kept.at(gid(e)).first;
  }
  return part;
}

void set_value(HostMesh& mesh, const char* name, double value) {
  soma::MeshValue v;
  v.type = soma::DataType::f64;
  v.ncomp = 1;
  v.global_value = soma::MeshValue::bytes_of(std::span<const double>(&value, 1));
  mesh.values().set(name, std::move(v));
}

double get_value(const HostMesh& mesh, const char* name) {
  if (!mesh.values().contains(name)) return 0;
  const auto v = mesh.values().at(name).global_as<double>();
  return v && v->size() == 1 ? (*v)[0] : 0;
}

}  // namespace

OceanMesh build_ocean_mesh(const soma::Comm& comm, const MeshOptions& o) {
  if (o.ne < 1 || o.order < 1 || !(o.seed_lat >= -90 && o.seed_lat <= 90) || !std::isfinite(o.seed_lon) ||
      (o.area_ratio != 0 && !(o.area_ratio >= 1))) {
    throw std::invalid_argument("build_ocean_mesh: needs ne >= 1, order >= 1, a seed latitude in [-90, 90], a "
                                "finite seed longitude and an area ratio bound of 0 or at least 1");
  }
  OceanMesh out;
  soma::sem::CubedSphereSpec spec;
  spec.ne = o.ne;
  spec.radius = 1;
  HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);

  // Refined toward the coast and smoothed, on the unit sphere.
  if (o.coast_km > 0) {
    soma::GradedOptions g;
    g.np = o.order + 1;
    g.transition = o.transition;
    g.reruns = o.reruns;
    g.spacing = o.coast_km;
    g.length = [](Real l) { return l * a / 1000; };
    std::vector<std::array<double, 3>> coast = coastline(comm, o.bathymetry.grid, o.coast_km, o.bathymetry.strip_rows);
    out.coastline_points = coast.size();
    out.refined = soma::refine_graded(comm, mesh, g, coast_target(std::move(coast)));
    mesh = std::move(out.refined.mesh);
    out.refined.mesh = HostMesh{};
  }
  optimize(comm, mesh, o);

  // On the Earth: the first space and bathymetry.
  {
    const auto x = mesh.coords();
    for (LO v = 0; v < mesh.count(0); ++v) {
      for (int k = 0; k < 3; ++k) x(v, k) *= a;
    }
  }
  mesh = soma::rebalance(comm, mesh);
  prepare(comm, mesh);
  {
    const std::unique_ptr<Space> space = make_space(comm, mesh, o.order);
    const Bathymetry first = average(comm, *space, mesh, o.bathymetry);
    const LO owned = space->owned_elements();
    const LO ppe = space->points_per_element();
    const auto mass = space->local_mass();
    // Per owned element: an ocean point or not, and its area (km^2, its points' masses in order).
    std::vector<std::uint8_t> ocean_element(static_cast<std::size_t>(owned), 0);
    std::vector<double> area(static_cast<std::size_t>(owned), 0);
    for (LO e = 0; e < owned; ++e) {
      for (LO k = 0; k < ppe; ++k) {
        const auto r = static_cast<std::size_t>(e * ppe + k);
        if (first.land[r] == 0) ocean_element[static_cast<std::size_t>(e)] = 1;
        area[static_cast<std::size_t>(e)] += mass(r, 0);
      }
      area[static_cast<std::size_t>(e)] /= 1e6;
    }
    const GO seed_gid = seed_element(comm, mesh, o, ocean_element);
    std::vector<std::uint8_t> seed(static_cast<std::size_t>(owned), 0);
    const auto gid = mesh.gid(2);
    for (LO e = 0; e < owned; ++e) seed[static_cast<std::size_t>(e)] = gid(e) == seed_gid ? 1 : 0;
    const std::vector<std::uint8_t> world = soma::edge_reachable(comm, mesh, ocean_element, seed);
    soma::ExactSums<2> removed;
    for (LO e = 0; e < owned; ++e) {
      const auto i = static_cast<std::size_t>(e);
      if (world[i] == 0) removed.add(ocean_element[i] == 0 ? 0 : 1, area[i]);
    }
    const auto [land_km2, disconnected_km2] = soma::allreduce_exact(comm, removed);
    out.removed_land_km2 = land_km2;
    out.removed_disconnected_km2 = disconnected_km2;
    HostMesh part = kept_part(comm, mesh, world, first.land, ppe);
    mesh = soma::rebalance(comm, part);
  }

  // The ocean's space and bathymetry.
  out.mesh = std::make_unique<HostMesh>(std::move(mesh));
  HostMesh& m = *out.mesh;
  prepare(comm, m);
  out.space = make_space(comm, m, o.order);
  out.bathymetry = average(comm, *out.space, m, o.bathymetry);
  {
    const auto flags = m.fields().get<std::uint8_t>(land_field, 2);
    const auto f = flags.view();
    const LO ppe = out.space->points_per_element();
    Bathymetry& b = out.bathymetry;
    for (LO e = 0; e < out.space->num_elements(); ++e) {
      for (LO k = 0; k < ppe; ++k) {
        const auto r = static_cast<std::size_t>(e * ppe + k);
        b.land[r] = f(e, k);
        if (b.land[r] != 0) b.H(r, 0) = o.bathymetry.H_min;
      }
    }
    m.fields().remove(flags.id());
  }
  smooth(comm, *out.space, out.bathymetry, o.bathymetry);
  return out;
}

void write_ocean_mesh(const soma::Comm& comm, const OceanMesh& ocean, const std::filesystem::path& path,
                      int aggregators) {
  HostMesh& mesh = *ocean.mesh;
  const auto& space = ocean.space->space();
  const Bathymetry& b = ocean.bathymetry;
  const std::pair<const char*, ConstValues> values[] = {{H_field, b.H}, {roughness_field, b.roughness}};
  for (const auto& [name, v] : values) {
    const auto field = mesh.fields().contains(name) ? mesh.fields().get<double>(name, space)
                                                    : mesh.fields().add<double>(name, space, 1);
    Kokkos::deep_copy(field.view(), v);
  }
  const auto land = mesh.fields().contains(land_field) ? mesh.fields().get<std::uint8_t>(land_field, space)
                                                       : mesh.fields().add<std::uint8_t>(land_field, space, 1);
  const auto l = land.view();
  for (std::size_t r = 0; r < b.land.size(); ++r) l(r, 0) = b.land[r];
  set_value(mesh, land_area_value, ocean.removed_land_km2);
  set_value(mesh, disconnected_area_value, ocean.removed_disconnected_km2);
  soma::SomaWriteOptions w;
  w.aggregators = aggregators;
  soma::write_soma(comm, mesh, path, w);
}

OceanMesh read_ocean_mesh(const soma::Comm& comm, const std::filesystem::path& path, int order, int aggregators) {
  if (order < 1) throw std::invalid_argument("read_ocean_mesh: needs order >= 1");
  OceanMesh out;
  soma::SomaReadOptions r;
  r.aggregators = aggregators;
  out.mesh = std::make_unique<HostMesh>(soma::read_soma(comm, path, r));
  HostMesh& m = *out.mesh;
  prepare(comm, m);
  out.space = make_space(comm, m, order);
  const LO ppe = out.space->points_per_element();
  const LO elements = out.space->num_elements();
  const auto rows = static_cast<std::size_t>(elements * ppe);
  Bathymetry& b = out.bathymetry;
  b.H = Values("H", rows, 1);
  b.roughness = Values("roughness", rows, 1);
  b.land.assign(rows, 0);
  // Each element's points from the file's fields; a missing or misshapen field is an error on every rank.
  std::int32_t bad = 0;
  const auto read = [&]<class T>(const char* name, auto&& put) {
    if (!m.fields().contains(name)) {
      bad = 1;
      return;
    }
    const auto field = m.fields().get<T>(name);
    const auto v = field.view();
    for (LO e = 0; e < elements; ++e) {
      const soma::RowRange rr = field.space().entity_rows(2, e);
      if (!rr.valid() || rr.size() != ppe) {
        bad = 1;
        return;
      }
      for (LO k = 0; k < ppe; ++k) put(static_cast<std::size_t>(e * ppe + k), v(rr.begin + k, 0));
    }
    m.fields().remove(field.id());
  };
  read.template operator()<double>(H_field, [&](std::size_t r, double x) { b.H(r, 0) = x; });
  read.template operator()<double>(roughness_field, [&](std::size_t r, double x) { b.roughness(r, 0) = x; });
  read.template operator()<std::uint8_t>(land_field, [&](std::size_t r, std::uint8_t x) { b.land[r] = x; });
  if (comm.allreduce(bad, soma::Op::max) != 0) {
    throw std::invalid_argument("read_ocean_mesh: " + path.string() + " holds no " + H_field + ", " +
                                roughness_field + " and " + land_field + " fields of " + std::to_string(ppe) +
                                " points per element (order " + std::to_string(order) + ")");
  }
  out.removed_land_km2 = get_value(m, land_area_value);
  out.removed_disconnected_km2 = get_value(m, disconnected_area_value);
  b.rx0 = max_rx0(comm, *out.space, b.H);
  return out;
}

}  // namespace ocean
