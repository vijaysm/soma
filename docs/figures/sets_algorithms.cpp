// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The sets of the algorithms pages (docs/algorithms, docs/amr and docs/io).
// Where a figure shows an example program's result (advect, poisson, remap,
// refine), its set makes the same library calls on the same input and checks
// the quantity the example's test checks.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../../tests/unit/sphere_meshes.hpp"
#include "figure_data.hpp"
#include "soma/algorithms/cg.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/algorithms/fv.hpp"
#include "soma/algorithms/grid_index.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/polygon.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/amr/box.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/flux_register.hpp"
#include "soma/amr/hierarchy.hpp"
#include "soma/amr/interlevel.hpp"
#include "soma/amr/level_field.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/testing/box_mesh.hpp"

namespace figures {

namespace {

using Host = Kokkos::HostSpace;
using View = soma::View1<double, Host>;
using CView = soma::View1<const double, Host>;

fs::path data_file(const char* name) { return fs::path(SOMA_SOURCE_DIR) / "data" / name; }

std::string number(double x) {
  char s[32];
  std::snprintf(s, sizeof s, "%.9g", x);
  return s;
}

// The centre (corners' mean) of cell e of the mesh's dimension.
std::array<double, 3> centre(const soma::HostMesh& mesh, soma::LO e) {
  const int d = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  const soma::LO n = m.num_vertices(d, e);
  std::array<double, 3> c{0, 0, 0};
  for (soma::LO k = 0; k < n; ++k) {
    for (int q = 0; q < 3; ++q) c[static_cast<std::size_t>(q)] += x(m.vertex(d, e, k), q) / static_cast<double>(n);
  }
  return c;
}

// A plane cell's corners as a JSON list of [x, y].
std::string corners_json(const soma::HostMesh& mesh, soma::LO e) {
  const int d = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  std::string s = "[";
  for (soma::LO k = 0; k < m.num_vertices(d, e); ++k) {
    const soma::LO v = m.vertex(d, e, k);
    s += (k ? ", [" : "[") + number(x(v, 0)) + ", " + number(x(v, 1)) + "]";
  }
  return s + "]";
}

// The owned cells of a plane mesh as a JSON list of corner lists.
std::string cells_json(const soma::HostMesh& mesh) {
  std::string s = "[";
  for (soma::LO e = 0; e < mesh.segment_table(mesh.dim()).owned_end; ++e) {
    s += (e ? ",\n    " : "\n    ") + corners_json(mesh, e);
  }
  return s + "]";
}

// The lower and upper corners of the box around the mesh's vertices, over every rank.
std::array<std::array<double, 3>, 2> bounds(const soma::Comm& comm, const soma::HostMesh& mesh) {
  const auto x = mesh.coords();
  std::array<std::array<double, 3>, 2> b{};
  for (int q = 0; q < 3; ++q) {
    double lo = HUGE_VAL, hi = -HUGE_VAL;
    for (soma::LO v = 0; v < mesh.view().count(0); ++v) {
      lo = std::fmin(lo, x(v, q));
      hi = std::fmax(hi, x(v, q));
    }
    b[0][static_cast<std::size_t>(q)] = comm.allreduce(lo, soma::Op::min);
    b[1][static_cast<std::size_t>(q)] = comm.allreduce(hi, soma::Op::max);
  }
  return b;
}

// smooth: a 12 x 12 box of unit-square quads, its inner vertices moved by up to
// 0.3 of a cell in each direction (a fixed hash of their gid), then smoothed by
// 10 sweeps as the laplacian_smooth example does; before.vtk and after.vtk with
// the cell field "area", each cell's area over the unperturbed cell's.
void set_smooth(const soma::Comm& comm, const fs::path& out) {
  constexpr int n = 12;
  constexpr double h = 1.0 / n;
  soma::HostMesh mesh = soma::testing::box_mesh(
      {.nx = n, .ny = n, .nz = 1, .element = soma::Topology::quad, .lx = 1, .ly = 1, .lz = 1});
  const auto x = mesh.coords();
  const auto gid = mesh.gid(0);
  const auto inner = [](double t) { return t > 1e-12 && t < 1 - 1e-12; };
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    if (!inner(x(v, 0)) || !inner(x(v, 1))) continue;
    // splitmix64 of the gid: the same moves on every machine.
    std::uint64_t z = static_cast<std::uint64_t>(gid(static_cast<std::size_t>(v))) * 0x9E3779B97F4A7C15ull;
    for (int q = 0; q < 2; ++q) {
      z += 0x9E3779B97F4A7C15ull;
      std::uint64_t r = z;
      r = (r ^ (r >> 30)) * 0xBF58476D1CE4E5B9ull;
      r = (r ^ (r >> 27)) * 0x94D049BB133111EBull;
      r ^= r >> 31;
      x(v, q) += 0.3 * h * (2 * static_cast<double>(r >> 11) / 9007199254740992.0 - 1);
    }
  }
  const auto areas = [&](const char* file) {
    const auto m = mesh.view();
    std::vector<double> a;
    for (soma::LO e = 0; e < mesh.count(2); ++e) {
      double s = 0;
      for (soma::LO k = 0; k < 4; ++k) {
        const soma::LO p = m.vertex(2, e, k), q = m.vertex(2, e, (k + 1) % 4);
        s += x(p, 0) * x(q, 1) - x(q, 0) * x(p, 1);
      }
      a.push_back(s / 2 / (h * h));
    }
    if (!mesh.fields().contains("area")) mesh.fields().add_tag<double>("area", 2, 1);
    const auto f = mesh.fields().get<double>("area", 2).view();
    for (soma::LO e = 0; e < mesh.count(2); ++e) f(e, 0) = a[static_cast<std::size_t>(e)];
    soma::write_file(comm, mesh, (out / file).string(), {.vtk_fields = std::vector<std::string>{"area"}});
    return std::pair{*std::min_element(a.begin(), a.end()), *std::max_element(a.begin(), a.end())};
  };
  const auto [lo0, hi0] = areas("before.vtk");
  std::vector<std::array<double, 2>> skin;
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    if (!inner(x(v, 0)) || !inner(x(v, 1))) skin.push_back({x(v, 0), x(v, 1)});
  }
  mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
  const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, {.iterations = 10});
  mesh.remove_ghosts(comm);
  const auto [lo1, hi1] = areas("after.vtk");
  std::size_t k = 0;
  bool kept = true;
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    if (!inner(x(v, 0)) || !inner(x(v, 1))) kept = kept && k < skin.size() && skin[k++] == std::array{x(v, 0), x(v, 1)};
  }
  require(report.iterations == 10, "smooth: 10 sweeps");
  require(kept && k == skin.size(), "smooth: the skin stays");
  // The caption's values, to its two decimals.
  const auto rounds_to = [](double v, double shown) { return std::abs(v - shown) <= 0.005; };
  require(rounds_to(lo0, 0.46) && rounds_to(hi0, 1.61) && rounds_to(lo1, 0.99) && rounds_to(hi1, 1.01),
          "smooth: areas 0.46 to 1.61 before, 0.99 to 1.01 after");
  write_json(comm, out / "smooth.json",
             "{\"before\": [" + number(lo0) + ", " + number(hi0) + "], \"after\": [" + number(lo1) + ", " +
                 number(hi1) + "]}");
}

struct One {
  KOKKOS_INLINE_FUNCTION double operator()(double, double, double) const { return 1; }
};

// poisson: -lap u = 1 with u = 0 on the boundary of data/64bricks_1khex.h5m,
// solved as the poisson example does; poisson.vtk with the vertex field "u".
void set_poisson(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh = soma::read_file(comm, data_file("64bricks_1khex.h5m"));
  mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
  double largest = 0;
  bool converged = false;
  {
    const soma::FemSpace<Host> space(comm, mesh, soma::Assembly::gather);
    soma::CsrMatrix<Host> a = space.laplacian(comm);
    auto b = space.load(comm, One{});
    const auto zero = space.vector();
    soma::apply_dirichlet(a, b, space.boundary(), CView(zero));
    auto u = space.vector();
    soma::CgOptions options;
    options.rtol = 1e-10;
    converged = soma::cg(comm, space, a, CView(b), u, options).converged;
    const auto values = mesh.fields().add_tag<double>("u", 0, 1).view();
    for (soma::LO v = 0; v < space.cols(); ++v) {
      values(v, 0) = u(v);
      if (v < space.rows()) largest = std::max(largest, u(v));
    }
  }
  largest = comm.allreduce(largest, soma::Op::max);
  mesh.remove_ghosts(comm);
  soma::write_file(comm, mesh, (out / "poisson.vtk").string(), {.vtk_fields = std::vector<std::string>{"u"}});
  char printed[32];
  std::snprintf(printed, sizeof printed, "%.6e", largest);
  require(converged, "poisson: CG converged");
  require(std::string(printed) == "9.090723e-01", "poisson: max u 9.090723e-01, as the example's test");
}

struct Rotation {
  double cx, cy;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double x, double y, double) const {
    constexpr double omega = 2 * 3.14159265358979323846;
    return {-omega * (y - cy), omega * (x - cx), 0};
  }
};

// advect: the advect example on data/64bricks_1khex.h5m: a Gaussian bump
// turned about the box's axis by 100 upwind steps at CFL 0.5; advect.vtk with
// the cell fields "start" and "u" (after).
void set_advect(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh = soma::read_file(comm, data_file("64bricks_1khex.h5m"));
  mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
  const auto [lo, hi] = bounds(comm, mesh);
  const Rotation a{(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2};
  const double bx = a.cx + (hi[0] - lo[0]) / 4, by = a.cy, bz = (lo[2] + hi[2]) / 2;
  const double width = (hi[0] - lo[0]) / 8;
  double before = 0, after = 0;
  {
    const soma::FvSpace<Host> space(comm, mesh);
    const auto phi = space.normal_flux(a);
    const CView fluxes(phi);
    auto u = space.vector();
    for (soma::LO e = 0; e < space.cols(); ++e) {
      const auto c = centre(mesh, e);
      const double r2 = (c[0] - bx) * (c[0] - bx) + (c[1] - by) * (c[1] - by) + (c[2] - bz) * (c[2] - bz);
      u(e) = std::exp(-r2 / (width * width));
    }
    const auto start = mesh.fields().add_tag<double>("start", 3, 1).view();
    for (soma::LO e = 0; e < space.cols(); ++e) start(e, 0) = u(e);
    before = space.total(comm, CView(u));
    const double dt = 0.5 * space.max_dt(comm, fluxes);
    for (int n = 0; n < 100; ++n) space.step(comm, fluxes, u, dt);
    after = space.total(comm, CView(u));
    const auto values = mesh.fields().add_tag<double>("u", 3, 1).view();
    for (soma::LO e = 0; e < space.cols(); ++e) values(e, 0) = u(e);
  }
  mesh.remove_ghosts(comm);
  soma::write_file(comm, mesh, (out / "advect.vtk").string(),
                   {.vtk_fields = std::vector<std::string>{"start", "u"}});
  // The example's test prints these to the last digit with --soma-reproducible.
  require(std::abs(before - 6.948772654115442e-01) < 1e-12, "advect: total 6.948772654115442e-01 before");
  require(std::abs(after - 4.446464981696669e-01) < 1e-12, "advect: total 4.446464981696669e-01 after");
}

// The remap example's target: n x n squares over [lo, hi] in the plane z =
// lo[2], each cut into two triangles along its diagonal from (i, j), on one rank.
soma::HostMesh plane_grid(const soma::Comm& comm, int n, const std::array<double, 3>& lo,
                          const std::array<double, 3>& hi) {
  const auto id = [&](int i, int j) { return static_cast<soma::GO>(j * (n + 1) + i + 1); };
  std::vector<soma::GO> cells, conn, verts;
  std::vector<double> xyz;
  soma::GO e = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const soma::GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      cells.insert(cells.end(), {e + 1, e + 2});
      conn.insert(conn.end(), {a, b, c, a, c, d});
      e += 2;
    }
  }
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      verts.push_back(id(i, j));
      xyz.insert(xyz.end(), {lo[0] + (hi[0] - lo[0]) * i / n, lo[1] + (hi[1] - lo[1]) * j / n, lo[2]});
    }
  }
  soma::MeshBuilder b(2);
  if (comm.rank() == 0) {
    b.add_vertices(verts, xyz);
    b.add_elements(soma::Topology::tri, cells, conn);
  }
  return soma::build_local_part(comm, std::move(b));
}

// The remap example's box: the source's bounding box widened by 1e-3 of its width in x and y.
std::array<std::array<double, 3>, 2> widened(const soma::Comm& comm, const soma::HostMesh& source) {
  auto b = bounds(comm, source);
  for (std::size_t q = 0; q < 2; ++q) {
    const double w = 1e-3 * (b[1][q] - b[0][q]);
    b[0][q] -= w;
    b[1][q] += w;
  }
  return b;
}

// The remap example's field: 2 + sin(x) cos(y) at each owned cell's corners' mean.
View remap_values(const soma::HostMesh& mesh) {
  View u("u", static_cast<std::size_t>(mesh.segment_table(mesh.dim()).owned_end));
  for (soma::LO e = 0; e < static_cast<soma::LO>(u.extent(0)); ++e) {
    const auto c = centre(mesh, e);
    u(e) = 2 + std::sin(c[0]) * std::cos(c[1]);
  }
  return u;
}

// remap: the remap example on data/hybrid_3quad_1tri.h5m: source.vtk and
// target.vtk (the 24 x 24 grid of triangle pairs) with the cell field "u".
void set_remap(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh source = soma::read_file(comm, data_file("hybrid_3quad_1tri.h5m"));
  const auto [lo, hi] = widened(comm, source);
  soma::HostMesh target = plane_grid(comm, 24, lo, hi);
  const soma::RemapOperator<Host> remap(comm, source, target, {});
  const soma::RemapOperator<Host> back(comm, target, source, {});
  const View u = remap_values(source);
  View ut("ut", static_cast<std::size_t>(remap.rows()));
  remap.apply(comm, CView(u), ut);
  double before = 0, after = 0;
  for (soma::LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(e);
  for (soma::LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * ut(e);
  before = comm.allreduce(before, soma::Op::sum);
  after = comm.allreduce(after, soma::Op::sum);
  put_cells(source, "u", [&](soma::LO e) { return u(e); });
  put_cells(target, "u", [&](soma::LO e) { return ut(e); });
  soma::write_file(comm, source, (out / "source.vtk").string(), {.vtk_fields = std::vector<std::string>{"u"}});
  soma::write_file(comm, target, (out / "target.vtk").string(), {.vtk_fields = std::vector<std::string>{"u"}});
  char printed[64];
  std::snprintf(printed, sizeof printed, "%.10e %.10e", before, after);
  require(std::string(printed) == "2.7464101622e+01 2.7464101622e+01",
          "remap: total 2.7464101622e+01 on the source and on the grid, as the example's test");
  require(target.count(2) == 2 * 24 * 24, "remap: 1152 target cells");
  bool zero = false;
  for (soma::LO e = 0; e < remap.rows(); ++e) zero = zero || ut(e) == 0;
  require(zero, "remap: target cells outside the source get 0");
}

soma::detail::Polygon polygon(const soma::HostMesh& mesh, soma::LO e) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  soma::detail::Polygon p;
  for (soma::LO k = 0; k < m.num_vertices(2, e); ++k) p.push(x(m.vertex(2, e, k), 0), x(m.vertex(2, e, k), 1));
  return p;
}

// overlap: the remap example's meshes with a 6 x 6 grid as the target; for
// the target cell nearest (0.3, 0.3), its overlaps with the source cells as
// remap's plan has them, each clipped again (soma::detail::clip) for its
// polygon, as overlap.json.
void set_overlap(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh source = soma::read_file(comm, data_file("hybrid_3quad_1tri.h5m"));
  const auto [lo, hi] = widened(comm, source);
  soma::HostMesh target = plane_grid(comm, 6, lo, hi);
  const soma::detail::RemapPlan plan = soma::detail::remap_plan(comm, source, target, {});
  soma::LO t = 0;
  double best = HUGE_VAL;
  for (soma::LO e = 0; e < target.count(2); ++e) {
    const auto c = centre(target, e);
    const double d = std::hypot(c[0] - 0.3, c[1] - 0.3);
    if (d < best) best = d, t = e;
  }
  const soma::detail::Polygon cell = polygon(target, t);
  const auto source_gid = source.gid(2);
  std::map<soma::GO, soma::LO> lid;
  for (soma::LO e = 0; e < source.count(2); ++e) lid[source_gid(static_cast<std::size_t>(e))] = e;
  std::string pieces = "[";
  double sum = 0;
  const auto row = static_cast<std::size_t>(t);
  for (soma::Offset k = plan.offsets[row]; k < plan.offsets[row + 1]; ++k) {
    const auto kk = static_cast<std::size_t>(k);
    const soma::GO g = plan.source_gid[static_cast<std::size_t>(plan.columns[kk])];
    const soma::detail::Polygon s = polygon(source, lid.at(g));
    require(soma::detail::twice_area(s) > 0, "overlap: source cells counterclockwise");
    soma::detail::Polygon piece;
    const double area = soma::detail::clip(s, cell, 1e-12, piece);
    require(std::abs(area - plan.overlap[kk]) < 1e-12 * plan.target_area[row], "overlap: clip gives the plan's area");
    sum += plan.overlap[kk];
    pieces += std::string(k > plan.offsets[row] ? ", " : "") + "{\"gid\": " + std::to_string(g) +
              ", \"area\": " + number(area) + ", \"source\": " + corners_json(source, lid.at(g)) + ", \"piece\": [";
    for (int i = 0; i < piece.n; ++i) {
      pieces += (i ? ", [" : "[") + number(piece.x[i]) + ", " + number(piece.y[i]) + "]";
    }
    pieces += "]}";
  }
  pieces += "]";
  const auto count = plan.offsets[row + 1] - plan.offsets[row];
  require(count == 6, "overlap: the cell meets six source cells");
  require(std::abs(sum - plan.target_area[row]) < 1e-12 * plan.target_area[row],
          "overlap: the pieces tile the target cell");
  write_json(comm, out / "overlap.json",
             "{\"source\": " + cells_json(source) + ",\n \"target\": " + cells_json(target) +
                 ",\n \"cell\": " + std::to_string(t) + ", \"area\": " + number(plan.target_area[row]) +
                 ",\n \"pieces\": " + pieces + "}");
}

// A mesh of one rank's sphere cells (tests/unit/sphere_meshes.hpp).
soma::HostMesh sphere_mesh(const soma::Comm& comm, const soma::test::SphereCells& g) {
  soma::MeshBuilder b(2);
  if (comm.rank() == 0) {
    std::vector<soma::GO> gids, conn;
    for (std::size_t e = 0; e < g.cells.size(); ++e) {
      gids.push_back(static_cast<soma::GO>(e + 1));
      conn.insert(conn.end(), g.cells[e].begin(), g.cells[e].end());
    }
    b.add_vertices(g.vg, g.xyz);
    b.add_elements(g.type, gids, conn);
  }
  return soma::build_local_part(comm, std::move(b));
}

// sphere: a field remapped on the unit sphere, from an icosphere of 1280
// triangles to the cubed sphere of data/sphere_quads_20.vtk (2400 quads);
// sphere-source.vtk and sphere-target.vtk with the cell field "u".
void set_sphere(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh source = sphere_mesh(comm, soma::test::icosphere(3, 1.0));
  soma::HostMesh target = soma::read_file(comm, data_file("sphere_quads_20.vtk"));
  const soma::RemapOptions options{.surface = soma::Surface::sphere, .radius = 1.0};
  const soma::RemapOperator<Host> remap(comm, source, target, options);
  const soma::RemapOperator<Host> back(comm, target, source, options);
  View u("u", static_cast<std::size_t>(source.segment_table(2).owned_end));
  for (soma::LO e = 0; e < static_cast<soma::LO>(u.extent(0)); ++e) {
    auto c = centre(source, e);
    const double r = std::hypot(c[0], c[1], c[2]);
    for (double& q : c) q /= r;
    u(e) = 2 + c[0] * c[0] - c[1] * c[1] + c[2] / 2;
  }
  View ut("ut", static_cast<std::size_t>(remap.rows()));
  remap.apply(comm, CView(u), ut);
  double before = 0, after = 0;
  for (soma::LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(e);
  for (soma::LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * ut(e);
  before = comm.allreduce(before, soma::Op::sum);
  after = comm.allreduce(after, soma::Op::sum);
  put_cells(source, "u", [&](soma::LO e) { return u(e); });
  put_cells(target, "u", [&](soma::LO e) { return ut(e); });
  soma::write_file(comm, source, (out / "sphere-source.vtk").string(), {.vtk_fields = std::vector<std::string>{"u"}});
  soma::write_file(comm, target, (out / "sphere-target.vtk").string(), {.vtk_fields = std::vector<std::string>{"u"}});
  require(source.count(2) == 1280 && target.count(2) == 2400, "sphere: 1280 triangles and 2400 quads");
  require(std::abs(after - before) < 1e-12 * before, "sphere: the total comes through");
}

// children: each of a tri, quad, tet and hex (its reference corners from the
// order-1 node table) refined once by soma::refine_uniform; the children's
// corners, as children.json.
void set_children(const soma::Comm& comm, const fs::path& out) {
  std::string js = "{";
  using soma::Topology;
  for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::hex}) {
    const soma::NodeTableHost table = soma::make_node_table(
        {.topology = t, .family = soma::NodeFamily::lagrange_equispaced, .order = 1,
         .ordering = soma::NodeOrdering::moab_native});
    const int dim = soma::dimension(t);
    std::vector<soma::GO> verts;
    std::vector<double> xyz;
    for (int i = 0; i < table.count; ++i) {
      verts.push_back(i + 1);
      for (int q = 0; q < 3; ++q) xyz.push_back(q < table.dim ? table.coord(i, q) : 0.0);
    }
    soma::MeshBuilder b(dim);
    if (comm.rank() == 0) {
      b.add_vertices(verts, xyz);
      const std::vector<soma::GO> cell{1};
      b.add_elements(t, cell, verts);
    }
    const soma::HostMesh mesh = soma::build_local_part(comm, std::move(b));
    const soma::HostMesh fine = soma::refine_uniform(comm, mesh, {.levels = 1}).mesh;
    const int expected = dim == 2 ? 4 : 8;
    require(fine.count(dim) == expected, std::string(soma::name(t)) + ": " + std::to_string(expected) + " children");
    const auto m = fine.view();
    const auto x = fine.coords();
    js += std::string(js.size() > 1 ? "," : "") + "\n  \"" + soma::name(t) + "\": {\"dim\": " +
          std::to_string(dim) + ", \"parent\": [";
    for (int i = 0; i < table.count; ++i) {
      js += std::string(i ? ", " : "") + "[" + number(xyz[3 * i]) + ", " + number(xyz[3 * i + 1]) + ", " +
            number(xyz[3 * i + 2]) + "]";
    }
    js += "], \"children\": [";
    for (soma::LO e = 0; e < fine.count(dim); ++e) {
      js += e ? ", [" : "[";
      for (soma::LO k = 0; k < m.num_vertices(dim, e); ++k) {
        const soma::LO v = m.vertex(dim, e, k);
        js += std::string(k ? ", " : "") + "[" + number(x(v, 0)) + ", " + number(x(v, 1)) + ", " + number(x(v, 2)) +
              "]";
      }
      js += "]";
    }
    js += "], \"topologies\": [";
    for (soma::LO e = 0; e < fine.count(dim); ++e) {
      js += std::string(e ? ", " : "") + "\"" + soma::name(m.topology(dim, e)) + "\"";
    }
    js += "]}";
  }
  write_json(comm, out / "children.json", js + "\n}");
}

// The refine example's indicator: h / (r + h), h a cell's largest distance
// from its centre to a corner and r its centre's distance from `corner`.
std::vector<double> indicator(const soma::HostMesh& mesh, const std::array<double, 3>& corner) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  std::vector<double> eta;
  for (soma::LO e = 0; e < mesh.segment_table(dim).owned_end; ++e) {
    const auto c = centre(mesh, e);
    double h = 0;
    for (soma::LO i = 0; i < m.num_vertices(dim, e); ++i) {
      const soma::LO v = m.vertex(dim, e, i);
      h = std::max(h, std::hypot(x(v, 0) - c[0], x(v, 1) - c[1], x(v, 2) - c[2]));
    }
    eta.push_back(h / (std::hypot(c[0] - corner[0], c[1] - corner[1], c[2] - corner[2]) + h));
  }
  return eta;
}

// marked: the refine example's marked refinement of data/hybrid_3quad_1tri.h5m
// (3 levels, a fraction 0.5 of the indicator towards the low corner), with
// hanging vertices (hanging.vtk, hanging.json) and conforming (conforming.vtk);
// the cell field "level" is each cell's refinement level.
void set_marked(const soma::Comm& comm, const fs::path& out) {
  for (const bool conforming : {false, true}) {
    soma::HostMesh mesh = soma::read_file(comm, data_file("hybrid_3quad_1tri.h5m"));
    const std::array<double, 3> corner = bounds(comm, mesh)[0];
    soma::HostMesh fine;
    const soma::HostMesh* from = &mesh;
    std::vector<std::array<double, 2>> hanging;
    for (int level = 0; level < 3; ++level) {
      const auto marks = soma::mark_fraction(comm, indicator(*from, corner), 0.5);
      soma::RefineResult r = soma::refine_marked(comm, *from, marks, {.conforming = conforming});
      hanging.clear();
      for (const soma::LO v : r.hanging) hanging.push_back({r.mesh.coords()(v, 0), r.mesh.coords()(v, 1)});
      fine = soma::rebalance(comm, r.mesh);
      from = &fine;
    }
    const auto level = fine.fields().get<std::int32_t>("refine:level", 2).view();
    put_cells(fine, "level", [&](soma::LO e) { return level(e, 0); });
    const char* name = conforming ? "conforming" : "hanging";
    soma::write_file(comm, fine, (out / (std::string(name) + ".vtk")).string(),
                     {.vtk_fields = std::vector<std::string>{"level"}});
    if (conforming) {
      // Each pass marks its own mesh, so the conforming run refines other cells than the default.
      require(hanging.empty() && fine.count(2) == 929, "marked: conforming: 929 cells, no hanging vertex");
      continue;
    }
    require(fine.count(2) == 730 && fine.count(0) == 751 && hanging.size() == 35,
            "marked: cells 64 -> 730, vertices 75 -> 751, 35 hanging, as the example's test");
    std::string js = "[";
    for (std::size_t i = 0; i < hanging.size(); ++i) {
      js += (i ? ", [" : "[") + number(hanging[i][0]) + ", " + number(hanging[i][1]) + "]";
    }
    write_json(comm, out / "hanging.json", js + "]");
  }
}

// search: the boxes around the cells of data/hybrid_3quad_1tri.h5m in a
// soma::GridIndex, and two query boxes; the boxes, the grid's bins and the
// pairs found, checked against all pairs, as search.json.
void set_search(const soma::Comm& comm, const fs::path& out) {
  const soma::HostMesh mesh = soma::read_file(comm, data_file("hybrid_3quad_1tri.h5m"));
  const soma::LO n = mesh.count(2);
  Kokkos::View<double**, Kokkos::LayoutLeft, Host> boxes("boxes", static_cast<std::size_t>(n), 6);
  const auto m = mesh.view();
  const auto x = mesh.coords();
  for (soma::LO e = 0; e < n; ++e) {
    for (int q = 0; q < 3; ++q) {
      boxes(e, q) = HUGE_VAL;
      boxes(e, q + 3) = -HUGE_VAL;
      for (soma::LO k = 0; k < m.num_vertices(2, e); ++k) {
        boxes(e, q) = std::fmin(boxes(e, q), x(m.vertex(2, e, k), q));
        boxes(e, q + 3) = std::fmax(boxes(e, q + 3), x(m.vertex(2, e, k), q));
      }
    }
  }
  const double query[2][6] = {{-0.8, -1.7, 0, -0.2, -0.9, 0}, {1.0, 1.0, 0, 1.8, 1.6, 0}};
  Kokkos::View<double**, Kokkos::LayoutLeft, Host> queries("queries", 2, 6);
  for (int q = 0; q < 2; ++q) {
    for (int a = 0; a < 6; ++a) queries(q, a) = query[q][a];
  }
  const soma::GridIndex<Host> index(boxes);
  const auto pairs = index.query(queries);
  // The bins split the span of the boxes evenly: bins() of them along each axis, as GridIndex
  // builds them (origin the boxes' low corner, width span / bins; see soma/algorithms/grid_index.hpp).
  const std::array<int, 3> bins = index.bins();
  double origin[2], width[2];
  for (int a = 0; a < 2; ++a) {
    double lo = HUGE_VAL, hi = -HUGE_VAL;
    for (soma::LO e = 0; e < n; ++e) lo = std::fmin(lo, boxes(e, a)), hi = std::fmax(hi, boxes(e, a + 3));
    origin[a] = lo;
    width[a] = (hi - lo) / bins[static_cast<std::size_t>(a)];
  }
  std::string js = "{\"boxes\": [";
  for (soma::LO e = 0; e < n; ++e) {
    js += std::string(e ? ",\n    [" : "\n    [") + number(boxes(e, 0)) + ", " + number(boxes(e, 1)) + ", " +
          number(boxes(e, 3)) + ", " + number(boxes(e, 4)) + "]";
  }
  js += "],\n \"cells\": " + cells_json(mesh) + ",\n \"grid\": {\"origin\": [" + number(origin[0]) + ", " +
        number(origin[1]) + "], \"width\": [" + number(width[0]) + ", " + number(width[1]) + "], \"n\": [" +
        std::to_string(bins[0]) + ", " + std::to_string(bins[1]) + ", " + std::to_string(bins[2]) +
        "]},\n \"queries\": [";
  for (int q = 0; q < 2; ++q) {
    std::vector<soma::LO> found, all;
    for (soma::Offset k = pairs.offsets(q); k < pairs.offsets(q + 1); ++k) found.push_back(pairs.boxes(k));
    for (soma::LO e = 0; e < n; ++e) {
      bool meets = true;
      for (int a = 0; a < 3; ++a) meets = meets && query[q][a] <= boxes(e, a + 3) && boxes(e, a) <= query[q][a + 3];
      if (meets) all.push_back(e);
    }
    require(found == all, "search: the index finds every pair and only those");
    require(!found.empty(), "search: each query meets some box");
    js += std::string(q ? ", " : "") + "{\"box\": [" + number(query[q][0]) + ", " + number(query[q][1]) + ", " +
          number(query[q][3]) + ", " + number(query[q][4]) + "], \"found\": [";
    for (std::size_t k = 0; k < found.size(); ++k) js += (k ? ", " : "") + std::to_string(found[k]);
    js += "]}";
  }
  require(bins == std::array<int, 3>{9, 8, 1}, "search: 9 x 8 bins, one across z for a plane mesh");
  write_json(comm, out / "search.json", js + "]}");
}

namespace amr = soma::amr;

// A 2D box as [lo i, lo j, hi i, hi j].
std::string box_json(const amr::Box& b) {
  return "[" + std::to_string(b.lo(0)) + ", " + std::to_string(b.lo(1)) + ", " + std::to_string(b.hi(0)) + ", " +
         std::to_string(b.hi(1)) + "]";
}

std::string boxes_json(const amr::BoxList& boxes) {
  std::string s = "[";
  for (std::size_t b = 0; b < boxes.size(); ++b) s += (b ? ", " : "") + box_json(boxes[b]);
  return s + "]";
}

// The cells the hierarchy set tags on a level of cell width dx over [0, 1]^2:
// those within 1.5 cells of the circle of radius 0.3 about the centre, and on
// level 1 only its upper half.
KOKKOS_INLINE_FUNCTION bool on_ring(int level, int i, int j, double dx) {
  const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
  const double r = Kokkos::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
  return Kokkos::fabs(r - 0.3) < 1.5 * dx && (level == 0 || y > 0.5);
}

// hierarchy: a 32 x 32 domain refined twice by ratio 2 (blocking factor 4,
// boxes of at most 16): make_grids on level 0 alone, then on the two levels
// it gave; the tagged cells of levels 0 and 1 and the boxes of every level,
// as hierarchy.json.
void set_hierarchy(const soma::Comm& comm, const fs::path& out) {
  const amr::Geometry geometry{amr::Box({{0, 0, 0}}, {{31, 31, 0}}, 2)};
  const amr::HierarchyOptions options{.max_level = 2, .ratio = 2, .blocking_factor = 4, .max_box = 16};
  const auto tagger = [](int level, const amr::TagField& tags) {
    const auto v = tags.view();
    const double dx = 1.0 / (32 << level);
    amr::for_each_point(tags, amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
      if (on_ring(level, i, j, dx)) v.patch(b)(i, j, k) = 1;
    });
  };
  const amr::Hierarchy first(geometry, options);
  const amr::Hierarchy second(geometry, options, first.make_grids(tagger));
  const amr::Hierarchy h(geometry, options, second.make_grids(tagger));
  require(h.num_levels() == 3, "hierarchy: three levels");
  // The cells tagged: on level 0 of the first hierarchy, on level 1 of the second.
  std::string js = "{\"domain\": " + box_json(h.domain(0)) + ", \"tags\": [";
  for (int level = 0; level < 2; ++level) {
    const amr::BoxArray& tagged = (level == 0 ? first : second).boxes(level);
    amr::BoxList cells;
    for (std::size_t b = 0; b < tagged.size(); ++b) {
      for (int j = tagged[b].lo(1); j <= tagged[b].hi(1); ++j) {
        for (int i = tagged[b].lo(0); i <= tagged[b].hi(0); ++i) {
          if (on_ring(level, i, j, h.dx(level, 0))) cells.push_back(amr::Box({{i, j, 0}}, {{i, j, 0}}, 2));
        }
      }
    }
    // The tags of a level lie under the next level of the final hierarchy.
    for (const amr::Box& c : cells) {
      require(amr::covers(h.boxes(level + 1).boxes(), c.refine(2)), "hierarchy: the next level covers the tags");
    }
    require(!cells.empty(), "hierarchy: cells tagged");
    js += std::string(level ? ",\n  " : "\n  ") + boxes_json(cells);
  }
  js += "],\n \"levels\": [";
  for (int level = 0; level < h.num_levels(); ++level) {
    require(amr::disjoint(h.boxes(level).boxes()), "hierarchy: a level's boxes are disjoint");
    js += std::string(level ? ",\n  " : "\n  ") + boxes_json(h.boxes(level).boxes());
  }
  write_json(comm, out / "hierarchy.json", js + "]}");
}

// coarsefine: a 16 x 16 level 0 under two fine boxes in an L (ratio 2); for a
// cell field with two ghosts, the ghost regions the coarse-fine plan fills
// from level 0 (Hierarchy::coarse_fine_plan) and the coarse faces of the flux
// register (Hierarchy::make_flux_register), as coarsefine.json.
void set_coarsefine(const soma::Comm& comm, const fs::path& out) {
  const amr::Geometry geometry{amr::Box({{0, 0, 0}}, {{15, 15, 0}}, 2)};
  const amr::HierarchyOptions options{.max_level = 1, .ratio = 2, .blocking_factor = 4, .max_box = 16};
  const amr::BoxList fine{amr::Box({{8, 8, 0}}, {{15, 23, 0}}, 2), amr::Box({{16, 8, 0}}, {{23, 15, 0}}, 2)};
  const amr::Hierarchy h(geometry, options, {fine});
  constexpr int ghosts = 2;
  const amr::CoarseFinePlan& plan = h.coarse_fine_plan(1, amr::IndexType::cell(), ghosts, amr::Interp::linear);
  const amr::BoxArray& level = h.boxes(1);
  std::string js = "{\"domain\": " + box_json(h.domain(0)) + ", \"ratio\": 2, \"ghosts\": " +
                   std::to_string(ghosts) + ",\n \"fine\": " + boxes_json(level.boxes()) + ",\n \"filled\": [";
  for (std::size_t k = 0; k < plan.ops().size(); ++k) {
    const amr::InterpOp& op = plan.ops()[k];
    const amr::Box& own = level[static_cast<std::size_t>(op.fine)];
    require(own.grow(ghosts).contains(op.region), "coarsefine: a filled region lies in its box's ghosts");
    for (std::size_t b = 0; b < level.size(); ++b) {
      require(!level[b].intersects(op.region), "coarsefine: no filled point is under the fine level");
    }
    js += std::string(k ? ", " : "") + "{\"fine\": " + std::to_string(op.fine) + ", \"region\": " +
          box_json(op.region) + "}";
  }
  require(!plan.ops().empty(), "coarsefine: some ghosts are filled from level 0");
  const amr::FluxRegister reg = h.make_flux_register(1, 1);
  js += "],\n \"faces\": [";
  for (std::size_t k = 0; k < reg.faces().size(); ++k) {
    const amr::RegisterFaces& f = reg.faces()[k];
    js += std::string(k ? ", " : "") + "{\"dir\": " + std::to_string(f.dir) + ", \"side\": " +
          std::to_string(f.side) + ", \"fine\": " + std::to_string(f.fine) + ", \"faces\": " + box_json(f.faces) +
          "}";
  }
  // The L under the fine level covers 48 coarse cells; its outline is 32 coarse faces long.
  require(reg.num_faces() == 32, "coarsefine: 32 register faces around the L");
  write_json(comm, out / "coarsefine.json", js + "]}");
}

// Per cell of the whole mesh (by its place in `gids`, ascending), value + 1
// on the rank owning it, summed over the ranks: 0 where no rank owns it.
std::vector<double> by_gid(const soma::Comm& comm, const soma::HostMesh* mesh, const std::vector<soma::GO>& gids,
                           double value) {
  std::vector<double> v(gids.size(), 0.0);
  if (mesh != nullptr) {
    const auto gid = mesh->gid(2);
    for (soma::LO e = 0; e < mesh->segment_table(2).owned_end; ++e) {
      const auto at = std::lower_bound(gids.begin(), gids.end(), gid(static_cast<std::size_t>(e)));
      require(at != gids.end() && *at == gid(static_cast<std::size_t>(e)), "readparts: a cell of the mesh read");
      v[static_cast<std::size_t>(at - gids.begin())] = value + 1;
    }
  }
  comm.allreduce(std::span<double>(v), soma::Op::sum);
  return v;
}

std::string parts_json(const std::vector<double>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(static_cast<int>(v[i]) - 1);
  return s + "]";
}

// readparts: data/hybrid_3quad_1tri.h5m written to SOMA files of 4 parts (on the 4
// ranks) and of 2 parts (on ranks 0 and 1), then read with rebalance off on 2
// ranks (N > P) and on 4 (N < P), and with rebalance on 4; for each cell by
// gid the part that wrote it and the rank that read it, as readparts.json.
void set_readparts(const soma::Comm& comm, const fs::path& out) {
  const fs::path in = data_file("hybrid_3quad_1tri.h5m");
  const soma::HostMesh mesh = soma::read_file(comm, in);
  // The whole mesh on every rank, for the cells' gids and corners.
  const soma::HostMesh whole = soma::read_file(soma::Comm::self(), in);
  std::vector<soma::GO> cells(static_cast<std::size_t>(whole.count(2)));
  for (std::size_t e = 0; e < cells.size(); ++e) cells[e] = whole.gid(2)(e);
  std::vector<soma::LO> order(cells.size());
  for (std::size_t e = 0; e < cells.size(); ++e) order[e] = static_cast<soma::LO>(e);
  std::sort(order.begin(), order.end(), [&](soma::LO a, soma::LO b) {
    return cells[static_cast<std::size_t>(a)] < cells[static_cast<std::size_t>(b)];
  });
  std::sort(cells.begin(), cells.end());
  soma::write_soma(comm, mesh, out / "four.soma");
  const std::vector<double> wrote4 = by_gid(comm, &mesh, cells, comm.rank());
  const soma::SplitComm pair = comm.split(comm.rank() < 2 ? 0 : -1, comm.rank());
  soma::HostMesh two, back;
  if (pair.member()) {
    back = soma::read_soma(pair.comm(), out / "four.soma", {.rebalance = false});
    two = soma::read_file(pair.comm(), in);
    soma::write_soma(pair.comm(), two, out / "two.soma");
  }
  const std::vector<double> read2 = by_gid(comm, pair.member() ? &back : nullptr, cells, comm.rank());
  const std::vector<double> wrote2 = by_gid(comm, pair.member() ? &two : nullptr, cells, comm.rank());
  const soma::HostMesh kept = soma::read_soma(comm, out / "two.soma", {.rebalance = false});
  const std::vector<double> read4 = by_gid(comm, &kept, cells, comm.rank());
  const soma::HostMesh spread = soma::read_soma(comm, out / "two.soma");
  const std::vector<double> rebalanced = by_gid(comm, &spread, cells, comm.rank());
  require(comm.size() == 4 && cells.size() == 64, "readparts: 64 cells on 4 ranks");
  // Part q of N goes to rank floor(qP/N).
  const auto at = [](const std::vector<double>& v, std::size_t i) { return static_cast<int>(v[i]) - 1; };
  for (std::size_t i = 0; i < cells.size(); ++i) {
    require(at(read2, i) == at(wrote4, i) * 2 / 4, "readparts: N = 4, P = 2: part q on rank floor(qP/N)");
    require(at(read4, i) == at(wrote2, i) * 4 / 2, "readparts: N = 2, P = 4: part q on rank floor(qP/N)");
    require(rebalanced[i] > 0, "readparts: every cell read");
  }
  const auto holds = [&](const std::vector<double>& v, int r) {
    return std::find(v.begin(), v.end(), r + 1.0) != v.end();
  };
  require(!holds(read4, 1) && !holds(read4, 3), "readparts: without rebalance ranks 1 and 3 stay empty");
  for (int r = 0; r < 4; ++r) require(holds(rebalanced, r), "readparts: with rebalance every rank holds cells");
  if (comm.rank() == 0) {
    std::string js = "{\"cells\": [";
    for (std::size_t i = 0; i < order.size(); ++i) js += (i ? ",\n    " : "\n    ") + corners_json(whole, order[i]);
    js += "],\n \"wrote4\": " + parts_json(wrote4) + ",\n \"read2\": " + parts_json(read2) +
          ",\n \"wrote2\": " + parts_json(wrote2) + ",\n \"read4\": " + parts_json(read4) +
          ",\n \"rebalanced\": " + parts_json(rebalanced) + "}";
    write_json(comm, out / "readparts.json", js);
  }
}

}  // namespace

void add_algorithm_sets(Sets& sets) {
  sets["advect"] = {1, set_advect};
  sets["children"] = {1, set_children};
  sets["coarsefine"] = {1, set_coarsefine};
  sets["hierarchy"] = {1, set_hierarchy};
  sets["marked"] = {1, set_marked};
  sets["overlap"] = {1, set_overlap};
  sets["readparts"] = {4, set_readparts};
  sets["poisson"] = {1, set_poisson};
  sets["remap"] = {1, set_remap};
  sets["search"] = {1, set_search};
  sets["smooth"] = {1, set_smooth};
  sets["sphere"] = {1, set_sphere};
}

}  // namespace figures
