// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma_figure_data <set> <outdir>
// soma_figure_data --list
//
// Writes the data a documentation figure is drawn from: meshes as legacy VTK
// files with the fields a figure colours by, and tables as small JSON files,
// all made by SOMA itself so a figure shows what the code does. Each set runs
// at the rank count it declares (`--list` prints "<set> <ranks>" per line,
// sorted by name), under `mpiexec -n <ranks>` and with --soma-reproducible.
// docs/figures/make_figures.sh runs the sets and renders the figures; see
// docs/developer/making-figures.md.
//
// Exits 0 on success; 1 when a set fails (the message names the check); 2 for
// bad arguments, an unknown set, or the wrong rank count.

#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/fem.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/segments.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/canonical.hpp"

namespace fs = std::filesystem;

namespace {

// A failed check of a set's data; main prints it and exits 1.
struct SetError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void require(bool ok, const std::string& what) {
  if (!ok) throw SetError(what);
}

// A cell field from a function of the cell's local id.
void put_cells(soma::HostMesh& mesh, const std::string& name, const std::function<double(soma::LO)>& value) {
  const int dim = mesh.dim();
  auto f = mesh.fields().add_tag<double>(name, dim, 1).view();
  for (soma::LO e = 0; e < mesh.count(dim); ++e) f(static_cast<std::size_t>(e), 0) = value(e);
}

// box: a 4 x 3 x 2 box of unit hexes (soma::testing::box_mesh), cells coloured by gid.
void set_box(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh = soma::testing::box_mesh({.nx = 4, .ny = 3, .nz = 2, .lx = 4, .ly = 3, .lz = 2});
  const auto gid = mesh.gid(3);
  put_cells(mesh, "gid", [&](soma::LO e) { return static_cast<double>(gid(static_cast<std::size_t>(e))); });
  require(mesh.count(3) == 24, "box: 24 cells");
  soma::write_file(comm, mesh, (out / "box.vtk").string(), {.vtk_fields = std::vector<std::string>{"gid"}});
}

// The text of a JSON file, written on the first rank.
void write_json(const soma::Comm& comm, const fs::path& path, const std::string& text) {
  if (comm.rank() != 0) return;
  std::ofstream f(path);
  f << text << "\n";
  require(static_cast<bool>(f), "cannot write " + path.string());
}

using Point = std::array<double, 3>;

// Reference coordinates of the cells, in MOAB's convention, with which the
// figures draw SOMA's numbering. set_reference checks them against SOMA: each
// face of soma::canonical must be planar with its right-hand normal pointing
// out of the cell, and the FEM shape functions must be 1 at their own corner.
const std::map<soma::Topology, std::vector<Point>>& reference_points() {
  using soma::Topology;
  static const std::map<Topology, std::vector<Point>> all = {
      {Topology::tri, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}},
      {Topology::quad, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}},
      {Topology::tet, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
      {Topology::pyramid, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}},
      {Topology::prism, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}},
      {Topology::hex, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}},
  };
  return all;
}

Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

std::vector<int> side_vertices(soma::Topology t, int d, int i) {
  const soma::canonical::SideDef side = soma::canonical::side(t, d, i);
  return std::vector<int>(side.v, side.v + side.num_vertices);
}

// The checks above, for one topology.
void check_reference(soma::Topology t, const std::vector<Point>& x) {
  const std::string name = soma::name(t);
  require(static_cast<int>(x.size()) == soma::num_vertices(t), name + ": vertex count");
  Point centre{0, 0, 0};
  for (const Point& p : x)
    for (int a = 0; a < 3; ++a) centre[a] += p[a] / static_cast<double>(x.size());
  const int dim = soma::dimension(t);
  if (dim == 2) {
    // The cell itself, counterclockwise seen from +z, as its edges run.
    require(cross(sub(x[1], x[0]), sub(x[2], x[0]))[2] > 0, name + ": counterclockwise");
  }
  for (int f = 0; dim == 3 && f < soma::canonical::num_sides(t, 2); ++f) {
    const std::vector<int> v = side_vertices(t, 2, f);
    const Point n = cross(sub(x[v[1]], x[v[0]]), sub(x[v.back()], x[v[0]]));
    for (std::size_t k = 3; k < v.size(); ++k) {
      require(std::abs(dot(n, sub(x[v[k]], x[v[0]]))) < 1e-12, name + ": face " + std::to_string(f) + " planar");
    }
    require(dot(n, sub(x[v[0]], centre)) > 0, name + ": face " + std::to_string(f) + " points out");
  }
  if (t == soma::Topology::tri || t == soma::Topology::tet || t == soma::Topology::quad || t == soma::Topology::hex) {
    const bool box = t == soma::Topology::quad || t == soma::Topology::hex;
    for (std::size_t a = 0; a < x.size(); ++a) {
      soma::Real xi[3], n[8], dn[8][3];
      for (int k = 0; k < 3; ++k) xi[k] = box ? 2 * x[a][k] - 1 : x[a][k];
      const int nodes = soma::fem::shape(t, xi, n, dn);
      require(nodes == static_cast<int>(x.size()), name + ": shape functions");
      for (int b = 0; b < nodes; ++b) {
        require(std::abs(n[b] - (static_cast<int>(a) == b ? 1 : 0)) < 1e-12, name + ": FEM corner order");
      }
    }
  }
}

// reference: for each cell, its corners (in the convention checked above) and
// its edges and faces from soma::canonical, as reference.json.
void set_reference(const soma::Comm& comm, const fs::path& out) {
  std::ostringstream js;
  js << "{";
  const char* sep = "";
  for (const auto& [t, x] : reference_points()) {
    check_reference(t, x);
    js << sep << "\n  \"" << soma::name(t) << "\": {\"dim\": " << soma::dimension(t) << ", \"points\": [";
    for (std::size_t a = 0; a < x.size(); ++a) {
      js << (a ? ", " : "") << "[" << x[a][0] << ", " << x[a][1] << ", " << x[a][2] << "]";
    }
    for (int d = 1; d <= 2; ++d) {
      js << "], \"" << (d == 1 ? "edges" : "faces") << "\": [";
      const int count = d < soma::dimension(t) ? soma::canonical::num_sides(t, d) : 0;
      for (int i = 0; i < count; ++i) {
        const std::vector<int> v = side_vertices(t, d, i);
        js << (i ? ", " : "") << "[";
        for (std::size_t k = 0; k < v.size(); ++k) js << (k ? ", " : "") << v[k];
        js << "]";
      }
    }
    js << "]}";
    sep = ",";
  }
  js << "\n}";
  require(soma::canonical::num_sides(soma::Topology::hex, 1) == 12, "hex: 12 edges");
  require(soma::canonical::num_sides(soma::Topology::hex, 2) == 6, "hex: 6 faces");
  require(soma::canonical::num_sides(soma::Topology::tet, 1) == 6, "tet: 6 edges");
  require(soma::canonical::num_sides(soma::Topology::tet, 2) == 4, "tet: 4 faces");
  write_json(comm, out / "reference.json", js.str());
}

// A JSON list of a view's or vector's values.
template <class V>
std::string json_list(const V& v, std::size_t begin, std::size_t end) {
  std::ostringstream js;
  js << "[";
  for (std::size_t i = begin; i < end; ++i) js << (i > begin ? ", " : "") << v(i);
  js << "]";
  return js.str();
}

const char* ownership_name(soma::Ownership o) {
  switch (o) {
    case soma::Ownership::interior: return "interior";
    case soma::Ownership::shared: return "shared";
    case soma::Ownership::shared_copy: return "shared_copy";
    case soma::Ownership::ghost: return "ghost";
  }
  return "?";
}

// segments: data/hybrid_3quad_1tri.h5m read on two ranks with one ghost layer;
// each rank's segment table of every dimension it holds, as segments-<rank>.json.
void set_segments(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh = soma::read_file(comm, fs::path(SOMA_SOURCE_DIR) / "data" / "hybrid_3quad_1tri.h5m");
  mesh.add_ghosts(comm, soma::GhostOptions{.layers = 1});
  std::ostringstream js;
  js << "{\"rank\": " << comm.rank() << ", \"ranks\": " << comm.size() << ", \"dims\": [";
  const char* sep = "";
  for (int d = 0; d <= mesh.dim(); ++d) {
    const soma::SegmentTable& t = mesh.segment_table(d);
    if (t.count == 0) continue;
    const std::array<soma::LO, 5> bounds = soma::kind_bounds(t);
    require(bounds[0] == 0 && bounds[1] == t.interior_end && bounds[2] == t.owned_end && bounds[3] == t.shared_end &&
                bounds[4] == t.count,
            "segments: kind_bounds and the table's ends agree");
    js << sep << "\n  {\"dim\": " << d << ", \"count\": " << t.count << ", \"interior_end\": " << t.interior_end
       << ", \"owned_end\": " << t.owned_end << ", \"shared_end\": " << t.shared_end << ", \"segments\": [";
    for (std::size_t k = 0; k < t.segments.size(); ++k) {
      const soma::Segment& g = t.segments[k];
      js << (k ? ", " : "") << "{\"begin\": " << g.begin << ", \"end\": " << g.end << ", \"topology\": \""
         << soma::name(g.topology) << "\", \"ownership\": \"" << ownership_name(g.ownership) << "\"}";
    }
    js << "]}";
    sep = ",";
  }
  js << "\n]}";
  // Every rank writes its own table.
  std::ofstream f(out / ("segments-" + std::to_string(comm.rank()) + ".json"));
  f << js.str() << "\n";
  require(static_cast<bool>(f), "segments: cannot write");
  const soma::SegmentTable& cells = mesh.segment_table(mesh.dim());
  const auto ghost_cells = static_cast<long long>(cells.count - cells.shared_end);
  require(comm.allreduce(ghost_cells, soma::Op::sum) > 0, "segments: some rank holds ghost cells");
}

// graphs: a 3 x 2 box of unit quads with its vertex-to-cell graph (0, 2) and its
// cell-to-cell bridge through edges (2, 1), as graphs.json.
void set_graphs(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh =
      soma::testing::box_mesh({.nx = 3, .ny = 2, .nz = 1, .element = soma::Topology::quad, .lx = 3, .ly = 2});
  soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(0, 2).add_bridge(2, 1));
  const auto x = mesh.coords();
  std::ostringstream js;
  js << "{\"points\": [";
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    const auto i = static_cast<std::size_t>(v);
    js << (v ? ", " : "") << "[" << x(i, 0) << ", " << x(i, 1) << "]";
  }
  const auto rows = [&](const soma::AdjacencyGraph<Kokkos::HostSpace>& g, soma::LO n) {
    std::ostringstream r;
    r << "[";
    for (soma::LO e = 0; e < n; ++e) {
      const auto i = static_cast<std::size_t>(e);
      r << (e ? ", " : "") << json_list(g.targets, static_cast<std::size_t>(g.offsets(i)),
                                        static_cast<std::size_t>(g.offsets(i + 1)));
    }
    return r.str() + "]";
  };
  js << "],\n \"cells\": " << rows(mesh.adjacency(2, 0), mesh.count(2))
     << ",\n \"vertex_cells\": " << rows(mesh.adjacency(0, 2), mesh.count(0))
     << ",\n \"cell_neighbours\": " << rows(mesh.bridge(2, 1), mesh.count(2)) << "}";
  require(mesh.count(2) == 6 && mesh.count(0) == 12, "graphs: 6 quads, 12 vertices");
  write_json(comm, out / "graphs.json", js.str());
}

struct Set {
  int ranks;
  std::function<void(const soma::Comm&, const fs::path&)> make;
};

const std::map<std::string, Set>& sets() {
  static const std::map<std::string, Set> all = {
      {"box", {1, set_box}},
      {"graphs", {1, set_graphs}},
      {"reference", {1, set_reference}},
      {"segments", {2, set_segments}},
  };
  return all;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  const bool root = comm.rank() == 0;
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() == 1 && args[0] == "--list") {
    if (root) {
      for (const auto& [name, set] : sets()) std::printf("%s %d\n", name.c_str(), set.ranks);
    }
    return 0;
  }
  const auto found = args.size() == 2 ? sets().find(args[0]) : sets().end();
  if (found == sets().end()) {
    if (root) {
      std::fprintf(stderr, "usage: soma_figure_data <set> <outdir> | --list\nsets:");
      for (const auto& [name, set] : sets()) std::fprintf(stderr, " %s", name.c_str());
      std::fprintf(stderr, "\n");
    }
    return 2;
  }
  if (comm.size() != found->second.ranks) {
    if (root) {
      std::fprintf(stderr, "soma_figure_data: %s runs on %d rank(s), not %d\n", found->first.c_str(),
                   found->second.ranks, comm.size());
    }
    return 2;
  }
  try {
    const fs::path out = args[1];
    if (root) fs::create_directories(out);
    comm.barrier();
    found->second.make(comm, out);
    if (root) std::printf("soma_figure_data: %s written to %s\n", found->first.c_str(), out.c_str());
    return 0;
  } catch (const std::exception& e) {
    if (root) std::fprintf(stderr, "soma_figure_data: %s: %s\n", found->first.c_str(), e.what());
    return 1;
  }
}
