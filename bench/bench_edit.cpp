// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Queries on edited meshes against the same queries on compact ones, and
// against MOAB. A fraction f of the hexes of an n^3 box is removed and
// created again with the same vertices, so the edited mesh holds the same
// hexes as the compact one, f of them masked and f in the tail:
//
//   * native: connectivity, the vertex-to-hex graph and a per-hex field read
//     and written, on the mesh after MeshEdit::apply, against the compact
//     mesh; the cost of apply and compress against building the box;
//   * compat: tag_get_data and tag_set_data (bulk and one handle at a time),
//     get_connectivity and vertex-to-hex get_adjacencies, on a
//     soma::compat::Core and a moab::Core that made the same calls.
//
// Every query is checksummed against its compact counterpart (and MOAB's);
// a mismatch exits with status 2. Queries run serially, as MOAB does.
//
//   bench_edit [--json] [--repeats N] [--box n] [--fractions f,f,...]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/Core.hpp"
#include "moab_box.hpp"
#include "soma/compat/core.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/build.hpp"

namespace {

namespace fs = std::filesystem;
namespace sc = soma::compat;
using soma::HostMesh;
using soma::LO;
using soma::Real;
using soma::Topology;
using H = moab::EntityHandle;

// Every k-th of n entities, k = round(1 / f); none for f = 0.
std::vector<LO> every_kth(LO n, double f) {
  std::vector<LO> out;
  if (f <= 0) return out;
  const LO k = std::max<LO>(1, static_cast<LO>(std::lround(1.0 / f)));
  for (LO e = 0; e < n; e += k) out.push_back(e);
  return out;
}

std::string label(double f) {
  std::ostringstream s;
  s << f * 100 << "%";
  return s.str();
}

// --- native ---------------------------------------------------------------

HostMesh base_box(int n) {
  HostMesh m = soma::testing::box_mesh({n, n, n, Topology::hex});
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(0, 3));
  m.fields().add<double>("e", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 3), soma::Layout::per_entity()),
                         1, {.default_value = std::vector<double>{1.0}});
  return m;
}

// Records into edit: the hexes `gone` removed and created again.
void churn(HostMesh& m, soma::MeshEdit& edit, const std::vector<LO>& gone) {
  const auto v = m.view();
  std::vector<LO> conn;
  conn.reserve(8 * gone.size());
  for (const LO e : gone) {
    for (LO q = 0; q < 8; ++q) conn.push_back(v.node(3, e, q));
  }
  edit.remove(3, gone);
  edit.add_elements(Topology::hex, conn);
}

std::int64_t conn_sum(const HostMesh& m) {
  const auto v = m.view();
  std::int64_t s = 0;
  for (LO e = 0; e < m.count(3); ++e) {
    if (!v.alive(3, e)) continue;
    for (LO q = 0; q < v.num_vertices(3, e); ++q) s += v.gid(0, v.node(3, e, q));
  }
  return s;
}

std::int64_t graph_sum(const HostMesh& m) {
  const auto g = m.adjacency(0, 3);
  const auto v = m.view();
  std::int64_t s = 0;
  for (LO r = 0; r + 1 < static_cast<LO>(g.offsets.extent(0)); ++r) {
    // Created hexes take new gids, so a row is summed by its vertex.
    for (auto i = g.offsets(r); i < g.offsets(r + 1); ++i) s += v.alive(3, g.targets(i)) ? v.gid(0, r) : 0;
  }
  return s;
}

double field_read(const HostMesh& m) {
  const auto f = m.fields().get<double>("e").view();
  const auto v = m.view();
  double s = 0;
  for (LO e = 0; e < m.count(3); ++e) {
    if (v.alive(3, e)) s += f(static_cast<std::size_t>(e), 0);
  }
  return s;
}

void field_write(HostMesh& m) {
  auto f = m.fields().get<double>("e").view();
  const auto v = m.view();
  for (LO e = 0; e < m.count(3); ++e) {
    if (v.alive(3, e)) f(static_cast<std::size_t>(e), 0) = 2.0;
  }
}

bool run_native(soma::bench::Runner& runner, int n, const std::vector<double>& fractions) {
  const HostMesh base = base_box(n);
  const std::int64_t hexes = base.count(3);
  const std::int64_t verts = base.count(0);
  const std::int64_t ref_conn = conn_sum(base), ref_graph = graph_sum(base);
  bool ok = true;
  const auto expect = [&](const std::string& what, double f, double got, double want) {
    if (got != want) {
      std::cerr << "[bench] native " << what << " at " << label(f) << ": " << got << " != " << want << "\n";
      ok = false;
    }
  };
  const auto none = [] {};
  for (const double f : fractions) {
    const std::string tag = "native_" + label(f);
    HostMesh m = base.clone();
    const std::vector<LO> gone = every_kth(base.count(3), f);
    if (!gone.empty()) {
      soma::MeshEdit edit(m);
      churn(m, edit, gone);
      edit.apply();
    }
    std::int64_t cs = 0, gs = 0;
    double fr = 0;
    runner.run(tag + "_connectivity", "soma", hexes, none, [&] { cs = conn_sum(m); });
    runner.run(tag + "_vertex_to_hex", "soma", verts, none, [&] { gs = graph_sum(m); });
    runner.run(tag + "_field_write", "soma", hexes, none, [&] { field_write(m); });
    runner.run(tag + "_field_read", "soma", hexes, none, [&] { fr = field_read(m); });
    expect("connectivity", f, static_cast<double>(cs), static_cast<double>(ref_conn));
    expect("vertex_to_hex", f, static_cast<double>(gs), static_cast<double>(ref_graph));
    expect("field", f, fr, 2.0 * static_cast<double>(hexes));
    if (gone.empty()) continue;

    HostMesh e;
    std::optional<soma::MeshEdit> edit;
    runner.run(tag + "_apply", "soma", static_cast<std::int64_t>(gone.size()),
               [&] {
                 edit.reset();
                 e = base.clone();
                 edit.emplace(e);
                 churn(e, *edit, gone);
               },
               [&] { edit->apply(); });
    edit.reset();
    runner.run(tag + "_compress", "soma", hexes, [&] { e = m.clone(); }, [&] { e.compress(); });
    expect("compressed connectivity", f, static_cast<double>(conn_sum(e)), static_cast<double>(ref_conn));
  }
  HostMesh b;
  runner.run("native_build", "soma", hexes, none, [&] { b = base_box(n); });
  return ok;
}

// --- compat against MOAB ----------------------------------------------------

// Loads file, removes the hexes `gone` and creates them again with the same
// vertices, and sets a dense double tag "t" to 1 on every hex. Core is
// moab::Core or soma::compat::Core; the calls are the same.
template <class Core, class Range, class Tag, class Code>
void prepare(Core& mb, const std::string& file, const std::vector<LO>& gone, Tag& t, Range& hexes, Code ok,
             auto hex, auto dense, auto creat, auto real) {
  if (mb.load_file(file.c_str()) != ok) throw std::runtime_error("load_file " + file);
  Range all;
  mb.get_entities_by_type(0, hex, all);
  std::vector<H> doomed;
  std::vector<std::vector<H>> conns;
  for (const LO e : gone) {
    const H h = all[e];
    std::vector<H> c;
    mb.get_connectivity(&h, 1, c);
    doomed.push_back(h);
    conns.push_back(c);
  }
  if (!doomed.empty() && mb.delete_entities(doomed.data(), static_cast<int>(doomed.size())) != ok) {
    throw std::runtime_error("delete_entities");
  }
  for (const auto& c : conns) {
    H h = 0;
    if (mb.create_element(hex, c.data(), 8, h) != ok) throw std::runtime_error("create_element");
  }
  const double one = 1.0;
  mb.tag_get_handle("t", 1, real, t, dense | creat, &one);
  mb.get_entities_by_type(0, hex, hexes);
  std::vector<double> ones(hexes.size(), 1.0);
  mb.tag_set_data(t, hexes, ones.data());
}

struct Sums {
  double tag = 0, single = 0;
  std::int64_t conn = 0, adj = 0;
};

template <class Core, class Range, class Tag>
void run_queries(soma::bench::Runner& runner, const std::string& tag_, const std::string& who, Core& mb,
                 Tag t, const Range& hexes, Sums& s, int hex_dim) {
  const auto none = [] {};
  const auto nh = static_cast<std::int64_t>(hexes.size());
  std::vector<double> vals(hexes.size());
  Range verts;
  mb.get_entities_by_dimension(0, 0, verts);
  runner.run(tag_ + "_tag_set_bulk", who, nh, none, [&] {
    std::fill(vals.begin(), vals.end(), 2.0);
    mb.tag_set_data(t, hexes, vals.data());
  });
  runner.run(tag_ + "_tag_get_bulk", who, nh, none, [&] {
    mb.tag_get_data(t, hexes, vals.data());
    s.tag = 0;
    for (const double x : vals) s.tag += x;
  });
  runner.run(tag_ + "_tag_get_single", who, nh, none, [&] {
    s.single = 0;
    for (const H h : hexes) {
      double x = 0;
      mb.tag_get_data(t, &h, 1, &x);
      s.single += x;
    }
  });
  runner.run(tag_ + "_connectivity", who, nh, none, [&] {
    s.conn = 0;
    std::vector<H> c;
    for (const H h : hexes) {
      c.clear();
      mb.get_connectivity(&h, 1, c);
      for (const H v : c) s.conn += static_cast<std::int64_t>(mb.id_from_handle(v));
    }
  });
  runner.run(tag_ + "_vertex_to_hex", who, static_cast<std::int64_t>(verts.size()), none, [&] {
    s.adj = 0;
    std::vector<H> adj;
    for (const H v : verts) {
      adj.clear();
      mb.get_adjacencies(&v, 1, hex_dim, false, adj);
      for (const H h : adj) s.adj += static_cast<std::int64_t>(mb.id_from_handle(h) % 7);
    }
  });
}

bool run_compat(soma::bench::Runner& runner, const std::string& file, int n, const std::vector<double>& fractions) {
  bool ok = true;
  Sums compact{};
  bool have_compact = false;
  for (const double f : fractions) {
    const std::string tag = "compat_" + label(f);
    const std::vector<LO> gone = every_kth(static_cast<LO>(n) * n * n, f);
    Sums sm{}, ss{};
    {
      moab::Core mb;
      moab::Tag t = nullptr;
      moab::Range hexes;
      prepare(mb, file, gone, t, hexes, moab::MB_SUCCESS, moab::MBHEX, moab::MB_TAG_DENSE, moab::MB_TAG_CREAT,
              moab::MB_TYPE_DOUBLE);
      run_queries(runner, tag, "moab", mb, t, hexes, sm, 3);
    }
    {
      sc::Core mb;
      sc::Tag t = nullptr;
      sc::Range hexes;
      prepare(mb, file, gone, t, hexes, sc::MB_SUCCESS, sc::MBHEX, sc::MB_TAG_DENSE, sc::MB_TAG_CREAT,
              sc::MB_TYPE_DOUBLE);
      run_queries(runner, tag, "compat", mb, t, hexes, ss, 3);
    }
    const auto same = [&](const char* what, double a, double b) {
      if (a != b) {
        std::cerr << "[bench] " << tag << " " << what << ": moab " << a << " compat " << b << "\n";
        ok = false;
      }
    };
    same("tag", sm.tag, ss.tag);
    same("single", sm.single, ss.single);
    same("connectivity", static_cast<double>(sm.conn), static_cast<double>(ss.conn));
    same("vertex_to_hex", static_cast<double>(sm.adj), static_cast<double>(ss.adj));
    // The same hexes, with other handles: the totals that do not see handle
    // ids match the compact instance's.
    if (!have_compact) {
      compact = ss;
      have_compact = true;
    } else {
      same("tag vs compact", compact.tag, ss.tag);
      same("connectivity vs compact", static_cast<double>(compact.conn), static_cast<double>(ss.conn));
    }
  }
  return ok;
}

std::vector<double> parse_fractions(const std::string& text) {
  std::vector<double> out;
  std::istringstream in(text);
  for (std::string item; std::getline(in, item, ',');) out.push_back(std::stod(item));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  int repeats = 5;
  int n = 40;
  std::vector<double> fractions{0.0, 0.01, 0.1, 0.5};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--box" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (a == "--fractions" && i + 1 < argc) {
      fractions = parse_fractions(argv[++i]);
    }
  }
  if (fractions.empty() || fractions.front() != 0.0) fractions.insert(fractions.begin(), 0.0);
  const std::string name = "box_" + std::to_string(n);
  const auto env = soma::bench::make_environment(comm, name, repeats);
  if (!json) soma::bench::print_environment(comm, env);
  soma::bench::Runner runner(comm, env, json);

  std::string pattern = (fs::temp_directory_path() / "soma_bench_edit_XXXXXX").string();
  if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
  const fs::path scratch = pattern;
  bool ok = run_native(runner, n, fractions);
  ok = run_compat(runner, soma::bench::write_box(scratch, n).string(), n, fractions) && ok;
  std::error_code ignored;
  fs::remove_all(scratch, ignored);
  return ok ? 0 : 2;
}
