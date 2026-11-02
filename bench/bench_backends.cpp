// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// SOMA against itself across the execution spaces of the build: Serial, the
// host parallel backend at the process's thread count, and the device. The
// meshes are generated boxes in curve order, so nothing here needs MOAB or a
// file. Every workflow checksums its result in the space it ran in; the
// spaces must agree or the process exits with status 2.
//
//   bench_backends [--json] [--repeats N] [--hex N...] [--quad N...] [--tri N...] [--no-serial]
//
// Copies (mirror_in, mirror_out) are attributed to a memory space, not an
// execution space: Mesh::mirror takes none, and a host copy runs on the
// default host backend whatever the caller intended.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/core/transfer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/build.hpp"

namespace soma::bench {

// A power-of-two scaling is exact, so the truncation is the same function of
// the bits in every space and the integer sums are order-independent.
KOKKOS_INLINE_FUNCTION std::int64_t fixed(double x) { return static_cast<std::int64_t>(x * 4096.0); }

template <class Exec>
std::int64_t gid_sum(const Exec& exec, MeshView<typename Exec::memory_space> v, int dim) {
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::bench::gid_sum", Kokkos::RangePolicy<Exec>(exec, 0, v.count(dim)),
      KOKKOS_LAMBDA(const LO e, std::int64_t& acc) { acc += static_cast<std::int64_t>(v.gid(dim, e)); },
      total);
  exec.fence();
  return total;
}

// Checksum of the pair (a, b), or of the bridge (a via b): every target
// weighted by its row, plus the row lengths, so a target in the wrong row or
// a row of the wrong length changes it.
template <class Exec>
std::int64_t target_sum(const Exec& exec, MeshView<typename Exec::memory_space> v, int a, int b,
                        bool bridge) {
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::bench::target_sum", Kokkos::RangePolicy<Exec>(exec, 0, v.count(a)),
      KOKKOS_LAMBDA(const LO e, std::int64_t& acc) {
        const Row<const LO> row = bridge ? v.bridge(a, b, e) : v.adj(a, b, e);
        acc += row.size;
        for (LO k = 0; k < row.size; ++k) acc += (static_cast<std::int64_t>(e) + 1) * row[k];
      },
      total);
  exec.fence();
  return total;
}

// Every derived dimension and every (D, k) graph of a derivation; a graph
// that was not built counts as -1 instead of aborting on an empty view.
template <class Exec>
std::int64_t derive_sum(const Exec& exec, const Mesh<typename Exec::memory_space>& m) {
  const int D = m.dim();
  std::int64_t total = 0;
  for (int k = 1; k < D; ++k) {
    total += 1000003 * static_cast<std::int64_t>(m.count(k));
    total += gid_sum(exec, m.view(), k);
    total += m.has_adjacency(D, k) ? target_sum(exec, m.view(), D, k, false) : -1;
  }
  return total;
}

// For each element, the coordinates of its vertices: the gather every
// element kernel starts with.
template <class Exec>
std::int64_t traverse(const Exec& exec, MeshView<typename Exec::memory_space> v) {
  const int dim = v.dim;
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::bench::traverse", Kokkos::RangePolicy<Exec>(exec, 0, v.count(dim)),
      KOKKOS_LAMBDA(const LO e, std::int64_t& acc) {
        for (LO k = 0; k < v.num_vertices(dim, e); ++k) {
          const LO vtx = v.vertex(dim, e, k);
          for (int c = 0; c < 3; ++c) acc += fixed(v.x(vtx, c));
        }
      },
      total);
  exec.fence();
  return total;
}

// For each vertex, the global ids of the elements around it: an upward walk
// through the (0, D) graph.
template <class Exec>
std::int64_t walk(const Exec& exec, MeshView<typename Exec::memory_space> v) {
  const int dim = v.dim;
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::bench::walk", Kokkos::RangePolicy<Exec>(exec, 0, v.count(0)),
      KOKKOS_LAMBDA(const LO vtx, std::int64_t& acc) {
        const Row<const LO> row = v.adj(0, dim, vtx);
        for (LO k = 0; k < row.size; ++k) acc += static_cast<std::int64_t>(v.gid(dim, row[k]));
      },
      total);
  exec.fence();
  return total;
}

struct Result {
  std::string workflow;
  std::string space;
  std::int64_t checksum_a = 0;
  std::int64_t checksum_b = 0;
  Timing timing;
};

AdjacencyRequest downward(int dim) {
  AdjacencyRequest r;
  for (int k = 1; k < dim; ++k) r.add(dim, k);
  return r;
}

// Runs every workflow in exec; the copy workflows only when `copies` is set
// (once per memory space).
template <class Exec>
void run_space(Runner& runner, const HostMesh& host, const Exec& exec, bool copies,
               std::vector<Result>& results) {
  using M = typename Exec::memory_space;
  const std::string space = Exec::name();
  // Kokkos names memory spaces like their execution spaces ("Host", "Cuda").
  const std::string memory = std::string(M::name()) + "Space";
  const int D = host.dim();
  const std::int64_t nv = host.count(0);
  const std::int64_t ne = host.count(D);
  Mesh<M> m;
  const auto lean = [&] { m = host.mirror<M>(); };
  const auto build = [&](AdjacencyRequest req) {
    return [&, req] {
      build_adjacencies(m, req, exec);
      exec.fence();
    };
  };

  if (copies) {
    Result r{"mirror_in", memory, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, nv + ne, [] {}, [&] {
      lean();
      exec.fence();
    });
    r.checksum_a = gid_sum(exec, m.view(), D);
    results.push_back(r);
  }
  {
    Result r{"derive", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, ne, lean, build(downward(D)));
    r.checksum_a = m.count(D - 1);
    r.checksum_b = derive_sum(exec, m);
    results.push_back(r);
  }
  {
    Result r{"vertex_to_element", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, nv, lean,
                          build(AdjacencyRequest{}.add(0, D)));
    r.checksum_a = target_sum(exec, m.view(), 0, D, false);
    results.push_back(r);
  }
  {
    Result r{"vertex_via_element", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, nv, lean,
                          build(AdjacencyRequest{}.add_bridge(0, D)));
    r.checksum_a = target_sum(exec, m.view(), 0, D, true);
    results.push_back(r);
  }
  {
    Result r{"element_via_vertex", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, ne, lean,
                          build(AdjacencyRequest{}.add_bridge(D, 0)));
    r.checksum_a = target_sum(exec, m.view(), D, 0, true);
    results.push_back(r);
  }
  // The kernels read a mesh that holds (0, D); it is built once, untimed.
  lean();
  build(AdjacencyRequest{}.add(0, D))();
  {
    Result r{"traverse", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, ne, [] {},
                          [&] { r.checksum_a = traverse(exec, m.view()); });
    results.push_back(r);
  }
  {
    Result r{"walk", space, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, nv, [] {},
                          [&] { r.checksum_a = walk(exec, m.view()); });
    results.push_back(r);
  }
  if (copies) {
    // Every graph the workflows above built, copied back with the topology.
    build(downward(D) | AdjacencyRequest{}.add_bridge(0, D).add_bridge(D, 0))();
    TransferOptions back = TransferOptions::everything();
    back.adjacencies = m.adjacencies();
    back.adj_source = TransferOptions::AdjSource::copy;
    HostMesh h;
    Result r{"mirror_out", memory, 0, 0, {}};
    r.timing = runner.run("backends", r.workflow, r.space, nv + ne, [] {}, [&] {
      h = m.template mirror<Kokkos::HostSpace>(back);
      Kokkos::fence();
    });
    r.checksum_a = gid_sum(Kokkos::DefaultHostExecutionSpace{}, h.view(), D);
    r.checksum_b = static_cast<std::int64_t>(h.adjacencies().bits());
    results.push_back(r);
  }
}

// Returns false when two spaces disagree on a checksum.
bool report(const Comm& comm, const std::string& mesh, const std::vector<Result>& results,
            const std::string& base, Runner& runner, bool json) {
  bool ok = true;
  std::map<std::string, std::vector<const Result*>> by_workflow;
  std::vector<std::string> order;
  for (const Result& r : results) {
    if (by_workflow[r.workflow].empty()) order.push_back(r.workflow);
    by_workflow[r.workflow].push_back(&r);
  }
  for (const std::string& w : order) {
    const auto& rows = by_workflow[w];
    const Result* ref = nullptr;
    for (const Result* r : rows) {
      if (r->space == base) ref = r;
    }
    if (!json && ref != nullptr) {
      for (const Result* r : rows) {
        if (r != ref) runner.compare(ref->timing, r->timing, base + "/" + r->space);
      }
    }
    for (const Result* r : rows) {
      const Result* first = rows.front();
      if (r->checksum_a != first->checksum_a || r->checksum_b != first->checksum_b) {
        ok = false;
        if (comm.rank() == 0) {
          std::cerr << "[bench] " << mesh << " " << w << ": checksum mismatch, " << first->space
                    << " " << first->checksum_a << "/" << first->checksum_b << " " << r->space
                    << " " << r->checksum_a << "/" << r->checksum_b << "\n";
        }
      }
    }
  }
  return ok;
}

bool run_mesh(const Comm& comm, const std::string& name, const HostMesh& host, int repeats,
              bool json, bool no_serial) {
  Runner runner(comm, make_environment(comm, name, repeats), json);
  if (!json) print_environment(comm, make_environment(comm, name, repeats));
  std::vector<Result> results;
  using HostExec = Kokkos::DefaultHostExecutionSpace;
  using DevExec = Kokkos::DefaultExecutionSpace;
  std::string base = HostExec::name();
  bool serial_ran = false;
#ifdef KOKKOS_ENABLE_SERIAL
  constexpr bool serial_is_host = std::is_same_v<HostExec, Kokkos::Serial>;
  if (!no_serial) {
    run_space(runner, host, Kokkos::Serial{}, serial_is_host, results);
    base = Kokkos::Serial::name();
    serial_ran = true;
  }
#else
  constexpr bool serial_is_host = false;
  (void)no_serial;
#endif
  // The host backend row is the Serial one when Serial is the host backend.
  if (!serial_is_host || !serial_ran) run_space(runner, host, HostExec{}, true, results);
  if constexpr (!std::is_same_v<DevExec, HostExec>) {
    run_space(runner, host, DevExec{}, true, results);
  }
  return report(comm, name, results, base, runner, json);
}

struct Box {
  std::string name;
  testing::BoxSpec spec;
};

}  // namespace soma::bench

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  bool no_serial = false;
  int repeats = 5;
  std::vector<int> hex, quad, tri;
  std::vector<int>* sizes = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
      sizes = nullptr;
    } else if (a == "--no-serial") {
      no_serial = true;
      sizes = nullptr;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
      sizes = nullptr;
    } else if (a == "--hex") {
      sizes = &hex;
    } else if (a == "--quad") {
      sizes = &quad;
    } else if (a == "--tri") {
      sizes = &tri;
    } else if (sizes != nullptr && std::atoi(a.c_str()) > 0) {
      sizes->push_back(std::atoi(a.c_str()));
    } else if (a.rfind("--kokkos-", 0) != 0) {
      if (comm.rank() == 0) std::cerr << "bench_backends: unknown argument " << a << "\n";
      return 1;
    }
  }
  if (hex.empty() && quad.empty() && tri.empty()) {
    hex = {32, 64};
    quad = {512, 1024};
    tri = {512};
  }
  std::vector<soma::bench::Box> boxes;
  const auto add = [&](const std::vector<int>& ns, soma::Topology t, const char* label) {
    for (const int n : ns) {
      soma::testing::BoxSpec s;
      s.nx = n;
      s.ny = n;
      s.nz = t == soma::Topology::hex ? n : 1;
      s.element = t;
      s.ordering = soma::Ordering::type_major_sfc;
      boxes.push_back({std::string(label) + "_" + std::to_string(n), s});
    }
  };
  add(hex, soma::Topology::hex, "hex");
  add(quad, soma::Topology::quad, "quad");
  add(tri, soma::Topology::tri, "tri");

  bool ok = true;
  for (const auto& b : boxes) {
    const soma::HostMesh host = soma::testing::box_mesh(b.spec);
    ok = soma::bench::run_mesh(comm, b.name, host, repeats, json, no_serial) && ok;
  }
  return ok ? 0 : 2;
}
