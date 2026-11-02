// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_MIRROR_CHECK_HPP_
#define SOMA_TESTS_MIRROR_CHECK_HPP_

// Helpers for comparing a mesh with its mirror in another memory space:
// every graph as host vectors, and checksums computed by kernels in the
// mesh's own execution space, so a device mirror is read where it lives.

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma_test {

// Every pair and bridge graph between dimensions 0..dim.
inline soma::AdjacencyRequest every_graph(int dim) {
  soma::AdjacencyRequest r;
  for (int a = 0; a <= dim; ++a) {
    for (int b = 0; b <= dim; ++b) {
      if (a == b) continue;
      if (b != 0) r.add(a, b);
      r.add_bridge(a, b);
    }
  }
  return r;
}

template <class M>
std::vector<HostGraph> all_graphs(const soma::Mesh<M>& mesh) {
  std::vector<HostGraph> out;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      if (mesh.has_adjacency(a, b)) out.push_back(to_host(mesh.adjacency(a, b)));
      if (mesh.has_bridge(a, b)) out.push_back(to_host(mesh.bridge(a, b)));
    }
  }
  return out;
}

// Scaling by a power of two is exact, so the truncation is the same function
// of the bits in every execution space and the sums are order-independent.
KOKKOS_INLINE_FUNCTION std::int64_t fixed(double x) {
  return static_cast<std::int64_t>(x * 4096.0);
}

struct MeshChecksum {
  std::array<std::int64_t, 4> gids{};
  std::int64_t coords = 0;
  std::array<std::int64_t, soma::adjacency_slots> targets{};
  std::array<std::int64_t, soma::adjacency_slots> rows{};
  friend bool operator==(const MeshChecksum&, const MeshChecksum&) = default;
};

template <class M>
std::int64_t gid_sum(soma::MeshView<M> v, int dim) {
  using Exec = typename M::execution_space;
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::test::gid_sum", Kokkos::RangePolicy<Exec>(0, v.count(dim)),
      KOKKOS_LAMBDA(const soma::LO e, std::int64_t& acc) {
        acc += static_cast<std::int64_t>(v.gid(dim, e));
      },
      total);
  return total;
}

template <class M>
std::int64_t coordinate_sum(soma::MeshView<M> v) {
  using Exec = typename M::execution_space;
  const int dim = v.dim;
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::test::coordinate_sum", Kokkos::RangePolicy<Exec>(0, v.count(dim)),
      KOKKOS_LAMBDA(const soma::LO e, std::int64_t& acc) {
        for (soma::LO k = 0; k < v.num_vertices(dim, e); ++k) {
          const soma::LO vtx = v.vertex(dim, e, k);
          for (int c = 0; c < 3; ++c) acc += fixed(v.x(vtx, c));
        }
      },
      total);
  return total;
}

// Sum of the targets and of the row lengths of one held graph: the pair
// (a, b) when bridge is false, the bridge (a via b) otherwise.
template <class M>
std::array<std::int64_t, 2> graph_sums(soma::MeshView<M> v, int a, int b, bool bridge,
                                       soma::LO rows) {
  using Exec = typename M::execution_space;
  std::int64_t targets = 0;
  std::int64_t lengths = 0;
  Kokkos::parallel_reduce(
      "soma::test::graph_sums", Kokkos::RangePolicy<Exec>(0, rows),
      KOKKOS_LAMBDA(const soma::LO e, std::int64_t& t, std::int64_t& l) {
        const soma::Row<const soma::LO> row = bridge ? v.bridge(a, b, e) : v.adj(a, b, e);
        l += row.size;
        for (soma::LO k = 0; k < row.size; ++k) t += row[k];
      },
      targets, lengths);
  return {targets, lengths};
}

template <class M>
MeshChecksum mesh_checksum(const soma::Mesh<M>& mesh) {
  MeshChecksum c;
  const auto v = mesh.view();
  for (int d = 0; d <= mesh.dim(); ++d) c.gids[static_cast<std::size_t>(d)] = gid_sum(v, d);
  c.coords = coordinate_sum(v);
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      if (b != 0 && mesh.has_adjacency(a, b)) {
        const int s = soma::pair_slot(a, b);
        const auto sums = graph_sums(v, a, b, false, mesh.count(a));
        c.targets[static_cast<std::size_t>(s)] = sums[0];
        c.rows[static_cast<std::size_t>(s)] = sums[1];
      }
      if (mesh.has_bridge(a, b)) {
        const int s = soma::bridge_slot(a, b);
        const auto sums = graph_sums(v, a, b, true, mesh.count(a));
        c.targets[static_cast<std::size_t>(s)] = sums[0];
        c.rows[static_cast<std::size_t>(s)] = sums[1];
      }
    }
  }
  return c;
}

template <class T>
KOKKOS_INLINE_FUNCTION std::int64_t as_checksum(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return fixed(static_cast<double>(x));
  } else {
    return static_cast<std::int64_t>(x);
  }
}

// Sum of a field's values, fixed-point for floating types, in M's space.
template <class T, class M>
std::int64_t field_checksum(soma::View2<T, M> values) {
  using Exec = typename M::execution_space;
  const int ncomp = static_cast<int>(values.extent(1));
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::test::field_checksum", Kokkos::RangePolicy<Exec>(0, values.extent(0)),
      KOKKOS_LAMBDA(const soma::LO r, std::int64_t& acc) {
        for (int c = 0; c < ncomp; ++c) acc += as_checksum(values(r, c));
      },
      total);
  return total;
}

// Three committed tags whose values depend on the local id: "velocity"
// (double, 3 components, vertices), "material" (int32, elements) and
// "flux" (double, the sides one dimension below the elements).
inline void add_fixture_fields(soma::HostMesh& m) {
  const int dim = m.dim();
  auto velocity = m.fields().add_tag<double>("velocity", 0, 3);
  for (soma::LO r = 0; r < velocity.rows(); ++r) {
    for (int c = 0; c < 3; ++c) velocity.view()(r, c) = 0.5 * r + c;
  }
  auto material = m.fields().add_tag<std::int32_t>("material", dim, 1);
  for (soma::LO r = 0; r < material.rows(); ++r) material.view()(r, 0) = r % 7;
  auto flux = m.fields().add_tag<double>("flux", dim - 1, 1);
  for (soma::LO r = 0; r < flux.rows(); ++r) flux.view()(r, 0) = 1.0 / (r + 1);
}

template <class M>
std::vector<std::string> field_names(const soma::Mesh<M>& m) {
  std::vector<std::string> names;
  for (const auto& info : m.fields().list()) names.push_back(info.name);
  std::sort(names.begin(), names.end());
  return names;
}

template <class M>
std::vector<double> coordinates(const soma::Mesh<M>& m) {
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.coords());
  std::vector<double> out;
  for (std::size_t v = 0; v < h.extent(0); ++v) {
    for (int c = 0; c < 3; ++c) out.push_back(h(v, c));
  }
  return out;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_MIRROR_CHECK_HPP_
