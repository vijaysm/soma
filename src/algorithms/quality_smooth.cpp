// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/quality_smooth.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "soma/core/view_types.hpp"
#include "soma/parallel/halo_exchange.hpp"

namespace soma::detail {

namespace {

// splitmix64: a vertex's priority from its gid alone.
std::uint64_t priority(GO gid) {
  std::uint64_t z = static_cast<std::uint64_t>(gid) + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

std::vector<std::int32_t> vertex_colours(const Comm& comm, const HostMesh& mesh, const VertexCells& cells) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto gid = mesh.gid(0);
  const LO n = mesh.count(0);
  const LO owned = mesh.segment_table(0).owned_end;
  const View2<std::int32_t, Kokkos::HostSpace> colour("soma::vertex_colours", static_cast<std::size_t>(n), 1);
  Kokkos::deep_copy(colour, -1);
  std::optional<HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, mesh, 0);
  // (priority, gid): a total order on the vertices.
  const auto beats = [&](LO a, LO b) {
    const std::uint64_t pa = priority(gid(a)), pb = priority(gid(b));
    return pa != pb ? pa > pb : gid(a) > gid(b);
  };
  std::vector<LO> winners;
  std::vector<char> taken;
  Real before = -1;
  for (;;) {
    Real left = 0;
    winners.clear();
    for (LO v = 0; v < owned; ++v) {
      if (colour(static_cast<std::size_t>(v), 0) >= 0) continue;
      left += 1;
      bool wins = true;
      const auto i0 = static_cast<std::size_t>(v);
      const Offset first = cells.offsets[i0], last = cells.offsets[i0 + 1];
      for (Offset k = first; wins && k < last; ++k) {
        const LO c = cells.cells[static_cast<std::size_t>(k)];
        for (LO i = 0; i < m.num_vertices(dim, c); ++i) {
          const LO u = m.vertex(dim, c, i);
          if (u != v && colour(static_cast<std::size_t>(u), 0) < 0 && beats(u, v)) {
            wins = false;
            break;
          }
        }
      }
      if (wins) winners.push_back(v);
    }
    const Real all = comm.allreduce(left, Op::sum);
    if (all == 0) break;
    // The vertex of highest priority left always wins, unless ranks disagree.
    if (all == before) throw std::logic_error("soma::detail::vertex_colours: a round coloured no vertex");
    before = all;
    // Winners share no cell, so each picks from colours fixed before the round.
    for (const LO v : winners) {
      taken.assign(taken.size(), 0);
      for (Offset k = cells.offsets[static_cast<std::size_t>(v)]; k < cells.offsets[static_cast<std::size_t>(v) + 1];
           ++k) {
        const LO c = cells.cells[static_cast<std::size_t>(k)];
        for (LO i = 0; i < m.num_vertices(dim, c); ++i) {
          const std::int32_t h = colour(static_cast<std::size_t>(m.vertex(dim, c, i)), 0);
          if (h < 0) continue;
          if (static_cast<std::size_t>(h) >= taken.size()) taken.resize(static_cast<std::size_t>(h) + 1, 0);
          taken[static_cast<std::size_t>(h)] = 1;
        }
      }
      std::int32_t least = 0;
      while (static_cast<std::size_t>(least) < taken.size() && taken[static_cast<std::size_t>(least)] != 0) ++least;
      colour(static_cast<std::size_t>(v), 0) = least;
    }
    if (halo) halo->update(comm, colour);
  }
  return {colour.data(), colour.data() + n};
}

}  // namespace soma::detail
