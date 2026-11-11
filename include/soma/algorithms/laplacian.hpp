// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_LAPLACIAN_HPP_
#define SOMA_ALGORITHMS_LAPLACIAN_HPP_

// Laplacian smoothing of vertex coordinates: Jacobi sweeps that move each
// free vertex to the mean of its neighbours' coordinates before the sweep,
// as MOAB's LaplacianSmoother example does.
//
//   mesh.add_ghosts(comm, soma::GhostOptions{1, 0});  // on more than one rank
//   soma::SmoothOptions o;
//   o.iterations = 10;
//   const soma::SmoothReport r = soma::laplacian_smooth(comm, mesh, o);
//
// The neighbours of a vertex are the other corners of the entities of
// dimension bridge_dim around it: the cells by default (all 26 around an
// interior hex vertex), 1 for edges. They are read off the cells' canonical
// sides, so the mesh need not hold those entities. Each vertex sums its
// neighbours in ascending gid order, so the result is the same, bit for bit,
// on any number of ranks and threads.
//
// Fixed vertices stay where they are: those of the mask, else the skin, the
// corners of the sides of dimension dim - 1 that only one cell has. On more
// than one rank the mesh must hold a ghost layer through vertices
// (GhostOptions{1, 0}): every cell around a held vertex is then held, so a
// vertex's neighbours are all here and the skin found here is the global
// one. Owners move their vertices; a halo update then gives copies and
// ghosts their owners' coordinates after every sweep.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/halo_exchange.hpp"

namespace soma {

struct SmoothOptions {
  int iterations = 1;
  // Vertices are neighbours through the entities of this dimension that
  // hold both: -1 the mesh's dimension (the cells), else 1 to dim.
  int bridge_dim = -1;
  // > 0: stop once no vertex moved farther than this in a sweep.
  Real tolerance = 0;
  // Per vertex lid, nonzero for a vertex that stays; empty: the skin stays.
  std::span<const std::uint8_t> fixed;
};

struct SmoothReport {
  int iterations = 0;     // sweeps made
  Real largest_move = 0;  // in the last sweep, over every rank; 0 unless tolerance > 0
};

namespace detail {

// The neighbour rows and the skin of a host mesh, built once per call.
struct SmoothingGraph {
  std::vector<Offset> offsets;     // vertex count + 1
  std::vector<LO> targets;         // each row ascending by gid
  std::vector<std::uint8_t> skin;  // per vertex: a corner of a side only one cell has
};

// bridge_dim in [1, mesh.dim()]; the mesh not edited. Host only.
SmoothingGraph smoothing_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim);

}  // namespace detail

// Collective. Throws std::invalid_argument on every rank if any rank passes
// a negative iteration count, a bridge_dim other than -1 or in [1, dim], a
// tolerance that is negative or not finite, a mask whose length is not the
// vertex count, a mesh of dimension 0 or one with unremoved edits, or, on
// more than one rank, a mesh without a ghost layer through vertices; or if
// ranks differ in iterations, bridge_dim or tolerance.
template <class MemSpace>
SmoothReport laplacian_smooth(const Comm& comm, Mesh<MemSpace>& mesh, const SmoothOptions& options = {}) {
  using Exec = typename MemSpace::execution_space;
  const int dim = mesh.dim();
  const LO n = mesh.count(0);
  const bool finite = std::isfinite(options.tolerance);
  const bool bad =
      options.iterations < 0 || (options.bridge_dim != -1 && (options.bridge_dim < 1 || options.bridge_dim > dim)) ||
      !finite || options.tolerance < 0 ||
      (!options.fixed.empty() && options.fixed.size() != static_cast<std::size_t>(n)) || dim < 1 || mesh.edited() ||
      (comm.size() > 1 && (mesh.ghost_layers() < 1 || mesh.ghost_bridge_dim() != 0));
  // One reduction: the flag, then each option and its negative, which
  // agree only if every rank passed the same.
  const Real tolerance = finite ? options.tolerance : 0;
  const auto it = static_cast<Real>(options.iterations), b = static_cast<Real>(options.bridge_dim);
  std::array<Real, 7> f{bad ? Real{1} : Real{0}, it, -it, b, -b, tolerance, -tolerance};
  comm.allreduce(std::span<Real>(f), Op::max);
  if (f[0] != 0 || f[1] != -f[2] || f[3] != -f[4] || f[5] != -f[6]) {
    throw std::invalid_argument(
        "laplacian_smooth: a rank passes a negative iteration count, a bridge_dim outside -1 and [1, dim], a "
        "negative or non-finite tolerance, a fixed mask not one per vertex, a mesh of dimension 0, with edits or, "
        "on more than one rank, without a ghost layer through vertices; or ranks differ in their options");
  }

  const int bridge_dim = options.bridge_dim == -1 ? dim : options.bridge_dim;
  const detail::SmoothingGraph g = [&] {
    if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      return detail::smoothing_graph(mesh, bridge_dim);
    } else {
      return detail::smoothing_graph(mesh.template mirror<Kokkos::HostSpace>(), bridge_dim);
    }
  }();
  // Owners move their vertices that are free and have neighbours.
  const LO owned = mesh.segment_table(0).owned_end;
  using HostU8 = Kokkos::View<std::uint8_t*, Kokkos::HostSpace>;
  HostU8 moving_host("soma::laplacian_smooth::moving", static_cast<std::size_t>(n));
  for (LO v = 0; v < owned; ++v) {
    const auto i = static_cast<std::size_t>(v);
    const bool fixed = options.fixed.empty() ? g.skin[i] != 0 : options.fixed[i] != 0;
    moving_host(i) = !fixed && g.offsets[i + 1] > g.offsets[i] ? 1 : 0;
  }
  const auto offsets = Kokkos::create_mirror_view_and_copy(
      MemSpace{}, Kokkos::View<const Offset*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                      g.offsets.data(), g.offsets.size()));
  const auto targets = Kokkos::create_mirror_view_and_copy(
      MemSpace{}, Kokkos::View<const LO*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                      g.targets.data(), g.targets.size()));
  const auto moving = Kokkos::create_mirror_view_and_copy(MemSpace{}, moving_host);

  const CoordView<MemSpace> x = mesh.coords();
  const CoordView<MemSpace> old(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::laplacian_smooth::old"),
                                static_cast<std::size_t>(n));
  std::optional<HaloExchange<MemSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, mesh, 0);
  const View2<Real, MemSpace> values(x.data(), static_cast<std::size_t>(n), 3);
  using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>;
  SmoothReport report;
  for (int sweep = 0; sweep < options.iterations; ++sweep) {
    Kokkos::deep_copy(Exec{}, old, x);
    Kokkos::parallel_for(
        "soma::laplacian_smooth::sweep", Policy(Exec{}, 0, owned), KOKKOS_LAMBDA(const LO v) {
          if (moving(v) == 0) return;
          Real sum[3] = {0, 0, 0};
          for (Offset k = offsets(v); k < offsets(v + 1); ++k) {
            const LO w = targets(k);
            for (int q = 0; q < 3; ++q) sum[q] += old(w, q);
          }
          const auto count = static_cast<Real>(offsets(v + 1) - offsets(v));
          for (int q = 0; q < 3; ++q) x(v, q) = sum[q] / count;
        });
    Exec{}.fence();
    ++report.iterations;
    if (halo) halo->update(comm, values);
    if (tolerance > 0) {
      Real move = 0;
      Kokkos::parallel_reduce(
          "soma::laplacian_smooth::move", Policy(Exec{}, 0, owned),
          KOKKOS_LAMBDA(const LO v, Real& most) {
            Real d2 = 0;
            for (int q = 0; q < 3; ++q) d2 += (x(v, q) - old(v, q)) * (x(v, q) - old(v, q));
            most = Kokkos::max(most, Kokkos::sqrt(d2));
          },
          Kokkos::Max<Real>(move));
      report.largest_move = comm.allreduce(move, Op::max);
      if (report.largest_move <= tolerance) break;
    }
  }
  return report;
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_LAPLACIAN_HPP_
