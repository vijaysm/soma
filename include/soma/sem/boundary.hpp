// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SEM_BOUNDARY_HPP_
#define SOMA_SEM_BOUNDARY_HPP_

/// \file
/// The boundary of a spectral-element space: which of its GLL points lie on
/// the mesh's skin (a coast), the unit outward normal there in the tangent
/// plane, and the points where the coast turns too sharply to have one.
///
/// ```
///   const soma::sem::SemBoundary<Mem> b = soma::sem::make_boundary(comm, mesh, space);
///   soma::sem::remove_normal(b, u);   // u: 3 components, continuous; no normal flow
/// ```
///
/// A skin edge is an edge of exactly one element over all ranks: each rank
/// counts its owned elements at each edge and the counts are summed over the
/// ranks sharing the edge. At the GLL points of a skin edge its element gives
/// the edge's outward unit normal, along -a^2, a^1, a^2 or -a^1 for sides
/// 0..3 (SemSpace's corner order), projected to the tangent plane. The
/// normals and a count of skin edges are summed at each point over the
/// elements holding it in dss's order (each scaled by 1 / local mass, dss
/// averaged, then multiplied by the assembled mass), so a shared point's sum
/// is the same, bit for bit, on any number of ranks. Then, per point:
/// ```
///   count 0                                  kind 0, interior, normal 0
///   count 1, or 2 with |n1 + n2| > sqrt(2)   kind 1, coast (slip), normal sum / |sum|
///     + 1e-12 (the edges turn by less than 90 degrees)
///   count 2 otherwise, or 3 or more          kind 2, no-slip, normal 0
/// ```
/// GLL points inside a skin edge have count 1; a corner of the coast where
/// it turns by 90 degrees or more, and a pinch point (two elements meeting at
/// one vertex only, count 4), are no-slip: there the velocity has no
/// direction that stays in the domain. Every element's rows, ghosts
/// included, hold the result.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/sem/sem_space.hpp"

namespace soma::sem {

template <class MemSpace>
struct SemBoundary {
  /// Per row of the space: 0 interior, 1 coast (slip), 2 no-slip.
  View1<std::uint8_t, MemSpace> kind;
  /// Per row, 3 components: the unit outward tangent-plane normal at kind 1,
  /// 0 elsewhere.
  View2<Real, MemSpace> normal;
  /// Distinct points of kind 1 and 2 that this rank owns (on its owned
  /// vertices and edges); the caller sums them over ranks.
  LO coast_points = 0, no_slip_points = 0;
};

/// Collective. The boundary of `space`, which must have been built on `mesh`
/// (as it is now). Throws std::invalid_argument on a rank whose mesh's
/// element count is not the space's, and may leave the others waiting.
template <class MemSpace>
SemBoundary<MemSpace> make_boundary(const Comm& comm, Mesh<MemSpace>& mesh, const SemSpace<MemSpace>& space) {
  if (mesh.count(2) != space.num_elements()) {
    throw std::invalid_argument("make_boundary: the mesh has " + std::to_string(mesh.count(2)) +
                                " elements, the space " + std::to_string(space.num_elements()));
  }
  std::optional<Mesh<Kokkos::HostSpace>> mirror;
  const Mesh<Kokkos::HostSpace>* host = nullptr;
  if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
    host = &mesh;
  } else {
    mirror.emplace(mesh.template mirror<Kokkos::HostSpace>(detail::sem_mirror_options(mesh)));
    host = &*mirror;
  }
  const int p = space.order(), np = p + 1;
  const LO ppe = space.points_per_element(), owned = space.owned_elements();
  const auto rows = static_cast<std::size_t>(space.num_elements()) * static_cast<std::size_t>(ppe);
  const detail::SemDssPlan plan = detail::sem_dss_plan(*host, p);

  // Skin edges: owned elements per edge, summed over the ranks sharing it.
  View2<int, Kokkos::HostSpace> at_edge("soma::sem::make_boundary::at_edge", static_cast<std::size_t>(host->count(1)),
                                        1);
  for (LO e = 0; e < owned; ++e) {
    for (int q = 0; q < 4; ++q) at_edge(plan.element_edge[static_cast<std::size_t>(4 * e + q)], 0) += 1;
  }
  if (comm.size() > 1) HaloExchange<Kokkos::HostSpace>(comm, *host, 1).reduce(comm, at_edge, Op::sum);

  // Each owned element's skin sides: their outward normals and a count of one,
  // over the local mass, at the side's points.
  const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, space.x());
  const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, space.contravariant());
  const auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, space.local_mass());
  const auto mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, space.mass());
  const bool sphere = space.geometry().kind == SemGeometry::Kind::sphere;
  View2<Real, Kokkos::HostSpace> sum("soma::sem::make_boundary::sum", rows, 4);
  const LO first[4] = {0, p, ppe - 1, p * np};
  const LO along[4] = {1, np, -1, -np};
  const int axis[4] = {1, 0, 1, 0};  // a^2, a^1, a^2, a^1
  const Real sign[4] = {-1, 1, 1, -1};
  for (LO e = 0; e < owned; ++e) {
    for (int q = 0; q < 4; ++q) {
      if (at_edge(plan.element_edge[static_cast<std::size_t>(4 * e + q)], 0) != 1) continue;
      for (int t = 0; t <= p; ++t) {
        const LO r = e * ppe + first[q] + along[q] * t;
        Real n[3];
        for (int k = 0; k < 3; ++k) n[k] = sign[q] * b(r, 3 * axis[q] + k);
        if (sphere) {
          const Real len = std::sqrt(x(r, 0) * x(r, 0) + x(r, 1) * x(r, 1) + x(r, 2) * x(r, 2));
          const Real dot = (n[0] * x(r, 0) + n[1] * x(r, 1) + n[2] * x(r, 2)) / len;
          for (int k = 0; k < 3; ++k) n[k] -= dot * x(r, k) / len;
        } else {
          n[2] = 0;
        }
        const Real len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) sum(r, k) += n[k] / len / m(r, 0);
        sum(r, 3) += 1 / m(r, 0);
      }
    }
  }
  const auto summed = Kokkos::create_mirror_view_and_copy(MemSpace{}, sum);
  space.dss(comm, summed);
  Kokkos::deep_copy(sum, summed);

  View1<std::uint8_t, Kokkos::HostSpace> kind("soma::sem::make_boundary::kind", rows);
  View2<Real, Kokkos::HostSpace> normal("soma::sem::make_boundary::normal", rows, 3);
  const Real sharp = std::sqrt(Real{2}) + 1e-12;
  for (std::size_t r = 0; r < rows; ++r) {
    Real s[3];
    for (int k = 0; k < 3; ++k) s[k] = sum(r, k) * mass(r, 0);
    const long count = std::lround(sum(r, 3) * mass(r, 0));
    const Real len = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (count == 0) continue;
    if (count == 1 || (count == 2 && len > sharp)) {
      kind(r) = 1;
      for (int k = 0; k < 3; ++k) normal(r, k) = s[k] / len;
    } else {
      kind(r) = 2;
    }
  }

  SemBoundary<MemSpace> out;
  // Distinct owned points: one row per owned vertex and per owned edge point.
  const auto tally = [&](LO r) {
    if (kind(static_cast<std::size_t>(r)) == 1) ++out.coast_points;
    if (kind(static_cast<std::size_t>(r)) == 2) ++out.no_slip_points;
  };
  for (std::size_t v = 0; v + 1 < plan.vertex_offsets.size(); ++v) {
    if (plan.vertex_offsets[v + 1] > plan.vertex_offsets[v]) {
      tally(plan.vertex_rows[static_cast<std::size_t>(plan.vertex_offsets[v])]);
    }
  }
  for (std::size_t g = 0; g + 1 < plan.edge_offsets.size(); ++g) {
    if (plan.edge_offsets[g + 1] == plan.edge_offsets[g]) continue;
    const auto t = static_cast<std::size_t>(plan.edge_offsets[g]);
    for (int s = 1; s < p; ++s) tally(plan.edge_base[t] + plan.edge_step[t] * s);
  }
  out.kind = Kokkos::create_mirror_view_and_copy(MemSpace{}, kind);
  out.normal = Kokkos::create_mirror_view_and_copy(MemSpace{}, normal);
  return out;
}

/// u: 3 components, a row per row of the boundary's space. Removes the
/// normal component at kind 1 points and zeroes u at kind 2 ones. Local, no
/// communication: continuous input stays continuous. Throws
/// std::invalid_argument for a row count other than the boundary's or fewer
/// than 3 components.
template <class MemSpace>
void remove_normal(const SemBoundary<MemSpace>& boundary, const View2<Real, MemSpace>& u) {
  if (u.extent(0) != boundary.kind.extent(0) || u.extent(1) < 3) {
    throw std::invalid_argument("remove_normal: u has " + std::to_string(u.extent(0)) + " rows of " +
                                std::to_string(u.extent(1)) + " components, not " +
                                std::to_string(boundary.kind.extent(0)) + " of 3");
  }
  using Exec = typename MemSpace::execution_space;
  const auto kind = boundary.kind;
  const auto n = boundary.normal;
  Kokkos::parallel_for(
      "soma::sem::remove_normal", Kokkos::RangePolicy<Exec>(0, u.extent(0)), KOKKOS_LAMBDA(const std::size_t r) {
        if (kind(r) == 1) {
          const Real d = u(r, 0) * n(r, 0) + u(r, 1) * n(r, 1) + u(r, 2) * n(r, 2);
          for (int k = 0; k < 3; ++k) u(r, k) -= d * n(r, k);
        } else if (kind(r) == 2) {
          for (int k = 0; k < 3; ++k) u(r, k) = 0;
        }
      });
  Exec{}.fence();
}

}  // namespace soma::sem

#endif  // SOMA_SEM_BOUNDARY_HPP_
