// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SEM_SEM_SPACE_HPP_
#define SOMA_SEM_SEM_SPACE_HPP_

/// \file
/// The spectral-element space of order p over a mesh of quads, in the plane or
/// on a sphere: (p + 1)^2 Gauss-Lobatto-Legendre (GLL) points per element,
/// each element storing its own values (HOMME's layout), and the geometry at
/// every point.
///
/// ```
///   soma::sem::CubedSphereSpec spec;
///   spec.ne = 30;
///   spec.radius = 6.371e6;
///   soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);
///   const soma::sem::SemSpace<Mem> space(comm, mesh, 3, soma::sem::SemGeometry::sphere(6.371e6));
///   auto h = mesh.fields().add<double>("h", space.space(), 1);   // a value per GLL point
/// ```
///
/// Point k = i + (p + 1) j of element e is row e (p + 1)^2 + k of space(),
/// with i along the element's edge from corner 0 to corner 1 and j along the
/// edge from corner 0 to corner 3, at reference coordinates (xi_i, eta_j) of
/// soma::gll_points_1d. The element map is bilinear in the corners or, when
/// the mesh carries 8 or 9 nodes per quad, biquadratic in them (Quad8
/// serendipity, Quad9 Lagrange; MOAB's node order). On a sphere each mapped
/// point y is projected radially, x = R y / |y|, and the tangent vectors are
/// the derivatives of the projected map. Vectors are stored as their three
/// Cartesian components:
/// ```
///   covariant      a_1 = dx/dxi, a_2 = dx/deta              (6 components)
///   contravariant  a^1, a^2: tangent, a^i . a_j = delta_ij  (6 components)
///   jacobian       |a_1 x a_2|
///   local_mass     w_i w_j |a_1 x a_2|, the element's quadrature weight
/// ```
/// Every element on the rank has its geometry, ghosts included; the owned
/// ones are the first owned_elements(). A space keeps views into the mesh: it
/// is valid while the mesh is alive and its topology unchanged.

#include <cstdint>
#include <optional>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::sem {

/// Where the elements lie: the plane z = 0 seen from +z, or the sphere of a
/// radius centred at the origin, seen from outside.
struct SemGeometry {
  enum class Kind : std::uint8_t { plane, sphere };
  Kind kind = Kind::plane;
  Real radius = 0;

  static SemGeometry plane() noexcept { return SemGeometry{}; }
  /// Throws std::invalid_argument for a radius not positive and finite.
  static SemGeometry sphere(Real radius);
};

namespace detail {

/// The geometry of every element of a host mesh, or why there is none.
struct SemGeometryData {
  enum class Error : std::uint8_t { none, dimension, order, topology, orientation };
  Error error = Error::none;
  GO gid = 0;  ///< the first offending element, for topology and orientation
  View2<Real, Kokkos::HostSpace> x, covariant, contravariant, jacobian, local_mass;
};

/// Host only; validates locally and computes when valid.
SemGeometryData sem_geometry(const Mesh<Kokkos::HostSpace>& mesh, int order, const SemGeometry& geometry);

/// Collective: throws std::invalid_argument on every rank naming the worst
/// error any rank found (and the smallest offending gid).
void check_sem_geometry(const Comm& comm, const SemGeometryData& data, int dim, int order);

}  // namespace detail

template <class MemSpace>
class SemSpace {
 public:
  /// Collective. Throws std::invalid_argument on every rank if any rank's
  /// mesh is not of dimension 2, `order` is outside [1, max_node_order], an
  /// element is not a quad of 4, 8 or 9 nodes, or one's normal points the
  /// wrong way (plane: not along +z; sphere: not outward) somewhere; the
  /// message names the smallest such element gid.
  SemSpace(const Comm& comm, Mesh<MemSpace>& mesh, int order, SemGeometry geometry)
      : order_(order), geometry_(geometry) {
    std::optional<Mesh<Kokkos::HostSpace>> mirror;
    const Mesh<Kokkos::HostSpace>* host = nullptr;
    if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      host = &mesh;
    } else {
      mirror.emplace(mesh.template mirror<Kokkos::HostSpace>());
      host = &*mirror;
    }
    const detail::SemGeometryData g = detail::sem_geometry(*host, order, geometry);
    detail::check_sem_geometry(comm, g, mesh.dim(), order);
    elements_ = mesh.count(2);
    owned_ = mesh.segment_table(2).owned_end;
    space_ = make_space(mesh, Domain<MemSpace>::all(mesh, 2), Layout::element_nodes(NodeFamily::gll, order));
    x_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.x);
    covariant_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.covariant);
    contravariant_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.contravariant);
    jacobian_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.jacobian);
    local_mass_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.local_mass);
  }

  int order() const noexcept { return order_; }
  int points_per_element() const noexcept { return (order_ + 1) * (order_ + 1); }
  /// Every element on the rank, ghosts included.
  LO num_elements() const noexcept { return elements_; }
  LO owned_elements() const noexcept { return owned_; }
  const Space<MemSpace>& space() const noexcept { return space_; }
  const SemGeometry& geometry() const noexcept { return geometry_; }

  /// Per point, num_elements() * points_per_element() rows each.
  View2<const Real, MemSpace> x() const { return x_; }
  View2<const Real, MemSpace> covariant() const { return covariant_; }
  View2<const Real, MemSpace> contravariant() const { return contravariant_; }
  View2<const Real, MemSpace> jacobian() const { return jacobian_; }
  View2<const Real, MemSpace> local_mass() const { return local_mass_; }

 private:
  int order_;
  SemGeometry geometry_;
  LO elements_ = 0, owned_ = 0;
  Space<MemSpace> space_;
  View2<Real, MemSpace> x_, covariant_, contravariant_, jacobian_, local_mass_;
};

}  // namespace soma::sem

#endif  // SOMA_SEM_SEM_SPACE_HPP_
