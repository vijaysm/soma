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
///
/// Direct stiffness summation (dss) makes element-local values continuous: a
/// point shared by several elements (on a vertex or an edge) takes
/// sum_e m_e u_e / sum_e m_e, m_e the elements' local masses there; interior
/// points are left as they are. Each shared vertex and edge is summed by its
/// owner over its elements in ascending gid, so the result is the same, bit
/// for bit, on any number of ranks and threads. On more than one rank the
/// mesh must hold a ghost layer through vertices and distributed edges:
/// ```
///   mesh.add_ghosts(comm, soma::GhostOptions{1, 0});
///   soma::build_distributed_adjacencies(comm, mesh, soma::AdjacencyRequest{}.add(2, 1));
/// ```
/// On one rank the space derives the edges itself when the mesh has none.
/// dss first gives ghost elements their owners' values, so only the owned
/// elements' values are read; afterwards ghosts hold their owners' results.

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/to_space.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/transfer.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/exact_sum.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/sem/gll.hpp"

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
/// Errors are ordered: the largest any rank finds is reported.
struct SemGeometryData {
  enum class Error : std::uint8_t { none, ghosts, orientation, radius, topology, edited, order, dimension };
  Error error = Error::none;
  GO gid = 0;  ///< the first offending element, for topology and orientation
  View2<Real, Kokkos::HostSpace> x, covariant, contravariant, jacobian, local_mass;
};

/// Host only; validates locally and computes when valid.
SemGeometryData sem_geometry(const Mesh<Kokkos::HostSpace>& mesh, int order, const SemGeometry& geometry);

/// Where direct stiffness summation reads and writes, built once on the host.
/// A point on an edge is addressed by its position s in 1..p-1 along the
/// edge from its vertex of smaller gid: row base + step s of the element.
struct SemDssPlan {
  /// Each owned vertex's corner points, its elements ascending by gid.
  std::vector<Offset> vertex_offsets{0};
  std::vector<LO> vertex_rows;
  /// Each owned edge's elements, ascending by gid: base and step.
  std::vector<Offset> edge_offsets{0};
  std::vector<LO> edge_base;
  std::vector<LO> edge_step;
  /// Per element and side q (corners q, q + 1): the corner's vertex, the
  /// side's edge, and the edge's base and step in the element.
  std::vector<LO> element_vertex, element_edge, element_base, element_step;
  bool complete = true;  ///< false: an element side has no edge here
};

/// Host only; the mesh holds its (2, 1) adjacency and passed sem_geometry.
SemDssPlan sem_dss_plan(const Mesh<Kokkos::HostSpace>& mesh, int order);

/// Owned sums of mass times value, per vertex (count(0) x components) and
/// per edge (count(1) x (p - 1) components); with `scatter`, writes
/// buf / mass (or buf when mass is empty) back to every element's points.
template <class MemSpace>
struct DssKernels {
  using Exec = typename MemSpace::execution_space;
  using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>;

  View1<Offset, MemSpace> vertex_offsets, edge_offsets;
  View1<LO, MemSpace> vertex_rows, edge_base, edge_step;
  View1<LO, MemSpace> element_vertex, element_edge, element_base, element_step;
  LO owned_vertices = 0, owned_edges = 0, elements = 0;
  int order = 1;

  void sum(const View2<const Real, MemSpace>& m, const View2<const Real, MemSpace>& u,
           const View2<Real, MemSpace>& vbuf, const View2<Real, MemSpace>& ebuf) const {
    const auto nc = static_cast<int>(u.extent(1));
    const int inner = order - 1;
    const auto vo = vertex_offsets;
    const auto vr = vertex_rows;
    Kokkos::parallel_for(
        "soma::sem::dss::vertices", Policy(0, owned_vertices), KOKKOS_LAMBDA(const LO v) {
          for (int c = 0; c < nc; ++c) {
            Real s = 0;
            for (Offset t = vo(v); t < vo(v + 1); ++t) s += m(vr(t), 0) * u(vr(t), c);
            vbuf(v, c) = s;
          }
        });
    const auto eo = edge_offsets;
    const auto eb = edge_base;
    const auto es = edge_step;
    Kokkos::parallel_for(
        "soma::sem::dss::edges", Policy(0, owned_edges), KOKKOS_LAMBDA(const LO e) {
          for (int c = 0; c < nc; ++c) {
            for (int s = 1; s <= inner; ++s) {
              Real sum = 0;
              for (Offset t = eo(e); t < eo(e + 1); ++t) {
                const LO r = eb(t) + es(t) * s;
                sum += m(r, 0) * u(r, c);
              }
              ebuf(e, (s - 1) + inner * c) = sum;
            }
          }
        });
    Exec{}.fence();
  }

  void scatter(const View2<const Real, MemSpace>& vbuf, const View2<const Real, MemSpace>& ebuf,
               const View2<const Real, MemSpace>& mass, const View2<Real, MemSpace>& u) const {
    const auto nc = static_cast<int>(u.extent(1));
    const int p = order, np = order + 1, inner = order - 1;
    const bool divide = mass.extent(0) > 0;
    const auto ev = element_vertex;
    const auto ee = element_edge;
    const auto eb = element_base;
    const auto es = element_step;
    Kokkos::parallel_for(
        "soma::sem::dss::scatter", Policy(0, elements), KOKKOS_LAMBDA(const LO e) {
          const LO first = e * np * np;
          const LO corner[4] = {first, first + p, first + np * np - 1, first + p * np};
          for (int q = 0; q < 4; ++q) {
            const LO v = ev(4 * e + q), r = corner[q];
            for (int c = 0; c < nc; ++c) u(r, c) = divide ? vbuf(v, c) / mass(r, 0) : vbuf(v, c);
            const LO edge = ee(4 * e + q);
            for (int s = 1; s <= inner; ++s) {
              const LO at = eb(4 * e + q) + es(4 * e + q) * s;
              for (int c = 0; c < nc; ++c) {
                const Real b = ebuf(edge, (s - 1) + inner * c);
                u(at, c) = divide ? b / mass(at, 0) : b;
              }
            }
          }
        });
    Exec{}.fence();
  }
};

/// What a space's host mirror of a mesh in another memory space must carry:
/// the (2, 1) graph when the mesh holds it, and the sharing and owner fields
/// of the halo exchanges when it has them.
template <class MemSpace>
TransferOptions sem_mirror_options(const Mesh<MemSpace>& mesh) {
  TransferOptions o;
  for (const char* name : {sharing_field, owner_field}) {
    if (mesh.fields().contains(name)) o.fields.emplace_back(name);
  }
  if (mesh.dim() == 2 && mesh.has_adjacency(2, 1)) {
    o.adjacencies = AdjacencyRequest{}.add(2, 1);
    o.adj_source = TransferOptions::AdjSource::copy;
  }
  return o;
}

/// Collective: throws std::invalid_argument on every rank naming the worst
/// error any rank found (and the smallest offending gid for the elements').
void check_sem_geometry(const Comm& comm, const SemGeometryData& data, int dim, int order);

}  // namespace detail

template <class MemSpace>
class SemSpace {
 public:
  /// Collective. Throws std::invalid_argument on every rank if any rank's
  /// mesh is not of dimension 2, `order` is outside [1, max_node_order], an
  /// element is not a quad of 4, 8 or 9 nodes, or one's normal points the
  /// wrong way (plane: not along +z; sphere: not outward) somewhere, or (on
  /// a sphere) one has a corner off the geometry's radius by more than 1e-8
  /// of it (the message names the smallest such element gid), the mesh has
  /// removed entities not yet compressed, or, on more than one
  /// rank, the mesh lacks the ghost layer through vertices or the
  /// distributed edges. On one rank, derives the mesh's edges if it has none.
  SemSpace(const Comm& comm, Mesh<MemSpace>& mesh, int order, SemGeometry geometry)
      : order_(order), geometry_(geometry) {
    if (comm.size() == 1 && mesh.dim() == 2 && !mesh.has_adjacency(2, 1)) {
      build_adjacencies(mesh, AdjacencyRequest{}.add(2, 1));
    }
    std::optional<Mesh<Kokkos::HostSpace>> mirror;
    const Mesh<Kokkos::HostSpace>* host = nullptr;
    if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      host = &mesh;
    } else {
      mirror.emplace(mesh.template mirror<Kokkos::HostSpace>(detail::sem_mirror_options(mesh)));
      host = &*mirror;
    }
    detail::SemGeometryData g;
    if (mesh.edited()) {
      g.error = detail::SemGeometryData::Error::edited;
    } else {
      g = detail::sem_geometry(*host, order, geometry);
    }
    detail::SemDssPlan plan;
    if (g.error == detail::SemGeometryData::Error::none) {
      const bool distributed = comm.size() == 1 || (mesh.ghost_layers() >= 1 && mesh.ghost_bridge_dim() == 0 &&
                                                    host->has_adjacency(2, 1) && has_sharing(*host, 1));
      if (distributed) plan = detail::sem_dss_plan(*host, order);
      if (!distributed || !plan.complete) g.error = detail::SemGeometryData::Error::ghosts;
    }
    detail::check_sem_geometry(comm, g, mesh.dim(), order);
    elements_ = mesh.count(2);
    owned_ = mesh.segment_table(2).owned_end;
    space_ = make_space(mesh, Domain<MemSpace>::all(mesh, 2), Layout::element_nodes(NodeFamily::gll, order));
    x_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.x);
    covariant_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.covariant);
    contravariant_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.contravariant);
    jacobian_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.jacobian);
    local_mass_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g.local_mass);
    if (g.error == detail::SemGeometryData::Error::none) {
      const std::vector<Real> d = gll_derivative_1d(order), w = gll_weights_1d(order);
      const auto np = static_cast<std::size_t>(order + 1);
      View2<Real, Kokkos::HostSpace> dh("soma::sem::SemSpace::derivative", np, np);
      for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t j = 0; j < np; ++j) dh(i, j) = d[i * np + j];
      }
      derivative_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, dh);
      weights_ = soma::detail::to_space<Real, MemSpace>(w, "soma::sem::SemSpace::weights");
    }

    k_.vertex_offsets = soma::detail::to_space<Offset, MemSpace>(plan.vertex_offsets, "soma::sem::dss::vertex_offsets");
    k_.vertex_rows = soma::detail::to_space<LO, MemSpace>(plan.vertex_rows, "soma::sem::dss::vertex_rows");
    k_.edge_offsets = soma::detail::to_space<Offset, MemSpace>(plan.edge_offsets, "soma::sem::dss::edge_offsets");
    k_.edge_base = soma::detail::to_space<LO, MemSpace>(plan.edge_base, "soma::sem::dss::edge_base");
    k_.edge_step = soma::detail::to_space<LO, MemSpace>(plan.edge_step, "soma::sem::dss::edge_step");
    k_.element_vertex = soma::detail::to_space<LO, MemSpace>(plan.element_vertex, "soma::sem::dss::element_vertex");
    k_.element_edge = soma::detail::to_space<LO, MemSpace>(plan.element_edge, "soma::sem::dss::element_edge");
    k_.element_base = soma::detail::to_space<LO, MemSpace>(plan.element_base, "soma::sem::dss::element_base");
    k_.element_step = soma::detail::to_space<LO, MemSpace>(plan.element_step, "soma::sem::dss::element_step");
    k_.owned_vertices = static_cast<LO>(plan.vertex_offsets.size() - 1);
    k_.owned_edges = static_cast<LO>(plan.edge_offsets.size() - 1);
    k_.elements = elements_;
    k_.order = order;
    vertices_ = mesh.count(0);
    edges_ = mesh.count(1);
    if (comm.size() > 1) {
      for (int d = 0; d < 3; ++d) halo_[static_cast<std::size_t>(d)].emplace(comm, mesh, d);
    }

    // The assembled mass: each shared point's local masses summed as dss sums.
    mass_ = View2<Real, MemSpace>("soma::sem::SemSpace::mass", x_.extent(0), 1);
    Kokkos::deep_copy(mass_, local_mass_);
    View2<Real, MemSpace> one("soma::sem::SemSpace::one", x_.extent(0), 1);
    Kokkos::deep_copy(one, Real{1});
    sum_and_scatter(comm, one, mass_, View2<const Real, MemSpace>{});
  }

  /// Collective: values (num_elements() * points_per_element() rows, any
  /// number of components) made continuous: each shared point's value
  /// becomes sum_e m_e u_e / sum_e m_e over the elements holding it. Owned
  /// elements' values are read; every element's are written.
  void dss(const Comm& comm, const View2<Real, MemSpace>& values) const {
    check_rows(values.extent(0), "dss");
    if (comm.size() > 1) refresh_ghosts(comm, values);
    sum_and_scatter(comm, values, values, mass_);
  }

  /// The assembled mass at each point: the sum of the local masses of the
  /// elements holding it.
  View2<const Real, MemSpace> mass() const { return mass_; }

  /// Collective: the integral of one component, the sum over owned elements'
  /// points of local mass times value, summed exactly (the same bits on any
  /// number of ranks and threads).
  Real integral(const Comm& comm, const View2<const Real, MemSpace>& values, int component = 0) const {
    check_rows(values.extent(0), "integral");
    if (component < 0 || static_cast<std::size_t>(component) >= values.extent(1)) {
      throw std::invalid_argument("SemSpace::integral: component " + std::to_string(component) + " of " +
                                  std::to_string(values.extent(1)));
    }
    const auto m = local_mass_;
    const ExactSums<1> local = reduce_exact<1, typename MemSpace::execution_space>(
        "soma::sem::SemSpace::integral", 0, owned_ * points_per_element(),
        KOKKOS_LAMBDA(const LO r, ExactSums<1>& acc) { acc.add(0, m(r, 0) * values(r, component)); });
    return allreduce_exact(comm, local)[0];
  }

  int order() const noexcept { return order_; }
  int points_per_element() const noexcept { return (order_ + 1) * (order_ + 1); }
  /// Every element on the rank, ghosts included.
  LO num_elements() const noexcept { return elements_; }
  LO owned_elements() const noexcept { return owned_; }
  const Space<MemSpace>& space() const noexcept { return space_; }
  const SemGeometry& geometry() const noexcept { return geometry_; }

  /// The 1D GLL derivative matrix, D(i, j) = l_j'(xi_i), and weights
  /// (soma/sem/gll.hpp).
  View2<const Real, MemSpace> derivative() const { return derivative_; }
  View1<const Real, MemSpace> weights() const { return weights_; }

  /// Per point, num_elements() * points_per_element() rows each.
  View2<const Real, MemSpace> x() const { return x_; }
  View2<const Real, MemSpace> covariant() const { return covariant_; }
  View2<const Real, MemSpace> contravariant() const { return contravariant_; }
  View2<const Real, MemSpace> jacobian() const { return jacobian_; }
  View2<const Real, MemSpace> local_mass() const { return local_mass_; }

 private:
  static bool has_sharing(const Mesh<Kokkos::HostSpace>& mesh, int dim) {
    try {
      (void)mesh.fields().template get_var<int>(sharing_field, dim);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  void check_rows(std::size_t rows, const char* who) const {
    if (rows != x_.extent(0)) {
      throw std::invalid_argument(std::string("SemSpace::") + who + ": " + std::to_string(rows) + " rows, not " +
                                  std::to_string(x_.extent(0)));
    }
  }

  /// Ghost elements take their owners' values.
  void refresh_ghosts(const Comm& comm, const View2<Real, MemSpace>& values) const {
    using Policy = typename detail::DssKernels<MemSpace>::Policy;
    const LO ppe = points_per_element();
    const auto nc = static_cast<LO>(values.extent(1));
    View2<Real, MemSpace> block(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::sem::dss::elements"),
                                static_cast<std::size_t>(elements_), static_cast<std::size_t>(ppe * nc));
    Kokkos::parallel_for(
        "soma::sem::dss::pack", Policy(0, elements_), KOKKOS_LAMBDA(const LO e) {
          for (LO c = 0; c < nc; ++c) {
            for (LO k = 0; k < ppe; ++k) block(e, k + ppe * c) = values(e * ppe + k, c);
          }
        });
    halo_[2]->update(comm, block);
    Kokkos::parallel_for(
        "soma::sem::dss::unpack", Policy(owned_, elements_), KOKKOS_LAMBDA(const LO e) {
          for (LO c = 0; c < nc; ++c) {
            for (LO k = 0; k < ppe; ++k) values(e * ppe + k, c) = block(e, k + ppe * c);
          }
        });
    typename MemSpace::execution_space{}.fence();
  }

  /// Owners sum m u over each shared vertex and edge, copies and ghosts take
  /// the owners' sums, and every element's shared points take sum / mass
  /// (the sums themselves when mass is empty).
  void sum_and_scatter(const Comm& comm, const View2<Real, MemSpace>& in, const View2<Real, MemSpace>& out,
                       const View2<const Real, MemSpace>& mass) const {
    const auto nc = static_cast<std::size_t>(in.extent(1));
    const auto inner = static_cast<std::size_t>(order_ - 1);
    View2<Real, MemSpace> vbuf("soma::sem::dss::vertex_sums", static_cast<std::size_t>(vertices_), nc);
    View2<Real, MemSpace> ebuf("soma::sem::dss::edge_sums", static_cast<std::size_t>(edges_), inner * nc);
    k_.sum(local_mass_, in, vbuf, ebuf);
    if (comm.size() > 1) {
      halo_[0]->update(comm, vbuf);
      if (inner > 0) halo_[1]->update(comm, ebuf);
    }
    k_.scatter(vbuf, ebuf, mass, out);
  }

  int order_;
  SemGeometry geometry_;
  LO elements_ = 0, owned_ = 0, vertices_ = 0, edges_ = 0;
  Space<MemSpace> space_;
  View2<Real, MemSpace> x_, covariant_, contravariant_, jacobian_, local_mass_, mass_, derivative_;
  View1<Real, MemSpace> weights_;
  detail::DssKernels<MemSpace> k_;
  std::array<std::optional<HaloExchange<MemSpace>>, 3> halo_;
};

}  // namespace soma::sem

#endif  // SOMA_SEM_SEM_SPACE_HPP_
