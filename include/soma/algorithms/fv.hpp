// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_FV_HPP_
#define SOMA_ALGORITHMS_FV_HPP_

/// \file
/// Cell-centred finite volumes: advection of a scalar u, du/dt + div(a u) = 0,
/// over a mesh of triangles, quads, tets or hexes, explicit in time, by one of
/// two schemes:
///
/// ```
///   upwind   first order, forward Euler (the default);
///   bounded  second order where u is smooth, and no new extrema: flux-corrected
///            transport, two stages (SSP-RK2, Heun) averaged with the start.
///
///   mesh.add_ghosts(comm, soma::GhostOptions{1, 0});  // on more than one rank
///   const soma::FvSpace<Mem> space(comm, mesh);      // or (comm, mesh, FvScheme::bounded)
///   const auto phi = space.normal_flux(a);           // a(x, y, z), device-callable
///   auto u = space.vector();                         // one value per cell
///   const Real dt = 0.5 * space.max_dt(comm, phi);
///   for (int n = 0; n < steps; ++n) space.step(comm, phi, u, dt, inflow);
/// ```
///
/// A face carries phi_f = a(x_f) . A_f, its centroid x_f and area vector A_f
/// (an edge's normal times its length in 2D; half the cross product of a
/// quad's diagonals, exact for a bilinear face), and the flux phi_f u from
/// its upwind side: the cell it leaves, or `inflow` where a enters the domain.
/// Each owned cell sums its faces' fluxes in its side order and divides by its
/// volume, so a face's flux leaves one cell and enters the other with opposite
/// signs, and the result is the same bit for bit on any number of ranks and
/// threads: a face's geometry is computed from its corners in an order fixed
/// by gid (from the least, towards its lesser neighbour), not in the order a
/// rank happens to hold them. A cell's volume is (1/d) sum_f A_f . (x_f - x_0)
/// over its outward faces, x_0 its first corner; a cell whose sides turn
/// inwards (a clockwise polygon) is taken the other way round.
///
/// A bounded stage (Boris and Book; Zalesak's limiter) takes the upwind result
/// u^L and adds to it, per face, alpha_f times the difference between a
/// second-order flux and the upwind one. The second-order flux carries the
/// upwind cell's value extended linearly to the face centroid, with the cell's
/// least-squares gradient over its face neighbours (zero where those do not
/// span the space, as at a corner cell with one neighbour). alpha_f in [0, 1]
/// is as large as keeps every cell within the least and greatest of u and u^L
/// over itself and its face neighbours. So, where the fluxes of each cell sum
/// to zero (a divergence-free flow: exactly so for a linear a on cells with
/// planar sides) and dt <= max_dt(), no stage makes a new extremum, and the
/// average of the two stages with the start makes none either: u stays within
/// its starting range (and the inflow's), to rounding (a value at an extremum
/// may pass it by a few units in the last place). Each correction leaves one cell and
/// enters the other, so the total is conserved as for upwind. In multiple
/// dimensions no scheme of second order is TVD (Goodman and LeVeque); this one
/// is second order away from extrema and clips them to first order. A stage
/// exchanges ghosts three times (gradients and u^L, the limiter, u); a step,
/// twice that.
///
/// The space keeps views into the mesh's geometry as it was when made: a mesh
/// whose coordinates move needs a new space. On more than one rank the mesh
/// needs a ghost layer (through any bridge), so that every face of an owned
/// cell is here; a step ends with one halo update of u. Vectors hold one entry
/// per cell here, owned cells first; a rank whose vectors are not of cols()
/// entries throws std::invalid_argument and leaves the others in the
/// exchange. Upwind keeps its results bit for bit; bounded is bit for bit
/// the same on any partition and thread count too.

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/to_space.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/exact_sum.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/runtime/options.hpp"

namespace soma {

enum class FvScheme { upwind, bounded };

namespace detail {

/// For tests only: bounded steps without the limiter (every alpha_f = 1),
/// which overshoot. Read at each stage; not for use while steps run.
inline bool fv_unlimited = false;

/// What an FvSpace works from, built once on the host.
struct FvPlan {
  bool supported = true;  ///< dimension 2 or 3, tri, quad, tet or hex cells, no nodes
  LO nonmanifold = 0;     ///< owned cells' sides that more than two cells have
  LO bad_cells = 0;       ///< owned cells of no positive finite volume, or turned
                          ///< against a neighbour
  std::vector<Real> area;      ///< 3 per face, from the left cell to the right
  std::vector<Real> centroid;  ///< 3 per face
  std::vector<LO> left;        ///< the cell A_f leaves
  std::vector<LO> right;       ///< the cell it enters; -1 on the boundary
  std::vector<Offset> cell_offsets;    ///< owned cells + 1
  std::vector<LO> cell_faces;          ///< each owned cell's faces in side order
  std::vector<std::int8_t> cell_sign;  ///< +1 where the cell is the face's left
  std::vector<Real> volume;            ///< per owned cell
  std::vector<Real> cell_centroid;     ///< 3 per cell here, owned and ghost: its corners' mean, summed in gid order
  std::vector<Real> lsq;               ///< 9 per owned cell: the inverse of sum_n d d^T over its face neighbours n,
                                       ///< d = x_n - x_c (2D: the xy block); zero where that is near-singular
};

/// Host only. A mesh not supported gives a plan with supported = false.
FvPlan fv_plan(const Mesh<Kokkos::HostSpace>& mesh);

}  // namespace detail

template <class MemSpace>
class FvSpace {
 public:
  using Exec = typename MemSpace::execution_space;

  /// Collective. Throws std::invalid_argument on every rank if any rank's
  /// mesh is not of dimension 2 or 3, has edits or high-order nodes, holds a
  /// cell other than a tri, quad, tet or hex, a side more than two cells
  /// share, or a cell of no positive volume or turned against a neighbour,
  /// or, on more than one rank, has no ghost layer; or if ranks differ in
  /// soma::reproducible() (which makes total() exact).
  FvSpace(const Comm& comm, const Mesh<MemSpace>& mesh, FvScheme scheme = FvScheme::upwind)
      : scheme_(scheme), reproducible_(soma::reproducible()) {
    std::optional<Mesh<Kokkos::HostSpace>> mirror;
    const Mesh<Kokkos::HostSpace>* host = nullptr;
    if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      host = &mesh;
    } else {
      mirror.emplace(mesh.template mirror<Kokkos::HostSpace>());
      host = &*mirror;
    }
    const bool bad = mesh.edited() || (comm.size() > 1 && mesh.ghost_layers() < 1);
    detail::FvPlan p;
    if (!bad) p = detail::fv_plan(*host);
    const Real exact = reproducible_ ? Real{1} : Real{0};
    Real flags[5] = {bad || !p.supported ? Real{1} : Real{0}, static_cast<Real>(p.nonmanifold),
                     static_cast<Real>(p.bad_cells), exact, -exact};
    comm.allreduce(std::span<Real>(flags), Op::max);
    if (flags[3] != -flags[4]) throw std::invalid_argument("FvSpace: ranks differ in soma::reproducible()");
    if (flags[0] != 0) {
      throw std::invalid_argument(
          "FvSpace: a rank's mesh is not of dimension 2 or 3, has edits or high-order nodes, holds a cell other "
          "than a tri, quad, tet or hex or, on more than one rank, has no ghost layer");
    }
    if (flags[1] != 0) throw std::invalid_argument("FvSpace: a rank holds a side that more than two cells share");
    if (flags[2] != 0) {
      throw std::invalid_argument("FvSpace: a rank holds a cell of no positive volume or turned against a neighbour");
    }
    const int dim = mesh.dim();
    rows_ = mesh.segment_table(dim).owned_end;
    cols_ = mesh.count(dim);
    faces_ = static_cast<LO>(p.left.size());
    area_ = detail::to_space<Real, MemSpace>(p.area, "soma::FvSpace::area");
    centroid_ = detail::to_space<Real, MemSpace>(p.centroid, "soma::FvSpace::centroid");
    left_ = detail::to_space<LO, MemSpace>(p.left, "soma::FvSpace::left");
    right_ = detail::to_space<LO, MemSpace>(p.right, "soma::FvSpace::right");
    cell_offsets_ = detail::to_space<Offset, MemSpace>(p.cell_offsets, "soma::FvSpace::cell_offsets");
    cell_faces_ = detail::to_space<LO, MemSpace>(p.cell_faces, "soma::FvSpace::cell_faces");
    cell_sign_ = detail::to_space<std::int8_t, MemSpace>(p.cell_sign, "soma::FvSpace::cell_sign");
    volume_ = detail::to_space<Real, MemSpace>(p.volume, "soma::FvSpace::volume");
    flux_ = View1<Real, MemSpace>("soma::FvSpace::flux", static_cast<std::size_t>(faces_));
    if (scheme_ == FvScheme::bounded) {
      const auto cols = static_cast<std::size_t>(cols_);
      cell_centroid_ = detail::to_space<Real, MemSpace>(p.cell_centroid, "soma::FvSpace::cell_centroid");
      lsq_ = detail::to_space<Real, MemSpace>(p.lsq, "soma::FvSpace::lsq");
      aux_ = View2<Real, MemSpace>("soma::FvSpace::gradient_low", cols, 4);
      limit_ = View2<Real, MemSpace>("soma::FvSpace::limit", cols, 2);
      anti_ = View1<Real, MemSpace>("soma::FvSpace::antidiffusive", static_cast<std::size_t>(faces_));
      stage1_ = View1<Real, MemSpace>("soma::FvSpace::stage1", cols);
      stage2_ = View1<Real, MemSpace>("soma::FvSpace::stage2", cols);
    }
    if (comm.size() > 1) halo_.emplace(comm, mesh, dim);
  }

  LO rows() const noexcept { return rows_; }    // owned cells
  LO cols() const noexcept { return cols_; }    // all cells here
  LO faces() const noexcept { return faces_; }  // the faces of the owned cells
  bool reproducible() const noexcept { return reproducible_; }
  FvScheme scheme() const noexcept { return scheme_; }

  /// Per owned cell, its volume (area in 2D).
  View1<const Real, MemSpace> volume() const { return volume_; }

  /// A vector of zeros, one entry per cell here.
  View1<Real, MemSpace> vector(const char* label = "soma::FvSpace::vector") const {
    return View1<Real, MemSpace>(label, static_cast<std::size_t>(cols_));
  }

  /// Per face, a(x_f) . A_f, from its left cell to its right; a(x, y, z)
  /// returns a Kokkos::Array<Real, 3> and is device-callable.
  template <class F>
  View1<Real, MemSpace> normal_flux(const F& a) const {
    View1<Real, MemSpace> phi("soma::FvSpace::normal_flux", static_cast<std::size_t>(faces_));
    const auto area = area_;
    const auto centroid = centroid_;
    Kokkos::parallel_for(
        "soma::FvSpace::normal_flux", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, faces_),
        KOKKOS_LAMBDA(const LO f) {
          const auto v = a(centroid(3 * f), centroid(3 * f + 1), centroid(3 * f + 2));
          phi(f) = v[0] * area(3 * f) + v[1] * area(3 * f + 1) + v[2] * area(3 * f + 2);
        });
    Exec{}.fence();
    return phi;
  }

  /// Collective: the largest stable step, min over owned cells of the volume
  /// over the outflow sum_f max(0, +-phi_f); infinite where nothing flows out
  /// anywhere. One allreduce.
  Real max_dt(const Comm& comm, const View1<const Real, MemSpace>& phi) const {
    check(phi.extent(0), static_cast<std::size_t>(faces_), "normal fluxes not of faces()");
    const auto offsets = cell_offsets_;
    const auto faces = cell_faces_;
    const auto sign = cell_sign_;
    const auto volume = volume_;
    Real least = Kokkos::Experimental::infinity_v<Real>;
    Kokkos::parallel_reduce(
        "soma::FvSpace::max_dt", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
        KOKKOS_LAMBDA(const LO c, Real& acc) {
          Real out = 0;
          for (Offset k = offsets(c); k < offsets(c + 1); ++k) {
            const Real flow = sign(k) * phi(faces(k));
            out += flow > 0 ? flow : 0;
          }
          if (out > 0) acc = Kokkos::min(acc, volume(c) / out);
        },
        Kokkos::Min<Real>(least));
    return comm.allreduce(least, Op::min);
  }

  /// Collective: upwind, u += dt * (the net inflow of each owned cell) / its
  /// volume, with `inflow` the value entering through the boundary; then u's
  /// ghosts from their owners (one halo update). Bounded, two such stages with
  /// the corrections above, averaged with u (six halo updates). u's ghosts
  /// must hold their owners' values on entry (fill all cols(), or call
  /// update() first). The scratch is the space's, shared by its copies: one
  /// step at a time.
  void step(const Comm& comm, const View1<const Real, MemSpace>& phi, const View1<Real, MemSpace>& u, Real dt,
            Real inflow = 0) const {
    check(phi.extent(0), static_cast<std::size_t>(faces_), "normal fluxes not of faces()");
    check(u.extent(0), static_cast<std::size_t>(cols_), "a vector not of cols()");
    if (scheme_ == FvScheme::bounded) {
      const View1<const Real, MemSpace> start(u);
      bounded_stage(comm, phi, start, stage1_, dt, inflow);
      bounded_stage(comm, phi, stage1_, stage2_, dt, inflow);
      const auto second = stage2_;
      Kokkos::parallel_for(
          "soma::FvSpace::heun", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, cols_),
          KOKKOS_LAMBDA(const LO c) { u(c) = (u(c) + second(c)) / 2; });
      Exec{}.fence();
      return;
    }
    const auto left = left_;
    const auto right = right_;
    const auto flux = flux_;
    Kokkos::parallel_for(
        "soma::FvSpace::face_flux", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, faces_),
        KOKKOS_LAMBDA(const LO f) {
          const Real p = phi(f);
          const LO r = right(f);
          flux(f) = p * (p >= 0 ? u(left(f)) : r >= 0 ? u(r) : inflow);
        });
    const auto offsets = cell_offsets_;
    const auto faces = cell_faces_;
    const auto sign = cell_sign_;
    const auto volume = volume_;
    Kokkos::parallel_for(
        "soma::FvSpace::update", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
        KOKKOS_LAMBDA(const LO c) {
          Real out = 0;
          for (Offset k = offsets(c); k < offsets(c + 1); ++k) out += sign(k) * flux(faces(k));
          if (out != 0) u(c) -= dt * out / volume(c);  // still, an infinite dt leaves u
        });
    Exec{}.fence();
    update(comm, u);
  }

  /// Collective: each owner's value overwrites its ghosts'.
  void update(const Comm& comm, const View1<Real, MemSpace>& u) const {
    check(u.extent(0), static_cast<std::size_t>(cols_), "a vector not of cols()");
    if (halo_) halo_->update(comm, View2<Real, MemSpace>(u.data(), u.extent(0), 1));
  }

  /// Collective: sum over owned cells of volume * u, the conserved quantity;
  /// exact (rounded once) when reproducible(). One allreduce.
  Real total(const Comm& comm, const View1<const Real, MemSpace>& u) const {
    check(u.extent(0), static_cast<std::size_t>(cols_), "a vector not of cols()");
    const auto volume = volume_;
    if (reproducible_) {
      const auto sums = reduce_exact<1, Exec>("soma::FvSpace::total_exact", 0, rows_,
                                              KOKKOS_LAMBDA(const LO c, ExactSums<1>& acc) {
                                                acc.add(0, volume(c) * u(c));
                                              });
      return allreduce_exact(comm, sums)[0];
    }
    Real s = 0;
    Kokkos::parallel_reduce(
        "soma::FvSpace::total", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
        KOKKOS_LAMBDA(const LO c, Real& acc) { acc += volume(c) * u(c); }, s);
    return comm.allreduce(s, Op::sum);
  }

 private:
  static void check(std::size_t have, std::size_t want, const char* what) {
    if (have != want) throw std::invalid_argument(std::string("FvSpace: ") + what);
  }

  /// One bounded stage, from `in` (ghosts current) to `out` (ghosts updated).
  void bounded_stage(const Comm& comm, const View1<const Real, MemSpace>& phi, const View1<const Real, MemSpace>& in,
                     const View1<Real, MemSpace>& out, Real dt, Real inflow) const {
    const bool unlimited = detail::fv_unlimited;
    const auto left = left_;
    const auto right = right_;
    const auto flux = flux_;
    const auto offsets = cell_offsets_;
    const auto faces = cell_faces_;
    const auto sign = cell_sign_;
    const auto volume = volume_;
    const auto centroid = centroid_;
    const auto cell = cell_centroid_;
    const auto lsq = lsq_;
    const auto aux = aux_;  // columns: the gradient, then u^L
    const auto limit = limit_;  // columns: R+, R-
    const auto anti = anti_;
    const auto faces_policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, faces_);
    const auto rows_policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_);
    // The upwind fluxes.
    Kokkos::parallel_for(
        "soma::FvSpace::low_flux", faces_policy, KOKKOS_LAMBDA(const LO f) {
          const Real p = phi(f);
          const LO r = right(f);
          flux(f) = p * (p >= 0 ? in(left(f)) : r >= 0 ? in(r) : inflow);
        });
    // Each owned cell's gradient and upwind result.
    Kokkos::parallel_for(
        "soma::FvSpace::gradient_low", rows_policy, KOKKOS_LAMBDA(const LO c) {
          Real rhs[3] = {0, 0, 0};
          Real net = 0;
          for (Offset k = offsets(c); k < offsets(c + 1); ++k) {
            const LO f = faces(k);
            net += sign(k) * flux(f);
            const LO n = left(f) == c ? right(f) : left(f);
            if (n < 0) continue;
            const Real du = in(n) - in(c);
            for (int q = 0; q < 3; ++q) rhs[q] += (cell(3 * n + q) - cell(3 * c + q)) * du;
          }
          for (int q = 0; q < 3; ++q) {
            aux(c, q) = lsq(9 * c + 3 * q) * rhs[0] + lsq(9 * c + 3 * q + 1) * rhs[1] + lsq(9 * c + 3 * q + 2) * rhs[2];
          }
          aux(c, 3) = net != 0 ? in(c) - dt * net / volume(c) : in(c);
        });
    Exec{}.fence();
    if (halo_) halo_->update(comm, aux);
    // Per face, dt times the second-order flux less the upwind one.
    Kokkos::parallel_for(
        "soma::FvSpace::antidiffusive", faces_policy, KOKKOS_LAMBDA(const LO f) {
          const Real p = phi(f);
          const LO up = p >= 0 ? left(f) : right(f);
          Real a = 0;
          if (up >= 0) {
            Real value = in(up);
            for (int q = 0; q < 3; ++q) value += aux(up, q) * (centroid(3 * f + q) - cell(3 * up + q));
            a = p * value - flux(f);
          }
          anti(f) = a != 0 ? dt * a : 0;
        });
    // Zalesak: the fraction of its incoming (R+) and outgoing (R-) corrections
    // each owned cell takes and stays within its neighbours' range.
    Kokkos::parallel_for(
        "soma::FvSpace::limiter", rows_policy, KOKKOS_LAMBDA(const LO c) {
          const Real low = aux(c, 3);
          Real hi = Kokkos::max(in(c), low), lo = Kokkos::min(in(c), low), gain = 0, loss = 0;
          for (Offset k = offsets(c); k < offsets(c + 1); ++k) {
            const LO f = faces(k);
            const LO n = left(f) == c ? right(f) : left(f);
            if (n >= 0) {
              hi = Kokkos::max(hi, Kokkos::max(in(n), aux(n, 3)));
              lo = Kokkos::min(lo, Kokkos::min(in(n), aux(n, 3)));
            }
            const Real into = -sign(k) * anti(f);
            if (into > 0) gain += into;
            if (into < 0) loss += into;
          }
          limit(c, 0) = gain > 0 ? Kokkos::min(Real{1}, volume(c) * (hi - low) / gain) : Real{1};
          limit(c, 1) = loss < 0 ? Kokkos::min(Real{1}, volume(c) * (lo - low) / loss) : Real{1};
        });
    Exec{}.fence();
    if (halo_) halo_->update(comm, limit);
    // Each face's correction scaled by what both its cells allow: the one it
    // leaves (R-) and the one it enters (R+).
    Kokkos::parallel_for(
        "soma::FvSpace::alpha", faces_policy, KOKKOS_LAMBDA(const LO f) {
          const Real a = anti(f);
          if (a == 0 || unlimited) return;
          const LO l = left(f), r = right(f);
          Real alpha = a > 0 ? limit(l, 1) : limit(l, 0);
          if (r >= 0) alpha = Kokkos::min(alpha, a > 0 ? limit(r, 0) : limit(r, 1));
          anti(f) = alpha * a;
        });
    Kokkos::parallel_for(
        "soma::FvSpace::correct", rows_policy, KOKKOS_LAMBDA(const LO c) {
          Real net = 0;
          for (Offset k = offsets(c); k < offsets(c + 1); ++k) net += sign(k) * anti(faces(k));
          out(c) = net != 0 ? aux(c, 3) - net / volume(c) : aux(c, 3);
        });
    Exec{}.fence();
    update(comm, out);
  }

  FvScheme scheme_ = FvScheme::upwind;
  bool reproducible_ = false;
  LO rows_ = 0;
  LO cols_ = 0;
  LO faces_ = 0;
  View1<Real, MemSpace> area_;
  View1<Real, MemSpace> centroid_;
  View1<LO, MemSpace> left_;
  View1<LO, MemSpace> right_;
  View1<Offset, MemSpace> cell_offsets_;
  View1<LO, MemSpace> cell_faces_;
  View1<std::int8_t, MemSpace> cell_sign_;
  View1<Real, MemSpace> volume_;
  View1<Real, MemSpace> flux_;
  View1<Real, MemSpace> cell_centroid_;  ///< bounded only, as the rest below
  View1<Real, MemSpace> lsq_;
  View2<Real, MemSpace> aux_;
  View2<Real, MemSpace> limit_;
  View1<Real, MemSpace> anti_;
  View1<Real, MemSpace> stage1_;
  View1<Real, MemSpace> stage2_;
  std::optional<HaloExchange<MemSpace>> halo_;
};

}  // namespace soma

#endif  // SOMA_ALGORITHMS_FV_HPP_
