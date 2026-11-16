// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_FV_HPP_
#define SOMA_ALGORITHMS_FV_HPP_

// Cell-centred finite volumes: first-order upwind advection of a scalar u,
// du/dt + div(a u) = 0, over a mesh of triangles, quads, tets or hexes,
// explicit (forward Euler) in time:
//
//   mesh.add_ghosts(comm, soma::GhostOptions{1, 0});  // on more than one rank
//   const soma::FvSpace<Mem> space(comm, mesh);
//   const auto phi = space.normal_flux(a);           // a(x, y, z), device-callable
//   auto u = space.vector();                         // one value per cell
//   const Real dt = 0.5 * space.max_dt(comm, phi);
//   for (int n = 0; n < steps; ++n) space.step(comm, phi, u, dt, inflow);
//
// A face carries phi_f = a(x_f) . A_f, its centroid x_f and area vector A_f
// (an edge's normal times its length in 2D; half the cross product of a
// quad's diagonals, exact for a bilinear face), and the flux phi_f u from
// its upwind side: the cell it leaves, or `inflow` where a enters the domain.
// Each owned cell sums its faces' fluxes in its side order and divides by its
// volume, so a face's flux leaves one cell and enters the other with opposite
// signs, and the result is the same bit for bit on any number of ranks and
// threads: a face's geometry is computed from its corners in an order fixed
// by gid (from the least, towards its lesser neighbour), not in the order a
// rank happens to hold them. A cell's volume is (1/d) sum_f A_f . (x_f - x_0)
// over its outward faces, x_0 its first corner; a cell whose sides turn
// inwards (a clockwise polygon) is taken the other way round.
//
// The space keeps views into the mesh's geometry as it was when made: a mesh
// whose coordinates move needs a new space. On more than one rank the mesh
// needs a ghost layer (through any bridge), so that every face of an owned
// cell is here; a step ends with one halo update of u. Vectors hold one entry
// per cell here, owned cells first; a rank whose vectors are not of cols()
// entries throws std::invalid_argument and leaves the others in the
// exchange.

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

namespace detail {

// What an FvSpace works from, built once on the host.
struct FvPlan {
  bool supported = true;  // dimension 2 or 3, tri, quad, tet or hex cells, no nodes
  LO nonmanifold = 0;     // owned cells' sides that more than two cells have
  LO bad_cells = 0;       // owned cells of no positive finite volume, or turned
                          // against a neighbour
  std::vector<Real> area;      // 3 per face, from the left cell to the right
  std::vector<Real> centroid;  // 3 per face
  std::vector<LO> left;        // the cell A_f leaves
  std::vector<LO> right;       // the cell it enters; -1 on the boundary
  std::vector<Offset> cell_offsets;    // owned cells + 1
  std::vector<LO> cell_faces;          // each owned cell's faces in side order
  std::vector<std::int8_t> cell_sign;  // +1 where the cell is the face's left
  std::vector<Real> volume;            // per owned cell
  std::vector<Real> cell_centroid;     // 3 per cell here, owned and ghost: its corners' mean, summed in gid order
  std::vector<Real> lsq;               // 9 per owned cell: the inverse of sum_n d d^T over its face neighbours n,
                                       // d = x_n - x_c (2D: the xy block); zero where that is near-singular
};

// Host only. A mesh not supported gives a plan with supported = false.
FvPlan fv_plan(const Mesh<Kokkos::HostSpace>& mesh);

}  // namespace detail

template <class MemSpace>
class FvSpace {
 public:
  using Exec = typename MemSpace::execution_space;

  // Collective. Throws std::invalid_argument on every rank if any rank's
  // mesh is not of dimension 2 or 3, has edits or high-order nodes, holds a
  // cell other than a tri, quad, tet or hex, a side more than two cells
  // share, or a cell of no positive volume or turned against a neighbour,
  // or, on more than one rank, has no ghost layer; or if ranks differ in
  // soma::reproducible() (which makes total() exact).
  FvSpace(const Comm& comm, const Mesh<MemSpace>& mesh) : reproducible_(soma::reproducible()) {
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
    if (comm.size() > 1) halo_.emplace(comm, mesh, dim);
  }

  LO rows() const noexcept { return rows_; }    // owned cells
  LO cols() const noexcept { return cols_; }    // all cells here
  LO faces() const noexcept { return faces_; }  // the faces of the owned cells
  bool reproducible() const noexcept { return reproducible_; }

  // Per owned cell, its volume (area in 2D).
  View1<const Real, MemSpace> volume() const { return volume_; }

  // A vector of zeros, one entry per cell here.
  View1<Real, MemSpace> vector(const char* label = "soma::FvSpace::vector") const {
    return View1<Real, MemSpace>(label, static_cast<std::size_t>(cols_));
  }

  // Per face, a(x_f) . A_f, from its left cell to its right; a(x, y, z)
  // returns a Kokkos::Array<Real, 3> and is device-callable.
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

  // Collective: the largest stable step, min over owned cells of the volume
  // over the outflow sum_f max(0, +-phi_f); infinite where nothing flows out
  // anywhere. One allreduce.
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

  // Collective: u += dt * (the net inflow of each owned cell) / its volume,
  // with `inflow` the value entering through the boundary; then u's ghosts
  // from their owners (one halo update). u's ghosts must hold their owners'
  // values on entry (fill all cols(), or call update() first). The flux
  // scratch is the space's, shared by its copies: one step at a time.
  void step(const Comm& comm, const View1<const Real, MemSpace>& phi, const View1<Real, MemSpace>& u, Real dt,
            Real inflow = 0) const {
    check(phi.extent(0), static_cast<std::size_t>(faces_), "normal fluxes not of faces()");
    check(u.extent(0), static_cast<std::size_t>(cols_), "a vector not of cols()");
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

  // Collective: each owner's value overwrites its ghosts'.
  void update(const Comm& comm, const View1<Real, MemSpace>& u) const {
    check(u.extent(0), static_cast<std::size_t>(cols_), "a vector not of cols()");
    if (halo_) halo_->update(comm, View2<Real, MemSpace>(u.data(), u.extent(0), 1));
  }

  // Collective: sum over owned cells of volume * u, the conserved quantity;
  // exact (rounded once) when reproducible(). One allreduce.
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
  std::optional<HaloExchange<MemSpace>> halo_;
};

}  // namespace soma

#endif  // SOMA_ALGORITHMS_FV_HPP_
