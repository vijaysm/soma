// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_FEM_SPACE_HPP_
#define SOMA_ALGORITHMS_FEM_SPACE_HPP_

/// \file
/// Linear finite elements over a mesh's vertices: one unknown per vertex, P1
/// on triangles and tets, Q1 on quads and hexes (soma/algorithms/fem.hpp);
/// hanging vertices constrained to their masters (see the constructor).
///
/// ```
///   mesh.add_ghosts(comm, soma::GhostOptions{1, 0});  // on more than one rank
///   const soma::FemSpace<Mem> space(comm, mesh);
///   soma::CsrMatrix<Mem> a = space.laplacian(comm);
///   auto b = space.load(comm, f);                   // f(x, y, z), device-callable
///   soma::apply_dirichlet(a, b, space.boundary(), g);  // g: a value per vertex
/// ```
///
/// A rank's rows are its owned vertices, its columns all its vertices
/// (soma/algorithms/csr.hpp); a vector holds one entry per vertex, of which
/// copies and ghosts hold their owners' values after update(). On more than
/// one rank the mesh must hold a ghost layer through vertices: every cell
/// around an owned vertex is then here, and owners assemble their rows whole,
/// without messages. It must also hold its sharing fields, as a mirror does
/// with TransferOptions::everything().
///
/// Assembly gathers by default: the element matrices are computed once per
/// cell, then each owned row sums its cells' entries, the cells ascending by
/// gid. The result is the same, bit for bit, on any number of ranks and
/// threads. Atomic and colored assembly (Assembly) compute each element
/// matrix where it is added, storing none. With soma::reproducible() on when
/// a space is made (soma/runtime/options.hpp), it gathers whatever it is
/// asked (assembly() tells) and dot() sums exactly: the same bit for bit on
/// any number of ranks and threads too.
///
/// A space keeps views into the mesh: it is valid while the mesh is alive and
/// its topology unchanged. Coordinates may move between assemblies, so long
/// as copies and ghosts hold their owners' (laplacian_smooth leaves them so):
/// owned rows read them. Vectors updated hold exactly cols() entries on every
/// rank; a rank whose do not throws std::invalid_argument and leaves the
/// others in the exchange.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/conformity.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/to_space.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/exact_sum.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/runtime/options.hpp"

namespace soma {

/// How a space assembles its matrices:
/// ```
///   gather   each owned row sums its cells' entries, the cells by gid: the
///            same, bit for bit, on any number of ranks and threads;
///   atomic   each cell with an owned corner adds its entries to the owned
///            rows with atomic additions: order and so rounding vary;
///   colored  cells sharing no vertex form a color; the colors take turns,
///            the cells of one adding without atomics: the same on any number
///            of threads, not of ranks.
/// ```
enum class Assembly : std::uint8_t { gather, atomic, colored };

namespace detail {

/// What a space works from, built once on the host.
struct FemPlan {
  std::vector<Offset> offsets;  ///< the pattern: owned rows, columns by gid
  std::vector<LO> columns;
  /// The cells around each owned vertex, ascending by gid, the vertex's
  /// corner in each, and where each corner of the cell lands in the row:
  /// entry t's slots are slots[slot_offsets[t], slot_offsets[t + 1]).
  std::vector<Offset> around_offsets;
  std::vector<LO> around_cells;
  std::vector<std::uint8_t> around_corner;
  std::vector<Offset> slot_offsets;
  std::vector<LO> slots;  ///< from the row's start
  std::vector<Offset> matrix_offsets;  ///< per cell: start of its n x n entries
  std::vector<Offset> vector_offsets;  ///< per cell: start of its n entries
  std::vector<std::uint8_t> skin;      ///< per vertex, this rank's verdict
  /// atomic and colored: the cells with an owned corner, by color (one
  /// group for atomic); for each of them in that order, where corner b lands
  /// in corner a's row (-1 where a is not an owned row), n x n from
  /// cell_slot_offsets[i].
  std::vector<Offset> group_offsets;
  std::vector<LO> group_cells;
  std::vector<Offset> cell_slot_offsets;
  std::vector<LO> cell_slots;
  LO orphans = 0;    ///< owned vertices no cell here has: their rows are empty
  bool fits = true;  ///< false: a size exceeds Offset, and nothing else is set

  /// Hanging constraints (u_h = sum_k w_k u_{m_k}): the space is that of the
  /// free vertices, the matrix P^T A P and the load P^T b on their rows; a
  /// hanging vertex's row is the identity's with load 0. Gather assembly
  /// from per-row terms: values[row start + term_slot] += term_weight *
  /// element[term_entry], the terms in an order fixed by gid.
  bool constrained = false;
  std::vector<Offset> term_offsets;  ///< per owned row
  std::vector<Offset> term_entry;
  std::vector<LO> term_slot;
  std::vector<Real> term_weight;
  std::vector<Offset> load_offsets;  ///< per owned row
  std::vector<Offset> load_entry;
  std::vector<Real> load_weight;
  std::vector<LO> fixed_rows;  ///< owned hanging vertices
  /// Every hanging vertex here, owned first, with its masters here.
  std::vector<LO> hang;
  std::vector<Offset> hang_offsets;
  std::vector<LO> hang_masters;
  std::vector<Real> hang_weights;
  std::vector<std::uint8_t> hanging;  ///< per vertex: 1 if it hangs
  std::vector<LO> star;               ///< per vertex: the cells around it here
  std::vector<LO> needed;             ///< vertices whose cells must all be here
  LO missing = 0;                     ///< masters a needed vertex has that are not here
  /// The balance of the hanging terms: sum over owned hanging h, its masters
  /// m, of w key(m, h), less, over owned rows m, each hanging h it took with
  /// weight w, w key(m, h). Zero, summed over the ranks, when every master's
  /// row took every hanging vertex.
  ExactSums<1> balance;
};

/// The mesh is one of dimension 2 or 3, without high-order nodes, whose
/// cells all have elements here.
bool fem_supported(const Mesh<Kokkos::HostSpace>& mesh);

/// Host only; the mesh supported. With hanging constraints (`hc`, from
/// hanging_constraints), the constrained plan; it gathers.
FemPlan fem_plan(const Mesh<Kokkos::HostSpace>& mesh, Assembly assembly, const HangingConstraints* hc = nullptr);

/// The number of cells of the mesh whose Jacobian is not positive throughout.
template <class MemSpace>
LO bad_cells(const MeshView<MemSpace>& view) {
  using Exec = typename MemSpace::execution_space;
  const int dim = view.dim;
  LO bad = 0;
  Kokkos::parallel_reduce(
      "soma::FemSpace::bad_cells", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, view.count(dim)),
      KOKKOS_LAMBDA(const LO e, LO& sum) { sum += fem::least_jacobian(fem::cell_of(view, dim, e)) > 0 ? 0 : 1; },
      bad);
  return bad;
}

/// The element stiffness matrices of every cell, at matrix_offsets; `bad`
/// counts the cells whose Jacobian is not positive throughout.
template <class MemSpace>
View1<Real, MemSpace> element_matrices(const MeshView<MemSpace>& view, const View1<Offset, MemSpace>& at,
                                       Offset total, const Kokkos::View<LO, MemSpace>& bad) {
  using Exec = typename MemSpace::execution_space;
  const int dim = view.dim;
  View1<Real, MemSpace> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::FemSpace::element_matrices"),
                            static_cast<std::size_t>(total));
  Kokkos::parallel_for(
      "soma::FemSpace::stiffness", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, view.count(dim)),
      KOKKOS_LAMBDA(const LO e) {
        const fem::Cell c = fem::cell_of(view, dim, e);
        Real k[fem::max_nodes][fem::max_nodes];
        if (!(fem::stiffness(c, k) > 0)) Kokkos::atomic_inc(&bad());
        for (int a = 0; a < c.n; ++a) {
          for (int b = 0; b < c.n; ++b) out(at(e) + a * c.n + b) = k[a][b];
        }
      });
  return out;
}

/// The element load vectors of every cell, at vector_offsets; `bad` counts
/// as element_matrices does.
template <class MemSpace, class F>
View1<Real, MemSpace> element_loads(const MeshView<MemSpace>& view, const View1<Offset, MemSpace>& at, Offset total,
                                    const F& f, const Kokkos::View<LO, MemSpace>& bad) {
  using Exec = typename MemSpace::execution_space;
  const int dim = view.dim;
  View1<Real, MemSpace> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::FemSpace::element_loads"),
                            static_cast<std::size_t>(total));
  Kokkos::parallel_for(
      "soma::FemSpace::load", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, view.count(dim)),
      KOKKOS_LAMBDA(const LO e) {
        const fem::Cell c = fem::cell_of(view, dim, e);
        Real r[fem::max_nodes];
        if (!(fem::load(c, f, r) > 0)) Kokkos::atomic_inc(&bad());
        for (int a = 0; a < c.n; ++a) out(at(e) + a) = r[a];
      });
  return out;
}

/// Each owned row sums its cells' entries, in the order of `around`.
template <class MemSpace>
void gather_rows(LO rows, const View1<Offset, MemSpace>& offsets, const View1<Offset, MemSpace>& around_offsets,
                 const View1<LO, MemSpace>& around_cells, const View1<std::uint8_t, MemSpace>& around_corner,
                 const View1<Offset, MemSpace>& slot_offsets, const View1<LO, MemSpace>& slots,
                 const View1<Offset, MemSpace>& matrix_offsets, const View1<Real, MemSpace>& element,
                 const View1<Real, MemSpace>& values) {
  using Exec = typename MemSpace::execution_space;
  Kokkos::parallel_for(
      "soma::FemSpace::gather", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows), KOKKOS_LAMBDA(const LO v) {
        for (Offset k = offsets(v); k < offsets(v + 1); ++k) values(k) = 0;
        for (Offset t = around_offsets(v); t < around_offsets(v + 1); ++t) {
          const Offset first = slot_offsets(t), n = slot_offsets(t + 1) - first;
          const Offset base = matrix_offsets(around_cells(t)) + around_corner(t) * n;
          for (Offset b = 0; b < n; ++b) values(offsets(v) + slots(first + b)) += element(base + b);
        }
      });
  Exec{}.fence();
}

/// Each owned vertex sums its cells' loads, in the order of `around`.
template <class MemSpace>
void gather_load(LO rows, const View1<Offset, MemSpace>& around_offsets, const View1<LO, MemSpace>& around_cells,
                 const View1<std::uint8_t, MemSpace>& around_corner, const View1<Offset, MemSpace>& vector_offsets,
                 const View1<Real, MemSpace>& element, const View1<Real, MemSpace>& b) {
  using Exec = typename MemSpace::execution_space;
  Kokkos::parallel_for(
      "soma::FemSpace::gather_load", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows),
      KOKKOS_LAMBDA(const LO v) {
        Real s = 0;
        for (Offset t = around_offsets(v); t < around_offsets(v + 1); ++t) {
          s += element(vector_offsets(around_cells(t)) + around_corner(t));
        }
        b(v) = s;
      });
  Exec{}.fence();
}

/// Cells cells[first, last) add their stiffness matrices to the owned rows,
/// atomically or not; `bad` counts as element_matrices does. The caller
/// fences.
template <bool Atomic, class MemSpace>
void scatter_cells(const MeshView<MemSpace>& view, LO rows, const View1<LO, MemSpace>& cells, Offset first,
                   Offset last, const View1<Offset, MemSpace>& offsets, const View1<Offset, MemSpace>& slot_offsets,
                   const View1<LO, MemSpace>& cell_slots, const View1<Real, MemSpace>& values,
                   const Kokkos::View<LO, MemSpace>& bad) {
  using Exec = typename MemSpace::execution_space;
  const int dim = view.dim;
  Kokkos::parallel_for(
      "soma::FemSpace::scatter", Kokkos::RangePolicy<Exec, Kokkos::IndexType<Offset>>(first, last),
      KOKKOS_LAMBDA(const Offset i) {
        const LO e = cells(i);
        const fem::Cell c = fem::cell_of(view, dim, e);
        Real k[fem::max_nodes][fem::max_nodes];
        if (!(fem::stiffness(c, k) > 0)) Kokkos::atomic_inc(&bad());
        for (int a = 0; a < c.n; ++a) {
          const LO v = view.vertex(dim, e, a);
          if (v >= rows) continue;
          const Offset row = offsets(v), base = slot_offsets(i) + a * c.n;
          for (int b = 0; b < c.n; ++b) {
            Real& entry = values(row + cell_slots(base + b));
            if constexpr (Atomic) {
              Kokkos::atomic_add(&entry, k[a][b]);
            } else {
              entry += k[a][b];
            }
          }
        }
      });
}

template <class MemSpace>
Real local_dot(LO rows, const View1<const Real, MemSpace>& x, const View1<const Real, MemSpace>& y) {
  using Exec = typename MemSpace::execution_space;
  Real s = 0;
  Kokkos::parallel_reduce(
      "soma::FemSpace::dot", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows),
      KOKKOS_LAMBDA(const LO i, Real& acc) { acc += x(i) * y(i); }, s);
  return s;
}

/// The products x[i] y[i] over the rows, summed exactly.
template <class MemSpace>
ExactSums<1> local_dot_exact(LO rows, const View1<const Real, MemSpace>& x, const View1<const Real, MemSpace>& y) {
  return reduce_exact<1, typename MemSpace::execution_space>(
      "soma::FemSpace::dot_exact", 0, rows, KOKKOS_LAMBDA(const LO i, ExactSums<1>& acc) { acc.add(0, x(i) * y(i)); });
}

}  // namespace detail

/// A global numbering of the vertices: owned vertices take first, first + 1,
/// ... rank by rank, copies and ghosts their owners' numbers; total over all.
template <class MemSpace>
struct Numbering {
  View1<GO, MemSpace> index;  ///< per vertex here
  GO first = 0;
  GO total = 0;
};

template <class MemSpace>
class FemSpace {
 public:
  /// Collective. Hanging vertices (refine_marked without conforming) are
  /// constrained to their masters (hanging_constraints): the space assembles
  /// P^T A P and P^T b on the free vertices' rows, a hanging vertex's row the
  /// identity's with load 0, by gather whatever `assembly`; distribute() sets
  /// the hanging values after a solve. On more than one rank that needs two
  /// ghost layers through vertices.
  ///
  /// Throws std::invalid_argument on every rank if any rank's mesh is not of
  /// dimension 2 or 3, has unremoved edits, holds a cell without elements here
  /// (only tri, quad, tet, hex) or one whose Jacobian is not positive
  /// throughout, a quad side that two triangles tile, a hanging vertex without
  /// a birth record ("refine:birth"; infer_births), or, on more than one rank,
  /// has no ghost layer through vertices or too few to hold the masters of the
  /// hanging vertices it needs and the cells around them; or if ranks differ
  /// in soma::reproducible().
  /// `assembly` is how laplacian() assembles, gather when reproducible.
  FemSpace(const Comm& comm, const Mesh<MemSpace>& mesh, Assembly assembly = Assembly::gather)
      : view_(mesh.view()), reproducible_(soma::reproducible()) {
    assembly_ = reproducible_ ? Assembly::gather : assembly;
    const Mesh<Kokkos::HostSpace>* host = nullptr;
    std::optional<Mesh<Kokkos::HostSpace>> mirror;
    if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      host = &mesh;
    } else {
      mirror.emplace(mesh.template mirror<Kokkos::HostSpace>());
      host = &*mirror;
    }
    const bool bad = mesh.edited() || !detail::fem_supported(*host) ||
                     (comm.size() > 1 && (mesh.ghost_layers() < 1 || mesh.ghost_bridge_dim() != 0));
    const Real exact = reproducible_ ? Real{1} : Real{0};
    {
      Real flags[3] = {bad ? Real{1} : Real{0}, exact, -exact};
      comm.allreduce(std::span<Real>(flags), Op::max);
      if (flags[1] != -flags[2]) throw std::invalid_argument("FemSpace: ranks differ in soma::reproducible()");
      if (flags[0] != 0) {
        throw std::invalid_argument(
            "FemSpace: a rank's mesh is not of dimension 2 or 3, has edits or high-order nodes, holds a cell other "
            "than a tri, quad, tet or hex or, on more than one rank, has no ghost layer through vertices");
      }
    }
    // Hanging vertices: constrained by their births; any found by geometry
    // without one, or a quad against two triangles, refused.
    const HangingConstraints hc = hanging_constraints(comm, *host);
    const detail::Nonconforming nc = detail::nonconforming(*host);
    LO unrecorded = 0;
    for (const LO v : nc.vertices) {
      unrecorded += std::binary_search(hc.vertex.begin(), hc.vertex.end(), v) ? 0 : 1;
    }
    constrained_ = comm.allreduce(static_cast<Real>(hc.vertex.size()), Op::max) > 0;
    if (constrained_) assembly_ = Assembly::gather;
    detail::FemPlan p = detail::fem_plan(*host, assembly_, constrained_ ? &hc : nullptr);
    Real flags[7] = {p.fits ? Real{0} : Real{1},
                     static_cast<Real>(p.orphans),
                     p.fits ? static_cast<Real>(detail::bad_cells(view_)) : Real{0},
                     static_cast<Real>(nc.split_quads),
                     static_cast<Real>(unrecorded),
                     static_cast<Real>(p.missing),
                     0};
    comm.allreduce(std::span<Real>(flags), Op::max);
    if (flags[0] != 0) throw std::invalid_argument("FemSpace: a rank's element matrices do not fit Offset");
    if (flags[1] != 0) throw std::invalid_argument("FemSpace: a rank owns a vertex that no cell there has");
    if (flags[2] != 0) throw std::invalid_argument("FemSpace: a rank holds an inverted or degenerate cell");
    if (flags[3] != 0) {
      throw std::invalid_argument(
          "FemSpace: a rank holds a quad side that two triangles tile (a hex against two tets): the space would not "
          "be continuous there");
    }
    if (flags[4] != 0) {
      throw std::invalid_argument(
          "FemSpace: a rank holds a hanging vertex without a record of the side it was born on (\"refine:birth\"): "
          "call infer_births, or refine with MarkedOptions::conforming");
    }
    if (flags[5] != 0) {
      throw std::invalid_argument(
          "FemSpace: a rank lacks a master of a hanging vertex it needs: add a ghost layer");
    }

    rows_ = mesh.segment_table(0).owned_end;
    cols_ = mesh.count(0);
    offsets_ = detail::to_space<Offset, MemSpace>(p.offsets, "soma::FemSpace::offsets");
    columns_ = detail::to_space<LO, MemSpace>(p.columns, "soma::FemSpace::columns");
    around_offsets_ = detail::to_space<Offset, MemSpace>(p.around_offsets, "soma::FemSpace::around_offsets");
    around_cells_ = detail::to_space<LO, MemSpace>(p.around_cells, "soma::FemSpace::around_cells");
    around_corner_ = detail::to_space<std::uint8_t, MemSpace>(p.around_corner, "soma::FemSpace::around_corner");
    slot_offsets_ = detail::to_space<Offset, MemSpace>(p.slot_offsets, "soma::FemSpace::slot_offsets");
    slots_ = detail::to_space<LO, MemSpace>(p.slots, "soma::FemSpace::slots");
    matrix_offsets_ = detail::to_space<Offset, MemSpace>(p.matrix_offsets, "soma::FemSpace::matrix_offsets");
    matrix_total_ = p.matrix_offsets.back();
    vector_total_ = p.vector_offsets.back();
    vector_offsets_ = detail::to_space<Offset, MemSpace>(p.vector_offsets, "soma::FemSpace::vector_offsets");
    cell_slot_offsets_ = detail::to_space<Offset, MemSpace>(p.cell_slot_offsets, "soma::FemSpace::cell_slot_offsets");
    cell_slots_ = detail::to_space<LO, MemSpace>(p.cell_slots, "soma::FemSpace::cell_slots");
    group_offsets_ = p.group_offsets;
    group_cells_ = detail::to_space<LO, MemSpace>(p.group_cells, "soma::FemSpace::group_cells");
    if (comm.size() > 1) halo_.emplace(comm, mesh, 0);
    if (constrained_) {
      // Every cell around a hanging vertex a row takes is here (its owner's
      // count), and every master's row took every hanging vertex it masters.
      View2<Real, Kokkos::HostSpace> star("soma::FemSpace::star", p.star.size(), 1);
      for (std::size_t v = 0; v < p.star.size(); ++v) star(v, 0) = static_cast<Real>(p.star[v]);
      if (halo_) {
        if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
          halo_->update(comm, star);
        } else {
          auto d = Kokkos::create_mirror_view_and_copy(MemSpace{}, star);
          halo_->update(comm, d);
          Kokkos::deep_copy(star, d);
        }
      }
      LO short_star = 0;
      for (const LO h : p.needed) {
        short_star += static_cast<Real>(p.star[static_cast<std::size_t>(h)]) == star(static_cast<std::size_t>(h), 0) ? 0 : 1;
      }
      if (comm.allreduce(static_cast<Real>(short_star), Op::max) != 0 ||
          allreduce_exact(comm, p.balance)[0] != 0) {
        throw std::invalid_argument(
            "FemSpace: a rank lacks cells around a hanging vertex a row it owns needs: add a ghost layer");
      }
      term_offsets_ = detail::to_space<Offset, MemSpace>(p.term_offsets, "soma::FemSpace::term_offsets");
      term_entry_ = detail::to_space<Offset, MemSpace>(p.term_entry, "soma::FemSpace::term_entry");
      term_slot_ = detail::to_space<LO, MemSpace>(p.term_slot, "soma::FemSpace::term_slot");
      term_weight_ = detail::to_space<Real, MemSpace>(p.term_weight, "soma::FemSpace::term_weight");
      load_offsets_ = detail::to_space<Offset, MemSpace>(p.load_offsets, "soma::FemSpace::load_offsets");
      load_entry_ = detail::to_space<Offset, MemSpace>(p.load_entry, "soma::FemSpace::load_entry");
      load_weight_ = detail::to_space<Real, MemSpace>(p.load_weight, "soma::FemSpace::load_weight");
      fixed_rows_ = detail::to_space<LO, MemSpace>(p.fixed_rows, "soma::FemSpace::fixed_rows");
      hang_ = detail::to_space<LO, MemSpace>(p.hang, "soma::FemSpace::hang");
      hang_offsets_ = detail::to_space<Offset, MemSpace>(p.hang_offsets, "soma::FemSpace::hang_offsets");
      hang_masters_ = detail::to_space<LO, MemSpace>(p.hang_masters, "soma::FemSpace::hang_masters");
      hang_weights_ = detail::to_space<Real, MemSpace>(p.hang_weights, "soma::FemSpace::hang_weights");
      hanging_ = detail::to_space<std::uint8_t, MemSpace>(p.hanging, "soma::FemSpace::hanging");
    }
    // Owners' verdicts: a ghost's far corners look like skin here.
    boundary_ = detail::to_space<std::uint8_t, MemSpace>(p.skin, "soma::FemSpace::boundary");
    if (halo_) halo_->update(comm, View2<std::uint8_t, MemSpace>(boundary_.data(), boundary_.extent(0), 1));
  }

  LO rows() const noexcept { return rows_; }  // owned vertices
  LO cols() const noexcept { return cols_; }  // all vertices here
  Assembly assembly() const noexcept { return assembly_; }  // as laplacian() assembles
  bool reproducible() const noexcept { return reproducible_; }  // soma::reproducible() when made

  /// A vector of zeros, one entry per vertex.
  View1<Real, MemSpace> vector(const char* label = "soma::FemSpace::vector") const {
    return View1<Real, MemSpace>(label, static_cast<std::size_t>(cols_));
  }

  /// The rows' sparsity, values zero: each owned vertex with the vertices
  /// that share a cell with it.
  CsrMatrix<MemSpace> pattern() const {
    return CsrMatrix<MemSpace>{rows_, cols_, offsets_, columns_,
                               View1<Real, MemSpace>("soma::FemSpace::values", columns_.extent(0))};
  }

  /// Collective: the stiffness matrix of -div grad (no boundary conditions).
  /// Throws std::invalid_argument on every rank if a cell it assembles has
  /// inverted since the space was made (gather assembles every cell here,
  /// atomic and colored those with an owned corner), once assembly is done.
  CsrMatrix<MemSpace> laplacian(const Comm& comm) const {
    CsrMatrix<MemSpace> a = pattern();
    const Kokkos::View<LO, MemSpace> bad("soma::FemSpace::bad");
    if (constrained_) {
      using Exec = typename MemSpace::execution_space;
      const auto element = detail::element_matrices(view_, matrix_offsets_, matrix_total_, bad);
      const auto offsets = offsets_, to = term_offsets_, entry = term_entry_;
      const auto slot = term_slot_;
      const auto weight = term_weight_;
      const auto values = a.values;
      Kokkos::parallel_for(
          "soma::FemSpace::gather_constrained", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
          KOKKOS_LAMBDA(const LO v) {
            for (Offset k = offsets(v); k < offsets(v + 1); ++k) values(k) = 0;
            for (Offset t = to(v); t < to(v + 1); ++t) values(offsets(v) + slot(t)) += weight(t) * element(entry(t));
          });
      const auto fixed = fixed_rows_;
      Kokkos::parallel_for(
          "soma::FemSpace::fixed_rows", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, static_cast<LO>(fixed.extent(0))),
          KOKKOS_LAMBDA(const LO i) { values(offsets(fixed(i))) = 1; });
      Exec{}.fence();
      check_cells(comm, bad);
      return a;
    }
    if (assembly_ == Assembly::gather) {
      const auto element = detail::element_matrices(view_, matrix_offsets_, matrix_total_, bad);
      detail::gather_rows(rows_, offsets_, around_offsets_, around_cells_, around_corner_, slot_offsets_, slots_,
                          matrix_offsets_, element, a.values);
      check_cells(comm, bad);
      return a;
    }
    // Launches on one instance run in turn: the colors need no fence apart.
    for (std::size_t g = 0; g + 1 < group_offsets_.size(); ++g) {
      const Offset first = group_offsets_[g], last = group_offsets_[g + 1];
      if (assembly_ == Assembly::atomic) {
        detail::scatter_cells<true>(view_, rows_, group_cells_, first, last, offsets_, cell_slot_offsets_,
                                    cell_slots_, a.values, bad);
      } else {
        detail::scatter_cells<false>(view_, rows_, group_cells_, first, last, offsets_, cell_slot_offsets_,
                                     cell_slots_, a.values, bad);
      }
    }
    typename MemSpace::execution_space{}.fence();
    check_cells(comm, bad);
    return a;
  }

  /// Collective: b[v] = integral of f N_v for the owned vertices (0 for the
  /// others); f(x, y, z) device-callable. Throws as laplacian() does, f having
  /// been called at the points of cells checked by then.
  template <class F>
  View1<Real, MemSpace> load(const Comm& comm, const F& f) const {
    View1<Real, MemSpace> b = vector("soma::FemSpace::load");
    const Kokkos::View<LO, MemSpace> bad("soma::FemSpace::bad");
    const auto element = detail::element_loads(view_, vector_offsets_, vector_total_, f, bad);
    if (constrained_) {
      using Exec = typename MemSpace::execution_space;
      const auto to = load_offsets_, entry = load_entry_;
      const auto weight = load_weight_;
      Kokkos::parallel_for(
          "soma::FemSpace::load_constrained", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
          KOKKOS_LAMBDA(const LO v) {
            Real s = 0;
            for (Offset t = to(v); t < to(v + 1); ++t) s += weight(t) * element(entry(t));
            b(v) = s;
          });
      Exec{}.fence();
    } else {
      detail::gather_load(rows_, around_offsets_, around_cells_, around_corner_, vector_offsets_, element, b);
    }
    check_cells(comm, bad);
    return b;
  }

  /// Per vertex, 1 on the mesh's boundary: a corner of a side of dimension
  /// dim - 1 that only one cell has, as its owner finds it.
  View1<const std::uint8_t, MemSpace> boundary() const { return boundary_; }

  /// Whether the mesh has hanging vertices: the space is then constrained
  /// (see the class comment), and assembles by gather.
  bool constrained() const noexcept { return constrained_; }

  /// Per vertex, 1 where it hangs (empty when not constrained()).
  View1<const std::uint8_t, MemSpace> hanging() const { return hanging_; }

  /// Collective: each hanging vertex's value from its masters', u_h =
  /// sum_k w_k u_{m_k}, then each owner's entries to its copies and ghosts
  /// (two updates). Call it on a solution of the constrained system, whose
  /// hanging rows hold no value of their own. Nothing when not constrained().
  void distribute(const Comm& comm, const View1<Real, MemSpace>& x) const {
    if (!constrained_) return;
    using Exec = typename MemSpace::execution_space;
    update(comm, x);
    const auto hang = hang_, offsets = hang_offsets_, masters = hang_masters_;
    const auto weights = hang_weights_;
    Kokkos::parallel_for(
        "soma::FemSpace::distribute", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, static_cast<LO>(hang.extent(0))),
        KOKKOS_LAMBDA(const LO i) {
          Real s = 0;
          for (Offset k = offsets(i); k < offsets(i + 1); ++k) s += weights(k) * x(masters(k));
          x(hang(i)) = s;
        });
    Exec{}.fence();
    update(comm, x);
  }

  /// Collective: each owner's entries of x overwrite its copies and ghosts.
  void update(const Comm& comm, const View1<Real, MemSpace>& x) const {
    if (x.extent(0) != static_cast<std::size_t>(cols_)) throw std::invalid_argument("FemSpace: a vector not of cols()");
    if (halo_) halo_->update(comm, View2<Real, MemSpace>(x.data(), x.extent(0), 1));
  }

  /// Collective: y = A x over the owned rows, after updating x.
  void multiply(const Comm& comm, const CsrMatrix<MemSpace>& a, const View1<Real, MemSpace>& x,
                const View1<Real, MemSpace>& y) const {
    update(comm, x);
    soma::multiply(a, View1<const Real, MemSpace>(x), y);
  }

  /// Collective: the vertices numbered rank by rank, as global matrices and
  /// vectors number their rows (one exscan, one allreduce, one update).
  Numbering<MemSpace> numbering(const Comm& comm) const {
    using Exec = typename MemSpace::execution_space;
    Numbering<MemSpace> out;
    out.first = comm.exscan(static_cast<GO>(rows_));
    out.total = comm.allreduce(static_cast<GO>(rows_), Op::sum);
    out.index = View1<GO, MemSpace>("soma::FemSpace::numbering", static_cast<std::size_t>(cols_));
    const auto index = out.index;
    const GO first = out.first;
    Kokkos::parallel_for(
        "soma::FemSpace::numbering", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
        KOKKOS_LAMBDA(const LO v) { index(v) = first + v; });
    Exec{}.fence();
    if (halo_) halo_->update(comm, View2<GO, MemSpace>(index.data(), index.extent(0), 1));
    return out;
  }

  /// Collective: the sum of x[i] y[i] over every rank's owned vertices; when
  /// reproducible(), the products' exact sum rounded once. One allreduce.
  Real dot(const Comm& comm, const View1<const Real, MemSpace>& x, const View1<const Real, MemSpace>& y) const {
    if (reproducible_) return allreduce_exact(comm, detail::local_dot_exact(rows_, x, y))[0];
    return comm.allreduce(detail::local_dot(rows_, x, y), Op::sum);
  }

 private:
  /// Collective, after the kernels that counted `bad` have finished.
  void check_cells(const Comm& comm, const Kokkos::View<LO, MemSpace>& bad) const {
    LO here = 0;
    Kokkos::deep_copy(here, bad);
    if (comm.allreduce(static_cast<Real>(here), Op::max) != 0) {
      throw std::invalid_argument("FemSpace: a rank holds an inverted or degenerate cell");
    }
  }

  MeshView<MemSpace> view_;
  bool reproducible_ = false;
  Assembly assembly_ = Assembly::gather;
  LO rows_ = 0;
  LO cols_ = 0;
  View1<Offset, MemSpace> offsets_;
  View1<LO, MemSpace> columns_;
  View1<Offset, MemSpace> around_offsets_;
  View1<LO, MemSpace> around_cells_;
  View1<std::uint8_t, MemSpace> around_corner_;
  View1<Offset, MemSpace> slot_offsets_;
  View1<LO, MemSpace> slots_;
  View1<Offset, MemSpace> matrix_offsets_;
  View1<Offset, MemSpace> vector_offsets_;
  Offset matrix_total_ = 0;
  Offset vector_total_ = 0;
  View1<Offset, MemSpace> cell_slot_offsets_;
  View1<LO, MemSpace> cell_slots_;
  std::vector<Offset> group_offsets_;
  View1<LO, MemSpace> group_cells_;
  View1<std::uint8_t, MemSpace> boundary_;
  std::optional<HaloExchange<MemSpace>> halo_;
  bool constrained_ = false;
  View1<Offset, MemSpace> term_offsets_;
  View1<Offset, MemSpace> term_entry_;
  View1<LO, MemSpace> term_slot_;
  View1<Real, MemSpace> term_weight_;
  View1<Offset, MemSpace> load_offsets_;
  View1<Offset, MemSpace> load_entry_;
  View1<Real, MemSpace> load_weight_;
  View1<LO, MemSpace> fixed_rows_;
  View1<LO, MemSpace> hang_;
  View1<Offset, MemSpace> hang_offsets_;
  View1<LO, MemSpace> hang_masters_;
  View1<Real, MemSpace> hang_weights_;
  View1<std::uint8_t, MemSpace> hanging_;
};

}  // namespace soma

#endif  // SOMA_ALGORITHMS_FEM_SPACE_HPP_
