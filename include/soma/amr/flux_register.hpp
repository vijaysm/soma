// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_FLUX_REGISTER_HPP_
#define SOMA_AMR_FLUX_REGISTER_HPP_

// A flux register keeps conservative schemes conservative across a
// coarse-fine interface. Its faces are the coarse faces on the boundary of
// the fine level whose coarse cell outside is not under the fine level (and
// not outside the domain). A step records the coarse fluxes there
// (set_coarse) and the fine fluxes through the same faces, averaged over the
// r^(dim-1) fine faces of each (add_fine, called once per fine step when
// fine steps are smaller); reflux then corrects the coarse cells outside:
//
//   u += -(fine - coarse)  for the cell below a face (the face is its high face)
//   u += +(fine - coarse)  for the cell above it
//
// with the fluxes scaled by the caller (dt / dx of the coarse level, times
// the fraction of the coarse step for each fine step). Fluxes are face
// fields, one per direction (IndexType::face(d)); the state is cell data.
// Each coarse cell is updated once per direction and side, so the result
// does not depend on the thread count.
//
// Over many ranks the rows of a face live twice: its coarse flux with the
// owner of the coarse box (set_coarse and reflux are local there), its fine
// sum with the owner of the fine box (add_fine is local there). reflux first
// sends the fine sums to the coarse side, one message per neighbour; every
// face has one fine box, so no sum crosses ranks.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

// Coarse faces `faces` (normal dir, face-typed, unshifted) on the boundary of
// fine box `fine`; the coarse cells outside lie in coarse box `coarse` at
// their position plus `shift` (a periodic image).
struct RegisterFaces {
  int coarse = 0;
  int fine = 0;
  Box faces;
  IntVect shift;
  int dir = 0;
  int side = 0;  // 0: the outside cell is below the face, 1: above
};

class FluxRegister {
 public:
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  static constexpr int default_tag = 32624;

  FluxRegister() = default;
  FluxRegister(const BoxArray& fine, const BoxArray& coarse, int ratio, Periodicity periodic, int ncomp);
  // The part of comm.rank(); reflux is then collective over the neighbours.
  FluxRegister(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner, const BoxArray& coarse,
               const Distribution& coarse_owner, int ratio, Periodicity periodic, int ncomp, int tag = default_tag);

  // Every piece of faces, on every rank, boxes by their ids.
  const std::vector<RegisterFaces>& faces() const { return pieces_; }
  int ratio() const { return ratio_; }
  int ncomp() const { return ncomp_; }
  std::int64_t num_faces() const { return start_.back(); }
  // This rank's rows: faces on the coarse boxes it owns, and on its fine boxes.
  std::int64_t num_coarse_rows() const { return coarse_start_.back(); }
  std::int64_t num_fine_rows() const { return fine_start_.back(); }

  // Zeroes both sides of the register.
  void reset() const;
  // coarse_flux: the coarse fluxes normal to dir, scaled by `scale`.
  template <class T>
  void set_coarse(int dir, const LevelField<T>& coarse_flux, double scale) const;
  // fine_flux: fine fluxes normal to dir; their average over each coarse
  // face, times scale, is added.
  template <class T>
  void add_fine(int dir, const LevelField<T>& fine_flux, double scale) const;
  // Corrects the coarse cells outside the fine level.
  template <class T>
  void reflux(const LevelField<T>& state) const;

 private:
  void build(const std::optional<Comm>& comm, const BoxArray& fine, const std::vector<int>& fine_owner,
             const BoxArray& coarse, const std::vector<int>& coarse_owner, Periodicity periodic, int tag);
  void check(const char* what, const std::vector<int>& owner, const Distribution& field_owner, int field_rank) const;

  using Rows = Kokkos::View<double**, Kokkos::LayoutLeft, DeviceSpace>;  // (face, component)
  using Index = Kokkos::View<const std::int64_t*, DeviceSpace>;

  int ratio_ = 2;
  int ncomp_ = 1;
  int dim_ = 3;
  int rank_ = 0;
  int tag_ = default_tag;
  std::optional<Comm> comm_;
  std::vector<int> fine_owner_, coarse_owner_;
  std::vector<RegisterFaces> pieces_;
  std::vector<std::int64_t> start_{0};
  // This rank's pieces on its coarse boxes and on its fine boxes, by the
  // fields' local indices, with their rows.
  std::vector<std::int64_t> coarse_start_{0}, fine_start_{0};
  Kokkos::View<const RegisterFaces*, DeviceSpace> d_coarse_pieces_, d_fine_pieces_;
  Index d_coarse_start_, d_fine_start_;
  Rows coarse_, fine_, fine_at_coarse_;
  // reflux's move of fine rows to coarse rows: local pairs, rows sent (by
  // destination) and rows received (by source).
  Index local_from_, local_to_, send_rows_, recv_rows_;
  transport::CommPlan messages_;
};

namespace detail {

// f(piece, face point, register row) over the faces of one direction and
// side (side < 0: both). A free function: CUDA refuses device lambdas in
// private member functions.
template <class F>
void for_register_faces(const Kokkos::View<const RegisterFaces*, FluxRegister::DeviceSpace>& pieces,
                        const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& start, std::int64_t num_faces,
                        int dir, int side, const char* label, const F& f) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int n = static_cast<int>(pieces.extent(0));
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, num_faces), KOKKOS_LAMBDA(std::int64_t p) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (start(mid) <= p) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        const RegisterFaces& pc = pieces(lo);
        if (pc.dir != dir || (side >= 0 && pc.side != side)) return;
        f(pc, detail::point_of(pc.faces, p - start(lo)), p);
      });
  Kokkos::fence(label);
}

// Copies rows `from` of a to rows `to` of b locally, and rows sent of a to
// the rows received of b on the other ranks along `messages`.
void move_rows(const Kokkos::View<double**, Kokkos::LayoutLeft, FluxRegister::DeviceSpace>& a,
               const Kokkos::View<double**, Kokkos::LayoutLeft, FluxRegister::DeviceSpace>& b,
               const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& from,
               const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& to,
               const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& sent,
               const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& received,
               const transport::CommPlan& messages, const std::optional<Comm>& comm, int tag);

}  // namespace detail

template <class T>
void FluxRegister::set_coarse(int dir, const LevelField<T>& coarse_flux, double scale) const {
  if (!(coarse_flux.type() == IndexType::face(dir)) || coarse_flux.ncomp() < ncomp_) {
    throw std::invalid_argument("FluxRegister::set_coarse: expected a face field normal to the direction");
  }
  check("set_coarse", coarse_owner_, coarse_flux.distribution(), coarse_flux.rank());
  const auto v = coarse_flux.view();
  const auto reg = coarse_;
  const int ncomp = ncomp_;
  detail::for_register_faces(d_coarse_pieces_, d_coarse_start_, num_coarse_rows(), dir, -1, "amr::FluxRegister::set_coarse", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
    const auto patch = v.patch(pc.coarse);
    for (int c = 0; c < ncomp; ++c) {
      reg(row, c) = scale * static_cast<double>(patch(q[0] + pc.shift[0], q[1] + pc.shift[1], q[2] + pc.shift[2], c));
    }
  });
}

template <class T>
void FluxRegister::add_fine(int dir, const LevelField<T>& fine_flux, double scale) const {
  if (!(fine_flux.type() == IndexType::face(dir)) || fine_flux.ncomp() < ncomp_) {
    throw std::invalid_argument("FluxRegister::add_fine: expected a face field normal to the direction");
  }
  check("add_fine", fine_owner_, fine_flux.distribution(), fine_flux.rank());
  const auto v = fine_flux.view();
  const auto reg = fine_;
  const int ncomp = ncomp_, r = ratio_, dim = dim_;
  detail::for_register_faces(d_fine_pieces_, d_fine_start_, num_fine_rows(), dir, -1, "amr::FluxRegister::add_fine", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
    const auto patch = v.patch(pc.fine);
    int len[3] = {1, 1, 1};
    for (int d = 0; d < dim; ++d) len[d] = d == dir ? 1 : r;
    for (int c = 0; c < ncomp; ++c) {
      double sum = 0;
      for (int k = 0; k < len[2]; ++k) {
        for (int j = 0; j < len[1]; ++j) {
          for (int i = 0; i < len[0]; ++i) sum += static_cast<double>(patch(q[0] * r + i, q[1] * r + j, dim > 2 ? q[2] * r + k : 0, c));
        }
      }
      reg(row, c) += scale * sum / (len[0] * len[1] * len[2]);
    }
  });
}

template <class T>
void FluxRegister::reflux(const LevelField<T>& state) const {
  if (!(state.type() == IndexType::cell()) || state.ncomp() < ncomp_) {
    throw std::invalid_argument("FluxRegister::reflux: expected cell data");
  }
  check("reflux", coarse_owner_, state.distribution(), state.rank());
  detail::move_rows(fine_, fine_at_coarse_, local_from_, local_to_, send_rows_, recv_rows_, messages_, comm_, tag_);
  const auto v = state.view();
  const auto rc = coarse_;
  const auto rf = fine_at_coarse_;
  const int ncomp = ncomp_;
  for (int dir = 0; dir < dim_; ++dir) {
    for (int side = 0; side < 2; ++side) {
      detail::for_register_faces(d_coarse_pieces_, d_coarse_start_, num_coarse_rows(), dir, side, "amr::FluxRegister::reflux", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
        IntVect cell = q + pc.shift;
        if (pc.side == 0) --cell[pc.dir];  // the cell below face q
        const auto patch = v.patch(pc.coarse);
        const double sign = pc.side == 0 ? -1.0 : 1.0;
        for (int c = 0; c < ncomp; ++c) {
          patch(cell[0], cell[1], cell[2], c) += static_cast<T>(sign * (rf(row, c) - rc(row, c)));
        }
      });
    }
  }
}

}  // namespace soma::amr

#endif  // SOMA_AMR_FLUX_REGISTER_HPP_
