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

#include <cstdint>
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

  FluxRegister() = default;
  FluxRegister(const BoxArray& fine, const BoxArray& coarse, int ratio, Periodicity periodic, int ncomp);

  const std::vector<RegisterFaces>& faces() const { return pieces_; }
  int ratio() const { return ratio_; }
  int ncomp() const { return ncomp_; }
  std::int64_t num_faces() const { return start_.back(); }

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
  template <class F>
  void for_faces(int dir, int side, const char* label, const F& f) const;

  int ratio_ = 2;
  int ncomp_ = 1;
  int dim_ = 3;
  std::vector<RegisterFaces> pieces_;
  std::vector<std::int64_t> start_{0};
  Kokkos::View<const RegisterFaces*, DeviceSpace> d_pieces_;
  Kokkos::View<const std::int64_t*, DeviceSpace> d_start_;
  Kokkos::View<double**, Kokkos::LayoutLeft, DeviceSpace> coarse_, fine_;  // (face, component)
};

// f(piece, face point, register row) over the faces of one direction and
// side (side < 0: both).
template <class F>
void FluxRegister::for_faces(int dir, int side, const char* label, const F& f) const {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto pieces = d_pieces_;
  const auto start = d_start_;
  const int n = static_cast<int>(pieces_.size());
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, num_faces()), KOKKOS_LAMBDA(std::int64_t p) {
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

template <class T>
void FluxRegister::set_coarse(int dir, const LevelField<T>& coarse_flux, double scale) const {
  if (!(coarse_flux.type() == IndexType::face(dir)) || coarse_flux.ncomp() < ncomp_) {
    throw std::invalid_argument("FluxRegister::set_coarse: expected a face field normal to the direction");
  }
  const auto v = coarse_flux.view();
  const auto reg = coarse_;
  const int ncomp = ncomp_;
  for_faces(dir, -1, "amr::FluxRegister::set_coarse", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
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
  const auto v = fine_flux.view();
  const auto reg = fine_;
  const int ncomp = ncomp_, r = ratio_, dim = dim_;
  for_faces(dir, -1, "amr::FluxRegister::add_fine", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
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
  const auto v = state.view();
  const auto rc = coarse_;
  const auto rf = fine_;
  const int ncomp = ncomp_;
  for (int dir = 0; dir < dim_; ++dir) {
    for (int side = 0; side < 2; ++side) {
      for_faces(dir, side, "amr::FluxRegister::reflux", KOKKOS_LAMBDA(const RegisterFaces& pc, const IntVect& q, std::int64_t row) {
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
