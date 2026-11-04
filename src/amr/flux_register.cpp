// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/flux_register.hpp"

#include <string>

#include "bins.hpp"
#include "soma/amr/box_list.hpp"

namespace soma::amr {

namespace {

template <class U>
Kokkos::View<const U*, FluxRegister::DeviceSpace> to_device(const std::vector<U>& v, const std::string& label) {
  Kokkos::View<U*, FluxRegister::DeviceSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

}  // namespace

FluxRegister::FluxRegister(const BoxArray& fine, const BoxArray& coarse, int ratio, Periodicity periodic, int ncomp)
    : ratio_(ratio), ncomp_(ncomp), dim_(coarse.domain().dim()) {
  if (ratio < 2 || ncomp < 1) throw std::invalid_argument("FluxRegister: the ratio must be at least 2 and ncomp at least 1");
  if (!(fine.domain() == coarse.domain().refine(ratio))) {
    throw std::invalid_argument("FluxRegister: the fine domain is not the coarse domain refined by the ratio");
  }
  const Box& domain = coarse.domain();
  // The fine level on the coarse grid, with its periodic images.
  std::vector<IntVect> shifts{IntVect{}};
  for (int d = 0; d < dim_; ++d) {
    if (!periodic[d]) continue;
    const auto n = shifts.size();
    for (const int s : {-1, 1}) {
      for (std::size_t i = 0; i < n; ++i) {
        IntVect x = shifts[i];
        x[d] = s * domain.length(d);
        shifts.push_back(x);
      }
    }
  }
  BoxList under;
  for (std::size_t f = 0; f < fine.size(); ++f) {
    for (const IntVect& s : shifts) under.push_back(fine[f].coarsen(ratio).shift(s));
  }
  const detail::Bins bins(coarse.boxes(), domain);
  for (std::size_t f = 0; f < fine.size(); ++f) {
    const Box fc = fine[f].coarsen(ratio);
    for (int d = 0; d < dim_; ++d) {
      for (int side = 0; side < 2; ++side) {
        // The layer of coarse cells just outside this side of the fine box.
        const int at = side == 0 ? fc.lo(d) - 1 : fc.hi(d) + 1;
        const Box outside = fc.grow(d, fc.lo(d) - at, at - fc.hi(d));
        IntVect shift;
        if (at < domain.lo(d) || at > domain.hi(d)) {
          if (!periodic[d]) continue;
          shift[d] = at < domain.lo(d) ? domain.length(d) : -domain.length(d);
        }
        for (const Box& part : subtract(BoxList{outside}, under)) {
          const Box image = part.shift(shift);
          if (!covers(coarse.boxes(), image)) {
            throw std::invalid_argument("FluxRegister: coarse cells next to a fine box are not in the coarse level");
          }
          for (const int c : bins.near(image)) {
            const Box cells = image & coarse[static_cast<std::size_t>(c)];
            if (cells.empty()) continue;
            // The faces between those cells and the fine box, unshifted.
            const Box back = cells.shift(IntVect{} - shift);
            const int face = side == 0 ? back.lo(d) + 1 : back.lo(d);
            IntVect lo = back.lo(), hi = back.hi();
            lo[d] = face;
            hi[d] = face;
            const Box faces(lo, hi, dim_, IndexType::face(d));
            pieces_.push_back(RegisterFaces{c, static_cast<int>(f), faces, shift, d, side});
            start_.push_back(start_.back() + faces.num_pts());
          }
        }
      }
    }
  }
  d_pieces_ = to_device(pieces_, "amr::FluxRegister.faces");
  d_start_ = to_device(start_, "amr::FluxRegister.start");
  coarse_ = Kokkos::View<double**, Kokkos::LayoutLeft, DeviceSpace>("amr::FluxRegister.coarse", static_cast<std::size_t>(start_.back()), static_cast<std::size_t>(ncomp));
  fine_ = Kokkos::View<double**, Kokkos::LayoutLeft, DeviceSpace>("amr::FluxRegister.fine", static_cast<std::size_t>(start_.back()), static_cast<std::size_t>(ncomp));
}

void FluxRegister::reset() const {
  Kokkos::deep_copy(coarse_, 0.0);
  Kokkos::deep_copy(fine_, 0.0);
}

}  // namespace soma::amr
