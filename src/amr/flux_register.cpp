// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/flux_register.hpp"

#include <algorithm>
#include <span>
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
  build(std::nullopt, fine, std::vector<int>(fine.size(), 0), coarse, std::vector<int>(coarse.size(), 0), periodic,
        default_tag);
}

FluxRegister::FluxRegister(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner, const BoxArray& coarse,
                           const Distribution& coarse_owner, int ratio, Periodicity periodic, int ncomp, int tag)
    : ratio_(ratio), ncomp_(ncomp), dim_(coarse.domain().dim()) {
  const auto bad = [&](const std::vector<int>& owner, std::size_t n) {
    return owner.size() != n || std::any_of(owner.begin(), owner.end(), [&](int r) { return r < 0 || r >= comm.size(); });
  };
  if (bad(fine_owner.owner, fine.size()) || bad(coarse_owner.owner, coarse.size())) {
    throw std::invalid_argument("FluxRegister: a distribution does not give every box an owner in the communicator");
  }
  build(comm, fine, fine_owner.owner, coarse, coarse_owner.owner, periodic, tag);
}

void FluxRegister::build(const std::optional<Comm>& comm, const BoxArray& fine, const std::vector<int>& fine_owner,
                         const BoxArray& coarse, const std::vector<int>& coarse_owner, Periodicity periodic, int tag) {
  const int ratio = ratio_;
  if (ratio < 2 || ncomp_ < 1) throw std::invalid_argument("FluxRegister: the ratio must be at least 2 and ncomp at least 1");
  if (!(fine.domain() == coarse.domain().refine(ratio))) {
    throw std::invalid_argument("FluxRegister: the fine domain is not the coarse domain refined by the ratio");
  }
  comm_ = comm;
  tag_ = tag;
  rank_ = comm ? comm->rank() : 0;
  fine_owner_ = fine_owner;
  coarse_owner_ = coarse_owner;
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
  // This rank's pieces and rows, and the moves of fine rows to coarse rows:
  // per neighbour in the order of the pieces, which both sides share.
  const int nranks = comm ? comm->size() : 1;
  const auto locals = [&](const std::vector<int>& owner) {
    std::vector<int> out(owner.size(), -1);
    for (std::size_t i = 0, n = 0; i < owner.size(); ++i) {
      if (owner[i] == rank_) out[i] = static_cast<int>(n++);
    }
    return out;
  };
  const std::vector<int> fine_local = locals(fine_owner), coarse_local = locals(coarse_owner);
  std::vector<RegisterFaces> coarse_pieces, fine_pieces;
  std::vector<std::int64_t> from, to;
  std::vector<std::vector<std::int64_t>> sent(static_cast<std::size_t>(nranks)), received(static_cast<std::size_t>(nranks));
  for (std::size_t i = 0; i < pieces_.size(); ++i) {
    const RegisterFaces& pc = pieces_[i];
    const int fo = fine_owner[static_cast<std::size_t>(pc.fine)], co = coarse_owner[static_cast<std::size_t>(pc.coarse)];
    const std::int64_t n = start_[i + 1] - start_[i];
    const std::int64_t f0 = fine_start_.back(), c0 = coarse_start_.back();
    if (fo == rank_) {
      RegisterFaces x = pc;
      x.fine = fine_local[static_cast<std::size_t>(pc.fine)];
      fine_pieces.push_back(x);
      fine_start_.push_back(f0 + n);
    }
    if (co == rank_) {
      RegisterFaces x = pc;
      x.coarse = coarse_local[static_cast<std::size_t>(pc.coarse)];
      coarse_pieces.push_back(x);
      coarse_start_.push_back(c0 + n);
    }
    for (std::int64_t k = 0; k < n; ++k) {
      if (fo == rank_ && co == rank_) {
        from.push_back(f0 + k);
        to.push_back(c0 + k);
      } else if (fo == rank_) {
        sent[static_cast<std::size_t>(co)].push_back(f0 + k);
      } else if (co == rank_) {
        received[static_cast<std::size_t>(fo)].push_back(c0 + k);
      }
    }
  }
  std::vector<std::int64_t> send_rows, recv_rows;
  std::vector<transport::CommPlan::Run> runs_out, runs_in;
  for (int q = 0; q < nranks; ++q) {
    const auto& o = sent[static_cast<std::size_t>(q)];
    const auto& in = received[static_cast<std::size_t>(q)];
    if (!o.empty()) runs_out.push_back({q, o.size()});
    if (!in.empty()) runs_in.push_back({q, in.size()});
    send_rows.insert(send_rows.end(), o.begin(), o.end());
    recv_rows.insert(recv_rows.end(), in.begin(), in.end());
  }
  if (comm) messages_ = transport::CommPlan::from_counts(*comm, runs_out, runs_in);
  d_coarse_pieces_ = to_device(coarse_pieces, "amr::FluxRegister.coarse_faces");
  d_fine_pieces_ = to_device(fine_pieces, "amr::FluxRegister.fine_faces");
  d_coarse_start_ = to_device(coarse_start_, "amr::FluxRegister.coarse_start");
  d_fine_start_ = to_device(fine_start_, "amr::FluxRegister.fine_start");
  local_from_ = to_device(from, "amr::FluxRegister.local_from");
  local_to_ = to_device(to, "amr::FluxRegister.local_to");
  send_rows_ = to_device(send_rows, "amr::FluxRegister.send_rows");
  recv_rows_ = to_device(recv_rows, "amr::FluxRegister.recv_rows");
  const auto nc = static_cast<std::size_t>(ncomp_);
  coarse_ = Rows("amr::FluxRegister.coarse", static_cast<std::size_t>(coarse_start_.back()), nc);
  fine_at_coarse_ = Rows("amr::FluxRegister.fine_at_coarse", static_cast<std::size_t>(coarse_start_.back()), nc);
  fine_ = Rows("amr::FluxRegister.fine", static_cast<std::size_t>(fine_start_.back()), nc);
}

void FluxRegister::check(const char* what, const std::vector<int>& owner, const Distribution& field_owner,
                         int field_rank) const {
  if (field_owner.owner != owner || field_rank != rank_) {
    throw std::invalid_argument(std::string("FluxRegister::") + what + ": the field is not distributed as the register's boxes");
  }
}

void detail::move_rows(const Kokkos::View<double**, Kokkos::LayoutLeft, FluxRegister::DeviceSpace>& a,
                       const Kokkos::View<double**, Kokkos::LayoutLeft, FluxRegister::DeviceSpace>& b,
                       const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& from,
                       const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& to,
                       const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& sent,
                       const Kokkos::View<const std::int64_t*, FluxRegister::DeviceSpace>& received,
                       const transport::CommPlan& messages, const std::optional<Comm>& comm, int tag) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using Buffer = Kokkos::View<double*, FluxRegister::DeviceSpace>;
  const auto nc = static_cast<std::int64_t>(a.extent(1));
  const Buffer out(Kokkos::view_alloc("amr::FluxRegister.out", Kokkos::WithoutInitializing), sent.extent(0) * a.extent(1));
  const Buffer in(Kokkos::view_alloc("amr::FluxRegister.in", Kokkos::WithoutInitializing), received.extent(0) * a.extent(1));
  Kokkos::parallel_for(
      "amr::FluxRegister.pack", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, static_cast<std::int64_t>(sent.extent(0))),
      KOKKOS_LAMBDA(std::int64_t i) {
        for (std::int64_t c = 0; c < nc; ++c) out(i * nc + c) = a(sent(i), c);
      });
  Kokkos::fence("amr::FluxRegister.pack");
  const auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  const auto in_h = Kokkos::create_mirror_view(in);
  transport::CommPlan::Pending pending;
  if (comm) {
    // One record per row: its ncomp values.
    pending = messages.begin_forward_bytes(
        *comm, std::as_bytes(std::span<const double>(out_h.data(), out_h.extent(0))),
        std::as_writable_bytes(std::span<double>(in_h.data(), in_h.extent(0))), a.extent(1) * sizeof(double), tag);
  }
  Kokkos::parallel_for(
      "amr::FluxRegister.local", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, static_cast<std::int64_t>(from.extent(0))),
      KOKKOS_LAMBDA(std::int64_t i) {
        for (std::int64_t c = 0; c < nc; ++c) b(to(i), c) = a(from(i), c);
      });
  pending.wait();
  Kokkos::deep_copy(in, in_h);
  Kokkos::parallel_for(
      "amr::FluxRegister.unpack", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, static_cast<std::int64_t>(received.extent(0))),
      KOKKOS_LAMBDA(std::int64_t i) {
        for (std::int64_t c = 0; c < nc; ++c) b(received(i), c) = in(i * nc + c);
      });
  Kokkos::fence("amr::FluxRegister.unpack");
}

void FluxRegister::reset() const {
  Kokkos::deep_copy(coarse_, 0.0);
  Kokkos::deep_copy(fine_, 0.0);
}

}  // namespace soma::amr
