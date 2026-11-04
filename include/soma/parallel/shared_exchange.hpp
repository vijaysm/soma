// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_SHARED_EXCHANGE_HPP_
#define SOMA_PARALLEL_SHARED_EXCHANGE_HPP_

// Exchange and reduction over the shared entities of one dimension of a
// distributed mesh (soma/parallel/distributed.hpp), built once from its
// soma:sharing field:
//
//   update(values)       each owner's values overwrite its copies;
//   reduce(values, op)   every holder's values are combined, the result on
//                        every holder.
//
// values is a View2 of count(dim) rows by any number of components. Both
// sides list the entities they share with a neighbour by gid, so the lists
// match without a handshake. reduce combines in ascending rank order, the
// holder's own value in its place, so every holder computes the same result
// bit for bit, whatever the op. Packing and unpacking are kernels over flat
// index arrays in the mesh's memory space; messages are staged through host
// memory (device-aware MPI is Phase 5) and use tags 32610 and 32611.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"

namespace soma {

namespace detail {

constexpr int update_tag = 32610;
constexpr int reduce_tag = 32611;

template <class T, class MemSpace>
void pack_rows(const View2<T, MemSpace>& values, const Kokkos::View<const LO*, MemSpace>& lids,
               const Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>& buf) {
  using Exec = typename MemSpace::execution_space;
  const auto ncomp = static_cast<LO>(values.extent(1));
  Kokkos::parallel_for(
      "soma::SharedExchange::pack", Kokkos::RangePolicy<Exec>(0, lids.extent(0)), KOKKOS_LAMBDA(const std::size_t k) {
        for (LO c = 0; c < ncomp; ++c) buf(k, c) = values(lids(k), c);
      });
}

template <class T, class MemSpace>
void unpack_rows(const View2<T, MemSpace>& values, const Kokkos::View<const LO*, MemSpace>& lids,
                 const Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>& buf) {
  using Exec = typename MemSpace::execution_space;
  const auto ncomp = static_cast<LO>(values.extent(1));
  Kokkos::parallel_for(
      "soma::SharedExchange::unpack", Kokkos::RangePolicy<Exec>(0, lids.extent(0)), KOKKOS_LAMBDA(const std::size_t k) {
        for (LO c = 0; c < ncomp; ++c) values(lids(k), c) = buf(k, c);
      });
}

template <class T>
KOKKOS_INLINE_FUNCTION T combine(T a, T b, Op op) {
  switch (op) {
    case Op::sum: return a + b;
    case Op::min: return b < a ? b : a;
    case Op::max: return a < b ? b : a;
    case Op::prod: return a * b;
  }
  return a;
}

// For shared entity i (entity first + i): the holders' values in ascending
// rank order, its own (rank me) in its place; holders other than me are
// rows offsets(i) .. offsets(i + 1) of `ranks`, their values at rows `at` of
// the receive buffer.
template <class T, class MemSpace>
void combine_rows(const View2<T, MemSpace>& values, LO first, const View1<const Offset, MemSpace>& offsets,
                  const View1<const int, MemSpace>& ranks, const Kokkos::View<const LO*, MemSpace>& at,
                  const Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>& buf, int me, Op op) {
  using Exec = typename MemSpace::execution_space;
  const auto ncomp = static_cast<LO>(values.extent(1));
  const auto n = offsets.extent(0) == 0 ? std::size_t{0} : offsets.extent(0) - 1;
  Kokkos::parallel_for(
      "soma::SharedExchange::combine", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const std::size_t i) {
        const LO e = first + static_cast<LO>(i);
        for (LO c = 0; c < ncomp; ++c) {
          const T own = values(e, c);
          bool placed = false;
          T acc{};
          bool any = false;
          for (Offset k = offsets(i); k < offsets(i + 1); ++k) {
            if (!placed && me < ranks(k)) {
              acc = any ? combine(acc, own, op) : own;
              any = placed = true;
            }
            const T v = buf(at(k), c);
            acc = any ? combine(acc, v, op) : v;
            any = true;
          }
          if (!placed) acc = any ? combine(acc, own, op) : own;
          values(e, c) = acc;
        }
      });
}

}  // namespace detail

template <class MemSpace>
class SharedExchange {
 public:
  // Throws std::invalid_argument if the mesh has no sharing fields on `dim`
  // (it was not built by build_local_part) or `dim` is out of range.
  SharedExchange(const Comm& comm, const Mesh<MemSpace>& mesh, int dim) : dim_(dim), me_(comm.rank()) {
    if (dim < 0 || dim > 3) throw std::invalid_argument("SharedExchange: dimension " + std::to_string(dim));
    const SegmentTable& t = mesh.segment_table(dim);
    count_ = mesh.count(dim);
    first_ = t.interior_end;
    const auto sharing = mesh.fields().template get_var<int>(sharing_field, dim);
    const auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sharing.offsets());
    const auto val = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sharing.values());
    const auto gid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.gid(dim));
    const LO nshared = t.shared_end - t.interior_end;
    if (static_cast<LO>(sharing.rows()) != nshared) {
      throw std::invalid_argument("SharedExchange: soma:sharing does not cover the shared entities");
    }
    // Per neighbour, (gid, entity) of what is sent and received.
    std::vector<int> nbrs(val.data(), val.data() + val.extent(0));
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    const auto slot = [&](int r) { return static_cast<std::size_t>(std::lower_bound(nbrs.begin(), nbrs.end(), r) - nbrs.begin()); };
    std::vector<std::vector<std::pair<GO, LO>>> up_send(nbrs.size()), up_recv(nbrs.size()), all(nbrs.size());
    for (LO i = 0; i < nshared; ++i) {
      const LO e = first_ + i;
      const auto row = static_cast<std::size_t>(i);
      for (auto k = static_cast<std::size_t>(off(row)); k < static_cast<std::size_t>(off(row + 1)); ++k) {
        const int r = val(k);
        all[slot(r)].emplace_back(gid(e), e);
        if (e < t.owned_end) up_send[slot(r)].emplace_back(gid(e), e);
      }
      // A copy's owner is the lowest other holder (copies are owned below this rank).
      if (e >= t.owned_end) up_recv[slot(val(static_cast<std::size_t>(off(row))))].emplace_back(gid(e), e);
    }
    for (auto* lists : {&up_send, &up_recv, &all}) {
      for (auto& l : *lists) std::sort(l.begin(), l.end());
    }
    nbrs_ = nbrs;
    flatten(up_send, update_send_, update_send_off_);
    flatten(up_recv, update_recv_, update_recv_off_);
    flatten(all, reduce_lids_, reduce_off_);
    // For each sharing entry (entity row i, other holder r): its row in the
    // reduce receive buffer.
    std::vector<LO> at(val.extent(0));
    for (std::size_t s = 0; s < nbrs.size(); ++s) {
      for (std::size_t k = 0; k < all[s].size(); ++k) {
        const auto row = static_cast<std::size_t>(all[s][k].second - first_);
        for (auto q = static_cast<std::size_t>(off(row)); q < static_cast<std::size_t>(off(row + 1)); ++q) {
          if (val(q) == nbrs[s]) at[q] = static_cast<LO>(reduce_off_[s] + k);
        }
      }
    }
    combine_offsets_ = sharing.offsets();
    combine_ranks_ = sharing.values();
    combine_at_ = to_space(at, "soma::SharedExchange::at");
  }

  [[nodiscard]] const std::vector<int>& neighbours() const noexcept { return nbrs_; }

  template <class T>
  void update(const Comm& comm, const View2<T, MemSpace>& values) const {
    check(values);
    exchange(comm, values, update_send_, update_send_off_, update_recv_, update_recv_off_, detail::update_tag, false,
             Op::sum);
  }

  template <class T>
  void reduce(const Comm& comm, const View2<T, MemSpace>& values, Op op) const {
    check(values);
    exchange(comm, values, reduce_lids_, reduce_off_, reduce_lids_, reduce_off_, detail::reduce_tag, true, op);
  }

 private:
  using Lids = Kokkos::View<const LO*, MemSpace>;

  static Lids to_space(const std::vector<LO>& v, const char* label) {
    Kokkos::View<LO*, Kokkos::HostSpace> h(label, v.size());
    std::copy(v.begin(), v.end(), h.data());
    return Kokkos::create_mirror_view_and_copy(MemSpace{}, h);
  }

  static void flatten(const std::vector<std::vector<std::pair<GO, LO>>>& lists, Lids& lids, std::vector<std::size_t>& off) {
    std::vector<LO> flat;
    off.assign(1, 0);
    for (const auto& l : lists) {
      for (const auto& [g, e] : l) flat.push_back(e);
      off.push_back(flat.size());
    }
    lids = to_space(flat, "soma::SharedExchange::lids");
  }

  template <class T>
  void check(const View2<T, MemSpace>& values) const {
    if (static_cast<LO>(values.extent(0)) != count_) {
      throw std::invalid_argument("SharedExchange: the values have " + std::to_string(values.extent(0)) +
                                  " rows, dimension " + std::to_string(dim_) + " has " + std::to_string(count_));
    }
  }

  template <class T>
  void exchange(const Comm& comm, const View2<T, MemSpace>& values, const Lids& send, const std::vector<std::size_t>& send_off,
                const Lids& recv, const std::vector<std::size_t>& recv_off, int tag, bool combine, Op op) const {
    using Buf = Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>;
    const std::size_t ncomp = values.extent(1);
    Buf out(Kokkos::view_alloc("soma::SharedExchange::send", Kokkos::WithoutInitializing), send.extent(0), ncomp);
    Buf in(Kokkos::view_alloc("soma::SharedExchange::recv", Kokkos::WithoutInitializing), recv.extent(0), ncomp);
    detail::pack_rows(values, send, out);
    Kokkos::fence("soma::SharedExchange::pack");
    const auto hout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    auto hin = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, in);
    std::vector<Request> r;
    for (std::size_t s = 0; s < nbrs_.size(); ++s) {
      const std::size_t a = recv_off[s] * ncomp, b = recv_off[s + 1] * ncomp;
      if (b > a) r.push_back(comm.irecv(std::span<T>(hin.data() + a, b - a), nbrs_[s], tag));
    }
    for (std::size_t s = 0; s < nbrs_.size(); ++s) {
      const std::size_t a = send_off[s] * ncomp, b = send_off[s + 1] * ncomp;
      if (b > a) r.push_back(comm.isend(std::span<const T>(hout.data() + a, b - a), nbrs_[s], tag));
    }
    Request::wait_all(r);
    Kokkos::deep_copy(in, hin);
    if (combine) {
      detail::combine_rows(values, first_, combine_offsets_, combine_ranks_, combine_at_, in, me_, op);
    } else {
      detail::unpack_rows(values, recv, in);
    }
    Kokkos::fence("soma::SharedExchange::unpack");
  }

  int dim_;
  int me_;
  LO count_ = 0;
  LO first_ = 0;
  std::vector<int> nbrs_;
  Lids update_send_, update_recv_, reduce_lids_;
  std::vector<std::size_t> update_send_off_, update_recv_off_, reduce_off_;
  View1<const Offset, MemSpace> combine_offsets_;
  View1<const int, MemSpace> combine_ranks_;
  Lids combine_at_;
};

}  // namespace soma

#endif  // SOMA_PARALLEL_SHARED_EXCHANGE_HPP_
