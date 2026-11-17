// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_COPY_HPP_
#define SOMA_AMR_COPY_HPP_

/// \file
/// Copies between two sets of boxes of one index type. A CopyPlan fills given
/// regions of the destination boxes from a source level: from its valid
/// points first (the first source along the curve wins where they share
/// points), then, if asked, from the ghost points of its boxes (which the
/// caller has filled), through periodic images too. What no source covers is
/// left as it is and listed as uncovered. The plan is computed once and
/// replayed by copy(); its copies of one destination box are disjoint, so the
/// result does not depend on the thread count.
///
/// Over many ranks the source and destination boxes have owners. Every rank
/// computes the same copies from the replicated boxes and keeps its part:
/// copies between boxes it owns, and per other rank the points it sends and
/// receives, in the same order on both sides, so no message sets the plan
/// up. A copy posts the receives, packs and sends (device kernels, staged
/// through host memory), copies locally, then waits and unpacks:
///
/// ```
///   auto pending = amr::begin_copy(src, dst, plan);   // work may go here
///   pending.end();
/// ```
///
/// Fills of one plan that are in flight together must begin in the same
/// order on every rank, as MPI collectives must.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/level_field.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/comm_plan.hpp"

namespace soma::amr {

struct Periodicity {
  bool periodic[3] = {false, false, false};
  KOKKOS_INLINE_FUNCTION constexpr bool operator[](int d) const { return periodic[d]; }
};

/// Points `region` of destination box dst take the values of points
/// region - shift of source box src.
struct CopyOp {
  int src = 0;
  int dst = 0;
  Box region;
  IntVect shift;
};

class CopyPlan {
 public:
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
  static constexpr int default_tag = 32620;

  CopyPlan() = default;
  /// regions[b]: the points of destination box b to fill, in `type`. With
  /// src_ghosts > 0, source ghost points up to that width serve after the
  /// valid ones. Every box on this process.
  CopyPlan(const BoxArray& src, const std::vector<BoxList>& regions, IndexType type, Periodicity periodic,
           int src_ghosts = 0);
  /// The part of rank comm.rank(), source box i owned by src.owner[i] and
  /// destination box b by dst.owner[b]; messages on `tag`. Throws
  /// std::invalid_argument unless each distribution gives every box an owner
  /// in the communicator.
  CopyPlan(const Comm& comm, const BoxArray& src, const Distribution& src_owner, const std::vector<BoxList>& regions,
           const Distribution& dst_owner, IndexType type, Periodicity periodic, int src_ghosts = 0,
           int tag = default_tag);

  IndexType type() const { return type_; }
  int src_ghosts() const { return src_ghosts_; }
  int tag() const { return tag_; }
  /// Every copy of the plan, on every rank, boxes by their ids.
  const std::vector<CopyOp>& copies() const { return copies_; }
  /// Per destination box, the points of its regions no source covered.
  const std::vector<BoxList>& uncovered() const { return uncovered_; }
  /// This rank's copies between boxes it owns, by the fields' local indices,
  /// and how many points they, the sends and the receives hold.
  const std::vector<CopyOp>& local_copies() const { return local_; }
  std::int64_t num_copied() const { return local_start_.back(); }
  std::int64_t num_sent() const { return send_start_.back(); }
  std::int64_t num_received() const { return recv_start_.back(); }
  const std::optional<Comm>& comm() const { return comm_; }
  /// The owner of every source and destination box.
  const std::vector<int>& src_owner() const { return src_owner_; }
  const std::vector<int>& dst_owner() const { return dst_owner_; }
  const transport::CommPlan& comm_plan() const { return messages_; }

  Kokkos::View<const CopyOp*, DeviceSpace> device_copies() const { return d_local_; }
  Kokkos::View<const std::int64_t*, DeviceSpace> device_start() const { return d_local_start_; }
  /// Sends: source box by local index, region in the destination's frame.
  Kokkos::View<const CopyOp*, DeviceSpace> device_sends() const { return d_send_; }
  Kokkos::View<const std::int64_t*, DeviceSpace> device_send_start() const { return d_send_start_; }
  /// Receives: destination box by local index.
  Kokkos::View<const CopyOp*, DeviceSpace> device_receives() const { return d_recv_; }
  Kokkos::View<const std::int64_t*, DeviceSpace> device_receive_start() const { return d_recv_start_; }

 private:
  void build(const BoxArray& src, const std::vector<BoxList>& regions, Periodicity periodic);
  void split(int rank, int nranks);

  IndexType type_{};
  int src_ghosts_ = 0;
  int tag_ = default_tag;
  std::optional<Comm> comm_;
  std::vector<int> src_owner_, dst_owner_;
  std::vector<CopyOp> copies_;
  std::vector<BoxList> uncovered_;
  std::vector<CopyOp> local_, send_, recv_;
  std::vector<std::int64_t> local_start_{0}, send_start_{0}, recv_start_{0};
  transport::CommPlan messages_;
  Kokkos::View<const CopyOp*, DeviceSpace> d_local_, d_send_, d_recv_;
  Kokkos::View<const std::int64_t*, DeviceSpace> d_local_start_, d_send_start_, d_recv_start_;
};

namespace detail {

/// The op holding point p of a list whose points start at start(i).
template <class Start>
KOKKOS_INLINE_FUNCTION int op_of(const Start& start, int n, std::int64_t p) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (start(mid) <= p) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

template <class T, class MemSpace>
void run_copies(const LevelView<T, MemSpace>& src, const LevelView<T, MemSpace>& dst, const CopyPlan& plan) {
  using Exec = typename MemSpace::execution_space;
  const auto copies = plan.device_copies();
  const auto start = plan.device_start();
  const int n = static_cast<int>(copies.extent(0));
  const int ncomp = dst.ncomp < src.ncomp ? dst.ncomp : src.ncomp;
  Kokkos::parallel_for(
      "amr::copy", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, plan.num_copied()),
      KOKKOS_LAMBDA(std::int64_t p) {
        const int i = op_of(start, n, p);
        const CopyOp& op = copies(i);
        const IntVect q = point_of(op.region, p - start(i));
        const auto to = dst.patch(op.dst);
        const auto from = src.patch(op.src);
        for (int c = 0; c < ncomp; ++c) {
          to(q[0], q[1], q[2], c) = from(q[0] - op.shift[0], q[1] - op.shift[1], q[2] - op.shift[2], c);
        }
      });
  Kokkos::fence("amr::copy");
}

/// Point p of the sends into buffer values p * ncomp ... p * ncomp + ncomp - 1.
template <class T, class MemSpace>
void pack_sends(const LevelView<T, MemSpace>& src, const CopyPlan& plan, const Kokkos::View<T*, MemSpace>& buffer,
                int ncomp) {
  using Exec = typename MemSpace::execution_space;
  const auto sends = plan.device_sends();
  const auto start = plan.device_send_start();
  const int n = static_cast<int>(sends.extent(0));
  Kokkos::parallel_for(
      "amr::copy.pack", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, plan.num_sent()),
      KOKKOS_LAMBDA(std::int64_t p) {
        const int i = op_of(start, n, p);
        const CopyOp& op = sends(i);
        const IntVect q = point_of(op.region, p - start(i));
        const auto from = src.patch(op.src);
        for (int c = 0; c < ncomp; ++c) {
          buffer(p * ncomp + c) = from(q[0] - op.shift[0], q[1] - op.shift[1], q[2] - op.shift[2], c);
        }
      });
  Kokkos::fence("amr::copy.pack");
}

template <class T, class MemSpace>
void unpack_receives(const LevelView<T, MemSpace>& dst, const Kokkos::View<const CopyOp*, CopyPlan::DeviceSpace>& recvs,
                     const Kokkos::View<const std::int64_t*, CopyPlan::DeviceSpace>& start, std::int64_t count,
                     const Kokkos::View<T*, MemSpace>& buffer, int ncomp) {
  using Exec = typename MemSpace::execution_space;
  const int n = static_cast<int>(recvs.extent(0));
  Kokkos::parallel_for(
      "amr::copy.unpack", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, count),
      KOKKOS_LAMBDA(std::int64_t p) {
        const int i = op_of(start, n, p);
        const CopyOp& op = recvs(i);
        const IntVect q = point_of(op.region, p - start(i));
        const auto to = dst.patch(op.dst);
        for (int c = 0; c < ncomp; ++c) to(q[0], q[1], q[2], c) = buffer(p * ncomp + c);
      });
  Kokkos::fence("amr::copy.unpack");
}

}  // namespace detail

/// A copy in flight: end() waits for the messages and unpacks them (a second
/// call does nothing). Destroying one before end() waits and drops what
/// arrived.
template <class T, class MemSpace = Kokkos::DefaultExecutionSpace::memory_space>
class PendingCopy {
 public:
  PendingCopy() = default;
  PendingCopy(const PendingCopy&) = delete;
  PendingCopy& operator=(const PendingCopy&) = delete;
  PendingCopy(PendingCopy&&) = default;
  PendingCopy& operator=(PendingCopy&&) = default;
  ~PendingCopy() = default;
  void end() {
    if (done_.value) return;
    done_.value = true;
    messages_.wait();
    if (count_ == 0) return;
    Kokkos::deep_copy(recv_, recv_host_);
    detail::unpack_receives(dst_, recvs_, start_, count_, recv_, ncomp_);
  }

 private:
  template <class U, class M>
  friend PendingCopy<U, M> begin_copy_views(const LevelView<U, M>& src, const LevelView<U, M>& dst, const CopyPlan& plan);

  /// Finished, or moved from: a moved-from copy has nothing left to end.
  struct Done {
    bool value = true;
    Done() = default;
    Done(Done&& o) noexcept : value(std::exchange(o.value, true)) {}
    Done& operator=(Done&& o) noexcept {
      value = std::exchange(o.value, true);
      return *this;
    }
  };
  Done done_;
  int ncomp_ = 1;
  std::int64_t count_ = 0;
  LevelView<T, MemSpace> dst_;
  Kokkos::View<const CopyOp*, CopyPlan::DeviceSpace> recvs_;
  Kokkos::View<const std::int64_t*, CopyPlan::DeviceSpace> start_;
  /// The buffers outlive the messages: declared before them, destroyed after.
  Kokkos::View<T*, MemSpace> send_, recv_;
  typename Kokkos::View<T*, MemSpace>::host_mirror_type send_host_, recv_host_;
  transport::CommPlan::Pending messages_;
};

template <class T, class MemSpace>
PendingCopy<T, MemSpace> begin_copy_views(const LevelView<T, MemSpace>& src, const LevelView<T, MemSpace>& dst,
                                          const CopyPlan& plan) {
  PendingCopy<T, MemSpace> out;
  out.done_.value = false;
  out.ncomp_ = dst.ncomp < src.ncomp ? dst.ncomp : src.ncomp;
  out.count_ = plan.num_received();
  out.dst_ = dst;
  out.recvs_ = plan.device_receives();
  out.start_ = plan.device_receive_start();
  const auto nc = static_cast<std::size_t>(out.ncomp_);
  out.send_ = Kokkos::View<T*, MemSpace>(Kokkos::view_alloc("amr::copy.send", Kokkos::WithoutInitializing),
                                         static_cast<std::size_t>(plan.num_sent()) * nc);
  out.recv_ = Kokkos::View<T*, MemSpace>(Kokkos::view_alloc("amr::copy.recv", Kokkos::WithoutInitializing),
                                         static_cast<std::size_t>(plan.num_received()) * nc);
  out.send_host_ = Kokkos::create_mirror_view(out.send_);
  out.recv_host_ = Kokkos::create_mirror_view(out.recv_);
  if (plan.comm()) {
    detail::pack_sends(src, plan, out.send_, out.ncomp_);
    Kokkos::deep_copy(out.send_host_, out.send_);
    const std::size_t width = nc * sizeof(T);
    out.messages_ = plan.comm_plan().begin_forward_bytes(
        *plan.comm(), std::as_bytes(std::span<const T>(out.send_host_.data(), out.send_host_.extent(0))),
        std::as_writable_bytes(std::span<T>(out.recv_host_.data(), out.recv_host_.extent(0))), width, plan.tag());
  }
  detail::run_copies(src, dst, plan);
  return out;
}

/// Starts a copy along the plan from src into dst (components up to the
/// smaller count); throws std::invalid_argument for fields that do not match
/// the plan's index type, source ghosts, owners or rank.
template <class T>
PendingCopy<T> begin_copy(const LevelField<T>& src, const LevelField<T>& dst, const CopyPlan& plan) {
  if (!(src.type() == plan.type()) || !(dst.type() == plan.type()) || src.ghosts() < plan.src_ghosts()) {
    throw std::invalid_argument("amr::copy: the fields do not match the plan's index type or source ghosts");
  }
  const int rank = plan.comm() ? plan.comm()->rank() : 0;
  if (src.distribution().owner != plan.src_owner() || dst.distribution().owner != plan.dst_owner() ||
      src.rank() != rank || dst.rank() != rank) {
    throw std::invalid_argument("amr::copy: the fields are not distributed as the plan's boxes");
  }
  return begin_copy_views(src.view(), dst.view(), plan);
}

template <class T>
void copy(const LevelField<T>& src, const LevelField<T>& dst, const CopyPlan& plan) {
  begin_copy(src, dst, plan).end();
}

}  // namespace soma::amr

#endif  // SOMA_AMR_COPY_HPP_
