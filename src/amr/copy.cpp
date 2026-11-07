// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/copy.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "bins.hpp"

namespace soma::amr {

namespace {

template <class U>
Kokkos::View<const U*, CopyPlan::DeviceSpace> to_device(const std::vector<U>& v, const std::string& label) {
  Kokkos::View<U*, CopyPlan::DeviceSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

// The cell box around a region of any index type, one cell wider.
Box around(const Box& r) {
  return Box(r.lo() - IntVect{{1, 1, 1}}, r.hi() + IntVect{{1, 1, 1}}, r.dim());
}

}  // namespace

CopyPlan::CopyPlan(const BoxArray& src, const std::vector<BoxList>& regions, IndexType type, Periodicity periodic,
                   int src_ghosts)
    : type_(type), src_ghosts_(src_ghosts) {
  src_owner_.assign(src.size(), 0);
  dst_owner_.assign(regions.size(), 0);
  build(src, regions, periodic);
  split(0, 1);
}

CopyPlan::CopyPlan(const Comm& comm, const BoxArray& src, const Distribution& src_owner, const std::vector<BoxList>& regions,
                   const Distribution& dst_owner, IndexType type, Periodicity periodic, int src_ghosts, int tag)
    : type_(type), src_ghosts_(src_ghosts), tag_(tag), comm_(comm), src_owner_(src_owner.owner),
      dst_owner_(dst_owner.owner) {
  const auto bad = [&](const std::vector<int>& owner, std::size_t n) {
    return owner.size() != n || std::any_of(owner.begin(), owner.end(), [&](int r) { return r < 0 || r >= comm.size(); });
  };
  if (bad(src_owner_, src.size()) || bad(dst_owner_, regions.size())) {
    throw std::invalid_argument("CopyPlan: a distribution does not give every box an owner in the communicator");
  }
  build(src, regions, periodic);
  split(comm.rank(), comm.size());
}

void CopyPlan::build(const BoxArray& src, const std::vector<BoxList>& regions, Periodicity periodic) {
  const int src_ghosts = src_ghosts_;
  const IndexType type = type_;
  if (src_ghosts < 0) throw std::invalid_argument("CopyPlan: source ghosts must not be negative");
  const Box& domain = src.domain();
  const int dim = domain.dim();
  // Periodic images: shifts by the domain length, each periodic direction
  // -1, 0 or +1, the unshifted one first.
  std::vector<IntVect> shifts{IntVect{}};
  for (int d = 0; d < dim; ++d) {
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
  const detail::Bins bins(grow(src.boxes(), src_ghosts), domain);
  uncovered_.resize(regions.size());
  for (std::size_t b = 0; b < regions.size(); ++b) {
    BoxList left = regions[b];
    std::vector<std::pair<int, std::size_t>> cand;  // (source, image)
    for (std::size_t s = 0; s < shifts.size(); ++s) {
      const IntVect back = IntVect{} - shifts[s];
      for (const Box& r : regions[b]) {
        for (const int i : bins.near(around(r).shift(back))) cand.emplace_back(i, s);
      }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    // Valid points first, then (if asked) the ghost points.
    for (const int width : src_ghosts > 0 ? std::vector<int>{0, src_ghosts} : std::vector<int>{0}) {
      for (const auto& [i, s] : cand) {
        if (left.empty()) break;
        const Box from = src[static_cast<std::size_t>(i)].convert(type).grow(width).shift(shifts[s]);
        for (const Box& r : intersect(left, from)) {
          copies_.push_back(CopyOp{i, static_cast<int>(b), r, shifts[s]});
        }
        left = subtract(left, BoxList{from});
      }
    }
    uncovered_[b] = std::move(left);
  }
}

void CopyPlan::split(int rank, int nranks) {
  // Local copies and, per neighbour, sends and receives in the plan's order,
  // which both sides share: receives by source rank, sends by destination.
  std::vector<int> src_local(src_owner_.size(), -1), dst_local(dst_owner_.size(), -1);
  for (std::size_t i = 0, n = 0; i < src_owner_.size(); ++i) {
    if (src_owner_[i] == rank) src_local[i] = static_cast<int>(n++);
  }
  for (std::size_t i = 0, n = 0; i < dst_owner_.size(); ++i) {
    if (dst_owner_[i] == rank) dst_local[i] = static_cast<int>(n++);
  }
  std::vector<std::vector<CopyOp>> to(static_cast<std::size_t>(nranks)), from(static_cast<std::size_t>(nranks));
  for (const CopyOp& op : copies_) {
    const int s = src_owner_[static_cast<std::size_t>(op.src)], d = dst_owner_[static_cast<std::size_t>(op.dst)];
    if (s == rank && d == rank) {
      local_.push_back(CopyOp{src_local[static_cast<std::size_t>(op.src)], dst_local[static_cast<std::size_t>(op.dst)], op.region, op.shift});
    } else if (s == rank) {
      to[static_cast<std::size_t>(d)].push_back(CopyOp{src_local[static_cast<std::size_t>(op.src)], op.dst, op.region, op.shift});
    } else if (d == rank) {
      from[static_cast<std::size_t>(s)].push_back(CopyOp{op.src, dst_local[static_cast<std::size_t>(op.dst)], op.region, op.shift});
    }
  }
  for (const CopyOp& op : local_) local_start_.push_back(local_start_.back() + op.region.num_pts());
  std::vector<transport::CommPlan::Run> sent, arrived;
  for (int q = 0; q < nranks; ++q) {
    const std::int64_t s0 = send_start_.back(), r0 = recv_start_.back();
    for (const CopyOp& op : to[static_cast<std::size_t>(q)]) {
      send_.push_back(op);
      send_start_.push_back(send_start_.back() + op.region.num_pts());
    }
    for (const CopyOp& op : from[static_cast<std::size_t>(q)]) {
      recv_.push_back(op);
      recv_start_.push_back(recv_start_.back() + op.region.num_pts());
    }
    if (send_start_.back() > s0) sent.push_back({q, static_cast<std::size_t>(send_start_.back() - s0)});
    if (recv_start_.back() > r0) arrived.push_back({q, static_cast<std::size_t>(recv_start_.back() - r0)});
  }
  if (comm_) messages_ = transport::CommPlan::from_counts(*comm_, sent, arrived);
  d_local_ = to_device(local_, "amr::CopyPlan.local");
  d_local_start_ = to_device(local_start_, "amr::CopyPlan.local_start");
  d_send_ = to_device(send_, "amr::CopyPlan.sends");
  d_send_start_ = to_device(send_start_, "amr::CopyPlan.send_start");
  d_recv_ = to_device(recv_, "amr::CopyPlan.receives");
  d_recv_start_ = to_device(recv_start_, "amr::CopyPlan.receive_start");
}

}  // namespace soma::amr
