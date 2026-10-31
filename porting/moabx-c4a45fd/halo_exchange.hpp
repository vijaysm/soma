#ifndef MOABX_MESH_HALO_EXCHANGE_HPP_
#define MOABX_MESH_HALO_EXCHANGE_HPP_

// Halo (ghost) exchange for distributed fields.
//
// Owners send their values; ghost holders receive and overwrite. After an
// exchange, every ghost equals its owner's value bitwise -- exchange moves
// bytes and performs no arithmetic, so there is no reassociation and the
// round-trip is exactly comparable. The tests rely on that.
//
// == Precomputed schedule ==
//
// The communication pattern is fixed for a given partition, so the pack and
// unpack index lists are built once in the constructor and reused by every
// exchange. A call is then gather -> isend/irecv -> scatter with no index
// computation and no allocation.
//
// == Ordering ==
//
// Send and receive lists for a neighbour pair must agree on element order.
// Both sides sort by *global id*, which is identical on every rank by
// definition. The sender orders its owned entities going to rank R by global
// id; the receiver orders its ghosts coming from R the same way. The two
// sequences therefore match without exchanging a schedule and without either
// side needing the other's local ids.
//
// == Overlap ==
//
// exchange_begin/exchange_end are split so callers can compute on interior
// entities while messages are in flight. exchange() is the fused form.

#include <algorithm>
#include <cstddef>
#include <cstddef>
#include <map>
#include <type_traits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "moabx/mesh/partition.hpp"
#include "moabx/mesh/tag_registry.hpp"
#include "moabx/pp/comm.hpp"

namespace moabx::mesh {

// Per-neighbour pack/unpack lists for one entity kind.
struct neighbor_schedule {
  int rank = invalid_rank;
  // Local ids of owned entities whose values are sent to `rank`, in send
  // order.
  std::vector<index_type> send_local_ids{};
  // Local ids of ghosts receiving from `rank`, in the order `rank` sends.
  std::vector<index_type> recv_local_ids{};

  std::size_t send_count() const noexcept { return send_local_ids.size(); }
  std::size_t recv_count() const noexcept { return recv_local_ids.size(); }
};

// Exchange schedule for one entity kind. Build once, reuse for every field.
class halo_exchange {
 public:
  halo_exchange() = default;

  // Derives the schedule from the partition. O(n log n) in the number of
  // shared entities, paid once.
  halo_exchange(const kind_partition& partition, int my_rank)
      : my_rank_(my_rank), entity_count_(partition.total_count()) {
    build(partition);
  }

  std::span<const neighbor_schedule> schedules() const noexcept {
    return std::span<const neighbor_schedule>(schedules_.data(), schedules_.size());
  }

  std::size_t neighbor_count() const noexcept { return schedules_.size(); }
  std::size_t entity_count() const noexcept { return entity_count_; }

  std::size_t total_send_count() const noexcept {
    std::size_t n = 0;
    for (const auto& s : schedules_) {
      n += s.send_count();
    }
    return n;
  }

  std::size_t total_recv_count() const noexcept {
    std::size_t n = 0;
    for (const auto& s : schedules_) {
      n += s.recv_count();
    }
    return n;
  }

  // Exchanges `components` values per entity over a flat array laid out as
  // entity-major (entity i occupies [i*components, (i+1)*components)).
  //
  // Values for owned entities are read; values for ghosts are overwritten.
  template <class T>
  void exchange(const pp::communicator& comm, std::span<T> field,
                std::size_t components = 1) const {
    exchange_begin(comm, field, components);
    exchange_end(field, components);
  }

  // Posts all receives and sends, then returns. `field` must stay valid and
  // its ghost region untouched until exchange_end.
  template <class T>
  void exchange_begin(const pp::communicator& comm, std::span<const T> field,
                      std::size_t components = 1) const {
    check_field(field.size(), components);
    if (schedules_.empty()) {
      return;
    }

    static_assert(std::is_trivially_copyable_v<T>,
                  "halo_exchange moves field values as raw bytes.");
    auto& buf = buffers_;
    buf.reset<T>(schedules_, components);

    // Receives first, so sends have somewhere to land and cannot force
    // unexpected-message buffering.
    for (std::size_t i = 0; i < schedules_.size(); ++i) {
      const auto& s = schedules_[i];
      if (s.recv_count() == 0) {
        continue;
      }
      buf.recv_requests[i] = comm.irecv(
          std::span<T>(buf.recv_at<T>(i), s.recv_count() * components), s.rank, kTag);
    }

    for (std::size_t i = 0; i < schedules_.size(); ++i) {
      const auto& s = schedules_[i];
      if (s.send_count() == 0) {
        continue;
      }
      T* dst = buf.send_at<T>(i);
      for (std::size_t e = 0; e < s.send_local_ids.size(); ++e) {
        const std::size_t src = static_cast<std::size_t>(s.send_local_ids[e]) * components;
        std::copy_n(field.begin() + static_cast<std::ptrdiff_t>(src), components,
                    dst + e * components);
      }
      buf.send_requests[i] = comm.isend(
          std::span<const T>(buf.send_at<T>(i), s.send_count() * components), s.rank, kTag);
    }
  }

  template <class T>
  void exchange_begin(const pp::communicator& comm, std::span<T> field,
                      std::size_t components = 1) const {
    exchange_begin(comm, std::span<const T>(field.data(), field.size()), components);
  }

  // Waits for all traffic and scatters received values into the ghost region.
  template <class T>
  void exchange_end(std::span<T> field, std::size_t components = 1) const {
    check_field(field.size(), components);
    if (schedules_.empty()) {
      return;
    }

    auto& buf = buffers_;
    for (auto& r : buf.recv_requests) {
      if (r.active()) {
        r.wait();
      }
    }

    for (std::size_t i = 0; i < schedules_.size(); ++i) {
      const auto& s = schedules_[i];
      const T* src = buf.recv_at<T>(i);
      for (std::size_t e = 0; e < s.recv_local_ids.size(); ++e) {
        const std::size_t dst = static_cast<std::size_t>(s.recv_local_ids[e]) * components;
        std::copy_n(src + e * components, components,
                    field.begin() + static_cast<std::ptrdiff_t>(dst));
      }
    }

    // Sends are completed last: the buffer must stay alive until MPI is done
    // reading it.
    for (auto& r : buf.send_requests) {
      if (r.active()) {
        r.wait();
      }
    }
  }

  // Typed convenience overload for a dense_tag.
  template <class T>
  void exchange_tag(const pp::communicator& comm, dense_tag<T>& tag) const {
    if (tag.entity_count() != entity_count_) {
      throw std::runtime_error(
          "halo_exchange: tag '" + tag.name() + "' has " +
          std::to_string(tag.entity_count()) + " entities, schedule expects " +
          std::to_string(entity_count_));
    }
    exchange(comm, tag.values(), tag.components());
  }

 private:
  static constexpr int kTag = 7701;

  void check_field(std::size_t size, std::size_t components) const {
    if (components == 0) {
      throw std::invalid_argument("halo_exchange: components must be > 0");
    }
    if (size != entity_count_ * components) {
      throw std::runtime_error(
          "halo_exchange: field has " + std::to_string(size) + " values, expected " +
          std::to_string(entity_count_ * components));
    }
  }

  void build(const kind_partition& partition) {
    const std::size_t n = partition.total_count();

    // Both directions are keyed by (global_id, local_id) and sorted on the
    // global id, which every rank agrees on.
    std::map<int, std::vector<std::pair<global_id_type, index_type>>> sends;
    std::map<int, std::vector<std::pair<global_id_type, index_type>>> recvs;

    for (std::size_t i = 0; i < n; ++i) {
      const auto local = static_cast<index_type>(i);
      const auto gid = partition.global_id(local);
      if (partition.is_owned(local)) {
        for (const auto& rc : partition.remote_copies(local)) {
          sends[rc.rank].emplace_back(gid, local);
        }
      } else {
        recvs[partition.owner_rank(local)].emplace_back(gid, local);
      }
    }

    std::vector<int> ranks;
    for (const auto& [r, _] : sends) {
      ranks.push_back(r);
    }
    for (const auto& [r, _] : recvs) {
      ranks.push_back(r);
    }
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

    schedules_.reserve(ranks.size());
    for (const int r : ranks) {
      neighbor_schedule s;
      s.rank = r;
      if (auto it = sends.find(r); it != sends.end()) {
        auto& pairs = it->second;
        std::sort(pairs.begin(), pairs.end());
        s.send_local_ids.reserve(pairs.size());
        for (const auto& [gid, lid] : pairs) {
          s.send_local_ids.push_back(lid);
        }
      }
      if (auto it = recvs.find(r); it != recvs.end()) {
        auto& pairs = it->second;
        std::sort(pairs.begin(), pairs.end());
        s.recv_local_ids.reserve(pairs.size());
        for (const auto& [gid, lid] : pairs) {
          s.recv_local_ids.push_back(lid);
        }
      }
      schedules_.push_back(std::move(s));
    }
  }

  // Scratch buffers, owned by this instance and reused across calls.
  //
  // Typed as raw bytes so one allocation serves every element type; a
  // per-type static registry would either leak or outlive the exchange
  // object it was keyed on. Element types are trivially copyable (enforced
  // below), so aliasing the byte buffer is well defined.
  struct exchange_buffers {
    std::vector<std::byte> send{};
    std::vector<std::byte> recv{};
    std::vector<std::size_t> send_offsets{};  // in elements, not bytes
    std::vector<std::size_t> recv_offsets{};
    std::vector<pp::request> send_requests{};
    std::vector<pp::request> recv_requests{};

    template <class T>
    void reset(const std::vector<neighbor_schedule>& schedules, std::size_t components) {
      const std::size_t k = schedules.size();
      send_offsets.assign(k + 1, 0);
      recv_offsets.assign(k + 1, 0);
      for (std::size_t i = 0; i < k; ++i) {
        send_offsets[i + 1] = send_offsets[i] + schedules[i].send_count() * components;
        recv_offsets[i + 1] = recv_offsets[i] + schedules[i].recv_count() * components;
      }
      send.resize(send_offsets[k] * sizeof(T));
      recv.resize(recv_offsets[k] * sizeof(T));
      send_requests.assign(k, pp::request{});
      recv_requests.assign(k, pp::request{});
    }

    template <class T>
    T* send_at(std::size_t i) {
      return reinterpret_cast<T*>(send.data()) + send_offsets[i];
    }

    template <class T>
    T* recv_at(std::size_t i) {
      return reinterpret_cast<T*>(recv.data()) + recv_offsets[i];
    }
  };

  // Mutable: exchange is logically const because it never alters the
  // schedule, but it must reuse the scratch space.
  mutable exchange_buffers buffers_{};

  int my_rank_ = 0;
  std::size_t entity_count_ = 0;
  std::vector<neighbor_schedule> schedules_{};
};

}  // namespace moabx::mesh

#endif  // MOABX_MESH_HALO_EXCHANGE_HPP_
