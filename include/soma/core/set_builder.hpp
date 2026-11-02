// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SET_BUILDER_HPP_
#define SOMA_CORE_SET_BUILDER_HPP_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/sets.hpp"
#include "soma/core/types.hpp"

namespace soma {

// One member as given to the builder: an entity (slot 0 to 3 and its local id)
// or a set (set_dim and its gid).
struct SetItem {
  int slot = 0;
  GO id = 0;
};

// One set: its gid, MOAB flags (within set_flag_bits), its contents in order
// and its child and parent sets by gid. An ordered set keeps its contents as
// given, repeats included; every set also gets sorted, unique rows per slot.
struct SetSpec {
  GO gid = 0;
  std::uint8_t flags = set_unique;
  std::vector<SetItem> contents;
  std::vector<GO> children;
  std::vector<GO> parents;
};

// Collects sets on the host and installs them on a mesh at once. install
// throws std::invalid_argument for a flag outside set_flag_bits, a repeated
// gid, an entity that is not in the mesh, or a member, child or parent that
// is not one of the sets; std::logic_error if the mesh has sets already.
// Nothing changes when it throws.
class SetBuilder {
 public:
  void add(SetSpec s) { specs_.push_back(std::move(s)); }

  template <class MemSpace>
  void install(Mesh<MemSpace>& mesh) && {
    if (mesh.sets_.count() > 0) {
      throw std::logic_error("SetBuilder: the mesh has sets already");
    }
    std::sort(specs_.begin(), specs_.end(),
              [](const SetSpec& a, const SetSpec& b) { return a.gid < b.gid; });
    for (std::size_t i = 0; i < specs_.size(); ++i) {
      const SetSpec& s = specs_[i];
      if ((s.flags & ~set_flag_bits) != 0) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " has flags " +
                                    hex(s.flags) + ", outside " + hex(set_flag_bits));
      }
      if (i > 0 && specs_[i - 1].gid == s.gid) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " is repeated");
      }
    }
    const auto n = static_cast<std::size_t>(detail::to_lo(specs_.size(), "number of sets"));
    const auto set_index = [&](GO owner, GO gid, const char* what) {
      const auto it = std::lower_bound(specs_.begin(), specs_.end(), gid,
                                       [](const SetSpec& s, GO g) { return s.gid < g; });
      if (it == specs_.end() || it->gid != gid) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(owner) + " " + what +
                                    " gid " + std::to_string(gid) + ", which is not a set");
      }
      return static_cast<LO>(it - specs_.begin());
    };

    EntitySets<Kokkos::HostSpace> h;
    h.count_ = static_cast<LO>(n);
    h.gid_ = View1<GO, Kokkos::HostSpace>("soma::sets::gid", n);
    for (std::size_t i = 0; i < n; ++i) h.gid_(i) = specs_[i].gid;
    std::array<std::vector<std::vector<LO>>, num_field_slots> rows;
    for (auto& r : rows) r.resize(n);
    h.flags_.resize(n);
    h.seq_offsets_.assign(n + 1, 0);
    h.child_offsets_.assign(n + 1, 0);
    h.parent_offsets_.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const SetSpec& s = specs_[i];
      h.flags_[i] = s.flags;
      const bool ordered = (s.flags & set_ordered) != 0;
      for (const SetItem& m : s.contents) {
        SetEntry e{m.slot, invalid_lo};
        if (m.slot == set_dim) {
          e.lid = set_index(s.gid, m.id, "contains");
        } else if (m.slot >= 0 && m.slot <= 3) {
          const LO count = mesh.count(m.slot);
          if (m.id < 0 || m.id >= count) {
            throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) +
                                        " lists entity " + std::to_string(m.id) + " of dimension " +
                                        std::to_string(m.slot) + ", which has " +
                                        std::to_string(count) + " entities");
          }
          e.lid = static_cast<LO>(m.id);
        } else {
          throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) +
                                      " lists a member of slot " + std::to_string(m.slot) +
                                      ", not in [0, 4]");
        }
        rows[static_cast<std::size_t>(e.slot)][i].push_back(e.lid);
        if (ordered) h.seq_.push_back(e);
      }
      h.seq_offsets_[i + 1] = detail::to_offset(h.seq_.size(), "sequence size");
      for (const GO c : s.children) h.children_.push_back(set_index(s.gid, c, "has child"));
      h.child_offsets_[i + 1] = detail::to_offset(h.children_.size(), "number of children");
      for (const GO p : s.parents) h.parents_.push_back(set_index(s.gid, p, "has parent"));
      h.parent_offsets_[i + 1] = detail::to_offset(h.parents_.size(), "number of parents");
    }
    for (std::size_t k = 0; k < rows.size(); ++k) {
      std::size_t total = 0;
      for (auto& r : rows[k]) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        total += r.size();
      }
      auto& slot = h.slots_[k];
      slot.offsets = View1<Offset, Kokkos::HostSpace>("soma::sets::offsets", n + 1);
      slot.members = View1<LO, Kokkos::HostSpace>("soma::sets::members", total);
      std::size_t at = 0;
      slot.offsets(0) = 0;
      for (std::size_t i = 0; i < n; ++i) {
        for (const LO l : rows[k][i]) slot.members(at++) = l;
        slot.offsets(i + 1) = detail::to_offset(at, "set membership size");
      }
      slot.host_offsets = slot.offsets;
      slot.host_members = slot.members;
    }
    h.host_gid_ = h.gid_;

    EntitySets<MemSpace> built = h.template mirror<MemSpace>({true, true, true, true}, true);
    // Nothing below throws.
    mesh.sets_ = std::move(built);
    mesh.fields_.set_entities(set_dim, mesh.sets_.count(), mesh.sets_.gid());
    specs_.clear();
  }

 private:
  static std::string hex(unsigned v) {
    static const char* digits = "0123456789abcdef";
    std::string s = "0x";
    s += digits[(v >> 4) & 0xF];
    s += digits[v & 0xF];
    return s;
  }

  std::vector<SetSpec> specs_;
};

}  // namespace soma

#endif  // SOMA_CORE_SET_BUILDER_HPP_
