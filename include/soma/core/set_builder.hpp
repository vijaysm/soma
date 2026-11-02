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

// count consecutive members from first on: local ids of an entity slot, or
// gids of sets (set_dim). Rows are built from runs without listing them.
struct SetRun {
  int slot = 0;
  GO first = 0;
  GO count = 0;
};

// Collects sets on the host and installs them on a mesh at once. install
// throws std::invalid_argument for a flag outside set_flag_bits, a repeated
// gid, an entity that is not in the mesh, a member, child or parent that is
// not one of the sets, a run of no members, or runs on an ordered set;
// std::logic_error if the mesh has sets already. Nothing changes when it
// throws.
class SetBuilder {
 public:
  void add(SetSpec s) { add(std::move(s), {}); }
  // The set with runs of members besides its contents (unordered sets only).
  void add(SetSpec s, std::vector<SetRun> runs) { specs_.push_back({std::move(s), std::move(runs)}); }

  template <class MemSpace>
  void install(Mesh<MemSpace>& mesh) && {
    if (mesh.sets_.count() > 0) {
      throw std::logic_error("SetBuilder: the mesh has sets already");
    }
    std::sort(specs_.begin(), specs_.end(),
              [](const Pending& a, const Pending& b) { return a.spec.gid < b.spec.gid; });
    for (std::size_t i = 0; i < specs_.size(); ++i) {
      const SetSpec& s = specs_[i].spec;
      if ((s.flags & ~set_flag_bits) != 0) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " has flags " +
                                    hex(s.flags) + ", outside " + hex(set_flag_bits));
      }
      if (i > 0 && specs_[i - 1].spec.gid == s.gid) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) +
                                    " is repeated");
      }
    }
    const auto n = static_cast<std::size_t>(detail::to_lo(specs_.size(), "number of sets"));
    const auto set_index = [&](GO owner, GO gid, const char* what) {
      const auto it = std::lower_bound(specs_.begin(), specs_.end(), gid,
                                       [](const Pending& s, GO g) { return s.spec.gid < g; });
      if (it == specs_.end() || it->spec.gid != gid) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(owner) + " " + what +
                                    " gid " + std::to_string(gid) + ", which is not a set");
      }
      return static_cast<LO>(it - specs_.begin());
    };

    const auto entities = [&](int slot) { return slot == set_dim ? static_cast<LO>(n) : mesh.count(slot); };

    EntitySets<Kokkos::HostSpace> h;
    h.count_ = static_cast<LO>(n);
    h.gid_ = View1<GO, Kokkos::HostSpace>("soma::sets::gid", n);
    for (std::size_t i = 0; i < n; ++i) h.gid_(i) = specs_[i].spec.gid;
    std::array<std::vector<std::vector<Interval>>, num_field_slots> rows;
    for (auto& r : rows) r.resize(n);
    h.flags_.resize(n);
    h.seq_offsets_.assign(n + 1, 0);
    h.child_offsets_.assign(n + 1, 0);
    h.parent_offsets_.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const SetSpec& s = specs_[i].spec;
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
        rows[static_cast<std::size_t>(e.slot)][i].push_back(Interval{e.lid, e.lid + 1});
        if (ordered) h.seq_.push_back(e);
      }
      if (ordered && !specs_[i].runs.empty()) {
        throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) +
                                    " is ordered and takes no runs");
      }
      for (const SetRun& r : specs_[i].runs) {
        if (r.slot < 0 || r.slot > set_dim) {
          throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " has a run of slot " +
                                      std::to_string(r.slot) + ", not in [0, 4]");
        }
        if (r.count < 1) {
          throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " has a run of " +
                                      std::to_string(r.count) + " members");
        }
        LO first = 0;
        if (r.slot == set_dim) {
          // Set gids are sorted and unique, so the run is all sets when its
          // last gid sits count - 1 places after its first.
          first = set_index(s.gid, r.first, "contains");
          const GO after = static_cast<GO>(n) - first;  // sets from first on
          if (r.count > after ||
              specs_[static_cast<std::size_t>(first + r.count - 1)].spec.gid != r.first + (r.count - 1)) {
            throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " contains the " +
                                        std::to_string(r.count) + " gids from " + std::to_string(r.first) +
                                        ", which are not all sets");
          }
        } else {
          const LO count = entities(r.slot);
          if (r.first < 0 || r.first >= count || r.count > count - r.first) {
            throw std::invalid_argument("SetBuilder: set gid " + std::to_string(s.gid) + " lists the " +
                                        std::to_string(r.count) + " entities from " + std::to_string(r.first) +
                                        " of dimension " + std::to_string(r.slot) + ", which has " +
                                        std::to_string(count) + " entities");
          }
          first = static_cast<LO>(r.first);
        }
        rows[static_cast<std::size_t>(r.slot)][i].push_back(Interval{first, first + static_cast<LO>(r.count)});
      }
      h.seq_offsets_[i + 1] = detail::to_offset(h.seq_.size(), "sequence size");
      for (const GO c : s.children) h.children_.push_back(set_index(s.gid, c, "has child"));
      h.child_offsets_[i + 1] = detail::to_offset(h.children_.size(), "number of children");
      for (const GO p : s.parents) h.parents_.push_back(set_index(s.gid, p, "has parent"));
      h.parent_offsets_[i + 1] = detail::to_offset(h.parents_.size(), "number of parents");
    }
    for (std::size_t k = 0; k < rows.size(); ++k) {
      for (auto& r : rows[k]) r = merged(std::move(r));
      h.slots_[k] = EntitySets<Kokkos::HostSpace>::Slot::packed(rows[k], entities(static_cast<int>(k)));
    }
    h.host_gid_ = h.gid_;

    EntitySets<MemSpace> built = h.template mirror<MemSpace>({true, true, true, true}, true);
    // Nothing below throws.
    mesh.sets_ = std::move(built);
    mesh.fields_.set_entities(set_dim, mesh.sets_.count(), mesh.sets_.gid());
    specs_.clear();
  }

 private:
  struct Pending {
    SetSpec spec;
    std::vector<SetRun> runs;
  };

  // Sorted runs with overlapping and touching ones joined.
  static std::vector<Interval> merged(std::vector<Interval> r) {
    std::sort(r.begin(), r.end(), [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
    std::size_t m = 0;
    for (std::size_t j = 1; j < r.size(); ++j) {
      if (r[j].begin <= r[m].end) {
        r[m].end = std::max(r[m].end, r[j].end);
      } else {
        r[++m] = r[j];
      }
    }
    if (!r.empty()) r.resize(m + 1);
    return r;
  }

  static std::string hex(unsigned v) {
    static const char* digits = "0123456789abcdef";
    std::string s = "0x";
    s += digits[(v >> 4) & 0xF];
    s += digits[v & 0xF];
    return s;
  }

  std::vector<Pending> specs_;
};

}  // namespace soma

#endif  // SOMA_CORE_SET_BUILDER_HPP_
