// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Interface: entity sets.

#include <algorithm>
#include <string>
#include <vector>

#include "core_state.hpp"
#include "soma/compat/core.hpp"

namespace soma::compat {

namespace {

using detail::CoreState;
using detail::SetData;

SetData* set_of(CoreState& s, EntityHandle h) {
  return s.is_set(h) ? &s.sets[static_cast<std::size_t>(id_of(h) - 1)] : nullptr;
}
const SetData* set_of(const CoreState& s, EntityHandle h) {
  return s.is_set(h) ? &s.sets[static_cast<std::size_t>(id_of(h) - 1)] : nullptr;
}

// Sets reached from `set` through `next` (children, parents or contained
// sets) within num_hops steps (all of them when num_hops <= 0), in the order
// they are found.
template <class Next>
std::vector<EntityHandle> reach(const CoreState& s, EntityHandle set, int num_hops, Next next) {
  std::vector<EntityHandle> out;
  std::vector<EntityHandle> level{set};
  for (int hop = 0; (num_hops <= 0 || hop < num_hops) && !level.empty(); ++hop) {
    std::vector<EntityHandle> following;
    for (const EntityHandle x : level) {
      for (const EntityHandle y : next(*set_of(s, x))) {
        if (y == set || std::find(out.begin(), out.end(), y) != out.end()) continue;
        out.push_back(y);
        following.push_back(y);
      }
    }
    level.swap(following);
  }
  return out;
}

std::vector<EntityHandle> contained(const SetData& d) {
  std::vector<EntityHandle> out;
  if (d.ordered()) {
    for (const EntityHandle h : d.sequence) {
      if (type_of(h) == MBENTITYSET && std::find(out.begin(), out.end(), h) == out.end()) out.push_back(h);
    }
  } else {
    const Range r = d.contents.subset_by_type(MBENTITYSET);
    out.assign(r.begin(), r.end());
  }
  return out;
}

}  // namespace

ErrorCode Interface::create_meshset(unsigned options, EntityHandle& set, int /*start_id*/) {
  if ((options & MESHSET_SET) != 0 && (options & MESHSET_ORDERED) != 0) {
    return s_->fail(MB_FAILURE, "create_meshset: a set is unique or ordered, not both");
  }
  const EntityID id = s_->alloc_one(MBENTITYSET, 0);
  if (s_->sets.size() < static_cast<std::size_t>(id)) {
    SetData gap;
    gap.alive = false;
    s_->sets.resize(static_cast<std::size_t>(id), gap);
  }
  SetData& d = s_->sets[static_cast<std::size_t>(id - 1)];
  d = SetData{};
  d.flags = options;
  set = make_handle(MBENTITYSET, id);
  s_->fill_dense(MBENTITYSET, id);
  return MB_SUCCESS;
}

ErrorCode Interface::clear_meshset(const EntityHandle* sets, int n) {
  for (int i = 0; i < n; ++i) {
    SetData* d = set_of(*s_, sets[i]);
    if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "clear_meshset: no set " + std::to_string(sets[i]));
    d->contents.clear();
    d->sequence.clear();
  }
  return MB_SUCCESS;
}

ErrorCode Interface::clear_meshset(const Range& sets) {
  const std::vector<EntityHandle> v(sets.begin(), sets.end());
  return clear_meshset(v.data(), static_cast<int>(v.size()));
}

ErrorCode Interface::get_meshset_options(EntityHandle set, unsigned& options) const {
  if (set == 0) {
    options = MESHSET_SET;
    return MB_SUCCESS;
  }
  const SetData* d = set_of(*s_, set);
  if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "get_meshset_options: no set " + std::to_string(set));
  options = d->flags;
  return MB_SUCCESS;
}

ErrorCode Interface::set_meshset_options(EntityHandle set, unsigned options) {
  SetData* d = set_of(*s_, set);
  if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "set_meshset_options: no set " + std::to_string(set));
  const bool was = d->ordered();
  d->flags = options;
  if (was && !d->ordered()) {
    for (const EntityHandle h : d->sequence) d->contents.insert(h);
    d->sequence.clear();
  } else if (!was && d->ordered()) {
    d->sequence.assign(d->contents.begin(), d->contents.end());
    d->contents.clear();
  }
  return MB_SUCCESS;
}

ErrorCode Interface::add_entities(EntityHandle set, const EntityHandle* es, int n) {
  s_->flush();
  SetData* d = set_of(*s_, set);
  if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "add_entities: no set " + std::to_string(set));
  for (int i = 0; i < n; ++i) {
    if (es[i] == 0 || !s_->valid(es[i])) return s_->fail(MB_ENTITY_NOT_FOUND, "add_entities: no entity " + std::to_string(es[i]));
  }
  if (d->ordered()) {
    d->sequence.insert(d->sequence.end(), es, es + n);
  } else {
    std::vector<EntityHandle> v(es, es + n);
    std::sort(v.begin(), v.end());
    for (const EntityHandle h : v) d->contents.insert(h);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::add_entities(EntityHandle set, const Range& es) {
  s_->flush();
  SetData* d = set_of(*s_, set);
  if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "add_entities: no set " + std::to_string(set));
  for (const EntityHandle h : es) {
    if (h == 0 || !s_->valid(h)) return s_->fail(MB_ENTITY_NOT_FOUND, "add_entities: no entity " + std::to_string(h));
  }
  if (d->ordered()) {
    d->sequence.insert(d->sequence.end(), es.begin(), es.end());
  } else {
    d->contents.merge(es);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::remove_entities(EntityHandle set, const EntityHandle* es, int n) {
  SetData* d = set_of(*s_, set);
  if (d == nullptr) return s_->fail(MB_ENTITY_NOT_FOUND, "remove_entities: no set " + std::to_string(set));
  for (int i = 0; i < n; ++i) {
    if (d->ordered()) {
      // The last occurrence, as MOAB removes it.
      const auto it = std::find(d->sequence.rbegin(), d->sequence.rend(), es[i]);
      if (it != d->sequence.rend()) d->sequence.erase(std::next(it).base());
    } else {
      d->contents.erase(es[i]);
    }
  }
  return MB_SUCCESS;
}

ErrorCode Interface::remove_entities(EntityHandle set, const Range& es) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return remove_entities(set, v.data(), static_cast<int>(v.size()));
}

bool Interface::contains_entities(EntityHandle set, const EntityHandle* es, int n, int op) {
  if (set == 0) {
    bool all = true;
    bool any = false;
    for (int i = 0; i < n; ++i) {
      const bool in = s_->valid(es[i]) && es[i] != 0;
      all = all && in;
      any = any || in;
    }
    return op == INTERSECT ? all : any;
  }
  const SetData* d = set_of(*s_, set);
  if (d == nullptr) return false;
  bool all = true;
  bool any = false;
  for (int i = 0; i < n; ++i) {
    const bool in = d->ordered() ? std::find(d->sequence.begin(), d->sequence.end(), es[i]) != d->sequence.end()
                                 : d->contents.find(es[i]) != d->contents.end();
    all = all && in;
    any = any || in;
  }
  return op == INTERSECT ? all : any;
}

ErrorCode Interface::get_parent_meshsets(EntityHandle set, std::vector<EntityHandle>& parents, int num_hops) const {
  if (set_of(*s_, set) == nullptr) return MB_ENTITY_NOT_FOUND;
  const auto r = reach(*s_, set, num_hops, [](const SetData& d) { return d.parents; });
  parents.insert(parents.end(), r.begin(), r.end());
  return MB_SUCCESS;
}

ErrorCode Interface::get_parent_meshsets(EntityHandle set, Range& parents, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_parent_meshsets(set, v, num_hops);
  std::sort(v.begin(), v.end());
  for (const EntityHandle h : v) parents.insert(h);
  return rc;
}

ErrorCode Interface::get_child_meshsets(EntityHandle set, std::vector<EntityHandle>& children, int num_hops) const {
  if (set_of(*s_, set) == nullptr) return MB_ENTITY_NOT_FOUND;
  const auto r = reach(*s_, set, num_hops, [](const SetData& d) { return d.children; });
  children.insert(children.end(), r.begin(), r.end());
  return MB_SUCCESS;
}

ErrorCode Interface::get_child_meshsets(EntityHandle set, Range& children, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_child_meshsets(set, v, num_hops);
  std::sort(v.begin(), v.end());
  for (const EntityHandle h : v) children.insert(h);
  return rc;
}

ErrorCode Interface::get_contained_meshsets(EntityHandle set, std::vector<EntityHandle>& sets, int num_hops) const {
  if (set == 0) {
    for (std::size_t i = 0; i < s_->sets.size(); ++i) {
      if (s_->sets[i].alive) sets.push_back(make_handle(MBENTITYSET, static_cast<EntityID>(i + 1)));
    }
    return MB_SUCCESS;
  }
  if (set_of(*s_, set) == nullptr) return MB_ENTITY_NOT_FOUND;
  const auto r = reach(*s_, set, num_hops, contained);
  sets.insert(sets.end(), r.begin(), r.end());
  return MB_SUCCESS;
}

ErrorCode Interface::get_contained_meshsets(EntityHandle set, Range& sets, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_contained_meshsets(set, v, num_hops);
  std::sort(v.begin(), v.end());
  for (const EntityHandle h : v) sets.insert(h);
  return rc;
}

ErrorCode Interface::num_parent_meshsets(EntityHandle set, int* n, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_parent_meshsets(set, v, num_hops);
  *n = static_cast<int>(v.size());
  return rc;
}

ErrorCode Interface::num_child_meshsets(EntityHandle set, int* n, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_child_meshsets(set, v, num_hops);
  *n = static_cast<int>(v.size());
  return rc;
}

ErrorCode Interface::num_contained_meshsets(EntityHandle set, int* n, int num_hops) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_contained_meshsets(set, v, num_hops);
  *n = static_cast<int>(v.size());
  return rc;
}

ErrorCode Interface::add_parent_meshset(EntityHandle child, EntityHandle parent) {
  SetData* c = set_of(*s_, child);
  if (c == nullptr || set_of(*s_, parent) == nullptr) return MB_ENTITY_NOT_FOUND;
  if (std::find(c->parents.begin(), c->parents.end(), parent) == c->parents.end()) c->parents.push_back(parent);
  return MB_SUCCESS;
}

ErrorCode Interface::add_child_meshset(EntityHandle parent, EntityHandle child) {
  SetData* p = set_of(*s_, parent);
  if (p == nullptr || set_of(*s_, child) == nullptr) return MB_ENTITY_NOT_FOUND;
  if (std::find(p->children.begin(), p->children.end(), child) == p->children.end()) p->children.push_back(child);
  return MB_SUCCESS;
}

ErrorCode Interface::add_parent_child(EntityHandle parent, EntityHandle child) {
  const ErrorCode rc = add_child_meshset(parent, child);
  return rc != MB_SUCCESS ? rc : add_parent_meshset(child, parent);
}

ErrorCode Interface::remove_parent_meshset(EntityHandle child, EntityHandle parent) {
  SetData* c = set_of(*s_, child);
  if (c == nullptr) return MB_ENTITY_NOT_FOUND;
  c->parents.erase(std::remove(c->parents.begin(), c->parents.end(), parent), c->parents.end());
  return MB_SUCCESS;
}

ErrorCode Interface::remove_child_meshset(EntityHandle parent, EntityHandle child) {
  SetData* p = set_of(*s_, parent);
  if (p == nullptr) return MB_ENTITY_NOT_FOUND;
  p->children.erase(std::remove(p->children.begin(), p->children.end(), child), p->children.end());
  return MB_SUCCESS;
}

ErrorCode Interface::remove_parent_child(EntityHandle parent, EntityHandle child) {
  const ErrorCode rc = remove_child_meshset(parent, child);
  return rc != MB_SUCCESS ? rc : remove_parent_meshset(child, parent);
}

}  // namespace soma::compat
