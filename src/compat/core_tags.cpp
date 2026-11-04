// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Interface: tags.

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "core_state.hpp"
#include "soma/compat/core.hpp"

namespace soma::compat {

int value_size(DataType t) {
  switch (t) {
    case MB_TYPE_INTEGER: return static_cast<int>(sizeof(int));
    case MB_TYPE_DOUBLE: return static_cast<int>(sizeof(double));
    case MB_TYPE_HANDLE: return static_cast<int>(sizeof(EntityHandle));
    default: return 1;  // opaque bytes, bit tags
  }
}

unsigned char* TagInfo::slot(EntityType t, EntityID id) {
  const auto k = static_cast<std::size_t>(t);
  const auto i = static_cast<std::size_t>(id - 1);
  auto& h = has[k];
  auto& d = data[k];
  if (h.size() <= i) h.resize(i + 1, 0);
  if (d.size() < (i + 1) * static_cast<std::size_t>(bytes)) d.resize((i + 1) * static_cast<std::size_t>(bytes), 0);
  unsigned char* p = d.data() + i * static_cast<std::size_t>(bytes);
  if (h[i] == 0 && has_def) std::memcpy(p, def.data(), static_cast<std::size_t>(bytes));
  return p;
}

void detail::allocate_dense(CoreState& s, TagInfo& t, EntityType type, EntityID id) {
  if (t.storage != MB_TAG_DENSE || t.varlen) return;
  const detail::IdBlock* b = s.block_of(type, id);
  if (b == nullptr) return;
  auto& allocated = t.dense_blocks[static_cast<std::size_t>(type)];
  if (std::find(allocated.begin(), allocated.end(), b->start) != allocated.end()) return;
  allocated.push_back(b->start);
  if (!t.has_def) return;
  for (EntityID x = b->start; x < b->start + b->capacity; ++x) {
    if (s.id_free(type, x) || t.has_value(type, x)) continue;
    t.slot(type, x);
    t.has[static_cast<std::size_t>(type)][static_cast<std::size_t>(x - 1)] = 1;
  }
}

namespace {

using detail::CoreState;

// A handle a tag can hold a value on: a live entity or set, or the root set.
bool taggable(const CoreState& s, EntityHandle h) { return s.valid(h); }

}  // namespace

ErrorCode Interface::tag_get_handle(const char* name, int size, DataType type, Tag& tag, unsigned flags,
                                    const void* default_value, bool* created) {
  if (created != nullptr) *created = false;
  if (name == nullptr || *name == '\0') return s_->fail(MB_FAILURE, "tag_get_handle: a tag needs a name");
  const bool varlen = (flags & MB_TAG_VARLEN) != 0;
  const bool in_bytes = (flags & MB_TAG_BYTES) != 0;
  const int vsize = value_size(type);
  if (type == MB_TYPE_BIT && (flags & MB_TAG_BIT) == 0 && (flags & (MB_TAG_DENSE | MB_TAG_SPARSE | MB_TAG_MESH)) != 0) {
    return s_->fail(MB_TYPE_OUT_OF_RANGE, "tag_get_handle: bit values need bit storage");
  }
  int length = size;
  if (!varlen && in_bytes && type != MB_TYPE_BIT) {
    if (size % vsize != 0) return s_->fail(MB_INVALID_SIZE, "tag_get_handle: size is no whole number of values");
    length = size / vsize;
  }
  if (TagInfo* t = s_->find_tag(name)) {
    if ((flags & MB_TAG_EXCL) != 0 && (flags & MB_TAG_CREAT) != 0) {
      return s_->fail(MB_ALREADY_ALLOCATED, std::string("tag '") + name + "' exists");
    }
    if ((flags & MB_TAG_ANY) == 0) {
      const bool opaque_ok = (type == MB_TYPE_OPAQUE || t->type == MB_TYPE_OPAQUE) && (flags & MB_TAG_NOOPQ) == 0;
      if (t->type != type && !opaque_ok) return s_->fail(MB_TYPE_OUT_OF_RANGE, std::string("tag '") + name + "' has another type");
      if (t->varlen != varlen) return s_->fail(MB_VARIABLE_DATA_LENGTH, std::string("tag '") + name + "' differs in length kind");
      if (!varlen) {
        const bool same = t->type == MB_TYPE_BIT ? t->length == size
                          : t->type == type      ? t->length == length
                                                 : t->bytes == (in_bytes ? size : size * vsize);
        if (!same) return s_->fail(MB_INVALID_SIZE, std::string("tag '") + name + "' has another size");
      }
      if ((flags & MB_TAG_STORE) != 0) {
        const int want = flags & (MB_TAG_DENSE | MB_TAG_SPARSE | MB_TAG_MESH);
        if (want != 0 ? (want & t->storage) == 0 : t->storage != MB_TAG_BIT) {
          return s_->fail(MB_TYPE_OUT_OF_RANGE, std::string("tag '") + name + "' has other storage");
        }
      }
    }
    tag = t;
    return MB_SUCCESS;
  }
  if ((flags & MB_TAG_CREAT) == 0) return s_->fail(MB_TAG_NOT_FOUND, std::string("no tag '") + name + "'");
  if (!varlen && size < 1) return s_->fail(MB_INVALID_SIZE, "tag_get_handle: size must be positive");
  if (type == MB_TYPE_BIT && size > 8) return s_->fail(MB_INVALID_SIZE, "tag_get_handle: at most 8 bits");
  auto t = std::make_unique<TagInfo>();
  t->name = name;
  t->type = type;
  t->varlen = varlen;
  t->storage = (flags & MB_TAG_DENSE) != 0  ? MB_TAG_DENSE
               : (flags & MB_TAG_MESH) != 0 ? MB_TAG_MESH
               : type == MB_TYPE_BIT        ? MB_TAG_BIT
                                            : MB_TAG_SPARSE;
  if (type == MB_TYPE_BIT) {
    t->length = size;
    t->bytes = 1;
  } else {
    t->length = length;
    t->bytes = varlen ? 0 : length * vsize;
  }
  if (default_value != nullptr) {
    const int n = varlen ? (in_bytes ? size : size * vsize) : t->bytes;
    const auto* p = static_cast<const unsigned char*>(default_value);
    t->def.assign(p, p + n);
    t->has_def = true;
  }
  tag = t.get();
  s_->tags.push_back(std::move(t));
  if (created != nullptr) *created = true;
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_handle(const char* name, int size, DataType type, Tag& tag, unsigned flags,
                                    const void* default_value) const {
  if ((flags & MB_TAG_CREAT) != 0) return s_->fail(MB_TAG_NOT_FOUND, "tag_get_handle: a const instance creates nothing");
  return const_cast<Interface*>(this)->tag_get_handle(name, size, type, tag, flags, default_value, nullptr);
}

ErrorCode Interface::tag_get_handle(const char* name, Tag& tag) const {
  TagInfo* t = name != nullptr ? s_->find_tag(name) : nullptr;
  if (t == nullptr) return s_->fail(MB_TAG_NOT_FOUND, std::string("no tag '") + (name ? name : "") + "'");
  tag = t;
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_name(Tag tag, std::string& name) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  name = t->name;
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_length(Tag tag, int& length) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  length = t->varlen ? MB_VARIABLE_LENGTH : t->length;
  return t->varlen ? MB_VARIABLE_DATA_LENGTH : MB_SUCCESS;
}

ErrorCode Interface::tag_get_bytes(Tag tag, int& bytes) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  bytes = t->varlen ? MB_VARIABLE_LENGTH : t->bytes;
  return t->varlen ? MB_VARIABLE_DATA_LENGTH : MB_SUCCESS;
}

ErrorCode Interface::tag_get_type(Tag tag, TagType& type) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  type = static_cast<TagType>(t->storage | (t->varlen ? MB_TAG_VARLEN : 0));
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_data_type(Tag tag, DataType& type) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  type = t->type;
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_default_value(Tag tag, void* value) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  if (!t->has_def) return MB_ENTITY_NOT_FOUND;
  if (t->varlen) return MB_VARIABLE_DATA_LENGTH;
  std::memcpy(value, t->def.data(), t->def.size());
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_default_value(Tag tag, const void*& value, int& size) const {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  if (!t->has_def) return MB_ENTITY_NOT_FOUND;
  value = t->def.data();
  size = static_cast<int>(t->def.size()) / (t->type == MB_TYPE_BIT ? 1 : value_size(t->type));
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_tags(std::vector<Tag>& tags) const {
  for (const auto& t : s_->tags) tags.push_back(t.get());
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_tags_on_entity(EntityHandle e, std::vector<Tag>& tags) const {
  s_->flush();
  if (!taggable(*s_, e)) return MB_ENTITY_NOT_FOUND;
  for (const auto& t : s_->tags) {
    if (e == 0 ? t->has_mesh : t->has_value(type_of(e), id_of(e))) tags.push_back(t.get());
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_data(Tag tag, const EntityHandle* es, int n, void* data) const {
  s_->flush();
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag_get_data: no such tag");
  if (t->varlen) return s_->fail(MB_VARIABLE_DATA_LENGTH, "tag_get_data: use tag_get_by_ptr for '" + t->name + "'");
  auto* out = static_cast<unsigned char*>(data);
  const auto b = static_cast<std::size_t>(t->bytes);
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    if (!taggable(*s_, h)) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_get_data: no entity " + std::to_string(h));
    const unsigned char* src = nullptr;
    if (h == 0) {
      if (t->has_mesh) src = t->mesh_value.data();
    } else if (t->has_value(type_of(h), id_of(h))) {
      src = t->data[static_cast<std::size_t>(type_of(h))].data() + static_cast<std::size_t>(id_of(h) - 1) * b;
    }
    if (src == nullptr && t->has_def) src = t->def.data();
    if (src == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag '" + t->name + "' has no value on " + std::to_string(h));
    std::memcpy(out + static_cast<std::size_t>(i) * b, src, b);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_data(Tag tag, const Range& es, void* data) const {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_get_data(tag, v.data(), static_cast<int>(v.size()), data);
}

ErrorCode Interface::tag_set_data(Tag tag, const EntityHandle* es, int n, const void* data) {
  s_->flush();
  TagInfo* t = s_->own(tag);
  if (t == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag_set_data: no such tag");
  if (t->varlen) return s_->fail(MB_VARIABLE_DATA_LENGTH, "tag_set_data: use tag_set_by_ptr for '" + t->name + "'");
  for (int i = 0; i < n; ++i) {
    if (!taggable(*s_, es[i])) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_set_data: no entity " + std::to_string(es[i]));
  }
  const auto* in = static_cast<const unsigned char*>(data);
  const auto b = static_cast<std::size_t>(t->bytes);
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    const unsigned char* src = in + static_cast<std::size_t>(i) * b;
    if (h == 0) {
      t->mesh_value.assign(src, src + b);
      t->has_mesh = true;
      continue;
    }
    std::memcpy(t->slot(type_of(h), id_of(h)), src, b);
    t->has[static_cast<std::size_t>(type_of(h))][static_cast<std::size_t>(id_of(h) - 1)] = 1;
    detail::allocate_dense(*s_, *t, type_of(h), id_of(h));
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_set_data(Tag tag, const Range& es, const void* data) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_set_data(tag, v.data(), static_cast<int>(v.size()), data);
}

ErrorCode Interface::tag_get_by_ptr(Tag tag, const EntityHandle* es, int n, const void** data, int* sizes) const {
  s_->flush();
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag_get_by_ptr: no such tag");
  const int per = t->type == MB_TYPE_BIT ? 1 : value_size(t->type);
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    if (!taggable(*s_, h)) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_get_by_ptr: no entity " + std::to_string(h));
    const unsigned char* src = nullptr;
    std::size_t len = t->varlen ? 0 : static_cast<std::size_t>(t->bytes);
    if (h == 0) {
      if (t->has_mesh) {
        src = t->mesh_value.data();
        len = t->mesh_value.size();
      }
    } else if (t->has_value(type_of(h), id_of(h))) {
      const auto k = static_cast<std::size_t>(type_of(h));
      const auto at = static_cast<std::size_t>(id_of(h) - 1);
      if (t->varlen) {
        src = t->vdata[k][at].data();
        len = t->vdata[k][at].size();
      } else {
        src = t->data[k].data() + at * static_cast<std::size_t>(t->bytes);
      }
    }
    if (src == nullptr && t->has_def) {
      src = t->def.data();
      len = t->def.size();
    }
    if (src == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag '" + t->name + "' has no value on " + std::to_string(h));
    data[i] = src;
    if (sizes != nullptr) sizes[i] = static_cast<int>(len) / per;
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_get_by_ptr(Tag tag, const Range& es, const void** data, int* sizes) const {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_get_by_ptr(tag, v.data(), static_cast<int>(v.size()), data, sizes);
}

ErrorCode Interface::tag_set_by_ptr(Tag tag, const EntityHandle* es, int n, void const* const* data, const int* sizes) {
  s_->flush();
  TagInfo* t = s_->own(tag);
  if (t == nullptr) return s_->fail(MB_TAG_NOT_FOUND, "tag_set_by_ptr: no such tag");
  if (t->varlen && sizes == nullptr) return s_->fail(MB_VARIABLE_DATA_LENGTH, "tag_set_by_ptr: sizes needed for '" + t->name + "'");
  for (int i = 0; i < n; ++i) {
    if (!taggable(*s_, es[i])) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_set_by_ptr: no entity " + std::to_string(es[i]));
  }
  const int per = t->type == MB_TYPE_BIT ? 1 : value_size(t->type);
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    const auto* src = static_cast<const unsigned char*>(data[i]);
    const std::size_t len = t->varlen ? static_cast<std::size_t>(sizes[i] * per) : static_cast<std::size_t>(t->bytes);
    if (h == 0) {
      t->mesh_value.assign(src, src + len);
      t->has_mesh = true;
      continue;
    }
    const auto k = static_cast<std::size_t>(type_of(h));
    const auto at = static_cast<std::size_t>(id_of(h) - 1);
    if (t->varlen) {
      if (t->vdata[k].size() <= at) t->vdata[k].resize(at + 1);
      if (t->has[k].size() <= at) t->has[k].resize(at + 1, 0);
      t->vdata[k][at].assign(src, src + len);
    } else {
      std::memcpy(t->slot(type_of(h), id_of(h)), src, len);
    }
    t->has[k][at] = 1;
    detail::allocate_dense(*s_, *t, type_of(h), id_of(h));
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_set_by_ptr(Tag tag, const Range& es, void const* const* data, const int* sizes) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_set_by_ptr(tag, v.data(), static_cast<int>(v.size()), data, sizes);
}

ErrorCode Interface::tag_clear_data(Tag tag, const EntityHandle* es, int n, const void* value, int size) {
  const TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  std::vector<const void*> ptrs(static_cast<std::size_t>(n), value);
  std::vector<int> sizes(static_cast<std::size_t>(n), size);
  return tag_set_by_ptr(tag, es, n, ptrs.data(), t->varlen ? sizes.data() : nullptr);
}

ErrorCode Interface::tag_clear_data(Tag tag, const Range& es, const void* value, int size) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_clear_data(tag, v.data(), static_cast<int>(v.size()), value, size);
}

ErrorCode Interface::tag_delete_data(Tag tag, const EntityHandle* es, int n) {
  s_->flush();
  TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    if (!taggable(*s_, h)) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_delete_data: no entity " + std::to_string(h));
    if (h == 0) {
      t->mesh_value.clear();
      t->has_mesh = false;
    } else {
      t->erase(type_of(h), id_of(h));
    }
  }
  return MB_SUCCESS;
}

ErrorCode Interface::tag_delete_data(Tag tag, const Range& es) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return tag_delete_data(tag, v.data(), static_cast<int>(v.size()));
}

ErrorCode Interface::tag_delete(Tag tag) {
  for (auto it = s_->tags.begin(); it != s_->tags.end(); ++it) {
    if (it->get() == tag) {
      s_->tags.erase(it);
      return MB_SUCCESS;
    }
  }
  return MB_TAG_NOT_FOUND;
}

ErrorCode Interface::tag_iterate(Tag tag, Range::const_iterator begin, Range::const_iterator end, int& count,
                                 void*& data, bool allocate) {
  s_->flush();
  TagInfo* t = s_->own(tag);
  if (t == nullptr) return MB_TAG_NOT_FOUND;
  if (t->storage != MB_TAG_DENSE || t->varlen) return s_->fail(MB_TYPE_OUT_OF_RANGE, "tag_iterate: dense fixed-length tags only");
  if (begin == end) {
    count = 0;
    data = nullptr;
    return MB_SUCCESS;
  }
  const EntityHandle first = *begin;
  const EntityType type = type_of(first);
  // A run of live entities of one type with consecutive ids, up to end.
  EntityHandle last = first;
  auto it = begin;
  ++it;
  while (it != end && *it == last + 1 && type_of(*it) == type && taggable(*s_, *it)) {
    last = *it;
    ++it;
  }
  if (!taggable(*s_, first) || first == 0) return s_->fail(MB_ENTITY_NOT_FOUND, "tag_iterate: no entity " + std::to_string(first));
  count = static_cast<int>(last - first + 1);
  const auto k = static_cast<std::size_t>(type);
  if (allocate) {
    for (EntityHandle h = first; h <= last; ++h) {
      t->slot(type, id_of(h));
      t->has[k][static_cast<std::size_t>(id_of(h) - 1)] = 1;
      detail::allocate_dense(*s_, *t, type, id_of(h));
    }
  } else if (!t->has_value(type, id_of(last))) {
    t->slot(type, id_of(last));
  }
  data = t->data[k].data() + static_cast<std::size_t>(id_of(first) - 1) * static_cast<std::size_t>(t->bytes);
  return MB_SUCCESS;
}

ErrorCode Interface::tag_commit(Tag tag) { return s_->own(tag) != nullptr ? MB_SUCCESS : MB_TAG_NOT_FOUND; }
ErrorCode Interface::tag_commit_all() { return MB_SUCCESS; }
bool Interface::tag_has_pending(Tag /*tag*/) const { return false; }

ErrorCode Interface::get_entities_by_type_and_tag(EntityHandle meshset, EntityType type, const Tag* tags,
                                                  const void* const* values, int num_tags, Range& out, int op,
                                                  bool recursive) const {
  Range candidates;
  const ErrorCode rc = get_entities_by_type(meshset, type, candidates, recursive);
  if (rc != MB_SUCCESS) return rc;
  for (int i = 0; i < num_tags; ++i) {
    if (s_->own(tags[i]) == nullptr) return MB_TAG_NOT_FOUND;
  }
  Range found;
  for (const EntityHandle h : candidates) {
    bool all = true;
    bool any = false;
    for (int i = 0; i < num_tags; ++i) {
      const TagInfo* t = tags[i];
      bool match = t->has_value(type_of(h), id_of(h));
      if (match && values != nullptr && values[i] != nullptr) {
        const auto k = static_cast<std::size_t>(type_of(h));
        const auto at = static_cast<std::size_t>(id_of(h) - 1);
        match = t->varlen ? false
                          : std::memcmp(t->data[k].data() + at * static_cast<std::size_t>(t->bytes), values[i],
                                        static_cast<std::size_t>(t->bytes)) == 0;
      }
      all = all && match;
      any = any || match;
    }
    if (op == INTERSECT ? all : any) found.insert(h);
  }
  out.merge(found);
  return MB_SUCCESS;
}

}  // namespace soma::compat
