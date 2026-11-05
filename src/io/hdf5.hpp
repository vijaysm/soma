// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_HDF5_HPP_
#define SOMA_SRC_IO_HDF5_HPP_

// Thin RAII over the HDF5 C API, private to the library. Every failure throws
// h5::Error with a message saying what could not be done; the HDF5 error stack
// is silenced while a Session lives, so nothing is printed.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hdf5.h>

namespace soma::h5 {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Owns one HDF5 identifier of any kind.
class Id {
 public:
  Id() = default;
  explicit Id(hid_t id) noexcept : id_(id) {}
  Id(const Id&) = delete;
  Id& operator=(const Id&) = delete;
  Id(Id&& o) noexcept : id_(std::exchange(o.id_, H5I_INVALID_HID)) {}
  Id& operator=(Id&& o) noexcept {
    Id tmp(std::move(o));
    std::swap(id_, tmp.id_);
    return *this;
  }
  ~Id() {
    if (id_ >= 0) H5Idec_ref(id_);
  }
  hid_t get() const noexcept { return id_; }

 private:
  hid_t id_ = H5I_INVALID_HID;
};

inline Id check(hid_t id, const std::string& what) {
  if (id < 0) throw Error("HDF5 could not " + what);
  return Id(id);
}

inline void check(herr_t rc, const std::string& what) {
  if (rc < 0) throw Error("HDF5 could not " + what);
}

// Turns off HDF5's automatic error printing for its lifetime.
class Session {
 public:
  Session() noexcept {
    H5Eget_auto2(H5E_DEFAULT, &func_, &data_);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }

 private:
  H5E_auto2_t func_ = nullptr;
  void* data_ = nullptr;
};

// True if every component of the relative path exists below loc.
inline bool exists(hid_t loc, const std::string& path) {
  std::size_t at = 0;
  while (at <= path.size()) {
    const std::size_t slash = path.find('/', at);
    const std::string prefix = path.substr(0, slash);
    if (H5Lexists(loc, prefix.c_str(), H5P_DEFAULT) <= 0) return false;
    if (slash == std::string::npos) break;
    at = slash + 1;
  }
  return true;
}

inline bool has_attribute(hid_t obj, const char* name) { return H5Aexists(obj, name) > 0; }

inline Id open_group(hid_t loc, const std::string& path) {
  return check(H5Gopen2(loc, path.c_str(), H5P_DEFAULT), "open the group " + path);
}

inline Id open_dataset(hid_t loc, const std::string& path) {
  return check(H5Dopen2(loc, path.c_str(), H5P_DEFAULT), "open the dataset " + path);
}

// Names of the links in a group, in name order.
inline std::vector<std::string> children(hid_t group, const std::string& where) {
  std::vector<std::string> names;
  const auto collect = [](hid_t, const char* name, const H5L_info2_t*, void* out) -> herr_t {
    // No exception may unwind through HDF5's C frames.
    try {
      static_cast<std::vector<std::string>*>(out)->emplace_back(name);
      return 0;
    } catch (...) {
      return -1;
    }
  };
  if (H5Literate2(group, H5_INDEX_NAME, H5_ITER_INC, nullptr, collect, &names) < 0) {
    throw Error("HDF5 could not list the members of " + where);
  }
  return names;
}

inline std::vector<hsize_t> extent(hid_t dataset, const std::string& where) {
  const Id space = check(H5Dget_space(dataset), "get the dataspace of " + where);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank < 0) throw Error("HDF5 could not get the rank of " + where);
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  if (rank > 0) check(H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr), "get the extent of " + where);
  return dims;
}

template <class T>
hid_t native() {
  if constexpr (std::is_same_v<T, std::int8_t>) return H5T_NATIVE_INT8;
  else if constexpr (std::is_same_v<T, std::uint8_t>) return H5T_NATIVE_UINT8;
  else if constexpr (std::is_same_v<T, std::int32_t>) return H5T_NATIVE_INT32;
  else if constexpr (std::is_same_v<T, std::uint32_t>) return H5T_NATIVE_UINT32;
  else if constexpr (std::is_same_v<T, std::int64_t>) return H5T_NATIVE_INT64;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return H5T_NATIVE_UINT64;
  else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
  else static_assert(sizeof(T) == 0, "no native HDF5 type");
}

// The transfer property list of the reads below: H5P_DEFAULT, or a
// collective one while a CollectiveReads lives, when every rank makes every
// read, rows or none.
struct Transfer {
  hid_t dxpl = H5P_DEFAULT;
  bool collective = false;
};

inline Transfer& transfer() {
  static thread_local Transfer t;
  return t;
}

class CollectiveReads {
 public:
  explicit CollectiveReads(hid_t dxpl) noexcept : saved_(transfer()) { transfer() = {dxpl, true}; }
  CollectiveReads(const CollectiveReads&) = delete;
  CollectiveReads& operator=(const CollectiveReads&) = delete;
  ~CollectiveReads() { transfer() = saved_; }

 private:
  Transfer saved_;
};

// Reads a whole dataset into out, which holds it in memory type `mem`.
inline void read(hid_t dataset, hid_t mem, void* out, const std::string& where) {
  static std::uint64_t none = 0;  // HDF5 wants a buffer even for no values
  check(H5Dread(dataset, mem, H5S_ALL, H5S_ALL, transfer().dxpl, out != nullptr ? out : &none), "read " + where);
}

// Rows [first, first + count) of a dataset of rank 1 or 2 (every column)
// into out, which holds them in memory type `mem`. With collective reads a
// rank reading no rows still takes part.
inline void read_rows(hid_t dataset, hid_t mem, hsize_t first, hsize_t count, void* out, const std::string& where) {
  if (count == 0) {
    if (!transfer().collective) return;
    const Id file = check(H5Dget_space(dataset), "get the dataspace of " + where);
    check(H5Sselect_none(file.get()), "select no rows of " + where);
    const Id memory = check(H5Scopy(file.get()), "copy the dataspace of " + where);
    check(H5Sselect_none(memory.get()), "select no rows of " + where);
    static std::uint64_t none = 0;
    check(H5Dread(dataset, mem, memory.get(), file.get(), transfer().dxpl, out != nullptr ? out : &none),
          "read no rows of " + where);
    return;
  }
  const Id file = check(H5Dget_space(dataset), "get the dataspace of " + where);
  const int rank = H5Sget_simple_extent_ndims(file.get());
  if (rank != 1 && rank != 2) throw Error(where + " must have rank 1 or 2 to be read in rows");
  hsize_t dims[2] = {0, 1};
  check(H5Sget_simple_extent_dims(file.get(), dims, nullptr), "get the extent of " + where);
  if (first > dims[0] || count > dims[0] - first) throw Error("rows beyond the end of " + where);
  const hsize_t start[2] = {first, 0};
  const hsize_t extent[2] = {count, rank == 2 ? dims[1] : 1};
  check(H5Sselect_hyperslab(file.get(), H5S_SELECT_SET, start, nullptr, extent, nullptr), "select rows of " + where);
  const Id memory = check(H5Screate_simple(rank, extent, nullptr), "make a dataspace for rows of " + where);
  check(H5Dread(dataset, mem, memory.get(), file.get(), transfer().dxpl, out), "read rows of " + where);
}

template <class T>
std::vector<T> read_rows(hid_t dataset, hsize_t first, hsize_t count, std::size_t columns, const std::string& where) {
  std::vector<T> out(static_cast<std::size_t>(count) * columns);
  read_rows(dataset, native<T>(), first, count, out.data(), where);
  return out;
}

template <class T>
std::vector<T> read_all(hid_t dataset, std::size_t count, const std::string& where) {
  std::vector<T> out(count);
  if (count > 0 || transfer().collective) read(dataset, native<T>(), count > 0 ? out.data() : nullptr, where);
  return out;
}

// H5Aread fills the whole attribute, so one holding more than a single value
// would overrun the caller's buffer.
inline void require_one_value(hid_t attribute, const std::string& what) {
  const Id space = check(H5Aget_space(attribute), "get the dataspace of " + what);
  if (H5Sget_simple_extent_npoints(space.get()) != 1) throw Error(what + " must hold exactly one value");
}

// A scalar attribute converted to T.
template <class T>
T read_attribute(hid_t obj, const char* name, const std::string& where) {
  const Id a = check(H5Aopen(obj, name, H5P_DEFAULT), std::string("open the attribute ") + name + " of " + where);
  require_one_value(a.get(), std::string("the attribute ") + name + " of " + where);
  T value{};
  check(H5Aread(a.get(), native<T>(), &value), std::string("read the attribute ") + name + " of " + where);
  return value;
}

}  // namespace soma::h5

#endif  // SOMA_SRC_IO_HDF5_HPP_
