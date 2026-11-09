// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_SOMA_LAYOUT_HPP_
#define SOMA_SRC_IO_SOMA_LAYOUT_HPP_

// What the SOMA format's writer and reader share: names, the
// version, the HDF5 types of value types, and the file access and transfer
// properties, MPI-IO and collective with a parallel HDF5 at more than one
// rank.

#include <cstdint>
#include <string>

#include <hdf5.h>

#include "hdf5.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::soma_file {

inline constexpr std::int64_t format_version = 1;
inline constexpr const char* root = "soma";

// Datasets of every part, concatenated in part order; /soma/parts/offsets
// holds where each part starts in each (columns named in its attribute
// "columns", one per line).
inline std::string dim_group(int d) { return "dim" + std::to_string(d); }
inline std::string field_group(std::size_t i) { return "fields/f" + std::to_string(i); }
inline std::string value_group(std::size_t i) { return "values/v" + std::to_string(i); }

// The HDF5 file type of a value type, little-endian as MOAB writes.
inline hid_t file_type(DataType t) {
  switch (t) {
    case DataType::i8: return H5T_STD_I8LE;
    case DataType::u8: return H5T_STD_U8LE;
    case DataType::i32: return H5T_STD_I32LE;
    case DataType::u32: return H5T_STD_U32LE;
    case DataType::i64: return H5T_STD_I64LE;
    case DataType::u64: return H5T_STD_U64LE;
    case DataType::f32: return H5T_IEEE_F32LE;
    case DataType::f64: return H5T_IEEE_F64LE;
  }
  return H5T_STD_U8LE;
}

inline hid_t memory_type(DataType t) {
  switch (t) {
    case DataType::i8: return H5T_NATIVE_INT8;
    case DataType::u8: return H5T_NATIVE_UINT8;
    case DataType::i32: return H5T_NATIVE_INT32;
    case DataType::u32: return H5T_NATIVE_UINT32;
    case DataType::i64: return H5T_NATIVE_INT64;
    case DataType::u64: return H5T_NATIVE_UINT64;
    case DataType::f32: return H5T_NATIVE_FLOAT;
    case DataType::f64: return H5T_NATIVE_DOUBLE;
  }
  return H5T_NATIVE_UINT8;
}

// Bytes of one value of type t.
inline std::size_t value_size(DataType t) {
  std::size_t n = 0;
  dispatch_data_type(t, [&]<class T>() { n = sizeof(T); });
  return n;
}

// True when the HDF5 is parallel: files at more than one rank are then
// opened with MPI-IO and moved with collective transfers.
constexpr bool parallel_hdf5() {
#ifdef H5_HAVE_PARALLEL
  return true;
#else
  return false;
#endif
}

// Why the hints and aggregator count cannot be used at `ranks` ranks, or
// nothing.
inline std::string access_error(const MpiIoHints& h, int aggregators, int ranks) {
  if (aggregators < 0 || aggregators > ranks) {
    return "aggregators must be in [0, " + std::to_string(ranks) + "], not " + std::to_string(aggregators);
  }
  if (aggregators > 1 && !parallel_hdf5()) return "more than one aggregator needs a parallel HDF5";
  if (h.cb_nodes < 0 || h.striping_factor < 0) return "a negative count in the MPI-IO hints";
  if ((h.alignment & (h.alignment - 1)) != 0) {
    return "alignment " + std::to_string(h.alignment) + " is not a power of two";
  }
  return "";
}

// The first rank of the group of rank r, of `m` aggregators over `p` ranks:
// group floor(rm/p), whose first rank is ceil(gp/m). A group's ranks are
// consecutive, so its parts' rows are contiguous in every dataset.
inline int leader_of(int r, int m, int p) {
  const auto g = static_cast<std::int64_t>(r) * m / p;
  return static_cast<int>((g * p + m - 1) / m);
}

// A hash of the options every rank must give alike.
inline std::uint64_t access_hash(const MpiIoHints& h, int aggregators) {
  std::uint64_t x = 1469598103934665603ULL;
  for (const std::uint64_t v : {static_cast<std::uint64_t>(aggregators), static_cast<std::uint64_t>(h.cb_nodes),
                                static_cast<std::uint64_t>(h.cb_buffer_size),
                                static_cast<std::uint64_t>(h.collective_buffering ? 1 + *h.collective_buffering : 0),
                                static_cast<std::uint64_t>(h.striping_factor),
                                static_cast<std::uint64_t>(h.striping_unit),
                                static_cast<std::uint64_t>(h.alignment)}) {
    x = (x ^ v) * 1099511628211ULL;
  }
  return x;
}

// A collective (MPI-IO) file over `comm` when `collective`, else the default
// driver; metadata is read and written collectively too unless
// `independent_metadata` (when ranks open different datasets). The hints go
// to MPI-IO; the alignment applies with either driver. Objects are created
// in the HDF5 1.14 file format, whatever newer library writes them.
inline h5::Id file_access(const Comm& comm, bool collective, bool independent_metadata = false,
                          const MpiIoHints& hints = {}) {
  h5::Id fapl = h5::check(H5Pcreate(H5P_FILE_ACCESS), "make a file access property list");
  h5::check(H5Pset_libver_bounds(fapl.get(), H5F_LIBVER_V114, H5F_LIBVER_LATEST), "use the HDF5 1.14 format");
  if (hints.alignment > 1) {
    h5::check(H5Pset_alignment(fapl.get(), hints.alignment, hints.alignment), "align datasets");
  }
#ifdef H5_HAVE_PARALLEL
  if (collective) {
    MPI_Info info = MPI_INFO_NULL;
    const auto set = [&](const char* key, const std::string& value) {
      if (info == MPI_INFO_NULL) MPI_Info_create(&info);
      MPI_Info_set(info, key, value.c_str());
    };
    if (hints.cb_nodes > 0) set("cb_nodes", std::to_string(hints.cb_nodes));
    if (hints.cb_buffer_size > 0) set("cb_buffer_size", std::to_string(hints.cb_buffer_size));
    if (hints.collective_buffering) {
      const char* v = *hints.collective_buffering ? "enable" : "disable";
      set("romio_cb_write", v);
      set("romio_cb_read", v);
    }
    if (hints.striping_factor > 0) set("striping_factor", std::to_string(hints.striping_factor));
    if (hints.striping_unit > 0) set("striping_unit", std::to_string(hints.striping_unit));
    const herr_t mpio = H5Pset_fapl_mpio(fapl.get(), comm.native(), info);  // copies the info
    if (info != MPI_INFO_NULL) MPI_Info_free(&info);
    h5::check(mpio, "use MPI-IO");
    if (!independent_metadata) {
      h5::check(H5Pset_all_coll_metadata_ops(fapl.get(), true), "read metadata collectively");
      h5::check(H5Pset_coll_metadata_write(fapl.get(), true), "write metadata collectively");
    }
  }
#else
  (void)comm;
  (void)collective;
  (void)independent_metadata;
#endif
  return fapl;
}

// Collective transfers when `collective`, else the default.
inline h5::Id transfer(bool collective) {
  h5::Id dxpl = h5::check(H5Pcreate(H5P_DATASET_XFER), "make a transfer property list");
#ifdef H5_HAVE_PARALLEL
  if (collective) h5::check(H5Pset_dxpl_mpio(dxpl.get(), H5FD_MPIO_COLLECTIVE), "transfer collectively");
#else
  (void)collective;
#endif
  return dxpl;
}

}  // namespace soma::soma_file

#endif  // SOMA_SRC_IO_SOMA_LAYOUT_HPP_
