// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_SOMA_LAYOUT_HPP_
#define SOMA_SRC_IO_SOMA_LAYOUT_HPP_

// What the SOMA format's writer and reader share (design §22.2): names, the
// version, the HDF5 types of value types, and the file access and transfer
// properties, MPI-IO and collective with a parallel HDF5 at more than one
// rank.

#include <cstdint>
#include <string>

#include <hdf5.h>

#include "hdf5.hpp"
#include "soma/fields/data_type.hpp"
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

// A collective (MPI-IO) file over `comm` when `collective`, else the default
// driver; metadata is read and written collectively too unless
// `independent_metadata` (when ranks open different datasets).
inline h5::Id file_access(const Comm& comm, bool collective, bool independent_metadata = false) {
  h5::Id fapl = h5::check(H5Pcreate(H5P_FILE_ACCESS), "make a file access property list");
#ifdef H5_HAVE_PARALLEL
  if (collective) {
    h5::check(H5Pset_fapl_mpio(fapl.get(), comm.native(), MPI_INFO_NULL), "use MPI-IO");
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
