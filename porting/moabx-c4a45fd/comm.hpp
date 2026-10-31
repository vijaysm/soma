#ifndef MOABX_PP_COMM_HPP_
#define MOABX_PP_COMM_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "moabx/pp/backend.hpp"

#if PP_USE_MPI
#include <mpi.h>
#endif

namespace moabx::pp {

class request {
 public:
  request() = default;

  bool active() const noexcept { return active_; }
  bool test();
  void wait();

 private:
  friend class communicator;
#if PP_USE_MPI
  explicit request(MPI_Request req) : request_(req), active_(true) {}
  MPI_Request request_ = MPI_REQUEST_NULL;
#endif
  bool active_ = false;
};

class communicator {
 public:
  communicator();

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool mpi_active() const noexcept { return mpi_active_; }
  void barrier() const;

  template <class T>
  request isend(std::span<const T> buf, int dst, int tag) const;

  template <class T>
  request irecv(std::span<T> buf, int src, int tag) const;

  template <class T>
  T allreduce_sum(T value) const;

  template <class T>
  T allreduce_min(T value) const;

  template <class T>
  T allreduce_max(T value) const;

 private:
#if PP_USE_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
  int rank_ = 0;
  int size_ = 1;
  bool mpi_active_ = false;
};

template <class T>
constexpr bool mpi_type_supported = false;

template <>
inline constexpr bool mpi_type_supported<int> = true;
template <>
inline constexpr bool mpi_type_supported<std::int64_t> = true;
template <>
inline constexpr bool mpi_type_supported<double> = true;

#if PP_USE_MPI
template <class T>
MPI_Datatype mpi_datatype();

template <>
inline MPI_Datatype mpi_datatype<int>() {
  return MPI_INT;
}
template <>
inline MPI_Datatype mpi_datatype<std::int64_t>() {
  return MPI_INT64_T;
}
template <>
inline MPI_Datatype mpi_datatype<double>() {
  return MPI_DOUBLE;
}
#endif

template <class T>
request communicator::isend(std::span<const T> buf, int dst, int tag) const {
  static_assert(mpi_type_supported<T>, "Unsupported MPI datatype.");
#if PP_USE_MPI
  if (!mpi_active_) {
    return request{};
  }
  MPI_Request req = MPI_REQUEST_NULL;
  MPI_Isend(const_cast<T*>(buf.data()), static_cast<int>(buf.size()), mpi_datatype<T>(), dst, tag, comm_, &req);
  return request(req);
#else
  (void)buf;
  (void)dst;
  (void)tag;
  throw std::runtime_error("MPI support is disabled.");
#endif
}

template <class T>
request communicator::irecv(std::span<T> buf, int src, int tag) const {
  static_assert(mpi_type_supported<T>, "Unsupported MPI datatype.");
#if PP_USE_MPI
  if (!mpi_active_) {
    return request{};
  }
  MPI_Request req = MPI_REQUEST_NULL;
  MPI_Irecv(buf.data(), static_cast<int>(buf.size()), mpi_datatype<T>(), src, tag, comm_, &req);
  return request(req);
#else
  (void)buf;
  (void)src;
  (void)tag;
  throw std::runtime_error("MPI support is disabled.");
#endif
}

template <class T>
T communicator::allreduce_sum(T value) const {
  static_assert(mpi_type_supported<T>, "Unsupported MPI datatype.");
#if PP_USE_MPI
  if (!mpi_active_) {
    return value;
  }
  T out{};
  MPI_Allreduce(&value, &out, 1, mpi_datatype<T>(), MPI_SUM, comm_);
  return out;
#else
  return value;
#endif
}

template <class T>
T communicator::allreduce_min(T value) const {
  static_assert(mpi_type_supported<T>, "Unsupported MPI datatype.");
#if PP_USE_MPI
  if (!mpi_active_) {
    return value;
  }
  T out{};
  MPI_Allreduce(&value, &out, 1, mpi_datatype<T>(), MPI_MIN, comm_);
  return out;
#else
  return value;
#endif
}

template <class T>
T communicator::allreduce_max(T value) const {
  static_assert(mpi_type_supported<T>, "Unsupported MPI datatype.");
#if PP_USE_MPI
  if (!mpi_active_) {
    return value;
  }
  T out{};
  MPI_Allreduce(&value, &out, 1, mpi_datatype<T>(), MPI_MAX, comm_);
  return out;
#else
  return value;
#endif
}

}  // namespace moabx::pp

#endif  // MOABX_PP_COMM_HPP_
