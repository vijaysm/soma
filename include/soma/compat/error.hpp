// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_ERROR_HPP_
#define SOMA_COMPAT_ERROR_HPP_

/// \file
/// MOAB's error macros, under SOMA_ names (the drop-in headers give them
/// MOAB's names). An error is reported on stderr as MOAB reports it:
///
/// ```
///   MOAB ERROR: --------------------- Error Message ------------------------------------
///   MOAB ERROR: <message>!
///   MOAB ERROR: <function>() line <n> in <file>
/// ```
///
/// and each SOMA_MB_CHK_ERR it passes through adds its own location line.
/// Unlike MOAB's macros, which evaluate a failing expression a second time,
/// these evaluate it once.

#include <sstream>
#include <string>

#include "soma/compat/types.hpp"

namespace soma::compat::detail {

/// new_error: print the banner and the message, then the location.
void report_error(bool new_error, const std::string& message, const char* function, int line, const char* file);

}  // namespace soma::compat::detail

#define SOMA_MB_ERROR_MESSAGE_(msg) \
  ([&]() {                          \
    std::ostringstream soma_mb_os_; \
    soma_mb_os_ << msg;             \
    return soma_mb_os_.str();       \
  }())

#define SOMA_MB_SET_ERR(code, msg)                                                                     \
  do {                                                                                                 \
    ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
    return code;                                                                                       \
  } while (false)

#define SOMA_MB_SET_ERR_RET(msg)                                                                       \
  do {                                                                                                 \
    ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
    return;                                                                                            \
  } while (false)

#define SOMA_MB_SET_ERR_RET_VAL(msg, val)                                                              \
  do {                                                                                                 \
    ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
    return val;                                                                                        \
  } while (false)

#define SOMA_MB_SET_ERR_CONT(msg) \
  ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__)

#define SOMA_MB_CHK_ERR(err)                                                              \
  do {                                                                                    \
    const ::soma::compat::ErrorCode soma_mb_err_ = (err);                                 \
    if (soma_mb_err_ != ::soma::compat::MB_SUCCESS) {                                     \
      ::soma::compat::detail::report_error(false, std::string(), __func__, __LINE__, __FILE__); \
      return soma_mb_err_;                                                                \
    }                                                                                     \
  } while (false)

#define SOMA_MB_CHK_ERR_RET(err)                                                          \
  do {                                                                                    \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                            \
      ::soma::compat::detail::report_error(false, std::string(), __func__, __LINE__, __FILE__); \
      return;                                                                             \
    }                                                                                     \
  } while (false)

#define SOMA_MB_CHK_ERR_RET_VAL(err, val)                                                 \
  do {                                                                                    \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                            \
      ::soma::compat::detail::report_error(false, std::string(), __func__, __LINE__, __FILE__); \
      return val;                                                                         \
    }                                                                                     \
  } while (false)

#define SOMA_MB_CHK_ERR_CONT(err)                                                         \
  do {                                                                                    \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                            \
      ::soma::compat::detail::report_error(false, std::string(), __func__, __LINE__, __FILE__); \
    }                                                                                     \
  } while (false)

#define SOMA_MB_CHK_SET_ERR(err, msg)                                                                    \
  do {                                                                                                   \
    const ::soma::compat::ErrorCode soma_mb_err_ = (err);                                                \
    if (soma_mb_err_ != ::soma::compat::MB_SUCCESS) {                                                    \
      ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
      return soma_mb_err_;                                                                               \
    }                                                                                                    \
  } while (false)

#define SOMA_MB_CHK_SET_ERR_RET(err, msg)                                                                \
  do {                                                                                                   \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                                           \
      ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
      return;                                                                                            \
    }                                                                                                    \
  } while (false)

#define SOMA_MB_CHK_SET_ERR_RET_VAL(err, msg, val)                                                       \
  do {                                                                                                   \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                                           \
      ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
      return val;                                                                                        \
    }                                                                                                    \
  } while (false)

#define SOMA_MB_CHK_SET_ERR_CONT(err, msg)                                                               \
  do {                                                                                                   \
    if ((err) != ::soma::compat::MB_SUCCESS) {                                                           \
      ::soma::compat::detail::report_error(true, SOMA_MB_ERROR_MESSAGE_(msg), __func__, __LINE__, __FILE__); \
    }                                                                                                    \
  } while (false)

#endif  // SOMA_COMPAT_ERROR_HPP_
