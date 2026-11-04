// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_DROPIN_MOAB_TYPES_HPP_
#define SOMA_COMPAT_DROPIN_MOAB_TYPES_HPP_

// Drop-in mode: MOAB's include paths name soma::compat as namespace moab,
// and MOAB's error macros are SOMA's.

#include "moab/MOABConfig.h"
#include "soma/compat/error.hpp"
#include "soma/compat/types.hpp"

namespace moab = soma::compat;

#define MB_SET_ERR(code, msg) SOMA_MB_SET_ERR(code, msg)
#define MB_SET_ERR_RET(msg) SOMA_MB_SET_ERR_RET(msg)
#define MB_SET_ERR_RET_VAL(msg, val) SOMA_MB_SET_ERR_RET_VAL(msg, val)
#define MB_SET_ERR_CONT(msg) SOMA_MB_SET_ERR_CONT(msg)
#define MB_CHK_ERR(err) SOMA_MB_CHK_ERR(err)
#define MB_CHK_ERR_RET(err) SOMA_MB_CHK_ERR_RET(err)
#define MB_CHK_ERR_RET_VAL(err, val) SOMA_MB_CHK_ERR_RET_VAL(err, val)
#define MB_CHK_ERR_CONT(err) SOMA_MB_CHK_ERR_CONT(err)
#define MB_CHK_SET_ERR(err, msg) SOMA_MB_CHK_SET_ERR(err, msg)
#define MB_CHK_SET_ERR_RET(err, msg) SOMA_MB_CHK_SET_ERR_RET(err, msg)
#define MB_CHK_SET_ERR_RET_VAL(err, msg, val) SOMA_MB_CHK_SET_ERR_RET_VAL(err, msg, val)
#define MB_CHK_SET_ERR_CONT(err, msg) SOMA_MB_CHK_SET_ERR_CONT(err, msg)

#endif  // SOMA_COMPAT_DROPIN_MOAB_TYPES_HPP_
