# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# cmake -DCOMMAND="<launcher;args;program>" -DREGEX=<stderr pattern> [-DOUT_REGEX=<stdout pattern>]
#       -P expect_failure.cmake
# Passes when the command exits non-zero, within the timeout, with REGEX in its stderr (and
# OUT_REGEX in its stdout: what the program wrote before failing is not lost), without a crash.

execute_process(COMMAND ${COMMAND} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 50)
if(rc STREQUAL "0")
  message(FATAL_ERROR "expected a failure, the command succeeded:\n${out}\n${err}")
endif()
if(rc MATCHES "timeout|Timeout")
  message(FATAL_ERROR "the command hung: ${rc}\n${out}\n${err}")
endif()
if(NOT err MATCHES "${REGEX}")
  message(FATAL_ERROR "stderr does not match '${REGEX}' (exit ${rc}):\n${out}\n${err}")
endif()
if(OUT_REGEX AND NOT out MATCHES "${OUT_REGEX}")
  message(FATAL_ERROR "stdout does not match '${OUT_REGEX}' (exit ${rc}):\n${out}\n${err}")
endif()
if(err MATCHES "mutex lock failed|terminating|Kokkos ERROR|Segmentation|signal")
  message(FATAL_ERROR "the failure ended in a crash, not a stop (exit ${rc}):\n${err}")
endif()
if(out MATCHES "not reached")
  message(FATAL_ERROR "the program went on after the failure:\n${out}")
endif()
message(STATUS "failed as expected (exit ${rc})")
