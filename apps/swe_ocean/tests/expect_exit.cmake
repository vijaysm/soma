# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs EXE with ARGS and checks that it exits with CODE and prints (on either stream) text matching MATCH.
separate_arguments(args UNIX_COMMAND "${ARGS}")
execute_process(COMMAND ${EXE} ${args} OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc TIMEOUT 30)
if(NOT rc EQUAL CODE)
  message(FATAL_ERROR "${EXE} ${ARGS} exited with ${rc}, not ${CODE}:\n${out}\n${err}")
endif()
if(NOT "${out}${err}" MATCHES "${MATCH}")
  message(FATAL_ERROR "${EXE} ${ARGS} printed nothing matching '${MATCH}':\n${out}\n${err}")
endif()
