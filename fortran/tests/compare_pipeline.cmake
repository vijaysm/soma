# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# cmake -DMPIEXEC=... -DNP=... -DCXX=<pipeline> -DFORTRAN=<pipeline_f> -DWORK=<dir> -P compare_pipeline.cmake
# Runs the pipeline example in C++ and in Fortran at NP ranks and requires the same output, byte for byte:
# the digests of the coordinates and the remapped values, and the integral.

file(MAKE_DIRECTORY "${WORK}/cxx" "${WORK}/fortran")
foreach(lang IN ITEMS cxx fortran)
  if(lang STREQUAL "cxx")
    set(program "${CXX}")
  else()
    set(program "${FORTRAN}")
  endif()
  execute_process(COMMAND "${MPIEXEC}" -n ${NP} "${program}" "${WORK}/${lang}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out_${lang} ERROR_VARIABLE err TIMEOUT 100)
  if(NOT rc STREQUAL "0")
    message(FATAL_ERROR "${lang} pipeline failed (${rc}):\n${out_${lang}}\n${err}")
  endif()
endforeach()
message(STATUS "C++:\n${out_cxx}Fortran:\n${out_fortran}")
if(NOT out_cxx STREQUAL out_fortran)
  message(FATAL_ERROR "the C++ and Fortran pipelines printed different results")
endif()
if(NOT out_cxx MATCHES "coords digest: [0-9a-f]+\nremap digest: [0-9a-f]+\nintegral: ")
  message(FATAL_ERROR "unexpected output:\n${out_cxx}")
endif()
