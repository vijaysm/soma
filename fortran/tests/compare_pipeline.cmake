# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# cmake [-DMPIEXEC=...] -DNP=... -DTHREADS="1|4" -DCXX=<pipeline> -DFORTRAN=<pipeline_f> -DWORK=<dir>
#       -P compare_pipeline.cmake
# Runs the pipeline example in C++ and in Fortran at NP ranks (under MPIEXEC; without it, one process)
# with each OMP_NUM_THREADS in THREADS, and requires the same output from every run, byte for byte:
# the digests of the coordinates and the remapped values, and the integral.

string(REPLACE "|" ";" threads "${THREADS}")
if(NOT threads)
  set(threads 1)
endif()
unset(reference)
foreach(t IN LISTS threads)
  foreach(lang IN ITEMS cxx fortran)
    if(lang STREQUAL "cxx")
      set(program "${CXX}")
    else()
      set(program "${FORTRAN}")
    endif()
    set(dir "${WORK}/${lang}.t${t}")
    file(MAKE_DIRECTORY "${dir}")
    if(MPIEXEC)
      set(launch "${MPIEXEC}" -n ${NP})
    else()
      set(launch "")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=${t} ${launch} "${program}" "${dir}"
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 100)
    if(NOT rc STREQUAL "0")
      message(FATAL_ERROR "${lang} pipeline at ${t} threads failed (${rc}):\n${out}\n${err}")
    endif()
    message(STATUS "${lang}, ${t} threads:\n${out}")
    if(NOT DEFINED reference)
      set(reference "${out}")
      set(reference_run "${lang}, ${t} threads")
      if(NOT out MATCHES "coords digest: [0-9a-f]+\nremap digest: [0-9a-f]+\nintegral: ")
        message(FATAL_ERROR "unexpected output:\n${out}")
      endif()
    elseif(NOT out STREQUAL reference)
      message(FATAL_ERROR "${lang} at ${t} threads printed different results from ${reference_run}")
    endif()
  endforeach()
endforeach()
