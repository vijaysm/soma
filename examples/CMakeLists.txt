# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Example programs, one per algorithm; they read and write meshes, so they
# need HDF5, and compute in Kokkos's default execution space: on the GPU of a
# GPU build. With tests on, each runs on a bundled mesh, and with MPI at
# several ranks too: two on the host, one for each GPU otherwise.
add_executable(laplacian_smooth laplacian_smooth.cpp)
target_link_libraries(laplacian_smooth PRIVATE soma soma_warnings)
soma_target_kokkos_sources(laplacian_smooth)
add_executable(poisson poisson.cpp)
target_link_libraries(poisson PRIVATE soma soma_warnings)
soma_target_kokkos_sources(poisson)
add_executable(advect advect.cpp)
target_link_libraries(advect PRIVATE soma soma_warnings)
soma_target_kokkos_sources(advect)

# The GPUs the tests may use, SOMA_TEST_GPUS: a count, or AUTO for those the
# vendor's tool lists on this node (one, said aloud, when it cannot tell).
function(soma_example_test_gpus out)
  set(_n "${SOMA_TEST_GPUS}")
  if(_n MATCHES "^[1-9][0-9]*$")
    set(${out} ${_n} PARENT_SCOPE)
    return()
  endif()
  string(TOUPPER "${_n}" _mode)
  if(NOT _mode STREQUAL "AUTO")
    message(FATAL_ERROR "SOMA_TEST_GPUS must be AUTO or a count of GPUs, 1 or more (got '${SOMA_TEST_GPUS}')")
  endif()
  set(_n 0)
  if(SOMA_KOKKOS_GPU_BACKEND STREQUAL "CUDA")
    find_program(SOMA_NVIDIA_SMI nvidia-smi)
    mark_as_advanced(SOMA_NVIDIA_SMI)
    if(SOMA_NVIDIA_SMI)
      execute_process(COMMAND ${SOMA_NVIDIA_SMI} --query-gpu=index --format=csv,noheader
        OUTPUT_VARIABLE _listed RESULT_VARIABLE _status ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_status EQUAL 0 AND NOT _listed STREQUAL "")
        string(REGEX MATCHALL "[^\n]+" _lines "${_listed}")
        list(LENGTH _lines _n)
      endif()
    endif()
  endif()
  if(_n EQUAL 0)
    message(STATUS "SOMA: cannot count the ${SOMA_KOKKOS_GPU_BACKEND} GPUs of this node; the example tests "
      "use one (set SOMA_TEST_GPUS to the number there are)")
    set(_n 1)
  endif()
  set(${out} ${_n} PARENT_SCOPE)
endfunction()

if(SOMA_ENABLE_TESTS)
  set(in "${PROJECT_SOURCE_DIR}/data/64bricks_1khex.h5m")
  # Ranks of the MPI runs (0: none). On the host, two. On GPUs, one rank for
  # each: none on a single GPU, where the plain runs below are already one
  # rank on that one device, and N on N GPUs, each rank of a node on a device
  # of its own (Kokkos's default; MPIEXEC_PREFLAGS place the ranks on nodes).
  set(ranks 0)
  if(SOMA_ENABLE_MPI)
    if(SOMA_KOKKOS_HAS_GPU)
      soma_example_test_gpus(gpus)
      if(gpus GREATER 1)
        set(ranks ${gpus})
      endif()
      message(STATUS "SOMA: example tests on ${gpus} ${SOMA_KOKKOS_GPU_BACKEND} GPU(s), one MPI rank each")
    else()
      set(ranks 2)
    endif()
  endif()
  set(launch ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS})

  add_test(NAME example_laplacian_smooth
    COMMAND laplacian_smooth --iterations 5 ${in} ${CMAKE_CURRENT_BINARY_DIR}/smoothed.vtk)
  if(ranks)
    add_test(NAME example_laplacian_smooth.np${ranks}
      COMMAND ${launch} $<TARGET_FILE:laplacian_smooth> ${MPIEXEC_POSTFLAGS} --iterations 5 --tolerance 1e-9 ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/smoothed_np${ranks}.soma)
    set_tests_properties(example_laplacian_smooth.np${ranks} PROPERTIES TIMEOUT 60 PROCESSORS ${ranks})
  endif()
  # Both runs give the solution's largest value, whatever the ranks and assembly.
  add_test(NAME example_poisson COMMAND poisson ${in} ${CMAKE_CURRENT_BINARY_DIR}/poisson.vtk)
  set_tests_properties(example_poisson PROPERTIES PASS_REGULAR_EXPRESSION "max u 9[.]090723e-01")
  if(ranks)
    add_test(NAME example_poisson.np${ranks}
      COMMAND ${launch} $<TARGET_FILE:poisson> ${MPIEXEC_POSTFLAGS} --assembly colored ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/poisson_np${ranks}.soma)
    set_tests_properties(example_poisson.np${ranks} PROPERTIES TIMEOUT 60 PROCESSORS ${ranks}
      PASS_REGULAR_EXPRESSION "max u 9[.]090723e-01")
  endif()
  # Reproducible, both runs give the same totals to the last digit: those of
  # a host build, which a GPU matches for certain only in a build that keeps
  # arithmetic as written (SOMA_REPRODUCIBLE=ON); the last three digits are
  # left open otherwise.
  if(SOMA_KOKKOS_HAS_GPU AND NOT SOMA_REPRODUCIBLE_BUILD)
    set(advected "total 6[.]948772654115[0-9]+e-01 before, 4[.]446464981696[0-9]+e-01 after")
  else()
    set(advected "total 6[.]948772654115442e-01 before, 4[.]446464981696669e-01 after")
  endif()
  add_test(NAME example_advect COMMAND advect --soma-reproducible ${in} ${CMAKE_CURRENT_BINARY_DIR}/advect.vtk)
  set_tests_properties(example_advect PROPERTIES PASS_REGULAR_EXPRESSION "${advected}")
  if(ranks)
    add_test(NAME example_advect.np${ranks}
      COMMAND ${launch} $<TARGET_FILE:advect> ${MPIEXEC_POSTFLAGS} --soma-reproducible ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/advect_np${ranks}.soma)
    set_tests_properties(example_advect.np${ranks} PROPERTIES TIMEOUT 60 PROCESSORS ${ranks}
      PASS_REGULAR_EXPRESSION "${advected}")
  endif()
endif()
