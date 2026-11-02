# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# bench_backends compares SOMA's execution spaces with each other and needs
# nothing but Kokkos. The MOAB comparisons need SOMA_ENABLE_MOAB_REFERENCE.

find_package(Git QUIET)
set(SOMA_GIT_DESCRIBE "unknown")
if(Git_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE _soma_git OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _soma_git_rc ERROR_QUIET)
  if(_soma_git_rc EQUAL 0 AND _soma_git)
    set(SOMA_GIT_DESCRIBE "${_soma_git}")
  endif()
endif()

# soma_add_bench(<name> <source> [LIBS ...])
function(soma_add_bench name source)
  cmake_parse_arguments(B "" "" "LIBS" ${ARGN})
  add_executable(${name} ${source})
  soma_target_kokkos_sources(${name})
  target_link_libraries(${name} PRIVATE soma soma_warnings ${B_LIBS})
  target_compile_definitions(${name} PRIVATE
    SOMA_BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    SOMA_BENCH_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}"
    SOMA_BENCH_BUILD_TYPE="$<IF:$<CONFIG:>,unset,$<CONFIG>>"
    SOMA_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    SOMA_BENCH_GIT="${SOMA_GIT_DESCRIBE}")
endfunction()

soma_add_bench(bench_backends bench_backends.cpp)

if(SOMA_ENABLE_MOAB_REFERENCE)
  soma_add_bench(bench_adjacency bench_adjacency.cpp LIBS ${SOMA_MOAB_TARGET})
  target_include_directories(bench_adjacency PRIVATE ${PROJECT_SOURCE_DIR}/tests/reference)
  if(SOMA_ENABLE_HDF5)
    soma_add_bench(bench_h5m_read bench_h5m_read.cpp LIBS ${SOMA_MOAB_TARGET})
    soma_add_bench(bench_h5m_write bench_h5m_write.cpp LIBS ${SOMA_MOAB_TARGET})
    if(SOMA_ENABLE_MOAB_COMPAT)
      soma_add_bench(bench_edit bench_edit.cpp LIBS soma_compat ${SOMA_MOAB_TARGET})
    endif()
  endif()
  if(SOMA_ENABLE_MPI AND SOMA_ENABLE_HDF5)
    soma_add_bench(bench_parallel bench_parallel.cpp LIBS ${SOMA_MOAB_TARGET})
  endif()
  if(SOMA_ENABLE_MPI)
    soma_add_bench(bench_transport bench_transport.cpp LIBS ${SOMA_MOAB_TARGET})
  endif()
endif()

if(SOMA_ENABLE_TESTS)
  # Keeps the benchmark compiling and its spaces agreeing in every verified
  # configuration; the sizes are too small to measure anything.
  if(SOMA_ENABLE_MPI)
    add_test(NAME bench.backends.smoke
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:bench_backends> ${MPIEXEC_POSTFLAGS}
              --hex 4 --quad 8 --tri 8 --repeats 1)
  else()
    add_test(NAME bench.backends.smoke
      COMMAND bench_backends --hex 4 --quad 8 --tri 8 --repeats 1)
  endif()
  set_tests_properties(bench.backends.smoke PROPERTIES TIMEOUT 120)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME bench.compare.selftest
      COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${PROJECT_SOURCE_DIR}/tests/bench)
  endif()
endif()
