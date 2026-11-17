# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The Fortran 2008 module `soma` over the C API.

add_library(soma_fortran soma_kinds.F90 soma_c_api.F90 soma.F90)
add_library(soma::soma_fortran ALIAS soma_fortran)
set_target_properties(soma_fortran PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod")
target_include_directories(soma_fortran PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/soma/fortran>)
target_compile_definitions(soma_fortran PRIVATE
  SOMA_LO_64=$<BOOL:${SOMA_LO_64}> SOMA_OFFSET_64=$<BOOL:${SOMA_OFFSET_64}>)
target_link_libraries(soma_fortran PUBLIC soma_c)

if(SOMA_ENABLE_TESTS)
  if(SOMA_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)
  endif()
  add_library(soma_fortran_check OBJECT tests/check.F90)
  set_target_properties(soma_fortran_check PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/testmod")
  target_include_directories(soma_fortran_check PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/testmod")
  target_compile_definitions(soma_fortran_check PUBLIC SOMA_TEST_MPI=$<BOOL:${SOMA_ENABLE_MPI}>)
  if(SOMA_ENABLE_MPI)
    target_link_libraries(soma_fortran_check PUBLIC MPI::MPI_Fortran)
  endif()

  # soma_add_fortran_test(<name> [RANKS 1 2 ...]): a test program under mpiexec at each rank count.
  function(soma_add_fortran_test name)
    cmake_parse_arguments(T "" "" "RANKS" ${ARGN})
    add_executable(fortran_${name} tests/${name}.F90)
    target_link_libraries(fortran_${name} PRIVATE soma_fortran soma_fortran_check)
    if(SOMA_ENABLE_MPI AND T_RANKS)
      foreach(np IN LISTS T_RANKS)
        add_test(NAME fortran.${name}.np${np}
          COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:fortran_${name}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(fortran.${name}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np}
          ENVIRONMENT OMP_NUM_THREADS=1)
      endforeach()
    else()
      add_test(NAME fortran.${name} COMMAND fortran_${name})
      set_tests_properties(fortran.${name} PROPERTIES TIMEOUT 60 ENVIRONMENT OMP_NUM_THREADS=1)
    endif()
  endfunction()

  soma_add_fortran_test(test_runtime RANKS 1 2 3 4)
  soma_add_fortran_test(test_mesh RANKS 1 2 3 4)

  # Without ierr a failure stops every rank with SOMA's message.
  add_executable(fortran_test_stop tests/test_stop.F90)
  target_link_libraries(fortran_test_stop PRIVATE soma_fortran soma_fortran_check)
  set(_stop_ranks 1)
  if(SOMA_ENABLE_MPI)
    set(_stop_ranks 1 2)
  endif()
  foreach(np IN LISTS _stop_ranks)
    if(SOMA_ENABLE_MPI)
      set(_cmd "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${np};$<TARGET_FILE:fortran_test_stop>")
    else()
      set(_cmd "$<TARGET_FILE:fortran_test_stop>")
    endif()
    add_test(NAME fortran.test_stop.np${np}
      COMMAND ${CMAKE_COMMAND} "-DCOMMAND=${_cmd}" "-DREGEX=soma: box_mesh: unsupported element topology"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake")
    set_tests_properties(fortran.test_stop.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np}
      ENVIRONMENT OMP_NUM_THREADS=1)
  endforeach()
endif()
