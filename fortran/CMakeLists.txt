# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The Fortran 2008 module `soma` over the C API.

add_library(soma_fortran soma_kinds.F90 soma_c_api.F90 soma.F90)
add_library(soma::soma_fortran ALIAS soma_fortran)
set_target_properties(soma_fortran PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod")
target_include_directories(soma_fortran PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/soma/fortran>)
# The length of the names list calls return (soma_name_len); longer names are an error, never cut off.
set(SOMA_FORTRAN_NAME_LENGTH 1024 CACHE STRING "Length of the names the Fortran module returns (soma_name_len)")
if(NOT SOMA_FORTRAN_NAME_LENGTH MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "SOMA_FORTRAN_NAME_LENGTH must be a positive integer, not '${SOMA_FORTRAN_NAME_LENGTH}'")
endif()
target_compile_definitions(soma_fortran PRIVATE
  SOMA_LO_64=$<BOOL:${SOMA_LO_64}> SOMA_OFFSET_64=$<BOOL:${SOMA_OFFSET_64}>
  SOMA_FORTRAN_NAME_LENGTH=${SOMA_FORTRAN_NAME_LENGTH})
target_link_libraries(soma_fortran PUBLIC soma_c)
# Standard Fortran only: gfortran rejects its own extensions here (the module is Fortran 2008,
# which gfortran checks under its 2018 mode; ifort -stand f08 accepts it without a warning).
target_compile_options(soma_fortran PRIVATE "$<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=f2018>")

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
  # With OUTPUT, the test gets an empty directory of its own in SOMA_TEST_OUT.
  function(soma_add_fortran_test name)
    cmake_parse_arguments(T "OUTPUT" "" "RANKS" ${ARGN})
    add_executable(fortran_${name} tests/${name}.F90)
    target_link_libraries(fortran_${name} PRIVATE soma_fortran soma_fortran_check)
    if(SOMA_ENABLE_MPI AND T_RANKS)
      foreach(np IN LISTS T_RANKS)
        add_test(NAME fortran.${name}.np${np}
          COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                  $<TARGET_FILE:fortran_${name}> ${MPIEXEC_POSTFLAGS})
        set(_env OMP_NUM_THREADS=1)
        if(T_OUTPUT)
          set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/${name}.np${np}")
          file(MAKE_DIRECTORY "${_out}")
          list(APPEND _env "SOMA_TEST_OUT=${_out}")
        endif()
        set_tests_properties(fortran.${name}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np}
          ENVIRONMENT "${_env}")
      endforeach()
    else()
      set(_env OMP_NUM_THREADS=1)
      if(T_OUTPUT)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/${name}")
        file(MAKE_DIRECTORY "${_out}")
        list(APPEND _env "SOMA_TEST_OUT=${_out}")
      endif()
      add_test(NAME fortran.${name} COMMAND fortran_${name})
      set_tests_properties(fortran.${name} PROPERTIES TIMEOUT 60 ENVIRONMENT "${_env}")
    endif()
  endfunction()

  soma_add_fortran_test(test_runtime RANKS 1 2 3 4)
  soma_add_fortran_test(test_mesh RANKS 1 2 3 4)
  soma_add_fortran_test(test_edit RANKS 1)
  soma_add_fortran_test(test_fields RANKS 1 2 3 4)
  soma_add_fortran_test(test_parallel RANKS 1 2 3 4)
  soma_add_fortran_test(test_algorithms RANKS 1 2 3 4)
  soma_add_fortran_test(test_handles RANKS 1)
  soma_add_fortran_test(test_uninit RANKS 1)
  if(SOMA_ENABLE_HDF5)
    soma_add_fortran_test(test_io RANKS 1 2 3 4 OUTPUT)
  endif()

  # Without ierr a failure stops every rank with SOMA's message, whoever initialized Kokkos.
  add_executable(fortran_test_stop tests/test_stop.F90)
  target_link_libraries(fortran_test_stop PRIVATE soma_fortran soma_fortran_check)
  add_executable(fortran_test_stop_host tests/test_stop.F90 tests/host_kokkos.cpp)
  target_link_libraries(fortran_test_stop_host PRIVATE soma_fortran soma_fortran_check)
  target_compile_definitions(fortran_test_stop_host PRIVATE SOMA_TEST_HOST_KOKKOS=1)
  set(_stop_ranks 1)
  if(SOMA_ENABLE_MPI)
    set(_stop_ranks 1 2)
  endif()
  foreach(variant IN ITEMS stop stop_host)
    foreach(np IN LISTS _stop_ranks)
      if(SOMA_ENABLE_MPI)
        set(_cmd "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${np};$<TARGET_FILE:fortran_test_${variant}>")
      else()
        set(_cmd "$<TARGET_FILE:fortran_test_${variant}>")
      endif()
      add_test(NAME fortran.test_${variant}.np${np}
        COMMAND ${CMAKE_COMMAND} "-DCOMMAND=${_cmd}" "-DREGEX=soma: box_mesh: unsupported element topology"
                "-DOUT_REGEX=before the failure" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake")
      set_tests_properties(fortran.test_${variant}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np}
        ENVIRONMENT OMP_NUM_THREADS=2)
    endforeach()
  endforeach()
endif()

# The pipeline example in Fortran: the C++ example's steps and digests, with MPI or without.
if(SOMA_ENABLE_EXAMPLES)
  add_executable(pipeline_f examples/pipeline.F90)
  target_link_libraries(pipeline_f PRIVATE soma_fortran pipeline_arith)
  target_compile_definitions(pipeline_f PRIVATE SOMA_PIPELINE_MPI=$<BOOL:${SOMA_ENABLE_MPI}>)
  if(SOMA_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)
    target_link_libraries(pipeline_f PRIVATE MPI::MPI_Fortran)
  endif()
  # Its arithmetic rounds as the C++ example's: no fused multiply-adds, by each compiler's own flag.
  target_compile_options(pipeline_f PRIVATE
    "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang>:-ffp-contract=off>"
    "$<$<COMPILE_LANG_AND_ID:Fortran,Intel,IntelLLVM>:-fp-model=precise;-no-fma>"
    "$<$<COMPILE_LANG_AND_ID:Fortran,NVHPC,PGI>:-Mnofma>")
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|LLVMFlang|Flang|Intel|IntelLLVM|NVHPC|PGI)$")
    message(WARNING "pipeline_f: no flag known to turn off fused multiply-adds for the "
      "${CMAKE_Fortran_COMPILER_ID} Fortran compiler; fortran.pipeline may not match C++ to the last bit")
  endif()
  if(SOMA_ENABLE_TESTS)
    # C++ and Fortran at each rank and thread count must print the same digests, to the last bit.
    if(SOMA_ENABLE_MPI)
      set(_pipeline_runs "np1:1|4" "np3:1")
    else()
      set(_pipeline_runs "serial:1|4")
    endif()
    foreach(run IN LISTS _pipeline_runs)
      string(REPLACE ":" ";" run "${run}")
      list(GET run 0 name)
      list(GET run 1 threads)
      set(np 1)
      if(name MATCHES "^np([0-9]+)$")
        set(np "${CMAKE_MATCH_1}")
      endif()
      add_test(NAME fortran.pipeline.${name}
        COMMAND ${CMAKE_COMMAND} "-DMPIEXEC=$<$<BOOL:${SOMA_ENABLE_MPI}>:${MPIEXEC_EXECUTABLE}>" -DNP=${np}
                "-DTHREADS=${threads}" -DCXX=$<TARGET_FILE:pipeline> -DFORTRAN=$<TARGET_FILE:pipeline_f>
                -DWORK=${CMAKE_CURRENT_BINARY_DIR}/pipeline_${name}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_pipeline.cmake")
      set_tests_properties(fortran.pipeline.${name} PROPERTIES TIMEOUT 240 PROCESSORS ${np})
    endforeach()
  endif()
endif()
