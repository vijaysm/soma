# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The MOAB example gate: each example of MOAB's share/examples is built against
# MOAB and against soma::compat (drop-in mode), both are run on the same
# input, and their output and written files must agree (compare_example.cmake).
# The example sources are read from the MOAB installation, not copied here.

set(SOMA_MOAB_EXAMPLES_DIR "" CACHE PATH "MOAB's share/examples directory, for the compat example gate")
if(NOT SOMA_MOAB_EXAMPLES_DIR)
  get_filename_component(_soma_moab_prefix "${MOAB_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${_soma_moab_prefix}/share/examples/basic/HelloMOAB.cpp")
    set(SOMA_MOAB_EXAMPLES_DIR "${_soma_moab_prefix}/share/examples")
  endif()
endif()
if(NOT SOMA_MOAB_EXAMPLES_DIR)
  message(STATUS "compat example gate: off (set SOMA_MOAB_EXAMPLES_DIR to MOAB's share/examples)")
  return()
endif()

# MOAB's MPI headers need MPI's, which the MOAB target does not carry.
find_package(MPI QUIET COMPONENTS CXX)

add_executable(compat_moab_describe compat/moab_describe.cpp)
target_link_libraries(compat_moab_describe PRIVATE ${SOMA_MOAB_TARGET})

set(_soma_gate_examples basic/HelloMOAB basic/GetEntities basic/SetsNTags basic/DeleteEdges
  basic/ReadPartFile basic/StructuredMeshSimple beginner/mbex1 beginner/mbex2 beginner/mbex3 beginner/mbex4)
foreach(ex IN LISTS _soma_gate_examples)
  get_filename_component(name ${ex} NAME)
  foreach(side ref soma)
    set(target compat_${side}_${name})
    add_executable(${target} "${SOMA_MOAB_EXAMPLES_DIR}/${ex}.cpp")
    target_compile_definitions(${target} PRIVATE MESH_DIR="${PROJECT_SOURCE_DIR}/data")
    # MOAB's code, compiled as it is: no warnings, and mbex4 initializes
    # doubles from unsigned loop counters.
    target_compile_options(${target} PRIVATE -w
      $<$<CXX_COMPILER_ID:Clang,AppleClang,IntelLLVM>:-Wno-c++11-narrowing>
      $<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>)
    if(side STREQUAL "ref")
      target_link_libraries(${target} PRIVATE ${SOMA_MOAB_TARGET})
      if(MPI_CXX_FOUND)
        target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
      endif()
    else()
      target_link_libraries(${target} PRIVATE soma_compat_dropin)
    endif()
  endforeach()
endforeach()

# soma_compat_gate(<test> <example> [ARGS a...] [INPUTS f...] [OUTPUTS f...]
#                  [PART_FILE n] [FIXTURE_SETUP f] [FIXTURE_REQUIRED f])
function(soma_compat_gate test example)
  cmake_parse_arguments(G "" "PART_FILE;FIXTURE_SETUP;FIXTURE_REQUIRED" "ARGS;INPUTS;OUTPUTS" ${ARGN})
  string(REPLACE ";" "|" args "${G_ARGS}")
  string(REPLACE ";" "|" inputs "${G_INPUTS}")
  string(REPLACE ";" "|" outputs "${G_OUTPUTS}")
  add_test(NAME compat.gate.${test}
    COMMAND ${CMAKE_COMMAND} -DREF=$<TARGET_FILE:compat_ref_${example}> -DSOMA=$<TARGET_FILE:compat_soma_${example}>
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/compat_gate/${test} -DDESCRIBE=$<TARGET_FILE:compat_moab_describe>
            -DARGS=${args} -DINPUTS=${inputs} -DOUTPUTS=${outputs} -DPART_FILE=${G_PART_FILE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/compat/compare_example.cmake)
  set_tests_properties(compat.gate.${test} PROPERTIES TIMEOUT 300)
  if(G_FIXTURE_SETUP)
    set_tests_properties(compat.gate.${test} PROPERTIES FIXTURES_SETUP ${G_FIXTURE_SETUP})
  endif()
  if(G_FIXTURE_REQUIRED)
    set_tests_properties(compat.gate.${test} PROPERTIES FIXTURES_REQUIRED ${G_FIXTURE_REQUIRED})
  endif()
endfunction()

set(_d ${PROJECT_SOURCE_DIR}/data)
soma_compat_gate(HelloMOAB.vtk HelloMOAB ARGS ${_d}/sphere_quads_20.vtk)
soma_compat_gate(HelloMOAB.h5m HelloMOAB ARGS ${_d}/64bricks_1khex.h5m)
soma_compat_gate(GetEntities GetEntities ARGS ${_d}/hexes_mixed.vtk)
soma_compat_gate(SetsNTags SetsNTags ARGS ${_d}/64bricks_1khex.h5m)
soma_compat_gate(mbex1 mbex1 OUTPUTS mbex1.vtk FIXTURE_SETUP compat_mbex1)
# mbex2 reads the mbex1.vtk MOAB's mbex1 wrote.
soma_compat_gate(mbex2 mbex2 INPUTS ${CMAKE_CURRENT_BINARY_DIR}/compat_gate/mbex1/ref/mbex1.vtk
  OUTPUTS mbex2.vtk FIXTURE_REQUIRED compat_mbex1)
soma_compat_gate(DeleteEdges DeleteEdges ARGS ${_d}/64bricks_1khex.h5m out.h5m OUTPUTS out.h5m)
soma_compat_gate(ReadPartFile ReadPartFile ARGS ${_d}/recMeshOcn128.h5m part.txt 4 out.h5m
  OUTPUTS out.h5m PART_FILE 7153)
soma_compat_gate(mbex3 mbex3 OUTPUTS mbex3.vtk)
soma_compat_gate(mbex4 mbex4 OUTPUTS mbex4.vtk)
soma_compat_gate(StructuredMeshSimple StructuredMeshSimple OUTPUTS structured_mesh.h5m)
soma_compat_gate(StructuredMeshSimple.n4 StructuredMeshSimple ARGS -d 3 -n 4 OUTPUTS structured_mesh.h5m)
