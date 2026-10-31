# Verifies that MOABX is consumable as an installed package.
#
# Installs the build tree to a scratch prefix, then configures, builds, and
# runs a standalone project that locates MOABX solely through
# find_package(moabx). This catches missing install rules, headers omitted
# from the install set, and absolute build-tree paths leaking into the
# exported targets -- none of which the in-tree tests can detect, because
# they see the source directory on the include path.
#
# Required: -DMOABX_BINARY_DIR -DMOABX_SOURCE_DIR -DSCRATCH_DIR
# Optional: -DCMAKE_PREFIX_PATH (to locate MOAB), -DBUILD_TYPE, -DGENERATOR

foreach(required MOABX_BINARY_DIR MOABX_SOURCE_DIR SCRATCH_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "run_packaging_test.cmake requires -D${required}")
  endif()
endforeach()

set(install_prefix "${SCRATCH_DIR}/install")
set(consumer_build "${SCRATCH_DIR}/consumer_build")

# Start clean so a stale prefix cannot mask a removed install rule.
file(REMOVE_RECURSE "${install_prefix}" "${consumer_build}")

function(run_step description)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${description} failed (exit ${rc})\n--- stdout ---\n${out}\n--- stderr ---\n${err}")
  endif()
endfunction()

run_step("install to scratch prefix"
  ${CMAKE_COMMAND} --install "${MOABX_BINARY_DIR}" --prefix "${install_prefix}"
)

# CMAKE_PREFIX_PATH is a list; the scratch prefix goes first so the freshly
# installed MOABX wins over any system copy.
set(prefix_path "${install_prefix}")
if(DEFINED CMAKE_PREFIX_PATH AND NOT CMAKE_PREFIX_PATH STREQUAL "")
  list(APPEND prefix_path ${CMAKE_PREFIX_PATH})
endif()
string(REPLACE ";" "|" prefix_path_arg "${prefix_path}")

set(configure_args
  -S "${MOABX_SOURCE_DIR}/tests/packaging"
  -B "${consumer_build}"
  -DCMAKE_PREFIX_PATH=${prefix_path_arg}
  -DMOABX_PACKAGING_LIST_SEPARATOR=|
)
if(DEFINED BUILD_TYPE AND NOT BUILD_TYPE STREQUAL "")
  list(APPEND configure_args -DCMAKE_BUILD_TYPE=${BUILD_TYPE})
endif()
if(DEFINED GENERATOR AND NOT GENERATOR STREQUAL "")
  list(APPEND configure_args -G "${GENERATOR}")
endif()

run_step("configure out-of-tree consumer" ${CMAKE_COMMAND} ${configure_args})
run_step("build out-of-tree consumer" ${CMAKE_COMMAND} --build "${consumer_build}")

# The consumer asserts its own results and returns non-zero on mismatch.
find_program(consumer_exe NAMES consumer
  PATHS "${consumer_build}" "${consumer_build}/${BUILD_TYPE}" NO_DEFAULT_PATH)
if(NOT consumer_exe)
  message(FATAL_ERROR "consumer executable not found under ${consumer_build}")
endif()

run_step("run out-of-tree consumer" "${consumer_exe}")

message(STATUS "packaging test passed: installed, configured, built, and ran via find_package(moabx)")
