# Standalone project: configured by tests/packaging/run_packaging_test.cmake
# against an install prefix, never added via add_subdirectory.
cmake_minimum_required(VERSION 3.20)
project(moabx_packaging_test LANGUAGES CXX)

# The driver passes CMAKE_PREFIX_PATH with '|' separators so the list survives
# being a single -D argument.
if(DEFINED MOABX_PACKAGING_LIST_SEPARATOR)
  string(REPLACE "${MOABX_PACKAGING_LIST_SEPARATOR}" ";" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endif()

find_package(moabx REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE moabx::moabx_runtime)
