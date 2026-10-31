# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

cmake_minimum_required(VERSION 3.24)
project(soma_packaging_test LANGUAGES CXX)

find_package(soma REQUIRED)
message(STATUS "consumer: soma_DIR=${soma_DIR} Kokkos_DIR=${Kokkos_DIR}")

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE soma::soma)
