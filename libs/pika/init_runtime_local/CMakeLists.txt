# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_local_headers
    pika/init_runtime_local/detail/init_logging.hpp
    pika/init_runtime_local/init_runtime_local.hpp pika/local/init.hpp
)

set(init_runtime_local_sources init_logging.cpp init_runtime_local.cpp)

include(pika_add_module)
pika_add_module(
  pika init_runtime_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${init_runtime_local_sources}
  HEADERS ${init_runtime_local_headers}
  MODULE_DEPENDENCIES
    pika_algorithms
    pika_command_line_handling_local
    pika_program_options
    pika_runtime_local
    pika_errors
    pika_filesystem
    pika_format
    pika_lock_registration
    pika_logging
    pika_schedulers
    pika_testing
    pika_threading_base
    pika_timing
  CMAKE_SUBDIRS examples tests
)
