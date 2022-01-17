# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(command_line_handling_headers
    pika/command_line_handling/command_line_handling.hpp
    pika/command_line_handling/late_command_line_handling.hpp
    pika/command_line_handling/parse_command_line.hpp
)

set(command_line_handling_sources
    command_line_handling.cpp late_command_line_handling.cpp
    parse_command_line.cpp
)

include(pika_add_module)
pika_add_module(
  pika command_line_handling
  GLOBAL_HEADER_GEN ON
  SOURCES ${command_line_handling_sources}
  HEADERS ${command_line_handling_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    pika_config
    pika_program_options
    pika_runtime_configuration
    pika_debugging
    pika_errors
    pika_filesystem
    pika_format
    pika_topology
    pika_util
    pika_version
  CMAKE_SUBDIRS examples tests
)
