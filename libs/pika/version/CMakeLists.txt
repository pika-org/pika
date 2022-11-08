# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_headers pika/version.hpp)

set(version_sources version.cpp)

include(pika_add_module)
pika_add_module(
  pika version
  GLOBAL_HEADER_GEN OFF
  SOURCES ${version_sources}
  HEADERS ${version_headers}
  CMAKE_SUBDIRS tests
  MODULE_DEPENDENCIES pika_config pika_prefix
)

target_include_directories(
  pika_version PRIVATE $<BUILD_INTERFACE:${PIKA_BINARY_DIR}>
)
