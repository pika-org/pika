# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(prefix_headers pika/prefix/find_prefix.hpp)

set(prefix_sources find_prefix.cpp)

include(pika_add_module)
pika_add_module(
  pika prefix
  GLOBAL_HEADER_GEN ON
  SOURCES ${prefix_sources}
  HEADERS ${prefix_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_errors
    pika_filesystem
    pika_string_util
    pika_type_support
  CMAKE_SUBDIRS tests
)
