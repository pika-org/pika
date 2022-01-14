# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(affinity_headers
    pika/affinity/affinity_data.hpp
    pika/affinity/parse_affinity_options.hpp
)

set(affinity_sources affinity_data.cpp parse_affinity_options.cpp)

include(pika_add_module)
pika_add_module(
  pika affinity
  GLOBAL_HEADER_GEN ON
  SOURCES ${affinity_sources}
  HEADERS ${affinity_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_datastructures
    pika_errors
    pika_format
    pika_topology
    pika_util
  DEPENDENCIES Hwloc::hwloc
  CMAKE_SUBDIRS tests
)
