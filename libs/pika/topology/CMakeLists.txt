# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/topology/include
set(topology_headers pika/topology/cpu_mask.hpp pika/topology/topology.hpp)

# Default location is $PIKA_ROOT/libs/topology/src
set(topology_sources cpu_mask.cpp topology.cpp)

include(pika_add_module)
pika_add_module(
  pika topology
  GLOBAL_HEADER_GEN ON
  SOURCES ${topology_sources}
  HEADERS ${topology_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config
    pika_concurrency
    pika_errors
    pika_format
    pika_logging
    pika_type_support
    pika_util
  DEPENDENCIES Hwloc::hwloc
  CMAKE_SUBDIRS examples tests
)
