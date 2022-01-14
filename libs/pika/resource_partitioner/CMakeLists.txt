# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(resource_partitioner_headers
    pika/resource_partitioner/detail/create_partitioner.hpp
    pika/resource_partitioner/detail/partitioner.hpp
    pika/resource_partitioner/partitioner.hpp
    pika/resource_partitioner/partitioner_fwd.hpp
)

set(resource_partitioner_sources detail_partitioner.cpp partitioner.cpp)

include(pika_add_module)
pika_add_module(
  pika resource_partitioner
  GLOBAL_HEADER_GEN ON
  SOURCES ${resource_partitioner_sources}
  HEADERS ${resource_partitioner_headers}
  MODULE_DEPENDENCIES
    pika_affinity
    pika_config_local
    pika_runtime_configuration
    pika_errors
    pika_format
    pika_ini
    pika_prefix
    pika_threading_base
    pika_synchronization
    pika_thread_pools
  CMAKE_SUBDIRS examples tests
)
