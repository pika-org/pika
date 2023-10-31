# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_manager_headers pika/modules/thread_manager.hpp
                           pika/thread_manager/thread_manager_fwd.hpp
)

set(thread_manager_sources thread_manager.cpp)

include(pika_add_module)
pika_add_module(
  pika thread_manager
  GLOBAL_HEADER_GEN OFF
  SOURCES ${thread_manager_sources}
  HEADERS ${thread_manager_headers}
  MODULE_DEPENDENCIES
    pika_program_options
    pika_resource_partitioner
    pika_runtime_configuration
    pika_errors
    pika_logging
    pika_schedulers
    pika_thread_pools
  CMAKE_SUBDIRS examples tests
)
