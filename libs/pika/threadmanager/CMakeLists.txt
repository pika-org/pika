# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threadmanager_headers pika/modules/threadmanager.hpp
                          pika/threadmanager/threadmanager_fwd.hpp
)

set(threadmanager_sources threadmanager.cpp)

include(pika_add_module)
pika_add_module(
  pika threadmanager
  GLOBAL_HEADER_GEN OFF
  SOURCES ${threadmanager_sources}
  HEADERS ${threadmanager_headers}
  MODULE_DEPENDENCIES
    pika_program_options
    pika_resource_partitioner
    pika_runtime_configuration
    pika_errors
    pika_logging
    pika_schedulers
    pika_async_combinators
    pika_thread_pools
    pika_thread_pool_util
  CMAKE_SUBDIRS examples tests
)
