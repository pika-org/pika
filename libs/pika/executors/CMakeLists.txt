# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/executors/include
set(executors_headers
    pika/executors/std_thread_scheduler.hpp pika/executors/thread_pool_scheduler.hpp
    pika/executors/thread_pool_scheduler_bulk.hpp
)

include(pika_add_module)
pika_add_module(
  pika executors
  GLOBAL_HEADER_GEN ON
  HEADERS ${executors_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_async_base
    pika_concurrency
    pika_config
    pika_coroutines
    pika_datastructures
    pika_errors
    pika_execution
    pika_execution_base
    pika_functional
    pika_itt_notify
    pika_memory
    pika_properties
    pika_threading
    pika_threading_base
    pika_timing
    pika_type_support
  CMAKE_SUBDIRS examples tests
)
