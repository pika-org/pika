# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/concurrency/include
set(concurrency_headers
    pika/concurrency/barrier.hpp
    pika/concurrency/cache_line_data.hpp
    pika/concurrency/concurrentqueue.hpp
    pika/concurrency/deque.hpp
    pika/concurrency/detail/contiguous_index_queue.hpp
    pika/concurrency/detail/freelist.hpp
    pika/concurrency/detail/tagged_ptr_pair.hpp
    pika/concurrency/spinlock.hpp
    pika/concurrency/spinlock_pool.hpp
)

# Default location is $PIKA_ROOT/libs/concurrency/src
set(concurrency_sources barrier.cpp)

include(pika_add_module)
pika_add_module(
  pika concurrency
  GLOBAL_HEADER_GEN ON
  SOURCES ${concurrency_sources}
  HEADERS ${concurrency_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_concepts
    pika_config_local
    pika_datastructures
    pika_execution_base
    pika_errors
    pika_hashing
    pika_itt_notify
    pika_lock_registration
    pika_thread_support
  CMAKE_SUBDIRS examples tests
)
