# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/synchronization/include
set(synchronization_headers
    pika/synchronization/async_rw_mutex.hpp
    pika/synchronization/barrier.hpp
    pika/synchronization/channel_mpmc.hpp
    pika/synchronization/channel_mpsc.hpp
    pika/synchronization/channel_spsc.hpp
    pika/synchronization/condition_variable.hpp
    pika/synchronization/counting_semaphore.hpp
    pika/synchronization/detail/condition_variable.hpp
    pika/synchronization/detail/counting_semaphore.hpp
    pika/synchronization/detail/sliding_semaphore.hpp
    pika/synchronization/event.hpp
    pika/synchronization/latch.hpp
    pika/synchronization/lock_types.hpp
    pika/synchronization/mutex.hpp
    pika/synchronization/no_mutex.hpp
    pika/synchronization/once.hpp
    pika/synchronization/recursive_mutex.hpp
    pika/synchronization/shared_mutex.hpp
    pika/synchronization/sliding_semaphore.hpp
    pika/synchronization/spinlock.hpp
    pika/synchronization/spinlock_no_backoff.hpp
    pika/synchronization/spinlock_pool.hpp
    pika/synchronization/stop_token.hpp
)

set(synchronization_sources
    detail/condition_variable.cpp detail/counting_semaphore.cpp
    detail/sliding_semaphore.cpp local_barrier.cpp mutex.cpp stop_token.cpp
)

include(pika_add_module)
pika_add_module(
  pika synchronization
  GLOBAL_HEADER_GEN ON
  SOURCES ${synchronization_sources}
  HEADERS ${synchronization_headers}
  MODULE_DEPENDENCIES
    pika_config_local
    pika_assertion
    pika_execution_base
    pika_concurrency
    pika_coroutines
    pika_errors
    pika_functional
    pika_hashing
    pika_itt_notify
    pika_memory
    pika_lock_registration
    pika_logging
    pika_memory
    pika_threading_base
    pika_thread_support
    pika_topology
    pika_timing
  CMAKE_SUBDIRS examples tests
)
