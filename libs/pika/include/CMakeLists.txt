# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(include_headers
    pika/algorithm.hpp
    pika/any.hpp
    pika/barrier.hpp
    pika/channel.hpp
    pika/chrono.hpp
    pika/condition_variable.hpp
    pika/exception.hpp
    pika/execution.hpp
    pika/functional.hpp
    pika/future.hpp
    pika/latch.hpp
    pika/memory.hpp
    pika/mutex.hpp
    pika/numeric.hpp
    pika/optional.hpp
    pika/runtime.hpp
    pika/semaphore.hpp
    pika/shared_mutex.hpp
    pika/stop_token.hpp
    pika/system_error.hpp
    pika/task_block.hpp
    pika/thread.hpp
    pika/tuple.hpp
    pika/type_traits.hpp
    pika/unwrap.hpp
)

include(pika_add_module)
pika_add_module(
  pika include
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_headers}
  MODULE_DEPENDENCIES
    pika_algorithms
    pika_async_base
    pika_async_combinators
    pika_async
    pika_errors
    pika_execution
    pika_executors
    pika_futures
    pika_lcos
    pika_runtime
  CMAKE_SUBDIRS examples tests
)
