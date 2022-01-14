# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(include_local_headers
    pika/local/algorithm.hpp
    pika/local/any.hpp
    pika/local/barrier.hpp
    pika/local/channel.hpp
    pika/local/chrono.hpp
    pika/local/condition_variable.hpp
    pika/local/exception.hpp
    pika/local/execution.hpp
    pika/local/functional.hpp
    pika/local/future.hpp
    pika/local/latch.hpp
    pika/local/memory.hpp
    pika/local/mutex.hpp
    pika/local/numeric.hpp
    pika/local/optional.hpp
    pika/local/runtime.hpp
    pika/local/semaphore.hpp
    pika/local/shared_mutex.hpp
    pika/local/stop_token.hpp
    pika/local/system_error.hpp
    pika/local/task_block.hpp
    pika/local/thread.hpp
    pika/local/tuple.hpp
    pika/local/type_traits.hpp
    pika/local/unwrap.hpp
)

include(pika_add_module)
pika_add_module(
  pika include_local
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_local_headers}
  MODULE_DEPENDENCIES
    pika_algorithms
    pika_async_base
    pika_async_combinators
    pika_async_local
    pika_errors
    pika_execution
    pika_executors
    pika_futures
    pika_lcos_local
    pika_runtime_local
  CMAKE_SUBDIRS examples tests
)
