# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(futures_headers
    pika/futures/future.hpp
    pika/futures/future_fwd.hpp
    pika/futures/futures_factory.hpp
    pika/futures/detail/future_data.hpp
    pika/futures/detail/future_transforms.hpp
    pika/futures/packaged_continuation.hpp
    pika/futures/packaged_task.hpp
    pika/futures/promise.hpp
    pika/futures/traits/acquire_future.hpp
    pika/futures/traits/acquire_shared_state.hpp
    pika/futures/traits/detail/future_await_traits.hpp
    pika/futures/traits/detail/future_traits.hpp
    pika/futures/traits/future_access.hpp
    pika/futures/traits/future_then_result.hpp
    pika/futures/traits/future_traits.hpp
    pika/futures/traits/get_remote_result.hpp
    pika/futures/traits/is_future.hpp
    pika/futures/traits/is_future_range.hpp
    pika/futures/traits/is_future_tuple.hpp
    pika/futures/traits/promise_local_result.hpp
    pika/futures/traits/promise_remote_result.hpp
)

set(futures_sources future_data.cpp)

include(pika_add_module)
pika_add_module(
  pika futures
  GLOBAL_HEADER_GEN ON
  SOURCES ${futures_sources}
  HEADERS ${futures_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "pika/futures/detail/future_data.hpp"
                             "pika/futures/detail/future_transforms.hpp"
  MODULE_DEPENDENCIES pika_async_base pika_config_local pika_allocator_support
                      pika_errors pika_memory pika_synchronization
  CMAKE_SUBDIRS examples tests
)
