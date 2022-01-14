# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(futures_headers
    hpx/futures/future.hpp
    hpx/futures/future_fwd.hpp
    hpx/futures/futures_factory.hpp
    hpx/futures/detail/future_data.hpp
    hpx/futures/detail/future_transforms.hpp
    hpx/futures/packaged_continuation.hpp
    hpx/futures/packaged_task.hpp
    hpx/futures/promise.hpp
    hpx/futures/traits/acquire_future.hpp
    hpx/futures/traits/acquire_shared_state.hpp
    hpx/futures/traits/detail/future_await_traits.hpp
    hpx/futures/traits/detail/future_traits.hpp
    hpx/futures/traits/future_access.hpp
    hpx/futures/traits/future_then_result.hpp
    hpx/futures/traits/future_traits.hpp
    hpx/futures/traits/get_remote_result.hpp
    hpx/futures/traits/is_future.hpp
    hpx/futures/traits/is_future_range.hpp
    hpx/futures/traits/is_future_tuple.hpp
    hpx/futures/traits/promise_local_result.hpp
    hpx/futures/traits/promise_remote_result.hpp
)

set(futures_sources future_data.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local futures
  GLOBAL_HEADER_GEN ON
  SOURCES ${futures_sources}
  HEADERS ${futures_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "hpx/futures/detail/future_data.hpp"
                             "hpx/futures/detail/future_transforms.hpp"
  MODULE_DEPENDENCIES hpx_async_base hpx_config_local hpx_allocator_support
                      hpx_errors hpx_memory hpx_synchronization
  CMAKE_SUBDIRS examples tests
)
