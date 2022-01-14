# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_base_headers
    pika/async_base/apply.hpp
    pika/async_base/async.hpp
    pika/async_base/dataflow.hpp
    pika/async_base/launch_policy.hpp
    pika/async_base/sync.hpp
    pika/async_base/traits/is_launch_policy.hpp
    pika/async_base/scheduling_properties.hpp
)

set(async_base_sources launch_policy.cpp)

include(pika_add_module)
pika_add_module(
  pika async_base
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_base_headers}
  SOURCES ${async_base_sources}
  MODULE_DEPENDENCIES pika_allocator_support pika_config_local pika_coroutines
                      pika_tag_invoke
  CMAKE_SUBDIRS examples tests
)
