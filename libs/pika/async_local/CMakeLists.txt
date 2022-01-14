# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_local_headers
    pika/async_local/apply.hpp pika/async_local/async.hpp
    pika/async_local/async_fwd.hpp pika/async_local/dataflow.hpp
    pika/async_local/sync.hpp pika/async_local/sync_fwd.hpp
)

include(pika_add_module)
pika_add_module(
  pika async_local
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_local_headers}
  MODULE_DEPENDENCIES pika_async_base pika_execution pika_executors
                      pika_config_local
  CMAKE_SUBDIRS examples tests
)
