# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_headers
    pika/async/apply.hpp pika/async/async.hpp
    pika/async/async_fwd.hpp pika/async/dataflow.hpp
    pika/async/sync.hpp pika/async/sync_fwd.hpp
)

include(pika_add_module)
pika_add_module(
  pika async
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_headers}
  MODULE_DEPENDENCIES pika_async_base pika_execution pika_executors
                      pika_config
  CMAKE_SUBDIRS examples tests
)
