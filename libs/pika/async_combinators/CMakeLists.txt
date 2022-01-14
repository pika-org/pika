# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_combinators_headers
    pika/async_combinators/detail/throw_if_exceptional.hpp
    pika/async_combinators/split_future.hpp
    pika/async_combinators/wait_all.hpp
    pika/async_combinators/wait_any.hpp
    pika/async_combinators/wait_each.hpp
    pika/async_combinators/wait_some.hpp
    pika/async_combinators/when_all.hpp
    pika/async_combinators/when_any.hpp
    pika/async_combinators/when_each.hpp
    pika/async_combinators/when_some.hpp
)

include(pika_add_module)
pika_add_module(
  pika async_combinators
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_combinators_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "pika/async_combinators/future_wait.hpp"
  MODULE_DEPENDENCIES pika_async_base pika_config_local pika_errors pika_futures
                      pika_memory pika_pack_traversal
  CMAKE_SUBDIRS examples tests
)
