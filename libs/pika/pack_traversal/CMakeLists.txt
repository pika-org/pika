# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(pack_traversal_headers
    pika/pack_traversal/detail/container_category.hpp
    pika/pack_traversal/detail/pack_traversal_async_impl.hpp
    pika/pack_traversal/detail/pack_traversal_impl.hpp
    pika/pack_traversal/detail/unwrap_impl.hpp
    pika/pack_traversal/pack_traversal.hpp
    pika/pack_traversal/pack_traversal_async.hpp
    pika/pack_traversal/traits/pack_traversal_rebind_container.hpp
    pika/pack_traversal/unwrap.hpp
)

include(pika_add_module)
pika_add_module(
  pika pack_traversal
  GLOBAL_HEADER_GEN ON
  HEADERS ${pack_traversal_headers}
  MODULE_DEPENDENCIES pika_futures pika_memory
  CMAKE_SUBDIRS examples tests
)
