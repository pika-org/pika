# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(pack_traversal_headers
    hpx/pack_traversal/detail/container_category.hpp
    hpx/pack_traversal/detail/pack_traversal_async_impl.hpp
    hpx/pack_traversal/detail/pack_traversal_impl.hpp
    hpx/pack_traversal/detail/unwrap_impl.hpp
    hpx/pack_traversal/pack_traversal.hpp
    hpx/pack_traversal/pack_traversal_async.hpp
    hpx/pack_traversal/traits/pack_traversal_rebind_container.hpp
    hpx/pack_traversal/unwrap.hpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local pack_traversal
  GLOBAL_HEADER_GEN ON
  HEADERS ${pack_traversal_headers}
  MODULE_DEPENDENCIES hpx_futures hpx_memory
  CMAKE_SUBDIRS examples tests
)
