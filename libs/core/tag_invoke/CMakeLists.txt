# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tag_invoke_headers
    hpx/functional/detail/invoke.hpp
    hpx/functional/detail/tag_fallback_invoke.hpp
    hpx/functional/detail/tag_priority_invoke.hpp
    hpx/functional/invoke_result.hpp
    hpx/functional/tag_invoke.hpp
    hpx/functional/traits/is_invocable.hpp
)

set(tag_invoke_sources)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local tag_invoke
  GLOBAL_HEADER_GEN ON
  SOURCES ${tag_invoke_sources}
  HEADERS ${tag_invoke_headers}
  MODULE_DEPENDENCIES hpx_config_local hpx_type_support
  CMAKE_SUBDIRS examples tests
)
