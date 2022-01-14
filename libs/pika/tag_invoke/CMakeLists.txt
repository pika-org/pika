# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tag_invoke_headers
    pika/functional/detail/invoke.hpp
    pika/functional/detail/tag_fallback_invoke.hpp
    pika/functional/detail/tag_priority_invoke.hpp
    pika/functional/invoke_result.hpp
    pika/functional/tag_invoke.hpp
    pika/functional/traits/is_invocable.hpp
)

set(tag_invoke_sources)

include(pika_add_module)
pika_add_module(
  pika tag_invoke
  GLOBAL_HEADER_GEN ON
  SOURCES ${tag_invoke_sources}
  HEADERS ${tag_invoke_headers}
  MODULE_DEPENDENCIES pika_config_local pika_type_support
  CMAKE_SUBDIRS examples tests
)
