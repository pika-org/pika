# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(allocator_support_headers
    pika/allocator_support/aligned_allocator.hpp
    pika/allocator_support/allocator_deleter.hpp
    pika/allocator_support/internal_allocator.hpp
    pika/allocator_support/traits/is_allocator.hpp
)

set(allocator_support_sources)

include(pika_add_module)
pika_add_module(
  pika allocator_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${allocator_support_sources}
  HEADERS ${allocator_support_headers}
  DEPENDENCIES pika_dependencies_allocator
  MODULE_DEPENDENCIES pika_concepts pika_config_local pika_preprocessor
  CMAKE_SUBDIRS examples tests
)
