# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(config_headers
    pika/config.hpp
    pika/config/attributes.hpp
    pika/config/autolink.hpp
    pika/config/branch_hints.hpp
    pika/config/compiler_fence.hpp
    pika/config/compiler_specific.hpp
    pika/config/constexpr.hpp
    pika/config/debug.hpp
    pika/config/emulate_deleted.hpp
    pika/config/export_definitions.hpp
    pika/config/forceinline.hpp
    pika/config/manual_profiling.hpp
    pika/config/static_call_operator.hpp
    pika/config/threads_stack.hpp
    pika/config/warnings_prefix.hpp
    pika/config/warnings_suffix.hpp
)

set(config_sources version.cpp)

include(pika_add_module)
pika_add_module(
  pika config CONFIG_FILES
  GLOBAL_HEADER_GEN OFF
  SOURCES ${config_sources}
  HEADERS ${config_headers}
  MODULE_DEPENDENCIES pika_preprocessor
  DEPENDENCIES pika_dependencies_boost
  CMAKE_SUBDIRS examples tests
)
