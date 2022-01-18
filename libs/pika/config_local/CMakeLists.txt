# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(config_local_headers
    pika/local/config.hpp
    pika/local/config/attributes.hpp
    pika/local/config/autolink.hpp
    pika/local/config/branch_hints.hpp
    pika/local/config/compiler_fence.hpp
    pika/local/config/compiler_specific.hpp
    pika/local/config/constexpr.hpp
    pika/local/config/debug.hpp
    pika/local/config/deprecation.hpp
    pika/local/config/detail/compat_error_code.hpp
    pika/local/config/emulate_deleted.hpp
    pika/local/config/endian.hpp
    pika/local/config/export_definitions.hpp
    pika/local/config/forceinline.hpp
    pika/local/config/forward.hpp
    pika/local/config/lambda_capture_this.hpp
    pika/local/config/manual_profiling.hpp
    pika/local/config/move.hpp
    pika/local/config/threads_stack.hpp
    pika/local/config/warnings_prefix.hpp
    pika/local/config/warnings_suffix.hpp
)

set(config_local_sources version.cpp)

include(pika_add_module)
pika_add_module(
  pika config_local CONFIG_FILES
  GLOBAL_HEADER_GEN OFF
  SOURCES ${config_local_sources}
  HEADERS ${config_local_headers}
  MODULE_DEPENDENCIES pika_preprocessor
  DEPENDENCIES pika_dependencies_boost
  CMAKE_SUBDIRS examples tests
)
