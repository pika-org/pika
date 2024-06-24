# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/logging/include
set(logging_headers pika/logging.hpp)

# Default location is $PIKA_ROOT/libs/logging/src
set(logging_sources logging.cpp)
# TODO: Logging disabled because of ICEs in GCC
set(logging_module_includes <type_traits> <utility> <cstddef> <tuple> <functional>) # <spdlog/common.h> <spdlog/sinks/sink.h>)

include(pika_add_module)
pika_add_module(
  pika logging
  GLOBAL_HEADER_GEN OFF
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  MODULE_INCLUDES ${logging_module_includes}
  MODULE_DEPENDENCIES pika_assertion pika_config pika_filesystem pika_string_util pika_timing
                      pika_thread_support
  CMAKE_SUBDIRS examples tests
)
