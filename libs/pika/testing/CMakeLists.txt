# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/testing/include
set(testing_headers pika/modules/testing.hpp pika/testing/performance.hpp)

# Default location is $PIKA_ROOT/libs/testing/src
set(testing_sources testing.cpp performance.cpp)

include(pika_add_module)
pika_add_module(
  pika testing
  GLOBAL_HEADER_GEN OFF
  SOURCES ${testing_sources}
  HEADERS ${testing_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config_local pika_format pika_functional
                      pika_preprocessor pika_util
  CMAKE_SUBDIRS examples tests
)
