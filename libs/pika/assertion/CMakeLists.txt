# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/assertion/include
set(assertion_headers
    pika/assert.hpp pika/assertion/current_function.hpp pika/assertion/assertion_handler.hpp
    pika/assertion/evaluate_assert.hpp pika/assertion/source_location.hpp
)

# Default location is $PIKA_ROOT/libs/assertion/src
set(assertion_sources assertion.cpp source_location.cpp)
set(assertion_module_includes <exception> <string> <type_traits> <utility>)

include(pika_add_module)
pika_add_module(
  pika assertion
  GLOBAL_HEADER_GEN OFF
  SOURCES ${assertion_sources}
  HEADERS ${assertion_headers}
  MODULE_INCLUDES ${assertion_module_includes}
  MODULE_DEPENDENCIES pika_config pika_preprocessor
  CMAKE_SUBDIRS examples tests
)
