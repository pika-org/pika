# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/format/include
set(format_headers
    pika/format.hpp pika/modules/format.hpp pika/util/bad_lexical_cast.hpp
    pika/util/from_string.hpp pika/util/to_string.hpp
)

# Default location is $PIKA_ROOT/libs/format/src
set(format_sources format.cpp util/bad_lexical_cast.cpp)

include(pika_add_module)
pika_add_module(
  pika format
  GLOBAL_HEADER_GEN OFF
  SOURCES ${format_sources}
  HEADERS ${format_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config_local pika_type_support
  CMAKE_SUBDIRS examples tests
)
