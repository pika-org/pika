# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_util_headers
    pika/string_util/bad_lexical_cast.hpp
    pika/string_util/case_conv.hpp
    pika/string_util/classification.hpp
    pika/string_util/from_string.hpp
    pika/string_util/split.hpp
    pika/string_util/to_string.hpp
    pika/string_util/trim.hpp
)

set(string_util_sources bad_lexical_cast.cpp)
set(string_util_module_includes <utility> <algorithm> <variant> <sstream> <fmt/format.h>)

include(pika_add_module)
pika_add_module(
  pika string_util
  GLOBAL_HEADER_GEN ON
  HEADERS ${string_util_headers}
  SOURCES ${string_util_sources}
  MODULE_INCLUDES ${string_util_module_includes}
  MODULE_DEPENDENCIES pika_config
  CMAKE_SUBDIRS examples tests
)
