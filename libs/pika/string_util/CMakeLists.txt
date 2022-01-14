# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_util_headers
    pika/string_util/case_conv.hpp pika/string_util/classification.hpp
    pika/string_util/split.hpp pika/string_util/trim.hpp
)

include(pika_add_module)
pika_add_module(
  pika string_util
  GLOBAL_HEADER_GEN ON
  HEADERS ${string_util_headers}
  CMAKE_SUBDIRS examples tests
)
