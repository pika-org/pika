# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_base_headers pika/async_base/scheduling_properties.hpp)

include(pika_add_module)
pika_add_module(
  pika async_base
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_base_headers}
  MODULE_DEPENDENCIES pika_tag_invoke
  CMAKE_SUBDIRS examples tests
)
