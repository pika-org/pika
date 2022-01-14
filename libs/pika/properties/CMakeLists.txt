# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(properties_headers pika/property.hpp pika/properties/property.hpp)

include(pika_add_module)
pika_add_module(
  pika properties
  GLOBAL_HEADER_GEN ON
  HEADERS ${properties_headers}
  DEPENDENCIES pika_tag_invoke
  CMAKE_SUBDIRS examples tests
)
