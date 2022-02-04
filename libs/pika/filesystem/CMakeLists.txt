# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(filesystem_headers pika/detail/filesystem.hpp)
set(filesystem_sources filesystem.cpp)

include(pika_add_module)
pika_add_module(
  pika filesystem
  GLOBAL_HEADER_GEN OFF
  HEADERS ${filesystem_headers}
  SOURCES ${filesystem_sources}
  MODULE_DEPENDENCIES pika_config
  CMAKE_SUBDIRS examples tests
)
