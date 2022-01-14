# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_headers pika/threading/jthread.hpp pika/threading/thread.hpp)

set(threading_sources thread.cpp)

include(pika_add_module)
pika_add_module(
  pika threading
  GLOBAL_HEADER_GEN ON
  SOURCES ${threading_sources}
  HEADERS ${threading_headers}
  MODULE_DEPENDENCIES pika_errors pika_futures pika_lock_registration pika_memory
  CMAKE_SUBDIRS examples tests
)
