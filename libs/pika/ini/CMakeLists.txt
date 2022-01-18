# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ini_headers pika/ini/ini.hpp)
set(ini_sources ini.cpp)

include(pika_add_module)
pika_add_module(
  pika ini
  GLOBAL_HEADER_GEN ON
  SOURCES ${ini_sources}
  HEADERS ${ini_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_errors
    pika_serialization
    pika_string_util
    pika_thread_support
    pika_concurrency
    pika_functional
    pika_util
  CMAKE_SUBDIRS examples tests
)
