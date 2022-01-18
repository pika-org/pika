# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(debugging_headers
    pika/debugging/attach_debugger.hpp pika/debugging/backtrace.hpp
    pika/debugging/backtrace/backtrace.hpp pika/debugging/demangle_helper.hpp
    pika/debugging/environ.hpp pika/debugging/print.hpp
)

set(debugging_sources attach_debugger.cpp backtrace.cpp print.cpp)

include(pika_add_module)
pika_add_module(
  pika debugging
  GLOBAL_HEADER_GEN ON
  SOURCES ${debugging_sources}
  HEADERS ${debugging_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "pika/debugging/backtrace/backtrace.hpp"
  MODULE_DEPENDENCIES pika_config_local
  CMAKE_SUBDIRS examples tests
)
