# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/timing/include
set(timing_headers
    pika/timing/high_resolution_clock.hpp pika/timing/high_resolution_timer.hpp
    pika/timing/scoped_timer.hpp pika/timing/steady_clock.hpp
    pika/timing/tick_counter.hpp
)

# cmake will not create a separate VS project without any source files, thus
# this adds a dummy (empty) source file to the target Default location is
# $PIKA_ROOT/libs/timing/src
set(timing_sources)

include(pika_add_module)
pika_add_module(
  pika timing
  GLOBAL_HEADER_GEN ON
  SOURCES ${timing_sources}
  HEADERS ${timing_headers}
  MODULE_DEPENDENCIES pika_config_local pika_hardware pika_type_support
  CMAKE_SUBDIRS examples tests
)
