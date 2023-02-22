# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/timing/include
set(timing_headers
    pika/timing/detail/timestamp.hpp
    pika/timing/detail/timestamp/bgq.hpp
    pika/timing/detail/timestamp/cuda.hpp
    pika/timing/detail/timestamp/linux_generic.hpp
    pika/timing/detail/timestamp/linux_x86_32.hpp
    pika/timing/detail/timestamp/linux_x86_64.hpp
    pika/timing/detail/timestamp/msvc.hpp
    pika/timing/high_resolution_timer.hpp
    pika/timing/steady_clock.hpp
    pika/timing/tick_counter.hpp
)

# cmake will not create a separate VS project without any source files, thus
# this adds a dummy (empty) source file to the target Default location is
# $PIKA_ROOT/libs/timing/src
set(timing_sources)

include(pika_add_module)
pika_add_module(
  pika timing
  SOURCES ${timing_sources}
  HEADERS ${timing_headers}
  MODULE_DEPENDENCIES pika_config pika_type_support
  CMAKE_SUBDIRS examples tests
)
