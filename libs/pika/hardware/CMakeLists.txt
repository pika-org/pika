# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/hardware/include
set(hardware_headers
    pika/hardware/bit_manipulation.hpp
)

# Default location is $PIKA_ROOT/libs/hardware/src
set(hardware_sources)

include(pika_add_module)

pika_add_module(
  pika hardware
  SOURCES ${hardware_sources}
  HEADERS ${hardware_headers}
  MODULE_DEPENDENCIES pika_config
  CMAKE_SUBDIRS examples tests
)
