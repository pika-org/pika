# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/logging/include
set(logging_headers
    pika/modules/logging.hpp
    pika/logging/detail/macros.hpp
    pika/logging/detail/logger.hpp
    pika/logging/format/destinations.hpp
    pika/logging/format/formatters.hpp
    pika/logging/format/named_write.hpp
    pika/logging/level.hpp
    pika/logging/logging.hpp
    pika/logging/manipulator.hpp
    pika/logging/message.hpp
)

# Default location is $PIKA_ROOT/libs/logging/src
set(logging_sources
    level.cpp
    logging.cpp
    manipulator.cpp
    format/named_write.cpp
    format/destination/defaults_destination.cpp
    format/destination/file.cpp
    format/formatter/high_precision_time.cpp
    format/formatter/defaults_formatter.cpp
    format/formatter/thread_id.cpp
)

include(pika_add_module)
pika_add_module(
  pika logging
  GLOBAL_HEADER_GEN OFF
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_format
    pika_filesystem
    pika_format
    pika_timing
    pika_thread_support
  CMAKE_SUBDIRS examples tests
)
