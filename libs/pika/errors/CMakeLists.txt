# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/errors/include
set(errors_headers
    pika/errors/error.hpp
    pika/errors/error_code.hpp
    pika/errors/exception.hpp
    pika/errors/exception_fwd.hpp
    pika/errors/exception_info.hpp
    pika/errors/exception_list.hpp
    pika/errors/throw_exception.hpp
    pika/errors/try_catch_exception_ptr.hpp
)

# Default location is $PIKA_ROOT/libs/errors/src
set(errors_sources error_code.cpp exception.cpp exception_list.cpp
                   throw_exception.cpp
)

include(pika_add_module)
pika_add_module(
  pika errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${errors_sources}
  HEADERS ${errors_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_filesystem
    pika_format
    pika_logging
    pika_preprocessor
    pika_thread_support
  CMAKE_SUBDIRS examples tests
)
