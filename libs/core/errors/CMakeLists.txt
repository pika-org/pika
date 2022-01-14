# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/errors/include
set(errors_headers
    hpx/errors/error.hpp
    hpx/errors/error_code.hpp
    hpx/errors/exception.hpp
    hpx/errors/exception_fwd.hpp
    hpx/errors/exception_info.hpp
    hpx/errors/exception_list.hpp
    hpx/errors/throw_exception.hpp
    hpx/errors/try_catch_exception_ptr.hpp
)

# Default location is $HPX_ROOT/libs/errors/src
set(errors_sources error_code.cpp exception.cpp exception_list.cpp
                   throw_exception.cpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${errors_sources}
  HEADERS ${errors_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config_local
    hpx_filesystem
    hpx_format
    hpx_logging
    hpx_preprocessor
    hpx_thread_support
  CMAKE_SUBDIRS examples tests
)
