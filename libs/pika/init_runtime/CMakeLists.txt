# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_headers
    pika/init_runtime/detail/init_logging.hpp pika/init_runtime/init_runtime.hpp
    pika/init_runtime/scoped_finalize.hpp pika/init.hpp
)

set(init_runtime_sources init_logging.cpp init_runtime.cpp scoped_finalize.cpp)

if(PIKA_WITH_MPI)
  list(APPEND init_runtime_additional_module_dependencies pika_mpi_base pika_async_mpi)
endif()

include(pika_add_module)
pika_add_module(
  pika init_runtime
  GLOBAL_HEADER_GEN ON
  SOURCES ${init_runtime_sources}
  HEADERS ${init_runtime_headers}
  MODULE_DEPENDENCIES
    pika_command_line_handling
    pika_debugging
    pika_errors
    pika_filesystem
    pika_lock_registration
    pika_logging
    pika_program_options
    pika_runtime
    pika_schedulers
    pika_threading_base
    pika_timing
    ${init_runtime_additional_module_dependencies}
  CMAKE_SUBDIRS examples tests
)
