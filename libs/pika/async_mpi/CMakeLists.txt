# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Note: PIKA_WITH_MPI is handled in the main CMakeLists.txt

# if the user does not want support, quit - the module will not be enabled
if(NOT ${PIKA_WITH_MPI})
  return()
endif()

# Default location is $PIKA_ROOT/libs/mpi/include
set(async_mpi_headers
    pika/async_mpi/mpi_exception.hpp pika/async_mpi/mpi_helpers.hpp
    pika/async_mpi/mpi_polling.hpp pika/async_mpi/dispatch_mpi.hpp
    pika/async_mpi/trigger_mpi.hpp pika/async_mpi/transform_mpi.hpp
)

# Default location is $PIKA_ROOT/libs/mpi/src
set(mpi_sources mpi_exception.cpp mpi_polling.cpp)

include(pika_add_module)
pika_add_module(
  pika async_mpi
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_sources}
  HEADERS ${async_mpi_headers}
  DEPENDENCIES MPI::MPI_CXX
  MODULE_DEPENDENCIES
    pika_concurrency
    pika_debugging
    pika_errors
    pika_execution_base
    pika_executors
    pika_memory
    pika_threading_base
    pika_mpi_base
    pika_runtime
    pika_config
  CMAKE_SUBDIRS examples tests
)
