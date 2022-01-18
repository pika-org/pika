# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Note: PIKA_WITH_ASYNC_MPI is handled in the main CMakeLists.txt

# if the user does not want mpi async futures, quit - the module will not be
# enabled
if(NOT ${PIKA_WITH_ASYNC_MPI})
  return()
endif()

# setup MPI, if necessary
if(NOT TARGET Mpi::mpi)
  include(pika_setup_mpi)
  pika_setup_mpi()
endif()

# Default location is $PIKA_ROOT/libs/mpi/include
set(async_mpi_headers
    pika/async_mpi/mpi_exception.hpp pika/async_mpi/mpi_executor.hpp
    pika/async_mpi/mpi_future.hpp pika/async_mpi/transform_mpi.hpp
)

# Default location is $PIKA_ROOT/libs/mpi/src
set(mpi_sources mpi_future.cpp)

include(pika_add_module)
pika_add_module(
  pika async_mpi
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_sources}
  HEADERS ${async_mpi_headers}
  DEPENDENCIES Mpi::mpi
  MODULE_DEPENDENCIES
    pika_concurrency
    pika_errors
    pika_execution_base
    pika_memory
    pika_threading_base
    pika_mpi_base
    pika_runtime_local
    pika_config_local
  CMAKE_SUBDIRS examples tests
)
