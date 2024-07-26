# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Note: PIKA_WITH_MPI is handled in the main CMakeLists.txt

# if the user does not want support, quit - the module will not be enabled
if(NOT ${PIKA_WITH_MPI})
  return()
endif()

# For testing and benchmarking, we would like to know if mpi continuations extension is present
try_compile(
  # Name of variable to store the compile result (TRUE or FALSE) in:
  MPIX_COMPILE_RESULT
  # Binary directory:
  ${CMAKE_CURRENT_BINARY_DIR}/
  # Source file to be compiled:
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/check_openmpi_continuations.cpp
  # Where to store the output produced during compilation:
  OUTPUT_VARIABLE TEST_COMPILE_OUTPUT
)

# Did compilation succeed and process return 0 (success)?
if("${MPIX_COMPILE_RESULT}")
  message(STATUS "MPIx Continuations detected")
  set(PIKA_WITH_MPIX_CONTINUATIONS True)
  set(PIKA_MPI_MODES_LOOP_COUNT 79)
else()
  message(STATUS "MPIx Continuations not detected")
  set(PIKA_WITH_MPIX_CONTINUATIONS False)
  set(PIKA_MPI_MODES_LOOP_COUNT 63)
endif()

# Default location is $PIKA_ROOT/libs/mpi/include
set(async_mpi_headers
    pika/async_mpi/mpi_exception.hpp pika/async_mpi/mpi_helpers.hpp pika/async_mpi/mpi_polling.hpp
    pika/async_mpi/dispatch_mpi.hpp pika/async_mpi/trigger_mpi.hpp pika/async_mpi/transform_mpi.hpp
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
    pika_resource_partitioner
    pika_threading_base
    pika_mpi_base
    pika_runtime
    pika_config
  CMAKE_SUBDIRS examples tests
)
