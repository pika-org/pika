# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT PIKA_WITH_MPI)
  return()
endif()

set(mpi_base_headers pika/mpi_base/mpi.hpp pika/mpi_base/mpi_environment.hpp
                     pika/mpi_base/mpi_exception.hpp
)
set(mpi_base_sources mpi_environment.cpp mpi_exception.cpp)

include(pika_add_module)
pika_add_module(
  pika mpi_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_base_sources}
  HEADERS ${mpi_base_headers}
  MODULE_DEPENDENCIES pika_logging pika_runtime_configuration pika_util
  DEPENDENCIES MPI::MPI_CXX
  CMAKE_SUBDIRS examples tests
)
