# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT PIKA_WITH_ASYNC_MPI)
  return()
endif()

include(pika_setup_mpi)
pika_setup_mpi()
set(additional_dependencies Mpi::mpi)

set(mpi_base_headers pika/mpi_base/mpi.hpp pika/mpi_base/mpi_environment.hpp)
set(mpi_base_sources mpi_environment.cpp)

include(pika_add_module)
pika_add_module(
  pika mpi_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_base_sources}
  HEADERS ${mpi_base_headers}
  MODULE_DEPENDENCIES pika_logging pika_runtime_configuration pika_util
  DEPENDENCIES ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
