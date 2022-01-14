# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_local_headers pika/local/version.hpp)

set(version_local_sources version.cpp)

if(PIKA_WITH_ASYNC_MPI)
  include(pika_setup_mpi)
  set(mpi_dependencies Mpi::mpi)
endif()

include(pika_add_module)
pika_add_module(
  pika version_local
  GLOBAL_HEADER_GEN OFF
  SOURCES ${version_local_sources}
  HEADERS ${version_local_headers}
  MODULE_DEPENDENCIES pika_config_local pika_format pika_prefix
  DEPENDENCIES Hwloc::hwloc ${mpi_dependencies}
)

target_include_directories(
  pika_version_local PRIVATE $<BUILD_INTERFACE:${PIKA_BINARY_DIR}>
)
