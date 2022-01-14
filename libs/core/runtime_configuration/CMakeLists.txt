# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_headers
    hpx/runtime_configuration/init_ini_data.hpp
    hpx/runtime_configuration/runtime_configuration.hpp
    hpx/runtime_configuration/runtime_configuration_fwd.hpp
    hpx/runtime_configuration/runtime_mode.hpp
)

set(runtime_configuration_sources init_ini_data.cpp runtime_configuration.cpp
                                  runtime_mode.cpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local runtime_configuration
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_configuration_sources}
  HEADERS ${runtime_configuration_headers}
  MODULE_DEPENDENCIES
    hpx_program_options
    hpx_errors
    hpx_filesystem
    hpx_itt_notify
    hpx_logging
    hpx_ini
    hpx_prefix
    hpx_coroutines
    hpx_version_local
    hpx_synchronization
  CMAKE_SUBDIRS examples tests
)
