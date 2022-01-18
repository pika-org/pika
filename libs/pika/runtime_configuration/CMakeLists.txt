# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_headers
    pika/runtime_configuration/init_ini_data.hpp
    pika/runtime_configuration/runtime_configuration.hpp
    pika/runtime_configuration/runtime_configuration_fwd.hpp
    pika/runtime_configuration/runtime_mode.hpp
)

set(runtime_configuration_sources init_ini_data.cpp runtime_configuration.cpp
                                  runtime_mode.cpp
)

include(pika_add_module)
pika_add_module(
  pika runtime_configuration
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_configuration_sources}
  HEADERS ${runtime_configuration_headers}
  MODULE_DEPENDENCIES
    pika_program_options
    pika_errors
    pika_filesystem
    pika_itt_notify
    pika_logging
    pika_ini
    pika_prefix
    pika_coroutines
    pika_version_local
    pika_synchronization
  CMAKE_SUBDIRS examples tests
)
