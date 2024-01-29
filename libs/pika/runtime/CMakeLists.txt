# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_headers
    pika/runtime/config_entry.hpp
    pika/runtime/custom_exception_info.hpp
    pika/runtime/debugging.hpp
    pika/runtime/detail/runtime_fwd.hpp
    pika/runtime/get_os_thread_count.hpp
    pika/runtime/get_thread_name.hpp
    pika/runtime/get_worker_thread_num.hpp
    pika/runtime/os_thread_type.hpp
    pika/runtime/report_error.hpp
    pika/runtime/runtime_handlers.hpp
    pika/runtime/runtime.hpp
    pika/runtime/runtime_fwd.hpp
    pika/runtime/state.hpp
    pika/runtime/shutdown_function.hpp
    pika/runtime/startup_function.hpp
    pika/runtime/thread_hooks.hpp
    pika/runtime/thread_mapper.hpp
    pika/runtime/thread_pool_helpers.hpp
    pika/runtime/thread_stacktrace.hpp
)

set(runtime_sources
    custom_exception_info.cpp
    debugging.cpp
    os_thread_type.cpp
    runtime_handlers.cpp
    runtime.cpp
    state.cpp
    thread_mapper.cpp
    thread_pool_helpers.cpp
    thread_stacktrace.cpp
)

if(PIKA_WITH_MPI)
  list(APPEND runtime_additional_module_dependencies pika_mpi_base)
endif()

include(pika_add_module)
pika_add_module(
  pika runtime
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_sources}
  HEADERS ${runtime_headers}
  MODULE_DEPENDENCIES
    pika_command_line_handling
    pika_debugging
    pika_errors
    pika_execution
    pika_logging
    pika_program_options
    pika_resource_partitioner
    pika_runtime_configuration
    pika_threading
    pika_threading_base
    pika_thread_manager
    pika_timing
    pika_topology
    ${runtime_additional_module_dependencies}
  CMAKE_SUBDIRS examples tests
)
