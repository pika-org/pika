# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_local_headers
    hpx/runtime_local/config_entry.hpp
    hpx/runtime_local/custom_exception_info.hpp
    hpx/runtime_local/debugging.hpp
    hpx/runtime_local/detail/runtime_local_fwd.hpp
    hpx/runtime_local/detail/serialize_exception.hpp
    hpx/runtime_local/get_locality_id.hpp
    hpx/runtime_local/get_locality_name.hpp
    hpx/runtime_local/get_num_all_localities.hpp
    hpx/runtime_local/get_os_thread_count.hpp
    hpx/runtime_local/get_thread_name.hpp
    hpx/runtime_local/get_worker_thread_num.hpp
    hpx/runtime_local/os_thread_type.hpp
    hpx/runtime_local/report_error.hpp
    hpx/runtime_local/run_as_hpx_thread.hpp
    hpx/runtime_local/runtime_handlers.hpp
    hpx/runtime_local/runtime_local.hpp
    hpx/runtime_local/runtime_local_fwd.hpp
    hpx/runtime_local/state.hpp
    hpx/runtime_local/shutdown_function.hpp
    hpx/runtime_local/startup_function.hpp
    hpx/runtime_local/thread_hooks.hpp
    hpx/runtime_local/thread_mapper.hpp
    hpx/runtime_local/thread_pool_helpers.hpp
    hpx/runtime_local/thread_stacktrace.hpp
)

set(runtime_local_sources
    custom_exception_info.cpp
    debugging.cpp
    get_locality_name.cpp
    os_thread_type.cpp
    runtime_handlers.cpp
    runtime_local.cpp
    serialize_exception.cpp
    state.cpp
    thread_mapper.cpp
    thread_pool_helpers.cpp
    thread_stacktrace.cpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local runtime_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_local_sources}
  HEADERS ${runtime_local_headers}
  MODULE_DEPENDENCIES
    hpx_command_line_handling_local
    hpx_debugging
    hpx_errors
    hpx_execution
    hpx_format
    hpx_futures
    hpx_logging
    hpx_program_options
    hpx_resource_partitioner
    hpx_runtime_configuration
    hpx_threading
    hpx_threading_base
    hpx_threadmanager
    hpx_timing
    hpx_topology
  CMAKE_SUBDIRS examples tests
)
