# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_local_headers
    pika/runtime_local/config_entry.hpp
    pika/runtime_local/custom_exception_info.hpp
    pika/runtime_local/debugging.hpp
    pika/runtime_local/detail/runtime_local_fwd.hpp
    pika/runtime_local/detail/serialize_exception.hpp
    pika/runtime_local/get_locality_id.hpp
    pika/runtime_local/get_locality_name.hpp
    pika/runtime_local/get_num_all_localities.hpp
    pika/runtime_local/get_os_thread_count.hpp
    pika/runtime_local/get_thread_name.hpp
    pika/runtime_local/get_worker_thread_num.hpp
    pika/runtime_local/os_thread_type.hpp
    pika/runtime_local/report_error.hpp
    pika/runtime_local/run_as_pika_thread.hpp
    pika/runtime_local/runtime_handlers.hpp
    pika/runtime_local/runtime_local.hpp
    pika/runtime_local/runtime_local_fwd.hpp
    pika/runtime_local/state.hpp
    pika/runtime_local/shutdown_function.hpp
    pika/runtime_local/startup_function.hpp
    pika/runtime_local/thread_hooks.hpp
    pika/runtime_local/thread_mapper.hpp
    pika/runtime_local/thread_pool_helpers.hpp
    pika/runtime_local/thread_stacktrace.hpp
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

include(pika_add_module)
pika_add_module(
  pika runtime_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_local_sources}
  HEADERS ${runtime_local_headers}
  MODULE_DEPENDENCIES
    pika_command_line_handling_local
    pika_debugging
    pika_errors
    pika_execution
    pika_format
    pika_futures
    pika_logging
    pika_program_options
    pika_resource_partitioner
    pika_runtime_configuration
    pika_threading
    pika_threading_base
    pika_threadmanager
    pika_timing
    pika_topology
  CMAKE_SUBDIRS examples tests
)
