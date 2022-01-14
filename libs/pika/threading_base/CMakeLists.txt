# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_base_headers
    pika/threading_base/annotated_function.hpp
    pika/threading_base/callback_notifier.hpp
    pika/threading_base/create_thread.hpp
    pika/threading_base/create_work.hpp
    pika/threading_base/detail/reset_backtrace.hpp
    pika/threading_base/detail/reset_lco_description.hpp
    pika/threading_base/detail/get_default_pool.hpp
    pika/threading_base/execution_agent.hpp
    pika/threading_base/external_timer.hpp
    pika/threading_base/network_background_callback.hpp
    pika/threading_base/print.hpp
    pika/threading_base/register_thread.hpp
    pika/threading_base/scheduler_base.hpp
    pika/threading_base/scheduler_mode.hpp
    pika/threading_base/scheduler_state.hpp
    pika/threading_base/scoped_annotation.hpp
    pika/threading_base/set_thread_state.hpp
    pika/threading_base/thread_data.hpp
    pika/threading_base/thread_data_stackful.hpp
    pika/threading_base/thread_data_stackless.hpp
    pika/threading_base/thread_description.hpp
    pika/threading_base/thread_helpers.hpp
    pika/threading_base/thread_init_data.hpp
    pika/threading_base/thread_num_tss.hpp
    pika/threading_base/thread_pool_base.hpp
    pika/threading_base/thread_queue_init_parameters.hpp
    pika/threading_base/thread_specific_ptr.hpp
    pika/threading_base/threading_base_fwd.hpp
)

set(threading_base_sources
    annotated_function.cpp
    create_thread.cpp
    create_work.cpp
    execution_agent.cpp
    external_timer.cpp
    get_default_pool.cpp
    print.cpp
    scheduler_base.cpp
    set_thread_state.cpp
    set_thread_state_timed.cpp
    thread_data.cpp
    thread_data_stackful.cpp
    thread_data_stackless.cpp
    thread_description.cpp
    thread_helpers.cpp
    thread_num_tss.cpp
    thread_pool_base.cpp
)

if(PIKA_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  set(additional_dependencies pika_debugging)
endif()

include(pika_add_module)
pika_add_module(
  pika threading_base
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER "pika/threading_base/thread_data_stackful.hpp"
                             "pika/threading_base/thread_data_stackless.hpp"
  SOURCES ${threading_base_sources}
  HEADERS ${threading_base_headers}
  MODULE_DEPENDENCIES
    pika_affinity
    pika_allocator_support
    pika_assertion
    pika_execution_base
    pika_config_local
    pika_concurrency
    pika_coroutines
    pika_debugging
    pika_errors
    pika_format
    pika_functional
    pika_hardware
    pika_itt_notify
    pika_lock_registration
    pika_logging
    pika_memory
    pika_timing
    pika_type_support
    ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
