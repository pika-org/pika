# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(execution_base_headers
    pika/execution_base/agent_base.hpp
    pika/execution_base/agent_ref.hpp
    pika/execution_base/any_sender.hpp
    pika/execution_base/context_base.hpp
    pika/execution_base/detail/spinlock_deadlock_detection.hpp
    pika/execution_base/execution.hpp
    pika/execution_base/completion_scheduler.hpp
    pika/execution_base/operation_state.hpp
    pika/execution_base/receiver.hpp
    pika/execution_base/resource_base.hpp
    pika/execution_base/sender.hpp
    pika/execution_base/this_thread.hpp
    pika/execution_base/traits/is_executor.hpp
    pika/execution_base/traits/is_executor_parameters.hpp
)

set(execution_base_sources agent_ref.cpp any_sender.cpp
                           spinlock_deadlock_detection.cpp this_thread.cpp
)

include(pika_add_module)
pika_add_module(
  pika execution_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${execution_base_sources}
  HEADERS ${execution_base_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_errors
    pika_format
    pika_functional
    pika_iterator_support
    pika_lock_registration
    pika_timing
    pika_type_support
  CMAKE_SUBDIRS examples tests
)
