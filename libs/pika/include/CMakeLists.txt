# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(include_headers
    pika/async_rw_mutex.hpp
    pika/barrier.hpp
    pika/channel.hpp
    pika/chrono.hpp
    pika/condition_variable.hpp
    pika/cuda.hpp
    pika/exception.hpp
    pika/execution.hpp
    pika/functional.hpp
    pika/future.hpp
    pika/latch.hpp
    pika/mpi.hpp
    pika/mutex.hpp
    pika/runtime.hpp
    pika/semaphore.hpp
    pika/shared_mutex.hpp
    pika/stop_token.hpp
    pika/system_error.hpp
    pika/thread.hpp
    pika/unwrap.hpp
)

set(include_additional_module_dependencies)

if(PIKA_WITH_GPU_SUPPORT)
  list(APPEND include_additional_module_dependencies pika_async_cuda)
endif()

if(PIKA_WITH_MPI)
  list(APPEND include_additional_module_dependencies pika_async_mpi)
endif()

include(pika_add_module)
pika_add_module(
  pika include
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_headers}
  MODULE_DEPENDENCIES
    pika_async_base
    pika_async_combinators
    pika_async
    pika_errors
    pika_execution
    pika_executors
    pika_futures
    pika_lcos
    pika_runtime
    pika_synchronization
    ${include_additional_module_dependencies}
  CMAKE_SUBDIRS examples tests
)
