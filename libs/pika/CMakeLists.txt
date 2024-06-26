# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(pika_message)

# cmake-format: off
set(_pika_modules
    # affinity
    # allocator_support
    assertion
    # async_base
    # async_cuda
    # async_cuda_base
    # async_mpi
    # command_line_handling
    concepts
    # concurrency
    config
    # coroutines
    datastructures
    debugging
    errors
    # execution
    # execution_base
    # executors
    filesystem
    functional
    # hashing
    # include
    # ini
    # init_runtime
    iterator_support
    itt_notify
    lock_registration
    logging
    # memory
    # mpi_base
    # prefix
    preprocessor
    # program_options
    # properties
    # resource_partitioner
    # runtime_configuration
    # runtime
    # schedulers
    string_util
    # synchronization
    tag_invoke
    # thread_pools
    thread_support
    # threading
    # threading_base
    # thread_manager
    timing
    # topology
    type_support
    util
    # version
)
# cmake-format: on

set(module_imports)
foreach(module ${_pika_modules})
  add_subdirectory(${module})

  set(module_imports "${module_imports}\nexport import pika.${module};")
endforeach()
set(module_includes)
set(module_name pika.all)
set(module_interface_unit "${CMAKE_CURRENT_BINARY_DIR}/module.cpp")
set(module_headers)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/module.cpp.in" "${module_interface_unit}" @ONLY
)
