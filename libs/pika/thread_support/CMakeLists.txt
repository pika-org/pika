# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(thread_support_headers
    pika/thread_support/assert_owns_lock.hpp
    pika/thread_support/atomic_count.hpp
    pika/thread_support/set_thread_name.hpp
    pika/thread_support/spinlock.hpp
    pika/thread_support/thread_specific_ptr.hpp
    pika/thread_support/unlock_guard.hpp
)

set(thread_support_sources set_thread_name.cpp spinlock.cpp)

include(pika_add_module)
pika_add_module(
  pika thread_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_support_sources}
  HEADERS ${thread_support_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config_local pika_concepts
                      pika_type_support
  CMAKE_SUBDIRS examples tests
)
