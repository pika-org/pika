# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/itt_notify/include
set(itt_notify_headers pika/modules/itt_notify.hpp pika/itt_notify/thread_name.hpp)

set(itt_notify_sources itt_notify.cpp thread_name.cpp)

if(TARGET pika_internal::amplifier)
  set(itt_notify_optional_dependencies pika_internal::amplifier)
endif()

include(pika_add_module)
pika_add_module(
  pika itt_notify
  GLOBAL_HEADER_GEN OFF
  SOURCES ${itt_notify_sources}
  HEADERS ${itt_notify_headers}
  DEPENDENCIES ${itt_notify_optional_dependencies}
  MODULE_DEPENDENCIES pika_config
  CMAKE_SUBDIRS examples tests
)
