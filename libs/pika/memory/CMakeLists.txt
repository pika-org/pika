# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Default location is $PIKA_ROOT/libs/memory/include
set(memory_headers pika/memory/intrusive_ptr.hpp
                   pika/memory/detail/sp_convertible.hpp
)

# Default location is $PIKA_ROOT/libs/memory/src
set(memory_sources)

include(pika_add_module)
pika_add_module(
  pika memory
  GLOBAL_HEADER_GEN ON
  SOURCES ${memory_sources}
  HEADERS ${memory_headers}
  MODULE_DEPENDENCIES pika_config pika_assertion
  CMAKE_SUBDIRS examples tests
)
