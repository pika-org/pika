# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(concepts_headers pika/concepts/concepts.hpp pika/concepts/has_member_xxx.hpp
                     pika/concepts/has_xxx.hpp
)

set(concepts_sources)

include(pika_add_module)
pika_add_module(
  pika concepts
  GLOBAL_HEADER_GEN ON
  SOURCES ${concepts_sources}
  HEADERS ${concepts_headers}
  MODULE_DEPENDENCIES pika_config_local pika_preprocessor pika_type_support
  CMAKE_SUBDIRS examples tests
)
