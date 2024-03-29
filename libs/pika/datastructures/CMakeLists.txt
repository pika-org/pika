# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(datastructures_headers
    pika/datastructures/detail/small_vector.hpp pika/datastructures/detail/variant.hpp
    pika/datastructures/member_pack.hpp pika/datastructures/traits/is_tuple_like.hpp
    pika/datastructures/variant.hpp
)

set(datastructures_sources)

include(pika_add_module)
pika_add_module(
  pika datastructures
  GLOBAL_HEADER_GEN ON
  SOURCES ${datastructures_sources}
  HEADERS ${datastructures_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config pika_concepts pika_type_support
  CMAKE_SUBDIRS examples tests
)
