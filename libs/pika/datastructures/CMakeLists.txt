# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Compatibility with using Boost.FileSystem, introduced in V1.4.0
pika_option(
  PIKA_DATASTRUCTURES_WITH_ADAPT_STD_TUPLE BOOL
  "Enable compatibility of pika::tuple with std::tuple. (default: ON)" ON
  ADVANCED
  CATEGORY "Modules"
  MODULE DATASTRUCTURES
)

if(PIKA_DATASTRUCTURES_WITH_ADAPT_STD_TUPLE)
  pika_add_config_define_namespace(
    DEFINE PIKA_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE NAMESPACE DATASTRUCTURES
  )
endif()

set(datastructures_headers
    pika/datastructures/any.hpp
    pika/datastructures/detail/small_vector.hpp
    pika/datastructures/detail/variant.hpp
    pika/datastructures/member_pack.hpp
    pika/datastructures/optional.hpp
    pika/datastructures/tuple.hpp
    pika/datastructures/traits/supports_streaming_with_any.hpp
    pika/datastructures/traits/is_tuple_like.hpp
    pika/datastructures/variant.hpp
    pika/datastructures/variant_helper.hpp
)

set(datastructures_sources)

include(pika_add_module)
pika_add_module(
  pika datastructures
  GLOBAL_HEADER_GEN ON
  SOURCES ${datastructures_sources}
  HEADERS ${datastructures_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "pika/datastructures/variant_helper.hpp"
  MODULE_DEPENDENCIES pika_assertion pika_config_local pika_concepts
                      pika_type_support
  CMAKE_SUBDIRS examples tests
)
