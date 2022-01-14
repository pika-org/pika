# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(iterator_support_headers
    pika/iterator_support/traits/is_iterator.hpp
    pika/iterator_support/traits/is_range.hpp
    pika/iterator_support/traits/is_sentinel_for.hpp
    pika/iterator_support/boost_iterator_categories.hpp
    pika/iterator_support/counting_iterator.hpp
    pika/iterator_support/counting_shape.hpp
    pika/iterator_support/generator_iterator.hpp
    pika/iterator_support/iterator_adaptor.hpp
    pika/iterator_support/iterator_facade.hpp
    pika/iterator_support/iterator_range.hpp
    pika/iterator_support/range.hpp
    pika/iterator_support/transform_iterator.hpp
    pika/iterator_support/zip_iterator.hpp
)

set(iterator_support_sources)

include(pika_add_module)
pika_add_module(
  pika iterator_support
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER "pika/iterator_support/traits/is_iterator.hpp"
                             "pika/iterator_support/traits/is_range.hpp"
  SOURCES ${iterator_support_sources}
  HEADERS ${iterator_support_headers}
  MODULE_DEPENDENCIES pika_config_local pika_datastructures pika_functional
                      pika_serialization pika_type_support
  CMAKE_SUBDIRS examples tests
)
