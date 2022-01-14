# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(util_headers
    pika/util/detail/reserve.hpp
    pika/util/get_and_reset_value.hpp
    pika/util/get_entry_as.hpp
    pika/util/insert_checked.hpp
    pika/util/ios_flags_saver.hpp
    pika/util/manage_config.hpp
    pika/util/max.hpp
    pika/util/min.hpp
)

set(util_sources manage_config.cpp)

include(pika_add_module)
pika_add_module(
  pika util
  GLOBAL_HEADER_GEN ON
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  MODULE_DEPENDENCIES pika_config_local pika_concepts pika_errors pika_format
                      pika_iterator_support
  CMAKE_SUBDIRS examples tests
)
