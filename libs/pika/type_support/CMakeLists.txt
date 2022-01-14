# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type_support_headers
    pika/type_support/detail/with_result_of.hpp
    pika/type_support/detail/wrap_int.hpp
    pika/type_support/always_void.hpp
    pika/type_support/decay.hpp
    pika/type_support/detected.hpp
    pika/type_support/empty_function.hpp
    pika/type_support/equality.hpp
    pika/type_support/identity.hpp
    pika/type_support/lazy_conditional.hpp
    pika/type_support/lazy_enable_if.hpp
    pika/type_support/pack.hpp
    pika/type_support/static.hpp
    pika/type_support/unwrap_ref.hpp
    pika/type_support/unused.hpp
    pika/type_support/void_guard.hpp
)

set(type_support_sources)

include(pika_add_module)
pika_add_module(
  pika type_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${type_support_sources}
  HEADERS ${type_support_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config_local
  CMAKE_SUBDIRS examples tests
)
