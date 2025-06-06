# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/functional/include
set(functional_headers
    pika/functional/bind.hpp
    pika/functional/bind_back.hpp
    pika/functional/bind_front.hpp
    pika/functional/deferred_call.hpp
    pika/functional/detail/basic_function.hpp
    pika/functional/detail/empty_function.hpp
    pika/functional/detail/function_registration.hpp
    pika/functional/detail/invoke_result_plain_function.hpp
    pika/functional/detail/reset_function.hpp
    pika/functional/detail/vtable/callable_vtable.hpp
    pika/functional/detail/vtable/copyable_vtable.hpp
    pika/functional/detail/vtable/function_vtable.hpp
    pika/functional/detail/vtable/vtable.hpp
    pika/functional/first_argument.hpp
    pika/functional/function.hpp
    pika/functional/invoke.hpp
    pika/functional/one_shot.hpp
    pika/functional/traits/get_function_address.hpp
    pika/functional/traits/get_function_annotation.hpp
    pika/functional/traits/is_bind_expression.hpp
    pika/functional/unique_function.hpp
)

# Default location is $PIKA_ROOT/libs/functional/src
set(functional_sources basic_function.cpp empty_function.cpp)

include(pika_add_module)
pika_add_module(
  pika functional
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
    "pika/functional/traits/get_function_address.hpp"
    "pika/functional/traits/get_function_annotation.hpp"
    "pika/functional/traits/is_bind_expression.hpp"
  SOURCES ${functional_sources}
  HEADERS ${functional_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config
    pika_datastructures
    pika_debugging
    pika_errors
    pika_tag_invoke
    pika_type_support
  CMAKE_SUBDIRS examples tests
)
