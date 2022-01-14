# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type_support_headers
    hpx/type_support/detail/with_result_of.hpp
    hpx/type_support/detail/wrap_int.hpp
    hpx/type_support/always_void.hpp
    hpx/type_support/decay.hpp
    hpx/type_support/detected.hpp
    hpx/type_support/empty_function.hpp
    hpx/type_support/equality.hpp
    hpx/type_support/identity.hpp
    hpx/type_support/lazy_conditional.hpp
    hpx/type_support/lazy_enable_if.hpp
    hpx/type_support/pack.hpp
    hpx/type_support/static.hpp
    hpx/type_support/unwrap_ref.hpp
    hpx/type_support/unused.hpp
    hpx/type_support/void_guard.hpp
)

set(type_support_sources)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local type_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${type_support_sources}
  HEADERS ${type_support_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config_local
  CMAKE_SUBDIRS examples tests
)
