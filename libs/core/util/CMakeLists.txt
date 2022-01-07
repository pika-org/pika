# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(util_headers
    hpx/util/detail/reserve.hpp
    hpx/util/get_and_reset_value.hpp
    hpx/util/get_entry_as.hpp
    hpx/util/insert_checked.hpp
    hpx/util/ios_flags_saver.hpp
    hpx/util/manage_config.hpp
    hpx/util/max.hpp
    hpx/util/min.hpp
)

# cmake-format: off
set(util_compat_headers
    hpx/util.hpp => hpx/include/util.hpp
)
# cmake-format: on

set(util_sources manage_config.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local util
  GLOBAL_HEADER_GEN ON
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  COMPAT_HEADERS ${util_compat_headers}
  MODULE_DEPENDENCIES hpx_config_local hpx_concepts hpx_errors hpx_format
                      hpx_iterator_support
  CMAKE_SUBDIRS examples tests
)
