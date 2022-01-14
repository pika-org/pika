# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(concepts_headers hpx/concepts/concepts.hpp hpx/concepts/has_member_xxx.hpp
                     hpx/concepts/has_xxx.hpp
)

set(concepts_sources)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local concepts
  GLOBAL_HEADER_GEN ON
  SOURCES ${concepts_sources}
  HEADERS ${concepts_headers}
  MODULE_DEPENDENCIES hpx_config_local hpx_preprocessor hpx_type_support
  CMAKE_SUBDIRS examples tests
)
