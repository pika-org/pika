# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeFindDependencyMacro)

function(pika_find_package)
  if(PIKA_FIND_PACKAGE)
    find_dependency(${ARGN})
  else()
    find_package(${ARGN})
  endif()
endfunction(pika_find_package)
